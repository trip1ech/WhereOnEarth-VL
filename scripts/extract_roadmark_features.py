"""Capture frozen Qwen vision features and score balanced road-marking questions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import math
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from phase1 import QWEN, QWEN_REV, clean_image

QUESTION = (
    "Inspect the marking in the middle of the road that separates traffic in opposite directions. "
    "Ignore shoulder lines and lane markings for traffic traveling in the same direction. "
    "Count solid and dashed lines the same way. Which option describes the center marking?\n"
)
CHOICES = (
    "A. One line\nB. Two parallel lines",
    "A. Two parallel lines\nB. One line",
)
CROP_FRACTION = 0.48
FIXED_CROP_CENTER = (0.5, 0.76)


def crop_view(image, center):
    x, y = center
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x, y)):
        raise ValueError(f"Invalid crop center: {center}")
    width, height = image.size
    w, h = round(width * CROP_FRACTION), round(height * CROP_FRACTION)
    left = max(0, min(round(x * width - w / 2), width - w))
    top = max(0, min(round(y * height - h / 2), height - h))
    return image.crop((left, top, left + w, top + h)).resize(image.size, Image.Resampling.BICUBIC)


def spatial_pool(x):
    channels_first = x.permute(2, 0, 1).unsqueeze(0)
    return F.adaptive_avg_pool2d(channels_first, (2, 2)).flatten().cpu().numpy()


def vision_vectors(captures, grids, merge):
    patch_counts = [t * h * w for t, h, w in grids]
    token_counts = [n // (merge ** 2) for n in patch_counts]
    pre, main = captures["main"]
    pre_pieces = pre.split(patch_counts)
    main_pieces = main.split(token_counts)
    early = [pair[1].split(token_counts) for name, pair in captures.items() if name != "main"]
    results = []
    for i, (t, h, w) in enumerate(grids):
        if t != 1 or h % merge or w % merge:
            raise ValueError(f"Unexpected image grid {t, h, w}")
        height, width = h // merge, w // merge
        # The vision tower orders unmerged patches by merger cell, then intra-cell offset.
        block = pre_pieces[i].reshape(height, width, merge ** 2, -1)
        vit = block.mean(dim=2)
        projected = main_pieces[i].reshape(height, width, -1)
        results.append({
            "vit": spatial_pool(vit),
            "projector": spatial_pool(projected),
            "vit_native": spatial_pool(block.flatten(2)),
            "vit_global": vit.mean(dim=(0, 1)).cpu().numpy(),
            "projector_global": projected.mean(dim=(0, 1)).cpu().numpy(),
            **{f"deepstack_{j}": spatial_pool(pieces[i].reshape(height, width, -1))
               for j, pieces in enumerate(early)},
        })
    return results


def label_tokens(tokenizer, letter):
    candidates = [tokenizer.encode(surface, add_special_tokens=False) for surface in (letter, f" {letter}")]
    tokens = sorted({ids[0] for ids in candidates if len(ids) == 1})
    if not tokens:
        raise ValueError(f"No single-token variant of {letter}")
    return tokens


def score_letters(logits, ids_a, ids_b, reversed_options):
    logits = logits.float()
    a = torch.logsumexp(logits[..., ids_a], dim=-1)
    b = torch.logsumexp(logits[..., ids_b], dim=-1)
    margin = a - b if reversed_options else b - a
    option_mass = torch.exp(torch.logaddexp(a, b) - torch.logsumexp(logits, dim=-1))
    return margin.cpu().tolist(), option_mass.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--view", choices=("full", "marked", "fixed"), default="full")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    manifest = (args.input / "inputs.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest.splitlines()]
    for row in rows:
        image = (args.input / row["image"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != row["sha256"]:
            raise ValueError(f"Wrong image bytes for {row['id']}")

    dtype = torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    processor.tokenizer.padding_side = "left"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quantization,
        dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, args.input / "adapter").eval()
    adapter_keys = [name for name, _ in model.named_parameters() if ".lora_" in name]
    if not adapter_keys or any(".language_model." not in key for key in adapter_keys):
        raise ValueError("Expected an LM-only adapter")

    prompts = [
        processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": QUESTION + choices + "\nAnswer only A or B."}
        ]}], tokenize=False, add_generation_prompt=True)
        for choices in CHOICES
    ]
    ids_a, ids_b = (label_tokens(processor.tokenizer, letter) for letter in ("A", "B"))
    if set(ids_a) & set(ids_b):
        raise ValueError("Answer token sets overlap")

    captures = {}
    active = False

    def hook_for(name):
        def hook(module, inputs, result):
            if active:
                captures[name] = (inputs[0].detach().float(), result.detach().float())
        return hook

    vision = base.visual
    hooks = [vision.merger.register_forward_hook(hook_for("main"))]
    hooks.extend(merger.register_forward_hook(hook_for(f"deepstack_{i}"))
                 for i, merger in enumerate(vision.deepstack_merger_list))
    arrays = defaultdict(list)
    decisions = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                images = [clean_image(args.input / row["image"]) for row in batch]
                if args.view == "marked":
                    images = [crop_view(image, row["crop_center"])
                              for image, row in zip(images, batch, strict=True)]
                elif args.view == "fixed":
                    images = [crop_view(image, FIXED_CROP_CENTER) for image in images]
                received = [{"id": row["id"], "base": [], "sft": []} for row in batch]
                for order, prompt in enumerate(prompts):
                    inputs = processor(text=[prompt] * len(batch), images=images,
                                       padding=True, return_tensors="pt").to("cuda")
                    active = order == 0
                    captures.clear()
                    with model.disable_adapter():
                        logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1]
                    active = False
                    if order == 0:
                        if len(captures) != 1 + len(vision.deepstack_merger_list):
                            raise RuntimeError("Not all visual mergers were captured")
                        vectors = vision_vectors(captures, inputs.image_grid_thw.tolist(),
                                                 vision.spatial_merge_size)
                        for item in vectors:
                            for name, feature in item.items():
                                arrays[name].append(feature)
                    margins, masses = score_letters(logits, ids_a, ids_b, bool(order))
                    for record, margin, mass in zip(received, margins, masses, strict=True):
                        record["base"].append({"double_margin": margin, "option_mass": mass})

                    logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1]
                    margins, masses = score_letters(logits, ids_a, ids_b, bool(order))
                    for record, margin, mass in zip(received, margins, masses, strict=True):
                        record["sft"].append({"double_margin": margin, "option_mass": mass})

                for image in images:
                    for name, picture in (("rgb_full", image),
                                          ("rgb_top", image.crop((0, 0, image.width, image.height * 3 // 10)))):
                        arrays[name].append(np.asarray(
                            picture.resize((12, 12), Image.Resampling.BILINEAR), dtype=np.float32
                        ).flatten() / 255)
                decisions.extend(received)
                print(f"Processed {len(decisions)}/{len(rows)} photos.", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    features = {name: np.stack(values) for name, values in arrays.items()}
    if any(x.shape[0] != len(rows) or not np.isfinite(x).all() for x in features.values()):
        raise RuntimeError("Incomplete or non-finite feature matrix")
    np.savez_compressed(output / "features.npz", ids=[row["id"] for row in rows], **features)
    (output / "lm_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions))
    (output / "configuration.json").write_text(json.dumps({
        "model": QWEN, "revision": QWEN_REV, "gpu": torch.cuda.get_device_name(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "adapter_sha256": hashlib.sha256((args.input / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
        "view": args.view, "crop_fraction": CROP_FRACTION, "fixed_crop_center": FIXED_CROP_CENTER,
        "vision_merge": vision.spatial_merge_size, "deepstack_layers": vision.deepstack_visual_indexes,
        "n": len(rows), "prompt": QUESTION, "choices": CHOICES,
        "answer_token_ids": {"A": ids_a, "B": ids_b},
        "feature_shapes": {name: list(values.shape) for name, values in features.items()},
    }, indent=2) + "\n")
    with ZipFile(output / "roadmark-features.zip", "w", compression=ZIP_DEFLATED) as archive:
        for filename in ("features.npz", "configuration.json", "lm_scores.jsonl"):
            archive.write(output / filename, filename)


if __name__ == "__main__":
    main()
