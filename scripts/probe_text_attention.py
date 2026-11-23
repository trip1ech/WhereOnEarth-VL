"""Inspect language-model attention and run matched text-region interventions."""
import argparse
import hashlib
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageFilter
import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from phase1 import QWEN, QWEN_REV, clean_image
from sft_colab import PROMPT

VIEWS = (
    ("geo_full", "full", "geo"), ("ocr_full", "full", "ocr"),
    ("ocr_crop", "crop", "ocr"), ("geo_text_blur", "text_blur", "geo"),
    ("geo_control_blur", "control_blur", "geo"), ("geo_transcribed", "full", "provided"),
    ("ocr_text_blur", "text_blur", "ocr"), ("ocr_control_blur", "control_blur", "ocr"),
)
MAPS = {"geo_full", "ocr_full"}


def pixel_box(box, size):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"Invalid box: {box}")
    width, height = size
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)


def prepare_views(image, row):
    regions = {key: pixel_box(row[key], image.size) for key in ("text_box", "control_box", "crop_box")}
    width = regions["text_box"][2] - regions["text_box"][0]
    height = regions["text_box"][3] - regions["text_box"][1]
    if (width, height) != (regions["control_box"][2] - regions["control_box"][0],
                          regions["control_box"][3] - regions["control_box"][1]):
        raise ValueError(f"Expected equal intervention sizes for {row['case']}: {regions}")
    blur_radius = max(4, min(width, height) * 0.40)
    views = {"full": image}
    for name in ("text", "control"):
        changed = image.copy()
        region = regions[f"{name}_box"]
        changed.paste(image.crop(region).filter(ImageFilter.GaussianBlur(blur_radius)), region)
        views[f"{name}_blur"] = changed

    cropped = image.crop(regions["crop_box"])
    zoom = min(4.0, 768 / max(cropped.size))
    views["crop"] = cropped.resize(tuple(round(v * zoom) for v in cropped.size), Image.Resampling.BICUBIC)
    return views, regions, blur_radius


def chat_prompt(processor, text):
    return processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": text}
    ]}], tokenize=False, add_generation_prompt=True)


def prompt_set(row):
    return {
        "geo": PROMPT,
        "ocr": (f"Transcribe the visible text {row['ocr_target']}. "
                "Copy the actual letters and digits. Do not infer missing text. "
                "Output only the transcription, or UNREADABLE if you cannot read it."),
        "provided": (PROMPT + "\nThe following is a transcription of text visible in the photo. "
                     "Use it as visual evidence if helpful:\n" + row["transcription"]),
    }


class TextAttention:
    def __init__(self, text_model):
        n = len(text_model.layers)
        self.layers = sorted({0, n // 3, 2 * n // 3, n - 1})
        self.image_tokens = None
        self.weights = {}
        self.hooks = [text_model.layers[i].self_attn.register_forward_hook(self.hook(i))
                      for i in self.layers]

    def hook(self, layer):
        def read_weights(_module, _args, output):
            if self.image_tokens is None:
                return
            weights = output[1]
            if weights is None:
                raise RuntimeError("No returned attention weights; use eager attention")
            head_mean = weights[0, :, -1, :].float().mean(dim=0)
            self.weights[layer].append(head_mean.index_select(0, self.image_tokens).cpu().numpy())
        return read_weights

    def start(self, image_tokens):
        self.weights = {i: [] for i in self.layers}
        self.image_tokens = image_tokens

    def stop(self):
        self.image_tokens = None
        return np.asarray([self.weights[layer] for layer in self.layers], dtype=np.float32)

    def close(self):
        for handle in self.hooks:
            handle.remove()


def selected_steps(tokenizer, generated, variant):
    pieces = [tokenizer.decode([int(token)], skip_special_tokens=True) for token in generated]
    expression = r"[0-9]" if variant == "geo_full" else r"[^\W_]"
    steps = [i for i, piece in enumerate(pieces) if re.search(expression, piece)]
    return steps or [i for i, piece in enumerate(pieces) if piece.strip()] or [0]


@torch.inference_mode()
def infer(model, processor, attention, prompt, image, variant, merge_size, token_id):
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    prefix = inputs.input_ids.shape[1]
    if variant in MAPS:
        image_tokens = torch.where(inputs.input_ids[0] == token_id)[0]
        frames, height, width = inputs.image_grid_thw[0].tolist()
        shape = (height // merge_size, width // merge_size)
        if frames != 1 or math.prod(shape) != image_tokens.numel():
            raise ValueError("Image tokens do not match the single-image spatial grid")
        attention.start(image_tokens)

    generated = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=48)[0, prefix:]
    raw = processor.tokenizer.decode(generated, skip_special_tokens=True)
    if variant not in MAPS:
        return raw, None, None
    weights = attention.stop()
    if weights.shape[1] != len(generated) or not np.isfinite(weights).all():
        raise ValueError(f"Attention records do not match decoded tokens: {weights.shape}, {len(generated)}")
    steps = selected_steps(processor.tokenizer, generated, variant)
    per_layer = weights[:, steps, :].mean(axis=1).reshape((len(attention.layers), *shape))
    return raw, per_layer, {
        "grid": list(shape), "layers": attention.layers, "answer_steps": steps,
        "answer_tokens": [processor.tokenizer.decode([int(generated[i])]) for i in steps],
        "image_mass": float(per_layer.sum(axis=(1, 2)).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = (args.bundle / "cases.json").read_bytes()
    rows = json.loads(manifest)
    for row in rows:
        if hashlib.sha256((args.bundle / row["image"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Wrong image bytes: {row['case']}")

    dtype = torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quant, dtype=dtype,
        device_map={"": 0}, attn_implementation="eager",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, args.bundle / "adapter").eval()
    attention = TextAttention(base.model.language_model)
    answers = []
    maps = {}
    try:
        for row in rows:
            original = clean_image(args.bundle / row["image"])
            images, regions, radius = prepare_views(original, row)
            prompts = {key: chat_prompt(processor, text) for key, text in prompt_set(row).items()}
            region_meta = {"image_size": original.size, "boxes": regions, "blur_radius": radius}
            for mode in ("base", "sft"):
                with model.disable_adapter() if mode == "base" else nullcontext():
                    for variant, view, task in VIEWS:
                        raw, grid, meta = infer(model, processor, attention, prompts[task], images[view],
                                                variant, base.visual.spatial_merge_size,
                                                base.config.image_token_id)
                        key = f"{row['case']}__{mode}__{variant}"
                        if grid is not None:
                            maps[key] = grid
                        answers.append({"case": row["case"], "mode": mode, "variant": variant,
                                        "answer": raw, "map_key": key if grid is not None else None,
                                        "attention": meta, **region_meta})
                        print(f"{row['case']} {mode} {variant} done", flush=True)
    finally:
        attention.close()

    (args.output / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
    np.savez_compressed(args.output / "attention.npz", **maps)
    (args.output / "configuration.json").write_text(json.dumps({
        "model": QWEN, "revision": QWEN_REV, "gpu": torch.cuda.get_device_name(),
        "case_sha256": hashlib.sha256(manifest).hexdigest(), "attention_backend": "eager",
        "adapter_sha256": hashlib.sha256((args.bundle / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
        "layers_zero_based": attention.layers, "views": VIEWS,
        "prompts_by_case": {row["case"]: prompt_set(row) for row in rows},
    }, indent=2) + "\n")
    with ZipFile(args.output / "text-attention-results.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name in ("answers.jsonl", "attention.npz", "configuration.json"):
            archive.write(args.output / name, name)


if __name__ == "__main__":
    main()
