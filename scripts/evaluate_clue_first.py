"""Compare direct geolocation with location prompts conditioned on visible text."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from PIL import Image
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from phase1 import QWEN, QWEN_REV, clean_image
from sft_colab import PROMPT

OCR_PROMPT = (
    "Transcribe up to three short location-relevant strings that are actually visible in this photo: "
    "place or street names, route numbers with a visible route sign, addresses on buildings, web "
    "domains, or distinctive words on road signs or storefronts. Keep the original alphabet and "
    "spelling. Use the clearest text. Omit any unreadable words, generic numbers, license plates, "
    "camera overlays and watermarks. Do not guess a language or place. "
    'Return only a JSON object with key "text" and an array of strings as its value. '
    'Use {"text":[]} when no relevant text is readable.'
)
EVIDENCE_PROMPT = (
    "\nConsider the following possible transcriptions from the same image, along with the rest "
    "of the scene. Treat these as observations and use them only if useful. They may contain "
    "reading errors; a sign pointing toward a destination need not be at that destination. "
    "Visible text: "
)


def read_text(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid", []
    if not isinstance(obj, dict) or set(obj) != {"text"} or not isinstance(obj["text"], list):
        return "invalid", []
    values = obj["text"]
    if len(values) > 3 or any(not isinstance(s, str) or len(s) > 100 or not s.strip() for s in values):
        return "invalid", []
    strings = list(dict.fromkeys(s.strip() for s in values))
    return ("nonempty" if strings else "empty"), strings


def conditioned_prompt(strings):
    return PROMPT + EVIDENCE_PROMPT + json.dumps(strings, ensure_ascii=False)


def marked_crop(photo, bounds):
    crop = photo.crop(tuple(bounds))
    zoom = min(4, 768 / max(crop.size))
    return crop.resize(tuple(round(value * zoom) for value in crop.size), Image.Resampling.BICUBIC)


def chat(processor, plain):
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": plain}]}],
        tokenize=False, add_generation_prompt=True,
    )


@torch.inference_mode()
def generate(model, processor, pairs, image_lookup, batch_size, cap, name):
    answers = {}
    for start in range(0, len(pairs), batch_size):
        group = pairs[start:start + batch_size]
        ids = [item[0] for item in group]
        prompts = [chat(processor, item[1]) for item in group]
        inputs = processor(text=prompts, images=[image_lookup[i] for i in ids],
                           padding=True, return_tensors="pt").to("cuda")
        tokens = model.generate(**inputs, max_new_tokens=cap, do_sample=False, use_cache=True)
        completion = processor.tokenizer.batch_decode(
            tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        answers.update(zip(ids, completion, strict=True))
        print(f"{name} {start + len(group)}/{len(pairs)}", flush=True)
    return answers


def read_images(directory, rows):
    photos = {}
    crops = {}
    oracle = {}
    for row in rows:
        path = directory / row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['id']}")
        photo = clean_image(path)
        photos[row["id"]] = photo
        if row.get("verified_text"):
            oracle[row["id"]] = row["verified_text"]
            crops[row["id"]] = marked_crop(photo, row["crop_pixels"])
    return photos, crops, oracle


def load_for_inference(adapter):
    dtype = torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    processor.tokenizer.padding_side = "left"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quant, device_map={"": 0},
        dtype=dtype, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter).eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    manifest = (args.input / "inputs.json").read_bytes()
    rows = json.loads(manifest)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated image IDs")
    photos, crops, oracle = read_images(args.input, rows)
    adapter = args.adapter or args.input / "adapter"
    model, processor = load_for_inference(adapter)

    direct = generate(model, processor, [(i, PROMPT) for i in ids], photos, 4, 40, "direct")
    read = generate(model, processor, [(i, OCR_PROMPT) for i in ids], photos, 4, 72, "read_full")
    text = {i: read_text(raw) for i, raw in read.items()}
    model_geo = generate(model, processor, [(i, conditioned_prompt(text[i][1])) for i in ids
                                           if text[i][1]], photos, 4, 40, "model_text")
    human_geo = generate(model, processor, [(i, conditioned_prompt(strings))
                                           for i, strings in oracle.items()], photos, 4, 40, "verified_text")
    crop_read = generate(model, processor, [(i, OCR_PROMPT) for i in crops], crops, 4, 72, "read_crop")
    crop_text = {i: read_text(raw) for i, raw in crop_read.items()}
    crop_geo = generate(model, processor, [(i, conditioned_prompt(strings))
                                          for i, (_, strings) in crop_text.items() if strings],
                        photos, 4, 40, "crop_text")

    cfg = {
        "model": QWEN, "revision": QWEN_REV, "gpu": torch.cuda.get_device_name(),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "direct_prompt": PROMPT, "ocr_prompt": OCR_PROMPT, "evidence_prompt_template": EVIDENCE_PROMPT,
        "batch_size": 4, "max_geo_tokens": 40, "max_ocr_tokens": 72, "temperature": "greedy",
        "n": len(ids), "verified_text_n": len(oracle), "model_text_n": len(model_geo), "crop_text_n": len(crop_geo),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for i in ids:
            row = {
                "id": i, "direct": direct[i], "read_full": read[i], "read_status": text[i][0],
                "model_strings": text[i][1], "model_text": model_geo.get(i, direct[i]),
                "used_model_text": i in model_geo, "verified_text": human_geo.get(i),
            }
            if i in crop_text:
                row.update(read_crop=crop_read[i], crop_strings=crop_text[i][1],
                           crop_status=crop_text[i][0], crop_text=crop_geo.get(i, direct[i]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "configuration.json").write_text(json.dumps(cfg, indent=2) + "\n")
    with ZipFile(args.output / "clue-first-results.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name in ("answers.jsonl", "configuration.json"):
            archive.write(args.output / name, name)


if __name__ == "__main__":
    main()
