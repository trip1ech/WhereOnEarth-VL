"""Run Qwen3-VL-8B-Instruct (4-bit NF4) on Colab GPU in two modes:
  1. Standalone zero-shot (no retrieval prior)
  2. Retrieval-augmented spatial KDE distribution reranking & coordinate refinement

Saves incremental JSONL outputs so partial runs can resume cleanly.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from kde_vlm_utils import (
    ZERO_SHOT_PROMPT,
    build_kde_rerank_prompt,
    haversine_km,
    parse_vlm_json,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_clean_image(path: Path, max_edge: int = 896) -> Image.Image:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    if max(rgb.size) > max_edge:
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


def load_model(model_id: str):
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {model_id} (4-bit NF4, compute={compute_dtype}) on {torch.cuda.get_device_name(0)}...", flush=True)
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    rev = getattr(model.config, "_commit_hash", "unknown")
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded {model_id} (rev={rev}) in {time.monotonic() - t0:.1f}s | VRAM={mem_gb:.2f} GB", flush=True)
    return model, processor, str(rev)


@torch.inference_mode()
def generate_batch(model, processor, images, prompt_texts, max_new_tokens: int):
    chat_texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompt_texts
    ]
    inputs = processor(
        text=chat_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)


def run_pass(
    label: str,
    model,
    processor,
    rows,
    img_dir: Path,
    cc_names,
    out_path: Path,
    *,
    prompt_fn,
    max_new_tokens: int,
    batch_size: int,
    max_edge: int,
):
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
        if done:
            print(f"  [{label}] Resuming from {len(done)}/{len(rows)} cached records", flush=True)

    todo = [r for r in rows if r["id"] not in done]
    t0 = time.monotonic()
    cur_bs = batch_size

    with out_path.open("a", encoding="utf-8") as fh:
        idx = 0
        while idx < len(todo):
            group = todo[idx:idx + cur_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=max_edge) for r in group]
            prompts = [prompt_fn(r) for r in group]
            try:
                raws = generate_batch(model, processor, imgs, prompts, max_new_tokens=max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    print(f"  [{label}] CUDA OOM -> reducing batch size to {cur_bs}", flush=True)
                    continue
                raise

            for r, raw in zip(group, raws):
                parsed = parse_vlm_json(raw, cc_names)
                coords = parsed["coordinates"]
                dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
                rec = {
                    "id": r["id"],
                    "true_country": r["country"],
                    "true_latitude": r["latitude"],
                    "true_longitude": r["longitude"],
                    "guess_country": parsed["country"],
                    "guess_coordinates": list(coords) if coords else None,
                    "distance_km": dist,
                    "selected_candidate": parsed["selected_candidate"],
                    "confidence": parsed["confidence"],
                    "visual_clues": parsed["visual_clues"],
                    "candidate_comparison": parsed["candidate_comparison"],
                    "parsed": parsed["parsed"],
                    "raw": raw,
                }
                done[r["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            idx += len(group)

            if idx == len(group) or idx % 20 < cur_bs or idx >= len(todo):
                all_recs = [done[r["id"]] for r in rows if r["id"] in done]
                n_cur = len(all_recs)
                cc_ok = sum(x["guess_country"] == x["true_country"] for x in all_recs)
                valid_d = [x["distance_km"] for x in all_recs if x["distance_km"] is not None]
                med = float(np.median(valid_d)) if valid_d else float("nan")
                pts = float(sum(5000.0 * np.exp(-d / 1492.7) for d in valid_d) / n_cur)
                elapsed = time.monotonic() - t0
                rate = idx / max(1e-3, elapsed)
                eta_m = (len(todo) - idx) / max(1e-3, rate) / 60.0
                print(
                    f"  [{label}] {n_cur:3d}/{len(rows)} | cc={cc_ok}/{n_cur} ({100*cc_ok/n_cur:.1f}%) | "
                    f"med={med:.1f}km | pts={pts:.1f} | {rate:.2f} img/s | ETA={eta_m:.1f}m",
                    flush=True,
                )

    ordered = [done[r["id"]] for r in rows]
    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered), encoding="utf-8")
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-edge", type=int, default=896)
    parser.add_argument("--skip-zeroshot", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    meta = json.loads((args.input_dir / "bundle_meta.json").read_text(encoding="utf-8"))
    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (args.input_dir / "kde_distributions.jsonl").read_text(encoding="utf-8").splitlines()]

    # Extract & verify images
    with ZipFile(args.input_dir / meta["image_zip"]) as z:
        for r in rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"Image SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(rows)} images for split={meta['split']} (max_edge={args.max_edge})", flush=True)

    model, processor, rev = load_model(args.model_id)

    config = {
        "split": meta["split"],
        "n": len(rows),
        "model_id": args.model_id,
        "model_revision": rev,
        "quantization": "4bit_nf4_double_quant",
        "max_edge": args.max_edge,
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print("\n=== Pass 1: Retrieval-Augmented Spatial KDE -> Qwen3-VL-8B-Instruct ===", flush=True)
    run_pass(
        "kde_rerank",
        model,
        processor,
        rows,
        img_dir,
        cc_names,
        args.output_dir / "qwen8b_kde_rerank.jsonl",
        prompt_fn=lambda r: build_kde_rerank_prompt(r["kde_distribution"]),
        max_new_tokens=260,
        batch_size=args.batch_size,
        max_edge=args.max_edge,
    )

    if not args.skip_zeroshot:
        print("\n=== Pass 2: Standalone Qwen3-VL-8B-Instruct Zero-Shot (No Retrieval Prior) ===", flush=True)
        run_pass(
            "zeroshot_8b",
            model,
            processor,
            rows,
            img_dir,
            cc_names,
            args.output_dir / "qwen8b_zeroshot.jsonl",
            prompt_fn=lambda _r: ZERO_SHOT_PROMPT,
            max_new_tokens=160,
            batch_size=args.batch_size,
            max_edge=args.max_edge,
        )

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nCompleted Qwen3-VL-8B-Instruct evaluation for {meta['split']}.", flush=True)


if __name__ == "__main__":
    main()
