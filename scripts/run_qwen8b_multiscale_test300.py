"""Colab GPU worker: Locked zero-touch evaluation on official_test_300 (300 unseen
photographers / 147 countries) with Top-8 Multi-Scale Spatial KDE CoT prompts.

Evaluates:
  1. The single locked winning QLoRA adapter (selected strictly on author_holdout_66)
  2. Exact controlled ablation: Base untrained Qwen3-VL-8B on the identical Top-8 prompts
     (via PeftModel.disable_adapter() without reloading the 8B weights)

Supports incremental JSONL resume and VRAM-aware OOM fallback.
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
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from multiscale_vlm_utils import (
    build_multiscale_cot_prompt,
    geoscore,
    haversine_km,
    parse_multiscale_json,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_clean_image(path: Path, max_edge: int = 896) -> Image.Image:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    if max(rgb.size) > max_edge:
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


@torch.inference_mode()
def run_eval_pass(
    model,
    processor,
    rows,
    img_dir: Path,
    cc_names: dict,
    out_path: Path,
    label: str,
    max_edge: int = 896,
    batch_size: int = 8,
):
    model.eval()
    processor.tokenizer.padding_side = "left"

    done = {}
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                done[x["id"]] = x
        if done:
            print(f"  [{label}] Resuming from {len(done)}/{len(rows)}", flush=True)

    todo = [r for r in rows if r["id"] not in done]
    t0 = time.monotonic()
    cur_bs = batch_size
    with out_path.open("a", encoding="utf-8") as fh:
        idx = 0
        while idx < len(todo):
            grp = todo[idx:idx + cur_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=max_edge) for r in grp]
            prompts = [build_multiscale_cot_prompt(r["kde_distribution"]) for r in grp]
            chat_texts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
            try:
                inputs = processor(text=chat_texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=False,
                    num_return_sequences=1,
                    use_cache=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    print(f"  [{label}] CUDA OOM -> reducing batch_size to {cur_bs}", flush=True)
                    continue
                raise

            prompt_len = inputs["input_ids"].shape[1]
            raws = processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
            for r, raw in zip(grp, raws):
                p = parse_multiscale_json(raw, cc_names)
                coords = p["coordinates"]
                dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
                rec = {
                    "id": r["id"],
                    "true_country": r["country"],
                    "true_latitude": r["latitude"],
                    "true_longitude": r["longitude"],
                    "guess_country": p["country"],
                    "guess_coordinates": list(coords) if coords else None,
                    "distance_km": dist,
                    "selected_cluster": p["selected_cluster"],
                    "confidence": p["confidence"],
                    "visual_clues": p["visual_clues"],
                    "reasoning": p["reasoning"],
                    "parsed": p["parsed"],
                    "raw": raw,
                }
                done[r["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            idx += len(grp)
            if idx == len(grp) or idx % 40 < cur_bs or idx >= len(todo):
                rate = idx / max(1e-3, time.monotonic() - t0)
                eta = (len(todo) - idx) / max(1e-3, rate) / 60.0
                print(f"  [{label}] {len(done):3d}/{len(rows)} | {rate:.2f} img/s | ETA={eta:.1f}m", flush=True)

    preds = [done[r["id"]] for r in rows]
    valid_d = [x["distance_km"] for x in preds if x["distance_km"] is not None]
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in preds)
    summary = {
        "label": label,
        "n": len(preds),
        "parsed": sum(1 for x in preds if x["parsed"]),
        "country_correct": cc_ok,
        "country_acc_pct": round(100.0 * cc_ok / len(preds), 2),
        "le_25km": sum(d <= 25.0 for d in valid_d),
        "le_200km": sum(d <= 200.0 for d in valid_d),
        "median_km": round(float(np.median(valid_d)), 1) if valid_d else None,
        "geoscore": round(float(sum(geoscore(d) for d in valid_d) / len(preds)), 1),
        "sel_top1": sum(1 for x in preds if x["selected_cluster"] == 1),
        "sel_2_to_8": sum(1 for x in preds if x["selected_cluster"] in range(2, 9)),
        "sel_override": sum(1 for x in preds if x["selected_cluster"] == 0),
    }
    print(f"  >>> {label} summary: {json.dumps(summary)}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--adapter-dir", default="locked_adapter")
    parser.add_argument("--adapter-label", default="locked_sft_8b_top8")
    parser.add_argument("--max-edge", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    test_rows = [json.loads(l) for l in (args.input_dir / "official_test_300_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    with ZipFile(args.input_dir / "official-test-300.zip") as z:
        for r in test_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"Test SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(test_rows)} official_test_300 images.", flush=True)

    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_ID} + {args.adapter_dir} on {torch.cuda.get_device_name(0)} (compute={compute_dtype})...", flush=True)
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    rev = str(getattr(base_model.config, "_commit_hash", "unknown"))
    model = PeftModel.from_pretrained(base_model, str(args.input_dir / args.adapter_dir), is_trainable=False).eval()
    print(f"Loaded in {time.monotonic() - t0:.1f}s | rev={rev} | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bs = max(args.batch_size, 12) if total_vram_gb >= 35.0 else (args.batch_size if total_vram_gb >= 20.0 else 4)

    # 1. Locked winning QLoRA adapter on official_test_300
    print(f"\n=== Step 1/2: Locked Winning Adapter ({args.adapter_label}) on official_test_300 (bs={bs}) ===", flush=True)
    s_locked = run_eval_pass(
        model, processor, test_rows, img_dir, cc_names,
        args.output_dir / "locked_finetuned_8b_test300_top8.jsonl",
        args.adapter_label, max_edge=args.max_edge, batch_size=bs,
    )

    # 2. Controlled ablation: Base untrained 8B on identical Top-8 prompts
    print(f"\n=== Step 2/2: Base Untrained Qwen3-VL-8B (disable_adapter) on official_test_300 (bs={bs}) ===", flush=True)
    with model.disable_adapter():
        s_base = run_eval_pass(
            model, processor, test_rows, img_dir, cc_names,
            args.output_dir / "base_8b_test300_top8.jsonl",
            "base_8b_top8", max_edge=args.max_edge, batch_size=bs,
        )

    report = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "adapter_label": args.adapter_label,
        "max_edge": args.max_edge,
        "gpu": torch.cuda.get_device_name(0),
        "locked_finetuned": s_locked,
        "base_untrained_top8": s_base,
    }
    (args.output_dir / "test300_eval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted locked official_test_300 evaluation!", flush=True)


if __name__ == "__main__":
    main()
