"""Phase C: Evaluate osv5m/baseline, Qwen3-VL-2B (zero-shot + 6k QLoRA + exact 211-country log-probs),
and dinov2-small features on the locked 300-image official OSV-5M test benchmark.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image, ImageOps
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from extract_vlm_country_priors import build_country_token_trie, score_countries_for_image, verify_cache_exactness
from phase1 import OSV_CODE, OSV_REV, QWEN, QWEN_REV, clean_image, coordinates, haversine
from retrieval import ENCODER as DINO_ENCODER, REVISION as DINO_REV
from sft_colab import PROMPT, parse_response


def make_512_jpeg(path, dst_path):
    """Exact same 512px RGB quality=90 preprocessing used in prepare_sft_data.py for Qwen3-VL."""
    im = clean_image(path)
    im.thumbnail((512, 512), Image.Resampling.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    dst_path.write_bytes(buf.getvalue())
    return dst_path


def run_osv5m_baseline(rows, img_dir, runtime_dir):
    print("\n=== 1/4: Running osv5m/baseline (official 5M-image geolocalizer) ===", flush=True)
    repo = runtime_dir / "osv5m-official"
    if not (repo / ".git").is_dir():
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    if not (repo / "models/huggingface.py").is_file():
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1",
                        "https://github.com/gastruc/osv5m.git", OSV_CODE], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    sys.path.insert(0, str(repo))
    from models.huggingface import Geolocalizer

    cwd = Path.cwd()
    try:
        os.chdir(repo)
        model = Geolocalizer.from_pretrained("osv5m/baseline", revision=OSV_REV).to("cuda").eval()
    finally:
        os.chdir(cwd)

    preds = []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            im = clean_image(img_dir / f"{row['id']}.jpg")
            x = model.transform(im).unsqueeze(0).to("cuda")
            lat_r, lon_r = model(x).detach().cpu().reshape(2).tolist()
            lat, lon = coordinates(math.degrees(lat_r), math.degrees(lon_r))
            dist = haversine(row["latitude"], row["longitude"], lat, lon)
            preds.append({
                "id": row["id"], "country": row["country"],
                "latitude": row["latitude"], "longitude": row["longitude"],
                "guess_latitude": lat, "guess_longitude": lon, "distance_km": dist,
            })
            if i == 1 or i % 50 == 0 or i == len(rows):
                med = float(np.median([p["distance_km"] for p in preds]))
                print(f"  osv5m/baseline: {i}/{len(rows)} (running median={med:.1f} km)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def run_dinov2(rows, img_dir):
    print("\n=== 2/4: Extracting dinov2-small features for official_test_300 ===", flush=True)
    proc = AutoImageProcessor.from_pretrained(DINO_ENCODER, revision=DINO_REV, use_fast=False, token=False)
    model = AutoModel.from_pretrained(DINO_ENCODER, revision=DINO_REV, token=False).to("cuda").eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            imgs = [clean_image(img_dir / f"{r['id']}.jpg") for r in rows[start:start + 32]]
            pix = proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda")
            emb = model(pixel_values=pix).last_hidden_state[:, 0, :]
            chunks.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy().astype(np.float16))
    feats = np.concatenate(chunks, axis=0)
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  dinov2-small encoded: {feats.shape}", flush=True)
    return feats


def run_qwen_all(rows, img512_dir, adapter_dir, countries):
    print("\n=== 3/4 & 4/4: Running Qwen3-VL-2B zero-shot + 6k QLoRA + exact country log-probs ===", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    processor.tokenizer.padding_side = "left"
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quant, dtype=dtype,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter_dir).eval()

    gen_prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )

    @torch.inference_mode()
    def eval_gen(label, disable_adapter):
        from contextlib import nullcontext
        ctx = model.disable_adapter() if disable_adapter else nullcontext()
        out = []
        with ctx:
            for start in range(0, len(rows), 4):
                group = rows[start:start + 4]
                imgs = [clean_image(img512_dir / f"{r['id']}.jpg") for r in group]
                inputs = processor(text=[gen_prompt] * len(group), images=imgs, padding=True, return_tensors="pt").to("cuda")
                toks = model.generate(**inputs, max_new_tokens=40, do_sample=False, use_cache=True)
                texts = processor.tokenizer.batch_decode(toks[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for r, raw in zip(group, texts):
                    cc, coords = parse_response(raw)
                    dist = haversine(r["latitude"], r["longitude"], *coords) if coords else None
                    out.append({
                        "id": r["id"], "country": r["country"], "latitude": r["latitude"], "longitude": r["longitude"],
                        "guess_country": cc, "guess_coordinates": coords, "distance_km": dist, "raw": raw,
                    })
                if start == 0 or (start // 4) % 10 == 0 or start + 4 >= len(rows):
                    cc_ok = sum(p["guess_country"] == p["country"] for p in out)
                    print(f"  {label}: {len(out)}/{len(rows)} (cc={cc_ok}/{len(out)})", flush=True)
        return out

    zs_preds = eval_gen("qwen_zero_shot", disable_adapter=True)
    sft_preds = eval_gen("qwen_sft_6000", disable_adapter=False)

    # Exact country log-probs from 6k QLoRA
    print("  Extracting exact 211+ country log-probs from Qwen3-VL 6k QLoRA...", flush=True)
    base_prefix, code_tokens = build_country_token_trie(processor.tokenizer, countries)
    score_prompt = gen_prompt + base_prefix
    priors = {}
    verified = False
    for idx, row in enumerate(rows, 1):
        im = clean_image(img512_dir / f"{row['id']}.jpg")
        inputs = processor(text=[score_prompt], images=[im], return_tensors="pt").to("cuda")
        scores = score_countries_for_image(model, inputs, code_tokens)
        if not verified:
            verify_cache_exactness(model, inputs, code_tokens, scores)
            verified = True
        vals = torch.tensor([scores[c] for c in countries], dtype=torch.float32)
        norm_lp = torch.nn.functional.log_softmax(vals, dim=0)
        probs = norm_lp.exp()
        top5 = sorted(((countries[i], float(probs[i])) for i in range(len(countries))), key=lambda x: -x[1])[:5]
        priors[row["id"]] = {
            "true_country": row["country"],
            "top1_country": top5[0][0],
            "top1_prob": top5[0][1],
            "log_probs": {countries[i]: round(float(norm_lp[i]), 5) for i in range(len(countries))},
        }
        if idx == 1 or idx % 50 == 0 or idx == len(rows):
            acc = sum(v["top1_country"] == v["true_country"] for v in priors.values()) / len(priors)
            print(f"  VLM country priors: {idx}/{len(rows)} (top1 cc={100*acc:.1f}%)", flush=True)

    return zs_preds, sft_preds, priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    img_dir = args.output / "imgs_raw"
    img512_dir = args.output / "imgs_512"
    img_dir.mkdir(exist_ok=True)
    img512_dir.mkdir(exist_ok=True)

    with ZipFile(args.input_dir / "official-test-300.zip") as z:
        rows = [json.loads(l) for l in z.read("test.jsonl").splitlines()]
        for r in rows:
            payload = z.read(r["image"])
            if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                raise ValueError(f"SHA mismatch: {r['id']}")
            p = img_dir / f"{r['id']}.jpg"
            p.write_bytes(payload)
            make_512_jpeg(p, img512_dir / f"{r['id']}.jpg")

    countries = json.loads((args.input_dir / "countries.json").read_text())
    osv_preds = run_osv5m_baseline(rows, img_dir, args.output / "runtime")
    dino_feats = run_dinov2(rows, img_dir)
    zs_preds, sft_preds, priors = run_qwen_all(rows, img512_dir, args.input_dir / "adapter", countries)

    (args.output / "osv5m_baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in osv_preds))
    (args.output / "qwen_zero_shot.jsonl").write_text("".join(json.dumps(r) + "\n" for r in zs_preds))
    (args.output / "qwen_sft_6000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sft_preds))
    (args.output / "vlm_country_priors_test300.json").write_text(json.dumps({"countries": countries, "predictions": priors}) + "\n")
    np.savez_compressed(args.output / "dinov2_test300.npz", ids=np.array([r["id"] for r in rows]), features=dino_feats)
    print(f"\nSaved all Phase C outputs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
