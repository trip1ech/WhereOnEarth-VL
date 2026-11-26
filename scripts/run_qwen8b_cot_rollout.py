"""Colab GPU worker: Generate K=4 multi-scale CoT rollouts per training image using
Qwen/Qwen3-VL-8B-Instruct (4-bit NF4), run STaR hindsight rationalization on hard
examples where all blind rollouts failed, and export:
  - cot_rollouts_raw.jsonl    (all K=4 blind rollouts + optional STaR CoT for inspection)
  - sft_cot_train.jsonl       (verified Rejection-Sampled + STaR CoT SFT targets)
  - dpo_cot_pairs.jsonl       (high-margin distance-reward chosen/rejected CoT pairs)

Also evaluates base Qwen3-VL-8B on Top-8 multi-scale clusters on author_holdout_66 for
an exact apples-to-apples pre-training baseline!
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

from multiscale_vlm_utils import (
    build_multiscale_cot_prompt,
    build_star_rationalization_prompt,
    compute_trajectory_reward,
    format_canonical_cot_json,
    geoscore,
    haversine_km,
    parse_multiscale_json,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_clean_image(path: Path, max_edge: int = 768) -> Image.Image:
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
    print(f"Loaded {model_id} (rev={rev}) in {time.monotonic() - t0:.1f}s | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    return model, processor, str(rev)


@torch.inference_mode()
def generate_samples(
    model,
    processor,
    images,
    prompt_texts,
    *,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    temperature: float = 0.75,
    top_p: float = 0.90,
    max_new_tokens: int = 230,
):
    # Explicitly repeat images/prompts rather than passing num_return_sequences>1
    # to avoid Qwen3-VL 3D M-RoPE / image_grid_thw expansion mismatches.
    rep_imgs = [im for im in images for _ in range(num_return_sequences)]
    rep_prompts = [p for p in prompt_texts for _ in range(num_return_sequences)]
    chat_texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in rep_prompts
    ]
    inputs = processor(text=chat_texts, images=rep_imgs, padding=True, return_tensors="pt").to("cuda")
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=1,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    out_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
    return [
        decoded[i * num_return_sequences:(i + 1) * num_return_sequences]
        for i in range(len(images))
    ]


STAR_LEAK_PHRASES = (
    "ground truth",
    "rationalization guide",
    "training guide",
    "provided location",
    "given coordinates",
    "as instructed",
    "do not mention",
)


def has_star_leak(visual_clues, reasoning) -> bool:
    txt = f"{visual_clues or ''} {reasoning or ''}".lower()
    return any(ph in txt for ph in STAR_LEAK_PHRASES)


def is_organic_success(parsed: dict, dist_km: float, rec: dict, d_top1: float) -> bool:
    """Category-aware verification for Rejection Sampling so same-country region flips
    only accept rollouts that genuinely move closer to the true sub-region (not lazy Top-1).
    """
    if not parsed["parsed"] or parsed["country"] != rec["country"] or dist_km is None:
        return False
    if not (parsed["visual_clues"] and parsed["reasoning"] and len(str(parsed["reasoning"])) >= 20):
        return False
    cat = rec["category"]
    if cat == "same_country_region_flip":
        return dist_km <= max(200.0, rec["best_cluster_dist_km"] + 100.0) and dist_km <= d_top1 - 100.0
    if cat == "cross_country_flip":
        return dist_km <= max(250.0, rec["best_cluster_dist_km"] + 150.0)
    if cat == "override_outside_top8":
        return dist_km <= 400.0
    # keep_top1
    return dist_km <= max(250.0, d_top1 + 100.0)


def evaluate_and_pack_datasets(rollout_records, cc_names, out_dir: Path):
    """Build clean SFT and DPO CoT datasets from completed blind rollouts + STaR rationalizations."""
    sft_rows = []
    dpo_rows = []
    stats = {
        "total_images": len(rollout_records),
        "sft_from_organic_rejection": 0,
        "sft_from_star_rationalization": 0,
        "star_rejected_leak_or_unparsed": 0,
        "dpo_pairs_total": 0,
        "dpo_pairs_cross_country": 0,
        "dpo_pairs_same_country_coord": 0,
        "dpo_pairs_keep_stability": 0,
        "dpo_pairs_chosen_organic": 0,
        "dpo_pairs_chosen_star": 0,
        "mean_dpo_margin": 0.0,
    }
    margins = []

    for rec in rollout_records:
        tlat, tlon, tcc = rec["latitude"], rec["longitude"], rec["country"]
        clusters = rec["kde_distribution"]["clusters"]
        c1 = clusters[0]
        d_top1 = haversine_km(tlat, tlon, c1["latitude"], c1["longitude"])

        # Score all K blind samples
        scored = []
        for s in rec["samples"]:
            p = parse_multiscale_json(s["raw"], cc_names)
            coords = p["coordinates"]
            dist = haversine_km(tlat, tlon, coords[0], coords[1]) if coords else None
            rew = compute_trajectory_reward(dist, p["country"], tcc, p["parsed"])
            has_text = bool(p["visual_clues"] and p["reasoning"] and len(str(p["reasoning"])) >= 20)
            scored.append({
                "raw": s["raw"],
                "parsed": p,
                "dist_km": dist,
                "reward": rew,
                "has_text": has_text,
            })

        valid_organic = [x for x in scored if x["parsed"]["parsed"] and x["has_text"]]
        valid_organic.sort(key=lambda x: -x["reward"])

        chosen_sft = None
        chosen_source = None
        chosen_reward = None
        chosen_dist = None
        if valid_organic:
            best = valid_organic[0]
            if is_organic_success(best["parsed"], best["dist_km"], rec, d_top1):
                p = best["parsed"]
                chosen_sft = format_canonical_cot_json(
                    p["visual_clues"],
                    p["reasoning"],
                    rec["target_cluster"] if p["selected_cluster"] is None else p["selected_cluster"],
                    tcc,
                    p["coordinates"][0],
                    p["coordinates"][1],
                    p["confidence"],
                )
                chosen_reward = best["reward"]
                chosen_dist = best["dist_km"]
                chosen_source = "rejection_sampled"
                stats["sft_from_organic_rejection"] += 1

        # Otherwise use STaR rationalized CoT (grounded to real coordinates!)
        if chosen_sft is None and rec.get("star_raw"):
            p_star = parse_multiscale_json(rec["star_raw"], cc_names)
            if (
                p_star["visual_clues"]
                and p_star["reasoning"]
                and len(str(p_star["reasoning"])) >= 20
                and not has_star_leak(p_star["visual_clues"], p_star["reasoning"])
            ):
                chosen_sft = format_canonical_cot_json(
                    p_star["visual_clues"],
                    p_star["reasoning"],
                    rec["target_cluster"],
                    tcc,
                    tlat,
                    tlon,
                    p_star["confidence"],
                )
                chosen_reward = compute_trajectory_reward(0.0, tcc, tcc, True)
                chosen_dist = 0.0
                chosen_source = "star_rationalized"
                stats["sft_from_star_rationalization"] += 1
            else:
                stats["star_rejected_leak_or_unparsed"] += 1

        if chosen_sft is not None:
            sft_rows.append({
                "id": rec["id"],
                "image": rec["image"],
                "category": rec["category"],
                "country": tcc,
                "latitude": tlat,
                "longitude": tlon,
                "target_cluster": rec["target_cluster"],
                "source": chosen_source,
                "prompt": rec["prompt"],
                "response": chosen_sft,
            })

            # Construct DPO pair: contrast chosen against worst valid organic rollout
            # so DPO learns geographic reasoning & coordinate precision (not JSON syntax).
            losers = [x for x in valid_organic if (chosen_reward - x["reward"]) >= 300.0]
            if losers:
                worst = min(losers, key=lambda x: x["reward"])
                pw = worst["parsed"]
                rejected_str = format_canonical_cot_json(
                    pw["visual_clues"],
                    pw["reasoning"],
                    pw["selected_cluster"] if pw["selected_cluster"] is not None else 1,
                    pw["country"],
                    pw["coordinates"][0],
                    pw["coordinates"][1],
                    pw["confidence"],
                )
                if rejected_str != chosen_sft:
                    margin = float(chosen_reward - worst["reward"])
                    margins.append(margin)
                    pair_type = (
                        "same_country_coord"
                        if pw["country"] == tcc
                        else ("keep_stability" if rec["category"] == "keep_top1" else "cross_country")
                    )
                    stats[f"dpo_pairs_{pair_type}"] += 1
                    stats["dpo_pairs_total"] += 1
                    if chosen_source == "rejection_sampled":
                        stats["dpo_pairs_chosen_organic"] += 1
                    else:
                        stats["dpo_pairs_chosen_star"] += 1
                    dpo_rows.append({
                        "id": rec["id"],
                        "image": rec["image"],
                        "category": rec["category"],
                        "pair_type": pair_type,
                        "chosen_source": chosen_source,
                        "chosen_dist_km": round(float(chosen_dist), 1),
                        "rejected_dist_km": round(float(worst["dist_km"]), 1),
                        "reward_margin": round(margin, 1),
                        "prompt": rec["prompt"],
                        "chosen": chosen_sft,
                        "rejected": rejected_str,
                    })

    if margins:
        stats["mean_dpo_margin"] = round(float(np.mean(margins)), 1)

    (out_dir / "sft_cot_train.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sft_rows), encoding="utf-8"
    )
    (out_dir / "dpo_cot_pairs.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dpo_rows), encoding="utf-8"
    )
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("\n=== Generated CoT SFT & DPO Dataset Stats ===", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-edge", type=int, default=768)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    train_rows = [json.loads(l) for l in (args.input_dir / "rollout_train_pool.jsonl").read_text(encoding="utf-8").splitlines()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines()]

    # Extract & verify images from bundled ZIPs
    with ZipFile(args.input_dir / "train_subset_imgs.zip") as z:
        for r in train_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"Train SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    with ZipFile(args.input_dir / "author-holdout-eval.zip") as z:
        for r in ah_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"AH SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(train_rows)} train images + {len(ah_rows)} author_holdout images.", flush=True)

    model, processor, rev = load_model(MODEL_ID)
    (args.output_dir / "rollout_config.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "model_revision": rev,
        "n_train_pool": len(train_rows),
        "k_samples": args.k_samples,
        "max_edge": args.max_edge,
        "gpu": torch.cuda.get_device_name(0),
    }, indent=2) + "\n", encoding="utf-8")

    # 1. Baseline greedy evaluation of UNTRAINED Qwen3-VL-8B on Top-8 multi-scale author_holdout_66
    print("\n=== Step 1/3: Untrained Qwen3-VL-8B baseline on Top-8 Multi-Scale author_holdout_66 ===", flush=True)
    ah_out_path = args.output_dir / "base_8b_author_holdout_top8.jsonl"
    if ah_out_path.exists():
        ah_preds = [json.loads(l) for l in ah_out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Loaded cached author_holdout_66 baseline ({len(ah_preds)} rows).", flush=True)
    else:
        ah_preds = []
        ah_bs = 8
        idx = 0
        while idx < len(ah_rows):
            grp = ah_rows[idx:idx + ah_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=args.max_edge) for r in grp]
            prompts = [build_multiscale_cot_prompt(r["kde_distribution"]) for r in grp]
            try:
                raws = [x[0] for x in generate_samples(model, processor, imgs, prompts, num_return_sequences=1, do_sample=False, max_new_tokens=320)]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if ah_bs > 1:
                    ah_bs = max(1, ah_bs // 2)
                    continue
                raise
            for r, raw in zip(grp, raws):
                p = parse_multiscale_json(raw, cc_names)
                coords = p["coordinates"]
                dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
                ah_preds.append({
                    "id": r["id"], "true_country": r["country"],
                    "true_latitude": r["latitude"], "true_longitude": r["longitude"],
                    "guess_country": p["country"], "guess_coordinates": list(coords) if coords else None,
                    "distance_km": dist, "selected_cluster": p["selected_cluster"],
                    "confidence": p["confidence"], "visual_clues": p["visual_clues"],
                    "reasoning": p["reasoning"], "parsed": p["parsed"], "raw": raw,
                })
            idx += len(grp)
        ah_out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ah_preds), encoding="utf-8")
    valid_d = [x["distance_km"] for x in ah_preds if x["distance_km"] is not None]
    parsed_ok = sum(1 for x in ah_preds if x["parsed"])
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in ah_preds)
    print(f"  Base 8B (Top-8 multi-scale) author_holdout_66: parsed={parsed_ok}/66, cc={cc_ok}/66 ({100*cc_ok/66:.1f}%), "
          f"<=25={sum(d<=25 for d in valid_d)}, <=200={sum(d<=200 for d in valid_d)}, "
          f"med={np.median(valid_d):.1f}km, pts={sum(geoscore(d) for d in valid_d)/66:.1f}", flush=True)

    # 2. Blind K rollouts on training pool (with incremental resume + VRAM-aware batching!)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_vram_gb >= 35.0:
        cur_bs = max(args.batch_size, 8)   # A100: ~24-32 sequences
    elif total_vram_gb >= 20.0:
        cur_bs = args.batch_size           # L4: ~16-18 sequences
    else:
        cur_bs = min(args.batch_size, 2)   # T4: ~6-8 sequences
    print(f"\n=== Step 2/3: Generating K={args.k_samples} blind CoT rollouts for {len(train_rows)} training images "
          f"(VRAM={total_vram_gb:.1f} GB, img_batch={cur_bs}, seq_batch={cur_bs * args.k_samples}) ===", flush=True)
    raw_path = args.output_dir / "cot_rollouts_blind.jsonl"
    done = {}
    if raw_path.exists():
        for l in raw_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rec = json.loads(l)
                done[rec["id"]] = rec
        if done:
            print(f"  Resuming blind rollouts from {len(done)}/{len(train_rows)}", flush=True)

    todo = [r for r in train_rows if r["id"] not in done]
    t0 = time.monotonic()
    with raw_path.open("a", encoding="utf-8") as fh:
        i = 0
        while i < len(todo):
            grp = todo[i:i + cur_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=args.max_edge) for r in grp]
            prompts = [build_multiscale_cot_prompt(r["kde_distribution"]) for r in grp]
            try:
                batch_samples = generate_samples(
                    model, processor, imgs, prompts,
                    num_return_sequences=args.k_samples, do_sample=True,
                    temperature=0.75, top_p=0.90, max_new_tokens=320,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    print(f"  CUDA OOM -> reducing batch_size to {cur_bs}", flush=True)
                    continue
                raise

            for r, prompt, raws in zip(grp, prompts, batch_samples):
                rec = dict(r)
                rec["prompt"] = prompt
                rec["samples"] = [{"raw": s} for s in raws]
                done[r["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            i += len(grp)
            if i == len(grp) or i % 40 < cur_bs or i >= len(todo):
                elapsed = time.monotonic() - t0
                rate = i / max(1e-3, elapsed)
                eta = (len(todo) - i) / max(1e-3, rate) / 60.0
                print(f"  Blind rollouts: {len(done):4d}/{len(train_rows)} | {rate:.2f} img/s | ETA={eta:.1f}m", flush=True)

    records = [done[r["id"]] for r in train_rows]

    # 3. STaR hindsight rationalization ONLY for hard examples where no blind rollout succeeded
    need_star = []
    for rec in records:
        tlat, tlon = rec["latitude"], rec["longitude"]
        c1 = rec["kde_distribution"]["clusters"][0]
        d_top1 = haversine_km(tlat, tlon, c1["latitude"], c1["longitude"])
        ok = False
        for s in rec["samples"]:
            p = parse_multiscale_json(s["raw"], cc_names)
            d = haversine_km(tlat, tlon, p["coordinates"][0], p["coordinates"][1]) if p["coordinates"] else None
            if is_organic_success(p, d, rec, d_top1):
                ok = True
                break
        if not ok:
            need_star.append(rec)

    star_bs = 16 if total_vram_gb >= 35.0 else (8 if total_vram_gb >= 20.0 else 4)
    print(f"\n=== Step 3/3: STaR hindsight rationalization for {len(need_star)}/{len(records)} hard examples (bs={star_bs}) ===", flush=True)
    star_path = args.output_dir / "cot_star_rationalizations.jsonl"
    star_done = {}
    if star_path.exists():
        for l in star_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                star_done[x["id"]] = x["star_raw"]
        if star_done:
            print(f"  Resuming STaR from {len(star_done)}/{len(need_star)}", flush=True)

    star_todo = [r for r in need_star if r["id"] not in star_done]
    t1 = time.monotonic()
    with star_path.open("a", encoding="utf-8") as fh:
        idx = 0
        while idx < len(star_todo):
            grp = star_todo[idx:idx + star_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=args.max_edge) for r in grp]
            prompts = [
                build_star_rationalization_prompt(
                    r["kde_distribution"], r["country"], cc_names.get(r["country"], r["country"]),
                    r["latitude"], r["longitude"], r["target_cluster"],
                )
                for r in grp
            ]
            try:
                raws = [x[0] for x in generate_samples(model, processor, imgs, prompts, num_return_sequences=1, do_sample=False, max_new_tokens=320)]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if star_bs > 1:
                    star_bs = max(1, star_bs // 2)
                    print(f"  CUDA OOM in STaR -> reducing batch_size to {star_bs}", flush=True)
                    continue
                raise
            for r, raw in zip(grp, raws):
                star_done[r["id"]] = raw
                fh.write(json.dumps({"id": r["id"], "star_raw": raw}, ensure_ascii=False) + "\n")
            fh.flush()
            idx += len(grp)
            if idx == len(grp) or idx % 40 < star_bs or idx >= len(star_todo):
                rate = idx / max(1e-3, time.monotonic() - t1)
                eta = (len(star_todo) - idx) / max(1e-3, rate) / 60.0
                print(f"  STaR rationalization: {len(star_done):4d}/{len(need_star)} | {rate:.2f} img/s | ETA={eta:.1f}m", flush=True)

    for rec in records:
        if rec["id"] in star_done:
            rec["star_raw"] = star_done[rec["id"]]

    (args.output_dir / "cot_rollouts_full.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8"
    )
    evaluate_and_pack_datasets(records, cc_names, args.output_dir)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted CoT rollout & dataset construction!", flush=True)


if __name__ == "__main__":
    main()
