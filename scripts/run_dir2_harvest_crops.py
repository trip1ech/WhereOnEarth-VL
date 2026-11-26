"""Direction 2 — Phase 1 Worker (Colab GPU):
  1. Harvests diverse, category-agnostic, ZERO-SKY local physical clue bounding boxes
     across ~900 country-balanced creator-disjoint opt_train images using batched
     Qwen/Qwen3-VL-8B-Instruct native Open-Ended Diagnostic Grounding (incremental resume).
  2. Runs TWO exact pre-training baselines on author_holdout_66 for apples-to-apples
     before/after comparison against Phase 2 (Unfrozen Projector + LM LoRA):
       (a) Isolated Local-Cue Shuffled Hard-Neighbor Probe (ZERO background, ZERO KDE prior,
           shuffled A..E options — measures exact pre-training local object <-> country accuracy,
           including non-text physical cues like roofs/fences/guardrails/poles/bollards/markings).
       (b) Image-First + Open-Ended Diagnostic Visual Fovea (Top-8 Multi-Scale KDE).
"""
import argparse
import gc
import hashlib
import json
import random
import re
import time
from collections import Counter
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
    geoscore,
    haversine_km,
    normalize_country_code,
    parse_multiscale_json,
)
from run_step0_grounding_attention import (
    crop_cue,
    generate_batch_single_img,
    generate_one_multi_img,
    load_clean_image,
    load_model,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SKY_BLOCK_PAT = re.compile(r"\b(sky|cloud|clouds|sun|sunlight|horizon|weather|overcast|atmosphere)\b", re.I)
TEXT_SIGN_PAT = re.compile(r"\b(text|sign|signage|billboard|banner|shop|store|plate|license|letter|script|street name)\b", re.I)


def build_open_harvest_prompt(cand_country_names) -> str:
    cand_str = ", ".join(cand_country_names[:5])
    return (
        f"Candidate countries for this street-level photo: {cand_str}.\n"
        "Locate 2 to 3 compact, physically concrete local regions of ANY object/structure type that best "
        "distinguish between these countries (e.g., building/roof style, fence/wall, guardrail, utility pole, "
        "streetlight, road sign, shopfront/text sign, road marking/curb, bollard, license plate, vehicle, "
        "or distinctive roadside tree/soil). NEVER select empty sky or clouds.\n"
        "Return ONLY a JSON array of 2-3 objects with normalized [0, 1000] integer boxes [x1, y1, x2, y2]:\n"
        '[{"label": "<short concrete cue name>", "bbox_2d": [x1, y1, x2, y2], "why": "<3-8 word detail>"}]'
    )


def parse_and_filter_local_boxes(raw_text: str):
    """Parse Open-Ended Grounding boxes and strictly filter out sky/clouds or overly global boxes."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    arr = None
    m = re.search(r"\[.*\]", cleaned, re.S)
    if m:
        try:
            cand = json.loads(m.group(0))
            if isinstance(cand, list):
                arr = cand
        except ValueError:
            arr = None
    if arr is None:
        arr = []
        dec = json.JSONDecoder()
        i = 0
        while i < len(cleaned):
            if cleaned[i] == "{":
                try:
                    obj, end = dec.raw_decode(cleaned[i:])
                    if isinstance(obj, dict) and ("bbox_2d" in obj or "bbox" in obj):
                        arr.append(obj)
                    i += end
                    continue
                except ValueError:
                    pass
            i += 1

    kept = []
    for item in arr[:5]:
        if not isinstance(item, dict):
            continue
        bb = item.get("bbox_2d") or item.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except (TypeError, ValueError):
            continue
        if max(x1, y1, x2, y2) <= 1.05:
            x1, y1, x2, y2 = x1 * 1000.0, y1 * 1000.0, x2 * 1000.0, y2 * 1000.0
        x1, x2 = max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))
        y1, y2 = max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))
        bw, bh = x2 - x1, y2 - y1
        if bw < 20.0 or bh < 20.0:
            continue
        area_frac = (bw * bh) / 1e6
        if area_frac < 0.004 or area_frac > 0.55:
            continue
        lbl = str(item.get("label", "local clue")).strip().lower()[:36]
        why = str(item.get("why", "")).strip()[:80]
        # Reject sky/cloud labels or pure upper-sky slices
        if SKY_BLOCK_PAT.search(lbl):
            continue
        cy = 0.5 * (y1 + y2)
        if cy < 200.0 and bh < 260.0:
            continue
        is_text_like = bool(TEXT_SIGN_PAT.search(lbl) or TEXT_SIGN_PAT.search(why))
        kept.append({
            "label": lbl,
            "bbox_1000": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "area_frac": round(area_frac, 4),
            "is_text_like": is_text_like,
            "why": why,
        })
        if len(kept) >= 3:
            break
    return kept


def build_shuffled_mcq_prompt(label: str, options_list, cc_names: dict) -> str:
    """Build the anti-cheat Isolated Local-Cue Shuffled Hard-Neighbor prompt (zero background, zero KDE)."""
    letters = "ABCDEF"
    lines = [
        f"You are inspecting ONLY a cropped local street-level clue (`{label}`) with all surrounding scenery removed.",
        "Which of these candidate countries best matches this local clue's physical style, design, or text?",
    ]
    for i, cc in enumerate(options_list):
        lines.append(f"  {letters[i]}. {cc_names.get(cc, cc)} ({cc})")
    lines.append(
        'Respond with ONLY one JSON object: {"option": "<letter>", "country": "<2-letter ISO code>"}'
    )
    return "\n".join(lines)


def parse_mcq_json(raw: str, options_list, cc_names: dict):
    letters = "ABCDEF"[:len(options_list)]
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    obj = {}
    m = re.search(r"\{.*\}", cleaned, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = {}
    opt = str(obj.get("option", "")).strip().upper()
    cc = normalize_country_code(obj.get("country"), cc_names)
    if len(opt) == 1 and opt in letters:
        cc_from_opt = options_list[letters.index(opt)]
        if cc is None or cc not in options_list:
            cc = cc_from_opt
    elif cc in options_list:
        opt = letters[options_list.index(cc)]
    else:
        opt_m = re.search(r'"option"\s*:\s*"([A-F])"', cleaned, re.I)
        if opt_m and opt_m.group(1).upper() in letters:
            opt = opt_m.group(1).upper()
            cc = options_list[letters.index(opt)]
    return opt if opt in letters else None, cc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    harvest_rows = [json.loads(l) for l in (args.input_dir / "harvest_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_open_g = {
        r["id"]: r
        for r in (json.loads(l) for l in (args.input_dir / "open_grounding_ah66.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }

    with ZipFile(args.input_dir / "harvest_imgs.zip") as z:
        for r in harvest_rows:
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
    print(f"Verified {len(harvest_rows)} opt_train harvest images + {len(ah_rows)} author_holdout_66 images.", flush=True)

    model, processor, rev = load_model(MODEL_ID)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bs = max(args.batch_size, 12) if total_vram_gb >= 35.0 else (args.batch_size if total_vram_gb >= 20.0 else 4)

    # =========================================================================
    # STEP 1/3: Pre-Training Baseline A — Isolated Local-Cue Shuffled MCQ on author_holdout_66
    # =========================================================================
    print(f"\n=== Step 1/3: Pre-Training Baseline — Isolated Local-Cue Shuffled Hard-Neighbor MCQ on author_holdout_66 (bs={bs}) ===", flush=True)
    mcq_path = args.output_dir / "base_8b_ah66_isolated_mcq.jsonl"
    if mcq_path.exists():
        mcq_results = [json.loads(l) for l in mcq_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Loaded cached isolated MCQ baseline ({len(mcq_results)} crops).", flush=True)
    else:
        rng = random.Random(20261125)
        mcq_items = []
        for r in ah_rows:
            im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
            raw_boxes = ah_open_g.get(r["id"], {}).get("boxes", [])
            # Apply the exact same anti-sky filter
            filt_boxes = []
            for b in raw_boxes:
                x1, y1, x2, y2 = b["bbox_1000"]
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh / 1e6
                if SKY_BLOCK_PAT.search(b["label"]) or area > 0.55 or (0.5 * (y1 + y2) < 200.0 and bh < 260.0):
                    continue
                is_txt = bool(TEXT_SIGN_PAT.search(b["label"]) or TEXT_SIGN_PAT.search(b.get("why", "")))
                filt_boxes.append({**b, "area_frac": round(area, 4), "is_text_like": is_txt})
            if not filt_boxes:
                filt_boxes = [{**b, "area_frac": round((b["bbox_1000"][2]-b["bbox_1000"][0])*(b["bbox_1000"][3]-b["bbox_1000"][1])/1e6, 4), "is_text_like": bool(TEXT_SIGN_PAT.search(b["label"]))} for b in raw_boxes[:2]]

            # Build 5-option hard-neighbor country set containing true country + 4 hardest StreetCLIP neighbors
            seen_c = []
            for cl in r["kde_distribution"]["clusters"]:
                if cl["country"] not in seen_c:
                    seen_c.append(cl["country"])
            for tx in r["kde_distribution"].get("top_text_countries", []):
                if tx["country"] not in seen_c:
                    seen_c.append(tx["country"])
            distractors = [c for c in seen_c if c != r["country"]][:4]
            opts = [r["country"]] + distractors
            for bi, b in enumerate(filt_boxes[:2]):
                shuf = list(opts)
                rng.shuffle(shuf)
                crop_im = crop_cue(im, b["bbox_1000"], pad_frac=0.08)
                prompt = build_shuffled_mcq_prompt(b["label"], shuf, cc_names)
                mcq_items.append((r, bi, b, shuf, crop_im, prompt))

        mcq_results = []
        idx = 0
        cur_bs = bs
        while idx < len(mcq_items):
            grp = mcq_items[idx:idx + cur_bs]
            try:
                raws = generate_batch_single_img(model, processor, [x[4] for x in grp], [x[5] for x in grp], max_new_tokens=40)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    continue
                raise
            for (r, bi, b, shuf, _, prompt), raw in zip(grp, raws):
                opt, pred_cc = parse_mcq_json(raw, shuf, cc_names)
                mcq_results.append({
                    "id": r["id"],
                    "crop_idx": bi,
                    "true_country": r["country"],
                    "label": b["label"],
                    "bbox_1000": b["bbox_1000"],
                    "is_text_like": b["is_text_like"],
                    "options": shuf,
                    "pred_option": opt,
                    "pred_country": pred_cc,
                    "correct": pred_cc == r["country"],
                    "raw": raw,
                })
            idx += len(grp)
        mcq_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mcq_results), encoding="utf-8")

    tot_ok = sum(x["correct"] for x in mcq_results)
    nontxt = [x for x in mcq_results if not x["is_text_like"]]
    txt = [x for x in mcq_results if x["is_text_like"]]
    print(
        f"  Base 8B Isolated Local-Cue Shuffled MCQ (chance=20%): "
        f"ALL={tot_ok}/{len(mcq_results)} ({100*tot_ok/max(1,len(mcq_results)):.1f}%) | "
        f"Non-Text={sum(x['correct'] for x in nontxt)}/{len(nontxt)} ({100*sum(x['correct'] for x in nontxt)/max(1,len(nontxt)):.1f}%) | "
        f"Text/Sign/Plate={sum(x['correct'] for x in txt)}/{len(txt)} ({100*sum(x['correct'] for x in txt)/max(1,len(txt)):.1f}%)",
        flush=True,
    )

    # =========================================================================
    # STEP 2/3: Pre-Training Baseline B — Image-First + Open-Ended Fovea on author_holdout_66
    # =========================================================================
    print("\n=== Step 2/3: Pre-Training Baseline — Image-First + Open-Ended Fovea on author_holdout_66 ===", flush=True)
    fov_path = args.output_dir / "base_8b_ah66_imgfirst_open_fovea.jsonl"
    if fov_path.exists():
        fov_preds = [json.loads(l) for l in fov_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"  Loaded cached Image-First Open-Fovea baseline ({len(fov_preds)} rows).", flush=True)
    else:
        fov_preds = []
        for r in ah_rows:
            full_im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
            o_boxes = ah_open_g.get(r["id"], {}).get("boxes", [])[:2]
            imgs = [full_im]
            blocks = [{"type": "image"}]
            notes = []
            for ci, b in enumerate(o_boxes, 1):
                imgs.append(crop_cue(full_im, b["bbox_1000"], pad_frac=0.10))
                blocks.append({"type": "image"})
                notes.append(f"Zoomed Detail #{ci}: `{b['label']}` ({b.get('why', '')})")
            kde_p = build_multiscale_cot_prompt(r["kde_distribution"])
            prefix = (
                "Image 1 is the full street-level photo. Additional zoomed-in crops of key diagnostic details are attached:\n"
                + "\n".join(f"  - {n}" for n in notes)
                + "\nInspect these zoomed details closely before deciding.\n\n"
            ) if notes else ""
            blocks.append({"type": "text", "text": prefix + kde_p})
            raw = generate_one_multi_img(model, processor, imgs, blocks, max_new_tokens=320)
            p = parse_multiscale_json(raw, cc_names)
            coords = p["coordinates"]
            dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
            fov_preds.append({
                "id": r["id"], "true_country": r["country"],
                "true_latitude": r["latitude"], "true_longitude": r["longitude"],
                "crop_labels": [b["label"] for b in o_boxes],
                "guess_country": p["country"], "guess_coordinates": list(coords) if coords else None,
                "distance_km": dist, "selected_cluster": p["selected_cluster"],
                "confidence": p["confidence"], "visual_clues": p["visual_clues"],
                "reasoning": p["reasoning"], "parsed": p["parsed"], "raw": raw,
            })
        fov_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in fov_preds), encoding="utf-8")
    valid_d = [x["distance_km"] for x in fov_preds if x["distance_km"] is not None]
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in fov_preds)
    print(
        f"  Base 8B Image-First + Open-Ended Fovea author_holdout_66: parsed={sum(x['parsed'] for x in fov_preds)}/66, "
        f"cc={cc_ok}/66 ({100*cc_ok/66:.1f}%), <=25={sum(d<=25 for d in valid_d)}, <=200={sum(d<=200 for d in valid_d)}, "
        f"med={np.median(valid_d):.1f}km, pts={sum(geoscore(d) for d in valid_d)/66:.1f}",
        flush=True,
    )

    # =========================================================================
    # STEP 3/3: Batched Open-Ended Zero-Sky Cue Harvesting on opt_train (incremental resume!)
    # =========================================================================
    print(f"\n=== Step 3/3: Harvesting Open-Ended Zero-Sky Local Cues on {len(harvest_rows)} opt_train images (bs={bs}) ===", flush=True)
    harv_path = args.output_dir / "harvested_opt_train_cues.jsonl"
    done = {}
    if harv_path.exists():
        for l in harv_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rec = json.loads(l)
                done[rec["id"]] = rec
        if done:
            print(f"  Resuming harvest from {len(done)}/{len(harvest_rows)}", flush=True)

    todo = [r for r in harvest_rows if r["id"] not in done]
    t0 = time.monotonic()
    cur_bs = bs
    with harv_path.open("a", encoding="utf-8") as fh:
        idx = 0
        while idx < len(todo):
            grp = todo[idx:idx + cur_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=512) for r in grp]
            prompts = [build_open_harvest_prompt(r["cand_country_names"]) for r in grp]
            try:
                raws = generate_batch_single_img(model, processor, imgs, prompts, max_new_tokens=170)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    print(f"  CUDA OOM -> reducing batch_size to {cur_bs}", flush=True)
                    continue
                raise
            for r, raw in zip(grp, raws):
                boxes = parse_and_filter_local_boxes(raw)
                rec = {
                    "id": r["id"],
                    "image": r["image"],
                    "country": r["country"],
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                    "cand_countries": r["cand_countries"],
                    "boxes": boxes,
                    "raw": raw,
                }
                done[r["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            idx += len(grp)
            if idx == len(grp) or idx % 80 < cur_bs or idx >= len(todo):
                rate = idx / max(1e-3, time.monotonic() - t0)
                eta = (len(todo) - idx) / max(1e-3, rate) / 60.0
                print(f"  Harvesting: {len(done):4d}/{len(harvest_rows)} | {rate:.2f} img/s | ETA={eta:.1f}m", flush=True)

    all_harv = [done[r["id"]] for r in harvest_rows]
    all_boxes = [b for rec in all_harv for b in rec["boxes"]]
    lbl_cnt = Counter(b["label"] for b in all_boxes)
    summary = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "base_ah66_isolated_mcq": {
            "n_crops": len(mcq_results),
            "all_acc_pct": round(100.0 * tot_ok / max(1, len(mcq_results)), 2),
            "nontxt_n": len(nontxt),
            "nontxt_acc_pct": round(100.0 * sum(x["correct"] for x in nontxt) / max(1, len(nontxt)), 2),
            "txt_n": len(txt),
            "txt_acc_pct": round(100.0 * sum(x["correct"] for x in txt) / max(1, len(txt)), 2),
        },
        "base_ah66_imgfirst_open_fovea": {
            "n": len(fov_preds),
            "parsed": sum(1 for x in fov_preds if x["parsed"]),
            "country_correct": cc_ok,
            "country_acc_pct": round(100.0 * cc_ok / len(fov_preds), 2),
            "le_25km": int(sum(d <= 25.0 for d in valid_d)),
            "le_200km": int(sum(d <= 200.0 for d in valid_d)),
            "median_km": round(float(np.median(valid_d)), 1),
            "geoscore": round(float(sum(geoscore(d) for d in valid_d) / len(fov_preds)), 1),
        },
        "harvest_stats": {
            "n_images": len(all_harv),
            "images_with_boxes": sum(1 for x in all_harv if x["boxes"]),
            "total_zero_sky_boxes": len(all_boxes),
            "nontxt_boxes": sum(1 for b in all_boxes if not b["is_text_like"]),
            "txt_boxes": sum(1 for b in all_boxes if b["is_text_like"]),
            "top_20_labels": lbl_cnt.most_common(20),
        },
    }
    (args.output_dir / "dir2_phase1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== Direction 2 Phase 1 Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted Direction 2 Phase 1 cue harvesting & pre-training baselines!", flush=True)


if __name__ == "__main__":
    main()
