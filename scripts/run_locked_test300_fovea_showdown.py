"""Locked Zero-Touch `official_test_300` Paired Showdown Worker (Colab L4).

Evaluates strictly head-to-head on identical Pass-1 Hybrid Native Grounding crops
across all 300 locked test images (zero test-set tuning, single-shot evaluation):
  1. Pass 1 — Hybrid Native Grounding (6-Class Step-0 Grounding + Zero-Sky Open-Ended
     Fallback ONLY where 6-Class finds 0 objects; batched bs=8).
  2. Pass 2A-i  — Zero-Training Hybrid Visual Fovea (CoT Prompt; AH66 Champion 3,612.3 / 3,627.5).
  3. Pass 2A-ii — Zero-Training Hybrid Visual Fovea (Direct JSON + Logit Argmax + MBR Expected-GS).
  4. Pass 2B    — Trained Hier-Fovea Ep1 LM LoRA (Direct JSON + Logit Argmax + MBR Expected-GS),
     loaded onto the same base model via PeftModel.from_pretrained (zero reload).
"""
import argparse
import gc
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from peft import PeftModel

from multiscale_vlm_utils import (
    build_direct_fovea_prompt,
    build_multiscale_cot_prompt,
    geoscore,
    haversine_km,
    parse_multiscale_json,
)
from run_dir2_harvest_crops import (
    build_open_harvest_prompt,
    parse_and_filter_local_boxes,
)
from run_step0_grounding_attention import (
    CUE_PRIORITY,
    GROUNDING_PROMPT,
    crop_cue,
    generate_batch_single_img,
    generate_one_multi_img,
    load_clean_image,
    load_model,
    parse_grounding_boxes,
)
from train_hier_fovea_lora import (
    get_cluster_digit_ids,
    pick_mbr_cluster,
    score_cluster_probs_multi_img,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

CAMERA_META_PAT = re.compile(
    r"\b(dashboard|hood|windshield|windscreen|steering|interior|mirror|wiper|sun\s*visor|camera|dashcam)\b",
    re.I,
)


def load_done_map(path: Path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            out[obj["id"]] = obj
    return out


def append_jsonl(path: Path, rows):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.inference_mode()
def generate_batch_multi_img(model, processor, batch_imgs, batch_blocks, max_new_tokens=240):
    """Batched multi-image greedy generation with safe automatic bs=1 fallback."""
    if len(batch_imgs) == 1:
        return [generate_one_multi_img(model, processor, batch_imgs[0], batch_blocks[0], max_new_tokens=max_new_tokens)]
    try:
        chat_texts = [
            processor.apply_chat_template(
                [{"role": "user", "content": blks}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for blks in batch_blocks
        ]
        flat_imgs = [im for sub in batch_imgs for im in sub]
        inputs = processor(text=chat_texts, images=flat_imgs, padding=True, return_tensors="pt").to("cuda")
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            use_cache=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        return processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
    except Exception as e:
        print(f"  [fallback bs=1 due to {type(e).__name__}: {e}]", flush=True)
        torch.cuda.empty_cache()
        return [
            generate_one_multi_img(model, processor, ims, blks, max_new_tokens=max_new_tokens)
            for ims, blks in zip(batch_imgs, batch_blocks)
        ]


def build_fovea_inputs(full_im, boxes, dist_info, mode: str):
    imgs = [full_im]
    blocks = [{"type": "image"}]
    notes = []
    for ci, b in enumerate(boxes[:2], 1):
        imgs.append(crop_cue(full_im, b["bbox_1000"], pad_frac=0.10))
        blocks.append({"type": "image"})
        extra = b.get("why") or b.get("note") or ""
        notes.append(f"Zoomed Detail #{ci}: `{b['label']}` ({extra})" if extra else f"Zoomed Detail #{ci}: `{b['label']}`")
    if mode == "cot":
        kde_p = build_multiscale_cot_prompt(dist_info)
        prefix = (
            "Image 1 is the full street-level photo. Additional zoomed-in crops of key diagnostic details are attached:\n"
            + "\n".join(f"  - {n}" for n in notes)
            + "\nInspect these zoomed details closely before deciding.\n\n"
        ) if notes else ""
        blocks.append({"type": "text", "text": prefix + kde_p})
    else:
        blocks.append({"type": "text", "text": build_direct_fovea_prompt(dist_info, notes)})
    return imgs, blocks


def summarize_preds(preds, d_key="distance_km", c_key="guess_country"):
    vd = [x[d_key] for x in preds if x[d_key] is not None]
    cc_ok = sum(x[c_key] == x["true_country"] for x in preds)
    return {
        "n": len(preds),
        "cc": cc_ok,
        "cc_pct": round(100.0 * cc_ok / max(1, len(preds)), 2),
        "le25": int(sum(d <= 25.0 for d in vd)),
        "le200": int(sum(d <= 200.0 for d in vd)),
        "med_km": round(float(np.median(vd)), 1) if vd else None,
        "geoscore": round(float(sum(geoscore(d) for d in vd) / max(1, len(preds))), 1),
    }


def run_pass2_eval(
    model,
    processor,
    test_rows,
    grounding_map,
    img_dir: Path,
    cc_names: dict,
    digit_ids,
    compute_dtype,
    out_path: Path,
    mode: str,
    tag: str,
    batch_size: int = 4,
    score_logits: bool = False,
):
    model.eval()
    processor.tokenizer.padding_side = "left"
    done = load_done_map(out_path)
    todo = [r for r in test_rows if r["id"] not in done]
    print(f"\n=== [{tag}] Running {mode} on {len(todo)}/{len(test_rows)} test_300 images (bs={batch_size}, logits={score_logits}) ===", flush=True)
    t0 = time.monotonic()

    # Group pending rows by crop count (0, 1, 2) so batched multi-image padding stays tight & fast
    by_k = {}
    for r in todo:
        k = min(2, len(grounding_map.get(r["id"], {}).get("boxes", [])))
        by_k.setdefault(k, []).append(r)

    max_toks = 260 if mode == "cot" else 96
    processed = 0
    for k_crops in sorted(by_k.keys()):
        group = by_k[k_crops]
        for i in range(0, len(group), batch_size):
            batch_r = group[i:i + batch_size]
            b_imgs, b_blocks, b_boxes = [], [], []
            for r in batch_r:
                full_im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
                boxes = grounding_map.get(r["id"], {}).get("boxes", [])[:2]
                imgs, blocks = build_fovea_inputs(full_im, boxes, r["kde_distribution"], mode=mode)
                b_imgs.append(imgs)
                b_blocks.append(blocks)
                b_boxes.append(boxes)

            with torch.autocast("cuda", dtype=compute_dtype):
                raws = generate_batch_multi_img(model, processor, b_imgs, b_blocks, max_new_tokens=max_toks)

            batch_out = []
            for r, imgs, blocks, boxes, raw in zip(batch_r, b_imgs, b_blocks, b_boxes, raws):
                p = parse_multiscale_json(raw, cc_names)
                coords = p["coordinates"]
                dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
                rec = {
                    "id": r["id"],
                    "true_country": r["country"],
                    "true_latitude": r["latitude"],
                    "true_longitude": r["longitude"],
                    "grounding_source": grounding_map.get(r["id"], {}).get("source", "none"),
                    "crop_labels": [b["label"] for b in boxes],
                    "guess_country": p["country"],
                    "guess_coordinates": list(coords) if coords else None,
                    "distance_km": dist,
                    "selected_cluster": p["selected_cluster"],
                    "confidence": p["confidence"],
                    "parsed": p["parsed"],
                    "raw": raw,
                }
                if score_logits:
                    cls = r["kde_distribution"]["clusters"]
                    cl_probs = score_cluster_probs_multi_img(model, processor, imgs, blocks, len(cls), digit_ids, compute_dtype)
                    argmax_j = int(np.argmax(cl_probs))
                    mbr_j, mbr_exp = pick_mbr_cluster(cls, cl_probs)
                    rec.update({
                        "cluster_probs": [round(x, 5) for x in cl_probs],
                        "argmax_cluster": argmax_j + 1,
                        "argmax_country": cls[argmax_j]["country"],
                        "argmax_distance_km": haversine_km(r["latitude"], r["longitude"], cls[argmax_j]["latitude"], cls[argmax_j]["longitude"]),
                        "mbr_cluster": mbr_j + 1,
                        "mbr_country": cls[mbr_j]["country"],
                        "mbr_distance_km": haversine_km(r["latitude"], r["longitude"], cls[mbr_j]["latitude"], cls[mbr_j]["longitude"]),
                        "mbr_expected_gs": round(5000.0 * mbr_exp, 1),
                    })
                batch_out.append(rec)
                done[r["id"]] = rec
            append_jsonl(out_path, batch_out)
            processed += len(batch_r)
            if processed % 40 < batch_size or processed == len(todo):
                print(f"  [{tag}] {processed}/{len(todo)} done ({(time.monotonic()-t0)/60:.1f}m)", flush=True)

    ordered = [done[r["id"]] for r in test_rows]
    summary = {
        "greedy": {
            **summarize_preds(ordered, "distance_km", "guess_country"),
            "parsed": sum(x["parsed"] for x in ordered),
            "sel_top1": sum(1 for x in ordered if x["selected_cluster"] == 1),
        }
    }
    if score_logits:
        summary["logit_argmax"] = {
            **summarize_preds(ordered, "argmax_distance_km", "argmax_country"),
            "sel_top1": sum(1 for x in ordered if x["argmax_cluster"] == 1),
        }
        summary["mbr_expected_gs"] = {
            **summarize_preds(ordered, "mbr_distance_km", "mbr_country"),
            "sel_top1": sum(1 for x in ordered if x["mbr_cluster"] == 1),
        }
    print(f"  [{tag}] SUMMARY: {json.dumps(summary)}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ground-bs", type=int, default=8)
    parser.add_argument("--eval-bs", type=int, default=4)
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
                    raise ValueError(f"test_300 SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(test_rows)} locked official_test_300 images.", flush=True)

    model, processor, rev = load_model(MODEL_ID)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    digit_ids = get_cluster_digit_ids(processor.tokenizer)

    # =========================================================================
    # PASS 1: Hybrid Native Grounding on official_test_300 (6-Class + Open Fallback)
    # =========================================================================
    g_path = args.output_dir / "test300_hybrid_grounding.jsonl"
    g_done = load_done_map(g_path)
    g_todo = [r for r in test_rows if r["id"] not in g_done]
    if g_todo:
        print(f"\n=== Pass 1A: Step-0 6-Class Grounding on {len(g_todo)} test_300 images (bs={args.ground_bs}) ===", flush=True)
        t0 = time.monotonic()
        g6_raw_map = {}
        for i in range(0, len(g_todo), args.ground_bs):
            batch_r = g_todo[i:i + args.ground_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768) for r in batch_r]
            prompts = [GROUNDING_PROMPT for _ in batch_r]
            with torch.autocast("cuda", dtype=compute_dtype):
                raws = generate_batch_single_img(model, processor, imgs, prompts, max_new_tokens=240)
            for r, im, raw in zip(batch_r, imgs, raws):
                boxes = parse_grounding_boxes(raw, im.size[0], im.size[1])
                boxes.sort(key=lambda b: (CUE_PRIORITY.get(b["label"], 99), -((b["bbox_1000"][2] - b["bbox_1000"][0]) * (b["bbox_1000"][3] - b["bbox_1000"][1]))))
                g6_raw_map[r["id"]] = (boxes, raw)
            if (i + len(batch_r)) % 80 < args.ground_bs or (i + len(batch_r)) == len(g_todo):
                print(f"  Pass 1A: {i + len(batch_r)}/{len(g_todo)} ({(time.monotonic()-t0)/60:.1f}m)", flush=True)

        need_open = [r for r in g_todo if not g6_raw_map[r["id"]][0]]
        print(f"\n=== Pass 1B: Zero-Sky Open-Ended Fallback on {len(need_open)}/{len(g_todo)} images with 0 6-class boxes ===", flush=True)
        open_raw_map = {}
        t1 = time.monotonic()
        for i in range(0, len(need_open), args.ground_bs):
            batch_r = need_open[i:i + args.ground_bs]
            imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768) for r in batch_r]
            prompts = []
            for r in batch_r:
                cands = []
                for cl in r["kde_distribution"]["clusters"]:
                    cc = cl["country"]
                    if cc not in cands:
                        cands.append(cc)
                cnames = [f"{cc_names.get(c, c)} ({c})" for c in cands[:5]]
                prompts.append(build_open_harvest_prompt(cnames))
            with torch.autocast("cuda", dtype=compute_dtype):
                raws = generate_batch_single_img(model, processor, imgs, prompts, max_new_tokens=240)
            for r, raw in zip(batch_r, raws):
                boxes = [b for b in parse_and_filter_local_boxes(raw) if not CAMERA_META_PAT.search(b.get("label", ""))]
                open_raw_map[r["id"]] = (boxes, raw)
        print(f"  Pass 1B finished in {(time.monotonic()-t1)/60:.1f}m", flush=True)

        new_g_rows = []
        for r in g_todo:
            b6, raw6 = g6_raw_map[r["id"]]
            if b6:
                rec = {"id": r["id"], "source": "g6", "boxes": b6[:4], "raw_g6": raw6}
            else:
                bo, rawo = open_raw_map.get(r["id"], ([], ""))
                rec = {"id": r["id"], "source": "open_fallback" if bo else "none", "boxes": bo[:4], "raw_g6": raw6, "raw_open": rawo}
            g_done[r["id"]] = rec
            new_g_rows.append(rec)
        append_jsonl(g_path, new_g_rows)

    src_counts = Counter(g_done[r["id"]]["source"] for r in test_rows)
    print(f"Pass 1 Hybrid Grounding complete: {dict(src_counts)}", flush=True)
    torch.cuda.empty_cache()

    # =========================================================================
    # PASS 2A: Zero-Training Base 8B Hybrid Visual Fovea (CoT + Direct/Logit/MBR)
    # =========================================================================
    base_cot_sum = run_pass2_eval(
        model, processor, test_rows, g_done, img_dir, cc_names, digit_ids, compute_dtype,
        args.output_dir / "test300_base_hybrid_fovea_cot.jsonl",
        mode="cot", tag="base_hybrid_cot", batch_size=args.eval_bs, score_logits=False,
    )
    torch.cuda.empty_cache()

    base_dir_sum = run_pass2_eval(
        model, processor, test_rows, g_done, img_dir, cc_names, digit_ids, compute_dtype,
        args.output_dir / "test300_base_hybrid_fovea_direct.jsonl",
        mode="direct", tag="base_hybrid_direct", batch_size=args.eval_bs, score_logits=True,
    )
    torch.cuda.empty_cache()

    # =========================================================================
    # PASS 2B: Trained Hier-Fovea Ep1 LM LoRA (zero reload via PeftModel!)
    # =========================================================================
    adapter_dir = args.input_dir / "hier_ep1_adapter"
    print(f"\nAttaching Hier-Fovea Ep1 LoRA adapter from {adapter_dir}...", flush=True)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
    hier_dir_sum = run_pass2_eval(
        peft_model, processor, test_rows, g_done, img_dir, cc_names, digit_ids, compute_dtype,
        args.output_dir / "test300_hier_ep1_hybrid_fovea_direct.jsonl",
        mode="direct", tag="hier_ep1_hybrid_direct", batch_size=args.eval_bs, score_logits=True,
    )

    report = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "grounding_sources": dict(src_counts),
        "base_hybrid_cot": base_cot_sum,
        "base_hybrid_direct": base_dir_sum,
        "hier_ep1_hybrid_direct": hier_dir_sum,
    }
    (args.output_dir / "test300_fovea_showdown_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del peft_model, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted Locked official_test_300 Paired Visual Fovea Showdown!", flush=True)


if __name__ == "__main__":
    main()
