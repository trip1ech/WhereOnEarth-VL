"""Direction 2 — Phase 2 Worker (Colab GPU):
Joint Two-Task Training with UNFROZEN Vision-Language Mergers (`visual.merger` +
`visual.deepstack_merger_list`) + LM LoRA on Qwen/Qwen3-VL-8B-Instruct:

  Task 1 (~70%): Zero-Sky Isolated Local-Cue -> Shuffled Hard-Neighbor Country (A..E)
                 (zero background, zero KDE percentages, uniform 20% option slots ->
                  physically impossible to cheat via ViT sky registers or Rank-1 shortcuts!)
  Task 2 (~30%): Image-First Visual Fovea Localization ([Full Image + 2 Local Crops] + Top-8 KDE)
                 with natural-prior calibrated ground-truth targets (zero STaR prose hallucination).

Evaluates on author_holdout_66 across 4 exact before-vs-after mechanistic metrics:
  1. Isolated Local-Cue Shuffled Hard-Neighbor Accuracy (ALL, Non-Text physical cues, Text/Sign/Plate)
     + calibrated cue logit probabilities for Bayes fusion with StreetCLIP KDE!
  2. End-to-End Image-First Open-Ended Visual Fovea Top-8 Localization (direct & CoT prompts)
  3. Eager LM Cross-Attention Heatmaps AFTER training (before-vs-after attention shift on local cues!)
"""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from multiscale_vlm_utils import (
    build_direct_fovea_prompt,
    build_multiscale_cot_prompt,
    build_shuffled_mcq_prompt,
    geoscore,
    haversine_km,
    parse_multiscale_json,
)
from run_dir2_harvest_crops import (
    SKY_BLOCK_PAT,
    TEXT_SIGN_PAT,
    parse_mcq_json,
)
from run_step0_grounding_attention import (
    compute_box_attention_stats,
    crop_cue,
    extract_visual_attention,
    generate_batch_single_img,
    generate_one_multi_img,
    load_clean_image,
    set_lm_attn_impl,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def set_peft_lm_attn_impl(model, impl: str):
    """Switch LM attention implementation reliably even when wrapped by PEFT."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    set_lm_attn_impl(base, impl)
    for obj in (model, base, getattr(base, "model", None), getattr(getattr(base, "model", None), "language_model", None)):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            for c in (cfg, getattr(cfg, "text_config", None)):
                if c is not None:
                    try:
                        c._attn_implementation = impl
                    except Exception:
                        pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jitter_bbox(bbox_1000, rng: random.Random, frac: float = 0.04):
    x1, y1, x2, y2 = bbox_1000
    bw, bh = x2 - x1, y2 - y1
    dx1 = rng.uniform(-frac, frac) * bw
    dy1 = rng.uniform(-frac, frac) * bh
    dx2 = rng.uniform(-frac, frac) * bw
    dy2 = rng.uniform(-frac, frac) * bh
    nx1 = max(0.0, min(980.0, x1 + dx1))
    ny1 = max(0.0, min(980.0, y1 + dy1))
    nx2 = max(nx1 + 20.0, min(1000.0, x2 + dx2))
    ny2 = max(ny1 + 20.0, min(1000.0, y2 + dy2))
    return [nx1, ny1, nx2, ny2]


def mild_color_jitter(im: Image.Image, rng: random.Random) -> Image.Image:
    # NEVER flip horizontally (would invert driving side and mirror sign text!)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.92, 1.08))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.92, 1.08))
    return im


def build_supervised_multi_inputs(processor, images, content_blocks, response_text: str):
    user_chat = processor.apply_chat_template(
        [{"role": "user", "content": content_blocks}],
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = processor.tokenizer.eos_token or "<|im_end|>"
    full_chat = user_chat + response_text + eos
    full_inputs = processor(text=[full_chat], images=images, return_tensors="pt")
    f_ids = full_inputs["input_ids"][0].tolist()

    marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    prefix_len = None
    for i in range(len(f_ids) - len(marker), -1, -1):
        if f_ids[i:i + len(marker)] == marker:
            prefix_len = i + len(marker)
            break
    if prefix_len is None:
        p_ids = processor(text=[user_chat], images=images, return_tensors="pt")["input_ids"][0].tolist()
        prefix_len = 0
        lim = min(len(p_ids), len(f_ids))
        while prefix_len < lim and p_ids[prefix_len] == f_ids[prefix_len]:
            prefix_len += 1

    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100
    if int((labels != -100).sum().item()) <= 0:
        raise ValueError("Empty response token mask")
    return full_inputs, labels


def load_unfrozen_merger_qlora_model(model_id: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    # CRITICAL: Keep visual.merger & deepstack_merger_list in native floating point so they can be UNFROZEN!
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["merger", "deepstack_merger_list", "lm_head"],
    )
    print(f"Loading {model_id} (4-bit NF4 LM + native FP mergers, compute={compute_dtype}) on {torch.cuda.get_device_name(0)}...", flush=True)
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    rev = str(getattr(model.config, "_commit_hash", "unknown"))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Attach LoRA to LM linear projections only (ViT backbone stays frozen)
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lm_targets = sorted({
        name for name, _ in model.named_modules()
        if any(name.endswith(f".{s}") for s in target_suffixes)
        and "visual" not in name
        and "vision" not in name
    })
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lm_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Explicitly UNFREEZE visual.merger + visual.deepstack_merger_list in FP32 master precision!
    merger_params = []
    main_merger_params = []
    deepstack_merger_params = []
    lora_params = []
    for name, p in model.named_parameters():
        if "merger" in name and p.is_floating_point():
            p.data = p.data.float()
            p.requires_grad_(True)
            merger_params.append(p)
            if "deepstack" in name:
                deepstack_merger_params.append(p)
            else:
                main_merger_params.append(p)
        elif p.requires_grad:
            lora_params.append(p)

    # Also ensure ViT inputs/outputs propagate gradients cleanly through unfrozen mergers
    vis_body_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and ("visual" in n or "vision" in n) and "merger" not in n
    )
    n_merger = sum(p.numel() for p in merger_params)
    n_main_m = sum(p.numel() for p in main_merger_params)
    n_ds_m = sum(p.numel() for p in deepstack_merger_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(
        f"Loaded {model_id} (rev={rev}) in {time.monotonic()-t0:.1f}s | "
        f"unfrozen merger params={n_merger/1e6:.2f}M (main={n_main_m/1e6:.2f}M/{len(main_merger_params)}t, "
        f"deepstack={n_ds_m/1e6:.2f}M/{len(deepstack_merger_params)}t) | "
        f"LM LoRA params={n_lora/1e6:.2f}M | ViT body trainable={vis_body_trainable} | "
        f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )
    if n_main_m == 0 or n_ds_m == 0:
        raise RuntimeError(
            f"Failed to unfreeze both visual.merger ({n_main_m}) and deepstack_merger_list ({n_ds_m})!"
        )
    return model, processor, rev, compute_dtype, merger_params, main_merger_params, deepstack_merger_params, lora_params


@torch.inference_mode()
def score_mcq_probs(model, processor, crop_im: Image.Image, prompt: str, options_list, compute_dtype):
    """Score exact calibrated softmax probabilities over option letters A..E (1 forward pass, zero hallucination!)."""
    letters = "ABCDEF"[:len(options_list)]
    prefix = '{"option": "'
    user_chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = user_chat + prefix
    inputs = processor(text=[full_text], images=[crop_im], return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=compute_dtype):
        out = model(**inputs, use_cache=False)
    last_logits = out.logits[0, -1, :].float()
    letter_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[0] for ch in letters]
    opt_logits = torch.stack([last_logits[tid] for tid in letter_ids])
    probs = F.softmax(opt_logits, dim=0).cpu().tolist()
    return {cc: round(float(pr), 5) for cc, pr in zip(options_list, probs)}


@torch.inference_mode()
def evaluate_all_ah66(
    model,
    processor,
    ah_rows,
    ah_mcq_spec,
    ah_open_g,
    step0_g,
    img_dir: Path,
    cc_names: dict,
    out_dir: Path,
    compute_dtype,
    tag: str,
):
    model.eval()
    processor.tokenizer.padding_side = "left"

    # 1. Isolated Local-Cue Shuffled Hard-Neighbor MCQ (exact same crops & shuffled options as Phase 1 baseline!)
    mcq_crops, mcq_prompts = [], []
    for item in ah_mcq_spec:
        im = load_clean_image(img_dir / f"{item['id']}.jpg", max_edge=768)
        mcq_crops.append(crop_cue(im, item["bbox_1000"], pad_frac=0.08))
        mcq_prompts.append(build_shuffled_mcq_prompt(item["label"], item["options"], cc_names))

    # Batched greedy decode (8x faster than bs=1) + exact 1-forward calibrated logit scoring
    mcq_raws = []
    for i in range(0, len(mcq_crops), 8):
        with torch.autocast("cuda", dtype=compute_dtype):
            mcq_raws.extend(
                generate_batch_single_img(model, processor, mcq_crops[i:i + 8], mcq_prompts[i:i + 8], max_new_tokens=36)
            )
    mcq_preds = []
    for item, crop_im, prompt, raw in zip(ah_mcq_spec, mcq_crops, mcq_prompts, mcq_raws):
        probs = score_mcq_probs(model, processor, crop_im, prompt, item["options"], compute_dtype)
        best_cc = max(probs.items(), key=lambda x: x[1])[0]
        opt, dec_cc = parse_mcq_json(raw, item["options"], cc_names)
        mcq_preds.append({
            **item,
            "logit_probs": probs,
            "logit_pred_country": best_cc,
            "logit_correct": best_cc == item["true_country"],
            "decode_pred_country": dec_cc,
            "decode_correct": dec_cc == item["true_country"],
            "raw": raw,
        })
    (out_dir / f"{tag}_ah66_isolated_mcq.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mcq_preds), encoding="utf-8"
    )
    n_all = len(mcq_preds)
    nontxt = [x for x in mcq_preds if not x["is_text_like"]]
    txt = [x for x in mcq_preds if x["is_text_like"]]
    mcq_summary = {
        "n_crops": n_all,
        "logit_all_acc_pct": round(100.0 * sum(x["logit_correct"] for x in mcq_preds) / max(1, n_all), 2),
        "logit_nontxt_acc_pct": round(100.0 * sum(x["logit_correct"] for x in nontxt) / max(1, len(nontxt)), 2),
        "logit_txt_acc_pct": round(100.0 * sum(x["logit_correct"] for x in txt) / max(1, len(txt)), 2),
        "decode_all_acc_pct": round(100.0 * sum(x["decode_correct"] for x in mcq_preds) / max(1, n_all), 2),
        "decode_nontxt_acc_pct": round(100.0 * sum(x["decode_correct"] for x in nontxt) / max(1, len(nontxt)), 2),
        "decode_txt_acc_pct": round(100.0 * sum(x["decode_correct"] for x in txt) / max(1, len(txt)), 2),
    }
    print(f"  [{tag}] Isolated Local-Cue Shuffled MCQ: {json.dumps(mcq_summary)}", flush=True)

    # 2. End-to-End Image-First Open-Ended Visual Fovea (Direct Prompt) on author_holdout_66
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
        prompt_txt = build_direct_fovea_prompt(r["kde_distribution"], notes)
        blocks.append({"type": "text", "text": prompt_txt})
        with torch.autocast("cuda", dtype=compute_dtype):
            raw = generate_one_multi_img(model, processor, imgs, blocks, max_new_tokens=120)
        p = parse_multiscale_json(raw, cc_names)
        coords = p["coordinates"]
        dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
        fov_preds.append({
            "id": r["id"], "true_country": r["country"],
            "true_latitude": r["latitude"], "true_longitude": r["longitude"],
            "crop_labels": [b["label"] for b in o_boxes],
            "guess_country": p["country"], "guess_coordinates": list(coords) if coords else None,
            "distance_km": dist, "selected_cluster": p["selected_cluster"],
            "confidence": p["confidence"], "parsed": p["parsed"], "raw": raw,
        })
    (out_dir / f"{tag}_ah66_direct_open_fovea.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in fov_preds), encoding="utf-8"
    )
    valid_d = [x["distance_km"] for x in fov_preds if x["distance_km"] is not None]
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in fov_preds)
    fov_summary = {
        "n": len(fov_preds),
        "parsed": sum(x["parsed"] for x in fov_preds),
        "country_correct": cc_ok,
        "country_acc_pct": round(100.0 * cc_ok / len(fov_preds), 2),
        "le_25km": int(sum(d <= 25.0 for d in valid_d)),
        "le_200km": int(sum(d <= 200.0 for d in valid_d)),
        "median_km": round(float(np.median(valid_d)), 1) if valid_d else None,
        "geoscore": round(float(sum(geoscore(d) for d in valid_d) / len(fov_preds)), 1),
        "sel_top1": sum(1 for x in fov_preds if x["selected_cluster"] == 1),
    }
    print(f"  [{tag}] Image-First Direct Open-Fovea: {json.dumps(fov_summary)}", flush=True)

    # 3. Also evaluate Image-First CoT Open-Fovea (verifying whether CoT & reasoning stayed intact!)
    cot_preds = []
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
        with torch.autocast("cuda", dtype=compute_dtype):
            raw = generate_one_multi_img(model, processor, imgs, blocks, max_new_tokens=320)
        p = parse_multiscale_json(raw, cc_names)
        coords = p["coordinates"]
        dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
        cot_preds.append({
            "id": r["id"], "true_country": r["country"],
            "true_latitude": r["latitude"], "true_longitude": r["longitude"],
            "crop_labels": [b["label"] for b in o_boxes],
            "guess_country": p["country"], "guess_coordinates": list(coords) if coords else None,
            "distance_km": dist, "selected_cluster": p["selected_cluster"],
            "confidence": p["confidence"], "visual_clues": p["visual_clues"],
            "reasoning": p["reasoning"], "parsed": p["parsed"], "raw": raw,
        })
    (out_dir / f"{tag}_ah66_cot_open_fovea.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cot_preds), encoding="utf-8"
    )
    vd_c = [x["distance_km"] for x in cot_preds if x["distance_km"] is not None]
    cc_c = sum(x["guess_country"] == x["true_country"] for x in cot_preds)
    cot_summary = {
        "n": len(cot_preds),
        "parsed": sum(x["parsed"] for x in cot_preds),
        "country_correct": cc_c,
        "country_acc_pct": round(100.0 * cc_c / len(cot_preds), 2),
        "le_25km": int(sum(d <= 25.0 for d in vd_c)),
        "le_200km": int(sum(d <= 200.0 for d in vd_c)),
        "median_km": round(float(np.median(vd_c)), 1) if vd_c else None,
        "geoscore": round(float(sum(geoscore(d) for d in vd_c) / len(cot_preds)), 1),
        "sel_top1": sum(1 for x in cot_preds if x["selected_cluster"] == 1),
    }
    print(f"  [{tag}] Image-First CoT Open-Fovea: {json.dumps(cot_summary)}", flush=True)

    # 4. After-Training Eager Cross-Attention Heatmaps (did unfrozen mergers shift attention onto local cues?!)
    set_peft_lm_attn_impl(model, "eager")
    processor.tokenizer.padding_side = "right"
    blind_prompt = "Identify the country where this street-level photo was taken. Respond with ONLY the 2-letter ISO country code:"
    attn_rows = []
    for r in ah_rows:
        im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        kde_p = build_multiscale_cot_prompt(r["kde_distribution"])
        with torch.autocast("cuda", dtype=compute_dtype):
            a_blind = extract_visual_attention(model, processor, im, blind_prompt)
            a_kde = extract_visual_attention(model, processor, im, kde_p)
        g6 = step0_g.get(r["id"], {}).get("boxes", [])
        go = ah_open_g.get(r["id"], {}).get("boxes", [])
        attn_rows.append({
            "id": r["id"], "true_country": r["country"],
            "blind": {**a_blind, "g6_stats": compute_box_attention_stats(a_blind["grid_norm"], g6), "open_stats": compute_box_attention_stats(a_blind["grid_norm"], go)},
            "kde": {**a_kde, "g6_stats": compute_box_attention_stats(a_kde["grid_norm"], g6), "open_stats": compute_box_attention_stats(a_kde["grid_norm"], go)},
        })
    (out_dir / f"{tag}_ah66_attention_maps.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in attn_rows), encoding="utf-8"
    )
    set_peft_lm_attn_impl(model, "sdpa")
    processor.tokenizer.padding_side = "left"

    attn_summary = {
        "blind_vis_share": round(float(np.mean([x["blind"]["total_vis_share"] for x in attn_rows])), 4),
        "kde_vis_share": round(float(np.mean([x["kde"]["total_vis_share"] for x in attn_rows])), 4),
        "blind_g6_conc": round(float(np.mean([x["blind"]["g6_stats"]["union_concentration"] for x in attn_rows if x["blind"]["g6_stats"]["union_concentration"] is not None])), 3),
        "kde_g6_conc": round(float(np.mean([x["kde"]["g6_stats"]["union_concentration"] for x in attn_rows if x["kde"]["g6_stats"]["union_concentration"] is not None])), 3),
        "blind_open_conc": round(float(np.mean([x["blind"]["open_stats"]["union_concentration"] for x in attn_rows if x["blind"]["open_stats"]["union_concentration"] is not None])), 3),
        "kde_open_conc": round(float(np.mean([x["kde"]["open_stats"]["union_concentration"] for x in attn_rows if x["kde"]["open_stats"]["union_concentration"] is not None])), 3),
    }
    print(f"  [{tag}] Attention Shift Summary: {json.dumps(attn_summary)}", flush=True)
    return {"mcq": mcq_summary, "direct_open_fovea": fov_summary, "cot_open_fovea": cot_summary, "attention": attn_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--merger-lr", type=float, default=6e-5)
    parser.add_argument("--lora-lr", type=float, default=4e-5)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    train_tasks = [json.loads(l) for l in (args.input_dir / "dir2_train_tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_mcq_spec = [json.loads(l) for l in (args.input_dir / "base_8b_ah66_isolated_mcq.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_open_g = {r["id"]: r for r in (json.loads(l) for l in (args.input_dir / "open_grounding_ah66.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    step0_g = {r["id"]: r for r in (json.loads(l) for l in (args.input_dir / "grounding_boxes_ah66.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    pool_sha = {r["id"]: r["sha256"] for r in (json.loads(l) for l in (args.input_dir / "harvest_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    with ZipFile(args.input_dir / "harvest_imgs.zip") as z:
        needed = {t["id"]: t["image"] for t in train_tasks}
        for rid, rel in needed.items():
            dst = img_dir / f"{rid}.jpg"
            if not dst.exists():
                payload = z.read(rel)
                if rid in pool_sha and hashlib.sha256(payload).hexdigest() != pool_sha[rid]:
                    raise ValueError(f"Train SHA mismatch: {rid}")
                dst.write_bytes(payload)
    with ZipFile(args.input_dir / "author-holdout-eval.zip") as z:
        for r in ah_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"AH SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(train_tasks)} training tasks {dict(Counter(t['task_type'] for t in train_tasks))} + {len(ah_rows)} author_holdout.", flush=True)

    model, processor, rev, compute_dtype, merger_params, main_merger_params, deepstack_merger_params, lora_params = (
        load_unfrozen_merger_qlora_model(MODEL_ID, args.lora_r, args.lora_alpha, 0.05)
    )

    # Pre-training baselines on author_holdout_66:
    # (A) Exact Logit MCQ probabilities + (B) Exact Direct Open-Fovea prompt (100% apples-to-apples!)
    print("\n=== Pre-Training Baselines on author_holdout_66 (Exact Logit MCQ + Direct Open-Fovea) ===", flush=True)
    model.eval()
    processor.tokenizer.padding_side = "left"
    base_logit_preds = []
    for item in ah_mcq_spec:
        im = load_clean_image(img_dir / f"{item['id']}.jpg", max_edge=768)
        crop_im = crop_cue(im, item["bbox_1000"], pad_frac=0.08)
        prompt = build_shuffled_mcq_prompt(item["label"], item["options"], cc_names)
        probs = score_mcq_probs(model, processor, crop_im, prompt, item["options"], compute_dtype)
        best_cc = max(probs.items(), key=lambda x: x[1])[0]
        base_logit_preds.append({**item, "logit_probs": probs, "logit_pred_country": best_cc, "logit_correct": best_cc == item["true_country"]})
    (args.output_dir / "base_ah66_isolated_mcq_logits.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in base_logit_preds), encoding="utf-8"
    )
    b_nt = [x for x in base_logit_preds if not x["is_text_like"]]
    b_tx = [x for x in base_logit_preds if x["is_text_like"]]
    base_mcq_summary = {
        "n_crops": len(base_logit_preds),
        "logit_all_acc_pct": round(100.0 * sum(x["logit_correct"] for x in base_logit_preds) / max(1, len(base_logit_preds)), 2),
        "logit_nontxt_acc_pct": round(100.0 * sum(x["logit_correct"] for x in b_nt) / max(1, len(b_nt)), 2),
        "logit_txt_acc_pct": round(100.0 * sum(x["logit_correct"] for x in b_tx) / max(1, len(b_tx)), 2),
    }
    print("  [BASE] Isolated Local-Cue Logit MCQ:", json.dumps(base_mcq_summary), flush=True)

    base_direct_preds = []
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
        prompt_txt = build_direct_fovea_prompt(r["kde_distribution"], notes)
        blocks.append({"type": "text", "text": prompt_txt})
        with torch.autocast("cuda", dtype=compute_dtype):
            raw = generate_one_multi_img(model, processor, imgs, blocks, max_new_tokens=120)
        p = parse_multiscale_json(raw, cc_names)
        coords = p["coordinates"]
        dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
        base_direct_preds.append({
            "id": r["id"], "true_country": r["country"],
            "true_latitude": r["latitude"], "true_longitude": r["longitude"],
            "crop_labels": [b["label"] for b in o_boxes],
            "guess_country": p["country"], "guess_coordinates": list(coords) if coords else None,
            "distance_km": dist, "selected_cluster": p["selected_cluster"],
            "confidence": p["confidence"], "parsed": p["parsed"], "raw": raw,
        })
    (args.output_dir / "base_ah66_direct_open_fovea.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in base_direct_preds), encoding="utf-8"
    )
    b_vd = [x["distance_km"] for x in base_direct_preds if x["distance_km"] is not None]
    b_cc = sum(x["guess_country"] == x["true_country"] for x in base_direct_preds)
    base_direct_summary = {
        "n": len(base_direct_preds),
        "parsed": sum(x["parsed"] for x in base_direct_preds),
        "country_correct": b_cc,
        "country_acc_pct": round(100.0 * b_cc / len(base_direct_preds), 2),
        "le_25km": int(sum(d <= 25.0 for d in b_vd)),
        "le_200km": int(sum(d <= 200.0 for d in b_vd)),
        "median_km": round(float(np.median(b_vd)), 1) if b_vd else None,
        "geoscore": round(float(sum(geoscore(d) for d in b_vd) / len(base_direct_preds)), 1),
        "sel_top1": sum(1 for x in base_direct_preds if x["selected_cluster"] == 1),
    }
    print("  [BASE] Image-First Direct Open-Fovea:", json.dumps(base_direct_summary), flush=True)

    # Differential LR PagedAdamW8bit: saves ~1.25 GB VRAM on 204M params + CPU paging safety!
    optimizer = bnb.optim.PagedAdamW8bit(
        [
            {"params": merger_params, "lr": args.merger_lr, "weight_decay": 0.01},
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.01},
        ]
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    all_trainable = merger_params + lora_params
    steps_per_epoch = math.ceil(len(train_tasks) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.08 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(
        f"\n=== Training Unfrozen Mergers (lr={args.merger_lr}) + LM LoRA (lr={args.lora_lr}) "
        f"with PagedAdamW8bit for {args.epochs} epochs ({total_steps} steps, eff_bs={args.grad_accum}) ===",
        flush=True,
    )
    history = []
    epoch_evals = {}
    rng = random.Random(args.seed)
    t0 = time.monotonic()
    global_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        processor.tokenizer.padding_side = "right"
        order = list(range(len(train_tasks)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        ep_losses, acc_mcq, acc_fov = [], [], []
        accum_count = 0

        for idx_i, ti in enumerate(order, 1):
            t = train_tasks[ti]
            if t["task_type"] == "isolated_cue_mcq":
                full_im = load_clean_image(img_dir / f"{t['id']}.jpg", max_edge=512)
                jbox = jitter_bbox(t["bbox_1000"], rng, frac=0.04)
                crop_im = mild_color_jitter(crop_cue(full_im, jbox, pad_frac=0.08, target_min_edge=224, max_edge=336), rng)
                imgs = [crop_im]
                blocks = [{"type": "image"}, {"type": "text", "text": t["prompt"]}]
            else:
                # Image-First Visual Fovea localization task (compact training crops keep peak VRAM ~14 GB on L4!)
                full_im = load_clean_image(img_dir / f"{t['id']}.jpg", max_edge=448)
                crops = [
                    mild_color_jitter(
                        crop_cue(full_im, jitter_bbox(bb, rng, frac=0.03), pad_frac=0.10, target_min_edge=196, max_edge=280),
                        rng,
                    )
                    for bb in t["crop_boxes"]
                ]
                imgs = [full_im] + crops
                blocks = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t["prompt"]}]

            inputs, labels = build_supervised_multi_inputs(processor, imgs, blocks, t["response"])
            n_resp = int((labels != -100).sum().item())
            inputs_cuda = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
            labels_cuda = labels.to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=compute_dtype):
                # Materialize lm_head ONLY for the response suffix (n_resp+1 tokens instead of ~1300 -> 50x less logit VRAM!)
                try:
                    out = model(**inputs_cuda, use_cache=False, logits_to_keep=n_resp + 1)
                except TypeError:
                    out = model(**inputs_cuda, use_cache=False)
                if out.logits.shape[1] == n_resp + 1:
                    shift_logits = out.logits[:, :-1, :].float()
                    shift_labels = labels_cuda[:, -n_resp:]
                else:
                    shift_logits = out.logits[:, -(n_resp + 1):-1, :].float()
                    shift_labels = labels_cuda[:, -n_resp:]
                loss_raw = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            (loss_raw / args.grad_accum).backward()

            lv = float(loss_raw.detach().item())
            del out, shift_logits, shift_labels, loss_raw, inputs_cuda, labels_cuda
            ep_losses.append(lv)
            if t["task_type"] == "isolated_cue_mcq":
                acc_mcq.append(lv)
            else:
                acc_fov.append(lv)
            accum_count += 1

            is_boundary = (accum_count == args.grad_accum) or (idx_i == len(order))
            if is_boundary:
                if accum_count < args.grad_accum:
                    scale = args.grad_accum / accum_count
                    for p in all_trainable:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                if global_step == 0:
                    def _gnorm(ps):
                        sq = sum(float(p.grad.detach().float().norm().item() ** 2) for p in ps if p.grad is not None)
                        return math.sqrt(sq)
                    gn_main = _gnorm(main_merger_params)
                    gn_ds = _gnorm(deepstack_merger_params)
                    gn_lora = _gnorm(lora_params)
                    print(
                        f"  [GRADIENT GATE step 1] main_merger_gnorm={gn_main:.4f} | "
                        f"deepstack_merger_gnorm={gn_ds:.4f} | lora_gnorm={gn_lora:.4f} | "
                        f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f} GB",
                        flush=True,
                    )
                    if gn_main <= 0.0 or gn_ds <= 0.0:
                        raise RuntimeError(
                            f"Dead visual merger gradient at step 1: main={gn_main}, deepstack={gn_ds}"
                        )
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 8 == 0:
                    torch.cuda.empty_cache()
                rec_h = {
                    "epoch": ep,
                    "step": global_step,
                    "loss_mcq": round(float(np.mean(acc_mcq)), 4) if acc_mcq else None,
                    "loss_fov": round(float(np.mean(acc_fov)), 4) if acc_fov else None,
                    "lr_merger": float(scheduler.get_last_lr()[0]),
                    "lr_lora": float(scheduler.get_last_lr()[1]),
                }
                history.append(rec_h)
                if global_step == 1 or global_step % 15 == 0 or idx_i == len(order):
                    elapsed = (time.monotonic() - t0) / 60.0
                    print(
                        f"  [Dir2 ep {ep}/{args.epochs} step {global_step:3d}/{total_steps}] "
                        f"mcq_nll={rec_h['loss_mcq']} | fov_nll={rec_h['loss_fov']} | "
                        f"lr_m={rec_h['lr_merger']:.2e} | peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB | elapsed={elapsed:.1f}m",
                        flush=True,
                    )
                acc_mcq, acc_fov = [], []
                accum_count = 0

        print(f"  Epoch {ep} mean NLL: {float(np.mean(ep_losses)):.4f}", flush=True)

        # Save BOTH LoRA adapter AND unfrozen merger weights per epoch!
        ckpt_dir = args.output_dir / f"ckpt_ep{ep}"
        ckpt_dir.mkdir(exist_ok=True)
        model.save_pretrained(ckpt_dir / "adapter")
        merger_sd = {
            n: p.detach().to(torch.bfloat16).cpu()
            for n, p in model.named_parameters()
            if "merger" in n and p.requires_grad
        }
        torch.save(merger_sd, ckpt_dir / "merger_weights_bf16.pt")
        print(f"Saved epoch {ep} adapter + merger weights ({len(merger_sd)} tensors) -> {ckpt_dir}", flush=True)

        torch.cuda.empty_cache()
        print(f"\n=== Evaluating Epoch {ep} (Unfrozen Mergers + LM LoRA) on author_holdout_66 ===", flush=True)
        epoch_evals[f"ep{ep}"] = evaluate_all_ah66(
            model, processor, ah_rows, ah_mcq_spec, ah_open_g, step0_g,
            img_dir, cc_names, args.output_dir, compute_dtype, tag=f"dir2_ep{ep}",
        )
        torch.cuda.empty_cache()

    # Identify best epoch (highest Direct Open-Fovea GeoScore, tie-break by isolated non-text MCQ)
    best_ep = max(
        epoch_evals.keys(),
        key=lambda k: (
            epoch_evals[k]["direct_open_fovea"]["geoscore"],
            epoch_evals[k]["mcq"]["logit_nontxt_acc_pct"],
        ),
    )
    (args.output_dir / "best_epoch.txt").write_text(best_ep + "\n", encoding="utf-8")
    print(f"Selected best_epoch={best_ep}", flush=True)

    report = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "hyperparams": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_mcq_logits": base_mcq_summary,
        "base_direct_open_fovea": base_direct_summary,
        "best_epoch": best_ep,
        "epoch_evals": epoch_evals,
        "history": history,
    }
    (args.output_dir / "dir2_train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted Direction 2 Phase 2 Unfrozen-Merger + LM LoRA Training & Evaluation!", flush=True)


if __name__ == "__main__":
    main()
