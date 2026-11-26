"""Hierarchical Multi-Scale Geographic Loss + Visual Fovea LM LoRA Worker (Colab L4).

Key design (directly informed by Step 0, Step 1, and Direction 2 mechanistic findings):
  1. FROZEN Vision Backbone + FROZEN Visual Mergers (`visual.merger` & `deepstack_merger_list`):
     Preserves 100% of Qwen3-VL-8B native OCR, grounding, and prevents Pixel[0,0] Attention Sink
     amplification seen in Direction 2 unfrozen-merger training.
  2. Lightweight LM LoRA (r=16, alpha=32, lr=3e-5) trained EXCLUSIVELY on multi-image
     Image-First Visual Fovea localization ([Full Image + up to 2 Zero-Sky Crops] + Top-8 KDE).
  3. Hierarchical Multi-Scale Geographic Loss at the `selected_cluster` (`1..K`) decision token:
       - Scale 1 (Macro / Continental, 2500 km): KL(q_2500 || p_theta)
       - Scale 2 (Country + 750 km):             KL(q_country || p_theta)
       - Scale 3 (Sub-Regional / 200 km):        KL(q_200 || p_theta)
       - Scale 4 (Expected GeoScore Regret):     max_j r_j - sum_k p_theta(k) * r_k,
         where r_k = exp(-d_k / 1492.7)
     The hard one-hot SFT CE label on the `selected_cluster` digit is masked (-100) so cluster
     selection receives pure hierarchical geographic gradients, while remaining JSON tokens
     (country, refined lat/lon, confidence) receive gentle SFT CE (only when true country is in Top-8).
  4. Evaluates BOTH Greedy JSON Decode AND Minimum Bayes Risk (MBR) Expected-GeoScore Decoding
     across Open-Fovea, 6-Class Fovea, and Hybrid Fovea crops on `author_holdout_66`, plus
     Isolated MCQ & Eager Cross-Attention anti-forgetting verification.
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

import bitsandbytes as bnb
import numpy as np
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
from run_step0_grounding_attention import (
    compute_box_attention_stats,
    crop_cue,
    extract_visual_attention,
    generate_one_multi_img,
    load_clean_image,
    set_lm_attn_impl,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def set_peft_lm_attn_impl(model, impl: str):
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


def jitter_bbox(bbox_1000, rng: random.Random, frac: float = 0.03):
    x1, y1, x2, y2 = bbox_1000
    bw, bh = x2 - x1, y2 - y1
    nx1 = max(0.0, min(980.0, x1 + rng.uniform(-frac, frac) * bw))
    ny1 = max(0.0, min(980.0, y1 + rng.uniform(-frac, frac) * bh))
    nx2 = max(nx1 + 20.0, min(1000.0, x2 + rng.uniform(-frac, frac) * bw))
    ny2 = max(ny1 + 20.0, min(1000.0, y2 + rng.uniform(-frac, frac) * bh))
    return [nx1, ny1, nx2, ny2]


def mild_color_jitter(im: Image.Image, rng: random.Random) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.93, 1.07))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.93, 1.07))
    return im


def load_frozen_vision_lora_model(model_id: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {model_id} (Frozen Vision + 4-bit NF4 LM LoRA, compute={compute_dtype}) on {torch.cuda.get_device_name(0)}...", flush=True)
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

    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lm_targets = sorted({
        name for name, _ in model.named_modules()
        if any(name.endswith(f".{s}") for s in target_suffixes)
        and "visual" not in name
        and "vision" not in name
        and "merger" not in name
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

    vis_trainable = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and ("visual" in n or "vision" in n or "merger" in n)
    )
    lora_params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    print(
        f"Loaded {model_id} (rev={rev}) in {time.monotonic()-t0:.1f}s | "
        f"LM LoRA trainable={n_lora/1e6:.2f}M | Vision+Merger trainable={vis_trainable} | "
        f"VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )
    if vis_trainable != 0:
        raise RuntimeError(f"Expected 0 vision/merger trainable params, got {vis_trainable}")
    return model, processor, rev, compute_dtype, lora_params


def get_cluster_digit_ids(tokenizer):
    ids = []
    for d in "12345678":
        toks = tokenizer.encode(d, add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(f"Digit {d!r} tokenized to {toks} (expected 1 token)")
        ids.append(toks[0])
    return ids


def build_supervised_hier_inputs(processor, images, content_blocks, response_text: str, target_cluster_idx: int, digit_ids):
    """Build multi-image inputs + labels AND locate exact token position of `selected_cluster` digit."""
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

    # Find the exact token position of the selected_cluster digit inside the first 12 response tokens
    target_tid = digit_ids[target_cluster_idx - 1]
    cluster_tok_pos = None
    for pos in range(prefix_len, min(len(f_ids), prefix_len + 12)):
        if f_ids[pos] == target_tid:
            cluster_tok_pos = pos
            break
    if cluster_tok_pos is None:
        resp_toks = [processor.tokenizer.decode([t]) for t in f_ids[prefix_len:prefix_len + 12]]
        raise RuntimeError(f"Could not find cluster digit {target_cluster_idx} (tid={target_tid}) in response prefix {resp_toks}")

    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100
    # Mask the hard one-hot selected_cluster digit from SFT CE so hierarchical loss owns it cleanly!
    labels[:, cluster_tok_pos] = -100
    return full_inputs, labels, prefix_len, cluster_tok_pos


def compute_hierarchical_cluster_loss(
    cluster_logits: torch.Tensor,
    cluster_dists_km: list,
    cluster_countries: list,
    true_country: str,
    w_macro: float = 0.25,
    w_country: float = 0.40,
    w_fine: float = 0.25,
    w_geo: float = 1.50,
):
    """Differentiable Multi-Scale Hierarchical Geographic Loss over K <= 8 candidate clusters."""
    k = len(cluster_dists_km)
    logits = cluster_logits[:k].float()
    log_p = F.log_softmax(logits, dim=0)
    p = log_p.exp()

    d = torch.tensor(cluster_dists_km, dtype=torch.float32, device=logits.device)

    # Scale 1: Macro / Continental distance distribution (tau = 2500 km)
    q_macro = F.softmax(-d / 2500.0, dim=0)
    l_macro = F.kl_div(log_p, q_macro, reduction="sum")

    # Scale 2: Country-level distribution (same country prioritized + 750 km decay)
    cc_mask = torch.tensor([1.0 if c == true_country else 0.0 for c in cluster_countries], dtype=torch.float32, device=logits.device)
    if float(cc_mask.sum().item()) > 0.0:
        scores_cc = (-d / 750.0).masked_fill(cc_mask == 0.0, -1e9)
        q_cc = F.softmax(scores_cc, dim=0)
    else:
        q_cc = F.softmax(-d / 750.0, dim=0)
    l_cc = F.kl_div(log_p, q_cc, reduction="sum")

    # Scale 3: Fine Sub-Regional distance distribution (tau = 200 km)
    q_fine = F.softmax(-d / 200.0, dim=0)
    l_fine = F.kl_div(log_p, q_fine, reduction="sum")

    # Scale 4: Direct Expected GeoScore Regret Loss (exact metric alignment: exp(-d / 1492.7))
    r_geo = torch.exp(-d / 1492.7)
    l_geo = r_geo.max() - torch.sum(p * r_geo)

    total = w_macro * l_macro + w_country * l_cc + w_fine * l_fine + w_geo * l_geo
    return total, {
        "l_macro": float(l_macro.detach().item()),
        "l_cc": float(l_cc.detach().item()),
        "l_fine": float(l_fine.detach().item()),
        "l_geo": float(l_geo.detach().item()),
    }


@torch.inference_mode()
def score_cluster_probs_multi_img(model, processor, imgs, blocks, n_clusters: int, digit_ids, compute_dtype):
    """Exact 1-forward calibrated probability distribution over candidate clusters 1..K."""
    prefix = '{"selected_cluster": '
    user_chat = processor.apply_chat_template(
        [{"role": "user", "content": blocks}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[user_chat + prefix], images=imgs, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=compute_dtype):
        try:
            out = model(**inputs, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = model(**inputs, use_cache=False)
    last_logits = out.logits[0, -1, :].float()
    cl_logits = torch.stack([last_logits[digit_ids[j]] for j in range(n_clusters)])
    probs = F.softmax(cl_logits, dim=0).cpu().tolist()
    return [float(x) for x in probs]


def pick_mbr_cluster(clusters, probs):
    """Minimum Bayes Risk (MBR) cluster maximizing expected GeoScore under p_theta(k)."""
    k = len(clusters)
    best_j, best_exp = 0, -1.0
    for j in range(k):
        cj = clusters[j]
        exp_gs = 0.0
        for m in range(k):
            cm = clusters[m]
            d_jm = haversine_km(cj["latitude"], cj["longitude"], cm["latitude"], cm["longitude"])
            exp_gs += probs[m] * math.exp(-d_jm / 1492.7)
        if exp_gs > best_exp:
            best_exp = exp_gs
            best_j = j
    return best_j, best_exp


@torch.inference_mode()
def score_mcq_probs(model, processor, crop_im: Image.Image, prompt: str, options_list, compute_dtype):
    letters = "ABCDEF"[:len(options_list)]
    prefix = '{"option": "'
    user_chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[user_chat + prefix], images=[crop_im], return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=compute_dtype):
        try:
            out = model(**inputs, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = model(**inputs, use_cache=False)
    last_logits = out.logits[0, -1, :].float()
    letter_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[0] for ch in letters]
    opt_logits = torch.stack([last_logits[tid] for tid in letter_ids])
    probs = F.softmax(opt_logits, dim=0).cpu().tolist()
    return {cc: round(float(pr), 5) for cc, pr in zip(options_list, probs)}


def select_boxes(rid: str, mode: str, open_g: dict, g6: dict):
    o_boxes = open_g.get(rid, {}).get("boxes", [])
    s_boxes = g6.get(rid, {}).get("boxes", [])
    if mode == "open":
        return o_boxes[:2]
    if mode == "g6":
        return s_boxes[:2]
    if mode == "hybrid":
        # Prefer high-precision 6-class boxes (signs/plates/poles/bollards/markings); fallback to open cues
        if s_boxes:
            return s_boxes[:2]
        return o_boxes[:2]
    raise ValueError(mode)


@torch.inference_mode()
def evaluate_fovea_suite(
    model,
    processor,
    ah_rows,
    open_g,
    g6,
    img_dir: Path,
    cc_names: dict,
    digit_ids,
    out_dir: Path,
    compute_dtype,
    tag: str,
    modes=("open", "g6"),
):
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    if hasattr(model, "config"):
        model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    results = {}
    all_mode_preds = {}

    def _agg_preds(preds_list):
        def _agg(d_key, c_key):
            vd = [x[d_key] for x in preds_list if x[d_key] is not None]
            cc_ok = sum(x[c_key] == x["true_country"] for x in preds_list)
            return {
                "cc_pct": round(100.0 * cc_ok / len(preds_list), 2),
                "cc": cc_ok,
                "le25": int(sum(d <= 25.0 for d in vd)),
                "le200": int(sum(d <= 200.0 for d in vd)),
                "med_km": round(float(np.median(vd)), 1) if vd else None,
                "geoscore": round(float(sum(geoscore(d) for d in vd) / len(preds_list)), 1),
            }
        return {
            "greedy": {
                **_agg("distance_km", "guess_country"),
                "parsed": sum(x["parsed"] for x in preds_list),
                "sel_top1": sum(1 for x in preds_list if x["selected_cluster"] == 1),
            },
            "logit_argmax": {
                **_agg("argmax_distance_km", "argmax_country"),
                "sel_top1": sum(1 for x in preds_list if x["argmax_cluster"] == 1),
            },
            "mbr_expected_gs": {
                **_agg("mbr_distance_km", "mbr_country"),
                "sel_top1": sum(1 for x in preds_list if x["mbr_cluster"] == 1),
            },
        }

    for mode in modes:
        preds = []
        for r in ah_rows:
            full_im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
            boxes = select_boxes(r["id"], mode, open_g, g6)
            imgs = [full_im]
            blocks = [{"type": "image"}]
            notes = []
            for ci, b in enumerate(boxes, 1):
                imgs.append(crop_cue(full_im, b["bbox_1000"], pad_frac=0.10))
                blocks.append({"type": "image"})
                extra = b.get("why") or b.get("note") or ""
                notes.append(f"Zoomed Detail #{ci}: `{b['label']}` ({extra})" if extra else f"Zoomed Detail #{ci}: `{b['label']}`")
            prompt_txt = build_direct_fovea_prompt(r["kde_distribution"], notes)
            blocks.append({"type": "text", "text": prompt_txt})

            cls = r["kde_distribution"]["clusters"]
            cl_probs = score_cluster_probs_multi_img(model, processor, imgs, blocks, len(cls), digit_ids, compute_dtype)
            argmax_j = int(np.argmax(cl_probs))
            mbr_j, mbr_exp = pick_mbr_cluster(cls, cl_probs)

            with torch.autocast("cuda", dtype=compute_dtype):
                raw = generate_one_multi_img(model, processor, imgs, blocks, max_new_tokens=96)
            p = parse_multiscale_json(raw, cc_names)
            coords = p["coordinates"]
            dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
            d_arg = haversine_km(r["latitude"], r["longitude"], cls[argmax_j]["latitude"], cls[argmax_j]["longitude"])
            d_mbr = haversine_km(r["latitude"], r["longitude"], cls[mbr_j]["latitude"], cls[mbr_j]["longitude"])

            preds.append({
                "id": r["id"],
                "true_country": r["country"],
                "true_latitude": r["latitude"],
                "true_longitude": r["longitude"],
                "crop_mode": mode,
                "crop_labels": [b["label"] for b in boxes],
                "guess_country": p["country"],
                "guess_coordinates": list(coords) if coords else None,
                "distance_km": dist,
                "selected_cluster": p["selected_cluster"],
                "confidence": p["confidence"],
                "parsed": p["parsed"],
                "cluster_probs": [round(x, 5) for x in cl_probs],
                "argmax_cluster": argmax_j + 1,
                "argmax_country": cls[argmax_j]["country"],
                "argmax_distance_km": d_arg,
                "mbr_cluster": mbr_j + 1,
                "mbr_country": cls[mbr_j]["country"],
                "mbr_distance_km": d_mbr,
                "mbr_expected_gs": round(5000.0 * mbr_exp, 1),
                "raw": raw,
            })

        (out_dir / f"{tag}_ah66_{mode}_fovea.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds), encoding="utf-8"
        )
        all_mode_preds[mode] = {x["id"]: x for x in preds}
        results[mode] = _agg_preds(preds)
        print(f"  [{tag}] {mode:6s} Fovea: {json.dumps(results[mode])}", flush=True)

    # Zero-cost exact Hybrid Fovea: g6 boxes where available (51/66), open-ended boxes fallback (15/66)
    if "open" in all_mode_preds and "g6" in all_mode_preds:
        hyb_preds = []
        for r in ah_rows:
            rid = r["id"]
            src = all_mode_preds["g6"][rid] if g6.get(rid, {}).get("boxes") else all_mode_preds["open"][rid]
            hyb_preds.append({**src, "crop_mode": "hybrid"})
        (out_dir / f"{tag}_ah66_hybrid_fovea.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in hyb_preds), encoding="utf-8"
        )
        results["hybrid"] = _agg_preds(hyb_preds)
        print(f"  [{tag}] hybrid Fovea: {json.dumps(results['hybrid'])}", flush=True)
    return results


@torch.inference_mode()
def evaluate_antiforgetting_mcq_and_attn(
    model,
    processor,
    ah_rows,
    ah_mcq_spec,
    open_g,
    g6,
    img_dir: Path,
    cc_names: dict,
    out_dir: Path,
    compute_dtype,
    tag: str,
    run_attn: bool = True,
):
    model.eval()
    processor.tokenizer.padding_side = "left"
    mcq_preds = []
    for item in ah_mcq_spec:
        im = load_clean_image(img_dir / f"{item['id']}.jpg", max_edge=768)
        crop_im = crop_cue(im, item["bbox_1000"], pad_frac=0.08)
        prompt = build_shuffled_mcq_prompt(item["label"], item["options"], cc_names)
        probs = score_mcq_probs(model, processor, crop_im, prompt, item["options"], compute_dtype)
        best_cc = max(probs.items(), key=lambda x: x[1])[0]
        mcq_preds.append({**item, "logit_probs": probs, "logit_pred_country": best_cc, "logit_correct": best_cc == item["true_country"]})
    (out_dir / f"{tag}_ah66_isolated_mcq_logits.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mcq_preds), encoding="utf-8"
    )
    nt = [x for x in mcq_preds if not x["is_text_like"]]
    tx = [x for x in mcq_preds if x["is_text_like"]]
    mcq_sum = {
        "n_crops": len(mcq_preds),
        "logit_all_acc_pct": round(100.0 * sum(x["logit_correct"] for x in mcq_preds) / max(1, len(mcq_preds)), 2),
        "logit_nontxt_acc_pct": round(100.0 * sum(x["logit_correct"] for x in nt) / max(1, len(nt)), 2),
        "logit_txt_acc_pct": round(100.0 * sum(x["logit_correct"] for x in tx) / max(1, len(tx)), 2),
    }
    print(f"  [{tag}] Anti-Forgetting Isolated MCQ Logits: {json.dumps(mcq_sum)}", flush=True)

    attn_sum = None
    if run_attn:
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
            bg = g6.get(r["id"], {}).get("boxes", [])
            bo = open_g.get(r["id"], {}).get("boxes", [])
            attn_rows.append({
                "id": r["id"], "true_country": r["country"],
                "blind": {**a_blind, "g6_stats": compute_box_attention_stats(a_blind["grid_norm"], bg), "open_stats": compute_box_attention_stats(a_blind["grid_norm"], bo)},
                "kde": {**a_kde, "g6_stats": compute_box_attention_stats(a_kde["grid_norm"], bg), "open_stats": compute_box_attention_stats(a_kde["grid_norm"], bo)},
            })
        (out_dir / f"{tag}_ah66_attention_maps.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in attn_rows), encoding="utf-8"
        )
        set_peft_lm_attn_impl(model, "sdpa")
        processor.tokenizer.padding_side = "left"
        p00_k = [float(np.array(x["kde"]["grid_norm"])[0, 0]) for x in attn_rows]
        attn_sum = {
            "blind_vis_share": round(float(np.mean([x["blind"]["total_vis_share"] for x in attn_rows])), 4),
            "kde_vis_share": round(float(np.mean([x["kde"]["total_vis_share"] for x in attn_rows])), 4),
            "kde_p00_sink": round(float(np.mean(p00_k)), 4),
            "blind_open_conc": round(float(np.mean([x["blind"]["open_stats"]["union_concentration"] for x in attn_rows if x["blind"]["open_stats"]["union_concentration"] is not None])), 3),
            "kde_open_conc": round(float(np.mean([x["kde"]["open_stats"]["union_concentration"] for x in attn_rows if x["kde"]["open_stats"]["union_concentration"] is not None])), 3),
        }
        print(f"  [{tag}] Anti-Forgetting Attention Check: {json.dumps(attn_sum)}", flush=True)
    return {"mcq": mcq_sum, "attention": attn_sum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora-lr", type=float, default=3e-5)
    parser.add_argument("--sft-weight", type=float, default=0.35)
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
    train_tasks = [json.loads(l) for l in (args.input_dir / "hier_fovea_train_tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_mcq_spec = [json.loads(l) for l in (args.input_dir / "base_8b_ah66_isolated_mcq.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_open_g = {r["id"]: r for r in (json.loads(l) for l in (args.input_dir / "open_grounding_ah66.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    step0_g = {r["id"]: r for r in (json.loads(l) for l in (args.input_dir / "grounding_boxes_ah66.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    pool_sha = {r["id"]: r["sha256"] for r in (json.loads(l) for l in (args.input_dir / "harvest_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    with ZipFile(args.input_dir / "harvest_imgs.zip") as z:
        for t in train_tasks:
            dst = img_dir / f"{t['id']}.jpg"
            if not dst.exists():
                payload = z.read(t["image"])
                if t["id"] in pool_sha and hashlib.sha256(payload).hexdigest() != pool_sha[t["id"]]:
                    raise ValueError(f"Train SHA mismatch: {t['id']}")
                dst.write_bytes(payload)
    with ZipFile(args.input_dir / "author-holdout-eval.zip") as z:
        for r in ah_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"AH SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(train_tasks)} Hier-Fovea tasks {dict(Counter(t['category'] for t in train_tasks))} + {len(ah_rows)} AH66.", flush=True)

    model, processor, rev, compute_dtype, lora_params = load_frozen_vision_lora_model(
        MODEL_ID, args.lora_r, args.lora_alpha, 0.05
    )
    digit_ids = get_cluster_digit_ids(processor.tokenizer)
    digit_ids_t = torch.tensor(digit_ids, dtype=torch.long, device="cuda")
    print(f"Verified cluster digit token IDs (1..8): {digit_ids}", flush=True)

    # Pre-training Baselines on author_holdout_66 across all 3 Fovea crop modes (Greedy + Logit + MBR)
    print("\n=== Pre-Training Zero-Shot Baselines on author_holdout_66 (Open / Hybrid / 6-Class Fovea) ===", flush=True)
    base_fovea = evaluate_fovea_suite(
        model, processor, ah_rows, ah_open_g, step0_g,
        img_dir, cc_names, digit_ids, args.output_dir, compute_dtype, tag="base",
    )

    optimizer = bnb.optim.PagedAdamW8bit(lora_params, lr=args.lora_lr, weight_decay=0.01)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    steps_per_epoch = math.ceil(len(train_tasks) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.10 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(
        f"\n=== Training Frozen-Vision Hier-Fovea LM LoRA (lr={args.lora_lr}, sft_w={args.sft_weight}) "
        f"for {args.epochs} epochs ({total_steps} steps, eff_bs={args.grad_accum}) ===",
        flush=True,
    )
    history = []
    epoch_evals = {}
    rng = random.Random(args.seed)
    t0 = time.monotonic()
    global_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"):
            model.config.use_cache = False
        processor.tokenizer.padding_side = "right"
        order = list(range(len(train_tasks)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        acc_tot, acc_hier, acc_geo, acc_sft = [], [], [], []
        accum_count = 0

        for idx_i, ti in enumerate(order, 1):
            t = train_tasks[ti]
            full_im = load_clean_image(img_dir / f"{t['id']}.jpg", max_edge=480)
            crops = [
                mild_color_jitter(
                    crop_cue(full_im, jitter_bbox(bb, rng, frac=0.03), pad_frac=0.10, target_min_edge=196, max_edge=280),
                    rng,
                )
                for bb in t["crop_boxes"]
            ]
            imgs = [full_im] + crops
            blocks = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t["prompt"]}]

            inputs, labels, prefix_len, cluster_tok_pos = build_supervised_hier_inputs(
                processor, imgs, blocks, t["response"], t["target_cluster"], digit_ids
            )
            seq_len = int(inputs["input_ids"].shape[1])
            n_keep = seq_len - (prefix_len - 1)  # covers prefix_len-1 .. seq_len-1
            inputs_cuda = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
            labels_cuda = labels.to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=compute_dtype):
                try:
                    out = model(**inputs_cuda, use_cache=False, logits_to_keep=n_keep)
                except TypeError:
                    out = model(**inputs_cuda, use_cache=False)

                if out.logits.shape[1] == n_keep:
                    suffix_logits = out.logits[0]  # pos prefix_len-1 .. seq_len-1
                    rel_pred_idx = (cluster_tok_pos - 1) - (prefix_len - 1)
                else:
                    suffix_logits = out.logits[0, prefix_len - 1:, :]
                    rel_pred_idx = (cluster_tok_pos - 1) - (prefix_len - 1)

                # 1. Exact Hierarchical Multi-Scale Geographic Loss at selected_cluster prediction logit
                cl_logits = suffix_logits[rel_pred_idx].index_select(0, digit_ids_t)
                l_hier, h_parts = compute_hierarchical_cluster_loss(
                    cl_logits, t["cluster_dists_km"], t["cluster_countries"], t["country"]
                )

                # 2. Gentle SFT CE on remaining JSON tokens (only when true country is in Top-8!)
                if t["apply_sft_rest"]:
                    shift_logits = suffix_logits[:-1, :].float()
                    shift_labels = labels_cuda[0, prefix_len:]
                    l_sft = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                else:
                    l_sft = torch.zeros((), dtype=torch.float32, device="cuda")

                loss_raw = l_hier + args.sft_weight * l_sft

            (loss_raw / args.grad_accum).backward()

            acc_tot.append(float(loss_raw.detach().item()))
            acc_hier.append(float(l_hier.detach().item()))
            acc_geo.append(h_parts["l_geo"])
            acc_sft.append(float(l_sft.detach().item()))
            del out, suffix_logits, cl_logits, l_hier, l_sft, loss_raw, inputs_cuda, labels_cuda
            accum_count += 1

            is_boundary = (accum_count == args.grad_accum) or (idx_i == len(order))
            if is_boundary:
                if accum_count < args.grad_accum:
                    scale = args.grad_accum / accum_count
                    for p in lora_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                if global_step == 0:
                    gn = math.sqrt(sum(float(p.grad.detach().float().norm().item() ** 2) for p in lora_params if p.grad is not None))
                    print(f"  [GRADIENT GATE step 1] lora_gnorm={gn:.4f} | peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
                    if gn <= 0.0 or math.isnan(gn):
                        raise RuntimeError(f"Invalid gradient norm at step 1: {gn}")
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 8 == 0:
                    torch.cuda.empty_cache()

                rec = {
                    "epoch": ep,
                    "step": global_step,
                    "loss_tot": round(float(np.mean(acc_tot)), 4),
                    "loss_hier": round(float(np.mean(acc_hier)), 4),
                    "loss_geo": round(float(np.mean(acc_geo)), 4),
                    "loss_sft": round(float(np.mean(acc_sft)), 4),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                history.append(rec)
                if global_step == 1 or global_step % 10 == 0 or idx_i == len(order):
                    elapsed = (time.monotonic() - t0) / 60.0
                    print(
                        f"  [Hier ep {ep}/{args.epochs} step {global_step:3d}/{total_steps}] "
                        f"tot={rec['loss_tot']:.4f} (hier={rec['loss_hier']:.4f}, geo_regret={rec['loss_geo']:.4f}, sft={rec['loss_sft']:.4f}) "
                        f"| lr={rec['lr']:.2e} | peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB | elapsed={elapsed:.1f}m",
                        flush=True,
                    )
                acc_tot, acc_hier, acc_geo, acc_sft = [], [], [], []
                accum_count = 0

        ckpt_dir = args.output_dir / f"ckpt_ep{ep}"
        ckpt_dir.mkdir(exist_ok=True)
        model.save_pretrained(ckpt_dir / "adapter")
        print(f"Saved epoch {ep} LoRA adapter -> {ckpt_dir / 'adapter'}", flush=True)

        torch.cuda.empty_cache()
        print(f"\n=== Evaluating Hier-Fovea Epoch {ep} on author_holdout_66 ===", flush=True)
        fov_res = evaluate_fovea_suite(
            model, processor, ah_rows, ah_open_g, step0_g,
            img_dir, cc_names, digit_ids, args.output_dir, compute_dtype, tag=f"hier_ep{ep}",
        )
        af_res = evaluate_antiforgetting_mcq_and_attn(
            model, processor, ah_rows, ah_mcq_spec, ah_open_g, step0_g,
            img_dir, cc_names, args.output_dir, compute_dtype, tag=f"hier_ep{ep}",
            run_attn=(ep == args.epochs),
        )
        epoch_evals[f"ep{ep}"] = {"fovea": fov_res, **af_res}
        torch.cuda.empty_cache()

    best_ep = max(
        epoch_evals.keys(),
        key=lambda k: max(
            epoch_evals[k]["fovea"]["hybrid"]["greedy"]["geoscore"],
            epoch_evals[k]["fovea"]["hybrid"]["mbr_expected_gs"]["geoscore"],
            epoch_evals[k]["fovea"]["open"]["greedy"]["geoscore"],
        ),
    )
    (args.output_dir / "best_epoch.txt").write_text(best_ep + "\n", encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "hyperparams": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_fovea": base_fovea,
        "best_epoch": best_ep,
        "epoch_evals": epoch_evals,
        "history": history,
    }
    (args.output_dir / "hier_fovea_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nCompleted Hierarchical-Loss Visual Fovea Training & Eval! best_epoch={best_ep}", flush=True)


if __name__ == "__main__":
    main()
