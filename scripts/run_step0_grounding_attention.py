"""Step 0 Diagnostic Worker (Colab GPU):
Tests Qwen/Qwen3-VL-8B-Instruct (4-bit NF4, untrained) on author_holdout_66 across 4 diagnostics:
  1. Native Visual Grounding: asks Qwen3-VL-8B directly for [0..1000] bbox_2d of GeoGuessr cues
     (utility pole, road sign, bollard, license plate, road marking, signage/text).
  2. Visual Cross-Attention Heatmaps: extracts exact layer-averaged LM attention over merged ViT
     patches under both blind-country and KDE-rerank prompts, measuring exact Attention Concentration
     Ratio inside detected cue boxes vs. image area.
  3. Object-Isolated Cue Probe: feeds ONLY high-res cropped cue objects (no background) to measure
     how much country signal Qwen3-VL-8B can already decode from isolated poles/signs/bollards/plates.
  4. Visual Fovea Injection (Zero-Training Test): feeds [Full Image + up to 2 Grounded Cue Crops]
     with the identical Top-8 Multi-Scale KDE prompt for an exact apples-to-apples comparison against
     full-image-only Base 8B (3,482.0 single-pass / 3,502.6 two-stage).
"""
import argparse
import gc
import hashlib
import json
import re
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
    geoscore,
    haversine_km,
    normalize_country_code,
    parse_multiscale_json,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

GROUNDING_PROMPT = (
    "Detect clearly visible GeoGuessr localization clues in this street-level photo from these categories:\n"
    "  - \"utility pole\" (wooden/concrete/metal electricity or telephone poles)\n"
    "  - \"road sign\" (traffic, highway, direction, street name, or warning signs)\n"
    "  - \"bollard\" (roadside reflector posts, delineators, or kilometer markers)\n"
    "  - \"license plate\" (vehicle license/number plates)\n"
    "  - \"road marking\" (painted center-lines, edge-lines, crosswalks, or arrows on road)\n"
    "  - \"text sign\" (shopfront signs, billboards, bus text, or banners with readable text)\n"
    "Return ONLY a valid JSON array of up to 5 most informative clearly visible objects "
    "(return [] if none are clearly visible). Use normalized [0, 1000] integer coordinates [x1, y1, x2, y2]:\n"
    '[{"label": "<category>", "bbox_2d": [x1, y1, x2, y2], "note": "<3-7 word visual description>"}]'
)

CUE_PRIORITY = {
    "text sign": 0,
    "road sign": 1,
    "license plate": 2,
    "utility pole": 3,
    "bollard": 4,
    "road marking": 5,
}


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
    rev = str(getattr(model.config, "_commit_hash", "unknown"))
    print(f"Loaded {model_id} (rev={rev}) in {time.monotonic() - t0:.1f}s | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    return model, processor, rev


def parse_grounding_boxes(raw_text: str, img_w: int, img_h: int):
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
        # Fallback: extract individual JSON objects
        arr = []
        dec = json.JSONDecoder()
        i = 0
        while i < len(cleaned):
            if cleaned[i] == "{":
                try:
                    obj, end = dec.raw_decode(cleaned[i:])
                    if isinstance(obj, dict) and "bbox_2d" in obj:
                        arr.append(obj)
                    i += end
                    continue
                except ValueError:
                    pass
            i += 1

    boxes = []
    for item in arr[:8]:
        if not isinstance(item, dict):
            continue
        bb = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except (TypeError, ValueError):
            continue
        # Detect if coords are [0..1] vs [0..1000]
        if max(x1, y1, x2, y2) <= 1.05:
            x1, y1, x2, y2 = x1 * 1000.0, y1 * 1000.0, x2 * 1000.0, y2 * 1000.0
        x1, x2 = max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))
        y1, y2 = max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))
        if x2 <= x1 + 8.0 or y2 <= y1 + 8.0:
            continue
        lbl = str(item.get("label", "other")).strip().lower()
        # Canonicalize common synonyms
        if "pole" in lbl:
            lbl = "utility pole"
        elif "bollard" in lbl or "delineator" in lbl or "marker" in lbl:
            lbl = "bollard"
        elif "plate" in lbl:
            lbl = "license plate"
        elif "marking" in lbl or "line" in lbl or "crosswalk" in lbl:
            lbl = "road marking"
        elif "text" in lbl or "shop" in lbl or "store" in lbl or "billboard" in lbl or "banner" in lbl:
            lbl = "text sign"
        elif "sign" in lbl:
            lbl = "road sign"
        boxes.append({
            "label": lbl,
            "bbox_1000": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "bbox_px": [
                int(round(x1 / 1000.0 * img_w)),
                int(round(y1 / 1000.0 * img_h)),
                int(round(x2 / 1000.0 * img_w)),
                int(round(y2 / 1000.0 * img_h)),
            ],
            "note": str(item.get("note", "")).strip(),
        })
    return boxes


def crop_cue(im: Image.Image, bbox_1000, pad_frac: float = 0.12, target_min_edge: int = 224, max_edge: int = 384) -> Image.Image:
    w, h = im.size
    x1 = bbox_1000[0] / 1000.0 * w
    y1 = bbox_1000[1] / 1000.0 * h
    x2 = bbox_1000[2] / 1000.0 * w
    y2 = bbox_1000[3] / 1000.0 * h
    bw, bh = max(12.0, x2 - x1), max(12.0, y2 - y1)
    # Add small context margin and ensure minimum crop size so ViT gets coherent geometry
    px = max(bw * pad_frac, max(0.0, (48.0 - bw) / 2.0))
    py = max(bh * pad_frac, max(0.0, (48.0 - bh) / 2.0))
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(w, int(round(x2 + px)))
    cy2 = min(h, int(round(y2 + py)))
    crop = im.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size
    scale = 1.0
    if min(cw, ch) < target_min_edge:
        scale = target_min_edge / max(1, min(cw, ch))
    if max(cw * scale, ch * scale) > max_edge:
        scale = max_edge / max(1, max(cw, ch))
    if abs(scale - 1.0) > 0.05:
        crop = crop.resize((max(28, int(round(cw * scale))), max(28, int(round(ch * scale)))), Image.Resampling.LANCZOS)
    return Image.frombytes("RGB", crop.size, crop.tobytes())


def set_lm_attn_impl(model, impl: str):
    """Switch LM attention implementation reliably across composite Qwen3VL configs."""
    try:
        model.set_attn_implementation(impl)
    except Exception as e:
        print(f"  Warning: model.set_attn_implementation({impl!r}) raised {e}; updating configs directly.", flush=True)
    for cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ):
        if cfg is not None:
            try:
                cfg._attn_implementation = impl
            except Exception:
                pass


@torch.inference_mode()
def generate_batch_single_img(model, processor, images, prompt_texts, max_new_tokens=260):
    chat_texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompt_texts
    ]
    inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt").to("cuda")
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)


@torch.inference_mode()
def generate_one_multi_img(model, processor, images, content_blocks, max_new_tokens=320):
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": content_blocks}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_text], images=images, return_tensors="pt").to("cuda")
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0]


@torch.inference_mode()
def extract_visual_attention(model, processor, image: Image.Image, prompt_text: str):
    """Extract layer-averaged cross-attention from the final decision prompt token to every
    merged visual patch token, reshaped onto the exact (grid_h, grid_w) spatial grid.
    """
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to("cuda")
    outputs = model(**inputs, output_attentions=True, use_cache=False)
    attns = outputs.attentions
    if not attns or attns[-1] is None:
        raise RuntimeError("Model returned None attentions; eager attention implementation did not activate!")

    input_ids = inputs["input_ids"][0]
    img_tok_id = getattr(model.config, "image_token_id", None)
    if img_tok_id is None:
        img_tok_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vis_mask = (input_ids == img_tok_id)
    vis_idx = torch.where(vis_mask)[0]

    grid_thw = inputs["image_grid_thw"][0].tolist()  # [T, H_vit, W_vit]
    merge = getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", 2)
    gh, gw = int(grid_thw[1]) // int(merge), int(grid_thw[2]) // int(merge)
    if len(vis_idx) != gh * gw:
        raise RuntimeError(f"Visual token count {len(vis_idx)} != {gh}*{gw} (grid_thw={grid_thw}, merge={merge})")

    # Average attention from the final prompt token (the exact position generating the answer!)
    # across middle-to-late layers (second half of LM layers) and all attention heads.
    n_layers = len(attns)
    start_l = n_layers // 2
    vis_scores = torch.zeros(len(vis_idx), dtype=torch.float32, device="cuda")
    total_vis_share = 0.0
    for l_idx in range(start_l, n_layers):
        # attns[l_idx]: [1, num_heads, seq_len, seq_len]
        last_tok_attn = attns[l_idx][0, :, -1, :].float().mean(dim=0)  # [seq_len]
        va = last_tok_attn[vis_idx]
        vis_scores += va
        total_vis_share += float(va.sum().item())
    n_used = n_layers - start_l
    vis_scores /= n_used
    total_vis_share /= n_used

    grid = vis_scores.reshape(gh, gw).cpu().numpy()
    grid_norm = grid / max(1e-12, float(grid.sum()))
    return {
        "grid_h": gh,
        "grid_w": gw,
        "total_vis_share": round(total_vis_share, 5),
        "grid_norm": [[round(float(v), 6) for v in row] for row in grid_norm],
    }


def compute_box_attention_stats(grid_norm_2d, boxes):
    arr = np.array(grid_norm_2d, dtype=np.float64)
    gh, gw = arr.shape
    results = []
    union_mask = np.zeros((gh, gw), dtype=bool)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox_1000"]
        c1 = max(0, min(gw - 1, int(np.floor(x1 / 1000.0 * gw))))
        c2 = max(c1 + 1, min(gw, int(np.ceil(x2 / 1000.0 * gw))))
        r1 = max(0, min(gh - 1, int(np.floor(y1 / 1000.0 * gh))))
        r2 = max(r1 + 1, min(gh, int(np.ceil(y2 / 1000.0 * gh))))
        mask = np.zeros((gh, gw), dtype=bool)
        mask[r1:r2, c1:c2] = True
        union_mask |= mask
        area_frac = float(mask.sum()) / float(gh * gw)
        attn_frac = float(arr[mask].sum())
        conc = attn_frac / max(1e-9, area_frac)
        results.append({
            "label": b["label"],
            "area_frac": round(area_frac, 4),
            "attn_frac": round(attn_frac, 4),
            "concentration": round(conc, 3),
        })
    union_area = float(union_mask.sum()) / float(gh * gw) if boxes else 0.0
    union_attn = float(arr[union_mask].sum()) if boxes else 0.0
    union_conc = (union_attn / union_area) if union_area > 0 else None
    return {
        "per_box": results,
        "union_area_frac": round(union_area, 4),
        "union_attn_frac": round(union_attn, 4),
        "union_concentration": round(union_conc, 3) if union_conc is not None else None,
    }


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
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    with ZipFile(args.input_dir / "author-holdout-eval.zip") as z:
        for r in ah_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"AH SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(ah_rows)} author_holdout_66 images.", flush=True)

    model, processor, rev = load_model(MODEL_ID)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bs = args.batch_size if total_vram_gb >= 20.0 else 4

    # =========================================================================
    # PART 1: Native Visual Grounding on all 66 images
    # =========================================================================
    print(f"\n=== Part 1/4: Native Visual Grounding on {len(ah_rows)} images (bs={bs}) ===", flush=True)
    t0 = time.monotonic()
    grounding_rows = []
    idx = 0
    while idx < len(ah_rows):
        grp = ah_rows[idx:idx + bs]
        imgs = [load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768) for r in grp]
        prompts = [GROUNDING_PROMPT] * len(grp)
        raws = generate_batch_single_img(model, processor, imgs, prompts, max_new_tokens=260)
        for r, im, raw in zip(grp, imgs, raws):
            boxes = parse_grounding_boxes(raw, im.size[0], im.size[1])
            grounding_rows.append({
                "id": r["id"],
                "true_country": r["country"],
                "width": im.size[0],
                "height": im.size[1],
                "boxes": boxes,
                "raw": raw,
            })
        idx += len(grp)
    (args.output_dir / "grounding_boxes_ah66.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in grounding_rows), encoding="utf-8"
    )
    from collections import Counter
    lbl_cnt = Counter(b["label"] for x in grounding_rows for b in x["boxes"])
    with_boxes = sum(1 for x in grounding_rows if x["boxes"])
    print(f"  Part 1 done in {time.monotonic()-t0:.1f}s: {with_boxes}/66 images have >=1 box, total={sum(lbl_cnt.values())} | {dict(lbl_cnt)}", flush=True)

    # =========================================================================
    # PART 2: Exact Visual Cross-Attention Heatmaps (Blind vs KDE prompt)
    # =========================================================================
    print("\n=== Part 2/4: Extracting LM Visual Cross-Attention Heatmaps (eager mode, all 66 images) ===", flush=True)
    t1 = time.monotonic()
    set_lm_attn_impl(model, "eager")
    processor.tokenizer.padding_side = "right"
    blind_prompt = "Identify the country where this street-level photo was taken. Respond with ONLY the 2-letter ISO country code:"

    attn_rows = []
    for r, g in zip(ah_rows, grounding_rows):
        im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        kde_prompt = build_multiscale_cot_prompt(r["kde_distribution"])
        a_blind = extract_visual_attention(model, processor, im, blind_prompt)
        a_kde = extract_visual_attention(model, processor, im, kde_prompt)
        s_blind = compute_box_attention_stats(a_blind["grid_norm"], g["boxes"])
        s_kde = compute_box_attention_stats(a_kde["grid_norm"], g["boxes"])
        attn_rows.append({
            "id": r["id"],
            "true_country": r["country"],
            "n_boxes": len(g["boxes"]),
            "blind": {**a_blind, "box_stats": s_blind},
            "kde": {**a_kde, "box_stats": s_kde},
        })
    (args.output_dir / "attention_maps_ah66.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in attn_rows), encoding="utf-8"
    )
    # Switch back to fast SDPA for Parts 3 & 4!
    set_lm_attn_impl(model, "sdpa")
    processor.tokenizer.padding_side = "left"
    torch.cuda.empty_cache()

    b_concs = [x["blind"]["box_stats"]["union_concentration"] for x in attn_rows if x["blind"]["box_stats"]["union_concentration"] is not None]
    k_concs = [x["kde"]["box_stats"]["union_concentration"] for x in attn_rows if x["kde"]["box_stats"]["union_concentration"] is not None]
    print(
        f"  Part 2 done in {time.monotonic()-t1:.1f}s | mean visual share: blind={np.mean([x['blind']['total_vis_share'] for x in attn_rows]):.3f}, "
        f"kde={np.mean([x['kde']['total_vis_share'] for x in attn_rows]):.3f} | "
        f"mean cue-box concentration: blind={np.mean(b_concs):.2f}x, kde={np.mean(k_concs):.2f}x",
        flush=True,
    )

    # =========================================================================
    # PART 3: Object-Isolated Cue Probe (Crop-Only, ZERO Background!)
    # =========================================================================
    print("\n=== Part 3/4: Object-Isolated Cue-Only Country Probe (zero background) ===", flush=True)
    t2 = time.monotonic()
    probe_items = []
    for r, g in zip(ah_rows, grounding_rows):
        if not g["boxes"]:
            continue
        im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        # Distinct candidate countries from Top-8 KDE
        seen_c = []
        for cl in r["kde_distribution"]["clusters"]:
            if cl["country"] not in seen_c:
                seen_c.append(cl["country"])
        cand_str = ", ".join(f"{cc_names.get(c, c)} ({c})" for c in seen_c[:6])

        # Select up to 2 highest-priority distinct-label cues per image
        sorted_b = sorted(g["boxes"], key=lambda b: (CUE_PRIORITY.get(b["label"], 9), -(b["bbox_1000"][2]-b["bbox_1000"][0])*(b["bbox_1000"][3]-b["bbox_1000"][1])))
        picked = []
        used_lbl = set()
        for b in sorted_b:
            if b["label"] not in used_lbl or len(picked) < 1:
                picked.append(b)
                used_lbl.add(b["label"])
            if len(picked) >= 2:
                break

        for b in picked:
            crop_im = crop_cue(im, b["bbox_1000"])
            p_txt = (
                f"You are viewing ONLY a zoomed-in crop of a `{b['label']}` from a street-level photo (no surrounding scenery).\n"
                f"Candidate countries: {cand_str}.\n"
                "1. Describe exact fine-grained diagnostic details of this object (e.g. text/script, pole material/holes/crossarm, "
                "bollard reflector shape/color, sign font/border, road line color/pattern, plate strip).\n"
                "2. Decide whether this isolated object alone is diagnostic of one of the candidate countries.\n"
                "Respond with ONLY one JSON object:\n"
                '{"detail": "<concise 10-20 word description>", "diagnostic": <true|false>, '
                '"best_country": "<2-letter ISO code or NONE>", "confidence": "<high|medium|low>"}'
            )
            probe_items.append((r, b, crop_im, p_txt, seen_c))

    crop_probe_rows = []
    idx = 0
    while idx < len(probe_items):
        grp = probe_items[idx:idx + bs]
        raws = generate_batch_single_img(model, processor, [x[2] for x in grp], [x[3] for x in grp], max_new_tokens=140)
        for (r, b, _, _, seen_c), raw in zip(grp, raws):
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            obj = {}
            try:
                m = re.search(r"\{.*\}", cleaned, re.S)
                if m:
                    obj = json.loads(m.group(0))
            except ValueError:
                pass
            pred_cc = normalize_country_code(obj.get("best_country"), cc_names)
            crop_probe_rows.append({
                "id": r["id"],
                "true_country": r["country"],
                "true_in_candidates": r["country"] in seen_c[:6],
                "top1_country": seen_c[0],
                "label": b["label"],
                "bbox_1000": b["bbox_1000"],
                "detail": obj.get("detail"),
                "diagnostic": bool(obj.get("diagnostic", False)),
                "pred_country": pred_cc,
                "confidence": str(obj.get("confidence", "low")).lower(),
                "correct": pred_cc == r["country"],
                "raw": raw,
            })
        idx += len(grp)

    (args.output_dir / "crop_probe_ah66.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in crop_probe_rows), encoding="utf-8"
    )
    diag_rows = [x for x in crop_probe_rows if x["diagnostic"] and x["pred_country"]]
    diag_ok = sum(x["correct"] for x in diag_rows)
    print(
        f"  Part 3 done in {time.monotonic()-t2:.1f}s: {len(crop_probe_rows)} isolated crops probed | "
        f"marked diagnostic={len(diag_rows)}, correct={diag_ok}/{len(diag_rows)} ({100*diag_ok/max(1,len(diag_rows)):.1f}%)",
        flush=True,
    )

    # =========================================================================
    # PART 4: Visual Fovea Injection ([Full Image + Grounded High-Res Crops])
    # =========================================================================
    print("\n=== Part 4/4: Visual Fovea Multi-Image Top-8 Test on author_holdout_66 (zero training) ===", flush=True)
    t3 = time.monotonic()
    fovea_preds = []
    for r, g in zip(ah_rows, grounding_rows):
        full_im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        sorted_b = sorted(g["boxes"], key=lambda b: (CUE_PRIORITY.get(b["label"], 9), -(b["bbox_1000"][2]-b["bbox_1000"][0])*(b["bbox_1000"][3]-b["bbox_1000"][1])))
        sel_boxes = sorted_b[:2]

        imgs = [full_im]
        blocks = [{"type": "image"}]
        crop_notes = []
        for ci, b in enumerate(sel_boxes, 1):
            imgs.append(crop_cue(full_im, b["bbox_1000"]))
            blocks.append({"type": "image"})
            crop_notes.append(f"Zoomed Crop #{ci}: `{b['label']}` ({b['note']})")

        base_prompt = build_multiscale_cot_prompt(r["kde_distribution"])
        if crop_notes:
            prefix = (
                "Image 1 is the full street-level photo. Additional zoomed-in crops of key detected cues are attached:\n"
                + "\n".join(f"  - {cn}" for cn in crop_notes)
                + "\nInspect these zoomed crops closely (especially text/script, sign design, pole structure, bollards, plates, road markings) before deciding.\n\n"
            )
            full_prompt = prefix + base_prompt
        else:
            full_prompt = base_prompt
        blocks.append({"type": "text", "text": full_prompt})

        raw = generate_one_multi_img(model, processor, imgs, blocks, max_new_tokens=320)
        p = parse_multiscale_json(raw, cc_names)
        coords = p["coordinates"]
        dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
        fovea_preds.append({
            "id": r["id"],
            "true_country": r["country"],
            "true_latitude": r["latitude"],
            "true_longitude": r["longitude"],
            "n_crops": len(sel_boxes),
            "crop_labels": [b["label"] for b in sel_boxes],
            "guess_country": p["country"],
            "guess_coordinates": list(coords) if coords else None,
            "distance_km": dist,
            "selected_cluster": p["selected_cluster"],
            "confidence": p["confidence"],
            "visual_clues": p["visual_clues"],
            "reasoning": p["reasoning"],
            "parsed": p["parsed"],
            "raw": raw,
        })

    (args.output_dir / "fovea_8b_author_holdout_top8.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in fovea_preds), encoding="utf-8"
    )
    valid_d = [x["distance_km"] for x in fovea_preds if x["distance_km"] is not None]
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in fovea_preds)
    fovea_summary = {
        "n": len(fovea_preds),
        "parsed": sum(1 for x in fovea_preds if x["parsed"]),
        "country_correct": cc_ok,
        "country_acc_pct": round(100.0 * cc_ok / len(fovea_preds), 2),
        "le_25km": int(sum(d <= 25.0 for d in valid_d)),
        "le_200km": int(sum(d <= 200.0 for d in valid_d)),
        "median_km": round(float(np.median(valid_d)), 1),
        "geoscore": round(float(sum(geoscore(d) for d in valid_d) / len(fovea_preds)), 1),
    }
    print(f"  Part 4 done in {time.monotonic()-t3:.1f}s | Visual Fovea Base 8B (Top-8): {json.dumps(fovea_summary)}", flush=True)

    summary = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "grounding_counts": dict(lbl_cnt),
        "images_with_boxes": with_boxes,
        "attention_blind_mean_vis_share": round(float(np.mean([x["blind"]["total_vis_share"] for x in attn_rows])), 4),
        "attention_kde_mean_vis_share": round(float(np.mean([x["kde"]["total_vis_share"] for x in attn_rows])), 4),
        "attention_blind_mean_box_concentration": round(float(np.mean(b_concs)), 3) if b_concs else None,
        "attention_kde_mean_box_concentration": round(float(np.mean(k_concs)), 3) if k_concs else None,
        "crop_probe_total": len(crop_probe_rows),
        "crop_probe_diagnostic_n": len(diag_rows),
        "crop_probe_diagnostic_acc_pct": round(100.0 * diag_ok / max(1, len(diag_rows)), 2),
        "fovea_top8_author_holdout_66": fovea_summary,
    }
    (args.output_dir / "step0_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted Step 0 Grounding + Attention + Crop-Probe + Visual Fovea diagnostic!", flush=True)


if __name__ == "__main__":
    main()
