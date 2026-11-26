"""Step 1 Mechanistic Experiment (Colab GPU):
Category-Agnostic Discriminative Patch-MIL Discovery + Image-Last Attention-Sink Recovery
+ Open-Ended Diagnostic Grounding on Qwen/Qwen3-VL-8B-Instruct (zero LM fine-tuning!).

Tests 4 concrete mechanistic questions on the SAME frozen 8B VLM:
  1. Patch-MIL Discovery (Zero Fixed Object Classes):
     Extracts Qwen3-VL-8B's own frozen merged visual patch tokens (H_g x W_g x D) across
     2,000 creator-disjoint opt_train images (+ 300 tune validation images) and trains a
     lightweight Gated Attention MIL country classifier (with patch dropout + anti-sink reg)
     to discover data-driven high-mutual-information regions of ANY type (roofs, vehicles,
     guardrails, poles, curbs, vegetation, signs, etc.).
  2. Attention-Sink Recovery (Image-First vs Image-Last):
     Measures exact LM Layer 18-35 visual cross-attention on all 66 author_holdout_66 images
     when placing the image AFTER the Top-8 KDE table (`Image-Last`) vs BEFORE (`Image-First`),
     testing whether causal recency decay caused the 7.9x attention collapse (6.87% -> 0.87%).
  3. Open-Ended Diagnostic Grounding (Zero 6-Category Limit):
     Asks Qwen3-VL-8B to locate the 2-3 most geographically distinctive regions of ANY kind
     to distinguish the competing KDE candidates.
  4. Apples-to-Apples Top-8 Multi-Scale Evaluation on author_holdout_66:
     Compares (a) Image-Last Full-Only, (b) Image-Last + MIL-Fovea (top-2 data-driven crops),
     and (c) Image-Last + Open-Ended Grounding Fovea against Step 0 baselines.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parse_multiscale_json,
)
from run_step0_grounding_attention import (
    compute_box_attention_stats,
    crop_cue,
    load_clean_image,
    load_model,
    set_lm_attn_impl,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class GatedPatchMIL(nn.Module):
    """Category-agnostic Gated Attention MIL over frozen Qwen3-VL visual patch vectors.
    Computes additive instance-level country logits weighted by gated attention so every
    spatial patch has an exact, interpretable discriminative saliency score.
    """
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, attn_dim: int = 128, dropout: float = 0.35):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Linear(hidden, attn_dim)
        self.attn_u = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.cls_head = nn.Linear(hidden, num_classes)

    def forward_one(self, patches: torch.Tensor, patch_dropout: float = 0.0):
        # patches: [N, D]
        n = patches.shape[0]
        h = self.proj(patches)  # [N, H]
        raw_a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)  # [N]
        if self.training and patch_dropout > 0.0 and n > 8:
            keep = torch.rand(n, device=patches.device) > patch_dropout
            if keep.sum() < 4:
                keep[:4] = True
            raw_a = raw_a.masked_fill(~keep, -1e9)
        attn = F.softmax(raw_a, dim=0)  # [N]
        patch_logits = self.cls_head(h)  # [N, C]
        bag_logits = torch.sum(attn.unsqueeze(-1) * patch_logits, dim=0)  # [C]
        return bag_logits, attn, patch_logits


def get_visual_module(model):
    if hasattr(model, "visual") and model.visual is not None:
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual") and model.model.visual is not None:
        return model.model.visual
    raise RuntimeError("Could not locate visual encoder on Qwen3VLForConditionalGeneration")


@torch.inference_mode()
def extract_qwen_visual_patches(model, processor, img_paths, batch_size: int = 16, max_edge: int = 512):
    """Extract frozen merged Qwen3-VL visual patch tokens (in LM embedding space) without running the LM."""
    vis_mod = get_visual_module(model)
    merge = int(getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", 2))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    out_patches = []
    out_grids = []
    t0 = time.monotonic()

    for start in range(0, len(img_paths), batch_size):
        grp_paths = img_paths[start:start + batch_size]
        imgs = [load_clean_image(p, max_edge=max_edge) for p in grp_paths]
        # Use image_processor directly (no text tokenization needed!)
        proc_out = processor.image_processor(images=imgs, return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to("cuda", dtype=dtype)
        grid_thw = proc_out["image_grid_thw"].to("cuda")

        vis_out = vis_mod(pixel_values, grid_thw=grid_thw)
        if isinstance(vis_out, tuple):
            vis_out = vis_out[0]
        elif hasattr(vis_out, "pooler_output") and vis_out.pooler_output is not None:
            vis_out = vis_out.pooler_output
        elif hasattr(vis_out, "last_hidden_state"):
            vis_out = vis_out.last_hidden_state
        # vis_out: [total_merged_tokens, D]
        split_sizes = [int(t * h * w // (merge * merge)) for t, h, w in grid_thw.tolist()]
        if sum(split_sizes) != vis_out.shape[0]:
            raise RuntimeError(f"Split mismatch: sum({split_sizes}) != {vis_out.shape[0]}")
        chunks = torch.split(vis_out.to(torch.float16).cpu(), split_sizes, dim=0)
        for ch, (t, h, w) in zip(chunks, grid_thw.tolist()):
            out_patches.append(ch)
            out_grids.append((int(h) // merge, int(w) // merge))

        done_n = min(start + len(grp_paths), len(img_paths))
        if start == 0 or done_n % 400 < batch_size or done_n >= len(img_paths):
            rate = done_n / max(1e-3, time.monotonic() - t0)
            print(f"  ViT patch extraction: {done_n:4d}/{len(img_paths)} | {rate:.1f} img/s", flush=True)
    return out_patches, out_grids


def train_gated_mil(train_patches, train_y, val_patches, val_y, num_classes: int, epochs: int = 18, lr: float = 1e-3):
    in_dim = train_patches[0].shape[-1]
    mil = GatedPatchMIL(in_dim=in_dim, num_classes=num_classes, hidden=512, attn_dim=128, dropout=0.35).cuda()
    opt = torch.optim.AdamW(mil.parameters(), lr=lr, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_acc = -1.0
    best_state = None
    rng = random.Random(42)
    n_train = len(train_patches)
    bs = 32

    print(f"\n=== Training Gated Patch-MIL on {n_train} opt_train images ({num_classes} classes, val={len(val_patches)}) ===", flush=True)
    for ep in range(1, epochs + 1):
        mil.train()
        order = list(range(n_train))
        rng.shuffle(order)
        ep_loss = []
        for s in range(0, n_train, bs):
            idxs = order[s:s + bs]
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for idx in idxs:
                p = train_patches[idx].to("cuda", dtype=torch.float32)
                y = torch.tensor(train_y[idx], dtype=torch.long, device="cuda")
                logits, attn, _ = mil.forward_one(p, patch_dropout=0.25)
                ce = F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0), label_smoothing=0.05)
                # Anti-sink L2 penalty prevents collapsing 100% weight onto a single corner pixel
                anti_sink = 0.15 * torch.sum(attn ** 2)
                batch_loss = batch_loss + (ce + anti_sink)
            (batch_loss / len(idxs)).backward()
            torch.nn.utils.clip_grad_norm_(mil.parameters(), 2.0)
            opt.step()
            ep_loss.append(float((batch_loss / len(idxs)).detach().item()))
        sched.step()

        # Evaluate on creator-disjoint tune split
        mil.eval()
        val_ok, val_top5 = 0, 0
        val_max_a = []
        with torch.inference_mode():
            for p_cpu, y_i in zip(val_patches, val_y):
                if y_i < 0:
                    continue
                p = p_cpu.to("cuda", dtype=torch.float32)
                logits, attn, _ = mil.forward_one(p, patch_dropout=0.0)
                top5 = torch.topk(logits, min(5, num_classes)).indices.tolist()
                if top5[0] == y_i:
                    val_ok += 1
                if y_i in top5:
                    val_top5 += 1
                val_max_a.append(float(attn.max().item()))
        n_valid_val = sum(1 for y_i in val_y if y_i >= 0)
        val_acc = val_ok / max(1, n_valid_val)
        val_t5 = val_top5 / max(1, n_valid_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in mil.state_dict().items()}
        if ep in (1, 5, 10, 15, epochs):
            print(
                f"  [MIL ep {ep:2d}/{epochs}] train_loss={np.mean(ep_loss):.3f} | "
                f"tune_top1={100*val_acc:.1f}% ({val_ok}/{n_valid_val}) | tune_top5={100*val_t5:.1f}% | "
                f"mean_max_patch_attn={np.mean(val_max_a):.3f}",
                flush=True,
            )

    mil.load_state_dict(best_state)
    mil.eval()
    return mil, round(100.0 * best_val_acc, 2)


def extract_top2_mil_boxes(saliency_2d: np.ndarray, box_frac_w: float = 0.28, box_frac_h: float = 0.32):
    """Extract 2 non-overlapping high-MI spatial crop boxes (normalized [0..1000]) centered on the
    top-2 discriminative saliency peaks discovered by Gated Patch-MIL (zero fixed categories!).
    """
    gh, gw = saliency_2d.shape
    sal = saliency_2d.copy()
    # Smooth slightly with 3x3 box kernel so crops center on coherent local objects/structures
    pad = np.pad(sal, 1, mode="edge")
    smooth = sum(
        pad[dr:dr + gh, dc:dc + gw]
        for dr in range(3) for dc in range(3)
    ) / 9.0

    boxes = []
    bw = box_frac_w * 1000.0
    bh = box_frac_h * 1000.0
    work = smooth.copy()
    for rank in range(2):
        flat_idx = int(np.argmax(work))
        pr, pc = divmod(flat_idx, gw)
        score = float(work[pr, pc])
        if score <= 0:
            break
        cx = (pc + 0.5) / gw * 1000.0
        cy = (pr + 0.5) / gh * 1000.0
        x1 = max(0.0, min(1000.0 - bw, cx - bw / 2.0))
        y1 = max(0.0, min(1000.0 - bh, cy - bh / 2.0))
        x2 = min(1000.0, x1 + bw)
        y2 = min(1000.0, y1 + bh)
        boxes.append({
            "label": f"mil_hotspot_{rank+1}",
            "bbox_1000": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "peak_rc": [int(pr), int(pc)],
            "score": round(score, 5),
        })
        # Suppress neighborhood around first peak so second crop inspects a genuinely distinct clue region
        r_lo = max(0, pr - max(2, gh // 4))
        r_hi = min(gh, pr + max(2, gh // 4) + 1)
        c_lo = max(0, pc - max(2, gw // 4))
        c_hi = min(gw, pc + max(2, gw // 4) + 1)
        work[r_lo:r_hi, c_lo:c_hi] = -1.0
    return boxes


def build_image_last_content(kde_prompt: str, extra_crops=None, crop_notes=None):
    """Place the long Top-8 KDE table BEFORE the image(s), and put a crisp short decision reminder
    right after the image(s) so visual patch tokens sit immediately adjacent to the decision token!
    """
    # Split build_multiscale_cot_prompt into candidate header + short decision instruction
    marker = "Reason step by step"
    if marker in kde_prompt:
        cand_part, instr_part = kde_prompt.split(marker, 1)
        instr_part = marker + instr_part
    else:
        cand_part, instr_part = kde_prompt, ""

    header = cand_part.strip() + "\n\nNow carefully inspect the street-level photo below"
    if extra_crops:
        header += " along with the zoomed-in high-information detail crops:\n" + "\n".join(f"  - {cn}" for cn in crop_notes)
    else:
        header += ":"

    blocks = [{"type": "text", "text": header}, {"type": "image"}]
    if extra_crops:
        for _ in extra_crops:
            blocks.append({"type": "image"})
    blocks.append({"type": "text", "text": "\n" + instr_part.strip()})
    return blocks


@torch.inference_mode()
def extract_visual_attention_blocks(model, processor, images, content_blocks):
    """Extract Layer 18-35 visual cross-attention for the FIRST image in arbitrary content_blocks."""
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": content_blocks}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_text], images=images, return_tensors="pt").to("cuda")
    outputs = model(**inputs, output_attentions=True, use_cache=False)
    attns = outputs.attentions
    if not attns or attns[-1] is None:
        raise RuntimeError("Attentions returned None in eager mode")

    input_ids = inputs["input_ids"][0]
    img_tok_id = getattr(model.config, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vis_idx = torch.where(input_ids == img_tok_id)[0]

    grid_thw = inputs["image_grid_thw"][0].tolist()
    merge = int(getattr(getattr(model.config, "vision_config", None), "spatial_merge_size", 2))
    gh, gw = int(grid_thw[1]) // merge, int(grid_thw[2]) // merge
    first_n = gh * gw
    first_vis_idx = vis_idx[:first_n]

    n_layers = len(attns)
    start_l = n_layers // 2
    vis_scores = torch.zeros(first_n, dtype=torch.float32, device="cuda")
    total_vis_share = 0.0
    for l_idx in range(start_l, n_layers):
        last_tok_attn = attns[l_idx][0, :, -1, :].float().mean(dim=0)
        va = last_tok_attn[first_vis_idx]
        vis_scores += va
        total_vis_share += float(last_tok_attn[vis_idx].sum().item())
    n_used = n_layers - start_l
    vis_scores /= n_used
    total_vis_share /= n_used

    grid = vis_scores.reshape(gh, gw).cpu().numpy()
    grid_norm = grid / max(1e-12, float(grid.sum()))
    return {
        "grid_h": gh,
        "grid_w": gw,
        "total_vis_share": round(total_vis_share, 5),
        "top_left_corner_share": round(float(grid_norm[0, 0]), 5),
        "grid_norm": [[round(float(v), 6) for v in row] for row in grid_norm],
    }


@torch.inference_mode()
def generate_one_blocks(model, processor, images, content_blocks, max_new_tokens=320):
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": content_blocks}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_text], images=images, return_tensors="pt").to("cuda")
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0]


def parse_open_boxes(raw_text: str):
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
    if not arr:
        return []
    boxes = []
    for item in arr[:3]:
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
        if x2 <= x1 + 10.0 or y2 <= y1 + 10.0:
            continue
        boxes.append({
            "label": str(item.get("label", "clue")).strip().lower()[:30],
            "bbox_1000": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "why": str(item.get("why", "")).strip()[:80],
        })
    return boxes


def summarize_preds(label, preds):
    valid_d = [x["distance_km"] for x in preds if x["distance_km"] is not None]
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in preds)
    s = {
        "label": label,
        "n": len(preds),
        "parsed": sum(1 for x in preds if x["parsed"]),
        "country_correct": cc_ok,
        "country_acc_pct": round(100.0 * cc_ok / len(preds), 2),
        "le_25km": int(sum(d <= 25.0 for d in valid_d)),
        "le_200km": int(sum(d <= 200.0 for d in valid_d)),
        "median_km": round(float(np.median(valid_d)), 1),
        "geoscore": round(float(sum(geoscore(d) for d in valid_d) / len(preds)), 1),
        "sel_top1": sum(1 for x in preds if x["selected_cluster"] == 1),
    }
    print(f"  >>> {label}: {json.dumps(s)}", flush=True)
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    mil_train = [json.loads(l) for l in (args.input_dir / "mil_train_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    mil_tune = [json.loads(l) for l in (args.input_dir / "mil_tune_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    step0_g = {
        r["id"]: r
        for r in (json.loads(l) for l in (args.input_dir / "grounding_boxes_ah66.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }

    with ZipFile(args.input_dir / "mil_imgs.zip") as z:
        for r in mil_train + mil_tune:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"MIL img SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    with ZipFile(args.input_dir / "author-holdout-eval.zip") as z:
        for r in ah_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                    raise ValueError(f"AH SHA mismatch: {r['id']}")
                dst.write_bytes(payload)
    print(f"Verified {len(mil_train)} MIL train + {len(mil_tune)} MIL tune + {len(ah_rows)} author_holdout images.", flush=True)

    model, processor, rev = load_model(MODEL_ID)

    # =========================================================================
    # PART 1: Category-Agnostic Gated Patch-MIL on Qwen3-VL's OWN Visual Tokens
    # =========================================================================
    print("\n=== Part 1/4: Extracting Qwen3-VL Frozen Visual Patches & Training Gated Patch-MIL ===", flush=True)
    cc_vocab = sorted({r["country"] for r in mil_train})
    cc_to_i = {c: i for i, c in enumerate(cc_vocab)}

    tr_paths = [img_dir / f"{r['id']}.jpg" for r in mil_train]
    tu_paths = [img_dir / f"{r['id']}.jpg" for r in mil_tune]
    ah_paths = [img_dir / f"{r['id']}.jpg" for r in ah_rows]

    tr_patches, _ = extract_qwen_visual_patches(model, processor, tr_paths, batch_size=16, max_edge=512)
    tu_patches, _ = extract_qwen_visual_patches(model, processor, tu_paths, batch_size=16, max_edge=512)
    ah_patches, ah_grids = extract_qwen_visual_patches(model, processor, ah_paths, batch_size=16, max_edge=512)

    tr_y = [cc_to_i[r["country"]] for r in mil_train]
    tu_y = [cc_to_i.get(r["country"], -1) for r in mil_tune]

    mil, tune_top1_pct = train_gated_mil(tr_patches, tr_y, tu_patches, tu_y, len(cc_vocab), epochs=18, lr=1e-3)
    # Free training patch RAM immediately
    del tr_patches, tu_patches
    gc.collect()

    # Compute Patch-MIL predictions & discriminative saliency maps on author_holdout_66
    mil_ah_rows = []
    ah_mil_ok = 0
    ah_mil_top5 = 0
    with torch.inference_mode():
        for r, p_cpu, (gh, gw) in zip(ah_rows, ah_patches, ah_grids):
            p = p_cpu.to("cuda", dtype=torch.float32)
            bag_logits, attn, patch_logits = mil.forward_one(p, patch_dropout=0.0)
            bag_probs = F.softmax(bag_logits, dim=0)
            top5_i = torch.topk(bag_probs, 5).indices.tolist()
            top5_cc = [(cc_vocab[i], round(float(bag_probs[i].item()), 4)) for i in top5_i]
            if top5_cc[0][0] == r["country"]:
                ah_mil_ok += 1
            if any(c == r["country"] for c, _ in top5_cc):
                ah_mil_top5 += 1

            # Discriminative saliency per patch: attn * (H_max - H(patch_probs))
            # Highlights patches that BOTH get high MIL attention AND have sharp country certainty!
            p_probs = F.softmax(patch_logits, dim=-1)
            p_ent = -(p_probs * torch.log(p_probs.clamp(min=1e-9))).sum(dim=-1)
            h_max = math.log(len(cc_vocab))
            sharpness = (h_max - p_ent).clamp(min=0.0)
            sal = (attn * sharpness).reshape(gh, gw).cpu().numpy()
            sal_norm = sal / max(1e-12, float(sal.sum()))
            attn_2d = attn.reshape(gh, gw).cpu().numpy()
            mil_boxes = extract_top2_mil_boxes(sal_norm)

            mil_ah_rows.append({
                "id": r["id"],
                "true_country": r["country"],
                "mil_top5": top5_cc,
                "grid_h": gh,
                "grid_w": gw,
                "mil_saliency_norm": [[round(float(v), 6) for v in row] for row in sal_norm],
                "mil_attn_2d": [[round(float(v), 6) for v in row] for row in attn_2d],
                "mil_boxes": mil_boxes,
            })

    (args.output_dir / "mil_saliency_ah66.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mil_ah_rows), encoding="utf-8"
    )
    print(
        f"  Patch-MIL on author_holdout_66 (from Qwen3-VL's OWN frozen patches, zero KDE!): "
        f"Top-1={ah_mil_ok}/66 ({100*ah_mil_ok/66:.1f}%), Top-5={ah_mil_top5}/66 ({100*ah_mil_top5/66:.1f}%)",
        flush=True,
    )
    del mil, ah_patches
    torch.cuda.empty_cache()

    # =========================================================================
    # PART 2: Attention-Sink Test — Image-First vs Image-Last Eager Attention
    # =========================================================================
    print("\n=== Part 2/4: Comparing Image-First vs Image-Last Cross-Attention on all 66 images ===", flush=True)
    t1 = time.monotonic()
    set_lm_attn_impl(model, "eager")
    processor.tokenizer.padding_side = "right"
    attn_cmp_rows = []
    for r, mrow in zip(ah_rows, mil_ah_rows):
        im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        kde_p = build_multiscale_cot_prompt(r["kde_distribution"])
        blocks_first = [{"type": "image"}, {"type": "text", "text": kde_p}]
        blocks_last = build_image_last_content(kde_p)
        a_first = extract_visual_attention_blocks(model, processor, [im], blocks_first)
        a_last = extract_visual_attention_blocks(model, processor, [im], blocks_last)

        g_boxes = step0_g.get(r["id"], {}).get("boxes", [])
        m_boxes = mrow["mil_boxes"]
        attn_cmp_rows.append({
            "id": r["id"],
            "true_country": r["country"],
            "img_first": {
                **a_first,
                "cue_box_stats": compute_box_attention_stats(a_first["grid_norm"], g_boxes),
                "mil_box_stats": compute_box_attention_stats(a_first["grid_norm"], m_boxes),
            },
            "img_last": {
                **a_last,
                "cue_box_stats": compute_box_attention_stats(a_last["grid_norm"], g_boxes),
                "mil_box_stats": compute_box_attention_stats(a_last["grid_norm"], m_boxes),
            },
        })
    (args.output_dir / "attention_first_vs_last_ah66.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in attn_cmp_rows), encoding="utf-8"
    )
    set_lm_attn_impl(model, "sdpa")
    processor.tokenizer.padding_side = "left"
    torch.cuda.empty_cache()

    f_vis = np.mean([x["img_first"]["total_vis_share"] for x in attn_cmp_rows])
    l_vis = np.mean([x["img_last"]["total_vis_share"] for x in attn_cmp_rows])
    f_tl = np.mean([x["img_first"]["top_left_corner_share"] for x in attn_cmp_rows])
    l_tl = np.mean([x["img_last"]["top_left_corner_share"] for x in attn_cmp_rows])
    f_cue = np.mean([x["img_first"]["cue_box_stats"]["union_concentration"] for x in attn_cmp_rows if x["img_first"]["cue_box_stats"]["union_concentration"] is not None])
    l_cue = np.mean([x["img_last"]["cue_box_stats"]["union_concentration"] for x in attn_cmp_rows if x["img_last"]["cue_box_stats"]["union_concentration"] is not None])
    f_mil = np.mean([x["img_first"]["mil_box_stats"]["union_concentration"] for x in attn_cmp_rows if x["img_first"]["mil_box_stats"]["union_concentration"] is not None])
    l_mil = np.mean([x["img_last"]["mil_box_stats"]["union_concentration"] for x in attn_cmp_rows if x["img_last"]["mil_box_stats"]["union_concentration"] is not None])
    print(
        f"  Part 2 done in {time.monotonic()-t1:.1f}s:\n"
        f"    Image-First KDE: vis_share={100*f_vis:.2f}%, top_left_sink={100*f_tl:.1f}%, cue_conc={f_cue:.2f}x, mil_conc={f_mil:.2f}x\n"
        f"    Image-Last  KDE: vis_share={100*l_vis:.2f}%, top_left_sink={100*l_tl:.1f}%, cue_conc={l_cue:.2f}x, mil_conc={l_mil:.2f}x",
        flush=True,
    )

    # =========================================================================
    # PART 3: Open-Ended Hypothesis-Driven Grounding (ZERO 6-Class Limit!)
    # =========================================================================
    print("\n=== Part 3/4: Open-Ended Hypothesis-Driven Diagnostic Grounding (any cue type!) ===", flush=True)
    t2 = time.monotonic()
    open_boxes_rows = []
    for r in ah_rows:
        im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        seen_c = []
        for cl in r["kde_distribution"]["clusters"]:
            if cl["country_name"] not in seen_c:
                seen_c.append(cl["country_name"])
        cand_str = ", ".join(seen_c[:5])
        og_prompt = (
            f"Candidate countries for this street-level photo: {cand_str}.\n"
            "Locate the 2 most geographically diagnostic regions of ANY type that best distinguish between these candidates "
            "(could be text/signs, building/roof style, vehicles, utility poles, streetlights, guardrails, road markings/curbs, "
            "fences, bollards, license plates, or distinctive soil/trees).\n"
            "Return ONLY a JSON array of 2 objects with normalized [0, 1000] coordinates [x1, y1, x2, y2]:\n"
            '[{"label": "<short object/feature name>", "bbox_2d": [x1, y1, x2, y2], "why": "<3-8 words why diagnostic>"}]'
        )
        raw = generate_one_blocks(model, processor, [im], [{"type": "image"}, {"type": "text", "text": og_prompt}], max_new_tokens=180)
        boxes = parse_open_boxes(raw)
        open_boxes_rows.append({"id": r["id"], "true_country": r["country"], "boxes": boxes, "raw": raw})
    (args.output_dir / "open_grounding_ah66.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in open_boxes_rows), encoding="utf-8"
    )
    og_cnt = Counter(b["label"] for x in open_boxes_rows for b in x["boxes"])
    print(f"  Part 3 done in {time.monotonic()-t2:.1f}s: {sum(1 for x in open_boxes_rows if x['boxes'])}/66 have boxes, top labels={og_cnt.most_common(12)}", flush=True)

    # =========================================================================
    # PART 4: Apples-to-Apples Top-8 Multi-Scale Eval on author_holdout_66
    #   (a) Image-Last Full-Only
    #   (b) Image-Last + MIL-Fovea (Top-2 Data-Driven High-MI Crops)
    #   (c) Image-Last + Open-Ended Diagnostic Grounding Fovea
    # =========================================================================
    print("\n=== Part 4/4: Evaluating Image-Last, MIL-Fovea, and Open-Grounding Fovea on author_holdout_66 ===", flush=True)
    t3 = time.monotonic()
    preds_last, preds_mil_fovea, preds_open_fovea = [], [], []

    def pack_pred(r, raw, crop_labels):
        p = parse_multiscale_json(raw, cc_names)
        coords = p["coordinates"]
        dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
        return {
            "id": r["id"], "true_country": r["country"],
            "true_latitude": r["latitude"], "true_longitude": r["longitude"],
            "crop_labels": crop_labels,
            "guess_country": p["country"], "guess_coordinates": list(coords) if coords else None,
            "distance_km": dist, "selected_cluster": p["selected_cluster"],
            "confidence": p["confidence"], "visual_clues": p["visual_clues"],
            "reasoning": p["reasoning"], "parsed": p["parsed"], "raw": raw,
        }

    for r, mrow, orow in zip(ah_rows, mil_ah_rows, open_boxes_rows):
        full_im = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=768)
        kde_p = build_multiscale_cot_prompt(r["kde_distribution"])

        # (a) Image-Last Full-Only
        raw_l = generate_one_blocks(model, processor, [full_im], build_image_last_content(kde_p), max_new_tokens=320)
        preds_last.append(pack_pred(r, raw_l, []))

        # (b) Image-Last + Data-Driven MIL Top-2 High-MI Crops
        m_boxes = mrow["mil_boxes"][:2]
        m_crops = [crop_cue(full_im, b["bbox_1000"], pad_frac=0.05) for b in m_boxes]
        m_notes = [f"Zoomed High-Information Region #{i+1} (auto-discovered salient area)" for i in range(len(m_crops))]
        raw_m = generate_one_blocks(model, processor, [full_im] + m_crops, build_image_last_content(kde_p, m_crops, m_notes), max_new_tokens=320)
        preds_mil_fovea.append(pack_pred(r, raw_m, [b["label"] for b in m_boxes]))

        # (c) Image-Last + Open-Ended Diagnostic Grounding Top-2 Crops
        o_boxes = orow["boxes"][:2]
        o_crops = [crop_cue(full_im, b["bbox_1000"], pad_frac=0.10) for b in o_boxes]
        o_notes = [f"Zoomed Diagnostic Region #{i+1}: `{b['label']}` ({b['why']})" for i, b in enumerate(o_boxes)]
        raw_o = generate_one_blocks(model, processor, [full_im] + o_crops, build_image_last_content(kde_p, o_crops, o_notes), max_new_tokens=320)
        preds_open_fovea.append(pack_pred(r, raw_o, [b["label"] for b in o_boxes]))

    (args.output_dir / "imglast_full_ah66.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds_last), encoding="utf-8")
    (args.output_dir / "imglast_mil_fovea_ah66.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds_mil_fovea), encoding="utf-8")
    (args.output_dir / "imglast_open_fovea_ah66.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds_open_fovea), encoding="utf-8")

    s_last = summarize_preds("Image-Last Full-Only", preds_last)
    s_mil = summarize_preds("Image-Last + MIL-Fovea", preds_mil_fovea)
    s_open = summarize_preds("Image-Last + Open-Grounding Fovea", preds_open_fovea)
    print(f"  Part 4 done in {time.monotonic()-t3:.1f}s.", flush=True)

    summary = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "mil_train_n": len(mil_train),
        "mil_classes": len(cc_vocab),
        "mil_tune_top1_pct": tune_top1_pct,
        "mil_ah66_top1_pct": round(100.0 * ah_mil_ok / 66.0, 2),
        "mil_ah66_top5_pct": round(100.0 * ah_mil_top5 / 66.0, 2),
        "attention_comparison": {
            "img_first_vis_share": round(float(f_vis), 5),
            "img_last_vis_share": round(float(l_vis), 5),
            "img_first_top_left_sink": round(float(f_tl), 5),
            "img_last_top_left_sink": round(float(l_tl), 5),
            "img_first_cue_conc": round(float(f_cue), 3),
            "img_last_cue_conc": round(float(l_cue), 3),
            "img_first_mil_conc": round(float(f_mil), 3),
            "img_last_mil_conc": round(float(l_mil), 3),
        },
        "eval_imglast_full": s_last,
        "eval_imglast_mil_fovea": s_mil,
        "eval_imglast_open_fovea": s_open,
    }
    (args.output_dir / "step1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted Step 1 Patch-MIL + Image-Last Attention-Sink + Open-Grounding experiment!", flush=True)


if __name__ == "__main__":
    main()
