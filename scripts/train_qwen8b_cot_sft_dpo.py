"""Colab GPU worker: Two-stage QLoRA post-training of Qwen/Qwen3-VL-8B-Instruct for
Multi-Scale Spatial KDE (Top-8) CoT Geolocalization:

  Stage 1 (SFT): QLoRA supervised fine-tuning on verified Rejection-Sampled + STaR CoT
                 trajectories (assistant-tokens-only cross-entropy loss, vision encoder frozen).
  Stage 2 (DPO): Exact multimodal offline Direct Preference Optimization + NLL anchor
                 starting from the SFT adapter, using precomputed SFT reference log-probs
                 on continuous geographic distance-reward (chosen, rejected) CoT pairs.

Evaluates both Stage 1 (SFT) and Stage 2 (SFT+DPO) greedily on creator-disjoint
author_holdout_66 with identical Top-8 multi-scale prompts for apples-to-apples comparison.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from multiscale_vlm_utils import (
    build_multiscale_cot_prompt,
    geoscore,
    haversine_km,
    parse_multiscale_json,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_clean_image(path: Path, max_edge: int = 768) -> Image.Image:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    if max(rgb.size) > max_edge:
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


def build_supervised_inputs(processor, image: Image.Image, prompt_text: str, response_text: str):
    """Tokenize (image, prompt, response) in a single processor call with exact assistant-only masking."""
    user_chat = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = processor.tokenizer.eos_token or "<|im_end|>"
    full_chat = user_chat + response_text + eos

    full_inputs = processor(text=[full_chat], images=[image], return_tensors="pt")
    f_ids = full_inputs["input_ids"][0].tolist()

    # Locate the final "<|im_start|>assistant\n" marker directly in full_ids (avoids running image processor twice)
    marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    prefix_len = None
    for i in range(len(f_ids) - len(marker), -1, -1):
        if f_ids[i:i + len(marker)] == marker:
            prefix_len = i + len(marker)
            break
    if prefix_len is None:
        prompt_inputs = processor(text=[user_chat], images=[image], return_tensors="pt")
        p_ids = prompt_inputs["input_ids"][0].tolist()
        prefix_len = 0
        lim = min(len(p_ids), len(f_ids))
        while prefix_len < lim and p_ids[prefix_len] == f_ids[prefix_len]:
            prefix_len += 1

    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100
    n_resp_tokens = int((labels != -100).sum().item())
    if n_resp_tokens <= 0:
        raise ValueError(f"Empty response token mask (prefix_len={prefix_len}, full_len={len(f_ids)})")
    return full_inputs, labels, n_resp_tokens


def compute_seq_logprob_and_nll(model, inputs, labels):
    """Forward pass returning summed assistant sequence log-prob and mean token NLL."""
    inputs_cuda = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
    labels_cuda = labels.to("cuda", non_blocking=True)

    outputs = model(**inputs_cuda, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels_cuda[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_lp = (token_lp * mask).sum(dim=-1)
    mean_nll = -seq_lp / mask.sum(dim=-1).clamp(min=1)
    return seq_lp.squeeze(0), mean_nll.squeeze(0)


@torch.inference_mode()
def evaluate_author_holdout(
    model,
    processor,
    ah_rows,
    img_dir: Path,
    cc_names: dict,
    out_path: Path,
    max_edge: int = 768,
    eval_bs: int = 8,
):
    model.eval()
    processor.tokenizer.padding_side = "left"
    preds = []
    idx = 0
    cur_bs = eval_bs
    while idx < len(ah_rows):
        grp = ah_rows[idx:idx + cur_bs]
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
                continue
            raise
        prompt_len = inputs["input_ids"].shape[1]
        raws = processor.tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
        for r, raw in zip(grp, raws):
            p = parse_multiscale_json(raw, cc_names)
            coords = p["coordinates"]
            dist = haversine_km(r["latitude"], r["longitude"], coords[0], coords[1]) if coords else None
            preds.append({
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
            })
        idx += len(grp)

    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds), encoding="utf-8")
    valid_d = [x["distance_km"] for x in preds if x["distance_km"] is not None]
    cc_ok = sum(x["guess_country"] == x["true_country"] for x in preds)
    summary = {
        "n": len(preds),
        "parsed": sum(1 for x in preds if x["parsed"]),
        "country_correct": cc_ok,
        "country_acc_pct": round(100.0 * cc_ok / len(preds), 2),
        "le_25km": sum(d <= 25.0 for d in valid_d),
        "le_200km": sum(d <= 200.0 for d in valid_d),
        "median_km": round(float(np.median(valid_d)), 1) if valid_d else None,
        "geoscore": round(float(sum(geoscore(d) for d in valid_d) / len(preds)), 1),
        "conf_high": sum(1 for x in preds if x["confidence"] == "high"),
        "conf_medium": sum(1 for x in preds if x["confidence"] == "medium"),
        "conf_low": sum(1 for x in preds if x["confidence"] == "low"),
    }
    return summary


def load_qlora_model(model_id: str, lora_r: int, lora_alpha: int, lora_dropout: float):
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

    # Attach LoRA strictly to language-model linear projections (vision encoder & merger stay frozen)
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lm_targets = sorted({
        name for name, _ in model.named_modules()
        if any(name.endswith(f".{s}") for s in target_suffixes)
        and "visual" not in name
        and "vision" not in name
    })
    if not lm_targets:
        raise RuntimeError("Failed to locate language-model linear layers for LoRA!")

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lm_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    vis_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("visual" in n or "vision" in n))
    print(
        f"Loaded {model_id} (rev={rev}) in {time.monotonic() - t0:.1f}s | "
        f"LM LoRA modules={len(lm_targets)} | trainable={trainable/1e6:.2f}M / {total/1e6:.1f}M "
        f"(visual_trainable={vis_trainable}) | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB",
        flush=True,
    )
    return model, processor, rev, compute_dtype


def run_sft_stage(
    model,
    processor,
    sft_rows,
    img_dir: Path,
    *,
    epochs: int,
    lr: float,
    grad_accum: int,
    max_edge: int,
    seed: int,
):
    print(f"\n=== Stage 1: QLoRA CoT SFT ({len(sft_rows)} examples x {epochs} epochs, lr={lr}, eff_bs={grad_accum}) ===", flush=True)
    model.train()
    processor.tokenizer.padding_side = "right"
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(sft_rows) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(0.08 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    rng = random.Random(seed)
    t0 = time.monotonic()
    global_step = 0

    for ep in range(1, epochs + 1):
        order = list(range(len(sft_rows)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        ep_losses = []
        accum_losses = []
        accum_count = 0

        for idx_i, row_i in enumerate(order, 1):
            r = sft_rows[row_i]
            img = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=max_edge)
            inputs, labels, _ = build_supervised_inputs(processor, img, r["prompt"], r["response"])
            _, mean_nll = compute_seq_logprob_and_nll(model, inputs, labels)
            loss = mean_nll / grad_accum
            loss.backward()

            lv = float(mean_nll.detach().item())
            ep_losses.append(lv)
            accum_losses.append(lv)
            accum_count += 1

            is_boundary = (accum_count == grad_accum) or (idx_i == len(order))
            if is_boundary:
                if accum_count < grad_accum:
                    scale = grad_accum / accum_count
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_loss = float(np.mean(accum_losses))
                history.append({
                    "stage": "sft",
                    "epoch": ep,
                    "step": global_step,
                    "loss": round(step_loss, 4),
                    "lr": float(scheduler.get_last_lr()[0]),
                })
                if global_step == 1 or global_step % 10 == 0 or idx_i == len(order):
                    elapsed = (time.monotonic() - t0) / 60.0
                    print(
                        f"  [SFT ep {ep}/{epochs} step {global_step:3d}/{total_steps}] "
                        f"nll={step_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | elapsed={elapsed:.1f}m",
                        flush=True,
                    )
                accum_losses = []
                accum_count = 0

        print(f"  SFT epoch {ep} mean NLL: {float(np.mean(ep_losses)):.4f}", flush=True)
    return history


@torch.inference_mode()
def precompute_ref_logprobs(model, processor, dpo_rows, img_dir: Path, max_edge: int):
    """Precompute exact reference log-probs under the frozen SFT adapter once."""
    print(f"\n=== Stage 2a: Precomputing SFT reference log-probs for {len(dpo_rows)} DPO pairs ===", flush=True)
    model.eval()
    processor.tokenizer.padding_side = "right"
    ref_cache = []
    t0 = time.monotonic()
    for i, r in enumerate(dpo_rows, 1):
        img = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=max_edge)
        in_w, lab_w, _ = build_supervised_inputs(processor, img, r["prompt"], r["chosen"])
        in_l, lab_l, _ = build_supervised_inputs(processor, img, r["prompt"], r["rejected"])
        lp_w, _ = compute_seq_logprob_and_nll(model, in_w, lab_w)
        lp_l, _ = compute_seq_logprob_and_nll(model, in_l, lab_l)
        ref_cache.append((float(lp_w.item()), float(lp_l.item())))
        if i == 1 or i % 100 == 0 or i == len(dpo_rows):
            rate = i / max(1e-3, time.monotonic() - t0)
            eta = (len(dpo_rows) - i) / max(1e-3, rate) / 60.0
            print(f"  Ref logprobs: {i:4d}/{len(dpo_rows)} | {rate:.2f} pair/s | ETA={eta:.1f}m", flush=True)
    return ref_cache


def run_dpo_stage(
    model,
    processor,
    dpo_rows,
    ref_cache,
    img_dir: Path,
    *,
    epochs: int,
    lr: float,
    beta: float,
    nll_weight: float,
    grad_accum: int,
    max_edge: int,
    seed: int,
    ah_rows=None,
    cc_names=None,
    output_dir: Path = None,
):
    print(
        f"\n=== Stage 2b: Offline Multimodal DPO ({len(dpo_rows)} pairs x {epochs} ep, "
        f"lr={lr}, beta={beta}, nll_w={nll_weight}, eff_bs={grad_accum}) ===",
        flush=True,
    )
    model.train()
    processor.tokenizer.padding_side = "right"
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(dpo_rows) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(0.10 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    epoch_evals = {}
    rng = random.Random(seed + 100)
    t0 = time.monotonic()
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        processor.tokenizer.padding_side = "right"
        order = list(range(len(dpo_rows)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        acc_dpo, acc_nll, acc_margin, acc_win = [], [], [], []
        accum_count = 0

        for idx_i, row_i in enumerate(order, 1):
            r = dpo_rows[row_i]
            ref_w, ref_l = ref_cache[row_i]
            img = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=max_edge)

            in_w, lab_w, _ = build_supervised_inputs(processor, img, r["prompt"], r["chosen"])
            in_l, lab_l, _ = build_supervised_inputs(processor, img, r["prompt"], r["rejected"])

            lp_w, nll_w = compute_seq_logprob_and_nll(model, in_w, lab_w)
            lp_l, _ = compute_seq_logprob_and_nll(model, in_l, lab_l)

            log_ratio_w = lp_w - ref_w
            log_ratio_l = lp_l - ref_l
            logits = beta * (log_ratio_w - log_ratio_l)
            dpo_loss = -F.logsigmoid(logits)
            total_loss = (dpo_loss + nll_weight * nll_w) / grad_accum
            total_loss.backward()

            acc_dpo.append(float(dpo_loss.detach().item()))
            acc_nll.append(float(nll_w.detach().item()))
            acc_margin.append(float(logits.detach().item()))
            acc_win.append(1.0 if float(logits.detach().item()) > 0.0 else 0.0)
            accum_count += 1

            is_boundary = (accum_count == grad_accum) or (idx_i == len(order))
            if is_boundary:
                if accum_count < grad_accum:
                    scale = grad_accum / accum_count
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                rec_h = {
                    "stage": "dpo",
                    "epoch": ep,
                    "step": global_step,
                    "dpo_loss": round(float(np.mean(acc_dpo)), 4),
                    "nll_w": round(float(np.mean(acc_nll)), 4),
                    "margin": round(float(np.mean(acc_margin)), 4),
                    "win_acc": round(float(np.mean(acc_win)), 3),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                history.append(rec_h)
                if global_step == 1 or global_step % 8 == 0 or idx_i == len(order):
                    elapsed = (time.monotonic() - t0) / 60.0
                    print(
                        f"  [DPO ep {ep}/{epochs} step {global_step:3d}/{total_steps}] "
                        f"dpo={rec_h['dpo_loss']:.4f} | nll_w={rec_h['nll_w']:.4f} | "
                        f"margin={rec_h['margin']:+.3f} | win={100*rec_h['win_acc']:.1f}% | "
                        f"lr={rec_h['lr']:.2e} | elapsed={elapsed:.1f}m",
                        flush=True,
                    )
                acc_dpo, acc_nll, acc_margin, acc_win = [], [], [], []
                accum_count = 0

        if output_dir is not None:
            ep_dir = output_dir / f"adapter_dpo_ep{ep}"
            model.save_pretrained(ep_dir)
            print(f"Saved DPO ep{ep} adapter -> {ep_dir}", flush=True)
            if ah_rows is not None and cc_names is not None:
                torch.cuda.empty_cache()
                print(f"\n=== Evaluating Stage 2 (SFT + DPO ep{ep}) on author_holdout_66 ===", flush=True)
                ev = evaluate_author_holdout(
                    model, processor, ah_rows, img_dir, cc_names,
                    output_dir / f"dpo_ep{ep}_8b_author_holdout_top8.jsonl",
                    max_edge=max_edge,
                )
                epoch_evals[f"ep{ep}"] = ev
                print(f"  SFT+DPO ep{ep} author_holdout_66:", json.dumps(ev), flush=True)
                torch.cuda.empty_cache()

    return history, epoch_evals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-lr", type=float, default=5e-5)
    parser.add_argument("--sft-grad-accum", type=int, default=16)
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-lr", type=float, default=8e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-nll-weight", type=float, default=0.1)
    parser.add_argument("--dpo-grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    sft_rows = [json.loads(l) for l in (args.input_dir / "sft_cot_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    dpo_rows = [json.loads(l) for l in (args.input_dir / "dpo_cot_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pool_sha = {
        r["id"]: r["sha256"]
        for r in (json.loads(l) for l in (args.input_dir / "rollout_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }

    with ZipFile(args.input_dir / "train_subset_imgs.zip") as z:
        needed = {r["id"]: r["image"] for r in sft_rows}
        needed.update({r["id"]: r["image"] for r in dpo_rows})
        for rid, img_rel in needed.items():
            dst = img_dir / f"{rid}.jpg"
            if not dst.exists():
                payload = z.read(img_rel)
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
    print(f"Verified images: {len(sft_rows)} SFT rows, {len(dpo_rows)} DPO pairs, {len(ah_rows)} author_holdout.", flush=True)

    model, processor, rev, compute_dtype = load_qlora_model(
        MODEL_ID, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    # Stage 1: QLoRA SFT
    sft_hist = run_sft_stage(
        model, processor, sft_rows, img_dir,
        epochs=args.sft_epochs, lr=args.sft_lr, grad_accum=args.sft_grad_accum,
        max_edge=args.max_edge, seed=args.seed,
    )
    sft_adapter_dir = args.output_dir / "adapter_sft"
    model.save_pretrained(sft_adapter_dir)
    print(f"Saved Stage 1 SFT adapter -> {sft_adapter_dir}", flush=True)

    print("\n=== Evaluating Stage 1 (SFT) on author_holdout_66 ===", flush=True)
    sft_eval = evaluate_author_holdout(
        model, processor, ah_rows, img_dir, cc_names,
        args.output_dir / "sft_8b_author_holdout_top8.jsonl",
        max_edge=args.max_edge,
    )
    print("  SFT author_holdout_66:", json.dumps(sft_eval), flush=True)

    # Stage 2: Exact Multimodal Offline DPO
    dpo_hist = []
    dpo_evals = {}
    if dpo_rows and args.dpo_epochs > 0:
        torch.cuda.empty_cache()
        ref_cache = precompute_ref_logprobs(model, processor, dpo_rows, img_dir, max_edge=args.max_edge)
        dpo_hist, dpo_evals = run_dpo_stage(
            model, processor, dpo_rows, ref_cache, img_dir,
            epochs=args.dpo_epochs, lr=args.dpo_lr, beta=args.dpo_beta,
            nll_weight=args.dpo_nll_weight, grad_accum=args.dpo_grad_accum,
            max_edge=args.max_edge, seed=args.seed,
            ah_rows=ah_rows, cc_names=cc_names, output_dir=args.output_dir,
        )

    report = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": str(compute_dtype),
        "hyperparams": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "sft_rows": len(sft_rows),
            "sft_epochs": args.sft_epochs,
            "sft_lr": args.sft_lr,
            "sft_grad_accum": args.sft_grad_accum,
            "dpo_pairs": len(dpo_rows),
            "dpo_epochs": args.dpo_epochs,
            "dpo_lr": args.dpo_lr,
            "dpo_beta": args.dpo_beta,
            "dpo_nll_weight": args.dpo_nll_weight,
            "dpo_grad_accum": args.dpo_grad_accum,
            "max_edge": args.max_edge,
            "seed": args.seed,
        },
        "eval_author_holdout_sft": sft_eval,
        "eval_author_holdout_dpo": dpo_evals,
        "history": sft_hist + dpo_hist,
    }
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted QLoRA SFT + Offline DPO training and evaluation!", flush=True)


if __name__ == "__main__":
    main()
