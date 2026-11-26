"""Colab GPU worker: Rank-1-Balanced Offline Multimodal DPO ablation starting directly from
the trained Stage 1 QLoRA CoT SFT adapter (adapter_sft/).

Tests the exact scientific hypothesis diagnosed from Phase C:
  Unbalanced offline DPO (140 rejected=#1 vs 47 chosen=#1) collapsed Cluster #1 retention
  from 54.5% -> 10.6%. Rebalancing DPO pairs to exact 1:1 Rank-1 symmetry (52 chosen=#1
  vs 52 rejected=#1, plus 63 rank-neutral pairs) with a stronger SFT NLL anchor (0.25)
  and gentler lr (5e-6) removes the rank-1 penalty bias while preserving multi-scale visual
  preference learning.
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
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from train_qwen8b_cot_sft_dpo import (
    MODEL_ID,
    build_supervised_inputs,
    compute_seq_logprob_and_nll,
    evaluate_author_holdout,
    load_clean_image,
    precompute_ref_logprobs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-lr", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-nll-weight", type=float, default=0.25)
    parser.add_argument("--dpo-grad-accum", type=int, default=8)
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.output_dir / "imgs"
    img_dir.mkdir(exist_ok=True)

    cc_names = json.loads((args.input_dir / "country_names.json").read_text(encoding="utf-8"))
    dpo_rows = [json.loads(l) for l in (args.input_dir / "dpo_balanced_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ah_rows = [json.loads(l) for l in (args.input_dir / "author_holdout_multiscale.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pool_sha = {
        r["id"]: r["sha256"]
        for r in (json.loads(l) for l in (args.input_dir / "rollout_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }

    with ZipFile(args.input_dir / "train_subset_imgs.zip") as z:
        for r in dpo_rows:
            dst = img_dir / f"{r['id']}.jpg"
            if not dst.exists():
                payload = z.read(r["image"])
                if r["id"] in pool_sha and hashlib.sha256(payload).hexdigest() != pool_sha[r["id"]]:
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
    print(f"Verified images: {len(dpo_rows)} Rank-1-Balanced DPO pairs, {len(ah_rows)} author_holdout.", flush=True)

    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_ID} + adapter_sft on {torch.cuda.get_device_name(0)} (compute={compute_dtype})...", flush=True)
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    rev = str(getattr(base_model.config, "_commit_hash", "unknown"))
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = PeftModel.from_pretrained(base_model, str(args.input_dir / "adapter_sft"), is_trainable=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded base+adapter_sft in {time.monotonic() - t0:.1f}s | trainable={trainable/1e6:.2f}M | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    ref_cache = precompute_ref_logprobs(model, processor, dpo_rows, img_dir, max_edge=args.max_edge)

    print(
        f"\n=== Rank-1-Balanced DPO ({len(dpo_rows)} pairs x {args.dpo_epochs} ep, "
        f"lr={args.dpo_lr}, beta={args.dpo_beta}, nll_w={args.dpo_nll_weight}, eff_bs={args.dpo_grad_accum}) ===",
        flush=True,
    )
    model.train()
    processor.tokenizer.padding_side = "right"
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.dpo_lr, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(dpo_rows) / args.dpo_grad_accum)
    total_steps = max(1, steps_per_epoch * args.dpo_epochs)
    warmup_steps = max(1, int(0.15 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    rng = random.Random(args.seed + 200)
    t_start = time.monotonic()
    global_step = 0

    for ep in range(1, args.dpo_epochs + 1):
        order = list(range(len(dpo_rows)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        acc_dpo, acc_nll, acc_margin, acc_win = [], [], [], []
        accum_count = 0

        for idx_i, row_i in enumerate(order, 1):
            r = dpo_rows[row_i]
            ref_w, ref_l = ref_cache[row_i]
            img = load_clean_image(img_dir / f"{r['id']}.jpg", max_edge=args.max_edge)
            in_w, lab_w, _ = build_supervised_inputs(processor, img, r["prompt"], r["chosen"])
            in_l, lab_l, _ = build_supervised_inputs(processor, img, r["prompt"], r["rejected"])

            lp_w, nll_w = compute_seq_logprob_and_nll(model, in_w, lab_w)
            lp_l, _ = compute_seq_logprob_and_nll(model, in_l, lab_l)

            logits = args.dpo_beta * ((lp_w - ref_w) - (lp_l - ref_l))
            dpo_loss = -F.logsigmoid(logits)
            total_loss = (dpo_loss + args.dpo_nll_weight * nll_w) / args.dpo_grad_accum
            total_loss.backward()

            acc_dpo.append(float(dpo_loss.detach().item()))
            acc_nll.append(float(nll_w.detach().item()))
            acc_margin.append(float(logits.detach().item()))
            acc_win.append(1.0 if float(logits.detach().item()) > 0.0 else 0.0)
            accum_count += 1

            is_boundary = (accum_count == args.dpo_grad_accum) or (idx_i == len(order))
            if is_boundary:
                if accum_count < args.dpo_grad_accum:
                    scale = args.dpo_grad_accum / accum_count
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                rec_h = {
                    "epoch": ep,
                    "step": global_step,
                    "dpo_loss": round(float(np.mean(acc_dpo)), 4),
                    "nll_w": round(float(np.mean(acc_nll)), 4),
                    "margin": round(float(np.mean(acc_margin)), 4),
                    "win_acc": round(float(np.mean(acc_win)), 3),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                history.append(rec_h)
                if global_step == 1 or global_step % 5 == 0 or idx_i == len(order):
                    elapsed = (time.monotonic() - t_start) / 60.0
                    print(
                        f"  [BalDPO ep {ep}/{args.dpo_epochs} step {global_step:2d}/{total_steps}] "
                        f"dpo={rec_h['dpo_loss']:.4f} | nll_w={rec_h['nll_w']:.4f} | "
                        f"margin={rec_h['margin']:+.3f} | win={100*rec_h['win_acc']:.1f}% | "
                        f"lr={rec_h['lr']:.2e} | elapsed={elapsed:.1f}m",
                        flush=True,
                    )
                acc_dpo, acc_nll, acc_margin, acc_win = [], [], [], []
                accum_count = 0

    out_adapter = args.output_dir / "adapter_dpo_balanced"
    model.save_pretrained(out_adapter)
    print(f"Saved Rank-1-Balanced DPO adapter -> {out_adapter}", flush=True)

    torch.cuda.empty_cache()
    print("\n=== Evaluating Rank-1-Balanced DPO on author_holdout_66 ===", flush=True)
    ev = evaluate_author_holdout(
        model, processor, ah_rows, img_dir, cc_names,
        args.output_dir / "dpo_balanced_8b_author_holdout_top8.jsonl",
        max_edge=args.max_edge,
    )
    print("  Balanced DPO author_holdout_66:", json.dumps(ev), flush=True)

    report = {
        "model_id": MODEL_ID,
        "model_revision": rev,
        "gpu": torch.cuda.get_device_name(0),
        "hyperparams": {
            "dpo_pairs": len(dpo_rows),
            "dpo_epochs": args.dpo_epochs,
            "dpo_lr": args.dpo_lr,
            "dpo_beta": args.dpo_beta,
            "dpo_nll_weight": args.dpo_nll_weight,
            "dpo_grad_accum": args.dpo_grad_accum,
            "seed": args.seed,
        },
        "eval_author_holdout_dpo_balanced": ev,
        "history": history,
    }
    (args.output_dir / "balanced_dpo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("\nCompleted Rank-1-Balanced DPO ablation!", flush=True)


if __name__ == "__main__":
    main()
