"""Step 1C: Extract exact Qwen3-VL-2B (6k QLoRA) country log-prob distributions over all gallery countries.

Uses single-pass KV-cache prefix scoring on '{"country":' -> '"{CC}"' so every country probability is
mathematically exact (zero sampling noise, zero retraining) and fast on Colab T4.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.nn import functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from phase1 import QWEN, QWEN_REV, clean_image
from sft_colab import PROMPT


def build_country_token_trie(tokenizer, country_codes):
    """Find cleanest BPE boundary right before the 2-letter ISO code to minimize unique prefixes (<=26)."""
    for base_prefix, fmt in (('{"country":"', "{cc}"), ('{"country":', '"{cc}')):
        base_ids = tokenizer.encode(base_prefix, add_special_tokens=False)
        code_tokens = {}
        ok = True
        for cc in country_codes:
            full_ids = tokenizer.encode(base_prefix + fmt.format(cc=cc), add_special_tokens=False)
            if full_ids[:len(base_ids)] != base_ids or len(full_ids) <= len(base_ids):
                ok = False
                break
            code_tokens[cc] = tuple(full_ids[len(base_ids):])
        if ok:
            return base_prefix, code_tokens
    raise ValueError("Could not find a clean BPE prefix boundary for ISO country codes")


@torch.inference_mode()
def score_countries_for_image(model, inputs, code_tokens):
    """Exact log P(suffix | image + prompt + '{"country":') using a single cached prefix forward pass."""
    # Single full multimodal forward pass to build KV-cache and obtain step-0 next-token log-probs
    out0 = model(**inputs, use_cache=True)
    logp0 = F.log_softmax(out0.logits[0, -1, :].float(), dim=-1)
    past0 = out0.past_key_values

    # Group countries by shared token prefixes to minimize cached continuation calls
    # Most codes have len 2 or 3 (e.g. ['"', 'US', '"'] or ['"U', 'S"']).
    # Collect all unique non-final prefixes of length 1, 2, ...
    prefix_to_next = defaultdict(set)
    for seq in code_tokens.values():
        for i in range(1, len(seq)):
            prefix_to_next[seq[:i]].add(seq[i])

    # Memoize log-prob tables for each prefix tuple using DynamicCache snapshots
    # Note: Transformers DynamicCache mutates in-place during forward; clone tensors cleanly.
    def clone_past(past):
        from transformers.cache_utils import DynamicCache
        if hasattr(past, "layers") and len(past.layers) > 0 and hasattr(past.layers[0], "keys"):
            c = DynamicCache()
            for layer_idx, layer in enumerate(past.layers):
                c.update(layer.keys.clone(), layer.values.clone(), layer_idx)
            return c
        if hasattr(past, "key_cache"):
            c = DynamicCache()
            for layer_idx in range(len(past.key_cache)):
                c.update(past.key_cache[layer_idx].clone(), past.value_cache[layer_idx].clone(), layer_idx)
            return c
        return tuple((k.clone(), v.clone()) for k, v in past)

    device = inputs["input_ids"].device
    base_len = inputs["input_ids"].shape[1]
    prefix_logp = {(): logp0}

    # Evaluate unique prefixes in increasing length order (usually only 1–28 unique prefixes total!)
    for length in sorted({len(p) for p in prefix_to_next}):
        prefixes = sorted(p for p in prefix_to_next if len(p) == length)
        for p in prefixes:
            past = clone_past(past0)
            tok = torch.tensor([list(p)], dtype=torch.long, device=device)
            attn = torch.ones((1, base_len + len(p)), dtype=inputs["attention_mask"].dtype, device=device)
            cache_pos = torch.arange(base_len, base_len + len(p), dtype=torch.long, device=device)
            out = model(input_ids=tok, attention_mask=attn, past_key_values=past,
                        cache_position=cache_pos, use_cache=True)
            prefix_logp[p] = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)

    scores = {}
    for cc, seq in code_tokens.items():
        lp = 0.0
        for i, tok_id in enumerate(seq):
            lp += float(prefix_logp[seq[:i]][tok_id])
        scores[cc] = lp
    return scores


@torch.inference_mode()
def verify_cache_exactness(model, inputs, code_tokens, cached_scores):
    """Sanity-check cached log-probs against full uncached forward pass on 3 sample countries."""
    device = inputs["input_ids"].device
    for cc in list(code_tokens)[:3]:
        seq = list(code_tokens[cc])
        full_ids = torch.cat([inputs["input_ids"], torch.tensor([seq], dtype=torch.long, device=device)], dim=1)
        full_attn = F.pad(inputs["attention_mask"], (0, len(seq)), value=1)
        kw = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        if "mm_token_type_ids" in kw:
            kw["mm_token_type_ids"] = F.pad(kw["mm_token_type_ids"], (0, len(seq)), value=0)
        logits = model(input_ids=full_ids, attention_mask=full_attn, use_cache=False, **kw).logits
        start = inputs["input_ids"].shape[1] - 1
        lp_ref = 0.0
        for i, tok_id in enumerate(seq):
            lp_ref += float(F.log_softmax(logits[0, start + i, :].float(), dim=-1)[tok_id])
        diff = abs(lp_ref - cached_scores[cc])
        if diff > 0.05:
            raise RuntimeError(f"M-RoPE / KV-cache mismatch for {cc}: ref={lp_ref:.4f} vs cached={cached_scores[cc]:.4f} (diff={diff:.4f})")
    print("Verified exact match between KV-cached scoring and full forward pass (diff < 0.05).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder with images/, manifest.json, adapter/, countries.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    countries = json.loads((args.input_dir / "countries.json").read_text())
    adapter = args.input_dir / "adapter"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quant, dtype=dtype,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter).eval()

    base_prefix, code_tokens = build_country_token_trie(processor.tokenizer, countries)
    unique_prefixes = {seq[:i] for seq in code_tokens.values() for i in range(1, len(seq))}
    print(f"Scoring {len(countries)} countries across {len(manifest)} images "
          f"(unique non-empty BPE prefixes: {len(unique_prefixes)})", flush=True)

    chat_prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    ) + base_prefix

    results = {}
    verified = False
    for idx, row in enumerate(manifest, 1):
        img_path = args.input_dir / row["image"]
        if hashlib.sha256(img_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Hash mismatch: {row['id']}")
        img = clean_image(img_path)
        inputs = processor(text=[chat_prompt], images=[img], return_tensors="pt").to("cuda")
        scores = score_countries_for_image(model, inputs, code_tokens)
        if not verified:
            verify_cache_exactness(model, inputs, code_tokens, scores)
            verified = True
        # Normalize log-probs over the country vocabulary
        vals = torch.tensor([scores[c] for c in countries], dtype=torch.float32)
        norm_lp = F.log_softmax(vals, dim=0)
        probs = norm_lp.exp()
        entropy = float(-(probs * norm_lp).sum())
        top10 = sorted(((countries[i], float(probs[i])) for i in range(len(countries))), key=lambda x: -x[1])[:10]
        results[row["id"]] = {
            "split": row["split"],
            "true_country": row["country"],
            "top1_country": top10[0][0],
            "top1_prob": top10[0][1],
            "entropy": entropy,
            "top10": top10,
            "log_probs": {countries[i]: round(float(norm_lp[i]), 5) for i in range(len(countries))},
        }
        if idx == 1 or idx % 25 == 0 or idx == len(manifest):
            top1_acc = sum(v["top1_country"] == v["true_country"] for v in results.values()) / len(results)
            print(f"VLM country priors: {idx}/{len(manifest)} (running top-1 country acc={100*top1_acc:.1f}%)", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "vlm_country_priors.json"
    out_file.write_text(json.dumps({"countries": countries, "predictions": results}) + "\n")
    print(f"Saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
