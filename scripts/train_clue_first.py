"""Continue the existing LM adapter on verified street-scene evidence and location replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from clue_visual import VISUAL_PROMPT, encode, visual_geo_prompt
from evaluate_clue_first import OCR_PROMPT, chat
from phase1 import QWEN, QWEN_REV
from sft_colab import PROMPT, train_adapter

SEED = 52711
LEARNING_RATE = 6e-5
ACCUMULATION = 8
PASSES = {"ocr_evidence": 3, "ocr_empty": 2, "visual": 2, "geo_direct": 2, "geo_visual": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = (args.input / "training.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest.splitlines()]
    keys = {(r["id"], r["task"]) for r in rows}
    if len(keys) != len(rows) or set(r["task"] for r in rows) != PASSES.keys():
        raise ValueError("Unexpected training tasks")

    adapter = args.input / "eval/adapter"
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    digest = {}
    jobs = []
    for row in rows:
        path = args.input / row["image"]
        if path not in digest:
            digest[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest[path] != row["sha256"]:
            raise ValueError(f"Training pixels differ for {row['id']}")
        row["path"] = path
        task = row["task"]
        if task.startswith("ocr_"):
            prompt = OCR_PROMPT
            row["target_text"] = encode({"text": row["evidence"]})
        elif task == "visual":
            prompt = VISUAL_PROMPT
            row["target_text"] = encode(row["observations"])
        elif task == "geo_visual":
            prompt = visual_geo_prompt(row["observations"])
        else:
            prompt = PROMPT
        row["prompt"] = chat(processor, prompt)
        jobs.extend([row] * PASSES[task])

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quant, dtype=dtype,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(".language_model." not in key or ".lora_" not in key for key in trainable):
        raise ValueError("Expected only LM LoRA weights to be trainable")
    config = {
        "model": QWEN, "revision": QWEN_REV, "gpu": torch.cuda.get_device_name(),
        "precision": str(dtype), "training_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "input_adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "unique_training_images": len(digest), "counts": dict(Counter(r["task"] for r in rows)),
        "passes": PASSES, "number_of_examples_seen": len(jobs), "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATION, "seed": SEED, "trainable_parameters": sum(trainable.values()),
    }
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)
    train_adapter(model, processor, chat(processor, PROMPT), jobs, args.output, dtype, SEED,
                  ACCUMULATION, 0, learning_rate=LEARNING_RATE)
    with ZipFile(args.output.parent / "clue-first-lm-final.zip", "w", compression=ZIP_DEFLATED) as archive:
        for relative in ("configuration.json", "training.jsonl", "adapter/adapter_config.json",
                         "adapter/adapter_model.safetensors"):
            archive.write(args.output / relative, relative)
    print("Exported final LM adapter", flush=True)


if __name__ == "__main__":
    main()
