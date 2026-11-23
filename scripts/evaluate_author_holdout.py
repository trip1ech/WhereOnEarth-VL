"""Evaluate the fixed base model and the saved 6,000-photo adapter on Colab."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from phase1 import QWEN, QWEN_REV
from sft_colab import PROMPT, evaluate, summarize


def main():
    root = Path("/content/geoguessr-author-holdout")
    inputs = root / "inputs"
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (inputs / "eval.jsonl").read_text().splitlines()]
    for row in rows:
        row["path"] = inputs / row["image"]
        if hashlib.sha256(row["path"].read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Unexpected input image: {row['id']}")

    dtype = torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quantization,
        dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, root / "adapter").eval()
    runs = {
        name: evaluate(model, processor, prompt, rows, output / f"{name}.jsonl", batch_size=4)
        for name in ("zero_shot", "sft")
    }
    metrics = [dict(model=name, **summarize(values)) for name, values in runs.items()]
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    pairs = [(a["distance_km"], b["distance_km"])
             for a, b in zip(runs["zero_shot"], runs["sft"], strict=True)
             if a["distance_km"] is not None and b["distance_km"] is not None]
    paired = {
        "valid_in_both": len(pairs),
        "zero_shot_median_km": median(a for a, b in pairs) if pairs else None,
        "sft_median_km": median(b for a, b in pairs) if pairs else None,
        "sft_closer": sum(b < a for a, b in pairs),
        "sft_farther": sum(b > a for a, b in pairs),
        "unchanged": sum(b == a for a, b in pairs),
    }
    (output / "paired_metrics.json").write_text(json.dumps(paired, indent=2) + "\n")
    (output / "configuration.json").write_text(json.dumps({
        "model": QWEN, "revision": QWEN_REV, "prompt": PROMPT,
        "gpu": torch.cuda.get_device_name(), "n": len(rows),
        "manifest_sha256": hashlib.sha256((inputs / "eval.jsonl").read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((root / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    with ZipFile(root / "author-holdout-results.zip", "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            bundle.write(path, path.name)
    print(json.dumps({"metrics": metrics, "paired": paired}), flush=True)


if __name__ == "__main__":
    main()
