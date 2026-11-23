"""Train a country-to-coordinate adapter with answer-only causal cross-entropy."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from phase1 import QWEN, QWEN_REV, clean_image, coordinates, haversine, parse_guess

PROMPT = (
    "From this street-level image, guess its location anywhere on Earth. "
    "Return exactly one JSON object with the two-letter country code first, then the "
    "decimal latitude and longitude. Make one best guess. "
    "Use exactly these keys: country, latitude, longitude."
)


def open_data(bundle, folder, *, load_train_images=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    metadata_hash = hashlib.sha256()
    splits = {}
    seen = {key: set() for key in ("id", "sequence", "original_sha256")}
    with ZipFile(bundle) as archive:
        for split in ("train", "eval"):
            manifest = archive.read(f"{split}.jsonl")
            metadata_hash.update(manifest)
            rows = [json.loads(line) for line in manifest.splitlines()]
            if not rows:
                raise ValueError(f"No records in {split}")
            for row in rows:
                image_name = row["id"]
                relative = f"images/{image_name}.jpg"
                if not re.fullmatch(r"[0-9]+", image_name) or row["image"] != relative:
                    raise ValueError(f"Unexpected image path in {split}")
                coords = coordinates(row["latitude"], row["longitude"])
                if not re.fullmatch(r"[A-Z]{2}", row["country"]) or not all(map(math.isfinite, coords)):
                    raise ValueError(f"Invalid geographic label in {split}")
                for key, existing in seen.items():
                    if not row[key] or row[key] in existing:
                        raise ValueError(f"Missing or overlapping {key} in {split}")
                    existing.add(row[key])
                if split == "train" and not load_train_images:
                    continue
                path = folder / relative
                payload = archive.read(relative)
                if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    raise ValueError(f"JPEG hash differs: {image_name}")
                if not path.exists() or path.read_bytes() != payload:
                    path.write_bytes(payload)
                row["path"] = path
            splits[split] = rows
    return splits, metadata_hash.hexdigest()


def format_answer(row):
    return (
        f'{{"country":"{row["country"]}","latitude":{row["latitude"]:.3f},'
        f'"longitude":{row["longitude"]:.3f}}}'
    )


def parse_response(raw):
    try:
        lat_lon = parse_guess(raw)
    except ValueError:
        lat_lon = None
    country_match = re.search(r'"country"\s*:\s*"([A-Za-z]{2})"', raw)
    return (country_match.group(1).upper() if country_match else None), lat_lon


def summarize(rows):
    n = len(rows)
    distances = [row["distance_km"] for row in rows if row["distance_km"] is not None]
    return {
        "n": n,
        "valid_coordinates": len(distances),
        "country_correct": sum(row["guess_country"] == row["country"] for row in rows),
        "within_25km": sum(km <= 25 for km in distances),
        "within_200km": sum(km <= 200 for km in distances),
        "median_km": statistics.median(distances) if distances else None,
    }


def evaluate(model, processor, prompt, records, destination, batch_size):
    import torch
    from contextlib import nullcontext

    existing = [json.loads(line) for line in destination.read_text().splitlines()] if destination.exists() else []
    if [row["id"] for row in existing] != [row["id"] for row in records[:len(existing)]]:
        raise ValueError(f"Cached predictions do not match {destination}")
    model.eval()
    adapters = model.disable_adapter() if destination.stem == "zero_shot" else nullcontext()
    processor.tokenizer.padding_side = "left"
    with torch.inference_mode(), adapters, destination.open("a") as handle:
        for start in range(len(existing), len(records), batch_size):
            group = records[start:start + batch_size]
            images = [clean_image(row["path"]) for row in group]
            inputs = processor(text=[prompt] * len(group), images=images,
                               padding=True, return_tensors="pt").to("cuda")
            tokens = model.generate(**inputs, max_new_tokens=40, do_sample=False, use_cache=True)
            completions = processor.tokenizer.batch_decode(
                tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            for row, raw in zip(group, completions, strict=True):
                country, coords = parse_response(raw)
                distance = haversine(row["latitude"], row["longitude"], *coords) if coords else None
                result = {key: row[key] for key in ("id", "country", "latitude", "longitude")}
                result.update(guess_country=country, guess_coordinates=coords, distance_km=distance, raw=raw)
                handle.write(json.dumps(result) + "\n")
                existing.append(result)
            handle.flush()
            if start == 0 or start // 24 != len(existing) // 24 or len(existing) == len(records):
                print(f"{destination.stem}: {len(existing)}/{len(records)} evaluated", flush=True)
    return existing


def answer_loss(model, processor, prompt, row, *, target_text=None):
    import torch
    import torch.nn.functional as F

    inputs = processor(text=[prompt], images=[clean_image(row["path"])], return_tensors="pt").to("cuda")
    target = format_answer(row) if target_text is None else target_text
    answer = processor.tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"]
    eos = torch.full((1, 1), processor.tokenizer.eos_token_id, dtype=answer.dtype)
    answer = torch.cat([answer, eos], dim=1).to("cuda")
    inputs["input_ids"] = torch.cat([inputs["input_ids"], answer], dim=1)
    inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, answer.shape[1]), value=1)
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = F.pad(inputs["mm_token_type_ids"], (0, answer.shape[1]), value=0)
    logits = model(**inputs, use_cache=False, logits_to_keep=answer.shape[1] + 1).logits
    return F.cross_entropy(logits[:, :-1, :].float().flatten(0, 1), answer.flatten())


def train_adapter(model, processor, prompt, rows, out, dtype, seed, accumulation, checkpoint_every,
                  completed_checkpoint=False, *, learning_rate=1e-4):
    import torch
    from transformers import get_linear_schedule_with_warmup

    adapter = out / "adapter"
    training_log = out / "training.jsonl"
    if (adapter / "adapter_model.safetensors").exists():
        raise FileExistsError("Use the saved adapter for inference or choose a fresh output directory.")
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=0.0)
    steps = math.ceil(len(rows) / accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 20), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    shuffled = random.Random(seed).sample(rows, len(rows))
    checkpoint = out.with_suffix(".checkpoint.pt")
    history = []
    start_time = time.monotonic()
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["weights"].keys() != trainable.keys():
            raise ValueError("Checkpoint belongs to a different set of trainable parameters")
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(saved["weights"][name])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        history = saved["history"]
        start_time -= history[-1]["elapsed_s"]
        print(f"Continuing from step {len(history)}/{steps}", flush=True)
    if completed_checkpoint and len(history) != steps:
        raise ValueError("Evaluation recovery requires the final training checkpoint")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start_row = len(history) * accumulation
    with training_log.open("w") as handle:
        for stats in history:
            handle.write(json.dumps(stats) + "\n")
        for step, start in enumerate(range(start_row, len(rows), accumulation), len(history) + 1):
            group = shuffled[start:start + accumulation]
            running = torch.zeros((), device="cuda")
            for row in group:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss = answer_loss(model, processor, row.get("prompt", prompt), row,
                                       target_text=row.get("target_text"))
                scaler.scale(loss / len(group)).backward()
                running += loss.detach()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            average_loss = float(running / len(group))
            if not math.isfinite(average_loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            stats = dict(step=step, steps=steps, loss=average_loss, elapsed_s=round(time.monotonic() - start_time))
            history.append(stats)
            handle.write(json.dumps(stats) + "\n")
            handle.flush()
            if step == 1 or step % 10 == 0 or step == steps:
                print(json.dumps(stats), flush=True)
            if checkpoint_every and (step % checkpoint_every == 0 or step == steps):
                temporary = checkpoint.with_suffix(".part")
                torch.save({"weights": {name: parameter.detach().cpu() for name, parameter in trainable.items()},
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "cpu_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state(), "history": history}, temporary)
                temporary.replace(checkpoint)
                with checkpoint.open("rb") as checkpoint_handle:
                    digest = hashlib.file_digest(checkpoint_handle, "sha256").hexdigest()
                print(f"GEOGUESSR_CHECKPOINT {step} {digest}", flush=True)
    temp_adapter = out / "adapter_in_progress"
    model.save_pretrained(temp_adapter)
    temp_adapter.rename(adapter)


def save_results(args, output, predictions, records):
    summary = []
    for name, values in predictions.items():
        summary.append(dict(model=name, **summarize(values)))
    if summary:
        with (output / "metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print((output / "metrics.csv").read_text(), flush=True)
    if "zero_shot" in predictions and "sft" in predictions:
        def relative(reference):
            counts = {"new_25km_hits": 0, "lost_25km_hits": 0}
            for before, after in zip(predictions[reference], predictions["sft"], strict=True):
                old_hit = before["distance_km"] is not None and before["distance_km"] <= 25
                new_hit = after["distance_km"] is not None and after["distance_km"] <= 25
                counts["new_25km_hits"] += int(new_hit and not old_hit)
                counts["lost_25km_hits"] += int(old_hit and not new_hit)
            return counts

        comparisons = relative("zero_shot")
        if "sft_2000" in predictions:
            comparisons["versus_sft_2000"] = relative("sft_2000")
        (output / "paired_metrics.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    for split, rows in records.items():
        with (output / f"{split}_labels.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps({k: row[k] for k in ("id", "sequence", "country", "latitude",
                                                           "longitude", "sha256", "original_sha256")}) + "\n")
    temp = args.export.with_suffix(".part")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and "adapter_in_progress" not in path.parts and path.parts[-1] != "README.md":
                bundle.write(path, path.relative_to(output))
    temp.replace(args.export)
    print(f"Saved {args.export}", flush=True)


def reuse_predictions(archive_path, records, output):
    """Bring forward measurements tied to the same model, prompt and JPEGs."""
    with ZipFile(archive_path) as archive:
        previous = json.loads(archive.read("configuration.json"))
        expected = {"model": QWEN, "revision": QWEN_REV, "prompt": PROMPT,
                    "train_n": 2000, "eval_n": len(records)}
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("Prior results have a different model, prompt, or split size")
        labels = [json.loads(line) for line in archive.read("eval_labels.jsonl").splitlines()]
        keys = ("id", "sequence", "country", "latitude", "longitude", "sha256", "original_sha256")
        expected_rows = [{key: row[key] for key in keys} for row in records]
        if labels != expected_rows:
            raise ValueError("Prior evaluation image order, bytes, or labels differ")
        predictions = {}
        for source, target in (("zero_shot", "zero_shot"), ("sft", "sft_2000")):
            payload = archive.read(f"{source}.jsonl")
            rows = [json.loads(line) for line in payload.splitlines()]
            if [row["id"] for row in rows] != [row["id"] for row in records]:
                raise ValueError(f"Prior predictions are unaligned: {source}")
            predictions[target] = rows
            (output / f"{target}.jsonl").write_bytes(payload)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/content/geoguessr_sft/run"))
    parser.add_argument("--export", type=Path, default=Path("/content/geoguessr-sft-results.zip"))
    parser.add_argument("--smoke", action="store_true", help="Run an 8-train/4-eval GPU test")
    parser.add_argument("--eval-batch-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--reuse-evaluation", type=Path, help="Use predictions from the matching 2,000-image run")
    parser.add_argument("--completed-checkpoint", action="store_true", help="Evaluate from a final checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Optimizer steps between resumable snapshots")
    parser.add_argument("--check-data", action="store_true", help="Validate input with CPU only")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    splits, data_hash = open_data(args.bundle, output.parent / "dataset",
                                  load_train_images=not args.completed_checkpoint)
    if args.check_data:
        print({name: len(rows) for name, rows in splits.items()}, data_hash, flush=True)
        return

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime for training or inference.")
    seed = 7818
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    if args.smoke:
        splits = {"train": splits["train"][:8], "eval": splits["eval"][:4]}
    accumulation = 2 if args.smoke else 8
    run = dict(model=QWEN, revision=QWEN_REV, gpu=torch.cuda.get_device_name(0), precision=str(dtype),
               objective="mean answer-token cross-entropy", dataset_hash=data_hash, prompt=PROMPT,
               train_n=len(splits["train"]), eval_n=len(splits["eval"]), epochs=1,
               rank=8, alpha=16, accumulation=accumulation, seed=seed)
    config = output / "configuration.json"
    if config.exists() and json.loads(config.read_text()) != run:
        raise ValueError("Output directory has a different configuration")
    config.write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run), flush=True)

    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quantization,
        dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    if (output / "adapter" / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, output / "adapter")
    else:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
    predictions = reuse_predictions(args.reuse_evaluation, splits["eval"], output) if args.reuse_evaluation else {}
    if not (output / "adapter" / "adapter_config.json").exists():
        train_adapter(model, processor, prompt, splits["train"], output, dtype, seed, accumulation,
                      args.checkpoint_every, completed_checkpoint=args.completed_checkpoint)
        save_results(args, output, predictions, splits)
        print("GEOGUESSR_ADAPTER_READY", flush=True)
    if "zero_shot" not in predictions:
        predictions["zero_shot"] = evaluate(model, processor, prompt, splits["eval"],
                                            output / "zero_shot.jsonl", args.eval_batch_size)
    predictions["sft"] = evaluate(model, processor, prompt, splits["eval"],
                                  output / "sft.jsonl", args.eval_batch_size)
    save_results(args, output, predictions, splits)


if __name__ == "__main__":
    main()
