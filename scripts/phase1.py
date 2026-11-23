"""Run and score the two frozen, single-image baselines."""
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

QWEN = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
OSV_REV = "71548b90ac4a1aa7c37839841f411a06da82b1a6"
OSV_CODE = "4e6075387ecde4255410785ffb83830c9aa099f6"
PROMPT = (
    "Estimate where this street-level photograph was taken from visual evidence. "
    'Return exactly one JSON object with decimal-degree coordinates in this format: '
    '{"latitude": 0.0, "longitude": 0.0}. Make your best single guess.'
)


def coordinates(lat, lon):
    if isinstance(lat, bool) or isinstance(lon, bool):
        raise ValueError("Boolean coordinate")
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Coordinate outside WGS84 bounds")
    return lat, lon


def parse_guess(text):
    for i, char in enumerate(text):
        if char == "{":
            try:
                value, _ = json.JSONDecoder().raw_decode(text[i:])
                return coordinates(value["latitude"], value["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
    raise ValueError("No valid latitude/longitude JSON object")


def haversine(lat, lon, guess_lat, guess_lon):
    lat, lon, guess_lat, guess_lon = map(math.radians, (lat, lon, guess_lat, guess_lon))
    a = math.sin((guess_lat - lat) / 2) ** 2
    a += math.cos(lat) * math.cos(guess_lat) * math.sin((guess_lon - lon) / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(max(0, a))))


def read_split(data, split, limit_per_source=None):
    manifest = data / f"{split}.jsonl"
    counts = {}
    records = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        source = row["source"]
        count = counts.get(source, 0)
        if limit_per_source is not None and count >= limit_per_source:
            continue
        row["path"] = data / row["image"]
        if hashlib.sha256(row["path"].read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
        coordinates(row["latitude"], row["longitude"])
        records.append(row)
        counts[source] = count + 1
    if set(counts) != {"osv5m", "panoramax"}:
        raise ValueError(f"Expected both sources in {manifest}")
    return records


def clean_image(path):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
    # Build a new image so the predictors receive only the pixel array.
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


def qwen_predictor():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(QWEN, revision=QWEN_REV)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN, revision=QWEN_REV, quantization_config=quantization, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    ).eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def predict(image):
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        tokens = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        raw = processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        try:
            return parse_guess(raw), raw
        except ValueError:
            return None, raw

    return predict


def osv_predictor(runtime):
    import torch

    repo = runtime / "osv5m-official"
    if not (repo / ".git").is_dir():
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    ready = (repo / "models/huggingface.py").is_file()
    if not ready:
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1",
                        "https://github.com/gastruc/osv5m.git", OSV_CODE], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    sys.path.insert(0, str(repo))
    from models.huggingface import Geolocalizer

    cwd = Path.cwd()
    try:
        os.chdir(repo)  # The checkpoint config uses a repo-relative quadtree CSV.
        model = Geolocalizer.from_pretrained("osv5m/baseline", revision=OSV_REV).to("cuda").eval()
    finally:
        os.chdir(cwd)

    @torch.inference_mode()
    def predict(image):
        rgb_tensor = model.transform(image).unsqueeze(0).to("cuda")
        lat_rad, lon_rad = model(rgb_tensor).detach().cpu().reshape(2).tolist()
        return coordinates(math.degrees(lat_rad), math.degrees(lon_rad)), ""

    return predict


def run_baseline(name, records, run_dir, runtime):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a VS Code Colab GPU runtime")
    destination = run_dir / f"{name}.jsonl"
    existing = [json.loads(line) for line in destination.read_text().splitlines()] if destination.exists() else []
    done = {(row["source"], row["id"]) for row in existing}
    pending = [row for row in records if (row["source"], row["id"]) not in done]
    if not pending:
        print(f"{name}: {len(done)} already recorded", flush=True)
        return
    print(f"{name}: loading on {torch.cuda.get_device_name(0)}", flush=True)
    predict = qwen_predictor() if name == "qwen_zero_shot" else osv_predictor(runtime)
    try:
        with destination.open("a") as handle:
            for row in pending:
                start = time.perf_counter()
                guess, raw = predict(clean_image(row["path"]))
                lat, lon = guess if guess is not None else (None, None)
                distance = haversine(row["latitude"], row["longitude"], lat, lon) if guess is not None else None
                result = {
                    **{key: row[key] for key in ("id", "source", "region", "sequence", "latitude", "longitude")},
                    "guess_latitude": lat, "guess_longitude": lon, "distance_km": distance,
                    "raw_output": raw, "seconds": round(time.perf_counter() - start, 3),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{name} | {row['source']} | {row['region']} | "
                      f"{distance:.1f} km" if distance is not None else
                      f"{name} | {row['source']} | {row['region']} | invalid coordinate", flush=True)
    finally:
        del predict
        gc.collect()
        torch.cuda.empty_cache()


def summarize(run_dir):
    methods = {}
    for method in ("osv5m_official", "qwen_zero_shot"):
        path = run_dir / f"{method}.jsonl"
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for source in sorted({row["source"] for row in records}):
            rows = [row for row in records if row["source"] == source]
            distances = [row["distance_km"] for row in rows if row["distance_km"] is not None]
            methods[(path.stem, source)] = {
                "method": path.stem, "source": source, "n": len(rows), "n_valid": len(distances),
                "hit_25": sum(distance <= 25 for distance in distances) / len(rows),
                "hit_200": sum(distance <= 200 for distance in distances) / len(rows),
                "median_km_valid": statistics.median(distances) if distances else None,
            }
    if methods:
        rows = list(methods.values())
        with (run_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print((run_dir / "metrics.csv").read_text(), flush=True)
    return methods
