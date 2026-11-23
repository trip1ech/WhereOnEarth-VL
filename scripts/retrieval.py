"""Frozen DINOv2 street-view retrieval and offline evaluation."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from phase1 import clean_image, coordinates, haversine, read_split

ENCODER = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
RAW_K = 100
GEO_K = 10
MERGE_KM = 5.0
HIT_KM = 25.0


def load_gallery(root):
    gallery = []
    with (root / "gallery.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["source"] != "osv5m_train":
                raise ValueError(f"Reference outside OSV train: {row['id']}")
            coordinates(row["latitude"], row["longitude"])
            image = root / row["image"]
            if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Gallery checksum mismatch: {row['id']}")
            gallery.append(dict(row, path=image))
    for key in ("id", "sequence"):
        if len({row[key] for row in gallery}) != len(gallery):
            raise ValueError(f"Duplicate reference {key}")
    if not gallery:
        raise ValueError("Gallery is empty")
    return gallery


def assert_disjoint(gallery, queries):
    for key in ("id", "sequence", "sha256"):
        overlap = {row.get(key) for row in gallery} & {row.get(key) for row in queries}
        if overlap - {None, ""}:
            raise ValueError(f"Gallery/query overlap on {key}: {len(overlap)}")


def feature_key(rows):
    return hashlib.sha256(
        "\n".join(f"{row['id']}:{row['sha256']}" for row in rows).encode()
    ).hexdigest()


def extract_features(rows, cache, *, device, batch_size=32):
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModel

    key = feature_key(rows)
    if cache.exists():
        with np.load(cache) as saved:
            expected = (key, REVISION, len(rows))
            actual = (str(saved["feature_key"]), str(saved["revision"]), len(saved["features"]))
            features = saved["features"]
        if actual != expected or features.shape != (len(rows), 384) or not np.isfinite(features).all():
            raise ValueError(f"Invalid or incompatible cached features: {cache}")
        return features
    processor = AutoImageProcessor.from_pretrained(ENCODER, revision=REVISION, use_fast=False, token=False)
    model = AutoModel.from_pretrained(ENCODER, revision=REVISION, token=False).to(device).eval()
    chunks = []
    print(f"Encoding {len(rows)} images with {ENCODER} on {device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = [clean_image(row["path"]) for row in rows[start:start + batch_size]]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            embedding = model(pixel_values=pixels).last_hidden_state[:, 0, :]
            chunks.append(torch.nn.functional.normalize(embedding.float(), dim=-1).cpu().numpy())
            if start % (batch_size * 10) == 0 or start + batch_size >= len(rows):
                print(f"Encoded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    features = np.concatenate(chunks).astype(np.float16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".part")
    with temporary.open("wb") as handle:
        np.savez(handle, feature_key=key, revision=REVISION, features=features)
    temporary.replace(cache)
    return features


def predict_candidates(reference_rows, similarities, raw_k=RAW_K, geo_k=GEO_K, merge_km=MERGE_KM):
    """Rank and merge reference locations without seeing the target GPS."""
    import numpy as np

    if (similarities.shape != (len(reference_rows),) or not len(reference_rows) or
            not np.isfinite(similarities).all() or raw_k < 1 or geo_k < 1 or merge_km < 0):
        raise ValueError("Invalid references or candidate settings")
    by_similarity = np.argsort(-similarities, kind="stable")[:raw_k]
    raw = [
        {
            "id": reference_rows[index]["id"],
            "latitude": reference_rows[index]["latitude"],
            "longitude": reference_rows[index]["longitude"],
            "score": float(similarities[index]),
        }
        for index in by_similarity
    ]
    grouped = []
    for reference in raw:
        merged = False
        for candidate in grouped:
            distance = haversine(reference["latitude"], reference["longitude"],
                                 candidate["latitude"], candidate["longitude"])
            if distance <= merge_km:
                candidate["member_ids"].append(reference["id"])
                merged = True
                break
        if not merged:
            grouped.append(dict(reference, member_ids=[reference["id"]]))
    return raw, grouped[:geo_k]


def evaluate_query(query, reference_rows, raw, candidates):
    def distance(row):
        return haversine(query["latitude"], query["longitude"], row["latitude"], row["longitude"])

    closest = min(reference_rows, key=distance)
    gallery_distance = distance(closest)
    raw_distance = min(map(distance, raw))
    evaluated = [dict(row, distance_km=distance(row)) for row in candidates]
    rank = next((i for i, row in enumerate(evaluated, 1) if row["distance_km"] <= HIT_KM), None)
    best = evaluated[0]
    if gallery_distance > HIT_KM:
        first_loss = "gallery_coverage"
    elif raw_distance > HIT_KM:
        first_loss = "raw_retrieval"
    elif rank is None:
        first_loss = "geographic_merge_or_top10"
    elif rank > 1:
        first_loss = "top1_selection"
    else:
        first_loss = "hit"
    return {
        **{k: query[k] for k in ("id", "source", "region", "latitude", "longitude")},
        "closest_reference_id": closest["id"], "gallery_min_km": gallery_distance,
        "raw_top100_min_km": raw_distance, "first_hit_candidate_rank": rank,
        "guess_latitude": best["latitude"], "guess_longitude": best["longitude"],
        "distance_km": best["distance_km"], "first_lost_stage": first_loss,
        "raw_count": len(raw), "candidate_count": len(candidates), "candidates": evaluated,
    }


def summarize_predictions(predictions, output):
    sources = ["all", *sorted({row["source"] for row in predictions})]
    metrics = []
    for source in sources:
        rows = [r for r in predictions if source == "all" or r["source"] == source]
        metrics.append({
            "source": source, "n": len(rows),
            "gallery_within_25": sum(r["gallery_min_km"] <= HIT_KM for r in rows),
            "raw_top100_within_25": sum(r["raw_top100_min_km"] <= HIT_KM for r in rows),
            "geo_top10_within_25": sum(r["first_hit_candidate_rank"] is not None for r in rows),
            "top1_within_25": sum(r["distance_km"] <= HIT_KM for r in rows),
            "top1_within_200": sum(r["distance_km"] <= 200 for r in rows),
            "median_km": statistics.median(r["distance_km"] for r in rows),
            "mean_world_points": statistics.mean(5000 * math.exp(-r["distance_km"] / 1492.7) for r in rows),
        })
    path = output / "metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(path.read_text(), flush=True)
    return metrics


def run_retrieval(gallery_root, query_root, output, *, device, batch_size=32):
    import numpy as np

    gallery = load_gallery(gallery_root)
    queries = read_split(query_root, "pilot_dev")
    assert_disjoint(gallery, queries)
    output.mkdir(parents=True, exist_ok=True)
    references = extract_features(gallery, output / "gallery_features.npz", device=device, batch_size=batch_size)
    query_features = extract_features(queries, output / "query_features.npz", device=device, batch_size=batch_size)
    references = references.astype(np.float32)
    references /= np.linalg.norm(references, axis=-1, keepdims=True)
    query_features = query_features.astype(np.float32)
    query_features /= np.linalg.norm(query_features, axis=-1, keepdims=True)
    predictions = []
    for query, vector in zip(queries, query_features):
        raw, candidates = predict_candidates(gallery, references @ vector)
        predictions.append(evaluate_query(query, gallery, raw, candidates))
    with (output / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "run_config.json").write_text(json.dumps({
        "encoder": ENCODER, "revision": REVISION, "raw_k": RAW_K, "geo_k": GEO_K,
        "merge_km": MERGE_KM, "hit_km": HIT_KM, "device": device, "gallery_n": len(gallery),
        "query_n": len(queries), "gallery_key": feature_key(gallery), "query_key": feature_key(queries),
    }, indent=2))
    return summarize_predictions(predictions, output)
