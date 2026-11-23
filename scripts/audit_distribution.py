"""Audit every official train/test geotag and compare its coverage with the sampled gallery."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from phase1 import haversine
from prepare_pilot import OSV, REGIONS, ROOT, fetch, in_bbox

# Counts for the pinned CSV release. The CVPR datasheet reports 4,894,685 train images.
TRAIN_ROWS = 4_894_684
TRAIN_ROWS_PAPER = 4_894_685
TEST_ROWS = 210_122
BLOCK_BYTES = 24 * 1024 * 1024
BOUNDARY_BYTES = 64 * 1024
RADIUS = 25.0
OUTPUT = ROOT / "results/distribution_audit"


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    part = path.with_suffix(".part")
    part.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    part.replace(path)


def coordinate_rows(records, indices):
    for record in records:
        if not record:
            continue
        image_id, lat, lon, country, sequence = (record[i] for i in indices)
        yield image_id, float(lat), float(lon), country or "UNKNOWN", sequence


def accumulate(rows, queries):
    country_counts, grid_counts, box_counts = Counter(), Counter(), Counter()
    near = [{"n": 0, "seqs": set(), "closest_km": None, "closest_id": None} for _ in queries]
    for image_id, lat, lon, country, seq in rows:
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Invalid coordinate in dataset: {image_id}")
        country_counts[country] += 1
        grid_counts[(min(89, int((lat + 90) // 2)), min(179, int((lon + 180) // 2)))] += 1
        if country == "FR":
            for key, bbox in REGIONS.items():
                if in_bbox(lat, lon, bbox):
                    box_counts[key] += 1
        # All audit queries are metropolitan France. These bounding
        # boxes conservatively contain every point within 25 km.
        if 42 <= lat <= 51 and -1 <= lon <= 8:
            for query, stats in zip(queries, near):
                if abs(query["latitude"] - lat) > 0.25 or abs(query["longitude"] - lon) > 0.45:
                    continue
                distance = haversine(lat, lon, query["latitude"], query["longitude"])
                if distance > RADIUS:
                    continue
                stats["n"] += 1
                if seq:
                    stats["seqs"].add(seq)
                if stats["closest_km"] is None or (distance, image_id) < (stats["closest_km"], stats["closest_id"]):
                    stats["closest_km"] = distance
                    stats["closest_id"] = image_id
    for cover in near:
        cover["seqs"] = sorted(cover["seqs"])
    return {"countries": dict(country_counts), "grid": [[*k, n] for k, n in grid_counts.items()],
            "regions": dict(box_counts), "coverage": near}


def merge(parts, queries):
    country, grid, boxes = Counter(), Counter(), Counter()
    cover = [{"n": 0, "seqs": set(), "closest_km": None, "closest_id": None} for _ in queries]
    for part in parts:
        country.update(part["countries"])
        grid.update({(a, b): n for a, b, n in part["grid"]})
        boxes.update(part["regions"])
        for dest, row in zip(cover, part["coverage"], strict=True):
            dest["n"] += row["n"]
            dest["seqs"].update(row["seqs"])
            if row["closest_km"] is not None and (dest["closest_km"] is None or
                    (row["closest_km"], row["closest_id"]) < (dest["closest_km"], dest["closest_id"])):
                dest["closest_km"], dest["closest_id"] = row["closest_km"], row["closest_id"]
    for stats in cover:
        stats["sequence_count"] = len(stats.pop("seqs"))
    return {"countries": country, "grid": grid, "regions": boxes, "coverage": cover}


def ranged_csv(url, start, end, size):
    read_start = max(0, start - 1)
    read_end = min(size, end + BOUNDARY_BYTES)
    payload, header, status = fetch(
        f"{url}?audit_range={read_start}-{read_end-1}", {"Range": f"bytes={read_start}-{read_end-1}"},
    )
    if (status != 206 or header.get("Content-Range") != f"bytes {read_start}-{read_end-1}/{size}" or
            len(payload) != read_end - read_start):
        raise OSError(f"Invalid range response for {start}-{end}")
    # A range owns rows whose first byte lies in [start, end).
    begin = payload.index(b"\n") + 1
    end_index = len(payload) if end == size else payload.index(b"\n", end - read_start - 1) + 1
    return csv.reader(io.StringIO(payload[begin:end_index].decode("utf-8"), newline=""))


def train_audit(queries, workers):
    url = f"{OSV}/train.csv"
    payload, header, status = fetch(f"{url}?audit_range=0-4095", {"Range": "bytes=0-4095"})
    if status != 206:
        raise OSError("Train metadata needs byte range support")
    size = int(header["Content-Range"].split("/")[-1])
    columns = next(csv.reader([payload.split(b"\n", 1)[0].decode()]))
    indices = [columns.index(k) for k in ("id", "latitude", "longitude", "country", "sequence")]
    query_key = hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest()[:12]
    cache = OUTPUT / f".train_parts_{size}_{BLOCK_BYTES}_{query_key}"
    cache.mkdir(exist_ok=True)
    ranges = [(a, min(a + BLOCK_BYTES, size)) for a in range(0, size, BLOCK_BYTES)]

    def shard(start, end):
        p = cache / f"{start:010}.json"
        if not p.exists():
            records = ranged_csv(url, start, end, size)
            atomic_json(p, accumulate(coordinate_rows(records, indices), queries))
        return json.loads(p.read_text())

    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(shard, start, end) for start, end in ranges]
        for i, future in enumerate(as_completed(pending), 1):
            parts.append(future.result())
            if i % 8 == 0 or i == len(pending):
                done = sum(sum(p["countries"].values()) for p in parts)
                print(f"Full train audit: {i}/{len(pending)} byte ranges, {done:,} rows", flush=True)
    return merge(parts, queries)


def test_audit(queries):
    with (ROOT / "data/raw/osv5m-test.csv").open(newline="", encoding="utf-8") as handle:
        docs = csv.reader(handle)
        columns = next(docs)
        positions = [columns.index(k) for k in ("id", "latitude", "longitude", "country", "sequence")]
        return merge([accumulate(coordinate_rows(docs, positions), queries)], queries)


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    queries = load_jsonl(ROOT / "data/pilot/pilot_dev.jsonl")
    if not all(42.3 <= q["latitude"] <= 50.7 and -0.4 <= q["longitude"] <= 7.4 for q in queries):
        raise ValueError("Pilot locations outside the coverage audit prefilter")
    gallery = load_jsonl(ROOT / "data/gallery_2000/gallery.jsonl")
    test = test_audit(queries)
    assert sum(test["countries"].values()) == TEST_ROWS
    gallery_stats = merge([accumulate(((r["id"], r["latitude"], r["longitude"], r["country"], r["sequence"])
                                       for r in gallery), queries)], queries)
    print("Full test metadata verified; scanning all train metadata ranges.", flush=True)
    train = train_audit(queries, args.workers)
    if sum(train["countries"].values()) != TRAIN_ROWS:
        raise ValueError(f"Unexpected train count: {sum(train['countries'].values())}")
    split_stats = {"train": train, "test": test, "gallery": gallery_stats}
    countries = sorted(set().union(*(stats["countries"] for stats in split_stats.values())))
    write_csv("countries.csv", [
        {"country": c, **{split: stats["countries"][c] for split, stats in split_stats.items()}}
        for c in countries
    ])
    grid = sorted(set().union(*(stats["grid"] for stats in split_stats.values())))
    write_csv("grid_2deg.csv", [
        {"south": a * 2 - 90, "west": b * 2 - 180,
         **{split: stats["grid"][(a, b)] for split, stats in split_stats.items()}}
        for a, b in grid
    ])
    write_csv("pilot_coverage.csv", [
        {**{k: q[k] for k in ("id", "source", "region")},
         **{f"{split}_{name}": value for split in ("train", "gallery")
            for name, value in split_stats[split]["coverage"][i].items()}}
        for i, q in enumerate(queries)
    ])
    atomic_json(OUTPUT / "summary.json", {
        "revision": OSV.split("/")[-1], "grid_degrees": 2, "coverage_radius_km": RADIUS,
        "paper_train_n": TRAIN_ROWS_PAPER,
        "splits": {split: {
            "n": sum(stats["countries"].values()), "country_codes": len(stats["countries"]),
            "top_countries": stats["countries"].most_common(12), "regions": stats["regions"],
            "covered_pilot": sum(k["n"] > 0 for k in stats["coverage"]),
        } for split, stats in split_stats.items()},
        "pilot_n": len(queries),
    })
    print((OUTPUT / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
