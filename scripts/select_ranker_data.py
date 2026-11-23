"""Select sequence-disjoint OSV train manifests by measured geographic coverage."""
import csv
import gzip
import hashlib
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_distribution import TRAIN_ROWS
from cache_train_metadata import CACHE
from prepare_gallery import write_rows
from prepare_pilot import OSV, ROOT

OUTPUT = ROOT / "data/ranker_preprocessing"
BUDGETS = (10_000, 25_000, 50_000, 100_000)
QUERY_COUNTS = {"coverage": 6_000, "train": 12_000, "validation": 2_000}
EARTH_KM = 6371.0088
RADIUS = 2 * math.sin(25 / (2 * EARTH_KM))


def tile(lat, lon, degrees, longitude_counts):
    band = min(len(longitude_counts) - 1, int((lat + 90) / degrees))
    return band, min(longitude_counts[band] - 1, int((lon + 180) * longitude_counts[band] / 360))


def band_widths(degrees):
    return [max(1, round(360 * math.cos(math.radians(-90 + (i + .5) * degrees)) / degrees))
            for i in range(round(180 / degrees))]


def unit_vectors(rows):
    lat, lon = np.deg2rad(np.array([(r[1], r[2]) for r in rows]).T)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def capture_year(value):
    try:
        timestamp = float(value)
        seconds = timestamp / 1000 if abs(timestamp) >= 1e11 else timestamp
        return datetime.fromtimestamp(seconds, tz=timezone.utc).year
    except (ValueError, OverflowError, OSError):
        return None


def scan(paths, min_year=None):
    fine_widths = band_widths(.1)
    coarse_widths = band_widths(2)
    cells = {key: {} for key in ("gallery", *QUERY_COUNTS)}
    rng = random.Random(7818)
    total = 0
    random_gallery = []
    random_sequences = set()
    years = Counter()
    eligible = 0
    for path in paths:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for image_id, lat, lon, country, sequence, captured_at, creator, creator_id in csv.reader(handle):
                total += 1
                year = capture_year(captured_at)
                years[str(year) if year is not None else "unknown"] += 1
                if min_year is not None and (year is None or year < min_year):
                    continue
                if not sequence:
                    continue
                eligible += 1
                # Assign every image in a capture sequence to the same partition.
                sequence_key = int.from_bytes(hashlib.blake2b(sequence.encode(), digest_size=8).digest(), "little")
                bucket = (sequence_key & 65535) % 100
                split = "coverage" if bucket < 4 else "validation" if bucket < 8 else "train" if bucket < 20 else "gallery"
                lat, lon = float(lat), float(lon)
                sample_key = sequence_key >> 16
                if (split == "gallery" and sequence not in random_sequences and
                        (len(random_gallery) < max(BUDGETS) or sample_key < -random_gallery[0][0])):
                    coarse = tile(lat, lon, 2, coarse_widths)
                    sample = (image_id, lat, lon, country, sequence, captured_at, creator or creator_id, coarse, 0)
                    entry = (-sample_key, sequence, sample)
                    if len(random_gallery) == max(BUDGETS):
                        random_sequences.remove(heapq.heapreplace(random_gallery, entry)[1])
                    else:
                        heapq.heappush(random_gallery, entry)
                    random_sequences.add(sequence)
                fine = tile(lat, lon, .1, fine_widths)
                priority = rng.random()
                pool = cells[split].get(fine)
                if pool is not None and (priority >= pool[-1][8] and len(pool) == 2 or
                                         any(r[4] == sequence and r[8] <= priority for r in pool)):
                    continue
                coarse = tile(lat, lon, 2, coarse_widths)
                row = (image_id, lat, lon, country, sequence, captured_at, creator or creator_id, coarse, priority)
                updated = [r for r in pool or () if r[4] != sequence] + [row]
                cells[split][fine] = sorted(updated, key=lambda r: r[8])[:2]
    if total != TRAIN_ROWS:
        raise ValueError(f"Expected complete train metadata, got {total:,} rows")
    return cells, [entry[2] for entry in sorted(random_gallery, reverse=True)], years, eligible


def round_robin(rows, count):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r[8], reverse=True):
        groups[row[7]].append(row)
    seen = set()
    active = list(groups.values())
    while active and len(seen) < count:
        remaining = []
        for group in active:
            while group and group[-1][4] in seen:
                group.pop()
            if group:
                row = group.pop()
                seen.add(row[4])
                yield row
                if len(seen) == count:
                    return
            if group:
                remaining.append(group)
        active = remaining


def cover_first(cells, budget):
    # Demand has one actual train photo per occupied ~11 km tile. Points in
    # a sparser coarse region get more weight per tile than points in hubs.
    demand = [pool[0] for pool in cells.values()]
    candidates = demand + [pool[1] for pool in cells.values() if len(pool) > 1]
    coarse_counts = Counter(row[7] for row in demand)
    weights = np.array([1 / math.sqrt(coarse_counts[row[7]]) for row in demand])
    tree = cKDTree(unit_vectors(demand))
    neighbors = [np.asarray(idx, dtype=np.int32)
                 for idx in tree.query_ball_point(unit_vectors(candidates), RADIUS, return_sorted=True)]
    pending = [(-weights[idx].sum(), row[8], i) for i, (row, idx) in enumerate(zip(candidates, neighbors))]
    heapq.heapify(pending)
    selected = []
    sequences = set()
    while pending and len(selected) < budget:
        upper, priority, i = heapq.heappop(pending)
        row = candidates[i]
        if row[4] in sequences:
            continue
        gain = weights[neighbors[i]].sum()
        if gain < -upper:
            heapq.heappush(pending, (-gain, priority, i))
            continue
        selected.append(row)
        sequences.add(row[4])
        weights[neighbors[i]] = 0
    if len(selected) < budget:
        raise ValueError(f"Only {len(selected)} unique gallery sequences")
    return selected


def measure(gallery, queries):
    tree = cKDTree(unit_vectors(gallery))
    distance, _ = tree.query(unit_vectors(queries))
    hit = distance <= RADIUS
    groups = defaultdict(list)
    for row, covered in zip(queries, hit.tolist()):
        groups[row[7]].append(covered)
    return {"n": len(queries), "hits": int(hit.sum()), "coverage": float(hit.mean()),
            "macro_coarse_tile_coverage": float(np.mean([np.mean(v) for v in groups.values()])),
            "countries": len({r[3] for r in gallery})}


def serialize(row):
    image_id, lat, lon, country, sequence, captured_at, attribution, _, _ = row
    return dict(id=image_id, latitude=lat, longitude=lon, country=country, sequence=sequence,
                captured_at=captured_at, attribution=attribution, source="osv5m_train",
                license="CC-BY-SA-4.0", image=f"images/{image_id}.jpg")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=CACHE / OSV.rsplit("/", 1)[-1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-year", type=int, help="Earliest included UTC capture year")
    args = parser.parse_args()
    output = args.output or (OUTPUT if args.min_year is None else OUTPUT / f"since_{args.min_year}")
    output.mkdir(parents=True, exist_ok=True)
    cells, random_gallery, years, eligible = scan(sorted(args.metadata.glob("[0-9]*.csv.gz")), args.min_year)
    print({split: len(tiles) for split, tiles in cells.items()}, flush=True)
    queries = {split: list(round_robin((r for pool in cells[split].values() for r in pool), count))
               for split, count in QUERY_COUNTS.items()}
    if {k: len(v) for k, v in queries.items()} != QUERY_COUNTS:
        raise ValueError("A query partition could not meet its sequence budget")
    unique_sequences = len({r[4] for pool in cells["gallery"].values() for r in pool})
    limit = min(max(BUDGETS), unique_sequences, len(random_gallery))
    budgets = [budget for budget in BUDGETS if budget < limit] + [limit]
    gallery = cover_first(cells["gallery"], limit)
    curve = []
    for budget in budgets:
        point = {"gallery_n": budget, **measure(gallery[:budget], queries["coverage"]),
                 "random_sequence": measure(random_gallery[:budget], queries["coverage"])}
        curve.append(point)
        print(json.dumps(point), flush=True)
    for split, rows in queries.items():
        write_rows(output / f"query_{split}.jsonl", map(serialize, rows))
    write_rows(output / "gallery_ordered.jsonl", map(serialize, gallery))
    temporary = output / "coverage_curve.part"
    temporary.write_text(json.dumps({"revision": OSV.rsplit("/", 1)[-1], "train_rows": TRAIN_ROWS,
                                     "min_capture_year": args.min_year, "eligible_rows": eligible,
                                     "train_capture_years": dict(sorted(years.items())),
                                     "query_counts": QUERY_COUNTS, "curve": curve}, indent=2) + "\n")
    temporary.replace(output / "coverage_curve.json")


if __name__ == "__main__":
    main()
