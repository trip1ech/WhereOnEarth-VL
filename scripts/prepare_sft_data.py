"""Prepare a deterministic OSV-5M QLoRA subset on the SSD (CPU only)."""
import argparse
import hashlib
import heapq
import json
import math
import random
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from PIL import Image

from audit_distribution import load_jsonl
from materialize_ranker_data import get_images, members, range_jobs
from phase1 import clean_image
from prepare_pilot import ROOT
from retrieval import assert_disjoint
from select_ranker_data import band_widths, tile

PARENT = ROOT / "results/ranker_preprocessing/since_2018/25000"
DESTINATION = ROOT / "data/qlora_since_2018"
SEED = 7818


def select_train(rows, size):
    """Favor coverage across countries and, within them, across geographic bins."""
    bands = band_widths(4)
    countries = defaultdict(lambda: defaultdict(deque))
    for row in rows:
        area = tile(row["latitude"], row["longitude"], 4, bands)
        countries[row["country"]][area].append(row)
    buckets = {country: deque(values.values()) for country, values in countries.items()}
    populations = Counter(row["country"] for row in rows)
    heap = [(1 / math.sqrt(n), country, 0) for country, n in populations.items()]
    heapq.heapify(heap)
    selected = []
    while len(selected) < size:
        _, country, selected_count = heapq.heappop(heap)
        country_buckets = buckets[country]
        area = country_buckets.popleft()
        selected.append(area.popleft())
        if area:
            country_buckets.append(area)
        if country_buckets:
            heapq.heappush(heap, ((selected_count + 2) / math.sqrt(populations[country]),
                                  country, selected_count + 1))
    return selected


def download_selected(rows, workers):
    by_archive = defaultdict(dict)
    for row in rows:
        if not row["path"].is_file():
            slot = int(row["archive_member"].split("/", 1)[0])
            by_archive[slot][row["id"]] = row
    jobs = []
    with ThreadPoolExecutor(max_workers=min(workers, 16)) as pool:
        for task in as_completed([pool.submit(members, slot, wanted) for slot, wanted in by_archive.items()]):
            jobs.extend(range_jobs(*task.result()))
    expected = sum(len(wanted) for wanted in by_archive.values())
    if sum(len(group) for _, _, _, _, group in jobs) != expected:
        raise ValueError("Original OSV-5M archive index is missing selected JPEGs")
    print(f"Found {expected} missing JPEGs in {len(jobs)} byte ranges.", flush=True)
    complete = 0
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task in as_completed([pool.submit(get_images, *job) for job in jobs]):
            try:
                complete += len(task.result())
            except OSError as exc:
                errors.append(exc)
            if complete and (complete % 100 < 3 or complete == expected):
                print(f"JPEGs downloaded {complete}/{expected}", flush=True)
    if errors:
        raise OSError(f"{len(errors)} byte ranges failed; completed JPEGs are cached") from errors[0]
    for row in rows:
        if hashlib.sha256(row["path"].read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image differs from locked manifest: {row['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--eval", type=int, default=300)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--bundle", type=Path, default=ROOT / "geoguessr_sft_upload.zip")
    parser.add_argument("--base-bundle", type=Path, help="Preserve the prior train prefix and exact evaluation images")
    args = parser.parse_args()
    train = load_jsonl(PARENT / "train_images.jsonl")
    validation = load_jsonl(PARENT / "validation_images.jsonl")
    if not 1 <= args.train <= len(train) or not 1 <= args.eval <= len(validation):
        parser.error("The requested train and evaluation sizes must fit the locked splits.")
    for rows in (train, validation):
        rows[:] = [row for row in rows if len(row.get("country", "")) == 2 and row["country"].isalpha()]
        rows.sort(key=lambda row: row["id"])
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(validation)
    if args.train > len(train) or args.eval > len(validation):
        raise ValueError("Too few rows with country labels.")
    subsets = {"train": select_train(train, args.train), "eval": validation[:args.eval]}
    assert_disjoint(subsets["train"], subsets["eval"])
    prior = {}
    if args.base_bundle:
        with ZipFile(args.base_bundle) as archive:
            for split, selection in subsets.items():
                previous = [json.loads(line) for line in archive.read(f"{split}.jsonl").splitlines()]
                size = len(previous)
                if split == "eval" and size != len(selection):
                    raise ValueError("The evaluation split must stay the same size.")
                if [row["id"] for row in previous] != [row["id"] for row in selection[:size]]:
                    raise ValueError(f"Selection changed in the previous {split} split")
                for old, selected in zip(previous, selection[:size], strict=True):
                    if old["original_sha256"] != selected["sha256"]:
                        raise ValueError(f"Original JPEG changed for {old['id']}")
                    prior[old["id"]] = old
                print(f"Preserving {size} exact {split} images from the prior run.", flush=True)
    (DESTINATION / "images").mkdir(parents=True, exist_ok=True)
    for split, selection in subsets.items():
        for row in selection:
            row["path"] = DESTINATION / row["image"]
        print(f"{split}: {len(selection)} photos; {len({r['country'] for r in selection})} country codes.", flush=True)
    download_selected(subsets["train"] + subsets["eval"], args.workers)
    bundle = args.bundle.resolve()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temp = bundle.with_suffix(".part")
    original = ZipFile(args.base_bundle) if args.base_bundle else nullcontext()
    with original as base, ZipFile(temp, "w", compression=ZIP_STORED) as archive:
        for split, selection in subsets.items():
            payload = []
            for row in selection:
                previous = prior.get(row["id"])
                if previous:
                    jpeg = base.read(row["image"])
                    if hashlib.sha256(jpeg).hexdigest() != previous["sha256"]:
                        raise ValueError(f"Prior bundle digest differs for {row['id']}")
                else:
                    image = clean_image(row["path"])
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=90)
                    jpeg = buffer.getvalue()
                archive.writestr(row["image"], jpeg)
                payload.append(dict(row, sha256=hashlib.sha256(jpeg).hexdigest(), original_sha256=row["sha256"]))
            manifest = "".join(json.dumps({k: v for k, v in row.items() if k != "path"}) + "\n" for row in payload)
            archive.writestr(f"{split}.jsonl", manifest, compress_type=ZIP_DEFLATED)
        for name, source in (("phase1.py", ROOT / "scripts/phase1.py"),
                             ("sft_colab.py", ROOT / "scripts/sft_colab.py"),
                             ("requirements-colab.txt", ROOT / "requirements-colab.txt")):
            archive.write(source, name, compress_type=ZIP_DEFLATED)
    temp.replace(bundle)
    print(f"Saved {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
