"""Get the selected OSV train JPEGs and build frozen-embedding pair datasets."""
import argparse
import csv
import hashlib
import json
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from zlib import crc32, decompress

import numpy as np

from audit_distribution import atomic_json, load_jsonl
from phase1 import haversine
from prepare_gallery import TRAIN_ARCHIVES, write_rows
from prepare_pilot import OSV, ROOT, RemoteZip, fetch
from retrieval import HIT_KM, RAW_K, REVISION, assert_disjoint, extract_features
from select_ranker_data import OUTPUT

RESULTS = ROOT / "results/ranker_preprocessing"


def members(slot, wanted):
    url = f"{OSV}/images/train/{slot:02}.zip"
    with RemoteZip(url) as stream, ZipFile(stream) as archive:
        selected = [(row, m) for m in archive.infolist() if m.filename.endswith(".jpg")
                    and (row := wanted.get(m.filename.rsplit("/", 1)[-1][:-4])) is not None]
        return url, stream.size, sorted(selected, key=lambda item: item[1].header_offset)


def range_jobs(url, size, records):
    group = []
    start = end = 0
    for row, member in records:
        offset = member.header_offset
        upper = min(size, offset + member.compress_size + 4096)
        if group and (offset > end + 32768 or upper - start > 8 * 1024 * 1024):
            yield url, size, start, end, group
            group = []
        if not group:
            start = offset
        group.append((row, member))
        end = max(end, upper)
    if group:
        yield url, size, start, end, group


def get_images(url, size, start, end, group):
    needed = [(r, m) for r, m in group if not r["path"].is_file()]
    if needed:
        data, headers, status = fetch(f"{url}?slice={start}-{end-1}", {"Range": f"bytes={start}-{end-1}"})
        if (status != 206 or len(data) != end - start or
                headers.get("Content-Range") != f"bytes {start}-{end-1}/{size}"):
            raise OSError(f"Invalid archive range: {start}-{end-1} ({url})")
        for row, member in needed:
            offset = member.header_offset - start
            if data[offset:offset + 4] != b"PK\x03\x04":
                raise ValueError(f"Wrong local ZIP header for {member.filename}")
            name_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
            offset += 30 + name_length + extra_length
            payload = data[offset:offset + member.compress_size]
            if member.compress_type == ZIP_DEFLATED:
                payload = decompress(payload, -15)
            elif member.compress_type != ZIP_STORED:
                raise ValueError(f"Unsupported ZIP compression: {member.compress_type}")
            if len(payload) != member.file_size or crc32(payload) != member.CRC or not payload.startswith(b"\xff\xd8"):
                raise ValueError(f"Invalid JPEG payload: {member.filename}")
            path = row["path"]
            temporary = path.with_suffix(".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
    verified = []
    for row, member in group:
        payload = row["path"].read_bytes()
        if len(payload) != member.file_size or crc32(payload) != member.CRC:
            raise ValueError(f"Invalid cached JPEG: {member.filename}")
        verified.append(dict(row, sha256=hashlib.sha256(payload).hexdigest(), source_url=url,
                             archive_member=member.filename))
    return verified


def download(rows, workers):
    wanted = {r["id"]: r for r in rows}
    if len(wanted) != len(rows):
        raise ValueError("Duplicate IDs in selected partitions")
    archives = []
    with ThreadPoolExecutor(max_workers=min(16, workers)) as pool:
        tasks = [pool.submit(members, slot, wanted) for slot in range(TRAIN_ARCHIVES)]
        for i, task in enumerate(as_completed(tasks), 1):
            archives.append(task.result())
            if i % 20 == 0 or i == len(tasks):
                print(f"Read ZIP indexes {i}/{len(tasks)}", flush=True)
    archives.sort(key=lambda archive: archive[0])
    ids = [row["id"] for _, _, records in archives for row, _ in records]
    if len(ids) != len(wanted) or set(ids) != wanted.keys():
        raise ValueError(f"Archive lookup matched {len(set(ids))}/{len(wanted)} unique selected IDs")
    print(f"Located {len(ids):,} JPEGs in {TRAIN_ARCHIVES} archives.", flush=True)
    verified = {}
    jobs = [job for url, size, records in archives for job in range_jobs(url, size, records)]
    for concurrency in (workers, max(1, workers // 2), max(1, workers // 4)):
        retry = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            tasks = {pool.submit(get_images, *job): job for job in jobs}
            for i, task in enumerate(as_completed(tasks), 1):
                try:
                    verified.update((r["id"], r) for r in task.result())
                except (OSError, IncompleteRead) as exc:
                    if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                        raise
                    if not retry:
                        print(f"Transient JPEG download error: {type(exc).__name__}: {exc}", flush=True)
                    retry.append(tasks[task])
                if i % 600 == 0 or i == len(tasks):
                    print(f"JPEGs {len(verified):,}/{len(rows):,} · ranges {i:,}/{len(tasks):,}"
                          f" · queued for retry {len(retry):,}", flush=True)
        if not retry:
            break
        jobs = retry
    if len(verified) != len(rows):
        raise OSError(f"Only {len(verified):,}/{len(rows):,} JPEGs could be verified")
    return [verified[r["id"]] for r in rows]


def normalized(features, device):
    import torch

    return torch.nn.functional.normalize(torch.as_tensor(features, device=device, dtype=torch.float32), dim=-1)


def make_pairs(gallery, queries, refs, features, output):
    import torch
    from scipy.spatial import cKDTree
    from select_ranker_data import RADIUS, unit_vectors

    def geo_vectors(rows):
        return unit_vectors([(r["id"], r["latitude"], r["longitude"]) for r in rows])

    gallery_distances, _ = cKDTree(geo_vectors(gallery)).query(geo_vectors(queries))

    out_rows, pairs = [], []
    with torch.inference_mode():
        for start in range(0, len(queries), 128):
            values, indices = torch.topk(features[start:start + 128] @ refs.T, min(RAW_K, len(gallery)))
            for i, (candidate_indices, similarities) in enumerate(zip(indices.cpu().tolist(), values.cpu().tolist()), start):
                query = queries[i]
                distances = [haversine(query["latitude"], query["longitude"],
                                       gallery[j]["latitude"], gallery[j]["longitude"]) for j in candidate_indices]
                positives = [j for j, km in zip(candidate_indices, distances) if km <= HIT_KM]
                negatives = [j for j, km in zip(candidate_indices, distances) if km > HIT_KM]
                count = min(8, len(positives) * len(negatives))
                for k in range(count):
                    # Rotate across positives while drawing from the highest-ranked negatives.
                    cycle, positive = divmod(k, len(positives))
                    negative = (cycle + positive) % len(negatives)
                    pairs.append((i, positives[positive], negatives[negative]))
                out_rows.append(dict(query_id=query["id"], country=query["country"],
                                     gallery_within_25=bool(gallery_distances[i] <= RADIUS), gallery_indices=candidate_indices,
                                     cosine=similarities, distance_km=distances, positive_n=len(positives), pair_n=count))
    write_rows(output.with_name(output.name + "_candidates.jsonl"), out_rows)
    temporary = output.with_name(output.name + "_pairs.part")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(pairs, dtype=np.int32).reshape(-1, 3))
    temporary.replace(output.with_name(output.name + "_pairs.npy"))

    def counts(rows):
        return dict(query_n=len(rows), gallery_within_25=sum(r["gallery_within_25"] for r in rows),
                    top100_hits=sum(r["positive_n"] > 0 for r in rows),
                    top1_hits=sum(r["distance_km"][0] <= HIT_KM for r in rows),
                    pair_query_n=sum(r["pair_n"] > 0 for r in rows), pair_n=sum(r["pair_n"] for r in rows))

    by_country = defaultdict(list)
    for row in out_rows:
        by_country[row["country"]].append(row)
    summary = counts(out_rows)
    with output.with_name(output.name + "_coverage_by_country.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["country", *summary])
        writer.writeheader()
        writer.writerows(dict(country=code, **counts(rows)) for code, rows in sorted(by_country.items()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT, help="Directory of the selected dataset manifests")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode", action="store_true", help="Also encode images and mine pairs")
    parser.add_argument("--smoke", action="store_true", help="Verify a small subset of the locked dataset")
    args = parser.parse_args()
    if args.encode and args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Connect this notebook to a Colab GPU before encoding.")
    data = args.data_dir.resolve()
    output = RESULTS / str(args.gallery) if data == OUTPUT.resolve() else RESULTS / data.name / str(args.gallery)
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    (data / "images").mkdir(exist_ok=True)
    locked_gallery = data / f"gallery_{args.gallery}.jsonl"
    gallery = load_jsonl(locked_gallery) if locked_gallery.exists() else load_jsonl(data / "gallery_ordered.jsonl")[:args.gallery]
    selections = {"gallery": gallery,
                  **{split: load_jsonl(data / f"query_{split}.jsonl") for split in ("train", "validation")}}
    if len(selections["gallery"]) != args.gallery or args.gallery < RAW_K:
        raise ValueError("Selected gallery does not match requested size")
    if (data / "dataset.json").exists():
        lock = json.loads((data / "dataset.json").read_text())
        paths = {"gallery": locked_gallery, "train": data / "query_train.jsonl",
                 "validation": data / "query_validation.jsonl"}
        for split, path in paths.items():
            expected = lock["manifests"][split]
            if (path.name != expected["file"] or len(selections[split]) != expected["n"] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]):
                raise ValueError(f"Locked manifest differs: {path}")
    if args.smoke:
        selections = {key: rows[: 256 if key == "gallery" else 32] for key, rows in selections.items()}
    rows = [dict(r, path=data / r["image"]) for selection in selections.values() for r in selection]
    manifest = output / "verified_images.jsonl"
    if manifest.exists():
        restored = load_jsonl(manifest)
        if [r["id"] for r in restored] != [r["id"] for r in rows]:
            raise ValueError("Cached manifest ordering does not match selection")
        indexed = {r["id"]: dict(r, path=data / r["image"]) for r in restored}
    else:
        indexed = {r["id"]: r for r in download(rows, args.workers)}
        write_rows(manifest, ({k: v for k, v in r.items() if k != "path"} for r in indexed.values()))
    splits = {}
    seen_images = set()
    for split, selection in selections.items():
        splits[split] = []
        for r in selection:
            row = indexed[r["id"]]
            if split == "gallery" or row["sha256"] not in seen_images:
                splits[split].append(row)
                seen_images.add(row["sha256"])
        if not splits[split]:
            raise ValueError(f"No distinct images for {split}")
        write_rows(output / f"{split}_images.jsonl",
                   ({k: v for k, v in r.items() if k != "path"} for r in splits[split]))
    for first, second in (("gallery", "train"), ("gallery", "validation"), ("train", "validation")):
        assert_disjoint(splits[first], splits[second])
    if not args.encode:
        return
    features = {}
    for split, records in splits.items():
        features[split] = normalized(extract_features(records, output / f"{split}_features.npz",
                                                        device=args.device, batch_size=args.batch_size), args.device)
    summary = dict(gallery_n=len(splits["gallery"]), raw_k=RAW_K, hit_km=HIT_KM, encoder_revision=REVISION,
                   device=args.device,
                   duplicates_removed={k: len(selections[k]) - len(splits[k]) for k in splits}, splits={})
    for split in ("train", "validation"):
        summary["splits"][split] = make_pairs(splits["gallery"], splits[split], features["gallery"],
                                              features[split], output / split)
    atomic_json(output / "pair_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
