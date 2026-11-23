"""Select, download, OCR-audit, and lock a clean 300-image OSV-5M official test benchmark.

Guarantees:
1. Strictly from official OSV-5M test split (osv5m-test.csv, >= 2018-01-01 UTC).
2. Zero creator (username or creator_id), sequence, or image ID overlap with
   gallery_25000, train_12000, or validation_2000 (from ranker-2018-pairs.zip).
3. One image per unique unseen creator; square-root country stratification + 4-degree tile diversity.
4. Apple Vision OCR scan to flag and exclude dashcam GPS coordinate HUD overlays before locking 300 images.
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import re
import struct
import subprocess
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from zlib import crc32, decompress

from prepare_pilot import OSV, ROOT, RemoteZip, fetch

def tile4(lat, lon):
    band = max(0, min(44, int((lat + 90.0) // 4.0)))
    cos_lat = max(0.15, math.cos(math.radians(-90.0 + (band + 0.5) * 4.0)))
    lon_step = 4.0 / cos_lat
    col = int(((lon + 180.0) % 360.0) // lon_step)
    return (band, col)

TEST_CSV = ROOT / "data/raw/osv5m-test.csv"
PAIRS_ZIP = ROOT / "results/ranker-2018-pairs.zip"
OUT_DIR = ROOT / "data/official_test_300"
REVIEW_SWIFT = ROOT / "results/text_attention/selection/review.swift"
SINCE_2018_MS = 1514764800000
SEED = 20261123
TEST_ARCHIVES = 5

# Regex to catch dashcam GPS coordinate HUD overlays (e.g., "N48.1648", "-28.7536S", "134.5008E", "48.8566, 2.3522")
GPS_HUD_RE = re.compile(
    r"(?:"
    r"[NSEW]\s*[-+]?\d{1,3}\.\d{3,}"
    r"|[-+]?\d{1,3}\.\d{3,}\s*[°]?\s*[NSEW]"
    r"|[-+]?\d{1,2}\.\d{4,}\s*[,/ ]\s*[-+]?\d{1,3}\.\d{4,}"
    r"|\b\d{1,3}\.\d\s*(?:KM/H|KPH|MPH)\b"
    r")",
    re.IGNORECASE,
)


def select_candidates(target_n=340):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(PAIRS_ZIP) as z:
        known_rows = [
            json.loads(l)
            for name in ("gallery_images.jsonl", "train_images.jsonl", "validation_images.jsonl")
            for l in z.read(name).splitlines()
        ]
    known_creators = {r["attribution"] for r in known_rows if r.get("attribution")}
    known_seqs = {r["sequence"] for r in known_rows if r.get("sequence")}
    known_ids = {r["id"] for r in known_rows}

    # Group eligible rows by unique unseen creator (pick 1 deterministic photo per creator)
    rng = random.Random(SEED)
    by_creator = defaultdict(list)
    with TEST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cc = (row["country"] or "").strip().upper()
            if len(cc) != 2 or not cc.isalpha():
                continue
            try:
                ts = int(float(row["captured_at"]))
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except (ValueError, TypeError):
                continue
            if ts < SINCE_2018_MS or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            uname = (row["creator_username"] or "").strip()
            cid = (row["creator_id"] or "").strip()
            attr = uname or cid
            seq = (row["sequence"] or "").strip()
            rid = (row["id"] or "").strip()
            if not attr or not seq or not rid:
                continue
            if (attr in known_creators or uname in known_creators or cid in known_creators
                    or seq in known_seqs or rid in known_ids):
                continue
            by_creator[attr].append({
                "id": rid,
                "latitude": lat,
                "longitude": lon,
                "country": cc,
                "sequence": seq,
                "captured_at": str(ts),
                "attribution": attr,
                "creator_username": uname,
                "creator_id": cid,
                "region": (row.get("region") or "").strip(),
                "city": (row.get("city") or "").strip(),
                "source": "osv5m_test",
                "license": "CC-BY-SA-4.0",
            })

    # Pick 1 photo per unique unseen creator deterministically
    one_per_creator = []
    for attr in sorted(by_creator):
        items = sorted(by_creator[attr], key=lambda r: r["id"])
        one_per_creator.append(rng.choice(items))
    rng.shuffle(one_per_creator)
    print(f"Eligible unseen-creator pool (1/creator): {len(one_per_creator)} creators across "
          f"{len({r['country'] for r in one_per_creator})} countries", flush=True)

    # Square-root country stratification + 4-degree tile round-robin (matching prepare_sft_data.py)
    countries = defaultdict(lambda: defaultdict(deque))
    for r in one_per_creator:
        area = tile4(r["latitude"], r["longitude"])
        countries[r["country"]][area].append(r)
    buckets = {cc: deque(areas.values()) for cc, areas in countries.items()}
    pops = Counter(r["country"] for r in one_per_creator)
    heap = [(1.0 / math.sqrt(n), cc, 0) for cc, n in pops.items()]
    heapq.heapify(heap)

    selected = []
    while len(selected) < target_n and heap:
        _, cc, cnt = heapq.heappop(heap)
        cb = buckets[cc]
        area = cb.popleft()
        selected.append(area.popleft())
        if area:
            cb.append(area)
        if cb:
            heapq.heappush(heap, ((cnt + 2) / math.sqrt(pops[cc]), cc, cnt + 1))

    cand_path = OUT_DIR / "candidates.jsonl"
    cand_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    print(f"Selected {len(selected)} candidates across {len({r['country'] for r in selected})} countries -> {cand_path}", flush=True)
    print("Top 12 countries in candidates:", Counter(r["country"] for r in selected).most_common(12), flush=True)


def members_test(slot, wanted):
    url = f"{OSV}/images/test/{slot:02}.zip"
    with RemoteZip(url) as stream, ZipFile(stream) as archive:
        found = [
            (row, m)
            for m in archive.infolist()
            if m.filename.endswith(".jpg") and (row := wanted.get(m.filename.rsplit("/", 1)[-1][:-4])) is not None
        ]
        return url, stream.size, sorted(found, key=lambda x: x[1].header_offset)


def range_jobs(url, size, records):
    group, start, end = [], 0, 0
    for row, m in records:
        off = m.header_offset
        upper = min(size, off + m.compress_size + 4096)
        if group and (off > end + 32768 or upper - start > 8 * 1024 * 1024):
            yield url, size, start, end, group
            group = []
        if not group:
            start = off
        group.append((row, m))
        end = max(end, upper)
    if group:
        yield url, size, start, end, group


def fetch_group(url, size, start, end, group, img_dir):
    needed = [(r, m) for r, m in group if not (img_dir / f"{r['id']}.jpg").is_file()]
    if needed:
        data, headers, status = fetch(f"{url}?slice={start}-{end-1}", {"Range": f"bytes={start}-{end-1}"})
        if status != 206 or len(data) != end - start or headers.get("Content-Range") != f"bytes {start}-{end-1}/{size}":
            raise OSError(f"Bad range {start}-{end-1} for {url}")
        for r, m in needed:
            off = m.header_offset - start
            if data[off:off + 4] != b"PK\x03\x04":
                raise ValueError(f"Bad ZIP header {m.filename}")
            nlen, elen = struct.unpack_from("<HH", data, off + 26)
            off += 30 + nlen + elen
            payload = data[off:off + m.compress_size]
            if m.compress_type == ZIP_DEFLATED:
                payload = decompress(payload, -15)
            elif m.compress_type != ZIP_STORED:
                raise ValueError(f"Unsupported compression {m.compress_type}")
            if len(payload) != m.file_size or crc32(payload) != m.CRC or not payload.startswith(b"\xff\xd8"):
                raise ValueError(f"Invalid JPEG {m.filename}")
            dest = img_dir / f"{r['id']}.jpg"
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
    out = []
    for r, m in group:
        payload = (img_dir / f"{r['id']}.jpg").read_bytes()
        if len(payload) != m.file_size or crc32(payload) != m.CRC:
            raise ValueError(f"CRC mismatch {m.filename}")
        out.append(dict(r, image=f"images/{r['id']}.jpg", sha256=hashlib.sha256(payload).hexdigest(),
                        source_url=url, archive_member=m.filename))
    return out


def download_candidates(workers=16):
    img_dir = OUT_DIR / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (OUT_DIR / "candidates.jsonl").read_text().splitlines()]
    wanted = {r["id"]: r for r in rows}
    archives = []
    with ThreadPoolExecutor(max_workers=TEST_ARCHIVES) as pool:
        for task in as_completed([pool.submit(members_test, s, wanted) for s in range(TEST_ARCHIVES)]):
            archives.append(task.result())
    found_ids = {r["id"] for _, _, recs in archives for r, _ in recs}
    print(f"Located {len(found_ids)}/{len(wanted)} JPEGs in {TEST_ARCHIVES} official test archives", flush=True)
    if len(found_ids) != len(wanted):
        raise ValueError(f"Missing {len(wanted) - len(found_ids)} IDs in official test archives")

    jobs = [job for url, size, recs in sorted(archives) for job in range_jobs(url, size, recs)]
    verified = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fetch_group, *job, img_dir) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            for rec in fut.result():
                verified[rec["id"]] = rec
            if i % 50 == 0 or i == len(futs):
                print(f"Downloaded {len(verified)}/{len(rows)} JPEGs ({i}/{len(futs)} ranges)", flush=True)

    ordered = [verified[r["id"]] for r in rows]
    (OUT_DIR / "candidates_verified.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    print(f"Verified all {len(ordered)} official test candidate JPEGs.", flush=True)


def filter_and_lock(final_n=300):
    img_dir = OUT_DIR / "raw_images"
    rows = [json.loads(l) for l in (OUT_DIR / "candidates_verified.jsonl").read_text().splitlines()]
    paths = [str(img_dir / f"{r['id']}.jpg") for r in rows]

    print(f"Running macOS Apple Vision OCR scan on {len(paths)} images to detect GPS HUD overlays...", flush=True)
    ocr_by_id = {}
    # Run in batches of 60 so CLI args stay short
    for start in range(0, len(paths), 60):
        batch = paths[start:start + 60]
        proc = subprocess.run(["swift", str(REVIEW_SWIFT), *batch], capture_output=True, text=True, check=True)
        for line in proc.stdout.splitlines():
            obj = json.loads(line)
            ocr_by_id[obj["id"]] = obj["lines"]

    flagged = []
    clean = []
    for r in rows:
        lines = ocr_by_id.get(r["id"], [])
        hud_hits = []
        for item in lines:
            txt = item["text"]
            y0, y1 = item["box"][1], item["box"][3]
            # Check for GPS coordinates anywhere, or speed telemetry near top/bottom edge
            if GPS_HUD_RE.search(txt):
                hud_hits.append(f"{txt} (y={y0:.2f}-{y1:.2f})")
            elif (y0 < 0.15 or y1 > 0.85) and re.search(r"\b(?:LAT|LON|GPS|ALT|KM/H|KPH)\b", txt, re.I):
                hud_hits.append(f"{txt} (edge y={y0:.2f}-{y1:.2f})")
        if hud_hits:
            flagged.append({"id": r["id"], "country": r["country"], "attribution": r["attribution"], "hits": hud_hits})
        else:
            clean.append(r)

    (OUT_DIR / "hud_audit.json").write_text(json.dumps({
        "scanned": len(rows), "flagged_hud": len(flagged), "clean_remaining": len(clean), "flagged": flagged,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"OCR HUD scan: {len(flagged)}/{len(rows)} flagged for GPS/telemetry overlay, {len(clean)} clean remaining.", flush=True)
    for item in flagged[:10]:
        print(f"  FLAGGED {item['id']} ({item['country']}, @{item['attribution']}): {item['hits']}", flush=True)

    if len(clean) < final_n:
        raise ValueError(f"Only {len(clean)} clean images left, need {final_n}")

    locked = clean[:final_n]
    # Write locked test.jsonl + official-test-300.zip
    (OUT_DIR / "test.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in locked))
    zpath = OUT_DIR / "official-test-300.zip"
    tmp = zpath.with_suffix(".part")
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED) as z:
        z.write(OUT_DIR / "test.jsonl", "test.jsonl")
        for r in locked:
            z.write(img_dir / f"{r['id']}.jpg", r["image"], compress_type=ZIP_STORED)
    tmp.replace(zpath)
    print(f"Locked official_test_300: {len(locked)} images across {len({r['country'] for r in locked})} countries, "
          f"{len({r['attribution'] for r in locked})} unique unseen creators -> {zpath} ({zpath.stat().st_size / 1e6:.2f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--filter-and-lock", action="store_true")
    parser.add_argument("--candidates", type=int, default=340)
    parser.add_argument("--final-n", type=int, default=300)
    args = parser.parse_args()
    if args.select_only:
        select_candidates(args.candidates)
    if args.download:
        download_candidates()
    if args.filter_and_lock:
        filter_and_lock(args.final_n)


if __name__ == "__main__":
    main()
