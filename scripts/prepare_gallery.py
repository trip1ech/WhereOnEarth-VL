"""Build a geographically stratified, train-only reference gallery on disk."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile
from zlib import crc32

from phase1 import coordinates
from prepare_pilot import OSV, ROOT, RemoteZip, fetch

SEED = "frozen-retrieval-v1"
BLOCKS = 144
BLOCK_BYTES = 96 * 1024
TRAIN_ARCHIVES = 98


def priority(identifier):
    return hashlib.sha256(f"{SEED}:{identifier}".encode()).digest()


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_metadata_window(url, size, start, length, columns):
    end = min(size, start + length)
    content, headers, status = fetch(
        f"{url}?slice={start}-{end-1}", {"Range": f"bytes={start}-{end-1}"},
    )
    if status != 206 or headers["Content-Range"] != f"bytes {start}-{end-1}/{size}" or len(content) != end - start:
        raise OSError(f"Wrong train.csv byte range: {start}-{end-1}")
    # The first and final physical lines can be truncated by the byte window.
    lines = content.decode("utf-8", errors="replace").splitlines()[1:-1]
    rows = []
    for values in csv.reader(lines):
        if len(values) != len(columns):
            continue
        row = dict(zip(columns, values))
        if not row["id"].isdigit() or not row["sequence"] or not row["country"]:
            continue
        lat, lon = coordinates(row["latitude"], row["longitude"])
        rows.append(dict(
            id=row["id"], sequence=row["sequence"], country=row["country"],
            latitude=lat, longitude=lon, captured_at=row["captured_at"],
            attribution=row["creator_username"] or row["creator_id"],
            source="osv5m_train", license="CC-BY-SA-4.0", image=f"images/{row['id']}.jpg",
        ))
    return rows


def sample_metadata(count, workers):
    url = f"{OSV}/train.csv"
    content, headers, status = fetch(url + "?slice=header", {"Range": "bytes=0-4095"})
    if status != 206 or len(content) != 4096:
        raise OSError("Expected a ranged train.csv header")
    columns = next(csv.reader([content.decode("utf-8", errors="replace").splitlines()[0]]))
    size = int(headers["Content-Range"].split("/")[-1])
    rng = random.Random(SEED)
    bytes_per_block = size // BLOCKS
    offsets = [index * bytes_per_block + rng.randrange(bytes_per_block - BLOCK_BYTES)
               for index in range(BLOCKS)]
    selected_by_id = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        requests = [pool.submit(read_metadata_window, url, size, start, BLOCK_BYTES, columns)
                    for start in offsets]
        for i, future in enumerate(as_completed(requests), 1):
            selected_by_id.update((r["id"], r) for r in future.result())
            if i % 24 == 0:
                print(f"Scanned {i}/{BLOCKS} metadata windows", flush=True)

    cells = defaultdict(list)
    for row in selected_by_id.values():
        grid = (int((row["latitude"] + 90) // 2), int((row["longitude"] + 180) // 2))
        cells[grid].append(row)
    pending = [deque(sorted(cells[key], key=lambda row: priority(row["id"])))
               for key in sorted(cells, key=lambda key: priority(str(key)))]
    selected, sequences = [], set()
    while pending and len(selected) < count:
        active = []
        for cell in pending:
            while cell and cell[0]["sequence"] in sequences:
                cell.popleft()
            if cell:
                row = cell.popleft()
                sequences.add(row["sequence"])
                selected.append(row)
                if len(selected) == count:
                    break
                if cell:
                    active.append(cell)
        pending = active
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} unique sequences available in sampled metadata")
    return selected


class GalleryArchive(RemoteZip):
    """Prefetch each selected ZIP record so its verified extraction uses one range."""
    cached = b""
    cache_offset = 0

    def read(self, count=-1):
        offset = self.position - self.cache_offset
        if count >= 0 and 0 <= offset <= len(self.cached) and offset + count <= len(self.cached):
            self.position += count
            return self.cached[offset:offset + count]
        return super().read(count)

    def prefetch(self, member):
        self.cache_offset = member.header_offset
        self.seek(self.cache_offset)
        self.cached = super().read(member.compress_size + 4096)


def extract_archive(slot, wanted, output):
    url = f"{OSV}/images/train/{slot:02}.zip"
    archive_path = output / "shards" / f"{slot:02}.jsonl"
    if archive_path.is_file():
        return [json.loads(line) for line in archive_path.read_text().splitlines()]
    stream = GalleryArchive(url)
    supplied = []
    with ZipFile(stream) as archive:
        for member in archive.infolist():
            image_id = Path(member.filename).stem
            if image_id not in wanted or not member.filename.endswith(".jpg"):
                continue
            row = dict(wanted[image_id], source_url=url, archive_member=member.filename)
            image_path = output / row["image"]
            if image_path.is_file():
                payload = image_path.read_bytes()
                checksum = crc32(payload)
                if checksum != member.CRC:
                    raise OSError(f"Existing gallery image failed ZIP checksum: {image_id}")
            else:
                stream.prefetch(member)
                payload = archive.read(member)
                partial = image_path.with_suffix(".part")
                partial.write_bytes(payload)
                partial.replace(image_path)
            if not payload.startswith(b"\xff\xd8"):
                raise ValueError(f"Not a JPEG in the train archive: {image_id}")
            supplied.append(dict(row, sha256=hashlib.sha256(payload).hexdigest()))
    write_rows(archive_path, supplied)
    return supplied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.count < 10 or args.workers < 1:
        parser.error("Count must be at least 10 and workers must be positive")
    output = ROOT / "data" / f"gallery_{args.count}"
    for name in ("images", "shards"):
        (output / name).mkdir(parents=True, exist_ok=True)
    selection = output / "selection.jsonl"
    if selection.is_file():
        rows = [json.loads(line) for line in selection.read_text().splitlines()]
    else:
        rows = sample_metadata(args.count, args.workers)
        write_rows(selection, rows)
    wanted = {row["id"]: row for row in rows}
    if len(wanted) != args.count:
        raise ValueError("Saved selection does not match requested count")
    print(f"Selected {len(rows)} train images in {len(set(r['country'] for r in rows))} countries", flush=True)
    found = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(extract_archive, i, wanted, output) for i in range(TRAIN_ARCHIVES)]
        for i, task in enumerate(as_completed(tasks), 1):
            found.update((r["id"], r) for r in task.result())
            if i % 8 == 0 or i == TRAIN_ARCHIVES:
                print(f"Checked {i}/{TRAIN_ARCHIVES} train archives, {len(found)}/{len(rows)} images ready", flush=True)
    if len(found) != args.count:
        raise RuntimeError(f"Missing {args.count - len(found)} selected train images")
    write_rows(output / "gallery.jsonl", (found[row["id"]] for row in rows))
    print(f"Gallery ready: {output} | country counts: {Counter(r['country'] for r in rows).most_common(8)}", flush=True)


if __name__ == "__main__":
    main()
