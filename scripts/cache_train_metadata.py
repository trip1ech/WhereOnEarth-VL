"""Cache only the train metadata fields needed for gallery and query manifests."""
import argparse
import csv
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

from audit_distribution import BLOCK_BYTES, TRAIN_ROWS, ranged_csv
from prepare_pilot import OSV, ROOT, fetch

FIELDS = ("id", "latitude", "longitude", "country", "sequence", "captured_at",
          "creator_username", "creator_id")
CACHE = ROOT / "data" / "ranker_preprocessing" / "train_metadata"


def cache_train(workers):
    url = f"{OSV}/train.csv"
    payload, headers, status = fetch(f"{url}?audit_range=0-4095", {"Range": "bytes=0-4095"})
    if status != 206:
        raise OSError("Train metadata needs byte range support")
    size = int(headers["Content-Range"].split("/")[-1])
    columns = next(csv.reader([payload.split(b"\n", 1)[0].decode()]))
    indices = [columns.index(field) for field in FIELDS]
    folder = CACHE / OSV.rsplit("/", 1)[-1]
    folder.mkdir(parents=True, exist_ok=True)

    def save(start):
        path = folder / f"{start:010d}.csv.gz"
        count_path = path.with_suffix(".count")
        if path.exists() and count_path.exists():
            return int(count_path.read_text())
        rows = ranged_csv(url, start, min(start + BLOCK_BYTES, size), size)
        temporary = path.with_suffix(".part")
        count = 0
        with gzip.open(temporary, "wt", compresslevel=2, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in rows:
                if row:
                    writer.writerow([row[i] for i in indices])
                    count += 1
        temporary.replace(path)
        temporary = count_path.with_suffix(".part")
        temporary.write_text(str(count))
        temporary.replace(count_path)
        return count

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(save, start) for start in range(0, size, BLOCK_BYTES)]
        total = 0
        for n, future in enumerate(as_completed(pending), 1):
            total += future.result()
            if n % 8 == 0 or n == len(pending):
                print(f"Metadata {n}/{len(pending)} ranges · {total:,} rows", flush=True)
    if total != TRAIN_ROWS:
        raise ValueError(f"Expected {TRAIN_ROWS:,} train rows, got {total:,}")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    print(cache_train(parser.parse_args().workers))
