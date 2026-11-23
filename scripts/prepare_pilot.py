"""Build a small, city-stratified pilot on the SSD with no training downloads."""
import csv
import hashlib
import io
import json
import re
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phase1 import haversine
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/pilot"
OSV = "https://huggingface.co/datasets/osv5m/osv5m/resolve/cff33609b56b54d8743b7ee7a416eb8433e9a681"
PANORAMAX = "https://panoramax.ign.fr/api/search"
REGIONS = {
    "paris_area": (1.80, 48.30, 2.90, 49.40),
    "lyon_area": (4.28, 45.20, 5.38, 46.30),
    "toulouse_area": (0.89, 43.05, 1.99, 44.15),
}
DEV_PER_CITY = 3
HOLDOUT_PER_CITY = 3
SEED = "geoguessr-phase01-v1"
_hub_cooldown_lock = threading.Lock()
_hub_cooldown_until = 0.0


def wait_for_hub_window():
    while True:
        with _hub_cooldown_lock:
            delay = _hub_cooldown_until - time.monotonic()
        if delay <= 0:
            return
        time.sleep(delay)


def pause_hub_requests(headers):
    global _hub_cooldown_until
    reset = re.search(r";\s*t=(\d+)", headers.get("RateLimit", ""))
    retry_after = headers.get("Retry-After", "").strip()
    delays = [int(reset[1])] if reset else []
    if retry_after.isdigit():
        delays.append(int(retry_after))
    seconds = max(delays, default=300) + 1
    now = time.monotonic()
    with _hub_cooldown_lock:
        if _hub_cooldown_until <= now:
            print(f"Hugging Face returned 429. Waiting {seconds}s for its request window to reset.", flush=True)
        _hub_cooldown_until = max(_hub_cooldown_until, now + seconds)


def fetch(url, headers=None):
    is_hub = url.startswith("https://huggingface.co/")
    attempt = throttles = 0
    while True:
        if is_hub:
            wait_for_hub_window()
        try:
            with urlopen(Request(url, headers=headers or {}), timeout=35) as response:
                return response.read(), response.headers, response.status
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            if is_hub and isinstance(exc, HTTPError) and exc.code == 429 and throttles < 6:
                throttles += 1
                pause_hub_requests(exc.headers)
                continue
            attempt += 1
            if attempt >= 3 or isinstance(exc, HTTPError) and exc.code in {400, 401, 403, 404, 416}:
                raise
            time.sleep(attempt)


class RemoteZip(io.RawIOBase):
    """Seek over a remote archive via validated HTTP ranges."""
    def __init__(self, url):
        self.url = url
        _, headers, status = fetch(url + "?slice=size", {"Range": "bytes=0-0"})
        if status != 206:
            raise OSError("Archive server did not honor a byte range")
        self.size = int(headers["Content-Range"].split("/")[-1])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.size)[whence]
        self.position = base + offset
        return self.position

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.position + count, self.size)
        if end <= self.position:
            return b""
        start = self.position
        payload, headers, status = fetch(
            f"{self.url}?slice={start}-{end-1}", {"Range": f"bytes={start}-{end-1}"}
        )
        if status != 206 or headers["Content-Range"] != f"bytes {start}-{end-1}/{self.size}" or len(payload) != end - start:
            raise OSError(f"Incomplete range {start}-{end-1} for {self.url}")
        self.position = end
        return payload


def rank(identifier):
    return hashlib.sha256(f"{SEED}:{identifier}".encode()).digest()


def in_bbox(lat, lon, bbox):
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east


def osv_rows():
    with (ROOT / "data/raw/osv5m-test.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["country"] != "FR":
                continue
            lat, lon = float(row["latitude"]), float(row["longitude"])
            for region, bbox in REGIONS.items():
                if in_bbox(lat, lon, bbox):
                    yield {
                        "id": row["id"], "latitude": lat, "longitude": lon, "region": region,
                        "sequence": row["sequence"], "license": "CC-BY-SA-4.0",
                        "attribution": row["creator_username"] or row["creator_id"],
                    }


def nearby_panorama_rows(osv_row):
    region = osv_row["region"]
    lat, lon = osv_row["latitude"], osv_row["longitude"]
    # Keep source comparison local to the same areas, with a radius recorded
    # in the manifest. This is a source comparison, not a device-only trial.
    for delta in (0.025, 0.06, 0.13):
        batch = []
        for x in (-1, 1):
            for y in (-1, 1):
                bbox = sorted((lon, lon + x * delta)) + sorted((lat, lat + y * delta))
                bbox = bbox[0], bbox[2], bbox[1], bbox[3]
                url = PANORAMAX + "?" + urlencode({"bbox": ",".join(map(str, bbox)), "limit": 35})
                content, _, _ = fetch(url)
                for feature in json.loads(content)["features"]:
                    props = feature["properties"]
                    lens = props.get("pers:interior_orientation") or {}
                    if lens.get("field_of_view") == 360 or "sd" not in feature["assets"]:
                        continue
                    item_lon, item_lat = feature["geometry"]["coordinates"][:2]
                    batch.append({
                        "id": feature["id"], "latitude": item_lat, "longitude": item_lon,
                        "region": region, "sequence": feature["collection"],
                        "license": props.get("license"), "attribution": props.get("geovisio:producer"),
                        "source_url": feature["assets"]["sd"]["href"],
                        "matched_osv_id": osv_row["id"],
                        "match_distance_km": haversine(lat, lon, item_lat, item_lon),
                    })
        yield from sorted(batch, key=lambda row: (row["match_distance_km"], rank(row["id"])))


def match_panorama_rows(rows):
    source_rows, external = [], []
    used_osv_sequences, used_panorama_sequences = set(), set()
    for region in REGIONS:
        count = 0
        for osv_row in sorted((row for row in rows if row["region"] == region), key=lambda row: rank(row["id"])):
            sequence = osv_row["sequence"]
            if not sequence or sequence in used_osv_sequences:
                continue
            match = next((row for row in nearby_panorama_rows(osv_row)
                          if row["sequence"] not in used_panorama_sequences
                          and row["license"] and row["attribution"]), None)
            if match is None:
                continue
            used_osv_sequences.add(sequence)
            used_panorama_sequences.add(match["sequence"])
            split = "pilot_dev" if count < DEV_PER_CITY else "pilot_holdout"
            for source, row in (("osv5m", osv_row), ("panoramax", match)):
                row.update(source=source, split=split, image=f"images/{source}-{row['id']}.jpg")
            source_rows.append(osv_row)
            external.append(match)
            count += 1
            print(f"{region} pair {count}: {match['match_distance_km']:.2f} km", flush=True)
            if count == DEV_PER_CITY + HOLDOUT_PER_CITY:
                break
        if count < DEV_PER_CITY + HOLDOUT_PER_CITY:
            raise RuntimeError(f"{region}: only {count} matches with disjoint sequences")
    return source_rows, external


def store_image(row, image):
    if not image.startswith(b"\xff\xd8"):
        raise ValueError(f"{row['id']} is not a JPEG")
    destination = PILOT / row["image"]
    destination.write_bytes(image)
    row["sha256"] = hashlib.sha256(image).hexdigest()


def osv_extract(url, rows):
    with zipfile.ZipFile(RemoteZip(url)) as archive:
        members = {Path(member).stem: member for member in archive.namelist() if member.lower().endswith(".jpg")}
        for row in rows:
            member = members.get(row["id"])
            if member:
                store_image(row, archive.read(member))
                row["source_url"] = url
                row["archive_member"] = member


def main():
    (PILOT / "images").mkdir(parents=True, exist_ok=True)
    print("Matching external sequences near OSV locations", flush=True)
    source_rows, external = match_panorama_rows(list(osv_rows()))
    print(f"Selected {len(source_rows)} OSV-5M records; extracting image ranges", flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda k: osv_extract(f"{OSV}/images/test/{k:02}.zip", source_rows), range(5)))
    if missing := [row["id"] for row in source_rows if "sha256" not in row]:
        raise RuntimeError(f"Missing OSV images: {missing}")
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda row: store_image(row, fetch(row["source_url"])[0]), external))
    rows = source_rows + external
    for split in ("pilot_dev", "pilot_holdout"):
        manifest = PILOT / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row["split"] == split))
    print(dict(Counter((row["source"], row["split"], row["region"]) for row in rows)))


if __name__ == "__main__":
    main()
