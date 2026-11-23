"""Stage a new-author OSV-5M test and produce contact sheets for visual review."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageOps

from phase1 import clean_image, haversine
from prepare_sft_data import DESTINATION, PARENT, download_selected

PROJECT = DESTINATION.parents[1]
OUTPUT = PROJECT / "data/author_holdout"
BUNDLE = DESTINATION / "geoguessr_sft_6000_upload.zip"


def read_jsonl(data):
    return [json.loads(line) for line in data]


def author(row):
    return (row.get("attribution") or "").strip().casefold()


def distance(a, b):
    return haversine(a["latitude"], a["longitude"], b["latitude"], b["longitude"])


def reproducible_order(row):
    return hashlib.sha256(f"author-holdout-v1:{row['id']}".encode()).digest()


def candidates():
    with ZipFile(BUNDLE) as bundle:
        train = read_jsonl(bundle.open("train.jsonl"))
        previous = read_jsonl(bundle.open("eval.jsonl"))
    with (PARENT / "validation_images.jsonl").open() as handle:
        pool = read_jsonl(handle)
    forbidden = {
        key: {row[key] for row in train + previous}
        for key in ("id", "sequence", "original_sha256")
    }
    seen_authors = {author(row) for row in train + previous}
    old_by_country = defaultdict(list)
    train_by_country = defaultdict(list)
    for row in previous:
        old_by_country[row["country"]].append(row)
    for row in train:
        train_by_country[row["country"]].append(row)
    eligible = defaultdict(list)
    for row in sorted(pool, key=reproducible_order):
        name = author(row)
        if (not name or name.isdecimal() or name in seen_authors
                or row["country"] not in old_by_country
                or row["country"] not in train_by_country
                or row["id"] in forbidden["id"]
                or row["sequence"] in forbidden["sequence"]
                or row["sha256"] in forbidden["original_sha256"]):
            continue
        nearest_train = min(distance(row, other) for other in train_by_country[row["country"]])
        if nearest_train < 1:
            continue
        nearest_old = min(distance(row, other) for other in old_by_country[row["country"]])
        eligible[row["country"]].append(dict(
            row, nearest_same_country_train_km=round(nearest_train, 3),
            nearest_same_country_previous_eval_km=round(nearest_old, 3),
        ))
    sizes = Counter(row["country"] for row in previous)
    for group in eligible.values():
        group.sort(key=lambda r: (
            r["nearest_same_country_previous_eval_km"] > 300,
            reproducible_order(r),
        ))
    used, result, quotas = set(), [], Counter()
    while eligible:
        country = min(eligible, key=lambda code: (
            quotas[code] / math.sqrt(sizes[code]), -sizes[code], code
        ))
        group = eligible[country]
        selected = next((row for row in group if author(row) not in used), None)
        if selected is None or quotas[country] == 8:
            del eligible[country]
            continue
        used.add(author(selected))
        quotas[country] += 1
        result.append(selected)
    return result


def sheet(rows, output, *, title, columns=3, label_offset=0):
    width, height = 340, 260
    canvas = Image.new("RGB", (width * columns, 42 + height * math.ceil(len(rows)/columns)), "#182637")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill="white", font_size=17)
    for index, row in enumerate(rows):
        path = row["path"]
        with Image.open(path) as source:
            preview = ImageOps.exif_transpose(source).convert("RGB")
            preview.thumbnail((width - 12, height - 32), Image.Resampling.LANCZOS)
        col, y = index % columns, 42 + (index // columns) * height
        x = col * width
        canvas.paste(preview, (x + (width - preview.width)//2, y))
        label = f"{label_offset+index+1:02}  {row.get('country', 'FR')}  {row['source']}  {row['id']}"
        draw.text((x + 5, y + height - 23), label, fill="white", font_size=14)
    canvas.save(output, quality=90)


def freeze():
    with (OUTPUT / "candidates.jsonl").open() as handle:
        rows = read_jsonl(handle)
    with (OUTPUT / "visual_review.csv").open() as handle:
        decisions = list(csv.DictReader(handle))
    if ([row["id"] for row in rows] != [decision["id"] for decision in decisions]
            or any(d["decision"] not in {"include", "exclude"} or not d["notes"] for d in decisions)):
        raise ValueError("The review must cover every staged image in order.")

    available = {row["id"]: row for row in candidates()}
    final, names, sequences, hashes = [], set(), set(), set()
    temporary = OUTPUT / "author-holdout-eval.part"
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as bundle:
        for row, decision in zip(rows, decisions):
            if decision["decision"] == "exclude":
                continue
            expected = available[row["id"]]
            if any(row[k] != expected[k] for k in expected if k != "image"):
                raise ValueError(f"Candidate metadata changed: {row['id']}")
            source = OUTPUT / row["image"]
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            name, sequence = author(row), row["sequence"]
            if (original_hash != row["sha256"] or original_hash in hashes
                    or name in names or sequence in sequences):
                raise ValueError(f"Repeated or unexpected image: {row['id']}")
            names.add(name)
            sequences.add(sequence)
            hashes.add(original_hash)
            photo = clean_image(source)
            photo.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            photo.save(buffer, format="JPEG", quality=90)
            data = buffer.getvalue()
            member = f"images/{row['id']}.jpg"
            bundle.writestr(member, data)
            final.append(dict(
                row, original_sha256=original_hash, sha256=hashlib.sha256(data).hexdigest(),
                image=member, review_note=decision["notes"],
            ))
        if not final:
            raise ValueError("No candidate passed review.")
        manifest = "".join(json.dumps(row) + "\n" for row in final)
        bundle.writestr("eval.jsonl", manifest)
    temporary.replace(OUTPUT / "author-holdout-eval.zip")
    (OUTPUT / "eval.jsonl").write_text(manifest)
    print(f"Frozen {len(final)} photos, {len(names)} authors, "
          f"{len({r['country'] for r in final})} countries.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--freeze", action="store_true", help="Pack images after the visual review")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = candidates()[:args.count]
    if len(selected) != args.count:
        raise ValueError(f"Only {len(selected)} eligible images")
    images = OUTPUT / "candidates"
    images.mkdir(exist_ok=True)
    rows = [dict(row, path=images / f"{row['id']}.jpg") for row in selected]
    print(f"Staging {len(rows)} photos for review, {len({r['country'] for r in rows})} countries.", flush=True)
    download_selected(rows, workers=10)
    payload = [{**row, "image": str(row["path"].relative_to(OUTPUT))} for row in rows]
    for row in payload:
        row.pop("path")
    (OUTPUT / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in payload))
    for start in range(0, len(rows), 12):
        sheet(rows[start:start+12], OUTPUT / f"osv-review-{start//12+1:02}.jpg",
              title="OSV-5M / new contributors / candidate images", label_offset=start)
    if args.count == 15:
        pilot = PROJECT / "data/pilot"
        with (pilot / "pilot_holdout.jsonl").open() as handle:
            external = [dict(row, path=pilot / row["image"])
                        for row in read_jsonl(handle) if row["source"] == "panoramax"]
        sheet(external, OUTPUT / "panoramax-review.jpg", title="Panoramax / existing France pilot")


if __name__ == "__main__":
    main()
