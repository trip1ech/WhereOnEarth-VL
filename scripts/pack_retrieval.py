"""Package the frozen retrieval experiment into bounded Colab upload archives."""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    gallery = ROOT / "data" / f"gallery_{args.count}"
    manifest = gallery / "gallery.jsonl"
    refs = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(refs) != args.count:
        raise ValueError("Gallery count does not match its manifest")
    pilot = ROOT / "data/pilot"
    queries = [json.loads(line) for line in (pilot / "pilot_dev.jsonl").read_text().splitlines()]
    output = ROOT / f"retrieval_{args.count}_upload"
    output.mkdir(exist_ok=True)
    # JPEGs are already compressed. Bound each upload well below 64 MiB.
    chunks = [refs[i:i + 250] for i in range(0, len(refs), 250)]
    parts = []
    for index, chunk in enumerate(chunks):
        path = output / f"retrieval_gallery_{args.count}_{index:02}.zip"
        with ZipFile(path, "w", compression=ZIP_STORED) as zip_file:
            for row in chunk:
                zip_file.write(gallery / row["image"], "gallery/" + row["image"])
        parts.append(path.name)
    with ZipFile(output / f"retrieval_code_{args.count}.zip", "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("gallery_parts.json", json.dumps(parts))
        zip_file.write(manifest, "gallery/gallery.jsonl")
        for name in ("retrieval.py", "phase1.py"):
            zip_file.write(ROOT / "scripts" / name, name)
        zip_file.write(pilot / "pilot_dev.jsonl", "query/pilot_dev.jsonl")
        zip_file.write(ROOT / "data/raw/ne_110m_admin_0_countries.geojson", "map.geojson")
        for row in queries:
            zip_file.write(pilot / row["image"], "query/" + row["image"])
    print(f"Prepared {len(parts)+1} upload files in {output}")


if __name__ == "__main__":
    main()
