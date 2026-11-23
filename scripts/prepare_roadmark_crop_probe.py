"""Package fixed and manually centered crops of the same road-marking inputs."""
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "results/roadmark_probe"
CROP = ROOT / "results/roadmark_crop_probe"


def main():
    with (CROP / "crop_centers.csv").open(newline="") as handle:
        locations = list(csv.DictReader(handle))
    centers = {row["image_id"]: [float(row["x"]), float(row["y"])] for row in locations}

    with ZipFile(PROBE / "inference-input.zip") as source:
        inputs = [json.loads(line) for line in source.open("inputs.jsonl")]
        if len(centers) != len(locations) or centers.keys() != {row["id"] for row in inputs}:
            raise ValueError("Expected one crop center for every locked probe image")
        with ZipFile(CROP / "inference-input.zip", "w", compression=ZIP_DEFLATED) as destination:
            for info in source.infolist():
                if info.filename not in {"inputs.jsonl", "extract_roadmark_features.py"}:
                    destination.writestr(info, source.read(info))
            for row in inputs:
                row["crop_center"] = centers[row["id"]]
            destination.writestr("inputs.jsonl", "".join(json.dumps(row) + "\n" for row in inputs))
            destination.write(ROOT / "scripts/extract_roadmark_features.py", "extract_roadmark_features.py")

    for view in ("marked", "fixed"):
        (CROP / view).mkdir(exist_ok=True)
    print(f"Prepared {len(inputs)} images with crop centers. Class labels remain on the SSD.")


if __name__ == "__main__":
    main()
