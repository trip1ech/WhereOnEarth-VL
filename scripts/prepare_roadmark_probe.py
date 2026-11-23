"""Lock road-marking annotations and package only the images needed for inference."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/roadmark_probe"
TRAIN = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
ANNOTATIONS = ROOT / "data/roadmark_annotations/labels.json"
ADAPTER = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads(ANNOTATIONS.read_bytes())
    selected = {
        row["image_id"]: row for row in labels.values()
        if row["dataset"] == "train" and row["label"] in ("single", "double")
    }
    records = []
    blind = []
    with ZipFile(TRAIN) as train, ZipFile(ADAPTER) as adapter, ZipFile(
        OUTPUT / "inference-input.zip", "w", compression=ZIP_DEFLATED
    ) as bundle:
        manifest = {row["id"]: row for row in map(json.loads, train.open("train.jsonl"))}
        for image_id, label in sorted(selected.items()):
            row = manifest[image_id]
            payload = train.read(row["image"])
            if digest(payload) != row["sha256"] or row["sha256"] != label["image_sha256"]:
                raise ValueError(f"Labeled input bytes changed: {image_id}")
            member = f"images/{image_id}.jpg"
            records.append({
                "id": image_id, "label": label["label"], "country": row["country"],
                "contributor": row["attribution"].strip().casefold(),
                "latitude": row["latitude"], "longitude": row["longitude"], "sha256": row["sha256"],
            })
            blind.append({"id": image_id, "image": member, "sha256": row["sha256"]})
            bundle.writestr(member, payload)

        if any(not row["contributor"] for row in records):
            raise ValueError("Missing contributor metadata")
        bundle.writestr("inputs.jsonl", "".join(json.dumps(row) + "\n" for row in blind))
        for member in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            bundle.writestr(member, adapter.read(member))
        for path in ("scripts/phase1.py", "scripts/extract_roadmark_features.py", "requirements-colab.txt"):
            bundle.write(ROOT / path, Path(path).name)

    (OUTPUT / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    print(f"Prepared {len(records)} input images. Annotations and geography stay on the SSD.")


if __name__ == "__main__":
    main()
