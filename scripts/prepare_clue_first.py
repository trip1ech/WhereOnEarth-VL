"""Lock audited photographs and package the exact no-move inference inputs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/clue_first"


def main():
    reviewed = json.loads((RUN / "review/verified.json").read_text())
    references = {r["ordinal"]: r for r in reviewed}
    if len(references) != len(reviewed):
        raise ValueError("Duplicate annotation")
    reviewed_ids = json.loads((RUN / "review/ids.json").read_text())
    rows = []
    with (ZipFile(ROOT / "data/author_holdout/author-holdout-eval.zip") as dataset,
          ZipFile(ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip") as trained,
          ZipFile(RUN / "inputs.zip", "w", compression=ZIP_DEFLATED) as bundle):
        labels = list(map(json.loads, dataset.open("eval.jsonl")))
        if [r["id"] for r in labels] != reviewed_ids:
            raise ValueError("Study photo order differs from reviewed photos")
        if not references.keys() <= set(range(1, len(labels) + 1)):
            raise ValueError("Review ordinal does not match the held-out set")
        for i, row in enumerate(labels, 1):
            image = dataset.read(row["image"])
            if hashlib.sha256(image).hexdigest() != row["sha256"]:
                raise ValueError(f"Photo hash differs for {i}")
            inference = {key: row[key] for key in ("id", "sha256", "image")}
            if i in references:
                marked = references[i]
                inference["verified_text"] = marked["text"]
                inference["crop_pixels"] = marked["crop_pixels"]
            rows.append(inference)
            bundle.writestr(row["image"], image)
        manifest = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
        bundle.writestr("inputs.json", manifest)
        for path in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            bundle.writestr(path, trained.read(path))
        for name in ("evaluate_clue_first.py", "phase1.py", "sft_colab.py"):
            bundle.write(ROOT / "scripts" / name, name)
        bundle.write(ROOT / "requirements-colab.txt", "requirements-colab.txt")
    (RUN / "inputs.json").write_text(manifest)
    print(f"Prepared {len(rows)} images and {len(references)} visually verified transcriptions.")


if __name__ == "__main__":
    main()
