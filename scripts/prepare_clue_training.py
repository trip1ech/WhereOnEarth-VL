"""Pack manually reviewed training targets and the unchanged evaluation images."""
import hashlib
import json
import random
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from clue_visual import load_visual_labels

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/clue_first_training"
REVIEW = RUN / "review"
HOLDOUT = ROOT / "results/clue_first"


def main():
    candidates = {r["ordinal"]: r for r in map(json.loads, (REVIEW / "candidates.jsonl").open())}
    text = {int(i): values for i, values in json.loads((REVIEW / "train-text.json").read_text()).items()}
    empty = {int(i) for i in (REVIEW / "train-empty.txt").read_text().split()}
    vision = load_visual_labels(REVIEW / "train-visual.tsv")
    if text.keys() & empty or not (text.keys() | empty | vision.keys()) <= candidates.keys():
        raise ValueError("Inconsistent train annotation IDs")
    replay = random.Random(52711).sample(sorted(vision.keys() - {406, 697, 771}), 54)
    geo_visual = [i for i in replay if vision[i]["surface"] != "unclear"][:34]
    training = []

    def add(i, task, **fields):
        data = candidates[i]
        training.append({**{key: data[key] for key in ("id", "sha256", "image")},
                         "task": task, **fields})

    for i, strings in sorted(text.items()):
        add(i, "ocr_evidence", evidence=strings)
    for i in sorted(empty):
        add(i, "ocr_empty", evidence=[])
    for i, observations in sorted(vision.items()):
        add(i, "visual", observations=observations)
    for task, pool in (("geo_direct", replay), ("geo_visual", geo_visual)):
        for i in pool:
            data = candidates[i]
            add(i, task, **{k: data[k] for k in ("latitude", "longitude", "country")},
                **({"observations": vision[i]} if task == "geo_visual" else {}))

    with (ZipFile(ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip") as source,
          ZipFile(HOLDOUT / "inputs.zip") as holdout,
          ZipFile(RUN / "training-input.zip", "w", compression=ZIP_DEFLATED) as bundle):
        manifest = json.loads(holdout.read("inputs.json"))
        ids = {r["id"] for r in training}
        if ids & {r["id"] for r in manifest} or {r["sha256"] for r in training} & {r["sha256"] for r in manifest}:
            raise ValueError("Training/evaluation overlap")
        for member in ("inputs.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                       *(r["image"] for r in manifest)):
            bundle.writestr("eval/" + member, holdout.read(member))
        images = {r["image"]: r["sha256"] for r in training}
        for path, expected in sorted(images.items()):
            jpg = source.read(path)
            if hashlib.sha256(jpg).hexdigest() != expected:
                raise ValueError(f"Training image differs: {path}")
            bundle.writestr(path, jpg)
        manifest = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in training)
        bundle.writestr("training.jsonl", manifest)
        for script in ("phase1.py", "sft_colab.py", "clue_visual.py", "evaluate_clue_first.py",
                       "train_clue_first.py", "evaluate_visual_clues.py"):
            bundle.write(ROOT / "scripts" / script, script)
        bundle.write(ROOT / "requirements-colab.txt", "requirements-colab.txt")
    (RUN / "training.jsonl").write_text(manifest)
    print(f"Packed {len(training)} task examples from {len(ids)} reviewed train photos.")


if __name__ == "__main__":
    main()
