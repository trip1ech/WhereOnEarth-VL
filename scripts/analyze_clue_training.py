"""Measure whether continuation improved evidence extraction and geolocation."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
from zipfile import ZipFile

from analyze_clue_first import paired, score
from clue_visual import load_visual_labels
from sft_colab import summarize

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/clue_first"
OUT = ROOT / "results/clue_first_training"

ANCHORS = {22: ("Βόλος", "Volos"), 24: ("residenti", "eccetto"), 31: ("汽车钥匙",),
           36: ("대아갈비",), 45: ("BOUAKE",)}


def read_bundle(path):
    with ZipFile(path) as archive:
        return json.loads(archive.read("configuration.json")), list(map(json.loads, archive.open("answers.jsonl")))


def plain(word):
    return "".join(c for c in unicodedata.normalize("NFKD", word.casefold()) if c.isalnum())


def keyed_answers(values):
    return {r["id"]: r for r in values}


def observations_metrics(rows, mode, annotations):
    recognized = 0
    surface, centerline = [], []
    exact, true_positive, false_positive, false_negative = 0, 0, 0, 0
    for row in rows:
        if row["ordinal"] not in annotations:
            continue
        expected = annotations[row["ordinal"]]
        found = row[f"{mode}_observations"]
        recognized += int(found is not None)
        observed = found or {}
        if expected["surface"] != "unclear":
            surface.append(observed.get("surface") == expected["surface"])
        if expected["centerline"] != "unclear":
            centerline.append(observed.get("centerline") == expected["centerline"])
        wanted, predicted = set(expected["roadside"]), set(observed.get("roadside", []))
        exact += int(found is not None and wanted == predicted)
        true_positive += len(wanted & predicted)
        false_positive += len(predicted - wanted)
        false_negative += len(wanted - predicted)
    denom = 2 * true_positive + false_positive + false_negative
    return {
        "n": len(annotations), "parsed": recognized,
        "surface": [sum(surface), len(surface)],
        "centerline": [sum(centerline), len(centerline)],
        "roadside_exact": exact,
        "roadside_micro_f1": 2 * true_positive / denom if denom else None,
    }


def text_metrics(rows, stage, no_readable):
    conditions = Counter(row[f"{stage}_read_status"] for row in rows)
    plain_cases = Counter(row[f"{stage}_read_status"] for row in rows if row["ordinal"] in no_readable)
    hit = {}
    for row in rows:
        if row["ordinal"] in ANCHORS:
            words = row[f"{stage}_strings"]
            hit[str(row["ordinal"])] = any(plain(ref) in plain(word) for ref in ANCHORS[row["ordinal"]]
                                           for word in words)
    return {"read_status": conditions, "no_readable_subset": plain_cases,
            "readable_anchor_hit": hit, "readable_anchor_total": sum(hit.values())}


def main():
    before_config, before = read_bundle(PILOT / "clue-first-results.zip")
    after_config, after = read_bundle(OUT / "clue-first-results.zip")
    visual_config, visual = read_bundle(OUT / "visual-results.zip")
    repaired_config, repaired = read_bundle(OUT / "format-control-results.zip")
    with ZipFile(OUT / "clue-first-lm-final.zip") as archive:
        final = json.loads(archive.read("configuration.json"))
        final_digest = hashlib.sha256(archive.read("adapter/adapter_model.safetensors")).hexdigest()
        train_log = list(map(json.loads, archive.open("training.jsonl")))
    if (final["input_adapter_sha256"] != before_config["adapter_sha256"]
            or final_digest in {final["input_adapter_sha256"], ""}
            or final_digest != after_config["adapter_sha256"]
            or final_digest != visual_config["final_adapter_sha256"]
            or len(train_log) != train_log[-1]["steps"]
            or final["training_manifest_sha256"] != hashlib.sha256((OUT / "training.jsonl").read_bytes()).hexdigest()):
        raise ValueError("Training or evaluation did not use the expected checkpoints")
    if (after_config["manifest_sha256"] != before_config["manifest_sha256"]
            or visual_config["manifest_sha256"] != before_config["manifest_sha256"]
            or repaired_config["manifest_sha256"] != before_config["manifest_sha256"]
            or repaired_config["adapter_sha256"] != before_config["adapter_sha256"]):
        raise ValueError("Evaluation manifest has changed")
    before, after, visual, repaired = [keyed_answers(values) for values in (before, after, visual, repaired)]

    with ZipFile(ROOT / "data/author_holdout/author-holdout-eval.zip") as dataset:
        labels = list(map(json.loads, dataset.open("eval.jsonl")))
    if not before.keys() == after.keys() == visual.keys() == repaired.keys() == {r["id"] for r in labels}:
        raise ValueError("Evaluation ID sets differ")
    rows = []
    for i, label in enumerate(labels, 1):
        key = label["id"]
        prior, updated, road = before[key], after[key], visual[key]
        cleaned = repaired[key]
        answers = {
            "direct": prior["direct"], "before_text": prior["model_text"],
            "before_visual": road["before"]["geo_visual"] or prior["direct"],
            "repaired_text": cleaned["geo_text"], "repaired_visual": cleaned["geo_road"],
            "after_direct": updated["direct"], "after_text": updated["model_text"],
            "after_visual": road["after"]["geo_visual"] or updated["direct"],
            "after_both": road["after"]["geo_both"],
        }
        rows.append({
            "ordinal": i, "id": key, "country": label["country"],
            "before_read_status": prior["read_status"], "after_read_status": updated["read_status"],
            "repaired_read_status": cleaned["read_status"], "repaired_strings": cleaned["strings"],
            "before_strings": prior["model_strings"], "after_strings": updated["model_strings"],
            "before_observations": road["before"]["observations"], "repaired_observations": cleaned["observations"],
            "after_observations": road["after"]["observations"],
            **{name: score(raw, label) for name, raw in answers.items()},
        })

    notes = OUT / "review"
    observations = load_visual_labels(notes / "holdout-visual.tsv")
    no_readable = {int(i) for i in (notes / "holdout-no-readable-text.txt").read_text().split()}
    methods = {name: summarize([row[name] for row in rows]) for name in answers}
    changes = {name: paired(rows, name) for name in answers if name != "direct"}
    visual_scores = {tag: observations_metrics(rows, tag, observations) for tag in ("before", "repaired", "after")}
    text_scores = {tag: text_metrics(rows, tag, no_readable) for tag in ("before", "repaired", "after")}
    result = {
        "training": final, "steps": len(train_log),
        "first_logged_loss": train_log[0]["loss"], "last_logged_loss": train_log[-1]["loss"],
        "mean_first_10_loss": sum(row["loss"] for row in train_log[:10]) / len(train_log[:10]),
        "mean_last_10_loss": sum(row["loss"] for row in train_log[-10:]) / len(train_log[-10:]),
        "geo": methods, "paired_geo": changes, "visual_read": visual_scores, "text_read": text_scores,
        "format_control": repaired_config,
        "original_and_final_adapter": [final["input_adapter_sha256"], final_digest],
        "rows": rows,
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "geo_metrics.csv").open("w", newline="") as file:
        rows_to_write = [dict(method=key, **row) for key, row in methods.items()]
        writer = csv.DictWriter(file, fieldnames=list(rows_to_write[0]))
        writer.writeheader()
        writer.writerows(rows_to_write)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, ensure_ascii=False))
    for row in rows:
        if row["ordinal"] in (*ANCHORS, 20, 47, 66):
            print(row["ordinal"], row["before_read_status"], repr(row["before_strings"]),
                  row["after_read_status"], repr(row["after_strings"]))


if __name__ == "__main__":
    main()
