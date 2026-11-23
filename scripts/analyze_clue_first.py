"""Score the locked street-view clue experiment and preserve per-photo outcomes."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from phase1 import haversine
from sft_colab import parse_response, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/clue_first"
METHODS = ("direct", "model_text", "verified_text", "crop_text")


def score(answer, label):
    country, pos = parse_response(answer)
    return {
        "country": label["country"],
        "guess_country": country,
        "guess_coordinates": pos,
        "distance_km": haversine(label["latitude"], label["longitude"], *pos) if pos else None,
    }


def paired(rows, candidate):
    shared = [(r["direct"]["distance_km"], r[candidate]["distance_km"]) for r in rows
              if r["direct"]["distance_km"] is not None and r[candidate]["distance_km"] is not None]
    countries = [(r["direct"]["guess_country"] == r["country"],
                  r[candidate]["guess_country"] == r["country"]) for r in rows]
    return {
        "valid_in_both": len(shared),
        "direct_median_km": median(a for a, _ in shared) if shared else None,
        "candidate_median_km": median(b for _, b in shared) if shared else None,
        "median_per_photo_change_km": median(b - a for a, b in shared) if shared else None,
        "closer_by_over_1_km": sum(b < a - 1 for a, b in shared),
        "farther_by_over_1_km": sum(b > a + 1 for a, b in shared),
        "within_1_km_change": sum(abs(b - a) <= 1 for a, b in shared),
        "country_gained": sum(not a and b for a, b in countries),
        "country_lost": sum(a and not b for a, b in countries),
    }


def main():
    inputs = (OUT / "inputs.json").read_bytes()
    cases = json.loads(inputs)
    review = {r["ordinal"]: r for r in json.loads((OUT / "review/verified.json").read_text())}
    with ZipFile(OUT / "clue-first-results.zip") as archive:
        config = json.loads(archive.read("configuration.json"))
        answers = list(map(json.loads, archive.open("answers.jsonl")))
    if config["manifest_sha256"] != hashlib.sha256(inputs).hexdigest():
        raise ValueError("Run manifest and reviewed manifest differ")
    with ZipFile(ROOT / "data/author_holdout/author-holdout-eval.zip") as archive:
        manifest = archive.read("eval.jsonl")
        labels = list(map(json.loads, manifest.splitlines()))
    with ZipFile(ROOT / "results/author_holdout/author-holdout-results.zip") as archive:
        historical_config = json.loads(archive.read("configuration.json"))
        historical = list(map(json.loads, archive.open("sft.jsonl")))

    checks = ("model", "revision", "adapter_sha256")
    if (case_ids := [r["id"] for r in cases]) != [r["id"] for r in labels] or any(
        [r["id"] for r in table] != case_ids for table in (answers, historical)
    ):
        raise ValueError("Inconsistent photo order")
    if (hashlib.sha256(manifest).hexdigest() != historical_config["manifest_sha256"]
            or historical_config["prompt"] != config["direct_prompt"]
            or any(config[key] != historical_config[key] for key in checks)):
        raise ValueError("Historical baseline did not use the same setting")

    rows = []
    for i, (answer, label, case) in enumerate(zip(answers, labels, cases), 1):
        if label["sha256"] != case["sha256"]:
            raise ValueError(f"Image hash mismatch for ordinal {i}")
        manual = review.get(i)
        if bool(manual) != (answer["verified_text"] is not None):
            raise ValueError(f"Missing verified-text prediction at {i}")
        row = {
            "ordinal": i, "id": label["id"], "country": label["country"],
            "reference": [label["latitude"], label["longitude"]],
            "review_kind": manual["kind"] if manual else "unverified",
            "verified_strings": manual["text"] if manual else [],
            "read_full": answer["read_full"], "read_status": answer["read_status"],
            "model_strings": answer["model_strings"], "used_model_text": answer["used_model_text"],
            "read_crop": answer.get("read_crop"), "crop_strings": answer.get("crop_strings"),
        }
        for name in METHODS:
            raw = answer.get(name)
            row[name] = {"raw": raw, **score(raw, label)} if raw is not None else None
        rows.append(row)

    cohorts = {
        "all": rows,
        "auto_used": [r for r in rows if r["used_model_text"]],
        "verified": [r for r in rows if r["verified_text"] is not None],
        "named_destinations": [r for r in rows if r["review_kind"] == "named_destination"],
        "language_only": [r for r in rows if r["review_kind"] == "language_only"],
    }
    metrics = {}
    for cohort, subset in cohorts.items():
        metrics[cohort] = {"n": len(subset), "summary": {}, "versus_direct": {}}
        if not subset:
            continue
        for key in METHODS:
            if all(row[key] is not None for row in subset):
                metrics[cohort]["summary"][key] = summarize([row[key] for row in subset])
                if key != "direct":
                    metrics[cohort]["versus_direct"][key] = paired(subset, key)

    read_counts = Counter(r["read_status"] for r in rows)
    history_matches = sum(parse_response(previous["raw"]) == parse_response(row["direct"]["raw"])
                          for previous, row in zip(historical, rows))
    report = {"configuration": config, "read_counts": dict(read_counts),
              "history_direct_matches": history_matches, "metrics": metrics, "rows": rows}
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    header = ["ordinal", "id", "country", "review_kind", "read_status", "model_strings",
              "verified_strings", "crop_strings"]
    metric_fields = [f"{m}_{s}" for m in METHODS for s in ("guess_country", "distance_km")]
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header + metric_fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row[field] for field in header}
            for field in ("model_strings", "verified_strings", "crop_strings"):
                flat[field] = json.dumps(flat[field], ensure_ascii=False) if flat[field] is not None else ""
            for method in METHODS:
                if row[method] is not None:
                    for metric in ("guess_country", "distance_km"):
                        flat[f"{method}_{metric}"] = row[method][metric]
            writer.writerow(flat)

    print("Read outcomes:", dict(read_counts), "Original direct predictions matched:", history_matches)
    for cohort, excerpt in metrics.items():
        print(cohort, "n", excerpt["n"])
        for key, result in excerpt["summary"].items():
            print(" ", key, result)
        for key, result in excerpt["versus_direct"].items():
            print(" ", key, result)
    for row in rows:
        if row["used_model_text"] or row["verified_text"] is not None:
            print(row["ordinal"], row["country"], repr(row["read_full"]), repr(row["read_crop"]))
            for method in METHODS:
                if row[method] is not None:
                    d = row[method]["distance_km"]
                    print(" ", method, row[method]["guess_country"], round(d, 1) if d is not None else None)


if __name__ == "__main__":
    main()
