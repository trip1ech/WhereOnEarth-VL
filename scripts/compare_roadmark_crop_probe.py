"""Compare the locked full-image probe and the two corresponding crop probes."""
from collections import defaultdict
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROP = ROOT / "results/roadmark_crop_probe"
BASE = ROOT / "results/roadmark_probe"
VIEWS = {"full": BASE, "fixed": CROP / "fixed", "marked": CROP / "marked"}
METHODS = ("vit", "vit_native", "projector", "projector_deepstack", "rgb_full", "lm_base", "lm_sft")


def read_predictions(directory):
    with (directory / "out_of_fold.csv").open(newline="") as handle:
        return {(row["cohort"], row["image_id"]): row for row in csv.DictReader(handle)}


def correct(row, method):
    return (float(row[method]) > 0) == (row["true_label"] == "double")


def main():
    reports = {view: json.loads((directory / "analysis.json").read_text())
               for view, directory in VIEWS.items()}
    full_config = reports["full"]["inference_configuration"]
    for view in ("fixed", "marked"):
        config = reports[view]["inference_configuration"]
        invariants = ("model", "revision", "adapter_sha256", "feature_shapes", "prompt", "choices", "gpu")
        if config["view"] != view or any(config[key] != full_config[key] for key in invariants):
            raise ValueError(f"Unexpected inference configuration for {view}")
        if reports[view]["matched_pairs"] != reports["full"]["matched_pairs"]:
            raise ValueError(f"Matched cases changed for {view}")
    all_rows = {view: read_predictions(directory) for view, directory in VIEWS.items()}
    full_rows = all_rows["full"]
    if any(rows.keys() != full_rows.keys() for rows in all_rows.values()):
        raise ValueError("The views do not contain the same rows")
    groups = defaultdict(list)
    for key, baseline in full_rows.items():
        groups[key[0]].append(key)
        for view, rows in all_rows.items():
            if any(
                rows[key][field] != baseline[field] for field in ("fold", "true_label", "country")
            ):
                raise ValueError(f"Incompatible folds or labels for {view}")
    with (CROP / "crop_centers.csv").open(newline="") as handle:
        uncertain = {r["image_id"] for r in csv.DictReader(handle) if r["review_uncertain"] == "1"}

    comparison = {}
    for cohort, keys in groups.items():
        print(cohort)
        table = {}
        for method in METHODS:
            values = {}
            baseline_right = [correct(full_rows[k], method) for k in keys]
            for view, rows in all_rows.items():
                stats = reports[view]["analyses"][cohort]["scores"][method]
                right = [correct(rows[k], method) for k in keys]
                audited = [correct(rows[k], method) for k in keys if k[1] not in uncertain]
                values[view] = {
                    "summary": stats,
                    "gained_vs_full": sum(not a and b for a, b in zip(baseline_right, right)),
                    "lost_vs_full": sum(a and not b for a, b in zip(baseline_right, right)),
                    "visually_clearer_count": len(audited),
                    "visually_clearer_correct": sum(audited),
                }
            table[method] = values
            print(method, " ".join(
                f"{view}={100*v['summary']['balanced_accuracy']:.1f}% "
                f"({v['summary']['correct']}/{len(keys)}, "
                f"+{v['gained_vs_full']}/-{v['lost_vs_full']}, pair {v['summary']['pair_wins']}, "
                f"clear {v['visually_clearer_correct']}/{v['visually_clearer_count']})"
                for view, v in values.items()
            ))
        comparison[cohort] = table
    (CROP / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
