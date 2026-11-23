"""Fit fixed linear readouts and score the same held-out road-marking photos."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from phase1 import haversine

METHODS = (
    "country_only", "rgb_top", "rgb_full", "vit", "projector", "projector_deepstack",
    "vit_native", "vit_global", "projector_global", "lm_base", "lm_sft",
)


def matched_pairs(rows):
    by_country = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        by_country[row["country"]][row["label"]].append(i)

    pairs = []
    for country, groups in sorted(by_country.items()):
        singles, doubles = groups["single"], groups["double"]
        if not singles or not doubles:
            continue
        distances = [[haversine(rows[s]["latitude"], rows[s]["longitude"],
                                rows[d]["latitude"], rows[d]["longitude"])
                      for d in doubles] for s in singles]
        si, di = linear_sum_assignment(distances)
        for s, d in zip(si, di):
            pairs.append((country, singles[s], doubles[d], distances[s][d]))
    return pairs


def normalized_parts(features):
    pieces = [features["projector"]]
    pieces.extend(features[key] for key in sorted(features) if key.startswith("deepstack_"))
    return np.concatenate([Normalizer().fit_transform(values) for values in pieces], axis=1)


def country_scores(y, countries, train, test):
    prior = y[train].mean()
    decision = []
    for country in countries[test]:
        peers = train[countries[train] == country]
        decision.append(y[peers].mean() - prior if len(peers) else 0.0)
    return np.array(decision)


def run_probes(y, groups, countries, universe, splitter, features, lm):
    results = {name: np.full(len(y), np.nan) for name in METHODS}
    folds = np.full(len(y), -1)
    for fold, (local_train, local_test) in enumerate(
        splitter.split(universe, y[universe], groups[universe]), 1
    ):
        train, test = universe[local_train], universe[local_test]
        if set(groups[train]) & set(groups[test]) or len(np.unique(y[train])) != 2:
            raise ValueError("Invalid split for binary readout")
        if set(folds[test]) != {-1}:
            raise ValueError("Duplicate held-out prediction")
        folds[test] = fold
        for name, scores in results.items():
            if name == "country_only":
                scores[test] = country_scores(y, countries, train, test)
            elif name.startswith("lm_"):
                scores[test] = lm[name][test].mean(axis=1)
            else:
                model = make_pipeline(
                    StandardScaler(), Normalizer(),
                    LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000)
                )
                model.fit(features[name][train], y[train])
                scores[test] = model.decision_function(features[name][test])

    if np.any(folds[universe] == -1) or any(not np.isfinite(s[universe]).all() for s in results.values()):
        raise ValueError("Missing held-out predictions")
    return folds, results


def summarize(y, universe, results, pairs=()):
    reference = y[universe]
    table = {}
    for name, scores in results.items():
        predicted = (scores[universe] > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(reference, predicted, labels=[0, 1]).ravel()
        pair_wins = sum(scores[double] > scores[single] for _, single, double, _ in pairs)
        pair_ties = sum(scores[double] == scores[single] for _, single, double, _ in pairs)
        table[name] = {
            "n": len(universe), "correct": int((reference == predicted).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(reference, predicted)),
            "single_correct": int(tn), "single_n": int(tn + fp),
            "double_correct": int(tp), "double_n": int(tp + fn),
            "pair_wins": int(pair_wins) if pairs else None,
            "pair_ties": int(pair_ties) if pairs else None,
        }
    return table


def compared(y, universe, scores, before, after):
    correct_before = (scores[before][universe] > 0) == y[universe]
    correct_after = (scores[after][universe] > 0) == y[universe]
    return {
        "gained": int(np.sum(~correct_before & correct_after)),
        "lost": int(np.sum(correct_before & ~correct_after)),
    }


def lm_diagnostics(y, universe, lm, scores):
    shift = (scores["lm_sft"] - scores["lm_base"])[universe]
    modes = {}
    for method, margins in lm.items():
        options = margins[universe] > 0
        modes[method] = {
            "choice_order_agreement": int(np.sum(options[:, 0] == options[:, 1])),
            "double_predictions_by_choice_order": options.sum(axis=0).tolist(),
            "double_predictions_after_averaging": int(np.sum(scores[method][universe] > 0)),
            "roc_auc_after_averaging": float(roc_auc_score(y[universe], scores[method][universe])),
        }
    return {
        "modes": modes,
        "images_with_lower_double_margin_after_lora": int(np.sum(shift < 0)),
        "median_double_margin_change_after_lora": float(np.median(shift)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/roadmark_probe"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--inference-input", type=Path)
    args = parser.parse_args()
    output = args.results
    labels = args.labels or output / "labels.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(row["label"] for row in rows) != {"single", "double"}:
        raise ValueError("Expected unique images and both road-marking labels")
    with ZipFile(output / "roadmark-features.zip") as archive:
        config = json.loads(archive.read("configuration.json"))
        inference = [json.loads(line) for line in archive.open("lm_scores.jsonl")]
        with np.load(io.BytesIO(archive.read("features.npz")), allow_pickle=False) as values:
            if ids != values["ids"].tolist() or ids != [r["id"] for r in inference]:
                raise ValueError("Feature and label image orders differ")
            features = {key: values[key] for key in values.files if key != "ids"}
    with ZipFile(args.inference_input or output / "inference-input.zip") as archive:
        manifest = archive.read("inputs.jsonl")
        blind_inputs = [json.loads(line) for line in manifest.splitlines()]
        if (hashlib.sha256(manifest).hexdigest() != config["manifest_sha256"]
                or [(row["id"], row["sha256"]) for row in blind_inputs]
                != [(row["id"], row["sha256"]) for row in rows]):
            raise ValueError("The inference bundle does not match the locked annotations")
        if hashlib.sha256(archive.read("adapter/adapter_model.safetensors")).hexdigest() != config["adapter_sha256"]:
            raise ValueError("The reported LoRA does not match the inference bundle")
    if config["n"] != len(ids) or any(
        list(x.shape) != config["feature_shapes"][key] or not np.isfinite(x).all()
        for key, x in features.items()
    ):
        raise ValueError("Unexpected feature shape or non-finite values")

    features["projector_deepstack"] = normalized_parts(features)
    lm = {f"lm_{mode}": np.array([[p["double_margin"] for p in row[mode]] for row in inference])
          for mode in ("base", "sft")}
    y = np.array([row["label"] == "double" for row in rows], dtype=int)
    contributors = np.array([row["contributor"] for row in rows])
    countries = np.array([row["country"] for row in rows])
    pairs = matched_pairs(rows)
    matched = np.array(sorted(i for _, single, double, _ in pairs for i in (single, double)))
    if (len(matched) != len(set(matched)) or len(set(countries[matched])) < 2
            or len(set(contributors[matched])) != len(matched)):
        raise ValueError("Matched leave-country-out set does not meet the split checks")

    analyses = {}
    cases = (
        ("all_author_held_out", np.arange(len(rows)), contributors,
         StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7818), []),
        ("matched_country_held_out", matched, countries, LeaveOneGroupOut(), pairs),
    )
    predictions = []
    for name, universe, groups, splitter, cohort_pairs in cases:
        folds, scores = run_probes(y, groups, countries, universe, splitter, features, lm)
        analyses[name] = {
            "scores": summarize(y, universe, scores, cohort_pairs),
            "before_after_main_merger": compared(y, universe, scores, "vit", "projector"),
            "before_after_lm_lora": compared(y, universe, scores, "lm_base", "lm_sft"),
            "lm_diagnostics": lm_diagnostics(y, universe, lm, scores),
        }
        for i in universe:
            predictions.append({
                "cohort": name, "fold": int(folds[i]), "image_id": ids[i],
                "country": countries[i], "true_label": rows[i]["label"],
                **{method: float(score[i]) for method, score in scores.items()},
            })

    report = {
        "analyses": analyses,
        "matched_pairs": [
            {"country": country, "single_id": ids[single], "double_id": ids[double], "km": distance}
            for country, single, double, distance in pairs
        ],
        "lm_inference_diagnostics": {mode: {
            "median_answer_option_mass": float(np.median([opt["option_mass"]
                                  for row in inference for opt in row[mode]])),
        } for mode in ("base", "sft")},
        "inference_configuration": config,
    }
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "out_of_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    for name, analysis in analyses.items():
        print(name)
        for method, score in analysis["scores"].items():
            print(f"  {method:22} {score['correct']:2}/{score['n']} "
                  f"{100 * score['balanced_accuracy']:5.1f}%  pairs: {score['pair_wins']}")
        print("  Projector gains/losses:", analysis["before_after_main_merger"])
        print("  LM LoRA gains/losses:", analysis["before_after_lm_lora"])


if __name__ == "__main__":
    main()
