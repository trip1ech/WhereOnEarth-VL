"""Evaluate Retrieval-Augmented Spatial KDE -> Qwen3-VL-8B-Instruct reranking & coordinate
refinement across all modes (pure retrieval, standalone 8B zero-shot, Mode A candidate selection,
Mode B VLM refined coordinates, Mode C confidence-gated hybrid, and narrow local KDE).
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from kde_vlm_utils import haversine_km

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")


def summarize(dists, pred_cc, true_cc):
    dists = np.asarray(dists, dtype=np.float64)
    n = len(dists)
    valid_mask = np.isfinite(dists)
    vd = dists[valid_mask]
    cc_ok = sum(str(p) == str(t) for p, t in zip(pred_cc, true_cc))
    return {
        "n": int(n),
        "valid": int(valid_mask.sum()),
        "hit_25": int((vd <= 25.0).sum()),
        "hit_25_rate": round(float((vd <= 25.0).sum() / n), 4),
        "hit_200": int((vd <= 200.0).sum()),
        "hit_200_rate": round(float((vd <= 200.0).sum() / n), 4),
        "hit_750": int((vd <= 750.0).sum()),
        "hit_750_rate": round(float((vd <= 750.0).sum() / n), 4),
        "country_correct": int(cc_ok),
        "country_rate": round(float(cc_ok / n), 4),
        "median_km": round(float(np.median(vd)), 2) if len(vd) else None,
        "mean_points": round(float(np.sum(5000.0 * np.exp(-vd / 1492.7)) / n), 2),
    }


def eval_split(split: str, out_subdir: str = None):
    out_dir = ROOT / "results/vlm_kde_rerank" / (out_subdir or split)
    dist_rows = [
        json.loads(l)
        for l in (out_dir / "kde_distributions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with ZipFile(out_dir / "vlm_rerank_results.zip") as z:
        run_cfg = json.loads(z.read("run_config.json"))
        rr_rows = [json.loads(l) for l in z.read("qwen8b_kde_rerank.jsonl").splitlines()]
        zs_rows = (
            [json.loads(l) for l in z.read("qwen8b_zeroshot.jsonl").splitlines()]
            if "qwen8b_zeroshot.jsonl" in z.namelist()
            else None
        )

    assert [r["id"] for r in dist_rows] == [r["id"] for r in rr_rows], "ID mismatch"

    true_cc = [r["country"] for r in dist_rows]

    # 1. Pure Retrieval Baseline: StreetCLIP 25k GeoProj + KDE + Text-Probe (Candidate #1)
    ret_d, ret_cc = [], []
    oracle6_d, oracle6_cc = [], []
    for r in dist_rows:
        cands = r["kde_distribution"]["candidates"]
        c1 = cands[0]
        ret_d.append(haversine_km(r["latitude"], r["longitude"], c1["latitude"], c1["longitude"]))
        ret_cc.append(c1["country"])
        ds = [haversine_km(r["latitude"], r["longitude"], c["latitude"], c["longitude"]) for c in cands]
        best_i = int(np.argmin(ds))
        oracle6_d.append(ds[best_i])
        oracle6_cc.append(r["country"] if r["country"] in [c["country"] for c in cands] else cands[best_i]["country"])

    results = {
        "split": split,
        "run_config": run_cfg,
        "methods": {},
    }
    results["methods"]["streetclip_25k_geoproj_kde_text"] = summarize(ret_d, ret_cc, true_cc)
    results["methods"]["oracle_top6_kde_clusters"] = summarize(oracle6_d, oracle6_cc, true_cc)

    # 2. Standalone Qwen3-VL-8B-Instruct Zero-Shot
    if zs_rows is not None:
        zs_d = [r["distance_km"] if r["distance_km"] is not None else np.nan for r in zs_rows]
        zs_cc = [r["guess_country"] for r in zs_rows]
        results["methods"]["qwen3_vl_8b_zeroshot"] = summarize(zs_d, zs_cc, true_cc)

    # Helper to resolve selected candidate cleanly (by selected_candidate index or matching country)
    def resolve_candidate(r_dist, r_vlm):
        cands = r_dist["kde_distribution"]["candidates"]
        sel = r_vlm.get("selected_candidate")
        if sel is not None and 1 <= sel <= len(cands):
            return sel, cands[sel - 1]
        if sel == 0:
            return 0, None
        # Fallback: match VLM country against candidate list
        vcc = r_vlm.get("guess_country")
        for c in cands:
            if c["country"] == vcc:
                return c["rank"], c
        return 1, cands[0]

    # Mode A: VLM Candidate Selection (KDE peak coordinate of selected cluster; VLM coords only on explicit override 0)
    # Mode A-narrow: Same selection, but using narrow local KDE (sigma=50km) inside the selected country
    # Mode B: VLM Refined Coordinates directly (fallback to selected candidate anchor if coords missing)
    # Mode C-sel: Confidence-gated selection (stay with KDE #1 when VLM confidence == "low")
    # Mode C-ref: Confidence-gated refined coordinates (stay with KDE #1 when VLM confidence == "low")
    # Mode D: Stay with KDE #1 unless VLM confidence == "high"
    # Mode E: No-override selection (never allow 0; if VLM says 0, keep KDE #1)
    # Mode F: Blend — use VLM refined coords when confidence == "high", else selected KDE anchor (low -> KDE #1)
    modes = {
        "kde_to_qwen8b_mode_a_select_kde_anchor": ([], []),
        "kde_to_qwen8b_mode_a_narrow_kde": ([], []),
        "kde_to_qwen8b_mode_b_vlm_refined_coords": ([], []),
        "kde_to_qwen8b_mode_c_gated_select": ([], []),
        "kde_to_qwen8b_mode_c_gated_refined": ([], []),
        "kde_to_qwen8b_mode_d_high_only_refined": ([], []),
        "kde_to_qwen8b_mode_e_no_override_select": ([], []),
        "kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine": ([], []),
    }

    flips = {"total_changed_from_1": 0, "override_0": 0, "improved_country": 0, "hurt_country": 0,
             "closer_than_ret_mode_b": 0, "farther_than_ret_mode_b": 0,
             "closer_than_ret_mode_a": 0, "farther_than_ret_mode_a": 0}
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    sample_examples = []

    for r_dist, r_vlm in zip(dist_rows, rr_rows):
        tlat, tlon, tcc = r_dist["latitude"], r_dist["longitude"], r_dist["country"]
        cands = r_dist["kde_distribution"]["candidates"]
        c1 = cands[0]
        sel_idx, sel_cand = resolve_candidate(r_dist, r_vlm)
        conf = r_vlm.get("confidence", "medium")
        conf_counts[conf] = conf_counts.get(conf, 0) + 1
        vcoords = r_vlm.get("guess_coordinates")
        vcc = r_vlm.get("guess_country")

        d_ret = haversine_km(tlat, tlon, c1["latitude"], c1["longitude"])

        # Mode A (select KDE anchor, allow override 0)
        if sel_cand is not None:
            a_lat, a_lon, a_cc = sel_cand["latitude"], sel_cand["longitude"], sel_cand["country"]
            an_lat, an_lon = sel_cand["narrow_latitude"], sel_cand["narrow_longitude"]
        elif vcoords is not None:
            a_lat, a_lon, a_cc = vcoords[0], vcoords[1], vcc or c1["country"]
            an_lat, an_lon = vcoords[0], vcoords[1]
        else:
            a_lat, a_lon, a_cc = c1["latitude"], c1["longitude"], c1["country"]
            an_lat, an_lon = c1["narrow_latitude"], c1["narrow_longitude"]
        d_a = haversine_km(tlat, tlon, a_lat, a_lon)
        d_an = haversine_km(tlat, tlon, an_lat, an_lon)
        modes["kde_to_qwen8b_mode_a_select_kde_anchor"][0].append(d_a)
        modes["kde_to_qwen8b_mode_a_select_kde_anchor"][1].append(a_cc)
        modes["kde_to_qwen8b_mode_a_narrow_kde"][0].append(d_an)
        modes["kde_to_qwen8b_mode_a_narrow_kde"][1].append(a_cc)

        # Mode B (VLM refined coords directly)
        if vcoords is not None:
            b_lat, b_lon, b_cc = vcoords[0], vcoords[1], vcc or a_cc
        else:
            b_lat, b_lon, b_cc = a_lat, a_lon, a_cc
        d_b = haversine_km(tlat, tlon, b_lat, b_lon)
        modes["kde_to_qwen8b_mode_b_vlm_refined_coords"][0].append(d_b)
        modes["kde_to_qwen8b_mode_b_vlm_refined_coords"][1].append(b_cc)

        # Mode C (gated: if conf == low, stay with KDE #1)
        if conf == "low":
            modes["kde_to_qwen8b_mode_c_gated_select"][0].append(d_ret)
            modes["kde_to_qwen8b_mode_c_gated_select"][1].append(c1["country"])
            modes["kde_to_qwen8b_mode_c_gated_refined"][0].append(d_ret)
            modes["kde_to_qwen8b_mode_c_gated_refined"][1].append(c1["country"])
        else:
            modes["kde_to_qwen8b_mode_c_gated_select"][0].append(d_a)
            modes["kde_to_qwen8b_mode_c_gated_select"][1].append(a_cc)
            modes["kde_to_qwen8b_mode_c_gated_refined"][0].append(d_b)
            modes["kde_to_qwen8b_mode_c_gated_refined"][1].append(b_cc)

        # Mode D (only move off KDE #1 when conf == high)
        if conf == "high":
            modes["kde_to_qwen8b_mode_d_high_only_refined"][0].append(d_b)
            modes["kde_to_qwen8b_mode_d_high_only_refined"][1].append(b_cc)
        else:
            modes["kde_to_qwen8b_mode_d_high_only_refined"][0].append(d_ret)
            modes["kde_to_qwen8b_mode_d_high_only_refined"][1].append(c1["country"])

        # Mode E (no override 0 allowed; fallback to KDE #1)
        if sel_cand is not None:
            modes["kde_to_qwen8b_mode_e_no_override_select"][0].append(d_a)
            modes["kde_to_qwen8b_mode_e_no_override_select"][1].append(a_cc)
        else:
            modes["kde_to_qwen8b_mode_e_no_override_select"][0].append(d_ret)
            modes["kde_to_qwen8b_mode_e_no_override_select"][1].append(c1["country"])

        # Mode F (high -> VLM refined; medium -> selected KDE anchor; low -> KDE #1)
        if conf == "high":
            modes["kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine"][0].append(d_b)
            modes["kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine"][1].append(b_cc)
        elif conf == "medium":
            modes["kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine"][0].append(d_a)
            modes["kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine"][1].append(a_cc)
        else:
            modes["kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine"][0].append(d_ret)
            modes["kde_to_qwen8b_mode_f_hybrid_anchor_and_high_refine"][1].append(c1["country"])

        # Track flips vs pure retrieval
        if sel_idx != 1:
            flips["total_changed_from_1"] += 1
        if sel_idx == 0:
            flips["override_0"] += 1
        if b_cc == tcc and c1["country"] != tcc:
            flips["improved_country"] += 1
        if b_cc != tcc and c1["country"] == tcc:
            flips["hurt_country"] += 1
        if d_b + 1.0 < d_ret:
            flips["closer_than_ret_mode_b"] += 1
        elif d_ret + 1.0 < d_b:
            flips["farther_than_ret_mode_b"] += 1
        if d_a + 1.0 < d_ret:
            flips["closer_than_ret_mode_a"] += 1
        elif d_ret + 1.0 < d_a:
            flips["farther_than_ret_mode_a"] += 1

        if sel_idx != 1 and len(sample_examples) < 12:
            sample_examples.append({
                "id": r_dist["id"],
                "true_cc": tcc,
                "ret_top1_cc": c1["country"],
                "ret_dist_km": round(d_ret, 1),
                "vlm_sel": sel_idx,
                "vlm_cc": b_cc,
                "vlm_mode_a_km": round(d_a, 1),
                "vlm_mode_b_km": round(d_b, 1),
                "conf": conf,
                "clues": r_vlm.get("visual_clues"),
                "comparison": r_vlm.get("candidate_comparison"),
            })

    for name, (ds, ccs) in modes.items():
        results["methods"][name] = summarize(ds, ccs, true_cc)

    # Two-stage locked system (validated on author_holdout_66):
    # If unbiased standalone 8B VLM has high confidence with valid coords, trust it;
    # otherwise use confidence-gated KDE-conditioned 8B rerank (Mode C).
    if zs_rows is not None:
        ts_d, ts_cc = [], []
        ts_ref_d, ts_ref_cc = [], []
        ts_narrow_d, ts_narrow_cc = [], []
        zs_high_used = 0
        for r_dist, r_vlm, r_zs in zip(dist_rows, rr_rows, zs_rows):
            tlat, tlon = r_dist["latitude"], r_dist["longitude"]
            cands = r_dist["kde_distribution"]["candidates"]
            c1 = cands[0]
            sel_idx, sel_cand = resolve_candidate(r_dist, r_vlm)
            conf = r_vlm.get("confidence", "medium")
            if r_zs.get("confidence") == "high" and r_zs.get("guess_coordinates") is not None:
                zs_high_used += 1
                lat, lon = r_zs["guess_coordinates"]
                cc = r_zs.get("guess_country") or c1["country"]
                ts_d.append(haversine_km(tlat, tlon, lat, lon))
                ts_cc.append(cc)
                ts_ref_d.append(haversine_km(tlat, tlon, lat, lon))
                ts_ref_cc.append(cc)
                ts_narrow_d.append(haversine_km(tlat, tlon, lat, lon))
                ts_narrow_cc.append(cc)
            elif conf != "low" and sel_cand is not None:
                ts_d.append(haversine_km(tlat, tlon, sel_cand["latitude"], sel_cand["longitude"]))
                ts_cc.append(sel_cand["country"])
                vc = r_vlm.get("guess_coordinates")
                if vc is not None:
                    ts_ref_d.append(haversine_km(tlat, tlon, vc[0], vc[1]))
                    ts_ref_cc.append(r_vlm.get("guess_country") or sel_cand["country"])
                else:
                    ts_ref_d.append(haversine_km(tlat, tlon, sel_cand["latitude"], sel_cand["longitude"]))
                    ts_ref_cc.append(sel_cand["country"])
                ts_narrow_d.append(haversine_km(tlat, tlon, sel_cand["narrow_latitude"], sel_cand["narrow_longitude"]))
                ts_narrow_cc.append(sel_cand["country"])
            else:
                ts_d.append(haversine_km(tlat, tlon, c1["latitude"], c1["longitude"]))
                ts_cc.append(c1["country"])
                ts_ref_d.append(haversine_km(tlat, tlon, c1["latitude"], c1["longitude"]))
                ts_ref_cc.append(c1["country"])
                ts_narrow_d.append(haversine_km(tlat, tlon, c1["narrow_latitude"], c1["narrow_longitude"]))
                ts_narrow_cc.append(c1["country"])
        results["methods"]["whereonearth_vl_two_stage_kde_anchor"] = summarize(ts_d, ts_cc, true_cc)
        results["methods"]["whereonearth_vl_two_stage_vlm_refined"] = summarize(ts_ref_d, ts_ref_cc, true_cc)
        results["methods"]["whereonearth_vl_two_stage_narrow_kde"] = summarize(ts_narrow_d, ts_narrow_cc, true_cc)
        flips["zs_high_overrides"] = zs_high_used

    results["diagnostics"] = {
        "parsed_count": sum(bool(r["parsed"]) for r in rr_rows),
        "confidence_distribution": conf_counts,
        "flips_vs_retrieval": flips,
        "sample_flips": sample_examples,
    }

    out_json = out_dir / "evaluation_summary.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== Evaluation Summary: {split} (n={len(dist_rows)}) ===")
    print(f"{'Method':52s} | {'CC%':>6s} | {'<=25':>5s} | {'<=200':>5s} | {'Med km':>8s} | {'GeoScore':>8s}")
    print("-" * 98)
    for mname, m in results["methods"].items():
        print(
            f"{mname:52s} | {100*m['country_rate']:5.1f}% | "
            f"{m['hit_25']:5d} | {m['hit_200']:5d} | {m['median_km']:8.1f} | {m['mean_points']:8.1f}"
        )
    print("\nDiagnostics:", json.dumps(results["diagnostics"]["flips_vs_retrieval"]),
          "conf:", conf_counts, "parsed:", results["diagnostics"]["parsed_count"])
    print(f"Saved: {out_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["author_holdout_66", "official_test_300"], required=True)
    parser.add_argument("--out-subdir", default=None)
    args = parser.parse_args()
    eval_split(args.split, args.out_subdir)


if __name__ == "__main__":
    main()
