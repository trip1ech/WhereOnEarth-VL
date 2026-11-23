"""Step 1C Analysis: Combine Step 1B (25k Diff-Creator Geo-Ranker) with Step 1C (Qwen3-VL 6k Country Priors)
using strict out-of-fold weight selection on qlora_eval_300 and zero-touch evaluation on author_holdout_66.
"""
import json
import math
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
PAIRS_ZIP = ROOT / "results/retrieval_v2/pairs_compact.zip"
STEP1BC_ZIP = ROOT / "results/retrieval_v2/step1bc_results.zip"
QLORA_6K_ZIP = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
AH_RESULTS_ZIP = ROOT / "results/author_holdout/author-holdout-results.zip"
OUT_DIR = ROOT / "results/retrieval_v2"


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def summarize(dist_km, pred_cc, true_cc):
    dist_km = np.asarray(dist_km, dtype=np.float64)
    return {
        "n": int(len(dist_km)),
        "hit_25": int((dist_km <= 25.0).sum()),
        "hit_25_rate": float((dist_km <= 25.0).mean()),
        "hit_200": int((dist_km <= 200.0).sum()),
        "hit_200_rate": float((dist_km <= 200.0).mean()),
        "hit_750": int((dist_km <= 750.0).sum()),
        "hit_750_rate": float((dist_km <= 750.0).mean()),
        "country_correct": int((pred_cc == true_cc).sum()),
        "country_rate": float((pred_cc == true_cc).mean()),
        "median_km": float(np.median(dist_km)),
        "mean_points": float(np.mean(5000.0 * np.exp(-dist_km / 1492.7))),
    }


def main():
    with zipfile.ZipFile(PAIRS_ZIP) as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        val = [json.loads(l) for l in z.read("validation_images.jsonl").splitlines()]
    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    val_id_to_idx = {r["id"]: i for i, r in enumerate(val)}

    with zipfile.ZipFile(STEP1BC_ZIP) as z:
        s1b = json.loads(z.read("step1b_ranker_results.json"))
        vlm_priors = json.loads(z.read("vlm_country_priors.json"))["predictions"]
        with z.open("val_projected_top100.npz") as f:
            npz = np.load(f)
            top_idx = npz["top_idx"]
            geo_sim = npz["geo_sim"]
            kde_score = npz["kde_score"]
            ranker_logits = npz["ranker_logits"]

    with zipfile.ZipFile(QLORA_6K_ZIP) as z:
        q300_vlm = {r["id"]: r for r in (json.loads(l) for l in z.read("sft.jsonl").splitlines())}
    with zipfile.ZipFile(AH_RESULTS_ZIP) as z:
        ah66_vlm = {r["id"]: r for r in (json.loads(l) for l in z.read("sft.jsonl").splitlines())}

    def build_arrays(pred_map):
        ids = list(pred_map.keys())
        vidx = np.array([val_id_to_idx[i] for i in ids], dtype=np.int32)
        t_lat = np.array([val[i]["latitude"] for i in vidx])
        t_lon = np.array([val[i]["longitude"] for i in vidx])
        t_cc = np.array([val[i]["country"] for i in vidx])
        vlm_lat = np.array([pred_map[i]["guess_coordinates"][0] for i in ids], dtype=np.float64)
        vlm_lon = np.array([pred_map[i]["guess_coordinates"][1] for i in ids], dtype=np.float64)
        vlm_cc = np.array([pred_map[i]["guess_country"] for i in ids])
        vlm_dist = np.array([pred_map[i]["distance_km"] for i in ids], dtype=np.float64)

        cand_idx = top_idx[vidx]  # (N, 100)
        cand_lat = g_lat[cand_idx]
        cand_lon = g_lon[cand_idx]
        cand_cc = g_cc[cand_idx]
        cand_d = hav(t_lat[:, None], t_lon[:, None], cand_lat, cand_lon)
        cand_rk = ranker_logits[vidx]
        cand_gs = geo_sim[vidx]
        # Log-softmax ranker score per query so scale is clean across queries
        cand_rk_lp = cand_rk - np.log(np.exp(cand_rk - cand_rk.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) - cand_rk.max(axis=1, keepdims=True)
        # Wait: exact log_softmax = x - max - log(sum(exp(x-max)))
        m = cand_rk.max(axis=1, keepdims=True)
        cand_rk_lp = (cand_rk - m) - np.log(np.exp(cand_rk - m).sum(axis=1, keepdims=True))

        # VLM country log-prob for each candidate's country
        cand_vlm_lp = np.full(cand_idx.shape, -15.0, dtype=np.float64)
        vlm_top1_cc = []
        for row_i, rid in enumerate(ids):
            lp_dict = vlm_priors[rid]["log_probs"]
            vlm_top1_cc.append(vlm_priors[rid]["top1_country"])
            for k in range(cand_idx.shape[1]):
                cand_vlm_lp[row_i, k] = lp_dict.get(str(cand_cc[row_i, k]), -15.0)
        # Distance from each candidate to VLM's predicted lat/lon (for spatial+retrieval fusion!)
        cand_to_vlm_km = hav(vlm_lat[:, None], vlm_lon[:, None], cand_lat, cand_lon)
        return dict(
            ids=ids, t_lat=t_lat, t_lon=t_lon, t_cc=t_cc,
            vlm_lat=vlm_lat, vlm_lon=vlm_lon, vlm_cc=vlm_cc, vlm_dist=vlm_dist,
            vlm_top1_cc=np.array(vlm_top1_cc),
            cand_idx=cand_idx, cand_lat=cand_lat, cand_lon=cand_lon, cand_cc=cand_cc,
            cand_d=cand_d, cand_rk=cand_rk, cand_rk_lp=cand_rk_lp, cand_gs=cand_gs,
            cand_vlm_lp=cand_vlm_lp, cand_to_vlm_km=cand_to_vlm_km,
        )

    d300 = build_arrays(q300_vlm)
    d66 = build_arrays(ah66_vlm)

    # 1. Tune fusion weights strictly via 5-fold CV on qlora_eval_300 (zero peek at author_holdout_66!)
    # Mode A: Country-prior only fusion: score = cand_rk_lp + lam_cc * cand_vlm_lp
    # Mode B: Country-prior + VLM spatial anchor: score = cand_rk_lp + lam_cc * cand_vlm_lp - lam_sp * (cand_to_vlm_km / 1000)
    # Mode C: Blend predicted coordinate with VLM coordinate when retrieval confidence is low
    lam_cc_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    lam_sp_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

    def eval_fusion(arr, lam_cc, lam_sp):
        score = arr["cand_rk_lp"] + lam_cc * arr["cand_vlm_lp"] - lam_sp * (arr["cand_to_vlm_km"] / 1000.0)
        pick = score.argmax(axis=1)
        rows = np.arange(len(pick))
        return arr["cand_d"][rows, pick], arr["cand_cc"][rows, pick], pick

    # 5-fold OOF on d300
    n300 = len(d300["ids"])
    folds = np.arange(n300) % 5
    oof_a_d, oof_a_cc = np.zeros(n300), np.empty(n300, dtype=object)
    oof_b_d, oof_b_cc = np.zeros(n300), np.empty(n300, dtype=object)
    fold_best_a, fold_best_b = [], []
    for f in range(5):
        tr = folds != f
        te = folds == f
        # Mode A (country prior only)
        best_a = max(lam_cc_grid, key=lambda lc: summarize(*eval_fusion({k: v[tr] if isinstance(v, np.ndarray) else v for k, v in d300.items()}, lc, 0.0)[:2], d300["t_cc"][tr])["mean_points"])
        fold_best_a.append(best_a)
        da, cca, _ = eval_fusion({k: v[te] if isinstance(v, np.ndarray) else v for k, v in d300.items()}, best_a, 0.0)
        oof_a_d[te], oof_a_cc[te] = da, cca

        # Mode B (country prior + VLM spatial anchor)
        best_b = max(
            ((lc, ls) for lc in lam_cc_grid for ls in lam_sp_grid),
            key=lambda pair: summarize(*eval_fusion({k: v[tr] if isinstance(v, np.ndarray) else v for k, v in d300.items()}, pair[0], pair[1])[:2], d300["t_cc"][tr])["mean_points"],
        )
        fold_best_b.append(best_b)
        db, ccb, _ = eval_fusion({k: v[te] if isinstance(v, np.ndarray) else v for k, v in d300.items()}, best_b[0], best_b[1])
        oof_b_d[te], oof_b_cc[te] = db, ccb

    # Fit global best weights on all 300 -> apply zero-touch to author_holdout_66
    global_a = max(lam_cc_grid, key=lambda lc: summarize(*eval_fusion(d300, lc, 0.0)[:2], d300["t_cc"])["mean_points"])
    global_b = max(
        ((lc, ls) for lc in lam_cc_grid for ls in lam_sp_grid),
        key=lambda pair: summarize(*eval_fusion(d300, pair[0], pair[1])[:2], d300["t_cc"])["mean_points"],
    )

    ah_a_d, ah_a_cc, _ = eval_fusion(d66, global_a, 0.0)
    ah_b_d, ah_b_cc, _ = eval_fusion(d66, global_b[0], global_b[1])

    # Pure retrieval ranker on 300 and 66
    ret300_d, ret300_cc, _ = eval_fusion(d300, 0.0, 0.0)
    ret66_d, ret66_cc, _ = eval_fusion(d66, 0.0, 0.0)

    summary = {
        "tuned_on_300": {"mode_a_lam_cc": global_a, "mode_b_lam_cc_sp": list(global_b),
                         "fold_a": fold_best_a, "fold_b": [list(x) for x in fold_best_b]},
        "qlora_eval_300": {
            "vlm_6k_only": summarize(d300["vlm_dist"], d300["vlm_cc"], d300["t_cc"]),
            "retrieval_v2_only": summarize(ret300_d, ret300_cc, d300["t_cc"]),
            "hybrid_vlm_country_oof": summarize(oof_a_d, oof_a_cc, d300["t_cc"]),
            "hybrid_vlm_country_spatial_oof": summarize(oof_b_d, oof_b_cc, d300["t_cc"]),
            "vlm_logprob_top1_country_acc": float((d300["vlm_top1_cc"] == d300["t_cc"]).mean()),
        },
        "author_holdout_66": {
            "vlm_6k_only": summarize(d66["vlm_dist"], d66["vlm_cc"], d66["t_cc"]),
            "retrieval_v2_only": summarize(ret66_d, ret66_cc, d66["t_cc"]),
            "hybrid_vlm_country_zero_touch": summarize(ah_a_d, ah_a_cc, d66["t_cc"]),
            "hybrid_vlm_country_spatial_zero_touch": summarize(ah_b_d, ah_b_cc, d66["t_cc"]),
            "vlm_logprob_top1_country_acc": float((d66["vlm_top1_cc"] == d66["t_cc"]).mean()),
        },
    }

    for split in ("qlora_eval_300", "author_holdout_66"):
        print(f"\n=== {split} ===")
        for k, m in summary[split].items():
            if isinstance(m, dict):
                print(f"  {k:38s}: <=25={m['hit_25']:2d} ({100*m['hit_25_rate']:5.2f}%), <=200={m['hit_200']:2d} ({100*m['hit_200_rate']:5.2f}%), "
                      f"cc={m['country_correct']:3d} ({100*m['country_rate']:5.1f}%), med={m['median_km']:6.1f}km, pts={m['mean_points']:6.1f}")
            else:
                print(f"  {k}: {100*m:.1f}%")

    (OUT_DIR / "step1c_hybrid_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved {OUT_DIR / 'step1c_hybrid_results.json'}")


if __name__ == "__main__":
    main()
