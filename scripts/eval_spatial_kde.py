"""Audit creator-shortcut leakage and run leak-free multi-scale spatial KDE on 25k DINOv2 retrieval."""
import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
PAIRS_ZIP = ROOT / "results/ranker-2018-pairs.zip"
QLORA_6K_ZIP = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
AUTHOR_HOLDOUT_EVAL = ROOT / "data/author_holdout/eval.jsonl"
OUT_DIR = ROOT / "results/retrieval_v2"


def haversine_matrix(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def geoguessr_points(dist_km):
    return float(np.mean(5000.0 * np.exp(-np.asarray(dist_km, dtype=np.float64) / 1492.7)))


def summarize_preds(dist_km, pred_cc, true_cc):
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
        "mean_points": geoguessr_points(dist_km),
    }


def load_cached_retrieval():
    with zipfile.ZipFile(PAIRS_ZIP) as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        trn = [json.loads(l) for l in z.read("train_images.jsonl").splitlines()]
        val = [json.loads(l) for l in z.read("validation_images.jsonl").splitlines()]
        with z.open("gallery_features.npz") as f:
            gf = np.load(f)["features"].astype(np.float32)
        with z.open("train_features.npz") as f:
            tf = np.load(f)["features"].astype(np.float32)
        with z.open("validation_features.npz") as f:
            vf = np.load(f)["features"].astype(np.float32)
    gf /= np.linalg.norm(gf, axis=-1, keepdims=True)
    tf /= np.linalg.norm(tf, axis=-1, keepdims=True)
    vf /= np.linalg.norm(vf, axis=-1, keepdims=True)
    return gal, trn, val, gf, tf, vf


def retrieve_topk(qf, q_attr, gf, g_attr, k=100, mask_same_creator=True, chunk=500):
    n = len(qf)
    top_idx = np.empty((n, k), dtype=np.int32)
    top_sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        s = qf[start:end] @ gf.T
        if mask_same_creator:
            qa = q_attr[start:end][:, None]
            same = (qa == g_attr[None, :]) & (qa != "")
            s = np.where(same, -2.0, s)
        part = np.argpartition(-s, k, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        order = np.argsort(-s[rows, part], axis=1)
        idx = part[rows, order]
        top_idx[start:end] = idx
        top_sim[start:end] = s[rows, idx]
    return top_idx, top_sim


def compute_gallery_local_density(g_lat, g_lon, sigma_km=200.0, chunk=1000):
    """Smooth local density of gallery images around each gallery point (for debiasing dense US/EU clusters)."""
    n = len(g_lat)
    density = np.zeros(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        d = haversine_matrix(g_lat[start:end], g_lon[start:end], g_lat, g_lon)
        density[start:end] = np.exp(-d / sigma_km).sum(axis=1)
    return density


def pairwise_top_dist(top_idx, g_lat, g_lon):
    lats = np.radians(g_lat[top_idx])
    lons = np.radians(g_lon[top_idx])
    dlat = lats[:, :, None] - lats[:, None, :]
    dlon = lons[:, :, None] - lons[:, None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lats[:, :, None]) * np.cos(lats[:, None, :]) * np.sin(dlon / 2) ** 2
    return (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)


def kde_predict(top_idx, top_sim, pw_d, g_lat, g_lon, g_cc, g_density, *, k=100, tau=0.05, sigma=200.0,
                density_alpha=0.0, refine_km=0.0):
    """Select best candidate via spatial KDE (optionally debiased by gallery density and locally refined)."""
    s = top_sim[:, :k]
    d = pw_d[:, :k, :k]
    idx = top_idx[:, :k]
    w = np.exp((s - s[:, :1]) / tau)
    if density_alpha > 0.0:
        w = w / (g_density[idx] ** density_alpha)
    K = np.exp(-d / sigma)
    scores = np.einsum("nj,nij->ni", w, K)
    pick = scores.argmax(axis=1)
    rows = np.arange(len(top_idx))
    chosen_g = idx[rows, pick]
    pred_lat = g_lat[chosen_g].copy()
    pred_lon = g_lon[chosen_g].copy()
    pred_cc = g_cc[chosen_g]

    if refine_km > 0.0:
        # Weighted circular/tangent average of neighbors within refine_km of winning candidate
        for i in range(len(top_idx)):
            p = pick[i]
            near = d[i, p] <= refine_km
            nw = w[i, near] * np.exp(-d[i, p, near] / max(25.0, sigma / 2.0))
            nw /= nw.sum()
            # 3D unit-sphere average (safe across dateline & poles)
            lats = np.radians(g_lat[idx[i, near]])
            lons = np.radians(g_lon[idx[i, near]])
            x = (nw * np.cos(lats) * np.cos(lons)).sum()
            y = (nw * np.cos(lats) * np.sin(lons)).sum()
            z = (nw * np.sin(lats)).sum()
            pred_lat[i] = math.degrees(math.atan2(z, math.hypot(x, y)))
            pred_lon[i] = math.degrees(math.atan2(y, x))
    return chosen_g, pred_lat, pred_lon, pred_cc, scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gal, trn, val, gf, tf, vf = load_cached_retrieval()

    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    g_attr = np.array([r["attribution"] for r in gal])

    t_lat = np.array([r["latitude"] for r in trn], dtype=np.float64)
    t_lon = np.array([r["longitude"] for r in trn], dtype=np.float64)
    t_cc = np.array([r["country"] for r in trn])
    t_attr = np.array([r["attribution"] for r in trn])

    v_lat = np.array([r["latitude"] for r in val], dtype=np.float64)
    v_lon = np.array([r["longitude"] for r in val], dtype=np.float64)
    v_cc = np.array([r["country"] for r in val])
    v_attr = np.array([r["attribution"] for r in val])

    # Load subset indices inside val_2000
    val_id_to_idx = {r["id"]: i for i, r in enumerate(val)}
    with zipfile.ZipFile(QLORA_6K_ZIP) as z:
        q300_ids = [json.loads(l)["id"] for l in z.read("eval_labels.jsonl").splitlines()]
    q300_idx = np.array([val_id_to_idx[i] for i in q300_ids], dtype=np.int32)
    ah_ids = [json.loads(l)["id"] for l in AUTHOR_HOLDOUT_EVAL.read_text().splitlines()]
    ah_idx = np.array([val_id_to_idx[i] for i in ah_ids], dtype=np.int32)

    # Creator-disjoint tuning subset from train_12000 (2,000 queries whose creators do NOT overlap val_2000?
    # Note: many creators appear in both train and val; pick 2,000 train queries strictly by deterministic seed for KDE tuning)
    rng = np.random.default_rng(7818)
    tune_idx = rng.choice(len(trn), size=2000, replace=False)

    print("Computing gallery local density (sigma=200km)...", flush=True)
    g_density = compute_gallery_local_density(g_lat, g_lon, sigma_km=200.0)

    print("Retrieving diff-creator Top-100 on tune (2000) and val (2000)...", flush=True)
    tune_top, tune_sim = retrieve_topk(tf[tune_idx], t_attr[tune_idx], gf, g_attr, k=100, mask_same_creator=True)
    val_top_diff, val_sim_diff = retrieve_topk(vf, v_attr, gf, g_attr, k=100, mask_same_creator=True)
    val_top_raw, val_sim_raw = retrieve_topk(vf, v_attr, gf, g_attr, k=100, mask_same_creator=False)

    tune_pw = pairwise_top_dist(tune_top, g_lat, g_lon)
    val_pw_diff = pairwise_top_dist(val_top_diff, g_lat, g_lon)

    # Sweep KDE parameters strictly on tune_2000 (diff-creator)
    best_cfg = None
    best_obj = None
    sweep = []
    for k in (50, 100):
        for tau in (0.03, 0.05, 0.07, 0.10):
            for sigma in (100.0, 250.0, 500.0, 800.0):
                for alpha in (0.0, 0.25, 0.5):
                    for refine in (0.0, 250.0):
                        _, plat, plon, pcc, _ = kde_predict(
                            tune_top, tune_sim, tune_pw, g_lat, g_lon, g_cc, g_density,
                            k=k, tau=tau, sigma=sigma, density_alpha=alpha, refine_km=refine,
                        )
                        dkm = haversine_vec(t_lat[tune_idx], t_lon[tune_idx], plat, plon)
                        m = summarize_preds(dkm, pcc, t_cc[tune_idx])
                        rec = dict(k=k, tau=tau, sigma=sigma, density_alpha=alpha, refine_km=refine, **m)
                        sweep.append(rec)
                        # Optimize GeoGuessr mean points (balanced metric across 25km/200km/median)
                        if best_obj is None or m["mean_points"] > best_obj:
                            best_obj = m["mean_points"]
                            best_cfg = rec

    print("\nBest KDE config on diff-creator TUNE (by GeoGuessr mean_points):")
    print(json.dumps(best_cfg, indent=2))

    # Also check raw top-1 vs diff-creator top-1 vs best KDE on val_2000, qlora_300, author_holdout_66
    results = {"best_tune_cfg": best_cfg, "splits": {}}

    def eval_subset(name, idxs):
        # Raw top-1 (leaky)
        raw_g = val_top_raw[idxs, 0]
        raw_d = haversine_vec(v_lat[idxs], v_lon[idxs], g_lat[raw_g], g_lon[raw_g])
        raw_m = summarize_preds(raw_d, g_cc[raw_g], v_cc[idxs])
        # Diff-creator top-1 (honest baseline)
        diff_g = val_top_diff[idxs, 0]
        diff_d = haversine_vec(v_lat[idxs], v_lon[idxs], g_lat[diff_g], g_lon[diff_g])
        diff_m = summarize_preds(diff_d, g_cc[diff_g], v_cc[idxs])
        # Diff-creator Top-100 recall
        all_d = haversine_vec(
            v_lat[idxs][:, None], v_lon[idxs][:, None],
            g_lat[val_top_diff[idxs]], g_lon[val_top_diff[idxs]],
        )
        recall = {
            "top100_25km": float((all_d.min(axis=1) <= 25.0).mean()),
            "top100_200km": float((all_d.min(axis=1) <= 200.0).mean()),
            "top100_country": float(np.any(g_cc[val_top_diff[idxs]] == v_cc[idxs][:, None], axis=1).mean()),
        }
        # Best KDE (no debiasing vs best with debiasing)
        _, plat_plain, plon_plain, pcc_plain, _ = kde_predict(
            val_top_diff[idxs], val_sim_diff[idxs], val_pw_diff[idxs], g_lat, g_lon, g_cc, g_density,
            k=best_cfg["k"], tau=best_cfg["tau"], sigma=best_cfg["sigma"], density_alpha=0.0, refine_km=0.0,
        )
        kde_plain_m = summarize_preds(haversine_vec(v_lat[idxs], v_lon[idxs], plat_plain, plon_plain), pcc_plain, v_cc[idxs])

        _, plat_best, plon_best, pcc_best, _ = kde_predict(
            val_top_diff[idxs], val_sim_diff[idxs], val_pw_diff[idxs], g_lat, g_lon, g_cc, g_density,
            k=best_cfg["k"], tau=best_cfg["tau"], sigma=best_cfg["sigma"],
            density_alpha=best_cfg["density_alpha"], refine_km=best_cfg["refine_km"],
        )
        kde_best_m = summarize_preds(haversine_vec(v_lat[idxs], v_lon[idxs], plat_best, plon_best), pcc_best, v_cc[idxs])

        results["splits"][name] = {
            "raw_top1_leaky": raw_m,
            "diff_top1_honest": diff_m,
            "diff_top100_recall": recall,
            "diff_kde_plain": kde_plain_m,
            "diff_kde_best": kde_best_m,
        }
        print(f"\n=== {name} (n={len(idxs)}) ===")
        for label, m in [
            ("Raw Top-1 (LEAKY)   ", raw_m),
            ("Diff-Creator Top-1  ", diff_m),
            ("Diff-Creator KDE    ", kde_plain_m),
            ("Diff-Creator KDE+Deb", kde_best_m),
        ]:
            print(f"  {label}: <=25km={m['hit_25']:3d} ({100*m['hit_25_rate']:5.2f}%), <=200km={m['hit_200']:3d} ({100*m['hit_200_rate']:5.2f}%), country={m['country_correct']:3d} ({100*m['country_rate']:5.1f}%), median={m['median_km']:6.1f} km, pts={m['mean_points']:6.1f}")
        print(f"  Diff-Creator Top-100 recall: <=25km={100*recall['top100_25km']:.2f}%, <=200km={100*recall['top100_200km']:.2f}%, country={100*recall['top100_country']:.2f}%")

    eval_subset("val_2000", np.arange(len(val)))
    eval_subset("qlora_eval_300", q300_idx)
    eval_subset("author_holdout_66", ah_idx)

    (OUT_DIR / "step1a_kde_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved {OUT_DIR / 'step1a_kde_results.json'}")


if __name__ == "__main__":
    main()
