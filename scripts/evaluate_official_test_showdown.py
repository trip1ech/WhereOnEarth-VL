"""Phase D: Complete apples-to-apples evaluation of 25k StreetCLIP Geo-Ranker + Official OSV-5M Test 300 Showdown
comparing osv5m/baseline, Qwen3-VL (zero-shot & 6k QLoRA), dinov2-small 25k, StreetCLIP 25k (raw / KDE+Text /
GeoProj+Listwise), and StreetCLIP 25k x Qwen3-VL 6k Hybrid (weights tuned via 5-fold CV on qlora_eval_300,
evaluated ZERO-TOUCH on author_holdout_66 and official_test_300).
"""
import json
import math
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
GEOJSON = ROOT / "data/raw/ne_110m_admin_0_countries.geojson"
SC25K_DIR = ROOT / "results/streetclip_25k"
OTEST_DIR = ROOT / "results/official_test_showdown"
RETV2_DIR = ROOT / "results/retrieval_v2"
QLORA_6K_ZIP = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
AH_RESULTS_ZIP = ROOT / "results/author_holdout/author-holdout-results.zip"
TEST_ZIP = ROOT / "data/official_test_300/official-test-300.zip"


def hav_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def summarize(dist_km, pred_cc=None, true_cc=None, valid_mask=None):
    dist_km = np.asarray(dist_km, dtype=np.float64)
    n = len(dist_km)
    if valid_mask is None:
        valid_mask = np.isfinite(dist_km)
    valid_d = dist_km[valid_mask]
    res = {
        "n": int(n),
        "valid": int(valid_mask.sum()),
        "hit_25": int((valid_d <= 25.0).sum()),
        "hit_25_rate": float((valid_d <= 25.0).sum() / n),
        "hit_200": int((valid_d <= 200.0).sum()),
        "hit_200_rate": float((valid_d <= 200.0).sum() / n),
        "hit_750": int((valid_d <= 750.0).sum()),
        "hit_750_rate": float((valid_d <= 750.0).sum() / n),
        "median_km": float(np.median(valid_d)) if len(valid_d) else None,
        "mean_points": float(np.sum(5000.0 * np.exp(-valid_d / 1492.7)) / n),
    }
    if pred_cc is not None and true_cc is not None:
        ok = sum(str(p) == str(t) for p, t in zip(pred_cc, true_cc))
        res["country_correct"] = int(ok)
        res["country_rate"] = float(ok / n)
    return res


def load_country_polygons():
    """Pure-Python point-in-polygon reverse geocoder using local Natural Earth 110m GeoJSON."""
    data = json.loads(GEOJSON.read_text(encoding="utf-8"))
    polys = []
    for feat in data["features"]:
        props = feat["properties"]
        iso = props.get("ISO_A2")
        if not iso or iso == "-99":
            iso = props.get("ISO_A2_EH") or props.get("POSTAL") or ""
        iso = str(iso).strip().upper()
        if len(iso) != 2:
            continue
        geom = feat["geometry"]
        rings_list = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in rings_list:
            outer = poly[0]
            xs = [p[0] for p in outer]
            ys = [p[1] for p in outer]
            polys.append((iso, min(xs), max(xs), min(ys), max(ys), outer, poly[1:]))
    return polys


def point_in_ring(x, y, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def reverse_geocode_nearest(lat, lon, polys, g_lat, g_lon, g_cc):
    for iso, minx, maxx, miny, maxy, outer, holes in polys:
        if minx <= lon <= maxx and miny <= lat <= maxy:
            if point_in_ring(lon, lat, outer) and not any(point_in_ring(lon, lat, h) for h in holes):
                return iso
    # Coastal / island fallback (110m resolution misses small islands): nearest 25k gallery country
    d = hav_vec(np.full(len(g_lat), lat), np.full(len(g_lon), lon), g_lat, g_lon)
    return str(g_cc[int(np.argmin(d))])


def log_softmax_rows(x):
    m = x.max(axis=1, keepdims=True)
    return (x - m) - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def kde_log_scores(top_idx, top_sim, g_lat, g_lon, tau=0.03, sigma=500.0):
    la = np.radians(g_lat[top_idx])
    lo = np.radians(g_lon[top_idx])
    dlat = la[:, :, None] - la[:, None, :]
    dlon = lo[:, :, None] - lo[:, None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin(dlon / 2) ** 2
    pw_d = (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)
    w = np.exp((top_sim - top_sim[:, :1]) / tau)
    K = np.exp(-pw_d / sigma)
    return np.log(np.maximum(np.einsum("nj,nij->ni", w, K), 1e-12))


def main():
    # 1. Load 25k gallery + val + official_test_300 metadata
    with zipfile.ZipFile(SC25K_DIR / "manifests_39k.zip") as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        val = [json.loads(l) for l in z.read("validation_images.jsonl").splitlines()]
    with zipfile.ZipFile(TEST_ZIP) as z:
        test_rows = [json.loads(l) for l in z.read("test.jsonl").splitlines()]

    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    val_id_to_idx = {r["id"]: i for i, r in enumerate(val)}

    # 2. Load Phase B (25k StreetCLIP) outputs
    with zipfile.ZipFile(SC25K_DIR / "streetclip_25k_results.zip") as z:
        sc25_json = json.loads(z.read("streetclip_25k_results.json"))
        with z.open("streetclip_25k_top100.npz") as f:
            sc_npz = dict(np.load(f))
    cc_vocab = [str(c) for c in sc_npz["country_vocab"]]
    cc_to_i = {c: i for i, c in enumerate(cc_vocab)}

    # 3. Load Phase C (Official Test 300 models) outputs
    with zipfile.ZipFile(OTEST_DIR / "phase_c_results.zip") as z:
        osv_preds = [json.loads(l) for l in z.read("osv5m_baseline.jsonl").splitlines()]
        qzs_preds = [json.loads(l) for l in z.read("qwen_zero_shot.jsonl").splitlines()]
        qsft_preds = [json.loads(l) for l in z.read("qwen_sft_6000.jsonl").splitlines()]
        vlm_priors_test = json.loads(z.read("vlm_country_priors_test300.json"))["predictions"]
        with z.open("dinov2_test300.npz") as f:
            dino_test_f = np.load(f)["features"].astype(np.float32)

    # Also load Step 1C VLM priors for qlora_eval_300 & author_holdout_66
    with zipfile.ZipFile(RETV2_DIR / "step1bc_results.zip") as z:
        vlm_priors_val = json.loads(z.read("vlm_country_priors.json"))["predictions"]
    with zipfile.ZipFile(QLORA_6K_ZIP) as z:
        q300_vlm = [json.loads(l) for l in z.read("sft.jsonl").splitlines()]
    with zipfile.ZipFile(AH_RESULTS_ZIP) as z:
        ah66_vlm = [json.loads(l) for l in z.read("sft.jsonl").splitlines()]

    # Also evaluate raw dinov2-small 25k on official_test_300 for complete baseline table
    with zipfile.ZipFile(RETV2_DIR / "pairs_compact.zip") as z:
        with z.open("gallery_features.npz") as f:
            dino_gf = np.load(f)["features"].astype(np.float32)
    dino_gf /= np.linalg.norm(dino_gf, axis=-1, keepdims=True)
    dino_test_f /= np.linalg.norm(dino_test_f, axis=-1, keepdims=True)
    dino_sim = dino_test_f @ dino_gf.T
    dino_top1 = dino_sim.argmax(axis=1)
    ts_lat = np.array([r["latitude"] for r in test_rows], dtype=np.float64)
    ts_lon = np.array([r["longitude"] for r in test_rows], dtype=np.float64)
    ts_cc = np.array([r["country"] for r in test_rows])
    dino_test_d = hav_vec(ts_lat, ts_lon, g_lat[dino_top1], g_lon[dino_top1])
    dino_test_m = summarize(dino_test_d, g_cc[dino_top1], ts_cc)

    # Reverse-geocode osv5m/baseline predicted coordinates
    polys = load_country_polygons()
    osv_cc = np.array([reverse_geocode_nearest(p["guess_latitude"], p["guess_longitude"], polys, g_lat, g_lon, g_cc) for p in osv_preds])
    osv_m = summarize([p["distance_km"] for p in osv_preds], osv_cc, ts_cc)

    qzs_d = np.array([p["distance_km"] if p["distance_km"] is not None else np.nan for p in qzs_preds])
    qzs_m = summarize(qzs_d, [p["guess_country"] for p in qzs_preds], ts_cc, valid_mask=np.isfinite(qzs_d))

    qsft_d = np.array([p["distance_km"] if p["distance_km"] is not None else np.nan for p in qsft_preds])
    qsft_m = summarize(qsft_d, [p["guess_country"] for p in qsft_preds], ts_cc, valid_mask=np.isfinite(qsft_d))

    # Build fusion helper for 25k StreetCLIP (both raw KDE+Text and Stage 2 Listwise Ranker) x Qwen3-VL country prior
    def build_fusion_split(ids, true_lat, true_lon, true_cc, raw_top, raw_sim, proj_top, proj_sim, rk_logits, txt_lp, priors_map):
        n = len(ids)
        # Raw StreetCLIP 25k Top-100 + KDE + Text-probe
        raw_d = hav_vec(true_lat[:, None], true_lon[:, None], g_lat[raw_top], g_lon[raw_top])
        raw_cc = g_cc[raw_top]
        raw_kde = kde_log_scores(raw_top, raw_sim, g_lat, g_lon, tau=0.03, sigma=500.0)
        raw_tlp = np.full(raw_top.shape, -15.0, dtype=np.float64)
        raw_vlp = np.full(raw_top.shape, -15.0, dtype=np.float64)
        for i, rid in enumerate(ids):
            vp = priors_map[rid]["log_probs"]
            for j in range(raw_top.shape[1]):
                cc = str(raw_cc[i, j])
                if cc in cc_to_i:
                    raw_tlp[i, j] = txt_lp[i, cc_to_i[cc]]
                raw_vlp[i, j] = vp.get(cc, -15.0)

        # Stage 1 GeoProj Top-100 + KDE + Text-probe + VLM
        proj_d = hav_vec(true_lat[:, None], true_lon[:, None], g_lat[proj_top], g_lon[proj_top])
        proj_cc = g_cc[proj_top]
        proj_kde = kde_log_scores(proj_top, proj_sim, g_lat, g_lon, tau=0.03, sigma=500.0)
        proj_tlp = np.full(proj_top.shape, -15.0, dtype=np.float64)
        rk_lp = log_softmax_rows(rk_logits.astype(np.float64))
        proj_vlp = np.full(proj_top.shape, -15.0, dtype=np.float64)
        for i, rid in enumerate(ids):
            vp = priors_map[rid]["log_probs"]
            for j in range(proj_top.shape[1]):
                cc = str(proj_cc[i, j])
                if cc in cc_to_i:
                    proj_tlp[i, j] = txt_lp[i, cc_to_i[cc]]
                proj_vlp[i, j] = vp.get(cc, -15.0)

        return dict(
            ids=ids, true_cc=true_cc,
            raw_d=raw_d, raw_cc=raw_cc, raw_kde=raw_kde, raw_tlp=raw_tlp, raw_vlp=raw_vlp,
            proj_d=proj_d, proj_cc=proj_cc, proj_kde=proj_kde, proj_tlp=proj_tlp,
            rk_lp=rk_lp, proj_vlp=proj_vlp,
        )

    q300_ids = [r["id"] for r in q300_vlm]
    q300_vidx = np.array([val_id_to_idx[i] for i in q300_ids], dtype=np.int32)
    ah66_ids = [r["id"] for r in ah66_vlm]
    ah66_vidx = np.array([val_id_to_idx[i] for i in ah66_ids], dtype=np.int32)
    test_ids = [r["id"] for r in test_rows]

    f300 = build_fusion_split(
        q300_ids, np.array([val[i]["latitude"] for i in q300_vidx]),
        np.array([val[i]["longitude"] for i in q300_vidx]), np.array([val[i]["country"] for i in q300_vidx]),
        sc_npz["val_raw_top"][q300_vidx], sc_npz["val_raw_sim"][q300_vidx],
        sc_npz["val_proj_top"][q300_vidx], sc_npz["val_proj_sim"][q300_vidx], sc_npz["val_rk_logits"][q300_vidx],
        sc_npz["val_txt_lp"][q300_vidx], vlm_priors_val,
    )
    f66 = build_fusion_split(
        ah66_ids, np.array([val[i]["latitude"] for i in ah66_vidx]),
        np.array([val[i]["longitude"] for i in ah66_vidx]), np.array([val[i]["country"] for i in ah66_vidx]),
        sc_npz["val_raw_top"][ah66_vidx], sc_npz["val_raw_sim"][ah66_vidx],
        sc_npz["val_proj_top"][ah66_vidx], sc_npz["val_proj_sim"][ah66_vidx], sc_npz["val_rk_logits"][ah66_vidx],
        sc_npz["val_txt_lp"][ah66_vidx], vlm_priors_val,
    )
    ftest = build_fusion_split(
        test_ids, ts_lat, ts_lon, ts_cc,
        sc_npz["test_raw_top"], sc_npz["test_raw_sim"],
        sc_npz["test_proj_top"], sc_npz["test_proj_sim"], sc_npz["test_rk_logits"],
        sc_npz["test_txt_lp"], vlm_priors_test,
    )

    # Tune fusion weights STRICTLY via 5-fold CV on qlora_eval_300 -> ZERO-TOUCH on author_holdout_66 & official_test_300
    folds = np.arange(len(q300_ids)) % 5

    def eval_raw_hyb(sp, wt, wv, mask=None):
        kde = sp["raw_kde"] if mask is None else sp["raw_kde"][mask]
        tlp = sp["raw_tlp"] if mask is None else sp["raw_tlp"][mask]
        vlp = sp["raw_vlp"] if mask is None else sp["raw_vlp"][mask]
        d = sp["raw_d"] if mask is None else sp["raw_d"][mask]
        cc = sp["raw_cc"] if mask is None else sp["raw_cc"][mask]
        p = (kde + wt * tlp + wv * vlp).argmax(axis=1)
        r0 = np.arange(len(p))
        return d[r0, p], cc[r0, p]

    def eval_rk_hyb(sp, wv, mask=None):
        rk = sp["rk_lp"] if mask is None else sp["rk_lp"][mask]
        vlp = sp["proj_vlp"] if mask is None else sp["proj_vlp"][mask]
        d = sp["proj_d"] if mask is None else sp["proj_d"][mask]
        cc = sp["proj_cc"] if mask is None else sp["proj_cc"][mask]
        p = (rk + wv * vlp).argmax(axis=1)
        r0 = np.arange(len(p))
        return d[r0, p], cc[r0, p]

    wt_grid = [0.0, 0.4, 0.8, 1.2]
    wv_grid = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]

    # OOF for raw KDE+Text (no VLM), raw KDE+Text+VLM, and Stage 2 Listwise+VLM
    def cv_raw(wv_list):
        oof_d, oof_cc = np.zeros(len(q300_ids)), np.empty(len(q300_ids), dtype=object)
        for f in range(5):
            tr, te = folds != f, folds == f
            best = max(((wt, wv) for wt in wt_grid for wv in wv_list),
                       key=lambda c: summarize(*eval_raw_hyb(f300, *c, mask=tr), f300["true_cc"][tr])["mean_points"])
            oof_d[te], oof_cc[te] = eval_raw_hyb(f300, *best, mask=te)
        best_all = max(((wt, wv) for wt in wt_grid for wv in wv_list),
                       key=lambda c: summarize(*eval_raw_hyb(f300, *c), f300["true_cc"])["mean_points"])
        return best_all, summarize(oof_d, oof_cc, f300["true_cc"]), summarize(*eval_raw_hyb(f66, *best_all), f66["true_cc"]), summarize(*eval_raw_hyb(ftest, *best_all), ftest["true_cc"]), eval_raw_hyb(ftest, *best_all)

    def cv_rk(wv_list):
        oof_d, oof_cc = np.zeros(len(q300_ids)), np.empty(len(q300_ids), dtype=object)
        for f in range(5):
            tr, te = folds != f, folds == f
            best = max(wv_list, key=lambda wv: summarize(*eval_rk_hyb(f300, wv, mask=tr), f300["true_cc"][tr])["mean_points"])
            oof_d[te], oof_cc[te] = eval_rk_hyb(f300, best, mask=te)
        best_all = max(wv_list, key=lambda wv: summarize(*eval_rk_hyb(f300, wv), f300["true_cc"])["mean_points"])
        return best_all, summarize(oof_d, oof_cc, f300["true_cc"]), summarize(*eval_rk_hyb(f66, best_all), f66["true_cc"]), summarize(*eval_rk_hyb(ftest, best_all), ftest["true_cc"]), eval_rk_hyb(ftest, best_all)

    def eval_proj_kde_hyb(sp, wt, wv, mask=None):
        kde = sp["proj_kde"] if mask is None else sp["proj_kde"][mask]
        tlp = sp["proj_tlp"] if mask is None else sp["proj_tlp"][mask]
        vlp = sp["proj_vlp"] if mask is None else sp["proj_vlp"][mask]
        d = sp["proj_d"] if mask is None else sp["proj_d"][mask]
        cc = sp["proj_cc"] if mask is None else sp["proj_cc"][mask]
        p = (kde + wt * tlp + wv * vlp).argmax(axis=1)
        r0 = np.arange(len(p))
        return d[r0, p], cc[r0, p]

    def cv_proj_kde(wv_list):
        oof_d, oof_cc = np.zeros(len(q300_ids)), np.empty(len(q300_ids), dtype=object)
        for f in range(5):
            tr, te = folds != f, folds == f
            best = max(((wt, wv) for wt in wt_grid for wv in wv_list),
                       key=lambda c: summarize(*eval_proj_kde_hyb(f300, *c, mask=tr), f300["true_cc"][tr])["mean_points"])
            oof_d[te], oof_cc[te] = eval_proj_kde_hyb(f300, *best, mask=te)
        best_all = max(((wt, wv) for wt in wt_grid for wv in wv_list),
                       key=lambda c: summarize(*eval_proj_kde_hyb(f300, *c), f300["true_cc"])["mean_points"])
        return best_all, summarize(oof_d, oof_cc, f300["true_cc"]), summarize(*eval_proj_kde_hyb(f66, *best_all), f66["true_cc"]), summarize(*eval_proj_kde_hyb(ftest, *best_all), ftest["true_cc"]), eval_proj_kde_hyb(ftest, *best_all)

    cfg_raw_no_vlm, m300_rkde, m66_rkde, mtest_rkde, _ = cv_raw([0.0])
    cfg_raw_vlm, m300_rhyb, m66_rhyb, mtest_rhyb, (test_rhyb_d, test_rhyb_cc) = cv_raw(wv_grid)
    cfg_pkde_no_vlm, m300_pkde, m66_pkde, mtest_pkde, _ = cv_proj_kde([0.0])
    cfg_pkde_vlm, m300_phyb, m66_phyb, mtest_phyb, (test_phyb_d, test_phyb_cc) = cv_proj_kde(wv_grid)
    cfg_s2_vlm, m300_s2hyb, m66_s2hyb, mtest_s2hyb, (test_s2hyb_d, test_s2hyb_cc) = cv_rk(wv_grid)

    # Complementarity on official_test_300: Our best Hybrid vs osv5m/baseline!
    osv_d = np.array([p["distance_km"] for p in osv_preds])
    cands_ours = [
        ("sc25k_geoproj_kde_text_vlm", mtest_phyb["mean_points"], test_phyb_d, test_phyb_cc),
        ("sc25k_stage2_plus_vlm", mtest_s2hyb["mean_points"], test_s2hyb_d, test_s2hyb_cc),
        ("sc25k_raw_kde_text_vlm", mtest_rhyb["mean_points"], test_rhyb_d, test_rhyb_cc),
    ]
    best_ours_name, _, best_ours_d, best_ours_cc = max(cands_ours, key=lambda x: x[1])
    ours_closer_osv = int((best_ours_d + 1.0 < osv_d).sum())
    osv_closer_ours = int((osv_d + 1.0 < best_ours_d).sum())
    ours_closer_qwen = int((best_ours_d + 1.0 < np.nan_to_num(qsft_d, nan=1e9)).sum())
    qwen_closer_ours = int((np.nan_to_num(qsft_d, nan=1e9) + 1.0 < best_ours_d).sum())

    # Check true 25k gallery coverage on official_test_300
    d_test_gal = hav_vec(ts_lat[:, None], ts_lon[:, None], g_lat[None, :], g_lon[None, :])
    cov25_test = float((d_test_gal.min(axis=1) <= 25.0).mean())
    cov200_test = float((d_test_gal.min(axis=1) <= 200.0).mean())
    cov_cc_test = float(np.mean([c in set(g_cc.tolist()) for c in ts_cc]))

    final = {
        "official_test_300_gallery_coverage": {"within_25km": cov25_test, "within_200km": cov200_test, "same_country": cov_cc_test},
        "tuned_weights_on_qlora_300": {
            "raw_kde_text": list(cfg_raw_no_vlm),
            "raw_kde_text_vlm": list(cfg_raw_vlm),
            "geoproj_kde_text": list(cfg_pkde_no_vlm),
            "geoproj_kde_text_vlm": list(cfg_pkde_vlm),
            "stage2_listwise_vlm": cfg_s2_vlm,
        },
        "official_test_300": {
            "osv5m_baseline_official": osv_m,
            "qwen3vl_2b_zero_shot": qzs_m,
            "qwen3vl_2b_sft_6000": qsft_m,
            "dinov2_small_25k_raw": dino_test_m,
            "streetclip_25k_raw_top1": sc25_json["splits"]["official_test_300"]["raw_streetclip_25k"],
            "streetclip_25k_raw_kde_text_zero_touch": mtest_rkde,
            "streetclip_25k_geoproj_top1": sc25_json["splits"]["official_test_300"]["stage1_geoproj_top1"],
            "streetclip_25k_geoproj_kde": sc25_json["splits"]["official_test_300"]["stage1_geoproj_kde"],
            "streetclip_25k_geoproj_kde_text_zero_touch": mtest_pkde,
            "streetclip_25k_stage2_listwise": sc25_json["splits"]["official_test_300"]["stage2_listwise_ranker"],
            "streetclip_25k_raw_kde_text_vlm_zero_touch": mtest_rhyb,
            "streetclip_25k_geoproj_kde_text_vlm_zero_touch": mtest_phyb,
            "streetclip_25k_stage2_plus_vlm_zero_touch": mtest_s2hyb,
            "zs_streetclip_text_country_acc": sc25_json["splits"]["official_test_300"]["zs_text_country_acc"],
            "paired_vs_osv5m": {"best_ours": best_ours_name, "ours_closer": ours_closer_osv, "osv5m_closer": osv_closer_ours},
            "paired_vs_qwen6k": {"best_ours": best_ours_name, "ours_closer": ours_closer_qwen, "qwen6k_closer": qwen_closer_ours},
        },
        "author_holdout_66_25k": {
            "streetclip_25k_raw_top1": sc25_json["splits"]["author_holdout_66"]["raw_streetclip_25k"],
            "streetclip_25k_raw_kde_text_zero_touch": m66_rkde,
            "streetclip_25k_geoproj_kde": sc25_json["splits"]["author_holdout_66"]["stage1_geoproj_kde"],
            "streetclip_25k_geoproj_kde_text_zero_touch": m66_pkde,
            "streetclip_25k_stage2_listwise": sc25_json["splits"]["author_holdout_66"]["stage2_listwise_ranker"],
            "streetclip_25k_raw_kde_text_vlm_zero_touch": m66_rhyb,
            "streetclip_25k_geoproj_kde_text_vlm_zero_touch": m66_phyb,
            "streetclip_25k_stage2_plus_vlm_zero_touch": m66_s2hyb,
        },
        "qlora_eval_300_25k": {
            "streetclip_25k_raw_top1": sc25_json["splits"]["qlora_eval_300"]["raw_streetclip_25k"],
            "streetclip_25k_raw_kde_text_oof": m300_rkde,
            "streetclip_25k_geoproj_kde": sc25_json["splits"]["qlora_eval_300"]["stage1_geoproj_kde"],
            "streetclip_25k_geoproj_kde_text_oof": m300_pkde,
            "streetclip_25k_stage2_listwise": sc25_json["splits"]["qlora_eval_300"]["stage2_listwise_ranker"],
            "streetclip_25k_raw_kde_text_vlm_oof": m300_rhyb,
            "streetclip_25k_geoproj_kde_text_vlm_oof": m300_phyb,
            "streetclip_25k_stage2_plus_vlm_oof": m300_s2hyb,
        },
        "val_2000_25k": sc25_json["splits"]["val_2000"],
        "val_unseen_opt_522_25k": sc25_json["splits"]["val_unseen_opt_522"],
    }

    print(f"official_test_300 25k gallery coverage: <=25km={100*cov25_test:.1f}%, <=200km={100*cov200_test:.1f}%, country={100*cov_cc_test:.1f}%")
    print(f"Tuned on qlora_300: raw_kde_text={cfg_raw_no_vlm}, raw_kde_text_vlm={cfg_raw_vlm}, stage2_vlm={cfg_s2_vlm}")

    for section in ("official_test_300", "author_holdout_66_25k", "qlora_eval_300_25k"):
        print(f"\n==================== {section} ====================")
        for k, m in final[section].items():
            if isinstance(m, dict) and "median_km" in m:
                cc_s = f"{m['country_correct']:3d} ({100*m['country_rate']:5.1f}%)" if "country_correct" in m else "   n/a      "
                print(f"  {k:44s}: <=25={m['hit_25']:3d} ({100*m['hit_25_rate']:5.2f}%), <=200={m['hit_200']:3d} ({100*m['hit_200_rate']:5.2f}%), "
                      f"cc={cc_s}, med={m['median_km']:6.1f}km, pts={m['mean_points']:6.1f}")
    print(f"\nPaired on official_test_300 ({best_ours_name}): vs osv5m/baseline={ours_closer_osv} closer / {osv_closer_ours} farther | vs Qwen3-VL 6k={ours_closer_qwen} closer / {qwen_closer_ours} farther")

    (OTEST_DIR / "showdown_summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"\nSaved {OTEST_DIR / 'showdown_summary.json'}")


if __name__ == "__main__":
    main()
