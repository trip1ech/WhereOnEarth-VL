"""Step 1D Analysis: Apples-to-apples ceiling-raising comparison on the clean 6,000-image gallery
between frozen dinov2-small vs geolocal/StreetCLIP (full vs bottom-12% HUD-stripped crop)
plus StreetCLIP spatial KDE, zero-shot text-probe country prior, and Qwen3-VL 6k Hybrid!

Evaluates strictly diff-creator on:
- author_holdout_66 (0% creator overlap with the 6,000 gallery AND 0% overlap with Qwen 6k SFT!)
- qlora_eval_300 (same-creator masked out of the 6,000 gallery)
"""
import json
import math
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
PAIRS_ZIP = ROOT / "results/retrieval_v2/pairs_compact.zip"
SFT_6K_UPLOAD = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
QLORA_6K_RESULTS = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
AH_RESULTS = ROOT / "results/author_holdout/author-holdout-results.zip"
STEP1BC_ZIP = ROOT / "results/retrieval_v2/step1bc_results.zip"
SC_NPZ = ROOT / "results/retrieval_v2/streetclip_features.npz"
OUT_DIR = ROOT / "results/retrieval_v2"


def hav_mat(lat1, lon1, lat2, lon2):
    lat1, lon1 = np.radians(lat1)[:, None], np.radians(lon1)[:, None]
    lat2, lon2 = np.radians(lat2)[None, :], np.radians(lon2)[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)


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


def eval_retrieval(q_emb, q_meta, g_emb, g_meta, *, vlm_priors=None, vlm_coords=None,
                   text_feats=None, text_codes=None, logit_scale=100.0,
                   kde_tau=0.04, kde_sigma=400.0, top_k=100):
    sim = q_emb @ g_emb.T
    q_attr = np.array([r["attribution"] for r in q_meta])
    g_attr = np.array([r["attribution"] for r in g_meta])
    same = (q_attr[:, None] == g_attr[None, :]) & (q_attr[:, None] != "")
    sim_diff = np.where(same, -2.0, sim)

    q_lat = np.array([r["latitude"] for r in q_meta], dtype=np.float64)
    q_lon = np.array([r["longitude"] for r in q_meta], dtype=np.float64)
    q_cc = np.array([r["country"] for r in q_meta])
    g_lat = np.array([r["latitude"] for r in g_meta], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in g_meta], dtype=np.float64)
    g_cc = np.array([r["country"] for r in g_meta])

    dist = hav_mat(q_lat, q_lon, g_lat, g_lon)
    dist_diff = np.where(same, np.inf, dist)

    # Top-K diff-creator
    part = np.argpartition(-sim_diff, top_k, axis=1)[:, :top_k]
    rows = np.arange(len(q_meta))[:, None]
    order = np.argsort(-sim_diff[rows, part], axis=1)
    top_idx = part[rows, order]
    top_sim = sim_diff[rows, top_idx]
    top_d = dist[rows, top_idx]
    top_cc = g_cc[top_idx]

    # Top-1 raw diff-creator
    out = {
        "coverage_diff_25km": float((dist_diff.min(axis=1) <= 25.0).mean()),
        "coverage_diff_200km": float((dist_diff.min(axis=1) <= 200.0).mean()),
        "top100_recall_25km": float((top_d.min(axis=1) <= 25.0).mean()),
        "top100_recall_200km": float((top_d.min(axis=1) <= 200.0).mean()),
        "top100_recall_country": float(np.any(top_cc == q_cc[:, None], axis=1).mean()),
        "top1": summarize(top_d[:, 0], top_cc[:, 0], q_cc),
    }

    # Spatial KDE over Top-K
    la = np.radians(g_lat[top_idx])
    lo = np.radians(g_lon[top_idx])
    dlat = la[:, :, None] - la[:, None, :]
    dlon = lo[:, :, None] - lo[:, None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin(dlon / 2) ** 2
    pw_d = (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)

    w = np.exp((top_sim - top_sim[:, :1]) / kde_tau)
    K = np.exp(-pw_d / kde_sigma)
    kde_scores = np.einsum("nj,nij->ni", w, K)
    kde_log = np.log(np.maximum(kde_scores, 1e-12))
    kde_pick = kde_scores.argmax(axis=1)
    r0 = np.arange(len(q_meta))
    out["kde"] = summarize(top_d[r0, kde_pick], top_cc[r0, kde_pick], q_cc)

    # Zero-shot StreetCLIP country text-probe (if provided)
    if text_feats is not None:
        t_logits = (q_emb @ text_feats.T) * min(logit_scale, 30.0)
        t_max = t_logits.max(axis=1, keepdims=True)
        t_lp = (t_logits - t_max) - np.log(np.exp(t_logits - t_max).sum(axis=1, keepdims=True))
        t_pred_cc = text_codes[t_logits.argmax(axis=1)]
        out["zeroshot_text_country_acc"] = float((t_pred_cc == q_cc).mean())
        cc_to_ti = {c: i for i, c in enumerate(text_codes)}
        cand_tlp = np.full(top_idx.shape, -15.0, dtype=np.float64)
        for i in range(len(q_meta)):
            for j in range(top_k):
                ti = cc_to_ti.get(str(top_cc[i, j]))
                if ti is not None:
                    cand_tlp[i, j] = t_lp[i, ti]
        # Combine KDE + StreetCLIP text country prior
        sc_comb = kde_log + 0.5 * cand_tlp
        cp = sc_comb.argmax(axis=1)
        out["kde_plus_textprobe"] = summarize(top_d[r0, cp], top_cc[r0, cp], q_cc)

    # Hybrid with Qwen3-VL 6k country log-probs (+ optional VLM spatial anchor)
    if vlm_priors is not None:
        cand_vlp = np.full(top_idx.shape, -15.0, dtype=np.float64)
        for i, r in enumerate(q_meta):
            lp_d = vlm_priors[r["id"]]["log_probs"]
            for j in range(top_k):
                cand_vlp[i, j] = lp_d.get(str(top_cc[i, j]), -15.0)
        # Mode 1: KDE + Qwen3-VL country log-prob
        h1 = kde_log + 0.75 * cand_vlp
        p1 = h1.argmax(axis=1)
        out["hybrid_kde_vlm_country"] = summarize(top_d[r0, p1], top_cc[r0, p1], q_cc)
        if text_feats is not None:
            # Mode 2: KDE + StreetCLIP text-probe + Qwen3-VL country log-prob (true ensemble!)
            h2 = kde_log + 0.4 * cand_tlp + 0.75 * cand_vlp
            p2 = h2.argmax(axis=1)
            out["hybrid_kde_textprobe_vlm_country"] = summarize(top_d[r0, p2], top_cc[r0, p2], q_cc)
        if vlm_coords is not None:
            vlat = np.array([vlm_coords[r["id"]][0] for r in q_meta])
            vlon = np.array([vlm_coords[r["id"]][1] for r in q_meta])
            c2v = hav_mat(vlat, vlon, g_lat, g_lon)[rows, top_idx]
            h3 = kde_log + 0.75 * cand_vlp - 0.5 * (c2v / 1000.0)
            p3 = h3.argmax(axis=1)
            out["hybrid_kde_vlm_country_spatial"] = summarize(top_d[r0, p3], top_cc[r0, p3], q_cc)

    return out


def main():
    with zipfile.ZipFile(PAIRS_ZIP) as z:
        trn_all = {r["id"]: r for r in (json.loads(l) for l in z.read("train_images.jsonl").splitlines())}
        val_all = {r["id"]: r for r in (json.loads(l) for l in z.read("validation_images.jsonl").splitlines())}
        with z.open("train_features.npz") as f:
            tf_all = np.load(f)["features"].astype(np.float32)
        with z.open("validation_features.npz") as f:
            vf_all = np.load(f)["features"].astype(np.float32)
        trn_ids_ord = [json.loads(l)["id"] for l in z.read("train_images.jsonl").splitlines()]
        val_ids_ord = [json.loads(l)["id"] for l in z.read("validation_images.jsonl").splitlines()]
    tf_all /= np.linalg.norm(tf_all, axis=-1, keepdims=True)
    vf_all /= np.linalg.norm(vf_all, axis=-1, keepdims=True)
    dino_map = {i: tf_all[k] for k, i in enumerate(trn_ids_ord)}
    dino_map.update({i: vf_all[k] for k, i in enumerate(val_ids_ord)})

    with zipfile.ZipFile(SFT_6K_UPLOAD) as z:
        g6k_ids = [json.loads(l)["id"] for l in z.read("train.jsonl").splitlines()]
        q300_ids = [json.loads(l)["id"] for l in z.read("eval.jsonl").splitlines()]
    ah66_ids = [json.loads(l)["id"] for l in (ROOT / "data/author_holdout/eval.jsonl").read_text().splitlines()]

    g6k_meta = [trn_all[i] for i in g6k_ids]
    q300_meta = [val_all[i] for i in q300_ids]
    ah66_meta = [val_all[i] for i in ah66_ids]

    # Verify 0% creator overlap between g6k and ah66
    g6k_creators = {r["attribution"] for r in g6k_meta if r["attribution"]}
    ah66_creators = {r["attribution"] for r in ah66_meta if r["attribution"]}
    print(f"Creator overlap g6k vs author_holdout_66: {len(g6k_creators & ah66_creators)} (expected 0)", flush=True)

    sc = np.load(SC_NPZ)
    sc_id_to_i = {str(i): k for k, i in enumerate(sc["ids"])}
    sc_full = sc["feats_full"].astype(np.float32)
    sc_crop = sc["feats_crop88"].astype(np.float32)
    sc_full /= np.linalg.norm(sc_full, axis=-1, keepdims=True)
    sc_crop /= np.linalg.norm(sc_crop, axis=-1, keepdims=True)
    text_feats = sc["text_feats"].astype(np.float32)
    text_codes = np.array([str(c) for c in sc["country_codes"]])
    logit_scale = float(sc["logit_scale"][0])

    vlm_priors = None
    if STEP1BC_ZIP.exists():
        with zipfile.ZipFile(STEP1BC_ZIP) as z:
            if "vlm_country_priors.json" in z.namelist():
                vlm_priors = json.loads(z.read("vlm_country_priors.json"))["predictions"]

    vlm_coords = {}
    with zipfile.ZipFile(QLORA_6K_RESULTS) as z:
        for l in z.read("sft.jsonl").splitlines():
            r = json.loads(l)
            vlm_coords[r["id"]] = r["guess_coordinates"]
    with zipfile.ZipFile(AH_RESULTS) as z:
        for l in z.read("sft.jsonl").splitlines():
            r = json.loads(l)
            vlm_coords[r["id"]] = r["guess_coordinates"]

    def gather(arr, ids):
        return np.stack([arr[sc_id_to_i[i]] for i in ids])

    def gather_dino(ids):
        return np.stack([dino_map[i] for i in ids])

    report = {}
    for split_name, q_ids, q_meta in [("qlora_eval_300", q300_ids, q300_meta), ("author_holdout_66", ah66_ids, ah66_meta)]:
        res = {}
        # 1. DINOv2-small on same 6k gallery
        res["dinov2_small_6k"] = eval_retrieval(
            gather_dino(q_ids), q_meta, gather_dino(g6k_ids), g6k_meta,
            vlm_priors=vlm_priors, vlm_coords=vlm_coords, kde_tau=0.07, kde_sigma=600.0,
        )
        # 2. StreetCLIP full on same 6k gallery
        res["streetclip_full_6k"] = eval_retrieval(
            gather(sc_full, q_ids), q_meta, gather(sc_full, g6k_ids), g6k_meta,
            vlm_priors=vlm_priors, vlm_coords=vlm_coords,
            text_feats=text_feats, text_codes=text_codes, logit_scale=logit_scale,
            kde_tau=0.04, kde_sigma=500.0,
        )
        # 3. StreetCLIP bottom-12% HUD-cropped on same 6k gallery
        res["streetclip_crop88_6k"] = eval_retrieval(
            gather(sc_crop, q_ids), q_meta, gather(sc_crop, g6k_ids), g6k_meta,
            vlm_priors=vlm_priors, vlm_coords=vlm_coords,
            text_feats=text_feats, text_codes=text_codes, logit_scale=logit_scale,
            kde_tau=0.04, kde_sigma=500.0,
        )
        report[split_name] = res

        print(f"\n==================== {split_name} (6k diff-creator gallery) ====================")
        for enc_name, d in res.items():
            print(f"\n  [{enc_name}] Top-100 recall: <=25={100*d['top100_recall_25km']:.1f}%, <=200={100*d['top100_recall_200km']:.1f}%, cc={100*d['top100_recall_country']:.1f}%"
                  + (f" | ZS text cc={100*d['zeroshot_text_country_acc']:.1f}%" if "zeroshot_text_country_acc" in d else ""))
            for mkey in ("top1", "kde", "kde_plus_textprobe", "hybrid_kde_vlm_country", "hybrid_kde_textprobe_vlm_country", "hybrid_kde_vlm_country_spatial"):
                if mkey in d:
                    m = d[mkey]
                    print(f"    {mkey:34s}: <=25={m['hit_25']:2d} ({100*m['hit_25_rate']:5.2f}%), <=200={m['hit_200']:2d} ({100*m['hit_200_rate']:5.2f}%), "
                          f"cc={m['country_correct']:3d} ({100*m['country_rate']:5.1f}%), med={m['median_km']:6.1f}km, pts={m['mean_points']:6.1f}")

    (OUT_DIR / "step1d_streetclip_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {OUT_DIR / 'step1d_streetclip_results.json'}")


if __name__ == "__main__":
    main()
