"""Finalize Step 1D: Strictly cross-validated (5-fold OOF on 300, zero-touch on author_holdout_66)
StreetCLIP retrieval + KDE + text-probe + Qwen3-VL hybrid, plus creator-shortcut & error complementarity audit.
"""
import json
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


def main():
    with zipfile.ZipFile(PAIRS_ZIP) as z:
        trn_all = {r["id"]: r for r in (json.loads(l) for l in z.read("train_images.jsonl").splitlines())}
        val_all = {r["id"]: r for r in (json.loads(l) for l in z.read("validation_images.jsonl").splitlines())}

    with zipfile.ZipFile(SFT_6K_UPLOAD) as z:
        g6k_ids = [json.loads(l)["id"] for l in z.read("train.jsonl").splitlines()]
        q300_ids = [json.loads(l)["id"] for l in z.read("eval.jsonl").splitlines()]
    ah66_ids = [json.loads(l)["id"] for l in (ROOT / "data/author_holdout/eval.jsonl").read_text().splitlines()]

    g6k_meta = [trn_all[i] for i in g6k_ids]
    q300_meta = [val_all[i] for i in q300_ids]
    ah66_meta = [val_all[i] for i in ah66_ids]

    sc = np.load(SC_NPZ)
    sc_id_to_i = {str(i): k for k, i in enumerate(sc["ids"])}
    sc_full = sc["feats_full"].astype(np.float32)
    sc_crop = sc["feats_crop88"].astype(np.float32)
    sc_full /= np.linalg.norm(sc_full, axis=-1, keepdims=True)
    sc_crop /= np.linalg.norm(sc_crop, axis=-1, keepdims=True)
    sc_mv = sc_full + sc_crop
    sc_mv /= np.linalg.norm(sc_mv, axis=-1, keepdims=True)
    text_feats = sc["text_feats"].astype(np.float32)
    text_codes = np.array([str(c) for c in sc["country_codes"]])
    cc_to_ti = {c: i for i, c in enumerate(text_codes)}

    with zipfile.ZipFile(STEP1BC_ZIP) as z:
        vlm_priors = json.loads(z.read("vlm_country_priors.json"))["predictions"]
    with zipfile.ZipFile(QLORA_6K_RESULTS) as z:
        vlm_300 = {r["id"]: r for r in (json.loads(l) for l in z.read("sft.jsonl").splitlines())}
    with zipfile.ZipFile(AH_RESULTS) as z:
        vlm_66 = {r["id"]: r for r in (json.loads(l) for l in z.read("sft.jsonl").splitlines())}

    def gather(arr, ids):
        return np.stack([arr[sc_id_to_i[i]] for i in ids])

    g_lat = np.array([r["latitude"] for r in g6k_meta], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in g6k_meta], dtype=np.float64)
    g_cc = np.array([r["country"] for r in g6k_meta])
    g_attr = np.array([r["attribution"] for r in g6k_meta])

    # 1. Check creator-shortcut sensitivity of StreetCLIP vs DINOv2 on qlora_eval_300
    print("=== CREATOR-SHORTCUT RESISTANCE CHECK (qlora_eval_300 vs 6k gallery) ===")
    q300_attr = np.array([r["attribution"] for r in q300_meta])
    q300_lat = np.array([r["latitude"] for r in q300_meta], dtype=np.float64)
    q300_lon = np.array([r["longitude"] for r in q300_meta], dtype=np.float64)
    q300_dmat = hav_mat(q300_lat, q300_lon, g_lat, g_lon)
    same300 = (q300_attr[:, None] == g_attr[None, :]) & (q300_attr[:, None] != "")
    for name, emb in [("StreetCLIP full", sc_full), ("StreetCLIP crop88", sc_crop), ("StreetCLIP multi-view", sc_mv)]:
        s = gather(emb, q300_ids) @ gather(emb, g6k_ids).T
        top1_raw = s.argmax(axis=1)
        is_same = same300[np.arange(len(q300_ids)), top1_raw]
        d_raw = q300_dmat[np.arange(len(q300_ids)), top1_raw]
        s_diff = np.where(same300, -2.0, s)
        top1_diff = s_diff.argmax(axis=1)
        d_diff = q300_dmat[np.arange(len(q300_ids)), top1_diff]
        print(f"  {name:22s}: raw Top-1 same-creator={is_same.sum():2d}/300 ({100*is_same.mean():4.1f}%), "
              f"raw <=25km={(d_raw<=25).sum():2d} (same={((d_raw<=25)&is_same).sum()}), "
              f"diff-creator <=25km={(d_diff<=25).sum():2d}, raw med={np.median(d_raw):.0f}km vs diff med={np.median(d_diff):.0f}km")

    # 2. Precompute Top-100 arrays for clean 5-fold OOF tuning on qlora_eval_300 -> zero-touch test on author_holdout_66
    def prep_split(emb, q_ids, q_meta, top_k=100):
        q_e = gather(emb, q_ids)
        g_e = gather(emb, g6k_ids)
        q_attr = np.array([r["attribution"] for r in q_meta])
        q_lat = np.array([r["latitude"] for r in q_meta], dtype=np.float64)
        q_lon = np.array([r["longitude"] for r in q_meta], dtype=np.float64)
        q_cc = np.array([r["country"] for r in q_meta])
        sim = q_e @ g_e.T
        same = (q_attr[:, None] == g_attr[None, :]) & (q_attr[:, None] != "")
        sim_diff = np.where(same, -2.0, sim)
        dist = hav_mat(q_lat, q_lon, g_lat, g_lon)

        part = np.argpartition(-sim_diff, top_k, axis=1)[:, :top_k]
        rows = np.arange(len(q_meta))[:, None]
        order = np.argsort(-sim_diff[rows, part], axis=1)
        top_idx = part[rows, order]
        top_sim = sim_diff[rows, top_idx]
        top_d = dist[rows, top_idx]
        top_cc = g_cc[top_idx]

        la = np.radians(g_lat[top_idx])
        lo = np.radians(g_lon[top_idx])
        dlat = la[:, :, None] - la[:, None, :]
        dlon = lo[:, :, None] - lo[:, None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin(dlon / 2) ** 2
        pw_d = (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)

        # Text-probe log-prob (temperature-scaled with standard CLIP scale=30)
        t_logits = (q_e @ text_feats.T) * 30.0
        tm = t_logits.max(axis=1, keepdims=True)
        t_lp = (t_logits - tm) - np.log(np.exp(t_logits - tm).sum(axis=1, keepdims=True))
        cand_tlp = np.full(top_idx.shape, -15.0, dtype=np.float64)
        cand_vlp = np.full(top_idx.shape, -15.0, dtype=np.float64)
        for i, r in enumerate(q_meta):
            vlp_d = vlm_priors[r["id"]]["log_probs"]
            for j in range(top_k):
                cc = str(top_cc[i, j])
                ti = cc_to_ti.get(cc)
                if ti is not None:
                    cand_tlp[i, j] = t_lp[i, ti]
                cand_vlp[i, j] = vlp_d.get(cc, -15.0)
        return dict(q_ids=q_ids, q_cc=q_cc, top_sim=top_sim, top_d=top_d, top_cc=top_cc,
                    pw_d=pw_d, cand_tlp=cand_tlp, cand_vlp=cand_vlp,
                    zs_text_acc=float((text_codes[t_logits.argmax(axis=1)] == q_cc).mean()))

    def score_with(sp, tau, sigma, w_text, w_vlm, mask=None):
        ts = sp["top_sim"] if mask is None else sp["top_sim"][mask]
        pw = sp["pw_d"] if mask is None else sp["pw_d"][mask]
        tlp = sp["cand_tlp"] if mask is None else sp["cand_tlp"][mask]
        vlp = sp["cand_vlp"] if mask is None else sp["cand_vlp"][mask]
        td = sp["top_d"] if mask is None else sp["top_d"][mask]
        tcc = sp["top_cc"] if mask is None else sp["top_cc"][mask]
        w = np.exp((ts - ts[:, :1]) / tau)
        K = np.exp(-pw / sigma)
        kde = np.log(np.maximum(np.einsum("nj,nij->ni", w, K), 1e-12))
        tot = kde + w_text * tlp + w_vlm * vlp
        pick = tot.argmax(axis=1)
        r0 = np.arange(len(pick))
        return td[r0, pick], tcc[r0, pick]

    results = {}
    for view_name, emb in [("streetclip_full", sc_full), ("streetclip_multiview", sc_mv)]:
        s300 = prep_split(emb, q300_ids, q300_meta)
        s66 = prep_split(emb, ah66_ids, ah66_meta)

        # Tune (tau, sigma, w_text, w_vlm) strictly via 5-fold CV on qlora_eval_300
        taus = [0.025, 0.04, 0.06]
        sigmas = [200.0, 400.0, 650.0]
        w_texts = [0.0, 0.3, 0.6, 1.0]
        w_vlms_only = [0.0]
        w_vlms_hyb = [0.0, 0.15, 0.3, 0.5, 0.75]

        folds = np.arange(len(q300_ids)) % 5

        def run_cv(w_vlm_list):
            oof_d = np.zeros(len(q300_ids))
            oof_cc = np.empty(len(q300_ids), dtype=object)
            for f in range(5):
                tr, te = folds != f, folds == f
                best = max(
                    ((t, sg, wt, wv) for t in taus for sg in sigmas for wt in w_texts for wv in w_vlm_list),
                    key=lambda cfg: summarize(*score_with(s300, *cfg, mask=tr), s300["q_cc"][tr])["mean_points"],
                )
                oof_d[te], oof_cc[te] = score_with(s300, *best, mask=te)
            best_all = max(
                ((t, sg, wt, wv) for t in taus for sg in sigmas for wt in w_texts for wv in w_vlm_list),
                key=lambda cfg: summarize(*score_with(s300, *cfg), s300["q_cc"])["mean_points"],
            )
            ah_d, ah_cc = score_with(s66, *best_all)
            return best_all, summarize(oof_d, oof_cc, s300["q_cc"]), summarize(ah_d, ah_cc, s66["q_cc"]), ah_d, ah_cc

        cfg_sc, oof_sc, ah_sc, ah_sc_d, ah_sc_cc = run_cv(w_vlms_only)
        cfg_hyb, oof_hyb, ah_hyb, ah_hyb_d, ah_hyb_cc = run_cv(w_vlms_hyb)

        results[view_name] = {
            "zs_text_acc_300": s300["zs_text_acc"],
            "zs_text_acc_66": s66["zs_text_acc"],
            "streetclip_only_cfg": cfg_sc,
            "streetclip_only_300_oof": oof_sc,
            "streetclip_only_66_zero_touch": ah_sc,
            "hybrid_cfg": cfg_hyb,
            "hybrid_300_oof": oof_hyb,
            "hybrid_66_zero_touch": ah_hyb,
        }
        print(f"\n=== {view_name} (5-fold OOF on 300 -> ZERO-TOUCH on author_holdout_66) ===")
        print(f"  ZS text country acc: 300={100*s300['zs_text_acc']:.1f}%, 66={100*s66['zs_text_acc']:.1f}%")
        print(f"  StreetCLIP+KDE+Text cfg={cfg_sc}:")
        print(f"    300 OOF       : <=25={oof_sc['hit_25']:2d} ({100*oof_sc['hit_25_rate']:5.2f}%), <=200={oof_sc['hit_200']:2d} ({100*oof_sc['hit_200_rate']:5.2f}%), cc={oof_sc['country_correct']:3d} ({100*oof_sc['country_rate']:5.1f}%), med={oof_sc['median_km']:5.1f}km, pts={oof_sc['mean_points']:.1f}")
        print(f"    66 ZERO-TOUCH : <=25={ah_sc['hit_25']:2d} ({100*ah_sc['hit_25_rate']:5.2f}%), <=200={ah_sc['hit_200']:2d} ({100*ah_sc['hit_200_rate']:5.2f}%), cc={ah_sc['country_correct']:3d} ({100*ah_sc['country_rate']:5.1f}%), med={ah_sc['median_km']:5.1f}km, pts={ah_sc['mean_points']:.1f}")
        print(f"  Hybrid (+Qwen3-VL 6k country) cfg={cfg_hyb}:")
        print(f"    300 OOF       : <=25={oof_hyb['hit_25']:2d} ({100*oof_hyb['hit_25_rate']:5.2f}%), <=200={oof_hyb['hit_200']:2d} ({100*oof_hyb['hit_200_rate']:5.2f}%), cc={oof_hyb['country_correct']:3d} ({100*oof_hyb['country_rate']:5.1f}%), med={oof_hyb['median_km']:5.1f}km, pts={oof_hyb['mean_points']:.1f}")
        print(f"    66 ZERO-TOUCH : <=25={ah_hyb['hit_25']:2d} ({100*ah_hyb['hit_25_rate']:5.2f}%), <=200={ah_hyb['hit_200']:2d} ({100*ah_hyb['hit_200_rate']:5.2f}%), cc={ah_hyb['country_correct']:3d} ({100*ah_hyb['country_rate']:5.1f}%), med={ah_hyb['median_km']:5.1f}km, pts={ah_hyb['mean_points']:.1f}")

        if view_name == "streetclip_multiview":
            # Complementarity check on author_holdout_66 vs Qwen3-VL 6k!
            vlm_d66 = np.array([vlm_66[i]["distance_km"] for i in ah66_ids])
            vlm_cc66 = np.array([vlm_66[i]["guess_country"] for i in ah66_ids])
            true_cc66 = s66["q_cc"]
            both_cc = int(((ah_sc_cc == true_cc66) & (vlm_cc66 == true_cc66)).sum())
            only_sc = int(((ah_sc_cc == true_cc66) & (vlm_cc66 != true_cc66)).sum())
            only_vlm = int(((ah_sc_cc != true_cc66) & (vlm_cc66 == true_cc66)).sum())
            neither = int(((ah_sc_cc != true_cc66) & (vlm_cc66 != true_cc66)).sum())
            sc_closer = int((ah_hyb_d + 1.0 < vlm_d66).sum())
            vlm_closer = int((vlm_d66 + 1.0 < ah_hyb_d).sum())
            print(f"\n=== Complementarity on author_holdout_66 (StreetCLIP-MV vs Qwen3-VL 6k) ===")
            print(f"  Country: both_right={both_cc}, only_StreetCLIP={only_sc}, only_VLM={only_vlm}, neither={neither} (oracle union={both_cc+only_sc+only_vlm}/66 = {100*(both_cc+only_sc+only_vlm)/66:.1f}%)")
            print(f"  Paired distance (Hybrid vs VLM 6k): Hybrid closer={sc_closer}/66, VLM closer={vlm_closer}/66")

    (OUT_DIR / "step1d_final_oof.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved {OUT_DIR / 'step1d_final_oof.json'}")


if __name__ == "__main__":
    main()
