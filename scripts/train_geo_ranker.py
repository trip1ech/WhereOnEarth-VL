"""Step 1B: Leak-free Full-Gallery Geo-Metric Projection + Listwise Geo-Ranker on cached 25k DINOv2 embeddings.

Key fixes over legacy scripts/train_ranker.py:
1. Strict diff-creator supervision & evaluation (masks same-creator nearby gallery items during training,
   uses same-creator distant items >300 km as hard negatives to unlearn camera/dashcam fingerprints,
   and masks ALL same-creator gallery items during evaluation).
2. Excludes all author_holdout_66 creators from optimization training & splits train/tune strictly by creator.
3. Stage 1 learns a shared residual Geo-Metric Projection over the FULL 25,000 gallery (raising Top-100 recall itself).
4. Stage 2 trains a stabilized Listwise Softmax Ranker over all 100 re-retrieved candidates using
   multi-scale spatial KDE, gallery-density debiasing, country consensus, and auxiliary country probabilities.
"""
import argparse
import copy
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 7818


def haversine_matrix_np(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)


def haversine_vec_np(lat1, lon1, lat2, lon2):
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


def compute_gallery_local_density(g_lat, g_lon, sigma_km=200.0, chunk=1000):
    n = len(g_lat)
    density = np.zeros(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        d = haversine_matrix_np(g_lat[start:end], g_lon[start:end], g_lat, g_lon)
        density[start:end] = np.exp(-d / sigma_km).sum(axis=1)
    return density


class GeoProjector(nn.Module):
    """Shared residual metric projection + auxiliary country head."""

    def __init__(self, dim=384, hidden=512, num_countries=180, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.linear_skip = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.linear_skip.weight)
        self.log_scale = nn.Parameter(torch.tensor(math.log(15.0)))
        self.country_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_countries),
        )

    def encode(self, x):
        z = self.linear_skip(x) + self.net(x)
        return F.normalize(z, dim=-1)

    def scale(self):
        return self.log_scale.clamp(math.log(4.0), math.log(45.0)).exp()


class ListwiseGeoRanker(nn.Module):
    """Stabilized listwise ranker over Top-100 candidates with bounded residual."""

    def __init__(self, feat_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.base_scale = nn.Parameter(torch.tensor(10.0))
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, geo_sim, feats):
        # geo_sim: (B, K), feats: (B, K, F)
        delta = self.mlp(feats).squeeze(-1)
        return self.base_scale.clamp(1.0, 40.0) * geo_sim + 6.0 * torch.tanh(delta / 6.0)


@torch.inference_mode()
def retrieve_diff_creator(q_emb, q_attr, g_emb, g_attr, k=100, chunk=512):
    """Exact diff-creator Top-K retrieval on GPU."""
    device = q_emb.device
    n = len(q_emb)
    top_idx = np.empty((n, k), dtype=np.int32)
    top_sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        s = q_emb[start:end] @ g_emb.T  # (b, G)
        qa = q_attr[start:end][:, None]
        same = (qa == g_attr[None, :]) & (qa != "")
        same_t = torch.as_tensor(same, device=device)
        s = s.masked_fill(same_t, -2.0)
        vals, idxs = torch.topk(s, k=k, dim=1)
        top_idx[start:end] = idxs.cpu().numpy().astype(np.int32)
        top_sim[start:end] = vals.cpu().numpy().astype(np.float32)
    return top_idx, top_sim


def build_candidate_features(top_idx, geo_sim, raw_sim, aux_cc_logp, g_lat, g_lon, g_cc_idx,
                             g_density, cc_freq_log):
    """Build scale-normalized per-candidate features for all (N, K) candidates."""
    n, k = top_idx.shape
    lats = np.radians(g_lat[top_idx])
    lons = np.radians(g_lon[top_idx])
    # Compute pairwise distances in chunks to keep RAM modest
    feats = np.zeros((n, k, 22), dtype=np.float32)
    kde_best_scores = np.zeros((n, k), dtype=np.float32)
    rank = np.arange(k, dtype=np.float32)[None, :]
    log_rank = np.log1p(rank)
    inv_rank = 1.0 / (rank + 1.0)

    chunk = 400
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        b = end - start
        c_idx = top_idx[start:end]
        gs = geo_sim[start:end]
        rs = raw_sim[start:end]
        la = lats[start:end]
        lo = lons[start:end]
        dlat = la[:, :, None] - la[:, None, :]
        dlon = lo[:, :, None] - lo[:, None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin(dlon / 2) ** 2
        pw_d = (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)

        dens = g_density[c_idx]
        inv_dens_25 = dens ** -0.25
        cc_i = g_cc_idx[c_idx]  # (b, k)
        same_cc = (cc_i[:, :, None] == cc_i[:, None, :]).astype(np.float32)

        w05 = np.exp((gs - gs[:, :1]) / 0.05)
        w05 /= w05.sum(axis=1, keepdims=True)
        w10 = np.exp((gs - gs[:, :1]) / 0.10)
        w10 /= w10.sum(axis=1, keepdims=True)
        w10_deb = w10 * inv_dens_25
        w10_deb /= w10_deb.sum(axis=1, keepdims=True)

        def norm_row(x):
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-6
            return (x - mu) / sd

        col = 0
        feats[start:end, :, col] = gs; col += 1
        feats[start:end, :, col] = gs - gs[:, :1]; col += 1
        feats[start:end, :, col] = norm_row(gs); col += 1
        feats[start:end, :, col] = rs; col += 1
        feats[start:end, :, col] = rs - rs[:, :1]; col += 1
        feats[start:end, :, col] = log_rank; col += 1
        feats[start:end, :, col] = inv_rank; col += 1

        # Multi-scale KDE features (raw & debiased)
        for sigma in (50.0, 200.0, 600.0, 1500.0):
            K = np.exp(-pw_d / sigma)
            s_raw = np.einsum("nj,nij->ni", w05, K)
            feats[start:end, :, col] = norm_row(s_raw); col += 1
            s_deb = np.einsum("nj,nij->ni", w10_deb, K)
            feats[start:end, :, col] = norm_row(s_deb); col += 1
            if sigma == 600.0:
                kde_best_scores[start:end] = np.einsum("nj,nij->ni", w10, K)

        # Country consensus in Top-100 (tau=0.05 and tau=0.10)
        cc_vote05 = np.einsum("nj,nij->ni", w05, same_cc)
        cc_vote10 = np.einsum("nj,nij->ni", w10, same_cc)
        feats[start:end, :, col] = norm_row(cc_vote05); col += 1
        feats[start:end, :, col] = norm_row(cc_vote10); col += 1
        # Gallery density & country frequency priors (for debiasing)
        feats[start:end, :, col] = np.log1p(dens); col += 1
        feats[start:end, :, col] = cc_freq_log[cc_i]; col += 1
        # Auxiliary country head log-prob for each candidate's country
        rows_b = np.arange(b)[:, None]
        c_lp = aux_cc_logp[start:end][rows_b, cc_i]
        feats[start:end, :, col] = c_lp; col += 1
        feats[start:end, :, col] = norm_row(c_lp); col += 1
        # Raw neighbor count within 200km
        feats[start:end, :, col] = np.log1p((pw_d <= 200.0).sum(axis=2)); col += 1
        assert col == 22, f"Expected 22 features, got {col}"

    return feats, kde_best_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_zip", type=Path)
    parser.add_argument("splits_json", type=Path, help="JSON with qlora_300_ids and author_holdout_66_ids")
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs-stage1", type=int, default=25)
    parser.add_argument("--epochs-stage2", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Running on {device}", flush=True)

    with zipfile.ZipFile(args.pairs_zip) as z:
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

    split_meta = json.loads(args.splits_json.read_text())
    val_id_to_idx = {r["id"]: i for i, r in enumerate(val)}
    q300_idx = np.array([val_id_to_idx[i] for i in split_meta["qlora_300_ids"]], dtype=np.int32)
    ah_idx = np.array([val_id_to_idx[i] for i in split_meta["author_holdout_66_ids"]], dtype=np.int32)
    ah_creators = {val[i]["attribution"] for i in ah_idx if val[i]["attribution"]}

    # Build country vocabulary over all splits
    all_ccs = sorted({r["country"] for r in gal + trn + val})
    cc_to_idx = {c: i for i, c in enumerate(all_ccs)}
    num_cc = len(all_ccs)

    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    g_cc_idx = np.array([cc_to_idx[c] for c in g_cc], dtype=np.int64)
    g_attr = np.array([r["attribution"] for r in gal])

    t_lat = np.array([r["latitude"] for r in trn], dtype=np.float64)
    t_lon = np.array([r["longitude"] for r in trn], dtype=np.float64)
    t_cc = np.array([r["country"] for r in trn])
    t_cc_idx = np.array([cc_to_idx[c] for c in t_cc], dtype=np.int64)
    t_attr = np.array([r["attribution"] for r in trn])

    v_lat = np.array([r["latitude"] for r in val], dtype=np.float64)
    v_lon = np.array([r["longitude"] for r in val], dtype=np.float64)
    v_cc = np.array([r["country"] for r in val])
    v_cc_idx = np.array([cc_to_idx[c] for c in v_cc], dtype=np.int64)
    v_attr = np.array([r["attribution"] for r in val])

    # Country frequency debiasing vector
    g_cc_counts = Counter(g_cc_idx.tolist())
    cc_freq_log = np.array([math.log(g_cc_counts.get(i, 1) / len(gal)) for i in range(num_cc)], dtype=np.float32)

    # STRICT CREATOR-DISJOINT SPLIT of train_12000 into opt_train (~85%) and tune (~15%),
    # explicitly excluding ALL author_holdout_66 creators from opt_train!
    eligible_mask = np.array([a not in ah_creators for a in t_attr], dtype=bool)
    eligible_creators = sorted({a for a, ok in zip(t_attr, eligible_mask) if ok})
    rng = np.random.default_rng(SEED)
    rng.shuffle(eligible_creators)
    n_tune_creators = max(1, int(0.15 * len(eligible_creators)))
    tune_creators = set(eligible_creators[:n_tune_creators])
    opt_idx = np.array([i for i, (a, ok) in enumerate(zip(t_attr, eligible_mask)) if ok and a not in tune_creators], dtype=np.int32)
    tune_idx = np.array([i for i, (a, ok) in enumerate(zip(t_attr, eligible_mask)) if ok and a in tune_creators], dtype=np.int32)
    opt_creators = {t_attr[i] for i in opt_idx if t_attr[i]}
    val_unseen_opt_idx = np.array([i for i, a in enumerate(v_attr) if a not in opt_creators], dtype=np.int32)

    print(f"Splits: opt_train={len(opt_idx)} ({len(opt_creators)} creators), "
          f"tune={len(tune_idx)} ({len(tune_creators)} disjoint creators), "
          f"excluded_ah_from_train={(~eligible_mask).sum()}, "
          f"val_unseen_opt={len(val_unseen_opt_idx)}/2000", flush=True)

    # Verify zero creator overlap between opt_train and tune, and opt_train and author_holdout_66
    assert len(opt_creators & tune_creators) == 0
    assert len(opt_creators & ah_creators) == 0

    print("Computing gallery local density...", flush=True)
    g_density = compute_gallery_local_density(g_lat, g_lon, sigma_km=200.0)

    # Move base tensors to GPU
    gf_t = torch.as_tensor(gf, device=device)
    tf_t = torch.as_tensor(tf, device=device)
    vf_t = torch.as_tensor(vf, device=device)
    g_cc_t = torch.as_tensor(g_cc_idx, device=device)
    g_cc_debias_w = torch.as_tensor(
        [1.0 / (max(1, g_cc_counts[c]) ** 0.25) for c in g_cc_idx], dtype=torch.float32, device=device
    )

    # ============================================================
    # STAGE 1: Full-Gallery Diff-Creator Geo-Metric Projection
    # ============================================================
    print("\n--- STAGE 1: Full-Gallery Diff-Creator Geo-Metric Projection ---", flush=True)
    proj = GeoProjector(dim=384, hidden=512, num_countries=num_cc, dropout=0.25).to(device)
    opt1 = torch.optim.AdamW(proj.parameters(), lr=1.5e-3, weight_decay=2e-2)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs_stage1)

    def eval_stage1_tune():
        proj.eval()
        with torch.inference_mode():
            g_emb = proj.encode(gf_t)
            q_emb = proj.encode(tf_t[tune_idx])
        top_i, _ = retrieve_diff_creator(q_emb, t_attr[tune_idx], g_emb, g_attr, k=100)
        d1 = haversine_vec_np(t_lat[tune_idx], t_lon[tune_idx], g_lat[top_i[:, 0]], g_lon[top_i[:, 0]])
        d100 = haversine_vec_np(
            t_lat[tune_idx][:, None], t_lon[tune_idx][:, None],
            g_lat[top_i], g_lon[top_i],
        )
        m = summarize(d1, g_cc[top_i[:, 0]], t_cc[tune_idx])
        m["top100_25km"] = float((d100.min(axis=1) <= 25.0).mean())
        m["top100_200km"] = float((d100.min(axis=1) <= 200.0).mean())
        m["top100_country"] = float(np.any(g_cc[top_i] == t_cc[tune_idx][:, None], axis=1).mean())
        return m

    init_m = eval_stage1_tune()
    print(f"Stage 1 Epoch 00 (frozen DINOv2): pts={init_m['mean_points']:.1f}, med={init_m['median_km']:.1f}km, "
          f"cc={100*init_m['country_rate']:.1f}%, <=25={100*init_m['hit_25_rate']:.2f}%, "
          f"Top100<=25={100*init_m['top100_25km']:.2f}%, Top100<=200={100*init_m['top100_200km']:.2f}%", flush=True)

    best_s1_pts = init_m["mean_points"]
    best_s1_state = copy.deepcopy(proj.state_dict())
    best_s1_epoch = 0
    s1_history = [dict(epoch=0, **init_m)]

    for epoch in range(1, args.epochs_stage1 + 1):
        proj.train()
        perm = rng.permutation(opt_idx)
        loss_sum = 0.0
        n_seen = 0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            b = len(batch_idx)
            # Compute exact geographic distance to all 25k gallery on CPU (fast: 256 x 25000 ~ 15ms)
            d_np = haversine_matrix_np(t_lat[batch_idx], t_lon[batch_idx], g_lat, g_lon)
            same_np = (t_attr[batch_idx][:, None] == g_attr[None, :]) & (t_attr[batch_idx][:, None] != "")
            d_t = torch.as_tensor(d_np, device=device)
            same_t = torch.as_tensor(same_np, device=device)
            q_cc_t = torch.as_tensor(t_cc_idx[batch_idx], device=device)

            # Multi-scale diff-creator soft geographic target
            diff_t = ~same_t
            same_cc_t = (q_cc_t[:, None] == g_cc_t[None, :])
            w = (
                4.0 * torch.exp(-d_t / 35.0)
                + 2.5 * torch.exp(-d_t / 180.0)
                + 1.0 * torch.exp(-d_t / 600.0)
                + 0.35 * same_cc_t.float()
            ) * diff_t.float() * g_cc_debias_w[None, :]
            w_sum = w.sum(dim=1, keepdim=True)
            valid = (w_sum.squeeze(1) > 1e-6)
            if not valid.any():
                continue
            target_p = w[valid] / w_sum[valid]

            q_in = tf_t[batch_idx[valid.cpu().numpy()]]
            q_emb = proj.encode(q_in)
            g_emb = proj.encode(gf_t)
            logits = proj.scale() * (q_emb @ g_emb.T)

            # Mask ONLY same-creator NEARBY (<=300km) so model gets ZERO shortcut reward,
            # while KEEPING same-creator FAR (>300km) as hard negatives to unlearn camera fingerprints!
            mask_near_same = same_t[valid] & (d_t[valid] <= 300.0)
            logits = logits.masked_fill(mask_near_same, -1e4)

            log_probs = F.log_softmax(logits, dim=-1)
            geo_loss = -(target_p * log_probs).sum(dim=-1).mean()
            cc_logits = proj.country_head(q_emb)
            cc_loss = F.cross_entropy(cc_logits, q_cc_t[valid], label_smoothing=0.05)
            loss = geo_loss + 0.30 * cc_loss

            opt1.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 2.0)
            opt1.step()
            loss_sum += float(loss.detach()) * int(valid.sum())
            n_seen += int(valid.sum())
        sched1.step()

        m = eval_stage1_tune()
        m["loss"] = loss_sum / max(1, n_seen)
        m["scale"] = float(proj.scale().detach())
        s1_history.append(dict(epoch=epoch, **m))
        print(f"Stage 1 Epoch {epoch:02d}: loss={m['loss']:.3f}, scale={m['scale']:.1f}, pts={m['mean_points']:.1f}, "
              f"med={m['median_km']:.1f}km, cc={100*m['country_rate']:.1f}%, <=25={100*m['hit_25_rate']:.2f}%, "
              f"Top100<=25={100*m['top100_25km']:.2f}%, Top100<=200={100*m['top100_200km']:.2f}%", flush=True)
        if m["mean_points"] > best_s1_pts:
            best_s1_pts = m["mean_points"]
            best_s1_state = copy.deepcopy(proj.state_dict())
            best_s1_epoch = epoch

    proj.load_state_dict(best_s1_state)
    proj.eval()
    print(f"Best Stage 1 epoch={best_s1_epoch}, tune mean_points={best_s1_pts:.1f}", flush=True)

    # Encode all splits with best GeoProjector & retrieve diff-creator Top-100
    with torch.inference_mode():
        g_proj = proj.encode(gf_t)
        t_proj = proj.encode(tf_t)
        v_proj = proj.encode(vf_t)
        t_aux_cc_lp = F.log_softmax(proj.country_head(t_proj), dim=-1).cpu().numpy()
        v_aux_cc_lp = F.log_softmax(proj.country_head(v_proj), dim=-1).cpu().numpy()

    opt_top, opt_gsim = retrieve_diff_creator(t_proj[opt_idx], t_attr[opt_idx], g_proj, g_attr, k=100)
    tune_top, tune_gsim = retrieve_diff_creator(t_proj[tune_idx], t_attr[tune_idx], g_proj, g_attr, k=100)
    val_top, val_gsim = retrieve_diff_creator(v_proj, v_attr, g_proj, g_attr, k=100)

    # Raw cosine for those exact Top-100 candidates
    def raw_cos_for_top(q_raw, top_i):
        out = np.empty(top_i.shape, dtype=np.float32)
        for s in range(0, len(q_raw), 500):
            e = min(len(q_raw), s + 500)
            out[s:e] = np.einsum("nd,nkd->nk", q_raw[s:e], gf[top_i[s:e]])
        return out

    opt_rsim = raw_cos_for_top(tf[opt_idx], opt_top)
    tune_rsim = raw_cos_for_top(tf[tune_idx], tune_top)
    val_rsim = raw_cos_for_top(vf, val_top)

    print("Building Stage 2 candidate features (multi-scale KDE + density + country consensus)...", flush=True)
    opt_feats, _ = build_candidate_features(
        opt_top, opt_gsim, opt_rsim, t_aux_cc_lp[opt_idx], g_lat, g_lon, g_cc_idx, g_density, cc_freq_log
    )
    tune_feats, tune_kde = build_candidate_features(
        tune_top, tune_gsim, tune_rsim, t_aux_cc_lp[tune_idx], g_lat, g_lon, g_cc_idx, g_density, cc_freq_log
    )
    val_feats, val_kde = build_candidate_features(
        val_top, val_gsim, val_rsim, v_aux_cc_lp, g_lat, g_lon, g_cc_idx, g_density, cc_freq_log
    )

    # ============================================================
    # STAGE 2: Stabilized Listwise Geo-Ranker over Re-Retrieved Top-100
    # ============================================================
    print("\n--- STAGE 2: Listwise Geo-Ranker over Projected Top-100 ---", flush=True)
    opt_d100 = haversine_vec_np(t_lat[opt_idx][:, None], t_lon[opt_idx][:, None], g_lat[opt_top], g_lon[opt_top])
    opt_same_cc = (g_cc_idx[opt_top] == t_cc_idx[opt_idx][:, None]).astype(np.float32)
    opt_w = (
        4.0 * np.exp(-opt_d100 / 35.0)
        + 2.5 * np.exp(-opt_d100 / 180.0)
        + 1.0 * np.exp(-opt_d100 / 600.0)
        + 0.3 * opt_same_cc
    ).astype(np.float32)
    opt_valid = opt_w.sum(axis=1) > 1e-6
    opt_target = (opt_w[opt_valid] / opt_w[opt_valid].sum(axis=1, keepdims=True)).astype(np.float32)

    opt_gsim_t = torch.as_tensor(opt_gsim[opt_valid], device=device)
    opt_feats_t = torch.as_tensor(opt_feats[opt_valid], device=device)
    opt_target_t = torch.as_tensor(opt_target, device=device)

    tune_gsim_t = torch.as_tensor(tune_gsim, device=device)
    tune_feats_t = torch.as_tensor(tune_feats, device=device)
    val_gsim_t = torch.as_tensor(val_gsim, device=device)
    val_feats_t = torch.as_tensor(val_feats, device=device)

    ranker = ListwiseGeoRanker(feat_dim=opt_feats.shape[-1], hidden=64, dropout=0.2).to(device)
    opt2 = torch.optim.AdamW(ranker.parameters(), lr=1e-3, weight_decay=2e-2)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs_stage2)

    def eval_stage2_tune():
        ranker.eval()
        with torch.inference_mode():
            logits = ranker(tune_gsim_t, tune_feats_t).cpu().numpy()
        pick = logits.argmax(axis=1)
        g_pick = tune_top[np.arange(len(tune_idx)), pick]
        d = haversine_vec_np(t_lat[tune_idx], t_lon[tune_idx], g_lat[g_pick], g_lon[g_pick])
        return summarize(d, g_cc[g_pick], t_cc[tune_idx])

    init_s2 = eval_stage2_tune()
    print(f"Stage 2 Epoch 00 (Stage 1 Top-1): pts={init_s2['mean_points']:.1f}, med={init_s2['median_km']:.1f}km, "
          f"cc={100*init_s2['country_rate']:.1f}%, <=25={100*init_s2['hit_25_rate']:.2f}%, <=200={100*init_s2['hit_200_rate']:.2f}%", flush=True)

    best_s2_pts = init_s2["mean_points"]
    best_s2_state = copy.deepcopy(ranker.state_dict())
    best_s2_epoch = 0
    s2_history = [dict(epoch=0, **init_s2)]

    n_opt2 = len(opt_target_t)
    for epoch in range(1, args.epochs_stage2 + 1):
        ranker.train()
        perm = torch.randperm(n_opt2, device=device)
        loss_sum = 0.0
        for batch in perm.split(args.batch_size):
            logits = ranker(opt_gsim_t[batch], opt_feats_t[batch])
            log_p = F.log_softmax(logits, dim=-1)
            loss = -(opt_target_t[batch] * log_p).sum(dim=-1).mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 2.0)
            opt2.step()
            loss_sum += float(loss.detach()) * len(batch)
        sched2.step()
        m = eval_stage2_tune()
        m["loss"] = loss_sum / n_opt2
        s2_history.append(dict(epoch=epoch, **m))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs_stage2:
            print(f"Stage 2 Epoch {epoch:02d}: loss={m['loss']:.3f}, pts={m['mean_points']:.1f}, med={m['median_km']:.1f}km, "
                  f"cc={100*m['country_rate']:.1f}%, <=25={100*m['hit_25_rate']:.2f}%, <=200={100*m['hit_200_rate']:.2f}%", flush=True)
        if m["mean_points"] > best_s2_pts:
            best_s2_pts = m["mean_points"]
            best_s2_state = copy.deepcopy(ranker.state_dict())
            best_s2_epoch = epoch

    ranker.load_state_dict(best_s2_state)
    ranker.eval()
    with torch.inference_mode():
        val_ranker_logits = ranker(val_gsim_t, val_feats_t).cpu().numpy()

    # Raw DINOv2 diff-creator baseline on val_2000 for exact side-by-side comparison
    val_raw_top, _ = retrieve_diff_creator(vf_t, v_attr, gf_t, g_attr, k=100)

    # Evaluate across all 4 validation views
    subsets = {
        "val_2000": np.arange(len(val), dtype=np.int32),
        "val_unseen_opt_creators": val_unseen_opt_idx,
        "qlora_eval_300": q300_idx,
        "author_holdout_66": ah_idx,
    }
    report = {
        "seed": SEED,
        "best_stage1_epoch": best_s1_epoch,
        "best_stage2_epoch": best_s2_epoch,
        "opt_train_n": int(len(opt_idx)),
        "tune_n": int(len(tune_idx)),
        "val_unseen_opt_n": int(len(val_unseen_opt_idx)),
        "stage1_history": s1_history,
        "stage2_history": s2_history,
        "splits": {},
    }

    for name, idxs in subsets.items():
        # 1. Raw frozen DINOv2 diff-creator
        rg = val_raw_top[idxs, 0]
        raw_d1 = haversine_vec_np(v_lat[idxs], v_lon[idxs], g_lat[rg], g_lon[rg])
        raw_d100 = haversine_vec_np(v_lat[idxs][:, None], v_lon[idxs][:, None], g_lat[val_raw_top[idxs]], g_lon[val_raw_top[idxs]])
        raw_m = summarize(raw_d1, g_cc[rg], v_cc[idxs])
        raw_m["top100_25km"] = float((raw_d100.min(axis=1) <= 25.0).mean())
        raw_m["top100_200km"] = float((raw_d100.min(axis=1) <= 200.0).mean())
        raw_m["top100_country"] = float(np.any(g_cc[val_raw_top[idxs]] == v_cc[idxs][:, None], axis=1).mean())

        # 2. Stage 1 Geo-Projected Top-1 + Top-100 recall
        pg = val_top[idxs, 0]
        p_d1 = haversine_vec_np(v_lat[idxs], v_lon[idxs], g_lat[pg], g_lon[pg])
        p_d100 = haversine_vec_np(v_lat[idxs][:, None], v_lon[idxs][:, None], g_lat[val_top[idxs]], g_lon[val_top[idxs]])
        s1_m = summarize(p_d1, g_cc[pg], v_cc[idxs])
        s1_m["top100_25km"] = float((p_d100.min(axis=1) <= 25.0).mean())
        s1_m["top100_200km"] = float((p_d100.min(axis=1) <= 200.0).mean())
        s1_m["top100_country"] = float(np.any(g_cc[val_top[idxs]] == v_cc[idxs][:, None], axis=1).mean())

        # 3. Stage 1 + KDE (sigma=600km)
        kp = val_kde[idxs].argmax(axis=1)
        kg = val_top[idxs][np.arange(len(idxs)), kp]
        kde_m = summarize(haversine_vec_np(v_lat[idxs], v_lon[idxs], g_lat[kg], g_lon[kg]), g_cc[kg], v_cc[idxs])

        # 4. Stage 1 + Stage 2 Listwise Ranker
        rp = val_ranker_logits[idxs].argmax(axis=1)
        rk_g = val_top[idxs][np.arange(len(idxs)), rp]
        s2_m = summarize(haversine_vec_np(v_lat[idxs], v_lon[idxs], g_lat[rk_g], g_lon[rk_g]), g_cc[rk_g], v_cc[idxs])

        report["splits"][name] = {
            "raw_dinov2_diff": raw_m,
            "stage1_geoproj_top1": s1_m,
            "stage1_geoproj_kde": kde_m,
            "stage2_listwise_ranker": s2_m,
        }
        print(f"\n=== {name} (n={len(idxs)}, STRICT DIFF-CREATOR) ===", flush=True)
        for lbl, mm in [
            ("Raw DINOv2 Top-1   ", raw_m),
            ("Stage1 GeoProj Top1", s1_m),
            ("Stage1 GeoProj+KDE ", kde_m),
            ("Stage2 Listwise    ", s2_m),
        ]:
            extra = f" | Top100 <=25={100*mm['top100_25km']:.2f}%, <=200={100*mm['top100_200km']:.1f}%, cc={100*mm['top100_country']:.1f}%" if "top100_25km" in mm else ""
            print(f"  {lbl}: <=25={mm['hit_25']:3d} ({100*mm['hit_25_rate']:5.2f}%), <=200={mm['hit_200']:3d} ({100*mm['hit_200_rate']:5.2f}%), "
                  f"cc={mm['country_correct']:3d} ({100*mm['country_rate']:5.1f}%), med={mm['median_km']:6.1f}km, pts={mm['mean_points']:6.1f}{extra}", flush=True)

    (args.output / "step1b_ranker_results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "val_projected_top100.npz",
        val_ids=np.array([r["id"] for r in val]),
        top_idx=val_top,
        geo_sim=val_gsim,
        raw_sim=val_rsim,
        kde_score=val_kde,
        ranker_logits=val_ranker_logits,
        aux_cc_logp=v_aux_cc_lp,
        country_vocab=np.array(all_ccs),
    )
    torch.save({"projector": best_s1_state, "ranker": best_s2_state, "country_vocab": all_ccs}, args.output / "geo_ranker_v2.pt")
    print(f"\nSaved Step 1B outputs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
