"""Phase B: Stream-encode all 39,000 (25k gallery + 12k train + 2k val) + 300 official_test images
with geolocal/StreetCLIP (reusing 6,366 already-encoded vectors, 1 HTTP request per shard -> 0 429s),
then train leak-free 25k Diff-Creator Geo-Metric Projection + Listwise Geo-Ranker.
"""
import argparse
import copy
import hashlib
import io
import json
import math
import os
import queue
import threading
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

SEED = 7818
STREETCLIP = "geolocal/StreetCLIP"
OSV = "https://huggingface.co/datasets/osv5m/osv5m/resolve/cff33609b56b54d8743b7ee7a416eb8433e9a681"


def clean_pil_from_bytes(payload):
    with Image.open(io.BytesIO(payload)) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


def crop_bottom(image, keep_ratio=0.88):
    w, h = image.size
    return image.crop((0, 0, w, max(32, int(round(h * keep_ratio)))))


@torch.inference_mode()
def encode_pil_list(model, processor, images, batch_size=128):
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feats = model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        feats = F.normalize(feats.float(), dim=-1)
        chunks.append(feats.cpu().numpy().astype(np.float16))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 768), dtype=np.float16)


@torch.inference_mode()
def encode_country_texts(model, processor, country_names):
    def enc(prompts):
        inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feats = model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        return F.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)

    single = enc([f"A Street View photo in {n}" for n in country_names])
    t2 = enc([f"A street-level photo in {n}" for n in country_names])
    t3 = enc([f"A road in {n}" for n in country_names])
    ens = single + t2 + t3
    ens /= np.linalg.norm(ens, axis=-1, keepdims=True)
    return single, ens


def download_and_extract_shard(slot, wanted_rows, tmp_dir):
    """Download 1 whole OSV-5M train shard (1 HTTP request), extract only wanted JPEGs, verify SHA-256, delete ZIP."""
    url = f"{OSV}/images/train/{slot:02}.zip"
    zpath = tmp_dir / f"shard_{slot:02}.zip"
    wanted_by_member = {r["archive_member"]: r for r in wanted_rows}
    t0 = time.monotonic()
    last_err = None
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as resp:
                resp.raise_for_status()
                with zpath.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            break
        except Exception as exc:
            last_err = exc
            if zpath.exists():
                zpath.unlink()
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed downloading shard {slot:02}: {last_err}")

    dl_s = time.monotonic() - t0
    t1 = time.monotonic()
    ids = []
    pil_imgs = []
    try:
        with zipfile.ZipFile(zpath) as z:
            for member, row in wanted_by_member.items():
                payload = z.read(member)
                if row.get("sha256") and hashlib.sha256(payload).hexdigest() != row["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {row['id']} in shard {slot:02}")
                ids.append(row["id"])
                pil_imgs.append(clean_pil_from_bytes(payload))
    finally:
        if zpath.exists():
            zpath.unlink()
    ext_s = time.monotonic() - t1
    return slot, ids, pil_imgs, dl_s, ext_s


def haversine_matrix_np(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)


def haversine_vec_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
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
    def __init__(self, dim=768, hidden=768, num_countries=220, dropout=0.25):
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
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.country_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_countries),
        )

    def encode(self, x):
        z = self.linear_skip(x) + self.net(x)
        return F.normalize(z, dim=-1)

    def scale(self):
        return self.log_scale.clamp(math.log(5.0), math.log(55.0)).exp()


class ListwiseGeoRanker(nn.Module):
    def __init__(self, feat_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.base_scale = nn.Parameter(torch.tensor(15.0))
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
        delta = self.mlp(feats).squeeze(-1)
        return self.base_scale.clamp(1.0, 50.0) * geo_sim + 8.0 * torch.tanh(delta / 8.0)


@torch.inference_mode()
def retrieve_diff_creator(q_emb, q_attr, g_emb, g_attr, k=100, chunk=512):
    device = q_emb.device
    n = len(q_emb)
    top_idx = np.empty((n, k), dtype=np.int32)
    top_sim = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        s = q_emb[start:end] @ g_emb.T
        qa = q_attr[start:end][:, None]
        same = (qa == g_attr[None, :]) & (qa != "")
        s = s.masked_fill(torch.as_tensor(same, device=device), -2.0)
        vals, idxs = torch.topk(s, k=k, dim=1)
        top_idx[start:end] = idxs.cpu().numpy().astype(np.int32)
        top_sim[start:end] = vals.cpu().numpy().astype(np.float32)
    return top_idx, top_sim


def build_candidate_features(top_idx, geo_sim, raw_sim, aux_cc_logp, text_cc_logp,
                             g_lat, g_lon, g_cc_idx, g_density, cc_freq_log):
    n, k = top_idx.shape
    lats = np.radians(g_lat[top_idx])
    lons = np.radians(g_lon[top_idx])
    feats = np.zeros((n, k, 24), dtype=np.float32)
    kde_scores = np.zeros((n, k), dtype=np.float32)
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
        la, lo = lats[start:end], lons[start:end]
        dlat = la[:, :, None] - la[:, None, :]
        dlon = lo[:, :, None] - lo[:, None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin(dlon / 2) ** 2
        pw_d = (6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))).astype(np.float32)

        dens = g_density[c_idx]
        inv_dens_25 = dens ** -0.25
        cc_i = g_cc_idx[c_idx]
        same_cc = (cc_i[:, :, None] == cc_i[:, None, :]).astype(np.float32)

        w03 = np.exp((gs - gs[:, :1]) / 0.03)
        w03 /= w03.sum(axis=1, keepdims=True)
        w06 = np.exp((gs - gs[:, :1]) / 0.06)
        w06 /= w06.sum(axis=1, keepdims=True)
        w06_deb = w06 * inv_dens_25
        w06_deb /= w06_deb.sum(axis=1, keepdims=True)

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

        for sigma in (50.0, 200.0, 500.0, 1200.0):
            K = np.exp(-pw_d / sigma)
            feats[start:end, :, col] = norm_row(np.einsum("nj,nij->ni", w03, K)); col += 1
            feats[start:end, :, col] = norm_row(np.einsum("nj,nij->ni", w06_deb, K)); col += 1
            if sigma == 500.0:
                kde_scores[start:end] = np.einsum("nj,nij->ni", w03, K)

        feats[start:end, :, col] = norm_row(np.einsum("nj,nij->ni", w03, same_cc)); col += 1
        feats[start:end, :, col] = norm_row(np.einsum("nj,nij->ni", w06, same_cc)); col += 1
        feats[start:end, :, col] = np.log1p(dens); col += 1
        feats[start:end, :, col] = cc_freq_log[cc_i]; col += 1
        rows_b = np.arange(b)[:, None]
        c_aux = aux_cc_logp[start:end][rows_b, cc_i]
        feats[start:end, :, col] = c_aux; col += 1
        feats[start:end, :, col] = norm_row(c_aux); col += 1
        c_txt = text_cc_logp[start:end][rows_b, cc_i]
        feats[start:end, :, col] = c_txt; col += 1
        feats[start:end, :, col] = norm_row(c_txt); col += 1
        feats[start:end, :, col] = np.log1p((pw_d <= 200.0).sum(axis=2)); col += 1
        assert col == 24, f"Expected 24 features, got {col}"

    return feats, kde_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output / "tmp_shards"
    tmp_dir.mkdir(exist_ok=True)

    # Load metadata manifests
    with zipfile.ZipFile(args.input_dir / "manifests_39k.zip") as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        trn = [json.loads(l) for l in z.read("train_images.jsonl").splitlines()]
        val = [json.loads(l) for l in z.read("validation_images.jsonl").splitlines()]

    with zipfile.ZipFile(args.input_dir / "official-test-300.zip") as z:
        test_rows = [json.loads(l) for l in z.read("test.jsonl").splitlines()]
        test_pils_full = []
        test_pils_crop = []
        for r in test_rows:
            payload = z.read(r["image"])
            if hashlib.sha256(payload).hexdigest() != r["sha256"]:
                raise ValueError(f"Test JPEG SHA mismatch: {r['id']}")
            im = clean_pil_from_bytes(payload)
            test_pils_full.append(im)
            test_pils_crop.append(crop_bottom(im, 0.88))

    split_meta = json.loads((args.input_dir / "splits.json").read_text())
    cc_names = json.loads((args.input_dir / "country_names.json").read_text())

    # Load existing 6,366 StreetCLIP vectors to skip re-downloading/encoding them
    existing_sc = np.load(args.input_dir / "streetclip_6k_seed.npz")
    emb_map = {str(i): existing_sc["feats_full"][k].astype(np.float16) for k, i in enumerate(existing_sc["ids"])}
    print(f"Seeded {len(emb_map)} already-encoded StreetCLIP vectors.", flush=True)

    # Load StreetCLIP model on GPU
    print(f"Loading {STREETCLIP} on {torch.cuda.get_device_name(0)}...", flush=True)
    processor = CLIPProcessor.from_pretrained(STREETCLIP)
    model = CLIPModel.from_pretrained(STREETCLIP).to(device).eval()

    # Encode official_test_300 (full & crop88) + country text probes
    test_f_full = encode_pil_list(model, processor, test_pils_full, batch_size=args.batch_size)
    test_f_crop = encode_pil_list(model, processor, test_pils_crop, batch_size=args.batch_size)
    del test_pils_full, test_pils_crop
    print(f"Encoded official_test_300: full={test_f_full.shape}, crop88={test_f_crop.shape}", flush=True)

    all_ccs = sorted({r["country"] for r in gal + trn + val + test_rows})
    cc_to_idx = {c: i for i, c in enumerate(all_ccs)}
    names_ordered = [cc_names.get(c, c) for c in all_ccs]
    text_single, text_ens = encode_country_texts(model, processor, names_ordered)

    # Group remaining unencoded 39k rows by train archive shard
    all_39k = gal + trn + val
    by_shard = defaultdict(list)
    for r in all_39k:
        if r["id"] not in emb_map:
            slot = int(r["archive_member"].split("/", 1)[0])
            by_shard[slot].append(r)

    shards_todo = sorted(by_shard.keys())
    total_todo = sum(len(by_shard[s]) for s in shards_todo)
    print(f"Streaming {total_todo:,} remaining images across {len(shards_todo)} shards (2-thread prefetch)...", flush=True)

    # 2-thread download prefetch queue (maxsize=2 keeps disk <= ~6 GB!)
    slot_q = queue.Queue()
    for s in shards_todo:
        slot_q.put(s)
    ready_q = queue.Queue(maxsize=2)
    stop_err = []

    def worker():
        while not stop_err:
            try:
                slot = slot_q.get_nowait()
            except queue.Empty:
                return
            try:
                res = download_and_extract_shard(slot, by_shard[slot], tmp_dir)
                ready_q.put(res)
            except Exception as exc:
                stop_err.append(exc)
                ready_q.put(None)
                return

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()

    done_imgs = 0
    t_stream0 = time.monotonic()
    for done_shards in range(1, len(shards_todo) + 1):
        item = ready_q.get()
        if item is None or stop_err:
            raise RuntimeError(f"Shard worker failed: {stop_err[0] if stop_err else 'unknown'}")
        slot, ids, pils, dl_s, ext_s = item
        te0 = time.monotonic()
        vecs = encode_pil_list(model, processor, pils, batch_size=args.batch_size)
        enc_s = time.monotonic() - te0
        for rid, v in zip(ids, vecs):
            emb_map[rid] = v
        done_imgs += len(ids)
        if done_shards == 1 or done_shards % 10 == 0 or done_shards == len(shards_todo):
            elapsed = time.monotonic() - t_stream0
            rate = done_imgs / max(1e-3, elapsed)
            eta_m = (total_todo - done_imgs) / max(1e-3, rate) / 60.0
            print(f"Shards {done_shards:2d}/{len(shards_todo)} | imgs {done_imgs:5d}/{total_todo} | "
                  f"last shard {slot:02} (dl={dl_s:.1f}s ext={ext_s:.1f}s enc={enc_s:.1f}s) | "
                  f"elapsed={elapsed/60:.1f}m ETA={eta_m:.1f}m", flush=True)

    for t in threads:
        t.join()

    # Free CLIP model before Stage 1/2 training
    del model, processor
    torch.cuda.empty_cache()

    # Assemble ordered feature matrices
    gf_fp16 = np.stack([emb_map[r["id"]] for r in gal])
    tf_fp16 = np.stack([emb_map[r["id"]] for r in trn])
    vf_fp16 = np.stack([emb_map[r["id"]] for r in val])
    feat_npz = args.output / "streetclip_39k_plus_test300.npz"
    np.savez_compressed(
        feat_npz,
        gallery_ids=np.array([r["id"] for r in gal]),
        train_ids=np.array([r["id"] for r in trn]),
        val_ids=np.array([r["id"] for r in val]),
        test_ids=np.array([r["id"] for r in test_rows]),
        gallery_features=gf_fp16,
        train_features=tf_fp16,
        val_features=vf_fp16,
        test_features_full=test_f_full,
        test_features_crop88=test_f_crop,
        country_codes=np.array(all_ccs),
        text_feats_single=text_single,
        text_feats_ens=text_ens,
    )
    print(f"Saved full 39k+300 StreetCLIP embeddings: {feat_npz} ({feat_npz.stat().st_size / 1e6:.2f} MB)", flush=True)

    # ============================================================
    # Train Leak-Free 25k StreetCLIP Geo-Projection + Listwise Ranker
    # ============================================================
    gf = gf_fp16.astype(np.float32); gf /= np.linalg.norm(gf, axis=-1, keepdims=True)
    tf = tf_fp16.astype(np.float32); tf /= np.linalg.norm(tf, axis=-1, keepdims=True)
    vf = vf_fp16.astype(np.float32); vf /= np.linalg.norm(vf, axis=-1, keepdims=True)
    tsf = test_f_full.astype(np.float32); tsf /= np.linalg.norm(tsf, axis=-1, keepdims=True)

    val_id_to_idx = {r["id"]: i for i, r in enumerate(val)}
    q300_idx = np.array([val_id_to_idx[i] for i in split_meta["qlora_300_ids"]], dtype=np.int32)
    ah_idx = np.array([val_id_to_idx[i] for i in split_meta["author_holdout_66_ids"]], dtype=np.int32)
    ah_creators = {val[i]["attribution"] for i in ah_idx if val[i]["attribution"]}

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
    v_attr = np.array([r["attribution"] for r in val])

    ts_lat = np.array([r["latitude"] for r in test_rows], dtype=np.float64)
    ts_lon = np.array([r["longitude"] for r in test_rows], dtype=np.float64)
    ts_cc = np.array([r["country"] for r in test_rows])
    ts_attr = np.array([r["attribution"] for r in test_rows])

    # Precompute StreetCLIP zero-shot text-probe country log-probs (using 30.0 scale)
    def text_logp(emb_np):
        logits = (emb_np @ text_ens.T) * 30.0
        mx = logits.max(axis=1, keepdims=True)
        return ((logits - mx) - np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))).astype(np.float32)

    t_txt_lp = text_logp(tf)
    v_txt_lp = text_logp(vf)
    ts_txt_lp = text_logp(tsf)

    g_cc_counts = Counter(g_cc_idx.tolist())
    cc_freq_log = np.array([math.log(g_cc_counts.get(i, 1) / len(gal)) for i in range(num_cc)], dtype=np.float32)

    # Same strict creator-disjoint split as Step 1B!
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
    assert len(opt_creators & tune_creators) == 0 and len(opt_creators & ah_creators) == 0

    g_density = compute_gallery_local_density(g_lat, g_lon, sigma_km=200.0)

    gf_t = torch.as_tensor(gf, device=device)
    tf_t = torch.as_tensor(tf, device=device)
    vf_t = torch.as_tensor(vf, device=device)
    tsf_t = torch.as_tensor(tsf, device=device)
    g_cc_t = torch.as_tensor(g_cc_idx, device=device)
    g_cc_debias_w = torch.as_tensor([1.0 / (max(1, g_cc_counts[c]) ** 0.25) for c in g_cc_idx], dtype=torch.float32, device=device)

    print("\n--- STAGE 1: Full-Gallery 25k StreetCLIP Geo-Metric Projection ---", flush=True)
    proj = GeoProjector(dim=768, hidden=768, num_countries=num_cc, dropout=0.30).to(device)
    opt1 = torch.optim.AdamW(proj.parameters(), lr=5e-4, weight_decay=5e-2)
    epochs_s1 = 20
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_s1)

    def eval_s1_tune():
        proj.eval()
        with torch.inference_mode():
            g_emb = proj.encode(gf_t)
            q_emb = proj.encode(tf_t[tune_idx])
        top_i, _ = retrieve_diff_creator(q_emb, t_attr[tune_idx], g_emb, g_attr, k=100)
        d1 = haversine_vec_np(t_lat[tune_idx], t_lon[tune_idx], g_lat[top_i[:, 0]], g_lon[top_i[:, 0]])
        d100 = haversine_vec_np(t_lat[tune_idx][:, None], t_lon[tune_idx][:, None], g_lat[top_i], g_lon[top_i])
        m = summarize(d1, g_cc[top_i[:, 0]], t_cc[tune_idx])
        m["top100_25km"] = float((d100.min(axis=1) <= 25.0).mean())
        m["top100_200km"] = float((d100.min(axis=1) <= 200.0).mean())
        m["top100_country"] = float(np.any(g_cc[top_i] == t_cc[tune_idx][:, None], axis=1).mean())
        return m

    init_m = eval_s1_tune()
    print(f"Stage 1 Epoch 00 (frozen StreetCLIP): pts={init_m['mean_points']:.1f}, med={init_m['median_km']:.1f}km, "
          f"cc={100*init_m['country_rate']:.1f}%, <=25={100*init_m['hit_25_rate']:.2f}%, "
          f"Top100<=25={100*init_m['top100_25km']:.2f}%, Top100<=200={100*init_m['top100_200km']:.1f}%", flush=True)

    best_s1_pts = init_m["mean_points"]
    best_s1_state = copy.deepcopy(proj.state_dict())
    best_s1_epoch = 0
    s1_hist = [dict(epoch=0, **init_m)]

    for epoch in range(1, epochs_s1 + 1):
        proj.train()
        perm = rng.permutation(opt_idx)
        loss_sum, n_seen = 0.0, 0
        for start in range(0, len(perm), 256):
            batch_idx = perm[start:start + 256]
            d_np = haversine_matrix_np(t_lat[batch_idx], t_lon[batch_idx], g_lat, g_lon)
            same_np = (t_attr[batch_idx][:, None] == g_attr[None, :]) & (t_attr[batch_idx][:, None] != "")
            d_t = torch.as_tensor(d_np, device=device)
            same_t = torch.as_tensor(same_np, device=device)
            q_cc_t = torch.as_tensor(t_cc_idx[batch_idx], device=device)

            w = (
                4.0 * torch.exp(-d_t / 35.0)
                + 2.5 * torch.exp(-d_t / 180.0)
                + 1.0 * torch.exp(-d_t / 600.0)
                + 0.35 * (q_cc_t[:, None] == g_cc_t[None, :]).float()
            ) * (~same_t).float() * g_cc_debias_w[None, :]
            w_sum = w.sum(dim=1, keepdim=True)
            valid = w_sum.squeeze(1) > 1e-6
            if not valid.any():
                continue
            target_p = w[valid] / w_sum[valid]
            q_emb = proj.encode(tf_t[batch_idx[valid.cpu().numpy()]])
            g_emb = proj.encode(gf_t)
            logits = proj.scale() * (q_emb @ g_emb.T)
            logits = logits.masked_fill(same_t[valid] & (d_t[valid] <= 300.0), -1e4)
            geo_loss = -(target_p * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            cc_loss = F.cross_entropy(proj.country_head(q_emb), q_cc_t[valid], label_smoothing=0.05)
            loss = geo_loss + 0.30 * cc_loss
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 2.0)
            opt1.step()
            loss_sum += float(loss.detach()) * int(valid.sum())
            n_seen += int(valid.sum())
        sched1.step()
        m = eval_s1_tune()
        m["loss"] = loss_sum / max(1, n_seen)
        m["scale"] = float(proj.scale().detach())
        s1_hist.append(dict(epoch=epoch, **m))
        print(f"Stage 1 Epoch {epoch:02d}: loss={m['loss']:.3f}, scale={m['scale']:.1f}, pts={m['mean_points']:.1f}, "
              f"med={m['median_km']:.1f}km, cc={100*m['country_rate']:.1f}%, <=25={100*m['hit_25_rate']:.2f}%, "
              f"Top100<=25={100*m['top100_25km']:.2f}%, Top100<=200={100*m['top100_200km']:.1f}%", flush=True)
        if m["mean_points"] > best_s1_pts:
            best_s1_pts = m["mean_points"]
            best_s1_state = copy.deepcopy(proj.state_dict())
            best_s1_epoch = epoch

    proj.load_state_dict(best_s1_state)
    proj.eval()
    print(f"Best Stage 1 epoch={best_s1_epoch}, tune pts={best_s1_pts:.1f}", flush=True)

    with torch.inference_mode():
        g_proj = proj.encode(gf_t)
        t_proj = proj.encode(tf_t)
        v_proj = proj.encode(vf_t)
        ts_proj = proj.encode(tsf_t)
        t_aux_lp = F.log_softmax(proj.country_head(t_proj), dim=-1).cpu().numpy()
        v_aux_lp = F.log_softmax(proj.country_head(v_proj), dim=-1).cpu().numpy()
        ts_aux_lp = F.log_softmax(proj.country_head(ts_proj), dim=-1).cpu().numpy()

    opt_top, opt_gsim = retrieve_diff_creator(t_proj[opt_idx], t_attr[opt_idx], g_proj, g_attr, k=100)
    tune_top, tune_gsim = retrieve_diff_creator(t_proj[tune_idx], t_attr[tune_idx], g_proj, g_attr, k=100)
    val_top, val_gsim = retrieve_diff_creator(v_proj, v_attr, g_proj, g_attr, k=100)
    ts_top, ts_gsim = retrieve_diff_creator(ts_proj, ts_attr, g_proj, g_attr, k=100)

    def raw_cos(q_raw, top_i):
        out = np.empty(top_i.shape, dtype=np.float32)
        for s in range(0, len(q_raw), 500):
            e = min(len(q_raw), s + 500)
            out[s:e] = np.einsum("nd,nkd->nk", q_raw[s:e], gf[top_i[s:e]])
        return out

    opt_feats, _ = build_candidate_features(opt_top, opt_gsim, raw_cos(tf[opt_idx], opt_top), t_aux_lp[opt_idx], t_txt_lp[opt_idx], g_lat, g_lon, g_cc_idx, g_density, cc_freq_log)
    tune_feats, tune_kde = build_candidate_features(tune_top, tune_gsim, raw_cos(tf[tune_idx], tune_top), t_aux_lp[tune_idx], t_txt_lp[tune_idx], g_lat, g_lon, g_cc_idx, g_density, cc_freq_log)
    val_feats, val_kde = build_candidate_features(val_top, val_gsim, raw_cos(vf, val_top), v_aux_lp, v_txt_lp, g_lat, g_lon, g_cc_idx, g_density, cc_freq_log)
    ts_feats, ts_kde = build_candidate_features(ts_top, ts_gsim, raw_cos(tsf, ts_top), ts_aux_lp, ts_txt_lp, g_lat, g_lon, g_cc_idx, g_density, cc_freq_log)

    print("\n--- STAGE 2: Listwise Geo-Ranker over 25k StreetCLIP Projected Top-100 ---", flush=True)
    opt_d100 = haversine_vec_np(t_lat[opt_idx][:, None], t_lon[opt_idx][:, None], g_lat[opt_top], g_lon[opt_top])
    opt_w = (4.0 * np.exp(-opt_d100 / 35.0) + 2.5 * np.exp(-opt_d100 / 180.0) + 1.0 * np.exp(-opt_d100 / 600.0)
             + 0.3 * (g_cc_idx[opt_top] == t_cc_idx[opt_idx][:, None]).astype(np.float32)).astype(np.float32)
    opt_valid = opt_w.sum(axis=1) > 1e-6
    opt_target_t = torch.as_tensor(opt_w[opt_valid] / opt_w[opt_valid].sum(axis=1, keepdims=True), device=device)
    opt_gsim_t = torch.as_tensor(opt_gsim[opt_valid], device=device)
    opt_feats_t = torch.as_tensor(opt_feats[opt_valid], device=device)

    ranker = ListwiseGeoRanker(feat_dim=opt_feats.shape[-1], hidden=64, dropout=0.2).to(device)
    opt2 = torch.optim.AdamW(ranker.parameters(), lr=1e-3, weight_decay=2e-2)
    epochs_s2 = 25
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_s2)

    tune_gsim_t = torch.as_tensor(tune_gsim, device=device)
    tune_feats_t = torch.as_tensor(tune_feats, device=device)

    def eval_s2_tune():
        ranker.eval()
        with torch.inference_mode():
            logits = ranker(tune_gsim_t, tune_feats_t).cpu().numpy()
        pick = logits.argmax(axis=1)
        gp = tune_top[np.arange(len(tune_idx)), pick]
        return summarize(haversine_vec_np(t_lat[tune_idx], t_lon[tune_idx], g_lat[gp], g_lon[gp]), g_cc[gp], t_cc[tune_idx])

    init_s2 = eval_s2_tune()
    best_s2_pts = init_s2["mean_points"]
    best_s2_state = copy.deepcopy(ranker.state_dict())
    best_s2_epoch = 0
    s2_hist = [dict(epoch=0, **init_s2)]
    print(f"Stage 2 Epoch 00: pts={init_s2['mean_points']:.1f}, med={init_s2['median_km']:.1f}km, cc={100*init_s2['country_rate']:.1f}%, <=25={100*init_s2['hit_25_rate']:.2f}%", flush=True)

    for epoch in range(1, epochs_s2 + 1):
        ranker.train()
        perm = torch.randperm(len(opt_target_t), device=device)
        loss_sum = 0.0
        for batch in perm.split(256):
            logits = ranker(opt_gsim_t[batch], opt_feats_t[batch])
            loss = -(opt_target_t[batch] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 2.0)
            opt2.step()
            loss_sum += float(loss.detach()) * len(batch)
        sched2.step()
        m = eval_s2_tune()
        m["loss"] = loss_sum / len(opt_target_t)
        s2_hist.append(dict(epoch=epoch, **m))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs_s2:
            print(f"Stage 2 Epoch {epoch:02d}: loss={m['loss']:.3f}, pts={m['mean_points']:.1f}, med={m['median_km']:.1f}km, "
                  f"cc={100*m['country_rate']:.1f}%, <=25={100*m['hit_25_rate']:.2f}%, <=200={100*m['hit_200_rate']:.2f}%", flush=True)
        if m["mean_points"] > best_s2_pts:
            best_s2_pts = m["mean_points"]
            best_s2_state = copy.deepcopy(ranker.state_dict())
            best_s2_epoch = epoch

    ranker.load_state_dict(best_s2_state)
    ranker.eval()
    with torch.inference_mode():
        val_rk_logits = ranker(torch.as_tensor(val_gsim, device=device), torch.as_tensor(val_feats, device=device)).cpu().numpy()
        ts_rk_logits = ranker(torch.as_tensor(ts_gsim, device=device), torch.as_tensor(ts_feats, device=device)).cpu().numpy()

    # Raw frozen StreetCLIP diff-creator Top-100 on val_2000 & official_test_300
    val_raw_top, val_raw_sim = retrieve_diff_creator(vf_t, v_attr, gf_t, g_attr, k=100)
    ts_raw_top, ts_raw_sim = retrieve_diff_creator(tsf_t, ts_attr, gf_t, g_attr, k=100)

    def eval_block(q_lat, q_lon, q_cc, raw_top, proj_top, kde_sc, rk_logits, txt_lp):
        rg = raw_top[:, 0]
        raw_d1 = haversine_vec_np(q_lat, q_lon, g_lat[rg], g_lon[rg])
        raw_d100 = haversine_vec_np(q_lat[:, None], q_lon[:, None], g_lat[raw_top], g_lon[raw_top])
        raw_m = summarize(raw_d1, g_cc[rg], q_cc)
        raw_m["top100_25km"] = float((raw_d100.min(axis=1) <= 25.0).mean())
        raw_m["top100_200km"] = float((raw_d100.min(axis=1) <= 200.0).mean())
        raw_m["top100_country"] = float(np.any(g_cc[raw_top] == q_cc[:, None], axis=1).mean())

        pg = proj_top[:, 0]
        p_d1 = haversine_vec_np(q_lat, q_lon, g_lat[pg], g_lon[pg])
        p_d100 = haversine_vec_np(q_lat[:, None], q_lon[:, None], g_lat[proj_top], g_lon[proj_top])
        s1_m = summarize(p_d1, g_cc[pg], q_cc)
        s1_m["top100_25km"] = float((p_d100.min(axis=1) <= 25.0).mean())
        s1_m["top100_200km"] = float((p_d100.min(axis=1) <= 200.0).mean())
        s1_m["top100_country"] = float(np.any(g_cc[proj_top] == q_cc[:, None], axis=1).mean())

        r0 = np.arange(len(q_lat))
        kg = proj_top[r0, kde_sc.argmax(axis=1)]
        kde_m = summarize(haversine_vec_np(q_lat, q_lon, g_lat[kg], g_lon[kg]), g_cc[kg], q_cc)

        rkg = proj_top[r0, rk_logits.argmax(axis=1)]
        s2_m = summarize(haversine_vec_np(q_lat, q_lon, g_lat[rkg], g_lon[rkg]), g_cc[rkg], q_cc)

        zs_acc = float((np.array(all_ccs)[txt_lp.argmax(axis=1)] == q_cc).mean())
        return {
            "zs_text_country_acc": zs_acc,
            "raw_streetclip_25k": raw_m,
            "stage1_geoproj_top1": s1_m,
            "stage1_geoproj_kde": kde_m,
            "stage2_listwise_ranker": s2_m,
        }

    report = {
        "best_stage1_epoch": best_s1_epoch,
        "best_stage2_epoch": best_s2_epoch,
        "stage1_history": s1_hist,
        "stage2_history": s2_hist,
        "splits": {
            "val_2000": eval_block(v_lat, v_lon, v_cc, val_raw_top, val_top, val_kde, val_rk_logits, v_txt_lp),
            "val_unseen_opt_522": eval_block(v_lat[val_unseen_opt_idx], v_lon[val_unseen_opt_idx], v_cc[val_unseen_opt_idx],
                                             val_raw_top[val_unseen_opt_idx], val_top[val_unseen_opt_idx],
                                             val_kde[val_unseen_opt_idx], val_rk_logits[val_unseen_opt_idx], v_txt_lp[val_unseen_opt_idx]),
            "qlora_eval_300": eval_block(v_lat[q300_idx], v_lon[q300_idx], v_cc[q300_idx],
                                         val_raw_top[q300_idx], val_top[q300_idx],
                                         val_kde[q300_idx], val_rk_logits[q300_idx], v_txt_lp[q300_idx]),
            "author_holdout_66": eval_block(v_lat[ah_idx], v_lon[ah_idx], v_cc[ah_idx],
                                            val_raw_top[ah_idx], val_top[ah_idx],
                                            val_kde[ah_idx], val_rk_logits[ah_idx], v_txt_lp[ah_idx]),
            "official_test_300": eval_block(ts_lat, ts_lon, ts_cc, ts_raw_top, ts_top, ts_kde, ts_rk_logits, ts_txt_lp),
        },
    }

    for sp, d in report["splits"].items():
        print(f"\n=== {sp} (25k StreetCLIP, STRICT DIFF-CREATOR | ZS text cc={100*d['zs_text_country_acc']:.1f}%) ===", flush=True)
        for mk in ("raw_streetclip_25k", "stage1_geoproj_top1", "stage1_geoproj_kde", "stage2_listwise_ranker"):
            m = d[mk]
            extra = f" | Top100 <=25={100*m['top100_25km']:.1f}%, <=200={100*m['top100_200km']:.1f}%, cc={100*m['top100_country']:.1f}%" if "top100_25km" in m else ""
            print(f"  {mk:24s}: <=25={m['hit_25']:3d} ({100*m['hit_25_rate']:5.2f}%), <=200={m['hit_200']:3d} ({100*m['hit_200_rate']:5.2f}%), "
                  f"cc={m['country_correct']:3d} ({100*m['country_rate']:5.1f}%), med={m['median_km']:6.1f}km, pts={m['mean_points']:6.1f}{extra}", flush=True)

    (args.output / "streetclip_25k_results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "streetclip_25k_top100.npz",
        val_ids=np.array([r["id"] for r in val]),
        val_raw_top=val_raw_top, val_raw_sim=val_raw_sim,
        val_proj_top=val_top, val_proj_sim=val_gsim,
        val_kde=val_kde, val_rk_logits=val_rk_logits, val_txt_lp=v_txt_lp, val_aux_lp=v_aux_lp,
        test_ids=np.array([r["id"] for r in test_rows]),
        test_raw_top=ts_raw_top, test_raw_sim=ts_raw_sim,
        test_proj_top=ts_top, test_proj_sim=ts_gsim,
        test_kde=ts_kde, test_rk_logits=ts_rk_logits, test_txt_lp=ts_txt_lp, test_aux_lp=ts_aux_lp,
        country_vocab=np.array(all_ccs),
    )
    torch.save({"projector": best_s1_state, "ranker": best_s2_state, "country_vocab": all_ccs}, args.output / "streetclip_25k_ranker.pt")
    print(f"\nSaved all Phase B outputs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
