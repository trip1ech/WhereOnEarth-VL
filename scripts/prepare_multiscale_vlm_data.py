"""Prepare Colab bundle for Phase B (Multi-Scale Top-8 Spatial KDE CoT Rollout +
STaR Rationalization) from the 5,174 creator-disjoint local opt_train images.

Balances all 4 multi-scale tasks (country + sub-national region + coordinate refinement):
  - cross_country_flip        (~35%)
  - same_country_region_flip  (~30%)
  - keep_top1                 (~25%, stratified close + far)
  - override_outside_top8     (~10%)
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiscale_vlm_utils import (
    build_multiscale_clusters,
    compute_kde_log_scores,
    haversine_km,
)

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
SFT_ZIP = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
OUT_DIR = ROOT / "results/vlm_multiscale_cot"


class GeoProjector(nn.Module):
    def __init__(self, dim=768, hidden=768, num_countries=220, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.linear_skip = nn.Linear(dim, dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.country_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_countries),
        )

    def encode(self, x):
        return F.normalize(self.linear_skip(x) + self.net(x), dim=-1)


def round_robin_by_country(items, limit, rng):
    by_cc = defaultdict(list)
    for it in items:
        by_cc[it["country"]].append(it)
    ccs = sorted(by_cc.keys())
    rng.shuffle(ccs)
    for c in ccs:
        rng.shuffle(by_cc[c])
    picked = []
    while len(picked) < limit:
        progressed = False
        for c in ccs:
            if by_cc[c]:
                picked.append(by_cc[c].pop())
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=400, help="Balanced rollout pool size from opt_train")
    parser.add_argument("--k-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = OUT_DIR / "transfer_rollout"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names = json.loads(z.read("country_names.json"))
    with ZipFile(SC25K_DIR / "manifests_39k.zip") as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        trn = [json.loads(l) for l in z.read("train_images.jsonl").splitlines()]
        val = [json.loads(l) for l in z.read("validation_images.jsonl").splitlines()]
    with ZipFile(SC25K_DIR / "streetclip_25k_results.zip") as z:
        with z.open("streetclip_25k_top100.npz") as f:
            sc_top = dict(np.load(f))
        with z.open("streetclip_25k_ranker.pt") as f:
            ckpt = torch.load(f, map_location="cpu")
    with ZipFile(AH_ZIP) as z:
        ah_rows = [json.loads(l) for l in z.read("eval.jsonl").splitlines()]
    with ZipFile(SFT_ZIP) as z:
        sft_train = [json.loads(l) for l in z.read("train.jsonl").splitlines()]

    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    g_attr = np.array([r["attribution"] for r in gal])
    cc_vocab = [str(c) for c in sc_top["country_vocab"]]

    # 1. Build multi-scale Top-8 distributions for author_holdout_66
    val_map = {r["id"]: i for i, r in enumerate(val)}
    ah_vidx = np.array([val_map[r["id"]] for r in ah_rows], dtype=np.int32)
    ah_kde, ah_pwd = compute_kde_log_scores(sc_top["val_proj_top"][ah_vidx], sc_top["val_proj_sim"][ah_vidx], g_lat, g_lon)
    ah_out = []
    for i, r in enumerate(ah_rows):
        dist_info = build_multiscale_clusters(
            sc_top["val_proj_top"][ah_vidx[i]], sc_top["val_proj_sim"][ah_vidx[i]],
            ah_kde[i], ah_pwd[i], sc_top["val_txt_lp"][ah_vidx[i]],
            g_lat, g_lon, g_cc, cc_vocab, cc_names,
        )
        ah_out.append({
            "id": r["id"], "image": r["image"], "sha256": r["sha256"],
            "country": r["country"], "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
            "kde_distribution": dist_info,
        })
    ah_jsonl = OUT_DIR / "author_holdout_multiscale.jsonl"
    ah_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ah_out), encoding="utf-8")

    # 2. Identify creator-disjoint opt_train subset inside sft_6000 (excluding tune & author_holdout)
    ah_creators = {r["attribution"] for r in ah_rows if r.get("attribution")}
    rng_split = np.random.default_rng(42)
    t_attr_all = np.array([r["attribution"] for r in trn])
    eligible_mask = np.array([a not in ah_creators for a in t_attr_all], dtype=bool)
    eligible_creators = sorted({a for a, ok in zip(t_attr_all, eligible_mask) if ok})
    rng_split.shuffle(eligible_creators)
    tune_creators = set(eligible_creators[:max(1, int(0.15 * len(eligible_creators)))])

    trn_map = {r["id"]: i for i, r in enumerate(trn)}
    opt_sft = [r for r in sft_train if trn[trn_map[r["id"]]]["attribution"] not in tune_creators and trn[trn_map[r["id"]]]["attribution"] not in ah_creators]
    opt_tidx = np.array([trn_map[r["id"]] for r in opt_sft], dtype=np.int32)
    print(f"Eligible creator-disjoint opt_train images in local sft_6000: {len(opt_sft)}", flush=True)

    # Project embeddings with Stage 1 GeoProjector
    sc_emb = np.load(SC25K_DIR / "streetclip_39k_plus_test300.npz")
    gf = sc_emb["gallery_features"].astype(np.float32); gf /= np.linalg.norm(gf, axis=-1, keepdims=True)
    tf = sc_emb["train_features"].astype(np.float32); tf /= np.linalg.norm(tf, axis=-1, keepdims=True)
    text_ens = sc_emb["text_feats_ens"].astype(np.float32)

    state = ckpt["projector"]
    proj = GeoProjector(dim=768, hidden=768, num_countries=state["country_head.2.weight"].shape[0])
    proj.load_state_dict(state)
    proj.eval()
    with torch.inference_mode():
        g_proj = proj.encode(torch.from_numpy(gf)).numpy()
        q_proj = proj.encode(torch.from_numpy(tf[opt_tidx])).numpy()

    q_raw = tf[opt_tidx]
    txt_logits = (q_raw @ text_ens.T) * 30.0
    txt_logits -= txt_logits.max(axis=1, keepdims=True)
    txt_lp = txt_logits - np.log(np.exp(txt_logits).sum(axis=1, keepdims=True))

    q_attr = np.array([trn[i]["attribution"] for i in opt_tidx])
    n = len(opt_tidx)
    top_idx = np.empty((n, 100), dtype=np.int32)
    top_sim = np.empty((n, 100), dtype=np.float32)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        sim = q_proj[s:e] @ g_proj.T
        qa = q_attr[s:e][:, None]
        same = (qa == g_attr[None, :]) & (qa != "")
        sim[same] = -2.0
        part = np.argpartition(-sim, 100, axis=1)[:, :100]
        rb = np.arange(e - s)[:, None]
        vals = sim[rb, part]
        ord_k = np.argsort(-vals, axis=1)
        top_idx[s:e] = part[rb, ord_k]
        top_sim[s:e] = vals[rb, ord_k]

    # Categorize all 5,174 opt_train images across the 4 multi-scale categories
    buckets = defaultdict(list)
    for s in range(0, n, 256):
        e = min(n, s + 256)
        kde_b, pwd_b = compute_kde_log_scores(top_idx[s:e], top_sim[s:e], g_lat, g_lon)
        for bi in range(e - s):
            gi = s + bi
            r = opt_sft[gi]
            tlat, tlon, tcc = float(r["latitude"]), float(r["longitude"]), str(r["country"])
            dist_info = build_multiscale_clusters(
                top_idx[gi], top_sim[gi], kde_b[bi], pwd_b[bi], txt_lp[gi],
                g_lat, g_lon, g_cc, cc_vocab, cc_names,
            )
            cls = dist_info["clusters"]
            ds = [haversine_km(tlat, tlon, c["latitude"], c["longitude"]) for c in cls]
            same_cc = [(j, ds[j]) for j, c in enumerate(cls) if c["country"] == tcc]
            if cls[0]["country"] == tcc:
                bj, bd = min(same_cc, key=lambda x: x[1])
                if bj != 0 and (ds[0] - bd) >= 150.0:
                    cat = "same_country_region_flip"
                    target_cl = bj + 1
                    best_d = bd
                else:
                    cat = "keep_top1_far" if ds[0] > 250.0 else "keep_top1_close"
                    target_cl = 1
                    best_d = ds[0]
            elif same_cc:
                bj, bd = min(same_cc, key=lambda x: x[1])
                cat = "cross_country_flip"
                target_cl = bj + 1
                best_d = bd
            else:
                cat = "override_outside_top8"
                target_cl = 0
                best_d = 0.0

            buckets[cat].append({
                "id": r["id"],
                "image": r["image"],
                "sha256": r["sha256"],
                "country": tcc,
                "latitude": tlat,
                "longitude": tlon,
                "category": "keep_top1" if cat.startswith("keep_top1") else cat,
                "sub_category": cat,
                "target_cluster": target_cl,
                "best_cluster_dist_km": round(float(best_d), 1),
                "kde_distribution": dist_info,
            })

    # Balanced country-stratified selection (35% cross-country, 30% same-country region, 25% keep, 10% override)
    rng = np.random.default_rng(20261125)
    n_cross = int(round(args.n_total * 0.35))
    n_reg = int(round(args.n_total * 0.30))
    n_keep_close = int(round(args.n_total * 0.125))
    n_keep_far = int(round(args.n_total * 0.125))
    n_over = args.n_total - (n_cross + n_reg + n_keep_close + n_keep_far)

    selected = (
        round_robin_by_country(buckets["cross_country_flip"], n_cross, rng)
        + round_robin_by_country(buckets["same_country_region_flip"], n_reg, rng)
        + round_robin_by_country(buckets["keep_top1_close"], n_keep_close, rng)
        + round_robin_by_country(buckets["keep_top1_far"], n_keep_far, rng)
        + round_robin_by_country(buckets["override_outside_top8"], n_over, rng)
    )
    rng.shuffle(selected)

    pool_jsonl = OUT_DIR / "rollout_train_pool.jsonl"
    pool_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")

    cc_set = {x["country"] for x in selected}
    cat_summary = {k: sum(1 for x in selected if x["sub_category"] == k) for k in sorted(buckets.keys())}
    print(f"Selected {len(selected)} balanced CoT rollout images across {len(cc_set)} countries: {cat_summary}", flush=True)

    # Package only the selected 800 JPEGs into a compact sub-ZIP (~32 MB instead of 260 MB!)
    sub_zip = OUT_DIR / "train_subset_imgs.zip"
    with ZipFile(SFT_ZIP) as src_z, ZipFile(sub_zip, "w", compression=ZIP_STORED) as dst_z:
        for r in selected:
            dst_z.writestr(r["image"], src_z.read(r["image"]))

    bundle = OUT_DIR / "cot_rollout_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(sub_zip, "train_subset_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", json.dumps(cc_names, ensure_ascii=False) + "\n")
        z.write(pool_jsonl, "rollout_train_pool.jsonl")
        z.write(ah_jsonl, "author_holdout_multiscale.jsonl")
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_qwen8b_cot_rollout.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)

    subprocess.run(["split", "-b", "5m", "-a", "2", str(bundle), str(transfer / "part-")], check=True)
    parts = sorted(transfer.glob("part-*"))
    print(f"Split into {len(parts)} parts in {transfer}", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/cot_rollout")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("cotroll-part-*"))
bundle = root / "cot_rollout_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Rollout bundle verified:", actual, flush=True)
with ZipFile(bundle) as z:
    z.extractall(root)

def run(cmd):
    with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, cmd)

print("Installing pinned deps...", flush=True)
run([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements-colab.txt"])
res = root / "results"
res.mkdir(exist_ok=True)
run([
    sys.executable, "-u", "run_qwen8b_cot_rollout.py",
    str(root), str(res),
    "--k-samples", "{args.k_samples}",
    "--batch-size", "{args.batch_size}",
])

out_zip = root / "cot_rollout_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in (
        "rollout_config.json",
        "base_8b_author_holdout_top8.jsonl",
        "cot_rollouts_full.jsonl",
        "sft_cot_train.jsonl",
        "dpo_cot_pairs.jsonl",
        "dataset_stats.json",
    ):
        p = res / fn
        if p.exists():
            z.write(p, fn)
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (transfer / "drive_colab.py").write_text(drive_py, encoding="utf-8")

    run_sh = f"""#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="woe-qwen8b-cot-rollout"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! "$cli" new --session "$name" --gpu {args.gpu}; then
    echo "Fallback GPU assignment -> trying L4 then T4..."
    "$cli" new --session "$name" --gpu L4 || "$cli" new --session "$name" --gpu T4
fi
for part in "$target"/transfer_rollout/part-*; do
    "$cli" upload --session "$name" "$part" "/content/cotroll-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer_rollout/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s (incremental JSONL resume enabled)..."
    sleep 10
done
"$cli" download --session "$name" /content/cot_rollout/cot_rollout_results.zip "$target/cot_rollout_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
