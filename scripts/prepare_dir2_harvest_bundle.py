"""Prepare Colab bundle for Direction 2 Phase 1:
  - 900 country-balanced creator-disjoint opt_train images + their exact StreetCLIP
    Top-8 multi-scale clusters & hard-neighbor country lists
  - author_holdout_66 + Step 1 open-ended grounding boxes for pre-training baselines
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

from multiscale_vlm_utils import build_multiscale_clusters, compute_kde_log_scores

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
SFT_ZIP = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
MS_DIR = ROOT / "results/vlm_multiscale_cot"
STEP1_DIR = ROOT / "results/vlm_mil_attention_step1"
OUT_DIR = ROOT / "results/vlm_local_cue_dir2"


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
    parser.add_argument("--n-harvest", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = OUT_DIR / "transfer_phase1"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names = json.loads(z.read("country_names.json"))
    with ZipFile(SC25K_DIR / "manifests_39k.zip") as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        trn = [json.loads(l) for l in z.read("train_images.jsonl").splitlines()]
    with ZipFile(SC25K_DIR / "streetclip_25k_results.zip") as z:
        with z.open("streetclip_25k_top100.npz") as f:
            sc_top = dict(np.load(f))
        with z.open("streetclip_25k_ranker.pt") as f:
            ckpt = torch.load(f, map_location="cpu")
    with ZipFile(AH_ZIP) as z:
        ah_rows = [json.loads(l) for l in z.read("eval.jsonl").splitlines()]
    with ZipFile(SFT_ZIP) as z:
        sft_train = [json.loads(l) for l in z.read("train.jsonl").splitlines()]

    # Exact creator-disjoint opt_train selection
    ah_creators = {r["attribution"] for r in ah_rows if r.get("attribution")}
    rng_split = np.random.default_rng(42)
    t_attr_all = np.array([r["attribution"] for r in trn])
    eligible_creators = sorted({a for a in t_attr_all if a not in ah_creators})
    rng_split.shuffle(eligible_creators)
    tune_creators = set(eligible_creators[:max(1, int(0.15 * len(eligible_creators)))])

    trn_map = {r["id"]: i for i, r in enumerate(trn)}
    opt_sft = [
        r for r in sft_train
        if trn[trn_map[r["id"]]]["attribution"] not in tune_creators
        and trn[trn_map[r["id"]]]["attribution"] not in ah_creators
    ]
    rng = np.random.default_rng(20261126)
    selected = round_robin_by_country(opt_sft, args.n_harvest, rng)
    sel_tidx = np.array([trn_map[r["id"]] for r in selected], dtype=np.int32)
    print(f"Selected {len(selected)} balanced opt_train images across {len({r['country'] for r in selected})} countries.", flush=True)

    # Compute exact Stage 1 GeoProjector Top-100 + Top-8 clusters & hard-neighbor countries
    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    g_attr = np.array([r["attribution"] for r in gal])
    cc_vocab = [str(c) for c in sc_top["country_vocab"]]

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
        q_proj = proj.encode(torch.from_numpy(tf[sel_tidx])).numpy()

    q_raw = tf[sel_tidx]
    txt_logits = (q_raw @ text_ens.T) * 30.0
    txt_logits -= txt_logits.max(axis=1, keepdims=True)
    txt_lp = txt_logits - np.log(np.exp(txt_logits).sum(axis=1, keepdims=True))

    q_attr = np.array([trn[i]["attribution"] for i in sel_tidx])
    n = len(sel_tidx)
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

    pool_out = []
    for s in range(0, n, 256):
        e = min(n, s + 256)
        kde_b, pwd_b = compute_kde_log_scores(top_idx[s:e], top_sim[s:e], g_lat, g_lon)
        for bi in range(e - s):
            gi = s + bi
            r = selected[gi]
            dist_info = build_multiscale_clusters(
                top_idx[gi], top_sim[gi], kde_b[bi], pwd_b[bi], txt_lp[gi],
                g_lat, g_lon, g_cc, cc_vocab, cc_names,
            )
            # Collect top-6 distinct hard-neighbor countries (from clusters + text probe + top100)
            cands = []
            for cl in dist_info["clusters"]:
                if cl["country"] not in cands:
                    cands.append(cl["country"])
            for tx in dist_info["top_text_countries"]:
                if tx["country"] not in cands:
                    cands.append(tx["country"])
            for j in top_idx[gi]:
                c = str(g_cc[j])
                if c not in cands:
                    cands.append(c)
                if len(cands) >= 6:
                    break
            pool_out.append({
                "id": r["id"],
                "image": r["image"],
                "sha256": r["sha256"],
                "country": str(r["country"]),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "cand_countries": cands[:6],
                "cand_country_names": [cc_names.get(c, c) for c in cands[:6]],
                "kde_distribution": dist_info,
            })

    pool_jsonl = OUT_DIR / "harvest_train_pool.jsonl"
    pool_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pool_out), encoding="utf-8")

    # Package only the 900 JPEGs (~36 MB)
    imgs_zip = OUT_DIR / "harvest_imgs.zip"
    with ZipFile(SFT_ZIP) as src_z, ZipFile(imgs_zip, "w", compression=ZIP_STORED) as dst_z:
        for r in pool_out:
            dst_z.writestr(r["image"], src_z.read(r["image"]))

    with ZipFile(STEP1_DIR / "step1_mil_results.zip") as z:
        open_g_bytes = z.read("open_grounding_ah66.jsonl")

    bundle = OUT_DIR / "dir2_phase1_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(imgs_zip, "harvest_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", json.dumps(cc_names, ensure_ascii=False) + "\n")
        z.write(pool_jsonl, "harvest_train_pool.jsonl")
        z.write(MS_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        z.writestr("open_grounding_ah66.jsonl", open_g_bytes)
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_step0_grounding_attention.py",
            ROOT / "scripts/run_dir2_harvest_crops.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)

    subprocess.run(["split", "-b", "10m", "-a", "2", str(bundle), str(transfer / "part-")], check=True)
    parts = sorted(transfer.glob("part-*"))
    print(f"Split into {len(parts)} parts in {transfer}", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/dir2_p1")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("d2p1-part-*"))
bundle = root / "dir2_phase1_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Dir2 Phase 1 bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "run_dir2_harvest_crops.py",
    str(root), str(res), "--batch-size", "{args.batch_size}",
])

out_zip = root / "dir2_phase1_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in (
        "dir2_phase1_summary.json",
        "base_8b_ah66_isolated_mcq.jsonl",
        "base_8b_ah66_imgfirst_open_fovea.jsonl",
        "harvested_opt_train_cues.jsonl",
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
name="woe-dir2-harvest"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assigned=0
for gpu_try in 1 2 3 4 5; do
    echo "=== GPU assign attempt $gpu_try (trying {args.gpu} -> A100) ==="
    if "$cli" new --session "$name" --gpu {args.gpu} < /dev/null; then
        assigned=1
        break
    fi
    if "$cli" new --session "$name" --gpu A100 < /dev/null; then
        assigned=1
        break
    fi
    if [ "$gpu_try" -ge 4 ]; then
        if "$cli" new --session "$name" --gpu T4 < /dev/null; then
            assigned=1
            break
        fi
    fi
    echo "L4/A100 busy, waiting 25s..."
    sleep 25
done
if [ "$assigned" -eq 0 ]; then
    echo "Could not acquire GPU."
    exit 1
fi
for part in "$target"/transfer_phase1/part-*; do
    "$cli" upload --session "$name" "$part" "/content/d2p1-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer_phase1/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s (incremental JSONL resume enabled)..."
    sleep 10
done
"$cli" download --session "$name" /content/dir2_p1/dir2_phase1_results.zip "$target/dir2_phase1_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
