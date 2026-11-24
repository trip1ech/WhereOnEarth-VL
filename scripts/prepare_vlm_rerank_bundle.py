"""Prepare Colab L4 bundle for Retrieval-Augmented Spatial KDE -> Qwen3-VL-8B-Instruct
reranking & coordinate refinement on either `author_holdout_66` (dev validation) or
`official_test_300` (locked official OSV-5M test).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from kde_vlm_utils import build_kde_distribution, compute_kde_log_scores

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["author_holdout_66", "official_test_300"], required=True)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-edge", type=int, default=896)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--out-subdir", default=None)
    args = parser.parse_args()

    out_dir = ROOT / "results/vlm_kde_rerank" / (args.out_subdir or args.split)
    transfer = out_dir / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    # Load gallery metadata, country names, and cached 25k StreetCLIP Stage 1 GeoProj Top-100
    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names = json.loads(z.read("country_names.json"))
    with ZipFile(SC25K_DIR / "manifests_39k.zip") as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
        val = [json.loads(l) for l in z.read("validation_images.jsonl").splitlines()]
    with ZipFile(SC25K_DIR / "streetclip_25k_results.zip") as z:
        with z.open("streetclip_25k_top100.npz") as f:
            sc = dict(np.load(f))

    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    cc_vocab = [str(c) for c in sc["country_vocab"]]

    if args.split == "author_holdout_66":
        img_zip = ROOT / "data/author_holdout/author-holdout-eval.zip"
        with ZipFile(img_zip) as z:
            rows = [json.loads(l) for l in z.read("eval.jsonl").splitlines()]
        val_id_to_idx = {r["id"]: i for i, r in enumerate(val)}
        vidx = np.array([val_id_to_idx[r["id"]] for r in rows], dtype=np.int32)
        top_idx = sc["val_proj_top"][vidx]
        top_sim = sc["val_proj_sim"][vidx]
        txt_lp = sc["val_txt_lp"][vidx]
    else:
        img_zip = ROOT / "data/official_test_300/official-test-300.zip"
        with ZipFile(img_zip) as z:
            rows = [json.loads(l) for l in z.read("test.jsonl").splitlines()]
        test_ids = [str(x) for x in sc["test_ids"]]
        assert [r["id"] for r in rows] == test_ids, "Order mismatch with streetclip_25k_top100.npz test_ids"
        top_idx = sc["test_proj_top"]
        top_sim = sc["test_proj_sim"]
        txt_lp = sc["test_txt_lp"]

    kde_log, pw_d = compute_kde_log_scores(top_idx, top_sim, g_lat, g_lon, tau=0.03, sigma=500.0)

    dist_rows = []
    for i, r in enumerate(rows):
        dist_info = build_kde_distribution(
            top_idx[i],
            top_sim[i],
            kde_log[i],
            pw_d[i],
            txt_lp[i],
            g_lat,
            g_lon,
            g_cc,
            cc_vocab,
            cc_names,
            wt=0.4,
            max_countries=6,
        )
        dist_rows.append({
            "id": r["id"],
            "image": r["image"],
            "sha256": r["sha256"],
            "country": r["country"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "kde_distribution": dist_info,
        })

    dist_jsonl = out_dir / "kde_distributions.jsonl"
    dist_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dist_rows), encoding="utf-8")

    # Quick local sanity report on Top-1 & Oracle Top-6
    top1_cc = sum(r["kde_distribution"]["candidates"][0]["country"] == r["country"] for r in dist_rows)
    top6_cc = sum(r["country"] in [c["country"] for c in r["kde_distribution"]["candidates"]] for r in dist_rows)
    print(f"Prepared {len(dist_rows)} KDE distributions for {args.split}: "
          f"Top-1 cc={top1_cc}/{len(dist_rows)} ({100*top1_cc/len(dist_rows):.1f}%), "
          f"Top-6 country recall={top6_cc}/{len(dist_rows)} ({100*top6_cc/len(dist_rows):.1f}%)", flush=True)

    bundle = out_dir / "vlm_rerank_input.zip"
    bundle_meta = {"split": args.split, "image_zip": img_zip.name, "n": len(dist_rows)}
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(img_zip, img_zip.name, compress_type=ZIP_STORED)
        z.writestr("bundle_meta.json", json.dumps(bundle_meta, indent=2) + "\n")
        z.writestr("country_names.json", json.dumps(cc_names, ensure_ascii=False) + "\n")
        z.write(dist_jsonl, "kde_distributions.jsonl")
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/kde_vlm_utils.py",
            ROOT / "scripts/run_qwen8b_kde_rerank.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)

    subprocess.run(["split", "-b", "5m", "-a", "2", str(bundle), str(transfer / "part-")], check=True)
    parts = sorted(transfer.glob("part-*"))
    print(f"Split into {len(parts)} parts in {transfer}", flush=True)

    prefix = f"vlmrr-{args.split[:6]}"
    extra_flag = ', "--skip-zeroshot"' if args.skip_zeroshot else ""
    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/vlm_rerank_{args.split}")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("{prefix}-part-*"))
bundle = root / "vlm_rerank_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "run_qwen8b_kde_rerank.py",
    str(root), str(res),
    "--batch-size", "{args.batch_size}",
    "--max-edge", "{args.max_edge}"{extra_flag},
])

out_zip = root / "vlm_rerank_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in ("run_config.json", "qwen8b_kde_rerank.jsonl", "qwen8b_zeroshot.jsonl"):
        p = res / fn
        if p.exists():
            z.write(p, fn)
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (transfer / "drive_colab.py").write_text(drive_py, encoding="utf-8")

    session_tag = (args.out_subdir or args.split).replace("_", "-")
    session_name = f"woe-qwen8b-{session_tag}"
    run_sh = f"""#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="{session_name}"
target="{out_dir}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! "$cli" new --session "$name" --gpu {args.gpu}; then
    echo "Fallback GPU assignment -> trying A100 then T4..."
    "$cli" new --session "$name" --gpu A100 || "$cli" new --session "$name" --gpu T4
fi
for part in "$target"/transfer/part-*; do
    "$cli" upload --session "$name" "$part" "/content/{prefix}-${{part##*/}}"
done
"$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer/drive_colab.py"
"$cli" download --session "$name" /content/vlm_rerank_{args.split}/vlm_rerank_results.zip "$target/vlm_rerank_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready to run: {sh}", flush=True)


if __name__ == "__main__":
    main()
