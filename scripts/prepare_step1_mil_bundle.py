"""Prepare Colab bundle for Step 1:
  - 2,000 country-stratified creator-disjoint opt_train images + 300 tune images for Patch-MIL
  - author_holdout_66 + Step 0 grounding boxes for exact Image-First vs Image-Last comparison
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
SFT_ZIP = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
MS_DIR = ROOT / "results/vlm_multiscale_cot"
STEP0_DIR = ROOT / "results/vlm_cue_grounding_step0"
OUT_DIR = ROOT / "results/vlm_mil_attention_step1"


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
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-tune", type=int, default=300)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = OUT_DIR / "transfer"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names_bytes = z.read("country_names.json")
    with ZipFile(SC25K_DIR / "manifests_39k.zip") as z:
        trn = [json.loads(l) for l in z.read("train_images.jsonl").splitlines()]
    with ZipFile(AH_ZIP) as z:
        ah_rows = [json.loads(l) for l in z.read("eval.jsonl").splitlines()]
    with ZipFile(SFT_ZIP) as z:
        sft_train = [json.loads(l) for l in z.read("train.jsonl").splitlines()]

    # Exact same creator-disjoint split as Stage 10/11/Phase B
    ah_creators = {r["attribution"] for r in ah_rows if r.get("attribution")}
    rng_split = np.random.default_rng(42)
    t_attr_all = np.array([r["attribution"] for r in trn])
    eligible_creators = sorted({a for a in t_attr_all if a not in ah_creators})
    rng_split.shuffle(eligible_creators)
    tune_creators = set(eligible_creators[:max(1, int(0.15 * len(eligible_creators)))])

    trn_map = {r["id"]: i for i, r in enumerate(trn)}
    opt_pool = [r for r in sft_train if trn[trn_map[r["id"]]]["attribution"] not in tune_creators and trn[trn_map[r["id"]]]["attribution"] not in ah_creators]
    tune_pool = [r for r in sft_train if trn[trn_map[r["id"]]]["attribution"] in tune_creators and trn[trn_map[r["id"]]]["attribution"] not in ah_creators]

    rng = np.random.default_rng(20261125)
    mil_train = round_robin_by_country(opt_pool, args.n_train, rng)
    mil_tune = round_robin_by_country(tune_pool, args.n_tune, rng)
    print(
        f"Selected {len(mil_train)} MIL train ({len({r['country'] for r in mil_train})} countries) + "
        f"{len(mil_tune)} MIL tune ({len({r['country'] for r in mil_tune})} countries) [0 creator overlap with author_holdout_66].",
        flush=True,
    )

    # Package only the selected 2,300 JPEGs (~85 MB)
    imgs_zip = OUT_DIR / "mil_imgs.zip"
    with ZipFile(SFT_ZIP) as src_z, ZipFile(imgs_zip, "w", compression=ZIP_STORED) as dst_z:
        for r in mil_train + mil_tune:
            dst_z.writestr(r["image"], src_z.read(r["image"]))

    with ZipFile(STEP0_DIR / "step0_results.zip") as z:
        step0_grounding_bytes = z.read("grounding_boxes_ah66.jsonl")

    bundle = OUT_DIR / "step1_mil_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(imgs_zip, "mil_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", cc_names_bytes)
        z.writestr("mil_train_manifest.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mil_train))
        z.writestr("mil_tune_manifest.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mil_tune))
        z.write(MS_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        z.writestr("grounding_boxes_ah66.jsonl", step0_grounding_bytes)
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_step0_grounding_attention.py",
            ROOT / "scripts/run_step1_mil_and_sink.py",
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

root = Path("/content/step1_mil")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("s1mil-part-*"))
bundle = root / "step1_mil_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Step 1 MIL bundle verified:", actual, flush=True)
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
run([sys.executable, "-u", "run_step1_mil_and_sink.py", str(root), str(res)])

out_zip = root / "step1_mil_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in (
        "step1_summary.json",
        "mil_saliency_ah66.jsonl",
        "attention_first_vs_last_ah66.jsonl",
        "open_grounding_ah66.jsonl",
        "imglast_full_ah66.jsonl",
        "imglast_mil_fovea_ah66.jsonl",
        "imglast_open_fovea_ah66.jsonl",
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
name="woe-step1-mil"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! "$cli" new --session "$name" --gpu {args.gpu} < /dev/null; then
    echo "Fallback GPU assignment -> trying A100 then T4..."
    "$cli" new --session "$name" --gpu A100 < /dev/null || "$cli" new --session "$name" --gpu T4 < /dev/null
fi
for part in "$target"/transfer/part-*; do
    "$cli" upload --session "$name" "$part" "/content/s1mil-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s..."
    sleep 10
done
"$cli" download --session "$name" /content/step1_mil/step1_mil_results.zip "$target/step1_mil_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
