"""Prepare Colab L4 bundle for Hierarchical Multi-Scale Geographic Loss + Visual Fovea LM LoRA.

Reuses the 855 clean zero-sky creator-disjoint `opt_train` Fovea crops already harvested in
Direction 2 Phase 1 (zero extra harvesting GPU cost!) and builds a strictly natural-prior
multi-scale training set with exact Haversine cluster distances for differentiable hierarchical
geographic loss (2500 km macro / country + 750 km / 200 km fine / expected GeoScore regret).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from multiscale_vlm_utils import build_direct_fovea_prompt, haversine_km

CAMERA_META_PAT = re.compile(
    r"\b(dashboard|hood|windshield|windscreen|steering|interior|mirror|wiper|sun\s*visor|camera|dashcam)\b",
    re.I,
)

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
MS_DIR = ROOT / "results/vlm_multiscale_cot"
STEP0_DIR = ROOT / "results/vlm_cue_grounding_step0"
STEP1_DIR = ROOT / "results/vlm_mil_attention_step1"
DIR2_DIR = ROOT / "results/vlm_local_cue_dir2"
OUT_DIR = ROOT / "results/vlm_hier_fovea"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora-lr", type=float, default=3e-5)
    parser.add_argument("--sft-weight", type=float, default=0.35)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names = json.loads(z.read("country_names.json"))
    pool_map = {
        r["id"]: r
        for r in (json.loads(l) for l in (DIR2_DIR / "harvest_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }
    with ZipFile(DIR2_DIR / "dir2_phase1_results.zip") as z:
        harv_rows = [json.loads(l) for l in z.read("harvested_opt_train_cues.jsonl").decode("utf-8").splitlines() if l.strip()]
        base_mcq_bytes = z.read("base_8b_ah66_isolated_mcq.jsonl")
    with ZipFile(STEP0_DIR / "step0_results.zip") as z:
        step0_g_bytes = z.read("grounding_boxes_ah66.jsonl")
    with ZipFile(STEP1_DIR / "step1_mil_results.zip") as z:
        open_g_bytes = z.read("open_grounding_ah66.jsonl")

    rng = random.Random(20261126)
    tasks = []
    skipped_far = 0

    for h in harv_rows:
        rid = h["id"]
        boxes = [b for b in h.get("boxes", []) if not CAMERA_META_PAT.search(b.get("label", ""))]
        if not boxes:
            continue
        meta = pool_map[rid]
        tcc = meta["country"]
        tlat, tlon = meta["latitude"], meta["longitude"]
        dist_info = meta["kde_distribution"]
        cls = dist_info["clusters"]
        ds = [haversine_km(tlat, tlon, c["latitude"], c["longitude"]) for c in cls]
        cl_ccs = [c["country"] for c in cls]

        same_cc = [(j, ds[j]) for j, c in enumerate(cls) if c["country"] == tcc]
        if same_cc:
            bj, bd = min(same_cc, key=lambda x: x[1])
            if cls[0]["country"] == tcc:
                if bj != 0 and (ds[0] - bd) >= 150.0:
                    cat = "same_country_region_flip"
                    target_cl = bj + 1
                else:
                    cat = "keep_top1"
                    target_cl = 1
            else:
                cat = "cross_country_flip"
                target_cl = bj + 1
            apply_sft_rest = True
        else:
            bj = int(min(range(len(ds)), key=lambda j: ds[j]))
            if ds[bj] > 1500.0:
                skipped_far += 1
                continue
            cat = "near_neighbor_<=1500km"
            target_cl = bj + 1
            apply_sft_rest = False  # Pure hierarchical geographic cluster loss (no inconsistent JSON country label)

        use_boxes = boxes[:2]
        notes = [f"Zoomed Detail #{ci}: `{b['label']}` ({b.get('why', '')})" for ci, b in enumerate(use_boxes, 1)]
        f_prompt = build_direct_fovea_prompt(dist_info, notes)
        has_txt = any(b.get("is_text_like", False) for b in use_boxes)
        top1_prob = 100.0 * float(cls[0].get("mass", 0.0))
        if has_txt or (cat == "keep_top1" and top1_prob >= 55.0):
            conf = "high"
        elif cat == "keep_top1" and top1_prob < 30.0:
            conf = "low"
        else:
            conf = "medium"

        f_resp = json.dumps({
            "selected_cluster": int(target_cl),
            "country": tcc if apply_sft_rest else cls[bj]["country"],
            "latitude": round(float(tlat if apply_sft_rest else cls[bj]["latitude"]), 2),
            "longitude": round(float(tlon if apply_sft_rest else cls[bj]["longitude"]), 2),
            "confidence": conf,
        }, ensure_ascii=False)

        tasks.append({
            "id": rid,
            "image": meta["image"],
            "category": cat,
            "country": tcc,
            "target_cluster": int(target_cl),
            "apply_sft_rest": apply_sft_rest,
            "cluster_dists_km": [round(x, 2) for x in ds],
            "cluster_countries": cl_ccs,
            "crop_boxes": [b["bbox_1000"] for b in use_boxes],
            "prompt": f_prompt,
            "response": f_resp,
        })

    rng.shuffle(tasks)
    print(
        f"Built Hierarchical Fovea training set: {len(tasks)} tasks (skipped >1500km={skipped_far}) | "
        f"categories={dict(Counter(t['category'] for t in tasks))} | "
        f"unique_countries={len({t['country'] for t in tasks})}",
        flush=True,
    )

    transfer = OUT_DIR / "transfer"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    bundle = OUT_DIR / "hier_fovea_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(DIR2_DIR / "harvest_imgs.zip", "harvest_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", json.dumps(cc_names, ensure_ascii=False) + "\n")
        z.write(DIR2_DIR / "harvest_train_pool.jsonl", "harvest_train_pool.jsonl")
        z.writestr("hier_fovea_train_tasks.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in tasks))
        z.write(MS_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        z.writestr("base_8b_ah66_isolated_mcq.jsonl", base_mcq_bytes)
        z.writestr("open_grounding_ah66.jsonl", open_g_bytes)
        z.writestr("grounding_boxes_ah66.jsonl", step0_g_bytes)
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_step0_grounding_attention.py",
            ROOT / "scripts/train_hier_fovea_lora.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)

    subprocess.run(["split", "-b", "10m", "-a", "2", str(bundle), str(transfer / "part-")], check=True)
    parts = sorted(transfer.glob("part-*"))
    print(f"Split into {len(parts)} parts in {transfer}", flush=True)

    drive_py = f"""import hashlib, os, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
root = Path("/content/hier_fov")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("hfov-part-*"))
bundle = root / "hier_fovea_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Hier-Fovea bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "train_hier_fovea_lora.py",
    str(root), str(res),
    "--epochs", "{args.epochs}",
    "--lora-lr", "{args.lora_lr}",
    "--sft-weight", "{args.sft_weight}",
    "--grad-accum", "{args.grad_accum}",
])

out_zip = root / "hier_fovea_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fp in sorted(res.rglob("*")):
        if fp.is_file() and not fp.is_relative_to(res / "imgs"):
            z.write(fp, str(fp.relative_to(res)))
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (transfer / "drive_colab.py").write_text(drive_py, encoding="utf-8")

    run_sh = f"""#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="woe-hier-fovea"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assigned=0
for gpu_try in 1 2 3 4 5; do
    echo "=== GPU assign attempt $gpu_try ({args.gpu} only) ==="
    if "$cli" new --session "$name" --gpu {args.gpu} < /dev/null; then
        assigned=1
        break
    fi
    echo "{args.gpu} busy, waiting 25s..."
    sleep 25
done
if [ "$assigned" -eq 0 ]; then
    echo "Could not acquire {args.gpu}."
    exit 1
fi
for part in "$target"/transfer/part-*; do
    "$cli" upload --session "$name" "$part" "/content/hfov-${{part##*/}}"
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
"$cli" download --session "$name" /content/hier_fov/hier_fovea_results.zip "$target/hier_fovea_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
