"""Prepare Colab bundle for Direction 2 — Phase 2:
Joint Two-Task Training with Unfrozen Visual Mergers (`visual.merger` + `deepstack_merger_list`)
+ LM LoRA on Qwen/Qwen3-VL-8B-Instruct.

Builds:
  - Task 1 (~70%): Zero-sky isolated local cue crops -> shuffled 5-way hard-neighbor MCQ (A..E)
  - Task 2 (~30%): Image-First Open-Ended Visual Fovea localization (natural-prior balanced,
                   zero STaR hallucination)
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from multiscale_vlm_utils import (
    build_direct_fovea_prompt,
    build_shuffled_mcq_prompt,
    haversine_km,
)

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
OUT_DIR = ROOT / "results/vlm_local_cue_dir2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--merger-lr", type=float, default=6e-5)
    parser.add_argument("--lora-lr", type=float, default=4e-5)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-mcq-per-img", type=int, default=2)
    parser.add_argument("--max-fovea", type=int, default=700)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--timeout", type=int, default=9000)
    args = parser.parse_args()

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names = json.loads(z.read("country_names.json"))
    pool_map = {
        r["id"]: r
        for r in (json.loads(l) for l in (OUT_DIR / "harvest_train_pool.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }
    with ZipFile(OUT_DIR / "dir2_phase1_results.zip") as z:
        harv_rows = [json.loads(l) for l in z.read("harvested_opt_train_cues.jsonl").decode("utf-8").splitlines() if l.strip()]
        base_mcq_bytes = z.read("base_8b_ah66_isolated_mcq.jsonl")

    rng = random.Random(20261126)
    letters = "ABCDE"
    mcq_tasks = []
    fovea_buckets = defaultdict(list)

    for h in harv_rows:
        rid = h["id"]
        boxes = [b for b in h.get("boxes", []) if not CAMERA_META_PAT.search(b.get("label", ""))]
        if not boxes:
            continue
        meta = pool_map[rid]
        tcc = meta["country"]
        tlat, tlon = meta["latitude"], meta["longitude"]
        dist_info = meta["kde_distribution"]

        # Task 1: Zero-sky isolated local cue -> 5-way shuffled hard-neighbor MCQ
        distractors = [c for c in meta["cand_countries"] if c != tcc][:4]
        if len(distractors) == 4:
            for b in boxes[:args.max_mcq_per_img]:
                opts = [tcc] + distractors
                rng.shuffle(opts)
                ans_idx = opts.index(tcc)
                ans_letter = letters[ans_idx]
                prompt = build_shuffled_mcq_prompt(b["label"], opts, cc_names)
                resp = json.dumps({"option": ans_letter, "country": tcc}, ensure_ascii=False)
                mcq_tasks.append({
                    "task_type": "isolated_cue_mcq",
                    "id": rid,
                    "image": meta["image"],
                    "country": tcc,
                    "label": b["label"],
                    "is_text_like": b.get("is_text_like", False),
                    "bbox_1000": b["bbox_1000"],
                    "ans_letter": ans_letter,
                    "prompt": prompt,
                    "response": resp,
                })

        # Task 2: Image-First Visual Fovea Localization (only when true country is in Top-8 -> clean grounded target!)
        cls = dist_info["clusters"]
        ds = [haversine_km(tlat, tlon, c["latitude"], c["longitude"]) for c in cls]
        same_cc = [(j, ds[j]) for j, c in enumerate(cls) if c["country"] == tcc]
        if not same_cc:
            continue
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
            "country": tcc,
            "latitude": round(float(tlat), 2),
            "longitude": round(float(tlon), 2),
            "confidence": conf,
        }, ensure_ascii=False)
        fovea_buckets[cat].append({
            "task_type": "direct_open_fovea",
            "category": cat,
            "id": rid,
            "image": meta["image"],
            "country": tcc,
            "crop_boxes": [b["bbox_1000"] for b in use_boxes],
            "prompt": f_prompt,
            "response": f_resp,
        })

    # Natural-prior balance for Task 2 (~44% keep_top1, ~30% cross_country, ~26% same_country_region)
    for k in fovea_buckets:
        rng.shuffle(fovea_buckets[k])
    n_keep = min(len(fovea_buckets["keep_top1"]), int(args.max_fovea * 0.44))
    n_cross = min(len(fovea_buckets["cross_country_flip"]), int(args.max_fovea * 0.30))
    n_reg = min(len(fovea_buckets["same_country_region_flip"]), args.max_fovea - n_keep - n_cross)
    fovea_tasks = (
        fovea_buckets["keep_top1"][:n_keep]
        + fovea_buckets["cross_country_flip"][:n_cross]
        + fovea_buckets["same_country_region_flip"][:n_reg]
    )

    all_tasks = mcq_tasks + fovea_tasks
    rng.shuffle(all_tasks)

    opt_dist = Counter(t["ans_letter"] for t in mcq_tasks)
    nt_cnt = sum(1 for t in mcq_tasks if not t["is_text_like"])
    print(
        f"Built Phase 2 training set: {len(all_tasks)} total tasks | "
        f"Task 1 MCQ={len(mcq_tasks)} (non-text={nt_cnt}, slot_balance={dict(sorted(opt_dist.items()))}) | "
        f"Task 2 Fovea={len(fovea_tasks)} {dict(Counter(t['category'] for t in fovea_tasks))}",
        flush=True,
    )

    transfer = OUT_DIR / "transfer_phase2"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    with ZipFile(STEP0_DIR / "step0_results.zip") as z:
        step0_g_bytes = z.read("grounding_boxes_ah66.jsonl")
    with ZipFile(STEP1_DIR / "step1_mil_results.zip") as z:
        open_g_bytes = z.read("open_grounding_ah66.jsonl")

    bundle = OUT_DIR / "dir2_phase2_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(OUT_DIR / "harvest_imgs.zip", "harvest_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", json.dumps(cc_names, ensure_ascii=False) + "\n")
        z.write(OUT_DIR / "harvest_train_pool.jsonl", "harvest_train_pool.jsonl")
        z.writestr("dir2_train_tasks.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_tasks))
        z.write(MS_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        z.writestr("base_8b_ah66_isolated_mcq.jsonl", base_mcq_bytes)
        z.writestr("open_grounding_ah66.jsonl", open_g_bytes)
        z.writestr("grounding_boxes_ah66.jsonl", step0_g_bytes)
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_step0_grounding_attention.py",
            ROOT / "scripts/run_dir2_harvest_crops.py",
            ROOT / "scripts/train_dir2_unfrozen_merger.py",
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
root = Path("/content/dir2_p2")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("d2p2-part-*"))
bundle = root / "dir2_phase2_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Dir2 Phase 2 bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "train_dir2_unfrozen_merger.py",
    str(root), str(res),
    "--epochs", "{args.epochs}",
    "--merger-lr", "{args.merger_lr}",
    "--lora-lr", "{args.lora_lr}",
    "--grad-accum", "{args.grad_accum}",
])

best_ep = (res / "best_epoch.txt").read_text(encoding="utf-8").strip() if (res / "best_epoch.txt").exists() else "ep2"
out_zip = root / "dir2_phase2_results.zip"
other_merger = None
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fp in sorted(res.rglob("*")):
        if not fp.is_file() or fp.is_relative_to(res / "imgs"):
            continue
        rel = str(fp.relative_to(res))
        if fp.name == "merger_weights_bf16.pt" and f"ckpt_{{best_ep}}/" not in rel:
            other_merger = fp
            continue
        z.write(fp, rel)
if other_merger is not None:
    shutil.copy2(other_merger, root / "dir2_phase2_other_merger.pt")
    print(f"Kept secondary merger at {{root / 'dir2_phase2_other_merger.pt'}} ({{other_merger.stat().st_size / 1e6:.1f}} MB)", flush=True)
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB, best_ep={{best_ep}})", flush=True)
"""
    (transfer / "drive_colab.py").write_text(drive_py, encoding="utf-8")

    run_sh = f"""#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="woe-dir2-train"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assigned=0
for gpu_try in 1 2 3 4 5; do
    echo "=== GPU assign attempt $gpu_try (trying {args.gpu} -> L4) ==="
    if "$cli" new --session "$name" --gpu {args.gpu} < /dev/null; then
        assigned=1
        break
    fi
    if "$cli" new --session "$name" --gpu L4 < /dev/null; then
        assigned=1
        break
    fi
    echo "A100 and L4 busy, waiting 25s..."
    sleep 25
done
if [ "$assigned" -eq 0 ]; then
    echo "Could not acquire A100 or L4."
    exit 1
fi
for part in "$target"/transfer_phase2/part-*; do
    "$cli" upload --session "$name" "$part" "/content/d2p2-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer_phase2/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s..."
    sleep 10
done
"$cli" download --session "$name" /content/dir2_p2/dir2_phase2_results.zip "$target/dir2_phase2_results.zip"
"$cli" download --session "$name" /content/dir2_p2/dir2_phase2_other_merger.pt "$target/dir2_phase2_other_merger.pt" || true
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
