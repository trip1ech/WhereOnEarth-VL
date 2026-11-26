"""Prepare Colab transfer bundle for Phase C: QLoRA CoT SFT -> Offline Multimodal DPO
on Qwen/Qwen3-VL-8B-Instruct using audited Phase B outputs from results/vlm_multiscale_cot/.
"""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
OUT_DIR = ROOT / "results/vlm_multiscale_cot"


def normalize_cot_json_str(raw_json: str) -> str:
    """Normalize coordinates to 2 decimals and ASCII punctuation in BOTH chosen and rejected
    outputs so DPO cannot exploit formatting/decimal-precision shortcuts.
    """
    obj = json.loads(raw_json)
    for k in ("visual_clues", "reasoning"):
        s = str(obj.get(k, "") or "")
        s = (
            s.replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .strip()
        )
        obj[k] = s
    obj["selected_cluster"] = int(obj["selected_cluster"])
    obj["country"] = str(obj["country"]).upper()
    obj["latitude"] = round(float(obj["latitude"]), 2)
    obj["longitude"] = round(float(obj["longitude"]), 2)
    obj["confidence"] = str(obj["confidence"]).lower()
    return json.dumps(obj, ensure_ascii=False)


def keep_grounded_sft(r: dict) -> bool:
    """Drop ungrounded low-confidence overrides and low-confidence STaR flips that contradict
    our calibration rule ('stay with #1 when visual cues are insufficient').
    """
    j = json.loads(r["response"])
    if r["category"] == "override_outside_top8":
        return j["confidence"] in ("high", "medium")
    if r["source"] == "star_rationalized" and r["category"] in ("cross_country_flip", "same_country_region_flip"):
        return j["confidence"] in ("high", "medium")
    return True


def keep_grounded_dpo(r: dict) -> bool:
    j = json.loads(r["chosen"])
    if r["category"] == "override_outside_top8":
        return j["confidence"] in ("high", "medium")
    if r["chosen_source"] == "star_rationalized" and r["category"] in ("cross_country_flip", "same_country_region_flip"):
        return j["confidence"] in ("high", "medium")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-lr", type=float, default=4e-5)
    parser.add_argument("--sft-grad-accum", type=int, default=8)
    parser.add_argument("--dpo-epochs", type=int, default=2)
    parser.add_argument("--dpo-lr", type=float, default=8e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-nll-weight", type=float, default=0.1)
    parser.add_argument("--dpo-grad-accum", type=int, default=8)
    parser.add_argument("--dpo-source-filter", choices=("all", "organic_only"), default="all")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--timeout", type=int, default=9000)
    args = parser.parse_args()

    rollout_zip = OUT_DIR / "cot_rollout_results.zip"
    if not rollout_zip.exists():
        raise FileNotFoundError(f"Missing Phase B output: {rollout_zip}")

    with ZipFile(rollout_zip) as z:
        sft_raw = [json.loads(l) for l in z.read("sft_cot_train.jsonl").decode("utf-8").splitlines() if l.strip()]
        dpo_raw = [json.loads(l) for l in z.read("dpo_cot_pairs.jsonl").decode("utf-8").splitlines() if l.strip()]

    # 1. Visual-Grounding Calibration Filter + Zero-Shortcut Normalization
    sft_rows = []
    for r in sft_raw:
        if not keep_grounded_sft(r):
            continue
        r2 = dict(r)
        r2["response"] = normalize_cot_json_str(r["response"])
        sft_rows.append(r2)

    dpo_rows = []
    for r in dpo_raw:
        if not keep_grounded_dpo(r):
            continue
        if args.dpo_source_filter == "organic_only" and r.get("chosen_source") != "rejection_sampled":
            continue
        r2 = dict(r)
        r2["chosen"] = normalize_cot_json_str(r["chosen"])
        r2["rejected"] = normalize_cot_json_str(r["rejected"])
        if r2["chosen"] != r2["rejected"]:
            dpo_rows.append(r2)

    # 2. Gentle natural-prior balancing so fine-tuned model does not over-flip away from Cluster #1
    rng = random.Random(42)
    keep_sft = [r for r in sft_rows if r["category"] == "keep_top1"]
    extra_keep = rng.sample(keep_sft, min(50, len(keep_sft)))
    sft_rows.extend(extra_keep)

    keep_dpo = [r for r in dpo_rows if r["pair_type"] == "keep_stability"]
    dpo_rows.extend(keep_dpo)  # 2x keep_stability (~19% of DPO pairs)

    print(
        f"Curated Phase C datasets: {len(sft_rows)} SFT CoTs {dict(Counter(r['category'] for r in sft_rows))} | "
        f"{len(dpo_rows)} DPO pairs {dict(Counter(r['pair_type'] for r in dpo_rows))}",
        flush=True,
    )

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names_bytes = z.read("country_names.json")

    transfer = OUT_DIR / "transfer_train"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    bundle = OUT_DIR / "cot_train_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(OUT_DIR / "train_subset_imgs.zip", "train_subset_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", cc_names_bytes)
        z.write(OUT_DIR / "rollout_train_pool.jsonl", "rollout_train_pool.jsonl")
        z.write(OUT_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        z.writestr("sft_cot_train.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sft_rows))
        z.writestr("dpo_cot_pairs.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dpo_rows))
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/train_qwen8b_cot_sft_dpo.py",
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

root = Path("/content/cot_train")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("cottrn-part-*"))
bundle = root / "cot_train_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Train bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "train_qwen8b_cot_sft_dpo.py",
    str(root), str(res),
    "--sft-epochs", "{args.sft_epochs}",
    "--sft-lr", "{args.sft_lr}",
    "--sft-grad-accum", "{args.sft_grad_accum}",
    "--dpo-epochs", "{args.dpo_epochs}",
    "--dpo-lr", "{args.dpo_lr}",
    "--dpo-beta", "{args.dpo_beta}",
    "--dpo-nll-weight", "{args.dpo_nll_weight}",
    "--dpo-grad-accum", "{args.dpo_grad_accum}",
])

out_zip = root / "cot_sft_dpo_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in (
        "train_report.json",
        "sft_8b_author_holdout_top8.jsonl",
        "dpo_ep1_8b_author_holdout_top8.jsonl",
        "dpo_ep2_8b_author_holdout_top8.jsonl",
        "dpo_8b_author_holdout_top8.jsonl",
    ):
        p = res / fn
        if p.exists():
            z.write(p, fn)
    for sub in ("adapter_sft", "adapter_dpo_ep1", "adapter_dpo_ep2", "adapter_dpo"):
        d = res / sub
        if d.exists():
            for fp in sorted(d.rglob("*")):
                if fp.is_file():
                    z.write(fp, f"{{sub}}/{{fp.relative_to(d)}}")
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (transfer / "drive_colab.py").write_text(drive_py, encoding="utf-8")

    run_sh = f"""#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="woe-qwen8b-cot-train"
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
    echo "A100 and L4 temporarily busy, waiting 25s before retry..."
    sleep 25
done
if [ "$assigned" -eq 0 ]; then
    echo "Could not acquire A100 or L4 after 5 attempts; aborting rather than timing out on T4."
    exit 1
fi
for part in "$target"/transfer_train/part-*; do
    "$cli" upload --session "$name" "$part" "/content/cottrn-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer_train/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s..."
    sleep 10
done
"$cli" download --session "$name" /content/cot_train/cot_sft_dpo_results.zip "$target/cot_sft_dpo_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
