"""Prepare Colab transfer bundle for Rank-1-Balanced Offline DPO ablation starting from
adapter_sft/ saved inside results/vlm_multiscale_cot/cot_sft_dpo_results.zip.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
OUT_DIR = ROOT / "results/vlm_multiscale_cot"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-lr", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-nll-weight", type=float, default=0.25)
    parser.add_argument("--dpo-grad-accum", type=int, default=8)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()

    # Load deduplicated, normalized grounded DPO pairs from cot_train_input.zip
    with ZipFile(OUT_DIR / "cot_train_input.zip") as z:
        dpo_all = [json.loads(l) for l in z.read("dpo_cot_pairs.jsonl").decode("utf-8").splitlines() if l.strip()]
    uniq, seen = [], set()
    for r in dpo_all:
        key = (r["id"], r["chosen"], r["rejected"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    g_tf = [r for r in uniq if json.loads(r["chosen"])["selected_cluster"] == 1 and json.loads(r["rejected"])["selected_cluster"] != 1]
    g_ft = [r for r in uniq if json.loads(r["chosen"])["selected_cluster"] != 1 and json.loads(r["rejected"])["selected_cluster"] == 1]
    g_ff = [r for r in uniq if json.loads(r["chosen"])["selected_cluster"] != 1 and json.loads(r["rejected"])["selected_cluster"] != 1]
    g_tt = [r for r in uniq if json.loads(r["chosen"])["selected_cluster"] == 1 and json.loads(r["rejected"])["selected_cluster"] == 1]

    # Exact 1:1 Rank-1 balance: 50 (chosen=#1, rej!=#1) vs 50 highest-margin (chosen!=#1, rej=#1) [25 cross + 25 same_cc]
    g_ft_cross = sorted([r for r in g_ft if r["pair_type"] == "cross_country"], key=lambda x: -x["reward_margin"])
    g_ft_same = sorted([r for r in g_ft if r["pair_type"] == "same_country_coord"], key=lambda x: -x["reward_margin"])
    n_bal = len(g_tf) * 2  # 50
    n_c = n_bal // 2       # 25
    n_s = n_bal - n_c      # 25

    balanced = g_tf + g_tf + g_ft_cross[:n_c] + g_ft_same[:n_s] + g_ff + g_tt
    ch1 = sum(json.loads(r["chosen"])["selected_cluster"] == 1 for r in balanced)
    rj1 = sum(json.loads(r["rejected"])["selected_cluster"] == 1 for r in balanced)
    assert ch1 == rj1, f"Rank-1 symmetry broken: chosen#1={ch1} != rejected#1={rj1}"
    print(
        f"Built Rank-1-Balanced DPO set: {len(balanced)} pairs | chosen#1={ch1} == rejected#1={rj1} | "
        f"types={dict(Counter(r['pair_type'] for r in balanced))}",
        flush=True,
    )

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names_bytes = z.read("country_names.json")

    transfer = OUT_DIR / "transfer_bal_dpo"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    bundle = OUT_DIR / "bal_dpo_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(OUT_DIR / "train_subset_imgs.zip", "train_subset_imgs.zip", compress_type=ZIP_STORED)
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", cc_names_bytes)
        z.write(OUT_DIR / "rollout_train_pool.jsonl", "rollout_train_pool.jsonl")
        z.write(OUT_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        z.writestr("dpo_balanced_pairs.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in balanced))
        # Copy adapter_sft/ directly from cot_sft_dpo_results.zip
        with ZipFile(OUT_DIR / "cot_sft_dpo_results.zip") as src_z:
            for nm in src_z.namelist():
                if nm.startswith("adapter_sft/") and not nm.endswith("/"):
                    z.writestr(nm, src_z.read(nm))
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/train_qwen8b_cot_sft_dpo.py",
            ROOT / "scripts/train_balanced_dpo.py",
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

root = Path("/content/bal_dpo")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("baldpo-part-*"))
bundle = root / "bal_dpo_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Balanced DPO bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "train_balanced_dpo.py",
    str(root), str(res),
    "--dpo-epochs", "{args.dpo_epochs}",
    "--dpo-lr", "{args.dpo_lr}",
    "--dpo-beta", "{args.dpo_beta}",
    "--dpo-nll-weight", "{args.dpo_nll_weight}",
    "--dpo-grad-accum", "{args.dpo_grad_accum}",
])

out_zip = root / "bal_dpo_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in ("balanced_dpo_report.json", "dpo_balanced_8b_author_holdout_top8.jsonl"):
        p = res / fn
        if p.exists():
            z.write(p, fn)
    d = res / "adapter_dpo_balanced"
    if d.exists():
        for fp in sorted(d.rglob("*")):
            if fp.is_file():
                z.write(fp, f"adapter_dpo_balanced/{{fp.relative_to(d)}}")
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (transfer / "drive_colab.py").write_text(drive_py, encoding="utf-8")

    run_sh = f"""#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="woe-qwen8b-bal-dpo"
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
    echo "Could not acquire A100 or L4."
    exit 1
fi
for part in "$target"/transfer_bal_dpo/part-*; do
    "$cli" upload --session "$name" "$part" "/content/baldpo-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer_bal_dpo/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s..."
    sleep 10
done
"$cli" download --session "$name" /content/bal_dpo/bal_dpo_results.zip "$target/bal_dpo_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
