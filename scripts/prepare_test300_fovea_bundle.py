"""Prepare Colab L4 bundle for the Locked Zero-Touch `official_test_300` Paired Showdown:
  1. Zero-Training Base 8B Hybrid Visual Fovea (CoT + Direct/Logit/MBR)
  2. Trained Hier-Fovea Ep1 LM LoRA Hybrid Visual Fovea (Direct/Logit/MBR)
on identical Pass-1 Hybrid Native Grounding crops (strictly Colab L4 only).
"""
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
MS_DIR = ROOT / "results/vlm_multiscale_cot"
HF_ADAPTER = ROOT / "results/vlm_hier_fovea/ep1_archive/ckpt_ep1/adapter"
OUT_DIR = ROOT / "results/vlm_test300_fovea_showdown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = OUT_DIR / "transfer"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    bundle = OUT_DIR / "test300_fovea_input.zip"
    with ZipFile(MS_DIR / "test300_multiscale_input.zip") as z_in, ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("official-test-300.zip", z_in.read("official-test-300.zip"), compress_type=ZIP_STORED)
        z.writestr("country_names.json", z_in.read("country_names.json"))
        z.write(MS_DIR / "official_test_300_multiscale.jsonl", "official_test_300_multiscale.jsonl")
        for fp in sorted(HF_ADAPTER.glob("*")):
            if fp.is_file():
                z.write(fp, f"hier_ep1_adapter/{fp.name}", compress_type=ZIP_STORED if fp.suffix == ".safetensors" else ZIP_DEFLATED)
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_step0_grounding_attention.py",
            ROOT / "scripts/run_dir2_harvest_crops.py",
            ROOT / "scripts/train_hier_fovea_lora.py",
            ROOT / "scripts/run_locked_test300_fovea_showdown.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)

    # Split into 10 MB chunks (proven reliable with Colab CLI upload)
    subprocess.run(["split", "-b", "10m", "-a", "2", str(bundle), str(transfer / "part-")], check=True)
    parts = sorted(transfer.glob("part-*"))
    print(f"Split into {len(parts)} parts in {transfer}", flush=True)

    drive_py = f"""import hashlib, os, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
root = Path("/content/t300_fov")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("t300f-part-*"))
bundle = root / "test300_fovea_input.zip"
if not bundle.exists():
    with bundle.open("wb") as dst:
        for p in parts:
            with p.open("rb") as src:
                shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Locked test_300 Fovea bundle verified:", actual, flush=True)
if not (root / "run_locked_test300_fovea_showdown.py").exists():
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
run([sys.executable, "-u", "run_locked_test300_fovea_showdown.py", str(root), str(res)])

out_zip = root / "test300_fovea_results.zip"
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
name="woe-t300-fovea"
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
    "$cli" upload --session "$name" "$part" "/content/t300f-${{part##*/}}"
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
"$cli" download --session "$name" /content/t300_fov/test300_fovea_results.zip "$target/test300_fovea_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
