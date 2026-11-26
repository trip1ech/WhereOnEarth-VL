"""Prepare compact Colab bundle (~4 MB, 1 part!) for Step 0:
Native Grounding + Visual Cross-Attention Heatmaps + Isolated Cue-Crop Probe + Visual Fovea Test
on author_holdout_66.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
MS_DIR = ROOT / "results/vlm_multiscale_cot"
OUT_DIR = ROOT / "results/vlm_cue_grounding_step0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = OUT_DIR / "transfer"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names_bytes = z.read("country_names.json")

    bundle = OUT_DIR / "step0_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", cc_names_bytes)
        z.write(MS_DIR / "author_holdout_multiscale.jsonl", "author_holdout_multiscale.jsonl")
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_step0_grounding_attention.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/step0_diag")
root.mkdir(exist_ok=True)
bundle = Path("/content/step0_input.zip")
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Step 0 bundle verified:", actual, flush=True)
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
run([sys.executable, "-u", "run_step0_grounding_attention.py", str(root), str(res)])

out_zip = root / "step0_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in (
        "step0_summary.json",
        "grounding_boxes_ah66.jsonl",
        "attention_maps_ah66.jsonl",
        "crop_probe_ah66.jsonl",
        "fovea_8b_author_holdout_top8.jsonl",
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
name="woe-step0-grounding"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! "$cli" new --session "$name" --gpu {args.gpu} < /dev/null; then
    echo "Fallback GPU assignment -> trying T4..."
    "$cli" new --session "$name" --gpu T4 < /dev/null
fi
"$cli" upload --session "$name" "$target/step0_input.zip" "/content/step0_input.zip"
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
"$cli" download --session "$name" /content/step0_diag/step0_results.zip "$target/step0_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
