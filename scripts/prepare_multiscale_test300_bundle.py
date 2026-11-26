"""Prepare Colab bundle for the single locked zero-touch evaluation on official_test_300
(300 unseen photographers / 147 countries) using Top-8 Multi-Scale Spatial KDE clusters
and the single winning QLoRA adapter chosen strictly on author_holdout_66.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from multiscale_vlm_utils import build_multiscale_clusters, compute_kde_log_scores

ROOT = Path("/Volumes/PortableSSD/Codex/GeoGuessr-Recognition")
SC25K_DIR = ROOT / "results/streetclip_25k"
TEST_ZIP = ROOT / "data/official_test_300/official-test-300.zip"
OUT_DIR = ROOT / "results/vlm_multiscale_cot"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-zip", type=Path, required=True, help="ZIP containing the winning adapter")
    parser.add_argument("--adapter-prefix", required=True, help="Folder prefix inside adapter-zip (e.g. adapter_sft or adapter_dpo_balanced)")
    parser.add_argument("--adapter-label", required=True, help="Human-readable label for report")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transfer = OUT_DIR / "transfer_test300"
    if transfer.exists():
        shutil.rmtree(transfer)
    transfer.mkdir(parents=True)

    with ZipFile(SC25K_DIR / "phase_b_input.zip") as z:
        cc_names = json.loads(z.read("country_names.json"))
    with ZipFile(SC25K_DIR / "manifests_39k.zip") as z:
        gal = [json.loads(l) for l in z.read("gallery_images.jsonl").splitlines()]
    with ZipFile(SC25K_DIR / "streetclip_25k_results.zip") as z:
        with z.open("streetclip_25k_top100.npz") as f:
            sc_top = dict(np.load(f))
    with ZipFile(TEST_ZIP) as z:
        test_rows = [json.loads(l) for l in z.read("test.jsonl").splitlines()]

    g_lat = np.array([r["latitude"] for r in gal], dtype=np.float64)
    g_lon = np.array([r["longitude"] for r in gal], dtype=np.float64)
    g_cc = np.array([r["country"] for r in gal])
    cc_vocab = [str(c) for c in sc_top["country_vocab"]]

    # Exact bit-for-bit Stage 1 GeoProjector Top-100 on official_test_300
    test_ids = [str(x) for x in sc_top["test_ids"]]
    assert [r["id"] for r in test_rows] == test_ids, "Order mismatch with streetclip_25k_top100.npz test_ids"
    top_idx = sc_top["test_proj_top"]
    top_sim = sc_top["test_proj_sim"]
    txt_lp = sc_top["test_txt_lp"]

    kde_log, pw_d = compute_kde_log_scores(top_idx, top_sim, g_lat, g_lon)
    out_rows = []
    for i, r in enumerate(test_rows):
        dist_info = build_multiscale_clusters(
            top_idx[i], top_sim[i], kde_log[i], pw_d[i], txt_lp[i],
            g_lat, g_lon, g_cc, cc_vocab, cc_names,
        )
        out_rows.append({
            "id": r["id"],
            "image": r["image"],
            "sha256": r["sha256"],
            "country": r["country"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "kde_distribution": dist_info,
        })
    test_jsonl = OUT_DIR / "official_test_300_multiscale.jsonl"
    test_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
    print(f"Wrote {len(out_rows)} Top-8 multi-scale distributions -> {test_jsonl}", flush=True)

    bundle = OUT_DIR / "test300_multiscale_input.zip"
    prefix = args.adapter_prefix.rstrip("/") + "/"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(TEST_ZIP, "official-test-300.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", json.dumps(cc_names, ensure_ascii=False) + "\n")
        z.write(test_jsonl, "official_test_300_multiscale.jsonl")
        copied = 0
        with ZipFile(args.adapter_zip) as az:
            for nm in az.namelist():
                if nm.startswith(prefix) and not nm.endswith("/"):
                    rel = nm[len(prefix):]
                    z.writestr(f"locked_adapter/{rel}", az.read(nm))
                    copied += 1
        if copied == 0:
            raise ValueError(f"No adapter files matched prefix {prefix!r} in {args.adapter_zip}")
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/multiscale_vlm_utils.py",
            ROOT / "scripts/run_qwen8b_multiscale_test300.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle} ({bundle.stat().st_size / 1e6:.2f} MB, {copied} adapter files), sha256={sha}", flush=True)

    subprocess.run(["split", "-b", "10m", "-a", "2", str(bundle), str(transfer / "part-")], check=True)
    parts = sorted(transfer.glob("part-*"))
    print(f"Split into {len(parts)} parts in {transfer}", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/test300_ms")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("t300ms-part-*"))
bundle = root / "test300_multiscale_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Test-300 bundle verified:", actual, flush=True)
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
    sys.executable, "-u", "run_qwen8b_multiscale_test300.py",
    str(root), str(res),
    "--adapter-dir", "locked_adapter",
    "--adapter-label", "{args.adapter_label}",
])

out_zip = root / "test300_multiscale_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in (
        "test300_eval_report.json",
        "locked_finetuned_8b_test300_top8.jsonl",
        "base_8b_test300_top8.jsonl",
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
name="woe-qwen8b-test300-ms"
target="{OUT_DIR}"
cleanup() {{
    "$cli" stop --session "$name" || "$cli" sessions
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assigned=0
for gpu_try in 1 2 3 4 5; do
    echo "=== GPU assign attempt $gpu_try (trying {args.gpu} -> L4 -> T4) ==="
    if "$cli" new --session "$name" --gpu {args.gpu} < /dev/null; then
        assigned=1
        break
    fi
    if "$cli" new --session "$name" --gpu L4 < /dev/null; then
        assigned=1
        break
    fi
    if [ "$gpu_try" -ge 3 ]; then
        if "$cli" new --session "$name" --gpu T4 < /dev/null; then
            assigned=1
            break
        fi
    fi
    echo "GPUs busy, waiting 25s before retry..."
    sleep 25
done
if [ "$assigned" -eq 0 ]; then
    echo "Could not acquire GPU."
    exit 1
fi
for part in "$target"/transfer_test300/part-*; do
    "$cli" upload --session "$name" "$part" "/content/t300ms-${{part##*/}}"
done
for attempt in 1 2 3; do
    echo "=== exec attempt $attempt ==="
    if "$cli" exec --session "$name" --timeout {args.timeout} --file "$target/transfer_test300/drive_colab.py" < /dev/null; then
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "All 3 exec attempts failed."
        exit 1
    fi
    echo "WebSocket hiccup, retrying in 10s (incremental JSONL resume enabled)..."
    sleep 10
done
"$cli" download --session "$name" /content/test300_ms/test300_multiscale_results.zip "$target/test300_multiscale_results.zip"
"""
    sh = transfer / "run.sh"
    sh.write_text(run_sh, encoding="utf-8")
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
