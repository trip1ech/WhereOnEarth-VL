"""Prepare compact (~22 MB) Colab T4/L4 bundle for Phase C:
Official Test 300 Showdown (osv5m/baseline + Qwen3-VL zero-shot & 6k QLoRA + exact country priors + dinov2-small).
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
PAIRS_ZIP = ROOT / "results/retrieval_v2/pairs_compact.zip"
QLORA_6K_RESULTS = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
TEST_ZIP = ROOT / "data/official_test_300/official-test-300.zip"
OUT = ROOT / "results/official_test_showdown"
TRANSFER = OUT / "transfer_c"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if TRANSFER.exists():
        shutil.rmtree(TRANSFER)
    TRANSFER.mkdir(parents=True)

    # Full country vocabulary across 25k gallery + 12k train + 2k val + official_test_300
    all_ccs = set()
    with ZipFile(PAIRS_ZIP) as z:
        for name in ("gallery_images.jsonl", "train_images.jsonl", "validation_images.jsonl"):
            for l in z.read(name).splitlines():
                cc = json.loads(l).get("country", "")
                if len(cc) == 2 and cc.isalpha():
                    all_ccs.add(cc.upper())
    with ZipFile(TEST_ZIP) as z:
        for l in z.read("test.jsonl").splitlines():
            cc = json.loads(l).get("country", "")
            if len(cc) == 2 and cc.isalpha():
                all_ccs.add(cc.upper())

    bundle = OUT / "phase_c_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(TEST_ZIP, "official-test-300.zip", compress_type=ZIP_STORED)
        z.writestr("countries.json", json.dumps(sorted(all_ccs)) + "\n")
        with ZipFile(QLORA_6K_RESULTS) as za:
            for member in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
                z.writestr(member, za.read(member), compress_type=ZIP_STORED)
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/phase1.py",
            ROOT / "scripts/sft_colab.py",
            ROOT / "scripts/retrieval.py",
            ROOT / "scripts/extract_vlm_country_priors.py",
            ROOT / "scripts/evaluate_official_test_models.py",
        ):
            z.write(src, src.name)

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle.name} ({bundle.stat().st_size / 1e6:.2f} MB), {len(all_ccs)} countries, sha256={sha}", flush=True)
    subprocess.run(["split", "-b", "5m", "-a", "2", str(bundle), str(TRANSFER / "part-")], check=True)
    parts = sorted(TRANSFER.glob("part-*"))
    print(f"Split into {len(parts)} parts in {TRANSFER}", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/official_test")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("otest-part-*"))
bundle = root / "phase_c_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Phase C bundle verified:", actual, flush=True)
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
run([sys.executable, "-u", "evaluate_official_test_models.py", str(root), str(res)])

out_zip = root / "phase_c_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in ("osv5m_baseline.jsonl", "qwen_zero_shot.jsonl", "qwen_sft_6000.jsonl",
               "vlm_country_priors_test300.json", "dinov2_test300.npz"):
        z.write(res / fn, fn)
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (TRANSFER / "drive_colab.py").write_text(drive_py)

    run_sh = """#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="geoguessr-otest-20261123"
target="/Volumes/PortableSSD/codex/GeoGuessr-Recognition/results/official_test_showdown"
cleanup() {
    "$cli" stop --session "$name" || "$cli" sessions
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$cli" new --session "$name" --gpu T4
for part in "$target"/transfer_c/part-*; do
    "$cli" upload --session "$name" "$part" "/content/otest-${part##*/}"
done
"$cli" exec --session "$name" --timeout 900 --file "$target/transfer_c/drive_colab.py"
"$cli" download --session "$name" /content/official_test/phase_c_results.zip "$target/phase_c_results.zip"
"""
    sh = TRANSFER / "run.sh"
    sh.write_text(run_sh)
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
