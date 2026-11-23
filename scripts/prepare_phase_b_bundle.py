"""Prepare ultra-compact (~28 MB) Colab L4 bundle for Phase B:
Full 39,000 + official_test_300 StreetCLIP encoding + 25k Diff-Creator Geo-Projection & Listwise Ranker.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from prepare_step1d_bundle import COUNTRY_NAMES

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
PAIRS_ZIP = ROOT / "results/ranker-2018-pairs.zip"
QLORA_6K_UPLOAD = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
AH_EVAL = ROOT / "data/author_holdout/eval.jsonl"
TEST_ZIP = ROOT / "data/official_test_300/official-test-300.zip"
SC_6K_NPZ = ROOT / "results/retrieval_v2/streetclip_features.npz"
OUT = ROOT / "results/streetclip_25k"
TRANSFER = OUT / "transfer_b"

KEEP_KEYS = ("id", "latitude", "longitude", "country", "attribution", "archive_member", "sha256")


def compact_jsonl(raw):
    out = []
    for line in raw.splitlines():
        r = json.loads(line)
        out.append(json.dumps({k: r.get(k, "") for k in KEEP_KEYS}, separators=(",", ":")))
    return ("\n".join(out) + "\n").encode("utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if TRANSFER.exists():
        shutil.rmtree(TRANSFER)
    TRANSFER.mkdir(parents=True)

    # 1. Compact manifests_39k.zip
    man_zip = OUT / "manifests_39k.zip"
    with ZipFile(PAIRS_ZIP) as zin, ZipFile(man_zip, "w", compression=ZIP_DEFLATED) as zout:
        for name in ("gallery_images.jsonl", "train_images.jsonl", "validation_images.jsonl"):
            zout.writestr(name, compact_jsonl(zin.read(name)))

    # 2. Seed StreetCLIP 6k full-view embeddings only (~9 MB)
    sc = np.load(SC_6K_NPZ)
    seed_npz = OUT / "streetclip_6k_seed.npz"
    np.savez_compressed(seed_npz, ids=sc["ids"], feats_full=sc["feats_full"])

    # 3. Splits JSON
    with ZipFile(QLORA_6K_UPLOAD) as z:
        q300_ids = [json.loads(l)["id"] for l in z.read("eval.jsonl").splitlines()]
    ah_ids = [json.loads(l)["id"] for l in AH_EVAL.read_text().splitlines()]
    splits = {"qlora_300_ids": q300_ids, "author_holdout_66_ids": ah_ids}

    # 4. Assemble phase_b_input.zip
    bundle = OUT / "phase_b_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as z:
        z.write(man_zip, "manifests_39k.zip", compress_type=ZIP_STORED)
        z.write(TEST_ZIP, "official-test-300.zip", compress_type=ZIP_STORED)
        z.write(seed_npz, "streetclip_6k_seed.npz", compress_type=ZIP_STORED)
        z.writestr("splits.json", json.dumps(splits, indent=2) + "\n")
        z.writestr("country_names.json", json.dumps(COUNTRY_NAMES, indent=2) + "\n")
        z.write(ROOT / "scripts/encode_and_train_streetclip_25k.py", "encode_and_train_streetclip_25k.py")

    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle.name} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha}", flush=True)
    subprocess.run(["split", "-b", "5m", "-a", "2", str(bundle), str(TRANSFER / "part-")], check=True)
    parts = sorted(TRANSFER.glob("part-*"))
    print(f"Split into {len(parts)} parts in {TRANSFER}", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/streetclip_25k")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("sc25k-part-*"))
bundle = root / "phase_b_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha}":
    raise RuntimeError(f"SHA mismatch: {{actual}} != {sha}")
print("Phase B bundle verified:", actual, flush=True)
with ZipFile(bundle) as z:
    z.extractall(root)

res = root / "results"
res.mkdir(exist_ok=True)
cmd = [sys.executable, "-u", "encode_and_train_streetclip_25k.py", str(root), str(res), "--batch-size", "128"]
with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, errors="replace", bufsize=1) as proc:
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, cmd)

small_zip = root / "streetclip_25k_results.zip"
with ZipFile(small_zip, "w", compression=ZIP_DEFLATED) as z:
    for fn in ("streetclip_25k_results.json", "streetclip_25k_top100.npz", "streetclip_25k_ranker.pt"):
        z.write(res / fn, fn)
print(f"Exported {{small_zip}} ({{small_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (TRANSFER / "drive_colab.py").write_text(drive_py)

    run_sh = """#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="geoguessr-sc25k-20261123"
target="/Volumes/PortableSSD/codex/GeoGuessr-Recognition/results/streetclip_25k"
cleanup() {
    "$cli" stop --session "$name" || "$cli" sessions
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$cli" new --session "$name" --gpu L4
for part in "$target"/transfer_b/part-*; do
    "$cli" upload --session "$name" "$part" "/content/sc25k-${part##*/}"
done
"$cli" exec --session "$name" --timeout 2400 --file "$target/transfer_b/drive_colab.py"
"$cli" download --session "$name" /content/streetclip_25k/streetclip_25k_results.zip "$target/streetclip_25k_results.zip"
"$cli" download --session "$name" /content/streetclip_25k/results/streetclip_39k_plus_test300.npz "$target/streetclip_39k_plus_test300.npz"
"""
    sh = TRANSFER / "run.sh"
    sh.write_text(run_sh)
    sh.chmod(0o755)
    print(f"Ready: {sh}", flush=True)


if __name__ == "__main__":
    main()
