"""Prepare compact, SHA-256-verified Colab upload bundle for Step 1B (25k Geo-Ranker) + Step 1C (VLM Country Priors)."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
PAIRS_ZIP = ROOT / "results/ranker-2018-pairs.zip"
QLORA_6K_RESULTS = ROOT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
QLORA_6K_UPLOAD = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
AH_UPLOAD = ROOT / "data/author_holdout/author-holdout-eval.zip"
OUT = ROOT / "results/retrieval_v2"
TRANSFER = OUT / "transfer_1bc"

KEEP_KEYS = ("id", "latitude", "longitude", "country", "attribution")


def compact_jsonl(raw_bytes):
    lines = []
    for line in raw_bytes.splitlines():
        r = json.loads(line)
        lines.append(json.dumps({k: r.get(k, "") for k in KEEP_KEYS}, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if TRANSFER.exists():
        shutil.rmtree(TRANSFER)
    TRANSFER.mkdir(parents=True)

    # 1. Compact pairs_compact.zip (only features + minimal metadata; drops 67MB candidate jsonls!)
    compact_pairs = OUT / "pairs_compact.zip"
    all_countries = set()
    with ZipFile(PAIRS_ZIP) as zin, ZipFile(compact_pairs, "w", compression=ZIP_DEFLATED) as zout:
        for name in ("gallery_images.jsonl", "train_images.jsonl", "validation_images.jsonl"):
            raw = zin.read(name)
            for l in raw.splitlines():
                cc = json.loads(l).get("country", "")
                if len(cc) == 2 and cc.isalpha():
                    all_countries.add(cc.upper())
            zout.writestr(name, compact_jsonl(raw))
        for name in ("gallery_features.npz", "train_features.npz", "validation_features.npz"):
            zout.writestr(name, zin.read(name), compress_type=ZIP_STORED)
    print(f"Built {compact_pairs.name} ({compact_pairs.stat().st_size / 1e6:.2f} MB), {len(all_countries)} countries", flush=True)

    # 2. Collect 300 qlora eval + 66 author_holdout JPEGs (identical 512px bytes used by the model!)
    with ZipFile(QLORA_6K_UPLOAD) as z:
        q300_rows = [json.loads(l) for l in z.read("eval.jsonl").splitlines()]
        q300_jpegs = {r["id"]: z.read(r["image"]) for r in q300_rows}
    with ZipFile(AH_UPLOAD) as z:
        ah_name = [n for n in z.namelist() if n.endswith(".jsonl")][0]
        ah_rows = [json.loads(l) for l in z.read(ah_name).splitlines()]
        ah_jpegs = {r["id"]: z.read(r["image"]) for r in ah_rows}

    assert len(set(q300_jpegs) & set(ah_jpegs)) == 0, "Unexpected overlap between q300 and author_holdout"

    splits_json = {
        "qlora_300_ids": [r["id"] for r in q300_rows],
        "author_holdout_66_ids": [r["id"] for r in ah_rows],
    }

    manifest = []
    for split, rows, jmap in (("qlora_eval_300", q300_rows, q300_jpegs), ("author_holdout_66", ah_rows, ah_jpegs)):
        for r in rows:
            jpeg = jmap[r["id"]]
            sha = hashlib.sha256(jpeg).hexdigest()
            manifest.append({
                "id": r["id"],
                "split": split,
                "country": r["country"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "image": f"images/{r['id']}.jpg",
                "sha256": sha,
            })

    # 3. Assemble step1bc_input.zip
    bundle = OUT / "step1bc_input.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as zout:
        zout.write(compact_pairs, "pairs_compact.zip", compress_type=ZIP_STORED)
        zout.writestr("splits.json", json.dumps(splits_json, indent=2) + "\n")
        zout.writestr("vlm_input/manifest.json", json.dumps(manifest, indent=2) + "\n")
        zout.writestr("vlm_input/countries.json", json.dumps(sorted(all_countries)) + "\n")
        for r in manifest:
            rid = r["id"]
            jpeg = q300_jpegs[rid] if rid in q300_jpegs else ah_jpegs[rid]
            zout.writestr(f"vlm_input/{r['image']}", jpeg, compress_type=ZIP_STORED)
        # Include 6k QLoRA adapter
        with ZipFile(QLORA_6K_RESULTS) as za:
            for member in ("adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
                zout.writestr(f"vlm_input/{member}", za.read(member), compress_type=ZIP_STORED)
        # Include scripts & pinned Colab requirements
        for src in (
            ROOT / "requirements-colab.txt",
            ROOT / "scripts/phase1.py",
            ROOT / "scripts/sft_colab.py",
            ROOT / "scripts/train_geo_ranker.py",
            ROOT / "scripts/extract_vlm_country_priors.py",
        ):
            zout.write(src, src.name)

    sha_bundle = hashlib.sha256(bundle.read_bytes()).hexdigest()
    print(f"Built {bundle.name} ({bundle.stat().st_size / 1e6:.2f} MB), sha256={sha_bundle}", flush=True)

    # Split into 5 MB chunks for reliable colab upload
    subprocess.run(["split", "-b", "5m", "-a", "2", str(bundle), str(TRANSFER / "part-")], check=True)
    parts = sorted(TRANSFER.glob("part-*"))
    print(f"Split into {len(parts)} parts in {TRANSFER}", flush=True)

    # Write Colab driver script
    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/content/retrieval_v2")
root.mkdir(exist_ok=True)
parts = sorted(Path("/content").glob("retv2-part-*"))
if not parts:
    raise RuntimeError("No uploaded parts found")
bundle = root / "step1bc_input.zip"
with bundle.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
if actual != "{sha_bundle}":
    raise RuntimeError(f"Bundle SHA-256 mismatch: {{actual}} != {sha_bundle}")
print("Bundle verified:", actual, flush=True)
with ZipFile(bundle) as z:
    z.extractall(root)

def run(cmd):
    with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, cmd)

print("=== Step 1B: Training leak-free 25k Geo-Projection + Listwise Ranker ===", flush=True)
res = root / "results"
res.mkdir(exist_ok=True)
run([sys.executable, "-u", "train_geo_ranker.py", str(root / "pairs_compact.zip"),
     str(root / "splits.json"), str(res)])

print("=== Installing pinned VLM deps for Step 1C ===", flush=True)
run([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements-colab.txt"])

print("=== Step 1C: Extracting exact Qwen3-VL 6k LoRA country log-prob priors ===", flush=True)
run([sys.executable, "-u", "extract_vlm_country_priors.py", str(root / "vlm_input"), str(res)])

out_zip = root / "step1bc_results.zip"
with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
    for fname in ("step1b_ranker_results.json", "val_projected_top100.npz", "geo_ranker_v2.pt", "vlm_country_priors.json"):
        p = res / fname
        if p.exists():
            z.write(p, fname)
print(f"Exported {{out_zip}} ({{out_zip.stat().st_size / 1e6:.2f}} MB)", flush=True)
"""
    (TRANSFER / "drive_colab.py").write_text(drive_py)

    run_sh = """#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="geoguessr-retv2-1bc-20261123"
target="/Volumes/PortableSSD/codex/GeoGuessr-Recognition/results/retrieval_v2"
cleanup() {
    "$cli" stop --session "$name" || "$cli" sessions
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$cli" new --session "$name" --gpu T4
for part in "$target"/transfer_1bc/part-*; do
    "$cli" upload --session "$name" "$part" "/content/retv2-${part##*/}"
done
"$cli" exec --session "$name" --timeout 900 --file "$target/transfer_1bc/drive_colab.py"
"$cli" download --session "$name" /content/retrieval_v2/step1bc_results.zip "$target/step1bc_results.zip"
"""
    sh_path = TRANSFER / "run.sh"
    sh_path.write_text(run_sh)
    sh_path.chmod(0o755)
    print(f"Ready: {sh_path}", flush=True)


if __name__ == "__main__":
    main()
