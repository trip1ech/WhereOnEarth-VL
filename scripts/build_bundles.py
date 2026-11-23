"""Create two separately uploadable Colab bundles on the SSD."""
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pilot = root / "data/pilot"
for split in ("pilot_dev", "pilot_holdout"):
    manifest = pilot / f"{split}.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    target = root / f"geoguessr-{split}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "scripts/phase1.py", "phase1.py")
        archive.write(manifest, f"data/{manifest.name}")
        for row in rows:
            archive.write(pilot / row["image"], "data/" + row["image"])
    print(f"{target.name}: {len(rows)} images, {target.stat().st_size:,} bytes")
