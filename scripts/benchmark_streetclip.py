"""Step 1D: Ceiling-raising benchmark comparing geolocal/StreetCLIP (full & HUD-cropped) vs dinov2-small
on the clean 6,000-image gallery against author_holdout_66 (0% creator overlap) and qlora_eval_300 (diff-creator).

Also extracts StreetCLIP zero-shot country text-probe probabilities and trains a creator-disjoint
Geo-Metric Projection on StreetCLIP embeddings to test how high the true leak-free ceiling goes.
"""
import argparse
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

from phase1 import clean_image

STREETCLIP = "geolocal/StreetCLIP"


def crop_bottom(image, keep_ratio=0.88):
    w, h = image.size
    return image.crop((0, 0, w, max(32, int(round(h * keep_ratio)))))


@torch.inference_mode()
def encode_images(model, processor, paths, *, batch_size=64, bottom_keep=1.0, label="full"):
    chunks = []
    n = len(paths)
    for start in range(0, n, batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs = [clean_image(p) for p in batch_paths]
        if bottom_keep < 1.0:
            imgs = [crop_bottom(im, bottom_keep) for im in imgs]
        inputs = processor(images=imgs, return_tensors="pt").to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feats = model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        feats = F.normalize(feats.float(), dim=-1)
        chunks.append(feats.cpu().numpy().astype(np.float16))
        if start == 0 or (start // batch_size) % 15 == 0 or start + batch_size >= n:
            print(f"StreetCLIP ({label}): {min(start + batch_size, n)}/{n}", flush=True)
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def encode_country_texts(model, processor, country_names):
    prompts = [f"A Street View photo in {name}" for name in country_names]
    inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        feats = model.get_text_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
    feats = F.normalize(feats.float(), dim=-1)
    scale = float(model.logit_scale.exp().detach().cpu())
    return feats.cpu().numpy().astype(np.float32), scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sft_zip", type=Path, help="geoguessr_sft_6000_upload.zip (6000 train + 300 eval)")
    parser.add_argument("ah_zip", type=Path, help="author-holdout-eval.zip (66 unseen creators)")
    parser.add_argument("country_names_json", type=Path, help="JSON mapping ISO2 -> English country name")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    work = args.output / "work_imgs"
    work.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.sft_zip) as z:
        train_rows = [json.loads(l) for l in z.read("train.jsonl").splitlines()]
        eval_rows = [json.loads(l) for l in z.read("eval.jsonl").splitlines()]
        for row in train_rows + eval_rows:
            dest = work / f"{row['id']}.jpg"
            if not dest.exists():
                dest.write_bytes(z.read(row["image"]))

    with ZipFile(args.ah_zip) as z:
        ah_name = [n for n in z.namelist() if n.endswith(".jsonl")][0]
        ah_rows = [json.loads(l) for l in z.read(ah_name).splitlines()]
        for row in ah_rows:
            dest = work / f"{row['id']}.jpg"
            if not dest.exists():
                dest.write_bytes(z.read(row["image"]))

    # Deduplicate paths to encode each unique image once
    all_rows = train_rows + eval_rows + ah_rows
    unique_ids = list(dict.fromkeys(r["id"] for r in all_rows))
    paths = [work / f"{i}.jpg" for i in unique_ids]
    print(f"Loading {STREETCLIP} for {len(unique_ids)} unique images...", flush=True)

    processor = CLIPProcessor.from_pretrained(STREETCLIP)
    model = CLIPModel.from_pretrained(STREETCLIP).to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", "unknown")
    print(f"Loaded {STREETCLIP} (commit={rev})", flush=True)

    feats_full = encode_images(model, processor, paths, batch_size=64, bottom_keep=1.0, label="full")
    feats_crop88 = encode_images(model, processor, paths, batch_size=64, bottom_keep=0.88, label="crop88")

    cc_map = json.loads(args.country_names_json.read_text())
    codes = sorted(cc_map.keys())
    names = [cc_map[c] for c in codes]
    text_feats, logit_scale = encode_country_texts(model, processor, names)

    args.output.mkdir(parents=True, exist_ok=True)
    out_npz = args.output / "streetclip_features.npz"
    np.savez_compressed(
        out_npz,
        ids=np.array(unique_ids),
        feats_full=feats_full,
        feats_crop88=feats_crop88,
        country_codes=np.array(codes),
        text_feats=text_feats,
        logit_scale=np.array([logit_scale], dtype=np.float32),
    )
    print(f"Saved {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
