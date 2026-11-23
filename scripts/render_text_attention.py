"""Score the three text case studies and render attention over the original images."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps

from phase1 import haversine
from sft_colab import parse_response

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/text_attention"
TITLES = {
    "canada": "Canada / dashcam coordinates",
    "guadeloupe": "Guadeloupe / French trail sign",
    "brazil": "Brazil / Portuguese garage sign",
}
MODES = {"base": "Base Qwen", "sft": "Location LoRA"}
MAPS = {"geo_full": "Location prompt", "ocr_full": "Transcription prompt"}
COLORS = {"text": "#27ecff", "control": "#f1f5f9"}


def overlap_weights(box, shape):
    height, width = shape
    x0, y0, x1, y1 = box
    x = np.arange(width) / width
    y = np.arange(height) / height
    dx = np.maximum(0, np.minimum(x + 1 / width, x1) - np.maximum(x, x0))
    dy = np.maximum(0, np.minimum(y + 1 / height, y1) - np.maximum(y, y0))
    return np.outer(dy * height, dx * width)


def attention_stats(weights, box, control):
    heat = weights.mean(axis=0)
    mass = float(heat.sum())
    if not 0 < mass <= 1.0001:
        raise ValueError(f"Invalid image-token attention mass: {mass}")
    density = heat / mass
    text_weights = overlap_weights(box, heat.shape)
    control_weights = overlap_weights(control, heat.shape)
    return {
        "image_mass": mass,
        "text_image_relative_mass": float(np.sum(density * text_weights)),
        "control_image_relative_mass": float(np.sum(density * control_weights)),
        "text_area": float(text_weights.mean()),
        "control_area": float(control_weights.mean()),
    }, density * heat.size


def draw_region(ax, box, size, kind):
    w, h = size
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                           fill=False, linewidth=1.2, linestyle="--" if kind == "control" else "-",
                           edgecolor=COLORS[kind]))


def draw_image(ax, image):
    w, h = image.size
    ax.imshow(image, extent=(0, w, h, 0))
    ax.axis("off")


def render(case, reference, photo, stats, densities):
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 5.9), facecolor="#111827")
    figure.subplots_adjust(left=0.025, right=0.975, top=0.85, bottom=0.13, hspace=0.32, wspace=0.045)
    figure.text(0.025, 0.944, TITLES[case["case"]], color="#f8fafc", fontsize=22, weight="bold")
    figure.text(0.025, 0.895, "Attention on image tokens while generating digits or transcribed text",
                color="#cbd5e1", fontsize=11)
    w, h = photo.size
    title_style = dict(color="#f8fafc", fontsize=11, loc="left", pad=6)
    draw_image(axes[0, 0], photo)
    axes[0, 0].set_title("Original image", **title_style)
    draw_region(axes[0, 0], case["text_box"], photo.size, "text")
    draw_region(axes[0, 0], case["control_box"], photo.size, "control")
    crop = photo.crop(tuple(round(v * n) for v, n in zip(case["crop_box"], (w, h, w, h))))
    draw_image(axes[1, 0], crop)
    axes[1, 0].set_title("Text crop for inspection", **title_style)

    scale_max = max(2.0, float(np.percentile(np.concatenate([a.ravel() for a in densities.values()]), 98)))
    for i, (model, model_label) in enumerate(MODES.items()):
        for j, (variant, label) in enumerate(MAPS.items(), 1):
            ax = axes[i, j]
            density = densities[model, variant]
            draw_image(ax, photo)
            opacity = np.clip((density / scale_max) ** 0.7 * 0.75, 0, 0.75)
            ax.imshow(density, cmap="inferno", interpolation="nearest", vmin=0, vmax=scale_max,
                      alpha=opacity, extent=(0, w, h, 0))
            draw_region(ax, case["text_box"], photo.size, "text")
            ax.set_title(f"{model_label} / {label}", **title_style)
            record = stats[model, variant]
            row = (f"Image  {record['image_mass']:.1%}     "
                   f"Text  {record['text_image_relative_mass']:.1%} of image attention")
            ax.text(0, -0.095, row, transform=ax.transAxes, color="#d1d5db", fontsize=9)
    figure.text(0.025, 0.036, "Cyan: text region   White dashed: equal-area control   "
                "Same heat scale across the four maps within this image",
                color="#cbd5e1", fontsize=9)
    figure.text(0.025, 0.008, f"Photo: {reference['attribution']} / OSV-5M / {reference['license']}",
                color="#9ca3af", fontsize=8)
    path = OUT / "maps" / f"{case['case']}.png"
    path.parent.mkdir(exist_ok=True)
    figure.savefig(path, dpi=145, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    roi_manifest = (OUT / "cases.json").read_bytes()
    cases = json.loads(roi_manifest)
    references = json.loads((OUT / "references.json").read_text())
    with ZipFile(OUT / "text-attention-results.zip") as result:
        config = json.loads(result.read("configuration.json"))
        if hashlib.sha256(roi_manifest).hexdigest() != config["case_sha256"]:
            raise ValueError("Result inputs differ from the reviewed cases")
        answers = [json.loads(line) for line in result.open("answers.jsonl")]
        with np.load(io.BytesIO(result.read("attention.npz")), allow_pickle=False) as data:
            maps = {key: data[key] for key in data.files}
    rows = {(r["case"], r["mode"], r["variant"]): r for r in answers}
    variants = [v[0] for v in config["views"]]
    if len(rows) != len(answers) or len(rows) != len(cases) * len(variants) * len(MODES):
        raise ValueError("Incomplete or duplicate answers")
    report = {"configuration": config, "cases": {}}
    with ZipFile(OUT / "inputs.zip") as source:
        if hashlib.sha256(source.read("adapter/adapter_model.safetensors")).hexdigest() != config["adapter_sha256"]:
            raise ValueError("Unexpected adapter")
        for case in cases:
            name = case["case"]
            reference = references[name]
            photo = ImageOps.exif_transpose(Image.open(io.BytesIO(source.read(case["image"])))).convert("RGB")
            photo.thumbnail((768, 768), Image.Resampling.LANCZOS)
            stats, densities, records = {}, {}, {}
            for mode in MODES:
                records[mode] = {}
                for variant in variants:
                    row = rows[name, mode, variant]
                    record = {"answer": row["answer"]}
                    if list(photo.size) != row["image_size"]:
                        raise ValueError(f"Photo dimensions differ for {name}")
                    if variant.startswith("geo_"):
                        country, coords = parse_response(row["answer"])
                        record.update(country=country, coordinates=coords, distance_km=None)
                        if coords is not None:
                            record["distance_km"] = haversine(reference["latitude"], reference["longitude"], *coords)
                    if variant in MAPS:
                        weight = maps[row["map_key"]]
                        if list(weight.shape) != [len(config["layers_zero_based"]), *row["attention"]["grid"]]:
                            raise ValueError(f"Unexpected grid shape for {name}, {mode}, {variant}")
                        stat, density = attention_stats(weight, case["text_box"], case["control_box"])
                        stats[mode, variant] = stat
                        densities[mode, variant] = density
                        record.update(attention=stat, attention_steps=row["attention"]["answer_tokens"])
                    records[mode][variant] = record
            render(case, reference, photo, stats, densities)
            report["cases"][name] = {"reference": reference, "human_transcription": case["transcription"],
                                     "predictions": records}

    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, details in report["cases"].items():
        print("\n", name, details["human_transcription"])
        for mode, variants in details["predictions"].items():
            print(mode)
            for variant, row in variants.items():
                stat = row.get("attention")
                roi = (f" | image {stat['image_mass']:.1%}, text {stat['text_image_relative_mass']:.1%}, "
                       f"control {stat['control_image_relative_mass']:.1%}") if stat else ""
                distance = f" | {row['distance_km']:.1f} km" if row.get("distance_km") is not None else ""
                print(f"  {variant:17} {row['answer']!r}{distance}{roi}")


if __name__ == "__main__":
    main()
