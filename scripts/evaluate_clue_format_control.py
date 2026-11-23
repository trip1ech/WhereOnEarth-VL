"""Recover harmless formatting from saved baseline evidence and test its geographic use."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from clue_visual import parse_observations, visual_geo_prompt
from evaluate_clue_first import conditioned_prompt, generate, load_for_inference, read_images, read_text


def strip_fence(raw):
    match = re.fullmatch(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", raw, flags=re.I | re.S)
    return match.group(1) if match else raw.strip()


def read_ignoring_trailing_quote(raw):
    return read_text(strip_fence(raw).rstrip('"').rstrip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = (args.input / "inputs.json").read_bytes()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    photos, _, _ = read_images(args.input, json.loads(manifest))
    reports = []
    for source in ("old-text.zip", "roads.zip"):
        with ZipFile(args.input / source) as archive:
            if json.loads(archive.read("configuration.json"))["manifest_sha256"] != manifest_hash:
                raise ValueError("Raw answers used different images")
            reports.append(list(map(json.loads, archive.open("answers.jsonl"))))
    text, visual = reports
    if [r["id"] for r in text] != list(photos) or [r["id"] for r in visual] != list(photos):
        raise ValueError("Raw answer order changed")

    old_text = {r["id"]: read_ignoring_trailing_quote(r["read_full"]) for r in text}
    road = {r["id"]: parse_observations(strip_fence(r["before"]["raw_observations"])) for r in visual}
    model, processor = load_for_inference(args.input / "adapter")
    geo_road = generate(model, processor, [(i, visual_geo_prompt(r)) for i, r in road.items() if r is not None],
                        photos, 4, 40, "format_control_road")
    geo_text = generate(model, processor, [(r["id"], conditioned_prompt(old_text[r["id"]][1])) for r in text
                                          if not r["used_model_text"] and old_text[r["id"]][1]],
                        photos, 4, 40, "format_control_text")
    rows = [{
        "id": r["id"],
        "read_status": old_text[r["id"]][0], "strings": old_text[r["id"]][1],
        "geo_text": geo_text.get(r["id"], r["model_text"]),
        "observations": road[r["id"]], "geo_road": geo_road.get(r["id"], r["direct"]),
    } for r in text]
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output / "format-control-results.zip", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("configuration.json", json.dumps({
            "manifest_sha256": manifest_hash,
            "adapter_sha256": hashlib.sha256(
                (args.input / "adapter/adapter_model.safetensors").read_bytes()).hexdigest(),
            "road_parsed": len(geo_road), "additional_text_cases": len(geo_text),
        }, indent=2) + "\n")
        archive.writestr("answers.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
