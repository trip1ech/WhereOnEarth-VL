"""Evaluate geographic answers and visible-road extraction before and after LM LoRA."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from clue_visual import VISUAL_PROMPT, parse_observations, visual_geo_prompt
from evaluate_clue_first import generate, load_for_inference, read_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("final_adapter", type=Path)
    parser.add_argument("new_text_results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = (args.input / "inputs.json").read_bytes()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    rows = json.loads(manifest)
    ids = [r["id"] for r in rows]
    photos, _, _ = read_images(args.input, rows)
    with ZipFile(args.new_text_results) as archive:
        if json.loads(archive.read("configuration.json"))["manifest_sha256"] != manifest_hash:
            raise ValueError("Text experiment used a different dataset")
        text = list(map(json.loads, archive.open("answers.jsonl")))
    if [r["id"] for r in text] != ids:
        raise ValueError("Text and visual evaluation rows differ")
    text_by_id = {r["id"]: r for r in text}

    model, processor = load_for_inference(args.input / "adapter")
    model.load_adapter(args.final_adapter, adapter_name="updated", is_trainable=False)
    model.eval()
    results = {i: {"id": i} for i in ids}
    for label, adapter in (("before", "default"), ("after", "updated")):
        model.set_adapter(adapter)
        raw = generate(model, processor, [(i, VISUAL_PROMPT) for i in ids],
                       photos, 4, 72, f"{label}_observe_road")
        observed = {i: parse_observations(raw[i]) for i in ids}
        active = [i for i in ids if observed[i] is not None]
        geo = generate(model, processor, [(i, visual_geo_prompt(observed[i])) for i in active],
                       photos, 4, 40, f"{label}_geo_road")
        if label == "after":
            useful = [i for i in active if text_by_id[i]["model_strings"]]
            both = generate(model, processor, [
                (i, visual_geo_prompt(observed[i], text_by_id[i]["model_strings"])) for i in useful
            ], photos, 4, 40, "after_geo_both")
        for i in ids:
            results[i][label] = {"raw_observations": raw[i], "observations": observed[i],
                                 "geo_visual": geo.get(i)}
            if label == "after":
                results[i][label]["geo_both"] = both.get(i, geo.get(i, text_by_id[i]["model_text"]))
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output / "visual-results.zip", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("configuration.json", json.dumps({
            "manifest_sha256": manifest_hash, "prompt": VISUAL_PROMPT, "n": len(ids),
            "final_adapter_sha256": hashlib.sha256(
                (args.final_adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        archive.writestr("answers.jsonl", "".join(json.dumps(results[i], ensure_ascii=False) + "\n" for i in ids))


if __name__ == "__main__":
    main()
