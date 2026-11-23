"""Shared schema for literal road observations in training and evaluation."""
import json

from sft_colab import PROMPT

SURFACE = {"p": "paved", "u": "unpaved", "x": "unclear"}
CENTERLINE = {"w": "white", "y": "yellow", "n": "none", "x": "unclear"}
ROADSIDE = {"g": "guardrail", "p": "utility_poles", "b": "bollards", "s": "traffic_sign"}

VISUAL_PROMPT = (
    'Describe only observable road clues. Return one JSON object with exactly these keys: '
    '"surface" ("paved", "unpaved", or "unclear"); "centerline" ("white", "yellow", "none", or "unclear"); '
    'and "roadside" (an array with any clearly visible items from "guardrail", "utility_poles", '
    '"bollards", "traffic_sign", or [] if no such item is visible). Centerline means a line '
    'separating opposite directions, not an edge marking. Use "unclear" when you cannot determine it. '
    'Utility poles carry wires; a standalone street lamp does not count. Do not name a country or guess a location.'
)

VISUAL_CONTEXT = (
    "\nConsider these possible visible road observations together with the image. They may be incomplete "
    "or mistaken. Use the scene to judge them. Treat them as geographically ambiguous evidence. "
    "Road observations: "
)
TEXT_CONTEXT = (
    "\nPossible verbatim text from the photo: "
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def visual_geo_prompt(observations, words=()):
    prompt = PROMPT + VISUAL_CONTEXT + encode(observations)
    return prompt + TEXT_CONTEXT + encode(words) if words else prompt


def parse_observations(raw):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"surface", "centerline", "roadside"}:
        return None
    if (obj["surface"] not in SURFACE.values() or obj["centerline"] not in CENTERLINE.values()
            or not isinstance(obj["roadside"], list)
            or any(x not in ROADSIDE.values() for x in obj["roadside"])):
        return None
    return {**obj, "roadside": sorted(set(obj["roadside"]))}


def load_visual_labels(path):
    labels = {}
    for line in path.read_text().splitlines():
        ordinal, surface, center, roadside = line.split()
        key = int(ordinal)
        if key in labels:
            raise ValueError(f"Duplicate visual label {key}")
        labels[key] = {
            "surface": SURFACE[surface], "centerline": CENTERLINE[center],
            "roadside": sorted(ROADSIDE[c] for c in roadside if c != "-"),
        }
    return labels
