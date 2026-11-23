# Blind Road Center-Line Annotator (`annotation_tool`)

A lightweight local web tool for blind annotation of visible road center-line markings. It streams the exact 512 px JPEGs directly from the project's local ZIP archives in a fixed shuffled order, hiding country, GPS coordinates, photographer attribution, and model predictions.

## Launch

On macOS, double-click [`launch_annotator.command`](launch_annotator.command) in Finder, or run from the repo root:

```sh
python3 annotation_tool/server.py
```

Then open <http://127.0.0.1:8766/> in your browser. Press `Control+C` to stop.

## Keyboard Shortcuts & Label Definitions

- `1` **Single line** — one solid or dashed line separating opposing traffic.
- `2` **Double line** — two parallel lines side by side (solid and/or dashed).
- `3` **No center line** — the center of a two-way road is clearly visible with no markings.
- `4` **Unclear / N/A** — occluded, blurred, no road visible, one-way street, or conflicting road segments.
- Count only lines separating opposing traffic; ignore edge lines and same-direction lane dividers.
- Use `←` / `→` arrow keys to browse and **Next unlabeled** to jump ahead. Selecting a label saves immediately and advances to the next unlabeled photo.

Annotations are atomically saved to [`data/roadmark_annotations/labels.json`](../data/roadmark_annotations/labels.json). The tool starts on **Batch 1 (40)** (first 40 shuffled training images), switches to **Batch 2 (30)** (30 enriched follow-up candidates), and also supports labeling the full training set or the 66-image unseen-creator holdout. **Export JSONL** downloads a copy containing image ID, dataset split, SHA-256, label (`single`, `double`, `none`, `unclear`), and timestamp.
