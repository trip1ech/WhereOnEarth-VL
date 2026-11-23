# Unseen-Creator Street-View Holdout (`author_holdout`)

This benchmark evaluates the frozen `Qwen3-VL-2B` zero-shot baseline vs. the 6,000-image QLoRA model on **strictly unseen photographers** from the same OSV-5M domain, leaving the original 300-image validation split untouched.

## 1. Visual Audit Before Locking

Before locking inputs, we inspected 15 OSV-5M candidates alongside all 9 local Panoramax `pilot_holdout` photos. Because Panoramax covered only 3 French metro areas across 6 creators (with forest paths and repeated windshields), it was kept separate for French cross-platform exploration rather than mixed into the global holdout.

All **72 OSV-5M candidates** were visually reviewed one by one and logged in [`visual_review.csv`](visual_review.csv). Exclusion criteria were strictly limited to:
1. Direct coordinate leaks (navigation HUD displaying latitude/longitude),
2. Non-street / non-traversable outdoor scenes with no recognizable road or public path, or
3. Severe blur rendering the scene uninterpretable.

Natural street-view artifacts (car hoods, wipers, windshield tint, reflections, dark roads, sparse rural tracks, and real signage) were intentionally retained. Local OCR also scanned every candidate for hidden coordinate overlays.

Six candidates were excluded and retained in the audit log:
- **#15 (`494529601695744`, ID):** navigation HUD displaying Indonesian coordinates and route map.
- **#43, #53, #57, #66:** covered building entrance, desert tourist shot with no road, private terrace, and foreground debris with no recognizable street/path.
- **#62:** severely out-of-focus night shot.

## 2. Split Hygiene & Sampling Bias

- **Origin:** Post-2018 validation candidates from OSV-5M official `train`, strictly disjoint from both the 6,000 QLoRA training images and the 300 evaluation images.
- **Zero `creator_id` overlap:** After sampling by creator username, we cross-checked raw OSV-5M metadata `creator_id` against all 6,000 training images and 300 eval images — confirming **zero overlap across all 66 unseen accounts** (1 image per photographer; disjoint image IDs, capture sequences, and raw JPEG SHA-256). Summary recorded in [`creator_id_check.json`](creator_id_check.json).
- **Geographic balance:** Restricted to countries present in both training and the 300-image eval split, prioritizing candidates within 300 km of an eval image in the same country while excluding any candidate within 1 km of a training image. The locked set spans **66 images, 66 unseen photographers, and 60 countries/territories** (2018–2022), with 52/66 within 300 km of an eval photo.
- **Input format:** Preserves full field of view, resized to 512 px longest edge with identical JPEG quality as `qlora_eval_300`.

*(Locked image archive `author-holdout-eval.zip` and visual contact sheets `osv-review-01..06.jpg` are stored locally and git-ignored.)*

## 3. Fixed-Model Evaluation Results

Evaluated with the exact `Qwen3-VL-2B` revision, 4-bit NF4 config, prompt, parser, and 6k QLoRA adapter used in Stage 4:

| Metric | Zero-Shot | 6,000-Image QLoRA |
|---|---:|---:|
| Country correct | 28 / 66 | **33 / 66** |
| Within 200 km | 8 / 66 | **14 / 66** |
| Within 25 km | 1 / 66 | 1 / 66 |
| Median error (65 shared valid) | 945 km | **589 km** |

Across the 65 images where both models returned valid coordinates (1 zero-shot coordinate parse failure), 6k QLoRA landed closer on **42 images** and farther on **23 images**. Full metrics are in [`results/author_holdout/metrics.csv`](../../results/author_holdout/metrics.csv). Colab T4 runtime was terminated upon completion.
