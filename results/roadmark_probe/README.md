# Qwen3-VL Road Center-Line Feature Probe

**Date:** 2026-11-21. This experiment tested whether frozen visual representations in `Qwen3-VL-2B` can linearly separate single vs. double road center lines, and how the base model vs. the 6k localization LoRA answer the same binary visual question directly.

## Data & Setup

- **Blind human labels:** 70 annotated images; binary classification used **23 single-line** and **20 double-line** images (excluding 16 no-line and 11 unclear). Because the second annotation batch specifically enriched double-line examples, proportions and scores reflect this diagnostic sample.
- **Splits:**
  - *43-image creator-disjoint 5-fold CV:* 43 distinct photographers partitioned into 5 creator-disjoint folds.
  - *22-image same-country paired leave-one-country-out:* 11 geographically matched single/double pairs spanning 8 countries (median within-pair distance 281 km, range 30–1,561 km), holding out an entire country per fold.
- **Probes & QA:** Using the exact same 4-bit NF4 `Qwen3-VL-2B` backbone and 512 px image preprocessing, we compared logistic regression probes (2×2 spatial pooling + identical standardization/hyperparameters) before vs. after the main vision-language projector, plus global-average / 4-patch ViT features and multi-layer DeepStack outputs. All vision features came from the frozen base encoder. For direct QA, we presented a binary single-vs-double choice, **swapped A/B option order on every image**, and averaged the two log-odds scores.
- **Scope note:** All 43 images came from the 6,000-image localization LoRA training pool (the center-line labels themselves were never used in training). Cross-country generalization of the localization LoRA must still be measured on unseen-creator holdouts.

## Results

Balanced accuracy equally averages single-line and double-line recall. The final column counts how many of the 11 same-country pairs assigned a higher double-line score to the true double-line photo.

| Readout Method | 43 Images (Creator 5-Fold) | 22 Images (Leave-One-Country-Out) | Same-Country Pair Ranking |
|---|---:|---:|---:|
| Known country label only | 56.1% (24/43) | 50.0% (11/22) | 11 ties |
| Downsampled full-image RGB | 51.1% (22/43) | 50.0% (11/22) | 8 / 11 |
| ViT pre-merger (2×2 pooled) | 49.2% (21/43) | 54.5% (12/22) | 5 / 11 |
| ViT preserving patch positions | 57.9% (25/43) | 59.1% (13/22) | 8 / 11 |
| Post-main projector | 43.6% (19/43) | 50.0% (11/22) | 9 / 11 |
| Main projector + DeepStack | 41.4% (18/43) | 54.5% (12/22) | 9 / 11 |
| Base Qwen3-VL direct QA | 54.0% (22/43) | 63.6% (14/22) | 10 / 11 |
| 6k localization LoRA direct QA | 62.7% (26/43) | 72.7% (16/22) | 10 / 11 |

- **No evidence of projector signal loss:** Moving from pre- to post-projector changed 6 wrong→right and 8 right→wrong on 43 images (3 wrong→right, 4 right→wrong on 22 images), while same-country pair ranking *improved* from 5/11 to 9/11.
- **LoRA mainly calibrated the double-line prior:** Both base and 6k LoRA got 19/20 double-line images right, with the 4 extra QA wins from LoRA all on single-line images (mean double-line logit shifted by median `−0.96`). AUROC rose `0.878 → 0.900` on all 43 images and `0.917 → 0.926` on the 22-image subset.
- **Strong A/B option-order bias:** Swapping A/B order kept the same answer on only `29/43` (base) and `16/43` (6k LoRA) images. When "double line" was option A, base chose double `41/43` times and LoRA `43/43` times — limiting raw binary accuracy interpretation.
- **Qualitative spot check:** Single dashed lines in Taiwan and Poland flipped from wrong to right with LoRA, while an Irish solid+dashed double line still scored lower than its paired single-line photo due to broader scene/lighting differences.

## Verdict & Reproduction

These 70 annotations do **not** justify unfreezing the ViT or projector yet (see the follow-up crop diagnosis in [`../roadmark_crop_probe/README.md`](../roadmark_crop_probe/README.md)).

Recompute [`analysis.json`](analysis.json) and [`out_of_fold.csv`](out_of_fold.csv) from local cached features (`roadmark-features.zip`) via:

```sh
python3 -B scripts/analyze_roadmark_probe.py
```

Feature extraction used a Colab Tesla T4 runtime (terminated immediately after downloading results; 0.00 compute units/hr).
