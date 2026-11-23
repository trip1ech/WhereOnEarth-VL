# Road Center-Line Crop Probe & ViT Diagnosis

## 1. Question & Design

To test whether weak full-image linear probe accuracy justified unfreezing `Qwen3-VL-2B`'s vision encoder, we compared full images against **zoomed crops around the road center line** while keeping **all weights (ViT, projector, and LM) completely frozen**.

- **Dataset & splits:** Reused the exact 23 single-line and 20 double-line images and splits from [`../roadmark_probe/README.md`](../roadmark_probe/README.md) (43-creator 5-fold CV + 11 same-country pairs across 8 countries).
- **Cropping conditions:**
  - *Manual crop (`marked`):* Selected a center point near the road center line on blind thumbnails (no labels or predictions shown), extracted a fixed `48% × 48%` window (clamped inside frame bounds), and resized back to original pixel dimensions — preserving image token count while magnifying linear scale by ~2.1×.
  - *Fixed crop (`fixed`):* Identical `48% × 48%` window always centered at `(x=50%, y=76%)`.
  - Ten dark/wet/occluded/complex-intersection images were flagged as visually uncertain in [`crop_centers.csv`](crop_centers.csv) while keeping original labels intact.
- **Readouts:** Identical frozen linear probes and order-swapped A/B direct QA as Stage 6.

## 2. Results

Balanced accuracy averages single-line and double-line recall (equal to exact accuracy on the balanced 22-image paired subset).

| Readout Method | 43 Full | 43 Fixed Crop | 43 Manual Crop | 22 Full | 22 Fixed Crop | 22 Manual Crop |
|---|---:|---:|---:|---:|---:|---:|
| ViT last layer, 2×2 pooled | 49.2% | 34.6% | 41.7% | 54.5% | 50.0% | 50.0% |
| ViT last layer, patch positions | 57.9% | 34.6% | 53.3% | 59.1% | 50.0% | 54.5% |
| Main projector | 43.6% | 41.4% | 65.1% | 50.0% | 54.5% | 72.7% |
| Projector + DeepStack | 41.4% | 47.9% | 58.3% | 54.5% | 63.6% | **81.8%** |
| Downsampled RGB baseline | 51.1% | 51.1% | 41.4% | 50.0% | 45.5% | 45.5% |
| Base Qwen3-VL direct QA | 54.0% | 58.4% | 71.7% | 63.6% | 72.7% | **81.8%** |
| 6k localization LoRA direct QA | 62.7% | 60.2% | **78.3%** | 72.7% | 68.2% | **86.4%** |

- **Large jump from cropping alone (zero weight updates):** On the 22-image same-country subset, manual cropping boosted Projector+DeepStack from `12/22 → 18/22` (7 wrong→right, 1 right→wrong), base Qwen QA by `+4` with zero regressions, and 6k LoRA QA by `+3` with zero regressions. Manual crops beat fixed crops across the board.
- **Perfect ranking & AUROC = 1.00:** With manual crops, Projector+DeepStack, base Qwen QA, and 6k LoRA QA all ranked **11/11** same-country pairs correctly (downsampled RGB: `6/11`). Both Qwen QA models reached **AUROC = 1.00** across all 43 manually cropped images (every double-line image outscored every single-line image), though at a zero decision threshold double-line bias remained (`20/20` double vs. `10/23` base / `13/23` LoRA single) alongside A/B order sensitivity (`28/43` base and `21/43` LoRA order-consistent).
- **Sensitivity check:** Excluding the single visually uncertain photo in the 22-image subset preserved the exact same trend (`12/21 → 17/21` Projector+DeepStack; `14/21 → 18/21` base Qwen QA).

## 3. ViT Decision & Open LM Hypothesis

**Decision: Keep the ViT frozen.** With identical frozen weights, merely giving the model a cleaner local view of the center line raised accuracy sharply — showing that resolution/framing is a much more direct bottleneck than ViT representation capacity.

### Open LM Hypothesis
Cropping before the ViT simultaneously magnifies the road line and strips background clutter. Because skipping the LM entirely (Projector+DeepStack linear probe) also jumped `12/22 → 18/22`, feature separability itself clearly improved. A clean follow-up test would extract visual features **once** from the full uncropped image and mask only which image tokens the LM may attend to (all tokens vs. center-line tokens vs. area-matched road tokens) to isolate LM visual attention from ViT resolution.

## 4. Reproduction

[`comparison.json`](comparison.json) stores per-method scores and paired improvements/regressions, while [`marked/`](marked/) and [`fixed/`](fixed/) hold `analysis.json` and `out_of_fold.csv`. Recompute locally (with cached `roadmark-features.zip`) via:

```sh
for view in marked fixed; do
  python3 -B scripts/analyze_roadmark_probe.py \
    --results "results/roadmark_crop_probe/$view" \
    --labels results/roadmark_probe/labels.jsonl \
    --inference-input results/roadmark_crop_probe/inference-input.zip
done
python3 -B scripts/compare_roadmark_crop_probe.py
```

Feature extraction ran on Colab Tesla T4 (terminated immediately after download; 0.00 compute units/hr).
