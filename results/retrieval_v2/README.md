# Anti-Leakage Retrieval v2: Shortcut Forensics, Geo-Projection Re-Ranking & StreetCLIP × VLM Hybrid

**Date:** 2026-11-23. Before training a new ranker, we manually inspected 15 matched query–gallery pairs and the contributor distribution of the 25k retrieval dataset — uncovering a massive same-creator shortcut leak. After enforcing a strict diff-creator protocol, we progressed through **Step 1A (Spatial KDE voting)**, **Step 1B (full-gallery Geo-Metric Projection + Listwise Ranker)**, **Step 1C (exact 211-country `Qwen3-VL` 6k log-prob fusion)**, and **Step 1D (swapping to `geolocal/StreetCLIP` + HUD-crop multi-view ablation)**.

---

## 1. Pre-Modeling Visual & Data Forensics (Four Critical Blind Spots)

Auditing `attribution` (photographer) and inspecting actual image pairs in the 25k retrieval cache (`results/ranker-2018-pairs.zip`: 25k gallery / 12k train / 2k val) revealed:

1. **97.2% of the original 25k `dinov2-small` hits (`14.5% @25 km` / `49.2% Top-100`) were same-photographer / same-vehicle shortcuts:**
   - **92.8% (1,855 / 2,000)** of `val_2000` creators also appeared in the 25k gallery; the top 20 prolific accounts alone made up **28.1%** of the gallery (`lvl5`: 2,284 images, `roadroid`: 888, `networklanman`: 422, `bob3bob3`: 298).
   - Out of 290 Top-1 ≤ 25 km hits on `val_2000`, **282 (97.2%)** were same-creator matches; on `qlora_eval_300`, **44/44 (100%)** were same-creator matches.
   - **Visual evidence from raw images:**
     - India `@natarajakaveeti` (`1101634303727287.jpg` vs. `1786588988190818.jpg`, cos=0.938, 0.0 km): exact same car, same triangular windshield sticker in the bottom-left corner, and same two cars ahead — merely split into separate Mapillary sequences.
     - Australia `@bob3bob3` (`2847551735459550.jpg` vs. `810202126300348.jpg`, cos=0.927, 7.2 km): same road trip 4 minutes apart with identical black dashcam GPS overlay bars.
     - USA `@networklanman` (`748528363325434.jpg` vs. `907192507133508.jpg`, cos=0.831, **647 km apart**): ranked Top-1 despite being 647 km away due to the identical `NETWORKLANMAN` GPS watermark and low-bumper camera mount.
   - **True diff-creator `dinov2-small` baseline (masking same-creator matches):**
     - Effective 25k gallery coverage: ≤ 25 km drops `95.8% → 52.1%`, ≤ 200 km `96.3%`, same-country `99.2%`.
     - True `val_2000` Top-1: ≤ 25 km **`0.45%` (9/2,000)**, ≤ 200 km **`3.80%` (76/2,000)**, country **`21.4%`**, median error **`3,965.0 km`**, mean GeoGuessr score **`1,189.4`**.
     - True `val_2000` Top-100 recall ceiling: ≤ 25 km is **`7.80%`** (not 49.2%), ≤ 200 km `50.1%`, country `83.2%`.

2. **General-purpose vision encoders are fooled by "composition / lighting twins":**
   - Italian tunnel matched to Japanese tunnel (`2986083211609576.jpg` vs. `484810106163607.jpg`, cos=0.848, **9,566 km** error).
   - Blurry Indian night scene matched to blurry Saudi night scene (`198463045436194.jpg` vs. `474829423581561.jpg`, cos=0.820, **3,755 km** error).
   - US desert yellow-dashed road matched to Australian red-soil white-dashed road (`768656080501644.jpg` vs. `553472652282790.jpg`, cos=0.833, **13,761 km** error).

3. **Scene-type mismatch within 25 km:**
   - Rural Danish gravel road (`1441389372865694.jpg`) has a true gallery neighbor 22.4 km away (`763751890877619.jpg`), but that neighbor is an urban canal harbor (cos = `-0.043`), while a straight rural road 7,349 km away in Angola scores `cos = 0.669`.

4. **Bottom-edge dashcam GPS watermarks on prolific accounts:**
   - Motivated the bottom-12% crop (`crop88`) and multi-view fusion ablation in Step 1D.

> **Anti-Leakage Iron Rule:** Every experiment in Steps 1A–1D **strictly masks gallery images sharing the query's `attribution`**. Step 1B training explicitly removes all `author_holdout_66` creators (42 rows excluded) and splits `opt_train` / `tune` by disjoint creators. Fusion weights in Steps 1C–1D are chosen solely via 5-fold OOF on `qlora_eval_300` and evaluated **zero-touch** on `author_holdout_66`.

---

## 2. Steps 1A & 1B: 25k `dinov2-small` Spatial KDE + Full-Gallery Geo-Projection + Listwise Ranker

- **Step 1A (local CPU, [`eval_spatial_kde.py`](../../scripts/eval_spatial_kde.py)):** Spatial kernel density voting over Top-50 diff-creator candidates (`τ=0.10, σ=800 km`, tuned on training subset).
- **Step 1B (Colab T4, [`train_geo_ranker.py`](../../scripts/train_geo_ranker.py)):**
  - **Stage 1 (Geo-Metric Projection):** Shared residual projection (384→512→384 + auxiliary country head) trained over all 25,000 gallery items with multi-scale geographic soft targets (35 / 180 / 600 km + same-country). **Masks same-creator pairs ≤ 300 km while retaining same-creator pairs > 300 km as hard negatives** to eliminate camera/watermark fingerprints.
  - **Stage 2 (Listwise Geo-Ranker):** Re-retrieves diff-creator Top-100 in projected space and trains a bounded residual ranker over 22 scale-normalized features (projected/raw cosine, multi-scale KDE, gallery-density debiasing, Top-100 country consensus, auxiliary country log-probs) with Listwise Softmax KL — completely fixing the legacy `train_ranker.py` Epoch-1 collapse (`best_stage1_epoch=15`, `best_stage2_epoch=2`).

| Eval Split (Strict Diff-Creator) | Method | Top-100 ≤ 25 km | Top-100 ≤ 200 km | ≤ 25 km | ≤ 200 km | Country Acc. | Median Error | GeoGuessr Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`val_2000`** (n=2,000) | Raw `dinov2-small` Top-1 | 7.80% | 50.0% | 9 (0.45%) | 76 (3.80%) | 429 (21.4%) | 3,965.0 km | 1,189.4 |
| | Step 1A: Spatial KDE | 7.80% | 50.0% | 8 (0.40%) | 92 (4.60%) | 488 (24.4%) | 2,736.8 km | 1,427.5 |
| | Step 1B Stage 1: GeoProj Top-1 | **11.90%** | **64.1%** | 7 (0.35%) | 114 (5.70%) | 779 (39.0%) | 1,476.3 km | 1,889.3 |
| | Step 1B Stage 1: GeoProj + KDE | 11.90% | 64.1% | 7 (0.35%) | 117 (5.85%) | 848 (42.4%) | 1,088.9 km | 2,126.0 |
| | **Step 1B Stage 2: Listwise Ranker** | 11.90% | 64.1% | **16 (0.80%)** | **166 (8.30%)** | **998 (49.9%)** | **1,032.1 km** | **2,213.2** |
| **`val_unseen_opt`** (n=522, unseen in train) | Raw `dinov2-small` Top-1 | 9.58% | 53.4% | 3 (0.57%) | 23 (4.41%) | 88 (16.9%) | 3,962.4 km | 1,229.7 |
| | **Step 1B Stage 2: Listwise Ranker** | **12.84%** | **57.3%** | **5 (0.96%)** | **34 (6.51%)** | **164 (31.4%)** | **1,384.9 km** | **1,909.6** |
| **`author_holdout_66`** (n=66, doubly unseen) | Raw `dinov2-small` Top-1 | 16.67% | 57.6% | 1 (1.52%) | 5 (7.58%) | 15 (22.7%) | 3,509.8 km | 1,287.4 |
| | **Step 1B Stage 2: Listwise Ranker** | **19.70%** | **59.1%** | **2 (3.03%)** | **5 (7.58%)** | **23 (34.8%)** | **1,404.4 km** | **1,988.9** |

---

## 3. Step 1C: Exact 211-Country `Qwen3-VL` 6k Priors × 25k `dinov2-small` Hybrid

Using [`extract_vlm_country_priors.py`](../../scripts/extract_vlm_country_priors.py), we read exact log-probabilities across all 211 country codes in a single KV-cache forward pass immediately after the `{"country":"` prefix (verified `< 0.05` error vs. full forward pass, reproducing exact Top-1 country accuracies of `52.0%` / `50.0%`), then ran 5-fold OOF fusion via [`evaluate_retrieval_v2_hybrid.py`](../../scripts/evaluate_retrieval_v2_hybrid.py):

- **`qlora_eval_300` (5-fold OOF):**
  - `Qwen3-VL` 6k alone: country `52.0%`, ≤ 200 km `14.7%`, median `831.6 km`, mean `2,556.2` pts.
  - 25k Retrieval v2 + VLM country priors: country **`58.3%` (175/300, +6.3%)**, median `885.6 km`, mean `2,521.7` pts.
  - 25k Retrieval v2 + VLM country priors + spatial anchor: country `57.3%`, ≤ 200 km `14.7%`, median **`782.0 km`**, mean **`2,580.6`** pts (surpassing both standalone VLM and standalone retrieval).
- **`author_holdout_66` (zero-touch):**
  - Adding VLM priors improved pure 25k `dinov2-small` retrieval from `1,404 km` to `721.7 km` (2/66 @25 km > VLM's 1/66, mean `2,644.5` pts), though 22M `dinov2-small` regional granularity still lagged standalone VLM (`600.3 km`, `2,788.6` pts).

---

## 4. Step 1D: Raising the Ceiling — `geolocal/StreetCLIP` (ViT-L/14) + HUD Crop × `Qwen3-VL`

To test how much a geo-pretrained backbone raises the ceiling, we compared frozen `dinov2-small` vs. `geolocal/StreetCLIP` (full image `full`, bottom-12% HUD crop `crop88`, and two-view average `multiview`) on the **exact same clean 6,000-image gallery** (4× smaller than 25k; **0% creator overlap** with `author_holdout_66`), combined with Spatial KDE, `StreetCLIP` zero-shot country text probes (`"A Street View photo in {country}"`), and `Qwen3-VL` 6k country log-probs. All weights were selected via 5-fold CV on `qlora_eval_300` and evaluated **zero-touch** on `author_holdout_66` ([`finalize_step1d_analysis.py`](../../scripts/finalize_step1d_analysis.py)).

### 4.1 `author_holdout_66` (66 unseen creators / 60 countries, **zero-touch**, 6,000-image gallery)

| Method | Top-100 ≤ 25 km | Top-100 ≤ 200 km | ≤ 25 km | ≤ 200 km | Country Acc. | Median Error | GeoGuessr Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dinov2-small` raw Top-1 (6k) | 22.7% | 62.1% | 0/66 (0.0%) | 2/66 (3.0%) | 11/66 (16.7%) | 3,478.5 km | 1,098.1 |
| `dinov2-small` GeoProj + Listwise (25k) | 19.7% | 59.1% | 2/66 (3.0%) | 5/66 (7.6%) | 23/66 (34.8%) | 1,404.4 km | 1,988.9 |
| **Previous best: `Qwen3-VL-2B` 6k QLoRA** | — | — | 1/66 (1.5%) | 14/66 (21.2%) | 33/66 (50.0%) | 600.3 km | 2,788.6 |
| `StreetCLIP` (full) raw Top-1 (6k) | **39.4%** | **90.9%** | 4/66 (6.1%) | 14/66 (21.2%) | 32/66 (48.5%) | 618.3 km | 2,854.8 |
| `StreetCLIP` (full) + KDE + Text-Probe (6k) | **39.4%** | **90.9%** | **7/66 (10.6%)** | **18/66 (27.3%)** | 43/66 (65.2%) | 461.1 km | 3,286.3 |
| **`StreetCLIP` (full) + KDE + Text + Qwen3-VL (6k)** | **39.4%** | **90.9%** | 6/66 (9.1%) | 16/66 (24.2%) | **46/66 (69.7%)** | 435.3 km | **3,395.4** |
| **`StreetCLIP` (full+crop88) + KDE + Text + Qwen3-VL (6k)** | — | — | 5/66 (7.6%) | 15/66 (22.7%) | **46/66 (69.7%)** | **389.7 km** | 3,374.6 |

### 4.2 `qlora_eval_300` (300 validation images, strict diff-creator 6k gallery, 5-fold OOF)

| Method | Top-100 ≤ 25 km | Top-100 ≤ 200 km | ≤ 25 km | ≤ 200 km | Country Acc. | Median Error | GeoGuessr Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dinov2-small` raw Top-1 (6k) | 8.3% | 42.7% | 0/300 (0.0%) | 11/300 (3.7%) | 41/300 (13.7%) | 5,972.8 km | 979.8 |
| `dinov2-small` GeoProj + Listwise (25k) | 10.7% | 65.0% | 3/300 (1.0%) | 27/300 (9.0%) | 143/300 (47.7%) | 1,077.9 km | 2,176.3 |
| **Previous best: `Qwen3-VL-2B` 6k QLoRA** | — | — | 8/300 (2.7%) | 44/300 (14.7%) | 156/300 (52.0%) | 831.6 km | 2,556.2 |
| `StreetCLIP` (full) + KDE + Text-Probe (6k, OOF) | **18.0%** | **72.0%** | **12/300 (4.0%)** | **60/300 (20.0%)** | 179/300 (59.7%) | **555.6 km** | 2,983.3 |
| **`StreetCLIP` (full) + KDE + Text + Qwen3-VL (6k, OOF)** | **18.0%** | **72.0%** | 11/300 (3.7%) | 56/300 (18.7%) | 191/300 (63.7%) | 615.9 km | **2,996.6** |
| **`StreetCLIP` (full+crop88) + KDE + Text + Qwen3-VL (6k, OOF)** | — | — | 11/300 (3.7%) | 53/300 (17.7%) | **193/300 (64.3%)** | 566.6 km | 2,988.5 |

### 4.3 Key Findings: HUD Cropping, Shortcuts & Complementarity
- **Massive recall jump:** On the exact same 6k gallery, swapping `dinov2-small` → `geolocal/StreetCLIP` jumped `author_holdout_66` Top-100 recall from `22.7% / 62.1%` to **`39.4% / 90.9%`** (and `qlora_eval_300` from `8.3% / 42.7%` to **`18.0% / 72.0%`**).
- **Multi-view HUD cropping (`full + crop88`) helps:** `crop88` alone reduced unmasked same-creator collisions on `qlora_eval_300` (`103 → 98/300`) and lifted zero-shot country text-probe accuracy on `author_holdout_66` (`59.1% → 60.6%`); averaging `full` + `crop88` pushed `author_holdout_66` hybrid median error down to **`389.7 km`** and `qlora_eval_300` country accuracy to **`64.3%`**.
- **StreetCLIP still exhibits creator shortcuts if unmasked:** Without diff-creator masking, raw Top-1 still matched the same photographer on `34.3%` (103/300) of `qlora_eval_300` (`raw median = 503 km` vs. `diff median = 945 km`), reinforcing the necessity of strict diff-creator or official `test` evaluation.
- **Genuine complementarity with VLM:** On `author_holdout_66`, both `StreetCLIP` and `Qwen3-VL` 6k got 30 countries right, `StreetCLIP`-only got 15, and VLM-only got 3 (oracle union **72.7%**). Pairwise, the hybrid landed closer than standalone `Qwen3-VL` on **38/66** images (+606.8 GeoGuessr pts, +19.7% country accuracy, 6× higher 25 km hit rate).

---

## 5. Next Steps Leading to Stage 11

These 6k proof-of-concept results directly motivated Stage 11 ([`official_test_showdown/README.md`](../official_test_showdown/README.md)):
1. Scale `StreetCLIP` from 6k to the full **25k coverage-first gallery** with leak-free Geo-Projection + Listwise Ranker.
2. Lock **`official_test_300`** from OSV-5M official `test` for a fair head-to-head comparison against `osv5m/baseline`.
3. Connect **OCR-gated sign crops + offline GeoNames gazetteer lookup** as a future follow-up to cross-check top candidate regions against readable signage.

---

## 6. Files & Reproduction

- **Step 1A:** `python3 -B scripts/eval_spatial_kde.py` → [`step1a_kde_results.json`](step1a_kde_results.json)
- **Step 1B + 1C (Colab T4, terminated):** [`scripts/prepare_retrieval_v2_bundle.py`](../../scripts/prepare_retrieval_v2_bundle.py), [`scripts/train_geo_ranker.py`](../../scripts/train_geo_ranker.py), [`scripts/extract_vlm_country_priors.py`](../../scripts/extract_vlm_country_priors.py); local fusion: `python3 -B scripts/evaluate_retrieval_v2_hybrid.py` → [`step1c_hybrid_results.json`](step1c_hybrid_results.json)
- **Step 1D (Colab T4, terminated):** [`scripts/prepare_step1d_bundle.py`](../../scripts/prepare_step1d_bundle.py), [`scripts/benchmark_streetclip.py`](../../scripts/benchmark_streetclip.py); local analysis: `python3 -B scripts/evaluate_step1d_streetclip.py` & `python3 -B scripts/finalize_step1d_analysis.py` → [`step1d_streetclip_results.json`](step1d_streetclip_results.json), [`step1d_final_oof.json`](step1d_final_oof.json)
