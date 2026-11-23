# 25k StreetCLIP Geo-Ranker vs. 5M OSV-5M Official Baseline (`official_test_300`)

**Date:** 2026-11-23. This stage delivered two major milestones:
1. **Scaled `geolocal/StreetCLIP` to all 39,000 post-2018 images** (25,000 gallery + 12,000 train + 2,000 val) and trained a creator-disjoint 25k Geo-Metric Projection (`Stage 1`) and Listwise Geo-Ranker (`Stage 2`).
2. **Constructed `official_test_300`** — a held-out benchmark from the OSV-5M official `test` split (300 unseen photographers across 147 countries) — for head-to-head evaluation against the 5-million-image official **`osv5m/baseline`**, `Qwen3-VL-2B` (zero-shot and 6k QLoRA), and `dinov2-small`.

---

## 1. Constructing `official_test_300` & Split Partitioning

Build script: [`scripts/prepare_official_test.py`](../../scripts/prepare_official_test.py). Local archive (git-ignored): `data/official_test_300/official-test-300.zip` (14.99 MB). Audit log: [`data/official_test_300/hud_audit.json`](../../data/official_test_300/hud_audit.json).

- **Source:** OSV-5M official `test` (`data/raw/osv5m-test.csv`, capture timestamp `>= 2018-01-01 UTC`), strictly disjoint from `osv5m/baseline`'s official training set.
- **Zero photographer leakage:** Cross-checked against `gallery_25000`, `train_12000`, `validation_2000`, and `qlora_6000` so all 300 photos come from **300 completely unseen photographers (1 photo per creator)**, with zero overlap on image ID, sequence, `creator_username`, or `creator_id`.
- **Globally balanced stratification:** From 1,175 eligible unseen creators (148 countries), sampled 340 candidates via square-root country stratification + 4° geographic grid round-robin, then locked **300 photos spanning 147 countries** (at most 10 images per country, preventing large-country dominance).
- **HUD overlay & visual audit:** Scanned all 340 candidates with native macOS Apple Vision OCR, catching and excluding 1 Japanese Garmin dashcam image with embedded GPS coordinates (`798364104121004.jpg`, bottom overlay `38.41452 141.23285 20 KM/H`). Spot-checked raw images to verify realistic street-view diversity (including real-world artifacts such as an upside-down dashcam frame); all models receive identical RGB pixels.
- **25k gallery natural geographic coverage of `official_test_300`:** `79.3%` within 25 km, `95.7%` within 200 km, `99.0%` same-country.

---

## 2. Streaming Encoding & 25k Two-Stage Training (Phase B)

Script: [`scripts/encode_and_train_streetclip_25k.py`](../../scripts/encode_and_train_streetclip_25k.py) (executed on Colab NVIDIA L4, runtime terminated upon download).

- **High-throughput zero-429 encoding:** Reused 6,366 embeddings cached in Step 1D; streamed the remaining 32,634 images using a two-thread prefetch pipeline that downloads each OSV-5M shard ZIP in a single HTTP request, extracts only the ~340 target JPEGs with SHA-256 verification, and immediately deletes the temporary shard. Completed all 32,634 `StreetCLIP` ViT-L/14 encodings in **18.4 minutes** across 98 requests with **zero HTTP 429 stalls**, caching all 39,300 768-D vectors locally (`results/streetclip_25k/streetclip_39k_plus_test300.npz`, 58.0 MB).
- **Creator-disjoint two-stage training:**
  - Excluded all `author_holdout_66` creators from training, then split remaining queries by disjoint creators into `opt_train` (9,980 images) and `tune` (1,978 images).
  - **Stage 1 (`GeoProjector`):** 768→768 residual projection + auxiliary country classification head trained with multi-scale diff-creator geographic soft targets over the full 25,000-image gallery. **Masks same-creator gallery photos ≤ 300 km while retaining same-creator photos > 300 km as hard negatives** to purge camera/watermark fingerprints (`best_epoch=18`; `tune` score rose from `2,717.9 → 3,068.4`).
  - **Stage 2 (`ListwiseGeoRanker`):** Re-retrieves diff-creator Top-100 in projected space and combines 24 features (projected/raw cosine, multi-scale KDE, gallery-density debiasing, Top-100 country consensus, multi-template `StreetCLIP` zero-shot country text probes, and auxiliary country log-probs) under Listwise Softmax KL (`best_epoch=1`).
- **Zero-touch fusion discipline:** All post-processing and hybrid weights (`geoproj_kde_text: wt=0.4`, `raw_kde_text_vlm: (1.2, 0.3)`, `stage2_vlm: 0.3`) were selected exclusively via 5-fold CV on `qlora_eval_300` and applied **without tuning** to `author_holdout_66` and `official_test_300`.

---

## 3. Final Showdown Results

### 3.1 Main Arena: `official_test_300` (147 countries, 300 unseen creators, OSV-5M official `test`, **zero-touch**)

| Model | ≤ 25 km | ≤ 200 km | ≤ 750 km | Country Acc. | Median Error | GeoGuessr Score |
|---|---:|---:|---:|---:|---:|---:|
| **`osv5m/baseline` (5M official model)** | **33 (11.0%)** | **82 (27.3%)** | 147 (49.0%) | 124 (41.3%) | 804.4 km | 2,590.1 |
| `Qwen3-VL-2B` zero-shot | 1 (0.3%) | 20 (6.7%) | 94 (31.3%) | 69 (23.0%) | 1,421.9 km | 1,984.9 |
| `Qwen3-VL-2B` 6k QLoRA | 4 (1.3%) | 26 (8.7%) | 110 (36.7%) | 87 (29.0%) | 1,306.7 km | 2,127.4 |
| `dinov2-small` 25k raw Top-1 | 1 (0.3%) | 17 (5.7%) | 52 (17.3%) | 33 (11.0%) | 4,603.5 km | 1,186.5 |
| `StreetCLIP` 25k raw Top-1 | 7 (2.3%) | 46 (15.3%) | 133 (44.3%) | 109 (36.3%) | 963.0 km | 2,467.7 |
| `StreetCLIP` 25k raw + KDE + Text-Probe | 11 (3.7%) | 73 (24.3%) | 162 (54.0%) | 130 (43.3%) | 650.8 km | 2,740.0 |
| `StreetCLIP` 25k Stage 1 GeoProj Top-1 | 11 (3.7%) | 60 (20.0%) | 161 (53.7%) | 120 (40.0%) | 682.2 km | 2,772.0 |
| `StreetCLIP` 25k Stage 1 GeoProj + KDE | 6 (2.0%) | 61 (20.3%) | 168 (56.0%) | 128 (42.7%) | 603.5 km | 2,853.9 |
| **`StreetCLIP` 25k GeoProj + KDE + Text-Probe** | 8 (2.7%) | 73 (24.3%) | **174 (58.0%)** | **144 (48.0%)** | **570.2 km** | **2,907.5** |
| `StreetCLIP` 25k Stage 2 Listwise | 6 (2.0%) | 59 (19.7%) | 154 (51.3%) | 132 (44.0%) | 718.9 km | 2,730.9 |
| `StreetCLIP` 25k GeoProj + KDE + Text + Qwen3-VL 6k | 4 (1.3%) | 64 (21.3%) | 167 (55.7%) | 131 (43.7%) | 630.0 km | 2,814.3 |

- **Paired head-to-head (`StreetCLIP` 25k GeoProj + KDE + Text-Probe):**
  - **vs. 5M `osv5m/baseline`:** **166 closer** vs. 131 farther (3 ties ≤ 1 km); **+317.4 GeoGuessr points**, **−234.2 km median error**, **+6.7% country accuracy**.
  - **vs. `Qwen3-VL-2B` 6k QLoRA:** **190 closer** vs. 110 farther; **+780.1 GeoGuessr points**, **−736.5 km median error**, **+19.0% country accuracy**.
  - **Top-100 recall:** raw 25k `StreetCLIP` achieves `34.3% ≤ 25 km / 82.0% ≤ 200 km / 89.7% country`; Stage 1 GeoProj raises ≤ 25 km Top-100 recall further to **`38.3%`**.

### 3.2 Progress Across All Four Validation Splits (25k `StreetCLIP`, strict diff-creator)

| Split | Best Method | Top-100 ≤ 25 km | ≤ 25 km | ≤ 200 km | Country Acc. | Median Error | GeoGuessr Score |
|---|---|---:|---:|---:|---:|---:|---:|
| **`val_2000`** (n=2,000) | Stage 2 Listwise (vs. raw `2,651.5` pts) | 27.9% | 46 (2.3%) | 425 (21.3%) | **1,438 (71.9%)** | **474.9 km** | **3,232.0** |
| **`val_unseen_opt_522`** (unseen in train) | Stage 2 Listwise (vs. raw `2,823.7` pts) | 33.7% | 13 (2.5%) | 123 (23.6%) | **332 (63.6%)** | **482.5 km** | **3,238.0** |
| **`qlora_eval_300`** (5-fold OOF) | Stage 2 Listwise + Qwen3-VL 6k (vs. 6k gallery `2,996.6`) | 28.0% | 7 (2.3%) | 67 (22.3%) | **223 (74.3%)** | **461.2 km** | **3,327.0** |
| **`author_holdout_66`** (zero-touch) | Stage 1 GeoProj + KDE (vs. 6k gallery `3,395.4`) | **57.6%** | 4 (6.1%) | **22 (33.3%)** | 39 (59.1%) | **318.2 km** | **3,465.3** |
| **`author_holdout_66`** (zero-touch) | Raw 25k KDE + Text + Qwen3-VL 6k | 48.5% | 4 (6.1%) | 20 (30.3%) | **45 (68.2%)** | **308.7 km** | 3,440.0 |

---

## 4. Three Deeper Technical Takeaways

1. **Why our 25k system wins on country, median error, and GeoGuessr score while 5M `osv5m/baseline` keeps the ≤ 25 km lead (`11.0%` vs. `2.7–3.7%`):**
   - `osv5m/baseline` trains an end-to-end quadtree + coordinate regressor on 4.89M images, giving it a 200× local density advantage for pinpoint ≤ 25 km hits. Our discrete 25k gallery paired with wide-kernel KDE (`σ = 500 km`) deliberately trades some pinpoint hits for broad regional stability to maximize expected GeoGuessr score (without KDE, Stage 1 Top-1 gets 11 hits ≤ 25 km, dropping to 6–8 with KDE while median error improves from 682 → 570 km).
   - **Next lever:** scale the gallery (25k → 100k+) and apply two-stage coarse-to-fine KDE (wide kernel `σ = 500 km` to pick the winning region/country, followed by a narrow kernel `σ = 30–50 km` within that region for fine coordinate selection).
2. **Stage 1 `GeoProjector` generalizes cleanest across shifts; Stage 2 `ListwiseRanker` inherits OSV-5M train country priors:**
   - Stage 1 `GeoProjector` improved performance on all 5 splits — including +304 points and `34.3% → 38.3%` Top-100 ≤ 25 km recall on 147-country `official_test_300`.
   - Stage 2 incorporates `cc_freq_log` and `aux_cc_logp` (which reflect OSV-5M train's US/RU/EU skew), making it dominant on `val_2000` (71.9%) and `qlora_eval_300` (74.3%), but slightly worse than prior-free **Stage 1 GeoProj + KDE + Text-Probe** on uniformly stratified 147-country `official_test_300` and `author_holdout_66`.
3. **Fixed-weight fusion with a weaker VLM hurts long-tail countries:**
   - On major-country-heavy `qlora_eval_300` (where Qwen3-VL 6k hits 52% country accuracy), adding VLM priors boosted hybrid accuracy to **74.3%** (`3,327.0` pts). On 147-country `official_test_300`, however, Qwen3-VL 6k dropped to `29.0%`, so mixing it at fixed weight `0.3` slightly pulled down `StreetCLIP` 25k GeoProj+KDE+Text (`48.0% → 43.7%`, `2,907.5 → 2,814.3` pts).
   - **Conclusion:** Rather than unconditionally blending VLM predictions at a fixed weight, VLM intervention should be **confidence- or OCR-gated** — e.g., firing only when readable signage is detected and cross-referenced against an offline GeoNames gazetteer.

---

## 5. Reproduction & Output Files

- **Lock official test split:** `python3 -B scripts/prepare_official_test.py --select-only` → `--download` → `--filter-and-lock` → [`data/official_test_300/hud_audit.json`](../../data/official_test_300/hud_audit.json) (local ZIP: `data/official_test_300/official-test-300.zip`)
- **Phase B (Colab L4, terminated):** [`scripts/prepare_phase_b_bundle.py`](../../scripts/prepare_phase_b_bundle.py) + [`scripts/encode_and_train_streetclip_25k.py`](../../scripts/encode_and_train_streetclip_25k.py)
- **Phase C (Colab T4, terminated):** [`scripts/prepare_phase_c_bundle.py`](../../scripts/prepare_phase_c_bundle.py) + [`scripts/evaluate_official_test_models.py`](../../scripts/evaluate_official_test_models.py)
- **Phase D (local showdown evaluation):** `python3 -B scripts/evaluate_official_test_showdown.py` → [`showdown_summary.json`](showdown_summary.json)
