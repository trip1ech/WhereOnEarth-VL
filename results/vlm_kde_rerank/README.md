# Retrieval-Augmented Spatial KDE → `Qwen3-VL-8B-Instruct` Reranking & Coordinate Refinement

This stage integrates **25k `StreetCLIP` Geo-Metric Retrieval + Spatial KDE** (System 1) with **`Qwen/Qwen3-VL-8B-Instruct`** (System 2, 4-bit NF4 on Colab L4/T4) so the VLM directly inspects the street-view image alongside the retrieval KDE posterior distribution, contrasts competing countries/regions, and outputs refined coordinates.

---

## 1. Architecture & Why Two-Stage Beats Single-Pass

1. **Spatial KDE Distribution Summarization ([`scripts/kde_vlm_utils.py`](../../scripts/kde_vlm_utils.py)):**
   For each query, we take the Top-100 creator-disjoint candidates from Stage 1 25k `StreetCLIP` `GeoProjector`, compute wide-kernel spatial KDE (`τ=0.03, σ=500 km`) + `0.4 × Text-Probe`, and condense them into the **Top-6 distinct countries** (each with KDE posterior share %, neighbor count, max similarity, primary peak coordinates, optional secondary regional cluster > 450 km away, and StreetCLIP Text-Probe rank).
   - Country recall inside Top-6: **`92.4%` (61/66)** on `author_holdout_66` and **`73.7%` (221/300)** on 147-country `official_test_300`.
2. **Development Finding on `author_holdout_66` (Single-Pass Anchoring vs. Two-Stage):**
   - **Single-pass conservative prompt (`v1`):** Clean confidence calibration (`high=4, medium=3, low=59`). Confidence-gated reranking (`Mode C`) improved `author_holdout_66` from `63.6% → 66.7%` country accuracy and `3,446.6 → 3,469.4` GeoScore (7 closer vs. 2 farther, flipping e.g. Sri Lanka 411 km → Kerala, India 26.7 km). However, when a clear sign's country had low KDE mass (e.g., `BOUAKÉ`, Côte d'Ivoire), the VLM sometimes anchored to a higher-KDE neighbor.
   - **Single-pass aggressive override prompt (`v2`):** Rescued `BOUAKÉ` (588 → 316 km) and Pakistan (419 → 184 km), but inflated `high` confidence (`4 → 32/66`), broke calibration, and caused two blurry-sign misreads (`3,440.2` pts).
   - **Locked Two-Stage System (`WhereOnEarth-VL`):** Run an **unbiased standalone visual read** (`Qwen3-VL-8B-Instruct` zero-shot) alongside the **v1 calibrated KDE-conditioned reranker**:
     - If the unbiased visual read returns **`confidence == "high"`** with valid coordinates (typically clear city/highway signage, postal codes, or unmistakable scripts — e.g., `BOUAKÉ`, `CORREOS DE CUBA C.P. 27200`, `Algeciras A-7`, `Pizza maru`, `Oeuvres Universitaires de Dakar`), trust the VLM's refined coordinates directly.
     - Otherwise, apply the **v1 confidence-gated KDE reranker** (accepting VLM reranking/refinement when `confidence != "low"`, and staying with KDE `#1` when clues are ambiguous).

---

## 2. Results

### 2.1 Development Validation: `author_holdout_66` (66 unseen photographers / 60 countries)

| Method | Country Acc. | ≤ 25 km | ≤ 200 km | Median Error | GeoScore |
|---|---:|---:|---:|---:|---:|
| `Qwen3-VL-8B-Instruct` zero-shot (no KDE) | 42.4% (28/66) | 3 (4.5%) | 10 (15.2%) | 869.1 km | 2,437.3 |
| `StreetCLIP` 25k GeoProj + KDE + Text (pure retrieval) | 63.6% (42/66) | 4 (6.1%) | 21 (31.8%) | 356.2 km | 3,446.6 |
| `KDE → Qwen3-VL-8B` v2 aggressive single-pass | 65.1% (43/66) | 4 (6.1%) | 21 (31.8%) | 344.0 km | 3,440.2 |
| `KDE → Qwen3-VL-8B` v1 calibrated + gated (Mode C) | 66.7% (44/66) | 4 (6.1%) | 22 (33.3%) | 344.0 km | 3,469.4 |
| **`WhereOnEarth-VL` Two-Stage (Locked)** | **68.2% (45/66)** | **5 (7.6%)** | **23 (34.8%)** | **313.6 km** | **3,494.9** |

### 2.2 Locked Final Evaluation: `official_test_300` (300 unseen photographers / 147 countries, zero tuning on test)

| Method | ≤ 25 km | ≤ 200 km | ≤ 750 km | Country Acc. | Median Error | GeoScore |
|---|---:|---:|---:|---:|---:|---:|
| `osv5m/baseline` (5M official model) | **33 (11.0%)** | **82 (27.3%)** | 147 (49.0%) | 41.3% (124) | 804.4 km | 2,590.1 |
| `Qwen3-VL-2B` zero-shot | 1 (0.3%) | 20 (6.7%) | 94 (31.3%) | 23.0% (69) | 1,421.9 km | 1,984.9 |
| `Qwen3-VL-2B` 6k QLoRA | 4 (1.3%) | 26 (8.7%) | 110 (36.7%) | 29.0% (87) | 1,306.7 km | 2,127.4 |
| `Qwen3-VL-8B-Instruct` zero-shot (no KDE) | 6 (2.0%) | 38 (12.7%) | 124 (41.3%) | 28.0% (84) | 1,088.4 km | 2,212.8 |
| `StreetCLIP` 25k GeoProj + KDE + Text (pure retrieval) | 8 (2.7%) | 73 (24.3%) | 174 (58.0%) | 48.0% (144) | 570.2 km | 2,907.5 |
| `KDE → Qwen3-VL-8B` (Mode D: high-conf rerank only) | 8 (2.7%) | 73 (24.3%) | 174 (58.0%) | **49.0% (147)** | 561.8 km | 2,921.5 |
| **`WhereOnEarth-VL` Two-Stage (Locked, VLM Refined)** | **10 (3.3%)** | **76 (25.3%)** | **174 (58.0%)** | 48.0% (144) | **556.7 km** | **2,945.8** |
| `WhereOnEarth-VL` Two-Stage (Narrow Local KDE, σ=50 km) | **13 (4.3%)** | 67 (22.3%) | 173 (57.7%) | 48.0% (144) | 579.0 km | 2,901.7 |

- **Head-to-head on `official_test_300`:**
  - **vs. 5M `osv5m/baseline`:** **170 closer** vs. 126 farther (4 ties ≤ 1 km) — **+355.7 GeoScore points**, **−247.7 km median error**, **+6.7% country accuracy**.
  - **vs. pure 25k `StreetCLIP` retrieval:** **+38.3 GeoScore points**, **−13.5 km median error**, **≤ 25 km hits `8 → 10`**, **≤ 200 km hits `73 → 76`** (16 closer vs. 13 farther).
  - **vs. standalone `Qwen3-VL-8B-Instruct` zero-shot:** **+733.0 GeoScore points** (`2,212.8 → 2,945.8`), **+20.0% country accuracy** (`28.0% → 48.0%`).

---

## 3. Notable Qualitative Wins on `official_test_300`

- **South Korea (`376978833655174`):** Pure retrieval guessed Taiwan (`1,544.3 km`); `Qwen3-VL-8B` read Korean signage (`"Pizza maru"`) and Korean license plates → **`10.0 km` error (−1,534 km)**.
- **Senegal (`259669532567430`):** Pure retrieval guessed Burkina Faso (`1,286.3 km`); `Qwen3-VL-8B` read `"Oeuvres Universitaires de Dakar"` on a bus → **`174.8 km` error (−1,112 km)**.
- **Iran (`360873282297562`):** Pure retrieval guessed Uzbekistan (`1,541.8 km`); `Qwen3-VL-8B` recognized Persian script on Islamic tilework → **`278.8 km` error (−1,263 km)**.
- **Spain (`909312326294088` & `229506101848921`):** Pure retrieval guessed Spain at `1,800.4 km` and `457.8 km`; `Qwen3-VL-8B` refined coordinates (reading `"Algeciras"` & `"A-7"` on the highway sign) → **`118.3 km` (−1,682 km)** and **`43.0 km` (−415 km)**.
- **Cuba (`314824470019196`):** Pure retrieval guessed Cuba at `651.9 km`; `Qwen3-VL-8B` read `"CORREOS DE CUBA C.P. 27200"` → **`71.4 km` (−580 km)**.
- **Malaysia (`1619395298260674`):** Pure retrieval guessed Malaysia at `255.9 km`; `Qwen3-VL-8B` refined to **`0.0 km` (−256 km)**.

---

## 4. Reproduction & Outputs

- Scripts: [`scripts/kde_vlm_utils.py`](../../scripts/kde_vlm_utils.py), [`scripts/run_qwen8b_kde_rerank.py`](../../scripts/run_qwen8b_kde_rerank.py), [`scripts/prepare_vlm_rerank_bundle.py`](../../scripts/prepare_vlm_rerank_bundle.py), [`scripts/evaluate_vlm_rerank.py`](../../scripts/evaluate_vlm_rerank.py).
- Summaries: [`author_holdout_66/evaluation_summary.json`](author_holdout_66/evaluation_summary.json), [`author_holdout_66_v2/evaluation_summary.json`](author_holdout_66_v2/evaluation_summary.json), [`official_test_300/evaluation_summary.json`](official_test_300/evaluation_summary.json).
- All Colab L4/T4 sessions were terminated immediately after downloading results (0 active sessions).
