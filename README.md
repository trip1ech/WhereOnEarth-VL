# WhereOnEarth-VL

> **Can a 25,000-image street-view retrieval & vision-language system beat a 5,000,000-image official geolocalization model — without relying on camera or photographer shortcuts?**

**WhereOnEarth-VL** is an end-to-end research log and codebase for **single-image global street-view geolocalization** (predicting country and GPS coordinates from one raw street photo — no EXIF, no metadata, no sequence context).

Over six days and **11 iterative experimental stages** (2026-11-18 → 2026-11-23), this project evolved from a 36-image French pilot into a systematic investigation of **Vision-Language Models (Qwen3-VL-2B QLoRA)**, **mechanistic visual & text-attention diagnostics**, **dataset leakage forensics**, and **anti-leakage metric learning (`geolocal/StreetCLIP` + Geo-Projection + Spatial KDE)** — culminating in a zero-touch head-to-head showdown on **300 unseen photographers across 147 countries** from the official OSV-5M test set.

---

## Headline Result (`official_test_300` — Zero-Touch Official OSV-5M Test)

Evaluated on **[`official_test_300`](results/official_test_showdown/README.md)** — 300 images from 300 strictly unseen photographers spanning **147 countries** (OSV-5M official `test`, ≥ 2018-01-01, Apple Vision OCR–audited to strip GPS HUD overlays, zero hyperparameter tuning on test):

| Model | Training / Gallery Scale | ≤ 25 km | ≤ 200 km | ≤ 750 km | Country Acc. | Median Error | GeoGuessr Score |
|---|---|---:|---:|---:|---:|---:|---:|
| **`StreetCLIP` 25k GeoProj + KDE + Text-Probe (Ours)** | **25k gallery + 10k proj.** | 2.7% (8) | 24.3% (73) | **58.0% (174)** | **48.0% (144)** | **570.2 km** | **2,907.5** |
| `osv5m/baseline` (Official Reference) | **4.89M train images** | **11.0% (33)** | **27.3% (82)** | 49.0% (147) | 41.3% (124) | 804.4 km | 2,590.1 |
| `StreetCLIP` 25k raw + KDE + Text-Probe | 25k gallery (frozen) | 3.7% (11) | 24.3% (73) | 54.0% (162) | 43.3% (130) | 650.8 km | 2,740.0 |
| `StreetCLIP` 25k GeoProj + KDE + Text + Qwen3-VL 6k | 25k + 6k QLoRA | 1.3% (4) | 21.3% (64) | 55.7% (167) | 43.7% (131) | 630.0 km | 2,814.3 |
| `Qwen3-VL-2B` 6k QLoRA (Pure VLM) | 6k SFT images | 1.3% (4) | 8.7% (26) | 36.7% (110) | 29.0% (87) | 1,306.7 km | 2,127.4 |
| `Qwen3-VL-2B` Zero-Shot | — | 0.3% (1) | 6.7% (20) | 31.3% (94) | 23.0% (69) | 1,421.9 km | 1,984.9 |
| `dinov2-small` 25k raw Top-1 (diff-creator) | 25k gallery (frozen) | 0.3% (1) | 5.7% (17) | 17.3% (52) | 11.0% (33) | 4,603.5 km | 1,186.5 |

- **Head-to-head vs. 5M `osv5m/baseline`:** Our 25k system lands closer on **166 of 300 images** (131 losses, 3 ties ≤ 1 km), outperforming the 5M-image official baseline by **+6.7% country accuracy**, **−234.2 km median error**, and **+317.4 GeoGuessr points**.
- **On the main-country OSV-5M validation distribution (`qlora_eval_300`, 5-fold OOF, diff-creator):** Fusing 25k `StreetCLIP` Stage 2 Listwise Ranker with `Qwen3-VL-2B` 6k country priors reaches **74.3% country accuracy (223/300)**, **461.2 km median error**, and **3,327.0 GeoGuessr points**.
- **On the hand-audited 66-photographer unseen holdout (`author_holdout_66`, zero-touch):** 25k `StreetCLIP` Stage 1 GeoProj + KDE hits **318.2 km median error**, **33.3% ≤ 200 km**, and **3,465.3 GeoGuessr points** (up from 2,788.6 for pure VLM).

---

## The Research Story: Five Acts

Rather than presenting only the final checkpoint, this repository preserves the full investigative arc — including dead ends, sanity checks, and a major data-leakage discovery that reshaped the entire architecture.

> Looking for the full 11-stage ledger and file index? See **[EXPERIMENTS.md](EXPERIMENTS.md)**.

### Act I · Honest Protocols & The Coverage Bottleneck (Stages 1–2)

We started by building a strict anti-leakage evaluation protocol: EXIF-oriented RGB pixels only, real GPS/country/EXIF withheld until scoring, and sequence-disjoint splits. Our first pilot paired 18 `pilot_dev` and 18 `pilot_holdout` photos across Paris, Lyon, and Toulouse from **two distinct platforms** (OSV-5M official test + Panoramax IGN, median cross-platform pair distance 1.97 km).

On this pilot, the specialized `osv5m/baseline` easily beat zero-shot `Qwen3-VL-2B` (22.2% vs. 0% @25 km on OSV). When we tested a naive 2,000-image frozen `dinov2-small` retrieval pipeline, it got **0/18 @25 km** — not because retrieval was broken, but because auditing all **4.89M OSV-5M train rows** revealed the 2k gallery only had a ≤ 25 km neighbor for **2/18** queries (vs. **18/18** in full train).

![OSV-5M global distribution across train, test, and the initial 2k gallery](results/distribution_audit/world_distribution.png)

That audit motivated a **coverage-first 25,000-image gallery** sampled from post-2018 OSV-5M train (~11 km grid candidates, greedily maximizing 25 km coverage): on 6,000 unseen post-2018 coverage queries, this gallery achieved **97.32%** 25 km coverage compared to **73.97%** for a random sequence-disjoint gallery of the same size.

---

### Act II · Fine-Tuning Qwen3-VL-2B — Strong First Jump, Fast Plateau (Stages 4–5)

Next, we asked how far supervised fine-tuning of a compact 2B VLM could go. Using 4-bit NF4 QLoRA (rank 8, LM attention layers, 1 epoch, 512 px longest edge), we trained `Qwen3-VL-2B-Instruct` to emit structured JSON `{"country","latitude","longitude"}` on stratified post-2018 subsets:

| Evaluation Set | Model | Valid Coords | Country Acc. | ≤ 25 km | ≤ 200 km | Median Error |
|---|---|---:|---:|---:|---:|---:|
| `qlora_eval_300` (n=300) | Zero-shot | 294 / 300 | 44.0% (132) | 1.3% (4) | 10.7% (32) | 1,141 km |
| `qlora_eval_300` (n=300) | 2k QLoRA | 300 / 300 | 51.3% (154) | 3.0% (9) | 14.7% (44) | 861 km |
| `qlora_eval_300` (n=300) | 6k QLoRA | 300 / 300 | **52.0% (156)** | 2.7% (8) | 14.7% (44) | **832 km** |
| `author_holdout` (66 unseen creators / 60 countries) | Zero-shot | 65 / 66 | 42.4% (28) | 1.5% (1) | 12.1% (8) | 945 km |
| `author_holdout` (66 unseen creators / 60 countries) | 6k QLoRA | 66 / 66 | **50.0% (33)** | 1.5% (1) | **21.2% (14)** | **589 km** (600.3 all) |

Two findings stood out:
1. **The gains genuinely generalized to unseen photographers** — on the hand-audited [`author_holdout`](data/author_holdout/README.md) split (66 photographers with verified zero `creator_id` overlap), 6k QLoRA moved predictions closer on **42/65** valid photos and cut median error by **356 km**.
2. **Tripling training data (2k → 6k) hit a wall** — gaining only +2 correct countries on 300 eval images. Plain token cross-entropy over 3-decimal coordinates has no geographic geometry (flipping a minus sign costs 1 token, the same as a 0.001° nudge).

---

### Act III · Opening the Black Box: Vision Bottleneck or Reasoning Bottleneck? (Stages 6–9)

Before spending compute unfreezing the vision encoder, we ran four targeted diagnostic studies:

1. **Road Center-Line Probes & Crop Diagnosis ([`roadmark_probe`](results/roadmark_probe/README.md), [`roadmark_crop_probe`](results/roadmark_crop_probe/README.md)):**
   We built a local blind [annotation tool](annotation_tool/README.md) and manually labeled 70 images (23 single-line, 20 double-line). On 11 same-country pairs (22 images), simply **cropping around the center line** without changing a single model weight jumped accuracy sharply:
   - Frozen Projector + DeepStack linear probe: **54.5% → 81.8%** (12 → 18/22)
   - Base Qwen3-VL QA: **63.6% → 81.8%** (14 → 18/22)
   - 6k LoRA QA: **72.7% → 86.4%** (16 → 19/22), ranking **11/11** same-country pairs correctly.
   - **Takeaway:** The frozen ViT already encodes fine-grained road cues once resolution/framing is adequate; there was no evidence to unfreeze ViT.

2. **Scene Text & LM Attention Interventions ([`text_attention`](results/text_attention/README.md)):**
   On three diagnostic cases (a Canadian dashcam with literal GPS coordinates `W111.421436 N50.36211`, a Guadeloupe French trail sign `POINTE DES CHATEAUX`, and a Brazilian storefront), **the 6k fine-tuned model already transcribed the text accurately from the full uncropped image**. Yet even when we injected **human-verified transcripts** directly into the prompt:
   - On Canada, the 6k model copied the latitude accurately but emitted `+111.421` instead of `-111.421` (8,102 km error, country `TM`).
   - On Guadeloupe, both base and fine-tuned models still guessed Europe (6,909 km error).
   - **Takeaway:** A 2B model lacks both deterministic sign consistency and world-scale toponym knowledge — it needs external geocoding/gazetteer lookup, not more ViT fine-tuning.

3. **Clue-First Prompting & Continued LM LoRA Training ([`clue_first`](results/clue_first/README.md), [`clue_first_training`](results/clue_first_training/README.md)):**
   Two-stage zero-shot prompting ("read signs first, then geolocate") **hurt** accuracy (`33 → 30/66` countries) because the model hallucinated place names on signless roads (e.g., inventing `Saray` on an Ethiopian mountain track).

   We then curated 104 manually verified photos (258 multi-task questions across sign transcription, empty-text abstention, structured road attributes, and clue-conditioned localization) and continued LM LoRA training for 69 steps:

   | Example Training Inputs (OSV-5M, CC BY-SA 4.0) | |
   |---|---|
   | ![Sign example: MERCADO DE PEIXE](results/clue_first_training/method_examples/market-sign.jpg)<br>*Sign transcription:* `{"text":["MERCADO DE PEIXE"]}` | ![Unpaved road example in Ghana](results/clue_first_training/method_examples/unpaved-road.jpg)<br>*Multi-task:* empty text `[]`, road JSON `{"surface":"unpaved","centerline":"none","roadside":[]}`, and GPS |

   - **Clue extraction soared:** roadside-object exact match jumped **7/29 → 27/29** (micro-F1 0.938), empty-text abstention hit **21/21** (up from 11/21), and valid JSON reached **66/66**.
   - **Localization regressed:** direct country accuracy dropped **33 → 30/66** (≤ 200 km `14 → 10/66`) due to task dilution and overly conservative OCR abstention (59/66 blank).
   - **Takeaway:** A single 2B adapter should not juggle clue extraction and coordinate prediction; keep them as separate specialized adapters.

---

### Act IV · The Plot Twist: Uncovering a 97.2% Same-Creator Shortcut Leak (Stages 3 & 10)

Returning to retrieval, our earlier 25k frozen `dinov2-small` run had achieved **14.5% @25 km (290/2,000)** and **49.2% Top-100 @25 km** on `val_2000` (while a naive pairwise softplus ranker had collapsed at Epoch 1 from 152 → 8 hits).

Before building on top of it, **we manually inspected 15 matched query–gallery photo pairs.** What we found changed everything ([`results/retrieval_v2/README.md`](results/retrieval_v2/README.md)):

- **97.2% (282 / 290)** of `dinov2-small`'s Top-1 ≤ 25 km hits — and **100% (44 / 44)** on `qlora_eval_300` — were **same-photographer / same-vehicle shortcuts**!
- Even though sequences were strictly disjoint, 92.8% of `val_2000` creators also appeared in the 25k gallery, and the top 20 prolific contributors accounted for 28.1% of gallery images. DINOv2 was matching identical windshield stickers, car hoods, and dashcam GPS watermarks across different Mapillary sequences — even matching the same dashcam watermark **647 km apart**.
- **Enforcing a strict diff-creator mask** exposed the real `dinov2-small` baseline on `val_2000`: Top-1 ≤ 25 km collapsed from `14.5%` to **`0.45%` (9/2,000)**, country accuracy fell to **`21.4%`**, and median error exploded to **`3,965.0 km`**.

From that moment on, every retrieval experiment obeyed an **iron anti-leakage rule** (always masking same-`attribution` gallery matches, excluding all `author_holdout_66` creators from training, and tuning fusion weights strictly via 5-fold OOF on `qlora_eval_300` with zero touch on holdout/test):

1. **Step 1A — Spatial KDE Voting:** Replacing greedy Top-1 with kernel density voting over Top-50 diff-creator neighbors immediately cut `dinov2-small` median error from `3,965 → 2,737 km`.
2. **Step 1B — Full-Gallery Geo-Metric Projection + Listwise Ranker:** We trained a shared residual projector with multi-scale geographic soft targets (35 / 180 / 600 km + same-country), **masking same-creator pairs ≤ 300 km while keeping same-creator pairs > 300 km as explicit hard negatives** to wash out camera fingerprints. Followed by a 22-feature scale-normalized Listwise Softmax KL ranker (fixing the Epoch-1 collapse), `dinov2-small` jumped on `val_2000` from **21.4% → 49.9%** country accuracy and **3,965 → 1,032 km** median error.
3. **Step 1C — Exact 211-Country VLM Log-Prob Fusion:** Using a single KV-cache forward pass after `{"country":"` on `Qwen3-VL-2B` 6k, we extracted exact log-probabilities over all 211 countries and fused them with retrieval (`qlora_eval_300` country accuracy `52.0% → 58.3%`).
4. **Step 1D — Swapping Backbone to `geolocal/StreetCLIP` (6k Proof of Concept):** On the *exact same* 6,000-image clean gallery (4× smaller than 25k), swapping `dinov2-small` for geo-pretrained `geolocal/StreetCLIP` (ViT-L/14) + bottom-12% HUD crop + Spatial KDE + zero-shot country text probe + Qwen3-VL priors catapulted zero-touch `author_holdout_66` from `16.7%` → **`69.7%` country accuracy**, **`389.7 km` median error**, and **`3,395.4` GeoGuessr points**.

---

### Act V · Scaling StreetCLIP to 25k & The Official Test Showdown (Stage 11)

With the proof of concept validated, we scaled up for a completely clean, leak-free showdown against the 5M-image official `osv5m/baseline` ([`results/official_test_showdown/README.md`](results/official_test_showdown/README.md)):

1. **Locked `official_test_300`:** Sampled 300 images from OSV-5M official `test` (≥ 2018-01-01) using square-root country stratification + 4° grid round-robin across **147 countries** and **300 completely unseen photographers** (zero overlap with 25k gallery, 12k train, 2k val, 6k QLoRA, and `osv5m/baseline` train). Audited all 340 candidates with macOS Apple Vision OCR, catching and removing 1 Garmin dashcam GPS overlay (`38.41452 141.23285 20 KM/H`).
2. **Zero-429 Streaming Encoding:** Streamed all 32,634 remaining images (reusing 6,366 cached vectors) via whole-shard HTTP downloads on Colab L4 (98 requests, zero rate-limit stalls, **18.4 minutes** total), caching all 39,300 768-D `StreetCLIP` embeddings.
3. **Anti-Leakage 25k Geo-Ranker:** Trained Stage 1 `GeoProjector` (768→768 residual + auxiliary country head, best epoch 18) and Stage 2 `ListwiseGeoRanker` (24 features, best epoch 1), then evaluated zero-touch on `official_test_300` and all four validation splits.

#### Three Key Insights from the Showdown
1. **Why our 25k system beats 5M `osv5m/baseline` globally (+317 pts) while `osv5m/baseline` keeps the ≤ 25 km crown (11.0% vs. 2.7–3.7%):** `osv5m/baseline` has 200× more training density for pinpoint localization, whereas our wide-kernel KDE (`σ = 500 km`) deliberately hedges toward regional safety to maximize expected GeoGuessr score (without KDE, Stage 1 Top-1 gets 11 hits ≤ 25 km vs. 8 with KDE, but median error degrades from 570 → 682 km). Coarse-to-fine KDE (`σ = 500 km` region selection → `σ = 30–50 km` pinpoint selection) plus scaling the gallery to 100k+ is the direct path to closing the ≤ 25 km gap.
2. **Stage 1 `GeoProjector` wins across uniform 147-country long-tail; Stage 2 `ListwiseRanker` wins on OSV-5M val distribution:** Stage 2 uses country-frequency features (`cc_freq_log`, `aux_cc_logp`) that excel on OSV-5M's US/RU/EU-heavy `val_2000` (**71.9%** country accuracy) and `qlora_eval_300` (**74.3%** with VLM), whereas unbiased **Stage 1 GeoProj + KDE + Text-Probe** generalizes strongest to 147-country `official_test_300` (**48.0%**).
3. **Don't fuse a weak VLM at fixed weight on long-tail countries:** Adding `Qwen3-VL-2B` 6k (52% accuracy on `qlora_eval_300`, but 29% on 147-country `official_test_300`) at fixed weight `0.3` boosted `qlora_eval_300` (+3.7% country) but pulled `official_test_300` down (`48.0% → 43.7%`). **Next step:** confidence/OCR-gated VLM intervention backed by an offline GeoNames gazetteer when readable signage is detected.

---

## Evaluation Splits & Anti-Leakage Map

| Split | Size | Source | Disjointness Guarantee | Role |
|---|---:|---|---|---|
| `pilot_dev` / `pilot_holdout` | 18 / 18 | OSV-5M `test` (FR) + Panoramax IGN | Disjoint sequences, ≥ 10.43 km split gap | Cross-platform pipeline & protocol check |
| `since_2018` ranker splits | 25k gallery / 12k train / 2k val / 6k cov | OSV-5M official `train` (≥ 2018) | Disjoint image IDs & sequences (diff-creator enforced at train/eval) | Retrieval & Geo-Ranker training / validation |
| `qlora_since_2018` | 2k & 6k train / 300 eval | Subsets of `since_2018` train & val | Disjoint IDs, sequences, SHA-256 | QLoRA SFT & 5-fold OOF fusion tuning |
| `author_holdout` | 66 (66 creators, 60 countries) | `since_2018` val (66/72 hand-audited) | Zero `creator_id` overlap with 6k train & 300 eval | Unseen-creator dev / diagnostic holdout |
| **`official_test_300`** | **300 (300 creators, 147 countries)** | **OSV-5M official `test` (≥ 2018, OCR-audited)** | **Zero overlap with all train/val/gallery & `osv5m/baseline` train** | **Zero-touch final benchmark** |
| `roadmark_annotations` | 70 (43 binary, 43 creators) | 6k QLoRA train subset (blind-labeled) | Disjoint creators across 5 folds | Center-line ViT/crop probe only |

---

## Repository Structure

```text
WhereOnEarth-VL/
├── README.md                          # Project story & research overview (this file)
├── EXPERIMENTS.md                     # Full 11-stage experiment index, metrics & bottlenecks
├── requirements-colab.txt             # Pinned Colab dependencies
├── notebooks/                         # 7 Colab notebooks (GPU smoke test through QLoRA SFT)
│   ├── 00_colab_gpu.ipynb
│   ├── 01_phase1_colab.ipynb
│   ├── 02_frozen_retrieval.ipynb
│   ├── 03_dataset_distribution.ipynb
│   ├── 04_ranker_preprocessing.ipynb
│   ├── 05_ranker_colab.ipynb
│   └── 06_sft_colab.ipynb
├── scripts/                           # 46 end-to-end data prep, training, probe & eval scripts
├── results/                           # Metrics, figures & stage-by-stage technical reports
│   ├── distribution_audit/            # OSV-5M 4.89M global distribution tables & plots
│   ├── roadmark_probe/README.md       # Frozen ViT / Projector / DeepStack center-line probe
│   ├── roadmark_crop_probe/README.md  # Full-image vs. fixed crop vs. manual crop diagnosis
│   ├── text_attention/README.md       # Scene-text OCR & LM attention heatmaps
│   ├── clue_first/README.md           # Two-stage clue-first prompting diagnosis
│   ├── clue_first_training/README.md  # Multi-task clue-extraction LM LoRA training report
│   ├── retrieval_v2/README.md         # 97.2% shortcut forensic audit & StreetCLIP 6k PoC
│   └── official_test_showdown/README.md # 25k StreetCLIP vs. 5M osv5m/baseline showdown
├── data/                              # Small docs, audit records & human annotations (media ignored)
│   ├── author_holdout/README.md       # Hand-audited 66-creator holdout report & review log
│   ├── official_test_300/hud_audit.json
│   └── roadmark_annotations/labels.json
├── webapp/                            # Offline Leaflet map comparator for 300-image VLM eval
└── annotation_tool/                   # Local blind road center-line annotation web app
```

---

## Reproducibility & Local Tools

> **Note on large artifacts:** Raw street-view images, OSV-5M CSVs, Colab upload ZIPs, cached embeddings (`streetclip_39k_plus_test300.npz`), and LoRA weight archives are excluded via [`.gitignore`](.gitignore). Every dataset split, manifest, and metric table can be regenerated deterministically from pinned OSV-5M and Hugging Face revisions using the scripts below.

- **Key script pipelines:**
  - *Pilot & distribution audit:* `scripts/prepare_pilot.py`, `scripts/build_bundles.py`, `scripts/phase1.py`, `scripts/audit_distribution.py`
  - *Coverage-first 25k gallery & retrieval v1:* `scripts/cache_train_metadata.py`, `scripts/select_ranker_data.py`, `scripts/materialize_ranker_data.py`, `scripts/train_ranker.py`
  - *Qwen3-VL QLoRA & unseen-creator eval:* `scripts/prepare_sft_data.py`, `scripts/sft_colab.py`, `scripts/prepare_author_holdout.py`, `scripts/evaluate_author_holdout.py`
  - *Visual/text probes & clue-first SFT:* `scripts/extract_roadmark_features.py`, `scripts/analyze_roadmark_probe.py`, `scripts/probe_text_attention.py`, `scripts/render_text_attention.py`, `scripts/train_clue_first.py`, `scripts/analyze_clue_training.py`
  - *Anti-leakage retrieval v2 & 25k StreetCLIP official showdown:* `scripts/eval_spatial_kde.py`, `scripts/train_geo_ranker.py`, `scripts/extract_vlm_country_priors.py`, `scripts/prepare_official_test.py`, `scripts/encode_and_train_streetclip_25k.py`, `scripts/evaluate_official_test_models.py`, `scripts/evaluate_official_test_showdown.py`
- **Interactive Prediction Viewer:** See [`webapp/README.md`](webapp/README.md) to rebuild data from local evaluation ZIPs and inspect Zero-shot vs. 2k vs. 6k QLoRA guesses side-by-side on synchronized Leaflet maps.
- **Blind Road-Marking Annotator:** See [`annotation_tool/README.md`](annotation_tool/README.md) to run the local keyboard-driven annotator at `http://127.0.0.1:8766/`.

## References

- [OSV-5M: OpenStreetView-5M Dataset & Baseline (CVPR 2024)](https://arxiv.org/abs/2404.18873) · [Pinned HF Snapshot](https://huggingface.co/datasets/osv5m/osv5m/tree/cff33609b56b54d8743b7ee7a416eb8433e9a681)
- [StreetCLIP (`geolocal/StreetCLIP`)](https://huggingface.co/geolocal/StreetCLIP)
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Panoramax STAC API](https://docs.panoramax.fr/backend/api/api/)
- [Plonk It GeoGuessr Guide](https://www.plonkit.net/beginners-guide)
