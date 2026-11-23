# Clue-First Prompting Diagnosis (Pre-Training)

## 1. Motivation & Visual Audit

In GeoGuessr ([Plonk It Beginner's Guide](https://www.plonkit.net/beginners-guide)), players separate language/script, directional place names, route numbers, road infrastructure, and driving side into distinct clues — noting that directional sign destinations can be far from the camera while script/language narrows the country set. Here we restricted our diagnostic to **visible scene text** on the 66-image unseen-creator holdout (`author_holdout`).

We manually inspected all 66 thumbnails and zoomed into candidate signage before inference, locking human transcripts on **5 clear-sign images** as a diagnostic subset:
- **#22:** directional sign reading `Βόλος` / `Volos` (Greece)
- **#45:** roadside sign reading `BOUAKE` (Côte d'Ivoire)
- **#24, #31, #36:** Italian (`residenti`, `eccetto`), Chinese (`汽车钥匙`), and Korean signage testing script/language cues

Remaining photos included unreadable distant signs, windshield stickers, and signless rural/residential roads. All 66 had already passed the GPS-HUD audit.

---

## 2. Method

All three inference conditions reused the fixed `Qwen3-VL-2B` (4-bit NF4) + `6k QLoRA` localization adapter (ViT, projector, and LM weights unchanged):
1. **Direct localization:** Original localization prompt on all 66 full images.
2. **Model OCR → localize:** Prompt the same model to extract up to three readable text snippets from the full image, then pass both the full image and extracted text to the localization prompt (falling back to direct prediction when OCR is empty or invalid JSON).
3. **Human transcript / zoomed OCR (5-image subset):** Feed human-verified transcripts alongside the full image, and separately run the OCR prompt on manually zoomed sign crops before feeding those crop transcripts back into localization.

Recompute [`analysis.json`](analysis.json) and [`predictions.csv`](predictions.csv) with `python3 -B scripts/analyze_clue_first.py` (image hashes locked in [`inputs.json`](inputs.json); raw outputs archived locally in `clue-first-results.zip`).

---

## 3. Results

| Method (66 Unseen-Creator Holdout Images) | Country Acc. | ≤ 200 km | ≤ 25 km | Median Error |
|---|---:|---:|---:|---:|
| Direct localization | **33 / 66** | **14 / 66** | 1 / 66 | **600.3 km** |
| With model-transcribed scene text | 30 / 66 | 11 / 66 | 1 / 66 | 615.7 km |

- Direct localization reproduced previous per-image predictions identically.
- Model OCR produced **24** parseable non-empty outputs, **21** empty arrays, and **21** invalid formats. Across the 24 non-empty cases (> 1 km change threshold), **8** improved, **13** worsened, and **3** were unchanged.
- **Hallucination drove regressions:** On a signless Ethiopian mountain road, the model invented `"Saray"` and flipped a correct Ethiopia guess to Turkmenistan. On image #47 (Denmark), it fabricated `"Hauptstraße"` and flipped Denmark to Germany.
- **5-image human-transcript subset:** Country hits remained `4/5`, while median error improved from `455 km → 370 km` (2 better, 2 worse, 1 unchanged):
  - *Volos (Greece):* directional sign destination pushed error from `47 km → 250 km`.
  - *BOUAKE (Côte d'Ivoire):* full-image OCR misread `BOUAKA` while zoomed crop fixed it to `BOUAKÉ`, yet even with exact text the 2B model still guessed Senegal.
  - *Chinese storefront (`汽车钥匙`):* already transcribed from the full image; providing that single verified cue cut error from `455 km → 59 km`.

These findings directly motivated the continued LM LoRA training for faithful transcription and empty-text abstention in [`results/clue_first_training/README.md`](../clue_first_training/README.md). Colab T4 runtime was terminated upon completion.
