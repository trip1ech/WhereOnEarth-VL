# GeoGuessr Field Clues & Continued LM LoRA Training

This experiment tested two questions: (1) Can we teach the model to report observable visual and textual street clues accurately without hallucinating? (2) Does feeding those extracted clues back into the same model improve geolocalization? GeoGuessr players combine place names, scripts, and road infrastructure ([Plonk It Beginner's Guide](https://www.plonkit.net/beginners-guide)), so we tested both scene-text transcription and structured road observation.

Here, the **base localization model** is `Qwen3-VL-2B` already fine-tuned on 6,000 street-view images (`6k QLoRA`), and the **continued LM LoRA** resumes training from that adapter. In the preceding [pre-training diagnosis](../clue_first/README.md) on 66 unseen-creator photos, direct guessing got 33/66 countries right, whereas two-stage "read text first, then geolocate" dropped to 30/66 — largely because the model hallucinated place names on signless photos. This motivated training explicit **faithful transcription** and **empty-text abstention**.

---

## 1. Method

### 1.1 Curating 104 Multi-Task Training Images

Starting from the 6,000-image training pool, we sampled across countries (at most 1 image per capture sequence) to assemble 3,140 candidates. Local OCR surfaced text-bearing candidates, which were manually inspected alongside zoomed crops to lock exact target transcripts. Separately, a subset of candidates was manually annotated for visible road attributes.

A single photo can generate multiple tasks (for example, an unpaved road serves as an empty-text abstention example, a road-attribute description example, and a geolocalization example), yielding **104 distinct photos and 258 distinct training questions**.

### 1.2 Task Formats & Ground-Truth Targets

**Market sign (OCR transcription):** Given the full image and asked to transcribe verifiable place names, route numbers, or signage, the human target is `{"text":["MERCADO DE PEIXE"]}` ("fish market" in Portuguese — an actionable language clue). Output stops at transcription.

![Market building wall reading MERCADO DE PEIXE](method_examples/market-sign.jpg)

*Photo source: OSV-5M, creator `zeylmans`, CC BY-SA 4.0.*

**Unpaved road in Ghana (4 tasks from the same full image):**

![Unpaved red-dirt road with no center line or signage](method_examples/unpaved-road.jpg)

*Photo source: OSV-5M, creator `mawutor`, CC BY-SA 4.0.*

| Prompt Task | Training Target |
|---|---|
| Transcribe readable scene text | `{"text":[]}` (no reliable readable text) |
| Describe road surface, center line, and listed roadside objects | `{"surface":"unpaved","centerline":"none","roadside":[]}` |
| Direct geolocalization | `{"country":"GH","latitude":8.838,"longitude":0.061}` |
| Clue-conditioned geolocalization (given surface/centerline/roadside) | Same country & GPS coordinates (road clues provided in prompt) |

For road tasks, `none` means the road center is clearly visible with no line, while `unclear` covers darkness, occlusion, or ambiguity between center and edge lines. `roadside` tracks only visible guardrails, utility poles, bollards, and traffic signs. Text tasks exclude illegible words and dashcam date/time overlays, encouraging empty arrays `[]` when no trustworthy scene text exists.

### 1.3 LoRA Training Schedule

Each training example pairs the full image and one prompt using standard answer-token cross-entropy SFT. ViT, projector, and base LM weights stay frozen; only the existing LM attention LoRA weights (3,211,264 parameters) are updated.

| Task Category | Images | Presentations per Task |
|---|---:|---:|
| Readable scene text (verbatim transcription) | 31 | 3 |
| No reliable text (empty array abstention) | 69 | 2 |
| Visible road description | 70 | 2 |
| Direct geolocalization (preserving base task) | 54 | 2 |
| Geolocalization conditioned on human-verified road clues | 34 | 2 |

Shuffled together, this totaled **547 presentations** (371 clue-extraction, 176 localization). With gradient accumulation of 8 (plus a final 3-sample step), training ran for **69 optimizer steps** at learning rate `6e-5`; average training loss fell from `0.693` (first 10 steps) to `0.382` (last 10 steps). Note that clue-conditioned training used **human-verified** road descriptions, while evaluation tested **model-extracted** text and road descriptions.

### 1.4 Evaluation Protocol

The 66 unseen-creator holdout images share zero image IDs or hashes with training data (note: this split was previously inspected during Stage 8 diagnosis, making this a controlled before/after comparison on the same holdout). In two-stage evaluation, the **full image always remains present** alongside the extracted text and/or road summary. When text output is empty or unparseable, the text-conditioned run falls back to direct prediction (only 7/66 photos yielded non-empty text from the continued model). To ensure fair comparison, Markdown code-fence wrappers were stripped from the base 6k model's JSON outputs before scoring.

We also manually audited three subsets within the 66 holdout images: **5 clear-sign images** (anchor-word recall), **21 textless images** (empty-array abstention), and **29 road images** (surface and roadside objects; 25/29 with unambiguous center-line ground truth).

Use [`scripts/analyze_clue_training.py`](../../scripts/analyze_clue_training.py) to recompute [`analysis.json`](analysis.json) and [`geo_metrics.csv`](geo_metrics.csv).

---

## 2. Results

### 2.1 Clue Extraction Quality

| Human-Audited Criterion | Base 6k LoRA | Continued LM LoRA |
|---|---:|---:|
| Road surface (paved vs. unpaved) | 28 / 29 | 28 / 29 |
| Recognizable center-line color / absence | 20 / 25 | 21 / 25 |
| Roadside-object set exact match | 7 / 29 | **27 / 29** |
| Empty array when no reliable text exists | 11 / 21 | **21 / 21** |
| Clear-sign anchor-word hit | 1 / 5 | 1 / 5 |

Stripping Markdown code fences yielded 63/66 valid road JSONs for the base 6k model vs. **66/66** for the continued model. The dramatic gain in roadside exact match (`7/29 → 27/29`, micro-F1 `0.938`) came mainly from eliminating hallucinated objects (false-positive utility poles fell from 13 to 1).

For scene text, valid JSON improved from 45/66 (54/66 after stripping code fences) to **66/66**, with 59 empty arrays and 7 non-empty transcripts. While `BOUAKA` improved to `BOUAKÉ`, the Chinese storefront dropped from transcribing `"汽车钥匙"` to only a brand name, and the Korean sign still read only a street-number fragment (`6-4`). Overall, training eliminated hallucination and formatting errors, but OCR recall became overly conservative.

### 2.2 End-to-End Geolocalization Impact

| 66 Unseen-Creator Holdout Images | Country Acc. | ≤ 200 km | ≤ 25 km | Valid Coords | Median Error (Valid) |
|---|---:|---:|---:|---:|---:|
| Base 6k LoRA — direct guess | **33 / 66** | **14 / 66** | 1 / 66 | 66 | 600.3 km |
| Base 6k LoRA + format-cleaned text | 29 / 66 | 8 / 66 | 1 / 66 | 66 | 665.4 km |
| Base 6k LoRA + format-cleaned road clues | 31 / 66 | 12 / 66 | 1 / 66 | 66 | 658.4 km |
| Continued LM LoRA — direct guess | 30 / 66 | 10 / 66 | **2 / 66** | 65 | **591.9 km** |
| Continued LM LoRA + text | 29 / 66 | 9 / 66 | **2 / 66** | 65 | **591.9 km** |
| Continued LM LoRA + road clues | 29 / 66 | 10 / 66 | 1 / 66 | 64 | 623.2 km |
| Continued LM LoRA + road clues & text | 29 / 66 | 10 / 66 | 1 / 66 | 64 | 623.2 km |

On the 65 shared valid images, direct median error shifted slightly (`589.0 → 591.9 km`). Adding both clue types moved median error on the 64 shared valid images from `586.9 → 623.2 km` (25 closer, 32 farther, 7 within 1 km). Both models ran FP16 with 4-bit NF4 quantization (base direct on T4, continued model on L4).

---

## 3. Verdict

The continued LM LoRA substantially reduced hallucination and serves as a strong **standalone clue extractor**, but **we keep the original 6k LoRA as the pure-VLM localization baseline**. Image #33 (a Finnish forest road) illustrates the gap clearly: the continued model accurately reported paved surface, yellow center line, and no roadside objects, yet still guessed Russia. Coarse road JSON alone does not teach exclusionary reasoning across multiple independent clues, and mixing clue extraction with localization in a single 2B adapter diluted localization accuracy.

Links: [Pre-training diagnosis](../clue_first/README.md) · [Metrics CSV](geo_metrics.csv) · [Analysis JSON](analysis.json) (final adapter ZIP archived locally). Colab runtime was terminated immediately after downloading results.
