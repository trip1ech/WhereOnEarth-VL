# Scene Text & LM Attention Case Studies

We selected three representative failure/diagnostic images from the 300-image validation set (`qlora_eval_300`): a **Canadian dashcam with GPS coordinates**, a **Guadeloupe French trail sign**, and a **Brazilian Portuguese storefront**.

## 1. Method

- Evaluated both **base `Qwen3-VL-2B`** and **`Qwen3-VL-2B` + 6k localization LoRA** (4-bit NF4, greedy decoding, original localization prompt) using eager attention to inspect per-layer attention weights.
- For each image, we ran: (1) full-image localization, (2) full-image text transcription, (3) zoomed-crop transcription, (4) full-image localization with **human-verified transcript injected**, and (5) **text-region blur** vs. **equal-area nearby control-region blur** for both localization and transcription.
- Attention heatmaps average all heads across LM layers `0, 9, 18, 27` (0-indexed), projecting image-token attention onto the spatial grid when generating coordinate digits (localization) or letters/digits (transcription). `Image` reports total attention mass on image tokens; `Text` reports the share of image attention falling inside the text bounding box (compared against an equal-area control box).

**Attention Heatmaps:** [Canada](maps/canada.png) · [Guadeloupe](maps/guadeloupe.png) · [Brazil](maps/brazil.png)

Structured metrics, distances, and region attention ratios are stored in [`diagnosis.json`](diagnosis.json) (inputs/references in [`cases.json`](cases.json) and [`references.json`](references.json)). Re-render maps locally from cached attention outputs (`text-attention-results.zip`) via `python3 -B scripts/render_text_attention.py`.

## 2. Observations

| Case | Base Full-Image Error | 6k LoRA Full-Image Error | 6k LoRA Full-Image OCR | 6k LoRA + Human Transcript |
|---|---:|---:|---|---|
| **Canada** | 1,507.3 km | 9,690.0 km | Accurately read `W111.421436 N50.36211` | Emitted `+111.421` (wrong hemisphere!), error **8,102.1 km** |
| **Guadeloupe** | 6,586.6 km | 6,023.7 km | Read `POINTE DES CHATEAUX 6,2 km` (missing accent only) | Still guessed Europe, error **6,909.3 km** |
| **Brazil** | 495.8 km | 146.1 km | Read `Garagem`, `MAQUINAS de VIDRO`, `MACANETAS` | Remained Brazil, error **146.1 km** |

### Canada (Dashcam GPS Overlay)
The 6k fine-tuned model transcribed the exact coordinates from the full image, while base Qwen had decimal/last-digit errors on the full image that disappeared when zoomed. Blurring the text region eliminated OCR output for both models; blurring the equal-area control box preserved OCR.

Injecting the exact human transcript dropped base Qwen's coordinate error to **~32 meters** (though it still wrote country `US`). In contrast, the 6k fine-tuned model copied the latitude accurately, **flipped `W` to positive longitude (`+111.421`)**, and wrote country `TM` — exposing hemisphere-sign and country-consistency failures rather than visual illiteracy.

### Guadeloupe (French Trail Sign)
Both models read the place name directly from the full image. During transcription, text-box share of image attention jumped from `2.3% → 14.1%` (base) and `2.4% → 15.3%` (6k LoRA). Blurring the target sign caused models to read other sign fragments, while control blur preserved transcription. Crucially, **even with the exact human transcript provided, both models still guessed metropolitan Europe** — pointing to a world-knowledge / gazetteer lookup gap rather than a vision failure.

### Brazil (Portuguese Storefront)
Both models read the main Portuguese shop words from the full image, allocating roughly a quarter of image attention to the sign during transcription; blurring the sign yielded `UNREADABLE` while control blur preserved transcription. Both models still predicted Brazil even with the sign blurred (language narrows the region, while the sign contains no city/address).

## 3. Implications for ViT Fine-Tuning

**Keep the ViT frozen.** The 6k model already transcribed key scene text from full uncropped images yet still failed at geolocalization even when handed exact human transcripts. Future gains on text-bearing scenes should focus on deterministic coordinate parsing (for N/S/E/W strings) and external gazetteer lookup for named places rather than ViT unfreezing. Colab T4 runtime was terminated upon download (0.00 compute units/hr).
