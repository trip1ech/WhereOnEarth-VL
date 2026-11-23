# GeoGuessr Recognition 實驗總覽（Experiment Index）

本文件彙整 `GeoGuessr-Recognition` 自 2026-11-18 至 2026-11-23 的所有實驗階段、資料切分、量化結果、已知瓶頸與檔案索引，方便快速回顧或接續後續實驗。

---

## 一、實驗演進總表

| 階段 | 日期 | 主題與方法 | 評估集（來源／規模） | 核心結果 | 關鍵結論 | 詳細報告／輸出 |
|---|---|---|---|---|---|---|
| **1. Phase 1 Pilot** | 11-18 | `osv5m/baseline` vs. `Qwen3-VL-2B` zero-shot | `pilot_dev`（法國 3 城周邊，OSV 9 + Panoramax 9） | OSV-5M 官方：OSV 22.2% @25km（中位 158 km）、Panoramax 11.1%（中位 548 km）<br>Qwen zero-shot：OSV 0% @25km（中位 280 km）、Panoramax 0%（中位 464 km） | 建立防洩漏與跨平台對照流程；小樣本下專門定位模型明顯優於 2B zero-shot | [metrics.csv](results/phase1-pilot_dev-results/metrics.csv) · [README](README.md) |
| **2. 分布稽核 & 2k 檢索管線** | 11-18 | 掃描 OSV-5M 全量 metadata（4.89M train / 210k test）＋凍結 `dinov2-small` 2,000 張圖庫測試 | `pilot_dev`（n=18） | 完整 train 覆蓋 18/18；2k 圖庫僅覆蓋 2/18，Top-1 0/18 @25km（中位 801 km） | 驗證檢索管線與覆蓋率是瓶頸，推導出需建立 25k 地理覆蓋優先圖庫 | [summary.json](results/distribution_audit/summary.json) · [metrics.csv](results/frozen-retrieval-2000-pilot_dev-results/metrics.csv) |
| **3. 25k 檢索圖庫 & 殘差排序器** | 11-19 | 2018 年起覆蓋優先選取 25k 圖庫（覆蓋率 97.32% vs. 隨機 73.97%）；凍結 `dinov2-small` Cosine Top-100 ＋殘差 MLP pairwise ranker | `since_2018` val（OSV train 切出，n=2,000） | **Cosine Top-100**：983/2,000（49.2%）≤25 km<br>**Cosine Top-1**：**290/2,000（14.5%）≤25 km**，569/2,000 ≤200 km，中位 1,252 km<br>**MLP Ranker**：Epoch 1 在 tuning set 從 152→8 hits 崩潰，退回 `best_epoch=0` | 凍結檢索在 25 km 命中率遠高於 VLM，但 pairwise softplus ranker 因尺度與分布不匹配在 Epoch 1 崩潰（詳見第四節） | [ranker-2018-evaluation.zip](results/ranker-2018-evaluation.zip) · [train_ranker.py](scripts/train_ranker.py) |
| **4. Qwen3-VL-2B QLoRA（2k / 6k）** | 11-20 | 4-bit NF4 + LM attention LoRA（r=8, α=16, 1 epoch, 512 px），輸出 `{"country","latitude","longitude"}` | `since_2018` val 子集（n=300） | **Zero-shot**：294 有效，國對 132，≤25km 4，≤200km 32，中位 1,141 km<br>**2k SFT**：300 有效，國對 154，≤25km 9，≤200km 44，中位 861 km<br>**6k SFT**：300 有效，國對 **156**，≤25km 8，≤200km 44，中位 **832 km** | 0→2k 有顯著提升（格式＋地理先驗），但 2k→6k 出現強烈邊際遞減（3 倍資料只多對 2 國） | [qlora/metrics.csv](results/qlora/metrics.csv) · [qlora_6000/metrics.csv](results/qlora_6000/metrics.csv) · [sft_colab.py](scripts/sft_colab.py) |
| **5. 未見拍攝者評估（Author Holdout）** | 11-20 | 人工逐張審核 72 張（剔除 HUD 座標洩漏、非道路、失焦共 6 張），核對 `creator_id` 零重疊 | `author_holdout`（66 張／66 位新拍攝者／60 國） | **Zero-shot**：65 有效，國對 28/66，≤200km 8/66，≤25km 1/66，中位 945 km<br>**6k SFT**：66 有效，國對 **33/66**，≤200km **14/66**，≤25km 1/66，中位 **589 km**（共同 65 張）／600.3 km（全 66 張） | 確認 6k QLoRA 的收益可泛化到未見拍攝者（42 張變近、23 張變遠），確立目前 VLM 定位基線 | [author_holdout/README.md](data/author_holdout/README.md) · [metrics.csv](results/author_holdout/metrics.csv) |
| **6. 中央標線 Probe & 裁切診斷** | 11-21 | 人工盲標 70 張（23 單線／20 雙線），比較凍結 ViT、Projector、DeepStack 與 Qwen QA 在整圖／固定裁切／人工裁切的辨識力 | 43 張拍攝者 5 折＋22 張同國配對（8 國 11 對） | 22 張同國配對整圖→人工裁切：<br>Projector+DeepStack `54.5% → 81.8%`（12→18/22）<br>原版 Qwen QA `63.6% → 81.8%`（14→18/22）<br>6k LoRA QA `72.7% → 86.4%`（16→19/22）；同國排序 11/11 | 凍結視覺特徵在放大關鍵區域後已具備良好辨識力，暫無足夠證據解凍 ViT；問題更偏向解析度／視野與 LM 讀取方式（另發現顯著 A/B 順序偏差） | [roadmark_probe/README.md](results/roadmark_probe/README.md) · [roadmark_crop_probe/README.md](results/roadmark_crop_probe/README.md) |
| **7. 街景文字與 Attention 個案** | 11-21 | 3 張代表性個案（加拿大 HUD 座標、瓜地洛普路牌、巴西招牌），做原圖／放大抄字、人工抄錄注入與文字區／控制區模糊干預 | 300 張評估集挑出 3 張個案 | 6k 微調版原圖即能抄對關鍵文字，但注入正確人工抄錄後仍可能定位失誤（加拿大 `W111.42` 誤輸出 `+111.421`；瓜地洛普 `POINTE DES CHATEAUX` 仍猜歐洲） | 再次支持暫時維持 ViT 凍結；瓶頸包含正負號／國家一致性與 2B 模型本身的地名世界知識不足 | [text_attention/README.md](results/text_attention/README.md) · [diagnosis.json](results/text_attention/diagnosis.json) |
| **8. 先讀線索再定位（推論診斷）** | 11-22 | 用 6k SFT 模型先抽文字再回饋給同一模型定位，另在 5 張清楚招牌測人工抄錄與裁切放大 | `author_holdout`（n=66；5 張已核對招牌子集） | 直接定位 `33/66`（中位 600.3 km）→ 加模型抄錄 `30/66`（中位 615.7 km）；21/66 格式不符，且無字場景會捏造地名（如衣索比亞捏造 `Saray`） | 直接兩階段 prompting 會因 OCR 幻覺與格式錯誤拉低定位；放大可把 `BOUAKA` 修成 `BOUAKÉ`，但 2B 模型仍誤猜塞內加爾 | [clue_first/README.md](results/clue_first/README.md) · [analysis.json](results/clue_first/analysis.json) |
| **9. 現場線索 LM LoRA 續訓** | 11-22 | 在 6k LoRA 上續訓 69 steps（104 張圖、258 題、547 次呈現，68% 為抽線索題、32% 為定位題） | `author_holdout`（n=66；並人工核對 29 張道路＋21 張無字＋5 張清楚招牌） | **抽線索大幅改善**：路側物件完全相符 `7/29 → 27/29`，無字正確留空 `11/21 → 21/21`，有效 JSON `66/66`<br>**定位反而退步**：直接猜 `33→30/66`（≤200km `14→10`）；加線索 `29/66`（中位 `623.2 km`）；OCR 過度保守（59/66 留空，anchor 仍僅 1/5） | 續訓版適合作為獨立的線索抽取器，但不適合直接覆蓋定位基線；單純把粗粒度道路 JSON 塞入 prompt 無法教會排除推理，且任務稀釋導致定位退步 | [clue_first_training/README.md](results/clue_first_training/README.md) · [geo_metrics.csv](results/clue_first_training/geo_metrics.csv) |
| **10. 方向 1：防洩漏檢索 v2 ＋ StreetCLIP 6k 概念驗證** | 11-23 | 抽查實圖揭露 25k 檢索 97.2% 同拍攝者捷徑後，全面採嚴格 diff-creator 協議：Step 1A 空間 KDE → Step 1B 25k `dinov2-small` GeoProj+Listwise → Step 1C Qwen3-VL 211 國精確先驗 → Step 1D 6k `StreetCLIP` × VLM 混合 | `val_2000`、`val_unseen_opt` (522)、`qlora_eval_300` (OOF)、`author_holdout_66` (ZERO-TOUCH) | **25k `dinov2-small` (diff-creator)**：`val_2000` 國對 `21.4% → 49.9%`、中位 `3,965 → 1,032 km`<br>**6k `StreetCLIP` + KDE + Text + Qwen3-VL**：`qlora_eval_300` 國對 `64.3%`／`2,997` 分；`author_holdout_66` 國對 `69.7%`／中位 `389.7 km`／`3,395` 分 | 證實地理預訓練骨幹 `StreetCLIP` 直接倍增召回天花板，且與 VLM 國家先驗互補 | [retrieval_v2/README.md](results/retrieval_v2/README.md) · [step1d_final_oof.json](results/retrieval_v2/step1d_final_oof.json) |
| **11. 25k `StreetCLIP` Geo-Ranker ＋ OSV‑5M 官方 `test`（300 張／147 國）正面對決** | 11-23 | 零 429 串流編碼全量 39k+300 張 `StreetCLIP`、訓練防洩漏 25k Geo-Projection & Listwise Ranker；鎖定全新 OSV‑5M 官方 test `official_test_300`（300 未見拍攝者、147 國、OCR 剔除 Garmin GPS HUD），與 500 萬張 `osv5m/baseline` 正面對決 | `official_test_300` (ZERO-TOUCH)、`val_2000`、`val_unseen_opt_522`、`qlora_eval_300`、`author_holdout_66` | **`official_test_300` 正面對決（無洩漏）**：<br>· **`StreetCLIP 25k GeoProj+KDE+Text`**：國對 **`48.0%`**（144/300）、中位 **`570.2 km`**、分 **`2,907.5`**（對決 `osv5m/baseline` **166 勝 131 負**）<br>· `osv5m/baseline`（5M 官方）：國對 `41.3%`、≤25km **`11.0%`**、中位 `804.4 km`、分 `2,590.1`<br>· `Qwen3-VL 6k`：國對 `29.0%`、中位 `1,306.7 km`、分 `2,127.4`<br>**其他切分新紀錄**：`val_2000` 國對 **`71.9%`**／`3,232` 分；`qlora_eval_300` (+VLM) 國對 **`74.3%`**／`3,327` 分；`author_holdout_66` 中位 **`308.7–318.2 km`**／**`3,465.3` 分** | **僅用 25k 圖庫即在官方 test 的國家準確率（+6.7%）、中位距離（-234 km）與 GeoGuessr 分數（+317 分）全面擊敗 500 萬張 `osv5m/baseline`**；`osv5m/baseline` 仍在 ≤25 km（11.0% vs 3.7%）保優。同時發現長尾 147 國下固定權重混入弱 VLM 會拖累強檢索，後續宜改為信心／招牌文字門控 | [official_test_showdown/README.md](results/official_test_showdown/README.md) · [showdown_summary.json](results/official_test_showdown/showdown_summary.json) |

---

## 二、目前各能力最佳基線

| 能力項目 | 目前最佳模型／版本 | 關鍵指標 | 權重／結果位置 |
|---|---|---|---|
| **全球街景定位（無洩漏官方 test 新紀錄）** | `geolocal/StreetCLIP`（25k 圖庫）+ Stage 1 Geo-Metric Projection + KDE + Text-Probe | `official_test_300`（147 國、ZERO-TOUCH）：國對 **`48.0%`**（144/300）、≤200km `24.3%`、中位 **`570.2 km`**、平均 **`2,907.5` 分**（擊敗 5M `osv5m/baseline` 的 `2,590.1` 分／`804.4 km`）<br>`author_holdout_66`（ZERO-TOUCH）：≤200km **`33.3%`**、中位 **`318.2 km`**、平均 **`3,465.3` 分** | [streetclip_25k_results.zip](results/streetclip_25k/streetclip_25k_results.zip) · [streetclip_39k_plus_test300.npz](results/streetclip_25k/streetclip_39k_plus_test300.npz) |
| **主要國家檢索 × VLM 混合（OSV‑5M val 分布）** | `StreetCLIP` 25k Stage 2 Listwise + Qwen3‑VL 6k 國家先驗 | `val_2000`（diff-creator，純檢索）：國對 **`71.9%`**、中位 `474.9 km`、平均 `3,232.0` 分<br>`qlora_eval_300`（5-fold OOF，+VLM）：國對 **`74.3%`（223/300）**、中位 `461.2 km`、平均 **`3,327.0` 分** | [official_test_showdown/README.md](results/official_test_showdown/README.md) |
| **5M 官方精準定位基準（≤25 km 標竿）** | `osv5m/baseline`（489 萬張訓練） | `official_test_300`：≤25km **`11.0%`（33/300）**、≤200km **`27.3%`**、國對 `41.3%`、中位 `804.4 km`、平均 `2,590.1` 分 | [phase_c_results.zip](results/official_test_showdown/phase_c_results.zip) |
| **純 VLM 單張定位基線** | Qwen3-VL-2B + 6k QLoRA | `official_test_300`：國對 `29.0%`、中位 `1,306.7 km`、平均 `2,127.4` 分<br>`author_holdout_66`：國對 `50.0%`、中位 `600.3 km`、平均 `2,788.6` 分 | [results/qlora_6000/geoguessr-sft-6000-results.zip](results/qlora_6000/geoguessr-sft-6000-results.zip) |
| **結構化道路／無字拒答抽取** | 6k LoRA + Clue-First 續訓版 | 路側物件完全相符 `27/29`（micro-F1 `0.938`）、鋪面 `28/29`、中央線 `21/25`、無字留空 `21/21` | [results/clue_first_training/clue-first-lm-final.zip](results/clue_first_training/clue-first-lm-final.zip) |

---

## 三、資料切分與評估紀律地圖

| 切分名稱 | 規模 | 原始來源 | 互斥保證 | 目前角色／注意事項 |
|---|---:|---|---|---|
| `pilot_dev` / `pilot_holdout` | 各 18 張（OSV 9 + Panoramax 9） | OSV-5M 官方 test（法國 3 城）+ Panoramax IGN | 拍攝序列互斥，切分間距 ≥10.43 km | 法國跨平台流程驗證；`pilot_holdout` 尚未大幅使用 |
| `since_2018` ranker splits | 圖庫 25k / train 12k / val 2k / cov 6k | OSV-5M **官方 train**（≥2018-01-01） | ID、拍攝序列互斥 | 檢索與排序器訓練／驗證；特徵與 Top-100 已快取於 `results/ranker-2018-pairs.zip` |
| `qlora_since_2018` train / eval | train 2k & 6k / eval 300 | 上述 `since_2018` train 與 val 子集 | ID、序列、原始 SHA-256 互斥 | QLoRA 訓練與主開發評估；**源自官方 train，不可與 `osv5m/baseline` 公平對比** |
| `author_holdout` | 66 張（66 位拍攝者、60 國） | `since_2018` val 候選（人工審核剔除 6 張） | 與 6k train、300 eval 之 `creator_id` 零重疊 | 已於階段 8–9 用於錯誤分析與選型，**後續應視為 dev／診斷集** |
| **`official_test_300`** | **300 張（300 位未見拍攝者、147 國）** | **OSV-5M 官方 `test`**（≥2018-01-01，OCR 剔除 1 張 Garmin GPS HUD） | 與 25k 圖庫、12k train、2k val、6k QLoRA 及 `osv5m/baseline` 訓練集**全面零重疊** | **全新官方無洩漏最終基準集**（ZERO-TOUCH），可直接與 `osv5m/baseline` 公平正面對決 |
| `roadmark_annotations` | 70 張（23 單 / 20 雙 / 16 無 / 11 不確定） | 6k QLoRA 訓練集子集（人工盲標） | 43 張二分類為 43 位不同拍攝者 | 僅供中央線特徵與裁切診斷，不可當作 6k LoRA 泛化測試 |

---

## 四、已知關鍵瓶頸（後續實驗設計依據）

1. **Ranker 訓練崩潰（實作可修復）**：[train_ranker.py](scripts/train_ranker.py) 用 pairwise `softplus(neg - pos)` 直接加在範圍極窄的 cosine 上，且只在正負配對上訓練而未對全部 Top-100 做 listwise softmax／InfoNCE，導致 `delta` 在 Epoch 1 淹沒 cosine 訊號；同時 [retrieval.py](scripts/retrieval.py) 尚未加入空間密度投票（KDE）。
2. **VLM 座標 SFT 邊際遞減**：[sft_colab.py](scripts/sft_colab.py) 對三位小數經緯度算無權重 token cross-entropy，缺乏地理距離幾何（正負號反轉僅扣 1 token）；且 512 px 縮圖抹掉路牌與標線細節。
3. **2B 模型的地名知識上限**：個案證明即使給人工正確抄錄（`POINTE DES CHATEAUX`、`BOUAKÉ`），2B 模型仍無法對應到正確地點，需要外部地名查詢（Geocoder / Gazetteer）而非硬塞參數記憶。
4. **單一 adapter 兼顧抽線索與定位會互相干擾**：續訓 69 步即拉低直接定位準確率，且 69 筆留空訓練讓 OCR 過度保守；後續應拆分「線索抽取 adapter」與「定位 adapter」。

---

## 五、專案目錄與工具索引

- **本地互動工具**
  - [webapp/](webapp/README.md)：離線 Leaflet 地圖比較器，可對照 300 張評估集上 zero-shot / 2k / 6k 的預測與真值。
  - [annotation_tool/](annotation_tool/README.md)：本機盲測道路中央線標註工具（寫入 `data/roadmark_annotations/labels.json`）。
- **核心腳本（`scripts/`）**
  - Pilot & 分布稽核：`prepare_pilot.py`, `phase1.py`, `build_bundles.py`, `audit_distribution.py`
  - 檢索 & 排序器：`prepare_gallery.py`, `retrieval.py`, `pack_retrieval.py`, `cache_train_metadata.py`, `select_ranker_data.py`, `materialize_ranker_data.py`, `train_ranker.py`
  - 防洩漏檢索 v2 & StreetCLIP 混合（方向 1）：`eval_spatial_kde.py`, `train_geo_ranker.py`, `extract_vlm_country_priors.py`, `prepare_retrieval_v2_bundle.py`, `evaluate_retrieval_v2_hybrid.py`, `benchmark_streetclip.py`, `prepare_step1d_bundle.py`, `evaluate_step1d_streetclip.py`, `finalize_step1d_analysis.py`
  - 25k StreetCLIP & 官方 test 對決：`prepare_official_test.py`, `encode_and_train_streetclip_25k.py`, `prepare_phase_b_bundle.py`, `evaluate_official_test_models.py`, `prepare_phase_c_bundle.py`, `evaluate_official_test_showdown.py`
  - QLoRA & 未見拍攝者：`prepare_sft_data.py`, `sft_colab.py`, `prepare_author_holdout.py`, `evaluate_author_holdout.py`
  - 視覺／文字診斷：`prepare_roadmark_probe.py`, `extract_roadmark_features.py`, `analyze_roadmark_probe.py`, `prepare_roadmark_crop_probe.py`, `compare_roadmark_crop_probe.py`, `probe_text_attention.py`, `render_text_attention.py`
  - 線索優先 & 續訓：`prepare_clue_first.py`, `evaluate_clue_first.py`, `analyze_clue_first.py`, `clue_visual.py`, `prepare_clue_training.py`, `train_clue_first.py`, `evaluate_visual_clues.py`, `evaluate_clue_format_control.py`, `analyze_clue_training.py`
- **Colab 遠端自動化**：後期實驗（`roadmark_probe`, `roadmark_crop_probe`, `text_attention`, `clue_first`, `clue_first_training`）均透過本機 `colab` CLI（`~/.local/bin/colab`）搭配各實驗目錄下的 `transfer/run.sh` 自動開機、切分上傳、執行、下載結果並於 `EXIT` 時終止 GPU session。
- **根目錄大型打包檔（~104 MB，皆為各階段 Colab 上傳 bundle，可依需求保留或移至封存資料夾）**：`geoguessr_sft_upload.zip`（92 MB）、`geoguessr-pilot_dev.zip`（5.0 MB）、`geoguessr-pilot_holdout.zip`（3.7 MB）、`ranker_training_upload.zip`（2.8 MB）、`distribution_audit_upload.zip`（473 KB）、`ranker_preprocessing_upload.zip`（25 KB）、`retrieval_2000_upload/`。
