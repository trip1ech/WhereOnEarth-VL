# 方向 1 實作報告：防洩漏街景檢索、Geo-Projection 重排與 StreetCLIP × VLM 混合定位

2026 年 11 月 23 日。本輪從頭到尾完成方向 1 的實作，並在建模前實際抽看 15 張街景配對與資料分布，先揭露並修正原本 25k 檢索的同拍攝者捷徑洩漏，再依序完成 **Step 1A（空間 KDE 投票）**、**Step 1B（防洩漏全圖庫 Geo-Metric Projection + Listwise Ranker）**、**Step 1C（Qwen3‑VL 6k 精確 211 國 log-prob 先驗融合）** 與 **Step 1D（換 `geolocal/StreetCLIP` 拉高天花板＋去 HUD 裁切對照）**。

---

## 一、建模前資料與看圖稽核（四大關鍵盲點）

在 [ranker-2018-pairs.zip](../ranker-2018-pairs.zip)（25k 圖庫／12k train／2k val）上檢查 `attribution`（拍攝者）與實際原圖配對：

1. **原本 25k `dinov2-small` 檢索的 `14.5% @25km`／`49.2% Top-100`，有 97.2% 是同拍攝者／同台車捷徑**
   - `val_2000` 中有 **92.8%（1,855/2,000）** 的拍攝者出現在 25k 圖庫裡；前 20 名高產帳號就佔整個圖庫 **28.1%**（`lvl5` 2,284 張、`roadroid` 888 張、`networklanman` 422 張、`bob3bob3` 298 張）。
   - 原本 290 個 Top‑1 ≤25 km 命中裡，**282 個（97.2%）是同拍攝者**；在 `qlora_eval_300` 上更是 **44/44（100%）** 全為同拍攝者。
   - **實圖抽查佐證**（本地 `data/qlora_since_2018/images/`）：
     - 印度 `@natarajakaveeti`（`1101634303727287.jpg` vs `1786588988190818.jpg`，cos=0.938，0.0 km）：同一台車、左下角同款玻璃三角貼紙與同兩台前車，僅在 Mapillary 被切成不同 `sequence`。
     - 澳洲 `@bob3bob3`（`2847551735459550.jpg` vs `810202126300348.jpg`，cos=0.927，7.2 km）：同一趟公路旅行隔 4 分鐘，且左下角均有同款行車記錄器黑條＋完整經緯度疊字。
     - 美國 `@networklanman`（`748528363325434.jpg` vs `907192507133508.jpg`，cos=0.831，相距 **647 km**）：相隔 647 km 仍衝上 Top‑1，主因是相同的 `NETWORKLANMAN` GPS 水印與低角度保險桿視角。
   - **嚴格遮除同拍攝者（diff-creator）後的真實 `dinov2-small` 基線**：
     - 25k 圖庫有效覆蓋率：≤25 km 從 `95.8% → 52.1%`，≤200 km 為 `96.3%`，同國 `99.2%`。
     - `val_2000` 真實 Top‑1：≤25 km **`0.45%`（9/2,000）**、≤200 km **`3.80%`（76/2,000）**、國家 **`21.4%`**、中位誤差 **`3,965.0 km`**、GeoGuessr 平均分 **`1,189.4`**。
     - `val_2000` 真實 Top‑100 召回上限：≤25 km 僅 **`7.80%`**（非 49.2%）、≤200 km `50.1%`、國家 `83.2%`。

2. **通用視覺模型被「構圖／光線雙胞胎」誤導**
   - 義大利隧道配日本隧道（`2986083211609576.jpg` vs `484810106163607.jpg`，cos=0.848，差 **9,566 km**）。
   - 印度模糊夜景配沙烏地模糊夜景（`198463045436194.jpg` vs `474829423581561.jpg`，cos=0.820，差 **3,755 km**）。
   - 美國沙漠黃虛線公路配澳洲紅土白虛線公路（`768656080501644.jpg` vs `553472652282790.jpg`，cos=0.833，差 **13,761 km**）。

3. **25 km 內「場景型態不同」的覆蓋極限**
   - 丹麥碎石荒郊路 `1441389372865694.jpg` 距 22.4 km 的同國真鄰居 `763751890877619.jpg` 是市區運河港口（cos = `-0.043`），反而是 7,349 km 外的安哥拉直路拿到 `cos = 0.669`。

4. **部分高產帳號底部含行車記錄器 GPS 水印**
   - 促使我們在 Step 1D 加入「裁切底部 12%（`crop88`）」的對照實驗與多視角融合。

> **防洩漏鐵則**：以下 Step 1A–1D 所有訓練與評估皆**強制遮除與 query 同 `attribution` 的圖庫影像**；Step 1B 訓練並明確剔除全部 `author_holdout_66` 拍攝者（排除 42 筆）且按拍攝者切分 `opt_train`／`tune`；Step 1C–1D 的融合權重僅以 `qlora_eval_300` 做 5-fold OOF 挑選，對 `author_holdout_66` 採 **zero-touch** 零調參評估。

---

## 二、Step 1A & 1B：25k `dinov2-small` 防洩漏空間 KDE ＋ 全圖庫 Geo-Projection ＋ Listwise Ranker

- **Step 1A（本機零 GPU，[eval_spatial_kde.py](../../scripts/eval_spatial_kde.py)）**：對 Top‑50 候選以空間核密度加權投票（`τ=0.10, σ=800 km`，於訓練子集調參）。
- **Step 1B（Colab T4，[train_geo_ranker.py](../../scripts/train_geo_ranker.py)）**：
  - **Stage 1（Geo-Metric Projection）**：在全部 25,000 圖庫上訓練共享殘差投影（384→512→384 + 輔助國家分類頭），以多尺度地理軟目標（35/180/600 km + 同國）做全圖庫對比訓練，**遮蔽同拍攝者 ≤300 km、同時保留同拍攝者 >300 km 當成硬負例**強迫模型消除相機／水印指紋。
  - **Stage 2（Listwise Geo-Ranker）**：於投影空間重撈 diff-creator Top‑100，輸入 22 維尺度正規化特徵（投影／原始 cosine、多尺度 KDE、圖庫密度去偏、Top‑100 國家共識、輔助國家 log-prob），以 Listwise Softmax KL 與受控殘差訓練，徹底解決舊版 `train_ranker.py` Epoch 1 崩潰問題（`best_stage1_epoch=15`, `best_stage2_epoch=2`）。

| 評估切分（皆嚴格排除同拍攝者） | 方法 | Top-100 ≤25km | Top-100 ≤200km | ≤25 km | ≤200 km | 國家正確 | 中位誤差 | GeoGuessr 分數 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`val_2000`** (n=2,000) | Raw `dinov2-small` Top‑1 | 7.80% | 50.0% | 9 (0.45%) | 76 (3.80%) | 429 (21.4%) | 3,965.0 km | 1,189.4 |
| | Step 1A：Spatial KDE | 7.80% | 50.0% | 8 (0.40%) | 92 (4.60%) | 488 (24.4%) | 2,736.8 km | 1,427.5 |
| | Step 1B Stage 1：GeoProj Top‑1 | **11.90%** | **64.1%** | 7 (0.35%) | 114 (5.70%) | 779 (39.0%) | 1,476.3 km | 1,889.3 |
| | Step 1B Stage 1：GeoProj + KDE | 11.90% | 64.1% | 7 (0.35%) | 117 (5.85%) | 848 (42.4%) | 1,088.9 km | 2,126.0 |
| | **Step 1B Stage 2：Listwise Ranker** | 11.90% | 64.1% | **16 (0.80%)** | **166 (8.30%)** | **998 (49.9%)** | **1,032.1 km** | **2,213.2** |
| **`val_unseen_opt`** (n=522, 訓練未見拍攝者) | Raw `dinov2-small` Top‑1 | 9.58% | 53.4% | 3 (0.57%) | 23 (4.41%) | 88 (16.9%) | 3,962.4 km | 1,229.7 |
| | **Step 1B Stage 2：Listwise Ranker** | **12.84%** | **57.3%** | **5 (0.96%)** | **34 (6.51%)** | **164 (31.4%)** | **1,384.9 km** | **1,909.6** |
| **`author_holdout_66`** (n=66, 雙重未見拍攝者) | Raw `dinov2-small` Top‑1 | 16.67% | 57.6% | 1 (1.52%) | 5 (7.58%) | 15 (22.7%) | 3,509.8 km | 1,287.4 |
| | **Step 1B Stage 2：Listwise Ranker** | **19.70%** | **59.1%** | **2 (3.03%)** | **5 (7.58%)** | **23 (34.8%)** | **1,404.4 km** | **1,988.9** |

---

## 三、Step 1C：Qwen3‑VL 6k 精確 211 國先驗 × 25k `dinov2-small` 混合重排

使用 [extract_vlm_country_priors.py](../../scripts/extract_vlm_country_priors.py) 以單次 KV‑cache 前向精確讀出 Qwen3‑VL 6k LoRA 在 `{"country":"` 後對全部 211 國的 log-prob（驗證與全前向誤差 `< 0.05`，Top‑1 國家正確率精確重現 `52.0%`／`50.0%`），再以 [evaluate_retrieval_v2_hybrid.py](../../scripts/evaluate_retrieval_v2_hybrid.py) 做 5-fold OOF 融合：

- **`qlora_eval_300`（5-fold OOF）**：
  - 單獨 Qwen3‑VL 6k：國家 `52.0%`、≤200km `14.7%`、中位 `831.6 km`、平均分 `2,556.2`
  - 25k Retrieval v2 + VLM 國家先驗：國家 **`58.3%`（175/300，+6.3%）**、中位 `885.6 km`、平均分 `2,521.7`
  - 25k Retrieval v2 + VLM 國家先驗 + 空間錨點：國家 `57.3%`、≤200km `14.7%`、中位 **`782.0 km`**、平均分 **`2,580.6`**（超越單獨 VLM 與單獨檢索）
- **`author_holdout_66`（Zero-touch）**：
  - 25k Retrieval v2 + VLM 國家先驗將純檢索從 `1,404 km` 拉進到 `721.7 km`（25 km 命中 2/66 > VLM 1/66、平均分 `2,644.5`），但受限於 `dinov2-small` 22M 的區域細粒度，仍略低於單獨 VLM（`600.3 km`, `2,788.6` 分）。

---

## 四、Step 1D：拉高天花板 —— `geolocal/StreetCLIP`（ViT-L/14）＋去 HUD 裁切 × Qwen3‑VL 混合

為了直接驗證「換地理預訓練骨幹能把天花板拉高多少」，我們在**完全相同的 6,000 張乾淨圖庫**上（比 25k 小 4 倍；與 `author_holdout_66` **0% 拍攝者重疊**），對照凍結 `dinov2-small` 與 `geolocal/StreetCLIP`（全圖 `full`、底部裁切 12% 去 HUD `crop88`、及雙視角平均 `multiview`），並結合空間 KDE、StreetCLIP 零樣本國家文字探針（`"A Street View photo in {country}"`）與 Qwen3‑VL 6k 國家 log-prob。所有融合權重均在 `qlora_eval_300` 做 5-fold CV 決定，對 `author_holdout_66` **零調參直接評估**（見 [finalize_step1d_analysis.py](../../scripts/finalize_step1d_analysis.py)）。

### 1. `author_holdout_66`（66 位全新未見拍攝者／60 國，**ZERO-TOUCH**，僅 6,000 張圖庫）

| 方法 | Top-100 ≤25km | Top-100 ≤200km | ≤25 km | ≤200 km | 國家正確 | 中位誤差 | GeoGuessr 分數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dinov2-small` raw Top‑1（6k） | 22.7% | 62.1% | 0/66 (0.0%) | 2/66 (3.0%) | 11/66 (16.7%) | 3,478.5 km | 1,098.1 |
| `dinov2-small` GeoProj + Listwise（25k） | 19.7% | 59.1% | 2/66 (3.0%) | 5/66 (7.6%) | 23/66 (34.8%) | 1,404.4 km | 1,988.9 |
| **原專案最佳：Qwen3‑VL‑2B 6k QLoRA** | — | — | 1/66 (1.5%) | 14/66 (21.2%) | 33/66 (50.0%) | 600.3 km | 2,788.6 |
| `StreetCLIP` (full) raw Top‑1（6k） | **39.4%** | **90.9%** | 4/66 (6.1%) | 14/66 (21.2%) | 32/66 (48.5%) | 618.3 km | 2,854.8 |
| `StreetCLIP` (full) + KDE + Text-Probe（6k） | **39.4%** | **90.9%** | **7/66 (10.6%)** | **18/66 (27.3%)** | 43/66 (65.2%) | 461.1 km | 3,286.3 |
| **`StreetCLIP` (full) + KDE + Text + Qwen3‑VL（6k）** | **39.4%** | **90.9%** | 6/66 (9.1%) | 16/66 (24.2%) | **46/66 (69.7%)** | 435.3 km | **3,395.4** |
| **`StreetCLIP` (full+crop88) + KDE + Text + Qwen3‑VL（6k）** | — | — | 5/66 (7.6%) | 15/66 (22.7%) | **46/66 (69.7%)** | **389.7 km** | 3,374.6 |

### 2. `qlora_eval_300`（300 張驗證集，嚴格 diff-creator 6,000 張圖庫，5-fold OOF）

| 方法 | Top-100 ≤25km | Top-100 ≤200km | ≤25 km | ≤200 km | 國家正確 | 中位誤差 | GeoGuessr 分數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dinov2-small` raw Top‑1（6k） | 8.3% | 42.7% | 0/300 (0.0%) | 11/300 (3.7%) | 41/300 (13.7%) | 5,972.8 km | 979.8 |
| `dinov2-small` GeoProj + Listwise（25k） | 10.7% | 65.0% | 3/300 (1.0%) | 27/300 (9.0%) | 143/300 (47.7%) | 1,077.9 km | 2,176.3 |
| **原專案最佳：Qwen3‑VL‑2B 6k QLoRA** | — | — | 8/300 (2.7%) | 44/300 (14.7%) | 156/300 (52.0%) | 831.6 km | 2,556.2 |
| `StreetCLIP` (full) + KDE + Text-Probe（6k, OOF） | **18.0%** | **72.0%** | **12/300 (4.0%)** | **60/300 (20.0%)** | 179/300 (59.7%) | **555.6 km** | 2,983.3 |
| **`StreetCLIP` (full) + KDE + Text + Qwen3‑VL（6k, OOF）** | **18.0%** | **72.0%** | 11/300 (3.7%) | 56/300 (18.7%) | 191/300 (63.7%) | 615.9 km | **2,996.6** |
| **`StreetCLIP` (full+crop88) + KDE + Text + Qwen3‑VL（6k, OOF）** | — | — | 11/300 (3.7%) | 53/300 (17.7%) | **193/300 (64.3%)** | 566.6 km | 2,988.5 |

### 3. 關鍵洞察：去 HUD 裁切、防捷徑與互補性
- **天花板直接暴漲**：同樣 6,000 張圖庫，把 `dinov2-small` 換成 `geolocal/StreetCLIP` 後，`author_holdout_66` 的 Top‑100 召回率直接從 `22.7% / 62.1%` 跳到 **`39.4% / 90.9%`**（`qlora_eval_300` 從 `8.3% / 42.7%` 跳到 **`18.0% / 72.0%`**）。
- **多視角去 HUD（`full + crop88`）有實效**：單獨 `crop88` 減少了 `qlora_eval_300` 上未遮蔽時的同拍攝者誤撞（`103 → 98/300`）並把 `author_holdout_66` 零樣本文字國家準確率從 `59.1% → 60.6%`；將 `full` 與 `crop88` 向量平均後，`author_holdout_66` 混合中位誤差下探到 **`389.7 km`**，`qlora_eval_300` 國家準確率達 **`64.3%`**。
- **StreetCLIP 仍有拍攝者捷徑傾向**：不遮蔽同拍攝者時，raw Top‑1 仍有 `34.3%`（103/300）撞到同拍攝者（`raw med=503 km` vs `diff med=945 km`），再次證明**未來所有檢索評估都必須維持嚴格 diff-creator 或官方 test 協議**。
- **與 VLM 的真實互補**：在 `author_holdout_66` 上，StreetCLIP 與 Qwen3‑VL 6k 兩者都答對國家 30 張、僅 StreetCLIP 對 15 張、僅 VLM 對 3 張（聯集上限 **72.7%**）；成對距離比較下混合模型有 **38/66** 張比單獨 Qwen3‑VL 更近、28/66 張更遠，零調參即比原 VLM 基線多出 **+606.8 GeoGuessr 分**、國家準確率 **+19.7%**、25 km 命中率 **6 倍**。

---

## 五、下一步如何繼續推高天花板

目前我們**只用 6,000 張圖庫、且完全未對 StreetCLIP 做 Geo-Projection 微調**就已經大幅刷新全專案紀錄。若要再往上推進：
1. **將 `StreetCLIP` 擴大到 25k（或 50k–100k）覆蓋圖庫**並套用 Step 1B 的**防洩漏 Geo-Metric Projection + Listwise Ranker**（預期 25 km 命中率與 200 km 命中率會因圖庫密度 4 倍＋投影去偏再跳一階）。
2. **鎖定 OSV‑5M 官方 `test`（2018 年後全球分層 300 張）**：作為全新未觸碰的最終基準，同時與 `osv5m/baseline` 做無洩漏正面比較。
3. **接上方向 2 的「自動文字裁切＋離線 GeoNames 地名查詢」**：把 `StreetCLIP + KDE + Qwen3‑VL` 產生的前幾名候選區域，與招牌文字查到的地名候選做交叉驗證。

---

## 六、檔案與重現

- Step 1A：`python3 -B scripts/eval_spatial_kde.py` → `step1a_kde_results.json`
- Step 1B + 1C（Colab T4，已完成並終止）：`scripts/prepare_retrieval_v2_bundle.py`、`transfer_1bc/run.sh` → `step1bc_results.zip`；本機融合分析：`python3 -B scripts/evaluate_retrieval_v2_hybrid.py` → `step1c_hybrid_results.json`
- Step 1D（Colab T4，已完成並終止）：`scripts/prepare_step1d_bundle.py`、`transfer_1d/run.sh` → `streetclip_features.npz`；本機分析：`python3 -B scripts/evaluate_step1d_streetclip.py`、`python3 -B scripts/finalize_step1d_analysis.py` → `step1d_streetclip_results.json`、`step1d_final_oof.json`
