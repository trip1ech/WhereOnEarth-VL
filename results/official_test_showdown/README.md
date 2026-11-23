# 25k StreetCLIP Geo-Ranker 與 OSV-5M 官方測試集（`official_test_300`）正面對決報告

2026 年 11 月 23 日。本輪完成兩項關鍵推進：
1. **擴充 `geolocal/StreetCLIP` 至全量 39,000 張（25,000 圖庫＋12,000 train＋2,000 val）＋訓練防洩漏 25k Geo-Metric Projection 與 Listwise Geo-Ranker**。
2. **建置並鎖定全新無洩漏的 OSV‑5M 官方 `test` 基準集 `official_test_300`（300 位全新未見拍攝者、147 國）**，首次在同一張官方無洩漏考卷上，讓我們的系統與 500 萬張資料訓練的官方 **`osv5m/baseline`**、`Qwen3‑VL‑2B`（zero-shot 與 6k QLoRA）及 `dinov2-small` 正面對決。

---

## 一、`official_test_300` 官方基準集建置與防洩漏保證

建置腳本：[scripts/prepare_official_test.py](../../scripts/prepare_official_test.py)，鎖定資料包：[data/official_test_300/official-test-300.zip](../../data/official_test_300/official-test-300.zip)（14.99 MB）。

- **來源**：OSV‑5M 官方 `test`（`data/raw/osv5m-test.csv`，拍攝時間 `>= 2018-01-01 UTC`），與 `osv5m/baseline` 的官方訓練集嚴格互斥。
- **零拍攝者洩漏**：比對 `gallery_25000`、`train_12000`、`validation_2000` 與 `qlora_6000`，確保 300 張照片來自 **300 位完全未見拍攝者（1 人 1 張）**，ID、序列、`creator_username`、`creator_id` 皆零重疊。
- **全球平衡分層**：從 1,175 位未見拍攝者（148 國）中以平方根國家分層＋4° 地理網格輪詢抽取 340 張候選，最終鎖定 **300 張、橫跨 147 國**（單一國家最多僅 10 張，無大國壟斷）。
- **HUD 疊字與品質稽核**：以 macOS 原生 Apple Vision OCR 掃描全部 340 張候選，成功抓出並剔除 1 張日本 Garmin 行車記錄器 GPS 疊字照片（`798364104121004.jpg`，底部印有 `38.41452 141.23285 20 KM/H`）；另抽看實圖確認路況涵蓋一般街景與真實世界雜訊（例如倒裝行車記錄器畫面），所有模型面對完全相同像素輸入。
- **25k 圖庫對 `official_test_300` 的天然地理覆蓋率**：≤25 km 覆蓋 `79.3%`、≤200 km 覆蓋 `95.7%`、同國覆蓋 `99.0%`。

---

## 二、全量 39k `StreetCLIP` 串流編碼與 25k 防洩漏訓練（Phase B）

腳本：[scripts/encode_and_train_streetclip_25k.py](../../scripts/encode_and_train_streetclip_25k.py)（Colab NVIDIA L4，已完成並終止）。

- **零 429 限流高速串流編碼**：復用 Step 1D 已編碼的 6,366 張向量；剩餘 32,634 張改採「雙執行緒預載、逐 shard 整包下載 1 次 HTTP 請求 → 解出約 340 張指定圖並核對 SHA‑256 → 立即刪除暫存 ZIP」架構，全程僅 98 次請求、**零 429 等待**，**18.4 分鐘**完成全部 32,634 張 `StreetCLIP` ViT‑L/14 編碼，永久快取全部 39,300 筆 768 維向量於 [streetclip_39k_plus_test300.npz](../streetclip_25k/streetclip_39k_plus_test300.npz)（58.0 MB）。
- **防洩漏兩階段訓練**：
  - 訓練切分明確剔除全部 `author_holdout_66` 拍攝者，再依拍攝者互斥切分 `opt_train`（9,980 張）／`tune`（1,978 張）。
  - **Stage 1（`GeoProjector`）**：768→768 殘差投影＋輔助國家頭，在全 25,000 圖庫上做多尺度 diff-creator 地理軟目標對比學習，遮蔽同拍攝者 ≤300 km、同時把同拍攝者 >300 km 留作硬負例以洗掉相機／水印指紋（`best_epoch=18`，`tune` 分數由 `2,717.9 → 3,068.4`）。
  - **Stage 2（`ListwiseGeoRanker`）**：於投影空間重撈 diff-creator Top‑100，融合 24 維特徵（投影／原始 cosine、多尺度 KDE、圖庫密度去偏、Top‑100 國家共識、StreetCLIP 多模板零樣本文字探針與輔助國家頭），以 Listwise Softmax KL 訓練（`best_epoch=1`）。
- **融合權重防洩漏**：所有後處理／混合權重（`geoproj_kde_text: wt=0.4`、`raw_kde_text_vlm: (1.2, 0.3)`、`stage2_vlm: 0.3`）均僅以 `qlora_eval_300` 做 5-fold CV 挑選，對 `author_holdout_66` 與 `official_test_300` 採 **ZERO-TOUCH 零調參評估**。

---

## 三、最終對決結果

### 1. 主戰場：`official_test_300`（147 國、300 位全新未見拍攝者，OSV‑5M 官方 test，**ZERO-TOUCH**）

| 模型 | ≤25 km | ≤200 km | ≤750 km | 國家正確 | 中位誤差 | GeoGuessr 分數 |
|---|---:|---:|---:|---:|---:|---:|
| **`osv5m/baseline`（500 萬張官方模型）** | **33 (11.0%)** | **82 (27.3%)** | 147 (49.0%) | 124 (41.3%) | 804.4 km | 2,590.1 |
| `Qwen3‑VL‑2B` zero-shot | 1 (0.3%) | 20 (6.7%) | 94 (31.3%) | 69 (23.0%) | 1,421.9 km | 1,984.9 |
| `Qwen3‑VL‑2B` 6k QLoRA | 4 (1.3%) | 26 (8.7%) | 110 (36.7%) | 87 (29.0%) | 1,306.7 km | 2,127.4 |
| `dinov2-small` 25k raw Top‑1 | 1 (0.3%) | 17 (5.7%) | 52 (17.3%) | 33 (11.0%) | 4,603.5 km | 1,186.5 |
| `StreetCLIP` 25k raw Top‑1 | 7 (2.3%) | 46 (15.3%) | 133 (44.3%) | 109 (36.3%) | 963.0 km | 2,467.7 |
| `StreetCLIP` 25k raw + KDE + Text-Probe | 11 (3.7%) | 73 (24.3%) | 162 (54.0%) | 130 (43.3%) | 650.8 km | 2,740.0 |
| `StreetCLIP` 25k Stage 1 GeoProj Top‑1 | 11 (3.7%) | 60 (20.0%) | 161 (53.7%) | 120 (40.0%) | 682.2 km | 2,772.0 |
| `StreetCLIP` 25k Stage 1 GeoProj + KDE | 6 (2.0%) | 61 (20.3%) | 168 (56.0%) | 128 (42.7%) | 603.5 km | 2,853.9 |
| **`StreetCLIP` 25k GeoProj + KDE + Text-Probe** | 8 (2.7%) | 73 (24.3%) | **174 (58.0%)** | **144 (48.0%)** | **570.2 km** | **2,907.5** |
| `StreetCLIP` 25k Stage 2 Listwise | 6 (2.0%) | 59 (19.7%) | 154 (51.3%) | 132 (44.0%) | 718.9 km | 2,730.9 |
| `StreetCLIP` 25k GeoProj + KDE + Text + Qwen3‑VL 6k | 4 (1.3%) | 64 (21.3%) | 167 (55.7%) | 131 (43.7%) | 630.0 km | 2,814.3 |

- **成對勝負（`StreetCLIP 25k GeoProj + KDE + Text-Probe`）**：
  - 對決 500 萬張 **`osv5m/baseline`**：**166 張更近** vs 131 張更遠（3 張差 ≤1 km）；GeoGuessr 分數領先 **+317.4 分**、中位誤差低 **234.2 km**、國家正確率高 **+6.7%**。
  - 對決 **`Qwen3‑VL 6k QLoRA`**：**190 張更近** vs 110 張更遠；分數領先 **+780.1 分**、中位誤差低 **736.5 km**、國家正確率高 **+19.0%**。
  - Top‑100 召回率：raw `StreetCLIP` 25k 為 `≤25km 34.3% / ≤200km 82.0% / 國家 89.7%`；經 Stage 1 GeoProj 後 `≤25km` 召回進一步拉升至 **`38.3%`**。

### 2. 其餘四個驗證切分的 25k `StreetCLIP` 進展（全嚴格 diff-creator）

| 切分 | 最佳方法 | Top‑100 ≤25km | ≤25 km | ≤200 km | 國家正確 | 中位誤差 | GeoGuessr 分數 |
|---|---|---:|---:|---:|---:|---:|---:|
| **`val_2000`** (n=2,000) | Stage 2 Listwise（對照 raw `2,651.5` 分） | 27.9% | 46 (2.3%) | 425 (21.3%) | **1,438 (71.9%)** | **474.9 km** | **3,232.0** |
| **`val_unseen_opt_522`** (訓練未見) | Stage 2 Listwise（對照 raw `2,823.7` 分） | 33.7% | 13 (2.5%) | 123 (23.6%) | **332 (63.6%)** | **482.5 km** | **3,238.0** |
| **`qlora_eval_300`** (5-fold OOF) | Stage 2 Listwise + Qwen3‑VL 6k（對照 6k 圖庫 `2,996.6`） | 28.0% | 7 (2.3%) | 67 (22.3%) | **223 (74.3%)** | **461.2 km** | **3,327.0** |
| **`author_holdout_66`** (ZERO-TOUCH) | Stage 1 GeoProj + KDE（對照 6k 圖庫 `3,395.4`） | **57.6%** | 4 (6.1%) | **22 (33.3%)** | 39 (59.1%) | **318.2 km** | **3,465.3** |
| **`author_holdout_66`** (ZERO-TOUCH) | Raw 25k KDE + Text + Qwen3‑VL 6k | 48.5% | 4 (6.1%) | 20 (30.3%) | **45 (68.2%)** | **308.7 km** | 3,440.0 |

---

## 四、三個深層技術洞察

1. **為什麼我們的 25k 系統在國家／中位距離／GeoGuessr 分數全面擊敗 500 萬張 `osv5m/baseline`，但 `osv5m/baseline` 仍贏在 ≤25 km（11.0% vs 2.7%–3.7%）？**
   - `osv5m/baseline` 以 489 萬張圖端到端訓練萬格 quadtree 與座標回歸，在「精準 25 km 落點」上有 200 倍資料密度優勢；而我們僅有 25,000 張離散圖庫，且 `σ=500 km` 寬核 KDE 刻意優先保大區域安全、拉高整體 GeoGuessr 期望分數（看 Stage 1 Top‑1 不做寬核平滑時 ≤25 km 有 11 張，做寬核後降為 6–8 張，但中位誤差從 682 壓到 570 km）。
   - 若要同步逼近 `osv5m/baseline` 的 ≤25 km 精準度，下一步最直接的槓桿是「擴大圖庫密度（25k → 100k+）」以及「先用寬核鎖定區域／國家，再在勝出區域內改用窄核（`σ=30–50 km`）細選座標」。
2. **Stage 1 `GeoProjector` 跨分布泛化最穩；Stage 2 `ListwiseRanker` 略微偏重 OSV‑5M train 國家先驗**
   - Stage 1 `GeoProjector` 在全部 5 個切分（包含 147 國的 `official_test_300`）全部穩定大漲（`official_test_300` 單靠 Stage 1 就跳升 `+304` 分、Top‑100 ≤25 km 從 `34.3% → 38.3%`）。
   - Stage 2 因為餵入了 `cc_freq_log` 與 `aux_cc_logp`（反映 OSV‑5M train 偏重美／俄／歐的分布），在 `val_2000`（71.9% 國對）與 `qlora_eval_300`（74.3% 國對）稱王，但在 147 國均勻分布的 `official_test_300` 與 `author_holdout_66` 上反而不如無國家頻率偏見的 **`Stage 1 GeoProj + KDE + Text-Probe`**。
3. **固定權重混入弱 VLM 會拖累長尾國家表現**
   - 在主要國家為主的 `qlora_eval_300`（VLM 國對 52%）上，加入 Qwen3‑VL 6k 把混合國家正確率一口氣推到 **74.3%**（`3,327.0` 分）；但在 147 國長尾分布的 `official_test_300` 上，Qwen3‑VL 6k 國對跌到 `29.0%`，以固定權重 `0.3` 混入已達 `48.0%` 的 `StreetCLIP 25k GeoProj+KDE+Text` 反而微降到 `43.7%`（`2,814.3` 分）。
   - 結論：**VLM 不該無條件以固定權重覆蓋強地理檢索**，而應該改成**信心／低熵門控（例如偵測到清晰招牌文字並接上離線 GeoNames 地名查詢時才介入）**——這正好把方向 1 與方向 2 完美接軌。

---

## 五、重現與產出檔案

- 官方測試集鎖定：`python3 -B scripts/prepare_official_test.py --select-only` → `--download` → `--filter-and-lock` → [official-test-300.zip](../../data/official_test_300/official-test-300.zip)、[hud_audit.json](../../data/official_test_300/hud_audit.json)
- Phase B（Colab L4，已終止）：[prepare_phase_b_bundle.py](../../scripts/prepare_phase_b_bundle.py) → [results/streetclip_25k/streetclip_25k_results.zip](../streetclip_25k/streetclip_25k_results.zip)、[streetclip_39k_plus_test300.npz](../streetclip_25k/streetclip_39k_plus_test300.npz)
- Phase C（Colab T4，已終止）：[prepare_phase_c_bundle.py](../../scripts/prepare_phase_c_bundle.py) → [phase_c_results.zip](phase_c_results.zip)
- Phase D（本機分析）：`python3 -B scripts/evaluate_official_test_showdown.py` → [showdown_summary.json](showdown_summary.json)
