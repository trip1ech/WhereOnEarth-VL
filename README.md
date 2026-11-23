# GeoGuessr Recognition

單張街景、全球經緯度預測。第一輪比較 Qwen3‑VL zero-shot 與 OSV‑5M 官方定位模型。

> 全階段實驗演進、指標總表、資料切分地圖與已知瓶頸整理見 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 0 · 評估協議

- OSV‑5M 官方 test 只作評估。Pilot 從巴黎、里昂、圖盧茲周邊抽樣，以固定 seed 每個拍攝序列取一張。
- Panoramax IGN 提供不同來源的街景。每個 OSV 樣本找鄰近的非全景影像，紀錄兩者的距離、來源、署名和授權。城市與地理鄰近程度受控。道路類型、拍攝時間及景物尚未人工配對。
- `pilot_dev` 與 `pilot_holdout` 採互斥拍攝序列。Holdout 獨立打包，在固定方法後才執行。法國小樣本用於驗證流程，正式全球評估須增加州國、城鄉與拍攝場景覆蓋。
- 同路段、同方向、近時間的跨裝置拍攝是後續 shortcut 對照。鄰近的跨平台影像提供跨來源測試初樣本。
- 主指標為 25 km 命中率，另外報 200 km、有效預測率與有效預測的中位公里誤差。無法解析的座標計入命中率分母。正式比較使用路段分群 bootstrap。
- 兩種模型共用 EXIF 方向校正後的 RGB 像素。真實 GPS、區域、來源、檔名與 EXIF 只在載入和評分端使用。圖片本身看得到的相機痕跡保留給後續消融。

實際的 pilot 共 36 張照片，開發集與保留集各 18 張，兩個來源各占一半。跨來源地理配對的中位距離約 1.97 km，最遠約 12.45 km。兩個切分之間最近影像約隔 10.43 km。

## 1 · VS Code Jupyter + 免費 Colab GPU

1. 在 VS Code 開啟這個 SSD 資料夾，再開 `notebooks/01_phase1_colab.ipynb`。
2. 選已配置的 T4 runtime，或選 **Select Kernel → Colab → New Colab Server → GPU → T4**。先跑 GPU 和套件檢查。
3. 在 VS Code Explorer 對 `geoguessr-pilot_dev.zip` 按右鍵，選 **Upload to Colab**。Notebook 由 ZIP 鑑別資料切分和影像校驗值。
4. 依序跑兩個 baseline。Notebook 逐筆寫入 JSONL，並輸出 CSV 和結果 ZIP。
5. 從 Colab Explorer 把 `phase1-pilot_dev-results.zip` 下載到本專案的 `results/`，再儲存 Notebook。Colab runtime 的資料、權重、結果均為暫存；本專案的原始資料與留存結果放 SSD。

切換到 `pilot_holdout` 需在 Notebook 設定格改 `SPLIT`，並上傳對應 ZIP。模型權重只在 Colab 取得。 `00_colab_gpu.ipynb` 是獨立 GPU 檢查。

## 本機資料準備

`data/raw/osv5m-test.csv` 來自 [OSV‑5M 固定版本](https://huggingface.co/datasets/osv5m/osv5m/tree/cff33609b56b54d8743b7ee7a416eb8433e9a681)。重建可用 `python3 scripts/prepare_pilot.py` 和 `python3 scripts/build_bundles.py`。準備器只下載必要的 ZIP byte ranges 與外部樣本，保留 data manifest 和影像 SHA‑256。Panoramax API 隨時間變動，重跑可能取得更新的外部樣本，請保留實驗使用的原始 manifest。

- [OSV‑5M 論文](https://arxiv.org/abs/2404.18873)
- [Panoramax STAC API](https://docs.panoramax.fr/backend/api/api/)
- [Qwen3‑VL‑2B‑Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## 2 · 檢索前的資料分布檢查

先開 `notebooks/03_dataset_distribution.ipynb` 看完整 OSV‑5M train、test 和 2,000 張圖庫的全球分布。Notebook 使用固定版本的 CSV 統計國家占比與 2° 網格，並比較兩個圖庫對 `pilot_dev` 的 25 km 覆蓋率。完整 train 覆蓋 18/18，2,000 張圖庫覆蓋 2/18。

在 VS Code 把 `distribution_audit_upload.zip` 上傳到 Colab，再執行 notebook 即可重現。完整 metadata 掃描用 CPU。SSD 上的 `results/distribution_audit/` 保留國家與網格統計、逐張覆蓋率和圖表，已具備這些結果時可直接使用。

## 3 · Frozen retrieval on Colab T4

Run `python3 -B scripts/prepare_gallery.py` on the SSD to build `data/gallery_2000`.
The deterministic bootstrap samples 144 byte windows of official train metadata and round-robins 2° geographic cells with distinct capture sequences.
It extracts chosen JPEGs directly from the original train archives. Successful shards are retained for resuming an interrupted download.

Run `python3 -B scripts/pack_retrieval.py` then upload the resulting `retrieval_2000_upload` folder from the VS Code Explorer to the Colab server.
Run `notebooks/02_frozen_retrieval.ipynb` and save the exported result ZIP back to SSD `results/`.
The evaluation bundles only `pilot_dev`. Result caches are keyed by ordered image hashes and encoder revision.
The 2,000-image run is a pipeline and coverage check. The gallery, visual top 100, geographic top 10, and final top 1 each have their own 25 km diagnostic.

## 4 · 排序器前處理

`notebooks/04_ranker_preprocessing.ipynb` 使用完整 OSV‑5M train metadata，從 2018-01-01 UTC 起的照片重新選擇圖庫和 query。先以約 11 km 網格選出候選，再優先選能補足 25 km 覆蓋的照片。在同一組 6,000 張 2018 年起的覆蓋率 query 上，25,000 張新版圖庫的覆蓋率為 97.32%，原圖庫為 97.22%，同大小的新版隨機抽序列對照為 73.97%。

新版 GPS 清單、雜湊與覆蓋率存於 `data/ranker_preprocessing/since_2018/`。`dataset.json` 鎖定 25,000 張圖庫、12,000 張排序器 train query、2,000 張 validation query 和 6,000 張覆蓋率 query。四份清單已核對圖片 ID 和拍攝序列互斥。Notebook 使用現有 GPS 清單，或在 Colab 上傳 `ranker_preprocessing_upload.zip` 後重新掃描公開 metadata。上層目錄保留全拍攝年份的清單供比較。

目前停在新版資料清單。新版圖庫與 train、validation 共 39,000 個 ID 已全部在官方 ZIP 索引找到。後續使用 `materialize_ranker_data.py` 時，將 `--data-dir` 指向 `data/ranker_preprocessing/since_2018`，以擷取新版 JPEG 並核對雜湊。需要明確加上 `--encode` 才會執行凍結 DINOv2 與 Top‑100 配對準備。若產生 `_pairs.npy`，三欄依序是 **query 列號、正例圖庫列號、負例圖庫列號**，均從 0 起算。

## 5 · 街景排序器

在 VS Code 將 `ranker_training_upload.zip` 上傳到 Colab，選擇 GPU kernel 並執行 `notebooks/05_ranker_colab.ipynb`。Notebook 先驗證 256 張圖庫與兩組各 32 張的 query，再跑完整圖庫。完整結果包含 GPS 覆蓋、Top‑100 和 Top‑1 命中、可配對的 query 數量與按國家拆分的診斷。

`train_ranker.py` 從固定的 cosine Top‑100 訓練殘差 MLP。在排序器 train 中留出一部分做 checkpoint 選擇，再在獨立的 validation 對照原始 cosine 和重排序。25 km 內的候選作正例，訓練 loss 對每張 query 等權。從 Colab 下載 `ranker-2018-pairs.zip` 與 `ranker-2018-evaluation.zip` 存入 SSD `results/`，可保存向量、候選與評估結果。

## 6 · Qwen3‑VL 單張街景 QLoRA

`scripts/prepare_sft_data.py` 從 2018 年起的已鎖定清單分層選 2,000 張訓練圖，另從既有驗證清單固定抽 300 張。分層使用國家與地理區塊，拍攝序列互斥。CPU 從 OSV‑5M 原始 ZIP 取得 JPEG，在 SSD 保留原圖與上傳 bundle。上傳圖保留方向校正後的 RGB 像素，最長邊為 512 px。座標保留原始精度計分，模型的標準回答座標以三位小數呈現。

在 VS Code 將 `geoguessr_sft_upload.zip` 上傳到已配置的 Colab GPU，再執行 `notebooks/06_sft_colab.ipynb`。單檔傳輸受限時，改上傳 `data/qlora_since_2018/sft_upload_parts/`，Notebook 會在 SHA-256 驗證後重組 bundle。Notebook 共用一個基底模型，分別比較停用 LoRA 的 zero-shot 和一輪回答 token cross-entropy 微調。先跑小型訓練驗證。正式流程完成微調時先匯出可恢復的 adapter，再批次評估同一組保留影像。

將 `geoguessr-sft-run-results.zip` 下載到 SSD 的 `results/`，核對 `metrics.csv`、逐張預測、訓練紀錄與 LoRA adapter，再終止 Colab runtime。

本次已完成的結果位於 `results/qlora/`。Notebook 頂部保留實測表格，可直接離線閱讀。

## 未見拍攝者的評估

新增資料先抽看有代表性的原圖，再鎖定抽樣、適用範圍與排除規則。記錄照片來源、拍攝者與序列重疊、地區分布、影像品質和可見的答案疊字，將跨來源測試的地理分布與場景差異一起報告。
這次的逐圖記錄及測試輸入見 [data/author_holdout/README.md](data/author_holdout/README.md)。

## 中央標線與 ViT 診斷

中央線裁切試驗及待驗證的 LM 假說記在[裁切報告](results/roadmark_crop_probe/README.md#待驗證的-lm-假說)。後續可固定整圖的視覺特徵，只控制 LM 能使用的影像位置，再判斷辨識瓶頸。

文字證據的定位對照與道路線索的 LM 續訓見[先讀線索再定位](results/clue_first/README.md)及[續訓報告](results/clue_first_training/README.md)。
