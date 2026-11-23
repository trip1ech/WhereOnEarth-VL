# 街景定位比較

用瀏覽器開啟 `dist/index.html`，即可切換同一批 300 張街景，對照
zero-shot、以 6,000 張照片微調的模型，以及以 2,000 張照片微調的模型。
地圖顯示真值和各自的預測，預設同步拖曳和縮放。關閉「同步地圖」即可獨立操作。
縮放可用地圖左側的 +／− 或觸控手勢。

直接開啟 HTML 可離線使用 Natural Earth 國界底圖。如果要看更細的道路底圖，
可以從任何目錄執行以下指令，再造訪 <http://127.0.0.1:8765/>。
道路底圖由 OpenStreetMap 在瀏覽器中載入。用完可以在終端機按 `Control+C` 停止。

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory "/Volumes/PortableSSD/Codex/GeoGuessr-Recognition/webapp/dist"
```

`python3 build_data.py` 會從專案內原有評估檔與照片重建網頁所用的資料。
網頁使用 Leaflet，授權條款見 `dist/assets/leaflet-LICENSE.txt`。
