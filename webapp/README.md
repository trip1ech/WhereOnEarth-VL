# Street-View Prediction Comparator (`webapp`)

An interactive Leaflet viewer to browse the 300-image validation split (`qlora_eval_300`) and compare **Zero-shot**, **2k QLoRA**, and **6k QLoRA** predictions against ground-truth coordinates side by side.

## Building & Running

1. First generate the static evaluation bundle (`dist/assets/photos/`, `data.js`, and `basemap.js` — git-ignored because they contain 300 street-view JPEGs) from your local evaluation archives:

   ```sh
   python3 webapp/build_data.py
   ```

2. Open `webapp/dist/index.html` directly for offline Natural Earth country-border basemaps, or serve locally over HTTP to enable OpenStreetMap road tiles:

   ```sh
   python3 -m http.server 8765 --bind 127.0.0.1 --directory webapp/dist
   ```

   Then open <http://127.0.0.1:8765/>. Press `Control+C` in the terminal to stop.

Maps pan and zoom in sync by default; uncheck **Sync maps** to inspect independently. Built with Leaflet (license in [`dist/assets/leaflet-LICENSE.txt`](dist/assets/leaflet-LICENSE.txt)).
