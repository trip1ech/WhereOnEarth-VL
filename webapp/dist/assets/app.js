"use strict";

const { examples, metrics } = window.GEO_DATA;
const naturalEarth = window.GEO_BASEMAP;
const element = id => document.getElementById(id);
const controls = {
  version: element("version"),
  filter: element("filter"),
  picture: element("picture"),
  previous: element("previous"),
  next: element("next"),
  reset: element("reset"),
  sync: element("sync"),
};
const photo = element("photo");
const photoError = element("photo-error");
const result = element("comparison-result");
const regions = new Intl.DisplayNames(["zh-Hant"], { type: "region", fallback: "none" });
const rounded = new Intl.NumberFormat("zh-Hant", { maximumFractionDigits: 0 });
const oneDecimal = new Intl.NumberFormat("zh-Hant", { maximumFractionDigits: 1 });
const swatches = { zero: "#a43929", fine: "#096c59" };
const originalIndex = new Map(examples.map((example, index) => [example.id, index + 1]));
const modelLabels = { sft: "微調 6,000 張", sft_2000: "微調 2,000 張" };
let matches = [];
let selected = null;
let extent = null;
let synchronizing = false;

function country(code) {
  if (!code) return "未提供";
  return /^[A-Z]{2}$/.test(code) ? regions.of(code) || code : code;
}

function kilometres(value) {
  return (value < 10 ? oneDecimal : rounded).format(value) + " km";
}

function relativePoint([latitude, longitude], referenceLongitude) {
  return [latitude, longitude + 360 * Math.round((referenceLongitude - longitude) / 360)];
}

function createMap(side) {
  const map = L.map("map-" + side, {
    minZoom: 1,
    maxZoom: 19,
    zoomSnap: 0.5,
    zoomControl: false,
    scrollWheelZoom: false,
  }).setView([20, 0], 1);
  const zoom = L.control.zoom({
    zoomInTitle: "放大",
    zoomOutTitle: "縮小",
  }).addTo(map);
  zoom.getContainer().setAttribute("aria-label", "地圖縮放");
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution("Natural Earth");
  map.createPane("country-outlines");
  const outlines = map.getPane("country-outlines");
  outlines.style.zIndex = "190";
  outlines.style.pointerEvents = "none";
  for (const offset of [-360, 0, 360]) {
    L.geoJSON(naturalEarth, {
      pane: "country-outlines",
      interactive: false,
      coordsToLatLng: point => L.latLng(point[1], point[0] + offset),
      style: {
        color: "#94a8b4",
        weight: 0.75,
        fill: true,
        fillOpacity: 1,
        fillColor: "#eef0e9",
      },
    }).addTo(map);
  }
  if (location.protocol === "http:" || location.protocol === "https:") {
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxNativeZoom: 19,
    }).addTo(map);
  }

  return { map, marks: L.layerGroup().addTo(map) };
}

const panels = { zero: createMap("zero"), fine: createMap("fine") };

function copyView(source, destination) {
  const center = source.getCenter();
  const zoom = source.getZoom();
  if (destination.getZoom() === zoom && destination.getCenter().equals(center, 0.000001)) {
    return;
  }
  synchronizing = true;
  destination.setView(center, zoom, { animate: false });
  synchronizing = false;
}

for (const [source, destination] of [["zero", "fine"], ["fine", "zero"]]) {
  panels[source].map.on("moveend", () => {
    if (controls.sync.checked && !synchronizing) {
      copyView(panels[source].map, panels[destination].map);
    }
  });
}

const observer = new ResizeObserver(() => {
  synchronizing = true;
  for (const { map } of Object.values(panels)) map.invalidateSize({ pan: false });
  synchronizing = false;
});
for (const { map } of Object.values(panels)) observer.observe(map.getContainer());

function showExtent() {
  if (!extent) return;
  const { zero, fine } = panels;
  synchronizing = true;
  zero.map.fitBounds(extent, { padding: [42, 42], maxZoom: 13, animate: false });
  fine.map.setView(zero.map.getCenter(), zero.map.getZoom(), { animate: false });
  synchronizing = false;
}

function putMarker(marks, mapPoint, sourcePoint, variant, label, countryCode) {
  const size = variant === "truth" ? 22 : 14;
  const icon = L.divIcon({
    className: "location-pin " + variant + "-pin",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
  const popup = document.createElement("div");
  const heading = document.createElement("strong");
  heading.textContent = label + " · " + country(countryCode);
  popup.append(heading, document.createElement("br"), sourcePoint.map(number => number.toFixed(5)).join(", "));
  L.marker(mapPoint, {
    icon,
    title: label,
    alt: label,
    zIndexOffset: variant === "truth" ? 0 : 100,
  }).bindPopup(popup).addTo(marks);
}

function plot(side, example, guess) {
  const { marks } = panels[side];
  marks.clearLayers();
  if (!example) return;
  if (guess.point) {
    const displayPoint = relativePoint(guess.point, example.point[1]);
    L.polyline([example.point, displayPoint], {
      color: swatches[side],
      opacity: 0.8,
      weight: 2,
      dashArray: "5 5",
      interactive: false,
    }).addTo(marks);
    putMarker(marks, displayPoint, guess.point, side, "預測", guess.country);
  }
  putMarker(marks, example.point, example.point, "truth", "真值", example.country);
}

function updateSummary(side, model) {
  const summary = metrics[model];
  const items = [
    `國家正確 ${summary.country_correct}/${summary.n}`,
    `25 km 內 ${summary.within_25km}/${summary.n}`,
    `中位誤差 ${kilometres(summary.median_km)}`,
  ];
  element("metrics-" + side).replaceChildren(...items.map(item => {
    const span = document.createElement("span");
    span.textContent = item;
    return span;
  }));
}

function predictionDetails(side, guess) {
  element("country-" + side).textContent = guess ? "猜測 " + country(guess.country) : "—";
  element("distance-" + side).textContent = guess?.distance == null
    ? "—"
    : oneDecimal.format(guess.distance) + " km";
}

function showCredit(example) {
  const credit = element("credit");
  credit.replaceChildren();
  if (!example) return;
  credit.append([example.attribution, "OSV-5M"].filter(Boolean).join(" · "));
  if (example.license) {
    const license = document.createElement("a");
    license.textContent = example.license;
    if (example.license === "CC-BY-SA-4.0") {
      license.href = "https://creativecommons.org/licenses/by-sa/4.0/";
      license.target = "_blank";
      license.rel = "noopener noreferrer";
    }
    credit.append(" · ", license);
  }
}

function showSelected() {
  const index = matches.findIndex(example => example.id === controls.picture.value);
  selected = matches[index] || null;
  controls.previous.disabled = index <= 0;
  controls.next.disabled = index === -1 || index >= matches.length - 1;
  controls.reset.disabled = !selected;
  const model = controls.version.value;
  const zero = selected?.guesses.zero_shot;
  const fine = selected?.guesses[model];
  element("fine-title").textContent = modelLabels[model];
  predictionDetails("zero", zero);
  predictionDetails("fine", fine);
  updateSummary("zero", "zero_shot");
  updateSummary("fine", model);
  plot("zero", selected, zero);
  plot("fine", selected, fine);
  showCredit(selected);
  result.dataset.tone = "";
  photoError.hidden = true;
  photo.hidden = !selected;
  if (!selected) {
    element("truth").textContent = "—";
    result.textContent = "沒有符合的照片";
    extent = null;
    return;
  }

  photo.src = selected.photo;
  photo.alt = `第 ${originalIndex.get(selected.id)} 張評估街景照片`;
  element("truth").textContent = country(selected.country);
  if (zero.distance == null || fine.distance == null) {
    const missing = [zero.distance == null ? "Zero-shot" : "", fine.distance == null ? "微調" : ""]
      .filter(Boolean).join("、");
    result.textContent = missing + " 缺少可用座標";
  } else {
    const change = fine.distance - zero.distance;
    if (Math.abs(change) < 0.1) {
      result.textContent = "這張照片的兩個預測誤差相近";
    } else {
      const improved = change < 0;
      result.dataset.tone = improved ? "better" : "worse";
      result.textContent = `這張照片微調後更${improved ? "近" : "遠"} ${kilometres(Math.abs(change))}`;
    }
  }

  extent = L.latLngBounds([selected.point]);
  for (const guess of [zero, fine]) {
    if (guess.point) extent.extend(relativePoint(guess.point, selected.point[1]));
  }
  showExtent();
}

function refreshPicker() {
  const model = controls.version.value;
  const filter = controls.filter.value;
  matches = examples.filter(example => {
    if (filter === "all") return true;
    const zero = example.guesses.zero_shot.distance;
    const fine = example.guesses[model].distance;
    const missing = zero == null || fine == null;
    if (filter === "missing") return missing;
    if (missing) return false;
    return filter === "better" ? zero - fine >= 0.1 : fine - zero >= 0.1;
  });
  const options = matches.map(example => {
    const label = `第 ${originalIndex.get(example.id)} / ${examples.length} 張`;
    return new Option(label, example.id);
  });
  controls.picture.replaceChildren(...(options.length ? options : [new Option("沒有符合的照片", "")]));
  controls.picture.disabled = !options.length;
  if (matches.includes(selected)) controls.picture.value = selected.id;
  showSelected();
}

photo.addEventListener("error", () => {
  if (!selected) return;
  photo.hidden = true;
  photoError.hidden = false;
});
controls.previous.addEventListener("click", () => {
  controls.picture.selectedIndex -= 1;
  showSelected();
});
controls.next.addEventListener("click", () => {
  controls.picture.selectedIndex += 1;
  showSelected();
});
controls.picture.addEventListener("change", showSelected);
controls.filter.addEventListener("change", refreshPicker);
controls.version.addEventListener("change", refreshPicker);
controls.sync.addEventListener("change", () => {
  if (controls.sync.checked) copyView(panels.zero.map, panels.fine.map);
});
controls.reset.addEventListener("click", showExtent);
element("total").textContent = examples.length;
refreshPicker();
