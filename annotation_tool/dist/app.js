"use strict";

const element = id => document.getElementById(id);
const datasetPicker = element("dataset");
const sample = element("sample");
const viewer = element("viewer");
const progress = element("progress");
const count = element("count");
const classCounts = element("class-counts");
const status = element("status");
const previous = element("previous");
const next = element("next");
const nextUnlabeled = element("next-unlabeled");
const clear = element("clear");
const exportLink = element("export");
const buttons = [...document.querySelectorAll(".label")];
const keyLabels = { 1: "single", 2: "double", 3: "none", 4: "unclear" };
const labelNames = { single: "單線", double: "雙線", none: "無中央線", unclear: "看不清／不適用" };

const resumes = new Map();
const resumeKey = "roadmark-annotation-view-with-followup";
const followupIds = [
  "524527566215015",
  "423752349436605",
  "3967126560029825",
  "761880321196618",
  "1062347254473266",
  "326967888929599",
  "2569715476670950",
  "1212145929242560",
  "138714041578549",
  "899701950761265",
  "172642221309076",
  "301133488220981",
  "450730583872952",
  "1069927146873984",
  "468565351113137",
  "940338356716443",
  "215443046804022",
  "809754973285749",
  "1386739581697239",
  "200056905290006",
  "4813025322082240",
  "3165981590285181",
  "214025490363489",
  "200169655275789",
  "2918408071772147",
  "1260801737648470",
  "202035325085945",
  "1103989746679970",
  "293997682375843",
  "1437322729948049",
];
let datasets = [];
let labels = {};
let current = null;
let index = 0;
let ready = false;
let saving = false;

function message(text, error = false) {
  status.textContent = text;
  status.dataset.error = String(error);
}

function findUnlabeled(after = -1) {
  const later = current.ids.findIndex((id, i) => i > after && !labels[id]);
  return later >= 0 ? later : current.ids.findIndex(id => !labels[id]);
}

function updateControls() {
  const imageId = current.ids[index];
  const tallies = { single: 0, double: 0, none: 0, unclear: 0 };
  for (const id of current.ids) {
    const label = labels[id];
    if (label) tallies[label] += 1;
  }
  const marked = Object.values(tallies).reduce((sum, total) => sum + total, 0);
  const destination = findUnlabeled(index);
  count.textContent = `已標註 ${marked.toLocaleString("zh-Hant")} / ${current.ids.length.toLocaleString("zh-Hant")} 張`;
  classCounts.textContent = `單線 ${tallies.single} · 雙線 ${tallies.double}`;
  if ((current.key === "pilot" || current.key === "followup") && marked === current.ids.length) {
    classCounts.textContent += " · 本批完成";
  }
  progress.max = current.ids.length;
  progress.value = marked;
  datasetPicker.disabled = saving;
  previous.disabled = saving || index === 0;
  next.disabled = saving || index === current.ids.length - 1;
  nextUnlabeled.disabled = saving || destination === -1 || destination === index;
  clear.disabled = saving || !labels[imageId];
  for (const button of buttons) {
    button.disabled = saving || !ready;
    button.setAttribute("aria-pressed", String(button.dataset.label === labels[imageId]));
  }
  const saved = Object.keys(labels).length > 0;
  exportLink.setAttribute("aria-disabled", String(!saved));
  if (saved) exportLink.href = "/api/export";
  else exportLink.removeAttribute("href");
}

function show(indexToShow) {
  index = indexToShow;
  ready = false;
  resumes.set(current.key, index);
  try {
    localStorage.setItem(resumeKey, JSON.stringify({ key: current.key, id: current.ids[index] }));
  } catch {
    // The SSD still holds annotations when browser storage is unavailable.
  }
  sample.textContent = `第 ${(index + 1).toLocaleString("zh-Hant")} / ${current.ids.length.toLocaleString("zh-Hant")} 張`;
  const image = document.createElement("img");
  image.alt = "待標註的街景照片";
  image.addEventListener("load", () => {
    if (viewer.firstElementChild !== image) return;
    ready = true;
    updateControls();
  });
  image.addEventListener("error", () => {
    if (viewer.firstElementChild !== image) return;
    const notice = document.createElement("p");
    notice.className = "photo-error";
    notice.textContent = "照片載入失敗。請確認 SSD 已連接，再重新整理。";
    viewer.replaceChildren(notice);
  });
  viewer.replaceChildren(image);
  image.src = "/photo/" + encodeURIComponent(current.ids[index]);
  updateControls();
}

function selectDataset(key) {
  current = datasets.find(dataset => dataset.key === key);
  const first = findUnlabeled();
  show(resumes.get(key) ?? (first >= 0 ? first : 0));
  message("");
}

async function save(label) {
  if (saving || (!ready && label !== null)) return;
  const imageId = current.ids[index];
  saving = true;
  updateControls();
  message("儲存中…");
  try {
    const response = await fetch("/api/label", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId, label }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    if (label === null) {
      delete labels[imageId];
      message("已清除這張的標註");
    } else {
      labels[imageId] = label;
      message(`已存到 SSD · ${labelNames[label]}`);
      const destination = findUnlabeled(index);
      if (destination >= 0) show(destination);
    }
  } catch (error) {
    message(error.message === "Failed to fetch" ? "無法連到本機服務，這次標註尚未確認。" : error.message, true);
  } finally {
    saving = false;
    updateControls();
  }
}

previous.addEventListener("click", () => { if (!previous.disabled) show(index - 1); });
next.addEventListener("click", () => { if (!next.disabled) show(index + 1); });
nextUnlabeled.addEventListener("click", () => {
  if (!nextUnlabeled.disabled) show(findUnlabeled(index));
});
clear.addEventListener("click", () => { if (!clear.disabled) save(null); });
for (const button of buttons) {
  button.addEventListener("click", () => save(button.dataset.label));
}
datasetPicker.addEventListener("change", () => selectDataset(datasetPicker.value));
document.addEventListener("keydown", event => {
  if (!current || event.repeat || event.altKey || event.ctrlKey || event.metaKey
      || event.target.closest?.("select, input, textarea, [contenteditable]")) return;
  if (keyLabels[event.key]) {
    event.preventDefault();
    save(keyLabels[event.key]);
  } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    (event.key === "ArrowLeft" ? previous : next).click();
  }
});

async function start() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) throw new Error("連線失敗");
    ({ datasets, labels } = await response.json());
    const train = datasets.find(dataset => dataset.key === "train");
    const pilot = { key: "pilot", name: "第一批", ids: train.ids.slice(0, 40) };
    const followup = { key: "followup", name: "第二批補標", ids: followupIds };
    const pilotComplete = pilot.ids.every(id => labels[id]);
    datasets.unshift(...(pilotComplete ? [followup, pilot] : [pilot, followup]));
    train.name = "完整訓練集";
    datasetPicker.replaceChildren(...datasets.map(dataset =>
      new Option(`${dataset.name}（${dataset.ids.length.toLocaleString("zh-Hant")}）`, dataset.key)
    ));
    let resume;
    try {
      resume = JSON.parse(localStorage.getItem(resumeKey));
    } catch {
      resume = null;
    }
    const dataset = datasets.find(item => item.key === resume?.key);
    if (dataset) {
      datasetPicker.value = dataset.key;
      const resumeIndex = dataset.ids.indexOf(resume.id);
      if (resumeIndex >= 0) resumes.set(dataset.key, resumeIndex);
    }
    selectDataset(datasetPicker.value);
  } catch {
    sample.textContent = "無法載入資料";
    message("請確認本機服務已啟動、SSD 已連接，再重新整理。", true);
  }
}
start();
