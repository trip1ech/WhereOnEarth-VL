"""Shared utilities for Multi-Scale Spatial KDE (Top-8 Clusters) -> Qwen3-VL-8B-Instruct
CoT reasoning, Rejection Sampling, STaR Rationalization, SFT, and Offline Distance DPO.
"""
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def geoscore(dist_km: float) -> float:
    return 5000.0 * math.exp(-dist_km / 1492.7)


def compute_trajectory_reward(
    dist_km: Optional[float],
    pred_cc: Optional[str],
    true_cc: str,
    parsed: bool,
) -> float:
    """Continuous multi-scale geographic reward used for Rejection Sampling & DPO pairs."""
    if not parsed or dist_km is None or not math.isfinite(dist_km):
        return -500.0
    r = geoscore(dist_km)
    if pred_cc == true_cc:
        r += 500.0
    if dist_km <= 200.0:
        r += 150.0
    if dist_km <= 25.0:
        r += 250.0
    return float(r)


def compute_kde_log_scores(
    top_idx: np.ndarray,
    top_sim: np.ndarray,
    g_lat: np.ndarray,
    g_lon: np.ndarray,
    tau: float = 0.03,
    sigma: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    la = np.radians(g_lat[top_idx])
    lo = np.radians(g_lon[top_idx])
    dlat = la[:, :, None] - la[:, None, :]
    dlon = lo[:, :, None] - lo[:, None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(la[:, :, None]) * np.cos(la[:, None, :]) * np.sin(dlon / 2) ** 2
    )
    pw_d = (
        6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))
    ).astype(np.float32)
    w = np.exp((top_sim - top_sim[:, :1]) / tau)
    k_mat = np.exp(-pw_d / sigma)
    kde_log = np.log(np.maximum(np.einsum("nj,nij->ni", w, k_mat), 1e-12))
    return kde_log, pw_d


def build_multiscale_clusters(
    top_idx_1: np.ndarray,
    top_sim_1: np.ndarray,
    kde_1: np.ndarray,
    pw_d_1: np.ndarray,
    txt_lp_1: np.ndarray,
    g_lat: np.ndarray,
    g_lon: np.ndarray,
    g_cc: np.ndarray,
    cc_vocab: List[str],
    cc_names: Dict[str, str],
    wt: float = 0.4,
    max_k: int = 8,
    same_cc_radius_km: float = 200.0,
    max_per_country: int = 3,
) -> Dict:
    """Build Top-8 dynamic multi-scale spatial KDE clusters (never merging different countries,
    merging same-country points within 200 km, capped at 3 regional modes per country so
    large countries expose distinct sub-regions without crowding out competing countries).
    """
    cc_to_i = {c: i for i, c in enumerate(cc_vocab)}
    ccs = [str(g_cc[i]) for i in top_idx_1]
    ccs_arr = np.array(ccs)
    tlp = np.array(
        [txt_lp_1[cc_to_i[c]] if c in cc_to_i else -15.0 for c in ccs],
        dtype=np.float64,
    )
    comb = kde_1.astype(np.float64) + wt * tlp
    w = np.exp(comb - comb.max())
    w /= w.sum()

    txt_probs = np.exp(txt_lp_1.astype(np.float64))
    txt_order = np.argsort(-txt_probs)
    txt_rank_map = {cc_vocab[int(idx)]: r + 1 for r, idx in enumerate(txt_order)}
    top_text_countries = [
        {
            "country": cc_vocab[int(idx)],
            "country_name": cc_names.get(cc_vocab[int(idx)], cc_vocab[int(idx)]),
            "prob": round(float(txt_probs[int(idx)]), 4),
        }
        for idx in txt_order[:3]
    ]

    used = np.zeros(len(top_idx_1), dtype=bool)
    cc_counts = Counter()
    clusters = []
    order = np.argsort(-comb)

    for idx in order:
        if used[idx]:
            continue
        seed_cc = ccs[idx]
        if cc_counts[seed_cc] >= max_per_country:
            continue
        members = np.where((~used) & (ccs_arr == seed_cc) & (pw_d_1[idx] <= same_cc_radius_km))[0]
        used[members] = True
        cc_counts[seed_cc] += 1
        g_peak = int(top_idx_1[idx])

        # Narrow local KDE (sigma=40km) within cluster for sharp pinpoint anchor
        m_d = pw_d_1[np.ix_(members, members)]
        m_w = np.exp((top_sim_1[members] - top_sim_1[members].max()) / 0.03)
        local_kde = (m_w[None, :] * np.exp(-m_d / 40.0)).sum(axis=1)
        g_narrow = int(top_idx_1[members[int(np.argmax(local_kde))]])

        clusters.append({
            "rank": len(clusters) + 1,
            "country": seed_cc,
            "country_name": cc_names.get(seed_cc, seed_cc),
            "latitude": round(float(g_lat[g_peak]), 4),
            "longitude": round(float(g_lon[g_peak]), 4),
            "narrow_latitude": round(float(g_lat[g_narrow]), 4),
            "narrow_longitude": round(float(g_lon[g_narrow]), 4),
            "mass": round(float(w[members].sum()), 4),
            "neighbors": int(len(members)),
            "max_sim": round(float(top_sim_1[members].max()), 3),
            "text_rank": int(txt_rank_map.get(seed_cc, 999)),
        })
        if len(clusters) >= max_k:
            break

    # Country-level KDE totals across Top-100 for global context
    cc_mass = {}
    for c, wi in zip(ccs, w):
        cc_mass[c] = cc_mass.get(c, 0.0) + float(wi)
    top_cc_summary = [
        {"country": c, "country_name": cc_names.get(c, c), "mass": round(m, 4)}
        for c, m in sorted(cc_mass.items(), key=lambda x: -x[1])[:4]
    ]

    return {
        "clusters": clusters,
        "top_country_totals": top_cc_summary,
        "top_text_countries": top_text_countries,
    }


def _fmt_coord(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


def build_multiscale_cot_prompt(dist_info: Dict) -> str:
    """Prompt for Multi-Scale Spatial KDE -> CoT visual comparison, country + regional cluster
    selection, and continuous coordinate refinement.
    """
    clusters = dist_info["clusters"]
    lines = [
        "You are an expert GeoGuessr analyst performing multi-scale visual geolocalization.",
        "Below are the top dynamic spatial KDE clusters retrieved from a 25,000-image global StreetCLIP index "
        "(note: large countries may appear multiple times representing distinct regional clusters):",
    ]
    for c in clusters:
        lines.append(
            f"  {c['rank']}. {c['country_name']} ({c['country']}) at ({c['latitude']:.2f}, {c['longitude']:.2f}) "
            f"[{_fmt_coord(c['latitude'], c['longitude'])}] — cluster KDE: {100.0 * c['mass']:.1f}%, "
            f"{c['neighbors']} neighbors, max sim {c['max_sim']:.3f}, text-probe #{c['text_rank']}"
        )

    cc_tot = ", ".join(
        f"{x['country_name']} ({100.0 * x['mass']:.0f}%)" for x in dist_info["top_country_totals"]
    )
    txt_top = ", ".join(
        f"{x['country_name']} ({x['country']})" for x in dist_info["top_text_countries"]
    )
    lines.append(f"Country-level KDE totals: {cc_tot}.")
    lines.append(f"StreetCLIP text-probe top-3: {txt_top}.")
    lines.append("")
    lines.append(
        "Reason step by step across BOTH country scale and regional/coordinate scale:\n"
        "1. Visual clues: list concrete observable evidence (text/script on signs, road lines & center-line color, "
        "bollards, utility poles, license plates, driving side, architecture, soil, terrain, vegetation, climate).\n"
        "2. Multi-scale comparison: compare competing countries AND competing sub-regions within the same country "
        "(e.g., coastal vs inland, north vs south climate/terrain/architecture) against your visual clues.\n"
        "3. Decision & coordinate refinement: select the best spatial cluster (`selected_cluster` = 1.."
        f"{len(clusters)}, or 0 ONLY if unmistakable direct visual/text evidence contradicts all listed clusters). "
        "Refine `latitude` and `longitude` to the most accurate location within that region using all visual and textual cues; "
        "if the scene lacks distinguishing cues, stay with cluster #1 and set `confidence` to \"low\".\n"
        "Respond with ONLY one valid JSON object matching this exact schema (keep it concise so the JSON is complete):\n"
        '{"visual_clues": "<concise comma-separated observable clues, max 25 words>", '
        '"reasoning": "<2 concise sentences comparing competing countries/sub-regions and explaining coordinates, max 45 words>", '
        '"selected_cluster": <integer 0..'
        f"{len(clusters)}>, "
        '"country": "<2-letter ISO-3166-1 alpha-2 code>", '
        '"latitude": <decimal degrees>, "longitude": <decimal degrees>, '
        '"confidence": "<high|medium|low>"}'
    )
    return "\n".join(lines)


def build_star_rationalization_prompt(
    dist_info: Dict,
    true_cc: str,
    true_cc_name: str,
    true_lat: float,
    true_lon: float,
    target_cluster: int,
) -> str:
    """STaR hindsight rationalization prompt used ONLY on hard training examples where
    all blind rollouts failed. Instructs the model to ground reasoning strictly in real
    observable visual features (never inventing signs/text not visible in the image).
    """
    base = build_multiscale_cot_prompt(dist_info)
    hint = (
        f"\n\n[TRAINING RATIONALIZATION GUIDE — DO NOT MENTION THIS GUIDE IN YOUR OUTPUT]\n"
        f"Ground truth location: {true_cc_name} ({true_cc}) at ({true_lat:.3f}, {true_lon:.3f}) "
        f"[{_fmt_coord(true_lat, true_lon)}], best matching cluster={target_cluster}.\n"
        "Inspect the actual image carefully and write a concise, honest GeoGuessr reasoning trace explaining "
        "which REAL observable visual features in the photo (road style, markings, vegetation, terrain, climate, "
        "architecture, driving side, poles, or genuinely visible text) support this country and region over the "
        "competing candidates. NEVER invent or hallucinate text, signs, or license plates that are not clearly visible. "
        f"Set `selected_cluster` to {target_cluster}, `country` to \"{true_cc}\", and `latitude`/`longitude` close to "
        f"({true_lat:.3f}, {true_lon:.3f})."
    )
    return base + hint


def _validate_coords(lat, lon) -> Optional[Tuple[float, float]]:
    if isinstance(lat, bool) or isinstance(lon, bool) or lat is None or lon is None:
        return None
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None
    return (lat_f, lon_f)


def normalize_country_code(raw_cc, cc_names: Dict[str, str]) -> Optional[str]:
    if not isinstance(raw_cc, str):
        return None
    val = raw_cc.strip()
    if len(val) == 2 and val.isalpha():
        return val.upper()
    low = val.lower()
    for code, name in cc_names.items():
        if name.lower() == low:
            return code
    if low in ("uk", "great britain", "england", "united kingdom"):
        return "GB"
    if low in ("usa", "united states", "united states of america"):
        return "US"
    return None


def parse_multiscale_json(text: str, cc_names: Dict[str, str]) -> Dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    obj = None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch == "{":
            try:
                cand, _ = decoder.raw_decode(cleaned[i:])
                if isinstance(cand, dict) and (
                    "country" in cand or "latitude" in cand or "selected_cluster" in cand
                ):
                    obj = cand
                    break
            except ValueError:
                continue

    if obj is None:
        cc_m = re.search(r'"country"\s*:\s*"([A-Za-z]{2})"', cleaned)
        lat_m = re.search(r'"latitude"\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        lon_m = re.search(r'"longitude"\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        sel_m = re.search(r'"selected_cluster"\s*:\s*(\d+)', cleaned)
        conf_m = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', cleaned, re.I)
        coords = _validate_coords(lat_m.group(1), lon_m.group(1)) if (lat_m and lon_m) else None
        return {
            "parsed": coords is not None and cc_m is not None,
            "visual_clues": None,
            "reasoning": None,
            "selected_cluster": int(sel_m.group(1)) if sel_m else None,
            "country": cc_m.group(1).upper() if cc_m else None,
            "coordinates": coords,
            "confidence": conf_m.group(1).lower() if conf_m else "medium",
        }

    sel = obj.get("selected_cluster", obj.get("selected_candidate"))
    try:
        sel_int = int(sel) if sel is not None and not isinstance(sel, bool) else None
    except (TypeError, ValueError):
        sel_int = None

    conf = str(obj.get("confidence", "medium")).strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"

    coords = _validate_coords(obj.get("latitude"), obj.get("longitude"))
    cc = normalize_country_code(obj.get("country"), cc_names)
    return {
        "parsed": coords is not None and cc is not None,
        "visual_clues": obj.get("visual_clues"),
        "reasoning": obj.get("reasoning", obj.get("candidate_comparison")),
        "selected_cluster": sel_int,
        "country": cc,
        "coordinates": coords,
        "confidence": conf,
    }


def format_canonical_cot_json(
    visual_clues: str,
    reasoning: str,
    selected_cluster: int,
    country: str,
    latitude: float,
    longitude: float,
    confidence: str,
) -> str:
    """Serialize a clean, canonical JSON CoT target for SFT/DPO training."""
    obj = {
        "visual_clues": str(visual_clues or "").strip(),
        "reasoning": str(reasoning or "").strip(),
        "selected_cluster": int(selected_cluster),
        "country": str(country).upper(),
        "latitude": round(float(latitude), 3),
        "longitude": round(float(longitude), 3),
        "confidence": str(confidence).lower() if str(confidence).lower() in ("high", "medium", "low") else "medium",
    }
    return json.dumps(obj, ensure_ascii=False)


def build_shuffled_mcq_prompt(label: str, options_list, cc_names: Dict[str, str]) -> str:
    """Build the anti-cheat Isolated Local-Cue Shuffled Hard-Neighbor prompt (zero background, zero KDE)."""
    letters = "ABCDEF"
    lines = [
        f"You are inspecting ONLY a cropped local street-level clue (`{label}`) with all surrounding scenery removed.",
        "Which of these candidate countries best matches this local clue's physical style, design, or text?",
    ]
    for i, cc in enumerate(options_list):
        lines.append(f"  {letters[i]}. {cc_names.get(cc, cc)} ({cc})")
    lines.append(
        'Respond with ONLY one JSON object: {"option": "<letter>", "country": "<2-letter ISO code>"}'
    )
    return "\n".join(lines)


def build_direct_fovea_prompt(dist_info: Dict, crop_notes) -> str:
    """Clean, concise Image-First Visual Fovea prompt without verbose template prose."""
    base = build_multiscale_cot_prompt(dist_info)
    marker = "Reason step by step"
    header = base.split(marker, 1)[0].strip() if marker in base else base
    prefix = ""
    if crop_notes:
        prefix = (
            "Image 1 is the full street-level photo. Additional zoomed-in local clue crops are attached:\n"
            + "\n".join(f"  - {n}" for n in crop_notes)
            + "\nUse the zoomed physical details (architecture, roofs, fences, poles, guardrails, road markings, signs, vehicles) "
              "together with the full scene to select the best cluster and coordinates.\n\n"
        )
    n_cl = len(dist_info["clusters"])
    instr = (
        f"\nSelect the best cluster (`selected_cluster` = 1..{n_cl}, or 0 ONLY if direct visual evidence contradicts all clusters) "
        "and output accurate coordinates. If cues are insufficient to distinguish, stay with cluster #1 and set confidence to \"low\".\n"
        'Respond with ONLY one valid JSON object: '
        f'{{"selected_cluster": <0..{n_cl}>, "country": "<2-letter ISO>", '
        '"latitude": <decimal>, "longitude": <decimal>, "confidence": "<high|medium|low>"}'
    )
    return prefix + header + instr
