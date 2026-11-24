"""Shared utilities for building 25k StreetCLIP spatial KDE distributions,
formatting prompts for Qwen3-VL-8B-Instruct reranking & coordinate refinement,
and robustly parsing structured JSON responses.
"""
import json
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(0.0, a))))


def compute_kde_log_scores(
    top_idx: np.ndarray,
    top_sim: np.ndarray,
    g_lat: np.ndarray,
    g_lon: np.ndarray,
    tau: float = 0.03,
    sigma: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute log spatial KDE score and pairwise Haversine distances for Top-K gallery items."""
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


def build_kde_distribution(
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
    max_countries: int = 6,
) -> Dict:
    """Summarize the Top-100 StreetCLIP GeoProj + KDE + Text-Probe distribution into
    Top-K distinct country hypotheses with posterior shares, peak coordinates,
    optional secondary spatial modes (for large countries), and text-probe summary.
    """
    cc_to_i = {c: i for i, c in enumerate(cc_vocab)}
    ccs = [str(g_cc[i]) for i in top_idx_1]
    tlp = np.array(
        [txt_lp_1[cc_to_i[c]] if c in cc_to_i else -15.0 for c in ccs],
        dtype=np.float64,
    )
    comb = kde_1.astype(np.float64) + wt * tlp

    # Posterior weights across all 100 retrieved neighbors
    w = np.exp(comb - comb.max())
    w /= w.sum()

    # StreetCLIP zero-shot text-probe top countries (rescaled to readable percentages)
    txt_probs = np.exp(txt_lp_1.astype(np.float64))
    txt_order = np.argsort(-txt_probs)
    txt_rank_map = {cc_vocab[int(idx)]: rank + 1 for rank, idx in enumerate(txt_order)}
    top_text_countries = [
        {
            "country": cc_vocab[int(idx)],
            "country_name": cc_names.get(cc_vocab[int(idx)], cc_vocab[int(idx)]),
            "prob": round(float(txt_probs[int(idx)]), 4),
        }
        for idx in txt_order[:3]
    ]

    # Group Top-100 by country; rank by peak combined score so Candidate #1 == exact baseline Top-1
    by_cc: Dict[str, List[int]] = {}
    for j, c in enumerate(ccs):
        by_cc.setdefault(c, []).append(j)

    cc_entries = []
    for c, idxs in by_cc.items():
        idxs_arr = np.array(idxs, dtype=np.int32)
        best_pos = int(idxs_arr[int(np.argmax(comb[idxs_arr]))])
        best_g = int(top_idx_1[best_pos])
        total_mass = float(w[idxs_arr].sum())

        # Local narrow KDE (sigma=50 km) around the primary mode (within 400 km)
        near_primary = idxs_arr[pw_d_1[best_pos, idxs_arr] <= 400.0]
        m_d = pw_d_1[np.ix_(near_primary, near_primary)]
        m_w = np.exp((top_sim_1[near_primary] - top_sim_1[near_primary].max()) / 0.03)
        local_kde = (m_w[None, :] * np.exp(-m_d / 50.0)).sum(axis=1)
        narrow_g = int(top_idx_1[int(near_primary[int(np.argmax(local_kde))])])

        # Check for a distinct secondary regional mode > 450 km away within the same country
        far_idxs = idxs_arr[pw_d_1[best_pos, idxs_arr] > 450.0]
        secondary = None
        if len(far_idxs) >= 2:
            sec_mass = float(w[far_idxs].sum())
            if sec_mass >= 0.15 * total_mass and sec_mass >= 0.02:
                sec_pos = int(far_idxs[int(np.argmax(comb[far_idxs]))])
                sec_g = int(top_idx_1[sec_pos])
                secondary = {
                    "latitude": round(float(g_lat[sec_g]), 3),
                    "longitude": round(float(g_lon[sec_g]), 3),
                    "mass": round(sec_mass, 4),
                    "neighbors": int(len(far_idxs)),
                }

        cc_entries.append({
            "country": c,
            "country_name": cc_names.get(c, c),
            "peak_comb": float(comb[best_pos]),
            "kde_score": round(float(kde_1[best_pos]), 3),
            "mass": round(total_mass, 4),
            "neighbors": int(len(idxs_arr)),
            "max_sim": round(float(top_sim_1[idxs_arr].max()), 3),
            "text_rank": int(txt_rank_map.get(c, 999)),
            "latitude": round(float(g_lat[best_g]), 4),
            "longitude": round(float(g_lon[best_g]), 4),
            "narrow_latitude": round(float(g_lat[narrow_g]), 4),
            "narrow_longitude": round(float(g_lon[narrow_g]), 4),
            "secondary_region": secondary,
        })

    # Sort countries by peak combined score so Candidate #1 matches the exact StreetCLIP 25k GeoProj+KDE+Text baseline.
    cc_entries.sort(key=lambda x: -x["peak_comb"])
    candidates = []
    for rank, entry in enumerate(cc_entries[:max_countries], start=1):
        item = dict(entry)
        item["rank"] = rank
        candidates.append(item)

    # Entropy of country-level KDE distribution (normalized 0..1)
    masses = np.array([e["mass"] for e in cc_entries if e["mass"] > 1e-9], dtype=np.float64)
    entropy = float(-np.sum(masses * np.log(masses))) if len(masses) > 1 else 0.0

    return {
        "candidates": candidates,
        "top_text_countries": top_text_countries,
        "top1_mass": candidates[0]["mass"] if candidates else 0.0,
        "top1_max_sim": candidates[0]["max_sim"] if candidates else 0.0,
        "country_entropy": round(entropy, 4),
        "num_countries_in_top100": len(cc_entries),
    }


def _fmt_coord(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


ZERO_SHOT_PROMPT = (
    "You are an expert GeoGuessr analyst. Determine where this street-level photograph was taken "
    "from observable visual cues (language/script on signs, road markings, bollards, utility poles, "
    "license plates, driving side, architecture, terrain, vegetation, and sun angle).\n"
    "Respond with ONLY one valid JSON object matching this exact schema:\n"
    '{"visual_clues": "<short concrete visual observations>", '
    '"country": "<2-letter ISO-3166-1 alpha-2 code>", '
    '"latitude": <decimal degrees>, "longitude": <decimal degrees>, '
    '"confidence": "<high|medium|low>"}'
)


def build_kde_rerank_prompt(dist_info: Dict) -> str:
    """Format the 25k StreetCLIP spatial KDE distribution into a clear, compact prompt
    for VLM visual comparison, candidate reranking, and coordinate refinement.
    """
    lines = [
        "You are an expert GeoGuessr analyst performing retrieval-augmented visual geolocalization.",
        "Below is the spatial KDE posterior distribution from a 25,000-image global StreetCLIP retrieval index:",
    ]
    for c in dist_info["candidates"]:
        coord_str = _fmt_coord(c["latitude"], c["longitude"])
        base = (
            f"  {c['rank']}. {c['country_name']} ({c['country']}) — "
            f"KDE weight: {100.0 * c['mass']:.1f}%, "
            f"peak at ({c['latitude']:.2f}, {c['longitude']:.2f}) [{coord_str}], "
            f"{c['neighbors']} neighbors, max sim {c['max_sim']:.3f}, text-probe rank #{c['text_rank']}"
        )
        sec = c.get("secondary_region")
        if sec is not None:
            base += (
                f"; secondary cluster at ({sec['latitude']:.2f}, {sec['longitude']:.2f}) "
                f"({100.0 * sec['mass']:.1f}%, {sec['neighbors']} neighbors)"
            )
        lines.append(base)

    txt_summary = ", ".join(
        f"{t['country_name']} ({t['country']})" for t in dist_info["top_text_countries"]
    )
    lines.append(f"StreetCLIP zero-shot text-probe top-3: {txt_summary}.")
    lines.append("")
    lines.append(
        "Instructions:\n"
        "1. Inspect the photo for concrete geographic clues: readable text/script, road markings & center-line color, "
        "traffic signs, bollards, utility poles, license plates, driving side, vehicle models, architecture, soil, and vegetation.\n"
        "2. Compare the candidate countries/regions above against these visual clues. Pay special attention to distinguishing "
        "nearby or visually similar candidates (e.g., neighboring countries with different scripts, sign styles, road markings, or driving side).\n"
        "3. Choose the best candidate (`selected_candidate` = 1.."
        f"{len(dist_info['candidates'])}). Use `selected_candidate` = 0 ONLY if unmistakable direct evidence "
        "(such as a clear readable place name, script, flag, or opposite driving side) strongly contradicts ALL listed candidates.\n"
        "4. Calibration rule: the StreetCLIP KDE retriever is strong on general landscape/road texture. "
        "If visual clues are weak or ambiguous, stay with candidate #1 and set `confidence` to \"low\". "
        "Override candidate #1 only when concrete visual features clearly favor another candidate.\n"
        "5. Output your final refined coordinates (`latitude`, `longitude`): anchor near the chosen candidate's peak coordinates "
        "(or its secondary cluster if that region matches better), and refine them further if signs, terrain, or urban cues pinpoint a more specific area.\n"
        "Respond with ONLY one valid JSON object matching this exact schema:\n"
        '{"visual_clues": "<concise observable clues>", '
        '"candidate_comparison": "<concise comparison of top candidates against the clues>", '
        '"selected_candidate": <integer 0..'
        f"{len(dist_info['candidates'])}>, "
        '"country": "<2-letter ISO-3166-1 alpha-2 code>", '
        '"latitude": <decimal degrees>, "longitude": <decimal degrees>, '
        '"confidence": "<high|medium|low>"}'
    )
    return "\n".join(lines)


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


def parse_vlm_json(text: str, cc_names: Dict[str, str]) -> Dict:
    """Extract structured fields from VLM text output (robust to code fences & extra prose)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    obj = None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch == "{":
            try:
                candidate, _ = decoder.raw_decode(cleaned[i:])
                if isinstance(candidate, dict) and (
                    "country" in candidate or "latitude" in candidate or "selected_candidate" in candidate
                ):
                    obj = candidate
                    break
            except ValueError:
                continue

    if obj is None:
        # Regex fallback for truncated outputs
        cc_m = re.search(r'"country"\s*:\s*"([A-Za-z]{2})"', cleaned)
        lat_m = re.search(r'"latitude"\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        lon_m = re.search(r'"longitude"\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
        sel_m = re.search(r'"selected_candidate"\s*:\s*(\d+)', cleaned)
        conf_m = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', cleaned, re.I)
        coords = _validate_coords(lat_m.group(1), lon_m.group(1)) if (lat_m and lon_m) else None
        return {
            "parsed": coords is not None or cc_m is not None or sel_m is not None,
            "visual_clues": None,
            "candidate_comparison": None,
            "selected_candidate": int(sel_m.group(1)) if sel_m else None,
            "country": cc_m.group(1).upper() if cc_m else None,
            "coordinates": coords,
            "confidence": conf_m.group(1).lower() if conf_m else "medium",
        }

    sel = obj.get("selected_candidate")
    try:
        sel_int = int(sel) if sel is not None and not isinstance(sel, bool) else None
    except (TypeError, ValueError):
        sel_int = None

    conf = str(obj.get("confidence", "medium")).strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"

    return {
        "parsed": True,
        "visual_clues": obj.get("visual_clues"),
        "candidate_comparison": obj.get("candidate_comparison"),
        "selected_candidate": sel_int,
        "country": normalize_country_code(obj.get("country"), cc_names),
        "coordinates": _validate_coords(obj.get("latitude"), obj.get("longitude")),
        "confidence": conf,
    }
