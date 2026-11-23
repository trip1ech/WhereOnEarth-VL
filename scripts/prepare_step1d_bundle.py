"""Prepare Colab upload bundle for Step 1D: geolocal/StreetCLIP ceiling-raising benchmark
(6,000 train gallery + 300 eval + 66 unseen-creator holdout, full image vs bottom-12% HUD-cropped).
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path("/Volumes/PortableSSD/codex/GeoGuessr-Recognition")
SFT_6K_ZIP = ROOT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
AH_ZIP = ROOT / "data/author_holdout/author-holdout-eval.zip"
OUT = ROOT / "results/retrieval_v2"
TRANSFER = OUT / "transfer_1d"

COUNTRY_NAMES = {
    "AE": "United Arab Emirates", "AF": "Afghanistan", "AG": "Antigua and Barbuda", "AI": "Anguilla",
    "AL": "Albania", "AM": "Armenia", "AO": "Angola", "AR": "Argentina", "AT": "Austria", "AU": "Australia",
    "AX": "Aland Islands", "AZ": "Azerbaijan", "BA": "Bosnia and Herzegovina", "BB": "Barbados",
    "BD": "Bangladesh", "BE": "Belgium", "BF": "Burkina Faso", "BG": "Bulgaria", "BH": "Bahrain",
    "BI": "Burundi", "BJ": "Benin", "BN": "Brunei", "BO": "Bolivia", "BQ": "Bonaire", "BR": "Brazil",
    "BS": "Bahamas", "BT": "Bhutan", "BW": "Botswana", "BY": "Belarus", "BZ": "Belize", "CA": "Canada",
    "CD": "Democratic Republic of the Congo", "CF": "Central African Republic", "CG": "Republic of the Congo",
    "CH": "Switzerland", "CI": "Ivory Coast", "CL": "Chile", "CM": "Cameroon", "CN": "China",
    "CO": "Colombia", "CR": "Costa Rica", "CU": "Cuba", "CV": "Cape Verde", "CW": "Curacao",
    "CX": "Christmas Island", "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany", "DJ": "Djibouti",
    "DK": "Denmark", "DM": "Dominica", "DO": "Dominican Republic", "DZ": "Algeria", "EC": "Ecuador",
    "EE": "Estonia", "EG": "Egypt", "ER": "Eritrea", "ES": "Spain", "ET": "Ethiopia", "FI": "Finland",
    "FJ": "Fiji", "FK": "Falkland Islands", "FM": "Micronesia", "FO": "Faroe Islands", "FR": "France",
    "GA": "Gabon", "GB": "United Kingdom", "GD": "Grenada", "GE": "Georgia", "GF": "French Guiana",
    "GG": "Guernsey", "GH": "Ghana", "GI": "Gibraltar", "GL": "Greenland", "GM": "Gambia", "GN": "Guinea",
    "GP": "Guadeloupe", "GR": "Greece", "GT": "Guatemala", "GW": "Guinea-Bissau", "GY": "Guyana",
    "HK": "Hong Kong", "HN": "Honduras", "HR": "Croatia", "HT": "Haiti", "HU": "Hungary", "ID": "Indonesia",
    "IE": "Ireland", "IL": "Israel", "IM": "Isle of Man", "IN": "India", "IQ": "Iraq", "IR": "Iran",
    "IS": "Iceland", "IT": "Italy", "JE": "Jersey", "JM": "Jamaica", "JO": "Jordan", "JP": "Japan",
    "KE": "Kenya", "KG": "Kyrgyzstan", "KH": "Cambodia", "KI": "Kiribati", "KM": "Comoros",
    "KN": "Saint Kitts and Nevis", "KR": "South Korea", "KW": "Kuwait", "KY": "Cayman Islands", "KZ": "Kazakhstan", "LA": "Laos",
    "LB": "Lebanon", "LC": "Saint Lucia", "LK": "Sri Lanka", "LR": "Liberia", "LS": "Lesotho",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya", "MA": "Morocco", "MD": "Moldova",
    "ME": "Montenegro", "MF": "Saint Martin", "MG": "Madagascar", "MK": "North Macedonia", "ML": "Mali",
    "MM": "Myanmar", "MN": "Mongolia", "MQ": "Martinique", "MR": "Mauritania", "MT": "Malta",
    "MU": "Mauritius", "MV": "Maldives", "MW": "Malawi", "MX": "Mexico", "MY": "Malaysia",
    "MZ": "Mozambique", "NC": "New Caledonia", "NE": "Niger", "NG": "Nigeria", "NI": "Nicaragua",
    "NL": "Netherlands", "NO": "Norway", "NP": "Nepal", "NZ": "New Zealand", "OM": "Oman", "PA": "Panama",
    "PE": "Peru", "PF": "French Polynesia", "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan",
    "PL": "Poland", "PR": "Puerto Rico", "PS": "Palestine", "PT": "Portugal", "PW": "Palau",
    "PY": "Paraguay", "QA": "Qatar", "RE": "Reunion", "RO": "Romania", "RS": "Serbia", "RU": "Russia",
    "RW": "Rwanda", "SA": "Saudi Arabia", "SB": "Solomon Islands", "SD": "Sudan", "SE": "Sweden",
    "SG": "Singapore", "SI": "Slovenia", "SJ": "Svalbard", "SK": "Slovakia", "SL": "Sierra Leone",
    "SM": "San Marino", "SN": "Senegal", "SO": "Somalia", "SR": "Suriname", "SS": "South Sudan", "ST": "Sao Tome and Principe",
    "SV": "El Salvador", "SX": "Sint Maarten", "SY": "Syria", "SZ": "Eswatini",
    "TC": "Turks and Caicos Islands", "TD": "Chad", "TG": "Togo", "TH": "Thailand", "TJ": "Tajikistan",
    "TL": "Timor-Leste", "TM": "Turkmenistan", "TN": "Tunisia", "TR": "Turkey", "TT": "Trinidad and Tobago",
    "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda", "US": "United States", "UY": "Uruguay",
    "UZ": "Uzbekistan", "VA": "Vatican City", "VC": "Saint Vincent and the Grenadines", "VE": "Venezuela",
    "VG": "British Virgin Islands", "VN": "Vietnam", "VU": "Vanuatu", "WS": "Samoa", "XK": "Kosovo",
    "YE": "Yemen", "YT": "Mayotte", "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if TRANSFER.exists():
        shutil.rmtree(TRANSFER)
    TRANSFER.mkdir(parents=True)

    # Bundle code + author_holdout + country_names into a tiny zip, and split SFT_6K_ZIP separately
    code_zip = TRANSFER / "step1d_code.zip"
    with ZipFile(code_zip, "w", compression=ZIP_DEFLATED) as z:
        z.write(ROOT / "scripts/phase1.py", "phase1.py")
        z.write(ROOT / "scripts/benchmark_streetclip.py", "benchmark_streetclip.py")
        z.write(AH_ZIP, "author-holdout-eval.zip", compress_type=ZIP_STORED)
        z.writestr("country_names.json", json.dumps(COUNTRY_NAMES, indent=2) + "\n")

    sft_sha = hashlib.sha256(SFT_6K_ZIP.read_bytes()).hexdigest()
    code_sha = hashlib.sha256(code_zip.read_bytes()).hexdigest()
    subprocess.run(["split", "-b", "6m", "-a", "2", str(SFT_6K_ZIP), str(TRANSFER / "sft-part-")], check=True)
    parts = sorted(TRANSFER.glob("sft-part-*"))
    print(f"Prepared Step 1D: code_zip={code_zip.stat().st_size / 1e6:.2f} MB, sft parts={len(parts)} (sha={sft_sha[:12]}...)", flush=True)

    drive_py = f"""import hashlib, shutil, subprocess, sys
from pathlib import Path
from zipfile import ZipFile

root = Path("/content/step1d")
root.mkdir(exist_ok=True)
code_zip = Path("/content/step1d_code.zip")
if hashlib.sha256(code_zip.read_bytes()).hexdigest() != "{code_sha}":
    raise RuntimeError("code_zip sha mismatch")
with ZipFile(code_zip) as z:
    z.extractall(root)

sft_zip = root / "sft_6000.zip"
parts = sorted(Path("/content").glob("sft1d-sft-part-*"))
with sft_zip.open("wb") as dst:
    for p in parts:
        with p.open("rb") as src:
            shutil.copyfileobj(src, dst)
actual = hashlib.sha256(sft_zip.read_bytes()).hexdigest()
if actual != "{sft_sha}":
    raise RuntimeError(f"sft_6000 sha mismatch: {{actual}} != {sft_sha}")
print("All inputs verified. Running StreetCLIP encoding (full + bottom-12% HUD crop)...", flush=True)

res = root / "results"
res.mkdir(exist_ok=True)
cmd = [sys.executable, "-u", "benchmark_streetclip.py", str(sft_zip),
       str(root / "author-holdout-eval.zip"), str(root / "country_names.json"), str(res)]
with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, errors="replace", bufsize=1) as proc:
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, cmd)
"""
    (TRANSFER / "drive_colab.py").write_text(drive_py)

    run_sh = """#!/bin/bash
set -euo pipefail
cli="/Users/ching-chichou/.local/bin/colab"
name="geoguessr-retv2-1d-20261123"
target="/Volumes/PortableSSD/codex/GeoGuessr-Recognition/results/retrieval_v2"
cleanup() {
    "$cli" stop --session "$name" || "$cli" sessions
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$cli" new --session "$name" --gpu T4
"$cli" upload --session "$name" "$target/transfer_1d/step1d_code.zip" "/content/step1d_code.zip"
for part in "$target"/transfer_1d/sft-part-*; do
    "$cli" upload --session "$name" "$part" "/content/sft1d-${part##*/}"
done
"$cli" exec --session "$name" --timeout 900 --file "$target/transfer_1d/drive_colab.py"
"$cli" download --session "$name" /content/step1d/results/streetclip_features.npz "$target/streetclip_features.npz"
"""
    sh_path = TRANSFER / "run.sh"
    sh_path.write_text(run_sh)
    sh_path.chmod(0o755)
    print(f"Ready: {sh_path}", flush=True)


if __name__ == "__main__":
    main()
