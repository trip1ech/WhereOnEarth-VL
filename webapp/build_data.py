"""Build the comparison data from the existing 300-image evaluation."""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

WEBAPP = Path(__file__).resolve().parent
PROJECT = WEBAPP.parent
ASSETS = WEBAPP / "dist" / "assets"
DATASET = PROJECT / "data/qlora_since_2018/geoguessr_sft_6000_upload.zip"
RESULTS = PROJECT / "results/qlora_6000/geoguessr-sft-6000-results.zip"
BASEMAP = PROJECT / "data/raw/ne_110m_admin_0_countries.geojson"
MODELS = ("zero_shot", "sft", "sft_2000")


def read_jsonl(archive, name):
    with archive.open(name) as rows:
        return [json.loads(line) for line in rows]


def write_js(filename, variable, data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (ASSETS / filename).write_text(f"window.{variable} = {payload};\n", encoding="utf-8")


def main():
    photos = ASSETS / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    examples = []
    with ZipFile(DATASET) as dataset, ZipFile(RESULTS) as results:
        rows = read_jsonl(dataset, "eval.jsonl")
        labels = read_jsonl(results, "eval_labels.jsonl")
        predictions = {
            model: read_jsonl(results, f"{model}.jsonl") for model in MODELS
        }
        ids = [row["id"] for row in rows]
        if len(ids) != 300 or len(set(ids)) != len(ids):
            raise ValueError("Expected 300 distinct evaluation images.")
        for name, records in {"eval_labels": labels, **predictions}.items():
            if [record["id"] for record in records] != ids:
                raise ValueError(f"Evaluation ID or order mismatch in {name}.")

        for index, (row, label) in enumerate(zip(rows, labels)):
            image_id = row["id"]
            for key in ("sha256", "country", "latitude", "longitude"):
                if row[key] != label[key]:
                    raise ValueError(f"Evaluation label mismatch for {image_id}: {key}.")
            photo = dataset.read(row["image"])
            if hashlib.sha256(photo).hexdigest() != row["sha256"]:
                raise ValueError(f"Image hash mismatch for {image_id}.")
            filename = f"{image_id}.jpg"
            photo_path = photos / filename
            if not photo_path.exists() or photo_path.read_bytes() != photo:
                photo_path.write_bytes(photo)

            guesses = {}
            for model, records in predictions.items():
                prediction = records[index]
                for key in ("country", "latitude", "longitude"):
                    if prediction[key] != row[key]:
                        raise ValueError(f"Truth mismatch for {image_id} in {model}: {key}.")
                guesses[model] = {
                    "country": prediction["guess_country"],
                    "point": prediction["guess_coordinates"],
                    "distance": prediction["distance_km"],
                }

            examples.append({
                "id": image_id,
                "photo": f"assets/photos/{filename}",
                "attribution": row["attribution"],
                "license": row["license"],
                "country": row["country"],
                "point": [row["latitude"], row["longitude"]],
                "guesses": guesses,
            })

        with results.open("metrics.csv") as file:
            metrics = {}
            for row in csv.DictReader(io.TextIOWrapper(file, encoding="utf-8")):
                model = row.pop("model")
                metrics[model] = {
                    key: float(value) if key == "median_km" else int(value)
                    for key, value in row.items()
                }

    write_js("data.js", "GEO_DATA", {"examples": examples, "metrics": metrics})
    basemap = json.loads(BASEMAP.read_text(encoding="utf-8"))
    features = [
        {"type": "Feature", "geometry": feature["geometry"], "properties": {}}
        for feature in basemap["features"]
        if feature["geometry"]
    ]
    write_js("basemap.js", "GEO_BASEMAP", {
        "type": "FeatureCollection",
        "features": features,
    })
    print(f"Built comparison for {len(examples)} verified evaluation images.")


if __name__ == "__main__":
    main()
