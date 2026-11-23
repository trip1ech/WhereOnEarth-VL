"""Local road-marking annotation tool backed by the project's existing ZIPs."""
import argparse
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DIST = ROOT / "dist"
LABEL_FILE = PROJECT / "data/roadmark_annotations/labels.json"
LABELS = {"single", "double", "none", "unclear"}
DATASETS = (
    ("train", "Training Set", "data/qlora_since_2018/geoguessr_sft_6000_upload.zip", "train.jsonl"),
    ("holdout", "Unseen-Creator Holdout", "data/author_holdout/author-holdout-eval.zip", "eval.jsonl"),
)
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class AnnotationStore:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()
        self.datasets, self.photos = self.read_datasets()
        self.labels = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        for image_id, entry in self.labels.items():
            row = self.photos.get(image_id)
            if not row or entry["image_sha256"] != row["sha256"] or entry["label"] not in LABELS:
                raise ValueError(f"Existing annotation does not match image {image_id}")

    @staticmethod
    def read_datasets():
        datasets, photos = [], {}
        for dataset, name, relative, manifest in DATASETS:
            archive = PROJECT / relative
            with ZipFile(archive) as source:
                rows = [json.loads(line) for line in source.open(manifest)]
            rows.sort(key=lambda row: row["id"])
            random.Random(7818).shuffle(rows)
            ids = [row["id"] for row in rows]
            if len(set(ids)) != len(ids) or not photos.keys().isdisjoint(ids):
                raise ValueError(f"Duplicate image IDs in {dataset}")
            photos.update((row["id"], {
                "dataset": dataset, "archive": archive,
                "member": row["image"], "sha256": row["sha256"],
            }) for row in rows)
            datasets.append({"key": dataset, "name": name, "ids": ids})
        return datasets, photos

    def save(self, image_id, label):
        with self.lock:
            updated = self.labels.copy()
            if label is None:
                updated.pop(image_id, None)
            else:
                updated[image_id] = {
                    "image_id": image_id,
                    "dataset": self.photos[image_id]["dataset"],
                    "image_sha256": self.photos[image_id]["sha256"],
                    "label": label,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".labels-", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(updated, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.labels = updated


class AnnotationHandler(BaseHTTPRequestHandler):
    def send_data(self, payload, mime_type, status=200, *, download=None, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data, status=200):
        self.send_data(json.dumps(data, ensure_ascii=False).encode(), "application/json", status)

    def local_request(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        return host in allowed and (origin is None or origin == f"http://{host}")

    def do_GET(self):
        if not self.local_request():
            self.send_json({"error": "Please open from the local address."}, 403)
            return
        path = urlsplit(self.path).path
        store = self.server.store
        try:
            if path in STATIC:
                name, mime = STATIC[path]
                self.send_data((DIST / name).read_bytes(), mime)
            elif path == "/api/state":
                self.send_json({
                    "datasets": store.datasets,
                    "labels": {image_id: row["label"] for image_id, row in store.labels.items()},
                })
            elif path == "/api/export":
                payload = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in sorted(store.labels.values(), key=lambda entry: entry["image_id"]))
                self.send_data(payload.encode(), "application/x-ndjson; charset=utf-8",
                               download="road-marking-labels.jsonl")
            elif path.startswith("/photo/") and (photo := store.photos.get(path.removeprefix("/photo/"))):
                with ZipFile(photo["archive"]) as archive:
                    payload = archive.read(photo["member"])
                self.send_data(payload, "image/jpeg", cache=True)
            else:
                self.send_json({"error": "Not found."}, 404)
        except (BadZipFile, FileNotFoundError, KeyError):
            self.send_json({"error": "Unable to read data archives; check that local files are accessible."}, 503)

    def do_POST(self):
        if not self.local_request():
            self.send_json({"error": "Please open from the local address."}, 403)
            return
        if self.path != "/api/label":
            self.send_json({"error": "Not found."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 1024 or self.headers.get_content_type() != "application/json":
                raise ValueError("Invalid request")
            data = json.loads(self.rfile.read(length))
            image_id, label = data["image_id"], data["label"]
            if not isinstance(image_id, str) or image_id not in self.server.store.photos:
                raise ValueError("Invalid image")
            if label is not None and (not isinstance(label, str) or label not in LABELS):
                raise ValueError("Invalid label")
        except (KeyError, TypeError, ValueError):
            self.send_json({"error": "Invalid annotation payload."}, 400)
            return

        try:
            self.server.store.save(image_id, label)
        except OSError:
            self.send_json({"error": "Unable to write annotation file; label was not saved."}, 503)
            return
        self.send_json({"image_id": image_id, "label": label})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--labels", type=Path, default=LABEL_FILE)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), AnnotationHandler) as server:
        server.store = AnnotationStore(args.labels)
        print(f"Road marking annotation at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
