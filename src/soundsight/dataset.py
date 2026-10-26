"""Official dataset acquisition and strict metadata validation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
import time

REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
URL = f"https://codeload.github.com/karolpiczak/ESC-50/zip/{REVISION}"


def download(destination, individual=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if individual:
        base = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{REVISION}/"
        def fetch(relative):
            out = destination / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(base + relative, timeout=90) as src:
                        data = src.read()
                    out.write_bytes(data)
                    return
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        fetch("meta/esc50.csv")
        with (destination / "meta/esc50.csv").open(newline="") as f:
            names = [r["filename"] for r in csv.DictReader(f)]
        if len(names) != 2000 or any(Path(n).name != n or "/" in n or "\\" in n for n in names):
            raise ValueError("Invalid upstream audio filenames.")
        print("Downloading 2000 official audio files (8 workers)...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, _ in enumerate(pool.map(fetch, ["audio/" + n for n in names]), 1):
                if i % 100 == 0:
                    print(f"Downloaded {i}/2000", flush=True)
        records(destination)
        (destination / "provenance.json").write_text(json.dumps({"base_url": base, "revision": REVISION, "method": "individual files; content hashes recorded during training"}, indent=2))
        return
    archive = destination.parent / "esc50-download.zip"
    print("Downloading official ESC-50 archive (~600 MB)...", flush=True)
    with urllib.request.urlopen(URL, timeout=90) as src, archive.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with zipfile.ZipFile(archive) as z:
        # Copy only expected dataset members; never extract arbitrary archive paths.
        for info in z.infolist():
            parts = Path(info.filename).parts
            if len(parts) == 3 and parts[1] in ("audio", "meta") and not info.is_dir():
                out = destination / parts[1] / parts[2]
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    records(destination)
    (destination / "provenance.json").write_text(json.dumps({"url": URL, "archive_sha256": digest}, indent=2))
    archive.unlink()
    print(f"Validated ESC-50 at {destination}", flush=True)


def records(root):
    root = Path(root)
    with (root / "meta" / "esc50.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2000:
        raise ValueError("Expected the full ESC-50 metadata (2000 rows).")
    names, labels, counts = set(), {}, {}
    for row in rows:
        filename = row["filename"]
        target, fold = int(row["target"]), int(row["fold"])
        if Path(filename).name != filename or filename in names or not (root / "audio" / filename).is_file():
            raise ValueError(f"Missing, duplicate or unsafe audio filename: {filename}")
        if target not in range(50) or fold not in range(1, 6):
            raise ValueError("Invalid target or fold.")
        if labels.setdefault(target, row["category"]) != row["category"]:
            raise ValueError("Inconsistent class labels.")
        counts[target, fold] = counts.get((target, fold), 0) + 1
        names.add(filename)
        row["target"], row["fold"] = target, fold
    if len(counts) != 250 or set(counts.values()) != {8}:
        raise ValueError("Expected eight examples per class per official fold.")
    return rows
