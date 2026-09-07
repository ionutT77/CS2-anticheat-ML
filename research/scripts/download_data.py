"""Download the original Kaggle archive and record provenance without credentials."""
from pathlib import Path
import hashlib
import json
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REF = "emstatsl/csgo-cheating-dataset"
BASE = "https://www.kaggle.com/api/v1/datasets"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for endpoint, filename in [(f"view/{REF}", "kaggle_metadata.json"),
                               (f"list/{REF}", "kaggle_files.json")]:
        r = requests.get(f"{BASE}/{endpoint}", timeout=90)
        r.raise_for_status()
        try:
            metadata = r.json()
            (RAW / filename).write_text(json.dumps(metadata, indent=2))
            print(filename, json.dumps(metadata)[:18000], flush=True)
        except ValueError:
            print(filename, "non-JSON response", r.text[:300], flush=True)
    url = f"{BASE}/download/{REF}?datasetVersionNumber=1"
    archive = RAW / "csgo-cheating-dataset-v1.zip"
    if not archive.exists():
        temp = archive.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            print("Downloading", r.headers.get("content-length"), "bytes", flush=True)
            written = 0
            last = time.monotonic()
            with temp.open("wb") as f:
                for chunk in r.iter_content(4 * 1024 * 1024):
                    f.write(chunk)
                    written += len(chunk)
                    if time.monotonic() - last > 15:
                        print(f"Downloaded {written / 1e6:.1f} MB", flush=True)
                        last = time.monotonic()
        if not zipfile.is_zipfile(temp):
            raise RuntimeError("Download was not a ZIP archive")
        temp.replace(archive)
    checksum = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    dest = RAW / "extracted"
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        for info in entries:
            path = (dest / info.filename).resolve()
            if not path.is_relative_to(dest.resolve()):
                raise ValueError(f"Unsafe archive entry: {info.filename}")
        print("Archive entries", len(entries), "uncompressed bytes", sum(e.file_size for e in entries), flush=True)
        print("Sample filenames", z.namelist()[:30], flush=True)
        z.extractall(dest)
    manifest = {"dataset": REF, "version": 1, "source_url": url,
                "archive": str(archive.relative_to(ROOT)), "sha256": checksum,
                "archive_bytes": archive.stat().st_size,
                "files": [{"name": e.filename, "bytes": e.file_size, "crc32": e.CRC} for e in entries]}
    (RAW / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Complete", checksum, flush=True)


if __name__ == "__main__":
    main()
