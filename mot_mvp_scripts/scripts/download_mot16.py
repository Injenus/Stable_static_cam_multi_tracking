#!/usr/bin/env python3
"""Download and extract MOT16 from MOTChallenge.

Goal: zero manual steps besides running this script.

It downloads the official archive (MOT16.zip) and extracts it under:
  <out_dir>/MOT16/

Expected structure after extraction:
  <out_dir>/MOT16/train/MOT16-02/img1/000001.jpg
  <out_dir>/MOT16/train/MOT16-02/gt/gt.txt

Usage:
  python scripts/download_mot16.py --out data

Tip: if you only need GT (no images), MOTChallenge also provides a tiny "no img" archive.
This script focuses on the full dataset because you need frames for running your tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request


DEFAULT_URL = "https://motchallenge.net/data/MOT16.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_stream(url: str, out_path: Path, *, user_agent: str = "Mozilla/5.0") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": user_agent})

    with urlopen(req) as resp:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total is not None else None

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        done = 0
        with tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total_n:
                    pct = 100.0 * done / total_n
                    sys.stdout.write(f"\rDownloaded {done/1e6:.1f}MB / {total_n/1e6:.1f}MB ({pct:.1f}%)")
                else:
                    sys.stdout.write(f"\rDownloaded {done/1e6:.1f}MB")
                sys.stdout.flush()

        sys.stdout.write("\n")
        tmp_path.replace(out_path)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and extract MOT16 dataset")
    ap.add_argument("--out", default="data", help="Output directory (default: data)")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Dataset URL (default: {DEFAULT_URL})")
    ap.add_argument("--skip-download", action="store_true", help="Skip downloading if zip already exists")
    ap.add_argument("--zip-name", default="MOT16.zip", help="Zip filename inside --out (default: MOT16.zip)")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    zip_path = out_dir / args.zip_name

    if zip_path.exists() and not args.skip_download:
        print(f"Zip already exists: {zip_path}\nUse --skip-download to reuse it, or delete it to re-download.")
        return

    if not zip_path.exists():
        print(f"Downloading: {args.url}")
        download_stream(args.url, zip_path)
    else:
        print(f"Using existing zip: {zip_path}")

    # Extract into <out>/MOT16/
    mot_root = out_dir / "MOT16"
    if mot_root.exists() and any(mot_root.iterdir()):
        print(f"Looks like already extracted: {mot_root} (non-empty). Skipping extraction.")
        return

    print(f"Extracting to: {mot_root}")
    extract_zip(zip_path, mot_root)

    # Basic sanity check
    expected = mot_root / "train" / "MOT16-02" / "img1"
    if expected.exists():
        print("OK: dataset looks extracted.")
        print(f"MOT root: {mot_root}")
    else:
        print("WARNING: extraction finished, but expected folder not found:")
        print(f"  {expected}")
        print("Check archive contents / download integrity.")


if __name__ == "__main__":
    main()
