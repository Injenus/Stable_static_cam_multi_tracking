#!/usr/bin/env python3
"""Download and extract a MOTChallenge dataset (MOT16 or MOT20).

This is the "no-manual" bootstrap:
  1) download official ZIP from motchallenge.net
  2) extract into <out_dir>/<DATASET>/...

Examples:
  python scripts/download_motchallenge.py --dataset MOT20 --out data
  python scripts/download_motchallenge.py --dataset MOT16 --out data

Notes:
- MOT20 is ~5.0 GB; MOT16 is smaller.
- The script uses requests if available; otherwise falls back to urllib.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

DATASET_URLS = {
    # Official download endpoints referenced from MOTChallenge "Data" pages.
    "MOT16": "https://motchallenge.net/data/MOT16.zip",
    "MOT20": "https://motchallenge.net/data/MOT20.zip",
}


def _download_with_requests(url: str, dst: Path) -> None:
    import requests  # type: ignore

    dst.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (MOT downloader; +https://motchallenge.net)",
    }

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or "0")
        done = 0
        t0 = time.time()

        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)

                # progress
                if total > 0:
                    pct = 100.0 * done / total
                    dt = max(1e-6, time.time() - t0)
                    speed = done / dt / (1024 * 1024)
                    print(f"\r  {pct:6.2f}%  {done/1e6:8.1f}MB / {total/1e6:8.1f}MB  ({speed:5.1f} MB/s)", end="")

    if total > 0:
        print()


def _download_with_urllib(url: str, dst: Path) -> None:
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dst))


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 10 * 1024 * 1024:
        print(f"[download] exists: {dst} ({dst.stat().st_size/1e6:.1f} MB) -> skip")
        return

    print(f"[download] {url}")
    print(f"[download] -> {dst}")

    try:
        _download_with_requests(url, dst)
    except Exception as e:
        print(f"[download] requests failed ({e}); trying urllib...")
        _download_with_urllib(url, dst)


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[extract] {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Safety: avoid ZipSlip
        for member in zf.infolist():
            p = out_dir / member.filename
            if not str(p.resolve()).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_URLS.keys()), default="MOT20")
    ap.add_argument("--out", default="data", help="Output directory (will contain <DATASET>/)")
    ap.add_argument("--keep-zip", action="store_true", help="Keep the downloaded ZIP")
    args = ap.parse_args()

    dataset = str(args.dataset)
    url = DATASET_URLS[dataset]

    out_dir = Path(args.out).expanduser().resolve()
    zip_path = out_dir / f"{dataset}.zip"

    download(url, zip_path)

    # Extract into out_dir. Most MOT zips contain a top-level folder with the same name.
    extract(zip_path, out_dir)

    # Normalize: ensure out_dir/<dataset>/ exists.
    ds_dir = out_dir / dataset
    if not ds_dir.exists():
        # Some archives might unpack without the top-level folder.
        # If so, create it and move train/test there.
        train = out_dir / "train"
        test = out_dir / "test"
        if train.exists() or test.exists():
            ds_dir.mkdir(parents=True, exist_ok=True)
            if train.exists():
                train.rename(ds_dir / "train")
            if test.exists():
                test.rename(ds_dir / "test")

    if not ds_dir.exists():
        print(f"[warn] Cannot find {ds_dir}. Check archive structure.")
    else:
        print(f"[ok] dataset dir: {ds_dir}")

    if not args.keep_zip:
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
