#!/usr/bin/env python3
"""One-command MVP experiments runner.

What it does:
- runs your tracker on a small set of MOTChallenge sequences (train split)
- evaluates each run with motmetrics
- writes a CSV summary

Default is tailored for "static camera / surveillance" => MOT20 train (4 sequences).
You can switch to MOT16 and select 3-5 sequences manually.

Example:
  python scripts/run_experiments_mvp.py --data data/MOT20 --dataset MOT20 --yolo jameslahm/yolov10m

Requires:
  pip install motmetrics pandas numpy

"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset folder, e.g. data/MOT20")
    ap.add_argument("--dataset", default=None, help="Dataset name for labeling only (MOT20/MOT16)")
    ap.add_argument("--split", default="train", choices=["train", "test"], help="Use train for GT evaluation")
    ap.add_argument("--yolo", default="jameslahm/yolov10m", help="Ultralytics model reference")
    ap.add_argument("--osnet", default=None, help="Path to OSNet weights (optional)")
    ap.add_argument("--max-frames", type=int, default=None, help="Debug: limit frames per sequence")
    ap.add_argument("--out", default="results", help="Base output folder")
    args = ap.parse_args()

    data = Path(args.data)
    dataset = args.dataset or data.name
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    # A tiny ablation set (MVP):
    #  A) baseline: no ReID, default KF R scaling
    #  B) no ReID, disable KF R scaling (R_scale_min=R_scale_max=1)
    #  C) ReID on (only if --osnet exists)
    variants: list[dict] = [
        {
            "name": "A_baseline",
            "no_reid": True,
            "extra": [],
        },
        {
            "name": "B_noRscale",
            "no_reid": True,
            "extra": ["--kf-r-min", "1.0", "--kf-r-max", "1.0"],
        },
    ]

    if args.osnet:
        variants.append(
            {
                "name": "C_reid",
                "no_reid": False,
                "extra": [],
            }
        )

    summary_rows: list[dict[str, str]] = []

    for v in variants:
        tag = f"{dataset}_{v['name']}"
        res_dir = out_base / tag
        res_dir.mkdir(parents=True, exist_ok=True)

        run_cmd = [
            "python",
            "scripts/run_tracker_motchallenge.py",
            "--data",
            str(data),
            "--split",
            args.split,
            "--out",
            str(res_dir),
            "--yolo",
            args.yolo,
        ]
        if args.max_frames is not None:
            run_cmd += ["--max-frames", str(args.max_frames)]
        if v["no_reid"]:
            run_cmd += ["--no-reid"]
        if args.osnet and (not v["no_reid"]):
            run_cmd += ["--osnet", args.osnet]
        run_cmd += v["extra"]

        run(run_cmd)

        eval_cmd = [
            "python",
            "scripts/eval_motchallenge.py",
            "--data",
            str(data),
            "--split",
            args.split,
            "--res",
            str(res_dir),
            "--out-csv",
            str(res_dir / "metrics.csv"),
        ]
        run(eval_cmd)

        # Read back the "OVERALL" row from metrics.csv
        overall = None
        with open(res_dir / "metrics.csv", "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Sequence") == "OVERALL":
                    overall = row
                    break
        if overall is None:
            raise RuntimeError(f"No OVERALL row produced for {tag}")

        overall = dict(overall)
        overall["Variant"] = v["name"]
        overall["Dataset"] = dataset
        summary_rows.append(overall)

    # Write combined summary
    out_csv = out_base / f"summary_{dataset}.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    print(f"\n[OK] Summary: {out_csv}")


if __name__ == "__main__":
    main()
