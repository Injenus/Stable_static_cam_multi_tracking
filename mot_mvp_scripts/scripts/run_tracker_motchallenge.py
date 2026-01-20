#!/usr/bin/env python3
"""Run your YOLO+Kalman+ByteTrack(-ish)+optional ReID tracker on MOTChallenge image sequences.

Outputs per-sequence tracker results in MOTChallenge text format:
  <out_dir>/<seq_name>.txt
Each line:
  frame, id, x, y, w, h, score, -1, -1, -1

This script DOES NOT require any manual annotation.
It assumes ground truth already exists in the dataset (train split).

Example (MOT20 train, 4 sequences):
  python scripts/run_tracker_motchallenge.py \
    --tracker ./object_tracker_pipeline.py \
    --data ./data/MOT20 --split train \
    --out ./results/mot20_baseline \
    --yolo jameslahm/yolov10m \
    --seq MOT20-01 MOT20-02 MOT20-03 MOT20-05 \
    --pred-k 1

Notes:
- --pred-k controls whether you output *predicted* boxes for short detection gaps.
  pred-k=0: only tracks updated on the current frame
  pred-k=1: also output tracks missed for 1 frame (KF prediction)

"""

from __future__ import annotations

import argparse
import configparser
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib

import numpy as np
import cv2

def _load_module_from_path(py_path: Path):
    

    # уникальное имя, чтобы не конфликтовать с другими импортами
    mod_name = "tracker_" + hashlib.md5(str(py_path).encode("utf-8")).hexdigest()

    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {py_path}")

    mod = importlib.util.module_from_spec(spec)

    # КРИТИЧНО для Python 3.12 + dataclasses:
    # dataclasses ищет sys.modules[cls.__module__]
    sys.modules[mod_name] = mod

    spec.loader.exec_module(mod)  # type: ignore
    return mod



def _read_seqinfo(seq_dir: Path) -> Dict[str, object]:
    ini = seq_dir / "seqinfo.ini"
    if not ini.exists():
        raise FileNotFoundError(f"Missing seqinfo.ini: {ini}")

    cp = configparser.ConfigParser()
    cp.read(str(ini))
    if "Sequence" not in cp:
        raise RuntimeError(f"Invalid seqinfo.ini (no [Sequence]): {ini}")

    s = cp["Sequence"]
    img_dir = s.get("imDir", "img1")
    frame_rate = float(s.get("frameRate", 25))
    seq_len = int(s.get("seqLength", 0))

    # img width/height are optional for reading; we infer from the first frame.
    return {
        "img_dir": img_dir,
        "fps": frame_rate,
        "seq_len": seq_len,
    }


def _iter_frames(seq_dir: Path, img_dir_name: str, seq_len: int) -> Iterable[Tuple[int, np.ndarray]]:
    img_dir = seq_dir / img_dir_name
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")

    # MOTChallenge frame index starts from 1; typical naming 000001.jpg
    for frame_id in range(1, seq_len + 1):
        img_path = img_dir / f"{frame_id:06d}.jpg"
        if not img_path.exists():
            # Some sequences may be shorter than seqLength; stop safely.
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        yield frame_id, img


def _clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    # ensure valid
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _xyxy_to_xywh(xyxy: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run tracker on MOTChallenge sequences and save MOT-format results")
    p.add_argument("--tracker", required=True, help="Path to object_tracker_pipeline.py")
    p.add_argument("--data", required=True, help="Path to dataset root, e.g. ./data/MOT20 or ./data/MOT16")
    p.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    p.add_argument("--seq", nargs="+", required=True, help="Sequence names, e.g. MOT20-01 MOT20-02")
    p.add_argument("--out", required=True, help="Output directory for result .txt files")

    p.add_argument("--yolo", default="jameslahm/yolov10m", help="Ultralytics model ref (HF repo or local .pt)")
    p.add_argument("--osnet", default=None, help="Path to OSNet weights (.pth) for torchreid (optional)")
    p.add_argument("--no-reid", action="store_true", help="Disable ReID even if --osnet is provided")

    # Output policy
    p.add_argument("--pred-k", type=int, default=0, help="Output confirmed tracks with time_since_update <= k")
    p.add_argument("--out-tentative", action="store_true", help="Also output tentative tracks (not recommended for metrics)")

    # Common knobs (pass-through to TrackingConfig)
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--det-iou", type=float, default=None)
    p.add_argument("--thr-high", type=float, default=None)
    p.add_argument("--thr-low", type=float, default=None)
    p.add_argument("--gate", type=float, default=None)
    p.add_argument("--max-lost", type=int, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    tracker_path = Path(args.tracker).resolve()
    if not tracker_path.exists():
        raise FileNotFoundError(tracker_path)

    mod = _load_module_from_path(tracker_path)

    # Required symbols from your tracker module
    TrackingConfig = getattr(mod, "TrackingConfig")
    Tracker = getattr(mod, "Tracker")
    YoloDetector = getattr(mod, "YoloDetector")
    PersonReID = getattr(mod, "PersonReID")
    resolve_ultralytics_device = getattr(mod, "resolve_ultralytics_device")
    make_det_dict = getattr(mod, "make_det_dict")

    data_root = Path(args.data).resolve() / args.split
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {data_root}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config
    cfg = TrackingConfig()
    # we track people only (COCO class 0)
    cfg.reid_enabled_classes = (0,)

    if args.det_conf is not None:
        cfg.det_conf_thr = float(args.det_conf)
    if args.det_iou is not None:
        cfg.det_iou_thr = float(args.det_iou)
    if args.thr_high is not None:
        cfg.thr_high = float(args.thr_high)
    if args.thr_low is not None:
        cfg.thr_low = float(args.thr_low)
    if args.gate is not None:
        cfg.gate_thresh = float(args.gate)
        cfg.lost_gate_thresh = float(args.gate)
    if args.max_lost is not None:
        cfg.max_time_lost = int(args.max_lost)

    # Detector
    device = resolve_ultralytics_device()
    detector = YoloDetector(args.yolo, device=device, imgsz=int(getattr(cfg, "imgsz", 960)))

    # ReID (optional)
    reid = None
    if (not args.no_reid) and args.osnet:
        try:
            reid_device = "cuda" if getattr(mod, "torch").cuda.is_available() else "cpu"
        except Exception:
            reid_device = "cuda" if "cuda" in str(device) else "cpu"
        reid = PersonReID(args.osnet, device=reid_device)

    pred_k = max(0, int(args.pred_k))
    out_tent = bool(args.out_tentative)

    for seq_name in args.seq:
        seq_dir = data_root / seq_name
        if not seq_dir.exists():
            raise FileNotFoundError(f"Missing sequence folder: {seq_dir}")

        info = _read_seqinfo(seq_dir)
        fps = float(info["fps"])
        img_dir = str(info["img_dir"])
        seq_len = int(info["seq_len"])

        print(f"[SEQ] {seq_name}  fps={fps:.2f}  len={seq_len}")

        tracker = Tracker(fps=fps, cfg=cfg)

        lines: List[str] = []

        for frame_id, frame in _iter_frames(seq_dir, img_dir, seq_len):
            # 1) detect
            boxes_xyxy, scores, cls = detector.predict(frame, conf=cfg.det_conf_thr, iou=cfg.det_iou_thr)

            # keep only person
            keep = (cls.astype(np.int32) == 0)
            boxes_xyxy, scores, cls = boxes_xyxy[keep], scores[keep], cls[keep]

            # 2) optional ReID features
            det_features = None
            det_feat_valid = None
            if reid is not None:
                feats, valid = reid.extract(frame, boxes_xyxy, cls, person_class=0)
                det_features = feats
                det_feat_valid = valid

            # 3) build det dict + update
            det = make_det_dict(frame, boxes_xyxy, scores, cls, cfg=cfg)
            det["frame_idx"] = frame_id
            tracks, _det_consumed = tracker.update(det, det_features, det_feat_valid)

            # 4) export in MOT format
            H, W = frame.shape[:2]
            for tr in tracks:
                if getattr(tr, "is_removed")():
                    continue

                is_conf = getattr(tr, "is_confirmed")()
                is_tent = getattr(tr, "is_tentative")()
                is_lost = getattr(tr, "is_lost")()

                if is_lost:
                    continue
                if (not is_conf) and (not (out_tent and is_tent)):
                    continue

                # time_since_update <= pred_k
                if int(getattr(tr, "time_since_update")) > pred_k:
                    continue

                xyxy = tr.current_xyxy()
                xyxy = _clip_xyxy(xyxy, W, H)
                x, y, w, h = _xyxy_to_xywh(xyxy)

                # MOT format: frame, id, x, y, w, h, score, -1, -1, -1
                line = f"{frame_id},{int(tr.track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{float(tr.score):.4f},-1,-1,-1"
                lines.append(line)

            if frame_id % 200 == 0:
                print(f"  frame {frame_id}/{seq_len}")

        out_file = out_dir / f"{seq_name}.txt"
        out_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[OK] wrote {out_file}  lines={len(lines)}")


if __name__ == "__main__":
    main()
