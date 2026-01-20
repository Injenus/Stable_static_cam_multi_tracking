# scripts/eval_motchallenge.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise RuntimeError(
        "Нужен scipy для Hungarian (linear_sum_assignment). Установите: pip install scipy"
    ) from e


# -------------------------
# MOT txt parsing
# -------------------------

@dataclass
class MotFrameData:
    ids: List[int]
    boxes_xywh: np.ndarray  # (N,4) in xywh float32


def _open_text(path: Path):
    # utf-8-sig помогает, если где-то BOM
    return path.open("r", encoding="utf-8-sig", errors="replace")


def _parse_mot_file(path: Path, *, is_gt: bool) -> Dict[int, MotFrameData]:
    """
    MOTChallenge txt format (common):
      frame, id, x, y, w, h, conf, class, vis, ...

    Returns dict: frame_idx (1-based) -> MotFrameData(ids, boxes_xywh)
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frames: Dict[int, Tuple[List[int], List[List[float]]]] = {}

    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            fr = int(float(parts[0]))
            tid = int(float(parts[1]))

            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # basic sanity
            if w <= 0 or h <= 0:
                continue

            conf = float(parts[6]) if len(parts) >= 7 else 1.0
            cls = int(float(parts[7])) if len(parts) >= 8 else 1

            if is_gt:
                # MOT GT: conf(mark) == 1 => valid, 0 => ignored region / not counted
                # Keep only positive IDs
                if tid < 0:
                    continue
                if conf < 0.5:
                    continue
                # For MOT20: class==1 is pedestrian (often). If class column exists – keep only 1.
                if len(parts) >= 8 and cls != 1:
                    continue

            frames.setdefault(fr, ([], []))
            frames[fr][0].append(tid)
            frames[fr][1].append([x, y, w, h])

    out: Dict[int, MotFrameData] = {}
    for fr, (ids, boxes) in frames.items():
        out[fr] = MotFrameData(ids=ids, boxes_xywh=np.asarray(boxes, dtype=np.float32))
    return out


def _find_seq_dirs(data_root: Path, split: str) -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    seq_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    seq_dirs.sort(key=lambda p: p.name)
    return seq_dirs


def _resolve_result_file(res_dir: Path, seq_name: str) -> Path:
    p1 = res_dir / f"{seq_name}.txt"
    if p1.exists():
        return p1
    p2 = res_dir / "data" / f"{seq_name}.txt"
    if p2.exists():
        return p2
    hits = list(res_dir.rglob(f"{seq_name}.txt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Result file for {seq_name} not found under: {res_dir}")


# -------------------------
# IoU + matching
# -------------------------

def _iou_matrix_xywh(gt_xywh: np.ndarray, pr_xywh: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU matrix between gt (G,4) and pr (P,4), both xywh.
    Returns (G,P) float32.
    """
    gt = np.asarray(gt_xywh, dtype=np.float32)
    pr = np.asarray(pr_xywh, dtype=np.float32)
    G = int(gt.shape[0])
    P = int(pr.shape[0])
    if G == 0 or P == 0:
        return np.zeros((G, P), dtype=np.float32)

    gt_x1 = gt[:, 0:1]
    gt_y1 = gt[:, 1:2]
    gt_x2 = gt_x1 + gt[:, 2:3]
    gt_y2 = gt_y1 + gt[:, 3:4]

    pr_x1 = pr[:, 0][None, :]
    pr_y1 = pr[:, 1][None, :]
    pr_x2 = (pr[:, 0] + pr[:, 2])[None, :]
    pr_y2 = (pr[:, 1] + pr[:, 3])[None, :]

    ix1 = np.maximum(gt_x1, pr_x1)
    iy1 = np.maximum(gt_y1, pr_y1)
    ix2 = np.minimum(gt_x2, pr_x2)
    iy2 = np.minimum(gt_y2, pr_y2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    gt_area = np.maximum(0.0, gt_x2 - gt_x1) * np.maximum(0.0, gt_y2 - gt_y1)  # (G,1)
    pr_area = np.maximum(0.0, pr_x2 - pr_x1) * np.maximum(0.0, pr_y2 - pr_y1)  # (1,P)

    union = gt_area + pr_area - inter + 1e-9
    iou = inter / union
    return iou.astype(np.float32)


def _match_one_frame(
    gt_ids: List[int],
    gt_boxes: np.ndarray,
    pr_ids: List[int],
    pr_boxes: np.ndarray,
    *,
    iou_th: float,
) -> Tuple[List[Tuple[int, int, float]], set[int], set[int]]:
    """
    Hungarian matching maximizing IoU with threshold gating.
    Returns:
      matches: list of (gt_index, pr_index, iou)
      matched_gt_idx_set
      matched_pr_idx_set
    """
    G = int(gt_boxes.shape[0])
    P = int(pr_boxes.shape[0])
    if G == 0 or P == 0:
        return [], set(), set()

    iou = _iou_matrix_xywh(gt_boxes, pr_boxes)  # (G,P)
    LARGE = 1e6
    cost = (1.0 - iou).astype(np.float32)
    cost[iou < float(iou_th)] = LARGE

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    mgt: set[int] = set()
    mpr: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if float(cost[r, c]) >= LARGE:
            continue
        v = float(iou[r, c])
        if v < float(iou_th):
            continue
        matches.append((int(r), int(c), v))
        mgt.add(int(r))
        mpr.add(int(c))

    return matches, mgt, mpr


# -------------------------
# Evaluation (self-contained, no motmetrics)
# -------------------------

def evaluate_one(gt_path: Path, res_path: Path, *, iou_th: float) -> dict:
    gt_by_fr = _parse_mot_file(gt_path, is_gt=True)
    pr_by_fr = _parse_mot_file(res_path, is_gt=False)

    max_fr = 0
    if gt_by_fr:
        max_fr = max(max_fr, max(gt_by_fr.keys()))
    if pr_by_fr:
        max_fr = max(max_fr, max(pr_by_fr.keys()))

    # Counters (CLEAR MOT style)
    num_objects = 0          # total GT detections across frames
    num_predictions = 0      # total predicted detections across frames
    num_matches = 0

    num_false_positives = 0
    num_misses = 0

    num_switches = 0
    num_fragmentations = 0

    iou_sum = 0.0

    # For IDSW and FRAG
    last_pred_for_gt: Dict[int, int] = {}
    prev_matched_for_gt: Dict[int, bool] = defaultdict(bool)
    ever_matched_for_gt: Dict[int, bool] = defaultdict(bool)

    # For IDF1 (global assignment on overlaps)
    overlap_counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)

    # Iterate frames
    for fr in range(1, max_fr + 1):
        gt = gt_by_fr.get(fr, MotFrameData([], np.zeros((0, 4), dtype=np.float32)))
        pr = pr_by_fr.get(fr, MotFrameData([], np.zeros((0, 4), dtype=np.float32)))

        gt_ids = gt.ids
        pr_ids = pr.ids
        gt_boxes = gt.boxes_xywh
        pr_boxes = pr.boxes_xywh

        G = len(gt_ids)
        P = len(pr_ids)

        num_objects += G
        num_predictions += P

        matches, matched_gt_idx, matched_pr_idx = _match_one_frame(
            gt_ids, gt_boxes, pr_ids, pr_boxes, iou_th=float(iou_th)
        )

        m = len(matches)
        num_matches += m
        num_false_positives += (P - m)
        num_misses += (G - m)

        # Sum IoU for MOTP
        for (g_i, p_i, v_iou) in matches:
            iou_sum += float(v_iou)

        # Build quick lookup: gt_index -> pred_index for matched
        gt_to_pr: Dict[int, int] = {g_i: p_i for (g_i, p_i, _v) in matches}

        # Update per-GT states (IDSW, FRAG)
        for g_i, gt_id in enumerate(gt_ids):
            cur_matched = (g_i in gt_to_pr)

            if cur_matched:
                p_i = gt_to_pr[g_i]
                pr_id = int(pr_ids[p_i])

                # IDSW: only if previous frame this GT was matched continuously
                if prev_matched_for_gt[gt_id]:
                    if (gt_id in last_pred_for_gt) and (last_pred_for_gt[gt_id] != pr_id):
                        num_switches += 1

                # FRAG: match resumes after a gap, but only after at least one match existed before
                if (not prev_matched_for_gt[gt_id]) and ever_matched_for_gt[gt_id]:
                    num_fragmentations += 1

                prev_matched_for_gt[gt_id] = True
                ever_matched_for_gt[gt_id] = True
                last_pred_for_gt[gt_id] = pr_id

                overlap_counts[(gt_id, pr_id)] += 1
            else:
                prev_matched_for_gt[gt_id] = False

    # Derived metrics
    if num_objects > 0:
        mota = 1.0 - (num_misses + num_false_positives + num_switches) / float(num_objects)
    else:
        mota = float("nan")

    motp = (iou_sum / float(num_matches)) if num_matches > 0 else float("nan")

    precision = (num_matches / float(num_matches + num_false_positives)) if (num_matches + num_false_positives) > 0 else 0.0
    recall = (num_matches / float(num_matches + num_misses)) if (num_matches + num_misses) > 0 else 0.0

    # IDF1 via global assignment on overlaps
    gt_id_set = sorted({gid for (gid, _pid) in overlap_counts.keys()} | set(last_pred_for_gt.keys()) | set(ever_matched_for_gt.keys()))
    pr_id_set = sorted({pid for (_gid, pid) in overlap_counts.keys()})

    idtp = 0.0
    if gt_id_set and pr_id_set:
        G = len(gt_id_set)
        P = len(pr_id_set)
        mat = np.zeros((G, P), dtype=np.float32)
        gt_index = {gid: i for i, gid in enumerate(gt_id_set)}
        pr_index = {pid: j for j, pid in enumerate(pr_id_set)}
        for (gid, pid), cnt in overlap_counts.items():
            mat[gt_index[gid], pr_index[pid]] = float(cnt)

        # maximize sum => minimize negative
        cost = -mat
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if r < G and c < P:
                idtp += float(mat[r, c])

    idfn = float(num_objects) - float(idtp)
    idfp = float(num_predictions) - float(idtp)

    idp = (idtp / (idtp + idfp)) if (idtp + idfp) > 0 else 0.0
    idr = (idtp / (idtp + idfn)) if (idtp + idfn) > 0 else 0.0
    idf1 = (2.0 * idtp / (2.0 * idtp + idfp + idfn)) if (2.0 * idtp + idfp + idfn) > 0 else 0.0

    return {
        "num_objects": float(num_objects),
        "num_predictions": float(num_predictions),
        "num_matches": float(num_matches),
        "num_false_positives": float(num_false_positives),
        "num_misses": float(num_misses),
        "num_switches": float(num_switches),
        "num_fragmentations": float(num_fragmentations),
        "mota": float(mota),
        "motp": float(motp),
        "precision": float(precision),
        "recall": float(recall),
        "idtp": float(idtp),
        "idfp": float(idfp),
        "idfn": float(idfn),
        "idp": float(idp),
        "idr": float(idr),
        "idf1": float(idf1),
    }


def _overall_from_rows(rows: List[dict]) -> dict:
    def s(k: str) -> float:
        return float(sum(float(r.get(k, 0.0)) for r in rows))

    out = {}
    for k in [
        "num_objects", "num_predictions", "num_matches",
        "num_false_positives", "num_misses",
        "num_switches", "num_fragmentations",
        "idtp", "idfp", "idfn",
    ]:
        out[k] = s(k)

    num_objects = out["num_objects"]
    fp = out["num_false_positives"]
    fn = out["num_misses"]
    idsw = out["num_switches"]
    matches = out["num_matches"]

    out["mota"] = 1.0 - (fn + fp + idsw) / max(1.0, num_objects)
    out["precision"] = matches / max(1.0, matches + fp)
    out["recall"] = matches / max(1.0, matches + fn)

    idtp = out["idtp"]
    idfp = out["idfp"]
    idfn = out["idfn"]
    out["idp"] = idtp / max(1.0, idtp + idfp)
    out["idr"] = idtp / max(1.0, idtp + idfn)
    out["idf1"] = (2.0 * idtp) / max(1.0, 2.0 * idtp + idfp + idfn)

    # MOTP корректно агрегировать без суммы IoU нельзя => оставим nan (или уберите из отчёта)
    out["motp"] = float("nan")
    return out


def _write_csv(path: Path, rows: List[dict], header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")

    if header is None:
        # stable header order
        header = [
            "seq",
            "mota", "motp", "idf1", "idp", "idr",
            "num_objects", "num_predictions", "num_matches",
            "num_false_positives", "num_misses", "num_switches", "num_fragmentations",
            "precision", "recall",
            "idtp", "idfp", "idfn",
        ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MOTChallenge results (self-contained, NumPy2-safe).")
    p.add_argument("--data", required=True, help="Path to dataset root, e.g. data/MOT20")
    p.add_argument("--split", default="train", choices=["train", "test"], help="Split (train has GT)")
    p.add_argument("--res", required=True, help="Directory with result txt files (MOT format), e.g. results/run1")
    p.add_argument("--out-csv", default=None, help="Where to write CSV summary")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching (default 0.5)")
    p.add_argument("--seq", nargs="+", default=None, help="Sequence names, e.g. MOT20-01 MOT20-02 ...")
    p.add_argument("--sequences", nargs="+", default=None, help="Alias for --seq")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    data_root = Path(args.data)
    split = str(args.split)
    res_dir = Path(args.res)

    if split != "train":
        raise RuntimeError("Для test split GT нет. Для test нужна отправка на MOTChallenge сервер.")

    seq_list = args.seq if args.seq is not None else args.sequences
    if seq_list is None:
        seq_dirs = _find_seq_dirs(data_root, split)
        seq_list = [p.name for p in seq_dirs]

    rows: List[dict] = []

    for seq in seq_list:
        seq = str(seq)
        gt_path = data_root / split / seq / "gt" / "gt.txt"
        res_path = _resolve_result_file(res_dir, seq)

        row = evaluate_one(gt_path, res_path, iou_th=float(args.iou))
        row_out = {"seq": seq}
        row_out.update(row)
        rows.append(row_out)

        print(
            f"[{seq}] "
            f"MOTA={row_out['mota']:.4f}  IDF1={row_out['idf1']:.4f}  "
            f"FP={int(row_out['num_false_positives'])}  FN={int(row_out['num_misses'])}  "
            f"IDSW={int(row_out['num_switches'])}  FRAG={int(row_out['num_fragmentations'])}"
        )

    overall = _overall_from_rows(rows)
    overall_row = {"seq": "OVERALL"}
    overall_row.update(overall)
    rows.append(overall_row)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        _write_csv(out_csv, rows)
        print("Wrote:", out_csv)

    print("\n=== SUMMARY ===")
    for r in rows:
        print(
            f"{r['seq']:>8} | "
            f"MOTA={float(r.get('mota', float('nan'))):.4f} "
            f"IDF1={float(r.get('idf1', float('nan'))):.4f} "
            f"IDSW={int(float(r.get('num_switches', 0)))} "
            f"FRAG={int(float(r.get('num_fragmentations', 0)))} "
            f"FP={int(float(r.get('num_false_positives', 0)))} "
            f"FN={int(float(r.get('num_misses', 0)))}"
        )


if __name__ == "__main__":
    main()
