#!/usr/bin/env python3
"""Generate a compact, visual report from MOT metrics.

This script is meant to be the final step after you have:
  1) tracker outputs in MOTChallenge format (results/<run>/<SEQ>.txt)
  2) metrics CSV from scripts/eval_motchallenge.py (metrics.csv)

It produces:
  - report.md (easy to paste into a thesis/report)
  - report.html (quick view in browser)
  - plots/*.png (optional; if matplotlib is available)

Usage:
  python scripts/make_report.py --metrics results/MOT20_run_baseline/metrics.csv --out results/MOT20_run_baseline/report

Optional convenience (run eval automatically):
  python scripts/make_report.py --data data/MOT20 --split train --res results/MOT20_run_baseline --out results/MOT20_run_baseline/report
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FLOAT_COLS = {
    "mota",
    "idf1",
    "motp",
    "precision",
    "recall",
}
INT_COLS = {
    "fp",
    "fn",
    "idsw",
    "frag",
    "num_matches",
    "num_misses",
    "num_false_positives",
    "num_objects",
}


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        # some CSV writers store ints as floats ("12.0")
        return int(float(s))
    except Exception:
        return None


def read_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rr: Dict[str, Any] = dict(r)
            # normalize column name for sequence
            if "seq" not in rr and "Sequence" in rr:
                rr["seq"] = rr.get("Sequence")
            # cast
            for k in list(rr.keys()):
                if k in FLOAT_COLS:
                    rr[k] = _to_float(rr[k])
                elif k in INT_COLS:
                    rr[k] = _to_int(rr[k])
            rows.append(rr)
    return rows


def find_overall(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in rows:
        if str(r.get("seq", "")).strip().upper() == "OVERALL":
            return r
    return None


def ensure_overall(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    overall = find_overall(rows)
    if overall is not None:
        # put overall last
        seq_rows = [r for r in rows if r is not overall]
        seq_rows.sort(key=lambda r: str(r.get("seq", "")))
        return seq_rows + [overall], overall

    # fallback: aggregate counts and approximate MOTA/precision/recall.
    # (This should not happen if metrics.csv came from eval_motchallenge.py.)
    agg: Dict[str, Any] = {"seq": "OVERALL"}
    for k in INT_COLS:
        agg[k] = sum(int(r.get(k) or 0) for r in rows)

    num_objects = int(agg.get("num_objects") or 0)
    fp = int(agg.get("fp") or 0)
    fn = int(agg.get("fn") or 0)
    idsw = int(agg.get("idsw") or 0)
    matches = int(agg.get("num_matches") or 0)

    if num_objects > 0:
        agg["mota"] = 1.0 - (fp + fn + idsw) / float(num_objects)
        agg["recall"] = matches / float(matches + fn) if (matches + fn) > 0 else 0.0
        agg["precision"] = matches / float(matches + fp) if (matches + fp) > 0 else 0.0
    else:
        agg["mota"] = None
        agg["precision"] = None
        agg["recall"] = None

    agg["idf1"] = None
    agg["motp"] = None

    seq_rows = list(rows)
    seq_rows.sort(key=lambda r: str(r.get("seq", "")))
    return seq_rows + [agg], agg


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{100.0 * float(v):.2f}%"


def _fmt_float(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{float(v):.4f}"


def _fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "-"
    return str(int(v))


def write_markdown(out_dir: Path, rows: List[Dict[str, Any]], *, title: str) -> Path:
    md_path = out_dir / "report.md"

    # primary metrics for an MVP report
    cols = [
        ("seq", "Sequence"),
        ("mota", "MOTA"),
        ("idf1", "IDF1"),
        ("motp", "MOTP"),
        ("precision", "Prec"),
        ("recall", "Rec"),
        ("fp", "FP"),
        ("fn", "FN"),
        ("idsw", "IDSW"),
        ("frag", "FRAG"),
        ("num_objects", "GT"),
    ]

    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Summary metrics\n")

    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)

    for r in rows:
        row_cells: List[str] = []
        for k, _h in cols:
            if k in {"mota", "idf1"}:
                row_cells.append(_fmt_pct(r.get(k)))
            elif k in {"precision", "recall"}:
                row_cells.append(_fmt_pct(r.get(k)))
            elif k == "motp":
                row_cells.append(_fmt_float(r.get(k)))
            elif k in INT_COLS:
                row_cells.append(_fmt_int(r.get(k)))
            else:
                row_cells.append(str(r.get(k, "-")))
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("\n## Metric notes (minimal)\n")
    lines.append("- **MOTA** = 1 − (FP + FN + IDSW) / GT. Higher is better.")
    lines.append("- **IDF1** is identity F1-score (ID precision/recall). Higher is better.")
    lines.append("- **MOTP** is localization precision (lower-level motmetrics definition). Higher is better.")
    lines.append("- **IDSW** counts identity switches; **FRAG** counts track fragmentations.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def write_html(out_dir: Path, rows: List[Dict[str, Any]], *, title: str) -> Path:
    html_path = out_dir / "report.html"

    # build an HTML table from the same fields as markdown
    fields = ["seq", "mota", "idf1", "motp", "precision", "recall", "fp", "fn", "idsw", "frag", "num_objects"]
    headers = ["Sequence", "MOTA", "IDF1", "MOTP", "Prec", "Rec", "FP", "FN", "IDSW", "FRAG", "GT"]

    def cell(k: str, r: Dict[str, Any]) -> str:
        if k in {"mota", "idf1", "precision", "recall"}:
            return _fmt_pct(r.get(k))
        if k == "motp":
            return _fmt_float(r.get(k))
        if k in INT_COLS:
            return _fmt_int(r.get(k))
        return str(r.get(k, "-"))

    # plots (optional)
    plot_imgs = []
    plots_dir = out_dir / "plots"
    for name in ["mota.png", "idf1.png", "idsw_frag.png"]:
        p = plots_dir / name
        if p.exists():
            plot_imgs.append(f"<img src='plots/{name}' style='max-width: 980px; width: 100%; height: auto; margin: 12px 0;' />")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #f4f4f4; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    code {{ background: #f7f7f7; padding: 2px 4px; }}
    .note {{ max-width: 980px; color: #444; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class='note'>Generated: {dt.datetime.now().isoformat(timespec='seconds')}</p>

  {''.join(plot_imgs)}

  <h2>Summary metrics</h2>
  <table>
    <thead>
      <tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr>
    </thead>
    <tbody>
      {''.join('<tr>' + ''.join(f'<td>{cell(k, r)}</td>' for k in fields) + '</tr>' for r in rows)}
    </tbody>
  </table>

  <h2>Metric notes (minimal)</h2>
  <div class='note'>
    <ul>
      <li><b>MOTA</b> = 1 − (FP + FN + IDSW) / GT. Higher is better.</li>
      <li><b>IDF1</b> is identity F1-score (ID precision/recall). Higher is better.</li>
      <li><b>MOTP</b> is localization precision (motmetrics definition). Higher is better.</li>
      <li><b>IDSW</b> counts identity switches; <b>FRAG</b> counts track fragmentations.</li>
    </ul>
  </div>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


def try_make_plots(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    seqs = [str(r.get("seq", "")) for r in rows]
    mota = [float(r.get("mota") or 0.0) * 100.0 for r in rows]
    idf1 = [float(r.get("idf1") or 0.0) * 100.0 for r in rows]

    # MOTA bar
    plt.figure()
    plt.bar(seqs, mota)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MOTA (%)")
    plt.tight_layout()
    plt.savefig(plots_dir / "mota.png", dpi=160)
    plt.close()

    # IDF1 bar
    plt.figure()
    plt.bar(seqs, idf1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("IDF1 (%)")
    plt.tight_layout()
    plt.savefig(plots_dir / "idf1.png", dpi=160)
    plt.close()

    # IDSW/FRAG (counts)
    idsw = [int(r.get("idsw") or 0) for r in rows]
    frag = [int(r.get("frag") or 0) for r in rows]

    x = list(range(len(seqs)))

    plt.figure()
    plt.bar([i - 0.2 for i in x], idsw, width=0.4, label="IDSW")
    plt.bar([i + 0.2 for i in x], frag, width=0.4, label="FRAG")
    plt.xticks(x, seqs, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "idsw_frag.png", dpi=160)
    plt.close()


def maybe_run_eval(args: argparse.Namespace) -> Optional[Path]:
    # If --metrics is given, use it. Else, try to run eval based on --data/--split/--res.
    if args.metrics is not None:
        return Path(args.metrics).expanduser().resolve()

    if args.data is None or args.res is None:
        return None

    # Import local evaluator (no extra install beyond motmetrics)
    import subprocess

    out_csv = Path(args.out).expanduser().resolve() / "metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(Path(__file__).parent / "eval_motchallenge.py"),
        "--data",
        str(Path(args.data).expanduser().resolve()),
        "--split",
        str(args.split),
        "--res",
        str(Path(args.res).expanduser().resolve()),
        "--out-csv",
        str(out_csv),
    ]
    subprocess.check_call(cmd)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None, help="Path to metrics.csv (from eval_motchallenge.py)")

    # convenience: run eval automatically
    ap.add_argument("--data", default=None, help="Dataset root, e.g. data/MOT20")
    ap.add_argument("--split", default="train", choices=["train", "test"], help="Split")
    ap.add_argument("--res", default=None, help="Results directory with <SEQ>.txt")

    ap.add_argument("--out", required=True, help="Output directory for report")
    ap.add_argument("--title", default=None, help="Report title")

    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = maybe_run_eval(args)
    if metrics_csv is None or not metrics_csv.exists():
        raise SystemExit(
            "Need either --metrics <metrics.csv> or (--data <...> --res <...>) to run eval automatically."
        )

    rows = read_metrics_csv(metrics_csv)
    if not rows:
        raise SystemExit(f"No rows in metrics CSV: {metrics_csv}")

    rows, overall = ensure_overall(rows)

    title = args.title
    if title is None:
        title = f"MOT report: {metrics_csv.parent.name}"

    try_make_plots(out_dir, rows)
    md = write_markdown(out_dir, rows, title=title)
    html = write_html(out_dir, rows, title=title)

    print(f"[OK] {md}")
    print(f"[OK] {html}")


if __name__ == "__main__":
    main()
