#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AUC faithfulness for all caption/VQA attribution results.

Scans every method subdirectory under the results directory, computes
insertion/deletion AUC metrics (including word-sensitivity AUC), writes a
per-method eval.json alongside the attribution results, and saves a combined
eval_summary.json at the top level.

Usage:
  python scripts/eval_caption.py [--results-dir DIR] [--sensitivity FLOAT]

Options:
  --results-dir DIR    Base results directory.
                       Default: ./caption_results/Qwen2.5-VL-3B-coco-caption
  --sensitivity FLOAT  Word sensitivity threshold for caption AUC  (default: 0.2)
  -h, --help           Show this help.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from attribution_research.evaluation.auc_faithfulness import AUCResult, aggregate_results  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def find_method_dirs(base_dir: Path) -> list[Path]:
    dirs = []
    if not base_dir.is_dir():
        return dirs
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and (d / "json").is_dir() and any((d / "json").glob("*.json")):
            dirs.append(d)
    return dirs


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "  n/a "


def _fmt_forward(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "  n/a "


def print_table(rows: list[tuple[str, AUCResult]]) -> None:
    col_method = max((len(name) for name, _ in rows), default=6)
    col_method = max(col_method, 6)

    header = (
        f"{'Method':<{col_method}}  {'N':>5}  "
        f"{'Ins AUC':>8}  {'Del AUC':>8}  "
        f"{'Highest':>8}  {'@30%':>8}  {'@50%':>8}  "
        f"{'SensIns':>8}  {'SensDel':>8}  {'AvgFwd':>10}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, r in rows:
        print(
            f"{name:<{col_method}}  {r.n_samples:>5}  "
            f"{_fmt(r.insertion_auc):>8}  {_fmt(r.deletion_auc):>8}  "
            f"{_fmt(r.average_highest):>8}  "
            f"{_fmt(r.average_highest_30pct_area):>8}  "
            f"{_fmt(r.average_highest_50pct_area):>8}  "
            f"{_fmt(r.insertion_sensitivity_auc):>8}  "
            f"{_fmt(r.deletion_sensitivity_auc):>8}  "
            f"{_fmt_forward(r.average_model_forward_calls):>10}"
        )
    print(sep)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate caption attribution results (AUC faithfulness).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="./caption_results/Qwen2.5-VL-3B-coco-caption",
        help="Base results directory "
             "(default: ./caption_results/Qwen2.5-VL-3B-coco-caption)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.2,
        help="Word sensitivity threshold for caption AUC  (default: 0.2)",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    if not base_dir.is_dir():
        print(f"[error] Results directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    method_dirs = find_method_dirs(base_dir)
    if not method_dirs:
        print(f"[error] No completed results found under {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(method_dirs)} method(s) under {base_dir}")

    rows: list[tuple[str, AUCResult]] = []
    summary: dict = {}

    for method_dir in method_dirs:
        method_name = method_dir.name
        print(f"  evaluating  {method_name} ...", end=" ", flush=True)
        try:
            result = aggregate_results(
                str(method_dir),
                sensitivity=args.sensitivity,
                show_progress=False,
            )
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        # Save per-method JSON
        eval_path = method_dir / "eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, indent=2)

        sens_str = (
            f"  sens_ins={result.insertion_sensitivity_auc:.4f}"
            if result.insertion_sensitivity_auc is not None else ""
        )
        fwd_str = (
            f"  fwd={result.average_model_forward_calls:.2f}"
            if result.average_model_forward_calls is not None else ""
        )
        print(
            f"n={result.n_samples}  "
            f"ins={result.insertion_auc:.4f}  del={result.deletion_auc:.4f}"
            f"{sens_str}{fwd_str}"
        )
        rows.append((method_name, result))
        summary[method_name] = dataclasses.asdict(result)

    if not rows:
        print("[error] All evaluations failed.", file=sys.stderr)
        sys.exit(1)

    # Save combined summary
    summary_path = base_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    print_table(rows)


if __name__ == "__main__":
    main()
