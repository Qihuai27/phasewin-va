#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate detection attribution results: AUC faithfulness + Point Game.

Scans every method subdirectory under the results directory, computes
AUC and Point Game metrics, writes a per-method eval.json alongside the
attribution results, and saves a combined eval_summary.json at the top level.

Usage:
  python scripts/eval_detection.py [--results-dir DIR] [--annotation-file FILE]

Options:
  --results-dir    DIR   Base results directory.
                         Default: ./detection_results/coco-groundingdino
  --annotation-file FILE JSON annotation file with {image_id, bbox} entries.
                         Default: ./datasets/coco_groundingdino_correct_detection.json
  -h, --help             Show this help.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from attribution_research.evaluation.auc_faithfulness import AUCResult, aggregate_results  # noqa: E402
from attribution_research.evaluation.point_game import evaluate_point_game               # noqa: E402


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


def print_table(rows: list[tuple[str, dict]]) -> None:
    col_method = max((len(name) for name, _ in rows), default=6)
    col_method = max(col_method, 6)

    header = (
        f"{'Method':<{col_method}}  {'N':>5}  "
        f"{'Ins AUC':>8}  {'Del AUC':>8}  "
        f"{'IoU AUC':>8}  {'CLS AUC':>8}  "
        f"{'PG':>7}  {'EPG':>7}  {'AvgFwd':>10}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, r in rows:
        auc = r["auc"]
        pg  = r.get("point_game", {})
        print(
            f"{name:<{col_method}}  {auc['n_samples']:>5}  "
            f"{_fmt(auc['insertion_auc']):>8}  {_fmt(auc['deletion_auc']):>8}  "
            f"{_fmt(auc['insertion_iou_auc']):>8}  {_fmt(auc['insertion_cls_auc']):>8}  "
            f"{pg.get('point_game', float('nan')):>7.4f}  "
            f"{pg.get('energy_point_game', float('nan')):>7.4f}  "
            f"{_fmt_forward(auc.get('average_model_forward_calls')):>10}"
        )
    print(sep)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detection attribution results (AUC + Point Game).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="./detection_results/coco-groundingdino",
        help="Base results directory (default: ./detection_results/coco-groundingdino)",
    )
    parser.add_argument(
        "--annotation-file",
        default="./datasets/coco_groundingdino_correct_detection.json",
        help="Annotation file with image_id + bbox entries for Point Game",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    if not base_dir.is_dir():
        print(f"[error] Results directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    annotation_file = Path(args.annotation_file)
    if not annotation_file.is_file():
        print(f"[error] Annotation file not found: {annotation_file}", file=sys.stderr)
        sys.exit(1)

    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    if isinstance(annotations, dict):
        annotations = annotations.get("annotations", list(annotations.values()))
    print(f"Loaded {len(annotations)} annotations from {annotation_file}")

    method_dirs = find_method_dirs(base_dir)
    if not method_dirs:
        print(f"[error] No completed results found under {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(method_dirs)} method(s) under {base_dir}")

    rows: list[tuple[str, dict]] = []
    summary: dict = {}

    for method_dir in method_dirs:
        method_name = method_dir.name
        print(f"  evaluating  {method_name} ...", end=" ", flush=True)

        entry: dict = {}

        # AUC faithfulness
        try:
            auc_result = aggregate_results(str(method_dir), show_progress=False)
            entry["auc"] = dataclasses.asdict(auc_result)
        except Exception as exc:
            print(f"AUC FAILED ({exc})")
            continue

        # Point Game
        try:
            pg_result = evaluate_point_game(
                str(method_dir),
                annotations,
                score_key="insertion_score",
            )
            entry["point_game"] = pg_result
        except Exception as exc:
            print(f"(Point Game failed: {exc})", end=" ")
            entry["point_game"] = {}

        # Save per-method JSON
        eval_path = method_dir / "eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

        auc = auc_result
        pg  = entry.get("point_game", {})
        fwd_str = (
            f"  fwd={auc.average_model_forward_calls:.2f}"
            if auc.average_model_forward_calls is not None else ""
        )
        print(
            f"n={auc.n_samples}  "
            f"ins={auc.insertion_auc:.4f}  del={auc.deletion_auc:.4f}  "
            f"pg={pg.get('point_game', float('nan')):.4f}{fwd_str}"
        )
        rows.append((method_name, entry))
        summary[method_name] = entry

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
