#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize aggregated metrics for one result root or one run directory.

Examples
--------
python scripts/summarize_result_metrics.py \
  --result-root classification_results/imagenet-clip-vitl

python scripts/summarize_result_metrics.py \
  --result-root classification_results/imagenet-clip-vitl \
  --output-csv summaries/classification_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attribution_research.evaluation.auc_faithfulness import aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize result metrics across run directories")
    parser.add_argument(
        "--result-root",
        required=True,
        help="Either a task result root containing many run dirs, or one run dir containing json/",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan json subdirectories inside each run dir",
    )
    return parser.parse_args()


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "json").is_dir()


def discover_run_dirs(result_root: Path) -> List[Path]:
    if _is_run_dir(result_root):
        return [result_root]
    run_dirs = sorted(path for path in result_root.iterdir() if _is_run_dir(path))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories with json/ found under {result_root}"
        )
    return run_dirs


def load_run_metadata(run_dir: Path, recursive: bool = False) -> Dict[str, str]:
    json_root = run_dir / "json"
    json_paths: Iterable[Path]
    if recursive:
        json_paths = sorted(json_root.rglob("*.json"))
    else:
        json_paths = sorted(json_root.glob("*.json"))
    first_json = next(iter(json_paths), None)
    if first_json is None:
        raise FileNotFoundError(f"No json files found in {json_root}")
    with first_json.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    return {
        "run_tag": run_dir.name,
        "algorithm": str(info.get("algorithm", "")),
        "family": str(info.get("algorithm_family", "")),
        "segmenter": str(info.get("segmenter", "")),
    }


def build_rows(run_dirs: List[Path], recursive: bool = False) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        meta = load_run_metadata(run_dir, recursive=recursive)
        agg = aggregate_results(str(run_dir), recursive=recursive, show_progress=False)
        rows.append(
            {
                "run_tag": meta["run_tag"],
                "algorithm": meta["algorithm"],
                "family": meta["family"],
                "segmenter": meta["segmenter"],
                "n_samples": str(agg.n_samples),
                "insertion_auc": f"{agg.insertion_auc:.4f}",
                "deletion_auc": f"{agg.deletion_auc:.4f}",
                "avg_highest": f"{agg.average_highest:.4f}",
                "avg_highest_30pct_area": f"{agg.average_highest_30pct_area:.4f}",
                "avg_highest_50pct_area": f"{agg.average_highest_50pct_area:.4f}",
                "avg_model_forward_calls": (
                    f"{agg.average_model_forward_calls:.2f}"
                    if agg.average_model_forward_calls is not None
                    else ""
                ),
            }
        )
    rows.sort(key=lambda row: (row["algorithm"], row["run_tag"]))
    return rows


def print_table(rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    widths = {name: len(name) for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            widths[name] = max(widths[name], len(row.get(name, "")))

    header = "  ".join(name.ljust(widths[name]) for name in fieldnames)
    separator = "  ".join("-" * widths[name] for name in fieldnames)
    print(header)
    print(separator)
    for row in rows:
        print("  ".join(row.get(name, "").ljust(widths[name]) for name in fieldnames))


def write_csv(rows: List[Dict[str, str]], fieldnames: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    if not result_root.exists():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")

    run_dirs = discover_run_dirs(result_root)
    rows = build_rows(run_dirs, recursive=args.recursive)
    fieldnames = [
        "run_tag",
        "algorithm",
        "family",
        "segmenter",
        "n_samples",
        "insertion_auc",
        "deletion_auc",
        "avg_highest",
        "avg_highest_30pct_area",
        "avg_highest_50pct_area",
        "avg_model_forward_calls",
    ]
    print_table(rows, fieldnames)

    if args.output_csv:
        output_path = Path(args.output_csv).resolve()
        write_csv(rows, fieldnames, output_path)
        print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
