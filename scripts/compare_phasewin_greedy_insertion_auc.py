#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare per-sample insertion AUC between PhaseWin and Greedy runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attribution_research.evaluation.auc_faithfulness import compute_auc_from_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count per-sample PhaseWin vs Greedy insertion-AUC outcomes for "
            "each result setting."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["results/classification", "results/caption"],
        help=(
            "Root directories to scan recursively. A setting is any directory "
            "whose immediate children include both greedy* and phasewin* result dirs."
        ),
    )
    parser.add_argument(
        "--point-scale",
        type=float,
        default=100.0,
        help="Scale applied to AUC deltas for point thresholds. Default: 100.",
    )
    parser.add_argument(
        "--near-points",
        type=float,
        default=3.0,
        help="Point threshold for +/- near comparison. Default: 3.",
    )
    parser.add_argument(
        "--large-points",
        type=float,
        default=10.0,
        help="Point threshold for large PhaseWin deficit. Default: 10.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV file for the summary table.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON file with summary rows and optional per-sample rows.",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Include per-sample deltas in --output-json.",
    )
    parser.add_argument(
        "--exclude-phasewin-behind-points",
        type=float,
        default=None,
        help=(
            "Also report filtered means after excluding samples where "
            "PhaseWin is behind Greedy by this many points or more. "
            "For example, 10 excludes delta_points <= -10."
        ),
    )
    parser.add_argument(
        "--extreme-stats",
        action="store_true",
        help=(
            "Report stats for extreme samples where PhaseWin trails Greedy by "
            "--large-points or more."
        ),
    )
    parser.add_argument(
        "--front-area",
        type=float,
        default=0.2,
        help="Released-area prefix used by --extreme-stats. Default: 0.2.",
    )
    return parser


def method_key_from_result_dir(path: Path) -> str:
    return path.name.split("-", 1)[0].strip().lower().replace("-", "_")


def is_result_dir(path: Path) -> bool:
    json_dir = path / "json"
    return path.is_dir() and json_dir.is_dir() and any(json_dir.glob("*.json"))


def select_latest_method_dir(setting_dir: Path, method_key: str) -> Path | None:
    candidates = [
        path for path in setting_dir.iterdir()
        if is_result_dir(path) and method_key_from_result_dir(path) == method_key
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def discover_settings(roots: Iterable[Path]) -> list[Path]:
    settings: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            if (
                select_latest_method_dir(path, "greedy") is not None
                and select_latest_method_dir(path, "phasewin") is not None
            ):
                settings.append(path)
                seen.add(resolved)
    return sorted(settings)


def _insertion_curve(payload: dict) -> tuple[list[float], list[float]]:
    x_values = [0.0] + [float(value) for value in payload["region_area"]]
    y_values = [float(payload["deletion_score"][-1])] + [
        float(value) for value in payload["insertion_score"]
    ]

    keep_x = [x_values[0]]
    keep_y = [y_values[0]]
    for x_value, y_value in zip(x_values[1:], y_values[1:]):
        if x_value <= keep_x[-1]:
            keep_x[-1] = max(keep_x[-1], x_value)
            keep_y[-1] = y_value
        else:
            keep_x.append(x_value)
            keep_y.append(y_value)
    return keep_x, keep_y


def _trapezoid_area(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.0
    return sum(
        (y_values[idx] + y_values[idx - 1])
        * 0.5
        * (x_values[idx] - x_values[idx - 1])
        for idx in range(1, len(x_values))
    )


def front_insertion_auc(payload: dict, area_limit: float) -> float:
    """Return insertion AUC over [0, area_limit], normalized by area_limit."""
    limit = float(area_limit)
    if limit <= 0:
        raise ValueError("area_limit must be positive")

    x_values, y_values = _insertion_curve(payload)
    prefix_x = [x_values[0]]
    prefix_y = [y_values[0]]

    for idx in range(1, len(x_values)):
        x_prev = x_values[idx - 1]
        y_prev = y_values[idx - 1]
        x_curr = x_values[idx]
        y_curr = y_values[idx]
        if x_curr < limit:
            prefix_x.append(x_curr)
            prefix_y.append(y_curr)
            continue

        if x_curr == x_prev:
            y_limit = y_curr
        else:
            ratio = (limit - x_prev) / (x_curr - x_prev)
            y_limit = y_prev + ratio * (y_curr - y_prev)
        prefix_x.append(limit)
        prefix_y.append(y_limit)
        break

    if prefix_x[-1] < limit:
        prefix_x.append(limit)
        prefix_y.append(y_values[-1])

    return _trapezoid_area(prefix_x, prefix_y) / limit


def highest_confidence(payload: dict) -> float:
    _, y_values = _insertion_curve(payload)
    return max(y_values)


def load_sample_metrics(method_dir: Path, *, front_area: float) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for json_path in sorted((method_dir / "json").glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = compute_auc_from_json(payload)
        values[json_path.stem] = {
            "insertion_auc": float(metrics["insertion_auc"]),
            "front_insertion_auc": front_insertion_auc(payload, front_area),
            "highest_confidence": highest_confidence(payload),
        }
    return values


def compare_setting(
    setting_dir: Path,
    *,
    point_scale: float,
    near_points: float,
    large_points: float,
    front_area: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    greedy_dir = select_latest_method_dir(setting_dir, "greedy")
    phasewin_dir = select_latest_method_dir(setting_dir, "phasewin")
    if greedy_dir is None or phasewin_dir is None:
        raise FileNotFoundError(f"Missing greedy/phasewin dirs under {setting_dir}")

    greedy_metrics = load_sample_metrics(greedy_dir, front_area=front_area)
    phasewin_metrics = load_sample_metrics(phasewin_dir, front_area=front_area)
    shared_ids = sorted(set(greedy_metrics) & set(phasewin_metrics))
    if not shared_ids:
        raise RuntimeError(f"No shared samples under {setting_dir}")

    sample_rows: list[dict[str, object]] = []
    for sample_id in shared_ids:
        greedy = greedy_metrics[sample_id]["insertion_auc"]
        phasewin = phasewin_metrics[sample_id]["insertion_auc"]
        delta_points = (phasewin - greedy) * point_scale
        sample_rows.append(
            {
                "setting": str(setting_dir),
                "sample_id": sample_id,
                "greedy_insertion_auc": greedy,
                "phasewin_insertion_auc": phasewin,
                "delta_points": delta_points,
                "greedy_front_insertion_auc": greedy_metrics[sample_id]["front_insertion_auc"],
                "phasewin_front_insertion_auc": phasewin_metrics[sample_id]["front_insertion_auc"],
                "greedy_highest_confidence": greedy_metrics[sample_id]["highest_confidence"],
                "phasewin_highest_confidence": phasewin_metrics[sample_id]["highest_confidence"],
            }
        )

    deltas = [float(row["delta_points"]) for row in sample_rows]
    greedy_values = [float(row["greedy_insertion_auc"]) for row in sample_rows]
    phasewin_values = [float(row["phasewin_insertion_auc"]) for row in sample_rows]

    row = {
        "setting": str(setting_dir),
        "greedy_dir": greedy_dir.name,
        "phasewin_dir": phasewin_dir.name,
        "n_shared": len(shared_ids),
        "phasewin_gt_greedy": sum(delta > 0.0 for delta in deltas),
        "phasewin_gt_greedy_3pts": sum(delta >= near_points for delta in deltas),
        "abs_diff_within_3pts": sum(abs(delta) <= near_points for delta in deltas),
        "phasewin_behind_greedy_3pts": sum(delta <= -near_points for delta in deltas),
        "phasewin_behind_greedy_10pts": sum(delta <= -large_points for delta in deltas),
        "mean_greedy_insertion_auc": sum(greedy_values) / len(greedy_values),
        "mean_phasewin_insertion_auc": sum(phasewin_values) / len(phasewin_values),
        "mean_delta_points": sum(deltas) / len(deltas),
    }
    return row, sample_rows


def add_filtered_summary(
    row: dict[str, object],
    sample_rows: list[dict[str, object]],
    *,
    exclude_phasewin_behind_points: float,
) -> None:
    kept = [
        sample for sample in sample_rows
        if float(sample["delta_points"]) > -float(exclude_phasewin_behind_points)
    ]
    row["filtered_exclude_phasewin_behind_points"] = float(exclude_phasewin_behind_points)
    row["filtered_excluded"] = len(sample_rows) - len(kept)
    row["filtered_n"] = len(kept)
    if not kept:
        row["filtered_mean_greedy_insertion_auc"] = None
        row["filtered_mean_phasewin_insertion_auc"] = None
        row["filtered_mean_delta_points"] = None
        return

    greedy_values = [float(sample["greedy_insertion_auc"]) for sample in kept]
    phasewin_values = [float(sample["phasewin_insertion_auc"]) for sample in kept]
    row["filtered_mean_greedy_insertion_auc"] = sum(greedy_values) / len(greedy_values)
    row["filtered_mean_phasewin_insertion_auc"] = sum(phasewin_values) / len(phasewin_values)
    row["filtered_mean_delta_points"] = (
        float(row["filtered_mean_phasewin_insertion_auc"])
        - float(row["filtered_mean_greedy_insertion_auc"])
    ) * 100.0


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def add_extreme_summary(
    row: dict[str, object],
    sample_rows: list[dict[str, object]],
    *,
    large_points: float,
    front_area: float,
) -> None:
    extreme = [
        sample for sample in sample_rows
        if float(sample["delta_points"]) <= -float(large_points)
    ]
    row["extreme_threshold_points"] = float(large_points)
    row["extreme_front_area"] = float(front_area)
    row["extreme_n"] = len(extreme)
    row["extreme_mean_greedy_front_insertion_auc"] = mean_or_none(
        [float(sample["greedy_front_insertion_auc"]) for sample in extreme]
    )
    row["extreme_mean_phasewin_front_insertion_auc"] = mean_or_none(
        [float(sample["phasewin_front_insertion_auc"]) for sample in extreme]
    )
    row["extreme_mean_greedy_highest_confidence"] = mean_or_none(
        [float(sample["greedy_highest_confidence"]) for sample in extreme]
    )
    row["extreme_mean_phasewin_highest_confidence"] = mean_or_none(
        [float(sample["phasewin_highest_confidence"]) for sample in extreme]
    )
    row["extreme_mean_delta_points"] = mean_or_none(
        [float(sample["delta_points"]) for sample in extreme]
    )


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def print_table(rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    widths = {name: len(name) for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            widths[name] = max(widths[name], len(format_value(row.get(name, ""))))

    print("  ".join(name.ljust(widths[name]) for name in fieldnames))
    print("  ".join("-" * widths[name] for name in fieldnames))
    for row in rows:
        print(
            "  ".join(
                format_value(row.get(name, "")).ljust(widths[name])
                for name in fieldnames
            )
        )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(root) for root in args.roots]
    settings = discover_settings(roots)
    if not settings:
        raise FileNotFoundError(f"No settings with greedy and phasewin found under: {args.roots}")

    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for setting_dir in settings:
        summary, samples = compare_setting(
            setting_dir,
            point_scale=args.point_scale,
            near_points=args.near_points,
            large_points=args.large_points,
            front_area=args.front_area,
        )
        if args.exclude_phasewin_behind_points is not None:
            add_filtered_summary(
                summary,
                samples,
                exclude_phasewin_behind_points=args.exclude_phasewin_behind_points,
            )
        if args.extreme_stats:
            add_extreme_summary(
                summary,
                samples,
                large_points=args.large_points,
                front_area=args.front_area,
            )
        summary_rows.append(summary)
        sample_rows.extend(samples)

    fieldnames = [
        "setting",
        "n_shared",
        "phasewin_gt_greedy",
        "phasewin_gt_greedy_3pts",
        "abs_diff_within_3pts",
        "phasewin_behind_greedy_3pts",
        "phasewin_behind_greedy_10pts",
        "mean_greedy_insertion_auc",
        "mean_phasewin_insertion_auc",
        "mean_delta_points",
    ]
    if args.exclude_phasewin_behind_points is not None:
        fieldnames.extend(
            [
                "filtered_excluded",
                "filtered_n",
                "filtered_mean_greedy_insertion_auc",
                "filtered_mean_phasewin_insertion_auc",
                "filtered_mean_delta_points",
            ]
        )
    if args.extreme_stats:
        fieldnames.extend(
            [
                "extreme_n",
                "extreme_front_area",
                "extreme_mean_greedy_front_insertion_auc",
                "extreme_mean_phasewin_front_insertion_auc",
                "extreme_mean_greedy_highest_confidence",
                "extreme_mean_phasewin_highest_confidence",
                "extreme_mean_delta_points",
            ]
        )
    print_table(summary_rows, fieldnames)

    if args.output_csv:
        write_csv(Path(args.output_csv), summary_rows, fieldnames)
        print(f"\nSaved CSV: {args.output_csv}")

    if args.output_json:
        payload: dict[str, object] = {"summary": summary_rows}
        if args.include_samples:
            payload["samples"] = sample_rows
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
