#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _suite_registry import select_presets
    from _suite_utils import REPO_ROOT, python_executable, resolve_result_root, run_or_print, write_json
except ImportError:  # pragma: no cover
    from scripts._suite_registry import select_presets
    from scripts._suite_utils import REPO_ROOT, python_executable, resolve_result_root, run_or_print, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate standardized caption suites.")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    parser.add_argument("--python", default=None)
    parser.add_argument("--results-root-base", default=None)
    parser.add_argument("--sensitivity", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python_bin = python_executable(args.python)
    suite_summary: dict[str, dict] = {}

    for preset in select_presets("caption", args.model):
        result_root = resolve_result_root(preset, args.results_root_base)
        cmd = [
            python_bin,
            str(REPO_ROOT / preset.evaluation_script),
            "--results-dir",
            str(result_root),
            "--sensitivity",
            str(args.sensitivity),
        ]

        print(f"[suite] evaluating {preset.display_name}")
        run_or_print(cmd, dry_run=args.dry_run)
        if args.dry_run:
            continue

        summary_path = result_root / "eval_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Expected eval summary not found: {summary_path}")
        suite_summary[preset.model_key] = json.loads(summary_path.read_text(encoding="utf-8"))

    if args.dry_run:
        return

    out_root = Path(args.results_root_base) if args.results_root_base else (REPO_ROOT / "caption_results")
    write_json(out_root / "eval_suite_summary.json", suite_summary)


if __name__ == "__main__":
    main()
