#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _suite_registry import select_presets
    from _suite_utils import (
        REPO_ROOT,
        python_executable,
        resolve_result_root,
        run_or_print,
        write_json,
    )
except ImportError:  # pragma: no cover
    from scripts._suite_registry import select_presets
    from scripts._suite_utils import (
        REPO_ROOT,
        python_executable,
        resolve_result_root,
        run_or_print,
        write_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate standardized classification suites.")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    parser.add_argument("--python", default=None)
    parser.add_argument("--results-root-base", default=None)
    parser.add_argument("--skip-mufidelity", action="store_true")
    parser.add_argument("--mu-limit", type=int, default=None)
    parser.add_argument("--mu-grid-size", type=int, default=9)
    parser.add_argument("--mu-subset-percent", type=float, default=0.2)
    parser.add_argument("--mu-nb-samples", type=int, default=200)
    parser.add_argument(
        "--mu-batch-size",
        type=int,
        default=8,
        help="Model batch size passed to MuFidelity (default: 8).",
    )
    parser.add_argument("--mu-sample-batch-size", type=int, default=8)
    parser.add_argument("--mu-baseline", default="0.0")
    parser.add_argument("--mu-score-key", default="insertion_score")
    parser.add_argument("--mu-device", default="cuda")
    parser.add_argument("--mu-tf-device", default="cpu")
    parser.add_argument("--mu-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python_bin = python_executable(args.python)
    suite_summary: dict[str, dict] = {}

    for preset in select_presets("classification", args.model):
        result_root = resolve_result_root(preset, args.results_root_base)
        cmd = [
            python_bin,
            str(REPO_ROOT / preset.evaluation_script),
            "--results-dir",
            str(result_root),
        ]
        if not args.skip_mufidelity:
            cmd.extend(
                [
                    "--mufidelity",
                    "--mu-grid-size",
                    str(args.mu_grid_size),
                    "--mu-subset-percent",
                    str(args.mu_subset_percent),
                    "--mu-nb-samples",
                    str(args.mu_nb_samples),
                    "--mu-batch-size",
                    str(args.mu_batch_size),
                    "--mu-sample-batch-size",
                    str(args.mu_sample_batch_size),
                    "--mu-baseline",
                    str(args.mu_baseline),
                    "--mu-score-key",
                    str(args.mu_score_key),
                    "--mu-device",
                    str(args.mu_device),
                    "--mu-tf-device",
                    str(args.mu_tf_device),
                    "--mu-seed",
                    str(args.mu_seed),
                ]
            )
            if args.mu_limit is not None:
                cmd.extend(["--mu-limit", str(args.mu_limit)])
            if "clip" in preset.model_key:
                cmd.extend(["--mu-model-family", "clip"])
                clip_type = "RN101" if preset.model_key == "clip_rn101" else "ViT-L/14"
                cmd.extend(["--mu-clip-type", clip_type])
            else:
                cmd.extend(["--mu-model-family", "torchvision", "--mu-arch", preset.model_key])

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

    out_root = Path(args.results_root_base) if args.results_root_base else (REPO_ROOT / "classification_results")
    write_json(out_root / "eval_suite_summary.json", suite_summary)


if __name__ == "__main__":
    main()
