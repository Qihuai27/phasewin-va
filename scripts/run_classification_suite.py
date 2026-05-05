#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _suite_registry import select_presets
    from _suite_utils import (
        REPO_ROOT,
        count_eval_items,
        parse_algorithms,
        python_executable,
        resolve_result_root,
        run_or_print,
        shard_bounds,
    )
except ImportError:  # pragma: no cover
    from scripts._suite_registry import select_presets
    from scripts._suite_utils import (
        REPO_ROOT,
        count_eval_items,
        parse_algorithms,
        python_executable,
        resolve_result_root,
        run_or_print,
        shard_bounds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standardized classification suites by model.")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    parser.add_argument("--split", default="true", choices=("true", "cause", "repair", "both"))
    parser.add_argument("--algorithms", default=None, help="Comma-separated algorithm override.")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--eval-list", default=None, help="Only used for split=true.")
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--save-root-base", default=None)
    parser.add_argument("--clip-download-root", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--igos-mask-size", type=int, default=None)
    parser.add_argument("--igos-steps", type=int, default=None)
    parser.add_argument("--igos-lr", type=float, default=None)
    parser.add_argument("--igos-blur-sigma", type=float, default=None)
    parser.add_argument("--igos-preserve-coeff", type=float, default=None)
    parser.add_argument("--igos-delete-coeff", type=float, default=None)
    parser.add_argument("--igos-area-coeff", type=float, default=None)
    parser.add_argument("--igos-tv-coeff", type=float, default=None)
    parser.add_argument("--igos-binary-coeff", type=float, default=None)
    parser.add_argument("--build-splits-if-missing", action="store_true")
    parser.add_argument("--build-semantic-if-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python_bin = python_executable(args.python)

    for preset in select_presets("classification", args.model):
        algorithms = parse_algorithms(args.algorithms, preset.methods)
        begin = args.begin
        end = args.end
        if begin is None and end is None and args.num_shards > 1:
            total = count_eval_items(
                preset,
                split=args.split,
                eval_list_override=args.eval_list,
            )
            begin, end = shard_bounds(total, args.num_shards, args.shard_index)
            print(
                f"[suite] {preset.model_key}: shard {args.shard_index + 1}/{args.num_shards} "
                f"-> [{begin}, {end}) / {total}"
            )

        cmd = [
            "bash",
            str(REPO_ROOT / preset.runner_script),
            *preset.runner_args,
            "--split",
            args.split,
            "--algorithms",
            ",".join(algorithms),
            "--python",
            python_bin,
            "--device",
            args.device,
        ]
        if begin is not None:
            cmd.extend(["--begin", str(begin)])
        if end is not None:
            cmd.extend(["--end", str(end)])
        if args.datasets:
            cmd.extend(["--datasets", args.datasets])
        if args.eval_list:
            cmd.extend(["--eval-list", args.eval_list])
        if args.generated_dir:
            cmd.extend(["--generated-dir", args.generated_dir])
        if args.save_root_base:
            cmd.extend(["--save-root", str(resolve_result_root(preset, args.save_root_base))])
        if args.clip_download_root:
            cmd.extend(["--clip-download-root", args.clip_download_root])
        if args.weights:
            cmd.extend(["--weights", args.weights])
        if args.igos_mask_size is not None:
            cmd.extend(["--igos-mask-size", str(args.igos_mask_size)])
        if args.igos_steps is not None:
            cmd.extend(["--igos-steps", str(args.igos_steps)])
        if args.igos_lr is not None:
            cmd.extend(["--igos-lr", str(args.igos_lr)])
        if args.igos_blur_sigma is not None:
            cmd.extend(["--igos-blur-sigma", str(args.igos_blur_sigma)])
        if args.igos_preserve_coeff is not None:
            cmd.extend(["--igos-preserve-coeff", str(args.igos_preserve_coeff)])
        if args.igos_delete_coeff is not None:
            cmd.extend(["--igos-delete-coeff", str(args.igos_delete_coeff)])
        if args.igos_area_coeff is not None:
            cmd.extend(["--igos-area-coeff", str(args.igos_area_coeff)])
        if args.igos_tv_coeff is not None:
            cmd.extend(["--igos-tv-coeff", str(args.igos_tv_coeff)])
        if args.igos_binary_coeff is not None:
            cmd.extend(["--igos-binary-coeff", str(args.igos_binary_coeff)])
        if args.build_splits_if_missing:
            cmd.append("--build-splits-if-missing")
        if args.build_semantic_if_missing:
            cmd.append("--build-semantic-if-missing")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"[suite] running {preset.display_name}")
        run_or_print(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
