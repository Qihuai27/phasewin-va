#!/usr/bin/env python3
from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Run standardized caption suites by model.")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    parser.add_argument("--algorithms", default=None, help="Comma-separated algorithm override.")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--eval-list", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--source-eval-list", default=None)
    parser.add_argument("--save-root-base", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--build-eval-list-if-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python_bin = python_executable(args.python)

    for preset in select_presets("caption", args.model):
        algorithms = parse_algorithms(args.algorithms, preset.methods)
        begin = args.begin
        end = args.end
        if begin is None and end is None and args.num_shards > 1:
            total = count_eval_items(preset, eval_list_override=args.eval_list)
            begin, end = shard_bounds(total, args.num_shards, args.shard_index)
            print(
                f"[suite] {preset.model_key}: shard {args.shard_index + 1}/{args.num_shards} "
                f"-> [{begin}, {end}) / {total}"
            )

        cmd = [
            "bash",
            str(REPO_ROOT / preset.runner_script),
            *preset.runner_args,
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
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        elif preset.model_name:
            cmd.extend(["--model-name", preset.model_name])
        if args.source_eval_list:
            cmd.extend(["--source-eval-list", args.source_eval_list])
        if args.save_root_base:
            cmd.extend(["--save-root", str(resolve_result_root(preset, args.save_root_base))])
        if args.max_new_tokens is not None:
            cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.build_eval_list_if_missing:
            cmd.append("--build-eval-list-if-missing")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"[suite] running {preset.display_name}")
        run_or_print(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
