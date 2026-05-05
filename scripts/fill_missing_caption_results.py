#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill missing caption attribution results for the default Qwen2.5-VL run.

The script scans the existing result directories under
``caption_results/Qwen2.5-VL-3B-coco-caption``, compares them against the eval
list, and reruns only the missing sample ranges for each algorithm.

It is intended to patch incomplete historical runs such as:
- the single sample missing from all methods; and
- the extra missing samples in ``gradient`` and ``llavacam``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_ALGORITHMS = ["greedy", "phasewin", "drise", "gradient", "llavacam"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill missing caption attribution results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="./caption_results/Qwen2.5-VL-3B-coco-caption",
        help="Base directory containing per-method result subdirectories.",
    )
    parser.add_argument(
        "--eval-list",
        default="datasets/Qwen2.5-VL-3B-coco-caption.json",
        help="Caption eval-list JSON used by the task script.",
    )
    parser.add_argument(
        "--task-script",
        default="tasks/caption_vqa/qwen25vl_coco_caption.py",
        help="Task entrypoint to invoke for backfilling.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to launch the task script.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device forwarded to the task script.",
    )
    parser.add_argument(
        "--datasets",
        default="datasets/coco/val2017",
        help="Image root forwarded to the task script.",
    )
    parser.add_argument(
        "--model-name",
        default="model_checkpoint/Qwen2.5-VL-3B-Instruct",
        help="Qwen2.5-VL checkpoint path forwarded to the task script.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=DEFAULT_ALGORITHMS,
        help="Algorithms to backfill.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def load_eval_ids(eval_list: Path) -> list[str]:
    with open(eval_list, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [Path(item["image_path"]).stem for item in items]


def resolve_method_dir(results_dir: Path, algorithm: str) -> Path:
    matches = sorted(
        path for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"{algorithm}-")
    )
    if not matches:
        raise FileNotFoundError(
            f"No result directory found for algorithm {algorithm!r} under {results_dir}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Multiple result directories found for algorithm {algorithm!r}: {names}"
        )
    return matches[0]


def existing_result_ids(method_dir: Path) -> set[str]:
    json_dir = method_dir / "json"
    npy_dir = method_dir / "npy"
    if not json_dir.is_dir() or not npy_dir.is_dir():
        return set()

    json_ids = {path.stem for path in json_dir.glob("*.json")}
    npy_ids = {path.stem for path in npy_dir.glob("*.npy")}
    return json_ids & npy_ids


def missing_indices(eval_ids: list[str], present_ids: set[str]) -> list[int]:
    return [idx for idx, image_id in enumerate(eval_ids) if image_id not in present_ids]


def contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = prev = idx
    ranges.append((start, prev + 1))
    return ranges


def build_command(
    args: argparse.Namespace,
    algorithm: str,
    begin: int,
    end: int,
) -> list[str]:
    return [
        args.python_bin,
        str((REPO_ROOT / args.task_script).resolve()),
        "--algorithm", algorithm,
        "--segmenter", "superpixel",
        "--superpixel-algorithm", "slico",
        "--division-number", "64",
        "--lambda1", "1.0",
        "--lambda2", "1.0",
        "--datasets", str((REPO_ROOT / args.datasets).resolve()),
        "--eval-list", str((REPO_ROOT / args.eval_list).resolve()),
        "--model-name", str((REPO_ROOT / args.model_name).resolve()),
        "--save-dir", str((REPO_ROOT / args.results_dir).resolve()),
        "--device", args.device,
        "--begin", str(begin),
        "--end", str(end),
        "--show-progress",
    ]


def print_command(command: list[str]) -> None:
    print("  $ " + " ".join(subprocess.list2cmdline([part]) for part in command))


def main() -> int:
    args = parse_args()

    results_dir = (REPO_ROOT / args.results_dir).resolve()
    eval_list = (REPO_ROOT / args.eval_list).resolve()

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not eval_list.is_file():
        raise FileNotFoundError(f"Eval-list not found: {eval_list}")

    eval_ids = load_eval_ids(eval_list)
    total_missing = 0
    planned_commands: list[tuple[str, list[str]]] = []

    print(f"Eval list size: {len(eval_ids)}")
    print(f"Results root  : {results_dir}")

    for algorithm in args.algorithms:
        method_dir = resolve_method_dir(results_dir, algorithm)
        present_ids = existing_result_ids(method_dir)
        missing = missing_indices(eval_ids, present_ids)
        ranges = contiguous_ranges(missing)
        total_missing += len(missing)

        print()
        print(f"{algorithm}:")
        print(f"  method dir : {method_dir.name}")
        print(f"  present    : {len(present_ids)}")
        print(f"  missing    : {len(missing)}")
        if missing:
            print(f"  ranges     : {ranges}")
            for begin, end in ranges:
                planned_commands.append((algorithm, build_command(args, algorithm, begin, end)))
        else:
            print("  ranges     : []")

    if not planned_commands:
        print()
        print("Nothing to backfill.")
        return 0

    print()
    print(f"Planned commands: {len(planned_commands)}  missing samples: {total_missing}")
    for algorithm, command in planned_commands:
        print(f"[{algorithm}]")
        print_command(command)

    if args.dry_run:
        return 0

    for algorithm, command in planned_commands:
        print()
        print(f"Running [{algorithm}]")
        print_command(command)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
