#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print the merged baseline inventory grouped by task / family / category / source / support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attribution_research.baselines import (
    SUPPORTED_BASELINE_GROUPS,
    SUPPORTED_BASELINE_STATUSES,
    group_baselines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List baseline inventory for the current repository")
    parser.add_argument(
        "--group-by",
        choices=SUPPORTED_BASELINE_GROUPS,
        default="task",
        help="Grouping axis for the printed inventory",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task filter: classification | detection | caption_vqa",
    )
    parser.add_argument(
        "--support",
        default="native,catalog",
        help="Comma-separated support filter. Default: native,catalog",
    )
    parser.add_argument(
        "--runnable-only",
        action="store_true",
        help="Show only baselines that currently have runnable CLI support in this repo",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print one summary line per baseline",
    )
    return parser.parse_args()


def _parse_support(csv: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(csv).split(",") if item.strip())
    if not values:
        raise ValueError("At least one support status is required")
    for value in values:
        if value not in SUPPORTED_BASELINE_STATUSES:
            raise ValueError(
                f"Unsupported support status: {value!r}. "
                f"Expected one of: {', '.join(SUPPORTED_BASELINE_STATUSES)}"
            )
    return values


def main() -> None:
    args = parse_args()
    support = _parse_support(args.support)
    grouped = group_baselines(
        group_by=args.group_by,
        task=args.task,
        support=support,
        runnable_only=args.runnable_only,
    )

    if not grouped:
        print("No baselines matched the requested filters.")
        return

    for group_name, specs in grouped.items():
        print(f"[{args.group_by}={group_name}]")
        for spec in specs:
            sources = ",".join(spec.sources)
            tasks = ",".join(spec.tasks)
            print(
                f"- {spec.name}"
                f" | support={spec.support}"
                f" | family={spec.family}"
                f" | category={spec.category}"
                f" | tasks={tasks}"
                f" | sources={sources}"
            )
            if args.details:
                print(f"  {spec.summary}")
        print()


if __name__ == "__main__":
    main()
