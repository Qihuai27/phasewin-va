#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_PATH = SCRIPT_DIR.parent
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

try:
    from _suite_registry import select_presets
    from _suite_utils import (
        REPO_ROOT,
        ordered_eval_ids,
        parse_algorithms,
        resolve_result_root,
        resolve_visualization_root,
        select_method_dirs,
        shared_result_ids,
        write_json,
    )
except ImportError:  # pragma: no cover
    from scripts._suite_registry import select_presets
    from scripts._suite_utils import (
        REPO_ROOT,
        ordered_eval_ids,
        parse_algorithms,
        resolve_result_root,
        resolve_visualization_root,
        select_method_dirs,
        shared_result_ids,
        write_json,
    )
from attribution_research.io.results import load_result
from attribution_research.visualization import render_caption_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render caption attribution reports.")
    parser.add_argument("--model", required=True, help="Caption model key.")
    parser.add_argument("--results-dir", default=None, help="Optional explicit result root.")
    parser.add_argument("--eval-list", default=None, help="Optional explicit eval list.")
    parser.add_argument("--methods", default=None, help="Comma-separated method list.")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    return parser


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else (REPO_ROOT / image_path)


def main() -> None:
    args = build_parser().parse_args()
    preset = select_presets("caption", args.model)[0]
    methods = parse_algorithms(args.methods, preset.methods)
    result_root = Path(args.results_dir) if args.results_dir else resolve_result_root(preset)
    output_root = Path(args.output_dir) if args.output_dir else resolve_visualization_root(preset)

    method_dirs = select_method_dirs(result_root, methods)
    if not method_dirs:
        raise FileNotFoundError(f"No completed caption result dirs found under {result_root}")

    ordered_ids = ordered_eval_ids(preset, eval_list_override=args.eval_list)
    sample_ids = shared_result_ids(method_dirs, ordered_ids, args.sample_count)
    if not sample_ids:
        raise RuntimeError(f"No shared sample ids found across methods in {result_root}")

    manifest = {
        "task": "caption",
        "model": preset.model_key,
        "display_name": preset.display_name,
        "result_root": str(result_root),
        "methods": {name: str(path) for name, path in method_dirs.items()},
        "sample_ids": sample_ids,
    }

    for method_name, method_dir in method_dirs.items():
        method_output = output_root / method_name
        for sample_id in sample_ids:
            masks, info = load_result(str(method_dir), sample_id)
            image_path = _resolve_image_path(info["image_path"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read source image for visualization: {image_path}")

            render_caption_report(
                image=image,
                masks=masks,
                info=info,
                sample_id=sample_id,
                method_name=method_name,
                model_label=preset.display_name,
                output_path=method_output / f"{sample_id}.png",
            )

    write_json(output_root / "manifest.json", manifest)
    print(f"Saved caption visualizations to {output_root}")


if __name__ == "__main__":
    main()
