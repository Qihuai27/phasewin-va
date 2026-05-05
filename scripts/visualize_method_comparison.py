#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_PATH = SCRIPT_DIR.parent
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))
MPLCONFIGDIR = REPO_ROOT_PATH / "tmp" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

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
from attribution_research.visualization import build_attribution_map


TASK_CMAPS = {
    "classification": "magma",
    "caption": "viridis",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render same-sample, multi-method attribution comparison figures."
    )
    parser.add_argument("--task", required=True, choices=("classification", "caption"))
    parser.add_argument("--model", required=True, help="Model key from scripts/_suite_registry.py.")
    parser.add_argument("--results-dir", default=None, help="Optional explicit result root.")
    parser.add_argument("--eval-list", default=None, help="Optional explicit eval list.")
    parser.add_argument("--methods", default=None, help="Comma-separated method list.")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--columns", type=int, default=4, help="Number of panels per row.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay alpha.")
    parser.add_argument(
        "--score-mode",
        default="positive_delta",
        choices=("raw", "delta", "positive_delta"),
        help="How cumulative insertion scores are converted to per-region weights.",
    )
    return parser


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _method_label(method: str) -> str:
    labels = {
        "grad_eclip": "Grad-ECLIP",
        "igos_pp": "IGOS++",
        "llavacam": "LLaVA-CAM",
        "phasewin": "PhaseWin",
        "dhsic": "D-HSIC",
        "drise": "D-RISE",
        "ig2": "IG2",
    }
    return labels.get(method, method.replace("_", "-").title())


def _caption_text(info: dict, *, max_chars: int = 150) -> str:
    words = info.get("words", [])
    if not words:
        return ""
    caption = "".join(str(word) for word in words).strip()
    if len(caption) <= max_chars:
        return caption
    return caption[: max_chars - 3].rstrip() + "..."


def _task_metadata(task: str, info: dict) -> str:
    if task == "classification":
        target = info.get("target_label")
        return f"target_label={target}" if target is not None else ""
    return _caption_text(info)


def _attribution_map(masks: np.ndarray, info: dict, *, score_mode: str) -> np.ndarray:
    attr_map = build_attribution_map(
        masks,
        info["insertion_score"],
        normalize=True,
        score_mode=score_mode,
        baseline_score=info.get("baseline_score"),
    )
    return np.clip(np.asarray(attr_map, dtype=np.float32), 0.0, 1.0)


def _overlay_rgb(image_bgr: np.ndarray, attr_map: np.ndarray, *, cmap_name: str, alpha: float) -> np.ndarray:
    import matplotlib

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if attr_map.shape != image_bgr.shape[:2]:
        attr_map = cv2.resize(attr_map, (image_bgr.shape[1], image_bgr.shape[0]))
    heat_rgb = matplotlib.colormaps[cmap_name](np.clip(attr_map, 0.0, 1.0))[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * image_rgb + alpha * heat_rgb, 0.0, 1.0)


def _panel_title(method: str, info: dict) -> str:
    calls = info.get("model_forward_calls", info.get("marginal_calls"))
    score = info.get("smdl_score_max")
    parts = [_method_label(method)]
    if score is not None:
        parts.append(f"peak={float(score):.3f}")
    if calls is not None:
        parts.append(f"calls={int(calls)}")
    return "\n".join(parts)


def render_comparison_figure(
    *,
    task: str,
    image: np.ndarray,
    method_results: dict[str, tuple[np.ndarray, dict]],
    sample_id: str,
    model_label: str,
    output_path: Path,
    columns: int,
    alpha: float,
    score_mode: str,
) -> Path:
    import matplotlib.pyplot as plt

    panels = [("original", None, next(iter(method_results.values()))[1])]
    panels.extend((method, payload, payload[1]) for method, payload in method_results.items())

    columns = max(1, int(columns))
    rows = int(math.ceil(len(panels) / columns))
    fig_width = max(4.0 * columns, 8.0)
    fig_height = 3.9 * rows + (1.0 if task == "caption" else 0.45)
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)
    cmap_name = TASK_CMAPS.get(task, "magma")

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (method, payload, info) in zip(axes.ravel(), panels):
        if payload is None:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            title = "Original"
        else:
            masks, method_info = payload
            attr_map = _attribution_map(masks, method_info, score_mode=score_mode)
            ax.imshow(_overlay_rgb(image, attr_map, cmap_name=cmap_name, alpha=alpha))
            title = _panel_title(method, method_info)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off")

    metadata = _task_metadata(task, next(iter(method_results.values()))[1])
    title = f"{model_label} | {sample_id}"
    if metadata:
        title += f"\n{textwrap.fill(metadata, width=110)}"
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    preset = select_presets(args.task, args.model)[0]
    requested_methods = parse_algorithms(args.methods, preset.methods)
    result_root = Path(args.results_dir) if args.results_dir else resolve_result_root(preset)
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_visualization_root(preset) / "method_comparison"
    )

    method_dirs = select_method_dirs(result_root, requested_methods)
    if not method_dirs:
        raise FileNotFoundError(f"No completed result dirs found under {result_root}")

    ordered_ids = ordered_eval_ids(preset, eval_list_override=args.eval_list)
    sample_ids = shared_result_ids(method_dirs, ordered_ids, args.sample_count)
    if not sample_ids:
        raise RuntimeError(f"No shared sample ids found across methods in {result_root}")

    missing_methods = [method for method in requested_methods if method not in method_dirs]
    manifest = {
        "task": args.task,
        "model": preset.model_key,
        "display_name": preset.display_name,
        "result_root": str(result_root),
        "output_root": str(output_root),
        "requested_methods": list(requested_methods),
        "missing_methods": missing_methods,
        "methods": {name: str(path) for name, path in method_dirs.items()},
        "sample_ids": sample_ids,
        "figures": [],
        "score_mode": args.score_mode,
        "alpha": args.alpha,
    }

    for sample_id in sample_ids:
        method_results: dict[str, tuple[np.ndarray, dict]] = {}
        for method_name, method_dir in method_dirs.items():
            method_results[method_name] = load_result(str(method_dir), sample_id)

        first_info = next(iter(method_results.values()))[1]
        image_path = _resolve_image_path(first_info["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read source image for visualization: {image_path}")

        output_path = output_root / f"{sample_id}.png"
        render_comparison_figure(
            task=args.task,
            image=image,
            method_results=method_results,
            sample_id=sample_id,
            model_label=preset.display_name,
            output_path=output_path,
            columns=args.columns,
            alpha=args.alpha,
            score_mode=args.score_mode,
        )
        manifest["figures"].append(str(output_path))

    write_json(output_root / "manifest.json", manifest)
    print(
        f"Saved {len(sample_ids)} {args.task} comparison figures to {output_root} "
        f"using methods: {', '.join(method_dirs)}"
    )
    if missing_methods:
        print(f"Skipped missing methods: {', '.join(missing_methods)}")


if __name__ == "__main__":
    main()
