# -*- coding: utf-8 -*-
"""
Runtime helpers for binding task-specific adapters to method families.

The task entrypoints should only provide:
1. task-specific data loading;
2. task-specific adapter builders; and
3. result serialization.

Algorithm-specific dispatch lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from attribution_research.adapters.base import ModelAdapter
from attribution_research.adapters.gradient import GradientAdapter
from attribution_research.methods.gradient import GradientExplainer
from attribution_research.methods.search import (
    DHSICExplainer,
    DRISEExplainer,
    GreedyExplainer,
    PhaseWinExplainer,
)
from attribution_research.registry import normalize_algorithm_name
from attribution_research.segmentation.base import RegionSet


_MAP_REPLAY_ALGORITHMS = {
    "gradient",
    "grad_eclip",
    "ig2",
    "igos_pp",
    "gradcam",
    "odam",
    "ssgrad_cam_pp",
    "llavacam",
    "input_grad",
    "dhsic",
}


@dataclass
class AttributionContext:
    """Per-sample context needed to run one attribution method."""

    args: Any
    image: np.ndarray
    regions: RegionSet
    target: Any
    build_search_adapter: Optional[Callable[[], ModelAdapter]] = None
    build_gradient_adapter: Optional[Callable[[], GradientAdapter]] = None
    build_dhsic_model: Optional[Callable[[], Any]] = None


def build_run_tag(args) -> str:
    """Build the method-specific output directory tag for one run."""
    from attribution_research.composition import segmenter_tag_from_args

    def _phasewin_window_tag() -> str:
        window_size = getattr(args, "window_size", None)
        if window_size is not None:
            return f"window-{int(window_size)}"
        frac = float(getattr(args, "phasewin_window_frac", 0.3))
        pct = int(round(frac * 100))
        if abs(frac - (pct / 100.0)) < 1e-9:
            return f"window-pct-{pct}"
        return f"window-frac-{str(frac).replace('.', 'p')}"

    tag = segmenter_tag_from_args(args)
    algo = normalize_algorithm_name(args.algorithm)
    if algo == "drise":
        return f"{algo}-{tag}-{args.lambda1}-{args.lambda2}-nmasks-{args.drise_n_masks}"
    if algo == "dhsic":
        tf_device = getattr(args, "dhsic_tf_device", "cpu")
        return f"{algo}-{tag}-batch-{args.dhsic_batch_size}-tf-{tf_device}"
    if algo == "phasewin":
        return f"{algo}-{tag}-{args.lambda1}-{args.lambda2}-{_phasewin_window_tag()}"
    return f"{algo}-{tag}-{args.lambda1}-{args.lambda2}"


def build_save_dir(base_dir: str, args) -> str:
    """Return the output directory for the current run."""
    from attribution_research.io.results import mkdir

    save_dir = str(base_dir)
    out = f"{save_dir.rstrip('/')}/{build_run_tag(args)}"
    mkdir(out)
    return out


def execute_attribution(context: AttributionContext) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Run one method on one image using task-specific adapter builders."""
    algo = normalize_algorithm_name(context.args.algorithm)
    if algo in _MAP_REPLAY_ALGORITHMS:
        return _run_map_replay(context)
    return _run_search(context)


def _require_builder(builder: Optional[Callable], label: str) -> Callable:
    if builder is None:
        raise ValueError(f"Missing builder for {label}")
    return builder


def _model_forward_calls(*objects: Any) -> int:
    total = 0
    for obj in objects:
        total += int(getattr(obj, "model_forward_calls", 0))
    return total


def _run_search(context: AttributionContext) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    args = context.args
    algo = normalize_algorithm_name(args.algorithm)
    adapter = _require_builder(context.build_search_adapter, "search adapter")()

    if algo == "greedy":
        explainer = GreedyExplainer(adapter, lambda1=args.lambda1, lambda2=args.lambda2)
        return explainer(
            image=context.image,
            masks=context.regions.binary_masks(),
            target=context.target,
            show_progress=getattr(args, "show_progress", False),
        )

    if algo == "phasewin":
        explainer = PhaseWinExplainer(
            adapter=adapter,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            model_type=args.model_type,
            n_greedy=args.n_greedy,
            pw_window_size=args.window_size,
            pw_window_frac=getattr(args, "phasewin_window_frac", 0.3),
            pw_beta_del=getattr(args, "phasewin_beta_del", 0.05),
            pw_alpha_sel=getattr(args, "phasewin_alpha_sel", 0.6),
            pw_random_frac=getattr(args, "phasewin_random_frac", 0.0),
            pw_window_policy=getattr(args, "phasewin_window_policy", "BA"),
            pw_enable_anneal=getattr(args, "phasewin_enable_anneal", True),
            pw_enable_hard_exit=getattr(args, "phasewin_enable_hard_exit", True),
            pw_hard_delta_thresh=args.hard_delta_thresh,
            pw_hard_phi_prev=args.hard_phi_prev,
        )
        return explainer(
            image=context.image,
            masks=context.regions.binary_masks(),
            target=context.target,
            show_progress=getattr(args, "show_progress", False),
        )

    if algo == "drise":
        explainer = DRISEExplainer(
            adapter=adapter,
            n_masks=args.drise_n_masks,
            grid_size=(args.drise_grid_cols, args.drise_grid_rows),
            prob_thresh=args.drise_prob_thresh,
            batch_size=getattr(args, "batch_size", 32),
            score_key=args.drise_score_key,
        )
        ordered_masks, json_dict, saliency = explainer(
            image=context.image,
            regions=context.regions,
            target=context.target,
            show_progress=getattr(args, "show_progress", False),
        )
        json_dict["saliency_map_max"] = float(np.max(saliency)) if saliency.size else 0.0
        return ordered_masks, json_dict

    raise ValueError(f"Unsupported search algorithm: {algo!r}")


def _run_map_replay(context: AttributionContext) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    args = context.args
    algo = normalize_algorithm_name(args.algorithm)
    evaluator = _require_builder(context.build_search_adapter, "search adapter")()

    if algo == "dhsic":
        model = _require_builder(context.build_dhsic_model, "dhsic model")()
        if hasattr(model, "reset_forward_counter"):
            model.reset_forward_counter()
        if hasattr(evaluator, "reset_forward_counter"):
            evaluator.reset_forward_counter()
        explainer = DHSICExplainer(
            model=model,
            num_classes=model.num_classes,
            evaluator=evaluator,
            batch_size=args.dhsic_batch_size,
            tf_device=getattr(args, "dhsic_tf_device", "cpu"),
        )
    else:
        grad_adapter = _require_builder(context.build_gradient_adapter, "gradient adapter")()
        if hasattr(grad_adapter, "reset_forward_counter"):
            grad_adapter.reset_forward_counter()
        if hasattr(evaluator, "reset_forward_counter"):
            evaluator.reset_forward_counter()
        explainer = GradientExplainer(
            adapter=grad_adapter,
            evaluator=evaluator,
        )

    ordered_masks, json_dict, saliency = explainer(
        image=context.image,
        regions=context.regions,
        target=context.target,
    )
    if algo == "dhsic":
        json_dict["dhsic_tf_device"] = getattr(args, "dhsic_tf_device", "cpu")
    elif algo == "igos_pp":
        saliency_forward_calls = _model_forward_calls(grad_adapter)
        eval_forward_calls = _model_forward_calls(evaluator)
        json_dict["model_forward_calls"] = saliency_forward_calls
        json_dict["saliency_model_forward_calls"] = saliency_forward_calls
        json_dict["eval_model_forward_calls"] = eval_forward_calls
        json_dict["total_model_forward_calls"] = saliency_forward_calls + eval_forward_calls
        json_dict["model_forward_count_mode"] = "equivalent_single_image_forwards"
        json_dict["model_forward_count_scope"] = "algorithm_only"
        json_dict["igos_mask_size"] = getattr(grad_adapter, "mask_size", None)
        json_dict["igos_steps"] = getattr(grad_adapter, "steps", None)
        json_dict["igos_lr"] = getattr(grad_adapter, "lr", None)
        json_dict["igos_blur_sigma"] = getattr(grad_adapter, "blur_sigma", None)
        json_dict["igos_preserve_coeff"] = getattr(grad_adapter, "preserve_coeff", None)
        json_dict["igos_delete_coeff"] = getattr(grad_adapter, "delete_coeff", None)
        json_dict["igos_area_coeff"] = getattr(grad_adapter, "area_coeff", None)
        json_dict["igos_tv_coeff"] = getattr(grad_adapter, "tv_coeff", None)
        json_dict["igos_binary_coeff"] = getattr(grad_adapter, "binary_coeff", None)
    json_dict["saliency_map_max"] = float(np.max(saliency)) if saliency.size else 0.0
    json_dict["map_method"] = algo
    return ordered_masks, json_dict
