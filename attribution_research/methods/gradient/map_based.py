# -*- coding: utf-8 -*-
"""
Map-based attribution workflow.

This bridges pixel saliency methods and region-based evaluation:
1. generate a saliency map;
2. aggregate it onto a RegionSet;
3. replay the resulting ordered masks through the task adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from attribution_research.adapters.base import ModelAdapter
from attribution_research.evaluation.replay import (
    evaluate_ordered_masks,
    order_masks_by_saliency,
)
from attribution_research.segmentation.base import RegionSet


class SaliencyMapExplainer:
    """
    Convert a saliency map into a region ranking and eval-compatible JSON.

    Parameters
    ----------
    evaluator : ModelAdapter
        Task-specific black-box evaluator used for insertion / deletion replay.
    reduction : str
        Region aggregation mode: "mean" | "sum" | "max".
    descending : bool
        Whether larger saliency scores indicate more important regions.
    """

    def __init__(
        self,
        evaluator: ModelAdapter,
        reduction: str = "mean",
        descending: bool = True,
    ):
        self.evaluator = evaluator
        self.reduction = reduction
        self.descending = descending

    def explain_from_map(
        self,
        image: np.ndarray,
        regions: RegionSet,
        saliency_map: np.ndarray,
        target: Any,
        **kwargs,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        ordered_masks, region_scores = order_masks_by_saliency(
            regions,
            saliency_map,
            reduction=self.reduction,
            descending=self.descending,
        )
        json_dict = evaluate_ordered_masks(
            self.evaluator,
            image,
            ordered_masks,
            target,
            **kwargs,
        )
        json_dict["region_saliency_score"] = region_scores
        json_dict["saliency_reduction"] = self.reduction
        return ordered_masks, json_dict
