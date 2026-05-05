# -*- coding: utf-8 -*-
"""
Unified gradient / saliency-map attribution workflow.

Gradient-family methods produce a pixel saliency map with a small number of
backward passes.  This explainer converts that map into the same ordered-region
representation used by the search pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from attribution_research.adapters.base import ModelAdapter
from attribution_research.adapters.gradient import GradientAdapter
from attribution_research.methods.gradient.map_based import SaliencyMapExplainer
from attribution_research.segmentation.base import RegionSet


class GradientExplainer:
    """
    Generate a saliency map with a GradientAdapter, then optionally replay it
    through a black-box evaluator adapter for standard insertion/deletion logs.
    """

    def __init__(
        self,
        adapter: GradientAdapter,
        evaluator: Optional[ModelAdapter] = None,
        reduction: str = "mean",
        descending: bool = True,
    ):
        self.adapter = adapter
        self.evaluator = evaluator
        self.reduction = reduction
        self.descending = descending
        self._map_explainer = (
            SaliencyMapExplainer(
                evaluator=evaluator,
                reduction=reduction,
                descending=descending,
            )
            if evaluator is not None
            else None
        )

    def saliency_map(
        self,
        image: np.ndarray,
        target: Any,
        **kwargs,
    ) -> np.ndarray:
        """Generate one saliency map for one image/target pair."""
        self.adapter.setup(image, target, **kwargs)
        try:
            saliency = np.asarray(self.adapter.saliency_map(**kwargs), dtype=np.float32)
        finally:
            self.adapter.teardown()

        if saliency.ndim == 3 and saliency.shape[2] == 1:
            saliency = saliency[:, :, 0]
        if saliency.ndim != 2:
            raise ValueError(f"gradient saliency_map must be 2-D, got {saliency.shape}")
        return saliency

    def __call__(
        self,
        image: np.ndarray,
        regions: Optional[RegionSet],
        target: Any,
        **kwargs,
    ):
        """
        If `regions` is None, return only the saliency map.
        Otherwise return `(ordered_masks, json_dict, saliency_map)`.
        """
        saliency = self.saliency_map(
            image=image,
            target=target,
            **kwargs,
        )
        if regions is None:
            return saliency

        if self._map_explainer is not None:
            ordered_masks, json_dict = self._map_explainer.explain_from_map(
                image=image,
                regions=regions,
                saliency_map=saliency,
                target=target,
                **kwargs,
            )
        else:
            region_scores = regions.region_scores(saliency, reduction=self.reduction)
            ordered_masks = regions.ordered_masks_from_scores(
                region_scores,
                descending=self.descending,
            )
            json_dict = {
                "region_saliency_score": region_scores,
                "saliency_reduction": self.reduction,
                "evaluation_skipped": True,
            }

        json_dict["method"] = getattr(self.adapter, "method_name", "gradient")
        return ordered_masks, json_dict, saliency
