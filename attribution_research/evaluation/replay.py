# -*- coding: utf-8 -*-
"""
Common helpers shared by multiple attribution families.

These utilities make it possible to:
1. turn a pixel saliency map into an ordered region ranking; and
2. evaluate any precomputed ordered mask sequence with the same insertion /
   deletion logging format used by greedy and PhaseWin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from attribution_research.adapters.base import ModelAdapter
from attribution_research.segmentation.base import RegionSet


def order_masks_by_saliency(
    regions: RegionSet,
    saliency_map: np.ndarray,
    reduction: str = "mean",
    descending: bool = True,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Rank regions by a pixel saliency map.

    Returns
    -------
    ordered_masks : list[(H, W, 1) uint8]
    scores        : list[float] region-level scores in region order
    """
    scores = regions.region_scores(saliency_map, reduction=reduction)
    ordered = regions.ordered_masks_from_scores(scores, descending=descending)
    return ordered, scores


def evaluate_ordered_masks(
    adapter: ModelAdapter,
    image: np.ndarray,
    ordered_masks: Sequence[np.ndarray],
    target: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate an externally produced ordered mask sequence.

    This is used by map-based algorithms such as gradient methods or D-RISE:
    the algorithm produces a saliency map or ranking, and this helper replays
    that ranking through the task adapter to export the standard per-step JSON.
    """
    h, w = image.shape[:2]
    baseline = np.zeros((h, w, 1), dtype=np.uint8)
    region_area = float(h * w)

    insertion_scores: List[float] = []
    deletion_scores: List[float] = []
    gain_scores: List[float] = []
    region_areas: List[float] = []
    extras: List[Dict[str, Any]] = []

    adapter.setup(image, target, **kwargs)
    try:
        for raw_mask in ordered_masks:
            mask = np.asarray(raw_mask)
            if mask.ndim == 2:
                mask = mask[:, :, None]
            mask = (mask > 0).astype(np.uint8)

            detail = adapter.score_single_detailed(mask, baseline)
            insertion_scores.append(detail.get("insertion_score", detail.get("smdl_score", 0.0)))
            deletion_scores.append(detail.get("deletion_score", 1.0))
            gain_scores.append(detail.get("smdl_score", 0.0))
            extras.append(detail)

            baseline = np.clip(
                baseline.astype(np.int32) + mask.astype(np.int32), 0, 1
            ).astype(np.uint8)
            region_areas.append(float(baseline.sum()) / region_area)
    finally:
        adapter.teardown()

    result: Dict[str, Any] = {
        "insertion_score": insertion_scores,
        "deletion_score": deletion_scores,
        "smdl_score": gain_scores,
        "region_area": region_areas,
        "sub-region_number": len(ordered_masks),
    }
    if insertion_scores:
        result["org_score"] = insertion_scores[-1]
        result["baseline_score"] = deletion_scores[-1]
    if gain_scores:
        result["smdl_score_max"] = max(gain_scores)
        result["smdl_score_max_index"] = gain_scores.index(max(gain_scores))

    extra_keys = set()
    for detail in extras:
        extra_keys.update(detail.keys())
    for key in extra_keys:
        if key not in ("insertion_score", "deletion_score", "smdl_score"):
            result[key] = [detail.get(key) for detail in extras]
    return result
