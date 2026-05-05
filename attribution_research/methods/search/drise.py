# -*- coding: utf-8 -*-
"""
D-RISE style random-mask attribution.

This implementation keeps the original Monte-Carlo spirit: sample many smooth
grid masks, score the masked images, accumulate a pixel saliency map, then
optionally convert that map into an ordered region explanation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from attribution_research.adapters.base import ModelAdapter
from attribution_research.methods.gradient.map_based import SaliencyMapExplainer
from attribution_research.segmentation.base import RegionSet


def generate_random_grid_mask(
    image_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    prob_thresh: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a smooth D-RISE mask in [0, 1].

    Parameters
    ----------
    image_size : (width, height)
    grid_size  : (grid_w, grid_h)
    """
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w = int(math.ceil(image_w / grid_w))
    cell_h = int(math.ceil(image_h / grid_h))
    up_w = (grid_w + 1) * cell_w
    up_h = (grid_h + 1) * cell_h

    coarse = (rng.uniform(0, 1, size=(grid_h, grid_w)) < prob_thresh).astype(np.float32)
    mask = cv2.resize(coarse, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = int(rng.integers(0, cell_w))
    offset_h = int(rng.integers(0, cell_h))
    mask = mask[offset_h : offset_h + image_h, offset_w : offset_w + image_w]
    return mask[:, :, None].astype(np.float32)


class DRISEExplainer:
    """
    Monte-Carlo random-mask attribution.

    Parameters
    ----------
    adapter : ModelAdapter
        Task-specific scoring adapter.
    n_masks : int
        Number of random masks to sample.
    grid_size : tuple[int, int]
        Coarse mask grid shape.
    prob_thresh : float
        Cell activation probability.
    batch_size : int
        Number of masks scored per batch.
    score_key : str
        Key from ``score_batch_detailed`` used to weight masks.  The classic
        D-RISE choice is ``insertion_score``.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        n_masks: int = 1000,
        grid_size: Tuple[int, int] = (16, 16),
        prob_thresh: float = 0.5,
        batch_size: int = 32,
        score_key: str = "insertion_score",
        reduction: str = "mean",
        descending: bool = True,
        rng_seed: int = 0,
    ):
        self.adapter = adapter
        self.n_masks = int(n_masks)
        self.grid_size = tuple(grid_size)
        self.prob_thresh = float(prob_thresh)
        self.batch_size = int(batch_size)
        self.score_key = score_key
        self._map_explainer = SaliencyMapExplainer(
            evaluator=adapter,
            reduction=reduction,
            descending=descending,
        )
        self._rng_seed = int(rng_seed)

    def saliency_map(
        self,
        image: np.ndarray,
        target: Any,
        show_progress: bool = True,
        **kwargs,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        baseline = np.zeros((h, w, 1), dtype=np.float32)
        saliency = np.zeros((h, w), dtype=np.float32)
        rng = np.random.default_rng(self._rng_seed)

        self.adapter.reset_forward_counter()
        self.adapter.setup(image, target, **kwargs)
        try:
            iterator = range(0, self.n_masks, self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="D-RISE")

            for start in iterator:
                count = min(self.batch_size, self.n_masks - start)
                masks = np.stack(
                    [
                        generate_random_grid_mask(
                            image_size=(w, h),
                            grid_size=self.grid_size,
                            prob_thresh=self.prob_thresh,
                            rng=rng,
                        )
                        for _ in range(count)
                    ],
                    axis=0,
                )
                details = self.adapter.score_batch_detailed(masks, baseline)
                if self.score_key in details:
                    scores = details[self.score_key].detach().cpu().numpy().astype(np.float32)
                else:
                    scores = self.adapter.score_batch(masks, baseline).detach().cpu().numpy().astype(np.float32)
                saliency += (masks[:, :, :, 0] * scores[:, None, None]).sum(axis=0)
        finally:
            self.adapter.teardown()

        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency

    def __call__(
        self,
        image: np.ndarray,
        regions: Optional[RegionSet],
        target: Any,
        show_progress: bool = True,
        **kwargs,
    ):
        saliency = self.saliency_map(
            image=image,
            target=target,
            show_progress=show_progress,
            **kwargs,
        )
        algorithm_forward_calls = self.adapter.model_forward_calls
        if regions is None:
            return saliency
        ordered_masks, json_dict = self._map_explainer.explain_from_map(
            image=image,
            regions=regions,
            saliency_map=saliency,
            target=target,
            **kwargs,
        )
        json_dict["method"] = "drise"
        json_dict["drise_grid_size"] = list(self.grid_size)
        json_dict["drise_n_masks"] = self.n_masks
        json_dict["drise_prob_thresh"] = self.prob_thresh
        json_dict["drise_score_key"] = self.score_key
        total_forward_calls = self.adapter.model_forward_calls
        json_dict["model_forward_calls"] = algorithm_forward_calls
        json_dict["saliency_model_forward_calls"] = algorithm_forward_calls
        json_dict["eval_model_forward_calls"] = total_forward_calls - algorithm_forward_calls
        json_dict["total_model_forward_calls"] = total_forward_calls
        json_dict["model_forward_count_mode"] = "equivalent_single_image_forwards"
        json_dict["model_forward_count_scope"] = "algorithm_only"
        return ordered_masks, json_dict, saliency
