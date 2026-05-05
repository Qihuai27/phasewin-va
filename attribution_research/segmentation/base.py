# -*- coding: utf-8 -*-
"""
Shared region and segmenter abstractions.

All algorithms in this repository ultimately operate on a collection of binary
regions.  Some methods consume those masks directly (greedy / PhaseWin),
and map-based methods need a stable way to aggregate pixel saliency into
region scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] != 1:
        raise ValueError(f"mask must have shape (H, W) or (H, W, 1), got {arr.shape}")
    return (arr > 0).astype(np.uint8)


@dataclass
class RegionSet:
    """
    Canonical region container shared by segmentation and attribution code.

    Parameters
    ----------
    masks : sequence of binary masks
        Each mask is (H, W) or (H, W, 1).  Masks are normalized to uint8.
    label_map : optional (H, W) int32
        Pixel-to-region assignment.  If omitted it is lazily derived from
        ``masks`` when needed.
    """

    masks: Sequence[np.ndarray]
    label_map: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self._masks: List[np.ndarray] = [_normalize_mask(mask) for mask in self.masks]
        self.masks = self._masks
        if self._masks:
            h, w = self._masks[0].shape[:2]
            for idx, mask in enumerate(self._masks):
                if mask.shape[:2] != (h, w):
                    raise ValueError(
                        f"all masks must share the same spatial size; "
                        f"mask[0]={self._masks[0].shape}, mask[{idx}]={mask.shape}"
                    )
        if self.label_map is not None:
            lm = np.asarray(self.label_map, dtype=np.int32)
            if self._masks and lm.shape != self._masks[0].shape[:2]:
                raise ValueError(
                    f"label_map shape {lm.shape} != masks shape {self._masks[0].shape[:2]}"
                )
            self.label_map = lm

    def __len__(self) -> int:
        return len(self._masks)

    @property
    def shape(self) -> Tuple[int, int]:
        if not self._masks:
            raise ValueError("RegionSet is empty")
        return self._masks[0].shape[:2]

    def binary_masks(self) -> List[np.ndarray]:
        """Return region masks as a list of (H, W, 1) uint8 arrays."""
        return [mask.copy() for mask in self._masks]

    def stack(self) -> np.ndarray:
        """Return stacked masks as (N, H, W, 1) uint8."""
        if not self._masks:
            return np.empty((0,), dtype=np.uint8)
        return np.stack(self._masks, axis=0).astype(np.uint8)

    def build_label_map(self) -> np.ndarray:
        """Build a best-effort label map from the stored masks."""
        if self.label_map is not None:
            return self.label_map.copy()
        if not self._masks:
            return np.empty((0, 0), dtype=np.int32)

        stacked = self.stack()[:, :, :, 0].astype(bool)
        covered = stacked.any(axis=0)
        label_map = np.full(stacked.shape[1:], -1, dtype=np.int32)
        if covered.any():
            label_map[covered] = stacked.argmax(axis=0)[covered].astype(np.int32)
        self.label_map = label_map
        return label_map.copy()

    def region_scores(
        self,
        saliency_map: np.ndarray,
        reduction: str = "mean",
    ) -> List[float]:
        """
        Aggregate a pixel saliency map into one score per region.

        Parameters
        ----------
        saliency_map : (H, W) or (H, W, 1)
        reduction    : "mean" | "sum" | "max"
        """
        sal = np.asarray(saliency_map, dtype=np.float32)
        if sal.ndim == 3 and sal.shape[2] == 1:
            sal = sal[:, :, 0]
        if sal.ndim != 2:
            raise ValueError(f"saliency_map must be 2-D, got {sal.shape}")
        if self._masks and sal.shape != self.shape:
            raise ValueError(f"saliency_map shape {sal.shape} != region shape {self.shape}")

        scores: List[float] = []
        for mask in self._masks:
            values = sal[mask[:, :, 0] > 0]
            if values.size == 0:
                scores.append(0.0)
            elif reduction == "sum":
                scores.append(float(values.sum()))
            elif reduction == "max":
                scores.append(float(values.max()))
            else:
                scores.append(float(values.mean()))
        return scores

    def ordered_masks_from_scores(
        self,
        scores: Sequence[float],
        descending: bool = True,
    ) -> List[np.ndarray]:
        """Return masks ordered by region score."""
        if len(scores) != len(self._masks):
            raise ValueError(f"expected {len(self._masks)} scores, got {len(scores)}")
        order = np.argsort(np.asarray(scores, dtype=np.float32))
        if descending:
            order = order[::-1]
        return [self._masks[int(idx)].copy() for idx in order]


class BaseSegmenter(ABC):
    """Abstract image partitioner returning a RegionSet."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> RegionSet:
        """Partition one image and return the corresponding RegionSet."""
