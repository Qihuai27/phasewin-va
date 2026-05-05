# -*- coding: utf-8 -*-
"""
Patch-grid segmentation.

This segmenter provides a regular square/grid partition. In VLM settings it is
also the natural token-aligned segmentation induced by patch-based visual
encoders. Map-based, search-based, and mask-optimization methods can all be
evaluated with this partition.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from attribution_research.segmentation.base import BaseSegmenter, RegionSet


class PatchSegmenter(BaseSegmenter):
    """
    Regular grid segmentation.

    Parameters
    ----------
    patch_size : int | None
        Square patch size in pixels.  If omitted, grid_shape or division_number
        is used instead.
    grid_shape : tuple[int, int] | None
        Explicit (rows, cols) grid shape.
    division_number : int | None
        Approximate total number of patches.  A near-square grid is derived.
    """

    def __init__(
        self,
        patch_size: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        division_number: Optional[int] = None,
    ):
        if patch_size is None and grid_shape is None and division_number is None:
            patch_size = 16
        self.patch_size = patch_size
        self.grid_shape = grid_shape
        self.division_number = division_number

    def _resolve_grid_shape(self, image: np.ndarray) -> Tuple[int, int]:
        h, w = image.shape[:2]
        if self.grid_shape is not None:
            rows, cols = self.grid_shape
            if rows < 1 or cols < 1:
                raise ValueError(f"grid_shape must be positive, got {self.grid_shape}")
            return int(rows), int(cols)

        if self.patch_size is not None:
            size = int(self.patch_size)
            if size < 1:
                raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
            return int(math.ceil(h / size)), int(math.ceil(w / size))

        assert self.division_number is not None
        n = max(1, int(self.division_number))
        rows = int(round(math.sqrt(n * h / max(w, 1))))
        rows = max(rows, 1)
        cols = int(math.ceil(n / rows))
        return rows, max(cols, 1)

    def segment(self, image: np.ndarray) -> RegionSet:
        h, w = image.shape[:2]
        rows, cols = self._resolve_grid_shape(image)
        y_edges = np.linspace(0, h, rows + 1, dtype=np.int32)
        x_edges = np.linspace(0, w, cols + 1, dtype=np.int32)

        masks = []
        label_map = np.full((h, w), -1, dtype=np.int32)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                y0, y1 = int(y_edges[i]), int(y_edges[i + 1])
                x0, x1 = int(x_edges[j]), int(x_edges[j + 1])
                if y1 <= y0 or x1 <= x0:
                    continue
                mask = np.zeros((h, w, 1), dtype=np.uint8)
                mask[y0:y1, x0:x1, 0] = 1
                label_map[y0:y1, x0:x1] = idx
                masks.append(mask)
                idx += 1

        return RegionSet(masks=masks, label_map=label_map)
