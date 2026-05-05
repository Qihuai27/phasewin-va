# -*- coding: utf-8 -*-
"""
Unified superpixel segmentation.

Two output modes
----------------
binary_masks(image)  -> list[(H,W,1) uint8]
    One binary mask per superpixel.  Used by search-based workflows
    (greedy / phasewin / other region-ranking methods).
"""

import math
import warnings
from typing import List, Tuple

import cv2
import numpy as np

from attribution_research.segmentation.base import BaseSegmenter, RegionSet


class Segmentor(BaseSegmenter):
    """
    Superpixel segmentation using SLICO or SEEDS (via opencv-contrib-python).

    Parameters
    ----------
    mode : "slico" | "seeds"
    region_size : int
        Approximate superpixel size for SLICO (in pixels, used as the size of
        the initial grid step).  Ignored when mode="seeds".
    division_number : int
        Target number of superpixels.  For SLICO the actual region_size is
        computed as sqrt(H*W / division_number); for SEEDS this is passed
        directly as num_superpixels.  Ignored when region_size is set
        explicitly (non-None).
    num_levels : int
        SEEDS histogram levels (default 3).  Ignored for SLICO.
    n_iterations : int
        Number of SLICO/SEEDS iterations (default 20).
    ruler : float
        SLICO spatial regularization (default 20.0).
    """

    def __init__(
        self,
        mode: str = "slico",
        region_size: int = None,
        division_number: int = 50,
        num_levels: int = 3,
        n_iterations: int = 20,
        ruler: float = 20.0,
    ):
        assert mode in ("slico", "seeds"), f"mode must be 'slico' or 'seeds', got {mode!r}"
        self.mode = mode
        self._region_size = region_size
        self.division_number = division_number
        self.num_levels = num_levels
        self.n_iterations = n_iterations
        self.ruler = ruler

    def _region_size_for(self, image: np.ndarray) -> int:
        if self._region_size is not None:
            return int(self._region_size)
        H, W = image.shape[:2]
        return max(1, int(math.sqrt(H * W / self.division_number)))

    def _run(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Run SLICO or SEEDS on image.

        Returns
        -------
        label_map : (H, W) int32  -- pixel-level superpixel IDs
        n         : int           -- number of superpixels
        """
        ximgproc = getattr(cv2, "ximgproc", None)
        if self.mode == "slico":
            if ximgproc is not None and hasattr(ximgproc, "createSuperpixelSLIC"):
                rs = self._region_size_for(image)
                slic = ximgproc.createSuperpixelSLIC(
                    image, region_size=rs, ruler=float(self.ruler)
                )
                slic.iterate(self.n_iterations)
                return slic.getLabels().astype(np.int32), slic.getNumberOfSuperpixels()
            return self._run_skimage_slic(image, requested_mode="slico")

        if ximgproc is not None and hasattr(ximgproc, "createSuperpixelSEEDS"):
            seeds = ximgproc.createSuperpixelSEEDS(
                image.shape[1],
                image.shape[0],
                image.shape[2],
                num_superpixels=self.division_number,
                num_levels=self.num_levels,
            )
            seeds.iterate(image, self.n_iterations)
            return seeds.getLabels().astype(np.int32), seeds.getNumberOfSuperpixels()
        return self._run_skimage_slic(image, requested_mode="seeds")

    def _run_skimage_slic(
        self,
        image: np.ndarray,
        requested_mode: str,
    ) -> Tuple[np.ndarray, int]:
        """Fallback when OpenCV superpixel operators are unavailable."""
        try:
            from skimage.segmentation import slic
        except ImportError as exc:
            raise RuntimeError(
                "No superpixel backend is available. "
                "Install opencv-contrib with ximgproc support or scikit-image."
            ) from exc

        if self._region_size is not None:
            h, w = image.shape[:2]
            n_segments = max(1, int(round((h * w) / float(self._region_size ** 2))))
        else:
            n_segments = max(1, int(self.division_number))

        warnings.warn(
            f"OpenCV superpixel backend is unavailable; falling back to skimage.slic for mode={requested_mode!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        label_map = slic(
            image_rgb,
            n_segments=n_segments,
            compactness=float(self.ruler),
            start_label=0,
            channel_axis=-1,
        ).astype(np.int32)
        return label_map, int(label_map.max()) + 1

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> RegionSet:
        """Segment image and return the canonical RegionSet representation."""
        label_map, n = self._run(image)
        masks = [
            (label_map == i)[:, :, np.newaxis].astype(np.uint8)
            for i in range(n)
        ]
        return RegionSet(masks=masks, label_map=label_map)

    def binary_masks(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment image and return binary masks.

        Parameters
        ----------
        image : (H, W, 3) uint8 BGR

        Returns
        -------
        masks : list of N arrays, each (H, W, 1) uint8 with values 0 or 1
        """
        return self.segment(image).binary_masks()

# ──────────────────────────────────────────────────────────────────────────────
# Legacy-compatible function (drop-in for original SubRegionDivision)
# ──────────────────────────────────────────────────────────────────────────────

def SubRegionDivision(
    image: np.ndarray,
    mode: str = "slico",
    region_size: int = 30,
    return_label_map: bool = False,
):
    """
    Legacy-compatible wrapper around Segmentor.

    When return_label_map=False  -> list of (H,W,1) int binary masks
    When return_label_map=True   -> (list of (H,W,1) int masks, (H,W) int32 label_map)
    """
    seg = Segmentor(mode=mode, region_size=region_size, n_iterations=20)
    regions = seg.segment(image)
    masks = [mask.astype(int) for mask in regions.binary_masks()]
    if return_label_map:
        return masks, regions.build_label_map()
    return masks
