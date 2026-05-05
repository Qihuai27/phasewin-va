# -*- coding: utf-8 -*-
"""
Shared composition helpers.

This module keeps task entrypoints thin by centralizing the common logic used
to combine:
1. algorithm family;
2. segmentation strategy; and
3. task-specific adapters.
"""

from __future__ import annotations

from typing import Optional, Tuple

from attribution_research.registry import algorithm_family, normalize_algorithm_name
from attribution_research.segmentation.patch import PatchSegmenter
from attribution_research.segmentation.superpixel import Segmentor


def build_segmenter(
    segmenter: str,
    *,
    superpixel_algorithm: str = "slico",
    region_size: Optional[int] = None,
    division_number: int = 50,
    patch_size: Optional[int] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
):
    """Instantiate either a superpixel segmenter or a regular patch grid."""
    segmenter_name = str(segmenter).strip().lower()
    if segmenter_name == "patch":
        grid_shape: Optional[Tuple[int, int]] = None
        if grid_rows is not None and grid_cols is not None:
            grid_shape = (int(grid_rows), int(grid_cols))
        return PatchSegmenter(
            patch_size=patch_size,
            grid_shape=grid_shape,
            division_number=division_number if grid_shape is None and patch_size is None else None,
        )
    if segmenter_name == "superpixel":
        return Segmentor(
            mode=superpixel_algorithm,
            region_size=region_size,
            division_number=division_number,
        )
    raise ValueError(f"Unsupported segmenter: {segmenter!r}")


def build_segmenter_from_args(args):
    """Argument-namespace wrapper around :func:`build_segmenter`."""
    return build_segmenter(
        args.segmenter,
        superpixel_algorithm=getattr(args, "superpixel_algorithm", "slico"),
        region_size=getattr(args, "region_size", None),
        division_number=getattr(args, "division_number", 50),
        patch_size=getattr(args, "patch_size", None),
        grid_rows=getattr(args, "grid_rows", None),
        grid_cols=getattr(args, "grid_cols", None),
    )


def segmenter_tag(
    segmenter: str,
    *,
    superpixel_algorithm: str = "slico",
    division_number: int = 50,
    patch_size: Optional[int] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> str:
    """Build a stable run-directory tag for the chosen segmenter."""
    segmenter_name = str(segmenter).strip().lower()
    if segmenter_name == "patch":
        if grid_rows is not None and grid_cols is not None:
            return f"patch-grid-{grid_rows}x{grid_cols}"
        if patch_size is not None:
            return f"patch-size-{patch_size}"
        return f"patch-division-{division_number}"
    if segmenter_name == "superpixel":
        return f"{superpixel_algorithm}-division-{division_number}"
    raise ValueError(f"Unsupported segmenter: {segmenter!r}")


def segmenter_tag_from_args(args) -> str:
    """Argument-namespace wrapper around :func:`segmenter_tag`."""
    return segmenter_tag(
        args.segmenter,
        superpixel_algorithm=getattr(args, "superpixel_algorithm", "slico"),
        division_number=getattr(args, "division_number", 50),
        patch_size=getattr(args, "patch_size", None),
        grid_rows=getattr(args, "grid_rows", None),
        grid_cols=getattr(args, "grid_cols", None),
    )


__all__ = [
    "algorithm_family",
    "build_segmenter",
    "build_segmenter_from_args",
    "normalize_algorithm_name",
    "segmenter_tag",
    "segmenter_tag_from_args",
]
