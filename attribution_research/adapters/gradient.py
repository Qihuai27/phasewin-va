# -*- coding: utf-8 -*-
"""
Gradient / saliency-map adapter interfaces.

These adapters expose model-specific internals needed by methods that generate
pixel-level or token-level attribution maps with a small number of backward
passes.  They are intentionally separate from search adapters, because search
methods only need black-box mask scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np

from attribution_research.adapters.base import BaseAdapter


class GradientAdapter(BaseAdapter, ABC):
    """Adapter contract for map-based gradient attribution methods."""

    @abstractmethod
    def setup(
        self,
        image: np.ndarray,
        target: Any,
        **kwargs,
    ) -> None:
        """Bind image + target context for subsequent saliency generation."""

    @abstractmethod
    def saliency_map(self, **kwargs) -> np.ndarray:
        """
        Return a pixel saliency map as (H, W) or (H, W, 1) float array.
        """

    def teardown(self) -> None:
        """Optional cleanup after one explanation."""


class CallableGradientAdapter(GradientAdapter):
    """
    Lightweight wrapper around a user-supplied saliency function.

    The callable receives ``image``, ``target``, and any extra keyword
    arguments passed to ``setup``.
    """

    model_name = "callable"
    task_type = "generic"

    def __init__(
        self,
        fn: Callable[..., np.ndarray],
        device: str = "",
    ):
        self._fn = fn
        self._device = device
        self._image: Optional[np.ndarray] = None
        self._target: Any = None
        self._kwargs = {}

    @property
    def device(self) -> str:
        return self._device

    def setup(self, image: np.ndarray, target: Any, **kwargs) -> None:
        self._image = image
        self._target = target
        self._kwargs = dict(kwargs)

    def saliency_map(self, **kwargs) -> np.ndarray:
        merged = dict(self._kwargs)
        merged.update(kwargs)
        return np.asarray(self._fn(self._image, self._target, **merged), dtype=np.float32)
