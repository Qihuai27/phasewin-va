# -*- coding: utf-8 -*-
"""
Abstract adapter interfaces for attribution algorithms.

Two families are modeled here:
  ModelAdapter / SearchAdapter -- for greedy / PhaseWin / D-RISE-style methods
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Shared bookkeeping mixin
# ──────────────────────────────────────────────────────────────────────────────

class ForwardCounterMixin:
    """
    Utility mixin for tracking per-sample model evaluation volume.

    The counter stores "equivalent single-image forwards" instead of Python-side
    forward invocations.  If one batched model call scores B images at once, the
    counter should be incremented by B so the metric stays invariant to the
    chosen micro-batch size.
    """

    def _ensure_forward_counter(self) -> None:
        if not hasattr(self, "_model_forward_calls"):
            self._model_forward_calls = 0

    def reset_forward_counter(self) -> None:
        self._model_forward_calls = 0

    def record_model_forward(self, count: int = 1) -> None:
        step = int(count)
        if step < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        self._ensure_forward_counter()
        self._model_forward_calls += step

    @property
    def model_forward_calls(self) -> int:
        self._ensure_forward_counter()
        return int(self._model_forward_calls)


class BaseAdapter(ForwardCounterMixin):
    """Shared metadata, device helpers, and forward-count tracking."""

    model_name: str = ""
    task_type: str = ""  # "classification" | "detection" | "grounding" | "caption_vqa"

    @property
    def device(self) -> str:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# ModelAdapter  (greedy / PhaseWin)
# ──────────────────────────────────────────────────────────────────────────────

class ModelAdapter(BaseAdapter, ABC):
    """
    Adapter contract for greedy / search explainers.

    Lifecycle per image
    -------------------
    1. ``setup(image, target, **kwargs)`` -- bind one image + target; prepare
       preprocessed tensors, class ids, boxes, token ids, etc.
    2. ``score_batch(masks, baseline)`` -- called many times by the selector loop.
    3. ``teardown()`` -- optional cleanup.

    The gain formula is part of the adapter (not the algorithm), because it
    depends on task-specific score components (IoU×cls vs token probability).
    Concretely, the adapter returns the already-combined gain:
        gain = lambda1 * insertion_score + lambda2 * (1 - deletion_score)
    using whatever definition of score is appropriate for the task.

    If you need to expose raw insertion/deletion scores for JSON logging, override
    ``score_single_detailed()`` (returns full dict).
    """

    @abstractmethod
    def setup(
        self,
        image: np.ndarray,   # (H, W, 3) uint8 BGR
        target: Any,         # task-specific target specification
        **kwargs,            # optional extras (e.g. image_proc for detection)
    ) -> None:
        """Bind image + target context for subsequent score_batch calls."""

    @abstractmethod
    def score_batch(
        self,
        masks: np.ndarray,      # (B, H, W, 1) uint8 binary masks (0 or 1)
        baseline: np.ndarray,   # (H, W, 1) uint8 accumulated selected-mask so far
    ) -> torch.Tensor:
        """
        Compute gain for a batch of candidate masks.

        The insertion image for mask_i is built as:
            alpha_i = clip(mask_i + baseline, 0, 1)
            ins_img  = alpha_i * source_image

        The deletion image is:
            del_img  = (1 - alpha_i) * source_image

        Returns
        -------
        (B,) float tensor (no grad) -- lambda1*ins + lambda2*(1-del) scores.
        """

    def score_single_detailed(
        self,
        mask: np.ndarray,       # (H, W, 1) uint8
        baseline: np.ndarray,   # (H, W, 1) uint8
    ) -> Dict[str, float]:
        """
        Score a single mask and return a dict with all task-specific components
        (insertion_score, deletion_score, insertion_iou, insertion_cls, …).

        Default: just wraps score_batch.  Override to return richer dicts.
        """
        gain = self.score_batch(mask[None], baseline)
        return {"smdl_score": float(gain[0].item())}

    def score_batch_detailed(
        self,
        masks: np.ndarray,      # (B, H, W, 1) uint8
        baseline: np.ndarray,   # (H, W, 1) uint8
    ) -> Dict[str, torch.Tensor]:
        """
        Batch version of ``score_single_detailed``.

        Subclasses can override this to expose insertion / deletion components
        efficiently for map-based algorithms such as D-RISE.  The default
        implementation is intentionally simple and may be slower.
        """
        details = [self.score_single_detailed(mask, baseline) for mask in masks]
        keys = set()
        for detail in details:
            keys.update(detail.keys())

        batched: Dict[str, torch.Tensor] = {}
        for key in keys:
            values = [detail.get(key) for detail in details]
            if all(isinstance(value, (float, int, np.floating, np.integer)) for value in values):
                batched[key] = torch.tensor(values, dtype=torch.float32)
        if "smdl_score" not in batched:
            batched["smdl_score"] = self.score_batch(masks, baseline).float().cpu()
        return batched

    def teardown(self) -> None:
        """Optional cleanup after processing one image."""


# Preferred name moving forward
SearchAdapter = ModelAdapter
GreedyAdapter = ModelAdapter

# Backwards-compatible alias
SubmodularAdapter = ModelAdapter
