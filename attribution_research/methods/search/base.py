# -*- coding: utf-8 -*-
"""
Base classes shared by all attribution algorithms.

Hierarchy
---------
BaseSelector          -- abstract selector (select V_set -> ordered subset)
  NaiveGreedySelector -- O(k·n) greedy (in greedy.py)
  PhaseWindowSelector -- O(k·w·p) phase-windowed (in phasewin.py)

BaseExplainer         -- drives a BaseSelector, manages state + JSON logging
  GreedyExplainer     -- uses NaiveGreedySelector (in greedy.py)
  PhaseWinExplainer   -- uses NaiveGreedy + PhaseWindowSelector (in phasewin.py)

Both GreedyExplainer and PhaseWinExplainer share the same __call__ signature,
so PhaseWin is a drop-in replacement for greedy across all three tasks
(classification, detection, caption/VQA).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from attribution_research.adapters.base import ModelAdapter


# ──────────────────────────────────────────────────────────────────────────────
# Abstract selector
# ──────────────────────────────────────────────────────────────────────────────

class BaseSelector(ABC):
    """
    Abstract subset selector.

    Parameters
    ----------
    k : int
        Maximum number of elements to select.
    """

    def __init__(self, k: int):
        self.k = int(k)

    @abstractmethod
    def select(
        self,
        V_set: Sequence[np.ndarray],
        marginal_gain: Callable[[np.ndarray], torch.Tensor],
        apply: Optional[Callable[[np.ndarray], None]] = None,
    ) -> List[np.ndarray]:
        """
        Select up to k elements from V_set using marginal_gain as the oracle.

        Parameters
        ----------
        V_set        : sequence of N elements (np.ndarray masks)
        marginal_gain: callable -- (B, H, W, 1) array -> (B,) Tensor of gains
        apply        : optional callback called for each accepted element

        Returns
        -------
        Ordered list of accepted elements (best first).
        """


# ──────────────────────────────────────────────────────────────────────────────
# BaseExplainer
# ──────────────────────────────────────────────────────────────────────────────

class BaseExplainer(ABC):
    """
    Base class for greedy-based explainers.

    Manages per-image state (baseline mask, score logging), builds the
    gain closure, and delegates element ordering to a BaseSelector.

    Subclasses implement ``_build_selector(k)`` to return the appropriate
    selector for the algorithm (NaiveGreedy or PhaseWindow).
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ):
        self.adapter = adapter
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # ── state ─────────────────────────────────────────────────────────────────

    def _init_state(self, image: np.ndarray) -> None:
        H, W = image.shape[:2]
        self.adapter.reset_forward_counter()
        self._baseline = np.zeros((H, W, 1), dtype=np.uint8)
        self._region_area = float(H * W)
        self._insertion_scores: List[float] = []
        self._deletion_scores: List[float] = []
        self._gain_scores: List[float] = []
        self._region_areas: List[float] = []
        self._extra_per_step: List[Dict[str, Any]] = []
        self._selected: List[np.ndarray] = []
        self._marginal_calls: int = 0
        self._selection_forward_calls: int = 0
        self._step_eval_forward_calls: int = 0

    # ── oracle ────────────────────────────────────────────────────────────────

    def _marginal_gain(self, candidate_masks: np.ndarray) -> torch.Tensor:
        """
        candidate_masks : (B, H, W, 1) uint8
        Returns (B,) gains (no grad).
        """
        self._marginal_calls += len(candidate_masks)
        before = self.adapter.model_forward_calls
        out = self.adapter.score_batch(candidate_masks, self._baseline)
        after = self.adapter.model_forward_calls
        self._selection_forward_calls += int(after - before)
        return out

    # ── apply (commit hook) ───────────────────────────────────────────────────

    def _apply_mask(self, mask: np.ndarray) -> None:
        """
        Called when a mask is accepted.  Records scores and updates baseline.
        """
        mask_u8 = mask.astype(np.uint8)
        before = self.adapter.model_forward_calls
        detail = self.adapter.score_single_detailed(mask_u8, self._baseline)
        after = self.adapter.model_forward_calls
        self._step_eval_forward_calls += int(after - before)

        self._insertion_scores.append(detail.get("insertion_score", detail.get("smdl_score", 0.0)))
        self._deletion_scores.append(detail.get("deletion_score", 1.0))
        self._gain_scores.append(detail.get("smdl_score", 0.0))
        self._extra_per_step.append(detail)

        self._baseline = np.clip(
            self._baseline.astype(np.int32) + mask_u8.astype(np.int32), 0, 1
        ).astype(np.uint8)
        self._region_areas.append(float(self._baseline.sum()) / self._region_area)
        self._selected.append(mask_u8)

    # ── selector factory ──────────────────────────────────────────────────────

    @abstractmethod
    def _build_selector(self, k: int) -> BaseSelector:
        """Return a configured selector for k elements."""

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _build_json(self) -> Dict[str, Any]:
        # Historical note:
        # - `candidate_evaluations` counts every candidate mask scored through
        #   `score_batch`, including the eventually accepted ones.
        # - `marginal_calls` is kept for backward compatibility and excludes the
        #   accepted masks, so older result summaries remain stable.
        total_forward_calls = self.adapter.model_forward_calls
        d: Dict[str, Any] = {
            "insertion_score": self._insertion_scores,
            "deletion_score":  self._deletion_scores,
            "smdl_score":      self._gain_scores,
            "region_area":     self._region_areas,
            "sub-region_number": len(self._selected),
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "candidate_evaluations": self._marginal_calls,
            "marginal_calls": self._marginal_calls - len(self._selected),
            # Fair search-efficiency comparisons should only count candidate
            # selection forwards. Per-step eval replay is reported separately.
            "model_forward_calls": self._selection_forward_calls,
            "selection_model_forward_calls": self._selection_forward_calls,
            "eval_model_forward_calls": self._step_eval_forward_calls,
            "step_eval_model_forward_calls": self._step_eval_forward_calls,
            "total_model_forward_calls": total_forward_calls,
            "model_forward_count_mode": "equivalent_single_image_forwards",
            "model_forward_count_scope": "algorithm_only",
        }
        if self._insertion_scores:
            d["org_score"]      = self._insertion_scores[-1]
            d["baseline_score"] = self._deletion_scores[-1]
        if self._gain_scores:
            d["smdl_score_max"]       = max(self._gain_scores)
            d["smdl_score_max_index"] = self._gain_scores.index(max(self._gain_scores))
        # Merge per-step extra fields (list-valued, keyed by field name)
        extra_keys: set = set()
        for step in self._extra_per_step:
            extra_keys.update(step.keys())
        for key in extra_keys:
            if key not in ("insertion_score", "deletion_score", "smdl_score"):
                d[key] = [step.get(key) for step in self._extra_per_step]
        return d

    # ── main entry point ──────────────────────────────────────────────────────

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        target: Any,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Run attribution on one image.

        Parameters
        ----------
        image         : (H, W, 3) uint8 BGR
        masks         : list of N binary masks, each (H, W, 1) uint8
        target        : task-specific target (adapter decides how to use it)
        show_progress : show tqdm progress bar
        **kwargs      : forwarded to adapter.setup() (e.g. image_proc for detection)

        Returns
        -------
        ordered_masks : list of accepted masks, best-first
        json_dict     : serializable metrics dict
        """


# Backwards-compatible alias
BaseSubmodularExplainer = BaseExplainer
