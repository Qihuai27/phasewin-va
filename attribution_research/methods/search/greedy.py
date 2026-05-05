# -*- coding: utf-8 -*-
"""
Greedy search explainer.

Implements a generic region-selection loop over any search adapter via the
BaseExplainer contract.

Algorithm
---------
For each of the N rounds:
  1. Evaluate gain for every remaining mask (batched through adapter).
  2. Select the mask with the highest gain.
  3. Call apply (update baseline, log scores).

Complexity: O(N²) oracle calls.

PhaseWinExplainer (phasewin.py) is a drop-in replacement for GreedyExplainer
that reduces adapter calls by 3-5x while preserving attribution quality.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from attribution_research.adapters.base import ModelAdapter
from attribution_research.methods.search.base import BaseExplainer, BaseSelector


# ──────────────────────────────────────────────────────────────────────────────
# NaiveGreedySelector
# ──────────────────────────────────────────────────────────────────────────────

class NaiveGreedySelector(BaseSelector):
    """
    Standard greedy selector: at each step evaluate all remaining elements and
    select the one with the highest gain.

    Complexity: O(k * n) oracle calls.
    """

    def select(self, V_set, marginal_gain, apply=None):
        remaining = list(V_set)
        selected: List[np.ndarray] = []

        for _ in range(self.k):
            if not remaining:
                break
            batch = np.stack(remaining, axis=0)         # (n_rem, H, W, 1)
            gains = marginal_gain(batch)                 # (n_rem,)
            idx = int(torch.argmax(gains).item())
            mask = remaining.pop(idx)
            selected.append(mask)
            if apply is not None:
                apply(mask)

        return selected


# ──────────────────────────────────────────────────────────────────────────────
# GreedyExplainer
# ──────────────────────────────────────────────────────────────────────────────

class GreedyExplainer(BaseExplainer):
    """
    Greedy search explainer for any task with a ModelAdapter.

    Unifies region-selection style search across:
    - caption / VQA attribution
    - classification attribution
    - detection attribution

    PhaseWinExplainer (phasewin.py) can replace this class with the same
    interface but significantly fewer model evaluations.

    Usage
    -----
    >>> adapter  = QwenVLAdapter(model, processor)
    >>> explainer = GreedyExplainer(adapter, lambda1=1.0, lambda2=1.0)
    >>> ordered_masks, json_dict = explainer(image, masks, target)
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ):
        super().__init__(adapter, lambda1=lambda1, lambda2=lambda2)

    def _build_selector(self, k: int) -> NaiveGreedySelector:
        return NaiveGreedySelector(k=k)

    def __call__(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        target: Any,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Parameters
        ----------
        image         : (H, W, 3) uint8 BGR
        masks         : list of N binary masks, each (H, W, 1) uint8
        target        : task-specific target (int label / dict / token ids)
        show_progress : whether to display a tqdm progress bar
        **kwargs      : forwarded to adapter.setup() (e.g. image_proc)

        Returns
        -------
        ordered_masks : list of N masks ordered by importance (best first)
        json_dict     : serializable dict for evaluation scripts
        """
        self._init_state(image)
        self.adapter.setup(image, target, **kwargs)

        remaining = [m.astype(np.uint8) for m in masks]
        iterator = tqdm(range(len(remaining)), desc="Greedy") if show_progress else range(len(remaining))
        selected: List[np.ndarray] = []

        for _ in iterator:
            if not remaining:
                break
            batch = np.stack(remaining, axis=0)
            gains = self._marginal_gain(batch)
            idx = int(torch.argmax(gains).item())
            mask = remaining.pop(idx)
            self._apply_mask(mask)
            selected.append(mask)

        self.adapter.teardown()
        return selected, self._build_json()


# Backwards-compatible alias
SubmodularExplainer = GreedyExplainer
