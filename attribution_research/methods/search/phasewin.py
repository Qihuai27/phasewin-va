# -*- coding: utf-8 -*-
"""
PhaseWin: Phase-Windowed Greedy Attribution.

Ports PhaseWindowSelector and the two-phase loop from phasewin-search,
refactored to accept a ModelAdapter so it works as a drop-in replacement
for GreedyExplainer across all three tasks:
  - Classification (CLIP classifiers)
  - Detection / Grounding (GroundingDINOAdapter, Florence2Adapter)
  - Caption / VQA (MLLMAdapter)

Algorithm (PhaseWindowSelector)
--------------------------------
Phase 0: n_greedy rounds of NaiveGreedy (optional warm-start).
Phase 1+: Windowed selection with:
  - Deletion threshold: prune elements with gain < beta_del * G_t
  - Selection threshold: tau_sel = alpha_sel_ratio * G_t
  - Window policy: LG (largest gain) | BA (batch accept) | T2 (top-2)
  - Hard exit: second-order saturation criterion
  - Annealing: temperature decay for deferred candidates

Speedup over pure greedy: 3-5x with <5% AUC degradation.
"""

import math
import random
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from attribution_research.adapters.base import ModelAdapter
from attribution_research.methods.search.base import BaseExplainer, BaseSelector
from attribution_research.methods.search.greedy import NaiveGreedySelector


# ──────────────────────────────────────────────────────────────────────────────
# PhaseWindowSelector
# ──────────────────────────────────────────────────────────────────────────────

class PhaseWindowSelector(BaseSelector):
    """
    Phase-windowed selector for greedy-style region selection objectives.

    Key mechanisms
    --------------
    - Deletion threshold  : tau_del = beta_del * G_t  (prune low-gain items)
    - Selection threshold : tau_sel = alpha_sel_ratio * G_t
    - Candidate set       : threshold-passing (sorted) + optional random sample
    - Phase start         : accept global top-1 first, then proceed in window
    - Annealing           : deferred window filling with temperature decay
    - Hard exit           : second-order robust acceleration criterion

    Window policies
    ---------------
    'LG' : Largest Gain  -- select only the maximum element each round
    'BA' : Batch Accept  -- select all elements above beta_win * G_window
    'T2' : Top-2         -- select top-2 when gain gap is small (< eta_top2)

    Parameters
    ----------
    k                  : maximum elements to select
    window_size        : window size w
    beta_del           : deletion threshold ratio (0 < beta_del < 1)
    alpha_sel_ratio    : selection threshold ratio (0 < alpha_sel_ratio < 1)
    random_frac        : fraction of below-threshold elements to add randomly
    rng_seed           : reproducibility seed (None = not seeded)
    window_policy      : 'LG' | 'BA' | 'T2'
    beta_win           : batch-acceptance cutoff ratio (BA policy)
    eta_top2           : relative gap threshold for T2 policy
    enable_anneal      : enable annealing temperature decay
    defer_T0           : initial annealing temperature
    defer_decay        : temperature decay factor per phase
    defer_max_per_phase: max deferrals per element per phase
    enable_hard_exit   : enable second-order hard termination
    hard_delta_thresh  : delta threshold triggering hard exit
    hard_phi_prev      : protection ratio for hard-exit acceptance
    """

    def __init__(
        self,
        k: int,
        # Thresholds
        window_size: int = 16,
        beta_del: float = 0.3,
        alpha_sel_ratio: float = 0.8,
        # Random sampling
        random_frac: float = 0.0,
        rng_seed: Optional[int] = None,
        # Window policy
        window_policy: str = "BA",
        beta_win: float = 0.9,
        eta_top2: float = 0.05,
        # Annealing
        enable_anneal: bool = True,
        defer_T0: float = 0.5,
        defer_decay: float = 0.85,
        defer_max_per_phase: int = 1,
        # Hard exit
        enable_hard_exit: bool = True,
        hard_delta_thresh: float = 0.025,
        hard_phi_prev: float = 0.95,
    ):
        super().__init__(k)
        assert window_size >= 1
        assert 0.0 < beta_del < 1.0
        assert 0.0 < alpha_sel_ratio < 1.0
        assert 0.0 <= random_frac <= 1.0
        assert 0.0 < beta_win <= 1.0
        assert hard_delta_thresh > 0.0

        self.w = int(window_size)
        self.beta_del = float(beta_del)
        self.alpha_sel_ratio = float(alpha_sel_ratio)
        self.random_frac = float(random_frac)
        self.rng = random.Random(rng_seed)
        self.window_policy = window_policy.upper().strip()
        assert self.window_policy in {"LG", "BA", "T2"}
        self.beta_win = float(beta_win)
        self.eta_top2 = float(eta_top2)
        self.enable_anneal = bool(enable_anneal)
        self.defer_T0 = float(defer_T0)
        self.defer_decay = float(defer_decay)
        self.defer_max_per_phase = int(defer_max_per_phase)
        self.enable_hard_exit = bool(enable_hard_exit)
        self.hard_delta_thresh = float(hard_delta_thresh)
        self.hard_phi_prev = float(hard_phi_prev)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _defer_prob(self, g: float, G: float, alpha_mult: float, T: float) -> float:
        if not self.enable_anneal or T <= 0:
            return 0.0
        x = (alpha_mult * G - g) / max(1e-12, T)
        if x > 40:
            return 1.0
        if x < -40:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _hard_exit_action(
        self, g_pp: Optional[float], g_p: Optional[float], g_i: float
    ) -> str:
        """
        Returns 'none' | 'exit_accept' | 'exit_reject'.

        Computes delta = (g_{t-1}/g_t) - (g_{t-2}/g_{t-1}).
        If delta > threshold:
          g_t >= phi * g_{t-1}  -> accept then exit
          else                  -> exit without accepting
        """
        if (not self.enable_hard_exit) or (g_pp is None) or (g_p is None):
            return "none"
        eps = 1e-12
        r_prev = g_pp / max(eps, g_p)
        r_curr = g_p  / max(eps, g_i)
        delta = r_curr - r_prev
        if delta > self.hard_delta_thresh:
            if g_i >= self.hard_phi_prev * g_p:
                return "exit_accept"
            return "exit_reject"
        return "none"

    # ── window policies ───────────────────────────────────────────────────────

    def _policy_LG(self, window, S_eval_fn, tau_sel):
        g = S_eval_fn(window)
        if g.numel() == 0:
            return [], g
        idx = int(torch.argmax(g).item())
        if float(g[idx].item()) >= tau_sel:
            return [idx], g
        return [], g

    def _policy_BA(self, window, S_eval_fn, tau_sel):
        g = S_eval_fn(window)
        if g.numel() == 0:
            return [], g
        gmax = float(torch.max(g).item())
        cut = max(tau_sel, self.beta_win * gmax)
        idxs = torch.nonzero(g >= cut).flatten().tolist()
        idxs_sorted = sorted(idxs, key=lambda i: float(g[i]), reverse=True)
        return idxs_sorted, g

    def _policy_T2(self, window, S_eval_fn, tau_sel):
        g = S_eval_fn(window)
        m = int(g.numel())
        if m == 0:
            return [], g
        vals, idxs = torch.topk(g, k=min(2, m))
        g1 = float(vals[0].item())
        i1 = int(idxs[0].item())
        if g1 < tau_sel:
            return [], g
        if m == 1:
            return [i1], g
        g2 = float(vals[1].item())
        i2 = int(idxs[1].item())
        rel_gap = abs(g1 - g2) / max(1e-12, max(g1, g2))
        if (g2 >= tau_sel) and (rel_gap <= self.eta_top2):
            return ([i1, i2] if g1 >= g2 else [i2, i1]), g
        return [i1], g

    def _get_policy(self):
        return {"LG": self._policy_LG, "BA": self._policy_BA, "T2": self._policy_T2}[
            self.window_policy
        ]

    # ── main select loop ──────────────────────────────────────────────────────

    def select(
        self,
        V_set: Sequence[np.ndarray],
        marginal_gain: Callable[[np.ndarray], torch.Tensor],
        apply: Optional[Callable[[np.ndarray], None]] = None,
    ) -> List[np.ndarray]:
        remaining = list(V_set)
        deleted: List[np.ndarray] = []
        S_out: List[np.ndarray] = []
        T = self.defer_T0
        policy = self._get_policy()

        def eval_vec(X: List[np.ndarray]) -> torch.Tensor:
            if not X:
                return torch.empty(0)
            return marginal_gain(np.stack(X, axis=0))

        g_prevprev: Optional[float] = None
        g_prev: Optional[float] = None

        while remaining and len(S_out) < self.k:
            # ── phase start: accept global top-1 ──────────────────────────
            g0 = eval_vec(remaining)
            if g0.numel() == 0:
                break
            G0 = float(torch.max(g0).item())
            if G0 <= 0:
                break
            idx0 = int(torch.argmax(g0).item())
            top_e = remaining.pop(idx0)
            if apply is not None:
                apply(top_e)
            S_out.append(top_e)
            g_prevprev = None
            g_prev = float(G0)
            if len(S_out) >= self.k:
                break

            # ── updated remaining + thresholds ────────────────────────────
            g_all = eval_vec(remaining)
            if g_all.numel() == 0:
                T *= self.defer_decay
                continue
            G_t = float(torch.max(g_all).item())
            if G_t <= 0:
                T *= self.defer_decay
                continue

            tau_del = self.beta_del * G_t
            tau_sel = self.alpha_sel_ratio * G_t

            # Delete low-gain items
            del_mask = g_all < tau_del
            if del_mask.any():
                keep_mask = ~del_mask
                deleted.extend([remaining[i] for i, f in enumerate(del_mask.tolist()) if f])
                remaining = [remaining[i] for i, f in enumerate(keep_mask.tolist()) if f]
                g_all = g_all[keep_mask]
                if not remaining:
                    T *= self.defer_decay
                    continue

            # Build candidate set
            cand_mask = (g_all >= tau_sel)
            rest_mask = ~cand_mask
            cand_vals = g_all[cand_mask]
            cand_list = [remaining[i] for i, f in enumerate(cand_mask.tolist()) if f]
            rest_indices = [i for i, f in enumerate(rest_mask.tolist()) if f]
            idx_sampled = []
            if rest_indices and self.random_frac > 0.0:
                k_samp = min(
                    len(rest_indices),
                    max(0, int(math.ceil(self.random_frac * len(rest_indices)))),
                )
                idx_sampled = self.rng.sample(rest_indices, k=k_samp)

            P_t: List[np.ndarray] = []
            if cand_list:
                order1 = torch.argsort(cand_vals, descending=True).tolist()
                P_t.extend([cand_list[i] for i in order1])
            if idx_sampled:
                sampled_list = [remaining[i] for i in idx_sampled]
                sampled_vals = g_all[idx_sampled]
                order2 = torch.argsort(sampled_vals, descending=True).tolist()
                P_t.extend([sampled_list[i] for i in order2])

            if not P_t:
                T *= self.defer_decay
                continue

            # ── window-based selection ─────────────────────────────────────
            window: Deque[np.ndarray] = deque(P_t[: self.w])
            tail: Deque[np.ndarray] = deque(P_t[self.w :])
            defer_count: Dict[int, int] = {}
            broke_phase = False

            def S_eval_fn(win_list):
                return eval_vec(win_list)

            while window and len(S_out) < self.k:
                # Refill with annealing
                while len(window) < self.w and tail:
                    e = tail.popleft()
                    p_defer = self._defer_prob(tau_sel, G_t, self.alpha_sel_ratio, T)
                    if self.enable_anneal and self.rng.random() < p_defer:
                        key = id(e)
                        cnt = defer_count.get(key, 0)
                        if cnt < self.defer_max_per_phase:
                            defer_count[key] = cnt + 1
                            tail.append(e)
                            continue
                    window.append(e)

                win_list = list(window)
                accept_positions, g_vec = policy(win_list, S_eval_fn, tau_sel)
                if not accept_positions:
                    break

                new_win: List[np.ndarray] = []
                for i, e in enumerate(win_list):
                    if (i in accept_positions) and len(S_out) < self.k:
                        g_i = float(g_vec[i].item())
                        action = self._hard_exit_action(g_prevprev, g_prev, g_i)
                        if action == "exit_reject":
                            new_win.extend(win_list[i:])
                            broke_phase = True
                            break
                        elif action == "exit_accept":
                            if apply is not None:
                                apply(e)
                            S_out.append(e)
                            g_prevprev = g_prev
                            g_prev = g_i
                            new_win.extend(win_list[i + 1:])
                            broke_phase = True
                            break
                        # Normal accept
                        if apply is not None:
                            apply(e)
                        S_out.append(e)
                        g_prevprev = g_prev
                        g_prev = g_i
                    else:
                        new_win.append(e)

                window = deque(new_win)
                if broke_phase:
                    break

                # Immediate refill
                while len(window) < self.w and tail and len(S_out) < self.k:
                    e = tail.popleft()
                    p_defer = self._defer_prob(tau_sel, G_t, self.alpha_sel_ratio, T)
                    if self.enable_anneal and self.rng.random() < p_defer:
                        key = id(e)
                        cnt = defer_count.get(key, 0)
                        if cnt < self.defer_max_per_phase:
                            defer_count[key] = cnt + 1
                            tail.append(e)
                            continue
                    window.append(e)

            # Phase end: return remaining window + tail + middle band
            middle_band = [
                v for v, g in zip(remaining, g_all.tolist())
                if tau_del <= g < tau_sel
            ]
            remaining = list(window) + list(tail) + middle_band
            T *= self.defer_decay

        # Recover from remaining (items never visited due to early phase exits)
        if len(S_out) < self.k and remaining:
            g_rem = eval_vec(remaining)
            if g_rem.numel() > 0:
                for i in torch.argsort(g_rem, descending=True).tolist():
                    if len(S_out) >= self.k:
                        break
                    e = remaining[i]
                    if apply is not None:
                        apply(e)
                    S_out.append(e)

        # Recover from deleted (items pruned by deletion threshold)
        if len(S_out) < self.k and deleted:
            g_del = eval_vec(deleted)
            if g_del.numel() > 0:
                for i in torch.argsort(g_del, descending=True).tolist():
                    if len(S_out) >= self.k:
                        break
                    e = deleted[i]
                    if apply is not None:
                        apply(e)
                    S_out.append(e)

        return S_out


# ──────────────────────────────────────────────────────────────────────────────
# PhaseWinExplainer
# ──────────────────────────────────────────────────────────────────────────────

# Model-specific default configurations
_MODEL_DEFAULTS: Dict[str, Dict] = {
    "default": {
        "n_greedy": 0,
        "pw_hard_delta_thresh": 0.02,
        "pw_hard_phi_prev": 0.98,
    },
    "florence": {
        "n_greedy": 18,
        "pw_hard_delta_thresh": 0.025,
        "pw_hard_phi_prev": 1.01,
    },
}


class PhaseWinExplainer(BaseExplainer):
    """
    Two-phase explainer: n_greedy warm-start rounds, then PhaseWindowSelector.

    Drop-in replacement for GreedyExplainer across all three tasks:
      - Classification : use with CLIPSearchAdapter (score_batch via CLIP confidence)
      - Detection      : use with GroundingDINOAdapter / Florence2Adapter
      - Caption / VQA  : use with MLLMAdapter

    The same __call__ signature as GreedyExplainer; swap by changing one line:
        explainer = GreedyExplainer(adapter)   # O(N²) calls
        explainer = PhaseWinExplainer(adapter) # ~3-5x fewer calls

    Usage
    -----
    >>> adapter  = GroundingDINOAdapter(model, preprocess_fn)
    >>> explainer = PhaseWinExplainer(adapter, model_type='default')
    >>> ordered_masks, json_dict = explainer(image, masks, target)
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        model_type: str = "default",
        n_greedy: Optional[int] = None,
        # PhaseWindowSelector kwargs
        pw_window_size: Optional[int] = None,
        pw_window_frac: float = 0.3,
        pw_beta_del: float = 0.05,
        pw_alpha_sel: float = 0.6,
        pw_random_frac: float = 0.0,
        pw_rng_seed: Optional[int] = None,
        pw_window_policy: str = "BA",
        pw_beta_win: float = 0.99,
        pw_eta_top2: float = 0.05,
        pw_enable_anneal: bool = True,
        pw_defer_T0: float = 0.5,
        pw_defer_decay: float = 0.85,
        pw_defer_max_per_phase: int = 1,
        pw_enable_hard_exit: bool = True,
        pw_hard_delta_thresh: Optional[float] = None,
        pw_hard_phi_prev: Optional[float] = None,
    ):
        super().__init__(adapter, lambda1=lambda1, lambda2=lambda2)

        if pw_window_size is not None and int(pw_window_size) < 1:
            raise ValueError("pw_window_size must be >= 1 when set")
        if not (0.0 < float(pw_window_frac) <= 1.0):
            raise ValueError("pw_window_frac must be in (0, 1]")

        defaults = _MODEL_DEFAULTS.get(model_type, _MODEL_DEFAULTS["default"])
        self.n_greedy = n_greedy if n_greedy is not None else defaults["n_greedy"]
        if pw_hard_delta_thresh is None:
            pw_hard_delta_thresh = defaults["pw_hard_delta_thresh"]
        if pw_hard_phi_prev is None:
            pw_hard_phi_prev = defaults["pw_hard_phi_prev"]

        self._pw_window_size_override = None if pw_window_size is None else int(pw_window_size)
        self._pw_window_frac = float(pw_window_frac)
        self._last_window_size: Optional[int] = None
        self._pw_cfg = dict(
            beta_del=pw_beta_del,
            alpha_sel_ratio=pw_alpha_sel,
            random_frac=pw_random_frac,
            rng_seed=pw_rng_seed,
            window_policy=pw_window_policy,
            beta_win=pw_beta_win,
            eta_top2=pw_eta_top2,
            enable_anneal=pw_enable_anneal,
            defer_T0=pw_defer_T0,
            defer_decay=pw_defer_decay,
            defer_max_per_phase=pw_defer_max_per_phase,
            enable_hard_exit=pw_enable_hard_exit,
            hard_delta_thresh=pw_hard_delta_thresh,
            hard_phi_prev=pw_hard_phi_prev,
        )

    def _resolve_window_size(self, candidate_count: int) -> int:
        if self._pw_window_size_override is not None:
            return max(1, min(self._pw_window_size_override, candidate_count))
        if candidate_count <= 0:
            return 1
        return max(1, int(math.floor(candidate_count * self._pw_window_frac)))

    def _build_selector(self, k: int) -> PhaseWindowSelector:
        resolved_window_size = self._resolve_window_size(k)
        self._last_window_size = resolved_window_size
        return PhaseWindowSelector(
            k=k,
            window_size=resolved_window_size,
            **self._pw_cfg,
        )

    def __call__(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        target: Any,
        image_proc: Optional[torch.Tensor] = None,  # detection models may need this
        show_progress: bool = True,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Parameters
        ----------
        image      : (H, W, 3) uint8 BGR
        masks      : list of N binary masks, each (H, W, 1) uint8
        target     : adapter-specific (e.g. {"label": int, "box": [x1,y1,x2,y2]})
        image_proc : optional preprocessed tensor, forwarded to adapter.setup()
        """
        self._init_state(image)
        setup_kwargs = {"image_proc": image_proc} if image_proc is not None else {}
        self.adapter.setup(image, target, **setup_kwargs)

        k = len(masks)
        remaining = [m.astype(np.uint8) for m in masks]
        selected: List[np.ndarray] = []

        # Phase 1: NaiveGreedy warm-start
        greedy_rounds = min(self.n_greedy, k)
        if greedy_rounds > 0:
            desc = f"Greedy warm-start ({greedy_rounds})"
            iterator = tqdm(range(greedy_rounds), desc=desc) if show_progress else range(greedy_rounds)
            for _ in iterator:
                if not remaining:
                    break
                batch = np.stack(remaining, axis=0)
                gains = self._marginal_gain(batch)
                idx = int(torch.argmax(gains).item())
                mask = remaining.pop(idx).astype(np.uint8)
                self._apply_mask(mask)
                selected.append(mask)

        # Phase 2: PhaseWindowSelector
        need = k - len(selected)
        if need > 0 and remaining:
            pw_sel = self._build_selector(k=len(remaining))
            if show_progress:
                print(f"PhaseWin selecting {need} from {len(remaining)} remaining...")
            chosen = pw_sel.select(
                remaining,
                self._marginal_gain,
                apply=lambda e: self._apply_mask(e.astype(np.uint8)),
            )
            selected.extend(chosen)

        self.adapter.teardown()

        json_dict = self._build_json()
        json_dict["phasewin_n_greedy"] = self.n_greedy
        json_dict["phasewin_effective_greedy_rounds"] = greedy_rounds
        json_dict["phasewin_window_size_override"] = self._pw_window_size_override
        json_dict["phasewin_window_frac"] = self._pw_window_frac
        json_dict["phasewin_effective_window_size"] = self._last_window_size
        json_dict["phasewin_beta_del"] = self._pw_cfg["beta_del"]
        json_dict["phasewin_alpha_sel"] = self._pw_cfg["alpha_sel_ratio"]
        json_dict["phasewin_random_frac"] = self._pw_cfg["random_frac"]
        json_dict["phasewin_window_policy"] = self._pw_cfg["window_policy"]
        json_dict["phasewin_beta_win"] = self._pw_cfg["beta_win"]
        json_dict["phasewin_eta_top2"] = self._pw_cfg["eta_top2"]
        json_dict["phasewin_enable_anneal"] = self._pw_cfg["enable_anneal"]
        json_dict["phasewin_defer_T0"] = self._pw_cfg["defer_T0"]
        json_dict["phasewin_defer_decay"] = self._pw_cfg["defer_decay"]
        json_dict["phasewin_defer_max_per_phase"] = self._pw_cfg["defer_max_per_phase"]
        json_dict["phasewin_enable_hard_exit"] = self._pw_cfg["enable_hard_exit"]
        json_dict["phasewin_hard_delta_thresh"] = self._pw_cfg["hard_delta_thresh"]
        json_dict["phasewin_hard_phi_prev"] = self._pw_cfg["hard_phi_prev"]
        return selected, json_dict
