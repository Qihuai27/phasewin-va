# -*- coding: utf-8 -*-
"""
Visualization utilities for attribution maps.

Shared visualization helpers for attribution results across tasks.
"""

import json
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Saliency map construction
# ──────────────────────────────────────────────────────────────────────────────

_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:
    _TRAPEZOID = np.trapz

def compute_step_scores(
    scores: Union[np.ndarray, List[float]],
    baseline_score: float | None = None,
    score_mode: str = "raw",
) -> np.ndarray:
    """
    Convert a cumulative score trajectory into per-step weights.

    Parameters
    ----------
    scores          : 1-D score sequence
    baseline_score  : reference score before the first released region
    score_mode      : ``raw`` keeps the original values, ``delta`` uses
                      step-to-step differences, and ``positive_delta`` clips
                      negative deltas to zero
    """
    score_arr = np.asarray(scores, dtype=np.float32)
    if score_arr.ndim != 1:
        raise ValueError("scores must be a 1-D sequence")
    if score_arr.size == 0:
        return score_arr

    if score_mode == "raw":
        return score_arr.copy()

    prev = np.empty_like(score_arr)
    prev[0] = 0.0 if baseline_score is None else float(baseline_score)
    prev[1:] = score_arr[:-1]
    delta = score_arr - prev

    if score_mode == "delta":
        return delta
    if score_mode == "positive_delta":
        return np.clip(delta, 0.0, None)
    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


def build_attribution_map(
    ordered_masks: Union[np.ndarray, List[np.ndarray]],
    scores: List[float],
    normalize: bool = True,
    score_mode: str = "raw",
    baseline_score: float | None = None,
) -> np.ndarray:
    """
    Build a 2-D attribution map by weighting each mask by its score.

    Parameters
    ----------
    ordered_masks : (N, H, W, 1) uint8 array or list of (H,W,1) arrays
    scores        : list of N float values (e.g. insertion_score or smdl_score)
    normalize     : whether to normalize by the maximum absolute magnitude
    score_mode    : ``raw``, ``delta``, or ``positive_delta``
    baseline_score: reference score used when ``score_mode`` is not ``raw``

    Returns
    -------
    attr_map : (H, W) float32
    """
    if isinstance(ordered_masks, list):
        ordered_masks = np.stack(ordered_masks, axis=0)

    score_values = compute_step_scores(
        scores,
        baseline_score=baseline_score,
        score_mode=score_mode,
    )
    n = min(len(ordered_masks), len(score_values))
    H, W = ordered_masks.shape[1:3]
    attr_map = np.zeros((H, W), dtype=np.float32)

    for i in range(n):
        attr_map += float(score_values[i]) * ordered_masks[i, :, :, 0].astype(np.float32)

    if normalize:
        vmax = float(np.max(np.abs(attr_map)))
        if vmax > 0:
            attr_map /= vmax

    return attr_map


def build_attribution_from_result(
    npy_path: str,
    json_path: str,
    score_key: str = "insertion_score",
    normalize: bool = True,
    score_mode: str = "raw",
    baseline_score_key: str = "baseline_score",
) -> np.ndarray:
    """Load saved result files and build attribution map."""
    masks = np.load(npy_path)
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    scores = info[score_key]
    return build_attribution_map(
        masks,
        scores,
        normalize=normalize,
        score_mode=score_mode,
        baseline_score=info.get(baseline_score_key),
    )


def find_early_peak(
    region_area: Union[np.ndarray, List[float]],
    scores: Union[np.ndarray, List[float]],
    area_limit: float = 0.3,
) -> Dict[str, float]:
    """
    Find the highest score whose cumulative released area stays within a limit.

    Returns a dict containing ``index``, ``area``, and ``score``.
    """
    areas = np.asarray(region_area, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    if areas.ndim != 1 or score_arr.ndim != 1:
        raise ValueError("region_area and scores must both be 1-D sequences")
    if len(areas) != len(score_arr):
        raise ValueError("region_area and scores must have the same length")
    if len(areas) == 0:
        raise ValueError("region_area is empty")

    candidates = np.flatnonzero(areas <= float(area_limit))
    if candidates.size == 0:
        candidates = np.array([0], dtype=np.int64)
    peak_idx = int(candidates[np.argmax(score_arr[candidates])])
    return {
        "index": peak_idx,
        "area": float(areas[peak_idx]),
        "score": float(score_arr[peak_idx]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap overlay
# ──────────────────────────────────────────────────────────────────────────────

def gen_cam(
    image: np.ndarray,
    attr_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlay a saliency heatmap on an image.

    Parameters
    ----------
    image    : (H, W, 3) uint8 BGR
    attr_map : (H, W) float in [0, 1]
    alpha    : blend factor (0 = image only, 1 = heatmap only)
    colormap : OpenCV colormap constant

    Returns
    -------
    blended  : (H, W, 3) uint8
    heatmap  : (H, W, 3) uint8
    """
    H, W = image.shape[:2]
    # Resize attr_map if needed
    if attr_map.shape != (H, W):
        attr_map = cv2.resize(attr_map, (W, H))

    heatmap_uint = (attr_map * 255).clip(0, 255).astype(np.uint8)
    heatmap_bgr  = cv2.applyColorMap(heatmap_uint, colormap)
    blended      = cv2.addWeighted(image.astype(np.uint8), 1 - alpha,
                                   heatmap_bgr, alpha, 0)
    return blended, heatmap_bgr


# ──────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_bbox(
    image: np.ndarray,
    bbox: Union[list, Tuple],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    """Draw a bounding box (xyxy format) on the image."""
    img = image.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Word-level saliency (caption tasks)
# ──────────────────────────────────────────────────────────────────────────────

def get_word_saliency(
    json_dict: Dict[str, Any],
    score_key: str = "insertion_word_score",
    mode: str = "final_delta",
) -> List[float]:
    """
    Compute per-word saliency from insertion/deletion word score lists.

    ``final_delta`` uses the last insertion-minus-deletion score.
    ``auc_delta`` integrates that difference over released area.
    """
    ins_scores = np.asarray(json_dict.get(score_key, []), dtype=np.float32)
    if ins_scores.size == 0:
        return []

    if ins_scores.ndim != 2:
        raise ValueError(f"{score_key} must be a 2-D step-by-word array")

    del_raw = json_dict.get("deletion_word_score", [])
    del_scores = (
        np.asarray(del_raw, dtype=np.float32)
        if len(del_raw) > 0
        else np.zeros_like(ins_scores, dtype=np.float32)
    )
    if del_scores.shape != ins_scores.shape:
        raise ValueError("deletion_word_score must match insertion_word_score shape")

    combined = ins_scores - del_scores
    if mode == "final_delta":
        return combined[-1].tolist()
    if mode == "auc_delta":
        areas = np.asarray([0.0] + list(json_dict.get("region_area", [])), dtype=np.float32)
        curves = np.concatenate(
            [np.zeros((1, combined.shape[1]), dtype=np.float32), combined],
            axis=0,
        )
        return _TRAPEZOID(curves, x=areas, axis=0).tolist()
    raise ValueError(f"Unsupported word saliency mode: {mode!r}")


def build_word_region_matrix(
    json_dict: Dict[str, Any],
    score_key: str = "insertion_word_score",
    mode: str = "delta",
) -> np.ndarray:
    """
    Build a word-by-release-step matrix for caption attribution visualization.

    Returns
    -------
    matrix : (num_words, num_steps) float32
        Each column corresponds to one released region step.
    """
    ins_scores = np.asarray(json_dict.get(score_key, []), dtype=np.float32)
    if ins_scores.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if ins_scores.ndim != 2:
        raise ValueError(f"{score_key} must be a 2-D step-by-word array")

    del_raw = json_dict.get("deletion_word_score", [])
    del_scores = (
        np.asarray(del_raw, dtype=np.float32)
        if len(del_raw) > 0
        else np.zeros_like(ins_scores, dtype=np.float32)
    )
    if del_scores.shape != ins_scores.shape:
        raise ValueError("deletion_word_score must match insertion_word_score shape")

    combined = ins_scores - del_scores
    if mode == "cumulative":
        matrix = combined
    else:
        prev = np.zeros_like(combined)
        prev[1:] = combined[:-1]
        matrix = combined - prev
        if mode == "positive_delta":
            matrix = np.clip(matrix, 0.0, None)
        elif mode != "delta":
            raise ValueError(f"Unsupported word-region matrix mode: {mode!r}")
    return matrix.T.astype(np.float32, copy=False)


def visualize_word_saliency(
    words: List[str],
    saliency_scores: List[float],
    max_words: int = 30,
) -> str:
    """
    Return a simple text representation of word saliency.
    (For rich visualization use matplotlib in the task scripts.)
    """
    pairs = list(zip(words, saliency_scores))[:max_words]
    lines = []
    for word, score in pairs:
        bar_len = max(0, int(score * 20))
        lines.append(f"{word:20s} {'█' * bar_len} ({score:.3f})")
    return "\n".join(lines)
