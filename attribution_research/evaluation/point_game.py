# -*- coding: utf-8 -*-
"""
Point Game and Energy Point Game evaluation metrics.

Consolidates point-based localization metrics used across the repository.

Metrics
-------
Point Game        : 1 if max saliency point falls inside the GT bbox/mask, else 0.
Energy Point Game : fraction of total saliency energy inside the GT bbox/mask.
"""

import argparse
import json
import os
from typing import Optional, Tuple, Union

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Saliency map construction
# ──────────────────────────────────────────────────────────────────────────────

def build_saliency_map(
    npy_path: str,
    json_path: str,
    score_key: str = "insertion_score",
) -> np.ndarray:
    """
    Build a 2-D saliency map from ordered masks and their scores.

    Parameters
    ----------
    npy_path   : path to .npy file with stacked masks (N, H, W, 1) uint8
    json_path  : path to companion .json file with score list
    score_key  : JSON key whose list of scores weight each mask

    Returns
    -------
    saliency : (H, W) float32, normalized to [0, 1]
    """
    masks  = np.load(npy_path)                        # (N, H, W, 1)
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    scores = np.array(info[score_key], dtype=np.float32)
    n = min(len(masks), len(scores))
    saliency = np.zeros(masks.shape[1:3], dtype=np.float32)
    for i in range(n):
        saliency += scores[i] * masks[i, :, :, 0].astype(np.float32)

    s_max = saliency.max()
    if s_max > 0:
        saliency /= s_max
    return saliency


# ──────────────────────────────────────────────────────────────────────────────
# Point game
# ──────────────────────────────────────────────────────────────────────────────

def point_game_box(
    saliency_map: np.ndarray,
    bbox: Union[list, Tuple[int, int, int, int]],
) -> int:
    """
    Returns 1 if the saliency maximum is inside the bounding box.

    Parameters
    ----------
    saliency_map : (H, W) float
    bbox         : (x1, y1, x2, y2) pixel coordinates
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    H, W = saliency_map.shape
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    in_box = np.zeros_like(saliency_map)
    in_box[y1:y2, x1:x2] = 1.0
    return int(in_box.max() == (in_box * saliency_map).max() and saliency_map.max() > 0)


def point_game_mask(
    saliency_map: np.ndarray,
    gt_mask: np.ndarray,
) -> int:
    """
    Returns 1 if the saliency maximum is inside the ground-truth mask.

    Parameters
    ----------
    saliency_map : (H, W) float
    gt_mask      : (H, W) binary mask (1 = object)
    """
    masked = saliency_map * (gt_mask > 0).astype(np.float32)
    return int(masked.max() == saliency_map.max() and saliency_map.max() > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Energy point game
# ──────────────────────────────────────────────────────────────────────────────

def energy_point_game_box(
    saliency_map: np.ndarray,
    bbox: Union[list, Tuple[int, int, int, int]],
) -> float:
    """
    Fraction of total saliency energy inside the bounding box.
    Returns 0.0 if total energy is zero.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    H, W = saliency_map.shape
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    total = float(saliency_map.sum())
    if total == 0 or x2 <= x1 or y2 <= y1:
        return 0.0
    inside = float(saliency_map[y1:y2, x1:x2].sum())
    return inside / total


def energy_point_game_mask(
    saliency_map: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Fraction of total saliency energy inside the ground-truth mask.
    """
    total = float(saliency_map.sum())
    if total == 0:
        return 0.0
    inside = float((saliency_map * (gt_mask > 0).astype(np.float32)).sum())
    return inside / total


# ──────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_point_game(
    result_dir: str,
    annotation_list: list,
    score_key: str = "insertion_score",
    use_mask: bool = False,
) -> dict:
    """
    Compute Point Game and Energy Point Game over a dataset.

    Parameters
    ----------
    result_dir       : directory with npy/ and json/ subdirs
    annotation_list  : list of dicts, each with:
                         "image_id"  : str (without extension)
                         "bbox"      : [x1, y1, x2, y2]
                         "mask_path" : str (optional, for mask mode)
    score_key        : JSON key for saliency scores
    use_mask         : use segmentation mask instead of bbox

    Returns
    -------
    dict with "point_game", "energy_point_game", "n_samples"
    """
    pg_scores, epg_scores = [], []

    for item in tqdm(annotation_list, desc="Point Game"):
        img_id   = item["image_id"]
        npy_path  = os.path.join(result_dir, "npy",  f"{img_id}.npy")
        json_path = os.path.join(result_dir, "json", f"{img_id}.json")

        if not (os.path.exists(npy_path) and os.path.exists(json_path)):
            continue

        saliency = build_saliency_map(npy_path, json_path, score_key=score_key)

        if use_mask and "mask_path" in item:
            import cv2
            gt_mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 0).astype(np.uint8)
            pg_scores.append(point_game_mask(saliency, gt_mask))
            epg_scores.append(energy_point_game_mask(saliency, gt_mask))
        else:
            bbox = item["bbox"]
            pg_scores.append(point_game_box(saliency, bbox))
            epg_scores.append(energy_point_game_box(saliency, bbox))

    return {
        "point_game":        float(np.mean(pg_scores))  if pg_scores  else 0.0,
        "energy_point_game": float(np.mean(epg_scores)) if epg_scores else 0.0,
        "n_samples":         len(pg_scores),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Point Game Evaluation")
    parser.add_argument("--explanation-dir", required=True)
    parser.add_argument("--annotation-file", required=True,
                        help="JSON file with list of {image_id, bbox} dicts")
    parser.add_argument("--use-mask",  action="store_true",
                        help="Use segmentation mask instead of bbox")
    parser.add_argument("--score-key", default="insertion_score")
    args = parser.parse_args()

    with open(args.annotation_file, "r") as f:
        annotations = json.load(f)
    if isinstance(annotations, dict):
        annotations = annotations.get("annotations", list(annotations.values()))

    results = evaluate_point_game(
        args.explanation_dir,
        annotations,
        score_key=args.score_key,
        use_mask=args.use_mask,
    )
    print(f"\n=== Point Game ({results['n_samples']} samples) ===")
    print(f"Point Game        : {results['point_game']:.4f}")
    print(f"Energy Point Game : {results['energy_point_game']:.4f}")


if __name__ == "__main__":
    main()
