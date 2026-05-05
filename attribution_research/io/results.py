# -*- coding: utf-8 -*-
"""
Output saving and loading utilities.

All attribution scripts produce:
  save_dir/
    npy/  <image_id>.npy    -- (N, H, W, 1) uint8 ordered masks
    json/ <image_id>.json   -- score metadata dict
"""

import json
import os
from typing import Any, Dict, List, Union

import numpy as np


def mkdir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_npy_json(
    ordered_masks: List[np.ndarray],
    json_dict: Dict[str, Any],
    save_dir: str,
    image_id: str,
    npy_subdir: str = "npy",
    json_subdir: str = "json",
) -> None:
    """
    Save attribution results for one image.

    Parameters
    ----------
    ordered_masks : list of N (H, W, 1) uint8 masks, best-first
    json_dict     : serializable metadata dict
    save_dir      : root directory
    image_id      : filename stem (without extension)
    """
    npy_dir  = os.path.join(save_dir, npy_subdir)
    json_dir = os.path.join(save_dir, json_subdir)
    mkdir(npy_dir)
    mkdir(json_dir)

    # Stack masks: (N, H, W, 1) uint8
    if ordered_masks:
        arr = np.stack(ordered_masks, axis=0).astype(np.uint8)
    else:
        arr = np.empty((0,), dtype=np.uint8)
    np.save(os.path.join(npy_dir, f"{image_id}.npy"), arr)

    with open(os.path.join(json_dir, f"{image_id}.json"), "w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=2)


def load_result(
    save_dir: str,
    image_id: str,
    npy_subdir: str = "npy",
    json_subdir: str = "json",
):
    """
    Load saved attribution results.

    Returns
    -------
    masks    : (N, H, W, 1) uint8 ndarray
    json_dict: dict
    """
    npy_path  = os.path.join(save_dir, npy_subdir,  f"{image_id}.npy")
    json_path = os.path.join(save_dir, json_subdir, f"{image_id}.json")
    masks = np.load(npy_path)
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return masks, info


def result_exists(save_dir: str, image_id: str) -> bool:
    """Return True if both .npy and .json files exist for image_id."""
    return (
        os.path.exists(os.path.join(save_dir, "npy",  f"{image_id}.npy")) and
        os.path.exists(os.path.join(save_dir, "json", f"{image_id}.json"))
    )
