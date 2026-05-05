# -*- coding: utf-8 -*-
"""
Unified AUC faithfulness evaluation.

Consolidates the task-specific AUC evaluation flows used by the repository.

Auto-detects task type from JSON keys (classification / detection / caption).
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn import metrics
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AUCResult:
    insertion_auc: float = 0.0
    deletion_auc:  float = 0.0
    # Detection-specific (None for other tasks)
    insertion_iou_auc: Optional[float] = None
    deletion_iou_auc:  Optional[float] = None
    insertion_cls_auc: Optional[float] = None
    deletion_cls_auc:  Optional[float] = None
    average_highest:   float = 0.0
    average_highest_30pct_area: float = 0.0
    average_highest_50pct_area: float = 0.0
    average_model_forward_calls: Optional[float] = None
    average_eval_model_forward_calls: Optional[float] = None
    average_total_model_forward_calls: Optional[float] = None
    mufidelity: Optional[float] = None
    mufidelity_n_samples: Optional[int] = None
    # Caption/VQA-specific
    insertion_sensitivity_auc: Optional[float] = None
    deletion_sensitivity_auc:  Optional[float] = None
    sensitivity_highest:       Optional[float] = None
    n_samples: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Per-file computation
# ──────────────────────────────────────────────────────────────────────────────

def _detect_task(json_dict: Dict[str, Any]) -> str:
    if "insertion_iou" in json_dict:
        return "detection"
    if "insertion_word_score" in json_dict:
        return "caption"
    return "classification"


def _highest_score_within_area_limit(
    region_area: List[float],
    scores: List[float],
    area_limit: float,
) -> float:
    areas = np.asarray(region_area, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    if areas.ndim != 1 or score_arr.ndim != 1:
        raise ValueError("region_area and scores must both be 1-D sequences")
    if len(areas) != len(score_arr):
        raise ValueError("region_area and scores must have the same length")
    if len(areas) == 0:
        return 0.0

    candidates = np.flatnonzero(areas <= float(area_limit))
    if candidates.size == 0:
        candidates = np.array([0], dtype=np.int64)
    return float(score_arr[candidates].max())


def compute_auc_from_json(
    saved_json: Dict[str, Any],
    sensitivity: float = 0.2,
) -> Dict[str, Any]:
    """
    Compute AUC metrics from one saved JSON dict.

    Returns a dict of per-sample metrics compatible with aggregate_results().
    """
    task = _detect_task(saved_json)
    result: Dict[str, Any] = {"task": task}

    insertion_area = np.array([0.0] + saved_json["region_area"])
    deletion_area  = 1.0 - insertion_area

    ins_score  = np.array([saved_json["deletion_score"][-1]] + saved_json["insertion_score"])
    del_score  = np.array([saved_json["insertion_score"][-1]] + saved_json["deletion_score"])

    result["insertion_auc"] = float(metrics.auc(insertion_area, ins_score))
    result["deletion_auc"]  = float(metrics.auc(deletion_area,  del_score))
    result["highest_score"] = float(ins_score.max())
    result["highest_score_30pct_area"] = _highest_score_within_area_limit(
        saved_json["region_area"],
        saved_json["insertion_score"],
        area_limit=0.3,
    )
    result["highest_score_50pct_area"] = _highest_score_within_area_limit(
        saved_json["region_area"],
        saved_json["insertion_score"],
        area_limit=0.5,
    )
    if "model_forward_calls" in saved_json:
        result["model_forward_calls"] = float(saved_json["model_forward_calls"])
    if "eval_model_forward_calls" in saved_json:
        result["eval_model_forward_calls"] = float(saved_json["eval_model_forward_calls"])
    if "total_model_forward_calls" in saved_json:
        result["total_model_forward_calls"] = float(saved_json["total_model_forward_calls"])
    elif "model_forward_calls" in result and "eval_model_forward_calls" in result:
        result["total_model_forward_calls"] = (
            result["model_forward_calls"] + result["eval_model_forward_calls"]
        )

    # ── detection extras ──────────────────────────────────────────────────────
    if task == "detection":
        for key, out in [
            ("insertion_iou", "insertion_iou"),
            ("deletion_iou",  "deletion_iou"),
            ("insertion_cls", "insertion_cls"),
            ("deletion_cls",  "deletion_cls"),
        ]:
            if key in saved_json:
                arr_ins = np.array([saved_json["deletion_" + key.split("_")[1]][-1]] + saved_json[key])
                arr_del = np.array([saved_json["insertion_" + key.split("_")[1]][-1]] + saved_json[key])
                result[f"{key}_auc"]     = float(metrics.auc(insertion_area, arr_ins))
                result[f"del_{key}_auc"] = float(metrics.auc(deletion_area,  arr_del))
        # Highest cls@IoU>0.5
        if "insertion_iou" in saved_json and "insertion_cls" in saved_json:
            iou_arr = np.array([saved_json["deletion_iou"][-1]]  + saved_json["insertion_iou"])
            cls_arr = np.array([saved_json["deletion_cls"][-1]]  + saved_json["insertion_cls"])
            result["highest_cls_50"] = float(((iou_arr > 0.5) * cls_arr).max())
            result["highest_cls_75"] = float(((iou_arr > 0.75) * cls_arr).max())

    # ── caption / VQA extras ─────────────────────────────────────────────────
    if task == "caption" and "insertion_word_score" in saved_json:
        last_ins_words = np.array(saved_json["insertion_word_score"][-1])
        last_del_words = np.array(saved_json["deletion_word_score"][-1])
        sensitive = (last_ins_words - last_del_words) > sensitivity
        if sensitive.sum() > 0:
            def _word_mean(scores_list, idx_mask):
                return [np.array(s)[idx_mask].mean() for s in scores_list]

            ins_sens = np.array(
                [last_del_words[sensitive].mean()]
                + _word_mean(saved_json["insertion_word_score"], sensitive)
            )
            del_sens = np.array(
                [last_ins_words[sensitive].mean()]
                + _word_mean(saved_json["deletion_word_score"], sensitive)
            )
            result["insertion_sensitivity_auc"] = float(metrics.auc(insertion_area, ins_sens))
            result["deletion_sensitivity_auc"]  = float(metrics.auc(deletion_area,  del_sens))
            result["sensitivity_highest"]       = float(ins_sens.max())

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_results(
    result_dir: str,
    sensitivity: float = 0.2,
    json_subdir: str = "json",
    recursive: bool = False,
    show_progress: bool = True,
) -> AUCResult:
    """
    Scan all JSON files under result_dir/json/, compute per-file AUC, return
    an AUCResult with aggregated means.

    Parameters
    ----------
    result_dir   : path to the method result directory (contains json/ subdir)
    sensitivity  : threshold for caption word sensitivity filtering
    json_subdir  : name of the JSON subdirectory (default "json")
    recursive    : whether to scan subdirectories of json_subdir
    """
    json_root = os.path.join(result_dir, json_subdir)
    if not os.path.isdir(json_root):
        raise FileNotFoundError(f"JSON directory not found: {json_root}")

    if recursive:
        all_paths = []
        for root, _, fnames in os.walk(json_root):
            for fn in fnames:
                if fn.endswith(".json"):
                    all_paths.append(os.path.join(root, fn))
    else:
        all_paths = [
            os.path.join(json_root, fn)
            for fn in os.listdir(json_root)
            if fn.endswith(".json")
        ]

    if not all_paths:
        raise FileNotFoundError(f"No .json files found under {json_root}")

    per_sample: List[Dict[str, Any]] = []
    iterator = tqdm(all_paths, desc="Computing AUC") if show_progress else all_paths
    for path in iterator:
        with open(path, "r", encoding="utf-8") as f:
            jd = json.load(f)
        per_sample.append(compute_auc_from_json(jd, sensitivity=sensitivity))

    def _mean(key: str) -> Optional[float]:
        vals = [s[key] for s in per_sample if key in s]
        return float(np.mean(vals)) if vals else None

    agg = AUCResult(
        insertion_auc     = _mean("insertion_auc") or 0.0,
        deletion_auc      = _mean("deletion_auc")  or 0.0,
        average_highest   = _mean("highest_score") or 0.0,
        average_highest_30pct_area = _mean("highest_score_30pct_area") or 0.0,
        average_highest_50pct_area = _mean("highest_score_50pct_area") or 0.0,
        average_model_forward_calls = _mean("model_forward_calls"),
        average_eval_model_forward_calls = _mean("eval_model_forward_calls"),
        average_total_model_forward_calls = _mean("total_model_forward_calls"),
        insertion_iou_auc = _mean("insertion_iou_auc"),
        deletion_iou_auc  = _mean("del_deletion_iou_auc"),
        insertion_cls_auc = _mean("insertion_cls_auc"),
        deletion_cls_auc  = _mean("del_deletion_cls_auc"),
        insertion_sensitivity_auc = _mean("insertion_sensitivity_auc"),
        deletion_sensitivity_auc  = _mean("deletion_sensitivity_auc"),
        sensitivity_highest       = _mean("sensitivity_highest"),
        n_samples         = len(per_sample),
    )
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faithfulness AUC Evaluation")
    parser.add_argument("--explanation-dir", required=True,
                        help="Result directory (contains json/ subdir)")
    parser.add_argument("--sensitivity", type=float, default=0.2,
                        help="Word sensitivity threshold for caption tasks")
    parser.add_argument("--recursive", action="store_true",
                        help="Scan JSON subdirectories recursively")
    args = parser.parse_args()

    agg = aggregate_results(
        args.explanation_dir,
        sensitivity=args.sensitivity,
        recursive=args.recursive,
    )

    print(f"\n=== AUC Results ({agg.n_samples} samples) ===")
    print(f"Insertion AUC : {agg.insertion_auc:.4f}")
    print(f"Deletion  AUC : {agg.deletion_auc:.4f}")
    print(f"Avg. highest  : {agg.average_highest:.4f}")
    print(f"Avg. highest @30% area : {agg.average_highest_30pct_area:.4f}")
    print(f"Avg. highest @50% area : {agg.average_highest_50pct_area:.4f}")
    if agg.average_model_forward_calls is not None:
        print(f"Avg. model forwards    : {agg.average_model_forward_calls:.2f}")
    if agg.average_eval_model_forward_calls is not None:
        print(f"Avg. eval forwards     : {agg.average_eval_model_forward_calls:.2f}")
    if agg.average_total_model_forward_calls is not None:
        print(f"Avg. total forwards    : {agg.average_total_model_forward_calls:.2f}")

    if agg.insertion_iou_auc is not None:
        print(f"\n-- Detection --")
        print(f"Insertion IoU AUC : {agg.insertion_iou_auc:.4f}")
        print(f"Insertion CLS AUC : {agg.insertion_cls_auc:.4f}")

    if agg.insertion_sensitivity_auc is not None:
        print(f"\n-- Caption sensitivity --")
        print(f"Ins sensitivity AUC : {agg.insertion_sensitivity_auc:.4f}")
        print(f"Del sensitivity AUC : {agg.deletion_sensitivity_auc:.4f}")


if __name__ == "__main__":
    main()
