# -*- coding: utf-8 -*-
"""
COCO caption attribution with composable Qwen2.5-VL pipelines.

Composition axes:
  task      : caption / token attribution
  adapter   : Qwen2.5-VL scorer + MLLM search / gradient adapter
  segmenter : superpixel or patch
  algorithm : greedy | phasewin | drise | gradient | llavacam | igos_pp
"""

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attribution_research import (
    MLLMAdapter,
    Qwen25VLGradientAdapter,
    Qwen25VLIGOSPPAdapter,
    Qwen25VLLLaVACAMAdapter,
    Qwen25VLTokenScorer,
    algorithm_family,
    build_segmenter_from_args,
    load_qwen25vl_model,
    normalize_algorithm_name,
    task_algorithm_choices,
)
from attribution_research.io.results import result_exists, save_npy_json
from attribution_research.runtime import AttributionContext, build_save_dir, execute_attribution


def parse_args():
    p = argparse.ArgumentParser(description="COCO caption attribution with composable Qwen2.5-VL pipelines")

    p.add_argument("--datasets", default="datasets/coco/val2017")
    p.add_argument(
        "--eval-list",
        default="datasets/Qwen2.5-VL-3B-coco-caption.json",
        help="JSON list with image_path, words, generated_ids, token positions",
    )
    p.add_argument("--model-name", default="model_checkpoint/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max-image-side",
        type=int,
        default=None,
        help="Optional longest-edge cap applied before Qwen image preprocessing.",
    )
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--save-dir", default="./caption_results/Qwen2.5-VL-3B-coco-caption")

    p.add_argument("--algorithm", choices=task_algorithm_choices("caption_vqa"), default="greedy")
    p.add_argument("--segmenter", choices=["superpixel", "patch"], default="superpixel")
    p.add_argument("--show-progress", action="store_true")

    p.add_argument("--superpixel-algorithm", choices=["slico", "seeds"], default="slico")
    p.add_argument("--division-number", type=int, default=64)
    p.add_argument("--region-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--grid-rows", type=int, default=None)
    p.add_argument("--grid-cols", type=int, default=None)

    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)

    p.add_argument("--model-type", choices=["default", "florence"], default="default")
    p.add_argument("--n-greedy", type=int, default=None)
    p.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Absolute PhaseWin window size. Default: unset, use floor(window_frac * candidate_regions).",
    )
    p.add_argument("--phasewin-window-frac", type=float, default=0.3)
    p.add_argument("--phasewin-beta-del", type=float, default=0.05)
    p.add_argument("--phasewin-alpha-sel", type=float, default=0.6)
    p.add_argument("--phasewin-random-frac", type=float, default=0.0)
    p.add_argument("--phasewin-window-policy", choices=["LG", "BA", "T2"], default="BA")
    p.add_argument("--phasewin-enable-anneal", dest="phasewin_enable_anneal", action="store_true")
    p.add_argument("--no-phasewin-anneal", dest="phasewin_enable_anneal", action="store_false")
    p.set_defaults(phasewin_enable_anneal=True)
    p.add_argument("--phasewin-enable-hard-exit", dest="phasewin_enable_hard_exit", action="store_true")
    p.add_argument("--no-phasewin-hard-exit", dest="phasewin_enable_hard_exit", action="store_false")
    p.set_defaults(phasewin_enable_hard_exit=True)
    p.add_argument("--hard-delta-thresh", type=float, default=None)
    p.add_argument("--hard-phi-prev", type=float, default=None)

    p.add_argument("--drise-n-masks", type=int, default=1000)
    p.add_argument("--drise-grid-rows", type=int, default=16)
    p.add_argument("--drise-grid-cols", type=int, default=16)
    p.add_argument("--drise-prob-thresh", type=float, default=0.5)
    p.add_argument("--drise-score-key", default="insertion_score")
    p.add_argument("--llavacam-layer-index", type=int, default=32)
    p.add_argument("--igos-mask-size", type=int, default=28)
    p.add_argument("--igos-steps", type=int, default=24)
    p.add_argument("--igos-lr", type=float, default=0.1)
    p.add_argument("--igos-blur-sigma", type=float, default=15.0)
    p.add_argument("--igos-preserve-coeff", type=float, default=2.0)
    p.add_argument("--igos-delete-coeff", type=float, default=1.0)
    p.add_argument("--igos-area-coeff", type=float, default=0.01)
    p.add_argument("--igos-tv-coeff", type=float, default=0.2)
    p.add_argument("--igos-binary-coeff", type=float, default=0.01)
    return p.parse_args()


def build_gradient_adapter(args, scorer):
    algo = normalize_algorithm_name(args.algorithm)
    if algo == "llavacam":
        return Qwen25VLLLaVACAMAdapter(
            scorer,
            layer_index=args.llavacam_layer_index,
        )
    if algo == "igos_pp":
        return Qwen25VLIGOSPPAdapter(
            scorer,
            mask_size=args.igos_mask_size,
            steps=args.igos_steps,
            lr=args.igos_lr,
            blur_sigma=args.igos_blur_sigma,
            preserve_coeff=args.igos_preserve_coeff,
            delete_coeff=args.igos_delete_coeff,
            area_coeff=args.igos_area_coeff,
            tv_coeff=args.igos_tv_coeff,
            binary_coeff=args.igos_binary_coeff,
        )
    return Qwen25VLGradientAdapter(scorer)


def _is_qwen25vl_7b(args, qwen_model) -> bool:
    candidates = [str(getattr(args, "model_name", ""))]
    config = getattr(qwen_model, "config", None)
    if config is not None:
        for attr in ("_name_or_path", "name_or_path"):
            value = getattr(config, attr, "")
            if value:
                candidates.append(str(value))
    return any("qwen2.5-vl-7b" in candidate.lower() for candidate in candidates)


def main():
    args = parse_args()
    save_dir = build_save_dir(args.save_dir, args)
    segmenter = build_segmenter_from_args(args)
    algo_name = normalize_algorithm_name(args.algorithm)
    algo_family = algorithm_family(args.algorithm)

    qwen_model, processor = load_qwen25vl_model(args.model_name, args.device)
    if _is_qwen25vl_7b(args, qwen_model) and algo_name in {"gradient", "llavacam", "igos_pp"}:
        for parameter in qwen_model.parameters():
            parameter.requires_grad_(False)
        if hasattr(qwen_model, "gradient_checkpointing_enable"):
            qwen_model.gradient_checkpointing_enable()
        if hasattr(qwen_model, "config"):
            qwen_model.config.use_cache = False
    qwen_scorer = Qwen25VLTokenScorer(
        qwen_model,
        processor,
        device=args.device,
        max_image_side=args.max_image_side,
    )
    mllm_adapter = MLLMAdapter(qwen_scorer, lambda1=args.lambda1, lambda2=args.lambda2)

    with open(args.eval_list, "r", encoding="utf-8") as f:
        items = json.load(f)
    items = items[args.begin : args.end]
    print(f"Processing {len(items)} captions -> {save_dir}")

    for item in items:
        img_path = os.path.join(args.datasets, item["image_path"]) if not os.path.isabs(item["image_path"]) else item["image_path"]
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        if result_exists(save_dir, image_id):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [warn] cannot read {img_path}")
            continue

        qwen_scorer.bind_target_from_item(item)
        regions = segmenter.segment(image)

        try:
            ordered_masks, json_dict = execute_attribution(
                AttributionContext(
                    args=args,
                    image=image,
                    regions=regions,
                    target=None,
                    build_search_adapter=lambda: mllm_adapter,
                    build_gradient_adapter=lambda: build_gradient_adapter(args, qwen_scorer),
                )
            )
        except Exception as exc:
            print(f"  [error] {image_id}: {exc}")
            continue

        json_dict["image_path"] = img_path
        json_dict["words"] = item.get("words", [])
        json_dict["selected_interpretation_token_id"] = item["selected_interpretation_token_id"]
        json_dict["selected_interpretation_token_word_id"] = item["selected_interpretation_token_word_id"]
        json_dict["algorithm"] = algo_name
        json_dict["algorithm_family"] = algo_family
        json_dict["segmenter"] = args.segmenter

        save_npy_json(ordered_masks, json_dict, save_dir, image_id)
        print(f"  [done] {image_id}  ({len(regions)} regions)")


if __name__ == "__main__":
    main()
