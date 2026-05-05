# -*- coding: utf-8 -*-
"""
ImageNet attribution with composable CLIP pipelines.

Composition axes:
  task      : ImageNet classification
  adapter   : CLIP search / gradient adapter
  segmenter : superpixel or patch
  algorithm : greedy | phasewin | drise | dhsic | gradient | grad_eclip | ig2 | igos_pp
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attribution_research import (
    CLIPGradEClipAdapter,
    CLIPGradientAdapter,
    CLIPIG2Adapter,
    CLIPIGOSPPAdapter,
    CLIPSearchAdapter,
    CLIPXpliqueWrapper,
    algorithm_family,
    build_segmenter_from_args,
    normalize_algorithm_name,
    task_algorithm_choices,
)
from attribution_research.io.results import result_exists, save_npy_json
from attribution_research.runtime import AttributionContext, build_save_dir, execute_attribution


def parse_args():
    p = argparse.ArgumentParser(description="ImageNet attribution with composable CLIP pipelines")

    p.add_argument("--datasets", default="datasets/imagenet/ILSVRC2012_img_val")
    p.add_argument(
        "--eval-list",
        default="datasets/imagenet/generated/clip_vitl14_true.txt",
        help=(
            "Text file: each line = <image_path> <class_id>. "
            "Default points to the generated model-specific true split."
        ),
    )
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--save-dir", default="./classification_results/imagenet-clip-vitl")

    p.add_argument(
        "--algorithm",
        choices=task_algorithm_choices("classification"),
        default="phasewin",
    )
    p.add_argument("--segmenter", choices=["superpixel", "patch"], default="superpixel")
    p.add_argument("--show-progress", action="store_true")

    p.add_argument("--clip-type", default="ViT-L/14")
    p.add_argument("--clip-download-root", default=".checkpoints/CLIP")
    p.add_argument(
        "--semantic-features",
        default="ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--superpixel-algorithm", choices=["slico", "seeds"], default="slico")
    p.add_argument("--region-size", type=int, default=None)
    p.add_argument("--division-number", type=int, default=50)

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

    p.add_argument("--dhsic-batch-size", type=int, default=32)
    p.add_argument(
        "--dhsic-tf-device",
        choices=["cpu", "auto", "gpu"],
        default="cpu",
        help=(
            "TensorFlow device policy for xplique. "
            "Default `cpu` avoids TensorFlow competing with PyTorch for CUDA memory."
        ),
    )
    p.add_argument("--gradient-score-mode", choices=["prob", "logit"], default="prob")
    p.add_argument("--grad-eclip-layer-span", type=int, default=1)
    p.add_argument("--ig2-steps", type=int, default=32)
    p.add_argument("--ig2-step-size", type=float, default=8.0)
    p.add_argument("--ig2-blur-sigmas", type=float, nargs="+", default=[3.0, 7.0, 15.0, 31.0])
    p.add_argument("--ig2-score-mode", choices=["prob", "logit"], default="prob")
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


def build_search_adapter(args):
    return CLIPSearchAdapter(
        clip_type=args.clip_type,
        semantic_feature_path=args.semantic_features,
        download_root=args.clip_download_root,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        batch_size=args.batch_size,
        device=args.device,
    )


def build_gradient_adapter(args):
    algo = normalize_algorithm_name(args.algorithm)
    if algo == "grad_eclip":
        return CLIPGradEClipAdapter(
            clip_type=args.clip_type,
            semantic_feature_path=args.semantic_features,
            download_root=args.clip_download_root,
            device=args.device,
            layer_span=args.grad_eclip_layer_span,
        )
    if algo == "ig2":
        return CLIPIG2Adapter(
            clip_type=args.clip_type,
            semantic_feature_path=args.semantic_features,
            download_root=args.clip_download_root,
            device=args.device,
            steps=args.ig2_steps,
            step_size=args.ig2_step_size,
            blur_sigmas=args.ig2_blur_sigmas,
            score_mode=args.ig2_score_mode,
        )
    if algo == "igos_pp":
        return CLIPIGOSPPAdapter(
            clip_type=args.clip_type,
            semantic_feature_path=args.semantic_features,
            download_root=args.clip_download_root,
            device=args.device,
            score_mode=args.gradient_score_mode,
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
    return CLIPGradientAdapter(
        clip_type=args.clip_type,
        semantic_feature_path=args.semantic_features,
        download_root=args.clip_download_root,
        device=args.device,
        score_mode=args.gradient_score_mode,
    )


def build_dhsic_model(args):
    return CLIPXpliqueWrapper(
        clip_type=args.clip_type,
        semantic_feature_path=args.semantic_features,
        download_root=args.clip_download_root,
        device=args.device,
    )


def main():
    args = parse_args()
    save_dir = build_save_dir(args.save_dir, args)
    segmenter = build_segmenter_from_args(args)
    algo_family = algorithm_family(args.algorithm)

    with open(args.eval_list, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    items = []
    for line in lines:
        parts = line.split()
        items.append({"path": parts[0], "label": int(parts[1]) if len(parts) > 1 else 0})
    items = items[args.begin : args.end]
    print(f"Processing {len(items)} images -> {save_dir}")

    for item in items:
        img_path = os.path.join(args.datasets, item["path"]) if not os.path.isabs(item["path"]) else item["path"]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        label = item["label"]

        if result_exists(save_dir, image_id):
            print(f"  [skip] {image_id}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [warn] cannot read {img_path}, skipping")
            continue

        image = cv2.resize(image, (224, 224))
        regions = segmenter.segment(image)

        try:
            ordered_masks, json_dict = execute_attribution(
                AttributionContext(
                    args=args,
                    image=image,
                    regions=regions,
                    target=label,
                    build_search_adapter=lambda: build_search_adapter(args),
                    build_gradient_adapter=lambda: build_gradient_adapter(args),
                    build_dhsic_model=lambda: build_dhsic_model(args),
                )
            )
        except Exception as exc:
            print(f"  [error] {image_id}: {exc}")
            continue

        json_dict["image_path"] = img_path
        json_dict["target_label"] = label
        json_dict["algorithm"] = normalize_algorithm_name(args.algorithm)
        json_dict["algorithm_family"] = algo_family
        json_dict["segmenter"] = args.segmenter

        save_npy_json(ordered_masks, json_dict, save_dir, image_id)
        print(f"  [done] {image_id}  ({len(regions)} regions)")


if __name__ == "__main__":
    main()
