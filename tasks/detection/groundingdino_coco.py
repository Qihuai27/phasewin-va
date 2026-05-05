# -*- coding: utf-8 -*-
"""
COCO detection attribution with composable GroundingDINO pipelines.

Composition axes:
  task      : object detection
  adapter   : GroundingDINO search / gradient adapter
  segmenter : superpixel or patch
  algorithm : greedy | phasewin | drise | gradient | gradcam | odam | ssgrad_cam_pp
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attribution_research import (
    COCO_TEXT_PROMPT,
    GroundingDINOAdapter,
    GroundingDINOCAMAdapter,
    GroundingDINOGradientAdapter,
    algorithm_family,
    build_segmenter_from_args,
    normalize_algorithm_name,
    task_algorithm_choices,
)
from attribution_research.io.results import result_exists, save_npy_json
from attribution_research.runtime import AttributionContext, build_save_dir, execute_attribution


def parse_args():
    p = argparse.ArgumentParser(description="COCO detection attribution with composable GroundingDINO pipelines")

    p.add_argument("--datasets", default="datasets/coco/val2017")
    p.add_argument(
        "--eval-list",
        default="datasets/coco_groundingdino_correct_detection.json",
        help="JSON list: [{image_path, class_id, bbox, caption?}, ...]",
    )
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--save-dir", default="./detection_results/coco-groundingdino")

    p.add_argument("--algorithm", choices=task_algorithm_choices("detection"), default="phasewin")
    p.add_argument("--segmenter", choices=["superpixel", "patch"], default="superpixel")
    p.add_argument("--show-progress", action="store_true")
    p.add_argument("--debug-traceback", action="store_true")

    p.add_argument("--groundingdino-config", default="config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--groundingdino-weights", default="ckpt/groundingdino_swint_ogc.pth")
    p.add_argument("--caption", default=COCO_TEXT_PROMPT)
    p.add_argument("--device", default="cuda")

    p.add_argument("--superpixel-algorithm", choices=["slico", "seeds"], default="slico")
    p.add_argument("--division-number", type=int, default=100)
    p.add_argument("--region-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--grid-rows", type=int, default=None)
    p.add_argument("--grid-cols", type=int, default=None)

    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--mode", choices=["object", "iou", "cls"], default="object")
    p.add_argument("--batch-size", type=int, default=4)

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
    p.add_argument("--cam-feature-level", type=int, default=2)
    return p.parse_args()


def _enable_hf_offline_if_local_bert_cache_exists() -> None:
    bert_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--bert-base-uncased"
    if bert_cache.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _install_groundingdino_transformers_compat() -> None:
    try:
        import inspect
        from transformers import BertModel
        from transformers.modeling_utils import ModuleUtilsMixin

        def _compat_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=self.dtype)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        def _compat_get_extended_attention_mask(self, attention_mask, input_shape, device=None):
            if device is not None:
                attention_mask = attention_mask.to(device)
            return ModuleUtilsMixin.get_extended_attention_mask(self, attention_mask, input_shape)

        if not hasattr(BertModel, "get_head_mask"):
            BertModel.get_head_mask = _compat_get_head_mask
        if "device" not in inspect.signature(BertModel.get_extended_attention_mask).parameters:
            BertModel.get_extended_attention_mask = _compat_get_extended_attention_mask
    except ImportError:
        pass


def load_groundingdino(config_path: str, weights_path: str, device: str):
    _enable_hf_offline_if_local_bert_cache_exists()
    _install_groundingdino_transformers_compat()
    try:
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO is not installed. "
            "Follow https://github.com/IDEA-Research/GroundingDINO to install."
        ) from exc
    return load_model(config_path, weights_path, device=device)


def build_preprocess():
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def preprocess(image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return transform(image_rgb)

    return preprocess


def resolve_caption(args, item) -> str:
    return item.get("caption") or args.caption


def build_search_adapter(args, model, preprocess, caption: str):
    return GroundingDINOAdapter(
        model=model,
        preprocess_fn=preprocess,
        caption=caption,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        batch_size=args.batch_size,
        mode=args.mode,
        device=args.device,
    )


def build_gradient_adapter(args, model, preprocess, caption: str):
    algo = normalize_algorithm_name(args.algorithm)
    if algo in {"gradcam", "odam", "ssgrad_cam_pp"}:
        return GroundingDINOCAMAdapter(
            model=model,
            preprocess_fn=preprocess,
            caption=caption,
            mode=args.mode,
            device=args.device,
            variant=algo,
            feature_level=args.cam_feature_level,
        )
    return GroundingDINOGradientAdapter(
        model=model,
        preprocess_fn=preprocess,
        caption=caption,
        mode=args.mode,
        device=args.device,
    )


def main():
    args = parse_args()
    save_dir = build_save_dir(args.save_dir, args)

    gd_model = load_groundingdino(args.groundingdino_config, args.groundingdino_weights, args.device)
    preprocess = build_preprocess()
    segmenter = build_segmenter_from_args(args)
    algo_family = algorithm_family(args.algorithm)

    with open(args.eval_list, "r", encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = items.get("annotations", list(items.values()))
    items = items[args.begin : args.end]
    print(f"Processing {len(items)} detections -> {save_dir}")

    for idx, item in enumerate(items):
        img_path = os.path.join(args.datasets, item["image_path"]) if not os.path.isabs(item["image_path"]) else item["image_path"]
        base_id = str(item.get("image_id", os.path.splitext(os.path.basename(img_path))[0]))
        # Append eval-list index to disambiguate multiple annotations per image.
        # The dataset has 394 items but only 378 unique image_ids; without this,
        # the second annotation for a shared image_id would be skipped by result_exists().
        image_id = f"{base_id}_{args.begin + idx}"
        caption = resolve_caption(args, item)

        if result_exists(save_dir, image_id):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [warn] cannot read {img_path}")
            continue

        regions = segmenter.segment(image)
        target = {"label": item["class_id"], "box": item["bbox"]}

        try:
            ordered_masks, json_dict = execute_attribution(
                AttributionContext(
                    args=args,
                    image=image,
                    regions=regions,
                    target=target,
                    build_search_adapter=lambda caption=caption: build_search_adapter(
                        args, gd_model, preprocess, caption
                    ),
                    build_gradient_adapter=lambda caption=caption: build_gradient_adapter(
                        args, gd_model, preprocess, caption
                    ),
                )
            )
        except Exception as exc:
            print(f"  [error] {image_id}: {exc}")
            if args.debug_traceback:
                import traceback
                traceback.print_exc()
            continue

        json_dict["image_path"] = img_path
        json_dict["caption"] = caption
        json_dict["algorithm"] = normalize_algorithm_name(args.algorithm)
        json_dict["algorithm_family"] = algo_family
        json_dict["segmenter"] = args.segmenter
        save_npy_json(ordered_masks, json_dict, save_dir, image_id)
        print(f"  [done] {image_id}  ({len(regions)} regions)")


if __name__ == "__main__":
    main()
