#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate CLIP ImageNet zero-shot semantic features.

This is mainly used to create the missing RN101 semantic feature file needed by
the first-round supplementary experiments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


CLIP_IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLIP ImageNet semantic features")
    parser.add_argument("--model", default="RN101", help="CLIP model variant, e.g. RN101 or ViT-L/14")
    parser.add_argument(
        "--download-root",
        default=".checkpoints/CLIP",
        help="CLIP cache directory",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Default: ckpt/semantic_features/clip_<variant>_imagenet_zeroweights.pt",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Optional debug limit for the number of ImageNet classes",
    )
    return parser.parse_args()


def default_out_path(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key == "rn101":
        tag = "rn101"
    elif key in {"vit-l/14", "vitl/14", "vit-l14", "vitl14"}:
        tag = "vitl"
    else:
        tag = key.replace("/", "").replace("-", "").replace(" ", "_")
    return f"ckpt/semantic_features/clip_{tag}_imagenet_zeroweights.pt"


def load_imagenet_categories() -> list[str]:
    import torchvision.models as tv_models

    weights = tv_models.ResNet50_Weights.DEFAULT
    return list(weights.meta["categories"])


def main() -> None:
    args = parse_args()

    try:
        import clip as openai_clip
    except ImportError as exc:
        raise ImportError(
            "openai-clip is required. Install with: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    out_path = args.out or default_out_path(args.model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    categories = load_imagenet_categories()
    if args.max_classes is not None:
        categories = categories[: int(args.max_classes)]

    print(f"Model      : {args.model}")
    print(f"Classes    : {len(categories)}")
    print(f"Templates  : {len(CLIP_IMAGENET_TEMPLATES)}")
    print(f"Device     : {args.device}")
    print(f"Output     : {out_path}")

    model, _ = openai_clip.load(
        args.model,
        device=args.device,
        download_root=args.download_root,
    )
    model = model.float().eval()

    all_weights = []
    with torch.no_grad():
        for idx, class_name in enumerate(categories):
            prompts = [template.format(class_name) for template in CLIP_IMAGENET_TEMPLATES]
            tokens = openai_clip.tokenize(prompts).to(args.device)
            text_features = model.encode_text(tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            class_feature = text_features.mean(dim=0)
            class_feature /= class_feature.norm()
            all_weights.append(class_feature)
            if (idx + 1) % 100 == 0 or idx + 1 == len(categories):
                print(f"  encoded {idx + 1}/{len(categories)} classes")

    weight_matrix = torch.stack(all_weights, dim=0) * 100.0
    torch.save(weight_matrix.cpu(), out_path)
    print(f"Saved semantic features: {tuple(weight_matrix.shape)} -> {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
