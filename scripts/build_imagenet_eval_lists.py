#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build model-specific ImageNet correct / error eval lists from one or more
ground-truth-labeled source lists.

Outputs:
  <prefix>_true.txt
  <prefix>_false_gt.txt
  <prefix>_false_pred.txt
  <prefix>_meta.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attribution_research.adapters.torchvision_imagenet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_torchvision_imagenet_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-specific ImageNet eval-list splits")
    parser.add_argument(
        "--model",
        choices=["clip_vitl14", "clip_rn101", "resnet101", "resnet50"],
        required=True,
        help="Model used to score the labeled source pool",
    )
    parser.add_argument(
        "--datasets",
        default="datasets/imagenet/ILSVRC2012_img_val",
        help="ImageNet validation image root",
    )
    parser.add_argument(
        "--input-lists",
        nargs="+",
        default=None,
        help="One or more labeled source lists. Each line: <image_path> <gt_label>",
    )
    parser.add_argument(
        "--ground-truth-file",
        default=None,
        help=(
            "Official ImageNet val ground-truth file. Each line is an official class id "
            "in the 1-1000 range, aligned with ILSVRC2012_val_*.JPEG ordering."
        ),
    )
    parser.add_argument(
        "--official-id2class-file",
        default=None,
        help=(
            "Official ImageNet id->class mapping file. Each line: "
            "<wnid> <official_id> <class_name>."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/imagenet",
        help="Output directory for generated lists",
    )
    parser.add_argument(
        "--save-prefix",
        required=True,
        help="Output prefix, e.g. val_resnet101_round1",
    )
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--true-quota", type=int, default=None)
    parser.add_argument("--false-quota", type=int, default=None)
    parser.add_argument(
        "--clip-download-root",
        default=".checkpoints/CLIP",
        help="CLIP cache directory",
    )
    parser.add_argument(
        "--semantic-features",
        default=None,
        help="Required for CLIP models. Path to the CLIP semantic feature matrix.",
    )
    parser.add_argument(
        "--weights",
        default="DEFAULT",
        help="torchvision weights enum name for pure ResNet models, DEFAULT or none",
    )
    return parser.parse_args()


def load_source_pool(paths: list[str]) -> list[dict]:
    merged: "OrderedDict[str, int]" = OrderedDict()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, label = line.split()[:2]
                merged.setdefault(image_path, int(label))
    return [{"image_path": image_path, "gt_label": label} for image_path, label in merged.items()]


def normalize_class_name(text: str) -> str:
    text = str(text).strip().lower().replace("_", " ").replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_official_id2class(path: str | Path) -> dict[int, dict[str, str]]:
    entries: dict[int, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            wnid, official_id, class_name = line.split(maxsplit=2)
            entries[int(official_id)] = {
                "wnid": wnid,
                "class_name": class_name,
            }
    if len(entries) != 1000:
        raise ValueError(
            f"Expected 1000 id->class entries in {path}, found {len(entries)}"
        )
    return entries


def build_official_to_internal_mapping(id2class_path: str | Path) -> dict[int, dict[str, object]]:
    import torchvision.models as tv_models

    official_entries = parse_official_id2class(id2class_path)
    internal_categories = list(tv_models.ResNet50_Weights.DEFAULT.meta["categories"])
    internal_by_name = {
        normalize_class_name(name): idx for idx, name in enumerate(internal_categories)
    }

    mapping: dict[int, dict[str, object]] = {}
    missing = []
    for official_id, entry in sorted(official_entries.items()):
        key = normalize_class_name(entry["class_name"])
        internal_idx = internal_by_name.get(key)
        if internal_idx is None:
            missing.append((official_id, entry["class_name"]))
            continue
        mapping[official_id] = {
            "official_id": official_id,
            "wnid": entry["wnid"],
            "official_class_name": entry["class_name"],
            "internal_class_index": int(internal_idx),
            "internal_class_name": internal_categories[int(internal_idx)],
        }

    if missing:
        preview = ", ".join(f"{official_id}:{name}" for official_id, name in missing[:10])
        raise ValueError(
            "Could not map some official ImageNet ids to internal class indices: "
            f"{preview}"
        )

    return mapping


def build_pool_from_official_ground_truth(
    ground_truth_file: str | Path,
    id2class_file: str | Path,
) -> tuple[list[dict], dict[int, dict[str, object]]]:
    mapping = build_official_to_internal_mapping(id2class_file)
    pool: list[dict] = []
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            text = raw_line.strip()
            if not text:
                continue
            official_id = int(text)
            if official_id not in mapping:
                raise ValueError(f"Official class id {official_id} is missing from id2class mapping")
            mapped = mapping[official_id]
            pool.append(
                {
                    "image_path": f"ILSVRC2012_val_{idx:08d}.JPEG",
                    "gt_label": int(mapped["internal_class_index"]),
                    "official_label": official_id,
                    "official_wnid": str(mapped["wnid"]),
                    "official_class_name": str(mapped["official_class_name"]),
                    "internal_class_name": str(mapped["internal_class_name"]),
                }
            )
    return pool, mapping


class ClipPredictor:
    def __init__(
        self,
        clip_type: str,
        *,
        semantic_features: str,
        clip_download_root: str,
        batch_size: int,
        device: str,
    ):
        try:
            import clip as openai_clip
        except ImportError as exc:
            raise ImportError(
                "openai-clip is required. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        if semantic_features is None:
            raise ValueError("--semantic-features is required for CLIP predictors")

        self.device = device
        self.batch_size = int(batch_size)
        self.model, _ = openai_clip.load(
            clip_type,
            device=device,
            download_root=clip_download_root,
        )
        self.model = self.model.float().eval()
        self.semantic_features = torch.load(semantic_features, map_location=device).float().to(device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _preprocess(self, images_bgr: np.ndarray) -> torch.Tensor:
        rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device)
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def predict(self, images_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        labels = []
        scores = []
        for start in range(0, len(images_bgr), self.batch_size):
            batch = self._preprocess(images_bgr[start : start + self.batch_size])
            feats = self.model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats @ self.semantic_features.T
            probs = torch.softmax(logits, dim=-1)
            pred_score, pred_label = probs.max(dim=-1)
            labels.append(pred_label.cpu())
            scores.append(pred_score.cpu())
        return torch.cat(labels, dim=0), torch.cat(scores, dim=0)


class TorchvisionPredictor:
    def __init__(
        self,
        arch: str,
        *,
        weights: str,
        batch_size: int,
        device: str,
    ):
        self.device = device
        self.batch_size = int(batch_size)
        self.model, _ = load_torchvision_imagenet_model(
            arch,
            weights=weights,
            device=device,
        )
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _preprocess(self, images_bgr: np.ndarray) -> torch.Tensor:
        rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self.device)
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def predict(self, images_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        labels = []
        scores = []
        for start in range(0, len(images_bgr), self.batch_size):
            batch = self._preprocess(images_bgr[start : start + self.batch_size])
            logits = self.model(batch).float()
            probs = torch.softmax(logits, dim=-1)
            pred_score, pred_label = probs.max(dim=-1)
            labels.append(pred_label.cpu())
            scores.append(pred_score.cpu())
        return torch.cat(labels, dim=0), torch.cat(scores, dim=0)


def build_predictor(args: argparse.Namespace):
    if args.model == "clip_vitl14":
        return ClipPredictor(
            "ViT-L/14",
            semantic_features=args.semantic_features or "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt",
            clip_download_root=args.clip_download_root,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.model == "clip_rn101":
        return ClipPredictor(
            "RN101",
            semantic_features=args.semantic_features or "ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt",
            clip_download_root=args.clip_download_root,
            batch_size=args.batch_size,
            device=args.device,
        )
    return TorchvisionPredictor(
        args.model,
        weights=args.weights,
        batch_size=args.batch_size,
        device=args.device,
    )


def write_eval_list(path: Path, rows: list[tuple[str, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for image_path, label in rows:
            f.write(f"{image_path} {label}\n")


def main() -> None:
    args = parse_args()
    if args.input_lists:
        pool = load_source_pool(args.input_lists)
        official_mapping = None
    elif args.ground_truth_file and args.official_id2class_file:
        pool, official_mapping = build_pool_from_official_ground_truth(
            args.ground_truth_file,
            args.official_id2class_file,
        )
    else:
        raise ValueError(
            "Provide either --input-lists, or both --ground-truth-file and "
            "--official-id2class-file."
        )

    if args.shuffle_seed is not None:
        rng = np.random.default_rng(int(args.shuffle_seed))
        order = rng.permutation(len(pool))
        pool = [pool[int(i)] for i in order]

    pool = pool[args.begin : args.end]
    if not pool:
        raise ValueError("No samples selected after applying begin/end")

    predictor = build_predictor(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {args.model}")
    print(f"Samples    : {len(pool)}")
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Prefix     : {args.save_prefix}")
    if args.shuffle_seed is not None:
        print(f"Shuffle    : seed={args.shuffle_seed}")
    if args.true_quota is not None or args.false_quota is not None:
        print(
            "Quotas     : "
            f"true={args.true_quota if args.true_quota is not None else 'all'} / "
            f"false={args.false_quota if args.false_quota is not None else 'all'}"
        )

    true_rows: list[tuple[str, int]] = []
    false_gt_rows: list[tuple[str, int]] = []
    false_pred_rows: list[tuple[str, int]] = []
    meta_rows: list[dict] = []
    processed_count = 0

    def quotas_reached() -> bool:
        true_done = args.true_quota is None or len(true_rows) >= int(args.true_quota)
        false_done = args.false_quota is None or len(false_gt_rows) >= int(args.false_quota)
        return true_done and false_done

    def handle_predictions(batch_items: list[dict], preds: torch.Tensor, scores: torch.Tensor) -> None:
        nonlocal processed_count
        for sample, pred_label, pred_score in zip(batch_items, preds.tolist(), scores.tolist()):
            gt_label = int(sample["gt_label"])
            row = {
                "image_path": sample["image_path"],
                "gt_label": gt_label,
                "pred_label": int(pred_label),
                "pred_score": float(pred_score),
                "correct": bool(int(pred_label) == gt_label),
            }
            if "official_label" in sample:
                row["official_label"] = int(sample["official_label"])
                row["official_wnid"] = sample["official_wnid"]
                row["official_class_name"] = sample["official_class_name"]
                row["internal_class_name"] = sample["internal_class_name"]

            selected = False
            if row["correct"]:
                if args.true_quota is None or len(true_rows) < int(args.true_quota):
                    true_rows.append((sample["image_path"], gt_label))
                    selected = True
            else:
                if args.false_quota is None or len(false_gt_rows) < int(args.false_quota):
                    false_gt_rows.append((sample["image_path"], gt_label))
                    false_pred_rows.append((sample["image_path"], int(pred_label)))
                    selected = True
            row["selected"] = selected
            meta_rows.append(row)
            processed_count += 1

    batch_items = []
    batch_images = []
    stop_requested = False
    for item in pool:
        image_path = Path(args.datasets) / item["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[warn] cannot read {image_path}, skipping")
            continue
        batch_items.append(item)
        batch_images.append(cv2.resize(image, (224, 224)))

        if len(batch_images) < args.batch_size:
            continue

        preds, scores = predictor.predict(np.stack(batch_images, axis=0))
        handle_predictions(batch_items, preds, scores)
        batch_items = []
        batch_images = []
        if quotas_reached():
            stop_requested = True
            break

    if batch_images and not stop_requested:
        preds, scores = predictor.predict(np.stack(batch_images, axis=0))
        handle_predictions(batch_items, preds, scores)

    if args.true_quota is not None and len(true_rows) < int(args.true_quota):
        raise RuntimeError(
            f"Unable to collect {args.true_quota} correct samples for {args.model}. "
            f"Collected {len(true_rows)} after processing {processed_count} samples."
        )
    if args.false_quota is not None and len(false_gt_rows) < int(args.false_quota):
        raise RuntimeError(
            f"Unable to collect {args.false_quota} incorrect samples for {args.model}. "
            f"Collected {len(false_gt_rows)} after processing {processed_count} samples."
        )

    true_path = out_dir / f"{args.save_prefix}_true.txt"
    false_gt_path = out_dir / f"{args.save_prefix}_false_gt.txt"
    false_pred_path = out_dir / f"{args.save_prefix}_false_pred.txt"
    meta_path = out_dir / f"{args.save_prefix}_meta.json"

    write_eval_list(true_path, true_rows)
    write_eval_list(false_gt_path, false_gt_rows)
    write_eval_list(false_pred_path, false_pred_rows)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, indent=2)

    if official_mapping is not None:
        mapping_path = out_dir / f"{args.save_prefix}_official_to_internal_map.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(official_mapping, f, indent=2, sort_keys=True)
        print(f"mapping    : {len(official_mapping)} -> {mapping_path}")

    print(f"processed  : {processed_count}")
    print(f"true       : {len(true_rows)} -> {true_path}")
    print(f"false_gt   : {len(false_gt_rows)} -> {false_gt_path}")
    print(f"false_pred : {len(false_pred_rows)} -> {false_pred_path}")
    print(f"meta       : {len(meta_rows)} -> {meta_path}")


if __name__ == "__main__":
    main()
