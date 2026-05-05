# -*- coding: utf-8 -*-
"""
GroundingDINO adapters.

This file provides:
1. a raw-image detection wrapper compatible with the original search code;
2. a black-box search adapter for greedy / PhaseWin / D-RISE; and
3. a gradient-family saliency adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

from attribution_research.adapters.base import SearchAdapter
from attribution_research.adapters.gradient import GradientAdapter


class GroundingDINODetector(torch.nn.Module):
    """
    Thin wrapper around the raw GroundingDINO model.

    It matches the interface assumed by the original search-based detection
    code: raw masked BGR images in, absolute xyxy boxes plus token logits out.
    """

    def __init__(
        self,
        model,
        preprocess_fn,
        caption: str,
        device: str = "cuda",
    ):
        super().__init__()
        if not caption:
            raise ValueError("GroundingDINODetector requires a non-empty caption prompt")
        self.model = model.to(device).eval()
        self.preprocess_fn = preprocess_fn
        self.caption = caption
        self.device = device

    def _prepare_batch(self, images_bgr: np.ndarray) -> torch.Tensor:
        batch = np.asarray(images_bgr)
        if batch.ndim == 3:
            batch = batch[None]
        tensors = [self.preprocess_fn(image.astype(np.uint8)) for image in batch]
        return torch.stack(tensors, dim=0).to(self.device)

    @torch.no_grad()
    def predict_boxes_logits(
        self,
        images_bgr: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_tensor = self._prepare_batch(images_bgr)
        outputs = self.model(batch_tensor, captions=[self.caption] * batch_tensor.shape[0])
        prediction_logits = outputs["pred_logits"].sigmoid().detach().cpu()
        prediction_boxes = outputs["pred_boxes"].detach().cpu()
        scale = torch.tensor([w, h, w, h], dtype=prediction_boxes.dtype)
        boxes = prediction_boxes * scale
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return xyxy, prediction_logits


class GroundingDINOAdapter(SearchAdapter):
    """
    GroundingDINO black-box scoring for search-based explainers.

    The target label is the GroundingDINO token index (or list of token
    indices) corresponding to the class phrase in the caption prompt.
    """

    model_name = "groundingdino"
    task_type = "detection"

    def __init__(
        self,
        model,
        preprocess_fn,
        caption: str,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        batch_size: int = 4,
        mode: str = "object",
        device: str = "cuda",
    ):
        if mode not in ("object", "iou", "cls"):
            raise ValueError(f"mode must be one of ('object', 'iou', 'cls'), got {mode!r}")
        self._device = device
        self._detector = (
            model
            if hasattr(model, "predict_boxes_logits")
            else GroundingDINODetector(
                model=model,
                preprocess_fn=preprocess_fn,
                caption=caption,
                device=device,
            )
        )
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.batch_size = int(batch_size)
        self.mode = mode

        self._source_image: Optional[np.ndarray] = None
        self._target_label: Optional[torch.Tensor] = None
        self._target_box: Optional[Any] = None
        self._h: int = 0
        self._w: int = 0

    @property
    def device(self) -> str:
        return self._device

    def setup(
        self,
        image: np.ndarray,
        target: Any,
        image_proc: Optional[torch.Tensor] = None,
    ) -> None:
        self._source_image = image.astype(np.uint8)
        self._h, self._w = self._source_image.shape[:2]
        self._target_label = torch.as_tensor(target["label"], dtype=torch.long)
        self._target_box = target["box"]

    def _build_masked_images(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = np.clip(masks.astype(np.float32) + baseline.astype(np.float32)[None], 0, 1)
        source = self._source_image[None].astype(np.float32)
        insertion = (alpha * source).astype(np.uint8)
        deletion = ((1.0 - alpha) * source).astype(np.uint8)
        return insertion, deletion

    def _run_batched(self, images_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        all_boxes = []
        all_logits = []
        for start in range(0, len(images_bgr), self.batch_size):
            batch = images_bgr[start : start + self.batch_size]
            boxes, logits = self._detector.predict_boxes_logits(batch, self._h, self._w)
            self.record_model_forward(len(batch))
            all_boxes.append(boxes)
            all_logits.append(logits)
        return torch.cat(all_boxes, dim=0), torch.cat(all_logits, dim=0)

    @staticmethod
    def _iou(batched_boxes: torch.Tensor, target_box) -> torch.Tensor:
        target_box_t = torch.as_tensor(
            target_box,
            device=batched_boxes.device,
            dtype=batched_boxes.dtype,
        )
        x1, y1, x2, y2 = (
            batched_boxes[..., 0],
            batched_boxes[..., 1],
            batched_boxes[..., 2],
            batched_boxes[..., 3],
        )
        tx1, ty1, tx2, ty2 = target_box_t
        ix1 = torch.maximum(x1, tx1)
        iy1 = torch.maximum(y1, ty1)
        ix2 = torch.minimum(x2, tx2)
        iy2 = torch.minimum(y2, ty2)
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        area1 = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area2 = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)
        return inter / (area1 + area2 - inter + 1e-6)

    def _target_cls_scores(self, logits: torch.Tensor) -> torch.Tensor:
        target = self._target_label.to(logits.device).view(-1)
        selected = logits.index_select(dim=2, index=target)
        return selected.amax(dim=-1)

    def _proposal_matrix(
        self,
        boxes: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        iou = self._iou(boxes, self._target_box)
        cls = self._target_cls_scores(logits)
        if self.mode == "iou":
            proposal = iou
        elif self.mode == "cls":
            proposal = (iou > 0.5).float() * cls
        else:
            proposal = iou * cls
        return proposal, iou, cls

    @torch.no_grad()
    def score_batch_detailed(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        img_ins, img_del = self._build_masked_images(masks, baseline)
        boxes_ins, logits_ins = self._run_batched(img_ins)
        boxes_del, logits_del = self._run_batched(img_del)

        proposal_ins, _, _ = self._proposal_matrix(boxes_ins, logits_ins)
        proposal_del, _, _ = self._proposal_matrix(boxes_del, logits_del)

        ins = proposal_ins.max(dim=-1)[0]
        dele = proposal_del.max(dim=-1)[0]
        gain = self.lambda1 * ins + self.lambda2 * (1.0 - dele)
        return {
            "insertion_score": ins.detach().cpu(),
            "deletion_score": dele.detach().cpu(),
            "smdl_score": gain.detach().cpu(),
        }

    @torch.no_grad()
    def score_batch(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> torch.Tensor:
        return self.score_batch_detailed(masks, baseline)["smdl_score"]

    @torch.no_grad()
    def score_single_detailed(
        self,
        mask: np.ndarray,
        baseline: np.ndarray,
    ) -> Dict[str, Any]:
        img_ins, img_del = self._build_masked_images(mask[None], baseline)
        boxes_ins, logits_ins = self._run_batched(img_ins)
        boxes_del, logits_del = self._run_batched(img_del)

        proposal_ins, iou_ins, cls_ins = self._proposal_matrix(boxes_ins, logits_ins)
        proposal_del, iou_del, cls_del = self._proposal_matrix(boxes_del, logits_del)

        ins_score = float(proposal_ins[0].max().item())
        del_score = float(proposal_del[0].max().item())
        smdl = self.lambda1 * ins_score + self.lambda2 * (1.0 - del_score)

        ins_idx = int(proposal_ins[0].argmax().item())
        del_idx = int(proposal_del[0].argmax().item())

        return {
            "insertion_score": ins_score,
            "deletion_score": del_score,
            "smdl_score": smdl,
            "insertion_iou": float(iou_ins[0, ins_idx].item()),
            "deletion_iou": float(iou_del[0, del_idx].item()),
            "insertion_cls": float(cls_ins[0, ins_idx].item()),
            "deletion_cls": float(cls_del[0, del_idx].item()),
            "insertion_box": boxes_ins[0, ins_idx].cpu().int().tolist(),
            "deletion_box": boxes_del[0, del_idx].cpu().int().tolist(),
        }


class GroundingDINOGradientAdapter(GradientAdapter):
    """Single-backward input-gradient saliency for GroundingDINO."""

    model_name = "groundingdino"
    task_type = "detection"

    def __init__(
        self,
        model,
        preprocess_fn,
        caption: str,
        mode: str = "object",
        device: str = "cuda",
    ):
        if mode not in ("object", "iou", "cls"):
            raise ValueError(f"mode must be one of ('object', 'iou', 'cls'), got {mode!r}")

        if hasattr(model, "model") and hasattr(model, "caption") and hasattr(model, "preprocess_fn"):
            self._model = model.model.to(device).eval()
            self._preprocess = model.preprocess_fn
            self.caption = model.caption
        else:
            self._model = model.to(device).eval()
            self._preprocess = preprocess_fn
            self.caption = caption

        if not self.caption:
            raise ValueError("GroundingDINOGradientAdapter requires a non-empty caption prompt")

        self._device = device
        self.mode = mode
        self._image: Optional[np.ndarray] = None
        self._target_label: Optional[torch.Tensor] = None
        self._target_box: Optional[Any] = None

    @property
    def device(self) -> str:
        return self._device

    def setup(self, image: np.ndarray, target: Any, **kwargs) -> None:
        self._image = image.astype(np.uint8)
        self._target_label = torch.as_tensor(target["label"], dtype=torch.long)
        self._target_box = target["box"]

    @staticmethod
    def _iou(boxes: torch.Tensor, target_box) -> torch.Tensor:
        target_box_t = torch.as_tensor(target_box, device=boxes.device, dtype=boxes.dtype)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        tx1, ty1, tx2, ty2 = target_box_t
        ix1 = torch.maximum(x1, tx1)
        iy1 = torch.maximum(y1, ty1)
        ix2 = torch.minimum(x2, tx2)
        iy2 = torch.minimum(y2, ty2)
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        area1 = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area2 = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)
        return inter / (area1 + area2 - inter + 1e-6)

    def _target_cls_scores(self, logits: torch.Tensor) -> torch.Tensor:
        target = self._target_label.to(logits.device).view(-1)
        selected = logits.index_select(dim=1, index=target)
        return selected.amax(dim=-1)

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        source_h, source_w = self._image.shape[:2]
        image_tensor = self._preprocess(self._image).unsqueeze(0).to(self._device)
        image_tensor = image_tensor.detach().clone().requires_grad_(True)

        self._model.zero_grad(set_to_none=True)
        outputs = self._model(image_tensor, captions=[self.caption])
        logits = outputs["pred_logits"][0].sigmoid()
        boxes = outputs["pred_boxes"][0]
        scale = torch.tensor([source_w, source_h, source_w, source_h], device=boxes.device, dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")

        iou = self._iou(boxes_xyxy, self._target_box)
        cls = self._target_cls_scores(logits)
        if self.mode == "iou":
            proposal = iou
        elif self.mode == "cls":
            proposal = (iou > 0.5).float() * cls
        else:
            proposal = iou * cls
        score = proposal.max()
        score.backward()

        grad = image_tensor.grad[0]
        saliency = (grad * image_tensor[0]).abs().mean(dim=0)
        saliency = saliency.detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (source_w, source_h),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class GroundingDINOCAMAdapter(GroundingDINOGradientAdapter):
    """
    Feature-map CAM baselines for GroundingDINO.

    Supported variants:
    - ``gradcam``
    - ``odam``
    - ``ssgrad_cam_pp``
    """

    def __init__(
        self,
        model,
        preprocess_fn,
        caption: str,
        mode: str = "object",
        device: str = "cuda",
        variant: str = "gradcam",
        feature_level: int = 2,
    ):
        super().__init__(
            model=model,
            preprocess_fn=preprocess_fn,
            caption=caption,
            mode=mode,
            device=device,
        )
        if variant not in {"gradcam", "odam", "ssgrad_cam_pp"}:
            raise ValueError(
                f"variant must be one of ('gradcam', 'odam', 'ssgrad_cam_pp'), got {variant!r}"
            )
        self.variant = variant
        self.feature_level = int(feature_level)

    @staticmethod
    def _normalize_01(saliency: torch.Tensor) -> np.ndarray:
        saliency = saliency.detach().cpu().numpy().astype(np.float32)
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency

    def _feature_tensors(self, outputs) -> Dict[int, torch.Tensor]:
        if not hasattr(self._model, "features"):
            raise RuntimeError(
                "GroundingDINO model does not expose cached backbone features. "
                "The CAM-style baselines require a model build that supports "
                "unset_image_tensor=False and model.features[*].tensors."
            )
        features: Dict[int, torch.Tensor] = {}
        for level, nested in enumerate(self._model.features):
            if not hasattr(nested, "tensors"):
                raise RuntimeError(f"GroundingDINO feature level {level} has no .tensors field")
            tensor = nested.tensors
            tensor.retain_grad()
            features[level] = tensor
        return features

    def _target_logit(self, logits: torch.Tensor, proposal_index: int) -> torch.Tensor:
        return self._target_cls_scores(logits[proposal_index : proposal_index + 1])[0]

    def _project_box_mask(self, target_box_xyxy, feature: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = [int(v) for v in torch.as_tensor(target_box_xyxy).detach().cpu().tolist()]
        source_h, source_w = self._image.shape[:2]
        mask = np.zeros((source_h, source_w), dtype=np.uint8)
        mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 1
        mask = cv2.resize(
            mask,
            (feature.shape[-1], feature.shape[-2]),
            interpolation=cv2.INTER_NEAREST,
        )
        return torch.from_numpy(mask[None, None]).to(feature.device, dtype=feature.dtype)

    def _gradcam_map(self, feature: torch.Tensor) -> torch.Tensor:
        grad = feature.grad
        if grad is None:
            raise RuntimeError("Missing feature gradients for GroundingDINO GradCAM")
        weights = grad.mean(dim=(-1, -2), keepdim=True)
        return torch.relu((weights * feature).sum(dim=1))[0]

    def _odam_map(self, feature: torch.Tensor) -> torch.Tensor:
        grad = feature.grad
        if grad is None:
            raise RuntimeError("Missing feature gradients for GroundingDINO ODAM")
        return torch.relu((grad * feature).sum(dim=1))[0]

    def _ssgrad_cam_pp_map(self, feature: torch.Tensor, target_box_xyxy) -> torch.Tensor:
        grad = feature.grad
        if grad is None:
            raise RuntimeError("Missing feature gradients for GroundingDINO SSGrad-CAM++")
        eps = torch.finfo(feature.dtype).eps
        mask = self._project_box_mask(target_box_xyxy, feature)
        grad_abs = grad.abs()
        grad_abs = grad_abs / grad_abs.max().clamp(min=eps)
        grad2 = grad.pow(2)
        grad3 = grad2 * grad
        denom = 2 * grad2 + feature.sum(dim=(-1, -2), keepdim=True) * mask * grad3
        alpha = grad2 / denom.clamp(min=eps)
        alpha = alpha * (alpha != 0).to(alpha.dtype)
        weighted = torch.relu(grad * grad_abs) * alpha
        weights = weighted.sum(dim=(-1, -2), keepdim=True)
        return torch.relu((weights * feature.detach()).sum(dim=1))[0]

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        source_h, source_w = self._image.shape[:2]
        image_tensor = self._preprocess(self._image).unsqueeze(0).to(self._device)

        self._model.zero_grad(set_to_none=True)
        outputs = self._model(image_tensor, captions=[self.caption], unset_image_tensor=False)
        features = self._feature_tensors(outputs)

        logits = outputs["pred_logits"][0].sigmoid()
        boxes = outputs["pred_boxes"][0]
        scale = torch.tensor([source_w, source_h, source_w, source_h], device=boxes.device, dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
        iou = self._iou(boxes_xyxy, self._target_box)
        proposal_index = int(iou.argmax().item())

        score_terms = [self._target_logit(logits, proposal_index)]
        if self.variant in {"gradcam", "odam"}:
            score_terms.extend(outputs["pred_boxes"][0, proposal_index, i].float() for i in range(4))

        maps = []
        selected_levels = list(features.keys()) if self.variant == "odam" else [self.feature_level]
        missing_levels = [level for level in selected_levels if level not in features]
        if missing_levels:
            raise ValueError(
                f"Requested GroundingDINO feature levels {missing_levels!r} are not available. "
                f"Found levels: {sorted(features.keys())}"
            )

        for score in score_terms:
            self._model.zero_grad(set_to_none=True)
            for feature in features.values():
                if feature.grad is not None:
                    feature.grad.zero_()
            score.backward(retain_graph=True)

            if self.variant == "odam":
                for level in selected_levels:
                    maps.append(self._odam_map(features[level]))
            elif self.variant == "gradcam":
                maps.append(self._gradcam_map(features[self.feature_level]))
            else:
                maps.append(self._ssgrad_cam_pp_map(features[self.feature_level], boxes_xyxy[proposal_index]))

        if hasattr(self._model, "unset_image_tensor"):
            self._model.unset_image_tensor()

        if not maps:
            raise RuntimeError("GroundingDINO CAM baseline produced no saliency maps")

        stacked = torch.stack(
            [
                torch.from_numpy(
                    cv2.resize(
                        self._normalize_01(cam),
                        (source_w, source_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
                for cam in maps
            ],
            dim=0,
        )
        return stacked.max(dim=0)[0].numpy().astype(np.float32)
