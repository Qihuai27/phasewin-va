# -*- coding: utf-8 -*-
"""
Torchvision ImageNet classification adapters.

These adapters provide black-box scoring and simple input-gradient saliency for
standard torchvision classifiers such as ResNet-50 / ResNet-101.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from attribution_research.adapters.base import ForwardCounterMixin, SearchAdapter
from attribution_research.adapters.gradient import GradientAdapter


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_TORCHVISION_ARCHES = ("resnet50", "resnet101")


def resolve_torchvision_weights(arch: str, weights: str | None):
    import torchvision.models as tv_models

    arch_name = str(arch).strip().lower()
    if arch_name == "resnet50":
        enum_cls = tv_models.ResNet50_Weights
    elif arch_name == "resnet101":
        enum_cls = tv_models.ResNet101_Weights
    else:
        raise ValueError(
            f"Unsupported torchvision ImageNet arch: {arch!r}. "
            f"Expected one of: {', '.join(SUPPORTED_TORCHVISION_ARCHES)}"
        )

    if weights is None:
        return None

    text = str(weights).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    if text.upper() == "DEFAULT":
        return enum_cls.DEFAULT
    try:
        return getattr(enum_cls, text)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported weights value {weights!r} for {arch_name}. "
            f"Use DEFAULT, none, or one of the enum names from torchvision."
        ) from exc


def load_torchvision_imagenet_model(
    arch: str = "resnet101",
    *,
    weights: str | None = "DEFAULT",
    device: str = "cuda",
) -> Tuple[torch.nn.Module, object | None]:
    import torchvision.models as tv_models

    arch_name = str(arch).strip().lower()
    resolved_weights = resolve_torchvision_weights(arch_name, weights)
    if arch_name == "resnet50":
        model = tv_models.resnet50(weights=resolved_weights)
    elif arch_name == "resnet101":
        model = tv_models.resnet101(weights=resolved_weights)
    else:
        raise ValueError(
            f"Unsupported torchvision ImageNet arch: {arch!r}. "
            f"Expected one of: {', '.join(SUPPORTED_TORCHVISION_ARCHES)}"
        )
    model = model.to(device).eval()
    return model, resolved_weights


class TorchvisionImageNetSearchAdapter(SearchAdapter):
    """Black-box insertion / deletion scoring for torchvision ImageNet models."""

    model_name = "torchvision"
    task_type = "classification"

    def __init__(
        self,
        arch: str = "resnet101",
        *,
        weights: str | None = "DEFAULT",
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.arch = str(arch).strip().lower()
        self._device = device
        self._model, self._resolved_weights = load_torchvision_imagenet_model(
            self.arch,
            weights=weights,
            device=device,
        )
        weight_tag = "random" if self._resolved_weights is None else "imagenet1k"
        self.model_name = f"{self.arch}_{weight_tag}"
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.batch_size = int(batch_size)

        self._source_image: Optional[np.ndarray] = None
        self._target_label: Optional[int] = None
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @property
    def device(self) -> str:
        return self._device

    def setup(
        self,
        image: np.ndarray,
        target: int,
        **kwargs,
    ) -> None:
        self._source_image = image.astype(np.uint8)
        self._target_label = int(target)

    def _preprocess_batch(self, images_bgr: np.ndarray) -> torch.Tensor:
        rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self._device)
        return (tensor - self._mean) / self._std

    def _compose_images(self, alpha: np.ndarray) -> np.ndarray:
        if self._source_image is None:
            raise RuntimeError("setup(image, target) must be called before scoring")
        return (alpha * self._source_image[None].astype(np.float32)).astype(np.uint8)

    @torch.no_grad()
    def _class_scores(self, images_bgr: np.ndarray) -> torch.Tensor:
        if self._target_label is None:
            raise RuntimeError("target label is not configured")
        outputs = []
        for start in range(0, len(images_bgr), self.batch_size):
            batch = self._preprocess_batch(images_bgr[start : start + self.batch_size])
            logits = self._model(batch).float()
            self.record_model_forward(int(batch.shape[0]))
            probs = torch.softmax(logits, dim=-1)[:, self._target_label]
            outputs.append(probs)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def score_batch_detailed(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        alpha = np.clip(masks.astype(np.float32) + baseline.astype(np.float32)[None], 0, 1)
        img_ins = self._compose_images(alpha)
        img_del = self._compose_images(1.0 - alpha)
        ins = self._class_scores(img_ins)
        dele = self._class_scores(img_del)
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
    ) -> Dict[str, float]:
        details = self.score_batch_detailed(mask[None], baseline)
        return {
            "insertion_score": float(details["insertion_score"][0].item()),
            "deletion_score": float(details["deletion_score"][0].item()),
            "smdl_score": float(details["smdl_score"][0].item()),
        }


class TorchvisionImageNetGradientAdapter(GradientAdapter):
    """Input-gradient saliency for torchvision ImageNet models."""

    model_name = "torchvision"
    task_type = "classification"

    def __init__(
        self,
        arch: str = "resnet101",
        *,
        weights: str | None = "DEFAULT",
        device: str = "cuda",
        score_mode: str = "prob",
    ):
        if score_mode not in {"prob", "logit"}:
            raise ValueError(f"Unsupported score_mode: {score_mode!r}")
        self.arch = str(arch).strip().lower()
        self._device = device
        self._model, self._resolved_weights = load_torchvision_imagenet_model(
            self.arch,
            weights=weights,
            device=device,
        )
        weight_tag = "random" if self._resolved_weights is None else "imagenet1k"
        self.model_name = f"{self.arch}_{weight_tag}"
        self.score_mode = score_mode

        self._image: Optional[np.ndarray] = None
        self._target_label: Optional[int] = None
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @property
    def device(self) -> str:
        return self._device

    def setup(self, image: np.ndarray, target: int, **kwargs) -> None:
        self._image = image.astype(np.uint8)
        self._target_label = int(target)

    def _preprocess_single(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return (tensor - self._mean) / self._std

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        image_tensor = self._preprocess_single(self._image).detach().clone().requires_grad_(True)
        self._model.zero_grad(set_to_none=True)
        logits = self._model(image_tensor).float()
        if self.score_mode == "logit":
            score = logits[0, self._target_label]
        else:
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, self._target_label]
        score.backward()

        grad = image_tensor.grad[0]
        saliency = (grad * image_tensor[0]).abs().mean(dim=0)
        saliency = saliency.detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class TorchvisionImageNetIG2Adapter(TorchvisionImageNetGradientAdapter):
    """IG^2 saliency for torchvision ImageNet models using blur-based references."""

    method_name = "ig2"

    def __init__(
        self,
        arch: str = "resnet101",
        *,
        weights: str | None = "DEFAULT",
        device: str = "cuda",
        score_mode: str = "prob",
        steps: int = 32,
        step_size: float = 8.0,
        blur_sigmas: Optional[List[float]] = None,
    ):
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps!r}")
        if float(step_size) <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size!r}")

        super().__init__(
            arch=arch,
            weights=weights,
            device=device,
            score_mode=score_mode,
        )
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.blur_sigmas = [float(s) for s in (blur_sigmas or [3.0, 7.0, 15.0, 31.0])]
        self._rep_output: Optional[torch.Tensor] = None
        self._rep_handle = None
        avgpool = getattr(self._model, "avgpool", None)
        if avgpool is not None:
            self._rep_handle = avgpool.register_forward_hook(self._capture_representation)

    def teardown(self) -> None:
        self._image = None
        self._target_label = None

    def _capture_representation(self, module, inputs, output) -> None:
        self._rep_output = output

    def _raw_rgb_batch(self, images_bgr: np.ndarray) -> torch.Tensor:
        batch = np.asarray(images_bgr, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None]
        rgb = batch[..., ::-1].copy()
        return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self._device)

    def _forward_scores_and_rep(self, raw_rgb_255: torch.Tensor):
        self._rep_output = None
        normalized = (raw_rgb_255 / 255.0 - self._mean) / self._std
        logits = self._model(normalized).float()
        self.record_model_forward(int(raw_rgb_255.shape[0]))
        rep = self._rep_output
        if rep is None:
            rep = logits
        if self._target_label is None:
            raise RuntimeError("target label is not configured")
        if self.score_mode == "logit":
            scores = logits[:, self._target_label]
        else:
            scores = torch.softmax(logits, dim=-1)[:, self._target_label]
        return scores, rep

    def _normalize_batch_l2(self, grads: torch.Tensor) -> torch.Tensor:
        flat = grads.view(grads.shape[0], -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return grads / norm.view(-1, 1, 1, 1)

    def _build_blur_references(self) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("setup(image, target) must be called before building references")
        refs = []
        for sigma in self.blur_sigmas:
            kernel = max(3, int(round(sigma * 4)) | 1)
            refs.append(cv2.GaussianBlur(self._image, (kernel, kernel), sigmaX=sigma))
        return np.stack(refs, axis=0)

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        self.reset_forward_counter()
        input_raw = self._raw_rgb_batch(self._image)
        references = self._raw_rgb_batch(self._build_blur_references())

        with torch.no_grad():
            _, baseline_rep = self._forward_scores_and_rep(references)
            baseline_rep = baseline_rep.detach()

        current = input_raw.repeat(references.shape[0], 1, 1, 1).detach()
        path = [current.detach().clone()]
        for _ in range(self.steps):
            current = current.detach().clone().requires_grad_(True)
            self._model.zero_grad(set_to_none=True)
            _, rep = self._forward_scores_and_rep(current)
            loss = -torch.nn.functional.mse_loss(rep, baseline_rep, reduction="mean")
            loss.backward()
            grad = current.grad.detach()
            current = torch.clamp(current + self._normalize_batch_l2(grad) * self.step_size, 0.0, 255.0)
            path.append(current.detach().clone())

        attr = torch.zeros_like(current)
        previous = path[0]
        for step_tensor in path[1:]:
            prev = previous.detach().clone().requires_grad_(True)
            self._model.zero_grad(set_to_none=True)
            scores, _ = self._forward_scores_and_rep(prev)
            scores.sum().backward()
            grad = prev.grad.detach()
            attr += (prev.detach() - step_tensor.detach()) * grad
            previous = step_tensor

        saliency = attr.mean(dim=0).abs().mean(dim=0).detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class TorchvisionImageNetIGOSPPAdapter(TorchvisionImageNetGradientAdapter):
    """IGOS++-style optimized mask saliency for torchvision ImageNet models."""

    method_name = "igos_pp"

    def __init__(
        self,
        arch: str = "resnet101",
        *,
        weights: str | None = "DEFAULT",
        device: str = "cuda",
        score_mode: str = "prob",
        mask_size: int = 28,
        steps: int = 24,
        lr: float = 0.1,
        blur_sigma: float = 15.0,
        preserve_coeff: float = 2.0,
        delete_coeff: float = 1.0,
        area_coeff: float = 0.01,
        tv_coeff: float = 0.2,
        binary_coeff: float = 0.01,
    ):
        if int(mask_size) < 2:
            raise ValueError(f"mask_size must be >= 2, got {mask_size!r}")
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps!r}")
        if float(lr) <= 0:
            raise ValueError(f"lr must be > 0, got {lr!r}")
        if float(blur_sigma) <= 0:
            raise ValueError(f"blur_sigma must be > 0, got {blur_sigma!r}")

        super().__init__(
            arch=arch,
            weights=weights,
            device=device,
            score_mode=score_mode,
        )
        self.mask_size = int(mask_size)
        self.steps = int(steps)
        self.lr = float(lr)
        self.blur_sigma = float(blur_sigma)
        self.preserve_coeff = float(preserve_coeff)
        self.delete_coeff = float(delete_coeff)
        self.area_coeff = float(area_coeff)
        self.tv_coeff = float(tv_coeff)
        self.binary_coeff = float(binary_coeff)

    def _build_blur_reference(self, image: np.ndarray) -> np.ndarray:
        kernel = max(3, int(round(self.blur_sigma * 4)) | 1)
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=self.blur_sigma)

    def _target_score(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._target_label is None:
            raise RuntimeError("target label is not configured")
        logits = self._model(pixel_values).float()
        self.record_model_forward(int(pixel_values.shape[0]))
        if self.score_mode == "logit":
            return logits[:, self._target_label].mean()
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self._target_label].mean()

    @staticmethod
    def _tv_loss(mask: torch.Tensor) -> torch.Tensor:
        tv_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
        tv_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        self.reset_forward_counter()
        orig_pixels = self._preprocess_single(self._image).detach()
        blur_pixels = self._preprocess_single(self._build_blur_reference(self._image)).detach()

        with torch.no_grad():
            orig_score = self._target_score(orig_pixels).detach()

        mask_logits = torch.zeros(
            (1, 1, self.mask_size, self.mask_size),
            dtype=torch.float32,
            device=orig_pixels.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([mask_logits], lr=self.lr)
        eps = 1e-6

        for _ in range(self.steps):
            mask = torch.sigmoid(mask_logits)
            mask_up = F.interpolate(
                mask,
                size=orig_pixels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(orig_pixels.dtype)

            keep_pixels = mask_up * orig_pixels + (1.0 - mask_up) * blur_pixels
            drop_pixels = (1.0 - mask_up) * orig_pixels + mask_up * blur_pixels

            self._model.zero_grad(set_to_none=True)
            keep_score = self._target_score(keep_pixels)
            drop_score = self._target_score(drop_pixels)

            preserve_loss = torch.relu(orig_score - keep_score)
            area_loss = mask.mean()
            tv_loss = self._tv_loss(mask)
            entropy = -(mask * torch.log(mask + eps) + (1.0 - mask) * torch.log(1.0 - mask + eps)).mean()
            loss = (
                self.preserve_coeff * preserve_loss
                + self.delete_coeff * drop_score
                + self.area_coeff * area_loss
                + self.tv_coeff * tv_loss
                + self.binary_coeff * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_mask = torch.sigmoid(mask_logits.detach())
        final_mask = F.interpolate(
            final_mask,
            size=orig_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        saliency = final_mask.float().detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class TorchvisionImageNetXpliqueWrapper(ForwardCounterMixin, torch.nn.Module):
    """
    Classification wrapper compatible with `xplique.wrappers.TorchWrapper`.

    Inputs are expected to be RGB images in NHWC or NCHW layout. The wrapper
    applies standard ImageNet normalization and returns class probabilities.
    """

    def __init__(
        self,
        arch: str = "resnet101",
        *,
        weights: str | None = "DEFAULT",
        device: str = "cuda",
    ):
        super().__init__()
        self.arch = str(arch).strip().lower()
        self.device = device
        self.model, self._resolved_weights = load_torchvision_imagenet_model(
            self.arch,
            weights=weights,
            device=device,
        )
        weight_tag = "random" if self._resolved_weights is None else "imagenet1k"
        self.model_name = f"{self.arch}_{weight_tag}"
        self.num_classes = 1000
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _prepare_inputs(self, inputs) -> torch.Tensor:
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs).to(self.device)
        else:
            tensor = inputs.to(self.device)

        if tensor.ndim != 4:
            raise ValueError(f"Expected 4-D inputs, got {tuple(tensor.shape)}")
        if tensor.shape[1] == 3:
            nchw = tensor.float()
        else:
            nchw = tensor.permute(0, 3, 1, 2).float()
        if float(nchw.detach().amax().item()) > 1.0:
            nchw = nchw / 255.0
        return (nchw - self._mean) / self._std

    def forward(self, vision_inputs):
        batch = self._prepare_inputs(vision_inputs)
        logits = self.model(batch).float()
        self.record_model_forward(int(batch.shape[0]))
        return torch.softmax(logits, dim=-1).float()


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUPPORTED_TORCHVISION_ARCHES",
    "resolve_torchvision_weights",
    "load_torchvision_imagenet_model",
    "TorchvisionImageNetGradientAdapter",
    "TorchvisionImageNetIG2Adapter",
    "TorchvisionImageNetIGOSPPAdapter",
    "TorchvisionImageNetSearchAdapter",
    "TorchvisionImageNetXpliqueWrapper",
]
