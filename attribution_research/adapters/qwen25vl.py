# -*- coding: utf-8 -*-
"""
Qwen2.5-VL adapters.

This file provides:
1. a task-specific token scorer used by the generic MLLM search adapter; and
2. task-specific map-based saliency adapters for the gradient family.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from attribution_research.adapters.gradient import GradientAdapter


def _to_long_tensor(values, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.to(device)


def _normalize_01_map(saliency: np.ndarray) -> np.ndarray:
    saliency = saliency.astype(np.float32)
    saliency -= float(saliency.min())
    vmax = float(saliency.max())
    if vmax > 0:
        saliency /= vmax
    return saliency


class Qwen25VLTokenScorer(torch.nn.Module):
    """
    Minimal Qwen2.5-VL token scorer used by search-based mask evaluation.

    The target is configured per sample via `bind_target(...)`.
    """

    def __init__(self, model, processor, device: str = "cuda", max_image_side: int | None = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self._device = device
        self.max_image_side = int(max_image_side) if max_image_side is not None else None
        self.generated_ids: Optional[torch.Tensor] = None
        self.target_token_position: List[int] = []
        self.selected_interpretation_token_word_id: List[int] = []

    @property
    def device(self) -> str:
        return self._device

    def bind_target(
        self,
        generated_ids,
        target_token_position: Sequence[int],
        selected_interpretation_token_word_id: Sequence[int],
    ) -> None:
        self.generated_ids = _to_long_tensor(generated_ids, self._device)
        self.selected_interpretation_token_word_id = [
            int(tok_id) for tok_id in selected_interpretation_token_word_id
        ]
        self.target_token_position = self._resolve_absolute_positions(
            self.generated_ids,
            target_token_position,
            self.selected_interpretation_token_word_id,
        )

    def bind_target_from_item(self, item: Dict[str, Any]) -> None:
        """Convenience wrapper for the dataset schema used by the task script."""
        self.bind_target(
            generated_ids=item["generated_ids"],
            target_token_position=item["selected_interpretation_token_id"],
            selected_interpretation_token_word_id=item["selected_interpretation_token_word_id"],
        )

    def _prepare_inputs(self, image_bgr: np.ndarray) -> Dict[str, torch.Tensor]:
        if self.generated_ids is None:
            raise RuntimeError("generated_ids are not configured; call bind_target(...) first")
        if not self.target_token_position:
            raise RuntimeError("target_token_position is empty after target binding")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.max_image_side is not None:
            h, w = image_rgb.shape[:2]
            longest = max(h, w)
            if longest > self.max_image_side:
                scale = float(self.max_image_side) / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = PILImage.fromarray(image_rgb)
        inputs = self.processor(images=pil_img, text="", return_tensors="pt")
        prefix_length = int(max(self.target_token_position))
        if prefix_length < 1:
            raise RuntimeError(f"Invalid Qwen target positions: {self.target_token_position}")
        input_ids = self.generated_ids[:prefix_length].unsqueeze(0)
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = torch.ones_like(input_ids)
        inputs.pop("mm_token_type_ids", None)
        return {key: value.to(self._device) for key, value in inputs.items()}

    def _select_token_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3:
            logits = logits[0]
        selected = []
        for pos, tok_id in zip(
            self.target_token_position,
            self.selected_interpretation_token_word_id,
        ):
            logit_pos = int(pos) - 1
            if 0 <= logit_pos < logits.shape[0]:
                row = logits[logit_pos].float()
                token_logit = row[tok_id]
                token_prob = torch.exp(token_logit - torch.logsumexp(row, dim=-1))
                selected.append(token_prob)
        if not selected:
            return torch.zeros(1, device=logits.device, dtype=torch.float32)
        return torch.stack(selected)

    def forward(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image_bgr = image.detach().cpu().numpy().astype(np.uint8)
        else:
            image_bgr = np.asarray(image, dtype=np.uint8)

        inputs = self._prepare_inputs(image_bgr)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True, use_cache=False)
        return self._select_token_probs(outputs.logits.float())

    @staticmethod
    def _resolve_absolute_positions(
        generated_ids: torch.Tensor,
        relative_positions: Sequence[int],
        selected_token_ids: Sequence[int],
    ) -> List[int]:
        """
        Map dataset-relative token positions onto absolute positions in the full
        generated token sequence saved by the caption eval-list preprocessing.
        """
        rel = [int(pos) for pos in relative_positions]
        toks = [int(tok) for tok in selected_token_ids]
        if len(rel) != len(toks):
            raise ValueError("target_token_position and selected token ids must have the same length")
        if not rel:
            return []

        seq = generated_ids.detach().cpu().tolist()
        max_rel = max(rel)
        candidates: List[int] = []
        for base in range(0, len(seq) - max_rel):
            if all(seq[base + rel_pos] == tok for rel_pos, tok in zip(rel, toks)):
                candidates.append(base)
        if not candidates:
            raise ValueError(
                "Could not align selected interpretation tokens within generated_ids. "
                "Check the eval-list token metadata."
            )
        base = candidates[-1]
        return [base + rel_pos for rel_pos in rel]


class Qwen25VLGradientAdapter(GradientAdapter):
    """
    Simple input-gradient saliency for Qwen2.5-VL token attribution.

    This is intentionally lightweight: a single backward pass on the selected
    token probabilities, projected back onto the processed image tensor.
    """

    model_name = "qwen25vl"
    task_type = "caption_vqa"
    method_name = "gradient"

    def __init__(self, scorer: Qwen25VLTokenScorer):
        self.scorer = scorer
        self._image: Optional[np.ndarray] = None

    @property
    def device(self) -> str:
        return self.scorer.device

    def setup(self, image: np.ndarray, target: Any = None, **kwargs) -> None:
        self._image = image.astype(np.uint8)
        if isinstance(target, dict):
            self.scorer.bind_target(
                generated_ids=target["generated_ids"],
                target_token_position=target["selected_interpretation_token_id"],
                selected_interpretation_token_word_id=target["selected_interpretation_token_word_id"],
            )

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        inputs = self.scorer._prepare_inputs(self._image)
        pixel_values = inputs["pixel_values"].detach().clone().requires_grad_(True)
        inputs["pixel_values"] = pixel_values

        self.scorer.model.zero_grad(set_to_none=True)
        outputs = self.scorer.model(**inputs, return_dict=True, use_cache=False)
        selected = self.scorer._select_token_probs(outputs.logits)
        score = selected.sum()
        score.backward()

        # Upcast to float32 after backward so the gradient × input product is
        # computed in full precision.  The model and pixel_values tensors stay in
        # bf16 during the forward/backward pass; we only promote the already-
        # detached outputs here, so memory overhead is limited to two small
        # (C, H, W) tensors rather than the full activation graph.
        grad = pixel_values.grad[0].float()
        saliency = (grad * pixel_values.detach()[0].float()).abs().mean(dim=0)
        saliency = saliency.detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return _normalize_01_map(saliency)


class Qwen25VLIGOSPPAdapter(GradientAdapter):
    """
    Repo-native IGOS++-style optimized mask baseline for Qwen2.5-VL.

    The adapter optimizes a low-resolution soft mask against a blurred image
    reference. The learned mask is then used as the saliency map that feeds the
    shared region replay evaluator.
    """

    model_name = "qwen25vl"
    task_type = "caption_vqa"
    method_name = "igos_pp"

    def __init__(
        self,
        scorer: Qwen25VLTokenScorer,
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

        self.scorer = scorer
        self.mask_size = int(mask_size)
        self.steps = int(steps)
        self.lr = float(lr)
        self.blur_sigma = float(blur_sigma)
        self.preserve_coeff = float(preserve_coeff)
        self.delete_coeff = float(delete_coeff)
        self.area_coeff = float(area_coeff)
        self.tv_coeff = float(tv_coeff)
        self.binary_coeff = float(binary_coeff)
        self._image: Optional[np.ndarray] = None

    @property
    def device(self) -> str:
        return self.scorer.device

    def teardown(self) -> None:
        self._image = None

    def setup(self, image: np.ndarray, target: Any = None, **kwargs) -> None:
        self._image = image.astype(np.uint8)
        if isinstance(target, dict):
            self.scorer.bind_target(
                generated_ids=target["generated_ids"],
                target_token_position=target["selected_interpretation_token_id"],
                selected_interpretation_token_word_id=target["selected_interpretation_token_word_id"],
            )

    def _build_blur_reference(self, image: np.ndarray) -> np.ndarray:
        kernel = max(3, int(round(self.blur_sigma * 4)) | 1)
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=self.blur_sigma)

    def _target_score(
        self,
        base_inputs: Dict[str, torch.Tensor],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.scorer.model(
            **base_inputs,
            pixel_values=pixel_values,
            return_dict=True,
            use_cache=False,
        )
        self.record_model_forward()
        selected = self.scorer._select_token_probs(outputs.logits)
        return selected.mean()

    @staticmethod
    def _tv_loss(mask: torch.Tensor) -> torch.Tensor:
        tv_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
        tv_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        self.reset_forward_counter()
        base_inputs = self.scorer._prepare_inputs(self._image)
        blurred_inputs = self.scorer._prepare_inputs(self._build_blur_reference(self._image))
        orig_pixels = base_inputs["pixel_values"].detach()
        blur_pixels = blurred_inputs["pixel_values"].detach()
        static_inputs = {key: value for key, value in base_inputs.items() if key != "pixel_values"}

        with torch.no_grad():
            orig_score = self._target_score(static_inputs, orig_pixels).detach()

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

            self.scorer.model.zero_grad(set_to_none=True)
            keep_score = self._target_score(static_inputs, keep_pixels)
            drop_score = self._target_score(static_inputs, drop_pixels)

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
        return _normalize_01_map(saliency)


class Qwen25VLLLaVACAMAdapter(GradientAdapter):
    """Hook-based LLaVA-CAM style saliency for Qwen2.5-VL."""

    model_name = "qwen25vl"
    task_type = "caption_vqa"
    method_name = "llavacam"

    def __init__(
        self,
        scorer: Qwen25VLTokenScorer,
        layer_index: int = 32,
    ):
        self.scorer = scorer
        self.layer_index = int(layer_index)
        self._image: Optional[np.ndarray] = None
        self._feature_maps: Optional[torch.Tensor] = None
        self._handles = []

    @property
    def device(self) -> str:
        return self.scorer.device

    def _resolve_target_layer(self):
        model = self.scorer.model
        # Try progressively deeper paths to find the decoder layer list.
        # Qwen2.5-VL (newer transformers): model.model.language_model.layers
        # Older layout: model.model.layers or model.layers
        if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
            layers = model.model.language_model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise RuntimeError("Could not resolve decoder layers for Qwen2.5-VL LLaVA-CAM")

        idx = self.layer_index
        if idx < 0:
            idx += len(layers)
        if not (0 <= idx < len(layers)):
            if self.layer_index == 32 and len(layers) > 0:
                idx = len(layers) - 1
            else:
                raise ValueError(f"Invalid Qwen layer index {self.layer_index!r}; found {len(layers)} layers")

        layer = layers[idx]
        if hasattr(layer, "post_attention_layernorm"):
            return layer.post_attention_layernorm
        raise RuntimeError("Selected Qwen decoder layer has no post_attention_layernorm submodule")

    def _register_hooks(self) -> None:
        target_layer = self._resolve_target_layer()

        def _save_feature_maps(module, inputs, output):
            # Save a reference to the intermediate activation so we can read
            # .grad after backward().  We do NOT cast or return a modified
            # tensor here — that would corrupt the fp16 computation graph and
            # cause dtype mismatches in downstream layers.
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                if tensor.requires_grad:
                    tensor.retain_grad()
                self._feature_maps = tensor

        self._handles.append(target_layer.register_forward_hook(_save_feature_maps))

    def teardown(self) -> None:
        self._image = None
        self._feature_maps = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def setup(self, image: np.ndarray, target: Any = None, **kwargs) -> None:
        self._image = image.astype(np.uint8)
        if not self._handles:
            self._register_hooks()
        if isinstance(target, dict):
            self.scorer.bind_target(
                generated_ids=target["generated_ids"],
                target_token_position=target["selected_interpretation_token_id"],
                selected_interpretation_token_word_id=target["selected_interpretation_token_word_id"],
            )

    @staticmethod
    def _normalize_01(saliency: np.ndarray) -> np.ndarray:
        return _normalize_01_map(saliency)

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        inputs = self.scorer._prepare_inputs(self._image)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.detach().clone().requires_grad_(True)
        image_grid = inputs.get("image_grid_thw")
        if image_grid is None:
            raise RuntimeError("Qwen2.5-VL processor output is missing image_grid_thw, required for LLaVA-CAM")

        spatial_merge_size = int(getattr(self.scorer.model.config.vision_config, "spatial_merge_size", 2))
        grid_h = int(image_grid[0, 1].item()) // spatial_merge_size
        grid_w = int(image_grid[0, 2].item()) // spatial_merge_size

        self._feature_maps = None
        self.scorer.model.zero_grad(set_to_none=True)
        outputs = self.scorer.model(**inputs, return_dict=True, use_cache=False)

        selected = []
        for pos, tok_id in zip(
            self.scorer.target_token_position,
            self.scorer.selected_interpretation_token_word_id,
        ):
            logit_pos = int(pos) - 1
            if 0 <= logit_pos < outputs.logits.shape[1]:
                selected.append(outputs.logits.float()[0, logit_pos, tok_id])
        if not selected:
            raise RuntimeError("No valid target token positions were found for Qwen LLaVA-CAM")

        target_score = torch.stack(selected).sum()
        target_score.backward(retain_graph=False)

        if self._feature_maps is None:
            raise RuntimeError("Qwen LLaVA-CAM hook did not capture feature maps")
        if self._feature_maps.grad is None:
            raise RuntimeError(
                "Qwen LLaVA-CAM: no gradient on feature maps after backward(). "
                "Ensure the model is not fully wrapped in torch.no_grad()."
            )

        image_token_id = int(getattr(self.scorer.model.config, "image_token_id", -1))
        if image_token_id < 0:
            raise RuntimeError("Qwen2.5-VL config is missing image_token_id")

        vision_mask = inputs["input_ids"] == image_token_id
        # Cast to float32 here (not in the hook) to avoid corrupting the fp16
        # computation graph with a dtype-mismatched intermediate tensor.
        feats = self._feature_maps[0][vision_mask[0]].float()
        grads = self._feature_maps.grad[0][vision_mask[0]].float()

        token_count = int(grid_h * grid_w)
        if feats.shape[0] < token_count or grads.shape[0] < token_count:
            side = int(np.sqrt(float(min(feats.shape[0], grads.shape[0]))))
            grid_h = side
            grid_w = side
            token_count = side * side
        feats = feats[:token_count].reshape(grid_h, grid_w, feats.shape[-1])
        grads = grads[:token_count].reshape(grid_h, grid_w, grads.shape[-1])

        pooled_gradients = torch.relu(grads).mean(dim=(0, 1))
        heatmap = (feats * pooled_gradients).mean(dim=-1)
        heatmap = heatmap.detach().cpu().numpy().astype(np.float32)
        heatmap = cv2.resize(
            heatmap,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return self._normalize_01(heatmap)


def load_qwen25vl_model(model_name: str, device: str):
    """Load Qwen2.5-VL model and processor.

    Loads in bfloat16 (the model's native training dtype) to avoid the fp16
    dynamic-range overflow that causes NaN in logit softmax computations.
    Attempts to enable FlashAttention-2 for faster, memory-efficient attention;
    falls back to eager attention if the flash-attn package is absent or the
    hardware does not support it.
    """
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("transformers with Qwen2-VL support is required") from exc

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModelClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenModelClass

    use_cuda = str(device).lower().startswith("cuda")
    attn_order = ("flash_attention_2", "eager") if use_cuda else ("eager",)

    # Try FlashAttention-2 first on CUDA; otherwise use eager attention.
    for attn_impl in attn_order:
        try:
            model = QwenModelClass.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            ).to(device).eval()
            print(
                f"[load_qwen25vl_model] Loaded with "
                f"{'FlashAttention-2' if attn_impl == 'flash_attention_2' else 'eager attention'}"
                f" (bfloat16)"
            )
            break
        except (ImportError, ValueError):
            if attn_impl == "eager":
                raise
            # flash-attn not installed or unsupported — retry with eager attention
            continue

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor
