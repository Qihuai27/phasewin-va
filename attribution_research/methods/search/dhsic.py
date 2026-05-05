# -*- coding: utf-8 -*-
"""
Optional D-HSIC baseline wrapper.

The original classification baseline uses `xplique.HsicAttributionMethod`
instead of a handwritten implementation.  This wrapper keeps that behavior
available without forcing `xplique` to be a hard dependency of the whole
repository.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from attribution_research.adapters.base import ModelAdapter
from attribution_research.methods.gradient.map_based import SaliencyMapExplainer
from attribution_research.segmentation.base import RegionSet


class DHSICExplainer:
    """
    Run `xplique.HsicAttributionMethod` on a model that returns class scores.

    Parameters
    ----------
    model : torch.nn.Module
        Classification model compatible with `xplique.wrappers.TorchWrapper`.
    num_classes : int
        Number of output classes.
    evaluator : ModelAdapter | None
        Optional black-box evaluator used to replay the resulting ranking.
    batch_size : int
        Forward batch size passed to `HsicAttributionMethod`.
    """

    def __init__(
        self,
        model,
        num_classes: int,
        evaluator: Optional[ModelAdapter] = None,
        batch_size: int = 32,
        tf_device: str = "cpu",
        reduction: str = "mean",
        descending: bool = True,
    ):
        self.model = model
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.tf_device = str(tf_device).strip().lower()
        self._map_explainer = (
            SaliencyMapExplainer(
                evaluator=evaluator,
                reduction=reduction,
                descending=descending,
            )
            if evaluator is not None
            else None
        )
        self.reduction = reduction
        self.descending = descending

    def _configure_tensorflow_runtime(self) -> None:
        """
        Configure TensorFlow device visibility before importing xplique.

        `xplique` depends on TensorFlow even when the wrapped model is PyTorch.
        On mixed TF+PyTorch CUDA setups, letting TensorFlow see the GPU can cause
        unnecessary memory reservation or PTX JIT overhead. Defaulting D-HSIC to
        TensorFlow-on-CPU avoids that contention while still allowing the wrapped
        PyTorch model to run on CUDA.
        """
        try:
            import tensorflow as tf
        except ImportError:
            return

        if self.tf_device == "cpu":
            try:
                tf.config.set_visible_devices([], "GPU")
            except RuntimeError:
                # Visible devices cannot be changed after runtime init.
                pass
            return

        if self.tf_device in {"auto", "gpu"}:
            try:
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
            return

        raise ValueError(
            f"Unsupported TensorFlow device policy for D-HSIC: {self.tf_device!r}. "
            "Expected one of: cpu, auto, gpu"
        )

    def saliency_map(
        self,
        image: np.ndarray,
        target: int,
    ) -> np.ndarray:
        """Generate one D-HSIC map."""
        self._configure_tensorflow_runtime()
        try:
            from xplique.attributions import HsicAttributionMethod
            from xplique.wrappers import TorchWrapper
        except ImportError as exc:
            raise ImportError(
                "D-HSIC requires the optional `xplique` dependency. "
                "Install it before using algorithm='dhsic'."
            ) from exc

        device = getattr(self.model, "device", "cpu")
        wrapped = TorchWrapper(self.model.eval(), device)
        explainer = HsicAttributionMethod(wrapped, batch_size=self.batch_size)

        image_batch = np.asarray(image[None], dtype=np.float32)
        labels = np.eye(self.num_classes, dtype=np.float32)[[int(target)]]
        explanation = explainer(image_batch, labels)
        if not isinstance(explanation, np.ndarray):
            explanation = explanation.numpy()

        saliency = np.asarray(explanation[0], dtype=np.float32)
        if saliency.ndim == 3 and saliency.shape[-1] == 1:
            saliency = saliency[:, :, 0]
        return saliency

    def __call__(
        self,
        image: np.ndarray,
        regions: Optional[RegionSet],
        target: int,
    ):
        saliency = self.saliency_map(image=image, target=target)
        algorithm_forward_calls = int(getattr(self.model, "model_forward_calls", 0))
        if regions is None:
            return saliency

        if self._map_explainer is not None:
            ordered_masks, json_dict = self._map_explainer.explain_from_map(
                image=image,
                regions=regions,
                saliency_map=saliency,
                target=target,
            )
        else:
            region_scores = regions.region_scores(saliency, reduction=self.reduction)
            ordered_masks = regions.ordered_masks_from_scores(
                region_scores,
                descending=self.descending,
            )
            json_dict = {
                "region_saliency_score": region_scores,
                "saliency_reduction": self.reduction,
                "evaluation_skipped": True,
            }

        eval_forward_calls = int(getattr(self._map_explainer.evaluator, "model_forward_calls", 0)) if self._map_explainer is not None else 0
        json_dict["method"] = "dhsic"
        json_dict["xplique_batch_size"] = self.batch_size
        json_dict["model_forward_calls"] = algorithm_forward_calls
        json_dict["saliency_model_forward_calls"] = algorithm_forward_calls
        json_dict["eval_model_forward_calls"] = eval_forward_calls
        json_dict["total_model_forward_calls"] = algorithm_forward_calls + eval_forward_calls
        json_dict["model_forward_count_mode"] = "equivalent_single_image_forwards"
        json_dict["model_forward_count_scope"] = "algorithm_only"
        return ordered_masks, json_dict, saliency
