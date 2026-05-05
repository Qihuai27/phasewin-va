# -*- coding: utf-8 -*-
"""
MLLM (Multimodal Large Language Model) adapter for greedy/PhaseWin explainers.

Supports caption attribution, VQA, and related multimodal token-attribution
tasks.

The adapter wraps any MLLM adaptor object that has a forward method:
    mllm_adaptor(image: Tensor_or_ndarray) -> (vocab_size,) logits

The score for a masked image is the mean log-probability of a set of target
token positions in the model's generated output.

Concrete MLLM adaptors (QwenVLAdaptor, InternVLAdaptor, LLaVAAdaptor) are
model-specific wrappers that handle tokenization, prompt construction, and
output decoding.  They are implemented per-task in the tasks/ directory.
To use a new model, implement the simple adaptor interface below.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from attribution_research.adapters.base import SearchAdapter


class MLLMAdapter(SearchAdapter):
    """
    Adapter for multimodal LLM token-attribution workflows.

    Parameters
    ----------
    mllm_adaptor : object
        A model-specific adaptor with:
            .forward(image) -> (vocab_size,) float tensor of softmax probs
            .device         -> torch.device or str
        Target token context (generated_ids, token positions) is configured
        externally on the adaptor before setup() is called.
    lambda1 : float
        Insertion weight.
    lambda2 : float
        Deletion weight.
    """

    model_name = "mllm"
    task_type  = "caption_vqa"

    def __init__(
        self,
        mllm_adaptor,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ):
        self._mllm = mllm_adaptor
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # set by setup()
        self._source_tensor: Optional[torch.Tensor] = None
        self._h: int = 0
        self._w: int = 0
        self._region_area: int = 1

    @property
    def device(self) -> str:
        d = self._mllm.device
        return str(d) if not isinstance(d, str) else d

    # ── state setup ───────────────────────────────────────────────────────────

    def setup(
        self,
        image: np.ndarray,
        target: Any = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        image  : (H, W, 3) uint8 BGR
        target : ignored; token context is configured on mllm_adaptor directly.
        """
        self._h, self._w, _ = image.shape
        self._region_area = self._h * self._w
        dev = self.device
        self._source_tensor = torch.from_numpy(image.astype(np.float32)).to(dev)

    # ── scoring ───────────────────────────────────────────────────────────────

    def _masked_images(
        self,
        masks: np.ndarray,      # (B, H, W, 1)
        baseline: np.ndarray,   # (H, W, 1)
    ):
        """Yield insertion and deletion images as (H,W,3) float tensors."""
        dev = self.device
        alpha = np.clip(masks.astype(np.float32) + baseline.astype(np.float32)[None], 0, 1)
        alpha_t     = torch.from_numpy(alpha).to(dev).expand(-1, -1, -1, 3)   # (B,H,W,3)
        src         = self._source_tensor.unsqueeze(0).expand_as(alpha_t)
        ins_images  = alpha_t * src                                             # (B,H,W,3)
        del_images  = (1.0 - alpha_t) * src
        return ins_images, del_images

    @torch.no_grad()
    def _batch_inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        images : (B, H, W, 3) float tensor
        Returns (B, n_target_tokens) softmax logits.
        """
        results = []
        for img in images:
            logits = self._mllm(img)   # (vocab_size,) or (n_tokens,)
            self.record_model_forward()
            results.append(logits)
        return torch.stack(results, dim=0)   # (B, ...)

    @torch.no_grad()
    def score_batch_detailed(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        ins_imgs, del_imgs = self._masked_images(masks, baseline)
        ins_scores = self._batch_inference(ins_imgs).float()   # (B, n_tok)
        del_scores = self._batch_inference(del_imgs).float()
        ins = ins_scores.mean(-1)
        dele = del_scores.mean(-1)
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
        ins_imgs, del_imgs = self._masked_images(mask[None], baseline)
        ins_scores = self._batch_inference(ins_imgs).float()[0]   # (n_tok,)
        del_scores = self._batch_inference(del_imgs).float()[0]

        ins  = float(ins_scores.mean().item())
        dele = float(del_scores.mean().item())
        smdl = self.lambda1 * ins + self.lambda2 * (1.0 - dele)

        return {
            "insertion_score":      ins,
            "deletion_score":       dele,
            "smdl_score":           smdl,
            "insertion_word_score": ins_scores.cpu().tolist(),
            "deletion_word_score":  del_scores.cpu().tolist(),
        }
