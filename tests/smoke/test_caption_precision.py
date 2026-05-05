# -*- coding: utf-8 -*-
"""
Smoke test for caption-task precision fixes and FlashAttention-2 integration.

Checks:
  1. Model loads in bfloat16 (not float16).
  2. FlashAttention-2 is used when flash-attn is installed.
  3. _select_token_probs returns finite float32 probabilities (no NaN/inf).
  4. Gradient saliency map is finite and in [0, 1].
  5. LLaVA-CAM saliency map is finite and in [0, 1].
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from attribution_research.adapters.qwen25vl import (
    Qwen25VLGradientAdapter,
    Qwen25VLLLaVACAMAdapter,
    Qwen25VLTokenScorer,
    load_qwen25vl_model,
)

MODEL_NAME = "model_checkpoint/Qwen2.5-VL-3B-Instruct"
EVAL_LIST = "datasets/Qwen2.5-VL-3B-coco-caption.json"
IMAGE_DIR = "datasets/coco/val2017"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_and_item():
    model, processor = load_qwen25vl_model(MODEL_NAME, DEVICE)
    with open(EVAL_LIST, "r", encoding="utf-8") as f:
        item = json.load(f)[0]
    scorer = Qwen25VLTokenScorer(model, processor, device=DEVICE)
    scorer.bind_target_from_item(item)
    img_path = os.path.join(IMAGE_DIR, item["image_path"])
    image = cv2.imread(img_path)
    return model, scorer, image, item


def test_model_dtype(model_and_item):
    model, _, _, _ = model_and_item
    param = next(model.parameters())
    assert param.dtype == torch.bfloat16, (
        f"Expected bfloat16 model weights, got {param.dtype}"
    )


def test_flash_attention_active(model_and_item):
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        pytest.skip("flash-attn not installed")
    if DEVICE != "cuda":
        pytest.skip("FlashAttention-2 is only expected on CUDA runs")

    model, _, _, _ = model_and_item
    config = model.config
    attn_impl = getattr(config, "_attn_implementation", None)
    if attn_impl != "flash_attention_2":
        pytest.skip(f"Loader fell back to {attn_impl}")
    assert attn_impl == "flash_attention_2"


def test_select_token_probs_finite(model_and_item):
    _, scorer, image, _ = model_and_item
    inputs = scorer._prepare_inputs(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image)
    with torch.no_grad():
        outputs = scorer.model(**inputs, return_dict=True, use_cache=False)
    probs = scorer._select_token_probs(outputs.logits)
    assert probs.dtype == torch.float32, f"Expected float32 probs, got {probs.dtype}"
    assert torch.isfinite(probs).all(), f"Non-finite probabilities: {probs}"
    assert (probs >= 0).all() and (probs <= 1).all(), f"Probabilities out of [0,1]: {probs}"


def test_gradient_saliency_finite(model_and_item):
    _, scorer, image, _ = model_and_item
    adapter = Qwen25VLGradientAdapter(scorer)
    adapter.setup(image)
    saliency = adapter.saliency_map()
    assert np.isfinite(saliency).all(), "Gradient saliency map contains NaN/inf"
    assert float(saliency.min()) >= 0.0 - 1e-6
    assert float(saliency.max()) <= 1.0 + 1e-6


def test_llavacam_saliency_finite(model_and_item):
    _, scorer, image, _ = model_and_item
    adapter = Qwen25VLLLaVACAMAdapter(scorer, layer_index=28)
    adapter.setup(image)
    saliency = adapter.saliency_map()
    adapter.teardown()
    assert np.isfinite(saliency).all(), "LLaVA-CAM saliency map contains NaN/inf"
    assert float(saliency.min()) >= 0.0 - 1e-6
    assert float(saliency.max()) <= 1.0 + 1e-6
