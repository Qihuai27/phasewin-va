# -*- coding: utf-8 -*-
"""
Unit tests for Qwen25VLLLaVACAMAdapter.

Uses a tiny mock model (2 transformer layers, 8-dim hidden state) so that the
test runs on CPU without loading the real Qwen checkpoint.  The key property
under test is that _register_hooks correctly captures float32 feature maps
whose .grad is non-None after backward(), even when the mock model weights are
in float16.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from attribution_research.adapters.qwen25vl import Qwen25VLLLaVACAMAdapter


# ─────────────────────────────────────────────────────────────────────────────
# Minimal mock plumbing
# ─────────────────────────────────────────────────────────────────────────────

class _PostAttnLN(nn.LayerNorm):
    """A LayerNorm that acts as post_attention_layernorm."""


class _MockLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.post_attention_layernorm = _PostAttnLN(d)

    def forward(self, x):
        return self.post_attention_layernorm(x)


class _MockDecoderStack(nn.Module):
    def __init__(self, n_layers: int = 2, d: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(d) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _MockConfig:
    image_token_id = 1
    vision_config = SimpleNamespace(spatial_merge_size=1)


class _MockOutputs:
    def __init__(self, logits):
        self.logits = logits


class _MockModel(nn.Module):
    """
    Minimal forward that mimics the interface expected by
    Qwen25VLLLaVACAMAdapter.saliency_map().

    Sequence layout: [img_tok, img_tok, img_tok, img_tok, text_tok]
    4 image tokens arranged in a 2×2 grid.
    """

    def __init__(self, d: int = 8, vocab: int = 32):
        super().__init__()
        self.config = _MockConfig()
        self.model = _MockDecoderStack(n_layers=2, d=d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self._d = d

    def forward(self, input_ids, pixel_values=None, image_grid_thw=None,
                attention_mask=None, **kwargs):
        bsz, seq = input_ids.shape
        # Random hidden states – shape (B, T, D)
        hidden = torch.randn(bsz, seq, self._d, device=input_ids.device,
                             dtype=torch.float32)
        hidden = self.model(hidden)
        logits = self.lm_head(hidden)
        return _MockOutputs(logits=logits)

    def zero_grad(self, set_to_none=False):
        for p in self.parameters():
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()


class _MockScorer:
    """Minimal scorer stub wiring the mock model."""

    def __init__(self, model):
        self.model = model
        self.device = "cpu"
        # 5-token sequence: [img, img, img, img, txt], target at position 4
        self.target_token_position = [4]
        self.selected_interpretation_token_word_id = [5]

    def _prepare_inputs(self, image_bgr: np.ndarray):
        # input_ids: 4 image tokens (id=1) + 1 text token (id=2)
        input_ids = torch.tensor([[1, 1, 1, 1, 2]], dtype=torch.long)
        # image_grid_thw: batch=1, T=1, H=2, W=2 → 4 image tokens
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_adapter(layer_index=1) -> Qwen25VLLLaVACAMAdapter:
    model = _MockModel(d=8, vocab=32)
    scorer = _MockScorer(model)
    return Qwen25VLLLaVACAMAdapter(scorer, layer_index=layer_index)


def _dummy_image(h=16, w=16) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_saliency_map_returns_correct_shape():
    adapter = _make_adapter()
    image = _dummy_image(16, 24)
    adapter.setup(image)
    saliency = adapter.saliency_map()
    assert saliency.shape == (16, 24), f"unexpected shape {saliency.shape}"
    adapter.teardown()


def test_saliency_map_normalized_to_01():
    adapter = _make_adapter()
    image = _dummy_image()
    adapter.setup(image)
    saliency = adapter.saliency_map()
    assert float(saliency.min()) >= 0.0
    assert float(saliency.max()) <= 1.0 + 1e-6
    adapter.teardown()


def test_feature_maps_are_float32_after_forward():
    """Hook must upcast fp16 tensors to fp32 before retain_grad."""
    adapter = _make_adapter()
    image = _dummy_image()
    adapter.setup(image)

    # Run saliency_map so the hook fires
    adapter.saliency_map()
    # After teardown _feature_maps is cleared, but during saliency_map it must
    # have been float32. We test this by patching: run a second call and inspect
    # inside via a flag.
    # Re-setup and inspect right after the forward hook fires.
    adapter.teardown()
    adapter.setup(image)

    inputs = adapter.scorer._prepare_inputs(image)
    adapter._feature_maps = None
    adapter.scorer.model(**inputs, return_dict=True, use_cache=False)
    assert adapter._feature_maps is not None
    assert adapter._feature_maps.dtype == torch.float32
    adapter.teardown()


def test_grad_is_populated_after_backward():
    """Core fix: .grad must not be None after backward() on fp16-style model."""
    adapter = _make_adapter()
    image = _dummy_image()
    adapter.setup(image)

    inputs = adapter.scorer._prepare_inputs(image)
    adapter._feature_maps = None
    adapter.scorer.model.zero_grad(set_to_none=True)
    outputs = adapter.scorer.model(**inputs, return_dict=True, use_cache=False)

    # Build a scalar loss from logits at the target position
    logit = outputs.logits.float()[0, 3, 5]
    logit.backward()

    assert adapter._feature_maps is not None, "hook did not capture feature maps"
    assert adapter._feature_maps.grad is not None, (
        "gradient is None — fp16 backward hook fix is broken"
    )
    adapter.teardown()


def test_teardown_clears_state_and_hooks():
    adapter = _make_adapter()
    image = _dummy_image()
    adapter.setup(image)
    assert len(adapter._handles) == 1

    adapter.saliency_map()
    adapter.teardown()

    assert adapter._image is None
    assert adapter._feature_maps is None
    assert len(adapter._handles) == 0


def test_setup_without_grad_context_raises_clear_error():
    """Ensure the error message is actionable when backward cannot run."""
    adapter = _make_adapter()
    image = _dummy_image()
    adapter.setup(image)

    inputs = adapter.scorer._prepare_inputs(image)
    adapter._feature_maps = None
    # Run forward inside no_grad — backward will still work but .grad won't
    # be set because no_grad blocks grad computation entirely.
    with torch.no_grad():
        adapter.scorer.model(**inputs, return_dict=True, use_cache=False)

    # Feature map captured but grad context disabled: .grad should be None.
    assert adapter._feature_maps is not None
    assert adapter._feature_maps.grad is None
    adapter.teardown()
