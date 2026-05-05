from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from attribution_research.adapters.qwen25vl import Qwen25VLIGOSPPAdapter


class _MockOutputs:
    def __init__(self, logits):
        self.logits = logits


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 16):
        super().__init__()
        self.config = SimpleNamespace()
        self._vocab = vocab

    def forward(self, input_ids, pixel_values=None, attention_mask=None, **kwargs):
        batch, seq_len = input_ids.shape
        signal = pixel_values.float().mean(dim=(1, 2, 3), keepdim=False)
        logits = torch.zeros((batch, seq_len, self._vocab), dtype=torch.float32, device=input_ids.device)
        logits[..., 3] = signal.unsqueeze(1)
        logits[..., 4] = (-signal).unsqueeze(1)
        return _MockOutputs(logits=logits)


class _MockScorer:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"
        self.target_token_position = [1]
        self.selected_interpretation_token_word_id = [3]

    def _prepare_inputs(self, image_bgr: np.ndarray):
        pixel_values = torch.from_numpy(image_bgr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
            "pixel_values": pixel_values,
        }

    def _select_token_probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float()[0], dim=-1)
        return torch.stack(
            [
                probs[int(pos) - 1, tok_id]
                for pos, tok_id in zip(
                    self.target_token_position,
                    self.selected_interpretation_token_word_id,
                )
            ]
        )


def _dummy_image(h=18, w=26):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_igospp_saliency_map_returns_image_shape():
    scorer = _MockScorer(_MockModel())
    adapter = Qwen25VLIGOSPPAdapter(
        scorer,
        mask_size=8,
        steps=3,
        lr=0.2,
        blur_sigma=5.0,
    )
    image = _dummy_image()
    adapter.setup(image)
    saliency = adapter.saliency_map()
    adapter.teardown()

    assert saliency.shape == image.shape[:2]
    assert np.isfinite(saliency).all()
    assert float(saliency.min()) >= 0.0
    assert float(saliency.max()) <= 1.0 + 1e-6
    assert adapter.model_forward_calls == 1 + 2 * 3


def test_igospp_invalid_configuration_raises():
    scorer = _MockScorer(_MockModel())
    try:
        Qwen25VLIGOSPPAdapter(scorer, mask_size=1)
    except ValueError as exc:
        assert "mask_size" in str(exc)
    else:
        raise AssertionError("Expected invalid mask_size to raise ValueError")
