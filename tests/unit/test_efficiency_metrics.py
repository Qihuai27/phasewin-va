import numpy as np
import torch

from attribution_research.adapters.base import ForwardCounterMixin, SearchAdapter
from attribution_research.adapters.gradient import GradientAdapter
from attribution_research.methods.search.drise import DRISEExplainer
from attribution_research.methods.search.dhsic import DHSICExplainer
from attribution_research.methods.search.greedy import GreedyExplainer
from attribution_research.methods.search.phasewin import PhaseWinExplainer
from attribution_research.runtime import AttributionContext, execute_attribution
from attribution_research.segmentation.base import RegionSet


class DummySearchAdapter(SearchAdapter):
    model_name = "dummy_search"
    task_type = "classification"

    @property
    def device(self) -> str:
        return "cpu"

    def setup(self, image, target=None, **kwargs) -> None:
        self._image = image

    def score_batch(self, masks, baseline) -> torch.Tensor:
        self.record_model_forward(2 * len(masks))
        scores = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float32)
        return torch.from_numpy(scores)

    def score_batch_detailed(self, masks, baseline):
        self.record_model_forward(2 * len(masks))
        scores = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float32)
        ins = torch.from_numpy(scores)
        dele = torch.zeros_like(ins)
        return {
            "insertion_score": ins,
            "deletion_score": dele,
            "smdl_score": ins,
        }

    def score_single_detailed(self, mask, baseline):
        self.record_model_forward(2)
        score = float(np.asarray(mask).sum())
        return {
            "insertion_score": score,
            "deletion_score": 0.0,
            "smdl_score": score,
        }


class DummyForwardModel(ForwardCounterMixin):
    num_classes = 2


class DummyIGOSAdapter(GradientAdapter):
    model_name = "dummy_igos"
    task_type = "classification"
    method_name = "igos_pp"
    mask_size = 4
    steps = 2
    lr = 0.1
    blur_sigma = 1.0

    @property
    def device(self) -> str:
        return "cpu"

    def setup(self, image, target=None, **kwargs) -> None:
        self._image = image

    def saliency_map(self, **kwargs) -> np.ndarray:
        self.record_model_forward(5)
        return np.ones(self._image.shape[:2], dtype=np.float32)


def test_greedy_json_records_model_forward_calls():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    masks = [
        np.array([[[1], [0]], [[0], [0]]], dtype=np.uint8),
        np.array([[[0], [1]], [[1], [0]]], dtype=np.uint8),
    ]

    explainer = GreedyExplainer(DummySearchAdapter())
    _, json_dict = explainer(image=image, masks=masks, target=0, show_progress=False)

    assert json_dict["candidate_evaluations"] == 3
    assert json_dict["model_forward_calls"] == 6
    assert json_dict["selection_model_forward_calls"] == 6
    assert json_dict["eval_model_forward_calls"] == 4
    assert json_dict["total_model_forward_calls"] == 10
    assert json_dict["model_forward_count_mode"] == "equivalent_single_image_forwards"
    assert json_dict["model_forward_count_scope"] == "algorithm_only"


def test_igos_runtime_resets_reused_evaluator_forward_counter():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    regions = RegionSet(
        [
            np.array([[[1], [0]], [[1], [0]]], dtype=np.uint8),
            np.array([[[0], [1]], [[0], [1]]], dtype=np.uint8),
        ]
    )
    args = type("Args", (), {"algorithm": "igos_pp"})()
    evaluator = DummySearchAdapter()

    def _run_once():
        _, json_dict = execute_attribution(
            AttributionContext(
                args=args,
                image=image,
                regions=regions,
                target=0,
                build_search_adapter=lambda: evaluator,
                build_gradient_adapter=lambda: DummyIGOSAdapter(),
            )
        )
        return json_dict

    first = _run_once()
    second = _run_once()

    assert first["model_forward_calls"] == 5
    assert first["saliency_model_forward_calls"] == 5
    assert first["eval_model_forward_calls"] == 4
    assert first["total_model_forward_calls"] == 9
    assert second["model_forward_calls"] == 5
    assert second["eval_model_forward_calls"] == 4
    assert second["total_model_forward_calls"] == 9
    assert second["model_forward_count_scope"] == "algorithm_only"


def test_drise_json_records_model_forward_calls():
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    regions = RegionSet(
        [
            np.array([[[1], [0]], [[1], [0]]], dtype=np.uint8),
            np.array([[[0], [1]], [[0], [1]]], dtype=np.uint8),
        ]
    )

    explainer = DRISEExplainer(
        adapter=DummySearchAdapter(),
        n_masks=4,
        grid_size=(2, 2),
        prob_thresh=0.5,
        batch_size=2,
        rng_seed=0,
    )
    _, json_dict, _ = explainer(image=image, regions=regions, target=0, show_progress=False)

    assert json_dict["model_forward_calls"] == 8
    assert json_dict["saliency_model_forward_calls"] == 8
    assert json_dict["eval_model_forward_calls"] == 4
    assert json_dict["total_model_forward_calls"] == 12
    assert json_dict["model_forward_count_mode"] == "equivalent_single_image_forwards"
    assert json_dict["model_forward_count_scope"] == "algorithm_only"


def test_dhsic_json_records_algorithm_and_eval_forward_calls(monkeypatch):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    regions = RegionSet(
        [
            np.array([[[1], [0]], [[1], [0]]], dtype=np.uint8),
            np.array([[[0], [1]], [[0], [1]]], dtype=np.uint8),
        ]
    )

    model = DummyForwardModel()
    evaluator = DummySearchAdapter()
    explainer = DHSICExplainer(model=model, num_classes=2, evaluator=evaluator, batch_size=4)

    def _fake_saliency_map(self, image, target):
        self.model.record_model_forward(5)
        return np.ones(image.shape[:2], dtype=np.float32)

    monkeypatch.setattr(DHSICExplainer, "saliency_map", _fake_saliency_map)

    _, json_dict, saliency = explainer(image=image, regions=regions, target=0)

    assert saliency.shape == image.shape[:2]
    assert json_dict["model_forward_calls"] == 5
    assert json_dict["saliency_model_forward_calls"] == 5
    assert json_dict["eval_model_forward_calls"] == 4
    assert json_dict["total_model_forward_calls"] == 9
    assert json_dict["model_forward_count_mode"] == "equivalent_single_image_forwards"
    assert json_dict["model_forward_count_scope"] == "algorithm_only"


def test_phasewin_json_records_model_forward_calls():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    masks = [
        np.array([[[1], [0]], [[0], [0]]], dtype=np.uint8),
        np.array([[[0], [1]], [[1], [0]]], dtype=np.uint8),
        np.array([[[0], [0]], [[0], [1]]], dtype=np.uint8),
    ]

    explainer = PhaseWinExplainer(
        DummySearchAdapter(),
        n_greedy=0,
        pw_window_size=2,
        pw_random_frac=0.0,
        pw_window_policy="LG",
        pw_enable_anneal=False,
        pw_enable_hard_exit=False,
    )
    _, json_dict = explainer(image=image, masks=masks, target=0, show_progress=False)

    assert json_dict["candidate_evaluations"] == 8
    assert json_dict["sub-region_number"] == 3
    assert json_dict["marginal_calls"] == 5
    assert json_dict["model_forward_calls"] == 16
    assert json_dict["selection_model_forward_calls"] == 16
    assert json_dict["eval_model_forward_calls"] == 6
    assert json_dict["total_model_forward_calls"] == 22
    assert json_dict["phasewin_effective_window_size"] == 2
    assert json_dict["phasewin_window_size_override"] == 2
    assert json_dict["phasewin_window_frac"] == 0.3
    assert json_dict["phasewin_random_frac"] == 0.0
    assert json_dict["phasewin_window_policy"] == "LG"
    assert json_dict["phasewin_enable_anneal"] is False
    assert json_dict["phasewin_enable_hard_exit"] is False
    assert json_dict["model_forward_count_mode"] == "equivalent_single_image_forwards"
    assert json_dict["model_forward_count_scope"] == "algorithm_only"


def test_phasewin_resolves_window_size_from_fraction():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    masks = [
        np.array([[[1], [0]], [[0], [0]]], dtype=np.uint8),
        np.array([[[0], [1]], [[0], [0]]], dtype=np.uint8),
        np.array([[[0], [0]], [[1], [0]]], dtype=np.uint8),
        np.array([[[0], [0]], [[0], [1]]], dtype=np.uint8),
        np.array([[[1], [1]], [[0], [0]]], dtype=np.uint8),
    ]

    explainer = PhaseWinExplainer(
        DummySearchAdapter(),
        n_greedy=0,
        pw_window_size=None,
        pw_window_frac=0.3,
        pw_random_frac=0.0,
        pw_enable_anneal=False,
        pw_enable_hard_exit=False,
    )
    _, json_dict = explainer(image=image, masks=masks, target=0, show_progress=False)

    assert json_dict["phasewin_window_size_override"] is None
    assert json_dict["phasewin_window_frac"] == 0.3
    assert json_dict["phasewin_effective_window_size"] == 1
