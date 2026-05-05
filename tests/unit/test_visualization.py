import numpy as np

from attribution_research.visualization import (
    build_attribution_map,
    build_word_region_matrix,
    compute_step_scores,
    find_early_peak,
    get_word_saliency,
)

TRAPEZOID = getattr(np, "trapezoid", getattr(np, "trapz", None))


def test_compute_step_scores_supports_delta_modes():
    scores = [0.2, 0.5, 0.4]

    assert np.allclose(
        compute_step_scores(scores, baseline_score=0.1, score_mode="delta"),
        np.array([0.1, 0.3, -0.1], dtype=np.float32),
    )
    assert np.allclose(
        compute_step_scores(scores, baseline_score=0.1, score_mode="positive_delta"),
        np.array([0.1, 0.3, 0.0], dtype=np.float32),
    )


def test_build_attribution_map_uses_positive_delta_weights():
    masks = np.zeros((3, 2, 2, 1), dtype=np.uint8)
    masks[0, 0, 0, 0] = 1
    masks[1, 0, 1, 0] = 1
    masks[2, 1, 0, 0] = 1

    attr_map = build_attribution_map(
        masks,
        [0.2, 0.5, 0.4],
        normalize=True,
        score_mode="positive_delta",
        baseline_score=0.1,
    )

    expected = np.array(
        [
            [1.0 / 3.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(attr_map, expected)


def test_find_early_peak_respects_area_limit():
    peak = find_early_peak([0.1, 0.2, 0.35], [0.2, 0.5, 0.9], area_limit=0.3)

    assert peak["index"] == 1
    assert peak["area"] == np.float32(0.2)
    assert peak["score"] == np.float32(0.5)


def test_word_region_helpers_capture_stepwise_word_contributions():
    info = {
        "region_area": [0.2, 0.5, 1.0],
        "insertion_word_score": [
            [0.1, 0.2],
            [0.4, 0.1],
            [0.5, 0.4],
        ],
        "deletion_word_score": [
            [0.0, 0.1],
            [0.1, 0.0],
            [0.2, 0.05],
        ],
    }

    matrix = build_word_region_matrix(info, mode="delta")
    expected_matrix = np.array(
        [
            [0.1, 0.2, 0.0],
            [0.1, 0.0, 0.25],
        ],
        dtype=np.float32,
    )
    assert np.allclose(matrix, expected_matrix)

    saliency = get_word_saliency(info, mode="auc_delta")
    curves = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.3, 0.1],
            [0.3, 0.35],
        ],
        dtype=np.float32,
    )
    areas = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
    expected_saliency = TRAPEZOID(curves, x=areas, axis=0)
    assert np.allclose(np.asarray(saliency, dtype=np.float32), expected_saliency)
