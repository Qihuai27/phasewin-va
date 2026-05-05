import json

import pytest

from attribution_research.evaluation.auc_faithfulness import (
    aggregate_results,
    compute_auc_from_json,
)


def test_compute_auc_from_json_adds_area_capped_highest_scores():
    saved_json = {
        "region_area": [0.1, 0.2, 0.4, 0.6],
        "insertion_score": [0.3, 0.8, 0.5, 0.9],
        "deletion_score": [0.4, 0.2, 0.1, 0.01],
        "model_forward_calls": 12,
        "eval_model_forward_calls": 3,
        "total_model_forward_calls": 15,
    }

    result = compute_auc_from_json(saved_json)

    assert result["highest_score"] == pytest.approx(0.9)
    assert result["highest_score_30pct_area"] == pytest.approx(0.8)
    assert result["highest_score_50pct_area"] == pytest.approx(0.8)
    assert result["model_forward_calls"] == 12.0
    assert result["eval_model_forward_calls"] == 3.0
    assert result["total_model_forward_calls"] == 15.0


def test_aggregate_results_averages_new_metrics(tmp_path):
    run_dir = tmp_path / "greedy-slico-division-50-1.0-1.0"
    json_dir = run_dir / "json"
    json_dir.mkdir(parents=True)

    sample_a = {
        "region_area": [0.1, 0.2, 0.4],
        "insertion_score": [0.2, 0.7, 0.6],
        "deletion_score": [0.5, 0.3, 0.1],
        "model_forward_calls": 10,
        "eval_model_forward_calls": 4,
        "total_model_forward_calls": 14,
    }
    sample_b = {
        "region_area": [0.1, 0.3, 0.5],
        "insertion_score": [0.4, 0.6, 0.9],
        "deletion_score": [0.6, 0.4, 0.2],
        "model_forward_calls": 20,
        "eval_model_forward_calls": 6,
        "total_model_forward_calls": 26,
    }

    with (json_dir / "a.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_a, handle)
    with (json_dir / "b.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_b, handle)

    agg = aggregate_results(str(run_dir))

    assert agg.n_samples == 2
    assert agg.average_highest == pytest.approx(0.8)
    assert agg.average_highest_30pct_area == pytest.approx(0.65)
    assert agg.average_highest_50pct_area == pytest.approx(0.8)
    assert agg.average_model_forward_calls == 15.0
    assert agg.average_eval_model_forward_calls == 5.0
    assert agg.average_total_model_forward_calls == 20.0
