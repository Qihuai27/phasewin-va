import json

import cv2
import numpy as np
import pytest

from attribution_research.evaluation.mufidelity import (
    collect_classification_samples,
    evaluate_classification_mufidelity,
    infer_classification_model_spec,
)


def test_infer_classification_model_spec_detects_clip_result_roots():
    vitl = infer_classification_model_spec(
        "/tmp/classification_results/imagenet-clip-vitl/greedy-slico-division-50-1.0-1.0",
        {},
    )
    rn101 = infer_classification_model_spec(
        "/tmp/classification_results/imagenet-clip-rn101/phasewin-slico-division-50-1.0-1.0-window-pct-30",
        {},
    )

    assert vitl.family == "clip"
    assert vitl.clip_type == "ViT-L/14"
    assert vitl.semantic_feature_path.endswith("clip_vitl_imagenet_zeroweights.pt")
    assert rn101.family == "clip"
    assert rn101.clip_type == "RN101"
    assert rn101.semantic_feature_path.endswith("clip_rn101_imagenet_zeroweights.pt")


def test_infer_classification_model_spec_uses_torchvision_metadata():
    spec = infer_classification_model_spec(
        "/tmp/classification_results/imagenet-resnet50/greedy-slico-division-50-1.0-1.0",
        {"model_arch": "resnet50", "model_weights": "DEFAULT"},
    )

    assert spec.family == "torchvision"
    assert spec.arch == "resnet50"
    assert spec.weights == "DEFAULT"


def test_infer_classification_model_spec_accepts_explicit_clip_override():
    spec = infer_classification_model_spec(
        "/tmp/custom-results/greedy-run",
        {},
        model_family="clip",
        clip_type="RN101",
    )

    assert spec.family == "clip"
    assert spec.clip_type == "RN101"
    assert spec.semantic_feature_path.endswith("clip_rn101_imagenet_zeroweights.pt")


def test_collect_and_evaluate_classification_mufidelity(monkeypatch, tmp_path):
    run_dir = tmp_path / "greedy-slico-division-50-1.0-1.0"
    json_dir = run_dir / "json"
    npy_dir = run_dir / "npy"
    json_dir.mkdir(parents=True)
    npy_dir.mkdir(parents=True)

    labels = [1, 4, 2]
    image_paths = []
    for idx, label in enumerate(labels):
        image = np.full((4, 4, 3), fill_value=idx * 50, dtype=np.uint8)
        image_path = tmp_path / f"sample_{idx}.png"
        cv2.imwrite(str(image_path), image)
        image_paths.append(str(image_path))

        masks = np.ones((1, 4, 4, 1), dtype=np.uint8)
        np.save(npy_dir / f"sample_{idx}.npy", masks)
        payload = {
            "image_path": str(image_path),
            "target_label": label,
            "model_arch": "resnet50",
            "model_weights": "DEFAULT",
            "insertion_score": [float(idx + 1)],
        }
        with (json_dir / f"sample_{idx}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    class DummyWrapper:
        num_classes = 5

    captured_batches = []

    class FakeMuFidelity:
        def __init__(self, model, inputs, targets, **kwargs):
            captured_batches.append((inputs.shape, targets.shape, kwargs))
            self.target_ids = np.argmax(targets, axis=1)
            self.inputs_shape = inputs.shape

        def __call__(self, explanations):
            assert explanations.shape == self.inputs_shape[:3]
            return float(np.mean(self.target_ids))

    monkeypatch.setattr(
        "attribution_research.evaluation.mufidelity._build_model_wrapper",
        lambda spec, device: DummyWrapper(),
    )
    monkeypatch.setattr(
        "attribution_research.evaluation.mufidelity._wrap_model_for_mufidelity",
        lambda model, device: model,
    )
    monkeypatch.setattr(
        "attribution_research.evaluation.mufidelity._get_mufidelity_class",
        lambda: FakeMuFidelity,
    )

    samples = collect_classification_samples(run_dir)
    assert [sample.target_label for sample in samples] == labels

    result = evaluate_classification_mufidelity(
        run_dir,
        batch_size=16,
        sample_batch_size=2,
        nb_samples=8,
        grid_size=4,
        subset_percent=0.25,
        baseline_mode="zero",
        device="cpu",
        show_progress=False,
    )

    assert result.n_samples == 3
    assert result.score == pytest.approx((2.5 * 2 + 2.0) / 3.0)
    assert result.batch_size == 16
    assert result.sample_batch_size == 1
    assert result.grid_size == 4
    assert result.subset_percent == pytest.approx(0.25)
    assert result.baseline_mode == pytest.approx(0.0)
    assert [batch[0] for batch in captured_batches] == [(1, 4, 4, 3)] * 3
    assert [batch[1] for batch in captured_batches] == [(1, 5)] * 3

    checkpoint_path = run_dir / "mufidelity_samples.jsonl"
    records = [
        json.loads(line)
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sample_id"] for record in records] == ["sample_0", "sample_1", "sample_2"]
    assert [record["score"] for record in records] == pytest.approx(labels)

    captured_batches.clear()
    resumed = evaluate_classification_mufidelity(
        run_dir,
        batch_size=16,
        sample_batch_size=2,
        nb_samples=8,
        grid_size=4,
        subset_percent=0.25,
        baseline_mode="zero",
        device="cpu",
        show_progress=False,
    )
    assert resumed.score == pytest.approx(result.score)
    assert resumed.n_samples == 3
    assert captured_batches == []
