# -*- coding: utf-8 -*-
"""
Classification-only MuFidelity evaluation.

This module reconstructs attribution maps from saved classification results,
rebuilds the underlying classifier, and evaluates explanation fidelity via
`xplique.metrics.MuFidelity`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from hashlib import sha1

import cv2
import numpy as np
from tqdm import tqdm

from attribution_research.evaluation.point_game import build_saliency_map


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_SEMANTIC_FEATURES = {
    "ViT-L/14": REPO_ROOT / "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt",
    "RN101": REPO_ROOT / "ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt",
}


@dataclass(frozen=True)
class ClassificationMuFidelitySample:
    image_path: str
    target_label: int
    json_path: Path
    npy_path: Path


@dataclass(frozen=True)
class ClassificationModelSpec:
    family: str
    clip_type: Optional[str] = None
    semantic_feature_path: Optional[str] = None
    arch: Optional[str] = None
    weights: Optional[str] = None


@dataclass(frozen=True)
class MuFidelityResult:
    score: float
    n_samples: int
    score_key: str
    grid_size: int
    subset_percent: float
    nb_samples: int
    batch_size: int
    sample_batch_size: int
    baseline_mode: float
    device: str
    tf_device: str
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_config_key: Optional[str] = None

    def config_dict(self) -> dict[str, Any]:
        return {
            "score_key": self.score_key,
            "grid_size": self.grid_size,
            "subset_percent": self.subset_percent,
            "nb_samples": self.nb_samples,
            "batch_size": self.batch_size,
            "sample_batch_size": self.sample_batch_size,
            "baseline_mode": self.baseline_mode,
            "device": self.device,
            "tf_device": self.tf_device,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_config_key": self.checkpoint_config_key,
        }


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield items[start : start + step]


def _parse_baseline_mode(value: float | str) -> float:
    if isinstance(value, (float, int)):
        return float(value)

    text = str(value).strip().lower()
    if text in {"zero", "black"}:
        return 0.0
    if text in {"one", "white"}:
        return 1.0
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            "Unsupported MuFidelity baseline value. Use a float, 'zero', or 'one'."
        ) from exc


def _configure_tensorflow_runtime(tf_device: str) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        return

    policy = str(tf_device).strip().lower()
    if policy == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        return

    if policy in {"auto", "gpu"}:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        return

    raise ValueError(
        f"Unsupported TensorFlow device policy for MuFidelity: {tf_device!r}. "
        "Expected one of: cpu, auto, gpu"
    )


def _checkpoint_config_key(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()


def _load_mufidelity_checkpoint(path: Path, config_key: str) -> dict[str, float]:
    if not path.is_file():
        return {}

    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("config_key") != config_key:
                continue
            sample_id = item.get("sample_id")
            score = item.get("score")
            if sample_id is None or score is None:
                continue
            scores[str(sample_id)] = float(score)
    return scores


def _append_mufidelity_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _get_mufidelity_class():
    try:
        from xplique.metrics import MuFidelity
    except ImportError as exc:
        raise ImportError(
            "MuFidelity evaluation requires the optional `xplique` dependency."
        ) from exc
    return MuFidelity


def _wrap_model_for_mufidelity(model, device: str):
    try:
        from xplique.wrappers import TorchWrapper
    except ImportError as exc:
        raise ImportError(
            "MuFidelity evaluation requires xplique's TorchWrapper bridge."
        ) from exc

    base_model = model.eval() if hasattr(model, "eval") else model
    return TorchWrapper(base_model, device)


def collect_classification_samples(
    result_dir: str | Path,
    *,
    limit: Optional[int] = None,
) -> list[ClassificationMuFidelitySample]:
    root = Path(result_dir)
    json_dir = root / "json"
    npy_dir = root / "npy"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not npy_dir.is_dir():
        raise FileNotFoundError(f"NPY directory not found: {npy_dir}")

    samples: list[ClassificationMuFidelitySample] = []
    for json_path in sorted(json_dir.glob("*.json")):
        npy_path = npy_dir / f"{json_path.stem}.npy"
        if not npy_path.is_file():
            continue
        with open(json_path, "r", encoding="utf-8") as handle:
            saved_json = json.load(handle)
        if "image_path" not in saved_json or "target_label" not in saved_json:
            continue
        samples.append(
            ClassificationMuFidelitySample(
                image_path=str(saved_json["image_path"]),
                target_label=int(saved_json["target_label"]),
                json_path=json_path,
                npy_path=npy_path,
            )
        )
        if limit is not None and len(samples) >= int(limit):
            break

    if not samples:
        raise FileNotFoundError(f"No complete classification result pairs found under {root}")
    return samples


def infer_classification_model_spec(
    result_dir: str | Path,
    sample_metadata: dict[str, Any],
    *,
    semantic_feature_path: str | None = None,
    model_family: str | None = None,
    clip_type: str | None = None,
    arch: str | None = None,
    weights: str | None = None,
) -> ClassificationModelSpec:
    family_hint = None if model_family is None else str(model_family).strip().lower()
    if family_hint in {"torchvision", "resnet", "cnn"}:
        resolved_arch = arch or sample_metadata.get("model_arch")
        if resolved_arch is None:
            raise ValueError("MuFidelity torchvision override requires an architecture name.")
        resolved_weights = weights if weights is not None else sample_metadata.get("model_weights", "DEFAULT")
        return ClassificationModelSpec(
            family="torchvision",
            arch=str(resolved_arch).strip().lower(),
            weights=str(resolved_weights) if resolved_weights is not None else None,
        )

    if family_hint in {"clip"}:
        resolved_clip_type = clip_type
        if resolved_clip_type is None:
            result_text = str(Path(result_dir)).lower()
            if "clip-rn101" in result_text or "clip_rn101" in result_text:
                resolved_clip_type = "RN101"
            elif "clip-vitl" in result_text or "clip_vitl" in result_text or "vitl14" in result_text:
                resolved_clip_type = "ViT-L/14"
        if resolved_clip_type is None:
            raise ValueError("MuFidelity CLIP override requires --mu-clip-type.")
        semantic_path = (
            Path(semantic_feature_path)
            if semantic_feature_path is not None
            else DEFAULT_CLIP_SEMANTIC_FEATURES[str(resolved_clip_type)]
        )
        return ClassificationModelSpec(
            family="clip",
            clip_type=str(resolved_clip_type),
            semantic_feature_path=str(semantic_path),
        )

    if "model_arch" in sample_metadata:
        arch = str(sample_metadata["model_arch"]).strip().lower()
        weights = sample_metadata.get("model_weights", "DEFAULT")
        return ClassificationModelSpec(
            family="torchvision",
            arch=arch,
            weights=str(weights) if weights is not None else None,
        )

    result_text = str(Path(result_dir)).lower()
    if "clip-rn101" in result_text or "clip_rn101" in result_text:
        clip_type = "RN101"
    elif "clip-vitl" in result_text or "clip_vitl" in result_text or "vitl14" in result_text:
        clip_type = "ViT-L/14"
    else:
        raise ValueError(
            "Unable to infer classification model from result directory. "
            "Expected a CLIP or torchvision ImageNet result root."
        )

    semantic_path = (
        Path(semantic_feature_path)
        if semantic_feature_path is not None
        else DEFAULT_CLIP_SEMANTIC_FEATURES[clip_type]
    )
    return ClassificationModelSpec(
        family="clip",
        clip_type=clip_type,
        semantic_feature_path=str(semantic_path),
    )


def _build_model_wrapper(spec: ClassificationModelSpec, device: str):
    if spec.family == "clip":
        from attribution_research.adapters.clip import CLIPXpliqueWrapper

        semantic_path = spec.semantic_feature_path
        if semantic_path is None:
            raise ValueError("CLIP MuFidelity requires semantic feature weights.")
        return CLIPXpliqueWrapper(
            clip_type=spec.clip_type or "ViT-L/14",
            semantic_feature_path=str(_resolve_repo_path(semantic_path)),
            download_root=str(REPO_ROOT / ".checkpoints/CLIP"),
            device=device,
        )

    if spec.family == "torchvision":
        from attribution_research.adapters.torchvision_imagenet import TorchvisionImageNetXpliqueWrapper

        return TorchvisionImageNetXpliqueWrapper(
            arch=spec.arch or "resnet101",
            weights=spec.weights,
            device=device,
        )

    raise ValueError(f"Unsupported classification model family for MuFidelity: {spec.family!r}")


def _model_spec_dict(spec: ClassificationModelSpec) -> dict[str, Any]:
    return {
        "family": spec.family,
        "clip_type": spec.clip_type,
        "semantic_feature_path": str(spec.semantic_feature_path) if spec.semantic_feature_path else None,
        "arch": spec.arch,
        "weights": spec.weights,
    }


def _load_batch_arrays(
    samples: list[ClassificationMuFidelitySample],
    *,
    num_classes: int,
    score_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    explanations: list[np.ndarray] = []
    labels: list[int] = []

    for sample in samples:
        saliency = build_saliency_map(
            str(sample.npy_path),
            str(sample.json_path),
            score_key=score_key,
        ).astype(np.float32)
        height, width = saliency.shape

        image_path = _resolve_repo_path(sample.image_path)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read source image for MuFidelity: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if image_rgb.shape[:2] != (height, width):
            image_rgb = cv2.resize(
                image_rgb,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        inputs.append(image_rgb.astype(np.float32))
        explanations.append(saliency)
        labels.append(int(sample.target_label))

    label_indices = np.asarray(labels, dtype=np.int64)
    targets = np.eye(int(num_classes), dtype=np.float32)[label_indices]
    return (
        np.stack(inputs, axis=0),
        targets,
        np.stack(explanations, axis=0),
    )


def evaluate_classification_mufidelity(
    result_dir: str | Path,
    *,
    score_key: str = "insertion_score",
    limit: Optional[int] = None,
    grid_size: int = 9,
    subset_percent: float = 0.2,
    nb_samples: int = 200,
    batch_size: int = 64,
    sample_batch_size: int = 8,
    baseline_mode: float | str = 0.0,
    semantic_feature_path: str | None = None,
    model_family: str | None = None,
    clip_type: str | None = None,
    arch: str | None = None,
    weights: str | None = None,
    device: str = "cuda",
    tf_device: str = "cpu",
    seed: Optional[int] = 0,
    use_checkpoint: bool = True,
    checkpoint_path: str | Path | None = None,
    show_progress: bool = True,
) -> MuFidelityResult:
    root = Path(result_dir)
    samples = collect_classification_samples(root, limit=limit)
    with open(samples[0].json_path, "r", encoding="utf-8") as handle:
        sample_metadata = json.load(handle)

    spec = infer_classification_model_spec(
        root,
        sample_metadata,
        semantic_feature_path=semantic_feature_path,
        model_family=model_family,
        clip_type=clip_type,
        arch=arch,
        weights=weights,
    )
    baseline_value = _parse_baseline_mode(baseline_mode)
    metric_checkpoint_config = {
        "score_key": score_key,
        "grid_size": int(grid_size),
        "subset_percent": float(subset_percent),
        "nb_samples": int(nb_samples),
        "baseline_mode": float(baseline_value),
        "seed": None if seed is None else int(seed),
        "model": _model_spec_dict(spec),
    }
    config_key = _checkpoint_config_key(metric_checkpoint_config)
    resolved_checkpoint_path: Path | None = None
    completed_scores: dict[str, float] = {}
    if use_checkpoint:
        resolved_checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else (root / "mufidelity_samples.jsonl")
        )
        completed_scores = _load_mufidelity_checkpoint(resolved_checkpoint_path, config_key)

    weighted_total = float(sum(completed_scores.values()))
    n_total = len(completed_scores)
    pending_samples = [
        sample for sample in samples
        if sample.json_path.stem not in completed_scores
    ]

    if use_checkpoint and not pending_samples and n_total > 0:
        return MuFidelityResult(
            score=weighted_total / float(n_total),
            n_samples=n_total,
            score_key=score_key,
            grid_size=int(grid_size),
            subset_percent=float(subset_percent),
            nb_samples=int(nb_samples),
            batch_size=int(batch_size),
            sample_batch_size=1,
            baseline_mode=float(baseline_value),
            device=str(device),
            tf_device=str(tf_device),
            seed=None if seed is None else int(seed),
            checkpoint_path=str(resolved_checkpoint_path),
            checkpoint_config_key=config_key,
        )

    _configure_tensorflow_runtime(tf_device)
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "MuFidelity evaluation requires tensorflow through `xplique`."
        ) from exc

    if seed is not None:
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))

    MuFidelity = _get_mufidelity_class()
    raw_model = _build_model_wrapper(spec, device=device)
    wrapped_model = _wrap_model_for_mufidelity(raw_model, device=device)

    if use_checkpoint:
        iterator = pending_samples
        if show_progress:
            iterator = tqdm(iterator, desc="MuFidelity", initial=n_total, total=len(samples))

        for sample in iterator:
            inputs, targets, explanations = _load_batch_arrays(
                [sample],
                num_classes=int(getattr(wrapped_model, "num_classes", 1000)),
                score_key=score_key,
            )
            metric = MuFidelity(
                wrapped_model,
                inputs,
                targets,
                batch_size=int(batch_size),
                grid_size=int(grid_size),
                subset_percent=float(subset_percent),
                baseline_mode=float(baseline_value),
                nb_samples=int(nb_samples),
            )
            score = float(metric(explanations))
            weighted_total += score
            n_total += 1
            if resolved_checkpoint_path is not None:
                _append_mufidelity_checkpoint(
                    resolved_checkpoint_path,
                    {
                        "config_key": config_key,
                        "sample_id": sample.json_path.stem,
                        "score": score,
                        "target_label": sample.target_label,
                        "json_path": str(sample.json_path),
                        "npy_path": str(sample.npy_path),
                        "created_at": time.time(),
                    },
                )
    else:
        iterator = list(_batched(samples, sample_batch_size))
        if show_progress:
            iterator = tqdm(iterator, desc="MuFidelity")

        for batch_samples in iterator:
            inputs, targets, explanations = _load_batch_arrays(
                batch_samples,
                num_classes=int(getattr(wrapped_model, "num_classes", 1000)),
                score_key=score_key,
            )
            metric = MuFidelity(
                wrapped_model,
                inputs,
                targets,
                batch_size=int(batch_size),
                grid_size=int(grid_size),
                subset_percent=float(subset_percent),
                baseline_mode=float(baseline_value),
                nb_samples=int(nb_samples),
            )
            score = float(metric(explanations))
            weighted_total += score * len(batch_samples)
            n_total += len(batch_samples)

    if n_total == 0:
        raise RuntimeError(f"MuFidelity did not evaluate any samples under {root}")

    return MuFidelityResult(
        score=weighted_total / float(n_total),
        n_samples=n_total,
        score_key=score_key,
        grid_size=int(grid_size),
        subset_percent=float(subset_percent),
        nb_samples=int(nb_samples),
        batch_size=int(batch_size),
        sample_batch_size=1 if use_checkpoint else int(sample_batch_size),
        baseline_mode=float(baseline_value),
        device=str(device),
        tf_device=str(tf_device),
        seed=None if seed is None else int(seed),
        checkpoint_path=str(resolved_checkpoint_path) if resolved_checkpoint_path is not None else None,
        checkpoint_config_key=config_key if use_checkpoint else None,
    )


__all__ = [
    "ClassificationModelSpec",
    "ClassificationMuFidelitySample",
    "MuFidelityResult",
    "collect_classification_samples",
    "evaluate_classification_mufidelity",
    "infer_classification_model_spec",
]
