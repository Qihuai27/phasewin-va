from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TaskModelPreset:
    task: str
    model_key: str
    display_name: str
    runner_script: str
    runner_args: tuple[str, ...]
    result_root: str
    datasets: str
    eval_list: str
    methods: tuple[str, ...]
    evaluation_script: str
    visualization_root: str
    common_metrics: tuple[str, ...]
    specific_metrics: tuple[str, ...]
    false_gt_list: str | None = None
    false_pred_list: str | None = None
    model_name: str | None = None


COMMON_CLASSIFICATION_METRICS = (
    "insertion_auc",
    "deletion_auc",
    "average_highest",
    "average_highest_30pct_area",
    "average_highest_50pct_area",
    "average_model_forward_calls",
)
COMMON_CAPTION_METRICS = (
    "insertion_auc",
    "deletion_auc",
    "average_highest",
    "average_highest_30pct_area",
    "average_highest_50pct_area",
    "average_model_forward_calls",
)


CLASSIFICATION_MODELS: dict[str, TaskModelPreset] = {
    "clip_vitl14": TaskModelPreset(
        task="classification",
        model_key="clip_vitl14",
        display_name="CLIP ViT-L/14",
        runner_script="scripts/run_classification.sh",
        runner_args=("--model", "clip_vitl14"),
        result_root="classification_results/imagenet-clip-vitl",
        datasets="datasets/imagenet/ILSVRC2012_img_val",
        eval_list="datasets/imagenet/generated/clip_vitl14_true.txt",
        false_gt_list="datasets/imagenet/generated/clip_vitl14_false_gt.txt",
        false_pred_list="datasets/imagenet/generated/clip_vitl14_false_pred.txt",
        methods=("greedy", "phasewin", "drise", "dhsic", "gradient", "grad_eclip", "ig2", "igos_pp"),
        evaluation_script="scripts/eval_classification.py",
        visualization_root="visualizations/classification/clip_vitl14",
        common_metrics=COMMON_CLASSIFICATION_METRICS,
        specific_metrics=("mufidelity",),
    ),
    "clip_rn101": TaskModelPreset(
        task="classification",
        model_key="clip_rn101",
        display_name="CLIP RN101",
        runner_script="scripts/run_classification.sh",
        runner_args=("--model", "clip_rn101"),
        result_root="classification_results/imagenet-clip-rn101",
        datasets="datasets/imagenet/ILSVRC2012_img_val",
        eval_list="datasets/imagenet/generated/clip_rn101_true.txt",
        false_gt_list="datasets/imagenet/generated/clip_rn101_false_gt.txt",
        false_pred_list="datasets/imagenet/generated/clip_rn101_false_pred.txt",
        methods=("greedy", "phasewin", "drise", "dhsic", "gradient", "ig2", "igos_pp"),
        evaluation_script="scripts/eval_classification.py",
        visualization_root="visualizations/classification/clip_rn101",
        common_metrics=COMMON_CLASSIFICATION_METRICS,
        specific_metrics=("mufidelity",),
    ),
    "resnet50": TaskModelPreset(
        task="classification",
        model_key="resnet50",
        display_name="ResNet-50",
        runner_script="scripts/run_classification.sh",
        runner_args=("--model", "resnet50"),
        result_root="classification_results/imagenet-resnet50",
        datasets="datasets/imagenet/ILSVRC2012_img_val",
        eval_list="datasets/imagenet/generated/resnet50_true.txt",
        false_gt_list="datasets/imagenet/generated/resnet50_false_gt.txt",
        false_pred_list="datasets/imagenet/generated/resnet50_false_pred.txt",
        methods=("greedy", "phasewin", "drise", "dhsic", "gradient", "ig2", "igos_pp"),
        evaluation_script="scripts/eval_classification.py",
        visualization_root="visualizations/classification/resnet50",
        common_metrics=COMMON_CLASSIFICATION_METRICS,
        specific_metrics=(),
    ),
    "resnet101": TaskModelPreset(
        task="classification",
        model_key="resnet101",
        display_name="ResNet-101",
        runner_script="scripts/run_classification.sh",
        runner_args=("--model", "resnet101"),
        result_root="classification_results/imagenet-resnet101",
        datasets="datasets/imagenet/ILSVRC2012_img_val",
        eval_list="datasets/imagenet/generated/resnet101_true.txt",
        false_gt_list="datasets/imagenet/generated/resnet101_false_gt.txt",
        false_pred_list="datasets/imagenet/generated/resnet101_false_pred.txt",
        methods=("greedy", "phasewin", "drise", "dhsic", "gradient", "ig2", "igos_pp"),
        evaluation_script="scripts/eval_classification.py",
        visualization_root="visualizations/classification/resnet101",
        common_metrics=COMMON_CLASSIFICATION_METRICS,
        specific_metrics=(),
    ),
}

CAPTION_MODELS: dict[str, TaskModelPreset] = {
    "qwen25vl_3b": TaskModelPreset(
        task="caption",
        model_key="qwen25vl_3b",
        display_name="Qwen2.5-VL-3B",
        runner_script="scripts/run_caption.sh",
        runner_args=("--model-size", "3b"),
        result_root="caption_results/Qwen2.5-VL-3B-coco-caption",
        datasets="datasets/coco/val2017",
        eval_list="datasets/Qwen2.5-VL-3B-coco-caption.json",
        methods=("greedy", "phasewin", "drise", "gradient", "llavacam", "igos_pp"),
        evaluation_script="scripts/eval_caption.py",
        visualization_root="visualizations/caption/qwen25vl_3b",
        common_metrics=COMMON_CAPTION_METRICS,
        specific_metrics=("insertion_sensitivity_auc", "deletion_sensitivity_auc", "sensitivity_highest"),
        model_name="model_checkpoint/Qwen2.5-VL-3B-Instruct",
    ),
    "qwen25vl_7b": TaskModelPreset(
        task="caption",
        model_key="qwen25vl_7b",
        display_name="Qwen2.5-VL-7B",
        runner_script="scripts/run_caption.sh",
        runner_args=("--model-size", "7b"),
        result_root="caption_results/Qwen2.5-VL-7B-coco-caption",
        datasets="datasets/coco/val2017",
        eval_list="datasets/Qwen2.5-VL-7B-coco-caption.json",
        methods=("greedy", "phasewin", "drise", "gradient", "llavacam", "igos_pp"),
        evaluation_script="scripts/eval_caption.py",
        visualization_root="visualizations/caption/qwen25vl_7b",
        common_metrics=COMMON_CAPTION_METRICS,
        specific_metrics=("insertion_sensitivity_auc", "deletion_sensitivity_auc", "sensitivity_highest"),
        model_name="model_checkpoint/Qwen2.5-VL-7B-Instruct",
    ),
}

TASK_MODEL_REGISTRY = {
    "classification": CLASSIFICATION_MODELS,
    "caption": CAPTION_MODELS,
}


def task_models(task: str) -> dict[str, TaskModelPreset]:
    normalized = str(task).strip().lower().replace("-", "_")
    if normalized == "caption_vqa":
        normalized = "caption"
    try:
        return TASK_MODEL_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported task: {task!r}") from exc


def select_presets(task: str, model: str = "all") -> list[TaskModelPreset]:
    registry = task_models(task)
    normalized_model = str(model).strip().lower()
    if normalized_model == "all":
        return [registry[key] for key in sorted(registry)]
    try:
        return [registry[normalized_model]]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported model {model!r} for task {task!r}. Expected one of: {choices}, all"
        ) from exc


__all__ = [
    "TaskModelPreset",
    "CLASSIFICATION_MODELS",
    "CAPTION_MODELS",
    "COMMON_CLASSIFICATION_METRICS",
    "COMMON_CAPTION_METRICS",
    "select_presets",
    "task_models",
]
