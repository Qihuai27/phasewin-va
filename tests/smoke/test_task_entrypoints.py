import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_help(relative_path: str) -> str:
    completed = subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_classification_entrypoint_help_runs():
    output = _run_help("tasks/classification/clip_imagenet.py")
    assert "--algorithm" in output
    assert "phasewin" in output
    assert "igos_pp" in output


def test_detection_entrypoint_help_runs():
    output = _run_help("tasks/detection/groundingdino_coco.py")
    assert "--algorithm" in output
    assert "gradcam" in output
    assert "odam" in output
    assert "ssgrad_cam_pp" in output


def test_caption_entrypoint_help_runs():
    output = _run_help("tasks/caption_vqa/qwen25vl_coco_caption.py")
    assert "--algorithm" in output
    assert "llavacam" in output
    assert "igos_pp" in output


def test_auc_eval_entrypoint_help_runs():
    output = _run_help("scripts/eval_auc_faithfulness.py")
    assert "--explanation-dir" in output


def test_classification_eval_entrypoint_help_runs():
    output = _run_help("scripts/eval_classification.py")
    assert "--results-dir" in output
    assert "--mufidelity" in output
    assert "default: 8" in output


def test_classification_suite_entrypoints_help_run():
    output = _run_help("scripts/run_classification_suite.py")
    assert "--num-shards" in output
    output = _run_help("scripts/eval_classification_suite.py")
    assert "--skip-mufidelity" in output
    assert "--mu-batch-size" in output
    output = _run_help("scripts/visualize_classification_results.py")
    assert "--sample-count" in output


def test_caption_suite_entrypoints_help_run():
    output = _run_help("scripts/run_caption_suite.py")
    assert "--num-shards" in output
    output = _run_help("scripts/eval_caption_suite.py")
    assert "--sensitivity" in output
    output = _run_help("scripts/visualize_caption_results.py")
    assert "--sample-count" in output


def test_point_game_eval_entrypoint_help_runs():
    output = _run_help("scripts/eval_point_game.py")
    assert "--annotation-file" in output


def test_summarize_metrics_entrypoint_help_runs():
    output = _run_help("scripts/summarize_result_metrics.py")
    assert "--result-root" in output
    assert "--output-csv" in output
