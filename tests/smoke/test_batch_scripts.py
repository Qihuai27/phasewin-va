import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_SCRIPT = REPO_ROOT / "scripts/run_classification.sh"
CAPTION_SCRIPT = REPO_ROOT / "scripts/run_caption.sh"
ROUND1_SCRIPT = REPO_ROOT / "scripts/run_round1_extension.sh"


def _run_batch_script(script: Path, *args: str) -> str:
    completed = subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_classification_batch_script_default_dry_run():
    output = _run_batch_script(CLASSIFICATION_SCRIPT, "--dry-run", "--begin", "0", "--end", "1")
    assert "Model key  : clip_vitl14" in output
    assert "Split      : true" in output
    assert "Algorithms : greedy phasewin drise dhsic gradient grad_eclip ig2 igos_pp" in output
    assert "Segmenter  : superpixel / slico / 50 divisions" in output
    assert "--algorithm greedy" in output
    assert "--algorithm phasewin" in output
    assert "--algorithm drise" in output
    assert "--algorithm gradient" in output
    assert "--algorithm dhsic" in output
    assert "--algorithm grad_eclip" in output
    assert "--algorithm ig2" in output
    assert "--algorithm igos_pp" in output
    assert "--segmenter superpixel" in output
    assert "--superpixel-algorithm slico" in output


def test_classification_batch_script_supports_integrated_rn101_mistake_dry_run():
    output = _run_batch_script(
        CLASSIFICATION_SCRIPT,
        "--dry-run",
        "--model",
        "clip_rn101",
        "--split",
        "both",
        "--begin",
        "0",
        "--end",
        "1",
    )
    assert "Model key  : clip_rn101" in output
    assert "Split      : both" in output
    assert "cause / greedy" in output
    assert "repair / greedy" in output
    assert "classification_results/imagenet-clip-rn101/mistake/cause" in output
    assert "classification_results/imagenet-clip-rn101/mistake/repair" in output


def test_classification_batch_script_supports_resnet50_and_split_generation_dry_run(tmp_path):
    generated_dir = tmp_path / "generated"
    eval_list = generated_dir / "resnet50_true.txt"
    output = _run_batch_script(
        CLASSIFICATION_SCRIPT,
        "--dry-run",
        "--build-if-missing",
        "--model",
        "resnet50",
        "--generated-dir",
        str(generated_dir),
        "--eval-list",
        str(eval_list),
        "--begin",
        "0",
        "--end",
        "1",
    )
    assert "Model key  : resnet50" in output
    assert "torchvision ResNet-50" in output
    assert "build_imagenet_eval_lists.py" in output
    assert "--model resnet50" in output
    assert "--arch resnet50" in output


def test_round1_extension_tracks_can_be_subset_in_dry_run():
    output = _run_batch_script(
        ROUND1_SCRIPT,
        "--dry-run",
        "--tracks",
        "clip_rn101,qwen7b",
        "--begin",
        "0",
        "--end",
        "1",
    )
    assert "Tracks     : clip_rn101 qwen7b" in output
    assert "scripts/run_classification.sh" in output
    assert "--model clip_rn101" in output
    assert "scripts/run_caption.sh" in output
    assert "--model-size 7b" in output
    assert "--model clip_vitl14" not in output


def test_caption_batch_script_includes_igospp_in_dry_run():
    output = _run_batch_script(CAPTION_SCRIPT, "--dry-run", "--begin", "0", "--end", "1")
    assert "Model size : 3b" in output
    assert "Algorithms : greedy phasewin drise gradient llavacam igos_pp" in output
    assert "--algorithm igos_pp" in output


def test_caption_batch_script_supports_integrated_7b_dry_run():
    output = _run_batch_script(CAPTION_SCRIPT, "--dry-run", "--model-size", "7b", "--begin", "0", "--end", "1")
    assert "Model size : 7b" in output
    assert "Qwen2.5-VL-7B-coco-caption" in output
