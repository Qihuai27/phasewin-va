import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/list_baselines.py"


def _run_script(*args: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_baseline_inventory_script_help_runs():
    output = _run_script("--help")
    assert "--group-by" in output
    assert "--runnable-only" in output


def test_baseline_inventory_script_surfaces_caption_support_states():
    output = _run_script("--task", "caption_vqa", "--group-by", "support")
    assert "[support=native]" in output
    assert "[support=catalog]" in output
    assert "[support=compat]" not in output
    assert "- gradient | support=native" in output
    assert "- igos_pp | support=native" in output
    assert "- llavacam | support=native" in output
    assert "- tam | support=catalog" in output
