from pathlib import Path

import pytest

from scripts._suite_utils import method_key_from_result_dir, shard_bounds, shared_result_ids


def test_shard_bounds_even_and_uneven():
    assert shard_bounds(10, 2, 0) == (0, 5)
    assert shard_bounds(10, 2, 1) == (5, 10)
    assert shard_bounds(10, 3, 0) == (0, 4)
    assert shard_bounds(10, 3, 1) == (4, 7)
    assert shard_bounds(10, 3, 2) == (7, 10)


def test_shard_bounds_rejects_invalid_index():
    with pytest.raises(ValueError):
        shard_bounds(10, 3, 3)


def test_method_key_from_result_dir_uses_prefix():
    assert method_key_from_result_dir(Path("phasewin-slico-division-50-1.0-1.0-window-pct-30")) == "phasewin"
    assert method_key_from_result_dir(Path("igos_pp-slico-division-64-1.0-1.0")) == "igos_pp"


def test_shared_result_ids_preserves_eval_order(monkeypatch):
    method_dirs = {
        "greedy": Path("/tmp/greedy"),
        "phasewin": Path("/tmp/phasewin"),
    }

    def fake_glob(self, pattern):
        if self == Path("/tmp/greedy/json"):
            return [Path("/tmp/greedy/json/b.json"), Path("/tmp/greedy/json/c.json")]
        if self == Path("/tmp/phasewin/json"):
            return [Path("/tmp/phasewin/json/a.json"), Path("/tmp/phasewin/json/c.json"), Path("/tmp/phasewin/json/b.json")]
        return []

    monkeypatch.setattr(Path, "glob", fake_glob)
    selected = shared_result_ids(method_dirs, ["c", "b", "a"], 2)

    assert selected == ["c", "b"]
