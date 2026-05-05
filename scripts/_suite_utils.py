from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from _suite_registry import REPO_ROOT, TaskModelPreset
except ImportError:  # pragma: no cover - package import path for tests
    from ._suite_registry import REPO_ROOT, TaskModelPreset


def parse_algorithms(csv_value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if csv_value is None or str(csv_value).strip() == "":
        return tuple(default)
    return tuple(part.strip() for part in str(csv_value).split(",") if part.strip())


def resolve_result_root(preset: TaskModelPreset, root_base: str | None = None) -> Path:
    if root_base is None:
        return REPO_ROOT / preset.result_root
    return Path(root_base) / Path(preset.result_root).name


def resolve_visualization_root(preset: TaskModelPreset, root_base: str | None = None) -> Path:
    if root_base is None:
        return REPO_ROOT / preset.visualization_root
    return Path(root_base) / Path(preset.visualization_root).name


def _load_text_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_path = line.split()[0]
        ids.append(Path(image_path).stem)
    return ids


def _load_json_ids(path: Path) -> list[str]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [Path(item["image_path"]).stem for item in items]


def ordered_eval_ids(
    preset: TaskModelPreset,
    *,
    split: str = "true",
    eval_list_override: str | None = None,
) -> list[str]:
    if preset.task == "classification":
        split_name = str(split).strip().lower()
        if split_name == "true":
            path = Path(eval_list_override) if eval_list_override is not None else (REPO_ROOT / preset.eval_list)
            return _load_text_ids(path)
        if split_name in {"cause", "repair", "both"}:
            source = preset.false_gt_list
            if source is None:
                raise ValueError(f"No generated false list configured for {preset.model_key}")
            return _load_text_ids(REPO_ROOT / source)
        raise ValueError(f"Unsupported classification split: {split!r}")

    path = Path(eval_list_override) if eval_list_override is not None else (REPO_ROOT / preset.eval_list)
    return _load_json_ids(path)


def count_eval_items(
    preset: TaskModelPreset,
    *,
    split: str = "true",
    eval_list_override: str | None = None,
) -> int:
    return len(ordered_eval_ids(preset, split=split, eval_list_override=eval_list_override))


def shard_bounds(total: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    base = total // num_shards
    rem = total % num_shards
    start = shard_index * base + min(shard_index, rem)
    stop = start + base + (1 if shard_index < rem else 0)
    return start, stop


def find_result_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "json").is_dir() and any((d / "json").glob("*.json"))
    )


def method_key_from_result_dir(path: Path) -> str:
    return path.name.split("-", 1)[0].strip().lower().replace("-", "_")


def select_method_dirs(root: Path, methods: tuple[str, ...]) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in find_result_dirs(root):
        grouped.setdefault(method_key_from_result_dir(path), []).append(path)

    selected: dict[str, Path] = {}
    for method in methods:
        candidates = grouped.get(method, [])
        if not candidates:
            continue
        selected[method] = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    return selected


def shared_result_ids(method_dirs: dict[str, Path], ordered_ids: list[str], sample_count: int) -> list[str]:
    if not method_dirs:
        return []

    available_sets = []
    for path in method_dirs.values():
        available_sets.append({json_path.stem for json_path in (path / "json").glob("*.json")})
    shared = set.intersection(*available_sets) if available_sets else set()

    selected = [sample_id for sample_id in ordered_ids if sample_id in shared]
    return selected[:sample_count]


def print_command(cmd: list[str]) -> None:
    print("  " + " ".join(shlex.quote(part) for part in cmd))


def run_or_print(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print_command(cmd)
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def python_executable(value: str | None = None) -> str:
    if value is not None:
        return value
    return sys.executable


__all__ = [
    "REPO_ROOT",
    "count_eval_items",
    "find_result_dirs",
    "method_key_from_result_dir",
    "ordered_eval_ids",
    "parse_algorithms",
    "print_command",
    "python_executable",
    "resolve_result_root",
    "resolve_visualization_root",
    "run_or_print",
    "select_method_dirs",
    "shard_bounds",
    "shared_result_ids",
    "write_json",
]
