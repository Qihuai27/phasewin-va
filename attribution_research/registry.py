# -*- coding: utf-8 -*-
"""
Algorithm taxonomy and task support registry.
"""

from __future__ import annotations

from typing import Dict, Tuple


def normalize_algorithm_name(name: str) -> str:
    """Normalize CLI-style algorithm names."""
    return str(name).strip().lower().replace("-", "_")


def normalize_task_name(task_name: str) -> str:
    """Normalize task keys used in the task support registry."""
    return str(task_name).strip().lower().replace("-", "_")


GRADIENT_ALGORITHMS = (
    "gradient",
    "grad_eclip",
    "ig2",
    "igos_pp",
    "gradcam",
    "odam",
    "ssgrad_cam_pp",
    "llavacam",
    "input_grad",
)
SEARCH_ALGORITHMS = (
    "greedy",
    "phasewin",
    "drise",
    "dhsic",
)

_ALGORITHM_FAMILIES: Dict[str, str] = {
    **{name: "gradient" for name in GRADIENT_ALGORITHMS},
    **{name: "search" for name in SEARCH_ALGORITHMS},
}

_TASK_ALGORITHMS: Dict[str, Tuple[str, ...]] = {
    "classification": (
        "greedy",
        "phasewin",
        "drise",
        "dhsic",
        "gradient",
        "grad_eclip",
        "ig2",
        "igos_pp",
    ),
    "detection": (
        "greedy",
        "phasewin",
        "drise",
        "gradient",
        "gradcam",
        "odam",
        "ssgrad_cam_pp",
    ),
    "caption_vqa": (
        "greedy",
        "phasewin",
        "drise",
        "gradient",
        "llavacam",
        "igos_pp",
    ),
}


def algorithm_family(name: str) -> str:
    """Return `gradient` or `search`."""
    normalized = normalize_algorithm_name(name)
    try:
        return _ALGORITHM_FAMILIES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported algorithm: {name!r}") from exc


def task_algorithm_choices(task_name: str) -> Tuple[str, ...]:
    """
    Return runnable CLI algorithms for a given task.

    This list is intentionally narrower than the full baseline inventory.
    See ``attribution_research.baselines`` for the merged catalog, including
    native methods and catalog-only legacy entries.
    """
    normalized = normalize_task_name(task_name)
    try:
        return _TASK_ALGORITHMS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported task: {task_name!r}") from exc


def task_supports_algorithm(task_name: str, algorithm_name: str) -> bool:
    """Check whether a task supports the requested algorithm."""
    normalized_task = normalize_task_name(task_name)
    normalized_algorithm = normalize_algorithm_name(algorithm_name)
    return normalized_algorithm in task_algorithm_choices(normalized_task)


def validate_task_algorithm(task_name: str, algorithm_name: str) -> str:
    """Normalize and validate one task/algorithm pairing."""
    normalized_task = normalize_task_name(task_name)
    normalized_algorithm = normalize_algorithm_name(algorithm_name)
    if not task_supports_algorithm(normalized_task, normalized_algorithm):
        choices = ", ".join(task_algorithm_choices(normalized_task))
        raise ValueError(
            f"Algorithm {algorithm_name!r} is not supported for task "
            f"{task_name!r}. Expected one of: {choices}"
        )
    return normalized_algorithm
