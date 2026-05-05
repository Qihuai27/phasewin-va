# -*- coding: utf-8 -*-
"""
Unified baseline catalog for the current repository.

The runtime only executes a subset of the full research inventory. This module
keeps that distinction explicit by recording whether a method is:

- ``native``: dedicated runnable implementation in this repo; or
- ``catalog``: tracked in the inventory but not yet wired into the unified
  runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple


SUPPORTED_BASELINE_STATUSES = ("native", "catalog")
SUPPORTED_BASELINE_GROUPS = ("task", "family", "category", "source", "support")


@dataclass(frozen=True)
class BaselineSpec:
    """One canonical baseline entry in the merged research inventory."""

    name: str
    family: str
    category: str
    tasks: Tuple[str, ...]
    sources: Tuple[str, ...]
    support: str
    summary: str
    aliases: Tuple[str, ...] = ()
    optional_dependencies: Tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.support == "native"


_CATALOG: Tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="greedy",
        family="search",
        category="search",
        tasks=("classification", "detection", "caption_vqa"),
        sources=("search",),
        support="native",
        summary="Region-by-region black-box greedy search baseline.",
    ),
    BaselineSpec(
        name="phasewin",
        family="search",
        category="search",
        tasks=("classification", "detection", "caption_vqa"),
        sources=("search",),
        support="native",
        summary="Phase-window accelerated search baseline.",
    ),
    BaselineSpec(
        name="drise",
        family="search",
        category="search",
        tasks=("classification", "detection", "caption_vqa"),
        sources=("search",),
        support="native",
        summary="Random-mask sampling baseline folded into the unified runtime.",
        aliases=("d_rise",),
    ),
    BaselineSpec(
        name="dhsic",
        family="search",
        category="search",
        tasks=("classification",),
        sources=("classification",),
        support="native",
        summary="Classification-only D-HSIC baseline via xplique replay.",
        optional_dependencies=("xplique", "tensorflow"),
    ),
    BaselineSpec(
        name="gradient",
        family="gradient",
        category="input_gradient",
        tasks=("classification", "detection", "caption_vqa"),
        sources=("unified",),
        support="native",
        summary="Generic input-gradient saliency replayed through the shared region evaluator.",
        aliases=("input_grad",),
    ),
    BaselineSpec(
        name="grad_eclip",
        family="gradient",
        category="attention_or_cam",
        tasks=("classification",),
        sources=("classification",),
        support="native",
        summary=(
            "Dedicated Grad-ECLIP dense-attention saliency adapter for CLIP classification."
        ),
        aliases=("grad_eclip", "grad-eclip", "grad_eclip_clip", "gradeclip"),
    ),
    BaselineSpec(
        name="gradcam",
        family="gradient",
        category="attention_or_cam",
        tasks=("detection",),
        sources=("search",),
        support="native",
        summary=(
            "Dedicated GroundingDINO GradCAM feature-map adapter."
        ),
        aliases=("grad_cam", "grad-cam"),
    ),
    BaselineSpec(
        name="llavacam",
        family="gradient",
        category="attention_or_cam",
        tasks=("caption_vqa",),
        sources=("caption",),
        support="native",
        summary=(
            "Dedicated Qwen2.5-VL hook-based LLaVA-CAM adapter."
        ),
        aliases=("llava_cam", "llava-cam", "llava cam"),
    ),
    BaselineSpec(
        name="ig2",
        family="gradient",
        category="path_gradient",
        tasks=("classification",),
        sources=("classification",),
        support="native",
        summary="Dedicated CLIP IG^2 adapter using blur-based references and iterative gradient-path search.",
        aliases=("ig_2",),
    ),
    BaselineSpec(
        name="igos_pp",
        family="gradient",
        category="optimization",
        tasks=("classification", "caption_vqa"),
        sources=("classification", "caption"),
        support="native",
        summary="Repo-native IGOS++-style optimized mask adapter for classification and Qwen2.5-VL caption attribution.",
        aliases=("igospp", "igos++", "igos"),
    ),
    BaselineSpec(
        name="vit_cx",
        family="gradient",
        category="attention_or_cam",
        tasks=("classification",),
        sources=("classification",),
        support="catalog",
        summary="ViT-CX causal feature clustering baseline is cataloged but not integrated.",
        aliases=("vit-cx", "vit cx", "vit_cx"),
    ),
    BaselineSpec(
        name="scorecam",
        family="gradient",
        category="attention_or_cam",
        tasks=("classification",),
        sources=("classification",),
        support="catalog",
        summary="ScoreCAM generation scripts exist upstream but are not integrated here.",
        aliases=("score_cam", "score-cam"),
    ),
    BaselineSpec(
        name="lime",
        family="gradient",
        category="sampling",
        tasks=("classification",),
        sources=("classification",),
        support="catalog",
        summary="LIME map generation exists upstream but is not wired into the unified runtime.",
    ),
    BaselineSpec(
        name="samp",
        family="gradient",
        category="sampling",
        tasks=("classification",),
        sources=("classification",),
        support="catalog",
        summary="SAMP / SAMP++ notebook baseline exists upstream but is not integrated.",
        aliases=("samp++", "samp_pp"),
    ),
    BaselineSpec(
        name="odam",
        family="gradient",
        category="attention_or_cam",
        tasks=("detection",),
        sources=("search",),
        support="native",
        summary="Dedicated GroundingDINO ODAM feature-map adapter.",
    ),
    BaselineSpec(
        name="ssgrad_cam_pp",
        family="gradient",
        category="attention_or_cam",
        tasks=("detection",),
        sources=("search",),
        support="native",
        summary="Dedicated GroundingDINO SSGrad-CAM++ feature-map adapter.",
        aliases=("ssgradcampp", "ssgradcam++", "ssgrad-cam++", "ssgrad_cam++"),
    ),
    BaselineSpec(
        name="tam",
        family="gradient",
        category="token_activation",
        tasks=("caption_vqa",),
        sources=("caption",),
        support="catalog",
        summary="Token Activation Map is tracked in the catalog but is not integrated into the unified Qwen task.",
    ),
)


def _normalize_baseline_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("++", "_pp")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


_BY_NAME: Dict[str, BaselineSpec] = {}
for spec in _CATALOG:
    keys = {spec.name, *spec.aliases}
    for key in keys:
        _BY_NAME[_normalize_baseline_name(key)] = spec


def baseline_catalog() -> Tuple[BaselineSpec, ...]:
    """Return the full merged baseline catalog."""
    return _CATALOG


def baseline_spec(name: str) -> BaselineSpec:
    """Resolve one canonical baseline entry from a user or upstream label."""
    normalized = _normalize_baseline_name(name)
    try:
        return _BY_NAME[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline: {name!r}") from exc


def baseline_names(
    *,
    task: str | None = None,
    support: Iterable[str] | None = None,
    runnable_only: bool = False,
) -> Tuple[str, ...]:
    """Return canonical baseline names filtered by task / support."""
    specs = filter_baselines(task=task, support=support, runnable_only=runnable_only)
    return tuple(spec.name for spec in specs)


def filter_baselines(
    *,
    task: str | None = None,
    support: Iterable[str] | None = None,
    runnable_only: bool = False,
) -> Tuple[BaselineSpec, ...]:
    """Return baseline specs filtered by task, support status, and runnability."""
    allowed_support = tuple(support) if support is not None else SUPPORTED_BASELINE_STATUSES
    for status in allowed_support:
        if status not in SUPPORTED_BASELINE_STATUSES:
            raise ValueError(
                f"Unsupported baseline status: {status!r}. "
                f"Expected one of: {', '.join(SUPPORTED_BASELINE_STATUSES)}"
            )

    task_name = str(task).strip().lower() if task is not None else None
    filtered: List[BaselineSpec] = []
    for spec in _CATALOG:
        if task_name is not None and task_name not in spec.tasks:
            continue
        if spec.support not in allowed_support:
            continue
        if runnable_only and not spec.runnable:
            continue
        filtered.append(spec)
    return tuple(filtered)


def group_baselines(
    *,
    group_by: str = "task",
    task: str | None = None,
    support: Iterable[str] | None = None,
    runnable_only: bool = False,
) -> Dict[str, Tuple[BaselineSpec, ...]]:
    """
    Group baseline specs by task, family, category, source, or support status.
    """
    if group_by not in SUPPORTED_BASELINE_GROUPS:
        raise ValueError(
            f"Unsupported baseline group: {group_by!r}. "
            f"Expected one of: {', '.join(SUPPORTED_BASELINE_GROUPS)}"
        )

    task_name = str(task).strip().lower() if task is not None else None
    grouped: Dict[str, List[BaselineSpec]] = {}
    specs = filter_baselines(task=task, support=support, runnable_only=runnable_only)
    for spec in specs:
        if group_by == "task":
            keys = (task_name,) if task_name is not None else spec.tasks
        elif group_by == "source":
            keys = spec.sources
        elif group_by == "family":
            keys = (spec.family,)
        elif group_by == "category":
            keys = (spec.category,)
        else:
            keys = (spec.support,)

        for key in keys:
            grouped.setdefault(key, []).append(spec)

    return {
        key: tuple(sorted(values, key=lambda item: item.name))
        for key, values in sorted(grouped.items(), key=lambda item: item[0])
    }
