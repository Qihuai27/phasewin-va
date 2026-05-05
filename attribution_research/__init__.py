# -*- coding: utf-8 -*-
"""
Attribution research package for unified task, method, and evaluation studies.
"""

from attribution_research.methods.gradient import GradientExplainer, SaliencyMapExplainer
from attribution_research.methods.search import (
    DHSICExplainer,
    DRISEExplainer,
    GreedyExplainer,
    NaiveGreedySelector,
    PhaseWindowSelector,
    PhaseWinExplainer,
    SubmodularExplainer,
)
from attribution_research.adapters.base import (
    GreedyAdapter,
    SearchAdapter,
    SubmodularAdapter,
)
from attribution_research.adapters.gradient import CallableGradientAdapter, GradientAdapter
from attribution_research.baselines import (
    BaselineSpec,
    baseline_catalog,
    baseline_names,
    baseline_spec,
    filter_baselines,
    group_baselines,
)
from attribution_research.adapters.clip import (
    CLIPGradEClipAdapter,
    CLIPGradientAdapter,
    CLIPIG2Adapter,
    CLIPIGOSPPAdapter,
    CLIPSearchAdapter,
    CLIPXpliqueWrapper,
)
from attribution_research.adapters.grounding_dino import (
    GroundingDINOAdapter,
    GroundingDINOCAMAdapter,
    GroundingDINODetector,
    GroundingDINOGradientAdapter,
)
from attribution_research.adapters.mllm import MLLMAdapter
from attribution_research.adapters.qwen25vl import (
    Qwen25VLGradientAdapter,
    Qwen25VLIGOSPPAdapter,
    Qwen25VLLLaVACAMAdapter,
    Qwen25VLTokenScorer,
    load_qwen25vl_model,
)
from attribution_research.adapters.torchvision_imagenet import (
    SUPPORTED_TORCHVISION_ARCHES,
    TorchvisionImageNetGradientAdapter,
    TorchvisionImageNetIG2Adapter,
    TorchvisionImageNetIGOSPPAdapter,
    TorchvisionImageNetSearchAdapter,
    TorchvisionImageNetXpliqueWrapper,
    load_torchvision_imagenet_model,
    resolve_torchvision_weights,
)
from attribution_research.composition import (
    algorithm_family,
    build_segmenter,
    build_segmenter_from_args,
    normalize_algorithm_name,
    segmenter_tag,
    segmenter_tag_from_args,
)
from attribution_research.data.prompts import COCO_TEXT_PROMPT
from attribution_research.registry import (
    GRADIENT_ALGORITHMS,
    SEARCH_ALGORITHMS,
    task_algorithm_choices,
    task_supports_algorithm,
    validate_task_algorithm,
)
from attribution_research.runtime import (
    AttributionContext,
    build_run_tag,
    build_save_dir,
    execute_attribution,
)
from attribution_research.segmentation.base import RegionSet
from attribution_research.segmentation.patch import PatchSegmenter
from attribution_research.segmentation.superpixel import Segmentor, SubRegionDivision

__version__ = "1.1.0"

__all__ = [
    "DHSICExplainer",
    "DRISEExplainer",
    "GradientExplainer",
    "GreedyExplainer",
    "NaiveGreedySelector",
    "PhaseWindowSelector",
    "PhaseWinExplainer",
    "SaliencyMapExplainer",
    "SubmodularExplainer",
    "GreedyAdapter",
    "SearchAdapter",
    "SubmodularAdapter",
    "CallableGradientAdapter",
    "GradientAdapter",
    "BaselineSpec",
    "baseline_catalog",
    "baseline_names",
    "baseline_spec",
    "filter_baselines",
    "group_baselines",
    "CLIPGradEClipAdapter",
    "CLIPGradientAdapter",
    "CLIPIG2Adapter",
    "CLIPIGOSPPAdapter",
    "CLIPSearchAdapter",
    "CLIPXpliqueWrapper",
    "GroundingDINOAdapter",
    "GroundingDINOCAMAdapter",
    "GroundingDINODetector",
    "GroundingDINOGradientAdapter",
    "MLLMAdapter",
    "Qwen25VLGradientAdapter",
    "Qwen25VLIGOSPPAdapter",
    "Qwen25VLLLaVACAMAdapter",
    "Qwen25VLTokenScorer",
    "load_qwen25vl_model",
    "SUPPORTED_TORCHVISION_ARCHES",
    "TorchvisionImageNetGradientAdapter",
    "TorchvisionImageNetIG2Adapter",
    "TorchvisionImageNetIGOSPPAdapter",
    "TorchvisionImageNetSearchAdapter",
    "TorchvisionImageNetXpliqueWrapper",
    "load_torchvision_imagenet_model",
    "resolve_torchvision_weights",
    "algorithm_family",
    "build_segmenter",
    "build_segmenter_from_args",
    "normalize_algorithm_name",
    "segmenter_tag",
    "segmenter_tag_from_args",
    "COCO_TEXT_PROMPT",
    "GRADIENT_ALGORITHMS",
    "SEARCH_ALGORITHMS",
    "task_algorithm_choices",
    "task_supports_algorithm",
    "validate_task_algorithm",
    "AttributionContext",
    "build_run_tag",
    "build_save_dir",
    "execute_attribution",
    "RegionSet",
    "PatchSegmenter",
    "Segmentor",
    "SubRegionDivision",
]
