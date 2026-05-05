from attribution_research.adapters.base import (
    SearchAdapter,
    GreedyAdapter,
    SubmodularAdapter,
)
from attribution_research.adapters.gradient import CallableGradientAdapter, GradientAdapter
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

__all__ = [
    "SearchAdapter",
    "GreedyAdapter",
    "SubmodularAdapter",
    "GradientAdapter",
    "CallableGradientAdapter",
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
]
