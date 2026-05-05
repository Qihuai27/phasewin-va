# Repository Overview

This document summarizes the current unified repository structure.

## Design Axes

The repository is organized around four decoupled axes:

1. task
2. algorithm family
3. segmentation
4. model-specific adapter

That means the main task scripts are intentionally thin. They compose reusable pieces from `attribution_research/` instead of re-implementing algorithm logic inline.

## Main Code Map

### `attribution_research/methods/search/`

- `greedy.py`
  - region-by-region search baseline
- `phasewin.py`
  - accelerated search with phase-window selection
- `drise.py`
  - random mask sampling baseline
- `dhsic.py`
  - optional `xplique` wrapper for classification
- `base.py`
  - shared selector / explainer base classes for greedy-style search

### `attribution_research/methods/gradient/`

- `gradient.py`
  - dense-map attribution replayed through unified region evaluation
- `map_based.py`
  - map-to-region ranking bridge shared by gradient-like workflows

Algorithm-specific saliency adapters now hang off the task adapters:

- classification / CLIP:
  - `gradient`
  - `grad_eclip`
  - `ig2`
  - `igos_pp`
- detection / GroundingDINO:
  - `gradient`
  - `gradcam`
  - `odam`
  - `ssgrad_cam_pp`
- caption / Qwen2.5-VL:
  - `gradient`
  - `llavacam`
  - `igos_pp`

### `attribution_research/adapters/`

- `clip.py`
  - CLIP classification adapters for search, gradient, and D-HSIC
- `grounding_dino.py`
  - detection scoring and gradient adapters
- `mllm.py`
  - generic token-scoring adapter for search algorithms
- `qwen25vl.py`
  - Qwen2.5-VL token scorer and gradient adapter

### `attribution_research/segmentation/`

- `patch.py`
  - regular grid / token-aligned segmentation
- `superpixel.py`
  - superpixel segmentation with OpenCV-first, `skimage.slic` fallback
- `base.py`
  - `RegionSet` and shared segmenter abstractions

### `attribution_research/registry.py`

- task support matrix and algorithm taxonomy

### `attribution_research/baselines.py`

- baseline inventory across the current repository tasks and catalog-only legacy methods
- explicit support status:
  - `native`: dedicated runnable implementation in this repo
  - `catalog`: tracked baseline that is not yet wired into the unified runtime

### `attribution_research/runtime.py`

- centralized method dispatch
- shared run-tag construction for task entrypoints

### `tasks/`

- `tasks/classification/clip_imagenet.py`
- `tasks/detection/groundingdino_coco.py`
- `tasks/caption_vqa/qwen25vl_coco_caption.py`

These are the three concrete runnable task entrypoints.

### `scripts/`

- canonical task runners:
  - `run_classification.sh`
  - `run_caption.sh`
  - `run_detection.sh`
- evaluation CLIs:
  - `eval_classification.py`
  - `eval_caption.py`
  - `eval_detection.py`
  - `eval_auc_faithfulness.py`
  - `eval_point_game.py`
- compatibility wrappers remain for model-fixed convenience, but the canonical
  flow should go through the three task runners above

### `scripts/list_baselines.py`

- prints the merged baseline inventory grouped by task / family / category / source / support
- makes it explicit which baselines are runnable now and which are still catalog-only

## Current Runtime Behavior

- `patch` is treated as a standard segmentation option, not a special-case path.
- gradient-style methods are evaluated by converting dense maps into ordered region masks under the chosen segmenter.
- search-style and gradient-style methods can use either `patch` or `superpixel`.
- if OpenCV lacks `ximgproc` superpixel APIs at runtime, the repo automatically falls back to `skimage.slic`.

## Local Asset Roots

- `datasets/`
- `ckpt/`
- `.checkpoints/`
- `config/`
- `model_checkpoint/`

See `docs/resource_inventory.md` for exact expected filenames and schemas.

## Output Layout

Every task writes results with the same structure:

```text
<result-root>/<run-tag>/
  npy/
    <image_id>.npy
  json/
    <image_id>.json
```

This shared layout is what allows one evaluation script to work across all three tasks.

## Current Validation Status

The current workspace has already been smoke-tested on one sample for:

- ImageNet CLIP classification
- COCO GroundingDINO detection
- COCO Qwen2.5-VL caption attribution

Matching notebook demos exist under `notebooks/`.

Those notebooks are now organized as single-sample research walkthroughs:

- configuration
- sample loading
- one-sample run command
- attribution heatmap
- insertion/deletion curves
- progressive reveal
- numerical summary
