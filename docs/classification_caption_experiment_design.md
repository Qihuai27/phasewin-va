# Classification + Caption Experiment Design

This document describes the public experiment design for the current
classification and caption/VQA attribution workflows.

## Goals

The repository keeps three concerns separated:

1. task entrypoints load data and build task-specific adapters;
2. reusable attribution methods live under `attribution_research/methods/`;
3. shared evaluation converts every method output into ordered region masks.

The public runtime currently exposes two algorithm families:

- `search`: `greedy`, `phasewin`, `drise`, and classification-only `dhsic`
- `gradient`: `gradient`, `grad_eclip`, `ig2`, `igos_pp`, `gradcam`, `odam`,
  `ssgrad_cam_pp`, and `llavacam`

## Tasks

### Classification

Classification entrypoints:

- `tasks/classification/clip_imagenet.py`
- `tasks/classification/torchvision_imagenet.py`

The CLIP entrypoint supports:

- `greedy`
- `phasewin`
- `drise`
- `dhsic`
- `gradient`
- `grad_eclip`
- `ig2`
- `igos_pp`

The torchvision entrypoint supports:

- `greedy`
- `phasewin`
- `drise`
- `dhsic`
- `gradient`
- `ig2`
- `igos_pp`

### Caption / VQA

Caption entrypoint:

- `tasks/caption_vqa/qwen25vl_coco_caption.py`

Supported algorithms:

- `greedy`
- `phasewin`
- `drise`
- `gradient`
- `llavacam`
- `igos_pp`

## Runtime Dispatch

All task entrypoints call `execute_attribution()` from
`attribution_research/runtime.py`.

- Search methods use `build_search_adapter`.
- Gradient-family methods use `build_gradient_adapter`, then replay the dense
  map through the same region evaluator used by search methods.
- `dhsic` uses `build_dhsic_model` plus the shared evaluator.

The task support matrix lives in `attribution_research/registry.py`. The
baseline catalog lives in `attribution_research/baselines.py`.

## Segmentation

Both public task families can use either:

- `superpixel`
- `patch`

The canonical `RegionSet` representation stores binary masks plus an optional
pixel-to-region label map. Search methods consume the masks directly.
Map-based methods aggregate pixel saliency into region scores and then replay
the resulting ordered masks for insertion/deletion evaluation.

## Evaluation

The shared result layout is:

```text
<result-root>/<run-tag>/
  npy/
    <image_id>.npy
  json/
    <image_id>.json
```

Classification and caption/VQA evaluations share insertion/deletion AUC and
highest-score summaries. Classification evaluation can also compute MuFidelity
when model-specific metadata is available.

## Standard Runners

- `scripts/run_classification.sh`
- `scripts/run_caption.sh`
- `scripts/run_detection.sh`
- `scripts/run_classification_suite.py`
- `scripts/run_caption_suite.py`

For CLIP models, semantic feature files can be generated with:

- `scripts/generate_clip_semantic_features.py`

For model-specific ImageNet splits:

- `scripts/build_imagenet_eval_lists.py`
