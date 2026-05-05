# Attribution Research

This repository organizes attribution experiments around one composable layout.

The design goal is to decouple four axes:

1. `task`
2. `algorithm family`
3. `segmentation`
4. `model-specific adapter`

## Structure

- `attribution_research/methods/gradient/`
  - `gradient.py`: unified gradient/map-based workflow
  - `map_based.py`: dense-map to ordered-region replay
- `attribution_research/methods/search/`
  - `greedy.py`, `phasewin.py`, `drise.py`
  - `dhsic.py`: optional xplique-based D-HSIC wrapper
  - `base.py`: shared selector / explainer base classes
- `attribution_research/adapters/`
  - `clip.py`: CLIP search, gradient, D-HSIC wrapper
  - `grounding_dino.py`: GroundingDINO search + gradient adapters
  - `mllm.py`: generic search adapter for multimodal token scorers
  - `qwen25vl.py`: Qwen2.5-VL scorer + gradient adapter
- `attribution_research/segmentation/`
  - `superpixel.py`
  - `patch.py`
  - `base.py`
- `attribution_research/composition.py`
  - common helpers for algorithm family detection and segmenter construction
- `attribution_research/registry.py`
  - task support matrix and algorithm taxonomy
- `attribution_research/runtime.py`
  - centralized method dispatch and run-tag construction shared by all tasks

## Top-Level Layout

- `tasks/`
  - task entrypoints for classification, detection, caption/VQA
- `attribution_research/`
  - reusable methods, adapters, segmentation, evaluation, visualization
- `datasets/`
  - repo-local evaluation images and curated eval lists
- `ckpt/`, `.checkpoints/`, `config/`, `model_checkpoint/`
  - task-specific checkpoints, CLIP cache, configs, local VLM snapshots
- `classification_results/`, `detection_results/`, `caption_results/`
  - unified output roots with `npy/` and `json/` subfolders
- `notebooks/`
  - richer single-sample research notebooks for the three main tasks
- `docs/`
  - install guide, resource inventory, repository overview, and experiment planning notes
- `scripts/generate_demo_notebooks.py`
  - regenerates the three demo notebooks from a checked-in template script

## Supported Algorithm Families

- Gradient family
  - `gradient`, `grad_eclip`, `ig2`, `igos_pp`, `gradcam`, `odam`, `ssgrad_cam_pp`, `llavacam`
  - can use `patch` or `superpixel`
- Search family
  - `greedy`
  - `phasewin`
  - `drise`
  - `dhsic` for classification via optional `xplique`
  - can use `patch` or `superpixel`

## Task Entry Points

- `tasks/classification/clip_imagenet.py`
  - CLIP classification
  - supports `greedy`, `phasewin`, `drise`, `dhsic`, `gradient`, `grad_eclip`, `ig2`, `igos_pp`
- `tasks/classification/torchvision_imagenet.py`
  - torchvision ImageNet classification (`resnet50` / `resnet101`)
  - supports `greedy`, `phasewin`, `drise`, `dhsic`, `gradient`, `ig2`, `igos_pp`
- `tasks/detection/groundingdino_coco.py`
  - GroundingDINO detection
  - supports `greedy`, `phasewin`, `drise`, `gradient`, `gradcam`, `odam`, `ssgrad_cam_pp`
  - requires a caption prompt; defaults to COCO classes prompt
- `tasks/caption_vqa/qwen25vl_coco_caption.py`
  - Qwen2.5-VL caption/token attribution
  - supports `greedy`, `phasewin`, `drise`, `gradient`, `llavacam`, `igos_pp`

## Setup Helpers

- Resource inventory: `docs/resource_inventory.md`
- Repository overview: `docs/repo_overview.md`
- Worklog: `docs/worklog/`
- Scripts overview: `scripts/README.md`
- Install guide for `lima2`: `docs/install_lima2.md`
- Classification / caption current design: `docs/classification_caption_experiment_design.md`
- Classification / caption extension re-evaluation (phasewin-first): `docs/classification_caption_extension_plan.md`
- Canonical run entrypoints:
  - `scripts/run_classification.sh`
  - `scripts/run_caption.sh`
  - `scripts/run_detection.sh`
- Preparation / orchestration helpers:
  - `scripts/generate_clip_semantic_features.py`
  - `scripts/build_imagenet_eval_lists.py`
  - `scripts/build_caption_eval_list.py`
  - `scripts/run_round1_extension.sh`
- Compatibility wrappers:
  - `scripts/run_classification_clip_rn101.sh`
  - `scripts/run_classification_clip_vitl_mistake.sh`
  - `scripts/run_classification_clip_rn101_mistake.sh`
  - `scripts/run_classification_resnet50.sh`
  - `scripts/run_classification_resnet101.sh`
  - `scripts/run_classification_mistake.sh`
  - `scripts/run_caption_qwen7b.sh`
- Evaluation wrappers:
  - `scripts/eval_auc_faithfulness.py`
  - `scripts/eval_point_game.py`
- Main runtime requirements: `requirements.txt`
- Optional D-HSIC extras: install `tensorflow` and `xplique` into the active environment
- Example notebooks:
  - `notebooks/classification_imagenet_clip_demo.ipynb`
  - `notebooks/detection_coco_groundingdino_demo.ipynb`
  - `notebooks/caption_qwen25vl_demo.ipynb`

## Notes

- Public package naming is now `attribution_research`.
- Method families are now aligned with the research structure: `gradient`, `search`.
- The runnable CLI support matrix lives in `attribution_research/registry.py`; the baseline inventory, including native and catalog-only methods, lives in `attribution_research/baselines.py` and `scripts/list_baselines.py`.
- `patch` is treated as a standard segmentation choice. Its special role is that, for patch-based vision encoders, it is naturally aligned with visual tokens.
- Gradient methods are evaluated by converting saliency maps to ordered region masks under the chosen segmenter, then replaying those masks through the corresponding black-box evaluator adapter.
- `dhsic` is intentionally optional because the original baseline depends on `xplique`.
- For `dhsic`, the classification task now defaults to `--dhsic-tf-device cpu`, which keeps TensorFlow off the GPU while allowing the wrapped CLIP model to use CUDA. Use `auto` or `gpu` only if you specifically want TensorFlow to see the GPU as well.
- If OpenCV does not expose `cv2.ximgproc` superpixel operators at runtime, the repository falls back to `skimage.slic` automatically.
- The provided local workspace has already been smoke-tested on one sample for classification, detection, and caption notebook flows.
