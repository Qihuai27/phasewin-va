# Scripts Overview

This directory contains the canonical experiment launchers, preparation
helpers, evaluation CLIs, and a small number of compatibility wrappers.

## Main Closed-Loop Scripts

- `run_classification.sh`
  - canonical classification runner
  - integrates `CLIP ViT-L/14`, `CLIP RN101`, `ResNet-50`, `ResNet-101`, and mistake splits
  - includes `igos_pp` in the repo-native classification algorithm set
- `run_caption.sh`
  - canonical caption runner
  - integrates `Qwen2.5-VL-3B` and `Qwen2.5-VL-7B`
  - includes `igos_pp` in the repo-native caption algorithm set
  - `Qwen2.5-VL-7B` now defaults to `--max-image-side 256`
    for the gradient-family methods so `gradient / llavacam / igos_pp`
    fit on a ~31 GB GPU
  - the backward-memory optimization path is only enabled for `Qwen2.5-VL-7B`;
    `Qwen2.5-VL-3B` keeps the original gradient runtime
  - when rebuilding a `7B` caption eval list, use the same image-side cap
    so generated token metadata stays aligned with runtime preprocessing
- `run_detection.sh`
  - base detection main table

## Preparation / Orchestration

- `run_round1_extension.sh`
  - convenience batch launcher over the canonical runners
- `generate_clip_semantic_features.py`
  - generate missing CLIP semantic features, mainly for `RN101`
- `build_imagenet_eval_lists.py`
  - derive model-specific `true / false_gt / false_pred` lists
  - supports canonical rebuilds from official ImageNet val GT + official id table
- `build_caption_eval_list.py`
  - rebuild caption eval metadata for a new Qwen model
  - accepts `--max-image-side` to keep eval-list token metadata aligned with
    any runtime image downscaling used by `run_caption.sh`

## Compatibility Wrappers

- `run_classification_clip_rn101.sh`
  - forwards to `run_classification.sh --model clip_rn101`
- `run_classification_clip_vitl_mistake.sh`
  - forwards to `run_classification.sh --model clip_vitl14 --split both`
- `run_classification_clip_rn101_mistake.sh`
  - forwards to `run_classification.sh --model clip_rn101 --split both`
- `run_classification_resnet101.sh`
  - forwards to `run_classification.sh --model resnet101`
- `run_classification_resnet50.sh`
  - forwards to `run_classification.sh --model resnet50`
- `run_classification_mistake.sh`
  - translates the older `--mode` interface into the canonical classification runner
- `run_caption_qwen7b.sh`
  - forwards to `run_caption.sh --model-size 7b`

## Evaluation

- `eval_auc_faithfulness.py`
- `eval_point_game.py`
- `eval_classification.py`
  - unified classification summary for AUC metrics
  - optional `--mufidelity` pass for `xplique`-based `MuFidelity`
  - MuFidelity defaults to `--mu-batch-size 8`; raise it only when there is enough GPU headroom
- `eval_caption.py`
- `eval_detection.py`

All runners are designed to stay repo-native:

- they call the task entrypoints under `tasks/`
- they keep outputs under this repo's result directories
- they reuse the unified replay evaluation scripts
