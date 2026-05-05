# Resource Inventory

This document collects the local assets expected by the unified repository:

1. datasets
2. model weights and configs
3. recommended storage locations
4. output directories
5. minimal input schema reminders

## Recommended Local Layout

```text
repo-root/
  datasets/
    imagenet/
      ILSVRC2012_img_val/
      ILSVRC2012_validation_ground_truth.txt
      imagenet1000_clsid_to_labels.txt
      generated/
        clip_vitl14_true.txt
        clip_vitl14_false_gt.txt
        clip_vitl14_false_pred.txt
        clip_rn101_true.txt
        clip_rn101_false_gt.txt
        clip_rn101_false_pred.txt
        resnet50_true.txt
        resnet50_false_gt.txt
        resnet50_false_pred.txt
        resnet101_true.txt
        resnet101_false_gt.txt
        resnet101_false_pred.txt
    coco/
      val2017/
    coco_groundingdino_correct_detection.json
    Qwen2.5-VL-3B-coco-caption.json
  ckpt/
    semantic_features/
      clip_vitl_imagenet_zeroweights.pt
    groundingdino_swint_ogc.pth
  config/
    GroundingDINO_SwinT_OGC.py
  model_checkpoint/
    Qwen2.5-VL-3B-Instruct/
  classification_results/
  detection_results/
  caption_results/
```

## Task 1: ImageNet Classification + CLIP

- Entry point: `tasks/classification/clip_imagenet.py`
- Required dataset root: `datasets/imagenet/ILSVRC2012_img_val`
- Official GT file: `datasets/imagenet/ILSVRC2012_validation_ground_truth.txt`
- Official id table: `datasets/imagenet/imagenet1000_clsid_to_labels.txt`
- Canonical CLIP ViT-L/14 true list: `datasets/imagenet/generated/clip_vitl14_true.txt`
- Canonical CLIP RN101 true list: `datasets/imagenet/generated/clip_rn101_true.txt`
- Canonical ResNet-50 true list: `datasets/imagenet/generated/resnet50_true.txt`
- Canonical ResNet-101 true list: `datasets/imagenet/generated/resnet101_true.txt`
- Canonical mistake lists:
  - `datasets/imagenet/generated/*_false_gt.txt`
  - `datasets/imagenet/generated/*_false_pred.txt`
- Required semantic features: `ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt`
- Supplementary semantic features:
  - `ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt`  (generate with `scripts/generate_clip_semantic_features.py --model RN101`)
- Default result root: `classification_results/imagenet-clip-vitl`

### Notes

- The eval list format is one line per sample:
- The current canonical classification lists are generated from the official
  `50000`-image val set with:
  - shuffle seed `0`
  - `5000` correct samples per model
  - `2000` incorrect samples per model
- `scripts/build_imagenet_eval_lists.py` also emits
  `*_official_to_internal_map.json` to make the official-id to internal-index
  mapping explicit.

```text
ILSVRC2012_val_00000293.JPEG 0
ILSVRC2012_val_00002138.JPEG 0
```

- The CLIP backbone itself is loaded through `openai-clip`.
- The task script exposes `--clip-download-root`, so you can keep the CLIP backbone snapshot under `.checkpoints/CLIP/`.
- The repo-local file you do need to manage explicitly is the semantic feature matrix at `ckpt/semantic_features/...`.

## Task 2: COCO Detection + GroundingDINO

- Entry point: `tasks/detection/groundingdino_coco.py`
- Required image root: `datasets/coco/val2017`
- Required eval list: `datasets/coco_groundingdino_correct_detection.json`
- Required config: `config/GroundingDINO_SwinT_OGC.py`
- Required weights: `ckpt/groundingdino_swint_ogc.pth`
- Default result root: `detection_results/coco-groundingdino`

### Notes

- The detection eval list is a JSON list. Minimal schema:

```json
[
  {
    "image_path": "000000000139.jpg",
    "class_id": [15],
    "bbox": [54, 133, 420, 355],
    "image_id": "139",
    "caption": "person . bicycle . car . ..."
  }
]
```

- `class_id` here is not a plain COCO category id. It must be the GroundingDINO token index or token-index list corresponding to the class phrase under the chosen `caption`.
- If `caption` is omitted per sample, the script falls back to `attribution_research.data.prompts.COCO_TEXT_PROMPT`.
- If OpenCV superpixel operators are unavailable in the active environment, the task falls back to `skimage.slic`.

## Task 3: COCO Caption / VQA + Qwen2.5-VL

- Entry point: `tasks/caption_vqa/qwen25vl_coco_caption.py`
- Required image root: `datasets/coco/val2017`
- Required eval list: `datasets/Qwen2.5-VL-3B-coco-caption.json`
- Supplementary eval list target: `datasets/Qwen2.5-VL-7B-coco-caption.json`
- Default model id: `Qwen/Qwen2.5-VL-3B-Instruct`
- Recommended local snapshot path: `model_checkpoint/Qwen2.5-VL-3B-Instruct`
- Supplementary local snapshot path: `model_checkpoint/Qwen2.5-VL-7B-Instruct`
- Default result root: `caption_results/Qwen2.5-VL-3B-coco-caption`

### Notes

- The caption eval list is a JSON list. Minimal schema:

```json
[
  {
    "image_path": "000000000139.jpg",
    "generated_ids": [151644, 894, 374, 264, 5679],
    "selected_interpretation_token_id": [3],
    "selected_interpretation_token_word_id": [5679],
    "words": ["dog"]
  }
]
```

- You can keep using the Hugging Face model id, but for reproducibility inside this repo it is cleaner to store a local snapshot under `model_checkpoint/` and pass that path through `--model-name`.
- If you do not pass a local path, the model will be downloaded into the Hugging Face cache outside the repository.
- The unified scorer now aligns target tokens against the saved `generated_ids` directly, so it no longer depends on `qwen_vl_utils` for the main task path.

## Optional Assets

- `dhsic` for classification additionally needs `xplique` and `tensorflow`.
- GroundingDINO additionally needs the external Python package installation referenced in `requirements.txt`.

## Shared Result Layout

All three task scripts save results with the same layout:

```text
<task-result-root>/<algorithm-and-segmenter-tag>/
  npy/
    <image_id>.npy
  json/
    <image_id>.json
```

Examples:

- `classification_results/imagenet-clip-vitl/phasewin-slico-division-50-1.0-1.0-window-pct-30/`
- `detection_results/coco-groundingdino/phasewin-slico-division-100-1.0-1.0-window-32/`
- `caption_results/Qwen2.5-VL-3B-coco-caption/greedy-slico-division-64-1.0-1.0/`

## Current Workspace Status

The current workspace is already populated with the main local assets:

- ImageNet validation subset and CLIP semantic feature files
- COCO `val2017`, GroundingDINO config, and GroundingDINO checkpoint
- local `Qwen2.5-VL-3B-Instruct` snapshot
- curated eval JSON / TXT lists for the three main tasks

One-sample smoke-test outputs have also been generated under:

- `classification_results/imagenet-clip-vitl/gradient-patch-size-16-1.0-1.0/`
- `detection_results/coco-groundingdino/phasewin-slico-division-100-1.0-1.0-window-32/`
- `caption_results/Qwen2.5-VL-3B-coco-caption/greedy-slico-division-64-1.0-1.0/`
