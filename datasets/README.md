# Datasets Directory

Put task inputs here.

Expected layout for the current unified tasks:

```text
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
```

Notes:

- `ILSVRC2012_validation_ground_truth.txt` is the official `50000`-line val GT
  file with official class ids in the `1-1000` range.
- `imagenet1000_clsid_to_labels.txt` is the official id-to-class table used to
  map official ImageNet ids into the repo's internal `0-999` class indices.
- Canonical classification eval lists now live under `datasets/imagenet/generated/`.
- The current generated policy is:
  - fixed shuffle seed `0`
  - `5000` correct samples per model
  - `2000` incorrect samples per model
  - paired `false_gt` / `false_pred` outputs for mistake experiments

See `docs/resource_inventory.md` for field-level details.
