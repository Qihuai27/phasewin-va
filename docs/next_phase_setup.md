# Next Phase Setup

This page compresses the current repository state into the minimum context needed
for the next work stage.

## Completed

- canonical runners are unified under:
  - `scripts/run_classification.sh`
  - `scripts/run_caption.sh`
  - `scripts/run_detection.sh`
- supplementary classification models are integrated:
  - `clip_vitl14`
  - `clip_rn101`
  - `resnet50`
  - `resnet101`
- search-family forward counts are standardized:
  - `model_forward_calls` = search-stage only
  - `eval_model_forward_calls` = replay / eval tail
  - `total_model_forward_calls` = end-to-end total
- `phasewin` defaults are calibrated to the current repo:
  - `window_frac=0.3`
  - `n_greedy=0` except Florence preset
  - `beta_del=0.05`
  - `alpha_sel=0.6`
  - `random_frac=0.0`
  - `window_policy=BA`
  - `hard_phi_prev=0.98`
  - `hard_delta_thresh=0.02`
- `Qwen2.5-VL-7B` gradient-family validation is now closed:
  - `gradient`, `llavacam`, and `igos_pp` fit on the current ~31 GB GPU
  - only the `7B` runtime path freezes model parameters for backward and only
    keeps gradients on the input / hook / mask path
  - `Qwen2.5-VL-3B` keeps the original gradient runtime for comparability
  - canonical `7B` caption runs now use `--max-image-side 256`
  - any rebuilt `7B` eval list must use the same image-side cap to keep token
    metadata aligned with runtime preprocessing

## Canonical ImageNet Assets

- official GT: `datasets/imagenet/ILSVRC2012_validation_ground_truth.txt`
- official id table: `datasets/imagenet/imagenet1000_clsid_to_labels.txt`
- canonical generated splits:
  - `datasets/imagenet/generated/clip_vitl14_true.txt`
  - `datasets/imagenet/generated/clip_vitl14_false_gt.txt`
  - `datasets/imagenet/generated/clip_vitl14_false_pred.txt`
  - `datasets/imagenet/generated/clip_rn101_true.txt`
  - `datasets/imagenet/generated/clip_rn101_false_gt.txt`
  - `datasets/imagenet/generated/clip_rn101_false_pred.txt`
  - `datasets/imagenet/generated/resnet50_true.txt`
  - `datasets/imagenet/generated/resnet50_false_gt.txt`
  - `datasets/imagenet/generated/resnet50_false_pred.txt`
  - `datasets/imagenet/generated/resnet101_true.txt`
  - `datasets/imagenet/generated/resnet101_false_gt.txt`
  - `datasets/imagenet/generated/resnet101_false_pred.txt`

Generation policy:

- fixed shuffle seed: `0`
- per-model quotas:
  - `5000` correct
  - `2000` incorrect

Observed processed counts:

- `clip_vitl14`: `7904`
- `clip_rn101`: `8864`
- `resnet50`: `9344`
- `resnet101`: `9856`

## Immediate Next Stage

Recommended focus: `phasewin` implementation validation, not more repo plumbing.

Suggested order:

1. classification small-slice validation
2. caption small-slice validation
3. detection spot check if classification/caption stay stable
4. only then decide whether any `phasewin` logic changes are needed

## Suggested Commands

Classification:

```bash
bash scripts/run_classification.sh --model clip_vitl14 --algorithms greedy,phasewin --begin 0 --end 50
bash scripts/run_classification.sh --model clip_rn101 --algorithms greedy,phasewin --begin 0 --end 50
```

Caption:

```bash
bash scripts/run_caption.sh --algorithms greedy,phasewin --begin 0 --end 20
bash scripts/run_caption.sh --model-size 7b --algorithms greedy,phasewin,drise,gradient,llavacam,igos_pp --begin 0 --end 20 --build-eval-list-if-missing
```

Evaluation:

```bash
python scripts/eval_classification.py --results-dir classification_results/imagenet-clip-vitl
python scripts/eval_auc_faithfulness.py --results-dir classification_results/imagenet-clip-vitl
```
