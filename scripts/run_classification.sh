#!/usr/bin/env bash
# Run ImageNet classification attribution baselines through one canonical entrypoint.
#
# Supported model keys:
#   - clip_vitl14
#   - clip_rn101
#   - resnet50
#   - resnet101
#
# Supported splits:
#   - true
#   - cause
#   - repair
#   - both
#
# Examples:
#   bash scripts/run_classification.sh
#   bash scripts/run_classification.sh --model clip_rn101
#   bash scripts/run_classification.sh --model clip_vitl14 --split both --build-splits-if-missing
#   bash scripts/run_classification.sh --model resnet50 --split repair --build-splits-if-missing
#
# Options:
#   --model NAME         clip_vitl14 | clip_rn101 | resnet50 | resnet101  (default: clip_vitl14)
#   --split NAME         true | cause | repair | both          (default: true)
#   --mode NAME          Alias for --split
#   --algorithms LIST    Comma-separated algorithm override
#   --eval-list PATH     Override eval list for split=true
#   --source-list PATH   Add one labeled source pool for building model-specific splits
#   --source-lists CSV   Comma-separated labeled source pools (alternative to repeated --source-list)
#   --generated-dir DIR  Output dir for generated false_gt / false_pred lists
#   --build-splits-if-missing  Build model-specific mistake splits automatically
#   --build-if-missing   Alias for --build-splits-if-missing
#   --semantic PATH      CLIP semantic feature path for CLIP models
#   --clip-download-root PATH  CLIP cache directory (default: .checkpoints/CLIP)
#   --build-semantic-if-missing  Generate missing CLIP semantic features automatically
#   --weights NAME       torchvision weights enum for resnet50/resnet101 (default: DEFAULT)
#   --datasets PATH      ImageNet image root (default: datasets/imagenet/ILSVRC2012_img_val)
#   --save-root DIR      Override result root
#   --igos-mask-size N   IGOS++ low-resolution mask size (default: 28)
#   --igos-steps N       IGOS++ optimization steps (default: 24)
#   --igos-lr LR         IGOS++ optimizer learning rate (default: 0.1)
#   --igos-blur-sigma S  IGOS++ Gaussian blur sigma (default: 15.0)
#   --begin N            First sample index, inclusive
#   --end N              Last sample index, exclusive
#   --device D           Torch device (default: cuda)
#   --python BIN         Python executable (default: $PYTHON_BIN or python)
#   --dry-run            Print commands without executing
#   -h, --help           Show this help

set -euo pipefail

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }

run_or_print() {
  printf '  '; printf '%q ' "$@"; printf '\n'
  (( DRY_RUN )) || "$@"
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CLIP_TASK="$REPO_ROOT/tasks/classification/clip_imagenet.py"
TV_TASK="$REPO_ROOT/tasks/classification/torchvision_imagenet.py"
BUILD_SPLITS_SCRIPT="$REPO_ROOT/scripts/build_imagenet_eval_lists.py"
BUILD_SEMANTIC_SCRIPT="$REPO_ROOT/scripts/generate_clip_semantic_features.py"
OFFICIAL_GT_FILE="datasets/imagenet/ILSVRC2012_validation_ground_truth.txt"
OFFICIAL_ID2CLASS_FILE="datasets/imagenet/imagenet1000_clsid_to_labels.txt"
SHUFFLE_SEED="0"
TRUE_QUOTA="5000"
FALSE_QUOTA="2000"

MODEL="clip_vitl14"
SPLIT="true"
ALGORITHMS_CSV=""
EVAL_LIST=""
SOURCE_LISTS=()
GENERATED_DIR="datasets/imagenet/generated"
BUILD_SPLITS_IF_MISSING=0
SEMANTIC_FEATURES=""
CLIP_DOWNLOAD_ROOT=".checkpoints/CLIP"
BUILD_SEMANTIC_IF_MISSING=0
WEIGHTS="DEFAULT"
DATASETS="datasets/imagenet/ILSVRC2012_img_val"
SAVE_ROOT=""
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
BEGIN=""
END=""
DRY_RUN=0
IGOS_MASK_SIZE=28
IGOS_STEPS=24
IGOS_LR=0.1
IGOS_BLUR_SIGMA=15.0
IGOS_PRESERVE_COEFF=2.0
IGOS_DELETE_COEFF=1.0
IGOS_AREA_COEFF=0.01
IGOS_TV_COEFF=0.2
IGOS_BINARY_COEFF=0.01

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --split|--mode) SPLIT="$2"; shift 2 ;;
    --algorithms) ALGORITHMS_CSV="$2"; shift 2 ;;
    --eval-list) EVAL_LIST="$2"; shift 2 ;;
    --source-list) SOURCE_LISTS+=("$2"); shift 2 ;;
    --source-lists)
      IFS=',' read -r -a _EXTRA_SOURCE_LISTS <<< "$2"
      for _src in "${_EXTRA_SOURCE_LISTS[@]}"; do
        [[ -n "$_src" ]] && SOURCE_LISTS+=("$_src")
      done
      shift 2
      ;;
    --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
    --build-splits-if-missing|--build-if-missing) BUILD_SPLITS_IF_MISSING=1; shift ;;
    --semantic) SEMANTIC_FEATURES="$2"; shift 2 ;;
    --clip-download-root) CLIP_DOWNLOAD_ROOT="$2"; shift 2 ;;
    --build-semantic-if-missing) BUILD_SEMANTIC_IF_MISSING=1; shift ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --begin) BEGIN="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --igos-mask-size) IGOS_MASK_SIZE="$2"; shift 2 ;;
    --igos-steps) IGOS_STEPS="$2"; shift 2 ;;
    --igos-lr) IGOS_LR="$2"; shift 2 ;;
    --igos-blur-sigma) IGOS_BLUR_SIGMA="$2"; shift 2 ;;
    --igos-preserve-coeff) IGOS_PRESERVE_COEFF="$2"; shift 2 ;;
    --igos-delete-coeff) IGOS_DELETE_COEFF="$2"; shift 2 ;;
    --igos-area-coeff) IGOS_AREA_COEFF="$2"; shift 2 ;;
    --igos-tv-coeff) IGOS_TV_COEFF="$2"; shift 2 ;;
    --igos-binary-coeff) IGOS_BINARY_COEFF="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$SPLIT" in
  true|cause|repair|both) ;;
  *) echo "Unsupported --split: $SPLIT" >&2; exit 1 ;;
esac

MODEL_LABEL=""
TASK_SCRIPT=""
RESULT_ROOT=""
TRUE_LIST=""
GENERATE_PREFIX=""
DEFAULT_SEMANTIC=""
DEFAULT_ALGORITHMS_CSV=""
CLIP_TYPE=""
ARCH=""

case "$MODEL" in
  clip_vitl14)
    MODEL_LABEL="CLIP ViT-L/14"
    TASK_SCRIPT="$CLIP_TASK"
    RESULT_ROOT="./classification_results/imagenet-clip-vitl"
    TRUE_LIST="datasets/imagenet/generated/clip_vitl14_true.txt"
    GENERATE_PREFIX="clip_vitl14"
    DEFAULT_SEMANTIC="ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
    DEFAULT_ALGORITHMS_CSV="greedy,phasewin,drise,dhsic,gradient,grad_eclip,ig2,igos_pp"
    CLIP_TYPE="ViT-L/14"
    ;;
  clip_rn101)
    MODEL_LABEL="CLIP RN101"
    TASK_SCRIPT="$CLIP_TASK"
    RESULT_ROOT="./classification_results/imagenet-clip-rn101"
    TRUE_LIST="datasets/imagenet/generated/clip_rn101_true.txt"
    GENERATE_PREFIX="clip_rn101"
    DEFAULT_SEMANTIC="ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt"
    DEFAULT_ALGORITHMS_CSV="greedy,phasewin,drise,dhsic,gradient,ig2,igos_pp"
    CLIP_TYPE="RN101"
    ;;
  resnet50)
    MODEL_LABEL="torchvision ResNet-50"
    TASK_SCRIPT="$TV_TASK"
    RESULT_ROOT="./classification_results/imagenet-resnet50"
    TRUE_LIST="datasets/imagenet/generated/resnet50_true.txt"
    GENERATE_PREFIX="resnet50"
    DEFAULT_ALGORITHMS_CSV="greedy,phasewin,drise,dhsic,gradient,ig2,igos_pp"
    ARCH="resnet50"
    ;;
  resnet101)
    MODEL_LABEL="torchvision ResNet-101"
    TASK_SCRIPT="$TV_TASK"
    RESULT_ROOT="./classification_results/imagenet-resnet101"
    TRUE_LIST="datasets/imagenet/generated/resnet101_true.txt"
    GENERATE_PREFIX="resnet101"
    DEFAULT_ALGORITHMS_CSV="greedy,phasewin,drise,dhsic,gradient,ig2,igos_pp"
    ARCH="resnet101"
    ;;
  *)
    echo "Unsupported --model: $MODEL" >&2
    exit 1
    ;;
esac

if [[ -z "$SAVE_ROOT" ]]; then
  SAVE_ROOT="$RESULT_ROOT"
fi
if [[ -z "$EVAL_LIST" ]]; then
  EVAL_LIST="$TRUE_LIST"
fi
if [[ -z "$SEMANTIC_FEATURES" ]]; then
  SEMANTIC_FEATURES="$DEFAULT_SEMANTIC"
fi
if [[ -z "$ALGORITHMS_CSV" ]]; then
  ALGORITHMS_CSV="$DEFAULT_ALGORITHMS_CSV"
fi
IFS=',' read -r -a ALGORITHMS <<< "$ALGORITHMS_CSV"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TASK_SCRIPT" ]]; then
  echo "Task script not found: $TASK_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$BUILD_SPLITS_SCRIPT" ]]; then
  echo "Split-builder script not found: $BUILD_SPLITS_SCRIPT" >&2
  exit 1
fi

algorithm_supported() {
  local model_name="$1"
  local algo="$2"
  case "$model_name" in
    clip_vitl14)
      case "$algo" in
        greedy|phasewin|drise|dhsic|gradient|grad_eclip|ig2|igos_pp) return 0 ;;
      esac
      ;;
    clip_rn101)
      case "$algo" in
        greedy|phasewin|drise|dhsic|gradient|ig2|igos_pp) return 0 ;;
      esac
      ;;
    resnet50)
      case "$algo" in
        greedy|phasewin|drise|dhsic|gradient|ig2|igos_pp) return 0 ;;
      esac
      ;;
    resnet101)
      case "$algo" in
        greedy|phasewin|drise|dhsic|gradient|ig2|igos_pp) return 0 ;;
      esac
      ;;
  esac
  return 1
}

for algo in "${ALGORITHMS[@]}"; do
  if ! algorithm_supported "$MODEL" "$algo"; then
    echo "Algorithm '$algo' is not supported for model '$MODEL'." >&2
    exit 1
  fi
done

repo_path_exists() {
  local path="$1"
  if [[ -z "$path" ]]; then
    return 1
  fi
  if [[ "$path" = /* ]]; then
    [[ -e "$path" ]]
  else
    [[ -e "$REPO_ROOT/$path" ]]
  fi
}

if [[ "$MODEL" == clip_* ]] && ! repo_path_exists "$SEMANTIC_FEATURES" && (( BUILD_SEMANTIC_IF_MISSING )); then
  if [[ ! -f "$BUILD_SEMANTIC_SCRIPT" ]]; then
    echo "Semantic-builder script not found: $BUILD_SEMANTIC_SCRIPT" >&2
    exit 1
  fi
  BUILD_SEMANTIC_CMD=(
    "$PYTHON_BIN" "$BUILD_SEMANTIC_SCRIPT"
    --model         "$CLIP_TYPE"
    --download-root "$CLIP_DOWNLOAD_ROOT"
    --out           "$SEMANTIC_FEATURES"
    --device        "$DEVICE"
  )
  echo "Preparing missing semantic features"
  run_or_print "${BUILD_SEMANTIC_CMD[@]}"
fi

if [[ "$MODEL" == clip_* ]] && ! repo_path_exists "$SEMANTIC_FEATURES"; then
  if (( DRY_RUN )); then
    echo "[warn] semantic feature file not found for dry-run: $SEMANTIC_FEATURES" >&2
  else
    echo "Missing semantic feature file: $SEMANTIC_FEATURES" >&2
    echo "Run with --build-semantic-if-missing or call scripts/generate_clip_semantic_features.py first." >&2
    exit 1
  fi
fi

FALSE_GT="$GENERATED_DIR/${GENERATE_PREFIX}_false_gt.txt"
FALSE_PRED="$GENERATED_DIR/${GENERATE_PREFIX}_false_pred.txt"
GENERATED_TRUE="$GENERATED_DIR/${GENERATE_PREFIX}_true.txt"

need_true_list=0
need_false_lists=0
if [[ "$SPLIT" == "true" ]] && ! repo_path_exists "$EVAL_LIST"; then
  need_true_list=1
fi
if [[ "$SPLIT" != "true" ]] && { ! repo_path_exists "$FALSE_GT" || ! repo_path_exists "$FALSE_PRED"; }; then
  need_false_lists=1
fi

if (( BUILD_SPLITS_IF_MISSING )) && (( need_true_list || need_false_lists )); then
  mkdir -p "$REPO_ROOT/$GENERATED_DIR"
  BUILD_CMD=(
    "$PYTHON_BIN" "$BUILD_SPLITS_SCRIPT"
    --model        "$MODEL"
    --datasets     "$DATASETS"
    --save-prefix  "$GENERATE_PREFIX"
    --out-dir      "$GENERATED_DIR"
    --device       "$DEVICE"
    --weights      "$WEIGHTS"
    --ground-truth-file "$OFFICIAL_GT_FILE"
    --official-id2class-file "$OFFICIAL_ID2CLASS_FILE"
    --shuffle-seed "$SHUFFLE_SEED"
    --true-quota "$TRUE_QUOTA"
    --false-quota "$FALSE_QUOTA"
  )
  if [[ ${#SOURCE_LISTS[@]} -gt 0 ]]; then
    BUILD_CMD=( "$PYTHON_BIN" "$BUILD_SPLITS_SCRIPT"
      --model        "$MODEL"
      --datasets     "$DATASETS"
      --input-lists  "${SOURCE_LISTS[@]}"
      --save-prefix  "$GENERATE_PREFIX"
      --out-dir      "$GENERATED_DIR"
      --device       "$DEVICE"
      --weights      "$WEIGHTS"
    )
  fi
  if [[ "$MODEL" == clip_* ]]; then
    BUILD_CMD+=(--semantic-features "$SEMANTIC_FEATURES" --clip-download-root "$CLIP_DOWNLOAD_ROOT")
  fi
  echo "Preparing model-specific mistake splits"
  run_or_print "${BUILD_CMD[@]}"
fi

if [[ "$SPLIT" == "true" ]] && ! repo_path_exists "$EVAL_LIST"; then
  if (( DRY_RUN )); then
    echo "[warn] eval list not found for dry-run: $EVAL_LIST" >&2
  else
    echo "Missing eval list: $EVAL_LIST" >&2
    echo "Run with --build-splits-if-missing or call scripts/build_imagenet_eval_lists.py first." >&2
    exit 1
  fi
fi

if [[ "$SPLIT" != "true" ]] && { ! repo_path_exists "$FALSE_GT" || ! repo_path_exists "$FALSE_PRED"; }; then
  if (( DRY_RUN )); then
    echo "[warn] generated mistake lists not found for dry-run:" >&2
    echo "[warn]   $FALSE_GT" >&2
    echo "[warn]   $FALSE_PRED" >&2
  else
    echo "Missing generated mistake lists:" >&2
    echo "  $FALSE_GT" >&2
    echo "  $FALSE_PRED" >&2
    echo "Run with --build-splits-if-missing or call scripts/build_imagenet_eval_lists.py first." >&2
    exit 1
  fi
fi

if (( ! DRY_RUN )) && printf '%s\n' "${ALGORITHMS[@]}" | grep -Fxq 'dhsic'; then
  if ! "$PYTHON_BIN" -c "import tensorflow, xplique" >/dev/null 2>&1; then
    echo "[warn] dhsic requires tensorflow + xplique (not found) — skipping dhsic." >&2
    FILTERED=()
    for algo in "${ALGORITHMS[@]}"; do
      [[ "$algo" == "dhsic" ]] || FILTERED+=("$algo")
    done
    ALGORITHMS=("${FILTERED[@]}")
  fi
fi

cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════"
echo " Task       : classification"
echo " Model      : $MODEL_LABEL"
echo " Model key  : $MODEL"
echo " Split      : $SPLIT"
echo " Datasets   : $DATASETS"
echo " Result root: $SAVE_ROOT"
echo " Algorithms : ${ALGORITHMS[*]}"
echo " Segmenter  : superpixel / slico / 50 divisions"
if [[ "$SPLIT" == "true" ]]; then
  echo " Eval list  : $EVAL_LIST"
else
  if [[ ${#SOURCE_LISTS[@]} -gt 0 ]]; then
    echo " Source     : ${SOURCE_LISTS[*]}"
  else
    echo " Source     : $OFFICIAL_GT_FILE + $OFFICIAL_ID2CLASS_FILE"
  fi
  echo " Generated  : $GENERATED_DIR"
fi
[[ "$MODEL" == resnet* ]] && echo " Weights    : $WEIGHTS"
[[ -n "$BEGIN" ]] && echo " Begin      : $BEGIN"
[[ -n "$END"   ]] && echo " End        : $END"
(( DRY_RUN )) && echo " Mode       : dry-run"
echo "════════════════════════════════════════════════════════════"

run_case() {
  local case_name="$1"
  local eval_list="$2"
  local save_dir="$3"

  for algo in "${ALGORITHMS[@]}"; do
    echo
    echo "  ── $case_name / $algo"
    if [[ "$MODEL" == clip_* ]]; then
      CMD=(
        "$PYTHON_BIN" "$TASK_SCRIPT"
        --datasets             "$DATASETS"
        --algorithm            "$algo"
        --clip-type            "$CLIP_TYPE"
        --clip-download-root   "$CLIP_DOWNLOAD_ROOT"
        --semantic-features    "$SEMANTIC_FEATURES"
        --eval-list            "$eval_list"
        --segmenter            superpixel
        --superpixel-algorithm slico
        --division-number      50
        --save-dir             "$save_dir"
        --device               "$DEVICE"
      )
    else
      CMD=(
        "$PYTHON_BIN" "$TASK_SCRIPT"
        --datasets             "$DATASETS"
        --algorithm            "$algo"
        --arch                 "$ARCH"
        --weights              "$WEIGHTS"
        --eval-list            "$eval_list"
        --segmenter            superpixel
        --superpixel-algorithm slico
        --division-number      50
        --save-dir             "$save_dir"
        --device               "$DEVICE"
      )
    fi
    [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
    [[ -n "$END"   ]] && CMD+=(--end   "$END")
    if [[ "$algo" == "igos_pp" ]]; then
      CMD+=(
        --igos-mask-size      "$IGOS_MASK_SIZE"
        --igos-steps          "$IGOS_STEPS"
        --igos-lr             "$IGOS_LR"
        --igos-blur-sigma     "$IGOS_BLUR_SIGMA"
        --igos-preserve-coeff "$IGOS_PRESERVE_COEFF"
        --igos-delete-coeff   "$IGOS_DELETE_COEFF"
        --igos-area-coeff     "$IGOS_AREA_COEFF"
        --igos-tv-coeff       "$IGOS_TV_COEFF"
        --igos-binary-coeff   "$IGOS_BINARY_COEFF"
      )
    fi
    run_or_print "${CMD[@]}"
  done
}

case "$SPLIT" in
  true)
    run_case "true" "$EVAL_LIST" "$SAVE_ROOT"
    ;;
  cause)
    run_case "cause" "$FALSE_PRED" "$SAVE_ROOT/mistake/cause"
    ;;
  repair)
    run_case "repair" "$FALSE_GT" "$SAVE_ROOT/mistake/repair"
    ;;
  both)
    run_case "cause" "$FALSE_PRED" "$SAVE_ROOT/mistake/cause"
    run_case "repair" "$FALSE_GT" "$SAVE_ROOT/mistake/repair"
    ;;
esac

echo
echo "════════════════════════════════════════════════════════════"
echo " Evaluate with:"
if [[ "$SPLIT" == "true" ]]; then
  echo "   python scripts/eval_classification.py --results-dir $SAVE_ROOT"
elif [[ "$SPLIT" == "both" ]]; then
  echo "   python scripts/eval_classification.py --results-dir $SAVE_ROOT/mistake/cause"
  echo "   python scripts/eval_classification.py --results-dir $SAVE_ROOT/mistake/repair"
else
  echo "   python scripts/eval_classification.py --results-dir $SAVE_ROOT/mistake/$SPLIT"
fi
echo "════════════════════════════════════════════════════════════"
