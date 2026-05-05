#!/usr/bin/env bash
# Run COCO caption/VQA attribution baselines through one canonical entrypoint.
#
# Supported model sizes:
#   - 3b
#   - 7b
#
# Examples:
#   bash scripts/run_caption.sh
#   bash scripts/run_caption.sh --model-size 7b --build-eval-list-if-missing
#
# Options:
#   --model-size NAME    3b | 7b  (default: 3b)
#   --variant NAME       Alias for --model-size
#   --datasets PATH      COCO image root (default: datasets/coco/val2017)
#   --eval-list PATH     Eval list JSON override
#   --model-name PATH    Model path or HF model id override
#   --max-image-side N   Optional longest-edge cap before Qwen preprocessing
#   --source-eval-list PATH  Source roster for rebuilding the eval list
#   --build-eval-list-if-missing  Generate the eval list automatically
#   --max-new-tokens N   Used only when auto-building the eval list
#   --algorithms LIST    Comma-separated algorithm override
#   --save-root DIR      Override result root
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
TASK_SCRIPT="$REPO_ROOT/tasks/caption_vqa/qwen25vl_coco_caption.py"
BUILD_SCRIPT="$REPO_ROOT/scripts/build_caption_eval_list.py"

MODEL_SIZE="3b"
DATASETS="datasets/coco/val2017"
EVAL_LIST=""
MODEL_NAME=""
MAX_IMAGE_SIDE=""
SOURCE_EVAL_LIST=""
SAVE_ROOT=""
ALGORITHMS_CSV="greedy,phasewin,drise,gradient,llavacam,igos_pp"
BUILD_EVAL_LIST_IF_MISSING=0
MAX_NEW_TOKENS=64
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
BEGIN=""
END=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size|--variant) MODEL_SIZE="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --eval-list) EVAL_LIST="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --max-image-side) MAX_IMAGE_SIDE="$2"; shift 2 ;;
    --source-eval-list) SOURCE_EVAL_LIST="$2"; shift 2 ;;
    --build-eval-list-if-missing) BUILD_EVAL_LIST_IF_MISSING=1; shift ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --algorithms) ALGORITHMS_CSV="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --begin) BEGIN="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

MODEL_LABEL=""
DEFAULT_EVAL_LIST=""
DEFAULT_MODEL_NAME=""
DEFAULT_MAX_IMAGE_SIDE=""
DEFAULT_SOURCE_EVAL_LIST=""
DEFAULT_SAVE_ROOT=""

case "$MODEL_SIZE" in
  3b)
    MODEL_LABEL="Qwen2.5-VL-3B-Instruct"
    DEFAULT_EVAL_LIST="datasets/Qwen2.5-VL-3B-coco-caption.json"
    DEFAULT_MODEL_NAME="model_checkpoint/Qwen2.5-VL-3B-Instruct"
    DEFAULT_MAX_IMAGE_SIDE=""
    DEFAULT_SOURCE_EVAL_LIST="datasets/Qwen2.5-VL-3B-coco-caption.json"
    DEFAULT_SAVE_ROOT="./caption_results/Qwen2.5-VL-3B-coco-caption"
    ;;
  7b)
    MODEL_LABEL="Qwen2.5-VL-7B-Instruct"
    DEFAULT_EVAL_LIST="datasets/Qwen2.5-VL-7B-coco-caption.json"
    DEFAULT_MODEL_NAME="model_checkpoint/Qwen2.5-VL-7B-Instruct"
    DEFAULT_MAX_IMAGE_SIDE="256"
    DEFAULT_SOURCE_EVAL_LIST="datasets/Qwen2.5-VL-3B-coco-caption.json"
    DEFAULT_SAVE_ROOT="./caption_results/Qwen2.5-VL-7B-coco-caption"
    ;;
  *)
    echo "Unsupported --model-size: $MODEL_SIZE" >&2
    exit 1
    ;;
esac

if [[ -z "$EVAL_LIST" ]]; then
  EVAL_LIST="$DEFAULT_EVAL_LIST"
fi
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$DEFAULT_MODEL_NAME"
fi
if [[ -z "$MAX_IMAGE_SIDE" ]]; then
  MAX_IMAGE_SIDE="$DEFAULT_MAX_IMAGE_SIDE"
fi
if [[ -z "$SOURCE_EVAL_LIST" ]]; then
  SOURCE_EVAL_LIST="$DEFAULT_SOURCE_EVAL_LIST"
fi
if [[ -z "$SAVE_ROOT" ]]; then
  SAVE_ROOT="$DEFAULT_SAVE_ROOT"
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
if [[ ! -f "$BUILD_SCRIPT" ]]; then
  echo "Build script not found: $BUILD_SCRIPT" >&2
  exit 1
fi

repo_path_exists() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    [[ -e "$path" ]]
  else
    [[ -e "$REPO_ROOT/$path" ]]
  fi
}

if [[ "$MODEL_NAME" == /* || "$MODEL_NAME" == model_checkpoint/* || "$MODEL_NAME" == ./* || "$MODEL_NAME" == ../* ]]; then
  if ! repo_path_exists "$MODEL_NAME"; then
    if (( DRY_RUN )); then
      echo "[warn] model path not found for dry-run: $MODEL_NAME" >&2
    else
      echo "Model path not found: $MODEL_NAME" >&2
      exit 1
    fi
  fi
fi

if ! repo_path_exists "$EVAL_LIST" && (( BUILD_EVAL_LIST_IF_MISSING )); then
  BUILD_CMD=(
    "$PYTHON_BIN" "$BUILD_SCRIPT"
    --datasets         "$DATASETS"
    --source-eval-list "$SOURCE_EVAL_LIST"
    --output           "$EVAL_LIST"
    --model-name       "$MODEL_NAME"
    --device           "$DEVICE"
    --max-new-tokens   "$MAX_NEW_TOKENS"
  )
  [[ -n "$MAX_IMAGE_SIDE" ]] && BUILD_CMD+=(--max-image-side "$MAX_IMAGE_SIDE")
  echo "Preparing missing caption eval list"
  run_or_print "${BUILD_CMD[@]}"
fi

if ! repo_path_exists "$EVAL_LIST"; then
  if (( DRY_RUN )); then
    echo "[warn] eval list not found for dry-run: $EVAL_LIST" >&2
  else
    echo "Eval list not found: $EVAL_LIST" >&2
    echo "Run with --build-eval-list-if-missing or call scripts/build_caption_eval_list.py first." >&2
    exit 1
  fi
fi

cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════"
echo " Task       : caption_vqa"
echo " Model      : $MODEL_LABEL"
echo " Model size : $MODEL_SIZE"
echo " Datasets   : $DATASETS"
echo " Eval list  : $EVAL_LIST"
echo " Save root  : $SAVE_ROOT"
echo " Algorithms : ${ALGORITHMS[*]}"
echo " Segmenter  : superpixel / slico / 64 divisions"
[[ -n "$BEGIN" ]] && echo " Begin      : $BEGIN"
[[ -n "$END"   ]] && echo " End        : $END"
(( DRY_RUN )) && echo " Mode       : dry-run"
echo "════════════════════════════════════════════════════════════"

for algo in "${ALGORITHMS[@]}"; do
  echo
  echo "  ── $algo"
  CMD=(
    "$PYTHON_BIN" "$TASK_SCRIPT"
    --algorithm            "$algo"
    --datasets             "$DATASETS"
    --segmenter            superpixel
    --superpixel-algorithm slico
    --division-number      64
    --eval-list            "$EVAL_LIST"
    --model-name           "$MODEL_NAME"
    --save-dir             "$SAVE_ROOT"
    --device               "$DEVICE"
  )
  [[ -n "$MAX_IMAGE_SIDE" ]] && CMD+=(--max-image-side "$MAX_IMAGE_SIDE")
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end   "$END")
  run_or_print "${CMD[@]}"
done

echo
echo "════════════════════════════════════════════════════════════"
echo " Evaluate with:"
echo "   python scripts/eval_caption.py --results-dir $SAVE_ROOT"
echo "════════════════════════════════════════════════════════════"
