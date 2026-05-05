#!/usr/bin/env bash
# Run the extension bundle through the canonical repo-native entrypoints.
#
# Default tracks:
#   1. CLIP RN101 main table
#   2. CLIP ViT-L/14 mistake (cause + repair)
#   3. CLIP RN101 mistake (cause + repair)
#   4. Qwen2.5-VL-7B main table
#
# Optional auxiliary track:
#   - resnet101
#
# Usage:
#   bash scripts/run_round1_extension.sh [options]
#
# Options:
#   --tracks LIST        Comma-separated subset of:
#                        clip_rn101,clip_vitl_mistake,clip_rn101_mistake,qwen7b,resnet101
#   --build-if-missing   Prepare missing assets automatically where supported
#   --source-list PATH   Optional legacy labeled pool for mistake split generation
#   --generated-dir DIR  Output dir for generated false_gt / false_pred lists
#   --clip-rn101-semantic PATH  RN101 semantic feature path
#   --qwen-model-name PATH      Qwen2.5-VL-7B checkpoint path or HF id
#   --qwen-eval-list PATH       Qwen2.5-VL-7B eval list path
#   --datasets PATH      COCO image root for Qwen2.5-VL-7B
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

CLASSIFICATION_SCRIPT="$REPO_ROOT/scripts/run_classification.sh"
CAPTION_SCRIPT="$REPO_ROOT/scripts/run_caption.sh"

TRACKS_CSV="clip_rn101,clip_vitl_mistake,clip_rn101_mistake,qwen7b"
BUILD_IF_MISSING=0
SOURCE_LIST=""
GENERATED_DIR="datasets/imagenet/generated"
CLIP_RN101_SEMANTIC="ckpt/semantic_features/clip_rn101_imagenet_zeroweights.pt"
QWEN_MODEL_NAME="model_checkpoint/Qwen2.5-VL-7B-Instruct"
QWEN_EVAL_LIST="datasets/Qwen2.5-VL-7B-coco-caption.json"
DATASETS="datasets/coco/val2017"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
BEGIN=""
END=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracks) TRACKS_CSV="$2"; shift 2 ;;
    --build-if-missing) BUILD_IF_MISSING=1; shift ;;
    --source-list) SOURCE_LIST="$2"; shift 2 ;;
    --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
    --clip-rn101-semantic) CLIP_RN101_SEMANTIC="$2"; shift 2 ;;
    --qwen-model-name) QWEN_MODEL_NAME="$2"; shift 2 ;;
    --qwen-eval-list) QWEN_EVAL_LIST="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --begin) BEGIN="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a TRACKS <<< "$TRACKS_CSV"

track_selected() {
  local target="$1"
  local item
  for item in "${TRACKS[@]}"; do
    if [[ "$item" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

for script in \
  "$CLASSIFICATION_SCRIPT" \
  "$CAPTION_SCRIPT"; do
  if [[ ! -f "$script" ]]; then
    echo "Required script not found: $script" >&2
    exit 1
  fi
done

echo "════════════════════════════════════════════════════════════"
echo " Task       : extension-bundle experiments"
echo " Tracks     : ${TRACKS[*]}"
echo " Device     : $DEVICE"
[[ -n "$BEGIN" ]] && echo " Begin      : $BEGIN"
[[ -n "$END"   ]] && echo " End        : $END"
(( BUILD_IF_MISSING )) && echo " Assets     : auto-build missing assets"
(( DRY_RUN )) && echo " Mode       : dry-run"
echo "════════════════════════════════════════════════════════════"

if track_selected clip_rn101; then
  echo
  echo "  ── clip_rn101"
  CMD=(
    bash "$CLASSIFICATION_SCRIPT"
    --model clip_rn101
    --semantic "$CLIP_RN101_SEMANTIC"
    --device "$DEVICE"
    --python "$PYTHON_BIN"
  )
  (( BUILD_IF_MISSING )) && CMD+=(--build-semantic-if-missing)
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end "$END")
  (( DRY_RUN )) && CMD+=(--dry-run)
  run_or_print "${CMD[@]}"
fi

if track_selected clip_vitl_mistake; then
  echo
  echo "  ── clip_vitl_mistake"
  CMD=(
    bash "$CLASSIFICATION_SCRIPT"
    --model clip_vitl14
    --split both
    --generated-dir "$GENERATED_DIR"
    --device "$DEVICE"
    --python "$PYTHON_BIN"
  )
  [[ -n "$SOURCE_LIST" ]] && CMD+=(--source-list "$SOURCE_LIST")
  (( BUILD_IF_MISSING )) && CMD+=(--build-splits-if-missing)
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end "$END")
  (( DRY_RUN )) && CMD+=(--dry-run)
  run_or_print "${CMD[@]}"
fi

if track_selected clip_rn101_mistake; then
  echo
  echo "  ── clip_rn101_mistake"
  CMD=(
    bash "$CLASSIFICATION_SCRIPT"
    --model clip_rn101
    --split both
    --generated-dir "$GENERATED_DIR"
    --semantic "$CLIP_RN101_SEMANTIC"
    --device "$DEVICE"
    --python "$PYTHON_BIN"
  )
  [[ -n "$SOURCE_LIST" ]] && CMD+=(--source-list "$SOURCE_LIST")
  if (( BUILD_IF_MISSING )); then
    CMD+=(--build-splits-if-missing --build-semantic-if-missing)
  fi
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end "$END")
  (( DRY_RUN )) && CMD+=(--dry-run)
  run_or_print "${CMD[@]}"
fi

if track_selected qwen7b; then
  echo
  echo "  ── qwen7b"
  CMD=(
    bash "$CAPTION_SCRIPT"
    --model-size 7b
    --datasets "$DATASETS"
    --eval-list "$QWEN_EVAL_LIST"
    --model-name "$QWEN_MODEL_NAME"
    --device "$DEVICE"
    --python "$PYTHON_BIN"
  )
  (( BUILD_IF_MISSING )) && CMD+=(--build-eval-list-if-missing)
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end "$END")
  (( DRY_RUN )) && CMD+=(--dry-run)
  run_or_print "${CMD[@]}"
fi

if track_selected resnet101; then
  echo
  echo "  ── resnet101"
  CMD=(
    bash "$CLASSIFICATION_SCRIPT"
    --model resnet101
    --device "$DEVICE"
    --python "$PYTHON_BIN"
  )
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end "$END")
  (( DRY_RUN )) && CMD+=(--dry-run)
  run_or_print "${CMD[@]}"
fi

echo
echo "════════════════════════════════════════════════════════════"
echo " Extension-bundle experiment launch finished."
echo "════════════════════════════════════════════════════════════"
