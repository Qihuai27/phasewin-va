#!/usr/bin/env bash
# Run all detection attribution baselines.
#
# Task      : GroundingDINO SwinT OGC on COCO
# Script    : tasks/detection/groundingdino_coco.py
# Save dir  : ./detection_results/coco-groundingdino/<run-tag>/
# Algorithms: greedy  phasewin  drise  gradient  gradcam  odam  ssgrad_cam_pp
# Segmenter : superpixel / SLICO / 100 divisions  (task-script default)
#
# Results that already exist are skipped automatically by the task script.
#
# Usage:
#   bash scripts/run_detection.sh [options]
#
# Options:
#   --begin N    First sample index, inclusive  (default: 0)
#   --end   N    Last  sample index, exclusive  (default: all)
#   --device D   Torch device                   (default: cuda)
#   --python BIN Python executable              (default: $PYTHON_BIN or python)
#   --dry-run    Print commands without executing
#   -h, --help   Show this help

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; }

run_or_print() {
  printf '  '; printf '%q ' "$@"; printf '\n'
  (( DRY_RUN )) || "$@"
}

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
TASK_SCRIPT="$REPO_ROOT/tasks/detection/groundingdino_coco.py"
SAVE_DIR="./detection_results/coco-groundingdino"

# ── defaults ──────────────────────────────────────────────────────────────────

ALGORITHMS=(greedy phasewin drise gradient gradcam odam ssgrad_cam_pp)
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
BEGIN=""
END=""
DRY_RUN=0

# ── argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --begin)   BEGIN="$2";      shift 2 ;;
    --end)     END="$2";        shift 2 ;;
    --device)  DEVICE="$2";     shift 2 ;;
    --python)  PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1;       shift   ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ── validation ────────────────────────────────────────────────────────────────

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2; exit 1
fi
if [[ ! -f "$TASK_SCRIPT" ]]; then
  echo "Task script not found: $TASK_SCRIPT" >&2; exit 1
fi

# ── summary ───────────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════"
echo " Task       : detection  (GroundingDINO SwinT OGC × COCO)"
echo " Algorithms : ${ALGORITHMS[*]}"
echo " Segmenter  : superpixel / slico / 100 divisions"
echo " Save dir   : $SAVE_DIR"
[[ -n "$BEGIN" ]] && echo " Begin      : $BEGIN"
[[ -n "$END"   ]] && echo " End        : $END"
(( DRY_RUN ))    && echo " Mode       : dry-run"
echo "════════════════════════════════════════════════════════════"

# ── per-algorithm run ─────────────────────────────────────────────────────────

for algo in "${ALGORITHMS[@]}"; do
  echo
  echo "  ── $algo"
  CMD=(
    "$PYTHON_BIN" "$TASK_SCRIPT"
    --algorithm            "$algo"
    --segmenter            superpixel
    --superpixel-algorithm slico
    --division-number      100
    --save-dir             "$SAVE_DIR"
    --device               "$DEVICE"
  )
  [[ -n "$BEGIN" ]] && CMD+=(--begin "$BEGIN")
  [[ -n "$END"   ]] && CMD+=(--end   "$END")
  run_or_print "${CMD[@]}"
done

echo
echo "════════════════════════════════════════════════════════════"
echo " All detection baselines finished."
echo " Evaluate with:"
echo "   python scripts/eval_detection.py"
echo "════════════════════════════════════════════════════════════"
