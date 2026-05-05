#!/usr/bin/env bash
# Compatibility wrapper for classification mistake runs.
#
# Canonical entrypoint:
#   bash scripts/run_classification.sh --model <model> --split <cause|repair|both>
#
# This wrapper keeps the older interface alive while delegating all logic to the
# standardized classification runner.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_classification_mistake.sh [options]

Options:
  --model NAME          clip_vitl14 | clip_rn101 | resnet101
  --mode NAME           cause | repair | both  (default: both)
  Any other option is forwarded to scripts/run_classification.sh.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_SCRIPT="$REPO_ROOT/scripts/run_classification.sh"

MODEL="clip_vitl14"
MODE="both"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

bash "$BASE_SCRIPT" --model "$MODEL" --split "$MODE" "${FORWARD_ARGS[@]}"
