#!/usr/bin/env bash
# Compatibility wrapper for CLIP RN101 mistake attribution experiments.
#
# Usage:
#   bash scripts/run_classification_clip_rn101_mistake.sh [options]
#
# Example:
#   bash scripts/run_classification_clip_rn101_mistake.sh --mode both --build-if-missing --build-semantic-if-missing

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_SCRIPT="$REPO_ROOT/scripts/run_classification.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

bash "$BASE_SCRIPT" --model clip_rn101 --split both "$@"
