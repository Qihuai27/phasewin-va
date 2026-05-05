#!/usr/bin/env bash
# Compatibility wrapper for CLIP RN101 classification runs.
#
# Canonical entrypoint:
#   bash scripts/run_classification.sh --model clip_rn101

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_SCRIPT="$REPO_ROOT/scripts/run_classification.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

bash "$BASE_SCRIPT" --model clip_rn101 "$@"
