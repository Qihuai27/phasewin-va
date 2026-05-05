#!/usr/bin/env bash
# Compatibility wrapper for Qwen2.5-VL-7B caption runs.
#
# Canonical entrypoint:
#   bash scripts/run_caption.sh --model-size 7b

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_SCRIPT="$REPO_ROOT/scripts/run_caption.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

bash "$BASE_SCRIPT" --model-size 7b "$@"
