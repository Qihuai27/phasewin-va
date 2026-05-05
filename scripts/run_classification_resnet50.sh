#!/usr/bin/env bash
# Compatibility wrapper for torchvision ResNet-50 classification runs.
#
# Canonical entrypoint:
#   bash scripts/run_classification.sh --model resnet50

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/run_classification.sh"

if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

bash "$BASE_SCRIPT" --model resnet50 "$@"
