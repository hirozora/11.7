#!/usr/bin/env bash
# Model view of each scope (src/build_model_data.py) and its acceptance checks (src/check_model_data.py), then the
# fixed-shape model arrays (src/model_inputs.py) and their checks (src/check_model_inputs.py).
# Requires the scope's tables from scripts/build_data.sh and the split manifests from src/make_model_split.py.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -eq 0 ]; then
  echo 'Usage: bash scripts/build_model_data.sh England_nontest [France_nontest ...]' >&2
  exit 2
fi
for scope in "$@"; do
  python src/build_model_data.py --config "configs/model/$scope.json"
  python src/check_model_data.py --config "configs/model/$scope.json"
  python src/model_inputs.py --config "configs/model/$scope.json"
  python src/check_model_inputs.py --config "configs/model/$scope.json"
done
