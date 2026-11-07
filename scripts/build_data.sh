#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -eq 0 ]; then
  echo 'Usage: bash scripts/build_data.sh England_train [France_nontest ...]' >&2
  exit 2
fi
for scope in "$@"; do
  python src/build_actions.py --config "configs/data/$scope.json"
  python src/build_candidates.py --config "configs/data/$scope.json"
done
