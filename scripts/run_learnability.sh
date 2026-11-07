#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f results/existence/pooled.json
bash scripts/build_data.sh England_nontest France_nontest Germany_nontest Italy_nontest Spain_nontest
for league in England France Germany Italy Spain; do
  cfg="configs/analysis/learn_${league}_nontest.json"
  python src/fit_offsets.py --config "$cfg" --head who
  python src/learn_population.py --config "$cfg"
  python src/learn_entities.py --config "$cfg"
done
python src/pool_learnability.py --config configs/analysis/pool_learnability.json
