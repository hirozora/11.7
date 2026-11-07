#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/build_data.sh England_train France_nontest Germany_nontest Italy_nontest Spain_nontest
cfg=configs/analysis/existence_England_train.json
python src/fit_offsets.py --config "$cfg" --head who
python src/fit_offsets.py --config "$cfg" --head event
python src/analyze_who.py --config "$cfg"
python src/analyze_events.py --config "$cfg"
python src/make_reference.py --who results/reference/who.json --event results/reference/event.json --out results/reference/england.json
for league in France Germany Italy Spain; do
  cfg="configs/analysis/existence_${league}_nontest.json"
  python src/fit_offsets.py --config "$cfg" --head who
  python src/fit_offsets.py --config "$cfg" --head event
  python src/analyze_who.py --config "$cfg"
  python src/analyze_events.py --config "$cfg"
done
python src/pool_existence.py --config configs/analysis/pool_existence.json
