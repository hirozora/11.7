#!/usr/bin/env bash
# Development runs of the who-acts head, scored on the validation period (docs/MODEL_PLAN.md, section 9, P1-P3).
# Requires the model arrays of the five non-test scopes (scripts/build_model_data.sh). Steps, in this order:
#   population  R1, R2, M0 and the M0p diagnostic (CPU)
#   m1          M1: five cross-fitting folds and the full training period (scripts/run_m1.sh; six GPU slots)
#   entities    entity and cell indices, history statistics and graphs, and the history checks
#   b0          B0-M1 and B0-g (CPU)
#   entity      the degenerate check and the entity variants over the GPUs, then the choice of the ridge scale c
#   control     England-only control: population, M1, entities, B0 and M2 on England's scope alone
#   england_rows  the England rows of the five-league scores (argument: the tag of the five-league M2's chosen c)
#   evaluate    configs/model/eval_dev.json and configs/model/eval_england_control.json
# Usage: bash scripts/run_model_dev.sh <step> [gpu ... | c tag]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo 'Usage: bash scripts/run_model_dev.sh population|m1|entities|b0|entity|control|england_rows|evaluate [gpu ...|c tag]' >&2
  exit 2
fi
step=$1; shift
seed=20260715
case "$step" in
  population)
    python src/fit_population.py --config configs/model/population.json
    python src/fit_population.py --config configs/model/population_m0_pooled.json ;;
  m1)
    bash scripts/run_m1.sh configs/model/m1.json "$seed" "results/model/m1/seed$seed" "$@" ;;
  entities)
    python src/entity_terms.py --config configs/model/entities.json
    python src/history_stats.py --config configs/model/history.json
    python src/check_history.py --config configs/model/history.json ;;
  b0)
    python src/fit_b0.py --config configs/model/b0_m1.json
    python src/fit_b0.py --config configs/model/b0_g.json ;;
  entity)
    python src/queue_models.py --jobs configs/model/degenerate_jobs.json --script entity_models.py --gpus "$1" \
      --jobs-per-gpu 1 --log-dir results/model/entity/logs_degenerate
    python src/queue_models.py --jobs configs/model/entity_jobs_ids.json --script entity_models.py --gpus "$@" \
      --jobs-per-gpu 2 --log-dir results/model/entity/logs_ids
    python src/queue_models.py --jobs configs/model/entity_jobs_hist.json --script entity_models.py --gpus "$@" \
      --jobs-per-gpu 2 --log-dir results/model/entity/logs_hist
    python src/select_entity.py --jobs configs/model/entity_jobs_ids.json configs/model/entity_jobs_hist.json \
      --out results/model/entity/selected.json ;;
  control)
    python src/fit_population.py --config configs/model/population_england_only.json
    bash scripts/run_m1.sh configs/model/m1_england_only.json "$seed" "results/model/m1_england_only/seed$seed" "$@"
    python src/entity_terms.py --config configs/model/entities_england_only.json
    python src/history_stats.py --config configs/model/history_england_only.json
    python src/check_history.py --config configs/model/history_england_only.json
    python src/fit_b0.py --config configs/model/b0_m1_england_only.json
    python src/fit_b0.py --config configs/model/b0_g_england_only.json
    python src/entity_models.py --config configs/model/m2_england_only.json --model degenerate --seed "$seed" \
      --device "cuda:$1" --out results/model/degenerate_england_only
    python src/queue_models.py --jobs configs/model/entity_england_only_jobs.json --script entity_models.py \
      --gpus "$1" --jobs-per-gpu 3 --log-dir results/model/entity_england_only/logs
    python src/select_entity.py --jobs configs/model/entity_england_only_jobs.json \
      --out results/model/entity_england_only/selected.json ;;
  england_rows)
    # argument: the tag (1-3, 1 or 3) of the ridge scale chosen for the five-league M2 (results/model/entity/selected.json)
    five=(configs/model/{England,France,Germany,Italy,Spain}_nontest.json)
    for pair in "m0:results/model/population/m0.npz" "m1:results/model/m1/seed$seed/offsets.npz" \
                "b0_m1:results/model/b0_m1/b0.npz" "m2:results/model/entity/m2_c$1/scores.npz"; do
      python src/slice_scores.py --scores "${pair#*:}" --from-scopes "${five[@]}" \
        --to-scopes configs/model/England_nontest.json --out "results/model/england_rows/${pair%%:*}.npz"
    done ;;
  evaluate)
    python src/evaluate_models.py --config configs/model/eval_dev.json
    python src/evaluate_models.py --config configs/model/eval_england_control.json ;;
  *)
    echo "unknown step $step" >&2; exit 2 ;;
esac
