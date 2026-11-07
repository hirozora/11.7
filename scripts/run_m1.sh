#!/usr/bin/env bash
# M1 for one seed: the five cross-fitting folds and the full training period, in parallel over the GPUs
# (src/train_seq.py), then the assembled offsets (src/assemble_offsets.py).
# Usage: bash scripts/run_m1.sh <config> <seed> <out_dir> <gpu for fold 0> ... <gpu for full>
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 9 ]; then
  echo 'Usage: bash scripts/run_m1.sh configs/model/m1.json 20260715 results/model/m1/seed20260715 0 0 1 1 2 3' >&2
  exit 2
fi
config=$1; seed=$2; out=$3; shift 3
gpus=("$@")
mkdir -p "$out"
pids=()
for i in 0 1 2 3 4 5; do
  fold=$i; [ "$i" -eq 5 ] && fold=full
  python src/train_seq.py --config "$config" --fold "$fold" --seed "$seed" --device "cuda:${gpus[$i]}" \
    --out "$out/fold_$fold" > "$out/fold_$fold.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do
  wait "$p" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "at least one M1 run failed; see $out/fold_*.log" >&2
  exit 1
fi
python src/assemble_offsets.py --config "$config" --runs "$out"
echo "M1 seed $seed done"
