#!/usr/bin/env bash
set -euo pipefail

SCENARIO="$1"
PREFIX="$2"
CONFIG_PATH="${3:-configs/revision.local.yaml}"

cd /home/ac/Dogtor_Project/DDPG
. .venv/bin/activate
export PATH=/home/ac/bin:$PATH
mkdir -p artifacts/publication/optimization

pids=()
run_shard() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  local suffix="$4"
  echo "launch ${suffix} gpu=${gpu} scenario=${SCENARIO} seeds=${start}:${end}" >&2
  CUDA_VISIBLE_DEVICES="$gpu" PAPER_REPRO_DEVICE=cuda:0 \
    python -m paper_repro.cli --config "$CONFIG_PATH" run-optimizers --ddpg-only --scenario "$SCENARIO" --seed-start "$start" --seed-end "$end" --output-suffix "$suffix" \
    > "artifacts/publication/optimization/${suffix}.log" 2>&1 &
  pids+=("$!")
}

run_shard 4 0 5  ${PREFIX}_00_05
run_shard 5 5 10 ${PREFIX}_05_10
run_shard 6 10 15 ${PREFIX}_10_15
run_shard 7 15 20 ${PREFIX}_15_20
wait "${pids[@]}"

for f in artifacts/publication/optimization/${PREFIX}_*.log; do
  echo "==== ${f} ===="
  tail -n 20 "$f"
done
