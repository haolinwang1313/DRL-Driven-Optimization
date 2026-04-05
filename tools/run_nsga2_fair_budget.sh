#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/revision.local.yaml}"
LOG_PATH="${2:-artifacts/publication/optimization/nsga2_fair_budget.log}"

cd /home/ac/Dogtor_Project/DDPG
. .venv/bin/activate
export PATH=/home/ac/bin:$PATH
mkdir -p "$(dirname "$LOG_PATH")"

python -m paper_repro.cli --config "$CONFIG_PATH" run-optimizers --nsga2-only > "$LOG_PATH" 2>&1
tail -n 40 "$LOG_PATH"
