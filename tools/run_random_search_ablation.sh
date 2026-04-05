#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/revision.yaml}"
OUTPUT_SUFFIX="${2:-remote_match}"
LOG_PATH="${3:-artifacts/publication/optimization/random_search_${OUTPUT_SUFFIX}.log}"

cd /home/ac/Dogtor_Project/DDPG
. .venv/bin/activate
export PATH=/home/ac/bin:$PATH
mkdir -p "$(dirname "$LOG_PATH")"

python -m paper_repro.cli --config "$CONFIG_PATH" run-optimizers --random-only --output-suffix "$OUTPUT_SUFFIX" > "$LOG_PATH" 2>&1
tail -n 40 "$LOG_PATH"
