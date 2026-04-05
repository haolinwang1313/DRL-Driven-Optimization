#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/revision.local.yaml}"

cd /home/ac/Dogtor_Project/DDPG

while pgrep -f "paper_repro.cli --config ${CONFIG_PATH} run-optimizers --ddpg-only --scenario Balanced_Performance" >/dev/null; do
  sleep 60
done
./tools/run_ddpg_revision_batch.sh Energy_Saving_Focus guard_es "$CONFIG_PATH"
./tools/run_ddpg_revision_batch.sh Energy_Generation_Focus guard_eg "$CONFIG_PATH"
