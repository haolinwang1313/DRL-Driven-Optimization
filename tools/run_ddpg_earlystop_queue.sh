#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/revision.earlystop.yaml}"

cd /home/ac/Dogtor_Project/DDPG

./tools/run_ddpg_revision_batch.sh Balanced_Performance stop_bal "$CONFIG_PATH"
./tools/run_ddpg_revision_batch.sh Energy_Saving_Focus stop_es "$CONFIG_PATH"
./tools/run_ddpg_revision_batch.sh Energy_Generation_Focus stop_eg "$CONFIG_PATH"
