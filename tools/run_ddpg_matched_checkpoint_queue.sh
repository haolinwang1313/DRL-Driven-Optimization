#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/revision.yaml}"

cd /home/ac/Dogtor_Project/DDPG

./tools/run_ddpg_revision_batch.sh Balanced_Performance match_bal "$CONFIG_PATH"
./tools/run_ddpg_revision_batch.sh Energy_Saving_Focus match_es "$CONFIG_PATH"
./tools/run_ddpg_revision_batch.sh Energy_Generation_Focus match_eg "$CONFIG_PATH"
