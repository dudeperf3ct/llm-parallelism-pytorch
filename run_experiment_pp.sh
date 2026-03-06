#!/bin/bash
set -exo pipefail

NUM_GPUS=${1:-}
if [[ -z "$NUM_GPUS" ]]; then
  echo "Usage: ./run_experiment_pp.sh <num_gpus>"
  exit 1
fi

echo "---Running with PP Choice: naive_pp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_pp.py --pp-choice naive_pp
sleep 5

echo "---Running with PP Choice: gpipe_pp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_pp.py --pp-choice gpipe_pp
sleep 5

echo "---Running with PP Choice: 1f1b_pp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_pp.py --pp-choice 1f1b_pp
sleep 5

echo "---Running with PP Choice: pytorch_gpipe_pp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_pp.py --pp-choice pytorch_gpipe_pp
sleep 5

echo "---Running with PP Choice: pytorch_1f1b_pp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_pp.py --pp-choice pytorch_1f1b_pp
