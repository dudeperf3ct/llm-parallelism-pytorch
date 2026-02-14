#!/bin/bash
set -exo pipefail

NUM_GPUS=${1:-}
if [[ -z "$NUM_GPUS" ]]; then
  echo "Usage: ./run_experiment.sh <num_gpus>"
  exit 1
fi

echo "---Running with Sharding Choice: simple_ddp + zero 1---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_shard.py --shard-choice zero1
sleep 5

sleep 5
echo "---Running with Sharding Choice: pytorch_ddp + pytorch_zero1---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_shard.py --shard-choice pytorch_zero1
