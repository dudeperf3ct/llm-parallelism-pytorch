#!/bin/bash
set -exo pipefail

NUM_GPUS=${1:-}
if [[ -z "$NUM_GPUS" ]]; then
  echo "Usage: ./run_experiment_shard.sh <num_gpus>"
  exit 1
fi

echo "---Running with Sharding Choice: baseline---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice baseline
sleep 5

echo "---Running with Sharding Choice: zero1---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice zero1
sleep 5

echo "---Running with Sharding Choice: zero2---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice zero2
sleep 5

echo "---Running with Sharding Choice: zero3---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice zero3
sleep 5

echo "---Running with Sharding Choice: pytorch_zero1---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice pytorch_zero1
sleep 5

echo "---Running with Sharding Choice: pytorch_zero2---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice pytorch_zero2
sleep 5

echo "---Running with Sharding Choice: pytorch_zero3---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp_shard.py --shard-choice pytorch_zero3
