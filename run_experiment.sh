#!/bin/bash
set -exo pipefail

NUM_GPUS=${1:-}
if [[ -z "$NUM_GPUS" ]]; then
  echo "Usage: ./run_experiment.sh <num_gpus>"
  exit 1
fi

echo "---Running with DDP Choice: simple_ddp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp
sleep 5

echo "---Running with DDP Choice: simple_ddp_ga---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp_ga
sleep 5

echo "---Running with DDP Choice: simple_ddp_hook---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp_hook
sleep 5

echo "---Running with DDP Choice: simple_ddp_async---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp_async
sleep 5

echo "---Running with DDP Choice: bucket_ddp_async---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice bucket_ddp_async

sleep 5
echo "---Running with DDP Choice: pytorch_ddp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice pytorch_ddp
