!#/bin/bash

NUM_GPUS=2
echo "---Running with DDP Choice: simple_ddp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp
echo "---Running with DDP Choice: simple_ddp_ga---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp_ga
echo "---Running with DDP Choice: simple_ddp_hook---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp_hook
echo "---Running with DDP Choice: simple_ddp_async---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp_async
echo "---Running with DDP Choice: bucket_ddp---"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice bucket_ddp_async
