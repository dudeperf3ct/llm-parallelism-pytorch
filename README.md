# Distributed Training Experiments

Implement and compare various data parallelism strategies on Yelp Review Full using `HuggingFaceTB/SmolLM2-360M-Instruct`.

Write up: 

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- Multiple GPUs
- uv

## Setup

```bash
# Install dependencies into .venv
uv sync
```

## Run (multi-GPU only)

`torchrun` sets up the distributed environment variables for you.

```bash
# Choose how many GPUs to use on the node
NUM_GPUS=4

torchrun --standalone --nproc_per_node=$NUM_GPUS main.py --ddp-choice simple_ddp
```

Notes:
- `GLOBAL_BATCH_SIZE` (1024) is split across ranks; adjust it if you change `NUM_GPUS`.
- Profiler traces land under `profile/<ddp_choice>/rank_<rank>/`.
- Logs print only on rank 0
- You can change `--ddp-choice` to try different strategies: `simple_ddp`, 