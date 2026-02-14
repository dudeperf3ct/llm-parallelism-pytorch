# Distributed Training Experiments

Implement and compare various data parallelism strategies on Yelp Review Full using `HuggingFaceTB/SmolLM2-360M-Instruct`.

* Data Parallelism write up: https://dudeperf3ct.github.io/posts/implement_data_parallelism/
* Sharding write up: https://dudeperf3ct.github.io/posts/implement_sharding/

## Requirements

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- Multiple GPUs
- uv

I used a 2 x Nvidia L40 (24 GB) instance using the Run Pod platform to run these experiments. It costs around $2/hour as of December 2025.

## Setup

```bash
# Install dependencies into .venv
uv sync
```

## Run (multi-GPU only)

To run all implemented strategies in one go:

### DDP

```bash
./run_experiment_ddp.sh 4
```

### DDP with Sharding

```bash

```


Following sections describe how to run each strategy individually. The `torchrun` CLI sets up the distributed environment variables for you.

```bash
# Choose how many GPUs to use on the node
NUM_GPUS=4

torchrun --standalone --nproc_per_node=$NUM_GPUS main_ddp.py --ddp-choice simple_ddp
```

Notes:
- `GLOBAL_BATCH_SIZE` (8) is split across ranks; adjust it if you change `NUM_GPUS` or use GPUs with larger memory.
- Profiler traces land under `profile/<ddp_choice>/rank_<rank>/`.
- Logs print only on rank 0
- You can change `--ddp-choice` to try different strategies: `simple_ddp`, `simple_ddp_ga`, `simple_ddp_hook`, `simple_ddp_async`, `bucket_ddp_async`, `pytorch_ddp`.

## Trace Analysis

Analyze PyTorch profiler traces with [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) (HTA) library. The script generates a single HTML dashboard and a compact CSV summary.

```bash
python scripts/analyze_traces.py --trace-dir profile/simple_ddp --select latest
```

By default, the output directory is inferred by replacing `profile/` with `reports/`, so the example above writes to `reports/simple_ddp/summary.html` and `reports/simple_ddp/summary.csv`.

Optional flags:
- `--select all` to analyze each trace window and save under `reports/<name>/run_<idx>_<ts>/`.
- `--enable-multiprocessing` to parse traces with multiprocessing.

>[!NOTE]
> Each experiment produces a trace file for each rank that can be viewed at [perfetto UI](https://ui.perfetto.dev/). This provides detailed breakdown of CUDA streams and CPU threads. It shows the compute time for all the operations taking place on GPU and CPU.