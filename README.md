# Distributed Training Experiments

Implement and compare various data parallelism strategies on Yelp Review Full using `HuggingFaceTB/SmolLM2-360M-Instruct`.

* Data Parallelism write up: https://dudeperf3ct.github.io/posts/implement_data_parallelism/
* Sharding write up: https://dudeperf3ct.github.io/posts/implement_sharding/

## Costs

**DDP**: I used a 2 x Nvidia L4 (24 GB) instance using the Run Pod platform to run these experiments. It costs around $0.79/hour as of December 2025. It costs about $2.25 to complete these experiments.

**Sharding**: I used a 2 x Nvidia L4 (24 GB) instance using the Run Pod platform to run these experiments. It costs around $0.78/hour as of Feburary 2026. It costs about $2 to complete these experiments.


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

To run all implemented strategies in one go:

### DDP

```bash
./run_experiment_ddp.sh 2
```

### DDP with Sharding

```bash
./run_experiment_shard.sh 2
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

Analyze PyTorch profiler traces with [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) (HTA).

### DDP traces

```bash
python scripts/analyze_traces_ddp.py --trace-dir profile/simple_ddp --select latest
```

Output is inferred by replacing `profile/` with `reports/`, so the example above writes:
- `reports/simple_ddp/summary.html`
- `reports/simple_ddp/summary.csv`

### Sharding traces
```bash
python scripts/analyze_traces_sharding.py --trace-dir profile_runpod/pytorch_zero3 --select latest
```

Output is inferred by replacing `profile/` with `reports_sharding/` when applicable. For the above command:
- `reports_sharding/pytorch_zero3/summary.html`
- `reports_sharding/pytorch_zero3/summary.csv`

Optional flags for both scripts:
- `--select all` to analyze each trace window and save under `run_<idx>_<ts>/`.
- `--enable-multiprocessing` to parse traces with multiprocessing.

>[!NOTE]
> Each experiment produces a trace file for each rank that can be viewed at [perfetto UI](https://ui.perfetto.dev/). This provides detailed breakdown of CUDA streams and CPU threads. It shows the compute time for all the operations taking place on GPU and CPU.
