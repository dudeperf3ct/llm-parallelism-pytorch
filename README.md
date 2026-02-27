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

<details>
<summary><strong>NCCL Hang (diagnosis + fix)</strong></summary>

Sometimes the training gets stuck on first epoch due to an NCCL hang. The fix involved disabling P2P but can also be optimized based on the topology. 

During training with 2x NVIDIA L4 GPUs, the run would freeze on the first epoch with both GPUs pegged at 100% utilization but no progress. No error was thrown — the process just hung indefinitely. This is a classic NCCL collective communication hang.

### Diagnosing the Topology

The first step was to inspect the physical interconnect topology using:

```bash
nvidia-smi topo -m
```

```bash
        GPU0    GPU1    NIC0    NIC1    NIC2    CPU Affinity    NUMA Affinity
GPU0     X      SYS     SYS     SYS     NODE    0-31,64-95      0
GPU1    SYS      X      NODE    NODE    SYS     32-63,96-127    1
NIC0    SYS     NODE     X      PIX     SYS
NIC1    SYS     NODE    PIX      X      SYS
NIC2    NODE    SYS     SYS     SYS      X

NIC Legend:
  NIC0: mlx5_2
  NIC1: mlx5_3
  NIC2: mlx5_bond_0
```

This matrix tells you the **quality of the physical path** between every pair of components. The key values to understand are:

| Value | Meaning |
|-------|---------|
| `NV#` | NVLink — fastest direct GPU-to-GPU link (not present here) |
| `PIX` | PCIe, single bridge hop — very fast |
| `NODE` | PCIe within the same NUMA node — fast |
| `SYS` | Crosses NUMA node boundary via QPI/UPI — slowest |

**What this reveals about the setup:**

`GPU0 ↔ GPU1 = SYS`: 2 GPUs are on **different NUMA nodes** with no NVLink. Every byte that travels directly between them must cross the slow QPI/UPI inter-socket bus. This is the worst possible GPU-to-GPU topology for distributed training.

There are three NICs available. Reading each NIC's column against the GPU rows shows their locality:

- `NIC0 (mlx5_2)` → `GPU1 = NODE`, `GPU0 = SYS` — physically close to GPU1 only
- `NIC1 (mlx5_3)` → `GPU1 = NODE`, `GPU0 = SYS` — physically close to GPU1 only  
- `NIC2 (mlx5_bond_0)` → `GPU0 = NODE`, `GPU1 = SYS` — physically close to GPU0 only

`NIC2` is also a **bonded interface**. It combines `mlx5_2` and `mlx5_3` into a single logical NIC, effectively doubling available bandwidth (up to 200 Gbps combined) and providing a single stable handle for NCCL to program against.

### Why NCCL Hung

With the topology understood, the next step was enabling NCCL debug logging to see what communication path it actually chose:

```bash
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 train.py 2>&1 | grep -E "NCCL|P2P|Channel"
```

The relevant output:

```
NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/CUMEM
NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
```

NCCL decided to use **P2P/CUMEM** — a mechanism that uses the CUDA virtual memory API (`cuMemCreate`) to map one GPU's memory directly into the other GPU's address space, allowing GPU-to-GPU transfers without CPU involvement.

The problem: the CUDA driver reported P2P as available (`isAllDirectP2p 1`), but the GPUs are on different NUMA nodes connected only via the slow SYS path. The CUMEM mapping either failed silently or the transfers stalled at the hardware level. NCCL's collective kernels on the GPU then entered a **spin-poll loop** — actively burning cycles waiting for data that never arrived — which explains the 100% GPU utilization despite no actual progress. NCCL has no timeout in blocking mode by default, so the process hung forever.

### The Fix

The solution has three parts:

```bash
NCCL_P2P_DISABLE=1 \
NCCL_IB_GID_INDEX=3 \
NCCL_IB_HCA=mlx5_bond_0 \
torchrun --standalone --nproc_per_node=2 train.py
```

**`NCCL_P2P_DISABLE=1`**  
Disables all direct GPU-to-GPU P2P memory access. NCCL stops trying to map GPU memory across the NUMA boundary and instead routes data through the NIC. This is the core fix.

**`NCCL_IB_HCA=mlx5_bond_0`**  
Tells NCCL which NIC to use. `mlx5_bond_0` is the right choice here for two reasons: it is the bonded interface combining both physical NICs (giving up to 200 Gbps vs 100 Gbps from either alone), and it has `NODE`-level connectivity to GPU0, making the receive path on GPU0 local — which is often the bottleneck in AllReduce operations.

**`NCCL_IB_GID_INDEX=3`**  
Selects GID index 3 on the Mellanox NIC, which corresponds to RoCEv2 (RDMA over Converged Ethernet). This is required for the NIC to operate in RDMA mode over an Ethernet fabric rather than native InfiniBand. Without this, NCCL may fail to establish an IB connection even when the hardware supports it.

### General Debugging Checklist for NCCL Hangs

1. **Run `nvidia-smi topo -m` first.** If GPUs show `SYS` connectivity, set `NCCL_P2P_DISABLE=1` preemptively.
2. **Enable `NCCL_DEBUG=INFO`** to see which communication path NCCL selected and whether IB/Socket is being used.
3. **Check for `P2P/CUMEM` in the channel output.** If present on a cross-NUMA topology, this is likely your hang.
4. **Pick the right NIC.** From the topo matrix, find the NIC with the best (lowest) connectivity value to your GPUs. Prefer bonded interfaces.
5. **Verify RDMA is active** in the debug log — you want to see `Using network IB` not `Using network Socket`. Socket fallback works but is significantly slower for large models.
</details>

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
