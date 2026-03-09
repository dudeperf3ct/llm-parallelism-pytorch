# ZeRO Sharding Summary (RunPod + HTA)

## Resources summarized

- Run log: `runpod_run.log`
- Traces: `profile_sharding/*`
- HTA outputs: `reports_sharding/*/summary.csv`, `reports_sharding/*/summary.html`
- Paper reference: ZeRO (Rajbhandari et al., 2019) - https://arxiv.org/abs/1910.02054

## 1) Model-state memory vs ZeRO paper expectations

From `runpod_run.log` (Epoch 10 end, averaged across rank 0/1):

| Mode | Model MB | Grad MB | Optim MB | Total state MB | vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 1380.26 | 1380.26 | 2760.51 | 5521.03 | 1.000 |
| zero1 | 1380.26 | 1380.26 | 1380.26 | 4140.78 | 0.750 |
| zero2 | 1380.26 | 690.12 | 1380.26 | 3450.65 | 0.625 |
| zero3 | 690.12 | 690.12 | 1380.26 | 2760.51 | 0.500 |
| pytorch_zero2 | 690.13 | 690.13 | 1380.26 | 2760.52 | 0.500 |
| pytorch_zero3 | 690.13 | 690.13 | 1380.26 | 2760.52 | 0.500 |

Expected trend from ZeRO paper for data parallel degree `N_d=2`:

- Stage 1: optimizer states shard -> ~75% of baseline model-state memory.
- Stage 2: optimizer + gradients shard -> ~62.5%.
- Stage 3: optimizer + gradients + parameters shard -> ~50%.

Result:

- Custom `zero1/zero2/zero3` follows the expected per-stage reduction pattern exactly.
- `pytorch_zero2/pytorch_zero3` also shows the expected ~50% state footprint vs baseline.
- `pytorch_zero1` shows `optim=0.00MB` in this log path, which under-reports optimizer state for this wrapper and should not be used as a literal Stage-1 state accounting value.

## 2) Rank balance observation

The custom implementations partition by contiguous parameter index ranges (not by tensor bytes), so shard sizes are imbalanced:

- `zero2` and `zero3` gradient/parameter state ratio rank0:rank1 is about `1.31x` (783.63 MB vs 596.62 MB).
- This is expected from parameter-level ownership by list index, not tensor-level equal-byte partitioning.

## 3) HTA trace-analysis insights (from `reports_sharding/*/summary.csv`)

Per-mode means across ranks:

| Mode | Peak alloc MB | P95 alloc MB | Collective ms | Comm overhead % | Comm/comp overlap % | Compute % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 14411.92 | 12846.57 | 1574.45 | 46.50 | 0.00 | 50.30 |
| zero1 | 13028.40 | 11500.45 | 1636.38 | 49.72 | 0.00 | 48.70 |
| zero2 | 13018.14 | 11167.29 | 2625.27 | 60.99 | 0.00 | 37.34 |
| zero3 | 13019.03 | 11132.91 | 2926.11 | 63.50 | 0.00 | 34.63 |
| pytorch_zero1 | 14391.37 | 12870.29 | 1960.90 | 51.82 | 34.64 | 55.02 |
| pytorch_zero2 | 13762.26 | 12040.26 | 1881.05 | 50.02 | 56.69 | 60.78 |
| pytorch_zero3 | 12662.63 | 11071.09 | 3017.95 | 57.28 | 56.46 | 57.87 |

Takeaways from the graphs/CSV:

- Peak memory: `zero1/zero2/zero3` all reduce peak allocated memory vs baseline, but improvement is much smaller than model-state-only reduction because activations/temporary buffers/allocator behavior remain dominant.
- Communication cost: custom `zero2/zero3` significantly increases collective time and communication overhead vs baseline (tradeoff for memory).
- Overlap behavior: custom `zero*` paths show near-zero comm/compute overlap, while PyTorch/FSDP paths show substantial overlap (around 35-57%), indicating better overlap scheduling in native stack.
- Collective pattern shifts as expected:
  - baseline: mostly all-reduce
  - zero1: all-reduce + broadcast
  - zero2/zero3: reduce-scatter + broadcast
  - pytorch_zero2/zero3: mainly all-gather + reduce-scatter

## 4) End-to-end training behavior

From `runpod_run.log`:

- All modes reached the same evaluation accuracy: `31.25%`.
- Runtime tradeoff (avg batch time):
  - faster: `pytorch_zero1` (~1.701 s), `baseline`/`zero1` (~1.785 s)
  - slower: `zero2` (~2.321 s), `zero3` (~2.458 s), `pytorch_zero3` (~2.553 s)

Overall conclusion:

- The custom ZeRO stage behavior matches the paper's model-state partitioning expectations.
- The main gap is efficiency: communication-heavy custom stage-2/3 implementations have low overlap and higher collective cost compared with native PyTorch sharding paths.
