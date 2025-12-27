from pathlib import Path

import torch

from .ddp_utils import get_dist_info


def pt_profiler(profiler_path, active_steps, wait_steps=1, warmup_steps=2, repeat=1):
    """PyTorch profiler to trace kernels, compute and comm overheads.

    Quick intuition for choosing these:
    - `wait`: 1–2 steps (skip startup oddities)
    - `warmup`: 1–2 steps (let kernels settle)
    - `active`: 8–20 steps for a stable view, smaller if profiling overhead is too heavy
    - `repeat`: how many trace files per rank you want
    """
    rank, world_size, local_rank = get_dist_info()
    file_path = Path(profiler_path) / f"rank_{rank}"
    Path.mkdir(file_path, exist_ok=True, parents=True)
    tensorboard_handler = torch.profiler.tensorboard_trace_handler(
        str(file_path), worker_name=f"worker_{rank}", use_gzip=True
    )

    def trace_handler(prof):
        # Saves the profiler trace to a file
        tensorboard_handler(prof)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    # wait: how many steps to wait before starting to record
    # warmup: how many steps to warm up before active profiling
    # active: how many steps to actively profile
    # Cycle length = wait + warmup + active
    # Ensure `wait + warmup + active <= total_steps` (total_steps = `epochs * len(data)`).
    schedule = torch.profiler.schedule(
        wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=repeat
    )
    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )
    return profiler  # noqa
