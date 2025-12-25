from pathlib import Path

import torch

from .ddp_utils import get_dist_info


def pt_profiler(profiler_path, steps):
    rank, world_size, local_rank = get_dist_info()
    file_path = Path(profiler_path) / f"rank_{rank}"
    Path.mkdir(file_path, exist_ok=True, parents=True)
    tensorboard_handler = torch.profiler.tensorboard_trace_handler(
        str(file_path), worker_name=f"worker_{rank}", use_gzip=True
    )
    sort_by_keyword = "self_cuda_time_total"

    def trace_handler(prof):
        # Saves the profiler trace to a file
        tensorboard_handler(prof)
        prof.export_chrome_trace(f"{file_path}_trace.json")
        output = prof.key_averages().table(sort_by=sort_by_keyword, row_limit=-1)
        print(output)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    # wait: how many steps to wait before starting to record
    # warmup: how many steps to warm up before active profiling
    # active: how many steps to actively profile
    schedule = torch.profiler.schedule(wait=1, warmup=2, active=steps)
    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )
    return profiler #noqa
