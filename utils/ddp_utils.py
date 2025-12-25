import os

import torch
import torch.distributed as dist


def ddp_initialize(backend="nccl"):
    """Initialize distributed training environment.

    Args:
        backend: Backend to use for distributed training. Default is 'nccl'.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # noqa: PLW1508
    # Set the device for the current process
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


def ddp_cleanup():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dist_info():
    """Get distributed training information.

    Returns:
        rank: Rank of the current process.
        world_size: Total number of processes.
        local_rank: Local rank of the current process.
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK", rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
            )
        )
        return rank, world_size, local_rank
    raise RuntimeError("Distributed process groups not initialized")
