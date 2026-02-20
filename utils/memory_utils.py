import torch

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - older torch versions may not expose DTensor
    DTensor = None


def _tensor_nbytes(tensor):
    """Return bytes resident on this rank for torch.Tensor or DTensor."""
    if tensor is None:
        return 0
    # Important: DTensor may expose global shape metadata.
    # For memory accounting we intentionally measure local shard bytes.
    if DTensor is not None and isinstance(tensor, DTensor):
        local_tensor = tensor.to_local()
        return local_tensor.element_size() * local_tensor.nelement()
    if torch.is_tensor(tensor):
        return tensor.element_size() * tensor.nelement()
    return 0


def _sum_tensor_nbytes(value):
    """Recursively sum tensor bytes from nested optimizer state containers."""
    if isinstance(value, dict):
        return sum(_sum_tensor_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_sum_tensor_nbytes(v) for v in value)
    return _tensor_nbytes(value)


def get_size_in_mb(tensor):
    """Get tensor memory in MB on this rank.

    Note:
        For DTensor inputs, this function reports local shard memory via `to_local()`.
        It does not report global logical tensor size across all ranks.
    """
    return _tensor_nbytes(tensor) / 1024**2


def get_optimizer_memory(optimizer):
    """Calculate local optimizer-state memory (MB) on this rank.

    Note:
        Optimizer state can be nested (dict/list/tuple/set), especially with
        sharded/distributed optimizers. This function recursively walks containers
        and sums tensor bytes, using DTensor local shard size when applicable.
    """
    total_memory = 0
    if hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    for state in optimizer.state.values():
        total_memory += _sum_tensor_nbytes(state) / 1024**2
    return total_memory


def get_model_memory(model):
    """Calculate local parameter memory (MB) on this rank.

    Note:
        In FSDP/DTensor modes, this is per-rank resident memory, not global model size.
    """
    return sum(get_size_in_mb(p) for p in model.parameters())


def get_gradient_memory(model):
    """Calculate local gradient memory (MB) on this rank.

    Note:
        In sharded training, this is the local shard gradient footprint.
    """
    return sum(get_size_in_mb(p.grad) for p in model.parameters() if p.grad is not None)


def reset_cuda_peak_memory(device):
    torch.cuda.reset_peak_memory_stats(device)


def get_memory_snapshot(model, optimizer, device):
    """Capture a per-rank memory snapshot.

    Returns local (rank-resident) state sizes and CUDA allocator counters.
    """
    return {
        "model_mb": get_model_memory(model),
        "grad_mb": get_gradient_memory(model),
        "optim_mb": get_optimizer_memory(optimizer),
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def print_memory_snapshot(prefix: str, model, optimizer, rank, device, sync_cuda=False):
    if sync_cuda:
        torch.cuda.synchronize(device)
    snapshot = get_memory_snapshot(model, optimizer, device)
    print(
        f"[Rank {rank}] {prefix} | "
        f"model={snapshot['model_mb']:.2f}MB "
        f"grad={snapshot['grad_mb']:.2f}MB "
        f"optim={snapshot['optim_mb']:.2f}MB "
        f"alloc={snapshot['cuda_allocated_mb']:.2f}MB "
        f"reserved={snapshot['cuda_reserved_mb']:.2f}MB "
        f"peak_alloc={snapshot['cuda_peak_allocated_mb']:.2f}MB "
        f"peak_reserved={snapshot['cuda_peak_reserved_mb']:.2f}MB"
    )
