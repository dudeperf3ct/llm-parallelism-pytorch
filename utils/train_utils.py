import random
import time

import numpy as np
import torch

from .ddp_utils import get_dist_info
from .memory_utils import print_memory_stats
from .pt_profiler import pt_profiler


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed for reproducibility in distributed training

    Args:
        seed: Random seed value, default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_loop(model, data, optimizer, device, epochs, profile_dir):  # noqa
    """Train a model while logging timing stats for each batch and epoch.

    Args:
        model: Model to train.
        data: DataLoader providing training batches.
        optimizer: Optimizer instance.
        device: Target device for training.
        epochs: Number of epochs to run.
        profile_dir: Directory to save profiler traces.
    """
    rank, _, _ = get_dist_info()
    log_on_rank0 = rank == 0

    total_batches = 0
    total_start = time.perf_counter()

    model.train()
    profiler_cm = pt_profiler(profile_dir, epochs - 3)

    # Use CUDA events for GPU timings without forcing full device syncs.
    if log_on_rank0:
        move_start_event = torch.cuda.Event(enable_timing=True)
        move_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)

    with profiler_cm as profiler:
        for epoch in range(epochs):
            data.sampler.set_epoch(epoch)
            epoch_start = time.perf_counter()
            epoch_move_times = []
            epoch_batch_times = []

            for batch_idx, batch in enumerate(data, start=1):
                move_time = 0.0
                batch_time = 0.0

                if log_on_rank0:
                    batch_start_event.record()
                    move_start_event.record()

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # noqa

                if log_on_rank0:
                    move_end_event.record()

                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                # Important step for SimpleDDP to sync gradients
                if hasattr(model, "sync_gradients"):
                    model.sync_gradients()

                optimizer.step()

                if log_on_rank0:
                    batch_end_event.record()
                    batch_end_event.synchronize()
                    move_time = move_start_event.elapsed_time(move_end_event) / 1000.0
                    batch_time = batch_start_event.elapsed_time(batch_end_event) / 1000.0

                if log_on_rank0:
                    epoch_move_times.append(move_time)
                    epoch_batch_times.append(batch_time)
                total_batches += 1

                profiler.step()

                if log_on_rank0:
                    num_batches = len(data)
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] "
                        f"[Batch {batch_idx}/{num_batches}] "
                        f"move_to_device={move_time * 1000:.2f}ms "
                        f"batch_time={batch_time:.3f}s "
                        f"loss={loss.item():.4f}"
                    )

            epoch_time = time.perf_counter() - epoch_start
            avg_move_time = (
                sum(epoch_move_times) / len(epoch_move_times) if epoch_move_times else 0.0
            )
            avg_batch_time = (
                sum(epoch_batch_times) / len(epoch_batch_times) if epoch_batch_times else 0.0
            )

            if log_on_rank0:
                print(
                    f"[Epoch {epoch + 1}] epoch_time={epoch_time:.3f}s "
                    f"avg_move_to_device={avg_move_time * 1000:.2f}ms "
                    f"avg_batch_time={avg_batch_time:.3f}s"
                )

            print_memory_stats(
                prefix=f"Epoch {epoch + 1} end",
                model=model,
                optimizer=optimizer,
                rank=rank,
                device=device,
            )

    total_time = time.perf_counter() - total_start
    avg_time_per_batch = total_time / total_batches if total_batches else 0.0
    if log_on_rank0:
        print(
            f"Training completed in {total_time:.3f}s across {total_batches} batches "
            f"(avg {avg_time_per_batch:.3f}s per batch)"
        )
