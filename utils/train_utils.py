import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .ddp_utils import get_dist_info
from .memory_utils import (
    print_memory_snapshot,
    reset_cuda_peak_memory,
)
from .pt_profiler import pt_profiler


def _build_profiler_config(total_steps: int, max_active: int = 10) -> tuple[int, int, int, int]:
    """Build a practical profiler schedule from the total number of training steps.

    Returns:
        A tuple `(wait_steps, warmup_steps, active_steps, repeat)` for
        `torch.profiler.schedule(...)`.

    Terms:
        - `wait_steps`: steps to skip before profiling starts (avoid startup noise).
        - `warmup_steps`: steps where profiler is enabled but events are not recorded
          into the final trace yet (let kernels/caches settle).
        - `active_steps`: steps that are actually recorded.
        - `repeat`: how many times this `(wait + warmup + active)` cycle is repeated.

    Logic:
        - For tiny runs (`total_steps <= 2`), profile everything:
          `wait=0`, `warmup=0`, `active=total_steps`.
        - Otherwise use a stable default: `wait=1`, `warmup=1`.
        - Keep `active_steps` bounded by `max_active`, but never below 1 and never
          beyond remaining steps after wait+warmup.
        - Compute one cycle length as `wait + warmup + active`.
        - Set `repeat` from how many full cycles fit into `total_steps`, clamped to `[1, 4]`
          to avoid creating too many trace files while still capturing repeated patterns.

    Example:
        If `total_steps=120` and `max_active=10`, then:
        `wait=1`, `warmup=1`, `active=10`, cycle=12, `repeat=min(4, 120//12)=4`.
    """
    total_steps = max(1, total_steps)
    if total_steps <= 2:
        wait_steps = 0
        warmup_steps = 0
        active_steps = total_steps
    else:
        wait_steps = 1
        warmup_steps = 1
        active_steps = min(max_active, total_steps - wait_steps - warmup_steps)
        active_steps = max(1, active_steps)

    cycle_steps = wait_steps + warmup_steps + active_steps
    repeat = max(1, min(4, total_steps // cycle_steps))
    return wait_steps, warmup_steps, active_steps, repeat


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


def train_step(batch, model, optimizer):
    """Perform a single training step: forward, backward, optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    with torch.profiler.record_function("forward"):
        outputs = model(**batch)
    loss = outputs.loss
    with torch.profiler.record_function("backward"):
        loss.backward()

    # If we are using PyTorch's DDP, we don't need to include this step
    # as it uses backward hook to automatically register and bucket gradients
    # Important step for SimpleDDP to sync gradients
    if hasattr(model, "sync_gradients"):
        with torch.profiler.record_function("grad_sync"):
            model.sync_gradients()

    with torch.profiler.record_function("optimizer_step"):
        optimizer.step()
    return model, optimizer, loss


def train_step_with_hook_ga_async(  # noqa
    batch, model, optimizer, grad_accum_steps, batch_idx, is_async=False, is_hook=False
):
    """Perform a single training step with gradient accumulation either using sync or async comms."""
    # Zero gradients at the start of accumulation
    if (batch_idx - 1) % grad_accum_steps == 0:
        optimizer.zero_grad(set_to_none=True)

    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
    # Determine if we should sync gradients this step
    should_sync = batch_idx % grad_accum_steps == 0
    # Perform backward and optimizer step based on accumulation
    if should_sync and not is_hook and not is_async:
        loss.backward()
        if hasattr(model, "sync_gradients"):
            model.sync_gradients()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elif should_sync and is_hook and not is_async:
        loss.backward()
        # We don't need to call sync_gradients explicitly here
        # Backward hooks will handle it before optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elif should_sync and is_async and is_hook:
        loss.backward()
        # For async, we don't sync gradients here
        # Wait for all async gradient syncs to complete
        model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # Use no_sync context to skip gradient sync during accumulation steps
    # This skips optimizer step and gradient sync until accumulation is done
    else:
        with model.no_sync():
            loss.backward()
    return model, optimizer, loss * grad_accum_steps  # Return original loss value


def train_loop(  # noqa
    model,
    data,
    optimizer,
    device,
    epochs,
    profile_dir,
    grad_accum_steps=None,
    is_async=False,
    is_hook=False,
    memory_log_interval=0,
):
    """Train a model while logging timing stats for each batch and epoch.

    Args:
        model: Model to train.
        data: DataLoader providing training batches.
        optimizer: Optimizer instance.
        device: Target device for training.
        epochs: Number of epochs to run.
        profile_dir: Directory to save profiler traces.
        grad_accum_steps: Number of steps for gradient accumulation (optional).
        is_async: Whether to use asynchronous gradient synchronization (optional).
        is_hook: Whether to use hook-based gradient synchronization (optional).
        memory_log_interval: Log memory every N batches on all ranks (0 disables per-batch logs).
    """
    rank, _, _ = get_dist_info()
    log_on_rank0 = rank == 0

    total_batches = 0
    total_start = time.perf_counter()

    model.train()
    total_steps = epochs * len(data)
    wait_steps, warmup_steps, active_steps, repeat = _build_profiler_config(total_steps)
    profiler_cm = pt_profiler(
        profile_dir,
        active_steps=active_steps,
        wait_steps=wait_steps,
        warmup_steps=warmup_steps,
        repeat=repeat,
    )

    # Use CUDA events for GPU timings without forcing full device syncs.
    if log_on_rank0:
        move_start_event = torch.cuda.Event(enable_timing=True)
        move_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)

    with profiler_cm as profiler:
        for epoch in range(epochs):
            data.sampler.set_epoch(epoch)
            reset_cuda_peak_memory(device)
            epoch_start = time.perf_counter()
            epoch_move_times = []
            epoch_batch_times = []

            last_batch_idx = 0
            for batch_idx, batch in enumerate(data, start=1):
                last_batch_idx = batch_idx
                move_time = 0.0
                batch_time = 0.0
                log_step_memory = memory_log_interval > 0 and batch_idx % memory_log_interval == 0

                if log_on_rank0:
                    batch_start_event.record()
                    move_start_event.record()

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # noqa

                if log_on_rank0:
                    move_end_event.record()

                with torch.profiler.record_function("train_step"):
                    if grad_accum_steps is not None:
                        model, optimizer, loss = train_step_with_hook_ga_async(
                            batch,
                            model,
                            optimizer,
                            grad_accum_steps,
                            batch_idx,
                            is_async=is_async,
                            is_hook=is_hook,
                        )
                    else:
                        model, optimizer, loss = train_step(batch, model, optimizer)

                if log_step_memory:
                    print_memory_snapshot(
                        prefix=f"Epoch {epoch + 1} Batch {batch_idx}",
                        model=model,
                        optimizer=optimizer,
                        rank=rank,
                        device=device,
                        sync_cuda=True,
                    )

                if log_on_rank0:
                    batch_end_event.record()
                    batch_end_event.synchronize()
                    move_time = move_start_event.elapsed_time(move_end_event) / 1000.0
                    batch_time = batch_start_event.elapsed_time(batch_end_event) / 1000.0

                if log_on_rank0:
                    epoch_move_times.append(move_time)
                    epoch_batch_times.append(batch_time)
                total_batches += 1

                # Take a profiler step after every batch
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

            # Handle any remaining gradients for gradient accumulation for both cases.
            # For async, there are no outstanding async handles on this tail
            # we use SimpleDDPAsyncHookGA sync_gradients to flush them.
            # Similarly for BucketDDPAsyncHookGA, we call sync_gradients to flush remaining grads.
            if (
                grad_accum_steps is not None
                and last_batch_idx % grad_accum_steps != 0
                and last_batch_idx != 0
            ):
                model.sync_gradients()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            print_memory_snapshot(
                prefix=f"Epoch {epoch + 1} end",
                model=model,
                optimizer=optimizer,
                rank=rank,
                device=device,
                sync_cuda=True,
            )

    total_time = time.perf_counter() - total_start
    avg_time_per_batch = total_time / total_batches if total_batches else 0.0
    if log_on_rank0:
        print(
            f"Training completed in {total_time:.3f}s across {total_batches} batches "
            f"(avg {avg_time_per_batch:.3f}s per batch)"
        )
    return model


def train_loop_pp(
    engine,
    data,
    device,
    epochs,
    profile_dir,
    memory_log_interval=0,
):
    """Train a PP engine with timing, profiling, and memory logging."""
    rank, _, _ = get_dist_info()
    pp_group = None
    if hasattr(engine, "pp_group"):
        pp_group = engine.pp_group
    elif hasattr(engine, "pipeline_impl"):
        pp_group = getattr(engine.pipeline_impl, "pp_group", None)
    if pp_group is None:
        pp_group = dist.group.WORLD
    pp_rank = dist.get_rank(pp_group)
    pp_world_size = dist.get_world_size(pp_group)
    log_on_last = pp_rank == pp_world_size - 1

    total_batches = 0
    total_start = time.perf_counter()

    total_steps = epochs * len(data)
    wait_steps, warmup_steps, active_steps, repeat = _build_profiler_config(total_steps)
    profiler_cm = pt_profiler(
        profile_dir,
        active_steps=active_steps,
        wait_steps=wait_steps,
        warmup_steps=warmup_steps,
        repeat=repeat,
    )

    if log_on_last:
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)

    with profiler_cm as profiler:
        for epoch in range(epochs):
            data.sampler.set_epoch(epoch)
            reset_cuda_peak_memory(device)
            epoch_start = time.perf_counter()
            epoch_batch_times = []
            epoch_loss_sum = 0.0
            epoch_loss_steps = 0

            for batch_idx, batch in enumerate(data, start=1):
                batch_time = 0.0
                log_step_memory = memory_log_interval > 0 and batch_idx % memory_log_interval == 0

                if log_on_last:
                    batch_start_event.record()

                with torch.profiler.record_function("pp.train_batch"):
                    step_loss = engine.train_batch(batch)

                if log_step_memory:
                    print_memory_snapshot(
                        prefix=f"Epoch {epoch + 1} Batch {batch_idx}",
                        model=engine.model_for_memory,
                        optimizer=engine.optimizer,
                        rank=rank,
                        device=device,
                        sync_cuda=True,
                    )

                if log_on_last and step_loss is not None:
                    epoch_loss_sum += step_loss
                    epoch_loss_steps += 1

                if log_on_last:
                    batch_end_event.record()
                    batch_end_event.synchronize()
                    batch_time = batch_start_event.elapsed_time(batch_end_event) / 1000.0
                    epoch_batch_times.append(batch_time)

                total_batches += 1
                profiler.step()

                if log_on_last:
                    loss_text = f"{step_loss:.4f}" if step_loss is not None else "n/a"
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] "
                        f"[Batch {batch_idx}/{len(data)}] "
                        f"batch_time={batch_time:.3f}s "
                        f"loss={loss_text}"
                    )

            epoch_time = time.perf_counter() - epoch_start
            avg_batch_time = (
                sum(epoch_batch_times) / len(epoch_batch_times) if epoch_batch_times else 0.0
            )
            avg_loss = epoch_loss_sum / max(1, epoch_loss_steps)

            if log_on_last:
                print(
                    f"[Epoch {epoch + 1}] epoch_time={epoch_time:.3f}s "
                    f"avg_batch_time={avg_batch_time:.3f}s "
                    f"avg_loss={avg_loss:.4f}"
                )

            print_memory_snapshot(
                prefix=f"Epoch {epoch + 1} end",
                model=engine.model_for_memory,
                optimizer=engine.optimizer,
                rank=rank,
                device=device,
                sync_cuda=True,
            )

    total_time = time.perf_counter() - total_start
    avg_time_per_batch = total_time / total_batches if total_batches else 0.0
    if log_on_last:
        print(
            f"Training completed in {total_time:.3f}s across {total_batches} batches "
            f"(avg {avg_time_per_batch:.3f}s per batch)"
        )


def evaluate(model, data_loader, device):
    """Run evaluation and return accuracy (0-1)."""
    rank, _, _ = get_dist_info()
    log_on_rank0 = rank == 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total += labels.numel()

    correct_tensor = torch.tensor(correct, device=device, dtype=torch.long)
    total_tensor = torch.tensor(total, device=device, dtype=torch.long)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    accuracy = correct_tensor.float() / total_tensor.clamp_min(1)

    if log_on_rank0:
        print(f"Evaluation accuracy: {accuracy.item() * 100:.2f}%")

    model.train()
    return accuracy.item()
