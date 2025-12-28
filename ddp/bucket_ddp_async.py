from contextlib import contextmanager

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class BucketDDPAsyncHookGA(torch.nn.Module):
    """Bucketed async DDP with gradient accumulation support.

    The flow is:
    - Register post-accumulate hooks per parameter.
    - Accumulate gradients into size-limited buckets.
    - Launch async all-reduce for each full bucket.
    - Call `finish_gradient_synchronization()` after backward to flush the
      remaining partial bucket and wait for all reductions to complete.

    A major concern here is we are building bucket on each ranky by hook firing order
    Hook order can differ across each rank, so flat buffers line up different parameters
    on different ranks. The all-reduce then sums mismatched params and might corrupt grads silently.
    """

    def __init__(self, model: torch.nn.Module, bucket_cap_mb: int = 25):
        super().__init__()
        self.model = model
        self.handles = []
        self.do_sync = True
        self.bucket, self.bucket_size = [], 0
        self.bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # broadcast parameters from rank 0 to all other ranks
        # This ensures all models start with the same parameters
        self.sync_parameters()
        # Register backward hooks to handle gradient synchronization
        self.register_bucket_hook()

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            torch.distributed.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _sync_gradients(self):
        """Asynchronously all-reduce the gradients in the current bucket."""
        grads = [g for g in self.bucket if g is not None]
        if not grads:
            return
        # Concatenate gradients into a single tensor for all-reduce
        # This reduces the overhead of multiple small all-reduce calls
        # Perform asynchronous all-reduce
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        handle = dist.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, grads, flat_grad))

    def _fill_buckets(self, param):
        """Fill buckets with gradients and trigger async all-reduce when full."""
        if not self.should_sync or param.grad is None:
            return

        # Fill the bucket with the current gradient
        # Calculate the size of the gradient in bytes
        grad = param.grad
        grad_size = grad.numel() * grad.element_size()
        self.bucket.append(grad)
        self.bucket_size += grad_size

        # If the bucket is full, sync the gradients
        if self.bucket_size >= self.bucket_cap_bytes:
            self._sync_gradients()
            # Clear the bucket
            self.bucket = []
            self.bucket_size = 0

    def register_bucket_hook(self):
        # Keep track of hooks to remove them later if needed.
        self.sync_hooks = []
        for p in self.model.parameters():
            if p.requires_grad:
                # Register a hook per parameter.
                # The hook will be called after all gradients for a tensor have been accumulated
                h = p.register_post_accumulate_grad_hook(self._fill_buckets)
                self.sync_hooks.append(h)

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed.
        Also flushes any remaining partial bucket.
        """
        # Ensure the final partial bucket is also synchronized.
        self.flush_buckets()
        for work, grads, flat_grad in self.handles:
            work.wait()
            # Unflatten the gradients back to their original shapes
            offset = 0
            for g in grads:
                numel = g.numel()
                g.copy_(flat_grad[offset : offset + numel].view_as(g))
                g.div_(self.world_size)
                offset += numel
        self.handles.clear()

    def flush_buckets(self):
        """Flush any remaining gradients in the bucket."""
        if self.bucket:
            self._sync_gradients()
            self.bucket = []
            self.bucket_size = 0

    @property
    def should_sync(self):
        """Indicate that gradient synchronization is needed."""
        return self.do_sync

    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization across ranks."""
        prev = self.do_sync
        self.do_sync = False
        try:
            yield
        finally:
            self.do_sync = prev

    def sync_gradients(self) -> None:
        """Synchronize gradients for last step if needed."""
        if not self.should_sync:
            return
        # Hooks were skipped during no_sync; bucket the current grads manually.
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                self._fill_buckets(p)
        self.finish_gradient_synchronization()
