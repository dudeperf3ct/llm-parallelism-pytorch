import torch
import torch.distributed as dist

from ddp.simple_ddp_hook import SimpleDDPHookGA


class SimpleDDPAsyncHookGA(SimpleDDPHookGA):
    """Asynchronous GradientAccumulation version of SimpleDDP using backward hooks.

    The flow is:

    - `register_backward_hook` registers `_sync_gradient` on each param.
    - Each hook fires after a param's grad is accumulated; it kicks off an async all-reduce
      and records the work handle plus the grad view.
    - Call `finish_gradient_synchronization` after backward to wait on all pending
      reductions and average the grads in place.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.handles = []

    def _sync_gradient(self, param):
        """Hook called after a param's grad is accumulated.
        Use asynchronous all-reduce to overlap communication with computation.
        """
        if not self.should_sync or param.grad is None:
            return

        # Asynchronously sum the gradient across all ranks and then average.
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, param.grad))

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed."""
        for work, grad in self.handles:
            work.wait()
            grad.div_(self.world_size)
        self.handles.clear()

    def sync_gradients(self) -> None:
        """Synchronize gradients for last step if needed."""
        if not self.should_sync:
            return
        for p in self.model.parameters():
            if p.grad is not None:
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, p.grad))
        self.finish_gradient_synchronization()
