import torch
import torch.distributed as dist

from ddp.simple_ddp_ga import SimpleDDPWithGA


class SimpleDDPHookGA(SimpleDDPWithGA):
    """GradientAccumulation version of SimpleDDP using backward hooks.

    The flow is:
    - `register_backward_hook` registers `self._sync_gradient` on each `Parameter`.
    - During backward, once a param’s grad is fully accumulated,
      PyTorch invokes the hook with that grad tensor.
    - `_sync_gradient` runs in-place on that tensor, `all_reduce`s, and divides by `world_size`,
      so `p.grad` ends up averaged across ranks.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.register_backward_hook()

    def _sync_gradient(self, grad):
        """Hook called after a param's grad is accumulated.
        `grad` is the actual gradient tensor (same storage as param.grad).
        """
        if not self.should_sync or grad is None:
            return

        # Sum the gradient across all ranks and then average.
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad /= self.world_size

    def register_backward_hook(self):
        # Keep track of hooks to remove them later if needed.
        self.sync_hooks = []
        for p in self.model.parameters():
            if p.requires_grad:
                # Register a hook per parameter.
                # The hook will be called after all gradients for a tensor have been accumulated
                h = p.register_post_accumulate_grad_hook(self._sync_gradient)
                self.sync_hooks.append(h)
