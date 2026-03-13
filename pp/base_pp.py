import torch
import torch.distributed as dist
from abc import ABC


class BasePipeline(ABC):
    def __init__(self, stage, num_stages, module, optimizer, loss_fn, num_microbatches, pp_group, device=None):
        self.stage = stage
        self.num_stages = num_stages
        self.stage_module = module
        self.stage_opt = optimizer
        self.loss_fn = loss_fn
        self.num_microbatches = num_microbatches
        self.pp_group = pp_group
        self.device = device if device is not None else torch.device(f"cuda:{stage}")
        self._p2p_initialized = False

    @property
    def is_first(self):
        return self.stage == 0

    @property
    def is_last(self):
        return self.stage == self.num_stages - 1

    def _initialize_p2p(self) -> None:
        """Pre-warm NCCL P2P channels with dummy tensors to avoid lazy-init deadlocks."""
        if self._p2p_initialized:
            return
        dummy = torch.zeros(1, device=self.device)
        ops: list[dist.P2POp] = []
        # Forward direction: recv from prev, send to next
        if not self.is_first:
            ops.append(dist.P2POp(dist.irecv, dummy.clone(), self.stage - 1, self.pp_group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.isend, dummy.clone(), self.stage + 1, self.pp_group))
        # Backward direction: recv from next, send to prev
        if not self.is_last:
            ops.append(dist.P2POp(dist.irecv, dummy.clone(), self.stage + 1, self.pp_group))
        if not self.is_first:
            ops.append(dist.P2POp(dist.isend, dummy.clone(), self.stage - 1, self.pp_group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
        self._p2p_initialized = True

    def run_batch(self, batch):
        """Run one training step for this rank; return scalar loss"""
        raise NotImplementedError()

    def step(self):
        """Step the optimizer."""
        self.stage_opt.step()

    def zero_grad(self):
        """Zero the optimizer gradients."""
        self.stage_opt.zero_grad(set_to_none=True)
