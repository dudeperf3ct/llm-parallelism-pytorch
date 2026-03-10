from abc import ABC


class BasePipeline(ABC):
    def __init__(self, stage, num_stages, module, optimizer, loss_fn, num_microbatches):
        self.stage = stage
        self.num_stages = num_stages
        self.stage_module = module
        self.stage_opt = optimizer
        self.loss_fn = loss_fn
        self.num_microbatches = num_microbatches

    @property
    def is_first(self):
        return self.stage == 0

    @property
    def is_last(self):
        return self.stage == self.num_stages - 1

    def run_batch(self, batch):
        """Run one training step for this rank; return scalar loss"""
        raise NotImplementedError()

    def step(self):
        """Step the optimizer."""
        self.stage_opt.step()

    def zero_grad(self):
        """Zero the optimizer gradients."""
        self.stage_opt.zero_grad(set_to_none=True)
