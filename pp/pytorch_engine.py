import torch
import torch.distributed as dist

from pp.engine import BasePPEngine


class PytorchPPEngine(BasePPEngine):
    """Adapter that runs torch.distributed.pipelining schedules."""

    def __init__(self, schedule, optimizer, device, model_for_memory=None, pp_group=None):
        self.schedule = schedule
        self._optimizer = optimizer
        self.device = device
        self._model_for_memory = model_for_memory
        self.pp_group = pp_group if pp_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.pp_group)
        self.is_last = self.rank == dist.get_world_size(self.pp_group) - 1

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model_for_memory(self):
        return self._model_for_memory

    def train_batch(self, batch: dict[str, torch.Tensor]) -> float | None:
        with torch.profiler.record_function("pp.optimizer_zero_grad"):
            self._optimizer.zero_grad(set_to_none=True)
        losses = []
        stage_module = getattr(getattr(self.schedule, "_stage", None), "submod", None)
        if hasattr(stage_module, "prepare_microbatch_attention_mask"):
            stage_module.prepare_microbatch_attention_mask(
                batch["attention_mask"].to(self.device, non_blocking=True),
                self.schedule._n_microbatches,
            )

        # For pipeline parallel training,
        # we need to call schedule.step() with the appropriate inputs and targets
        # on the first and last stages,
        # and just step through the schedule on the intermediate stages.
        if self.rank == 0 and self.is_last:
            with torch.profiler.record_function("pp.forward"):
                self.schedule.step(
                    batch["input_ids"].to(self.device, non_blocking=True),
                    target=batch["labels"].to(self.device, non_blocking=True),
                    losses=losses,
                )
        elif self.rank == 0:
            with torch.profiler.record_function("pp.forward"):
                self.schedule.step(
                    batch["input_ids"].to(self.device, non_blocking=True),
                )
        elif self.is_last:
            with torch.profiler.record_function("pp.forward"):
                self.schedule.step(
                    target=batch["labels"].to(self.device, non_blocking=True),
                    losses=losses,
                )
        else:
            with torch.profiler.record_function("pp.forward"):
                self.schedule.step()

        with torch.profiler.record_function("pp.optimizer_step"):
            self._optimizer.step()

        if self.is_last and losses:
            return torch.stack([loss.detach() for loss in losses]).mean().item()
        return None
