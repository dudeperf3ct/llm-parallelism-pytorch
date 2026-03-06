"""Pipeline Engine for custom pipeline implementations like Naive, GPipe, and 1F1B."""

from pp.engine import BasePPEngine


class ScratchPPEngine(BasePPEngine):
    def __init__(self, pipeline_impl):
        self.pipeline_impl = pipeline_impl

    @property
    def optimizer(self):
        return self.pipeline_impl.stage_opt

    @property
    def model_for_memory(self):
        return self.pipeline_impl.stage_module

    def train_batch(self, batch):
        return self.pipeline_impl.run_batch(batch)
