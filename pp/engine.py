from abc import ABC, abstractmethod

import torch


class BasePPEngine(ABC):
    """Minimal engine interface for a unified PP training loop."""

    @property
    @abstractmethod
    def optimizer(self):
        raise NotImplementedError

    @property
    def model_for_memory(self):
        return None

    @abstractmethod
    def train_batch(self, batch: dict[str, torch.Tensor]) -> float | None:
        """Run one train step for this rank; return scalar loss on the logging rank."""
        raise NotImplementedError
