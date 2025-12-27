from contextlib import contextmanager

import torch

from ddp.simple_ddp import SimpleDDP


class SimpleDDPWithGA(SimpleDDP):
    """GradientAccumulation version of SimpleDDP."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.do_sync = True

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

    def sync_gradients(self):
        """Synchronize gradients across ranks if enabled."""
        if not self.should_sync:
            return
        super().sync_gradients()
