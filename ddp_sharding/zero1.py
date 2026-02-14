"""Zero1 sharding."""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class ZeroOneSharding:
    """Zero-1 sharding optimizer.

    In Zero-1 sharding, optimizer states are sharded across available GPUs.
    Model parameters remain replicated on all ranks.
    During the optimizer step, each GPU only updates its local parameters and their corresponding states.
    After the local optimizer step, we need to synchronize the updated parameters across all ranks
    to ensure that all ranks have the same parameter values for the next iteration.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Get parameter groups stored in the optimizer
        # Parameter group contains the parameters and
        # their corresponding hyperparameters (like learning rate, momentum, etc.)
        self.original_param_groups = self.optimizer.param_groups
        # Get list of all parameters
        self._all_params = [
            parameter
            for param_group in self.original_param_groups
            for parameter in param_group["params"]
        ]
        # Shard optimizer states across ranks
        self._shard_optimizer_states()

    @staticmethod
    def _build_shard_bounds(total: int, world_size: int) -> list[tuple[int, int]]:
        """Build contiguous [start, end) shard bounds for each rank."""
        base, remainder = total // world_size, total % world_size
        shard_sizes = [base + int(rank < remainder) for rank in range(world_size)]

        bounds: list[tuple[int, int]] = []
        start = 0
        for size in shard_sizes:
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

    def _shard_optimizer_states(self):
        """Shard optimizer states across ranks.

        Each rank keeps only a portion of the optimizer states, determined by the rank and world size.
        """
        total_params = len(self._all_params)
        shard_bounds = self._build_shard_bounds(total=total_params, world_size=self.world_size)

        start_idx, end_idx = shard_bounds[self.rank]
        self.local_param_indices = list(range(start_idx, end_idx))
        self.local_params = [self._all_params[i] for i in self.local_param_indices]
        local_param_ids = {id(parameter) for parameter in self.local_params}
        self._param_owner = {}
        # Build a mapping from parameter ID to the rank that owns it, based on the shard bounds
        for owner_rank, (owner_start_idx, owner_end_idx) in enumerate(shard_bounds):
            for parameter in self._all_params[owner_start_idx:owner_end_idx]:
                self._param_owner[id(parameter)] = owner_rank

        # Keep only the local parameters in the optimizer's param_groups for the current rank
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"] if id(parameter) in local_param_ids
            ]

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        """Perform single optimizer step and sync parameters across all ranks.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        """
        # Perform optimization only on the local parameters
        self.optimizer.step(closure=closure, **kwargs)
        # After the local optimizer step, synchronize the updated parameters across all ranks
        # This ensures that all ranks have the same parameter values for the next iteration
        for param in self._all_params:
            # Broadcast the updated parameter values from the rank that owns them to all other ranks
            owner_rank = self._param_owner[id(param)]
            dist.broadcast(tensor=param.data, src=owner_rank)

    def zero_grad(self, set_to_none: bool = True):
        # Clear gradients for all model params, not only the local optimizer shard.
        for parameter in self._all_params:
            parameter.grad = None
