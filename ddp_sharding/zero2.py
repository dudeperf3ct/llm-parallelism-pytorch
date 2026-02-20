"""Zero2 sharding."""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class Zero2Sharding:
    """Zero-2 sharding optimizer.

    In Zero-2 sharding, both model gradients and optimizer states are sharded across available GPUs.
    Each GPU handles a portion of model gradients and their corresponding optimizer states.

    During the backward pass, in a DPP setup gradients are synchronized using all_reduce
    but in this case since gradients are sharded, we only need to synchronize the local gradients on each GPU.
    To update only the local gradients, reduce_scatter can be used to sum the gradients across all ranks
    and scatter the results back to the local shards.

    Note:
        The behaviour for optimizer states is the same as Zero-1 sharding
        where each GPU only updates its local parameters and their corresponding states.

        This reference implementation uses parameter-level partitioning: whole ``Parameter`` objects
        are assigned to owners. It does not split individual tensors across ranks.
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

    def shard_gradients(self):
        """Shard gradients across ranks using reduce-scatter.

        This function is used to override the default gradient sync behaviour in DDP
        where all gradients are synchronized across ranks using all_reduce.

        Gradient update:
            We use zero tensors as placeholders when a local grad is missing so that
            all ranks still execute the same collectives in the same order. This is
            not the same as a true `None` grad semantically: `None` means "unused this
            step", while zero means "used, but gradient value is 0".

            ``reduce_scatter`` is used to sum gradients and return owner-local shards.
            Here it is done per parameter with explicit zero placeholders for non-owners,
            which is simple but less efficient than bucketized implementations.
        """
        with torch.profiler.record_function("zero2_reduce_scatter"):
            for parameter in self._all_params:
                owner_rank = self._param_owner[id(parameter)]
                # Not optimal to zero out the gradients on non-owner ranks,
                # but it keeps the implementation simple as zero tensor instead of None
                grad = (
                    parameter.grad.detach().contiguous()
                    if parameter.grad is not None
                    else torch.zeros_like(parameter.data)
                )
                output = torch.empty_like(grad)
                input_list = []
                for rank in range(self.world_size):
                    if rank == owner_rank:
                        input_list.append(grad)
                    else:
                        input_list.append(torch.zeros_like(grad))

                dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
                if self.rank == owner_rank:
                    output /= self.world_size
                    parameter.grad = output
                else:
                    parameter.grad = None

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        """Perform single optimizer step and sync parameters across all ranks.

        Parameter update:
            In this implementation, we use per-parameter ``broadcast`` from each
            owner rank to synchronize updated parameters after the local optimizer step.

            ZeRO Stage-2 is often described as synchronizing updated parameter partitions
            with all-gather style communication. With this code's parameter-level ownership
            and no bucketization, repeated broadcasts are a simpler equivalent.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        """
        # Perform optimization only on the local parameters
        self.optimizer.step(closure=closure, **kwargs)
        # After the local optimizer step, synchronize the updated parameters across all ranks
        # This ensures that all ranks have the same parameter values for the next iteration
        with torch.profiler.record_function("zero2_param_broadcast"):
            for param in self._all_params:
                # Broadcasting each parameter separately is simple but not communication-optimal.
                # Broadcast the updated parameter values from the rank that owns them to all other ranks
                owner_rank = self._param_owner[id(param)]
                dist.broadcast(tensor=param.data, src=owner_rank)

    def zero_grad(self, set_to_none: bool = True):
        # Clear gradients for all model params, not only the local optimizer shard.
        for parameter in self._all_params:
            parameter.grad = None
