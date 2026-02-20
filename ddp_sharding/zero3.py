"""Zero3 sharding."""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class Zero3Sharding:
    """Zero-3 sharding optimizer.

    In Zero-3 sharding, parameters, gradients, and optimizer states are sharded
    across available GPUs.
    - Parameters are sharded by ownership (non-owner ranks keep empty tensors).
      This is parameter-level partitioning: whole ``Parameter`` objects are assigned by index.
      It does not split each parameter tensor across ranks.
    - Full parameters are materialized before forward via owner broadcast.
    - Gradients are sharded using reduce-scatter similar to Zero2.
    - Full parameters are resharded after backward to free memory.
    - Optimizer step updates only local-owner parameters/states similar to Zero1 and Zero2.

    Note:
        The behaviour for optimizer states is the same as Zero-1 and Zero-2 sharding
        where each GPU only updates its local parameters and their corresponding states.

        Native PyTorch FSDP/FSDP2 uses tensor-level (intra-parameter) sharding semantics,
        while this class is intentionally a simpler parameter-ownership reference.
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
        # Keep track of parameter shapes for reshaping during gather/scatter steps
        self._param_shapes = {
            id(parameter): tuple(parameter.shape) for parameter in self._all_params
        }
        # Shard optimizer states across ranks
        self._shard_optimizer_states()
        # Shard model parameters across ranks
        # Free memory of non-owner parameters by keeping empty tensors,
        # and only keep local shard of parameters on each rank
        self.reshard_model_parameters()

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

    def gather_full_parameters(self):
        """Materialize full parameters on all ranks before forward.

        Parameter gather:
            Each rank starts with only its local shard of parameters.
            We broadcast the full parameters from their owner ranks to all other ranks before forward pass,
            so that the full model is materialized on each rank for the forward and backward computations.

        Note:
            For this parameter-ownership layout, ``broadcast`` is the natural primitive because each
            full parameter lives on one owner rank. If each rank held tensor slices of every parameter
            (tensor-level partitioning), ``all_gather`` of slices would be the natural primitive.
        """
        with torch.profiler.record_function("zero3_param_gather"):
            for parameter in self._all_params:
                owner_rank = self._param_owner[id(parameter)]
                full_shape = self._param_shapes[id(parameter)]
                if self.rank != owner_rank and tuple(parameter.data.shape) != full_shape:
                    parameter.data = torch.empty(
                        full_shape, device=parameter.device, dtype=parameter.dtype
                    )
                dist.broadcast(tensor=parameter.data, src=owner_rank)

    def reshard_model_parameters(self):
        """Keep only owner parameters materialized; free non-owner parameter storage.

        This is used to free the memory of non-owner parameters after forward and backward communication steps.
        """
        for parameter in self._all_params:
            owner_rank = self._param_owner[id(parameter)]
            if self.rank != owner_rank:
                parameter.data = torch.empty(0, device=parameter.device, dtype=parameter.dtype)
                parameter.grad = None

    def shard_gradients(self):
        """Shard gradients across ranks using reduce-scatter.

        This function is used to override the default gradient sync behaviour in DDP
        where all gradients are synchronized across ranks using all_reduce.

        Gradient update:
            We use zero tensors as placeholders when a local grad is missing so that
            all ranks still execute the same collectives in the same order. This is
            not the same as a true `None` grad semantically: `None` means "unused this
            step", while zero means "used, but gradient value is 0".

            Reduce scatter is used to sum the gradients across all ranks and scatter the results
            back to the local shards.
            After backward communication, we free the memory of non-owner parameters and gradients.
        """
        with torch.profiler.record_function("zero3_reduce_scatter"):
            for parameter in self._all_params:
                owner_rank = self._param_owner[id(parameter)]
                grad = (
                    parameter.grad.detach().contiguous()
                    if parameter.grad is not None
                    else torch.zeros(
                        self._param_shapes[id(parameter)],
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
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

        # After backward communication, free the memory of non-owner parameters and gradients
        self.reshard_model_parameters()

    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any):
        """Perform optimizer step on local shards only."""
        self.optimizer.step(closure=closure, **kwargs)

    def zero_grad(self, set_to_none: bool = True):
        for parameter in self._all_params:
            parameter.grad = None
