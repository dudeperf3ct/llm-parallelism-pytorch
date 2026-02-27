import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class SimpleDDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # Options for custom gradient synchronization function,
        # used in ZeRO2 and ZeRO3 where gradients are sharded across ranks
        self._gradient_sync_fn = None
        # Used in Zero3 to materialize full parameters before forward via owner broadcast
        self._pre_forward_fn = None
        # broadcast parameters from rank 0 to all other ranks
        self.sync_parameters()

    def set_gradient_sync_fn(self, gradient_sync_fn):
        """Set a custom gradient synchronization function."""
        self._gradient_sync_fn = gradient_sync_fn

    def set_pre_forward_fn(self, pre_forward_fn):
        """Set a callback to run before each forward pass."""
        self._pre_forward_fn = pre_forward_fn

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            # distributed comm: broadcast
            # broadcast parameter values across all ranks
            # to be same as that of rank 0
            dist.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        # These are used to override the default forward behaviour in DDP
        # For example, in ZeRO3, the full parameters are materialized on each rank
        # before forward pass
        if self._pre_forward_fn is not None:
            self._pre_forward_fn()
        return self.model(*args, **kwargs)

    def sync_gradients(self):
        """Average gradients across ranks."""
        # These are used to override the default gradient sync behaviour
        # For example, in ZeRO2 and Zero3, the gradients are sharded across ranks
        # and the synchronization is performed using reduce-scatter semantics.
        if self._gradient_sync_fn is not None:
            self._gradient_sync_fn()
            return

        # Synchronize gradients across all ranks
        for param in self.model.parameters():
            # distributed comm: All Reduce
            # To perform syncronization, we
            # first need to gather gradients from all ranks
            # sum all the gathered gradients
            # broadcast the summed results to all ranks
            # All this can be performed using single all_reduce operation
            if param.grad is not None:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
                # Average the gradients by all ranks
                param.grad /= self.world_size
