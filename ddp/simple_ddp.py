import torch
import torch.distributed as dist

from utils.ddp_utils import get_dist_info


class SimpleDDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.rank, self.world_size, self.local_rank = get_dist_info()
        # broadcast parameters from rank 0 to all other ranks
        self.sync_parameters()

    def sync_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks."""
        for param in self.model.parameters():
            # distributed comm: broadcast
            # broadcast parameter values across all ranks
            # to be same as that of rank 0
            dist.broadcast(tensor=param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def sync_gradients(self):
        """Average gradients across ranks."""
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
