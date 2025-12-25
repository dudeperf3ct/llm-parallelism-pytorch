import argparse

import torch

from data import prepare_data
from ddp.simple_ddp import SimpleDDP
from model import get_model
from utils.ddp_utils import ddp_cleanup, ddp_initialize, get_dist_info
from utils.train_utils import set_seed, train_loop

GLOBAL_BATCH_SIZE = 1024


parser = argparse.ArgumentParser(description="Distributed Training Example")
parser.add_argument(
    "--ddp-choice", type=str, choices=["simple_ddp", "pytorch_ddp"], default="simple_ddp"
)
args = parser.parse_args()

if __name__ == "__main__":
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")

    set_seed(42)
    ddp_initialize()
    rank, world_size, local_rank = get_dist_info()
    PER_DEVICE_BATCH_SIZE = GLOBAL_BATCH_SIZE // world_size
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    device = torch.device(f"cuda:{local_rank}")

    print(f"Number of devices: {world_size}")
    print(f"Number of batches per device: {PER_DEVICE_BATCH_SIZE}")
    print(f"Preparing data on rank {rank}...")
    train_loader, eval_loader = prepare_data(PER_DEVICE_BATCH_SIZE, rank, world_size)
    model = get_model()
    model.to(device)

    if args.ddp_choice == "pytorch_ddp":
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    elif args.ddp_choice == "simple_ddp":
        model = SimpleDDP(model)

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    profile_dir = f"profile/{args.ddp_choice}"
    train_loop(model, train_loader, optim, device=device, epochs=10, profile_dir=profile_dir)

    ddp_cleanup()
