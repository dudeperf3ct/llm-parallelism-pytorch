import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data import prepare_data
from ddp.bucket_ddp_async import BucketDDPAsyncHookGA
from ddp.simple_ddp import SimpleDDP
from ddp.simple_ddp_async import SimpleDDPAsyncHookGA
from ddp.simple_ddp_ga import SimpleDDPWithGA
from ddp.simple_ddp_hook import SimpleDDPHookGA
from model import get_model
from utils.ddp_utils import ddp_cleanup, ddp_initialize, get_dist_info
from utils.train_utils import evaluate, set_seed, train_loop

GLOBAL_BATCH_SIZE = 8
EPOCHS = 10
GRAD_ACCUM_STEPS = 2
SEED = 42

parser = argparse.ArgumentParser(description="Distributed Training Example")
parser.add_argument(
    "--ddp-choice",
    type=str,
    choices=[
        "simple_ddp",
        "simple_ddp_ga",
        "simple_ddp_hook",
        "simple_ddp_async",
        "bucket_ddp_async",
        "pytorch_ddp",
    ],
    default="simple_ddp",
)
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(SEED)
    ddp_initialize()
    rank, world_size, local_rank = get_dist_info()
    per_device_batch = GLOBAL_BATCH_SIZE // world_size
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"Number of devices: {world_size}")
        print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
        print(f"Number of batches per device: {per_device_batch}")

    print(f"Preparing data on rank {rank}...")
    train_loader, eval_loader = prepare_data(per_device_batch, rank, world_size)
    model = get_model()
    model.to(device)

    is_async = False
    is_hook = False
    if args.ddp_choice == "simple_ddp_ga":
        model = SimpleDDPWithGA(model)
        grad_accum_steps = GRAD_ACCUM_STEPS
    elif args.ddp_choice == "simple_ddp_hook":
        model = SimpleDDPHookGA(model)
        is_hook = True
        grad_accum_steps = GRAD_ACCUM_STEPS
    elif args.ddp_choice == "simple_ddp":
        model = SimpleDDP(model)
        grad_accum_steps = None
    elif args.ddp_choice == "simple_ddp_async":
        model = SimpleDDPAsyncHookGA(model)
        grad_accum_steps = GRAD_ACCUM_STEPS
        is_async = True
        is_hook = True
    elif args.ddp_choice == "bucket_ddp_async":
        model = BucketDDPAsyncHookGA(model)
        grad_accum_steps = GRAD_ACCUM_STEPS
        is_async = True
        is_hook = True
    elif args.ddp_choice == "pytorch_ddp":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        grad_accum_steps = None

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    profile_dir = f"profile/{args.ddp_choice}"
    print(f"Training started on rank {rank}...")
    train_loop(
        model,
        train_loader,
        optim,
        device=device,
        epochs=EPOCHS,
        profile_dir=profile_dir,
        grad_accum_steps=grad_accum_steps,
        is_async=is_async,
        is_hook=is_hook,
    )
    evaluate(model, eval_loader, device=device)

    ddp_cleanup()
