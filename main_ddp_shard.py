import argparse

import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from data import prepare_data
from ddp.simple_ddp import SimpleDDP
from ddp_sharding.zero1 import ZeroOneSharding
from ddp_sharding.zero2 import Zero2Sharding
from ddp_sharding.zero3 import Zero3Sharding
from model import get_model
from utils.ddp_utils import ddp_cleanup, ddp_initialize, get_dist_info
from utils.train_utils import evaluate, set_seed, train_loop

GLOBAL_BATCH_SIZE = 14
EPOCHS = 10
SEED = 42

parser = argparse.ArgumentParser(description="Distributed Training Example using Sharding")
parser.add_argument(
    "--shard-choice",
    type=str,
    choices=[
        "baseline",  # No sharding, just DDP
        "zero1",
        "zero2",
        "zero3",
        "pytorch_zero1",
        "pytorch_zero2",
        "pytorch_zero3",  # Implemented as FSDP2 in PyTorch different from DeepSpeed ZeRO3
        # Native PyTorch FSDP2 uses tensor-level sharding semantics.
        # This repo's custom zero3 path is a simpler parameter-level ownership implementation.
    ],
    default="baseline",
)


args = parser.parse_args()


def wrap_with_fsdp2(model, reshard_after_forward: bool):
    """Apply FSDP2 fully_shard per decoder block, then on the full module."""
    for layer in model.model.layers:  # decoder blocks
        fully_shard(layer, reshard_after_forward=reshard_after_forward)
    return fully_shard(model, reshard_after_forward=reshard_after_forward)


if __name__ == "__main__":
    set_seed(SEED)
    ddp_initialize()
    global_rank, world_size, local_rank = get_dist_info()
    per_device_batch = GLOBAL_BATCH_SIZE // world_size
    print(f"Rank: {global_rank}, World Size: {world_size}, Local Rank: {local_rank}")
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"Number of devices: {world_size}")
        print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
        print(f"Number of batches per device: {per_device_batch}")

    print(f"Preparing data on rank {global_rank}...")
    train_loader, eval_loader = prepare_data(per_device_batch, local_rank, world_size)
    model = get_model()
    # For PyTorch ZeRO2/ZeRO3 we use FSDP2 fully_shard, so skip eager device placement.
    if args.shard_choice not in {"pytorch_zero2", "pytorch_zero3"}:
        model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    if args.shard_choice == "baseline":
        model = SimpleDDP(model)
        grad_accum_steps = None
    elif args.shard_choice == "zero1":
        model = SimpleDDP(model)
        optim = ZeroOneSharding(optim)
        grad_accum_steps = None
    elif args.shard_choice == "zero2":
        model = SimpleDDP(model)
        optim = Zero2Sharding(optim)
        # Override the default gradient synchronization function in DDP to use
        # the custom one defined in Zero2Sharding
        model.set_gradient_sync_fn(optim.shard_gradients)
        grad_accum_steps = None
    elif args.shard_choice == "zero3":
        model = SimpleDDP(model)
        optim = Zero3Sharding(optim)
        # Override the default pre-forward function in DDP to materialize full parameters before forward
        model.set_pre_forward_fn(optim.gather_full_parameters)
        model.set_gradient_sync_fn(optim.shard_gradients)
        grad_accum_steps = None
    elif args.shard_choice == "pytorch_zero1":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        grad_accum_steps = None
        # Ideally we would like to set overlap_with_ddp=True to overlap the communication of sharding
        # with DDP's all-reduce of gradients.
        optim = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.AdamW, lr=5e-5, overlap_with_ddp=False
        )
    elif args.shard_choice == "pytorch_zero2":
        # FSDP2 ZeRO2-like mode: keep full params after forward to reduce all-gather frequency.
        model = wrap_with_fsdp2(model, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
        grad_accum_steps = None
    elif args.shard_choice == "pytorch_zero3":
        # FSDP2 ZeRO3-like mode: reshard after forward for maximum memory savings.
        model = wrap_with_fsdp2(model, reshard_after_forward=True)
        optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
        grad_accum_steps = None

    profile_dir = f"profile/{args.shard_choice}"
    print(f"Training started on rank {local_rank}...")
    model = train_loop(
        model,
        train_loader,
        optim,
        device=device,
        epochs=EPOCHS,
        profile_dir=profile_dir,
        grad_accum_steps=grad_accum_steps,
        is_async=False,
        is_hook=False,
        memory_log_interval=1,  # log memory every batch
    )
    evaluate(model, eval_loader, device=device)

    ddp_cleanup()
