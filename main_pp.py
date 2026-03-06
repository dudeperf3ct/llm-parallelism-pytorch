import argparse
from importlib import import_module

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe, SplitPoint, pipeline
from transformers.models.llama.modeling_llama import create_causal_mask

from data import prepare_data
from model import get_model
from pp.gpipe_pp import GPipePipeline
from pp.naive_pp import NaivePipeline
from pp.pytorch_engine import PytorchPPEngine
from pp.scratch_engine import ScratchPPEngine
from utils.ddp_utils import ddp_cleanup, ddp_initialize, get_dist_info
from utils.train_utils import set_seed, train_loop_pp

# Module name starts with a digit, so load via importlib.
OneFOneBPipeline = import_module("pp.1f1b_pp").OneFOneBPipeline

GLOBAL_BATCH_SIZE = 32
NUM_MICROBATCHES = 4
# Create a fixed length dataset to scratch implementation of pipelines easy
SCRATCH_FIXED_SEQ_LEN = 256
EPOCHS = 10
SEED = 42

parser = argparse.ArgumentParser(
    description="Distributed Training Example using Pipeline Parallelism"
)
parser.add_argument(
    "--pp-choice",
    type=str,
    choices=[
        "naive_pp",
        "gpipe_pp",
        "1f1b_pp",
        "pytorch_gpipe_pp",
        "pytorch_1f1b_pp",
    ],
    default="pytorch_gpipe_pp",
)
args = parser.parse_args()


def stage_bounds(n_layers: int, num_stages: int, rank: int) -> tuple[int, int]:
    # even split with remainder on early ranks
    base = n_layers // num_stages
    rem = n_layers % num_stages
    start = rank * base + min(rank, rem)
    end = start + base + int(rank < rem)
    return start, end


def split_model_for_scratch(model: torch.nn.Module, num_stages: int, rank: int):
    """Build rank-local module shard for scratch PP."""

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected model.model.layers for splitting the model.")

    n_layers = len(model.model.layers)
    start, end = stage_bounds(n_layers, num_stages, rank)
    is_first = rank == 0
    is_last = rank == num_stages - 1

    class ScratchStageModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = model.model.embed_tokens if is_first else None
            self.layers = nn.ModuleList(model.model.layers[start:end])
            self.norm = getattr(model.model, "norm", None) if is_last else None
            self.dropout = getattr(model, "dropout", None) if is_last else None
            self.classifier = getattr(model, "score", None) if is_last else None

        def forward(self, x, attention_mask=None):
            """Run this stage shard.

            Args:
                x: First stage expects token ids [B, S]; other stages expect hidden states [B, S, H].
                attention_mask: Optional mask [B, S] propagated across stages.

            Returns:
                Hidden states [B, S, H] for non-last stages, or classifier output on last stage.
            """
            # Stage 0: token ids [B, S] -> embeddings [B, S, H].
            # Other stages: x is already hidden states [B, S, H].
            hidden_states = self.embed_tokens(x) if self.embed_tokens is not None else x
            if attention_mask is None:
                attention_mask = torch.ones(
                    hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long
                )
            else:
                attention_mask = attention_mask.to(hidden_states.device, non_blocking=True)

            # Llama decoder blocks expect causal mask + RoPE position embeddings
            # prepared at model scope; rebuild them for this stage-local path.
            seq_len = hidden_states.shape[1]
            cache_position = torch.arange(seq_len, device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask = create_causal_mask(
                config=model.model.config,
                input_embeds=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

            for layer in self.layers:
                # Decoder block preserves hidden shape: [B, S, H] -> [B, S, H].
                out = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out

            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.dropout is not None:
                hidden_states = self.dropout(hidden_states)
            if self.classifier is not None:
                # Current head path produces token-level logits [B, S, C].
                return self.classifier(hidden_states)
            # Non-last stages send hidden states [B, S, H] to the next stage.
            return hidden_states

    return ScratchStageModule()


def build_uniform_split_spec(model: torch.nn.Module, num_stages: int) -> dict[str, SplitPoint]:
    """Build split points by evenly partitioning decoder layers across stages."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected model.model.layers for building split points.")

    n_layers = len(model.model.layers)
    layers_per_stage = max(1, n_layers // num_stages)
    split_spec: dict[str, SplitPoint] = {}
    for stage_idx in range(1, num_stages):
        boundary = stage_idx * layers_per_stage
        if boundary < n_layers:
            split_spec[f"model.layers.{boundary}"] = SplitPoint.BEGINNING
    return split_spec


def wrap_pytorch_pipeline(
    model: torch.nn.Module,
    example_input_ids: torch.Tensor,
    example_attention_mask: torch.Tensor,
    rank: int,
    device: torch.device,
    pp_group,
):
    """Wrap the model to split it across ranks.

    Here build_uniform_split_spec is used to split the model into equal parts
    across the pipeline stages.
    """
    # The split spec defines where to split the model for pipeline parallelism.
    split_spec = build_uniform_split_spec(model, dist.get_world_size(pp_group))
    pipe = pipeline(
        module=model,
        mb_args=(example_input_ids,),
        mb_kwargs={"attention_mask": example_attention_mask},
        split_spec=split_spec,
    )
    return pipe.build_stage(rank, device, pp_group)


def pp_loss_fn(outputs, labels, attention_mask=None):
    """Cross entropy from last-stage output for sequence classification."""
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict) and "logits" in outputs:
        logits = outputs["logits"]
    else:
        logits = outputs

    # Scratch split can produce token-level logits [B, S, C].
    # Convert to sequence logits [B, C] using the last non-pad token when mask is available.
    if logits.ndim == 3:
        if attention_mask is not None:
            attention_mask = attention_mask.to(logits.device)
            last_token_idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        else:
            last_token_idx = torch.full(
                (logits.size(0),), logits.size(1) - 1, device=logits.device, dtype=torch.long
            )
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        logits = logits[batch_idx, last_token_idx]

    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    set_seed(SEED)
    ddp_initialize()
    global_rank, world_size, local_rank = get_dist_info()
    pp_group = dist.group.WORLD
    pp_rank = dist.get_rank(pp_group)
    pp_world_size = dist.get_world_size(pp_group)
    per_stage_batch = GLOBAL_BATCH_SIZE
    use_static_shapes = args.pp_choice in {"naive_pp", "gpipe_pp", "1f1b_pp"}
    print(f"Rank: {global_rank}, World Size: {world_size}, Local Rank: {local_rank}")
    device = torch.device(f"cuda:{local_rank}")
    num_stages = pp_world_size

    if global_rank == 0:
        print(f"Number of devices: {world_size}")
        print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
        print(f"Per-stage batch size: {per_stage_batch}")
        print(f"Number of stages: {num_stages}")

    # PP is model parallel, not data parallel:
    # every stage must process the same samples, so use GLOBAL_BATCH_SIZE per stage
    # and disable dataset sharding across ranks.
    # NOTE: Scratch PP uses fixed-size recv buffers, so we enforce static sequence
    # length and drop incomplete train batches for stable boundary tensor shapes.
    print(f"Preparing data on rank {global_rank}...")
    train_loader, _ = prepare_data(
        per_stage_batch,
        global_rank,
        world_size,
        shard_data=False,
        static_shapes=use_static_shapes,
        fixed_seq_len=SCRATCH_FIXED_SEQ_LEN if use_static_shapes else None,
        drop_last_train=use_static_shapes,
    )
    # Sample a batch to use for tracing the model and building the pipeline stages.
    # It infers the shape of activations that will be sent between stages,
    # which is needed for setting up the pipeline schedule.
    # This is used to allocate buffers for receiving activations from the previous stage.
    sample_batch = next(iter(train_loader))
    microbatch_size = max(1, sample_batch["input_ids"].shape[0] // NUM_MICROBATCHES)
    example_input_ids = sample_batch["input_ids"][:microbatch_size].clone()
    example_attention_mask = sample_batch["attention_mask"][:microbatch_size].clone()

    model = get_model()

    if args.pp_choice == "pytorch_gpipe_pp":
        stage = wrap_pytorch_pipeline(
            model, example_input_ids, example_attention_mask, pp_rank, device, pp_group
        )
        schedule = ScheduleGPipe(stage, n_microbatches=NUM_MICROBATCHES, loss_fn=pp_loss_fn)
        optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=5e-5)
        engine = PytorchPPEngine(
            schedule=schedule,
            optimizer=optimizer,
            device=device,
            model_for_memory=stage.submod,
            pp_group=pp_group,
        )
    elif args.pp_choice == "pytorch_1f1b_pp":
        stage = wrap_pytorch_pipeline(
            model, example_input_ids, example_attention_mask, pp_rank, device, pp_group
        )
        schedule = Schedule1F1B(stage, n_microbatches=NUM_MICROBATCHES, loss_fn=pp_loss_fn)
        optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=5e-5)
        engine = PytorchPPEngine(
            schedule=schedule,
            optimizer=optimizer,
            device=device,
            model_for_memory=stage.submod,
            pp_group=pp_group,
        )
    elif args.pp_choice == "naive_pp":
        stage_module = split_model_for_scratch(model, num_stages, pp_rank).to(device)
        optimizer = torch.optim.AdamW(stage_module.parameters(), lr=5e-5)

        # For the naive pipeline, num_microbatches = 1
        micro_batch_size = per_stage_batch // 1
        seq_len = example_input_ids.shape[1]
        hidden = model.config.hidden_size  # for transformer hidden activations

        # We are fixing activation shape for scratch comm buffers.
        # This assumes fixed seq_len across steps; dynamic padding can violate it.
        # stage 0 gets token ids, stage>0 gets hidden states
        activation_shape = (micro_batch_size, seq_len, hidden)
        in_shape = None if pp_rank == 0 else activation_shape
        grad_shape = activation_shape  # gradients match boundary activation shape

        naive_pp_pipeline = NaivePipeline(
            num_microbatches=1,
            optimizer=optimizer,
            module=stage_module,
            stage=pp_rank,
            num_stages=num_stages,
            loss_fn=pp_loss_fn,
            in_shape=in_shape,
            grad_shape=grad_shape,
            pp_group=pp_group,
            device=device,
        )
        engine = ScratchPPEngine(pipeline_impl=naive_pp_pipeline)

    elif args.pp_choice == "gpipe_pp":
        stage_module = split_model_for_scratch(model, num_stages, pp_rank).to(device)
        optimizer = torch.optim.AdamW(stage_module.parameters(), lr=5e-5)

        seq_len = example_input_ids.shape[1]
        hidden = model.config.hidden_size  # for transformer hidden activations

        # We are fixing activation shape for scratch comm buffers.
        # This assumes fixed seq_len across steps; dynamic padding can violate it.
        # stage 0 gets token ids, stage>0 gets hidden states
        micro_batch_size = per_stage_batch // NUM_MICROBATCHES
        activation_shape = (micro_batch_size, seq_len, hidden)
        in_shape = None if pp_rank == 0 else activation_shape
        grad_shape = activation_shape  # gradients match boundary activation shape

        gpipe_pp_pipeline = GPipePipeline(
            num_microbatches=NUM_MICROBATCHES,
            optimizer=optimizer,
            module=stage_module,
            stage=pp_rank,
            num_stages=num_stages,
            loss_fn=pp_loss_fn,
            in_shape=in_shape,
            grad_shape=grad_shape,
            pp_group=pp_group,
            device=device,
        )
        engine = ScratchPPEngine(pipeline_impl=gpipe_pp_pipeline)
    elif args.pp_choice == "1f1b_pp":
        stage_module = split_model_for_scratch(model, num_stages, pp_rank).to(device)
        optimizer = torch.optim.AdamW(stage_module.parameters(), lr=5e-5)

        seq_len = example_input_ids.shape[1]
        hidden = model.config.hidden_size
        micro_batch_size = per_stage_batch // NUM_MICROBATCHES
        activation_shape = (micro_batch_size, seq_len, hidden)
        in_shape = None if pp_rank == 0 else activation_shape
        grad_shape = activation_shape

        onef1b_pipeline = OneFOneBPipeline(
            num_microbatches=NUM_MICROBATCHES,
            optimizer=optimizer,
            module=stage_module,
            stage=pp_rank,
            num_stages=num_stages,
            loss_fn=pp_loss_fn,
            in_shape=in_shape,
            grad_shape=grad_shape,
            pp_group=pp_group,
            device=device,
        )
        engine = ScratchPPEngine(pipeline_impl=onef1b_pipeline)

    profile_dir = f"profile/{args.pp_choice}"
    print(f"Training started on global rank {global_rank} (pp rank {pp_rank})...")
    train_loop_pp(
        engine=engine,
        data=train_loader,
        device=device,
        epochs=EPOCHS,
        profile_dir=profile_dir,
        memory_log_interval=1,
    )
    if global_rank == 0:
        print("Pipeline training done. Eval path is not wired yet for stage-only modules.")

    ddp_cleanup()
