import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from transformers.models.distilbert.modeling_distilbert import _prepare_4d_attention_mask_for_sdpa

from data import prepare_data
from model import get_pp_model, get_pp_tokenizer
from pp.gpipe_pp import GPipePipeline
from pp.naive_pp import NaivePipeline
from pp.onef1b_pp import OneFOneBPipeline
from pp.pytorch_engine import PytorchPPEngine
from pp.scratch_engine import ScratchPPEngine
from utils.ddp_utils import ddp_cleanup, ddp_initialize, get_dist_info
from utils.train_utils import set_seed, train_loop_pp

GLOBAL_BATCH_SIZE = 24
NUM_MICROBATCHES = 8
# Create a fixed length dataset to scratch implementation of pipelines easy
SCRATCH_FIXED_SEQ_LEN = 256
EPOCHS = 20
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


def get_pp_layers(model: torch.nn.Module) -> nn.ModuleList:
    """Return the encoder layers used for PP splitting."""
    if not hasattr(model, "distilbert") or not hasattr(model.distilbert, "transformer"):
        raise ValueError("Expected DistilBERT backbone for pipeline parallelism.")
    return model.distilbert.transformer.layer


def get_pp_hidden_size(model: torch.nn.Module) -> int:
    """Return the hidden width of the PP model."""
    return getattr(model.config, "dim", model.config.hidden_size)


def stage_bounds(n_layers: int, num_stages: int, rank: int) -> tuple[int, int]:
    # even split with remainder on early ranks
    base = n_layers // num_stages
    rem = n_layers % num_stages
    start = rank * base + min(rank, rem)
    end = start + base + int(rank < rem)
    return start, end


def build_stage_module(
    model: torch.nn.Module,
    num_stages: int,
    rank: int,
):
    """Build one DistilBERT stage module for either scratch or PyTorch PP."""
    layers = get_pp_layers(model)
    n_layers = len(layers)
    start, end = stage_bounds(n_layers, num_stages, rank)
    is_first = rank == 0
    is_last = rank == num_stages - 1

    class ScratchStageModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = model.distilbert.embeddings if is_first else None
            self.layers = nn.ModuleList(layers[start:end])
            self.pre_classifier = getattr(model, "pre_classifier", None) if is_last else None
            self.dropout = getattr(model, "dropout", None) if is_last else None
            self.classifier = getattr(model, "classifier", None) if is_last else None
            self._attention_mask_chunks: tuple[torch.Tensor, ...] = ()
            self._next_mask_idx = 0

        def prepare_microbatch_attention_mask(
            self, attention_mask: torch.Tensor, num_microbatches: int
        ) -> None:
            """Cache the local attention-mask chunks for PyTorch PP schedules."""
            device = next(self.parameters()).device
            self._attention_mask_chunks = tuple(
                chunk.to(device, non_blocking=True)
                for chunk in attention_mask.chunk(num_microbatches, dim=0)
            )
            self._next_mask_idx = 0

        def _resolve_attention_mask(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            """Use explicit mask when present, otherwise consume the next cached chunk."""
            if attention_mask is None:
                if self._next_mask_idx < len(self._attention_mask_chunks):
                    attention_mask = self._attention_mask_chunks[self._next_mask_idx]
                    self._next_mask_idx += 1
                else:
                    attention_mask = torch.ones(
                        hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool
                    )
            return attention_mask

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
            hidden_states = self.embeddings(x) if self.embeddings is not None else x
            attention_mask = self._resolve_attention_mask(hidden_states, attention_mask)
            attention_mask_2d = attention_mask.to(
                hidden_states.device, dtype=torch.bool, non_blocking=True
            )
            attention_mask = attention_mask_2d
            if model.config._attn_implementation == "sdpa":
                attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask,
                    hidden_states.dtype,
                    tgt_len=hidden_states.shape[1],
                )

            for layer in self.layers:
                # Encoder block preserves hidden shape: [B, S, H] -> [B, S, H].
                out = layer(hidden_states, attn_mask=attention_mask)
                hidden_states = out[0] if isinstance(out, tuple) else out

            if self.classifier is not None:
                pooled_output = hidden_states[:, 0]
                if self.pre_classifier is not None:
                    pooled_output = self.pre_classifier(pooled_output)
                    pooled_output = F.relu(pooled_output)
                if self.dropout is not None:
                    pooled_output = self.dropout(pooled_output)
                return self.classifier(pooled_output)
            # Scratch stages send hidden states [B, S, H] only.
            return hidden_states

    return ScratchStageModule()


def split_model_for_scratch(model: torch.nn.Module, num_stages: int, rank: int):
    """Build rank-local module shard for scratch PP."""
    return build_stage_module(
        model,
        num_stages,
        rank,
    )


def build_pytorch_stage(
    model: torch.nn.Module,
    rank: int,
    device: torch.device,
    pp_group,
):
    """Build a manual PipelineStage for PyTorch PP.

    The automatic `pipeline(...)` frontend traces the full graph and then infers
    stage boundaries. On this DistilBERT classifier path it fails during
    backward setup with `Backward of skip connections not supported yet`.
    Manual stage construction keeps the cross-stage graph linear by explicitly
    returning `(hidden_states, attention_mask)` between stages.
    """
    stage_module = build_stage_module(
        model,
        dist.get_world_size(pp_group),
        rank,
    )
    stage_module = stage_module.to(device)
    stage = PipelineStage(
        stage_module,
        rank,
        dist.get_world_size(pp_group),
        device,
        group=pp_group,
    )
    return stage


def pp_loss_fn(outputs, labels, attention_mask=None):
    """Cross entropy from last-stage output for sequence classification."""
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, dict) and "logits" in outputs:
        logits = outputs["logits"]
    else:
        logits = outputs

    # Scratch split for decoder-style models can produce token-level logits [B, S, C].
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
    pp_tokenizer = get_pp_tokenizer()
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
        tokenizer=pp_tokenizer,
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

    model = get_pp_model()

    if args.pp_choice == "pytorch_gpipe_pp":
        stage = build_pytorch_stage(model, pp_rank, device, pp_group)
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
        stage = build_pytorch_stage(model, pp_rank, device, pp_group)
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
        hidden = get_pp_hidden_size(model)

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
        hidden = get_pp_hidden_size(model)

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
        hidden = get_pp_hidden_size(model)
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
