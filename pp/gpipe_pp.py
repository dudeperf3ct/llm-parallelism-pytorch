"""Gpipe pipeline parallel implementation."""

import torch
import torch.distributed as dist

from pp.base_pp import BasePipeline


class GPipePipeline(BasePipeline):
    """GPipe pipeline parallel.

    Batch is split into microbatches m0..mN.
    First fill pipeline with forwards, then drain with backwards.

    Example (4 stages, 4 microbatches, 1 batch):
        t0: S0[F0] S1[  ] S2[  ] S3[  ]
        t1: S0[F1] S1[F0] S2[  ] S3[  ]
        t2: S0[F2] S1[F1] S2[F0] S3[  ]
        t3: S0[F3] S1[F2] S2[F1] S3[F0]
        t4: S0[  ] S1[F3] S2[F2] S3[F1]
        t5: S0[  ] S1[  ] S2[F3] S3[F2]
        t6: S0[  ] S1[  ] S2[  ] S3[F3]
        t7: S0[  ] S1[  ] S2[  ] S3[B3]
        t8: S0[  ] S1[  ] S2[B3] S3[B2]
        t9: S0[  ] S1[B3] S2[B2] S3[B1]
        t10:S0[B3] S1[B2] S2[B1] S3[B0]
    """

    def __init__(
        self,
        stage,
        num_stages,
        module,
        optimizer,
        loss_fn,
        num_microbatches,
        pp_group,
        in_shape,
        grad_shape,
        act_dtype=torch.float32,
        device=None,
    ):
        super().__init__(stage, num_stages, module, optimizer, loss_fn, num_microbatches)
        self.pp_group = pp_group
        self.device = device if device is not None else torch.device(f"cuda:{stage}")
        self.fwd_cache = [
            torch.empty(in_shape, dtype=act_dtype, device=self.device) if in_shape else None
            for _ in range(self.num_microbatches)
        ]
        self.bwd_cache = [
            torch.empty(grad_shape, dtype=act_dtype, device=self.device) if grad_shape else None
            for _ in range(self.num_microbatches)
        ]
        self._saved_input = [None] * self.num_microbatches
        self._saved_output = [None] * self.num_microbatches
        self.losses = [None] * self.num_microbatches

    def _recv(self, buf: torch.Tensor, src: int) -> torch.Tensor:
        """Receive a tensor from `src` into a preallocated buffer.

        Args:
            buf: Destination tensor to write into. Its shape/dtype/device must match sender tensor.
            src: Source rank within `self.pp_group`.

        Returns:
            The same buffer `buf`, filled with received values.

        Note:
            This is a blocking point-to-point receive (`dist.recv`) scoped to `self.pp_group`.
        """
        with torch.profiler.record_function("pp.comm.recv"):
            dist.recv(buf, src=src, group=self.pp_group)
        return buf

    def _send(self, inp: torch.Tensor, dst: int) -> None:
        """Send a tensor to `dst` using point-to-point communication.

        Args:
            inp: Tensor to send. It is sent as `inp.contiguous()` for communication safety.
            dst: Destination rank within `self.pp_group`.

        Note:
            This is a blocking point-to-point send (`dist.send`) scoped to `self.pp_group`.
        """
        with torch.profiler.record_function("pp.comm.send"):
            dist.send(inp.contiguous(), dst=dst, group=self.pp_group)

    def run_batch(self, batch):
        """Run one GPipe fill-and-drain step over `num_microbatches`.

        Forward phase:
            - Split input batch tensors along batch dimension into microbatches.
            - For each microbatch index (0..m-1), run local forward and communicate boundary
              activations to the next stage.
            - Store per-microbatch boundary activations needed for backward.
            - Last stage computes and stores one loss tensor per microbatch.

        Backward phase:
            - Iterate microbatches in reverse order (m-1..0).
            - Last stage backprops each stored loss and sends boundary gradients upstream.
            - Earlier stages receive upstream gradients and backprop through saved activations.

        Optimizer:
            - One `optimizer.step()` per stage after all microbatch backward passes complete.

        Returns:
            Final microbatch loss scalar on last stage, otherwise `None`.
        """
        assert self.num_microbatches > 1, "GPipe requires num_microbatches>1"

        self.stage_opt.zero_grad(set_to_none=True)

        # Chunk the batch into microbatches and perform forward and backward pass
        # for each microbatch sequentially.
        # Simplicity assumption microbatches is divisible by batch size
        assert batch["input_ids"].size(0) % self.num_microbatches == 0, (
            "Batch size must be divisible by num_microbatches"
        )
        chunks = {k: v.chunk(self.num_microbatches, dim=0) for k, v in batch.items()}
        micro_batches = [{k: chunks[k][i] for k in chunks} for i in range(self.num_microbatches)]

        def forward_micro(micro_batch_idx: int) -> None:
            """Run one microbatch forward for this stage."""
            micro_batch = micro_batches[micro_batch_idx]
            # First stage, we run the forward pass on the input batch
            # and send the activations to the next stage.
            if self.is_first:
                input_ids = micro_batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = micro_batch["attention_mask"].to(self.device, non_blocking=True)
                out = self.stage_module(input_ids, attention_mask)
                self._saved_output[micro_batch_idx] = out
                self._send(out, dst=self.stage + 1)
            # Last stage, we receive the activations from the previous stage,
            # run the forward pass to get logits and calculate the loss with the labels.
            elif self.is_last:
                buf = self._recv(buf=self.fwd_cache[micro_batch_idx], src=self.stage - 1)
                buf = buf.detach()
                buf.requires_grad_()
                self._saved_input[micro_batch_idx] = buf
                attention_mask = micro_batch["attention_mask"].to(self.device, non_blocking=True)
                logits = self.stage_module(buf, attention_mask=attention_mask)
                labels = micro_batch["labels"].to(self.device, non_blocking=True)
                self.losses[micro_batch_idx] = self.loss_fn(
                    logits, labels, attention_mask=attention_mask
                )
            # Intermediate stage, we receive the activations from the previous stage,
            # run the forward pass, and send the activations to the next stage.
            else:
                buf = self._recv(buf=self.fwd_cache[micro_batch_idx], src=self.stage - 1)
                buf = buf.detach()
                buf.requires_grad_()
                self._saved_input[micro_batch_idx] = buf
                attention_mask = micro_batch["attention_mask"].to(self.device, non_blocking=True)
                out = self.stage_module(buf, attention_mask=attention_mask)
                self._saved_output[micro_batch_idx] = out
                self._send(out, dst=self.stage + 1)

        def backward_micro(micro_batch_idx: int) -> None:
            """Run one microbatch backward for this stage."""
            # Last stage starts the backward pass by calling `loss.backward()`,
            # then sends the input gradient to the previous stage.
            if self.is_last:
                # Match full-batch mean-loss scaling across microbatches.
                (self.losses[micro_batch_idx] / self.num_microbatches).backward()
                grad_to_send = self._saved_input[micro_batch_idx].grad
                self._send(grad_to_send, dst=self.stage - 1)
            # Intermediate stage receives the input gradient from the next stage,
            # runs backward on the intermediate activation,
            # and sends the gradient of the input activation to the previous stage.
            elif not self.is_first:
                grad_to_recv = self._recv(buf=self.bwd_cache[micro_batch_idx], src=self.stage + 1)
                self._saved_output[micro_batch_idx].backward(grad_to_recv)
                grad_to_send = self._saved_input[micro_batch_idx].grad
                self._send(grad_to_send, dst=self.stage - 1)
            # First stage receives the input gradient from the next stage
            # and runs backward on the input activation.
            else:
                grad_to_recv = self._recv(buf=self.bwd_cache[micro_batch_idx], src=self.stage + 1)
                # For stage 0, saved activation is the output we sent onward.
                self._saved_output[micro_batch_idx].backward(grad_to_recv)

        # Forward pass and calculate loss
        with torch.profiler.record_function("pp.forward"):
            for micro_batch_idx in range(self.num_microbatches):
                forward_micro(micro_batch_idx)

        # Backward pass in reverse
        with torch.profiler.record_function("pp.backward"):
            for micro_batch_idx in range(self.num_microbatches - 1, -1, -1):
                backward_micro(micro_batch_idx)

        # Optimizer step for particular stage
        with torch.profiler.record_function("pp.optimizer_step"):
            self.stage_opt.step()
        self._saved_input = [None] * self.num_microbatches
        self._saved_output = [None] * self.num_microbatches
        final_loss = None
        if self.is_last:
            loss_vals = [loss.detach() for loss in self.losses if loss is not None]
            final_loss = torch.stack(loss_vals).mean().item() if loss_vals else None
        self.losses = [None] * self.num_microbatches
        return final_loss
