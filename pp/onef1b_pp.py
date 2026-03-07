"""1F1B pipeline parallel implementation."""

import torch
import torch.distributed as dist

from pp.base_pp import BasePipeline


class OneFOneBPipeline(BasePipeline):
    """One forward One backward pipeline parallel.

    Batch is split into microbatches m0..mN.
    1F1B runs in three phases:
        1. Warmup: forward-only to fill the pipeline.
        2. Steady state: alternate one backward and one forward per step.
        3. Drain: backward-only for remaining microbatches.

    Warmup count per stage:
        warmup_steps = min(num_stages - stage - 1, num_microbatches)

    Example (`num_stages=4`, `num_microbatches=8`):
        - stage 0: warmup=3, steady=5, drain=3
        - stage 1: warmup=2, steady=6, drain=2
        - stage 2: warmup=1, steady=7, drain=1
        - stage 3: warmup=0, steady=8, drain=0

    Example (`num_stages=4`, `num_microbatches=4`):
        S0: warmup[F0 F1 F2] steady[B0 F3] drain[B1 B2 B3]
        S1: warmup[F0 F1]    steady[B0 F2 B1 F3] drain[B2 B3]
        S2: warmup[F0]       steady[B0 F1 B1 F2 B2 F3] drain[B3]
        S3: warmup[]         steady[F0 B0 F1 B1 F2 B2 F3 B3] drain[]

    Time-axis view (`num_stages=4`, `num_microbatches=4`):
        t0: S0[F0] S1[  ] S2[  ] S3[  ]
        t1: S0[F1] S1[F0] S2[  ] S3[  ]
        t2: S0[F2] S1[F1] S2[F0] S3[  ]
        t3: S0[  ] S1[  ] S2[  ] S3[F0]
        t4: S0[  ] S1[  ] S2[  ] S3[B0]
        t5: S0[  ] S1[  ] S2[B0] S3[F1]
        t6: S0[  ] S1[B0] S2[F1] S3[B1]
        t7: S0[B0] S1[F2] S2[B1] S3[F2]
        t8: S0[F3] S1[B1] S2[F2] S3[B2]
        t9: S0[B1] S1[F3] S2[B2] S3[F3]
        t10:S0[  ] S1[B2] S2[F3] S3[B3]
        t11:S0[B2] S1[  ] S2[B3] S3[  ]
        t12:S0[  ] S1[B3] S2[  ] S3[  ]
        t13:S0[B3] S1[  ] S2[  ] S3[  ]
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
        self.activation_recv_buffers = [
            torch.empty(in_shape, dtype=act_dtype, device=self.device) if in_shape else None
            for _ in range(self.num_microbatches)
        ]
        self.gradient_recv_buffers = [
            torch.empty(grad_shape, dtype=act_dtype, device=self.device) if grad_shape else None
            for _ in range(self.num_microbatches)
        ]
        self._saved_input = [None] * self.num_microbatches
        self._saved_output = [None] * self.num_microbatches

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
        """Run one non-interleaved 1F1B step over `num_microbatches`.

        Training steps:
            1. Split batch tensors along dim=0 into microbatches.
            2. Warmup: run `warmup_steps` forward-only micros for this stage.
            3. Steady state: each step performs one backward micro and one forward micro.
               (Ordering is backward->forward on non-last stages to avoid deadlocks with
               blocking point-to-point send/recv.)
            4. Drain: run remaining backward-only micros.
            5. Run one optimizer step on this stage.

        Returns:
            Final microbatch loss scalar on last stage, otherwise `None`.
        """
        assert self.num_microbatches > 1, "1F1B requires num_microbatches>1"

        self.stage_opt.zero_grad(set_to_none=True)

        # Chunk the batch into microbatches and perform forward and backward pass
        # across warmup/steady/drain phases.
        assert batch["input_ids"].size(0) % self.num_microbatches == 0, (
            "Batch size must be divisible by num_microbatches"
        )
        chunks = {k: v.chunk(self.num_microbatches, dim=0) for k, v in batch.items()}
        micro_batches = [{k: chunks[k][i] for k in chunks} for i in range(self.num_microbatches)]

        self.losses = [None] * self.num_microbatches

        def forward_micro(micro_batch_idx: int) -> None:
            """Run one microbatch forward for this stage."""
            micro_batch = micro_batches[micro_batch_idx]

            # First stage, we run the forward pass on the input batch
            # and send the activations to the next stage.
            if self.is_first:
                input_ids = micro_batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = micro_batch["attention_mask"].to(self.device, non_blocking=True)
                out = self.stage_module(input_ids, attention_mask=attention_mask)
                self._saved_output[micro_batch_idx] = out
                if not self.is_last:
                    self._send(out, dst=self.stage + 1)
            # Last stage, we receive the activations from the previous stage,
            # run the forward pass to get logits and calculate the loss with the labels.
            # Intermediate stage, we receive the activations from the previous stage,
            # run the forward pass, and send the activations to the next stage.
            else:
                buf = self._recv(
                    buf=self.activation_recv_buffers[micro_batch_idx], src=self.stage - 1
                )
                buf = buf.detach()
                buf.requires_grad_()
                self._saved_input[micro_batch_idx] = buf

                attention_mask = micro_batch["attention_mask"].to(self.device, non_blocking=True)
                out = self.stage_module(buf, attention_mask=attention_mask)

                if self.is_last:
                    labels = micro_batch["labels"].to(self.device, non_blocking=True)
                    self.losses[micro_batch_idx] = self.loss_fn(
                        out, labels, attention_mask=attention_mask
                    )
                else:
                    self._saved_output[micro_batch_idx] = out
                    self._send(out, dst=self.stage + 1)

        def backward_micro(micro_batch_idx: int) -> None:
            """Run one microbatch backward for this stage."""
            # Last stage starts the backward pass by calling `loss.backward()`,
            # then sends the input gradient to the previous stage.
            if self.is_last:
                # Match full-batch mean-loss scaling across microbatches.
                (self.losses[micro_batch_idx] / self.num_microbatches).backward()
                if not self.is_first:
                    self._send(self._saved_input[micro_batch_idx].grad, dst=self.stage - 1)
            # Intermediate stage receives the input gradient from the next stage,
            # runs backward on the intermediate activation,
            # and sends the gradient of the input activation to the previous stage.
            # First stage receives the input gradient from the next stage
            # and runs backward on the input activation.
            else:
                grad_to_recv = self._recv(
                    buf=self.gradient_recv_buffers[micro_batch_idx], src=self.stage + 1
                )
                self._saved_output[micro_batch_idx].backward(grad_to_recv)
                if not self.is_first:
                    self._send(self._saved_input[micro_batch_idx].grad, dst=self.stage - 1)

        # A stage can only start backward after gradients arrive from downstream stages.
        # Earlier stages therefore need more forward-only warmup steps than later stages.
        warmup_steps = min(self.num_stages - self.stage - 1, self.num_microbatches)
        steady_steps = self.num_microbatches - warmup_steps

        # Warmup: forward-only.
        with torch.profiler.record_function("pp.forward_warmup"):
            for micro_batch_idx in range(warmup_steps):
                forward_micro(micro_batch_idx)

        # Steady state: 1 backward + 1 forward per step.
        with torch.profiler.record_function("pp.1f1b_steady"):
            for i in range(steady_steps):
                if self.is_last:
                    # Last stage has no warmup dependency on backward gradients.
                    forward_micro(i + warmup_steps)
                    backward_micro(i)
                else:
                    # Non-last stages receive gradients first to avoid send/send deadlocks
                    # with blocking point-to-point communication.
                    backward_micro(i)
                    forward_micro(i + warmup_steps)

        # Drain: backward-only for remaining micros.
        with torch.profiler.record_function("pp.backward_drain"):
            for micro_batch_idx in range(steady_steps, self.num_microbatches):
                backward_micro(micro_batch_idx)

        # Optimizer step for particular stage
        with torch.profiler.record_function("pp.optimizer_step"):
            self.stage_opt.step()
        final_loss = None
        if self.is_last:
            loss_vals = [loss.detach() for loss in self.losses if loss is not None]
            final_loss = torch.stack(loss_vals).mean().item() if loss_vals else None
        self._saved_input = [None] * self.num_microbatches
        self._saved_output = [None] * self.num_microbatches
        self.losses = [None] * self.num_microbatches
        return final_loss
