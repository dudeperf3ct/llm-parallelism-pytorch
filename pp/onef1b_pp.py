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

    Communication strategy (following PyTorch's ``Schedule1F1B``):
        Uses ``dist.batch_isend_irecv`` to fuse complementary send+recv ops
        into a single batched call.  This avoids NCCL deadlocks because:
        1. All ops in the batch are submitted as one coalesced group, so there
           is no per-op lazy communicator creation that requires lockstep.
        2. isend/irecv within a batch are non-blocking and independent, so
           there is no FIFO serialization conflict on the same communicator.
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
        self._p2p_initialized = False

    # ── P2P initialization ──────────────────────────────────────────

    def _initialize_p2p(self) -> None:
        """Pre-warm NCCL P2P channels by exchanging dummy tensors with neighbors.

        Following PyTorch's ``_get_init_p2p_neighbors_ops``: the first
        ``batch_isend_irecv`` on a communicator must involve ALL ranks so that
        NCCL can lazily create the P2P channels without deadlocking.  We send a
        small dummy in both directions (fwd & bwd) between every pair of
        adjacent stages in one batched call.
        """
        if self._p2p_initialized:
            return
        dummy = torch.zeros(1, device=self.device)
        ops: list[dist.P2POp] = []
        # Forward direction: recv from prev, send to next
        if not self.is_first:
            ops.append(dist.P2POp(dist.irecv, dummy.clone(), self.stage - 1, self.pp_group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.isend, dummy.clone(), self.stage + 1, self.pp_group))
        # Backward direction: recv from next, send to prev
        if not self.is_last:
            ops.append(dist.P2POp(dist.irecv, dummy.clone(), self.stage + 1, self.pp_group))
        if not self.is_first:
            ops.append(dist.P2POp(dist.isend, dummy.clone(), self.stage - 1, self.pp_group))
        self._exec_p2p(ops)
        self._p2p_initialized = True

    # ── P2P helpers (build ops, don't execute) ────────────────────────

    def _fwd_recv_ops(self, micro_batch_idx: int) -> list[dist.P2POp]:
        """Return irecv op for receiving activation from prev stage."""
        if self.is_first:
            return []
        buf = self.activation_recv_buffers[micro_batch_idx]
        return [dist.P2POp(dist.irecv, buf, self.stage - 1, self.pp_group)]

    def _fwd_send_ops(self, micro_batch_idx: int) -> list[dist.P2POp]:
        """Return isend op for sending activation to next stage."""
        if self.is_last:
            return []
        out = self._saved_output[micro_batch_idx]
        return [dist.P2POp(dist.isend, out.contiguous(), self.stage + 1, self.pp_group)]

    def _bwd_recv_ops(self, micro_batch_idx: int) -> list[dist.P2POp]:
        """Return irecv op for receiving gradient from next stage."""
        if self.is_last:
            return []
        buf = self.gradient_recv_buffers[micro_batch_idx]
        return [dist.P2POp(dist.irecv, buf, self.stage + 1, self.pp_group)]

    def _bwd_send_ops(self, micro_batch_idx: int) -> list[dist.P2POp]:
        """Return isend op for sending gradient to prev stage."""
        if self.is_first:
            return []
        grad = self._saved_input[micro_batch_idx].grad
        return [dist.P2POp(dist.isend, grad.contiguous(), self.stage - 1, self.pp_group)]

    @staticmethod
    def _exec_p2p(ops: list[dist.P2POp]) -> None:
        """Submit batched P2P ops and wait for completion."""
        if not ops:
            return
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # ── Compute helpers (no communication) ────────────────────────────

    def _forward_compute(self, micro_batch_idx: int, micro_batches: list[dict]) -> None:
        """Run forward computation for one microbatch (no P2P)."""
        micro_batch = micro_batches[micro_batch_idx]

        if self.is_first:
            input_ids = micro_batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = micro_batch["attention_mask"].to(self.device, non_blocking=True)
            out = self.stage_module(input_ids, attention_mask=attention_mask)
            self._saved_output[micro_batch_idx] = out
        else:
            buf = self.activation_recv_buffers[micro_batch_idx]
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

    def _backward_compute(self, micro_batch_idx: int) -> None:
        """Run backward computation for one microbatch (no P2P)."""
        if self.is_last:
            (self.losses[micro_batch_idx] / self.num_microbatches).backward()
        else:
            grad = self.gradient_recv_buffers[micro_batch_idx]
            self._saved_output[micro_batch_idx].backward(grad)

    def run_batch(self, batch):
        """Run one non-interleaved 1F1B step over `num_microbatches`.

        Uses ``batch_isend_irecv`` to fuse complementary send+recv ops,
        following the same pattern as PyTorch's ``Schedule1F1B``.

        Returns:
            Final microbatch loss scalar on last stage, otherwise `None`.
        """
        assert self.num_microbatches > 1, "1F1B requires num_microbatches>1"

        self._initialize_p2p()
        self.stage_opt.zero_grad(set_to_none=True)

        assert batch["input_ids"].size(0) % self.num_microbatches == 0, (
            "Batch size must be divisible by num_microbatches"
        )
        chunks = {k: v.chunk(self.num_microbatches, dim=0) for k, v in batch.items()}
        micro_batches = [{k: chunks[k][i] for k in chunks} for i in range(self.num_microbatches)]

        self.losses = [None] * self.num_microbatches

        warmup_steps = min(self.num_stages - self.stage - 1, self.num_microbatches)
        steady_steps = self.num_microbatches - warmup_steps
        fwd_idx = 0
        bwd_idx = 0

        # ── Warmup: forward-only ──────────────────────────────────────
        # Each iteration: recv activation, compute forward, send activation.
        fwd_sends: list[dist.P2POp] = []
        with torch.profiler.record_function("pp.forward_warmup"):
            for _ in range(warmup_steps):
                # Recv activation from prev stage (if not first).
                fwd_recvs = self._fwd_recv_ops(fwd_idx)
                self._exec_p2p(fwd_recvs)

                self._forward_compute(fwd_idx, micro_batches)

                # Send activation to next stage — hold the ops for fusing
                # with the first backward recv in steady state.
                fwd_sends = self._fwd_send_ops(fwd_idx)
                # For all warmup steps except the last, fire immediately.
                if fwd_idx != warmup_steps - 1:
                    self._exec_p2p(fwd_sends)
                    fwd_sends = []

                fwd_idx += 1

        # ── Last-stage first forward ─────────────────────────────────
        # The last stage has warmup_steps=0, so no forward has run yet.
        # We must do one forward before the steady-state backward-first loop.
        if warmup_steps == 0:
            fwd_recvs = self._fwd_recv_ops(fwd_idx)
            self._exec_p2p(fwd_recvs)
            self._forward_compute(fwd_idx, micro_batches)
            fwd_sends = self._fwd_send_ops(fwd_idx)
            fwd_idx += 1

        # ── Steady state: 1B + 1F per step ────────────────────────────
        # Following PyTorch's Schedule1F1B pattern:
        #   1. Fuse last fwd_send + bwd_recv  →  execute  →  backward compute
        #   2. Fuse bwd_send + fwd_recv        →  execute  →  forward compute
        #   3. Save fwd_send for next iteration (or drain)
        with torch.profiler.record_function("pp.1f1b_steady"):
            for i in range(steady_steps):
                # ── Backward half ─────────────────────────────────────
                # Fuse: send prev forward's activation + recv gradient.
                bwd_recvs = self._bwd_recv_ops(bwd_idx)
                self._exec_p2p(fwd_sends + bwd_recvs)

                self._backward_compute(bwd_idx)
                bwd_sends = self._bwd_send_ops(bwd_idx)
                bwd_idx += 1

                # ── Forward half ──────────────────────────────────────
                if fwd_idx < self.num_microbatches:
                    # Fuse: send gradient + recv next activation.
                    fwd_recvs = self._fwd_recv_ops(fwd_idx)
                    self._exec_p2p(bwd_sends + fwd_recvs)

                    self._forward_compute(fwd_idx, micro_batches)
                    fwd_sends = self._fwd_send_ops(fwd_idx)
                    fwd_idx += 1
                else:
                    # No more forwards; just send the gradient.
                    self._exec_p2p(bwd_sends)
                    fwd_sends = []

        # ── Drain: backward-only ──────────────────────────────────────
        with torch.profiler.record_function("pp.backward_drain"):
            for _ in range(warmup_steps):
                # Fuse: send prev forward's activation + recv gradient.
                bwd_recvs = self._bwd_recv_ops(bwd_idx)
                self._exec_p2p(fwd_sends + bwd_recvs)
                fwd_sends = []

                self._backward_compute(bwd_idx)
                bwd_sends = self._bwd_send_ops(bwd_idx)
                self._exec_p2p(bwd_sends)
                bwd_idx += 1

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
