"""Naive pipeline parallel implementation."""

import torch
import torch.distributed as dist

from pp.base_pp import BasePipeline


class NaivePipeline(BasePipeline):
    """Naive pipeline parallel.

    One full batch runs end-to-end forward, then backward.
    This leaves pipeline bubbles (idle slots) on most stages.

    Example (4 stages, 1 batch, 1 microbatch):
        t0: S0[F]  S1[ ]  S2[ ]  S3[ ]
        t1: S0[ ]  S1[F]  S2[ ]  S3[ ]
        t2: S0[ ]  S1[ ]  S2[F]  S3[ ]
        t3: S0[ ]  S1[ ]  S2[ ]  S3[F]
        t4: S0[ ]  S1[ ]  S2[ ]  S3[B]
        t5: S0[ ]  S1[ ]  S2[B]  S3[ ]
        t6: S0[ ]  S1[B]  S2[ ]  S3[ ]
        t7: S0[B]  S1[ ]  S2[ ]  S3[ ]
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
        grad_shape=None,
        act_dtype=torch.float32,
        device=None,
    ):
        super().__init__(stage, num_stages, module, optimizer, loss_fn, num_microbatches)
        self.pp_group = pp_group
        self.device = device if device is not None else torch.device(f"cuda:{stage}")
        # Placeholders used to recieve
        self.fwd_cache = (
            torch.empty(in_shape, dtype=act_dtype, device=self.device) if in_shape else None
        )
        self.bwd_cache = (
            torch.empty(grad_shape, dtype=act_dtype, device=self.device) if grad_shape else None
        )
        # Responsible for peak memory
        self._saved_activation = None
        self.loss = None

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
        """Run one full-batch pipeline step with a single microbatch.

        Per-stage behavior:
            - First stage: runs local forward from input tokens and sends activations downstream.
            - Middle stages: receive boundary activations, run local forward, send downstream.
            - Last stage: receive activations, compute loss from labels.

        Backward:
            - Last stage calls `loss.backward()` and sends input-activation gradients upstream.
            - Middle/first stages receive gradients from next stage and backprop through their
              saved boundary activation.

        Optimizer:
            - One `optimizer.step()` per stage after backward completes.

        Returns:
            Loss scalar on last stage, otherwise `None`.
        """
        assert self.num_microbatches == 1, "NaivePipeline only supports num_microbatches=1"

        self.stage_opt.zero_grad(set_to_none=True)
        self.loss = None

        def forward_step() -> None:
            """Run stage-local forward for the single microbatch."""
            # First stage, we run the forward pass on the input batch
            # and send the activations to the next stage.
            if self.is_first:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                out = self.stage_module(input_ids, attention_mask)
                self._saved_activation = out
                self._send(out, dst=self.stage + 1)
            # Last stage, we receive the activations from the previous stage,
            # run the forward pass to get logits and calculate the loss with the labels.
            elif self.is_last:
                buf = self._recv(buf=self.fwd_cache, src=self.stage - 1)
                # Explicitly marking require grads as cross rank communication breaks autograd history
                buf = buf.detach()
                buf.requires_grad_()
                self._saved_activation = buf
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                logits = self.stage_module(buf, attention_mask=attention_mask)
                labels = batch["labels"].to(self.device, non_blocking=True)
                self.loss = self.loss_fn(logits, labels, attention_mask=attention_mask)
            # Intermediate stage, we receive the activations from the previous stage,
            # run the forward pass, and send the activations to the next stage.
            else:
                buf = self._recv(buf=self.fwd_cache, src=self.stage - 1)
                # Explicitly marking require grads as cross rank communication breaks autograd history
                buf = buf.detach()
                buf.requires_grad_()
                self._saved_activation = buf
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                out = self.stage_module(buf, attention_mask=attention_mask)
                self._send(out, dst=self.stage + 1)

        def backward_step() -> None:
            """Run stage-local backward for the single microbatch."""
            # Last stage starts the backward pass by calling `loss.backward()`,
            # then sends the input gradient to the previous stage.
            if self.is_last:
                self.loss.backward()
                grad_to_send = self._saved_activation.grad
                self._send(grad_to_send, dst=self.stage - 1)
            # Intermediate stage receives the input gradient from the next stage,
            # runs backward on the intermediate activation,
            # and sends the gradient of the input activation to the previous stage.
            elif not self.is_first:
                grad_to_recv = self._recv(buf=self.bwd_cache, src=self.stage + 1)
                self._saved_activation.backward(grad_to_recv)
                grad_to_send = self._saved_activation.grad
                self._send(grad_to_send, dst=self.stage - 1)
            # First stage receives the input gradient from the next stage
            # and runs backward on the input activation.
            else:
                grad_to_recv = self._recv(buf=self.bwd_cache, src=self.stage + 1)
                # For stage 0, saved activation is the output we sent onward.
                self._saved_activation.backward(grad_to_recv)

        # Forward pass and calculate loss
        with torch.profiler.record_function("pp.forward"):
            forward_step()

        # Backward pass
        with torch.profiler.record_function("pp.backward"):
            backward_step()

        # Optimizer step for particular stage
        with torch.profiler.record_function("pp.optimizer_step"):
            self.stage_opt.step()
        self._saved_activation = None
        final_loss = self.loss.item() if self.is_last and self.loss is not None else None
        self.loss = None
        return final_loss
