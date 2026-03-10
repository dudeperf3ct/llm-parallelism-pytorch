# Debugging Scratch 1F1B Pipeline Parallelism

## Problem

The scratch implementation of 1F1B (One Forward One Backward) pipeline parallelism
(`pp/onef1b_pp.py`) deadlocked during training when run with 3 GPUs:

```bash
torchrun --standalone --nproc_per_node=3 main_pp.py --pp-choice 1f1b_pp
```

**Symptoms:** GPUs 0 and 1 at 100% utilization (spinning), GPU 2 at 0%. Training
never produced any output. The naive and GPipe scratch implementations worked fine
on the same setup.

---

## Root Cause Summary

Three distinct issues were identified, each building on the previous fix:

1. **NCCL lazy P2P channel initialization deadlock** — the primary deadlock
2. **Last-stage scheduling bug** — a forward-before-backward ordering issue

---

## Approach 1: Blocking `dist.send`/`dist.recv` → Send-Send Deadlock

### Original Code Pattern

```python
# Steady state: each stage does backward then forward
if not self.is_last:
    dist.send(activation, dst=self.stage + 1)  # send forward activation
if not self.is_first:
    dist.recv(grad_buf, src=self.stage - 1)     # recv backward gradient
```

### Problem

In the 1F1B steady state, adjacent stages need to exchange data in **both
directions simultaneously**. Stage 0 tries to `send` F1 to Stage 1, while Stage 1
tries to `send` B0 gradient to Stage 0. Both calls are blocking — each waits for
the other to post a matching `recv`, creating a classic **send-send deadlock**.

This doesn't happen in naive PP (fully sequential) or GPipe (all forwards then all
backwards) because they never have simultaneous bidirectional communication.

### Fix Attempted

Changed `dist.send` to `dist.isend` (non-blocking sends). **Result: still stuck.**

### Why It Failed

Non-blocking sends alone don't solve the problem. NCCL has deeper issues with P2P
operations on groups with more than 2 ranks (see Approach 2).

---

## Approach 2: Separate 2-Rank Subgroups → FIFO Serialization Deadlock

### Hypothesis

With a 3-rank group, each individual `isend`/`irecv` triggers **lazy NCCL
communicator creation** that requires ALL ranks to participate at the same sequence
point. Different ranks calling P2P ops in different orders deadlocks during comm
initialization.

**Evidence:** `NCCL_DEBUG=INFO` showed only channels `0→1` were created, never
`1→2`. GPU 2 never participated.

### Fix Attempted

Created dedicated 2-rank subgroups (`dist.new_group([i, i+1])`) for each pair of
adjacent stages. This avoids lazy init on the 3-rank world group.

### Result: Still stuck!

NCCL debug showed both subgroup comms were created successfully, but training still
deadlocked.

### Why It Failed

**NCCL FIFO serialization:** All operations from the same rank on the same
communicator are processed in FIFO order. On the single `(0,1)` subgroup, Stage 0
enqueues: `isend(F0)`, `isend(F1)`, `irecv(B0)`. The `irecv(B0)` can't start
until `isend(F1)` completes, but `isend(F1)` needs Stage 1's matching `recv`, which
Stage 1 can't post until its `isend(B0)` completes, which needs Stage 0's
`irecv(B0)` — **circular dependency**.

---

## Approach 3: 4 Separate Groups Per Pair (Not Tested)

### Idea

Create 4 separate groups per adjacent pair (fwd_send, fwd_recv, bwd_send,
bwd_recv), each used in only ONE direction per rank, to break FIFO serialization.

### Why We Moved On

This approach adds significant complexity (8+ process groups for 3 stages). The
question arose: **how does PyTorch's `Schedule1F1B` do this correctly without all
these groups?**

---

## Approach 4: `batch_isend_irecv` (PyTorch's Pattern) — The Solution

### Investigation

Reading PyTorch's `Schedule1F1B` source
(`torch/distributed/pipelining/schedules.py`)
revealed the key pattern:

1. **`dist.batch_isend_irecv`** fuses complementary send+recv ops into a single
   batched call using NCCL's `ncclGroupStart()`/`ncclGroupEnd()`.
2. **Complementary ops are always fused:**
   - `batch_isend_irecv(fwd_sends + bwd_recvs)` — send activation + recv gradient
   - `batch_isend_irecv(bwd_sends + fwd_recvs)` — send gradient + recv activation
3. This avoids all three deadlock issues:
   - No blocking send/recv conflicts (all ops are non-blocking within the batch)
   - No FIFO serialization (send and recv are submitted as one atomic group)
   - Lazy init is handled by a **P2P pre-warming step** (see below)

### Complete Rewrite

Separated P2P communication from computation:

- **Op builders** (`_fwd_recv_ops`, `_fwd_send_ops`, `_bwd_recv_ops`,
  `_bwd_send_ops`): return `dist.P2POp` objects without executing them.
- **`_exec_p2p(ops)`**: calls `dist.batch_isend_irecv(ops)` and waits.
- **Compute helpers** (`_forward_compute`, `_backward_compute`): pure computation,
  no P2P.
- **`run_batch`**: restructured with fused communication following the 1F1B pattern.

### Sub-issue: NCCL Lazy P2P Channel Init (Still Deadlocked!)

Even with `batch_isend_irecv`, the first call still deadlocked. The NCCL debug
output showed P2P channels were only created in one direction.

**Root cause:** The first `batch_isend_irecv` on a communicator triggers lazy NCCL
P2P channel creation. Different ranks calling `batch_isend_irecv` with different
peer sets at different times causes the lazy init to deadlock.

**Solution — P2P pre-warming** (following PyTorch's `_get_init_p2p_neighbors_ops`):

```python
def _initialize_p2p(self) -> None:
    """Pre-warm NCCL P2P channels with dummy tensors."""
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
```

This single call creates ALL P2P channels (both directions) between ALL adjacent
stages simultaneously. Every rank participates, so NCCL's lazy init succeeds.

**NCCL debug confirmed all 4 channels created:**

```
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
Channel 00 : 1[1] -> 2[2] via SHM/direct/direct
Channel 00 : 2[2] -> 1[1] via SHM/direct/direct
Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
```

### Sub-issue: Last-Stage Scheduling Bug

After fixing the deadlock, a new error appeared:

```
TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
```

at `self.losses[micro_batch_idx] / self.num_microbatches` on the last stage.

**Root cause:** The last stage has `warmup_steps = 0` (formula:
`min(num_stages - stage - 1, num_microbatches)`). The steady-state loop does
backward before forward. But with no warmup forwards, `losses[0]` is `None` when
the first backward tries to use it.

**Fix:** Added a pre-forward for the last stage before the steady-state loop:

```python
if warmup_steps == 0:
    fwd_recvs = self._fwd_recv_ops(fwd_idx)
    self._exec_p2p(fwd_recvs)
    self._forward_compute(fwd_idx, micro_batches)
    fwd_sends = self._fwd_send_ops(fwd_idx)
    fwd_idx += 1
```

---

## Final Working Solution

The complete fix required two changes:

1. **P2P pre-warming** (`_initialize_p2p`): A one-time `batch_isend_irecv` with
   dummy tensors in all 4 directions (fwd/bwd × send/recv) between adjacent stages.
   Called once at the start of the first `run_batch`.

2. **Last-stage pre-forward**: Before the backward-first steady-state loop, the
   last stage does one forward to compute `losses[0]`, so the first backward has
   something to work with.

3. **`batch_isend_irecv` for all P2P**: Every communication step fuses
   complementary send+recv ops into a single batched call, avoiding all NCCL
   deadlock scenarios.

### Key Takeaway

**NCCL P2P operations on groups with >2 ranks require careful initialization.**
PyTorch's pipeline schedules handle this with `_get_init_p2p_neighbors_ops()` — a
pre-warming step that creates all P2P channels before real work begins. Without
this, the first `batch_isend_irecv` calls trigger lazy channel creation that can
deadlock when different ranks contact different peers at different times.

---

## Extending Fixes to Naive PP and GPipe

### Problem

After fixing 1F1B, running the full experiment script (`run_experiment_pp.sh`)
revealed that naive PP and GPipe also deadlocked on 3 GPUs. They had previously
worked by lucky timing — the NCCL lazy init deadlock is **non-deterministic**.

The warning confirmed the issue:

```
An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.
In lazy initialization mode, this will result in a new 2-rank NCCL communicator
to be created.
```

### Why They Were Vulnerable

Both used unbatched `dist.send`/`dist.recv`, which on a group with >2 ranks
creates **separate 2-rank communicators** via lazy initialization. Each such
creation requires both involved ranks to call it at the same time. With 3+ stages
calling P2P ops in different orders, this is a race condition that sometimes
works and sometimes deadlocks.

### Fix Attempt: P2P Pre-warming Only

Added `_initialize_p2p()` (same as 1F1B) to both naive PP and GPipe. **Still
stuck!** The pre-warming initializes the main group communicator via
`batch_isend_irecv`, but the unbatched `dist.send`/`dist.recv` calls bypass it
entirely — they create separate per-pair communicators through a different NCCL
code path.

### Working Fix: `batch_isend_irecv` for All P2P

Changed `_send` and `_recv` in both naive PP and GPipe to use
`batch_isend_irecv` (even for single ops) instead of unbatched
`dist.send`/`dist.recv`. This ensures all P2P goes through the pre-warmed
communicator:

```python
def _send(self, inp: torch.Tensor, dst: int) -> None:
    ops = [dist.P2POp(dist.isend, inp.contiguous(), dst, self.pp_group)]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()

def _recv(self, buf: torch.Tensor, src: int) -> torch.Tensor:
    ops = [dist.P2POp(dist.irecv, buf, src, self.pp_group)]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    return buf
```

Combined with `_initialize_p2p()` pre-warming, this eliminates all lazy
communicator creation and works reliably across any number of stages.

---

## Why Naive PP and GPipe Originally Appeared to Work

Both use **sequential communication patterns** that are less likely to trigger
the NCCL lazy init deadlock — but are not immune to it:

- **Naive PP**: One microbatch flows through all stages sequentially. Each stage
  does recv→forward→send→recv_grad→backward→send_grad in order. Only one P2P
  operation is active at a time, so ranks often happen to call P2P in a compatible
  order — but this is not guaranteed.

- **GPipe**: All forwards complete before any backward starts. Communication is
  unidirectional within each phase — first all activations flow forward, then all
  gradients flow backward. No simultaneous bidirectional communication, but the
  lazy per-pair communicator creation can still deadlock non-deterministically.

**1F1B is fundamentally different** because adjacent stages communicate in **both
directions simultaneously** during the steady state (Stage N sends activation
forward while receiving gradient backward). This makes the deadlock deterministic
rather than a race condition, which is why it was the first to fail consistently.

**The universal fix**: Use `batch_isend_irecv` for all P2P communication (even
single ops) and pre-warm all channels with `_initialize_p2p()`. This applies to
all scratch pipeline implementations, not just 1F1B.
