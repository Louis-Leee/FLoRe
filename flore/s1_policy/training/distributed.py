"""Process-group plumbing for multi-GPU S1 training (``algo='act'`` and ``'diffusion'``).

Every distributed concern lives HERE rather than being sprinkled through
:mod:`flore.s1_policy.training.trainer`, so the single-GPU path stays readable and a
non-``torchrun`` invocation provably behaves exactly as before: with no ``WORLD_SIZE`` in
the environment :func:`get_dist_info` reports ``world_size=1`` and every helper below is a
pass-through that touches neither ``torch.distributed`` nor the tensors it is handed.

Why native DDP and not a framework: ``trainer.py`` owns a mid-epoch-resumable sampler, a
full-config equality refusal, split-provenance + norm-stats hash gates, atomic safetensors
writes, EMA-as-deploy-weights and four W&B artifact types — each documented as the fix for
a real bug. A framework that owns the training loop would have to reproduce all of them.
DDP only needs three things from the loop (wrap for forward/backward, shard the sampler,
guard rank-0 side effects), which is what this module provides.

Nothing in this module is algorithm-specific: which presets are ALLOWED to use it is decided
by ``trainer._DDP_ALGOS``, on the strength of a per-algorithm gradient-parity test.

Launch with::

    torchrun --standalone --nproc-per-node 4 scripts/train_s1.py --algo act ...
    torchrun --standalone --nproc-per-node 8 scripts/train_s1.py --algo diffusion ...
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

#: NCCL's own default is 30 min; 10 is long enough for a multi-GB rank-0 checkpoint write
#: plus barrier, and short enough that a genuine deadlock surfaces within a coffee break
#: instead of looking like slow training.
_TIMEOUT = datetime.timedelta(minutes=10)


@dataclass(frozen=True)
class DistInfo:
    """Resolved topology. ``world_size == 1`` means every helper here is a no-op."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        """True when a process group is in play (i.e. launched under ``torchrun``)."""
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """True on the ONE rank allowed to write checkpoints and talk to W&B."""
        return self.rank == 0


def get_dist_info() -> DistInfo:
    """Read the topology ``torchrun`` publishes, without initializing anything.

    Deliberately env-only and side-effect free: the trainer needs the world size while
    building its dataloaders, which happens before (and independently of) any decision to
    create a process group.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistInfo()
    return DistInfo(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=world_size,
    )


def init_process_group(info: DistInfo) -> None:
    """Join the process group and bind this rank to its own GPU.

    ``set_device`` before any collective is load-bearing: without it every rank would
    default to ``cuda:0``, so all N processes would allocate on one card (OOM) and NCCL
    would deadlock trying to communicate with itself.
    """
    if not info.enabled or dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
    kwargs = {}
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        # Binding the group to this rank's device lets barrier() pick the right one instead
        # of inferring it from ambient context (which warns, and can guess wrong).
        kwargs["device_id"] = torch.device(f"cuda:{info.local_rank}")
    dist.init_process_group(backend=backend, timeout=_TIMEOUT, **kwargs)
    logger.info(
        "DDP rank %d/%d initialized (local_rank=%d, backend=%s)",
        info.rank, info.world_size, info.local_rank, backend,
    )


def destroy_process_group() -> None:
    """Tear the group down. Safe to call unconditionally, including after a crash."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:  # noqa: BLE001 - teardown must never mask the real error
            logger.warning("destroy_process_group failed: %s", exc)


def barrier() -> None:
    """Hold every rank until all arrive. Used after rank-0-only filesystem writes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across ranks, returning a new tensor (the input is untouched).

    Returned rather than done in place because callers hold float64 metric accumulators
    they re-use; mutating those under the caller's feet would make a second reduction
    double-count. Single-process callers get a plain clone, so the call site needs no
    branch.
    """
    if not dist.is_initialized():
        return tensor.clone()
    # NCCL only reduces device tensors. A CPU float64 accumulator (which is exactly what
    # the trainer's metric storage holds) must therefore make the round trip, and coming
    # back to the ORIGINAL device keeps the caller's arithmetic dtype/device-stable.
    work = tensor.detach().clone()
    if dist.get_backend() == "nccl" and work.device.type != "cuda":
        work = work.cuda()
    dist.all_reduce(work, op=dist.ReduceOp.SUM)
    return work.to(tensor.device)


def any_rank_flagged(flag: bool) -> bool:
    """True on EVERY rank if ``flag`` is true on ANY rank.

    This is what turns a one-rank numerical failure into a clean whole-job abort. Without
    it, the rank that sees a NaN raises and leaves the process group; the others block at
    the next collective until the NCCL timeout, so a fast, legible crash becomes a hang
    that looks like slow training. Cheap enough (one scalar) to run every step.
    """
    if not dist.is_initialized():
        return bool(flag)
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    signal = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(signal, op=dist.ReduceOp.MAX)
    return bool(signal.item() > 0)


def broadcast_flag(flag: bool, src: int = 0) -> bool:
    """Give every rank ``src``'s value of ``flag``.

    Needed for decisions that only rank 0 can make but ALL ranks must branch on
    identically. The live case is "is there a W&B run to publish to": only rank 0 calls
    ``wandb.init``, yet every rank has to agree on whether the epoch-artifact cadence fires,
    because that branch contains a :func:`barrier`. If the ranks disagreed, the ones that
    skipped the branch would never reach that barrier and rank 0 would block there until the
    NCCL timeout.
    """
    if not dist.is_initialized():
        return bool(flag)
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    signal = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.broadcast(signal, src=src)
    return bool(signal.item() > 0)


def wrap_model(model: torch.nn.Module, info: DistInfo) -> torch.nn.Module:
    """Return a DDP view of ``model`` for forward/backward, or ``model`` itself.

    **The caller MUST keep its own unwrapped reference** and use it for ``state_dict()``,
    EMA, gradient clipping, saving and evaluation. Saving from the wrapper prefixes every
    key with ``module.``, and both ``scripts/eval_checkpoints_physical.py`` and
    ``flore.s1_policy.inference.common`` load with ``strict=True`` — so a wrapped save
    produces a checkpoint that is written successfully and then refused by every reader.

    ``find_unused_parameters=False`` is deliberate and MEASURED, per model:

    * **ACT** — all 180 named parameter tensors receive a gradient. Decoder layers 1-6
      receive exact ZEROS, since ``_decode`` reads ``hs[0]``, but a zero gradient is still a
      gradient, so DDP's reducer sees every bucket.
    * **diffusion** — all 306 tensors of the shipped ``vit_b`` + ``ConditionalUnet1D`` build
      receive one (157 on the tiny test build, 126 with the DiT decoder).
      ``force_encoder.invalid_embed`` is the zero-gradient analogue of ACT's dead decoder
      layers: it enters as ``(1 - valid) * embed``, so an all-valid batch multiplies it by
      zero yet still builds the grad path.

    Setting True would add a full graph traversal per step for nothing. This is the opposite
    of the MuSe/UVA force model, which genuinely leaves parameters untouched and dies on
    step 2 without it — hence the pinning tests in ``tests/test_ddp_training.py`` and
    ``tests/test_ddp_diffusion.py``.
    """
    if not info.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel

    # Keyed on where the MODEL actually lives, not on whether the host has a GPU: DDP
    # rejects device_ids for a CPU module, and `torch.cuda.is_available()` is True on this
    # box even when the module is on CPU (every gloo-backed test). Reading the parameters
    # cannot disagree with reality the way an availability probe can.
    on_cuda = any(p.is_cuda for p in model.parameters())
    device_ids = [info.local_rank] if on_cuda else None
    # NEITHER model has a buffer that CHANGES during training, so a per-forward buffer
    # broadcast is pure overhead. ACT: its BatchNorm is FrozenBatchNorm2d (ImageNet
    # statistics held as constants) and `pos_table` is a deterministic sinusoid table.
    # Diffusion: the only buffers are `_last_actions` (non-persistent, written by
    # `sample_actions` and never read by the training forward) and, with the DiT decoder,
    # `decoder.rope.inv_freq` (a constant) -- its normalization is GroupNorm/LayerNorm, which
    # keep no running statistics, and the DDIM schedule is a plain attribute, not a buffer.
    # `tests/test_ddp_diffusion.py` pins that no buffer moves across a step. torch 2.13
    # renamed the knob
    # and warns on the old name, while `forward_sync_buffers` does not exist before it --
    # and pyproject allows torch>=2.4 -- so pick whichever this build actually accepts
    # rather than emitting a FutureWarning every run or crashing on an older torch.
    import inspect

    sync_kwarg = (
        "forward_sync_buffers"
        if "forward_sync_buffers" in inspect.signature(DistributedDataParallel.__init__).parameters
        else "broadcast_buffers"
    )
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=info.local_rank if on_cuda else None,
        find_unused_parameters=False,
        **{sync_kwarg: False},
    )
