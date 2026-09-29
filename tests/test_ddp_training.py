"""Multi-GPU (DDP) ACT training: gradient correctness, shard geometry, failure modes.

CPU + gloo throughout, so the whole file runs in CI with no GPU. What that does and does
not prove:

* **Proves** the DDP wiring — gradient averaging, shard partitioning, collective ordering,
  rank-0 guards, coordinated abort. These are process-level properties, and the backend
  does not change them.
* **Does not prove** NCCL-specific behaviour or throughput. Those are measured separately
  on the 8x L40S box; see the runbook section.

Every test spawns real processes via ``torch.multiprocessing.spawn``. A thread-based fake
would not exercise the reducer, which is the part that can silently be wrong.
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from fuji.s1_policy.model.dp_act import DPACTConfig, DPACTModel
from fuji.s1_policy.training.trainer import EpochShuffleSampler

# Small but STRUCTURALLY faithful: 7 decoder layers so the hs[0] gradient pattern (layer 0
# live, 1-6 exactly zero) is present, since that pattern is what makes
# find_unused_parameters=False a claim worth pinning.
ACTION_DIM = 16
ACTION_HORIZON = 50
STATE_DIM = 7
HIDDEN = 64


class StubBackbone(nn.Module):
    """Deterministic trunk with no download; mirrors tests/test_act_model.py's stub."""

    def __init__(self, num_channels: int = 32, hidden_dim: int = HIDDEN):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(3, num_channels, kernel_size=32, stride=32)

    def forward(self, x):
        f = self.conv(x)
        pos = torch.zeros(1, self.hidden_dim, f.shape[2], f.shape[3], dtype=f.dtype, device=f.device)
        return f, pos


@contextlib.contextmanager
def _pinned_latent_noise():
    """Force the CVAE reparameterization noise to ZERO (so ``z == mu``).

    Required for any exact cross-process gradient comparison, and NOT a weakening of the
    test. ``DPACTModel.forward`` draws ``torch.randn_like(mu)``, whose draw COUNT depends on
    the batch shape: a full batch of 4 consumes 4x8 values while each rank of a 2-way split
    consumes 2x8 from its own identically-seeded stream. The two streams therefore cannot be
    aligned by seeding, so a raw comparison would measure RNG alignment rather than whether
    DDP reduces gradients correctly. Pinning the one stochastic node leaves the entire graph
    -- backbone, style encoder, decoder, every mask -- still exercised.
    """
    original = torch.randn_like
    torch.randn_like = lambda tensor, *a, **k: torch.zeros_like(tensor)
    try:
        yield
    finally:
        torch.randn_like = original


def _model(seed: int = 0) -> DPACTModel:
    """An identically-initialized tiny ACT on every call (and in every process).

    Returned in ``eval()`` mode: ACT carries ``dropout=0.1`` in both its encoder and
    decoder, which is a second source of cross-process divergence. Gradients still flow
    (ACT's BatchNorm is frozen either way), so this changes nothing about what is measured.
    """
    torch.manual_seed(seed)
    config = DPACTConfig(
        hidden_dim=HIDDEN, dim_feedforward=128, nheads=4, enc_layers=1, dec_layers=7,
        vae_enc_layers=1, latent_dim=8, action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON,
        state_dim=STATE_DIM, obs_history_len=1, force_hist_len=1, force_hist_channels=6,
        force_enc_width=16, force_enc_out_dim=8, layout_mode="left_only", pretrained=False,
    )
    return DPACTModel(config, encoder=StubBackbone()).eval()


def _batch(batch_size: int, seed: int = 1234) -> dict:
    """A fixed batch. Image resolution is 64x64: the stub's stride-32 conv keeps the token
    grid meaningful while making the test fast."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    return {
        "images": [randn(batch_size, 3, 64, 64)],
        "img_masks": [torch.ones(batch_size, dtype=torch.bool)],
        "state": randn(batch_size, 1, STATE_DIM),
        "state_valid": torch.ones(batch_size, 1, STATE_DIM),
        "actions": randn(batch_size, ACTION_HORIZON, ACTION_DIM),
        "action_mask": torch.ones(batch_size, ACTION_HORIZON, ACTION_DIM),
        "force_history": randn(batch_size, 1, 6),
        "force_history_valid": torch.ones(batch_size, dtype=torch.bool),
        "_skip_scalar_metrics": True,
    }


def _slice_batch(batch: dict, start: int, stop: int) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, list):
            out[key] = [t[start:stop] for t in value]
        elif torch.is_tensor(value):
            out[key] = value[start:stop]
        else:
            out[key] = value
    return out


def _spawn(worker, world_size: int, *args, timeout: int = 300):
    """Run ``worker`` in ``world_size`` processes; fail the test on any child error.

    ``join=True`` inside a bounded timeout is deliberate: a DEADLOCK is one of the failure
    modes under test (unequal shards, mismatched collectives), and it must surface as a
    test failure rather than a hung CI job.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    procs = []
    for rank in range(world_size):
        p = context.Process(target=_worker_entry, args=(worker, rank, world_size, queue, args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=timeout)
    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        p.terminate()
        p.join(5)
    results = []
    while not queue.empty():
        results.append(queue.get())
    if alive:
        pytest.fail(f"{len(alive)} of {world_size} ranks did not finish within {timeout}s "
                    f"(deadlock); results so far: {results}")
    errors = [r for r in results if isinstance(r, tuple) and r and r[0] == "ERROR"]
    if errors:
        pytest.fail("worker error:\n" + "\n".join(str(e[1]) for e in errors))
    exit_codes = {p.exitcode for p in procs}
    if exit_codes != {0}:
        pytest.fail(f"ranks exited with {exit_codes}; results: {results}")
    return results


def _worker_entry(worker, rank: int, world_size: int, queue, args) -> None:
    """Set the env torchrun would set, join the group, run, and always tear down."""
    import traceback

    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": os.environ.get("_TEST_PORT", "29517"),
    })
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        queue.put(worker(rank, world_size, *args))
    except Exception:
        queue.put(("ERROR", f"rank {rank}: {traceback.format_exc()}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# THE test: does DDP compute the same gradient as one process on the full batch?
# ---------------------------------------------------------------------------
def _grad_parity_worker(rank: int, world_size: int, total_batch: int):
    from torch.nn.parallel import DistributedDataParallel

    from fuji.s1_policy.training.distributed import DistInfo, wrap_model

    model = _model()
    ddp = wrap_model(model, DistInfo(rank=rank, local_rank=rank, world_size=world_size))
    assert isinstance(ddp, DistributedDataParallel)

    per_rank = total_batch // world_size
    shard = _slice_batch(_batch(total_batch), rank * per_rank, (rank + 1) * per_rank)
    with _pinned_latent_noise():
        ddp(shard)["loss"].backward()
    # Read gradients off the UNWRAPPED module: same tensors, and it is the reference every
    # non-forward call site (clipping, saving, EMA) uses.
    #
    # Returned as plain floats rather than tensors: torch's multiprocessing reductions send
    # tensors through a shared-memory file handle whose owning process may exit before the
    # parent reads it ("connection refused"). Summary statistics per parameter are enough to
    # catch a wrong reduction, and they survive the queue.
    return {name: (float(p.grad.sum()), float(p.grad.abs().sum()), float(p.grad.pow(2).sum()))
            for name, p in model.named_parameters() if p.grad is not None}


def test_ddp_gradient_matches_single_process_on_the_same_data():
    """A 2-rank DDP step must produce the SAME averaged gradient as one process over the
    whole batch.

    This is the test that the DDP setup is correct rather than merely running. DDP averages
    (not sums) gradients across ranks, so with every sample equally masked, rank-mean of
    per-shard means == the full-batch mean. If the wrapping, the loss reduction, or the
    shard split were wrong, this is where it shows up -- and nowhere else would, because a
    subtly-wrong gradient still trains, just to a different model.
    """
    total_batch = 4
    results = _spawn(_grad_parity_worker, 2, total_batch)
    assert len(results) == 2, results

    reference = _model()
    with _pinned_latent_noise():
        reference(_batch(total_batch))["loss"].backward()
    expected = {n: (float(p.grad.sum()), float(p.grad.abs().sum()), float(p.grad.pow(2).sum()))
                for n, p in reference.named_parameters() if p.grad is not None}

    live = [n for n, stats in expected.items() if stats[1] > 0]
    assert len(live) > 20, f"only {len(live)} parameters have a non-zero gradient; too weak a test"

    for rank_index, rank_grads in enumerate(results):
        assert set(rank_grads) == set(expected)
        for name, stats in rank_grads.items():
            # Every rank must also agree with the reference -- that IS the all-reduce.
            #
            # Tolerances are scaled by the gradient's own MAGNITUDE (abs_sum), not absolute:
            # summation order differs between "one pass over 4 samples" and "two 2-sample
            # passes averaged", so float32 non-associativity leaves a relative residue.
            # `sum` additionally cancels to near zero on a symmetric gradient, which makes
            # any fixed absolute bound either vacuous or impossible -- hence abs_sum as the
            # scale. A genuinely wrong reduction (missing all-reduce, sum instead of mean,
            # bad shard) shifts these by O(1) relative, which this still catches: the
            # pre-pinning failure was 0.60 vs -0.13.
            scale = max(expected[name][1], 1e-6)
            for label, got, want in zip(("sum", "abs_sum", "sq_sum"), stats, expected[name]):
                assert got == pytest.approx(want, rel=1e-3, abs=scale * 1e-4), (
                    f"rank {rank_index}: gradient {label} mismatch for {name}: "
                    f"{got} != {want} (gradient scale {scale:.3e})")


def _shard_disagreement_worker(rank: int, world_size: int, total_batch: int):
    """Each rank backprops only ITS shard, and reports the resulting gradient scale.

    Complements the parity test above, which cannot tell correct sharding from no sharding
    at all: if every rank averaged the *whole* batch the averaged gradient would still equal
    the single-process one. Verified by mutation -- replacing the shard with the full batch
    leaves the parity test green and this one red.
    """
    from fuji.s1_policy.training.distributed import DistInfo, wrap_model

    model = _model()
    ddp = wrap_model(model, DistInfo(rank=rank, local_rank=rank, world_size=world_size))
    per_rank = total_batch // world_size
    shard = _slice_batch(_batch(total_batch), rank * per_rank, (rank + 1) * per_rank)
    with _pinned_latent_noise():
        out = ddp(shard)
    # The LOCAL loss, before any reduction: it depends on which samples this rank holds, so
    # two ranks with different shards must report different values.
    return {"rank": rank, "local_loss": float(out["loss"].detach()),
            "n_samples": int(shard["actions"].shape[0])}


def test_each_rank_trains_on_a_different_shard():
    """Ranks must see DISJOINT data, not N copies of the same batch.

    Without this, a sharding regression would be invisible: N ranks each averaging the full
    batch produce the correct averaged gradient while doing N times the work for nothing --
    the entire speedup silently gone, with no error and no wrong number to notice.
    """
    total_batch = 4
    results = _spawn(_shard_disagreement_worker, 2, total_batch)
    assert len(results) == 2, results
    assert all(r["n_samples"] == total_batch // 2 for r in results), results
    losses = [r["local_loss"] for r in results]
    assert losses[0] != pytest.approx(losses[1], rel=1e-6), (
        f"both ranks computed the same local loss ({losses}); they are training on identical "
        "data, so the run does N times the work for a 1x speedup")


def _training_steps_worker(rank: int, world_size: int):
    """Three real DDP optimizer steps, reporting everything the two assertions below need.

    ONE spawn serves both tests: each process pays model construction plus a process-group
    handshake, so splitting these across two spawns doubled the cost of this file's slowest
    fixture to extract different fields from an identical run.

    Reports the PRE-clip norm (what ``clip_grad_norm_`` returns) and, separately, the norm
    recomputed AFTER clipping -- the two answer different questions, and conflating them is
    why the earlier version of this test claimed to check ``max_norm`` without doing so.
    """
    import math

    from fuji.s1_policy.training.distributed import DistInfo, wrap_model

    max_norm = 1.0
    model = _model()
    ddp = wrap_model(model, DistInfo(rank=rank, local_rank=rank, world_size=world_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    preclip, postclip = [], []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        # Different data per rank per step, as a real sharded run has.
        ddp(_batch(2, seed=1000 + step * 10 + rank))["loss"].backward()
        preclip.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)))
        postclip.append(math.sqrt(sum(float(p.grad.pow(2).sum())
                                      for p in model.parameters() if p.grad is not None)))
        optimizer.step()
    return {
        "preclip": preclip,
        "postclip": postclip,
        "max_norm": max_norm,
        "n_no_grad": sum(1 for p in model.parameters() if p.grad is None),
        "n_params": sum(1 for _ in model.parameters()),
    }


@pytest.fixture(scope="module")
def ddp_training_steps():
    """Run the 2-rank training worker ONCE and share it across the tests below."""
    results = _spawn(_training_steps_worker, 2)
    assert len(results) == 2, results
    return results


def test_find_unused_parameters_false_is_correct_for_act(ddp_training_steps):
    """DDP with ``find_unused_parameters=False`` must survive repeated steps.

    Pinned because this is the OPPOSITE of the MuSe/UVA force model, where the same setting
    kills the run on step 2 -- and the two live in the same repo. ACT reads only ``hs[0]``,
    so decoder layers 1-6 get exact ZERO gradients, but a zero gradient still marks the
    bucket ready, so no parameter is "unused" in DDP's sense. If a future edit made part of
    the graph genuinely conditional, DDP would raise here instead of 20 minutes into a run.
    """
    for result in ddp_training_steps:
        assert result["n_no_grad"] == 0, (
            f"{result['n_no_grad']} of {result['n_params']} parameters received no gradient; "
            "find_unused_parameters=False is no longer safe")


def test_ddp_gradient_norms_stay_finite_and_bounded(ddp_training_steps):
    """Gradient stability across DDP steps: finite, non-zero, and actually clipped.

    The user's explicit requirement ("gradient 不要爆炸"). Clipping runs AFTER backward, by
    which point DDP has already averaged, so the post-clip norm must respect ``max_norm``
    exactly as single-GPU does -- and that is checked on the RECOMPUTED norm, because
    ``clip_grad_norm_`` returns the norm as it was BEFORE clipping.
    """
    import math

    # The bound is a LITERAL, not `result["max_norm"]`: reading the clip threshold back from
    # the run under test makes the assertion self-fulfilling (verified -- mutating the worker
    # to max_norm=inf left the test green, since nothing exceeds infinity). The worker's
    # `max_norm` is cross-checked against this instead, so a change there fails loudly.
    CLIP = 1.0
    for result in ddp_training_steps:
        assert result["max_norm"] == CLIP, (
            f"worker clipped at {result['max_norm']} but this test asserts the {CLIP} bound; "
            "update both together")
        max_norm = CLIP
        for step, norm in enumerate(result["preclip"]):
            assert math.isfinite(norm), f"step {step}: non-finite pre-clip gradient norm {norm}"
            assert norm > 0.0, f"step {step}: zero gradient norm -- nothing is learning"
        for step, norm in enumerate(result["postclip"]):
            assert math.isfinite(norm), f"step {step}: non-finite post-clip gradient norm {norm}"
            # The clip is the anti-explosion guarantee, so assert it on the real post-clip
            # value rather than trusting the returned pre-clip one.
            assert norm <= max_norm * (1 + 1e-4), (
                f"step {step}: post-clip norm {norm} exceeds max_norm {max_norm}")

    # Both ranks see the same AVERAGED gradient, so they must agree on its norm. A missing
    # all-reduce would let the ranks drift apart here.
    left, right = (r["preclip"] for r in ddp_training_steps)
    for step, (a, b) in enumerate(zip(left, right)):
        assert a == pytest.approx(b, rel=1e-3), (
            f"step {step}: ranks disagree on the gradient norm ({a} vs {b}); "
            "the gradient was not averaged across ranks")


def _nan_abort_worker(rank: int, world_size: int):
    """Rank 1 alone reports a non-finite loss; both ranks must conclude "abort"."""
    from fuji.s1_policy.training.distributed import any_rank_flagged

    saw_nan_locally = rank == 1
    return {"rank": rank, "local": saw_nan_locally,
            "should_abort": any_rank_flagged(saw_nan_locally)}


def _uneven_collective_worker(rank: int, world_size: int):
    """Reproduce the shape of the validation loop: unequal per-rank batch counts, then ONE
    reduction after the loop.

    Regression guard for a real defect found during review of this work: ``evaluate`` used to
    RAISE inside its batch loop on a non-finite loss. Because ranks hold different numbers of
    validation batches, one rank raising is a collective its peers never make -- they walk
    into the post-loop all-reduce and block until the 10-minute NCCL timeout. A hang, not a
    crash. The fix records the condition and reduces it once, after the loop, which is what
    this asserts: per-rank work may differ, collective COUNT may not.
    """
    from fuji.s1_policy.training.distributed import all_reduce_sum, any_rank_flagged

    # Deliberately uneven, exactly as a contiguous validation shard can be.
    n_batches = 2 + rank
    local_bad = rank == world_size - 1        # only the LAST rank sees trouble
    total = torch.zeros((), dtype=torch.float64)
    for _ in range(n_batches):
        total += 1.0                          # no collective inside the loop

    should_abort = any_rank_flagged(local_bad)     # exactly one collective per rank
    reduced = float(all_reduce_sum(total))         # exactly one more
    return {"rank": rank, "n_batches": n_batches, "should_abort": should_abort,
            "reduced": reduced}


def test_uneven_per_rank_work_still_makes_matching_collectives():
    """Ranks may do different AMOUNTS of work, but must make the same NUMBER of collectives.

    If this test hangs, the deadlock it guards against has returned -- `_spawn` bounds the
    join and fails rather than hanging CI.
    """
    world_size = 3
    results = _spawn(_uneven_collective_worker, world_size)
    assert len(results) == world_size, results
    assert sorted(r["n_batches"] for r in results) == [2, 3, 4], "shards should be uneven"
    # Every rank agrees on the abort decision, and on the reduced total (2+3+4).
    assert all(r["should_abort"] for r in results), results
    for result in results:
        assert result["reduced"] == pytest.approx(9.0), result


def test_a_nan_on_one_rank_aborts_every_rank():
    """One rank's numerical failure must stop the whole job, not hang it.

    Without the all-reduce, the rank that raises leaves the group and its peers block at the
    next collective until the NCCL timeout (10 min) -- so a legible crash becomes a hang
    that reads as slow training.
    """
    results = _spawn(_nan_abort_worker, 2)
    assert len(results) == 2, results
    assert all(r["should_abort"] for r in results), results
    # And exactly one rank actually saw it, so the flag really did propagate.
    assert sum(r["local"] for r in results) == 1, results


# ---------------------------------------------------------------------------
# Shard geometry -- pure logic, no processes needed
# ---------------------------------------------------------------------------
def test_shards_are_disjoint_equal_and_cover_the_truncated_permutation():
    """Every rank must get an equal, disjoint slice.

    Equal lengths are what prevent the classic silent DDP deadlock: with drop_last=True,
    unequal shards mean one rank runs fewer batches, exits the loop early, and never reaches
    the final gradient all-reduce.
    """
    size, world_size = 1000, 3
    samplers = [EpochShuffleSampler(size, seed=7, rank=r, world_size=world_size)
                for r in range(world_size)]
    for sampler in samplers:
        sampler.set_epoch(4)
    shards = [list(iter(s)) for s in samplers]

    assert {len(s) for s in shards} == {size // world_size}
    for sampler, shard in zip(samplers, shards):
        assert len(shard) == len(sampler) == sampler.shard_size
    flat = [i for shard in shards for i in shard]
    assert len(set(flat)) == len(flat), "shards overlap: a sample would be seen twice"
    # The union is the permutation truncated to a multiple of world_size.
    assert len(flat) == (size // world_size) * world_size
    assert set(flat) <= set(range(size))
    # Truncation drops fewer than world_size samples.
    assert size - len(flat) == size % world_size < world_size


def test_all_ranks_shard_the_same_underlying_permutation():
    """The ranks partition ONE global permutation without communicating.

    They agree because the permutation is seeded by (seed + epoch) alone. If a rank's order
    depended on its rank, the shards could overlap or miss samples silently.
    """
    size, world_size = 60, 4
    union = set()
    for rank in range(world_size):
        sampler = EpochShuffleSampler(size, seed=11, rank=rank, world_size=world_size)
        sampler.set_epoch(2)
        union |= set(iter(sampler))
    solo = EpochShuffleSampler(size, seed=11, rank=0, world_size=1)
    solo.set_epoch(2)
    assert union == set(iter(solo)), "sharded union differs from the single-process order"


def test_a_different_epoch_drops_a_different_truncated_tail():
    """The truncation must not exclude the same samples every epoch.

    With a fixed permutation, `size % world_size` samples would be permanently invisible.
    Reseeding per epoch makes the dropped tail rotate.
    """
    size, world_size = 101, 4  # 101 % 4 == 1, so exactly one sample is dropped per epoch
    dropped = set()
    for epoch in range(12):
        seen = set()
        for rank in range(world_size):
            sampler = EpochShuffleSampler(size, seed=3, rank=rank, world_size=world_size)
            sampler.set_epoch(epoch)
            seen |= set(iter(sampler))
        missing = set(range(size)) - seen
        assert len(missing) == 1
        dropped |= missing
    assert len(dropped) > 1, "the same sample is dropped every epoch"


def test_mid_epoch_offset_preserves_the_remaining_shard_order():
    """Resume must continue a rank's shard, not restart or reshuffle it."""
    sampler = EpochShuffleSampler(120, seed=5, rank=1, world_size=3)
    sampler.set_epoch(2)
    full = list(iter(sampler))
    sampler.set_epoch(2, offset=7)
    assert list(iter(sampler)) == full[7:]
    assert len(sampler) == len(full) - 7


def test_offset_is_bounded_by_the_shard_not_the_dataset():
    """An offset valid for the whole dataset but past this rank's shard must be refused.

    Accepting it would silently yield an empty rank while its peers still had batches --
    i.e. a deadlock at the next collective.
    """
    sampler = EpochShuffleSampler(120, seed=5, rank=0, world_size=4)
    assert sampler.shard_size == 30
    sampler.set_epoch(0, offset=30)          # exactly at the end is legal
    with pytest.raises(ValueError, match=r"outside \[0, 30\]"):
        sampler.set_epoch(0, offset=31)


def test_resume_refuses_a_different_world_size():
    """The shard depends on world_size, so resuming on a different one replays other data."""
    saved = EpochShuffleSampler(120, seed=5, rank=0, world_size=4)
    saved.set_epoch(3, offset=2)
    state = saved.state_dict()
    assert state["world_size"] == 4

    with pytest.raises(ValueError, match="world_size=4"):
        EpochShuffleSampler(120, seed=5, rank=0, world_size=2).load_state_dict(state)

    same = EpochShuffleSampler(120, seed=5, rank=0, world_size=4)
    same.load_state_dict(state)
    assert (same.epoch, same.offset) == (3, 2)


def test_a_pre_ddp_checkpoint_still_resumes():
    """A state written before DDP existed has no world_size key; treat it as 1."""
    legacy = {"size": 64, "seed": 5, "epoch": 2, "offset": 3}   # no world_size
    sampler = EpochShuffleSampler(64, seed=5)
    sampler.load_state_dict(legacy)
    assert (sampler.epoch, sampler.offset, sampler.world_size) == (2, 3, 1)


def test_single_process_sampler_is_unchanged_by_the_ddp_plumbing():
    """world_size=1 must behave exactly as before: full permutation, no truncation.

    Guards the promise that a non-torchrun run is bit-identical to the pre-DDP trainer.
    """
    sampler = EpochShuffleSampler(97, seed=13)
    sampler.set_epoch(6)
    order = list(iter(sampler))
    expected = torch.randperm(97, generator=torch.Generator().manual_seed(13 + 6)).tolist()
    assert order == expected
    assert sampler.shard_size == 97 and len(order) == 97


def test_sampler_rejects_an_impossible_topology():
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        EpochShuffleSampler(10, seed=0, world_size=0)
    with pytest.raises(ValueError, match=r"rank 4 is outside \[0, 4\)"):
        EpochShuffleSampler(10, seed=0, rank=4, world_size=4)


# ---------------------------------------------------------------------------
# Single-process no-op guarantees
# ---------------------------------------------------------------------------
def test_distributed_helpers_are_inert_without_a_process_group():
    """Every helper must be safe and side-effect-free in a plain single-process run."""
    from fuji.s1_policy.training.distributed import (
        all_reduce_sum,
        any_rank_flagged,
        barrier,
        broadcast_flag,
        destroy_process_group,
        get_dist_info,
        wrap_model,
    )

    info = get_dist_info()
    assert not info.enabled and info.is_main and info.world_size == 1

    barrier()                      # must not raise
    destroy_process_group()        # must not raise
    assert any_rank_flagged(True) is True
    assert any_rank_flagged(False) is False
    assert broadcast_flag(True) is True

    tensor = torch.tensor([2.0, 3.0], dtype=torch.float64)
    out = all_reduce_sum(tensor)
    torch.testing.assert_close(out, tensor)
    assert out is not tensor, "must not alias the caller's accumulator"

    model = _model()
    assert wrap_model(model, info) is model, "single process must not be DDP-wrapped"


def test_wrap_model_keeps_state_dict_keys_unprefixed():
    """A DDP-wrapped save would prefix every key with 'module.'.

    Both scripts/eval_checkpoints_physical.py and fuji.s1_policy.inference.common load with
    strict=True, so such a checkpoint is written successfully and then refused by every
    reader. The trainer therefore keeps `self.model` unwrapped and saves from it; this pins
    the property that makes that necessary.
    """
    from fuji.s1_policy.training.distributed import DistInfo

    model = _model()
    unwrapped_keys = set(model.state_dict())
    assert not any(k.startswith("module.") for k in unwrapped_keys)

    # Simulate what a wrapped save would produce, without needing a process group.
    class _FakeWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    wrapped_keys = set(_FakeWrapper(model).state_dict())
    assert all(k.startswith("module.") for k in wrapped_keys)
    assert wrapped_keys != unwrapped_keys
    assert DistInfo(world_size=4).enabled and not DistInfo(rank=1, world_size=4).is_main
