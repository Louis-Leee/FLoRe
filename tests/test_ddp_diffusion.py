"""Multi-GPU (DDP) DP-Diffusion training: gradient parity, EMA parity, convergence, loaders.

The diffusion sibling of ``tests/test_ddp_training.py`` (which covers ACT). Same harness and
the same caveat: CPU + gloo throughout, so the whole file runs in CI with no GPU.

* **Proves** the parts that can be silently WRONG — gradient averaging, that the EMA shadow
  (diffusion's deploy weights) stays identical to a single-process run, that every parameter
  receives a gradient so ``find_unused_parameters=False`` is sound, that no buffer drifts
  between ranks, that the loaders shard disjointly, that sharded validation re-aggregates to
  the exact single-process objective, and that the loss actually falls.
* **Does not prove** NCCL behaviour or throughput. Those are measured on the 8x L40S box;
  see ``docs/RUNBOOK_S1_WORKSTATION.md`` section 8c.

Why diffusion needs its own file rather than a parametrization of the ACT one: the two
models have different stochastic surfaces (``epsilon``/``t`` draws vs one CVAE
reparameterization), different "structurally zero gradient" parameters, and diffusion alone
carries an EMA shadow — which is what a diffusion checkpoint actually deploys, so a
replica-induced drift there would ship silently.

Three process spawns, deliberately: each rank pays a fresh interpreter plus a process-group
handshake, so the tests are grouped by the run they need rather than one spawn per assertion.
"""

from __future__ import annotations

import contextlib
import math
import os

import pytest
import torch
import torch.nn as nn

from fuji.s1_policy.model.diffusion_base import DiffusionVLABase
from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel
from fuji.s1_policy.training.distributed import DistInfo, wrap_model

# Own rendezvous port so this file can never collide with tests/test_ddp_training.py.
_PORT = "29621"

# HL-v2 left_only widths, at a small horizon. down_dims has TWO levels on purpose, so the
# UNet's downsample/skip/upsample path is genuinely exercised rather than a single block.
ACTION_DIM = 16
ACTION_HORIZON = 16
STATE_DIM = 7
IMAGE_HW = 64
FORCE_HIST_LEN = 8
FORCE_CHANNELS = 6


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_torch():
    """Pin intra-op threads to 1 for this module, then restore.

    Not a micro-optimization: on a 96-thread box a tiny CPU model spends ~30x more time in
    thread launch than in arithmetic (measured 305 ms/step at 96 threads vs 11 ms at 1), and
    two ranks each fanning out to 96 threads thrash each other. The spawned workers set it
    again for themselves -- child processes do not inherit a fixture.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class StubEncoder(nn.Module):
    """Minimal pluggable image encoder: (N, 3, H, W) -> (N, num_features). No downloads."""

    def __init__(self, num_features: int = 48):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(3, num_features, kernel_size=16, stride=16)

    def forward(self, x):
        return self.conv(x).mean(dim=(-1, -2))


def _model(diffusion_net: str = "unet", seed: int = 0) -> DPDiffusionModel:
    """An identically-initialized tiny DP-Diffusion on every call (and in every process).

    Kept in TRAIN mode, unlike the ACT fixture: diffusion has no dropout anywhere in this
    build, so train mode is both faithful to the real loop and reproducible across processes.
    """
    torch.manual_seed(seed)
    extra = (
        dict(down_dims=(32, 64), kernel_size=3, n_groups=8)
        if diffusion_net == "unet"
        else dict(model_channels=32, num_blocks=2, n_heads=4)
    )
    config = DPDiffusionConfig(
        diffusion_net=diffusion_net,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        state_dim=STATE_DIM,
        obs_history_len=1,
        force_hist_len=FORCE_HIST_LEN,
        force_hist_channels=FORCE_CHANNELS,
        force_enc_width=16,
        force_enc_out_dim=8,
        layout_mode="left_only",
        diffusion_step_embed_dim=32,
        pretrained=False,
        **extra,
    )
    return DPDiffusionModel(config, encoder=StubEncoder())


def _batch(batch_size: int, seed: int = 1234, force_valid: bool = True) -> dict:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    return {
        "images": [randn(batch_size, 3, IMAGE_HW, IMAGE_HW)],
        "img_masks": [torch.ones(batch_size, dtype=torch.bool)],
        "state": randn(batch_size, 1, STATE_DIM),
        "state_valid": torch.ones(batch_size, 1, STATE_DIM),
        "actions": randn(batch_size, ACTION_HORIZON, ACTION_DIM),
        "action_mask": torch.ones(batch_size, ACTION_HORIZON, ACTION_DIM),
        "force_history": randn(batch_size, FORCE_HIST_LEN, FORCE_CHANNELS),
        "force_history_valid": torch.full((batch_size,), force_valid, dtype=torch.bool),
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


@contextlib.contextmanager
def _pinned_draws(row_offset: int = 0):
    """Make ``epsilon`` and ``t`` deterministic functions of the GLOBAL row index.

    Required for ANY exact cross-process comparison, and not a weakening of the test. The
    diffusion objective draws three random things per forward -- the target noise, the DDPM-IP
    perturbation, and one integer timestep per row -- and their draw COUNT depends on the
    batch shape, so a full batch of 4 and two 2-row shards consume differently-shaped chunks
    of an identically-seeded stream. Aligning them by seeding is impossible; a raw comparison
    would measure RNG alignment rather than whether DDP reduces gradients correctly.

    Keying on ``row_offset + local_row`` instead means rank 1's row 0 gets exactly what the
    single-process run's row 2 gets, so the ONLY remaining difference between the two runs is
    the reduction. Everything else -- encoder, force encoder, scheduler, masking, decoder --
    is fully exercised at a genuinely per-row noise level and timestep.

    ``call_index`` separates the two ``_sample_noise`` calls, so DDPM-IP still perturbs with
    an independent draw rather than a copy of the target. It is per-CONTEXT, so enter this
    context once per forward when comparing multi-step runs.
    """
    calls = {"n": 0}

    def sample_noise(self, shape, device):
        call_index = calls["n"]
        calls["n"] += 1
        rows = shape[0]
        out = torch.empty(shape, dtype=torch.float32, device=device)
        for row in range(rows):
            seed = 7919 * (row_offset + row) + 17 * call_index + 1
            out[row] = torch.randn(
                shape[1:], generator=torch.Generator().manual_seed(seed)
            ).to(device)
        return out

    def sample_timesteps(self, count, device):
        limit = self.noise_scheduler.config.num_train_timesteps
        values = [(13 * (row_offset + row) + 5) % limit for row in range(count)]
        return torch.tensor(values, device=device, dtype=torch.long)

    original = (DiffusionVLABase._sample_noise, DiffusionVLABase._sample_timesteps)
    DiffusionVLABase._sample_noise = sample_noise
    DiffusionVLABase._sample_timesteps = sample_timesteps
    try:
        yield
    finally:
        DiffusionVLABase._sample_noise, DiffusionVLABase._sample_timesteps = original


def _grad_stats(model: nn.Module) -> dict[str, tuple[float, float, float]]:
    """Per-parameter gradient summary as plain floats.

    Floats, not tensors: torch's multiprocessing reduction sends tensors through a
    shared-memory handle whose owning process may exit before the parent reads it. Three
    statistics per tensor are enough to catch a wrong reduction and they survive the queue.
    """
    return {
        name: (float(p.grad.sum()), float(p.grad.abs().sum()), float(p.grad.pow(2).sum()))
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def _param_stats(model: nn.Module) -> dict[str, tuple[float, float]]:
    return {name: (float(p.detach().sum()), float(p.detach().abs().sum()))
            for name, p in model.named_parameters()}


def _assert_stats_match(got: dict, expected: dict, label: str, *, rel: float = 1e-3) -> None:
    """Compare per-tensor statistics with tolerances scaled by the tensor's own magnitude.

    Summation order differs between "one pass over N rows" and "two N/2-row passes averaged",
    so float32 non-associativity leaves a relative residue. ``sum`` additionally cancels to
    near zero on a symmetric tensor, which makes any fixed absolute bound either vacuous or
    impossible -- hence abs_sum as the scale. A genuinely wrong reduction (missing all-reduce,
    sum instead of mean, wrong shard) shifts these by O(1) relative, which this still catches.
    """
    assert set(got) == set(expected), (
        f"{label}: tensor names differ "
        f"(missing {sorted(set(expected) - set(got))[:5]}, "
        f"extra {sorted(set(got) - set(expected))[:5]})")
    for name, values in got.items():
        scale = max(abs(expected[name][1]), 1e-6)
        for axis, (actual, want) in enumerate(zip(values, expected[name])):
            assert actual == pytest.approx(want, rel=rel, abs=scale * 1e-4), (
                f"{label}: statistic {axis} mismatch for {name}: {actual} != {want} "
                f"(scale {scale:.3e})")


def _spawn(worker, world_size: int, *args, timeout: int = 600):
    """Run ``worker`` in ``world_size`` processes on this file's own rendezvous port.

    Reuses the harness from tests/test_ddp_training.py rather than copying it: a DEADLOCK is
    one of the failure modes under test, and that harness already bounds the join so a hang
    surfaces as a test failure instead of a stuck CI job.
    """
    from tests.test_ddp_training import _spawn as _spawn_ranks

    previous = os.environ.get("_TEST_PORT")
    os.environ["_TEST_PORT"] = _PORT
    try:
        return _spawn_ranks(worker, world_size, *args, timeout=timeout)
    finally:
        if previous is None:
            os.environ.pop("_TEST_PORT", None)
        else:
            os.environ["_TEST_PORT"] = previous


# ---------------------------------------------------------------------------
# Preconditions for find_unused_parameters=False and broadcast_buffers=False.
# Pure single-process checks -- no reason to pay a spawn for them.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("diffusion_net", ["unet", "dit"])
def test_every_diffusion_parameter_receives_a_gradient(diffusion_net):
    """``find_unused_parameters=False`` is only sound if NO parameter is left out.

    DDP's fast reducer pre-registers one bucket per parameter and expects every one to be
    marked ready during backward; a parameter that never receives a gradient makes it raise
    (typically on step 2, ~minutes into a run). Checked for both decoders because
    ``--diffusion.diffusion-net`` selects between them at launch.

    ``force_encoder.invalid_embed`` is the interesting case and the reason this asserts
    "receives a gradient" rather than "receives a non-zero gradient": it enters as
    ``(1 - valid) * embed``, so an all-valid batch multiplies it by zero. That is a ZERO
    gradient, not a MISSING one -- the bucket is still marked ready, exactly like ACT's
    decoder layers 1-6. If a future edit turned such a term into a Python ``if``, the
    parameter would disappear from the graph and this test would fail here instead of on the
    cluster.
    """
    torch.manual_seed(0)
    model = _model(diffusion_net)
    model(_batch(4))["loss"].backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, (
        f"{len(missing)} parameter(s) received no gradient, so "
        f"find_unused_parameters=False would raise mid-run: {missing[:10]}")
    total = sum(1 for _ in model.parameters())
    live = [n for n, p in model.named_parameters() if float(p.grad.abs().sum()) > 0]
    assert len(live) > total * 0.8, (
        f"only {len(live)} of {total} parameters have a NON-zero gradient; the model is "
        "mostly dead and this file would be testing very little")


@pytest.mark.parametrize("diffusion_net", ["unet", "dit"])
def test_no_buffer_changes_during_a_training_step(diffusion_net):
    """``broadcast_buffers=False`` is only safe if no buffer carries training state.

    With the per-forward buffer sync off, a buffer that ACCUMULATED during training (a
    BatchNorm running mean, say) would drift apart on every rank, and rank 0's copy -- the
    one that gets saved -- would describe a 1/N slice of the data. The diffusion model's
    normalization is GroupNorm/LayerNorm (no running statistics) and its DDIM schedule is a
    plain attribute rather than a buffer, so the only buffers are the non-persistent RTC
    action cache and, with the DiT, a constant RoPE frequency table. This pins that.
    """
    torch.manual_seed(0)
    model = _model(diffusion_net)
    before = {name: buf.detach().clone() for name, buf in model.named_buffers()}
    assert before, "expected at least the _last_actions cache"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(_batch(4, seed=500 + step))["loss"].backward()
        optimizer.step()

    for name, buf in model.named_buffers():
        assert torch.equal(buf, before[name]), (
            f"buffer {name!r} changed during training; broadcast_buffers=False would let the "
            "ranks drift apart and only rank 0's copy would be saved")


# ---------------------------------------------------------------------------
# Spawn A: one 2-rank process group, three independent backward passes.
# ---------------------------------------------------------------------------
def _single_step_worker(rank: int, world_size: int, total_batch: int):
    """Gradient parity, shard disagreement, and the lopsided-mask case, in one group.

    Three separate forward/backward pairs; both ranks run all three in the same order, so the
    reducer sees matching collectives throughout.
    """
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    model = _model()
    ddp = wrap_model(model, DistInfo(rank=rank, local_rank=rank, world_size=world_size))
    assert isinstance(ddp, DistributedDataParallel)

    per_rank = total_batch // world_size
    start, stop = rank * per_rank, (rank + 1) * per_rank
    result = {"rank": rank, "n_samples": per_rank}

    # (1) Parity: equal masks, so the per-rank valid counts match and the reduction is exact.
    shard = _slice_batch(_batch(total_batch), start, stop)
    model.zero_grad(set_to_none=True)
    with _pinned_draws(row_offset=start):
        out = ddp(shard)
    out["loss"].backward()
    # Read gradients off the UNWRAPPED module: the same tensors, and the reference every
    # non-forward call site (clipping, saving, EMA) uses.
    result["grads"] = _grad_stats(model)

    # (2) The LOCAL loss, before any reduction: it depends on which rows this rank holds, so
    # two ranks with different shards must report different values.
    result["local_loss"] = float(out["loss"].detach())

    # (3) Lopsided masks: rank r supervises only the first (r+1) action dims, so the ranks
    # hold deliberately different valid-element counts.
    lopsided = _slice_batch(_batch(total_batch, seed=99), start, stop)
    mask = torch.zeros_like(lopsided["action_mask"])
    mask[..., : rank + 1] = 1.0
    lopsided["action_mask"] = mask
    model.zero_grad(set_to_none=True)
    with _pinned_draws(row_offset=start):
        ddp(lopsided)["loss"].backward()
    result["lopsided_grads"] = _grad_stats(model)
    result["lopsided_valid"] = float(mask.sum())
    return result


@pytest.fixture(scope="module")
def single_step_ddp():
    results = _spawn(_single_step_worker, 2, 4)
    assert len(results) == 2, results
    return sorted(results, key=lambda r: r["rank"])


def test_ddp_gradient_matches_single_process_on_the_same_data(single_step_ddp):
    """A 2-rank DDP step must produce the SAME averaged gradient as one process over the
    whole batch.

    This is THE test that the diffusion DDP setup is correct rather than merely running. DDP
    averages (not sums) gradients across ranks, so with every row equally masked, the mean of
    the per-shard masked means equals the full-batch masked mean. If the wrapping, the loss
    reduction, or the shard split were wrong this is where it shows up -- and nowhere else
    would, because a subtly-wrong gradient still trains, just to a different model.
    """
    torch.manual_seed(0)
    reference = _model()
    with _pinned_draws(row_offset=0):
        reference(_batch(4))["loss"].backward()
    expected = _grad_stats(reference)

    live = [name for name, stats in expected.items() if stats[1] > 0]
    assert len(live) > 20, f"only {len(live)} parameters have a non-zero gradient; too weak"

    for result in single_step_ddp:
        _assert_stats_match(result["grads"], expected, f"rank {result['rank']} gradient")


def test_each_rank_trains_on_a_different_shard(single_step_ddp):
    """Ranks must see DISJOINT rows, not N copies of the same batch.

    The parity test above cannot tell correct sharding from no sharding at all: if every rank
    forwarded the WHOLE batch the averaged gradient would still equal the single-process one,
    while the run did N times the work for a 1x speedup -- no error, no wrong number.
    """
    assert all(r["n_samples"] == 2 for r in single_step_ddp), single_step_ddp
    losses = [r["local_loss"] for r in single_step_ddp]
    assert losses[0] != pytest.approx(losses[1], rel=1e-6), (
        f"both ranks computed the same local loss ({losses}); they are training on identical "
        "data, so the run does N times the work for a 1x speedup")


def test_unequal_valid_counts_reduce_to_the_average_of_per_rank_means(single_step_ddp):
    """Pin the EXACT semantics when the ranks hold different numbers of valid elements.

    Every S1 loss is a masked mean, and DDP averages the per-rank LOSSES. So the reduced
    gradient is ``mean_r(S_r / C_r)``, whereas a single process over the concatenated batch
    computes ``(sum_r S_r) / (sum_r C_r)``. The two agree only when the counts agree.

    This asserts both halves of that: the DDP gradient equals the average of the two
    single-shard gradients (the claim), and does NOT equal the pooled masked-mean gradient
    (so the claim is not vacuous). On real HL-v2 batches the counts differ only by temporal
    padding and force validity inside a 32x50x16 mask, i.e. by a fraction of a percent -- the
    masks here are deliberately extreme (1 supervised dim vs 2 out of 16) to make the
    difference measurable at all.
    """
    counts = [r["lopsided_valid"] for r in single_step_ddp]
    assert counts[0] != counts[1], "the shards were supposed to have different valid counts"

    # The claim: average of the per-shard gradients.
    per_shard = []
    for rank in (0, 1):
        torch.manual_seed(0)
        model = _model()
        shard = _slice_batch(_batch(4, seed=99), rank * 2, rank * 2 + 2)
        mask = torch.zeros_like(shard["action_mask"])
        mask[..., : rank + 1] = 1.0
        shard["action_mask"] = mask
        with _pinned_draws(row_offset=rank * 2):
            model(shard)["loss"].backward()
        per_shard.append({n: p.grad.detach().clone() for n, p in model.named_parameters()})
    averaged = {name: (per_shard[0][name] + per_shard[1][name]) / 2 for name in per_shard[0]}
    expected = {name: (float(g.sum()), float(g.abs().sum()), float(g.pow(2).sum()))
                for name, g in averaged.items()}
    for result in single_step_ddp:
        _assert_stats_match(result["lopsided_grads"], expected,
                            f"rank {result['rank']} lopsided gradient")

    # Not vacuous: the pooled masked mean is a DIFFERENT gradient.
    torch.manual_seed(0)
    pooled_model = _model()
    pooled = _batch(4, seed=99)
    pooled_mask = torch.zeros_like(pooled["action_mask"])
    pooled_mask[:2, :, :1] = 1.0
    pooled_mask[2:, :, :2] = 1.0
    pooled["action_mask"] = pooled_mask
    with _pinned_draws(row_offset=0):
        pooled_model(pooled)["loss"].backward()
    differs = [
        name for name, p in pooled_model.named_parameters()
        if float(p.grad.abs().sum()) > 0
        and abs(float(p.grad.abs().sum()) - expected[name][1]) > 1e-6 * max(expected[name][1], 1e-6)
    ]
    assert differs, (
        "the pooled masked-mean gradient is indistinguishable from the average of per-rank "
        "means, so this test proves nothing about the semantics it claims to pin")


# ---------------------------------------------------------------------------
# Spawn B: several real optimizer steps -- weights, EMA shadow, clipping, convergence.
# ---------------------------------------------------------------------------
PARITY_STEPS = 3
CONVERGENCE_STEPS = 40
CLIP = 1.0


def _optimizer_and_ema(model):
    from diffusers.training_utils import EMAModel

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.95, 0.999),
                                  weight_decay=1e-6)
    ema = EMAModel(model.parameters(), decay=0.9, use_ema_warmup=True, power=0.75)
    return optimizer, ema


def _ema_stats(ema, model) -> dict[str, tuple[float, float]]:
    return {name: (float(shadow.sum()), float(shadow.abs().sum()))
            for (name, _), shadow in zip(model.named_parameters(), ema.shadow_params)}


def _multi_step_worker(rank: int, world_size: int, total_batch: int):
    """PARITY_STEPS real updates (optimizer + EMA + clipping), then a convergence run.

    ONE spawn for both phases: each rank pays interpreter startup plus a process-group
    handshake, and the two phases need the same group. The convergence phase rebuilds the
    model from the same seed, so it starts from the same weights the parity phase did rather
    than inheriting its optimizer trajectory.
    """
    torch.set_num_threads(1)
    per_rank = total_batch // world_size
    start, stop = rank * per_rank, (rank + 1) * per_rank
    info = DistInfo(rank=rank, local_rank=rank, world_size=world_size)

    model = _model()
    ddp = wrap_model(model, info)
    optimizer, ema = _optimizer_and_ema(model)
    preclip, postclip = [], []
    for step in range(PARITY_STEPS):
        shard = _slice_batch(_batch(total_batch, seed=300 + step), start, stop)
        optimizer.zero_grad(set_to_none=True)
        # Fresh context per step: the call counter that separates the two noise draws is
        # per-context, so entering once per forward keeps rank r's row k aligned with the
        # single-process run's row start+k at EVERY step.
        with _pinned_draws(row_offset=1_000_000 * step + start):
            ddp(shard)["loss"].backward()
        preclip.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)))
        postclip.append(math.sqrt(sum(float(p.grad.pow(2).sum())
                                      for p in model.parameters() if p.grad is not None)))
        optimizer.step()
        ema.step(model.parameters())

    result = {
        "rank": rank,
        "params": _param_stats(model),
        "ema": _ema_stats(ema, model),
        "preclip": preclip,
        "postclip": postclip,
        "clip": CLIP,
        "n_no_grad": sum(1 for p in model.parameters() if p.grad is None),
        "n_params": sum(1 for _ in model.parameters()),
    }

    # Convergence: overfit ONE fixed batch with the draws pinned, so the objective is a fixed
    # deterministic function of the weights and a falling loss means learning rather than a
    # lucky timestep. A live run's loss is dominated by the fresh (epsilon, t) draw each step,
    # which is exactly what makes "did it converge" unassertable without this.
    torch.manual_seed(0)
    fresh = _model()
    fresh_ddp = wrap_model(fresh, info)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=3e-3)
    fixed = _slice_batch(_batch(total_batch, seed=777), start, stop)
    losses = []
    for _ in range(CONVERGENCE_STEPS):
        fresh_optimizer.zero_grad(set_to_none=True)
        with _pinned_draws(row_offset=start):
            loss = fresh_ddp(fixed)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fresh.parameters(), CLIP)
        fresh_optimizer.step()
        losses.append(float(loss.detach()))
    result["losses"] = losses
    return result


@pytest.fixture(scope="module")
def multi_step_ddp():
    results = _spawn(_multi_step_worker, 2, 4)
    assert len(results) == 2, results
    return sorted(results, key=lambda r: r["rank"])


@pytest.fixture(scope="module")
def multi_step_reference():
    """The same PARITY_STEPS run in ONE process over the full batch."""
    torch.manual_seed(0)
    model = _model()
    optimizer, ema = _optimizer_and_ema(model)
    for step in range(PARITY_STEPS):
        optimizer.zero_grad(set_to_none=True)
        with _pinned_draws(row_offset=1_000_000 * step):
            model(_batch(4, seed=300 + step))["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        ema.step(model.parameters())
    return {"params": _param_stats(model), "ema": _ema_stats(ema, model)}


def test_ddp_reaches_the_same_weights_as_one_process(multi_step_ddp, multi_step_reference):
    """After several real updates, 2 ranks x 2 rows must equal 1 process x 4 rows.

    A single-step gradient comparison leaves the rest of the update unchecked: gradient
    clipping runs on ``self.model.parameters()`` (the tensors DDP wraps, AFTER it has
    averaged), and AdamW's moments compound any per-step error. This walks the whole update
    path three times and compares the weights themselves, which is the thing that actually
    gets deployed.

    Tolerances are looser than the single-step test's because three Adam steps amplify
    float32 reassociation; a wrong reduction moves these by O(1) relative, not by 1e-3.
    """
    for result in multi_step_ddp:
        _assert_stats_match(result["params"], multi_step_reference["params"],
                            f"rank {result['rank']} weights after {PARITY_STEPS} steps",
                            rel=5e-3)


def test_ddp_ema_shadow_matches_one_process(multi_step_ddp, multi_step_reference):
    """The EMA shadow -- diffusion's DEPLOY weights -- must not drift under DDP.

    ``save_checkpoint`` writes ``model.safetensors`` from the EMA parameters whenever EMA is
    enabled, and it is enabled by default for diffusion. The shadow is maintained per rank
    from the local (post-all-reduce) parameters and only rank 0's copy is ever saved, so a
    replica-induced difference here would ship silently: training curves would look normal
    because they are computed from the LIVE weights, and the deployed checkpoint would be
    something no rank's training ever validated.
    """
    for result in multi_step_ddp:
        _assert_stats_match(result["ema"], multi_step_reference["ema"],
                            f"rank {result['rank']} EMA shadow", rel=5e-3)
    left, right = multi_step_ddp
    _assert_stats_match(left["ema"], right["ema"], "EMA shadow across ranks", rel=1e-6)


def test_ddp_gradient_norms_stay_finite_and_clipped(multi_step_ddp):
    """Gradient stability across DDP steps: finite, non-zero, and actually clipped.

    The user's explicit requirement ("gradient 不要乱掉"). Clipping runs AFTER backward, by
    which point DDP has averaged, so the post-clip norm must respect ``max_norm`` exactly as
    single-GPU does -- asserted on the RECOMPUTED norm, because ``clip_grad_norm_`` returns
    the norm as it was BEFORE clipping.
    """
    # A literal, not `result["clip"]`: reading the threshold back from the run under test
    # makes the assertion self-fulfilling (with max_norm=inf nothing exceeds infinity).
    bound = 1.0
    for result in multi_step_ddp:
        assert result["clip"] == bound, (
            f"worker clipped at {result['clip']} but this test asserts {bound}; update both")
        assert result["n_no_grad"] == 0, (
            f"{result['n_no_grad']} of {result['n_params']} parameters received no gradient; "
            "find_unused_parameters=False is no longer safe")
        for step, norm in enumerate(result["preclip"]):
            assert math.isfinite(norm), f"step {step}: non-finite pre-clip norm {norm}"
            assert norm > 0.0, f"step {step}: zero gradient norm -- nothing is learning"
        for step, norm in enumerate(result["postclip"]):
            assert math.isfinite(norm), f"step {step}: non-finite post-clip norm {norm}"
            assert norm <= bound * (1 + 1e-4), (
                f"step {step}: post-clip norm {norm} exceeds max_norm {bound}")

    # Both ranks see the same AVERAGED gradient, so they must agree on its norm. A missing
    # all-reduce would let them drift apart here.
    left, right = (r["preclip"] for r in multi_step_ddp)
    for step, (a, b) in enumerate(zip(left, right)):
        assert a == pytest.approx(b, rel=1e-6), (
            f"step {step}: ranks disagree on the gradient norm ({a} vs {b}); the gradient "
            "was not averaged across ranks")


def test_ddp_diffusion_loss_actually_falls(multi_step_ddp):
    """The user's "无法收敛" requirement: DDP training must LEARN, not merely run.

    Every failure this file guards against elsewhere is silent, and so is this one -- a
    correctly-reduced gradient applied through a broken schedule still produces plausible
    logs. On a fixed batch with the noise and timesteps pinned, the objective is deterministic
    and an epsilon-prediction model has enough capacity to fit 2 rows, so the loss must fall
    substantially and stay finite.
    """
    for result in multi_step_ddp:
        losses = result["losses"]
        assert len(losses) == CONVERGENCE_STEPS
        assert all(math.isfinite(v) for v in losses), (
            f"rank {result['rank']}: non-finite loss in {losses[:5]}...")
        first, last = losses[0], min(losses[-5:])
        assert last < 0.5 * first, (
            f"rank {result['rank']}: loss only moved {first:.4f} -> {last:.4f} over "
            f"{CONVERGENCE_STEPS} steps; DDP training is not converging")


# ---------------------------------------------------------------------------
# Spawn C: the full Trainer under DDP -- loaders, validation reduction, rank-0 writes.
# ---------------------------------------------------------------------------
class _DeterministicModel(nn.Module):
    """A stand-in whose loss is a fixed function of the batch CONTENT.

    Deliberately RNG-free where the loss is concerned: the sharded validation objective is
    compared against a single-process one below, and a ``torch.rand_like`` in the loss would
    make that comparison measure RNG alignment instead of the reduction.

    It does draw ONE value per forward, recorded and returned but never used, purely so the
    test can see whether each rank got its own RNG stream inside the real training loop.

    TWO parameters, not one, and that is load-bearing. With a single scalar the gradient has
    only a magnitude, and ``clip_grad_norm_`` rescales it to exactly ``grad_clip`` on every
    rank -- so two ranks descending their OWN un-averaged shard gradients still land on
    identical weights, and the assertion in
    ``test_train_step_goes_through_the_ddp_view_and_keeps_the_ranks_identical`` passes for the
    wrong reason (verified: the mutation that forwards through the unwrapped module went
    undetected). A 2-vector gradient has a DIRECTION, which clipping preserves and which
    differs per shard.
    """

    def __init__(self, _config=None):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.draws: list[float] = []
        self._last_actions_valid = False

    def forward(self, batch):
        self.draws.append(float(torch.rand(())))
        error = batch["actions"] * self.scale + self.bias
        mask = batch.get("action_mask", torch.ones_like(error, dtype=torch.bool)).to(error.dtype)
        num = (error.square() * mask).sum()
        den = mask.sum()
        sums = {"overall": (num, den), "force": (num, den)}
        return {"loss": num / den.clamp(min=1), "metrics": {}, "metric_sums": sums}

    def reset(self):
        self._last_actions_valid = False

    def sample_actions(self, batch, generator=None, **_kwargs):
        return torch.zeros_like(batch["actions"])


def _trainer_config(zarr_path, output_dir, batch_size: int):
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    return TrainConfig(
        algo="diffusion",
        # No norm stats in this fixture, and TrainConfig refuses clip_sample against
        # unnormalized actions (DDIM's per-step x0 clamp would crush every real pose). The
        # stub model never samples, so the flag is irrelevant to what is under test here.
        diffusion=DPDiffusionConfig(clip_sample=False),
        hl_data=HLDataConfig(
            zarr_paths=(str(zarr_path),),
            val_last_episodes=1,
            drop_glitch_windows=False,
            obs_history_len=1,
        ),
        batch_size=batch_size,
        num_workers=0,
        device="cpu",
        bf16=False,
        ema=False,
        max_steps=6,
        save_every=0,
        eval_seeds=(0, 1),
        sample_every_epochs=0,
        sample_diagnostic_batches=0,
        artifact_every_epochs=0,
        epoch_artifact_every_epochs=0,
        output_dir=str(output_dir),
    )


def _trainer_worker(rank: int, world_size: int, zarr_path: str, output_dir: str, batch_size: int):
    """Build and run the REAL Trainer under a process group, with a stub model.

    A stub model, not a real diffusion net: what is under test here is the trainer's
    plumbing -- shard geometry, epoch accounting, the validation all-reduce, and which rank
    touches the filesystem. Substituted by NAME through the module globals, which is exactly
    how ``_ALGO_MODELS`` resolves it.
    """
    from pathlib import Path

    from fuji.s1_policy.training import trainer as trainer_module

    torch.set_num_threads(1)
    trainer_module.DPDiffusionModel = _DeterministicModel
    trainer = trainer_module.Trainer(_trainer_config(zarr_path, output_dir, batch_size))
    assert trainer.device.type == "cpu", (
        f"--device cpu must be honoured under torchrun, got {trainer.device}")
    _, val_loader = trainer.build_dataloaders()
    shard = list(iter(trainer.train_sampler))
    val_rows = sum(int(b["actions"].shape[0]) for b in val_loader)
    # BEFORE train(): its `finally` closes the dataset, and the reference below is an
    # untrained model too, so this is the comparison that isolates the reduction.
    val_means = trainer.evaluate()
    # Validation draws are seed-pinned by `isolated_rng`, hence identical on every rank.
    # Drop them so `first_draws` below is unambiguously the TRAINING stream.
    trainer.model.draws.clear()
    trainer.train()

    from torch.nn.parallel import DistributedDataParallel

    out = Path(output_dir)
    step_dirs = sorted(out.glob("step_*"))
    return {
        "rank": rank,
        "fwd_is_ddp": isinstance(trainer._fwd_model, DistributedDataParallel),
        "model_is_unwrapped": not any(
            k.startswith("module.") for k in trainer.model.state_dict()),
        # The trained parameters, bit-exact. Their values are the whole point: the stub's loss
        # depends on the batch CONTENT and the ranks hold disjoint rows, so identical weights
        # after 6 real train_step() calls can only mean the gradient was averaged.
        "weights": (float(trainer.model.scale.detach()), float(trainer.model.bias.detach())),
        "shard": shard,
        "shard_size": trainer.train_sampler.shard_size,
        "batches_per_epoch": trainer.train_sampler.shard_size // batch_size,
        "val_rows": val_rows,
        "val_is_sharded": trainer._val_is_sharded,
        "val_means": val_means,
        "global_step": trainer.global_step,
        "epoch": trainer.epoch,
        "step_dirs": [p.name for p in step_dirs],
        "best_dirs": sorted(p.name for p in out.glob("best_*")),
        "training_state_world_size": (
            torch.load(step_dirs[0] / "training_state.pt", map_location="cpu",
                       weights_only=False)["world_size"] if step_dirs else None
        ),
        "first_draws": trainer.model.draws[:3],
    }


@pytest.fixture(scope="module")
def trainer_ddp(tmp_path_factory):
    """Run the 2-rank Trainer once and share it; also compute the single-process reference."""
    from tests.test_hl_dataset import _write_hl_zarr

    root = tmp_path_factory.mktemp("ddp_trainer")
    zarr_path = _write_hl_zarr(root / "ddp.zarr.zip", n_frames=90, n_episodes=3)
    batch_size = 4
    results = _spawn(_trainer_worker, 2, str(zarr_path), str(root / "out_ddp"), batch_size)
    assert len(results) == 2, results

    # Single process, same dataset and batch size: the val objective must come out identical.
    from fuji.s1_policy.training import trainer as trainer_module

    saved = trainer_module.DPDiffusionModel
    trainer_module.DPDiffusionModel = _DeterministicModel
    try:
        solo = trainer_module.Trainer(_trainer_config(zarr_path, root / "out_solo", batch_size))
        solo.build_dataloaders()
        reference = {
            "val_means": solo.evaluate(),
            "train_size": len(solo.train_loader.dataset),
            "val_rows": sum(int(b["actions"].shape[0]) for b in solo.val_loader),
            "val_is_sharded": solo._val_is_sharded,
        }
        # Same sequence the workers run, so `draws` below is the TRAINING stream and is
        # comparable to rank 0's: only the global RNG feeds it, never the sample content.
        solo.model.draws.clear()
        solo.train()
        reference["draws"] = list(solo.model.draws)
    finally:
        trainer_module.DPDiffusionModel = saved
    return sorted(results, key=lambda r: r["rank"]), reference


def test_trainer_shards_the_train_set_disjointly_and_equally(trainer_ddp):
    """The loaders must partition the train subset: no row twice, no rank starved.

    "dataloader 错误加载" has two shapes and this covers both. OVERLAP means a row is trained
    on twice per epoch, which no metric reveals. UNEQUAL shard lengths mean one rank runs
    fewer batches with ``drop_last=True``, leaves the loop early, and never reaches the last
    gradient all-reduce -- the classic silent DDP deadlock.
    """
    results, reference = trainer_ddp
    shards = [r["shard"] for r in results]
    assert {len(s) for s in shards} == {reference["train_size"] // 2}, (
        f"shard sizes {[len(s) for s in shards]} for a train set of {reference['train_size']}")
    for result in results:
        assert len(result["shard"]) == result["shard_size"]
    flat = [i for shard in shards for i in shard]
    assert len(set(flat)) == len(flat), "shards overlap: a row would be trained on twice"
    assert set(flat) <= set(range(reference["train_size"]))
    # Truncation to a multiple of world_size drops fewer than world_size rows, and every rank
    # therefore runs the same number of updates.
    assert reference["train_size"] - len(flat) == reference["train_size"] % 2
    assert len({r["batches_per_epoch"] for r in results}) == 1


def test_train_step_goes_through_the_ddp_view_and_keeps_the_ranks_identical(trainer_ddp):
    """The real ``train_step`` must forward through ``_fwd_model``, not ``self.model``.

    Forwarding through the unwrapped module is the one DDP regression that leaves everything
    else looking right: the loss is computed, ``backward`` runs, the optimizer steps, the log
    lines are plausible -- but no reducer was armed, so each rank descends its OWN shard's
    gradient and the replicas silently become N different models. Rank 0's is the one saved,
    so the run costs N GPUs and delivers a 1-GPU model trained on 1/N of the data.

    The stub's loss is a function of the batch content and the ranks hold disjoint rows, so
    after six real updates an averaged gradient is the ONLY way both ranks can hold bit-equal
    weights. Also pins the unwrapped-``self.model`` invariant that makes checkpoints loadable
    with ``strict=True``.
    """
    results, _ = trainer_ddp
    for result in results:
        assert result["fwd_is_ddp"], "the trainer did not build a DDP view under torchrun"
        assert result["model_is_unwrapped"], (
            "self.model is DDP-wrapped, so every saved state_dict key gains a 'module.' "
            "prefix and every strict=True reader refuses the checkpoint")
    weights = [r["weights"] for r in results]
    assert weights[0] != pytest.approx((1.0, 0.0)), "nothing moved; nothing was trained"
    assert weights[0] == weights[1], (
        f"the ranks hold different weights after training ({weights}); the gradient was not "
        "averaged, so the replicas have diverged into different models")


def test_sharded_validation_reduces_to_the_single_process_objective(trainer_ddp):
    """Sharded validation must be the SAME NUMBER, not merely a similar one.

    ``evaluate`` all-reduces the (numerator, denominator) pairs and divides afterwards, so the
    result is the exact global ratio rather than a mean of per-rank means. If it reduced means
    instead -- or reduced already-duplicated sums -- the objective would shift, and since
    checkpoint selection reads ``val/objective`` the run would silently keep the wrong
    ``best_overall``.
    """
    results, reference = trainer_ddp
    assert all(r["val_is_sharded"] for r in results), (
        "validation was not sharded, so this proves nothing about the reduction")
    assert not reference["val_is_sharded"]
    assert sum(r["val_rows"] for r in results) == reference["val_rows"], (
        "the validation shards do not add up to the whole validation set")
    assert min(r["val_rows"] for r in results) > 0
    for result in results:
        assert set(result["val_means"]) == set(reference["val_means"])
        for key, value in result["val_means"].items():
            assert value == pytest.approx(reference["val_means"][key], rel=1e-9), (
                f"rank {result['rank']}: val/{key} is {value} but a single process over the "
                f"same anchors gets {reference['val_means'][key]}")


def test_only_rank_zero_writes_checkpoints(trainer_ddp):
    """N ranks writing byte-identical multi-GB directories to one path is a race and waste.

    Both ranks still agree on the STEP the checkpoint belongs to and both record the world
    size, because the sampler refuses a resume at a different one.
    """
    results, _ = trainer_ddp
    rank0, rank1 = results
    assert rank0["step_dirs"], "rank 0 wrote no checkpoint"
    assert rank0["best_dirs"], "rank 0 wrote no best_* checkpoint"
    assert rank1["step_dirs"] == rank0["step_dirs"], (
        "the ranks disagree about which step dirs exist; they share one output dir, so this "
        "means rank 1 wrote its own")
    assert len({r["global_step"] for r in results}) == 1
    assert len({r["epoch"] for r in results}) == 1
    assert rank0["training_state_world_size"] == 2, (
        "training_state.pt must record the world size; the sampler refuses a resume at a "
        "different one and needs it to say so")


def test_each_rank_gets_its_own_stochastic_stream(trainer_ddp):
    """Ranks must not draw the SAME random numbers, and rank 0 must match a 1-GPU run.

    Every rank seeds from ``config.seed`` so the replicas build identical weights, which left
    alone would make the per-step draws identical too -- and for diffusion those draws ARE the
    objective (one ``(epsilon, t)`` pair per row). Sharing them across 8 ranks would give 32
    distinct pairs at an advertised effective batch of 256, so most of the added GPUs would
    buy data diversity and no noise diversity. ``_decorrelate_rng_across_ranks`` fixes that,
    and leaves rank 0 untouched so a single-GPU run stays bit-identical.
    """
    results, reference = trainer_ddp
    rank0, rank1 = (r["first_draws"] for r in results)
    assert rank0 and rank1, "the stub model recorded no draws"
    assert rank0 != rank1, (
        f"both ranks drew the same values ({rank0}); the per-step noise, timesteps and image "
        "augmentation would be shared across every rank")
    assert rank0 == pytest.approx(reference["draws"][: len(rank0)]), (
        f"rank 0 ({rank0}) diverged from a single-process run "
        f"({reference['draws'][: len(rank0)]}); world_size=1 parity is the promise that makes "
        "this offset safe to add")


# ---------------------------------------------------------------------------
# The guard itself.
# ---------------------------------------------------------------------------
def test_flow_is_still_refused_under_torchrun(monkeypatch):
    """Widening DDP to diffusion must not quietly widen it to flow as well.

    Flow's training forward draws a per-sample RTC delay that excludes a variable-length
    prefix from the loss, so its per-rank valid counts differ BY CONSTRUCTION rather than
    incidentally -- and DDP averages per-rank means. Nothing has measured how far that pushes
    the reduced gradient, so it stays refused. The refusal happens before any process group is
    created, which is why this test can run in-process.
    """
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig
    from fuji.s1_policy.training.trainer import _DDP_ALGOS, Trainer

    assert "diffusion" in _DDP_ALGOS and "act" in _DDP_ALGOS
    assert "flow" not in _DDP_ALGOS

    for name, value in (("WORLD_SIZE", "2"), ("RANK", "0"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    config = TrainConfig(
        algo="flow",
        hl_data=HLDataConfig(zarr_paths=("unused.zarr.zip",)),
        device="cpu",
    )
    with pytest.raises(ValueError, match=r"implemented and verified for algo in"):
        Trainer(config)
    import torch.distributed as dist

    assert not dist.is_initialized(), "the refusal must precede process-group creation"
