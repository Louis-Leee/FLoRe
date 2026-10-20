"""The funnel force-recovery augmentation: the paper's math, and the traps it walked into.

What is pinned here, and why each one earned a test:

1. **The funnel schedule is 1001-Demos'.** ``w_k = w_min + (w_max-w_min)(k/T)^power``, normalized.
   Asserted as the SHAPE property that makes it a funnel (monotone, ~0 at the mouth, largest at
   the throat) plus the exact cubic ratio, so a change from ``**3`` to ``**2`` fails loudly
   instead of quietly widening every trajectory.
2. **Both endpoints are hard constraints.** ``u_0`` is the sampled start and ``u_{H-1}`` is the
   demo's last wrench, BIT-EXACT. The paper writes them as constraints
   (``x^1 = x_init, x^T = x_goal``), not as penalties, and a penalty would let the trajectory
   end somewhere the policy was never meant to land.
3. **The solve preserves ``T = r x F``.** This is the load-bearing property behind the fix in
   :func:`~flore.utils.force_funnel.sample_start`: the solve is linear and shares one matrix
   across all six axes, so a consistent input implies a consistent output *exactly*. If a
   future edit weights force and torque differently, or solves them with different matrices,
   this test is what catches it — and the consequence would be a 50% rejection rate, which is
   what the independent-perturbation version actually measured.
4. **The generated trajectory is a FUNNEL, not a snap or a drift.** Sampled starts must
   disperse, and every one must end up on the demo: variance across samples must fall
   monotonically-ish from the mouth to the throat.
5. **Rejections are counted, not hidden.** A generator that returns fewer trajectories than
   asked without saying so reads as "the data supports this everywhere".
6. **The duplicated constant matches the layout.** ``WRENCH_DIM`` is a copy of
   ``layout.FORCE_DIM``, because this module must stay importable from the collection venv
   (numpy only) — the same arrangement, pinned the same way, as ``SENSOR_TO_TCP`` in
   ``test_force_frame.py``.
7. **The module really is numpy-only**, by AST, so a future convenience import of torch or of
   ``flore.s1_policy`` cannot silently break the replay viewer that imports it across the venv
   split.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from flore.utils.force_funnel import (
    FORCE_SLICE,
    LAMBDA_ACC_BY_TARGET_MODE,
    MIN_HORIZON,
    TARGET_MODES,
    TORQUE_SLICE,
    WRENCH_DIM,
    FunnelConfig,
    WindowGate,
    check_admissible,
    funnel_reference,
    funnel_weights,
    generate_recovery_chunks,
    moment_arm_for_wrench,
    linear_reference,
    merge_step,
    observation_for,
    sample_in_cone,
    sample_start,
    select_windows,
    solve_funnel_trajectory,
    wrench_consistency,
)

H = 50
DT = 1.0 / 30.0


def _demo_chunk(horizon: int = H, *, seed: int = 0, arm=(0.0, 0.0, 0.2)) -> np.ndarray:
    """A synthetic contact window that IS ``T = r x F`` consistent, with a curved force ramp.

    Curved on purpose: the real chunks are not lines (``max|F - line|`` is 0.59x the excursion
    at the median), so a test built on a straight ramp would not exercise the thing that made
    the straight-line shortcut wrong.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, horizon)[:, None]
    force = np.array([0.4, -1.2, 0.6]) + np.array([2.5, 3.0, -1.5]) * t
    force += 0.6 * np.sin(3.0 * math.pi * t) * np.array([1.0, -0.5, 0.8])
    force += 0.02 * rng.standard_normal(force.shape)
    torque = np.cross(np.broadcast_to(np.asarray(arm, float), force.shape), force)
    return np.concatenate([force, torque], axis=1)


# ---------------------------------------------------------------------------
# 1. the funnel schedule
# ---------------------------------------------------------------------------
def test_funnel_weights_are_the_papers_cubic_schedule():
    w = funnel_weights(H, w_min=0.0, w_max=1.0, power=3.0)
    assert w.shape == (H,)
    assert w.sum() == pytest.approx(1.0)
    # A funnel: free at the mouth, binding at the throat, monotone in between.
    assert w[0] == pytest.approx(0.0)
    assert np.all(np.diff(w) >= 0.0)
    assert w[-1] == w.max()
    # The exact cubic, not a quadratic or a linear ramp: w_k/w_last == (k/(H-1))**3.
    for k in (10, 25, 40):
        assert w[k] / w[-1] == pytest.approx((k / (H - 1)) ** 3, rel=1e-9)


def test_funnel_weights_w_min_lifts_the_mouth():
    """``w_min > 0`` pulls even step 0 toward the demo, which narrows the mouth."""
    lo = funnel_weights(H, w_min=0.0, w_max=1.0)
    hi = funnel_weights(H, w_min=0.5, w_max=1.0)
    assert lo[0] == pytest.approx(0.0)
    assert hi[0] > 0.0
    assert hi[0] / hi[-1] > lo[0] / lo[-1]


def test_funnel_weights_reject_a_degenerate_schedule():
    with pytest.raises(ValueError, match="horizon"):
        funnel_weights(MIN_HORIZON - 1)
    with pytest.raises(ValueError, match="summed to zero"):
        funnel_weights(H, w_min=0.0, w_max=0.0)


# ---------------------------------------------------------------------------
# 2. the endpoints are constraints, not penalties
# ---------------------------------------------------------------------------
def test_endpoints_are_bit_exact():
    demo = _demo_chunk()
    start = np.array([-8.0, 2.0, 1.0, 0.1, 0.2, 0.3])
    # Both modes: the terminal is the REFERENCE's last row, which differs between them.
    anchor = solve_funnel_trajectory(demo, start)                       # target_mode="anchor"
    assert anchor.shape == (H, WRENCH_DIM)
    # Bit-exact, not approximately: they are written into the solution, never solved for.
    assert np.array_equal(anchor[0], start)
    assert np.array_equal(anchor[-1], demo[0])
    dmode = solve_funnel_trajectory(demo, start, cfg=FunnelConfig(target_mode="demo"))
    assert np.array_equal(dmode[0], start)
    assert np.array_equal(dmode[-1], demo[-1])


def test_endpoints_hold_for_every_lambda_and_power():
    """A weight change may reshape the interior; it must never move the endpoints."""
    demo = _demo_chunk(seed=3)
    start = np.array([12.0, -6.0, 3.0, -0.4, 0.5, 0.1])
    for lam in (0.0, 4.0, 1e4):
        for power in (1.0, 3.0, 6.0):
            for mode, terminal in (("anchor", demo[0]), ("demo", demo[-1])):
                cfg = FunnelConfig(lambda_acc=max(lam, 1e-6), funnel_power=power,
                                   target_mode=mode)
                out = solve_funnel_trajectory(demo, start, cfg=cfg)
                assert np.array_equal(out[0], start)
                assert np.array_equal(out[-1], terminal)


def test_an_unperturbed_start_on_a_smooth_demo_is_an_exact_fixed_point():
    """No perturbation must mean no augmentation.

    Exact only where BOTH objective terms are already zero — i.e. a demo with no curvature to
    smooth. That is the sharp statement of "the funnel does not distort a trajectory it has
    nothing to correct"; on a curved demo the smoothness term is nonzero at the demo itself, so
    the fixed point is a mildly smoothed demo rather than the demo (next test).
    """
    horizon = H
    t = np.linspace(0.0, 1.0, horizon)[:, None]
    demo = np.concatenate([np.array([0.5, -1.0, 0.2]) + np.array([3.0, 2.0, -1.0]) * t] * 2, axis=1)
    out = solve_funnel_trajectory(demo, demo[0], cfg=FunnelConfig(target_mode="demo"))
    assert np.allclose(out, demo, atol=1e-9)


def test_an_unperturbed_start_on_a_curved_demo_only_smooths_it():
    """The residual on a real (curved, noisy) demo is a SMOOTHING, not a different trajectory."""
    demo = _demo_chunk(seed=11)
    out = solve_funnel_trajectory(demo, demo[0], cfg=FunnelConfig(target_mode="demo"))
    excursion = float(np.linalg.norm(demo[-1, FORCE_SLICE] - demo[0, FORCE_SLICE]))
    assert np.abs(out - demo).max() < 0.05 * excursion
    # And it is smoother than what it came from — that is the only thing it changed.
    assert (np.abs(np.diff(out, n=2, axis=0)).sum() < np.abs(np.diff(demo, n=2, axis=0)).sum())


def test_both_lambdas_zero_is_refused_because_it_is_not_a_funnel():
    """With no smoothness the objective is diagonal: every step snaps to the demo and step 0
    is left as a lone discontinuity. Refused at config time, where the cause is legible."""
    with pytest.raises(ValueError, match="not a funnel"):
        FunnelConfig(lambda_vel=0.0, lambda_acc=0.0)


def test_singular_system_backstop_names_its_cause():
    """The numerical guard behind that refusal, reached with weights that underflow to zero."""
    from flore.utils.force_funnel import _funnel_system

    weights = funnel_weights(H, w_min=0.0, w_max=1.0, power=400.0)
    assert weights[1] == 0.0                       # genuinely underflowed, not merely tiny
    with pytest.raises(ValueError, match="singular"):
        _funnel_system(H, weights, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 3. the solve preserves single-contact consistency, exactly
# ---------------------------------------------------------------------------
def test_solve_preserves_T_equals_r_cross_F():
    """The property that dropped the rejection rate from 50% to 13%.

    ``T = r x F`` is linear in F, and the solve applies ONE matrix to all six axes, so a
    consistent demo plus a consistent start gives a consistent trajectory at every step.
    """
    arm = np.array([0.03, -0.11, 0.19])
    demo = _demo_chunk(arm=arm)
    f_start = np.array([-8.0, 2.0, 1.0])
    start = np.concatenate([f_start, np.cross(arm, f_start)])
    out = solve_funnel_trajectory(demo, start, cfg=FunnelConfig(target_mode="demo"))
    expected_torque = np.cross(np.broadcast_to(arm, (H, 3)), out[:, FORCE_SLICE])
    assert np.allclose(out[:, TORQUE_SLICE], expected_torque, atol=1e-9)
    assert wrench_consistency(out)[0] < 1e-6


def test_axes_solve_independently_so_units_need_no_normalizer():
    """Scaling one axis scales only that axis' solution — the basis of "N vs N.m is free"."""
    demo = _demo_chunk(seed=5)
    start = demo[0] + np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    base = solve_funnel_trajectory(demo, start, cfg=FunnelConfig(target_mode="demo"))
    scaled_demo = demo.copy()
    scaled_demo[:, 3:] *= 1000.0                     # N.m -> N.mm, say
    scaled_start = start.copy()
    scaled_start[3:] *= 1000.0
    scaled = solve_funnel_trajectory(scaled_demo, scaled_start, cfg=FunnelConfig(target_mode="demo"))
    assert np.allclose(scaled[:, :3], base[:, :3], atol=1e-9)          # force untouched
    assert np.allclose(scaled[:, 3:], base[:, 3:] * 1000.0, atol=1e-6)  # torque exactly scaled


def test_wrench_consistency_recovers_a_known_contact_point():
    arm = np.array([0.0, 0.05, 0.24])
    demo = _demo_chunk(arm=arm)
    residual, fitted = wrench_consistency(demo)
    assert residual < 1e-6
    # r is only identifiable up to a multiple of F, so pin what is observable: the torque it
    # reproduces, not the vector itself.
    assert np.allclose(np.cross(fitted, demo[0, FORCE_SLICE]), demo[0, TORQUE_SLICE], atol=1e-6)


def test_wrench_consistency_flags_an_impossible_wrench():
    """A torque PARALLEL to its force cannot come from any lever: r x F is always perpendicular."""
    chunk = np.zeros((10, WRENCH_DIM))
    chunk[:, FORCE_SLICE] = np.array([0.0, 0.0, 5.0])
    chunk[:, TORQUE_SLICE] = np.array([0.0, 0.0, 1.0])
    assert wrench_consistency(chunk)[0] > 0.9


def test_wrench_consistency_is_silent_on_a_torque_free_window():
    chunk = np.zeros((10, WRENCH_DIM))
    chunk[:, FORCE_SLICE] = np.array([1.0, 2.0, 3.0])
    residual, arm = wrench_consistency(chunk)
    assert residual == 0.0 and np.allclose(arm, 0.0)


# ---------------------------------------------------------------------------
# 4. it is a funnel: dispersed at the mouth, converged at the throat
# ---------------------------------------------------------------------------
def test_generated_trajectories_form_a_funnel():
    demo = _demo_chunk(seed=7)
    batch = generate_recovery_chunks(demo, dt=DT,
                                     cfg=FunnelConfig(n_aug=24, target_mode="demo"), rng=0)
    assert batch.n >= 1
    spread = batch.chunks[:, :, FORCE_SLICE].std(axis=0).mean(axis=1)   # (H,) across samples
    # Wide mouth, closed throat, and closing overall rather than fluctuating.
    assert spread[0] > 1.0
    assert spread[-1] == pytest.approx(0.0, abs=1e-9)
    assert spread[0] > 5.0 * spread[H // 2]
    # Every trajectory lands on the demo's last wrench.
    assert np.allclose(batch.chunks[:, -1, :], demo[-1], atol=1e-9)


def test_starts_are_out_of_distribution_by_construction():
    demo = _demo_chunk(seed=9)
    cfg = FunnelConfig(n_aug=32, force_radius_scale=2.0, force_radius_min_frac=0.35,
                       target_mode="demo")
    batch = generate_recovery_chunks(demo, dt=DT, cfg=cfg, rng=1)
    gap = np.linalg.norm(batch.starts[:, FORCE_SLICE] - demo[0, FORCE_SLICE], axis=1)
    excursion = float(np.linalg.norm(demo[-1, FORCE_SLICE] - demo[0, FORCE_SLICE]))
    radius = float(np.clip(cfg.force_radius_scale * excursion,
                           cfg.force_radius_min_n, cfg.force_radius_max_n))
    # The annulus floor is what stops the generator wasting draws on barely-OOD starts.
    assert gap.min() >= cfg.force_radius_min_frac * radius - 1e-9
    assert gap.max() <= radius + 1e-9
    assert batch.starts.shape == (batch.n, WRENCH_DIM)
    assert np.array_equal(batch.starts, batch.chunks[:, 0, :])


def test_merge_step_reports_where_the_gap_actually_closes():
    demo = _demo_chunk(seed=13)
    start = demo[0] + np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cfg = FunnelConfig(target_mode="demo")
    out = solve_funnel_trajectory(demo, start, cfg=cfg)
    k = merge_step(out, funnel_reference(demo, cfg.target_mode))
    gap = np.linalg.norm(out - demo, axis=1)
    assert 0 < k < H - 1                       # it merges, and not only on the last step
    assert gap[k] <= 0.1 * gap[0]
    assert gap[k - 1] > 0.1 * gap[0]           # k is the FIRST such step


def test_merge_step_is_zero_when_there_is_no_gap():
    demo = _demo_chunk()
    assert merge_step(demo, demo) == 0


def test_a_wider_cone_disperses_the_torque_more():
    demo = _demo_chunk(seed=17)
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    narrow = np.stack([sample_start(demo, rng_a, cfg=FunnelConfig(torque_cone_deg=5.0))
                       for _ in range(64)])
    wide = np.stack([sample_start(demo, rng_b, cfg=FunnelConfig(torque_cone_deg=90.0))
                     for _ in range(64)])
    # Same force draws (same seed, same call order), so the difference is the cone alone.
    assert np.allclose(narrow[:, FORCE_SLICE], wide[:, FORCE_SLICE])
    assert wide[:, TORQUE_SLICE].std(axis=0).mean() > narrow[:, TORQUE_SLICE].std(axis=0).mean()


def test_cone_sampling_respects_its_half_angle():
    axis = np.array([0.3, -0.7, 0.5])
    axis /= np.linalg.norm(axis)
    rng = np.random.default_rng(2)
    theta = math.radians(35.0)
    dirs = np.stack([sample_in_cone(axis, theta, rng) for _ in range(2000)])
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    ang = np.arccos(np.clip(dirs @ axis, -1.0, 1.0))
    assert ang.max() <= theta + 1e-9
    assert ang.max() > 0.9 * theta               # the rim is reached, not just the axis
    # Uniform in SOLID angle: E[cos psi] is (1+cos theta)/2. Sampling the ANGLE uniformly
    # instead would give E[cos psi] = sin(theta)/theta, which is a different number — that is
    # the mistake this pins, since it under-covers the rim we specifically want.
    assert float((dirs @ axis).mean()) == pytest.approx((1.0 + math.cos(theta)) / 2.0, abs=0.02)


def test_cone_about_a_zero_axis_falls_back_to_isotropic():
    rng = np.random.default_rng(3)
    dirs = np.stack([sample_in_cone(np.zeros(3), math.radians(10.0), rng) for _ in range(200)])
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    assert dirs.std(axis=0).min() > 0.3          # genuinely spread over the sphere


# ---------------------------------------------------------------------------
# 5. admissibility, and rejections that are counted
# ---------------------------------------------------------------------------
def test_admissibility_names_the_limit_it_hit():
    arm = np.array([0.0, 0.0, 0.15])
    ok = _demo_chunk(arm=arm)
    assert check_admissible(ok, dt=DT, reference_residual=0.3) is None

    loud = ok.copy()
    loud[:, FORCE_SLICE] *= 100.0
    loud[:, TORQUE_SLICE] *= 100.0
    assert check_admissible(loud, dt=DT, reference_residual=0.3) == "force_over_max"

    # A one-step spike of 8 N at 30 Hz is 240 N/s, over the 200 N/s cap, while |F| stays
    # under 10 N — so this isolates the RATE check from the magnitude check.
    spiky = ok.copy()
    spiky[H // 2, 0] += 8.0
    assert float(np.linalg.norm(spiky[:, FORCE_SLICE], axis=1).max()) < FunnelConfig.max_force_n
    assert check_admissible(spiky, dt=DT, reference_residual=0.3) == "dforce_over_max"

    bad = ok.copy()
    bad[:, TORQUE_SLICE] = bad[:, FORCE_SLICE] * 0.1     # torque parallel to force
    assert check_admissible(bad, dt=DT, reference_residual=0.0) == "inconsistent_wrench"


def test_consistency_budget_is_relative_to_the_source_window():
    """An augmentation is never held to a higher physical standard than its own demo."""
    chunk = _demo_chunk(arm=(0.0, 0.0, 0.15))
    chunk[:, TORQUE_SLICE] += 0.35 * chunk[:, FORCE_SLICE]      # deliberately inconsistent
    residual = wrench_consistency(chunk)[0]
    assert residual > 0.5
    cfg = FunnelConfig(consistency_slack=1.5, consistency_floor=0.5)
    assert check_admissible(chunk, dt=DT, reference_residual=0.0, cfg=cfg) == "inconsistent_wrench"
    assert check_admissible(chunk, dt=DT, reference_residual=residual, cfg=cfg) is None


def test_rejections_are_reported_not_absorbed():
    """Make every draw fail, and check the batch SAYS so instead of returning a quiet zero."""
    demo = _demo_chunk()
    cfg = FunnelConfig(n_aug=3, max_force_n=1e-6, max_tries_per_sample=2)
    batch = generate_recovery_chunks(demo, dt=DT, cfg=cfg, rng=0)
    assert batch.n == 0
    assert batch.n_rejected == 3 * 2
    assert batch.reject_reasons == {"force_over_max": 6}
    assert batch.chunks.shape == (0, H, WRENCH_DIM)      # shaped, so a consumer can concatenate


def test_generation_is_reproducible_from_its_seed():
    demo = _demo_chunk(seed=21)
    a = generate_recovery_chunks(demo, dt=DT, cfg=FunnelConfig(n_aug=6), rng=99)
    b = generate_recovery_chunks(demo, dt=DT, cfg=FunnelConfig(n_aug=6), rng=99)
    c = generate_recovery_chunks(demo, dt=DT, cfg=FunnelConfig(n_aug=6), rng=100)
    assert np.array_equal(a.chunks, b.chunks)
    assert not np.array_equal(a.chunks, c.chunks)


def test_config_rejects_nonsense():
    for kwargs, match in (
        ({"n_aug": 0}, "n_aug"),
        ({"w_min": 1.5}, r"w_min/w_max"),
        ({"w_max": 0.0}, "w_max must be"),
        ({"funnel_power": 0.0}, "funnel_power"),
        ({"lambda_acc": -1.0}, "smoothness"),
        ({"force_radius_min_frac": 1.0}, "force_radius_min_frac"),
        ({"torque_cone_deg": 0.0}, "torque_cone_deg"),
        ({"torque_mag_scale_max": 0.5}, "torque_mag_scale_max"),
        ({"max_tries_per_sample": 0}, "max_tries"),
    ):
        with pytest.raises(ValueError, match=match):
            FunnelConfig(**kwargs)


# ---------------------------------------------------------------------------
# the observation, the window gate, and the naive reference
# ---------------------------------------------------------------------------
def test_observation_is_the_starting_wrench():
    demo = _demo_chunk()
    start = demo[0] + 6.0
    chunk = solve_funnel_trajectory(demo, start)
    obs = observation_for(chunk, 1)
    assert obs.shape == (1, WRENCH_DIM)
    assert np.array_equal(obs[0], chunk[0])
    # L > 1 holds the start: a flat history where the real one had slope (documented weakness).
    held = observation_for(chunk, 6)
    assert held.shape == (6, WRENCH_DIM)
    assert np.all(held == chunk[0])
    with pytest.raises(ValueError, match="force_hist_len"):
        observation_for(chunk, 0)


def test_linear_reference_is_a_line_and_is_not_what_we_generate():
    """It exists to be DRAWN next to the funnel, so pin that it is straight and different."""
    demo = _demo_chunk(seed=23)
    start = demo[0] + np.array([7.0, -3.0, 1.0, 0.0, 0.0, 0.0])
    line = linear_reference(demo, start, "demo")
    assert np.array_equal(line[0], start)
    assert np.allclose(line[-1], demo[-1])
    assert np.allclose(np.diff(line, axis=0).std(axis=0), 0.0, atol=1e-9)   # constant slope
    funnel = solve_funnel_trajectory(demo, start, cfg=FunnelConfig(target_mode="demo"))
    # The curved demo is nothing like the line, so the funnel must not be either.
    assert np.abs(funnel - line).max() > 0.5


def test_window_gate_applies_every_criterion_and_the_stride():
    n, horizon = 40, H
    valid = np.ones((n, horizon), dtype=np.float32)
    hist = np.ones((n, 1), dtype=np.float32)
    peak = np.full(n, 5.0)
    ramp = np.full(n, 3.0)
    anchor = np.full(n, 4.0)
    target = np.full(n, 4.0)
    valid[3, 7] = 0.0          # one unsupervised step is enough to drop the anchor
    hist[5] = 0.0              # no force observation
    peak[9] = 0.4              # never touches anything
    ramp[11] = 0.1             # force does not move
    anchor[13] = 0.3           # NOT IN CONTACT AT t0 — the defect this gate was added for
    target[15] = 0.2           # the recovery target is ~no force, which is not a target
    kw = dict(anchor_force_n=anchor, target_force_n=target)
    idx = select_windows(valid, hist, peak, ramp, gate=WindowGate(stride=1), **kw)
    for dropped in (3, 5, 9, 11, 13, 15):
        assert dropped not in idx
    assert len(idx) == n - 6

    strided = select_windows(valid, hist, peak, ramp, gate=WindowGate(stride=7), **kw)
    assert np.array_equal(strided, idx[::7])


def test_contact_at_the_ENDPOINTS_is_gated_separately_from_peak_contact():
    """The 7.6% defect: a window can reach 1.5 N somewhere and still start in free space.

    ``contact_min_n`` looks at the window's PEAK, so on its own it accepts an anchor whose t0
    is out of contact — and a recovery toward ~0 N is not a contact behaviour. Pinned as the
    difference the two new gates make, so removing them fails here rather than silently
    reintroducing degenerate windows."""
    valid = np.ones((3, H), dtype=np.float32)
    hist = np.ones((3, 1), dtype=np.float32)
    peak = np.full(3, 5.0)      # every window DOES reach contact somewhere
    ramp = np.full(3, 3.0)
    anchor = np.array([4.0, 0.3, 4.0])   # window 1 starts in free space
    target = np.array([4.0, 4.0, 0.2])   # window 2 ends in free space
    peak_only = select_windows(valid, hist, peak, ramp, gate=WindowGate(stride=1))
    assert np.array_equal(peak_only, np.array([0, 1, 2]))     # the old gate takes all three
    both = select_windows(valid, hist, peak, ramp, anchor_force_n=anchor,
                          target_force_n=target, gate=WindowGate(stride=1))
    assert np.array_equal(both, np.array([0]))


def test_window_gate_can_keep_partially_valid_anchors():
    valid = np.ones((4, H), dtype=np.float32)
    valid[1, -1] = 0.0
    kw = dict(force_hist_valid=np.ones((4, 1)), peak_force_n=np.full(4, 5.0),
              ramp_n=np.full(4, 3.0))
    strict = select_windows(valid, kw["force_hist_valid"], kw["peak_force_n"], kw["ramp_n"],
                            gate=WindowGate(stride=1, require_all_valid=True))
    loose = select_windows(valid, kw["force_hist_valid"], kw["peak_force_n"], kw["ramp_n"],
                           gate=WindowGate(stride=1, require_all_valid=False))
    assert 1 not in strict and 1 in loose


def test_project_torque_makes_it_consistent_but_moves_the_endpoint():
    """The experimental arm's cost, pinned so it cannot be enabled by default unnoticed."""
    demo = _demo_chunk(seed=29, arm=(0.0, 0.0, 0.2))
    demo[:, TORQUE_SLICE] += 0.25 * demo[:, FORCE_SLICE]      # a demo that is NOT consistent
    cfg = FunnelConfig(n_aug=4, project_torque=True, consistency_floor=10.0)
    batch = generate_recovery_chunks(demo, dt=DT, cfg=cfg, rng=0)
    assert batch.n > 0
    # Compared against the REFERENCE's last row, so this holds under either target_mode.
    target = batch.reference[-1]
    for chunk in batch.chunks:
        assert wrench_consistency(chunk)[0] < 1e-6                      # consistent by force
        assert np.allclose(chunk[-1, FORCE_SLICE], target[FORCE_SLICE], atol=1e-9)
        assert not np.allclose(chunk[-1, TORQUE_SLICE], target[TORQUE_SLICE], atol=1e-3)


# ---------------------------------------------------------------------------
# cross-module contracts
# ---------------------------------------------------------------------------
def test_wrench_dim_matches_the_action_layout():
    """``WRENCH_DIM`` is a hand copy of ``layout.FORCE_DIM``; keep them equal."""
    from flore.s1_policy.data import layout

    assert WRENCH_DIM == layout.FORCE_DIM
    # And the slices must match how a hand block actually lays force out.
    slots = layout.get_layout().slots(layout.get_layout().force_roles[0])
    assert slots.force is not None
    assert slots.force.stop - slots.force.start == WRENCH_DIM
    assert FORCE_SLICE.stop - FORCE_SLICE.start == 3
    assert TORQUE_SLICE.start == 3 and TORQUE_SLICE.stop == WRENCH_DIM


def test_module_is_numpy_only_so_the_collection_venv_can_import_it():
    """The replay viewer imports this module from a DIFFERENT venv (no torch, no zarr).

    Checked by AST rather than by importing, because the import would succeed in FLoRe's own
    venv and prove nothing about the collection one.
    """
    src = Path(__file__).resolve().parents[1] / "flore" / "utils" / "force_funnel.py"
    tree = ast.parse(src.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    allowed = {"numpy", "dataclasses", "math", "__future__"}
    assert imported <= allowed, f"force_funnel.py must stay numpy-only; found {imported - allowed}"


# ---------------------------------------------------------------------------
# target_mode: WHAT the funnel converges onto — the one real modelling choice
# ---------------------------------------------------------------------------
def test_anchor_mode_converges_every_trajectory_onto_the_t0_wrench():
    """The default, and 1001-Demos' own shape: ONE fixed goal, not a moving reference.

    The paper's ``x_goal`` is a single pre-contact pose that every generated trajectory
    converges to ("converge consistently to **the same** pre-contact pose"). In wrench space
    that goal is ``d[0]`` — the nominal force for this contact, and the vector the replay
    viewer's red ``F^cmd`` arrow draws. So the recovery is visibly "come back to the red arrow".
    """
    demo = _demo_chunk(seed=31)
    batch = generate_recovery_chunks(demo, dt=DT, cfg=FunnelConfig(n_aug=8), rng=0)
    assert batch.n > 0
    assert np.allclose(batch.chunks[:, -1, :], demo[0], atol=1e-9)     # all on the SAME point
    assert np.allclose(batch.reference, demo[0], atol=0.0)             # constant reference
    # ...and that point is emphatically NOT where the demo chunk ends.
    assert np.linalg.norm(demo[0, FORCE_SLICE] - demo[-1, FORCE_SLICE]) > 1.0


def test_demo_mode_tracks_the_stroke_instead():
    demo = _demo_chunk(seed=31)
    batch = generate_recovery_chunks(demo, dt=DT,
                                     cfg=FunnelConfig(n_aug=8, target_mode="demo"), rng=0)
    assert np.allclose(batch.chunks[:, -1, :], demo[-1], atol=1e-9)
    assert np.array_equal(batch.reference, demo)


def test_anchor_mode_paths_are_straight_and_only_their_TIMING_is_tunable():
    """Why the viewer's purple bundle looks like a fan of straight rays, and why that is right.

    One FIXED target plus a solve that is linear in the boundary data forces
    ``u[k] = target + a[k] * (start - target)`` with a scalar ``a`` that depends on the horizon,
    the weights and the lambdas — but NOT on the start. So under ``anchor``:

    * every trajectory is exactly a straight segment from its own OOD start to the target, and
    * ``funnel_power`` / ``lambda_acc`` can only reshape the SPEED along that segment.

    Worth pinning because it bounds what the augmentation can teach: the shape of a recovery is
    not a free parameter here (1001-Demos gets curvature from its collision/render terms, which
    have no wrench-space analogue), and a future term that DID bend the path would silently
    break this and want re-measuring rather than a quiet test edit.
    """
    demo = _demo_chunk(seed=41)
    cfg = FunnelConfig(target_mode="anchor")
    target = demo[0]
    profiles = []
    for shift in ([-6.0, 3.0, 2.0, 0.3, -0.2, 0.4], [5.0, -4.0, 1.0, -0.1, 0.5, 0.2],
                  [0.0, 0.0, -7.0, 0.2, 0.2, -0.3]):
        start = target + np.asarray(shift)
        chunk = solve_funnel_trajectory(demo, start, cfg=cfg)
        offset = chunk - target                      # (H, 6), zero at the end
        span = start - target
        unit = span / np.linalg.norm(span)
        along = offset @ unit
        # No component perpendicular to the start->target direction: a straight segment.
        assert np.abs(offset - np.outer(along, unit)).max() < 1e-9
        profiles.append(along / np.linalg.norm(span))
    # ...and ONE shared time profile, whatever the start: 1 at the mouth, 0 at the throat.
    assert np.abs(np.asarray(profiles) - profiles[0]).max() < 1e-9
    assert profiles[0][0] == pytest.approx(1.0)
    assert profiles[0][-1] == pytest.approx(0.0)
    # It is NOT monotone: the acceleration penalty makes it sail a little past the setpoint and
    # settle back, which is what a real recovery does. Small and bounded (1.2% of the gap at the
    # default power, 3.3% at the loosest schedule), and it never re-approaches the start.
    assert -0.05 < profiles[0].min() < -1e-4
    assert profiles[0][1:].max() < 0.95
    # The knobs move the timing only. Larger power = later merge (weights lift later).
    def _steps_to_10pct(power: float) -> int:
        c = solve_funnel_trajectory(demo, target + np.array([-6.0, 0, 0, 0, 0, 0]),
                                    cfg=FunnelConfig(target_mode="anchor", funnel_power=power))
        return int(np.argmax(np.abs(c[:, 0] - target[0]) / 6.0 < 0.10))
    assert _steps_to_10pct(1.0) < _steps_to_10pct(3.0) < _steps_to_10pct(8.0)


def test_demo_mode_bends_the_path_because_the_target_moves():
    """The complement of the test above: a per-step reference is what buys curvature, and it is
    the reason the two modes look nothing alike in the viewer."""
    demo = _demo_chunk(seed=41)
    start = demo[0] + np.array([-6.0, 3.0, 2.0, 0.3, -0.2, 0.4])
    chunk = solve_funnel_trajectory(demo, start, cfg=FunnelConfig(target_mode="demo"))
    span = demo[-1] - start
    unit = span / np.linalg.norm(span)
    offset = chunk - demo[-1]
    perpendicular = offset - np.outer(offset @ unit, unit)
    assert np.abs(perpendicular).max() > 0.5          # newtons off the straight line


def test_funnel_reference_is_the_single_source_of_the_target():
    demo = _demo_chunk(seed=33)
    assert np.array_equal(funnel_reference(demo, "demo"), demo)
    anchor = funnel_reference(demo, "anchor")
    assert anchor.shape == demo.shape
    assert np.all(anchor == demo[0])
    with pytest.raises(ValueError, match="target_mode"):
        funnel_reference(demo, "sideways")


def test_batch_keeps_demo_and_reference_apart():
    """A viewer that drew ``demo`` and called it "the target" would draw the wrong curve in the
    default mode, so the two are separate fields rather than one."""
    demo = _demo_chunk(seed=35)
    batch = generate_recovery_chunks(demo, dt=DT, cfg=FunnelConfig(n_aug=2), rng=0)
    assert np.array_equal(batch.demo, demo)             # untouched context
    assert not np.array_equal(batch.reference, demo)    # and NOT the same thing


def test_merge_step_is_measured_against_the_reference_not_the_demo():
    """Under anchor mode the demo is a trajectory the funnel never aimed at; scoring against it
    would report a merge that did not happen."""
    demo = _demo_chunk(seed=37)
    batch = generate_recovery_chunks(demo, dt=DT, cfg=FunnelConfig(n_aug=1), rng=2)
    chunk = batch.chunks[0]
    assert merge_step(chunk, batch.reference) == batch.merge_steps[0]
    # The demo-scored number is a different (and here, wrong) answer.
    assert merge_step(chunk, demo) != batch.merge_steps[0]


def test_lambda_acc_defaults_per_mode_and_can_be_overridden():
    """One shared lambda would leave anchor mode 22% slower than any human demo."""
    assert FunnelConfig().lambda_acc is None
    assert FunnelConfig().resolved_lambda_acc == LAMBDA_ACC_BY_TARGET_MODE["anchor"]
    assert FunnelConfig(target_mode="demo").resolved_lambda_acc == LAMBDA_ACC_BY_TARGET_MODE["demo"]
    assert LAMBDA_ACC_BY_TARGET_MODE["anchor"] != LAMBDA_ACC_BY_TARGET_MODE["demo"]
    assert set(LAMBDA_ACC_BY_TARGET_MODE) == set(TARGET_MODES)
    assert FunnelConfig(lambda_acc=7.0).resolved_lambda_acc == 7.0
    with pytest.raises(ValueError, match="target_mode"):
        FunnelConfig(target_mode="nope")


def test_radius_basis_follows_the_target_so_a_held_setpoint_still_perturbs():
    """In anchor mode the reference does not move, so the chunk's excursion is the wrong scale.

    Build a demo whose force ENDS where it started: its excursion is ~0, so a
    ``|d[H-1] - d[0]|`` radius would collapse to the floor while ``|d[0]|`` stays meaningful.
    """
    t = np.linspace(0.0, 1.0, H)[:, None]
    force = np.array([4.0, 0.0, 0.0]) + np.array([0.0, 3.0, 0.0]) * np.sin(np.pi * t)
    demo = np.concatenate([force, np.cross(np.broadcast_to([0.0, 0.0, 0.2], force.shape), force)],
                          axis=1)
    assert np.linalg.norm(demo[-1, FORCE_SLICE] - demo[0, FORCE_SLICE]) < 1e-6   # no excursion
    cfg = FunnelConfig(n_aug=16, force_radius_min_n=0.01)      # floor out of the way
    gap = np.linalg.norm(
        generate_recovery_chunks(demo, dt=DT, cfg=cfg, rng=0).starts[:, FORCE_SLICE]
        - demo[0, FORCE_SLICE], axis=1)
    # |d[0]| is 4 N, so at scale 2.0 the annulus reaches several newtons out.
    assert gap.max() > 2.0
    demo_mode = np.linalg.norm(
        generate_recovery_chunks(demo, dt=DT,
                                 cfg=dataclasses.replace(cfg, target_mode="demo"),
                                 rng=0).starts[:, FORCE_SLICE] - demo[0, FORCE_SLICE], axis=1)
    assert demo_mode.max() < 0.02        # collapsed to the floor, as the excursion implies


def test_moment_arm_for_wrench_reproduces_the_perpendicular_torque_exactly():
    """The single-pair lever. Its residual is exactly the force-PARALLEL torque, which no lever
    can produce — which is why the admissibility budget must come from the reference."""
    f = np.array([3.0, -1.0, 2.0])
    t_perp = np.cross(np.array([0.0, 0.1, 0.2]), f)
    r = moment_arm_for_wrench(np.concatenate([f, t_perp]))
    assert np.allclose(np.cross(r, f), t_perp, atol=1e-12)
    # Add a torque along the force: irreducible, and it shows up as the whole residual.
    t_par = 0.5 * f / np.linalg.norm(f)
    r2 = moment_arm_for_wrench(np.concatenate([f, t_perp + t_par]))
    assert np.allclose(np.cross(r2, f), t_perp, atol=1e-12)
    assert moment_arm_for_wrench(np.zeros(WRENCH_DIM)).tolist() == [0.0, 0.0, 0.0]
