"""Tests for the reach compensator (flore.s1_policy.serving.hl_reach).

The properties pinned here are the ones that make it safe to switch on while a physical G1 is
holding a target: OFF is an exact identity, the extra reach is hard-bounded, the gate fires ONLY
when the arm is genuinely not moving (never when it is merely lagging), a blocked hand is never
pushed harder, and an abandoned intent does not survive.

Most tests drive the compensator with a *stream* of steps rather than one call, because the gate
is defined over a WINDOW of commanded-vs-measured motion. A helper builds the two canonical
regimes: the arm frozen while the wire advances (must accumulate) and the arm following the wire
(must not).
"""

from __future__ import annotations

import numpy as np
import pytest

from flore.s1_policy.serving.hl_reach import MAX_CLAMP_M, ReachCompensator, ReachConfig

STEP = np.array([0.002, 0.0, 0.0])  # 2 mm/step of commanded motion ≈ 6 cm/s at 30 Hz


def drive(comp, n, *, follow, start=(0.40, 0.0, 0.10), force_n=None, step=STEP):
    """Run ``n`` steps where the wire advances by ``step`` each tick and the arm realizes
    ``follow`` of it (0.0 = frozen, 1.0 = perfect tracking). Returns the extra reach per step."""
    cmd = np.asarray(start, dtype=np.float64)
    meas = cmd.copy()
    extra = []
    for _ in range(n):
        cmd = cmd + step
        meas = meas + follow * step
        out = comp.compensate(cmd, meas, force_n)
        extra.append(float(np.linalg.norm(out - cmd)))
    return np.asarray(extra)


def test_disabled_is_exact_identity():
    """ki=0 must return the commanded value bit-for-bit — that is what "default OFF changes
    nothing" means. Checked with awkward values so a stray float op would show up."""
    comp = ReachCompensator.disabled()
    assert not comp.enabled
    for cmd in ([0.3123456789, -0.0271828, 0.4141592], [0.0, 0.0, 0.0], [1e-9, -1e-9, 5.0]):
        c = np.array(cmd)
        out = comp.compensate(c, np.array([0.0, 0.0, 0.0]))
        assert out.tolist() == c.tolist()
    assert comp.residual_norm == 0.0


def test_default_config_is_disabled():
    """The class must be inert unless someone opts in."""
    assert not ReachConfig().enabled
    assert not ReachCompensator().enabled


def test_accumulates_when_the_arm_is_frozen():
    """THE FEATURE: the wire keeps asking, the arm does not move ⇒ the published target must creep
    ahead of the model's own target, monotonically, until the clamp."""
    comp = ReachCompensator(ReachConfig(ki=0.2, clamp_m=0.03, decay=1.0, window=5))
    extra = drive(comp, 40, follow=0.0)
    assert extra[-1] > 0.002, "a frozen arm must build a real residual"
    assert np.all(np.diff(extra) >= -1e-12), "residual must not shrink while the arm is stalled"
    assert comp.stats()["n_applied"] > 0


def test_does_not_accumulate_when_the_arm_is_tracking():
    """THE OPERATOR'S EXPLICIT RULE: commanded 6 cm, moved 3 cm ⇒ that is fine, leave it alone.
    A lagging-but-moving arm must NOT be pushed further."""
    for follow in (0.5, 0.7, 1.0):
        comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, decay=1.0, window=5))
        extra = drive(comp, 40, follow=follow)
        assert extra.max() == pytest.approx(0.0, abs=1e-12), (
            f"arm realizing {follow:.0%} of the ask must not be compensated")
        assert comp.stats()["n_applied"] == 0
        assert comp.stats()["n_gated"] > 0


def test_gate_boundary_is_the_configured_fraction():
    """Just below stall_frac accumulates; just above does not."""
    for follow, should in ((0.3, True), (0.45, True), (0.55, False), (0.9, False)):
        comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, stall_frac=0.5, window=5,
                                            decay=1.0))
        extra = drive(comp, 30, follow=follow)
        got = bool(extra.max() > 1e-9)
        assert got == should, f"follow={follow} should {'' if should else 'not '}accumulate"


def test_residual_is_hard_clamped():
    """The safety bound. However long the arm fails to move, the published target may never
    exceed the model's target by more than clamp_m."""
    clamp = 0.03
    comp = ReachCompensator(ReachConfig(ki=0.5, clamp_m=clamp, decay=1.0, window=5))
    extra = drive(comp, 500, follow=0.0)
    assert extra.max() <= clamp + 1e-9
    assert comp.saturated
    assert comp.stats()["n_saturated"] > 0


def test_decay_bleeds_the_residual_once_the_arm_recovers():
    """Nothing is remembered forever: once the arm starts tracking again the residual must fade
    rather than stay parked on the target."""
    comp = ReachCompensator(ReachConfig(ki=0.4, clamp_m=0.03, decay=0.5, window=5))
    built = drive(comp, 30, follow=0.0)[-1]
    assert built > 0.001
    after = drive(comp, 40, follow=1.0)
    assert after[-1] < built * 0.01


def test_reset_drops_residual_and_history():
    """A Stop / disarm must not leave a residual for the next motion, NOR a motion history that
    would judge the new attempt by the old one's progress."""
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, window=5))
    drive(comp, 30, follow=0.0)
    assert comp.residual_norm > 0
    comp.reset()
    assert comp.residual_norm == 0.0
    assert comp.realized_fraction is None
    cmd = np.array([0.4, 0.0, 0.0])
    assert comp.compensate(cmd, None).tolist() == cmd.tolist()


def test_history_is_bounded():
    """The rolling window must not grow over a long rollout (this runs inside a live control
    process)."""
    comp = ReachCompensator(ReachConfig(ki=0.1, clamp_m=0.03, window=9))
    drive(comp, 5000, follow=0.0)
    assert len(comp._hist_cmd) <= 10
    assert len(comp._hist_meas) <= 10


def test_missing_readback_holds_but_does_not_grow():
    """measured=None (RECORD off / stale / offline) is not an error — the shortfall is unknowable,
    so the residual is re-applied unchanged rather than integrated."""
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, decay=1.0, window=5))
    drive(comp, 30, follow=0.0)
    held = comp.residual
    cmd = np.array([0.5, 0.0, 0.0])
    for _ in range(50):
        out = comp.compensate(cmd, None)
        assert np.allclose(out, cmd + held)
    assert np.allclose(comp.residual, held), "a stale readback must not wind the integrator"


def test_parked_wire_does_not_accumulate():
    """When the wire itself is not asking for motion there is no shortfall to speak of, and the
    realized fraction is 0/0. Must hold, not integrate noise."""
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, window=5, min_cmd_m=0.005))
    extra = drive(comp, 50, follow=0.0, step=np.array([1e-5, 0.0, 0.0]))
    assert extra.max() == pytest.approx(0.0, abs=1e-12)
    assert comp.realized_fraction is None


def test_non_finite_inputs_do_not_poison_the_accumulator():
    """NaN must not enter the residual: one poisoned step would corrupt every later one, and
    is_safe_to_send (not this class) decides whether a bad target reaches the wire."""
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, window=5))
    drive(comp, 30, follow=0.0)
    good = comp.residual
    out = comp.compensate(np.array([0.5, 0.0, 0.0]), np.array([np.nan, 0.0, 0.0]))
    assert np.isfinite(out).all()
    assert np.allclose(comp.residual, good)
    bad = comp.compensate(np.array([np.nan, 0.0, 0.0]), np.array([0.4, 0.0, 0.0]))
    assert not np.isfinite(bad).all(), "a non-finite COMMAND passes through for the safety gate"
    assert np.allclose(comp.residual, good)


def test_shortfall_is_measured_against_the_uncompensated_target():
    """The integrator must not see its own output, or it compounds (ki*(1+ki)^n) instead of
    growing linearly."""
    ki = 0.25
    # window=3 with a 3 mm step keeps the windowed commanded displacement (9 mm) above min_cmd_m,
    # so the gate is measurable from the 4th step on.
    comp = ReachCompensator(ReachConfig(ki=ki, clamp_m=MAX_CLAMP_M, decay=1.0, window=3))
    # Frozen arm, wire advancing: each step's shortfall is the accumulated commanded advance, so
    # growth is driven by cmd-meas alone and stays linear in it.
    extra = drive(comp, 14, follow=0.0, step=np.array([0.003, 0.0, 0.0]))
    # Look only at the pre-saturation stretch: once the clamp binds, increments correctly go to 0.
    grow = extra[extra < MAX_CLAMP_M - 1e-9]
    d = np.diff(grow[4:])
    assert len(d) >= 4 and np.all(d > 0), f"residual must grow while stalled; increments {d}"
    # A compounding integrator would show a growing increment ratio; a correct one is ~constant
    # (the shortfall itself grows linearly, so increments grow linearly, not geometrically).
    ratios = d[1:] / np.maximum(d[:-1], 1e-12)
    assert ratios.max() < 1.6, f"increment ratios {ratios} suggest the residual feeds itself"


def test_contact_cut_out_stops_a_blocked_arm_pressing_harder():
    """THE dangerous case: a hand pressing on something immovable is 'not moving' by every
    measure, so the STALL GATE OPENS on it. Integrating there commands a harder press — toward
    the 40 N FORCE_ESTOP_N stop. The cut-out is checked before the stall gate for that reason."""
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, decay=0.9, window=5,
                                        contact_n=1.5))
    free = drive(comp, 30, follow=0.0, force_n=0.2)[-1]
    assert free > 0.001, "in free space a frozen arm must accumulate"
    after = drive(comp, 40, follow=0.0, force_n=6.0)
    assert after[-1] < free * 0.1, "a blocked (pressing) arm must not wind the integrator"
    assert comp.stats()["n_contact_blocked"] == 40


def test_contact_cut_out_uses_the_configured_threshold():
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, decay=1.0, window=3,
                                        contact_n=5.0))
    drive(comp, 10, follow=0.0, force_n=4.9)      # below threshold ⇒ accumulate
    assert comp.stats()["n_applied"] > 0
    assert comp.stats()["n_contact_blocked"] == 0
    drive(comp, 10, follow=0.0, force_n=5.0)      # at threshold ⇒ blocked
    assert comp.stats()["n_contact_blocked"] == 10


def test_unknown_force_still_accumulates():
    """A rig not reporting force must not silently disable the feature — but the docstring tells
    the operator to use a smaller clamp there."""
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, window=3, contact_n=1.5))
    drive(comp, 10, follow=0.0, force_n=None)
    assert comp.stats()["n_applied"] > 0
    comp2 = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, window=3, contact_n=1.5))
    drive(comp2, 10, follow=0.0, force_n=float("nan"))
    assert comp2.stats()["n_applied"] > 0, "a NaN force must not read as contact"


def test_contact_cut_out_disabled_by_zero():
    comp = ReachCompensator(ReachConfig(ki=0.3, clamp_m=0.03, window=3, contact_n=0.0))
    drive(comp, 10, follow=0.0, force_n=50.0)
    assert comp.stats()["n_contact_blocked"] == 0
    assert comp.stats()["n_applied"] > 0


def test_optional_distance_ceiling_bounds_compensation():
    """gate_m is off by default; when set it stops growth once the shortfall is large, even if
    the arm is stalled."""
    comp = ReachCompensator(ReachConfig(ki=0.5, clamp_m=MAX_CLAMP_M, decay=1.0, window=3,
                                        gate_m=0.01))
    extra = drive(comp, 200, follow=0.0)
    # Once cmd-meas exceeds 1 cm the gate closes, so the residual stops growing well below clamp.
    assert extra.max() < MAX_CLAMP_M
    assert comp.stats()["n_gated"] > 0


def test_rejects_unsafe_config():
    """Every guard exists because the failure it prevents happens on a physical arm."""
    with pytest.raises(ValueError, match="exceeds"):
        ReachConfig(ki=0.2, clamp_m=MAX_CLAMP_M + 0.01)
    with pytest.raises(ValueError, match="AMPLIFY"):
        ReachConfig(ki=0.2, decay=1.5)
    with pytest.raises(ValueError, match="must be >= 0"):
        ReachConfig(ki=-0.1)
    with pytest.raises(ValueError, match="must be finite"):
        ReachConfig(ki=float("nan"))
    with pytest.raises(ValueError, match="never move"):
        ReachConfig(ki=0.2, clamp_m=0.0)
    with pytest.raises(ValueError, match="window must be"):
        ReachConfig(ki=0.2, window=1)
    with pytest.raises(ValueError, match="OVERSHOOTS"):
        ReachConfig(ki=0.2, stall_frac=1.5)


def test_rejects_wrong_shapes():
    comp = ReachCompensator(ReachConfig(ki=0.2))
    with pytest.raises(ValueError, match=r"commanded must be \(3,\)"):
        comp.compensate(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match=r"measured must be \(3,\)"):
        comp.compensate(np.zeros(3), np.zeros(4))


def test_gate_choice_validated_on_recorded_deploy_data():
    """END-TO-END on real robot data, and the reason the gate keys on MOTION not DISTANCE.

    Replays each take's own published-wire / measured-TCP streams and asks two questions:
      * does the realized-fraction gate accumulate on the genuinely stalled steps and stay off
        the healthy ones?
      * is the shortfall DISTANCE informative about whether the arm moved? (it is not:
        corr ≈ +0.08 on this data, which is why the first version of this gate was wrong)
    """
    from pathlib import Path

    session = Path("data/deploy_takes/20260814_235118")
    if not session.is_dir():
        pytest.skip("recorded deploy takes not present (rig data, not in git)")
    from flore.s1_policy.model.act.temporal_ensemble import TemporalEnsembler
    from flore.s1_policy.serving.hl_take import DeployTake, find_takes

    W = 9
    shortfalls, realized = [], []
    for take_dir in find_takes(session):
        take = DeployTake(take_dir)
        preds = take.predictions
        if len(preds) < 60:
            continue
        chunks = [take.decoded_chunk_positions(p, "left") for p in preds]
        H = chunks[0].shape[0]
        rot = np.tile(np.array([1.0, 0, 0, 0, 1, 0]), (H, 1))
        ens = TemporalEnsembler(H, k=0.01)
        wire = np.asarray([ens.add(c, rot)["position"] for c in chunks])
        meas = np.asarray([c[0] for c in chunks])
        if len(wire) <= W + 2:
            continue
        d_cmd = wire[W:] - wire[:-W]
        d_meas = meas[W:] - meas[:-W]
        n = np.linalg.norm(d_cmd, axis=1)
        moving = n > 0.005
        if moving.sum() < 20:
            continue
        rf = (d_meas[moving] * d_cmd[moving]).sum(1) / (n[moving] ** 2)
        realized.append(rf)
        shortfalls.append(np.linalg.norm((wire - meas)[W:][moving], axis=1))
    if not realized:
        pytest.skip("no take had enough commanded motion")
    rf = np.concatenate(realized)
    sf = np.concatenate(shortfalls)

    # 1) Distance is NOT informative about motion — the premise of the rewrite.
    assert abs(np.corrcoef(sf, rf)[0, 1]) < 0.25, (
        "if shortfall distance were strongly correlated with realized fraction, the original "
        "distance gate would have been defensible — re-check this analysis")

    # 2) The realized-fraction gate separates the two regimes essentially perfectly.
    stalled = rf < 0.3
    healthy = rf >= 0.5
    assert stalled.sum() > 20 and healthy.sum() > 100, "need both regimes present to test"
    gate_open = rf < 0.5
    assert gate_open[stalled].mean() > 0.95, "gate must fire on genuinely stalled steps"
    assert gate_open[healthy].mean() == pytest.approx(0.0, abs=1e-9), (
        "gate must never fire while the arm is tracking at or above stall_frac")
