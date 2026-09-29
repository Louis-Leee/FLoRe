"""Unit tests for the TCP precision rewards + gripper-mass DR.

Plan: docs/2026-08-05-tcp-precision-translation-orientation-plan.md. Applied IN PLACE to
`exp:g1-wbt-3pt-hybrid-unified[-anneal]` (user 2026-08-05), not to a `-tcp` fork.

Two things these tests exist to protect, beyond the terms' own math:

  1. **The obs must not move.** `pfl96opy` warm-starts into these presets, so any change
     to the 170D actor obs would silently mis-feed a loaded checkpoint. Pinned hard.
  2. **Eq.2's offset cancellation is the PREMISE of the whole change.** If someone later
     "fixes" `_force_ref_pos_w` to use the reference quaternion, `tcp_position_error_exp`
     becomes a duplicate of Eq.2 and the double pricing should be removed. The
     cancellation is therefore asserted directly — the test goes red when the premise dies.

All numeric/config assertions; no IsaacSim runtime (the reward funcs are exercised against
hand-built tensor doubles, which is exactly what the "PURE READER" contract buys).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import os
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
_YAM_URDF = _REPO_ROOT / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"

_HANDS = ("left_wrist_yaw_link", "right_wrist_yaw_link")

# Phase 1 is the DEFAULT shipping config; `WBT_HAND_MASS_DR=1` deliberately changes
# randomization / robot / the cfg carriers, so the phase-1 invariant tests below cannot hold
# in that mode. Skip them there rather than leaving a phase-2 operator with a red suite —
# the phase-2 assembly has its own coverage (see `_overrides_with_dr`), which does NOT depend
# on the env var because it patches the flag directly.
_phase1_only = pytest.mark.skipif(
    os.environ.get("WBT_HAND_MASS_DR", "0").strip().lower() in ("1", "true", "yes"),
    reason="phase-1 invariant; WBT_HAND_MASS_DR=1 intentionally changes these",
)


# ---------------------------------------------------------------------------
# Env double. Carries exactly the tensors the pure readers touch.
# ---------------------------------------------------------------------------
class _MotionCmd:
    def __init__(self, pos_rel, quat_rel_xyzw, robot_quat_xyzw):
        self.body_pos_relative_w = pos_rel
        self.body_quat_relative_w = quat_rel_xyzw
        self.robot_body_quat_w = robot_quat_xyzw
        self.motion_cfg = type("_MC", (), {"body_names_to_track": list(_HANDS)})()


class _CmdMgr:
    def __init__(self, mc):
        self._mc = mc

    def get_state(self, _name):
        return self._mc


class _Robot:
    def __init__(self, pos_w, quat_w_wxyz):
        self.data = type("_D", (), {"body_pos_w": pos_w, "body_quat_w": quat_w_wxyz})()


class _Sim:
    def __init__(self, robot):
        self._robot = robot
        self.body_names = list(_HANDS)

    def find_rigid_body_indice(self, name):
        return list(_HANDS).index(name)


class _Env:
    """Minimal env exposing only what the TCP reward readers consume."""

    def __init__(self, *, wrist_pos_w, wrist_quat_wxyz, ref_pos_w, ref_quat_xyzw, contact_phase=None, warmup=0):
        n = wrist_pos_w.shape[0]
        self.device = "cpu"
        self.num_envs = n
        self.simulator = _Sim(_Robot(wrist_pos_w, wrist_quat_wxyz))
        self.command_manager = _CmdMgr(_MotionCmd(ref_pos_w, ref_quat_xyzw, _wxyz_to_xyzw(wrist_quat_wxyz)))
        self._wbt_contact_phase = torch.zeros(n, len(_HANDS)) if contact_phase is None else contact_phase
        self.episode_length_buf = torch.full((n,), 1000, dtype=torch.long)
        self._wbt_hybrid_cfg = type("_HC", (), {"contact_phase_warmup_steps": warmup})()


def _wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def _quat_axis_angle_wxyz(axis, angle, shape):
    """(w,x,y,z) rotation of `angle` about unit `axis`, broadcast to `shape + (4,)`."""
    axis = torch.tensor(axis, dtype=torch.float)
    axis = axis / axis.norm()
    q = torch.cat([torch.cos(torch.tensor(angle / 2.0)).reshape(1), torch.sin(torch.tensor(angle / 2.0)) * axis])
    return q.expand(*shape, 4).clone()


def _identity_wxyz(shape):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    return q.expand(*shape, 4).clone()


def _env_with_orientation_error(theta: float, *, contact_phase=None, warmup=0, n=1):
    """Wrist ORIGIN perfectly tracked; wrist ATTITUDE off by `theta` about +z.

    Nothing else differs, so the entire TCP position error is the orientation lever.
    """
    h = len(_HANDS)
    pos = torch.zeros(n, h, 3)
    return _Env(
        wrist_pos_w=pos,
        wrist_quat_wxyz=_quat_axis_angle_wxyz((0.0, 0.0, 1.0), theta, (n, h)),
        ref_pos_w=pos.clone(),
        ref_quat_xyzw=_wxyz_to_xyzw(_identity_wxyz((n, h))),
        contact_phase=contact_phase,
        warmup=warmup,
    )


# ---------------------------------------------------------------------------
# 1. The premise: Eq.2's FK offset cancels, so it does NOT price the TCP.
# ---------------------------------------------------------------------------
def test_eq2_offset_really_cancels_so_the_tcp_is_unpriced():
    """`force_track_eq2_servo` must be INVARIANT to the FK offset.

    This is the load-bearing premise of the whole change (plan §1.2): both
    `_force_ref_pos_w` and `_force_cur_pos_w` rotate the offset by the SAME live
    `body_quat_w`, so `R·d` subtracts out and Eq.2 constrains the wrist_yaw ORIGIN.

    Run with a wrist deliberately 0.2 rad off — i.e. the one regime where a
    TCP-aware term would differ — and with d=0.242 vs d=0. Bit-identical ⇒ Eq.2
    genuinely cannot see the TCP, so `tcp_position_error_exp` is not duplicating it.

    If this test ever fails because Eq.2 was changed to use the REFERENCE quaternion,
    the two terms now measure the same thing and the double pricing must be revisited.
    """
    from wbt_training.config_values.force_tracking import HandForceConfig
    from wbt_training.config_values.hybrid import force_track_eq2_servo

    n, h = 2, len(_HANDS)
    env = _env_with_orientation_error(0.2, n=n)
    # Eq.2 reads these buffers directly.
    env._wbt_force_cmd_w = torch.zeros(n, h, 3)
    env._wbt_force_ext_w = torch.zeros(n, h, 6)
    env._wbt_delta_x = torch.zeros(n, h, 3)
    env._wbt_hybrid_cfg = type("_HC", (), {"servo_enabled": True, "contact_phase_warmup_steps": 0})()

    out = []
    for d in (0.242, 0.0):
        env._wbt_force_offsets_b = torch.tensor([(d, 0.0, 0.0)] * h)
        # drop the per-env index caches so each pass re-resolves cleanly
        for attr in ("_wbt_force_motion_indices", "_wbt_force_isaac_indices"):
            if hasattr(env, attr):
                delattr(env, attr)
        out.append(force_track_eq2_servo(env, cfg=HandForceConfig(body_names=_HANDS)))
    assert torch.equal(out[0], out[1]), "Eq.2 became offset-SENSITIVE — re-check the double-pricing argument"
    assert float(out[0][0]) == pytest.approx(1.0), "origin is perfectly tracked, so Eq.2 should read ~1.0"


# ---------------------------------------------------------------------------
# 2. The position term prices the orientation lever (the thing Eq.2 drops).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.05, 0.10, 0.20, 0.40])
def test_tcp_position_term_prices_the_orientation_lever(theta):
    """With the origin perfect and attitude off by θ, the implied TCP error must be the
    rigid-lever chord 2·d·sin(θ/2) — 1.2 cm at 0.05 rad up to 9.6 cm at 0.4 rad (plan §1.3).

    Invert the kernel to recover the distance the reward actually saw, rather than
    asserting a magic reward number: r = exp(−err²/σ²) ⇒ err = σ·sqrt(−ln r).
    """
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    d, sigma = 0.242, 0.10
    env = _env_with_orientation_error(theta)
    r = tcp_position_error_exp(env, sigma=sigma, tcp_offset_x=d, servo_shifted=False)
    implied = sigma * math.sqrt(-math.log(float(r[0])))
    assert implied == pytest.approx(2.0 * d * math.sin(theta / 2.0), rel=1e-4)


def test_tcp_position_term_is_one_when_perfect():
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    env = _env_with_orientation_error(0.0)
    assert float(tcp_position_error_exp(env, servo_shifted=False)[0]) == pytest.approx(1.0)


def test_tcp_position_term_prices_pure_translation_too():
    """A pure origin offset (attitude perfect) must still be priced — the term is not
    only an orientation proxy. 3 cm on one axis at σ=0.10 ⇒ exp(−0.0009/0.01)."""
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    n, h = 1, len(_HANDS)
    env = _Env(
        wrist_pos_w=torch.tensor([[[0.03, 0.0, 0.0], [0.03, 0.0, 0.0]]]),
        wrist_quat_wxyz=_identity_wxyz((n, h)),
        ref_pos_w=torch.zeros(n, h, 3),
        ref_quat_xyzw=_wxyz_to_xyzw(_identity_wxyz((n, h))),
    )
    r = float(tcp_position_error_exp(env, sigma=0.10, servo_shifted=False)[0])
    assert r == pytest.approx(math.exp(-(0.03**2) / 0.10**2), rel=1e-6)


# ---------------------------------------------------------------------------
# 3. The servo-Δx target: agrees with Eq.2 in contact, and is NOT reward-hackable.
# ---------------------------------------------------------------------------
def _env_with_offsets(cur_offsets, *, delta_x=None, n=1):
    """Attitude perfect, ref at the origin, per-hand CURRENT positions = `cur_offsets`."""
    h = len(_HANDS)
    cur = torch.zeros(n, h, 3)
    for hand, dx in enumerate(cur_offsets):
        cur[:, hand, 0] = dx
    env = _Env(
        wrist_pos_w=cur,
        wrist_quat_wxyz=_identity_wxyz((n, h)),
        ref_pos_w=torch.zeros(n, h, 3),
        ref_quat_xyzw=_wxyz_to_xyzw(_identity_wxyz((n, h))),
    )
    if delta_x is not None:
        env._wbt_delta_x = delta_x
    return env


def test_tcp_position_target_follows_the_servo_delta_x():
    """The target must be `ref_tcp + Δx`, the SAME setpoint Eq.2 chases.

    In contact the integrator pushes the position target INTO the wall — that overshoot IS
    the UniFP force mechanism — so a hand sitting at `ref + Δx` is CORRECT and must score
    ~1.0, not be penalised for the 4 cm it is away from the kinematic ref. That agreement
    is what lets this term coexist with Eq.2 without a contact gate.
    """
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    n, h, sigma = 1, len(_HANDS), 0.10
    dx = torch.zeros(n, h, 3)
    dx[:, :, 0] = 0.04  # servo pushed the setpoint 4 cm into the wall

    # Hand sits exactly on the SHIFTED setpoint → perfect.
    on_target = _env_with_offsets((0.04, 0.04), delta_x=dx)
    assert float(tcp_position_error_exp(on_target, sigma=sigma)[0]) == pytest.approx(1.0, abs=1e-6)

    # Same geometry, but ignoring Δx would score it as 4 cm off — proving the shift is live.
    assert float(tcp_position_error_exp(on_target, sigma=sigma, servo_shifted=False)[0]) == pytest.approx(
        math.exp(-(0.04**2) / sigma**2), rel=1e-6
    )

    # Hand at the KINEMATIC ref while the servo asked for +4 cm → priced as 4 cm off.
    at_ref = _env_with_offsets((0.0, 0.0), delta_x=dx)
    assert float(tcp_position_error_exp(at_ref, sigma=sigma)[0]) == pytest.approx(
        math.exp(-(0.04**2) / sigma**2), rel=1e-6
    )

    # Free space (Δx = 0) must be identical to the pure kinematic form.
    free = _env_with_offsets((0.02, 0.02), delta_x=torch.zeros(n, h, 3))
    assert float(tcp_position_error_exp(free, sigma=sigma)[0]) == pytest.approx(
        float(tcp_position_error_exp(free, sigma=sigma, servo_shifted=False)[0]), rel=1e-9
    )


def test_tcp_position_is_not_reward_hackable_by_making_contact():
    """REGRESSION GUARD for the gate this term used to have (2026-08-05).

    The first version weighted per-hand error by `g = 1 − contact_phase` and normalised by
    `Σg` (clamped at 1e-3). As both hands approached contact the numerator → 0 while the
    denominator hit the clamp, so r → exp(0) = 1.0: the term paid its FULL w=4.0 no matter
    how badly the TCP tracked. Worth up to +2.5/step at 10 cm error and +3.9 at 20 cm
    (vs feet_slip = −2.0), and strongest in early training when errors are largest — i.e.
    "press the wall, stop tracking, collect full marks".

    So: contact_phase must have NO influence, and a badly-tracking hand must stay penalised
    at every contact_phase value.
    """
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    sigma = 0.10
    want = math.exp(-(0.10**2) / sigma**2)  # both hands 10 cm off ⇒ ~0.368, NOT 1.0
    rewards = []
    for phase in (0.0, 0.5, 0.999, 1.0):
        env = _env_with_offsets((0.10, 0.10))
        env._wbt_contact_phase = torch.full((1, 2), phase)
        r = float(tcp_position_error_exp(env, sigma=sigma)[0])
        assert r == pytest.approx(want, rel=1e-6), f"contact_phase={phase} changed the reward"
        rewards.append(r)
    assert max(rewards) - min(rewards) < 1e-9, "contact_phase must not influence this term at all"
    assert max(rewards) < 0.5, "a 10 cm error must stay penalised, never pay full marks"


def test_tcp_position_prices_each_hand_independently():
    """ASYMMETRIC errors: left 1 cm, right 8 cm. The reward must be the mean of the two
    SQUARED errors — catching a per-hand mix-up, a wrong reduction axis, or a mean taken
    over distances instead of squares. Kernel inverted so a failure names the distance."""
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    sigma = 0.10
    r = tcp_position_error_exp(_env_with_offsets((0.01, 0.08)), sigma=sigma)
    implied = sigma * math.sqrt(-math.log(float(r[0])))
    assert implied == pytest.approx(math.sqrt((0.01**2 + 0.08**2) / 2.0), rel=1e-5)
    # Swapping the two hands must not change the (symmetric) mean.
    r2 = tcp_position_error_exp(_env_with_offsets((0.08, 0.01)), sigma=sigma)
    assert float(r2[0]) == pytest.approx(float(r[0]), rel=1e-9)
    # A single bad hand must NOT be masked by a good one.
    one_bad = float(tcp_position_error_exp(_env_with_offsets((0.0, 0.08)), sigma=sigma)[0])
    both_good = float(tcp_position_error_exp(_env_with_offsets((0.0, 0.0)), sigma=sigma)[0])
    assert one_bad < 0.9 * both_good


def test_force_alpha_needs_both_init_and_step_to_pin_full_strength():
    """Pinning force_alpha at 1.0 requires alpha_step=0.0 as well as force_alpha_init=1.0.

    `force_alpha_init` only sets the STARTING value. `ForceAlphaCurriculum.reset()` then keeps
    stepping α DOWN by `alpha_step` whenever `average_episode_length < level_down_threshold`,
    so init alone walks α back off 1.0 during any rough patch — i.e. "full strength from step
    0" silently becomes "full strength until the first bad stretch". The repo's own
    curriculum-disabled convention (base `g1_wbt_3pt_hybrid`) is init=1.0 AND step=0.0.

    Simulates the ramp arithmetic rather than booting a curriculum manager, so it pins the
    REASON the second flag is needed.
    """
    from wbt_training.config_values.experiment import DEFAULTS

    hf = DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"].curriculum.setup_terms["force_alpha"].params["hand_force_config"]

    def ramp(init, step, avg_lengths):
        a = init
        for avg in avg_lengths:
            if avg >= hf.level_up_threshold:
                a += step
            elif avg < hf.level_down_threshold:
                a -= step
            a = max(0.0, min(hf.force_alpha_max, a))
        return a

    rough = [500.0] * 3 + [80.0] * 30  # healthy, then episodes collapse
    assert ramp(1.0, hf.alpha_step, rough) < 0.75, "init alone does NOT hold α at 1.0"
    assert ramp(1.0, 0.0, rough) == 1.0, "init + step=0.0 pins it"
    # The shipped default really is a live ramp from zero (so the override is meaningful).
    assert (hf.force_alpha_init, hf.alpha_step) == (0.0, 0.01)
    # And the base preset is the precedent for the disabled form.
    base = DEFAULTS["g1_wbt_3pt_hybrid"].curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    assert (base.force_alpha_init, base.alpha_step) == (1.0, 0.0)


def test_servo_anneal_is_force_alpha_gated():
    """Pinning force_alpha=1.0 OPENS the R1 servo-anneal gate at iteration 0 — a real side
    effect of running full force, worth pinning so it is not rediscovered mid-run.

    `ServoAuthorityCurriculum.reset()` returns early while α < 0.999, so with the default ramp
    the servo cannot retire until the force task reaches full strength. It also means the
    documented way to hold the servo at full authority (`force_err_threshold_n=0.0`) works
    because the error EMA is a norm and can never be < 0.0.
    """
    import inspect

    from wbt_training.config_values.hybrid_unified import ServoAuthorityCurriculum

    src = inspect.getsource(ServoAuthorityCurriculum.reset)
    assert "_wbt_force_alpha" in src and "0.999" in src, "the force-alpha readiness gate must still exist"
    assert "if self._err_ema < self.err_threshold_n:" in src, "threshold=0.0 must be unsatisfiable for a norm"


def test_delta_x_carries_a_free_space_preload_not_zero():
    """Δx is NOT zero in free space — it is `F_cmd/K`, and both terms carry it.

    An earlier docstring claimed Δx vanishes without contact. It does not:
    `integrator_observe_step` gates only the INTEGRAL (`new_integ = gate·clip(…)`), while
    `task_wrench = S⊙(F_cmd + I)` and `Δx = task_wrench/K` — so with the hard contact gate
    at 0 the integral is 0 but Δx is still F_cmd/K. That is intentional UniFP pre-load (the
    arm extends toward the wall by F/K so contact is met with the right force), and it is
    up to 12.5 cm for ~84% of an episode.

    Pinned because the WRONG claim is the dangerous one: it would make someone read
    `tcp_position_error_exp` raw as distance-to-clip, and it would make "the two terms agree
    in free space because there is no shift" sound like a proof when the real reason is that
    both read the same buffer.
    """
    from wbt_training.config_values.hybrid import HybridContactConfig, integrator_observe_step

    cfg = HybridContactConfig()
    f_cmd = torch.zeros(1, 2, 3)
    f_cmd[..., 0] = 8.0  # 8 N commanded along +x
    free_space_gate = torch.zeros(1, 2)  # hard contact gate = 0
    new_integ, delta_x, task_wrench = integrator_observe_step(
        f_cmd, torch.zeros(1, 2, 3), free_space_gate, torch.zeros(1, 2, 3), cfg, dt=0.005
    )
    assert torch.equal(new_integ, torch.zeros(1, 2, 3)), "the INTEGRAL must be gated off"
    assert task_wrench[0, 0, 0] == pytest.approx(8.0), "but the feedforward F_cmd is NOT gated"
    assert float(delta_x[0, 0, 0]) == pytest.approx(8.0 / cfg.stiffness_K, rel=1e-6)
    assert float(delta_x[0, 0, 0]) > 0.03, "so free-space Δx is centimetres, not zero"

    # Worst case over the K_env range: the reachable-force cap sets the pre-load.
    worst = min(cfg.f_cmd_max, 400.0 * cfg.max_penetration_m) / cfg.stiffness_K
    assert worst == pytest.approx(0.125, rel=1e-6), "12.5 cm pre-load at the F_cmd ceiling"

    # And the reward really does follow that shifted target rather than the kinematic ref.
    n, h, sigma = 1, len(_HANDS), 0.10
    dx = torch.zeros(n, h, 3)
    dx[:, :, 0] = 0.125
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    on_preload = _env_with_offsets((0.125, 0.125), delta_x=dx)
    assert float(tcp_position_error_exp(on_preload, sigma=sigma)[0]) == pytest.approx(1.0, abs=1e-6)
    # ...while `servo_shifted=False` is the ablation that measures distance-to-clip.
    assert float(tcp_position_error_exp(on_preload, sigma=sigma, servo_shifted=False)[0]) == pytest.approx(
        math.exp(-(0.125**2) / sigma**2), rel=1e-6
    )


def test_roll_about_the_offset_axis_is_invisible_to_the_position_term():
    """The `2·d·sin(θ/2)` lever is an UPPER BOUND, attained only for rotation axes ⊥ d.

    A pure ROLL about the 0.242 m offset axis moves the TCP by EXACTLY zero, so
    `tcp_position_error_exp` cannot see it — only `tcp_orientation_error_exp` prices it.
    This is why both terms are needed, and it is a caveat for reading T1: if roll error
    dominates, the position term looks healthy while the hand is still mis-oriented.
    Pins the claim so the docstrings can't drift back to stating the bound as an identity.
    """
    from wbt_training.config_values.tcp_precision import tcp_orientation_error_exp, tcp_position_error_exp

    theta, d = 0.2, 0.242
    # ROLL: rotate about +x, which IS the offset direction.
    roll = _Env(
        wrist_pos_w=torch.zeros(1, 2, 3),
        wrist_quat_wxyz=_quat_axis_angle_wxyz((1.0, 0.0, 0.0), theta, (1, 2)),
        ref_pos_w=torch.zeros(1, 2, 3),
        ref_quat_xyzw=_wxyz_to_xyzw(_identity_wxyz((1, 2))),
    )
    assert float(tcp_position_error_exp(roll, tcp_offset_x=d, servo_shifted=False)[0]) == pytest.approx(1.0, abs=1e-6)
    # ...but the orientation term DOES see it, so the pair still covers the error.
    assert float(tcp_orientation_error_exp(roll)[0]) < 0.9

    # Sweep the axis: the implied error must follow 2·d·sin(θ/2)·sin∠(axis, d), so the
    # quoted bound is attained ONLY perpendicular and an oblique axis lands strictly between.
    sigma = 0.10
    for axis, name in (((0.0, 1.0, 0.0), "pitch ⊥"), ((0.0, 0.0, 1.0), "yaw ⊥"), ((1.0, 0.0, 1.0), "45° oblique")):
        env = _Env(
            wrist_pos_w=torch.zeros(1, 2, 3),
            wrist_quat_wxyz=_quat_axis_angle_wxyz(axis, theta, (1, 2)),
            ref_pos_w=torch.zeros(1, 2, 3),
            ref_quat_xyzw=_wxyz_to_xyzw(_identity_wxyz((1, 2))),
        )
        r = tcp_position_error_exp(env, sigma=sigma, tcp_offset_x=d, servo_shifted=False)
        implied = sigma * math.sqrt(-math.log(float(r[0])))
        k = torch.tensor(axis, dtype=torch.float)
        k = k / k.norm()
        sin_angle = math.sqrt(max(0.0, 1.0 - float(k[0]) ** 2))  # sin∠(axis, d), d ∥ +x
        assert implied == pytest.approx(2 * d * math.sin(theta / 2) * sin_angle, rel=1e-4), name
    # The oblique case must be strictly inside (0, bound) — i.e. the bound is not an identity.
    assert 0.0 < 2 * d * math.sin(theta / 2) * math.sin(math.pi / 4) < 2 * d * math.sin(theta / 2)


def test_orientation_amplification_is_18x_not_36x():
    """Pin the de-dilution factor the docstring quotes, by finite difference.

    The per-body gradient of `exp(−mean_h err_h/σ²)` scales as `w/(N·σ²)`:
        old 14-body σ=0.4  → 2.0/(14·0.16)   = 0.893
        new  2-hand σ=0.25 → 2.0/(2·0.0625)  = 16.0     ⇒ 17.92×
    An earlier docstring claimed ~36× (= 14 × 2.56), which would only hold if the new term
    meaned over ONE body; it means over TWO, so the true factor is (14/2) × 2.56 — exactly
    2× smaller. Real but modest overstatement; pinned here so it cannot creep back.
    """
    w_old, n_old, s_old = 2.0, 14, 0.4
    w_new, n_new, s_new = 2.0, 2, 0.25
    eps = 1e-9

    def grad(w, n, s):
        base = math.exp(-0.0 / n / s**2)
        pert = math.exp(-eps / n / s**2)
        return w * (base - pert) / eps

    ratio = grad(w_new, n_new, s_new) / grad(w_old, n_old, s_old)
    assert ratio == pytest.approx(17.92, rel=1e-3)
    # Matches the closed form (N_old/N_new)·(σ_old/σ_new)². rel=1e-6, not tighter: this is a
    # one-sided finite difference, so O(eps) truncation error is expected in the last digits.
    assert ratio == pytest.approx((n_old / n_new) * (s_old / s_new) ** 2, rel=1e-6)
    assert ratio < 20.0, "not ~36x — that figure assumed a 1-body mean"


def test_index_caches_are_keyed_on_body_names():
    """`_ensure_force_{isaac,motion}_indices` must key their cache on `body_names`.

    They used to store a bare tensor and return it BEFORE inspecting `body_names`, so the
    first caller's body list won process-wide and any later caller with a different list got
    the wrong bodies — silently, with the right shape. Benign while every caller passed the
    same two wrists, which is exactly why it needed pinning (review 2026-08-05).
    """
    from wbt_training.config_values.force_tracking import _ensure_force_isaac_indices

    class _S:
        body_ids = list(range(40))

        def find_rigid_body_indice(self, n):
            return {"left_wrist_yaw_link": 24, "right_wrist_yaw_link": 31, "torso_link": 11}[n]

    class _E:
        device = "cpu"
        simulator = _S()

    env = _E()
    wrists = _ensure_force_isaac_indices(env, _HANDS)
    assert wrists.tolist() == [24, 31]
    # A DIFFERENT body list must resolve independently, not inherit the wrists.
    assert _ensure_force_isaac_indices(env, ("torso_link",)).tolist() == [11]
    # ...and the original entry must survive, still cached (same object, no re-resolve).
    assert _ensure_force_isaac_indices(env, _HANDS) is wrists

    # The legacy bare-tensor escape hatch (pre-seeded by test_hybrid*.py) still wins.
    class _E2:
        device = "cpu"
        simulator = _S()

    env2 = _E2()
    env2._wbt_force_isaac_indices = torch.tensor([0, 1], dtype=torch.long)
    assert _ensure_force_isaac_indices(env2, ("torso_link",)).tolist() == [0, 1]


def test_tcp_position_survives_a_missing_delta_x_buffer():
    """No `_wbt_delta_x` (unit-test env, or a preset without the injector) must fall back to
    the pure kinematic target rather than raising."""
    from wbt_training.config_values.tcp_precision import tcp_position_error_exp

    env = _env_with_offsets((0.03, 0.03))
    assert not hasattr(env, "_wbt_delta_x")
    assert float(tcp_position_error_exp(env, sigma=0.10)[0]) == pytest.approx(math.exp(-(0.03**2) / 0.10**2), rel=1e-6)


# ---------------------------------------------------------------------------
# 4. The orientation term: same math as upstream, minus both dilutions.
# ---------------------------------------------------------------------------
def test_tcp_orientation_matches_the_upstream_mirror_on_two_bodies():
    """Numerically identical to the repo's mirror of the upstream 14-body orientation term
    when both are restricted to the same 2 bodies at the same σ — proving the ONLY deltas
    are the mean SET and σ, not a different error metric or quaternion convention.

    Compared against `force_tracking.motion_relative_body_orientation_error_exp_no_wrist`
    (documented as "math identical to upstream", `exclude_body_names=()` ⇒ all tracked
    bodies) rather than upstream itself: upstream's
    `_get_motion_command_and_assert_type` hard-asserts `isinstance(_, MotionCommand)`, so it
    cannot run against a test double. `test_upstream_orientation_math_is_still_the_mirror`
    below keeps that mirror honest by diffing the two source formulas.
    """
    from wbt_training.config_values.force_tracking import motion_relative_body_orientation_error_exp_no_wrist as mirror

    from wbt_training.config_values.tcp_precision import tcp_orientation_error_exp

    env = _env_with_orientation_error(0.2)
    mine = tcp_orientation_error_exp(env, sigma=0.4)
    theirs = mirror(env, sigma=0.4, exclude_body_names=())  # keeps both hands
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_upstream_orientation_math_is_still_the_mirror():
    """The mirror above is only a valid stand-in while upstream's formula matches it.

    Pin the two facts the comparison rests on, straight out of upstream's source: the
    squared `quat_error_magnitude` and the `exp(-mean/σ²)` kernel. If upstream changes its
    error metric, this goes red and the comparison above must be re-derived.
    """
    import inspect

    from holosoma.managers.reward.terms.wbt import motion_relative_body_orientation_error_exp as upstream

    src = inspect.getsource(upstream)
    assert "quat_error_magnitude" in src and "** 2" in src
    assert "torch.exp(-error.mean(-1) / sigma**2)" in src


@pytest.mark.parametrize("theta,expected", [(0.10, 0.85214), (0.20, 0.52729), (0.30, 0.23693), (0.40, 0.07730)])
def test_tcp_orientation_dilution_is_removed(theta, expected):
    """The σ=0.25, 2-body term must produce the plan §1.3 right-hand column.

    Contrast the diluted status quo (14-body mean at σ=0.4): 0.2 rad costs 0.0354 of total
    reward there, vs 1 − 0.527 = 0.47 × w=2.0 ≈ 0.95 here.
    """
    from wbt_training.config_values.tcp_precision import tcp_orientation_error_exp

    env = _env_with_orientation_error(theta)
    assert float(tcp_orientation_error_exp(env, sigma=0.25)[0]) == pytest.approx(expected, abs=2e-5)


def test_tcp_orientation_is_one_when_perfect():
    from wbt_training.config_values.tcp_precision import tcp_orientation_error_exp

    assert float(tcp_orientation_error_exp(_env_with_orientation_error(0.0))[0]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Preset wiring, phase 1 (the default).
# ---------------------------------------------------------------------------
def test_tcp_terms_wired_into_both_local_unified_presets():
    from wbt_training.config_values.experiment import DEFAULTS

    for key in ("g1_wbt_3pt_hybrid_unified", "g1_wbt_3pt_hybrid_unified_anneal"):
        terms = DEFAULTS[key].reward.terms
        pos, orn = terms["tcp_position_error_exp"], terms["tcp_orientation_error_exp"]
        assert pos.func == "wbt_training.config_values.tcp_precision:tcp_position_error_exp"
        assert orn.func == "wbt_training.config_values.tcp_precision:tcp_orientation_error_exp"
        assert (pos.weight, orn.weight) == (4.0, 2.0)
        assert pos.params == {"sigma": 0.10, "tcp_offset_x": 0.242, "servo_shifted": True}
        assert orn.params == {"sigma": 0.25}
        # Non-zero weights matter: RewardManager SKIPS weight==0.0 terms entirely at init
        # (reward/manager.py:61-62), so a zero-weighted term is absent, not merely inert.
        assert pos.weight != 0.0 and orn.weight != 0.0


@_phase1_only
def test_tcp_change_is_purely_additive_on_reward():
    """Every inherited unified term must survive with its weight untouched — the fix adds
    pricing, it does not rebalance (Eq.2 at 3.0 in particular is the main regression risk
    being watched in the A/B, so it must not move here)."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import (
        g1_29dof_wbt_unified_anneal_reward,
        g1_29dof_wbt_unified_reward,
    )

    for key, base in (
        ("g1_wbt_3pt_hybrid_unified", g1_29dof_wbt_unified_reward),
        ("g1_wbt_3pt_hybrid_unified_anneal", g1_29dof_wbt_unified_anneal_reward),
    ):
        got = DEFAULTS[key].reward.terms
        assert set(got) - set(base.terms) == {"tcp_position_error_exp", "tcp_orientation_error_exp"}
        assert not set(base.terms) - set(got), "an inherited reward term was dropped"
        for k, v in base.terms.items():
            assert got[k] is v, f"inherited term {k} was mutated"
        assert got["force_track_pos_eq2"].weight == 3.0
        assert got["motion_relative_body_orientation_error_exp"].weight == 2.0


_phase2_only = pytest.mark.skipif(
    os.environ.get("WBT_HAND_MASS_DR", "0").strip().lower() not in ("1", "true", "yes"),
    reason="phase-2 wiring; run with WBT_HAND_MASS_DR=1",
)


@_phase2_only
def test_phase2_narrows_mass_dr_to_the_wrists_per_preset():
    """Phase 2 must REPLACE `randomize_link_body_names` with the two wrists — and must
    derive the RobotConfig PER PRESET.

    Two distinct hazards:
      * Sharing one RobotConfig across both twins is correct only while
        `g1_wbt_3pt_hybrid_unified_anneal` happens to inherit R2's `robot` object. The moment
        anyone gives the anneal twin its own `robot=`, a shared object would silently clobber
        it. So the two must be DIFFERENT objects with the same content.
      * The field is replaced, not extended — the robot default is 9 pelvis/hip/knee links,
        so leaving it would scatter the DR over the legs and never touch the hand.
    """
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.tcp_precision import HAND_MASS_BODY_NAMES

    u, a = DEFAULTS["g1_wbt_3pt_hybrid_unified"], DEFAULTS["g1_wbt_3pt_hybrid_unified_anneal"]
    for exp in (u, a):
        assert list(exp.robot.randomize_link_body_names) == list(HAND_MASS_BODY_NAMES) == list(_HANDS)
        assert "randomize_mass_startup" in exp.randomization.setup_terms
        inj = exp.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
        assert inj.sync_distal_mass_from_sim is True
    assert u.robot is not a.robot, "the twins must not share one RobotConfig object"

    # And the sibling presets keep the robot default — the narrowing must not leak.
    for key in ("g1_wbt_3pt_hybrid_global_vel_unified", "g1_wbt_3pt_hybrid", "g1_wbt_2pt_hybrid"):
        assert DEFAULTS[key].robot.randomize_link_body_names[0] == "pelvis"
        assert "randomize_mass_startup" not in DEFAULTS[key].randomization.setup_terms


def test_obs_is_untouched_so_pfl96opy_warm_starts():
    """THE checkpoint-compatibility guard. pfl96opy (170D) is resumed into these presets,
    so the actor obs must remain the very same ObservationManagerCfg object at 170D."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid as expected

    for key in ("g1_wbt_3pt_hybrid_unified", "g1_wbt_3pt_hybrid_unified_anneal"):
        obs = DEFAULTS[key].observation
        assert obs is expected
        actor = obs.groups["actor_obs"]
        assert len(actor.terms) == 14
        assert list(actor.terms) == sorted(actor.terms)
        assert actor.history_length == 4
        assert "obs01_vr_3point_pos_tcp" in actor.terms


@_phase1_only
def test_no_mass_dr_by_default():
    """Phase 1 must ship WITHOUT mass DR: it lowers peak accuracy, so co-launching would
    make a flat precision result unattributable (plan §5 intro, §6)."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.tcp_precision import HAND_MASS_DR_ENABLED

    assert HAND_MASS_DR_ENABLED is False, "WBT_HAND_MASS_DR is set in this environment"
    for key in ("g1_wbt_3pt_hybrid_unified", "g1_wbt_3pt_hybrid_unified_anneal"):
        exp = DEFAULTS[key]
        assert "randomize_mass_startup" not in exp.randomization.setup_terms
        assert exp.robot.randomize_link_body_names[0] == "pelvis"  # robot default, untouched
        inj = exp.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
        assert inj.sync_distal_mass_from_sim is False


def test_sibling_presets_unperturbed():
    """Blast radius: the 185D global-vel unified twins and the base 3pt-hybrid must be
    byte-identical (same carrier OBJECTS), since only the local twins were asked for."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid import (
        g1_29dof_wbt_2pt_hybrid_reward,
        g1_29dof_wbt_hybrid_command,
        g1_29dof_wbt_hybrid_termination,
    )
    from wbt_training.config_values.hybrid_unified import (
        g1_29dof_wbt_unified_anneal_command,
        g1_29dof_wbt_unified_anneal_reward,
        g1_29dof_wbt_unified_command,
        g1_29dof_wbt_unified_reward,
        g1_29dof_wbt_unified_termination,
    )

    gp, ga = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified"], DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified_anneal"]
    assert gp.reward is g1_29dof_wbt_unified_reward
    assert gp.command is g1_29dof_wbt_unified_command
    assert gp.termination is g1_29dof_wbt_unified_termination
    assert ga.reward is g1_29dof_wbt_unified_anneal_reward
    assert ga.command is g1_29dof_wbt_unified_anneal_command
    for e in (gp, ga):
        assert "tcp_position_error_exp" not in e.reward.terms
        assert "randomize_mass_startup" not in e.randomization.setup_terms

    base = DEFAULTS["g1_wbt_3pt_hybrid"]
    assert base.reward is g1_29dof_wbt_2pt_hybrid_reward
    assert base.command is g1_29dof_wbt_hybrid_command
    assert base.termination is g1_29dof_wbt_hybrid_termination
    assert "tcp_position_error_exp" not in base.reward.terms


@_phase1_only
def test_local_twins_no_longer_share_reward_with_global_vel():
    """Documents the ONE intentional break from the previous verbatim-sharing invariant
    (test_hybrid_3pt_unified.py::test_local_unified_shares_carriers_with_global_vel_unified):
    reward now differs by design, while command/curriculum/termination still share."""
    from wbt_training.config_values.experiment import DEFAULTS

    lp, gp = DEFAULTS["g1_wbt_3pt_hybrid_unified"], DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified"]
    assert lp.reward is not gp.reward
    assert lp.command is gp.command
    assert lp.curriculum is gp.curriculum
    assert lp.termination is gp.termination


# ---------------------------------------------------------------------------
# 6. Distal-mass sync (phase 2 physics), and the scalar path's bit-identity.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s", [0.8, 1.0, 1.2, 1.5])
def test_distal_mass_sync_formula(s):
    """m_distal(s) = s·(ft+hand) + fingers. The fingers are separate PRISMATIC bodies, so
    scaling wrist_yaw must NOT scale them (plan §5.5)."""
    from wbt_training.config_values.tcp_precision import (
        GRIPPER_FINGERS_KG,
        GRIPPER_SCALED_PART_KG,
        MERGED_WRIST_DEFAULT_KG,
        distal_mass_from_body_mass,
    )

    got = distal_mass_from_body_mass(torch.tensor([s * MERGED_WRIST_DEFAULT_KG], dtype=torch.float64))
    assert float(got[0]) == pytest.approx(s * GRIPPER_SCALED_PART_KG + GRIPPER_FINGERS_KG, abs=1e-9)


def test_distal_mass_at_nominal_matches_the_hardcoded_scalar():
    """s=1 must land on the 1.5 kg the cfg has always used, or turning the sync on would
    shift F_meas even with DR off."""
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.tcp_precision import MERGED_WRIST_DEFAULT_KG, distal_mass_from_body_mass

    got = float(distal_mass_from_body_mass(torch.tensor([MERGED_WRIST_DEFAULT_KG], dtype=torch.float64))[0])
    assert got == pytest.approx(HybridContactConfig().distal_mass_kg, abs=1e-4)


def test_desync_error_exceeds_the_contact_threshold():
    """Quantifies WHY the sync is mandatory: at scale 1.5 and a=3 m/s² the m·a error is
    ~2.04 N, i.e. above contact_phase_theta_on=2.0 N and 4× sensor_noise_std=0.5 N — it
    would corrupt the hysteresis label, force_error_direct, and Eq.2's force target."""
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.tcp_precision import MERGED_WRIST_DEFAULT_KG, distal_mass_from_body_mass

    cfg = HybridContactConfig()
    true_m = float(distal_mass_from_body_mass(torch.tensor([1.5 * MERGED_WRIST_DEFAULT_KG], dtype=torch.float64))[0])
    err_n = abs(true_m - cfg.distal_mass_kg) * 3.0
    assert err_n == pytest.approx(2.04, abs=0.02)
    assert err_n > cfg.contact_phase_theta_on
    assert err_n > 4.0 * cfg.sensor_noise_std


def test_scalar_distal_mass_path_is_bit_identical():
    """`distal_mass=None` must reproduce the pre-change arithmetic EXACTLY — this is what
    lets the tensorization land without touching any existing preset's behavior (and keeps
    test_hybrid.py's scalar assertions valid)."""
    from wbt_training.config_values.hybrid import HybridContactConfig, simulated_measured_force

    cfg = HybridContactConfig(sensor_noise_std=0.0, sensor_bias_std=0.0)
    f_ext = torch.tensor([[1.0, -2.0, 0.5]])
    acc = torch.tensor([[2.0, 3.0, -1.0]])
    zero = torch.zeros(1, 3)
    legacy = f_ext + cfg.distal_mass_kg * acc
    assert torch.equal(simulated_measured_force(f_ext, acc, zero, cfg, zero), legacy)
    # An explicit nominal tensor must agree with the scalar path too.
    nominal = torch.full((1, 1), cfg.distal_mass_kg)
    assert torch.equal(simulated_measured_force(f_ext, acc, zero, cfg, zero, distal_mass=nominal), legacy)


def test_f_meas_uses_per_env_per_hand_distal_mass():
    """Two envs with different hand masses must yield different m·a in F_meas — i.e. the
    buffer really is per-(env, hand) and broadcasts correctly against (n,h,3)."""
    from wbt_training.config_values.hybrid import HybridContactConfig, simulated_measured_force

    cfg = HybridContactConfig(sensor_noise_std=0.0, sensor_bias_std=0.0)
    f_ext = torch.zeros(2, 2, 3)
    acc = torch.zeros(2, 2, 3)
    acc[..., 0] = 1.0  # 1 m/s² on +x everywhere
    m = torch.tensor([[[1.2], [1.5]], [[2.0], [1.5]]])  # (n=2, h=2, 1)
    out = simulated_measured_force(f_ext, acc, torch.zeros(2, 2, 3), cfg, torch.zeros(2, 2, 3), distal_mass=m)
    assert out[0, 0, 0] == pytest.approx(1.2)
    assert out[0, 1, 0] == pytest.approx(1.5)
    assert out[1, 0, 0] == pytest.approx(2.0)


def test_distal_mass_readback_uses_the_physx_index_space():
    """`_sync_distal_mass` must index `get_masses()` in PHYSX body order.

    THE SILENT-FAILURE GUARD. Two index spaces coexist: CONFIG order
    (`robot_config.body_names`, 32 entries, IsaacGym/holosoma order) and IsaacSim's PHYSX
    DFS order, related by `sim.body_ids[cfg_i] = physx_i`. That map is NOT the identity
    (see the worked example in isaacsim.py `find_rigid_body_indice`'s docstring), so
    indexing masses with a CONFIG index reads a DIFFERENT BODY'S MASS — with no error, no
    shape mismatch, and no log line. It would just quietly desync F_meas.

    Here the wrists sit at CONFIG 24/31 but PHYSX 3/7, and `get_masses()` is filled so that
    only the PHYSX slots hold the intended values. Reading CONFIG slots yields the decoy
    masses and the assertion fails.
    """
    from wbt_training.config_values.hybrid import HybridContactConfig, HybridContactInjector
    from wbt_training.config_values.tcp_precision import MERGED_WRIST_DEFAULT_KG, distal_mass_from_body_mass

    n_bodies, cfg_idx, physx_idx = 32, (24, 31), (3, 7)
    masses = torch.full((2, n_bodies), 99.0)  # decoys everywhere
    scales = (0.8, 1.5)
    for slot, s in zip(physx_idx, scales):
        masses[:, slot] = s * MERGED_WRIST_DEFAULT_KG

    class _View:
        def get_masses(self):
            return masses.clone()  # CPU, as IsaacLab returns

    class _Rbt:
        root_physx_view = _View()

    class _Simu:
        _robot = _Rbt()

    class _E:
        device = "cpu"
        num_envs = 2
        simulator = _Simu()

    env = _E()
    inj = object.__new__(HybridContactInjector)  # bypass CommandTermBase.__init__
    inj.env = env
    inj.hc_cfg = HybridContactConfig(sync_distal_mass_from_sim=True)
    inj.body_names = list(_HANDS)
    inj._isaac_body_ids = torch.tensor(physx_idx, dtype=torch.long)
    env._wbt_distal_mass = torch.full((2, 2, 1), inj.hc_cfg.distal_mass_kg)

    inj._sync_distal_mass()

    got = env._wbt_distal_mass
    assert got.shape == (2, 2, 1), "must stay (n, h, 1) to broadcast against (n, h, 3) accel"
    for hand, s in enumerate(scales):
        want = float(distal_mass_from_body_mass(torch.tensor([s * MERGED_WRIST_DEFAULT_KG]))[0])
        assert float(got[0, hand, 0]) == pytest.approx(want, rel=1e-6)
    # And prove the decoy would have been caught: a CONFIG-order read gives 99 kg.
    assert masses[0, cfg_idx[0]] == 99.0
    assert float(got.max()) < 5.0, "read a decoy body — index space is wrong"


def test_distal_mass_sync_raises_when_the_physx_view_is_missing():
    """With DR explicitly ON, an unreadable mass view must RAISE, not warn-and-continue.

    Silently keeping the nominal m_distal under active mass DR reintroduces the ≤2.04 N
    F_meas desync the sync exists to prevent — above contact_phase_theta_on=2.0 N, so it
    corrupts the CONTACT LABEL rather than just adding noise. In an 8-GPU run a single
    logger.warning scrolls past unseen and the whole run is quietly invalid, so this has to
    be loud. `sync_distal_mass_from_sim` is never set by accident.
    """
    from wbt_training.config_values.hybrid import HybridContactConfig, HybridContactInjector

    class _E:
        device = "cpu"
        num_envs = 1
        simulator = type("_S", (), {})()

    env = _E()
    inj = object.__new__(HybridContactInjector)
    inj.env = env
    inj.hc_cfg = HybridContactConfig(sync_distal_mass_from_sim=True)
    inj.body_names = list(_HANDS)
    inj._isaac_body_ids = torch.tensor([3, 7], dtype=torch.long)
    env._wbt_distal_mass = torch.full((1, 2, 1), inj.hc_cfg.distal_mass_kg)

    with pytest.raises(RuntimeError, match="sync_distal_mass_from_sim"):
        inj._sync_distal_mass()


def test_hybrid_declares_its_loguru_dependency():
    """`hybrid.py` imports loguru at module scope and is on EVERY preset's import path via
    experiment.py, so it must be a declared dependency rather than a transitive accident."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert any(d.split("[")[0].split(">")[0].split("=")[0].strip() == "loguru" for d in deps), deps


def test_gripper_mass_constants_match_the_urdf():
    """The three mirrored constants must equal the URDF, or the sync silently mis-scales.

    Also pins `hybrid.HybridContactConfig.distal_mass_kg` against the URDF total, and the
    94.1%-gripper claim the "wrist_yaw IS the gripper" argument rests on (plan §5.1).
    """
    if not _YAM_URDF.is_file():
        pytest.skip("holosoma g1 assets not checked out (run scripts/bootstrap-holosoma.sh)")
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.tcp_precision import (
        GRIPPER_FINGERS_KG,
        GRIPPER_SCALED_PART_KG,
        MERGED_WRIST_DEFAULT_KG,
    )

    root = ET.fromstring(_YAM_URDF.read_text())
    mass = {
        link.get("name"): float(link.find("inertial").find("mass").get("value"))
        for link in root.findall("link")
        if link.find("inertial") is not None
    }
    merged = mass["left_wrist_yaw_link"] + mass["left_ft_sensor_link"] + mass["left_rubber_hand"]
    scaled = mass["left_ft_sensor_link"] + mass["left_rubber_hand"]
    fingers = mass["left_gripper_finger1_link"] + mass["left_gripper_finger2_link"]

    assert MERGED_WRIST_DEFAULT_KG == pytest.approx(merged, abs=1e-8)
    assert GRIPPER_SCALED_PART_KG == pytest.approx(scaled, abs=1e-8)
    assert GRIPPER_FINGERS_KG == pytest.approx(fingers, abs=1e-8)
    # The nominal m_distal is the FULL gripper (ft + hand + both fingers).
    assert HybridContactConfig().distal_mass_kg == pytest.approx(scaled + fingers, abs=1e-3)
    # "wrist_yaw IS the gripper": the gripper is >90% of the merged body's mass.
    assert scaled / merged > 0.90

    # The joint TYPES are what decide which links merge: two fixed (absorbed into
    # wrist_yaw) + two prismatic (separate bodies, hence excluded from the scaling).
    jtype = {j.get("name"): j.get("type") for j in root.findall("joint")}
    assert jtype["left_ft_sensor_joint"] == "fixed"
    assert jtype["left_hand_palm_joint"] == "fixed"
    assert jtype["left_gripper_finger1_joint"] == "prismatic"
    assert jtype["left_gripper_finger2_joint"] == "prismatic"


def test_hand_mass_dr_targets_only_the_two_wrist_bodies():
    """Guards against someone reinstating `experiment._ee_mass_body_names`, which spreads
    the DR over pelvis/hip/knee/ankle/wrist_roll/pitch: at scale [0.7,1.4] that is 1.010–
    2.020 kg of swing on wrist_yaw but only 0.399 kg across roll+pitch, plus leg DR that is
    pure variance for this objective (plan §5.3)."""
    from wbt_training.config_values.tcp_precision import (
        HAND_MASS_BODY_NAMES,
        HAND_MASS_SCALE_RANGE,
        g1_29dof_wbt_randomization_hand_mass as dr,
    )

    assert tuple(HAND_MASS_BODY_NAMES) == _HANDS
    term = dr.setup_terms["randomize_mass_startup"]
    assert term.func == "holosoma.managers.randomization.terms.locomotion:randomize_mass_startup"
    assert term.params["enable_link_mass"] is True
    assert term.params["enable_base_mass"] is False, "the torso must stay nominal"
    assert term.params["link_mass_range"] == list(HAND_MASS_SCALE_RANGE) == [0.8, 1.5]
    # Every base randomization term (pushes / friction / COM / dof bias) is preserved.
    from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization as base

    for k, v in base.setup_terms.items():
        assert dr.setup_terms[k] is v
    assert dr.reset_terms == base.reset_terms and dr.step_terms == base.step_terms


def test_ee_mass_randomization_alias_is_a_shell():
    """Records WHY we build our own cfg: pinned holosoma exports no `_ee_mass` variant, so
    experiment.py's try/except is in its fallback branch and `g1_wbt_maskedmimic_ee_mass`
    currently has NO mass DR at all. If upstream ever adds the real one, this goes red and
    §5.4's cleanup can proceed."""
    import holosoma.config_values.wbt.g1.randomization as up

    assert not hasattr(up, "g1_29dof_wbt_randomization_ee_mass")
    from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization as base

    from wbt_training.config_values.experiment import g1_29dof_wbt_randomization_ee_mass as alias

    assert alias is base
    assert "randomize_mass_startup" not in alias.setup_terms


# ---------------------------------------------------------------------------
# 7. Phase 2 assembly (built unconditionally; selected by WBT_HAND_MASS_DR).
# ---------------------------------------------------------------------------
def test_phase2_overrides_sync_every_cfg_carrier():
    """DR and the m_distal sync must move TOGETHER, and EVERY carrier that holds a
    HybridContactConfig (4 reward terms + termination + command + the R1 servo curriculum)
    must hold the SAME synced object — a split would leave the injector syncing while a
    reward term reads the stale cfg."""
    from wbt_training.config_values import tcp_precision as tp

    for anneal in (False, True):
        ov = _overrides_with_dr(anneal=anneal)
        assert set(ov) >= {"reward", "command", "termination", "randomization"}
        assert ov["randomization"] is tp.g1_29dof_wbt_randomization_hand_mass

        inj = ov["command"].setup_terms["hybrid_contact_injector"].params["hybrid_config"]
        assert inj.sync_distal_mass_from_sim is True
        assert inj.servo_observe_only is (not anneal)  # R2 observe-only, R1 full servo
        assert inj.servo_enabled is True
        assert ov["termination"].terms["force_runaway"].params["cfg"] is inj

        carriers = [v.params["cfg"] for v in ov["reward"].terms.values() if "cfg" in (v.params or {})]
        hybrid_carriers = [c for c in carriers if hasattr(c, "sync_distal_mass_from_sim")]
        assert len(hybrid_carriers) == 4  # direct / init_bonus / approach / rate_penalty
        assert all(c is inj for c in hybrid_carriers)
        # TCP terms come along unchanged.
        assert {"tcp_position_error_exp", "tcp_orientation_error_exp"} <= set(ov["reward"].terms)
        if anneal:
            sa = ov["curriculum"].setup_terms["servo_authority"].params
            assert sa["hybrid_config"] is inj
            assert sa["force_err_threshold_n"] == 2.0 and sa["alpha_step"] == 0.02


def _overrides_with_dr(*, anneal: bool):
    """`unified_tcp_preset_overrides` reads a module-level flag set at import time, so
    exercise the phase-2 branch by patching the flag rather than by re-importing."""
    from wbt_training.config_values import tcp_precision as tp

    saved = tp.HAND_MASS_DR_ENABLED
    tp.HAND_MASS_DR_ENABLED = True
    try:
        return tp.unified_tcp_preset_overrides(anneal=anneal)
    finally:
        tp.HAND_MASS_DR_ENABLED = saved


@_phase1_only
def test_phase1_overrides_touch_reward_only():
    """Phase 1 must change NOTHING but reward — so obs/physics/DR are identical to
    pfl96opy's and the A/B isolates the reward change."""
    from wbt_training.config_values import tcp_precision as tp

    for anneal in (False, True):
        ov = tp.unified_tcp_preset_overrides(anneal=anneal)
        assert set(ov) == {"reward"}


def test_phase2_keeps_the_obs_at_170d():
    """DR changes physics, not the obs layout — so a phase-1 checkpoint warm-starts into
    phase 2 as well. Nothing in the phase-2 override touches `observation`."""
    from wbt_training.config_values import tcp_precision as tp

    for anneal in (False, True):
        assert "observation" not in _overrides_with_dr(anneal=anneal)
    assert not hasattr(tp, "g1_29dof_wbt_observation_3pt_unified_hybrid_massdr")
