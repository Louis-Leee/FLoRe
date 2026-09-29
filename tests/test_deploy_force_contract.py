"""The force half of the hardware deploy contract, across the FUJI/holosoma seam.

Four things must agree between two repos with two different venvs, so none of them can be
checked by importing both sides:

1. **The frame.** FUJI's measured ``TCP_FROM_SENSOR`` and holosoma's ``R_SENSOR`` are the
   SAME rotation, and both equal the URDF ``{side}_ft_sensor_joint rpy=(0, -pi/2, pi)``.
   That is what makes the wire's sensor-frame ``F^cmd`` land where the planner drew it: the
   planner does not rotate the command at all, the low level rotates it with ``R_SENSOR``,
   and the arrows are drawn with ``TCP_FROM_SENSOR``. If those two constants ever diverge,
   every arrow would point one way and every commanded force push the other — silently,
   because each side is internally consistent.
1b. **The sign.** The models predict ``F_meas``; the low level's servo error is
   ``F_cmd + F_meas``, driven to zero. So the planner must publish ``-F_meas``, and it must be
   the ONLY side that negates. Both halves are pinned below, against holosoma's own
   ``servo_step`` — the same "each side is internally consistent" failure mode as the frame,
   which is why the derivation is asserted rather than the number alone.
2. **The gate.** ONE number (1.5 N) has to appear as the planner's default gate, the low
   level's wire deadband, and (wire on) the F_meas-on-F^cmd threshold. Below it the whole
   stack is position-only; a copy left at the old value would leave the gate half-applied.
   The sign does not interact with it: ``‖F^cmd‖ == ‖F_meas‖``.
3. **The opt-in.** Force reaching the robot requires BOTH the viser switch AND
   ``FORCE_CMD_WIRE=1``. The default launch must be pure position.

The holosoma side is read as SOURCE (``ast`` / text), the same technique
``test_force_frame.py`` uses for the UMI copy of the relabel and for the same reason: the
submodule's imports are not installed in FUJI's venv.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from fuji.utils.force_frame import CMD_FROM_MEAS_SIGN, TCP_FROM_SENSOR, world_force_from_sensor

_REPO = Path(__file__).resolve().parents[1]
_HOLOSOMA = _REPO / "third_party" / "holosoma_extensions"
_FORCE_MATH = _HOLOSOMA / "src/extensions/wbt_wrappers/wbt_wrappers_inference/utils/force_deploy_math.py"
_VIVE_LAUNCHER = _HOLOSOMA / "force_deploy/3_run_policy_vive_force.sh"
_CLIP_LAUNCHER = _HOLOSOMA / "force_deploy/3_run_policy_clip_force.sh"
_POLICY = _HOLOSOMA / "src/extensions/wbt_wrappers/wbt_wrappers_inference/policies/holosoma_wbt.py"
_PLANNER = _REPO / "scripts" / "hl_serving.py"

#: The single gate value the whole stack must agree on (N).
GATE_N = 1.5
#: The hard F/T stop (N) — above this the low level runs its own joystick-'B' handler.
#: Raised 20 → 40 on 2026-08-15: 20 was under BOTH the 25 N keyboard ``F^cmd`` clamp and the
#: 18.5 N hardest press in the 283-episode 0814 demo set, so a normal wipe tripped it (at
#: 25.3 N on the rig). See ``docs/RUNBOOK_hl_serving.md`` § the three force thresholds.
ESTOP_N = 40.0

_needs_holosoma = pytest.mark.skipif(
    not _FORCE_MATH.is_file(), reason="holosoma_extensions submodule not checked out"
)
_needs_launchers = pytest.mark.skipif(
    not (_VIVE_LAUNCHER.is_file() and _CLIP_LAUNCHER.is_file()),
    reason="the robot-deploy launchers (force_deploy/) are not part of this checkout",
)


def _np_array_literal(path: Path, name: str) -> np.ndarray:
    """Read ``name = np.array([...])`` out of a source file without importing it."""
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Assign) and not isinstance(node, ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        assert isinstance(value, ast.Call), f"{name} in {path.name} is not an np.array(...) call"
        return np.asarray(ast.literal_eval(value.args[0]), dtype=np.float64)
    raise AssertionError(f"{path.name} has no module-level assignment to {name}")


def _shell_default(path: Path, name: str) -> str:
    """Extract ``NAME="${NAME:-default}"`` from a shell script."""
    match = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', path.read_text(), re.MULTILINE)
    assert match is not None, f"{path.name} has no `{name}=${{{name}:-...}}` default"
    return match.group(1)


# ---------------------------------------------------------------------------
# 1. The frame
# ---------------------------------------------------------------------------
@_needs_holosoma
def test_fuji_relabel_is_the_same_rotation_as_the_low_levels_r_sensor():
    """``TCP_FROM_SENSOR`` (measured on the rig) == ``R_SENSOR`` (read off the URDF).

    Two independent derivations of one physical mounting, and the reason the display fix
    was display-ONLY: the low level was already rotating both F_meas and F^cmd correctly,
    so only FUJI's arrows were wrong. A future edit to either constant that does not touch
    the other is a frame split between what the operator SEES and what the robot DOES.
    """
    np.testing.assert_array_equal(_np_array_literal(_FORCE_MATH, "R_SENSOR"), TCP_FROM_SENSOR)


@_needs_holosoma
def test_that_rotation_is_the_urdf_ft_sensor_joint_rpy():
    """Both constants equal ``rpy=(0, -pi/2, pi)`` — the actual joint in the yam URDF.

    This is the third, independent witness: the number is not a convention either repo
    chose, it is where the Bota is bolted.
    """
    r, p, y = 0.0, -np.pi / 2.0, np.pi
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    np.testing.assert_allclose(rz @ ry @ rx, TCP_FROM_SENSOR, atol=1e-12)


def test_the_planner_publishes_the_command_unrotated():
    """``_gate_wire_force`` must hand the wire the RAW sensor triple.

    The low level rotates sensor→pelvis with its own live wrist FK, which the planner
    cannot do (it has no joint readback) — that is WHY the command travels sensor-frame.
    A planner-side rotation would double-apply the mounting and, being a 180° involution,
    do so invisibly.
    """
    src = _PLANNER.read_text()
    body = src[src.index("def _gate_wire_force"):]
    body = body[: body.index("\n    def ")]
    assert "wrench_row[0:3]" in body, "the gate no longer takes the raw sensor force triple"
    for forbidden in ("TCP_FROM_SENSOR", "tcp_from_sensor_force", "tcp_from_sensor_wrench"):
        assert forbidden not in body, f"_gate_wire_force applies {forbidden}; the low level already rotates"


# ---------------------------------------------------------------------------
# 1b. The SIGN: the model predicts F_meas, the wire must carry -F_meas
# ---------------------------------------------------------------------------
@_needs_holosoma
def test_the_low_levels_force_loop_makes_the_command_oppose_the_measurement():
    """``servo_step``'s error is ``F_cmd + F_meas``, driven to zero — the ONLY reason FUJI
    negates.

    This is the load-bearing fact behind :data:`fuji.utils.force_frame.CMD_FROM_MEAS_SIGN`, and
    it lives in the OTHER repo, so it is read as source. If holosoma ever changes to
    ``F_cmd - F_meas`` (the other common convention), the planner's flip becomes the bug it
    currently prevents — and nothing else in either repo would notice, because each side stays
    internally consistent.
    """
    body = _FORCE_MATH.read_text()
    body = body[body.index("def servo_step"):]
    assert re.search(r"^\s*e = f_cmd \+ f_meas", body, re.MULTILINE), (
        "servo_step's force error is no longer `f_cmd + f_meas`; re-derive CMD_FROM_MEAS_SIGN"
    )
    assert CMD_FROM_MEAS_SIGN == -1.0, "the planner no longer negates; the wire would fight the servo"


def test_the_planner_negates_the_prediction_before_publishing_it():
    """``_gate_wire_force`` must apply ``cmd_from_meas_force`` — once, and via the shared helper.

    The model's output is a MEASUREMENT (``F_meas`` from the demonstrations). Publishing it
    verbatim asks the arm to pull away from the surface the demo pressed on, and leaves the
    servo error at ``2·F_meas``, so the integrator winds up rather than settling. An inline
    ``-force3`` would work but would be a second spelling of the sign — the point of routing it
    through :mod:`fuji.utils.force_frame` is that the wire and the arrows cannot disagree.
    """
    src = _PLANNER.read_text()
    body = src[src.index("def _gate_wire_force"):]
    body = body[: body.index("\n    def ")]
    assert body.count("cmd_from_meas_force(") == 1, (
        "_gate_wire_force must apply the meas→cmd sign exactly once, through the shared helper"
    )
    assert "from fuji.utils.force_frame import" in src and "cmd_from_meas_force" in src


@_needs_holosoma
def test_the_drawn_measured_force_is_the_vector_the_servo_consumes():
    """FUJI's ``world_force_from_sensor(f, R_rel)`` == holosoma's
    ``sensor_force_to_pelvis(f, R_rel, +1)``, vector for vector.

    This is what makes the planner's CYAN ``F_meas`` arrow trustworthy. It is drawn at the live
    robot's FK'd TCP by composing ``R_world_tcp @ TCP_FROM_SENSOR @ f_sensor``, where
    ``R_world_tcp`` is the URDF wrist rotation in the pelvis frame — and that is *definitionally*
    holosoma's ``R_rel`` (``oMf[wrist].rotation`` from identity-root FK on the measured joints).
    Since the planner's viser world is the pelvis frame with identity orientation, the two
    expressions must be the same arrow. If they diverge, the operator would be judging "is the
    force loop tracking?" against a vector the controller never saw.

    It is also why the page can offer the two ``F_meas`` sources (RecordTap pelvis vs raw
    ``:8771`` sensor) as a live cross-check of the F/T chain: they are the same composition
    evaluated at the same ``R_rel``, so any difference is tare / RNEA compensation / the
    deploy-side EMA, never geometry.

    ``sensor_force_to_pelvis`` is re-expressed here from ``R_SENSOR`` read out of holosoma's own
    source (no import across the venv split), so the equality is checked against that repo's
    constant rather than against a copy of it.
    """
    r_sensor = _np_array_literal(_FORCE_MATH, "R_SENSOR")
    rng = np.random.default_rng(11)
    for _ in range(8):
        f_sensor = rng.normal(size=3)
        r_rel = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(r_rel) < 0:
            r_rel[:, 0] *= -1.0                      # keep it a proper rotation
        # holosoma: f_pelvis = sign * R_rel @ (R_SENSOR @ f_sensor), with sign = +1 on this rig.
        low_level = r_rel @ (r_sensor @ f_sensor)
        np.testing.assert_allclose(world_force_from_sensor(f_sensor, r_rel), low_level, atol=1e-12)


def test_the_sign_is_applied_planner_side_only_never_twice():
    """The low level must NOT also negate: it applies the same ``_force_meas_sign`` to the wire's
    F^cmd as to its own F_meas, so a second flip there would cancel this one.

    Pinned as the DERIVATION (``sensor_force_to_pelvis(..., self._force_meas_sign)``) rather than
    a literal, because ``HOLOSOMA_FORCE_MEAS_SIGN`` is a gantry-resolved hardware unknown — it is
    allowed to change; what is not allowed is the command and the measurement taking different
    signs into the servo.
    """
    body = _POLICY.read_text()
    body = body[body.index("def _poll_force_cmd_wire"):]
    body = body[: body.index("\n    def ")]
    assert "sensor_force_to_pelvis(" in body and "self._force_meas_sign" in body, (
        "the wire's F^cmd no longer shares the F_meas sign/rotation chain"
    )
    assert "-f_sensor" not in body and "* -1" not in body, "the low level negates F^cmd too — double flip"


# ---------------------------------------------------------------------------
# 2. The gate: one number, three places
# ---------------------------------------------------------------------------
def test_planner_gate_defaults_to_the_contract_value():
    src = _PLANNER.read_text()
    match = re.search(r'"--force-gate-n",\s*type=float,\s*default=([0-9.]+)', src)
    assert match is not None, "hl_serving no longer declares --force-gate-n with a float default"
    assert float(match.group(1)) == GATE_N


@_needs_holosoma
@_needs_launchers
def test_low_level_wire_deadband_mirrors_the_planner_gate():
    """Defense in depth: the low level hard-zeros a sub-gate command BEFORE it reaches the
    buffer, so a planner bug cannot push force through below the gate."""
    assert float(_shell_default(_VIVE_LAUNCHER, "FORCE_CMD_WIRE_GATE_N")) == GATE_N


@_needs_holosoma
@_needs_launchers
def test_the_meas_gate_is_derived_from_the_wire_deadband_not_retyped():
    """With the wire on, GATE_CMD_N must come FROM ``$FORCE_CMD_WIRE_GATE_N``.

    This is the half of "no force" that the operator cannot see: below the gate the
    measured force must also stay out of the actor obs and the servo, so the robot is
    position-only however hard the Bota is pressed. Asserting the DERIVATION rather than
    the literal is the point — a second hand-typed 1.5 is how the two drift apart.
    """
    text = _VIVE_LAUNCHER.read_text()
    assert 'GATE_CMD_N="${GATE_CMD_N:-$FORCE_CMD_WIRE_GATE_N}"' in text, (
        "GATE_CMD_N is no longer derived from FORCE_CMD_WIRE_GATE_N when the wire drives F^cmd"
    )
    # ...and the keyboard path keeps its own, deliberately higher default.
    assert 'GATE_CMD_N="${GATE_CMD_N:-6}"' in text


# ---------------------------------------------------------------------------
# 3. The opt-in, and the hard stop
# ---------------------------------------------------------------------------
@_needs_holosoma
@_needs_launchers
def test_the_default_launch_is_pure_position():
    """``FORCE_CMD_WIRE`` off by default: whatever the planner publishes, the low level
    ignores the wire's f_cmd entirely, so only translation+rotation reach the robot."""
    assert _shell_default(_VIVE_LAUNCHER, "FORCE_CMD_WIRE") == "0"


def test_the_planner_force_switch_is_off_by_default():
    """The other half of the two-key rule: ``--enable-force-cmd`` is opt-in (store_true)."""
    src = _PLANNER.read_text()
    match = re.search(r'"--enable-force-cmd",\s*action="(\w+)"', src)
    assert match is not None and match.group(1) == "store_true"


@_needs_holosoma
@pytest.mark.parametrize("launcher", ["vive", "clip"])
@_needs_launchers
def test_the_hard_ft_stop_is_armed_at_40n_in_both_launchers(launcher):
    """A limit that protects the robot must not depend on which script started it."""
    path = _VIVE_LAUNCHER if launcher == "vive" else _CLIP_LAUNCHER
    assert float(_shell_default(path, "FORCE_ESTOP_N")) == ESTOP_N


@_needs_holosoma
@pytest.mark.parametrize("launcher", ["vive", "clip"])
@_needs_launchers
def test_the_hard_stop_fires_below_the_runaway_valve(launcher):
    """The stop must come first, or the runaway valve pre-empts it.

    ``HOLOSOMA_FORCE_RUNAWAY_N`` only warns and freezes the optional servo, so it must never
    sit below the stop — otherwise the tick between the two thresholds gets a behaviour the
    rig has never exercised (in a unified observe-only preset a runaway trip HOLDS the servo
    integrator, and I/Δx are that policy's only force observation). It used to be a literal
    35 against a 20 N stop, which held the ordering by luck; raising the stop to 40 would
    have inverted it. It is DERIVED from the stop now, so the ordering holds for every
    threshold an operator can type, and that is what this pins.
    """
    src = (_VIVE_LAUNCHER if launcher == "vive" else _CLIP_LAUNCHER).read_text()
    assert 'HOLOSOMA_FORCE_RUNAWAY_N="$FORCE_RUNAWAY_N"' in src, "exported as a literal again"
    derived = re.search(r"FORCE_RUNAWAY_N=.*e > 0 \? e \+ ([0-9.]+) : ([0-9.]+)", src)
    assert derived is not None, "the runaway valve is no longer derived from FORCE_ESTOP_N"
    offset, stop_off_fallback = float(derived.group(1)), float(derived.group(2))
    assert offset > 0.0, "the runaway valve would sit at or below the hard stop"
    # With the stop OFF the valve is the only valve left, so it keeps an absolute value —
    # "0 + offset" would put it inside the task's own force range and fire on every press.
    assert stop_off_fallback > offset


@_needs_holosoma
def test_the_hard_stop_runs_the_joystick_b_handler():
    """It must reuse ``_handle_stop_policy``, not a bespoke shutdown.

    Two payoffs: one stopped state to reason about, and the FUJI planner detects it for
    free — its LOW-LEVEL-LOST watchdog spots joystick B by the RecordTap stream stopping,
    which is exactly what this produces, so no new wire field is needed.
    """
    src = _POLICY.read_text()
    body = src[src.index("def _trip_force_estop"):]
    body = body[: body.index("\n    def ")]
    assert "self._handle_stop_policy()" in body
    # And the planner still has the watchdog that observes it.
    assert "LOW LEVEL LOST" in _PLANNER.read_text()


@_needs_holosoma
def test_the_hard_stop_is_checked_before_any_force_command_is_folded():
    """Order matters within the tick: the check must precede ``_poll_force_cmd_wire``, so a
    tripping tick cannot also slew a fresh force command into the buffer on its way out."""
    src = _POLICY.read_text()
    body = src[src.index("def rl_inference"):]
    assert body.index("_check_force_estop") < body.index("_poll_force_cmd_wire")
    assert body.index("_check_force_estop") < body.index("_build_observation")


# ---------------------------------------------------------------------------
# 4. LEFT-only: what the RIGHT hand is allowed to receive
# ---------------------------------------------------------------------------
def test_the_planner_publishes_a_left_only_force_command():
    """``_gate_wire_force`` returns a **(3,)** — LEFT force only, never a 6-wide [L, R].

    The low level splits on width (`f[0:3]` alone when 3), so a 3-wide command leaves the
    right hand's F^cmd at zero structurally. If this ever widened to 6, the right hand would
    start receiving whatever sat in the reserved columns of a `left_only` checkpoint —
    columns the model never learned to fill.
    """
    src = _PLANNER.read_text()
    body = src[src.index("def _gate_wire_force"):]
    body = body[: body.index("\n    def ")]
    assert "wrench_row[0:3]" in body and "(3,)" in body
    assert "0:6" not in body and "[3:6]" not in body, "the wire force is no longer LEFT-only"


def test_the_right_hand_pose_is_held_never_predicted():
    """`full_wire` writes the model's LEFT prediction into slot 0 and HOLDS slots 1/2.

    The hold pose is re-synced to the robot's MEASURED right/head at each Predict, so the
    un-predicted arm commands where it already is rather than flying to a stale home target.
    That is the "no action" half of LEFT-only, and it is a property of this one function.
    """
    src = _PLANNER.read_text()
    body = src[src.index("def full_wire"):]
    body = body[: body.index("\n\ndef ")]
    assert "self.hold_pos_wire[1]" in body and "self.hold_pos_wire[2]" in body
    assert "self.hold_quat_xyzw[1]" in body and "self.hold_quat_xyzw[2]" in body
    # Only the LEFT slot comes from the argument.
    assert body.count("lc_pos") >= 1 and "self.hold_pos_wire[0]" not in body


def test_the_model_only_ingests_the_force_roles_of_its_layout():
    """`_pick_force_channels` iterates ``force_roles`` — ('left',) for every HL-v2 checkpoint
    — so the RIGHT sensor's wrench never enters the observation the model conditions on."""
    src = _PLANNER.read_text()
    body = src[src.index("def _pick_force_channels"):]
    body = body[: body.index("\n\n#")]
    assert "for role in force_roles:" in body
    assert '"right"' not in body.split('"""')[2], "the force ingest hardcodes a right-hand key"


@_needs_holosoma
def test_the_low_level_masks_the_unused_hand_at_the_source():
    """`HOLOSOMA_FORCE_HANDS` must be applied where the sensor is READ, not per consumer.

    There are five independent consumers of the pelvis-force cache (obs, contact_phase,
    servo, runaway valve, hard stop) plus the F^cmd wire. Masking at the read point covers
    all of them at once; masking per consumer is how one keeps a live channel nobody
    remembers — the shape of bug the earlier F/T-gate review found in the runaway valve.
    """
    src = _POLICY.read_text()
    body = src[src.index("def _update_force_meas_and_servo"):]
    body = body[: body.index("\n    def ")]
    read_at = body.index("self._force_live.latest()")
    mask_at = body.index("_force_hands_enabled")
    ema_at = body.index("_force_meas_ema_alpha")
    gate_at = body.index("_force_meas_gate_on_cmd")
    assert read_at < mask_at < ema_at, "the mask must sit between the sensor read and the EMA"
    assert mask_at < gate_at, "the mask must precede the F^cmd gate, not depend on it"
    # The hard stop reads the sensor directly, so it carries the mask itself.
    stop = src[src.index("def _check_force_estop"):]
    stop = stop[: stop.index("\n    def ")]
    assert "_force_hands_enabled" in stop, "a masked hand could still halt the robot"


@_needs_holosoma
@_needs_launchers
def test_the_s1_planner_path_defaults_to_left_only_sensors():
    """`HOLOSOMA_VIVE_RELATIVE=0` (the planner feeding absolute targets) → `FORCE_HANDS=left`.

    Not merely tidy: this rig's RIGHT RNEA `wrench_sign` is still seeded/UNVALIDATED (see
    `2_ft_bridge.sh`), so a sign error there makes the reading grow with orientation instead
    of staying near zero. An untrusted sensor must not reach the actor obs, and must not be
    able to trip the hard stop.
    """
    text = _VIVE_LAUNCHER.read_text()
    assert 'FORCE_HANDS="${FORCE_HANDS:-left}"' in text
    assert 'FORCE_HANDS="${FORCE_HANDS:-both}"' in text, "direct-VIVE teleop must stay two-handed"
    assert 'HOLOSOMA_FORCE_HANDS="$FORCE_HANDS"' in text, "the selector never reaches the policy"
