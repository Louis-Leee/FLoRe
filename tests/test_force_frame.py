"""The bota-sensor -> yam-TCP force relabel, the meas->cmd SIGN, and the arrows drawn with them.

Four things are pinned here:

1. **The measured mapping itself.** ``sensor +x -> TCP +z``, ``+y -> TCP -y``, ``+z -> TCP +x``.
   Spelled out as three axis assertions rather than a matrix literal compared to a matrix
   literal, so a transposed or re-signed edit fails on the SENTENCE it violates.
2. **It is a proper rotation and its own inverse.** A mirror would flip torque handedness, and
   a second hand-written inverse constant is how the two directions drift apart.
3. **The measured->commanded sign.** Every viewer draws ``F^cmd = -F_meas``, because the low
   level closes ``e = F^cmd + F_meas -> 0`` and the models predict the MEASUREMENT. That the
   sign is what the low level's own servo implies is pinned across the repo seam in
   ``test_deploy_force_contract.py``; what is pinned HERE is that the arrow actually carries it,
   exactly once, and that it can be turned off to see the raw measurement.
4. **The UMI Collection copy agrees** on both the rotation and the sign. ``replay_common.py``
   cannot import from ``flore`` (the submodule has its own venv), so it carries copies — the same
   arrangement as the triad offsets in ``test_inference_display.py``, and pinned the same way.

The empirical provenance of the matrix lives in ``flore/utils/force_frame.py``'s docstring; the
one property asserted here is the CONSEQUENCE that makes the old behaviour a bug — the relabel
is ~180 deg from identity, so drawing the raw triple against a TCP-oriented frame pointed the
force the wrong way, not merely along a permuted axis.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from flore.s1_policy.inference.display import (
    FORCE_ARROW_MAX_LEN_M,
    FORCE_ARROW_MIN_N,
    FORCE_ARROW_SCALE_M_PER_N,
    force_arrow_local_segment,
)
from flore.utils.force_frame import (
    CMD_FROM_MEAS_NOTE,
    CMD_FROM_MEAS_SIGN,
    TCP_FROM_SENSOR,
    TCP_FROM_SENSOR_NOTE,
    cmd_from_meas_force,
    cmd_from_meas_wrench,
    sensor_from_tcp_force,
    tcp_from_sensor_force,
    tcp_from_sensor_wrench,
    world_force_from_sensor,
)

_REPO = Path(__file__).resolve().parents[1]
_UMI_REPLAY = _REPO / "third_party" / "UMI_collection" / "replay"

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def _literal_from_source(path: Path, name: str):
    """Read one module-level literal assignment out of a source file without importing it."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            continue
    raise AssertionError(f"{path.name} has no module-level literal assignment to {name}")


def _np_array_literal_from_source(path: Path, name: str) -> np.ndarray:
    """Read a module-level ``name = np.array([...], dtype=...)`` as a numpy array.

    ``ast.literal_eval`` cannot evaluate the ``np.array(...)`` call, so the nested-list
    argument is lifted out and evaluated on its own. Reading the SOURCE (rather than importing)
    is what keeps this test runnable in FLoRe's venv, which has no viser for the submodule.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        value = node.value
        assert isinstance(value, ast.Call), f"{name} in {path.name} is not an np.array(...) call"
        return np.asarray(ast.literal_eval(value.args[0]), dtype=np.float64)
    raise AssertionError(f"{path.name} has no module-level assignment to {name}")


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------
def test_relabel_maps_each_sensor_axis_where_the_measurement_says():
    """The three sentences from the rig measurement, one assertion each."""
    np.testing.assert_allclose(tcp_from_sensor_force(_X), _Z, atol=0)      # +x -> TCP +z
    np.testing.assert_allclose(tcp_from_sensor_force(_Y), -_Y, atol=0)     # +y -> TCP -y
    np.testing.assert_allclose(tcp_from_sensor_force(_Z), _X, atol=0)      # +z -> TCP +x
    # And the note the GUIs display says the same thing, so a label can't go stale silently.
    assert TCP_FROM_SENSOR_NOTE == "sensor +x->TCP +z, +y->TCP -y, +z->TCP +x"


def test_relabel_is_a_proper_rotation():
    """det == +1 and orthonormal. A det == -1 mirror would invert every torque's handedness."""
    np.testing.assert_allclose(np.linalg.det(TCP_FROM_SENSOR), 1.0, atol=1e-12)
    np.testing.assert_allclose(TCP_FROM_SENSOR @ TCP_FROM_SENSOR.T, np.eye(3), atol=1e-12)


def test_relabel_is_its_own_inverse():
    """``R == R.T == R^-1``, which is why there is no second inverse constant to drift."""
    np.testing.assert_array_equal(TCP_FROM_SENSOR, TCP_FROM_SENSOR.T)
    np.testing.assert_allclose(TCP_FROM_SENSOR @ TCP_FROM_SENSOR, np.eye(3), atol=1e-12)
    rng = np.random.default_rng(0)
    force = rng.normal(size=(7, 3))
    np.testing.assert_allclose(sensor_from_tcp_force(tcp_from_sensor_force(force)), force, atol=1e-12)


def test_relabel_is_a_180_degree_turn_away_from_identity():
    """The property that makes the pre-fix arrows a BUG rather than a cosmetic axis swap.

    Identity is what every viewer implicitly used before, and it is not "close" to the truth:
    the two differ by a half turn, so the drawn force pointed into the wiped surface instead of
    out of it. Pinned so nobody "simplifies" the matrix back toward identity.
    """
    cosine = (np.trace(TCP_FROM_SENSOR) - 1.0) / 2.0
    np.testing.assert_allclose(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 180.0, atol=1e-9)
    # ...about (1, 0, 1)/sqrt(2): that axis is the one direction the relabel leaves alone.
    axis = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(tcp_from_sensor_force(axis), axis, atol=1e-12)


def test_wrench_rotates_both_triples_and_preserves_magnitudes():
    rng = np.random.default_rng(1)
    wrench = rng.normal(size=(5, 6))
    rotated = tcp_from_sensor_wrench(wrench)
    np.testing.assert_allclose(rotated[:, 0:3], tcp_from_sensor_force(wrench[:, 0:3]), atol=1e-12)
    np.testing.assert_allclose(rotated[:, 3:6], tcp_from_sensor_force(wrench[:, 3:6]), atol=1e-12)
    # A rotation is an isometry on each triple independently.
    for cols in (slice(0, 3), slice(3, 6)):
        np.testing.assert_allclose(
            np.linalg.norm(rotated[:, cols], axis=-1), np.linalg.norm(wrench[:, cols], axis=-1), atol=1e-12
        )


def test_world_force_composes_both_legs_and_broadcasts():
    """``R_world_tcp @ TCP_FROM_SENSOR @ f``, elementwise over a stack of rotations."""
    rng = np.random.default_rng(2)
    force = rng.normal(size=(4, 3))
    rotations = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(4)])
    rotations[np.linalg.det(rotations) < 0] *= -1.0
    expected = np.einsum("tij,tj->ti", rotations, tcp_from_sensor_force(force))
    np.testing.assert_allclose(world_force_from_sensor(force, rotations), expected, atol=1e-12)
    # A single shared rotation broadcasts against a stack of forces.
    np.testing.assert_allclose(
        world_force_from_sensor(force, rotations[0]),
        np.einsum("ij,tj->ti", rotations[0], tcp_from_sensor_force(force)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "bad, func",
    [
        (np.zeros((2, 2)), tcp_from_sensor_force),
        (np.zeros((3, 5)), tcp_from_sensor_wrench),
        (np.zeros(3), tcp_from_sensor_wrench),
    ],
)
def test_wrong_widths_are_refused(bad, func):
    with pytest.raises(ValueError):
        func(bad)


def test_world_force_refuses_a_non_rotation_shape():
    with pytest.raises(ValueError):
        world_force_from_sensor(np.zeros(3), np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# The measured -> commanded sign
# ---------------------------------------------------------------------------
def test_the_command_is_the_negated_measurement():
    """``F^cmd = -F_meas``, elementwise, on both triples of a wrench.

    The one sentence the whole feature rests on. The low level's force loop is
    ``e = F^cmd + F_meas -> 0``, so at the fixed point the measurement OPPOSES the command; a
    command equal to the measurement would leave the error at ``2*F_meas``.
    """
    assert CMD_FROM_MEAS_SIGN == -1.0
    force = np.array([3.0, -4.0, 12.0])
    np.testing.assert_array_equal(cmd_from_meas_force(force), -force)
    # e = F_cmd + F_meas is exactly zero for this pair — the property, not just the sign.
    np.testing.assert_array_equal(cmd_from_meas_force(force) + force, np.zeros(3))
    wrench = np.arange(12, dtype=np.float64).reshape(2, 6)
    np.testing.assert_array_equal(cmd_from_meas_wrench(wrench), -wrench)
    # And the note the GUIs display says the same thing, so a label cannot go stale silently.
    assert CMD_FROM_MEAS_NOTE == "F^cmd = -F_meas (the low level drives F^cmd + F_meas -> 0)"


def test_the_sign_is_its_own_inverse_and_leaves_magnitude_alone():
    """Why there is no ``MEAS_FROM_CMD``: applying it twice is the identity.

    That is also the hazard — a second, accidental application is INVISIBLE, which is why the
    flip lives in one place per surface. And ``‖F^cmd‖ == ‖F_meas‖``, so every magnitude gate
    (the planner's force gate, the low level's deadband, the arrow's noise floor) means exactly
    what it did before the flip existed.
    """
    rng = np.random.default_rng(3)
    force = rng.normal(size=(6, 3))
    np.testing.assert_array_equal(cmd_from_meas_force(cmd_from_meas_force(force)), force)
    np.testing.assert_allclose(
        np.linalg.norm(cmd_from_meas_force(force), axis=-1), np.linalg.norm(force, axis=-1), atol=0
    )


def test_the_sign_commutes_with_the_relabel():
    """A scalar commutes with a rotation, which is WHY the flip may live planner-side.

    The planner must not rotate (the low level owns sensor->pelvis, it has the joint readback),
    but it may negate: the composed result is identical whichever order the two are applied in,
    so flipping before the wire and rotating after it is the same transform as doing both in
    either place.
    """
    rng = np.random.default_rng(4)
    force = rng.normal(size=(5, 3))
    np.testing.assert_allclose(
        tcp_from_sensor_force(cmd_from_meas_force(force)),
        cmd_from_meas_force(tcp_from_sensor_force(force)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "bad, func",
    [
        (np.zeros((4, 2)), cmd_from_meas_force),
        (np.zeros(3), cmd_from_meas_wrench),
    ],
)
def test_the_sign_helpers_refuse_the_wrong_width(bad, func):
    """Same width discipline as the rotations: a 6-wide wrench handed to the 3-wide force
    helper would silently negate only its first three columns."""
    with pytest.raises(ValueError):
        func(bad)


def test_the_sign_helper_does_not_alias_its_input():
    """The callers publish or draw the result; aliasing the model's own output buffer would let
    a viewer's flip mutate the array the next stage reads."""
    force = np.array([1.0, 2.0, 3.0])
    out = cmd_from_meas_force(force)
    out[0] = 99.0
    np.testing.assert_array_equal(force, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Arrow geometry
# ---------------------------------------------------------------------------
def test_arrow_points_along_the_commanded_force_from_the_trajectory_point():
    """Tail exactly at the origin, tip along the TCP-frame COMMAND direction, length |F| * scale.

    The tail must be EXACTLY zero: the node is parented under the frame driven to the
    trajectory point, so a nonzero tail would detach the arrow from its own datum — the same
    failure the TCP-triad offset test guards against for the triads.

    A measured sensor ``+z`` is the TCP's ``+x``, and the COMMAND that reproduces it is the
    negation, so the drawn tip is TCP ``-x``. Both conversions in one assertion, on purpose:
    they are applied together in one function and this is the composed result a viewer sees.
    """
    force = np.array([0.0, 0.0, 6.0])  # measured sensor +z -> TCP +x -> commanded TCP -x
    segment = force_arrow_local_segment(force, scale_m_per_n=0.01, max_len_m=1.0, min_n=0.2)
    assert segment.shape == (2, 3)
    np.testing.assert_array_equal(segment[0], np.zeros(3))
    np.testing.assert_allclose(segment[1], np.array([-0.06, 0.0, 0.0]), atol=1e-12)


def test_arrow_draws_the_raw_measurement_when_asked():
    """``as_command=False`` is the escape hatch, and it must be the EXACT negation of the
    default — otherwise the two modes differ by something other than the documented sign."""
    force = np.array([0.0, 0.0, 6.0])
    kwargs = {"scale_m_per_n": 0.01, "max_len_m": 1.0, "min_n": 0.2}
    measured = force_arrow_local_segment(force, as_command=False, **kwargs)
    commanded = force_arrow_local_segment(force, as_command=True, **kwargs)
    np.testing.assert_allclose(measured[1], np.array([0.06, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(commanded[1], -measured[1], atol=1e-12)


def test_arrow_length_is_clamped_but_direction_is_not():
    """A spike may not fill the scene, yet must still say which way it points."""
    force = np.array([40.0, 0.0, 0.0])
    segment = force_arrow_local_segment(force, scale_m_per_n=0.01, max_len_m=0.25, min_n=0.2)
    np.testing.assert_allclose(np.linalg.norm(segment[1]), 0.25, atol=1e-12)
    np.testing.assert_allclose(
        segment[1] / 0.25, tcp_from_sensor_force(cmd_from_meas_force(force)) / 40.0, atol=1e-12
    )


def test_arrow_is_hidden_below_the_noise_floor_and_for_non_finite_input():
    """None means "draw nothing", which the viewer renders as a hidden node.

    Sub-threshold must be indistinguishable from no-contact rather than a jittering stub, and a
    NaN wrench must never become an arrow of undefined direction.
    """
    assert force_arrow_local_segment(np.array([0.05, 0.0, 0.0]), min_n=0.2) is None
    assert force_arrow_local_segment(np.zeros(3)) is None
    assert force_arrow_local_segment(np.array([np.nan, 0.0, 0.0])) is None
    assert force_arrow_local_segment(np.array([np.inf, 0.0, 0.0])) is None
    # Exactly AT the threshold draws (the comparison is strict-less-than).
    assert force_arrow_local_segment(np.array([0.0, 0.0, 0.2]), min_n=0.2) is not None


def test_arrow_defaults_are_sane_for_this_rig():
    """Defaults must draw the session's peak force at a readable, non-scene-filling size."""
    assert FORCE_ARROW_SCALE_M_PER_N == 0.01          # parity with UMI's WrenchArrows default
    assert 0.0 < FORCE_ARROW_MIN_N < 0.5              # under the free-space |F| median (0.42 N)
    # The 0812 session peaks at 18.5 N; at the default scale that is 18.5 cm, under the clamp,
    # so the clamp only engages on a genuine outlier rather than flattening normal contact.
    assert 18.5 * FORCE_ARROW_SCALE_M_PER_N < FORCE_ARROW_MAX_LEN_M


def test_arrow_rejects_a_non_3_vector():
    with pytest.raises(ValueError):
        force_arrow_local_segment(np.zeros(6))


def test_arrow_colours_are_exactly_the_colours_of_the_trajectories_they_belong_to():
    """GT arrow == the blue target line, pred arrow == the orange prediction line, playback
    arrow == the green recorded-episode line.

    This is the whole readability argument: the operator already learned blue/orange/green from
    the three polylines, so an arrow's colour must answer "whose force is this?" with no second
    legend. Derived from the line constants rather than re-typed, and pinned here so a tweak to
    a line colour cannot silently leave its arrow behind. Imported at module level in
    ``viewer.py``, which stays importable without viser.
    """
    from flore.s1_policy.inference import viewer as v

    assert v._FORCE_TARGET_COLOR == tuple(int(c) for c in v._TARGET_BLUE)
    assert v._FORCE_PREDICTION_COLOR == tuple(int(c) for c in v._PREDICTION_ORANGE)
    assert v._FORCE_PLAYBACK_COLOR == tuple(int(c) for c in v._EPISODE_GREEN)
    # Distinct, and plain int tuples (viser's add_arrows wants a colour, not a uint8 array).
    colours = (v._FORCE_TARGET_COLOR, v._FORCE_PREDICTION_COLOR, v._FORCE_PLAYBACK_COLOR)
    assert len(set(colours)) == 3, f"force arrow colours are not distinct: {colours}"
    for colour in colours:
        assert all(type(c) is int and 0 <= c <= 255 for c in colour), colour
    # The GT arrow is the THINNER of the pair, so a near-perfect prediction occludes it rather
    # than the reverse — the same reason and the same number as the target triads.
    assert v._FORCE_ARROW_TARGET_SCALE == v._TARGET_TRIAD_SCALE < 1.0


# ---------------------------------------------------------------------------
# Parity with the UMI Collection copy
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _UMI_REPLAY.is_dir(), reason="UMI_collection submodule not checked out")
def test_umi_replay_copy_of_the_relabel_matches():
    """``replay_common.SENSOR_TO_TCP`` is the same matrix as :data:`TCP_FROM_SENSOR`.

    The zarr replay and the raw replay draw their arrows through that copy, so a drift here
    means the offline inference viewer and the two replay tools disagree about which way the
    contact force points — the exact class of split this repo's triad-offset test exists to
    prevent. Imported by SOURCE because the submodule's module-level ``import viser`` is not
    installed in FLoRe's venv.
    """
    copy = _np_array_literal_from_source(_UMI_REPLAY / "replay_common.py", "SENSOR_TO_TCP")
    np.testing.assert_array_equal(copy, TCP_FROM_SENSOR)


@pytest.mark.skipif(not _UMI_REPLAY.is_dir(), reason="UMI_collection submodule not checked out")
def test_umi_replay_copy_of_the_meas_to_cmd_sign_matches():
    """``replay_common.CMD_FROM_MEAS_SIGN`` is the same number as :data:`CMD_FROM_MEAS_SIGN`.

    Same venv-split argument as the relabel above, and the same consequence if it drifts: the
    zarr replay and the raw replay would draw the demonstrated contact one way while the offline
    viewer and the wire went the other, with each tool internally consistent.
    """
    copy = _literal_from_source(_UMI_REPLAY / "replay_common.py", "CMD_FROM_MEAS_SIGN")
    assert float(copy) == CMD_FROM_MEAS_SIGN


@pytest.mark.skipif(not _UMI_REPLAY.is_dir(), reason="UMI_collection submodule not checked out")
def test_umi_wrench_arrows_apply_the_relabel_and_the_sign_exactly_once():
    """``WrenchArrows.update`` must compose BOTH the relabel and the sign itself, and no caller
    may pre-apply either.

    Both halves are needed for the arrow to be right: the class applying them makes all three
    tools correct at once, and a caller ALSO applying one would double-apply it — and both are
    involutions, so the second application is silently the identity, i.e. back to the original
    bug with nothing on screen to say so.
    """
    source = (_UMI_REPLAY / "replay_common.py").read_text()
    assert "as_matrix() @ SENSOR_TO_TCP" in source, "WrenchArrows.update no longer composes the relabel"
    assert "CMD_FROM_MEAS_SIGN * R" in source, "WrenchArrows.update no longer composes the meas->cmd sign"
    for tool in ("replay_zarr_viz.py", "replay_raw_viz.py"):
        text = (_UMI_REPLAY / tool).read_text()
        assert "SENSOR_TO_TCP" not in text, f"{tool} pre-rotates the wrench; update() already does"
        assert "CMD_FROM_MEAS_SIGN" not in text, f"{tool} pre-negates the wrench; update() already does"
    planner = (_REPO / "scripts" / "hl_serving.py").read_text()
    assert "tcp_from_sensor_wrench" not in planner, (
        "hl_serving pre-rotates the wrench before WrenchArrows.update, which double-rotates it"
    )
    # hl_serving DOES import cmd_from_meas_wrench (for its charts) — what it must not do is hand
    # a pre-negated wrench to WrenchArrows.update, which negates for itself.
    call = planner[planner.index("wrench.update("):]
    call = call[: call.index("\n")]
    assert "cmd_from_meas" not in call, (
        "hl_serving pre-negates the wrench passed to WrenchArrows.update, which double-negates it"
    )


# ---------------------------------------------------------------------------
# The planner's force arrow rides the GHOST hand (frame + anchor agreement)
# ---------------------------------------------------------------------------
def test_planner_force_arrow_rides_the_predicted_ghost_hand_not_follow():
    """The F^cmd arrow is anchored at the PREDICTED per-step **TCP** (the ghost's fingertip as it
    plays the loop), NOT at /follow (the live robot) and no longer at the wrist.

    Two properties in one:

    * **the ghost, not /follow** — the force is read at the pose the plan acts on, like a demo's
      F/T in replay, and /wrench stays a TOP-LEVEL node so it is given absolute world coordinates
      rather than coordinates that /follow's position would double.
    * **the TCP, not the wrist** — ``path`` is ``base_xyz + the model's absolute TCP``, which is
      where contact happens and where the "tcp" triad is drawn; the wrist origin it used to use is
      24.2 cm back along +x, so the arrow stood off both. Anchoring on the TCP is also what makes
      this page agree with the offline inference viewer, whose arrows are children of a frame
      driven to the TCP.
    """
    source = (_REPO / "scripts" / "hl_serving.py").read_text()
    assert "_ap, _aq = (path[i], wwxyz[i])" in source, (
        "the F^cmd arrow no longer rides the predicted ghost hand's TCP"
    )
    assert "force_quiver_world_segments" not in source, "the removed force quiver is back"
    assert "/rollout/force" not in source, "the force quiver node is back"


def test_planner_draws_the_measured_wrench_at_the_real_robots_tcp():
    """The F_meas counterpart: a SECOND arrow, un-negated, on the LIVE robot's FK'd TCP.

    The pair is the force loop — the low level drives ``F^cmd + F_meas -> 0``, so anti-parallel
    equal arrows are its fixed point. For that reading to mean anything, three things must hold,
    and each is a separate way to get it silently wrong:

    1. the measured instance must NOT negate (``as_command=False``) — otherwise both arrows point
       the same way and the tracking check inverts;
    2. it must anchor on ``live_tcp`` (the RecordTap FK), not on the ghost or on a stale pose, so
       a live reading is never attached to a pose that is not the robot's;
    3. the two sources take DIFFERENT frame chains — the low level's ``F_meas`` is already
       pelvis-frame and must go through ``update_world`` (no relabel), while the raw ``:8771``
       wrench is sensor-frame and must go through ``update`` (relabel applied). Swapping them
       rotates an already-rotated vector, or fails to rotate one that needs it.
    """
    source = (_REPO / "scripts" / "hl_serving.py").read_text()
    assert 'name="/wrench_meas"' in source and "as_command=False" in source, (
        "the measured-wrench arrow is gone, or it now negates like the command"
    )
    body = source[source.index("_meas_arrow_n = None"):]
    body = body[: body.index("wrench_meas.set_visible(")]
    assert "wrench_meas.update_world(live_tcp[0]" in body, (
        "the pelvis-frame F_meas no longer goes through update_world — it would be double-rotated"
    )
    assert "wrench_meas.update(live_tcp[0]" in body, (
        "the raw sensor-frame F_meas no longer goes through update() — it would be un-relabelled"
    )
    assert "live_fmeas_ungated" in body, (
        "the arrow reads the GATED F_meas, which GATE_CMD=1 pins to zero for all of pure position"
    )


# ---------------------------------------------------------------------------
# The live F_meas trace behind the planner's scrolling charts
# ---------------------------------------------------------------------------
def test_ft_trace_emits_the_interval_MEAN_not_a_decimated_sample():
    """Each plotted point must be the mean of every raw sample in its interval.

    This is the whole reason the trace is filled on the F/T socket thread instead of polled by the
    render loop: the sensor runs ~200 Hz and the chart shows 30, so sampling the latest-value
    register (what the collection viz does) would ALIAS — a contact spike could land at the wrong
    amplitude, or between samples and vanish. Asserted on a signal whose mean and whose
    last-sample differ, so a decimating implementation cannot pass.
    """
    import scripts.hl_serving as hs

    trace = hs._FtDisplayTrace(window_s=1.0, hz=10.0)   # 100 ms intervals
    # Six samples inside one interval: values 0..5 on Fx. mean = 2.5, last = 5.
    for k in range(6):
        trace.push(np.array([float(k), 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0 + k * 0.015)
    trace.push(np.zeros(6), 0.10)                       # crosses the boundary → emit
    trace.push(np.zeros(6), 0.20)                       # a second point, so series() is non-None
    out = trace.series()
    assert out is not None
    times, wrench, _peak = out
    np.testing.assert_allclose(wrench[0, 0], (0 + 1 + 2 + 3 + 4 + 5 + 0) / 7.0, atol=1e-12)
    assert times[0] == pytest.approx(0.10)              # t is RELATIVE to the first sample


def test_ft_trace_reports_the_UNFLATTENED_peak():
    """‖F‖ peak over the window, not the norm of the means.

    The charts show means by construction, so a sub-interval spike is flattened THERE — which is
    exactly why the peak is carried separately and printed on the page: it is the number an
    operator compares against the 40 N hard stop.
    """
    import scripts.hl_serving as hs

    trace = hs._FtDisplayTrace(window_s=1.0, hz=10.0)
    trace.push(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0)
    trace.push(np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.01)   # the spike
    trace.push(np.zeros(6), 0.10)
    trace.push(np.zeros(6), 0.20)
    times, wrench, peak = trace.series()
    assert peak == pytest.approx(30.0)
    assert np.linalg.norm(wrench[0, 0:3]) < 30.0, "the mean should have flattened the spike"


def test_ft_trace_ignores_unreadable_samples_rather_than_zero_filling():
    """A missing or non-finite frame means "unknown". Charting it as 0 N would read as "no
    contact", which is the one wrong answer — the operator is watching this to decide whether the
    tool is touching anything."""
    import scripts.hl_serving as hs

    trace = hs._FtDisplayTrace(window_s=1.0, hz=10.0)
    for bad in (None, np.zeros(3), np.array([np.nan] + [0.0] * 5), np.array([np.inf] + [0.0] * 5)):
        trace.push(bad, 0.0)
    assert trace.series() is None, "a rejected sample started the trace anyway"
    # ...and a good sample after them still works, with t rebased on IT.
    trace.push(np.ones(6), 5.0)
    trace.push(np.ones(6), 5.2)
    trace.push(np.ones(6), 5.4)
    times, wrench, _ = trace.series()
    assert times[0] == pytest.approx(0.2)
    np.testing.assert_allclose(wrench[0], np.ones(6), atol=1e-12)


def test_ft_trace_window_is_bounded():
    """It runs for the whole session, so it must not grow. ~window_s * hz points, no more."""
    import scripts.hl_serving as hs

    trace = hs._FtDisplayTrace(window_s=1.0, hz=10.0)
    for k in range(2000):
        trace.push(np.ones(6), k * 0.05)                # 100 s of samples into a 1 s window
    times, wrench, _ = trace.series()
    assert len(times) <= 12, f"window grew to {len(times)} points"
    assert wrench.shape[1] == 6


def test_planner_ghost_is_a_translucent_distinct_colour():
    """The ghost's RGBA is semi-transparent and its hue is none of the scene's trajectory
    colours (yellow plan / green demo / red trace), so a translucent robot is never read as a
    line — the reason a distinct violet was chosen over the requested green."""
    import scripts.hl_serving as hs

    assert len(hs._GHOST_RGBA) == 4, "the ghost colour must carry an alpha (translucent)"
    r, g, b, a = hs._GHOST_RGBA
    assert 0.0 < a < 1.0, "the ghost must be semi-transparent, not opaque or invisible"
    for chan in (r, g, b):
        assert 0.0 <= chan <= 1.0, "ViserUrdf mesh_color_override channels are 0-1 floats"
    # violet: blue dominant, red present, green suppressed → clearly not the green demo colour.
    assert b > g and r > g, "the ghost hue is not the violet that distinguishes it from the demo green"
