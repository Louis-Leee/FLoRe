"""The offline-sample driver: the page protocol, the cursor, the ghost, and the 3pt wire.

Everything here is CPU-only and model-free. The end-to-end path (a real checkpoint, a real
dataset, a real ``:8770`` socket) is ``scripts/smoke_test_hl_offline.py``, which drives the same
protocol from a WebSocket client; these are the properties that must hold without one.
"""

from __future__ import annotations

import numpy as np
import pytest

from flore.client.s0_driver import _RESERVED_WIRE_KEYS, _merge_extra
from flore.s1_policy.data.layout import get_layout
from flore.s1_policy.serving import hl_offline
from flore.s1_policy.serving.hl_dense import KEYPOINT_GATE_TOL_M, PostAnchorConfig


# ---------------------------------------------------------------------------
# the page protocol
# ---------------------------------------------------------------------------
def _state(n_samples: int = 100, n_episodes: int = 5) -> dict:
    return {"n_samples": n_samples, "n_episodes": n_episodes, "sample": 0,
            "req": None, "want_sample": None, "want_episode": None, "stop": False}


@pytest.mark.parametrize("action", ["predict", "run", "reset", "auto"])
def test_simple_verbs_queue_a_request(action):
    st = _state()
    line = hl_offline.apply_control(st, {"cmd": "flore", "action": action})
    assert st["req"] == action
    assert action in line


def test_stop_is_a_latch_that_a_later_verb_cannot_overwrite():
    """The one verb whose value is that it cannot be lost — see ``apply_control``."""
    st = _state()
    hl_offline.apply_control(st, {"cmd": "flore", "action": "stop"})
    assert st["stop"] is True
    hl_offline.apply_control(st, {"cmd": "flore", "action": "predict"})
    assert st["req"] == "predict"
    assert st["stop"] is True, "a later verb cleared a latched STOP"


def test_sample_out_of_range_is_refused_and_names_the_range():
    st = _state(n_samples=10)
    line = hl_offline.apply_control(st, {"cmd": "flore", "action": "sample", "index": 10})
    assert st["want_sample"] is None and st["req"] is None
    assert "0..9" in line


def test_episode_out_of_range_is_refused():
    st = _state(n_episodes=3)
    line = hl_offline.apply_control(st, {"cmd": "flore", "action": "episode", "index": -1})
    assert st["want_episode"] is None and st["req"] is None
    assert "0..2" in line


@pytest.mark.parametrize("msg", [
    {"cmd": "flore", "action": "sample"},                  # no index
    {"cmd": "flore", "action": "sample", "index": "x"},     # not an int
    {"cmd": "flore", "action": "sample", "index": None},
])
def test_malformed_index_requests_are_refused(msg):
    st = _state()
    line = hl_offline.apply_control(st, msg)
    assert st["want_sample"] is None and st["req"] is None
    assert "malformed" in line


def test_unknown_action_is_refused_by_name():
    st = _state()
    line = hl_offline.apply_control(st, {"cmd": "flore", "action": "fly"})
    assert st["req"] is None
    assert "'fly'" in line


def test_another_producers_verb_is_silent_not_an_error():
    """One socket, two vocabularies: the MC replay bridge's ``episode`` must not be answered."""
    st = _state()
    assert hl_offline.apply_control(st, {"cmd": "episode", "index": 3}) == ""
    assert st["req"] is None and st["want_episode"] is None


def test_recalibrate_is_mapped_to_reset():
    """The page's Recalibrate button exists on every live obs class and cannot be hidden."""
    st = _state()
    line = hl_offline.apply_control(st, {"cmd": "recalibrate"})
    assert st["req"] == "reset"
    assert "reset" in line


def test_non_object_message_is_refused():
    st = _state()
    assert "non-object" in hl_offline.apply_control(st, ["flore", "predict"])


# ---------------------------------------------------------------------------
# the cursor
# ---------------------------------------------------------------------------
class _Ref:
    def __init__(self, ep: int, frame: int, split: str = "train") -> None:
        self.global_episode_id = ep
        self.episode_frame = frame
        self.split = split


def _refs_with_holes():
    """Three episodes whose anchors are NOT contiguous ranges.

    Which is the real shape: ``drop_glitch_windows`` and the force/validity filters remove
    individual anchors, so an episode is a SET of indices with holes. A cursor that assumed
    ``[start, start + n)`` would land outside its own episode on any QC'd dataset.
    """
    return [_Ref(0, 0), _Ref(0, 1), _Ref(2, 5), _Ref(0, 9), _Ref(2, 6), _Ref(7, 0, "validation")]


def test_cursor_groups_episodes_rather_than_assuming_a_range():
    c = hl_offline.SampleCursor(_refs_with_holes())
    assert len(c) == 6
    assert c.n_episodes == 3
    at = c.at()
    assert (at.index, at.episode, at.episode_index, at.episode_frames, at.episode_offset) == \
        (0, 0, 0, 3, 0)
    # episode 0's anchors are {0, 1, 3}: the third one is index 3, not index 2.
    assert c.set_index(3)
    assert c.at().episode == 0 and c.at().episode_offset == 2


def test_set_episode_lands_on_that_episodes_first_ANCHOR():
    c = hl_offline.SampleCursor(_refs_with_holes())
    assert c.set_episode(1)                     # the 2nd episode present, id 2
    assert c.index == 2 and c.at().episode == 2
    assert c.set_episode(2)
    assert c.index == 5 and c.at().split == "validation"


def test_step_clamps_instead_of_wrapping():
    c = hl_offline.SampleCursor(_refs_with_holes())
    assert not c.step(-1), "stepping back from 0 should be a no-op, not a wrap"
    assert c.index == 0
    assert c.step(99) and c.index == 5
    assert not c.step(1)


def test_step_episode_clamps():
    c = hl_offline.SampleCursor(_refs_with_holes())
    assert c.step_episode(+1) and c.at().episode == 2
    assert c.step_episode(+5) and c.at().episode == 7
    assert not c.step_episode(+1)


def test_out_of_range_and_unchanged_index_both_report_no_move():
    c = hl_offline.SampleCursor(_refs_with_holes())
    assert not c.set_index(0)          # already there
    assert not c.set_index(6)          # past the end
    assert not c.set_index(-1)
    assert c.index == 0


def test_an_empty_dataset_is_refused():
    with pytest.raises(ValueError, match="no usable anchors"):
        hl_offline.SampleCursor([])


# ---------------------------------------------------------------------------
# the wire keys
# ---------------------------------------------------------------------------
def test_ghost_extras_publishes_an_identity_root_and_names_the_joints():
    q = np.linspace(-1.0, 1.0, 29)
    names = [f"j{i}" for i in range(29)]
    out = hl_offline.ghost_extras(q, names)
    assert out["mc_joint_names"] == names
    assert out["mc_qpos"] == pytest.approx(q.tolist())
    # The page composes the root onto the REFERENCE CLIP's root, so identity == "stand at the
    # reference pelvis and let the 29 joints be the content". A wholebody chunk has no root.
    assert out["mc_root_pos"] == [0.0, 0.0, 0.0]
    assert out["mc_root_quat"] == [1.0, 0.0, 0.0, 0.0]


def test_ghost_extras_refuses_a_width_mismatch_because_the_page_would_be_silent():
    with pytest.raises(ValueError, match="BY NAME"):
        hl_offline.ghost_extras(np.zeros(29), [f"j{i}" for i in range(28)])


def test_ghost_extras_refuses_non_finite():
    q = np.zeros(3)
    q[1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        hl_offline.ghost_extras(q, ["a", "b", "c"])


def _cursor_at() -> hl_offline.CursorAt:
    return hl_offline.CursorAt(index=7, episode=3, episode_frame=11, episode_index=2,
                              episode_frames=40, episode_offset=5, split="train")


def test_page_extras_carries_the_gate_key_and_the_ack():
    out = hl_offline.page_extras(cursor=_cursor_at(), n_samples=1234, n_episodes=9,
                                state="staged", message="hi", layout_note="wholebody",
                                chunk_steps=50, exec_steps=16, step=3, ghost_step=4, ack=2)
    assert out[hl_offline.PAGE_GATE_KEY] == 1234
    assert out["flore_ack"] == 2
    assert out["flore_sample"] == 7 and out["flore_episode"] == 3
    assert out["flore_episode_index"] == 2 and out["flore_split"] == "train"
    assert out["flore_state"] == "staged" and out["flore_step"] == 3


def test_page_extras_refuses_an_unknown_state():
    with pytest.raises(ValueError, match="state must be one of"):
        hl_offline.page_extras(cursor=_cursor_at(), n_samples=1, n_episodes=1, state="wat",
                               message="", layout_note="")


def test_page_extras_keys_never_collide_with_the_wire_and_are_json_safe():
    """The gate/telemetry keys ride on the SAME frame as the command, so a collision would be a
    frame corruption rather than a UI bug — ``S0Driver`` refuses them, and this pins that none of
    ours is in that set to begin with."""
    import json

    out = hl_offline.page_extras(cursor=_cursor_at(), n_samples=1, n_episodes=1, state="hold",
                                message="x" * 400, layout_note="y" * 400)
    out.update(hl_offline.ghost_extras(np.zeros(2), ["a", "b"]))
    assert not _RESERVED_WIRE_KEYS.intersection(out)
    json.loads(json.dumps(out))       # no numpy scalars, no NaN
    assert len(out["flore_msg"]) <= 240 and len(out["flore_layout"]) <= 120


@pytest.mark.parametrize("key", sorted(_RESERVED_WIRE_KEYS))
def test_the_wire_refuses_extras_that_would_shadow_its_own_fields(key):
    with pytest.raises(ValueError, match="collide"):
        _merge_extra({"mode": "dense"}, {key: 1})


def test_merge_extra_is_the_identity_when_nothing_is_passed():
    """Every existing deploy passes nothing, so the payload must be byte-identical."""
    base = {"mode": "dense", "seq": 1}
    assert _merge_extra(dict(base), None) == base
    assert _merge_extra(dict(base), {}) == base


# ---------------------------------------------------------------------------
# the ramp
# ---------------------------------------------------------------------------
def test_ramp_is_per_element_and_arrives_exactly():
    cur = np.zeros(3)
    tgt = np.array([1.0, -0.05, 0.0])
    q, done = hl_offline.ramp_step(cur, tgt, 0.1)
    assert not done
    # PER-ELEMENT: the small channel arrives in one step while the big one is still clipped. An
    # L2-normalised step would have moved it by 0.005 instead.
    assert q == pytest.approx([0.1, -0.05, 0.0])
    for _ in range(200):
        q, done = hl_offline.ramp_step(q, tgt, 0.1)
        if done:
            break
    assert done and q == pytest.approx(tgt)


def test_a_nonpositive_or_nonfinite_step_teleports_rather_than_stalling():
    """A zero rate must not be a wire that never arrives; it collapses to "no ramp"."""
    for bad in (0.0, -1.0, np.nan, np.inf):
        q, done = hl_offline.ramp_step(np.zeros(2), np.ones(2), bad)
        assert done and q == pytest.approx([1.0, 1.0])


def test_ramp_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        hl_offline.ramp_step(np.zeros(3), np.zeros(4), 0.1)


# ---------------------------------------------------------------------------
# the 3pt wire
# ---------------------------------------------------------------------------
_READY_POS = np.array([0.4418, 0.1486, 0.0952,      # LEFT hand, the 0.242 m force TCP
                       0.4418, -0.1486, 0.0952,     # RIGHT hand
                       -0.004, 0.0, 0.394])         # head/torso, torso_link + 0.35 z
_READY_ORN = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), 3)


def _wire(**kw) -> hl_offline.ThreePointWire:
    return hl_offline.ThreePointWire(_READY_POS, _READY_ORN, **kw)


def test_three_point_labels_are_the_first_three_of_the_five():
    from flore.s1_policy.serving.hl_dense import FIVE_POINT_LABELS

    assert hl_offline.THREE_POINT_N == 3
    assert hl_offline.THREE_POINT_LABELS == FIVE_POINT_LABELS[:3]


def test_the_ready_frame_passes_its_own_gate_by_construction():
    """THE load-bearing property. A 3pt chunk is decoded against the ready pose, so its step 0
    lands ON the pre-arm gate's reference and the gate closes without an operator posture."""
    w = _wire()
    ok, note = w.gate_report(w.ready_frame)
    assert ok and "PASS" in note
    assert "LEFT hand 0 mm" in note


def test_the_gate_refuses_and_names_the_point_that_is_out():
    w = _wire()
    pos = _READY_POS.copy()
    pos[3] += 2.0 * KEYPOINT_GATE_TOL_M          # the RIGHT hand's x
    ok, note = w.gate_report((pos, _READY_ORN))
    assert not ok and "REFUSE" in note
    assert "RIGHT hand 140 mm" in note


def test_plan_is_the_identity_with_the_default_post_anchor():
    w = _wire()
    pos = np.tile(_READY_POS, (5, 1)) + np.linspace(0, 0.1, 5)[:, None]
    orn = np.tile(_READY_ORN, (5, 1))
    plan = w.plan(pos, orn)
    assert plan["n"] == 5
    assert plan["pos"] == pytest.approx(pos)
    assert plan["orn"] == pytest.approx(orn)


def test_post_anchor_command_carries_ALL_THREE_points():
    """Unlike the 5pt wire, which carries only 3 of 5: there every point past the third is the
    completion's inferred FOOT, and re-basing the stance moved the feet 66-105 mm. Here all three
    are the policy's own predictions, so all three are carried."""
    w = _wire(post=PostAnchorConfig(mode="command"))
    assert w.clutch.n_points == 3
    assert w.clutch.n_carried == 3
    pos = np.tile(_READY_POS, (4, 1))
    pos[:, 0] += np.arange(4) * 0.01
    plan = w.plan(pos, np.tile(_READY_ORN, (4, 1)))
    # step 0 lands on the base (the ready frame) exactly; the motion is preserved.
    assert plan["pos"][0] == pytest.approx(_READY_POS)
    assert plan["pos"][3][0] - plan["pos"][0][0] == pytest.approx(0.03)


def test_the_idle_hold_never_chases_a_readback():
    """A keypoint wire's gate reference is the FROZEN reference clip, not the encoders, so
    tracking measured would move the wire AWAY from what the gate wants."""
    w = _wire()
    assert w.measured(object()) is None
    assert w.idle()[0] == pytest.approx(_READY_POS)


def test_publish_gates_non_finite_and_denormalised_quaternions():
    sent = []

    class _S0:
        def send(self, pos, orn, f_cmd=None, extra=None):
            sent.append((pos, orn, f_cmd, extra))

    w, s0 = _wire(), _S0()
    bad = _READY_POS.copy()
    bad[4] = np.nan
    ok, why = w.publish(s0, (bad, _READY_ORN), None)
    assert not ok and "non-finite" in why
    ok, why = w.publish(s0, (_READY_POS, _READY_ORN * 0.5), None)
    assert not ok and "norm" in why
    assert sent == [], "an unsafe frame reached the socket"
    ok, why = w.publish(s0, w.ready_frame, None, extra={"flore_state": "hold"})
    assert ok and why == ""
    assert sent[0][3] == {"flore_state": "hold"}


def test_a_wrong_width_ready_frame_is_refused_at_construction():
    with pytest.raises(ValueError, match="3pt ready frame"):
        hl_offline.ThreePointWire(np.zeros(15), np.zeros(20))


def test_reset_returns_the_clutch_to_the_ready_frame():
    w = _wire(post=PostAnchorConfig(mode="command"))
    w.commit((_READY_POS + 0.05, _READY_ORN))
    assert w.clutch.drift_m() > 0.04
    w.clutch.reset()
    assert w.clutch.drift_m() == pytest.approx(0.0)


def test_status_reports_the_gap_against_ready():
    w = _wire()
    pos = _READY_POS.copy()
    pos[0] += 0.123
    assert "123 mm" in w.status((pos, _READY_ORN))


# ---------------------------------------------------------------------------
# the decode contract the 3pt driver rests on
# ---------------------------------------------------------------------------
def test_a_zero_relative_chunk_decodes_ONTO_the_anchor():
    """Why the 3pt path needs no wrist->contact lever of its own.

    ``decode_chunk`` is ``p_abs = R_anchor @ p_rel + p_anchor``, so a chunk whose row 0 is zero —
    which is what a t0-inclusive grid produces, the observation frame being step 0 — lands exactly
    on whatever it was decoded against. Decode against ``resolve_deploy_anchor`` and step 0 IS the
    ready pose, which IS the pre-arm gate's reference. Applying a lever on top would double it.
    """
    from flore.s1_policy.serving.hl_online import decode_chunk
    from flore.utils.rot6d import rotmat_to_rot6d

    layout = get_layout("both_hands_head_dual")
    roles = tuple(layout.roles)
    assert set(roles) >= {"left", "right", "head"}
    anchor_pos = np.stack([_READY_POS.reshape(3, 3)[i % 3] for i in range(len(roles))])
    anchor_rot6d = np.stack([rotmat_to_rot6d(np.eye(3)) for _ in roles])

    pred = np.zeros((4, layout.action_dim), dtype=np.float64)
    for role in roles:
        # A valid rot6d for every row: the first two ROWS of the identity.
        pred[:, layout.slots(role).rot6d] = rotmat_to_rot6d(np.eye(3))
    chunk = decode_chunk(pred, layout, anchor_pos, anchor_rot6d,
                         force_enabled=False, gate_n=1.5)
    for ri, rc in enumerate(chunk.roles):
        assert rc.position == pytest.approx(np.tile(anchor_pos[ri], (4, 1)))


def test_the_three_wholebody_and_keypoint_families_map_to_distinct_wires():
    """A layout->family mapping that ever collapsed would publish a 43-D chunk as keypoints (or
    the reverse), which the low level answers by dropping every frame — i.e. silently."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "_hl_offline_drive", pathlib.Path(__file__).resolve().parents[1]
        / "scripts" / "hl_offline_drive.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.FAMILY_BY_LAYOUT["wholebody"] == "dense"
    assert mod.FAMILY_BY_LAYOUT["both_hands_head_dual"] == "3pt"
    assert mod.FAMILY_BY_LAYOUT["both_hands_head"] == "3pt"
    assert set(mod.FAMILY_BY_LAYOUT.values()) <= set(mod.FAMILIES)
    # left_only / both_hands are DELIBERATELY absent: they predict a subset of the three points,
    # so a 3pt wire built from them would publish a stale home for the rest.
    assert "left_only" not in mod.FAMILY_BY_LAYOUT
    assert "both_hands" not in mod.FAMILY_BY_LAYOUT
    # And the geom family, which no deploy path builds the observation for yet.
    assert not any(k.startswith("wholebody_geom") for k in mod.FAMILY_BY_LAYOUT)


def test_post_anchor_command_ACCUMULATES_across_chunks_and_never_reads_the_ready_pose_again():
    """The property ``ANCHOR=command`` exists for: chunk N+1's deltas are added to the last
    EXECUTED step of chunk N, not to the ready pose and not to any measured value.

    With ``off`` every chunk restarts from the ready frame, so a two-chunk reach would publish the
    same absolute poses twice and the robot would never get anywhere; the difference is invisible
    in one chunk, which is why it is pinned over two.
    """
    w = _wire(post=PostAnchorConfig(mode="command"))
    orn = np.tile(_READY_ORN, (4, 1))

    # CHUNK 1: 3 cm of +x on the left hand. Step 0 lands on the ready frame (bit-exact no-op).
    pos = np.tile(_READY_POS, (4, 1))
    pos[:, 0] += np.arange(4) * 0.01
    p1 = w.plan(pos, orn)
    assert p1["pos"][0] == pytest.approx(_READY_POS)
    # EXECUTED to step 2 (2 cm), not to the chunk end: an interrupted chunk's tail was never asked
    # for, which is what makes this an integrator over commands rather than over predictions.
    w.commit(w.frame(p1, 2))

    # CHUNK 2: the SAME relative motion again. It must now start where chunk 1 stopped.
    p2 = w.plan(pos, orn)
    assert p2["pos"][0][0] == pytest.approx(_READY_POS[0] + 0.02), "the base did not carry"
    assert p2["pos"][3][0] == pytest.approx(_READY_POS[0] + 0.05)
    # Nothing measured was consulted: sync with no readback holds the carry rather than clamping.
    assert "FROZEN" in w.sync(None, None)
    assert w.clutch.drift_m() == pytest.approx(0.02)

    # ... and `off` is the control: the same two chunks publish the same absolute poses.
    w_off = _wire()
    q1 = w_off.plan(pos, orn)
    w_off.commit(w_off.frame(q1, 2))
    w_off.sync(None, None)
    q2 = w_off.plan(pos, orn)
    assert q2["pos"][0] == pytest.approx(q1["pos"][0])


# ---------------------------------------------------------------------------
# the jaws: an OUT-OF-BAND channel, not the :8770 wire
# ---------------------------------------------------------------------------
class _FakeSub:
    """Stands in for ``hl_serving._WSSubscriber``: records what would go to a jaw bridge."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def send(self, obj: dict) -> None:
        self.sent.append(dict(obj))


def _jaws(**kw):
    """A ``_Jaws`` over fake bridges, so nothing touches a socket."""
    import argparse
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "_hl_offline_drive_jaws", pathlib.Path(__file__).resolve().parents[1]
        / "scripts" / "hl_offline_drive.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    args = argparse.Namespace(
        gripper_ws="ws://127.0.0.1:8784", gripper_right_ws="ws://127.0.0.1:8785",
        grip_overshoot_mm=kw.pop("overshoot_mm", 10.0),
        grip_close_below_mm=kw.pop("close_below_mm", 70.0),
        grip_confirm_steps=kw.pop("confirm_steps", 1))
    j = mod._Jaws(args, kw.pop("roles", ("left", "right")))
    j.subs = {r: _FakeSub() for r in j.roles}
    return j


def test_a_jaw_command_is_one_target_frac_per_hand():
    """One send per hand per action step, and the value is the CALIBRATION's fraction of travel
    (90 mm of a 0-95 mm span = 0.947), not the width in metres."""
    j = _jaws()
    j.command({"left": 0.090, "right": 0.090})
    for sub in j.subs.values():
        assert len(sub.sent) == 1
        assert sub.sent[0]["target_frac"] == pytest.approx(0.090 / 0.095, abs=1e-6)


def test_an_absent_or_nonfinite_width_is_SKIPPED_not_published_as_zero():
    """0.0 is a legal fully-closed target, so a fabricated value would be invisible in the
    numbers — and a masked-out gripper column is exactly the case that produces one."""
    j = _jaws()
    j.command({"left": float("nan")})              # present but not measured
    j.command({})                                  # absent entirely
    assert all(not s.sent for s in j.subs.values())


def test_each_hand_has_its_OWN_confirm_counter():
    """``build_grip_command``'s run-length state lives in the dict it is handed, so a shared one
    lets the LEFT jaw's sustained close engage the RIGHT jaw's overshoot."""
    j = _jaws(confirm_steps=3)
    for _ in range(5):
        j.command({"left": 0.045})                 # LEFT closes, sustained
    j.command({"right": 0.045})                    # RIGHT's FIRST close
    left, right = j.last["left"], j.last["right"]
    assert left < right, ("the right jaw inherited the left's engaged overshoot "
                          f"(left {left:.3f} vs right {right:.3f})")


def test_open_all_is_inert_until_something_has_been_commanded():
    """With the switch off the jaws were never driven; forcing them open on a Reset would move
    hardware this process never asked to move."""
    j = _jaws()
    j.open_all("reset")
    assert all(not s.sent for s in j.subs.values())
    j.command({"left": 0.045, "right": 0.045})
    j.open_all("reset")
    assert all(s.sent[-1] == {"target_frac": 1.0} for s in j.subs.values())


def test_a_role_with_no_bridge_url_is_refused_by_name():
    with pytest.raises(SystemExit, match="head"):
        _jaws(roles=("left", "head"))
