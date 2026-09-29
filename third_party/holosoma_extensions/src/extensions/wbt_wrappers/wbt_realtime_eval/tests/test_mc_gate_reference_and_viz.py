"""The pose-match viewer's two halves: the reference dump (hssim) and the gap arithmetic.

The viewer itself needs viser, which this env does not have — but everything that can be
WRONG in it is pure numpy, so those functions are imported by path and tested here. What is
deliberately NOT tested is the drawing; what IS tested is that the numbers it draws are the
gate's numbers.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[4].parents[0]
_VIVE = Path(__file__).resolve().parents[1] / "wbt_realtime_eval" / "vive"


def _load_viz():
    """Import the viewer BY PATH.

    It is written to be runnable as a script in UMI's `.venv` (its package `__init__` pulls
    in mjswan + holosoma, which are not there), so a path import is also how a test must
    reach it — and doing it this way means the test fails if someone adds a package-level
    import that would break the only interpreter that can actually render.
    """
    spec = importlib.util.spec_from_file_location("_mc_viz", _VIVE / "mc_teleop_gate_viz.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------

def test_gate_reference_matches_the_policys_own_fk(monkeypatch) -> None:
    """THE point of the module: the drawn target must be the GATED target.

    `mc_gate_reference` drives `HolosomaWBTPolicy._compute_vr_fk` rather than restating the
    FK, so this asserts the thing that would break if someone "simplified" it into a local
    copy — a viewer that shows a plausible pose the gate disagrees with is worse than no
    viewer, because the operator would trust it.
    """
    monkeypatch.setenv("HOLOSOMA_INIT_READY_POSE", "1")
    from wbt_realtime_eval.vive import mc_gate_reference as gr

    q, names = gr.ready_pose_qpos()
    kp, bodies = gr.reference_keypoints(q, "5pt")
    assert kp.shape == (5, 3), kp.shape

    # Independently: build the policy's FK the way the POLICY does and compare.
    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import (
        _VR_5POINT_OFFSETS_TCP,
        HolosomaWBTPolicy,
    )

    p = object.__new__(HolosomaWBTPolicy)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, gr._holosoma_yam_urdf().read_text())
    p._vr_frame_ids = [p.pinocchio_robot.robot_model.getFrameId(b) for b in bodies]
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._vr_orn_6d = False
    want, _ = p._compute_vr_fk(q.astype(np.float32), _VR_5POINT_OFFSETS_TCP)
    np.testing.assert_allclose(kp.reshape(-1), np.asarray(want, dtype=np.float64), atol=1e-9)


def test_reference_is_pelvis_local_so_the_anchor_cancels(monkeypatch) -> None:
    """The dump claims the base pose is irrelevant because the readout is pelvis-LOCAL. If
    that were false the drawn target would move with wherever the robot happens to stand."""
    monkeypatch.setenv("HOLOSOMA_INIT_READY_POSE", "1")
    from wbt_realtime_eval.vive import mc_gate_reference as gr

    q, _ = gr.ready_pose_qpos()
    a, _ = gr.reference_keypoints(q, "5pt")

    import wbt_realtime_eval.vive.mc_gate_reference as _gr_mod

    real = _gr_mod.reference_keypoints
    # Re-run with a different base height/heading by patching what the helper installs.
    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_BODIES, g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import (
        _VR_5POINT_OFFSETS_TCP,
        HolosomaWBTPolicy,
    )

    p = object.__new__(HolosomaWBTPolicy)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, gr._holosoma_yam_urdf().read_text())
    p._vr_frame_ids = [p.pinocchio_robot.robot_model.getFrameId(b) for b in VR_5POINT_BODIES]
    p._ref_pos_xyz_t = np.array([[3.0, -2.0, 1.11]], dtype=np.float32)   # elsewhere entirely
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.7071068, 0.7071068]], dtype=np.float32)  # +90° yaw
    p._vr_orn_6d = False
    moved, _ = p._compute_vr_fk(q.astype(np.float32), _VR_5POINT_OFFSETS_TCP)
    np.testing.assert_allclose(a.reshape(-1), np.asarray(moved, dtype=np.float64), atol=1e-6)
    assert real is _gr_mod.reference_keypoints  # nothing monkeypatched away


def test_ready_pose_zeroes_exactly_the_arm_joints_and_nothing_else(monkeypatch) -> None:
    """The reference is `default_dof_angles` with the 14 ARM joints zeroed. If it silently
    zeroed a leg too, the drawn ghost would stand differently from the robot."""
    monkeypatch.setenv("HOLOSOMA_INIT_READY_POSE", "1")
    from wbt_realtime_eval.vive import mc_gate_reference as gr

    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    q, names = gr.ready_pose_qpos()
    default = np.asarray(g1_29dof_wbt.default_dof_angles, dtype=np.float64).reshape(-1)
    arms = set(HolosomaWBTPolicy._READY_POSE_ARM_JOINTS)
    assert len(arms) == 14, len(arms)
    for i, n in enumerate(names):
        if n in arms:
            assert q[i] == 0.0, n
        else:
            assert q[i] == default[i], n
    # And the crouch that makes the FEET ungateable is really in there.
    assert q[names.index("left_knee_joint")] == pytest.approx(0.669, abs=1e-3)


def test_dump_refuses_when_the_hold_is_not_the_ready_pose(monkeypatch) -> None:
    """With HOLOSOMA_INIT_READY_POSE=0 the policy holds the CLIP's frame 0 instead, so a
    ready-pose drawing would be a confidently wrong target. Refuse, don't guess."""
    monkeypatch.setenv("HOLOSOMA_INIT_READY_POSE", "0")
    from wbt_realtime_eval.vive import mc_gate_reference as gr

    assert gr.ready_pose_enabled() is False
    with pytest.raises(SystemExit) as ei:
        gr.build("5pt")
    assert "frame 0" in str(ei.value)


def test_dump_marks_only_the_gated_points_and_joints(monkeypatch) -> None:
    """The viewer draws gated vs derived differently, so the dump must agree with the gate's
    own split rather than carrying a second hardcoded count."""
    monkeypatch.setenv("HOLOSOMA_INIT_READY_POSE", "1")
    from wbt_realtime_eval.vive import mc_gate_reference as gr

    from wbt_wrappers_inference.policies.holosoma_wbt import (
        _VR_5POINT_OPERATOR_PLACED,
        _VR_DENSE_DERIVED_JOINT_KEYS,
    )

    five = gr.build("5pt")
    assert five["n_gated_points"] == _VR_5POINT_OPERATOR_PLACED
    assert len(five["keypoints"]) == 5
    assert five["point_labels"][:3] == ["LEFT hand", "RIGHT hand", "head/torso"]
    assert "foot" in five["point_labels"][3].lower()

    dense = gr.build("dense")
    assert dense["keypoints"] == []          # no keypoints exist for dense; don't invent any
    gated = set(dense["gated_joints"])
    assert "left_shoulder_pitch_joint" in gated
    for j in ("left_knee_joint", "right_hip_yaw_joint", "left_ankle_roll_joint", "waist_pitch_joint"):
        assert j not in gated, j
    assert all(not any(k in j for k in _VR_DENSE_DERIVED_JOINT_KEYS) for j in gated)

    # 3pt gates every point (one puck each) — the exemption is 5pt-specific.
    three = gr.build("3pt")
    assert three["n_gated_points"] == len(three["keypoints"]) == 3


def test_dump_round_trips_through_json(monkeypatch, tmp_path) -> None:
    """The viewer reads this over a process boundary, so it must be JSON-clean — a stray
    numpy scalar raises inside json.dumps and only at launch time."""
    monkeypatch.setenv("HOLOSOMA_INIT_READY_POSE", "1")
    from wbt_realtime_eval.vive import mc_gate_reference as gr

    doc = gr.build("5pt")
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(doc))
    back = json.loads(p.read_text())
    assert back["n_gated_points"] == doc["n_gated_points"]
    np.testing.assert_allclose(back["keypoints"], doc["keypoints"])


# ---------------------------------------------------------------------------
# the viewer's arithmetic
# ---------------------------------------------------------------------------

def test_viz_verdict_ignores_the_derived_points_like_the_gate_does() -> None:
    """The viewer must not say REFUSE when the policy would ARM (or vice versa) — an operator
    chasing a green panel while the robot refuses is the failure this whole change is about."""
    viz = _load_viz()
    # The rig's measured numbers: hands+head fine, feet out.
    gaps = np.array([0.030, 0.050, 0.020, 0.101, 0.062])
    ok, over = viz.verdict(gaps, n_gated=3, tol=0.07)
    assert ok is True and over == []
    # A gated point out -> refuse, and it is named by index.
    gaps[0] = 0.15
    ok, over = viz.verdict(gaps, n_gated=3, tol=0.07)
    assert ok is False and over == [0]


def test_viz_verdict_agrees_with_the_policy_gate_on_random_frames() -> None:
    """Cross-check the mirror against the real thing rather than trusting two readings of the
    same rule. Both must answer identically for every frame."""
    viz = _load_viz()
    rng = np.random.default_rng(0)
    tol = 0.07
    for _ in range(200):
        gaps = rng.uniform(0.0, 0.25, size=5)
        mine, _ = viz.verdict(gaps, n_gated=3, tol=tol)
        theirs = bool(np.all(gaps[:3] <= tol))  # exactly holosoma_wbt's expression
        assert mine == theirs, gaps


def test_viz_keypoint_gaps_tolerate_a_short_frame() -> None:
    """A 3-tracker frame reaching a 5pt reference must read as 3 gaps, not an index error —
    the same min-in-whole-3-vectors rule the gate uses."""
    viz = _load_viz()
    ref = np.arange(15, dtype=float).reshape(5, 3)
    live = ref[:3] + 0.1
    g = viz.keypoint_gaps(live.reshape(-1), ref)
    assert g.shape == (3,)
    np.testing.assert_allclose(g, np.full(3, np.sqrt(3) * 0.1))


def test_viz_joint_gaps_remap_by_name_never_by_index() -> None:
    """MC's joint order and the policy's are both real orders and are NOT the same. A
    REVERSED live order is the only input that tells a name map from an index map."""
    viz = _load_viz()
    ref_names = [f"j{i}" for i in range(5)]
    ref_q = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    live_names = list(reversed(ref_names))
    live_q = np.array([4.0, 3.0, 2.0, 1.0, 0.0])  # same VALUES per name
    gaps, names = viz.joint_gaps(live_q, live_names, ref_q, ref_names)
    assert names == ref_names
    np.testing.assert_allclose(gaps, np.zeros(5)), "a name map sees zero gap; an index map would not"
    # A joint the live frame omits is not evidence of a mismatch.
    gaps2, _ = viz.joint_gaps([0.0], ["j0"], ref_q, ref_names)
    np.testing.assert_allclose(gaps2, np.zeros(5))


# ---------------------------------------------------------------------------
# the wire flag
# ---------------------------------------------------------------------------

def test_publish_mc_qpos_is_a_real_flag_on_the_bridges_own_parser(monkeypatch, capsys) -> None:
    """Drive the REAL parser via `--help` rather than inventing an accessor for it.

    The parser is built inline in `main()`, so `--help` is the only way to reach it without
    refactoring production code for a test's convenience. `pysurvive` is injected by the
    launcher via PYTHONPATH, not pip, so it is stubbed exactly as the station suite does.
    """
    import sys as _sys
    import types as _types

    if "pysurvive" not in _sys.modules:
        m = _types.ModuleType("pysurvive")
        m.SimpleContext = object
        m.survive_simple_serial_number = lambda *_a, **_k: b"?"
        _sys.modules["pysurvive"] = m
    from wbt_realtime_eval.vive import vive_teleop_bridge as vtb

    monkeypatch.setattr(_sys, "argv", ["vive_teleop_bridge", "--help"])
    with pytest.raises(SystemExit) as ei:
        vtb.main()
    assert ei.value.code == 0
    import re as _re

    # argparse RE-WRAPS help, so a phrase can be split across lines ("OFF by\ndefault").
    # Flatten before matching, the same way the station suite's `_flat()` does.
    helptext = _re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "--publish-mc-qpos" in helptext, "the flag is not on the bridge's parser"
    # store_true => default False => the 5pt payload is byte-identical unless asked.
    assert "mc_qpos" in helptext and "OFF by default" in helptext


def _bridge_stub(mode: str, *, publish: bool):
    """A bridge with just enough state for `_mc_snapshot`, and a fake completion.

    `object.__new__` because the real `__init__` claims tracker serials and spawns the MC
    backend — neither of which a payload-assembly test has any business doing.
    """
    import sys as _sys
    import types as _types

    if "pysurvive" not in _sys.modules:
        m = _types.ModuleType("pysurvive")
        m.SimpleContext = object
        m.survive_simple_serial_number = lambda *_a, **_k: b"?"
        _sys.modules["pysurvive"] = m
    from wbt_realtime_eval.vive import vive_teleop_bridge as vtb

    class _FakeMC:
        def reset(self):  # pragma: no cover - not reached in these cases
            pass

        def push(self, *_a, **_k):
            pass

        def latest(self, now=None):
            return {
                "pos": list(range(15)),
                "orn": list(range(20)),
                "names": [f"b{i}" for i in range(5)],
                "joint_pos": [0.1 * i for i in range(29)],
                "joint_names": [f"j{i}" for i in range(29)],
                "joint_vel": [0.0] * 29,
                "root_pos": [0.11, 0.22, 0.33],
                "root_quat": [1.0, 0.0, 0.0, 0.0],
                "frame": 42,
                "lag_s": 0.667,
            }

    b = object.__new__(vtb.ViveTeleopBridge) if hasattr(vtb, "ViveTeleopBridge") else None
    if b is None:  # class name differs -> find the one owning _mc_snapshot
        cands = [
            v for v in vars(vtb).values()
            if isinstance(v, type) and hasattr(v, "_mc_snapshot")
        ]
        assert cands, "no class on the bridge module owns _mc_snapshot"
        b = object.__new__(cands[0])
    b._mc = _FakeMC()
    b._mc_needs_reset = False
    b._seq = 0
    b._mode = mode
    b._args = _types.SimpleNamespace(publish_mc_qpos=publish)
    return b


#: What `--publish-mc-qpos` adds, in EITHER mode. The root pair is not decoration: without it
#: a viewer can only flex the joints in place, so a completion that takes a STEP draws as one
#: that stands still — which is the single thing the ghost exists to show.
_GHOST_KEYS = {"mc_qpos", "mc_joint_names", "mc_root_pos", "mc_root_quat"}


@pytest.mark.parametrize("mode", ["5pt", "dense"])
def test_publish_mc_qpos_adds_exactly_the_ghost_keys_and_nothing_else(mode: str) -> None:
    """The flag must be purely ADDITIVE: off -> the payload the policy has always seen; on ->
    the same payload plus the four ghost keys. Anything else is a wire change.

    BOTH modes, deliberately: the ghost draws the completion, and a dense policy is if
    anything more interesting to compare against, since a joint-space command cannot express
    a step at all. dense's `mc_qpos` therefore duplicates its own `joint_pos` — see
    `wire_payload` for why one uniformly-named ghost array beats a mode-dependent one.
    """
    off = _bridge_stub(mode, publish=False)._mc_snapshot(1.0, (0.0, None, None, None), True)
    on = _bridge_stub(mode, publish=True)._mc_snapshot(1.0, (0.0, None, None, None), True)
    assert set(on) - set(off) == _GHOST_KEYS
    assert set(off) - set(on) == set()
    for k in set(off):
        assert off[k] == on[k], f"{k} changed when the flag was turned on"
    assert len(on["mc_qpos"]) == 29 and len(on["mc_joint_names"]) == 29
    assert len(on["mc_root_pos"]) == 3 and len(on["mc_root_quat"]) == 4


def test_the_flag_stays_off_by_default_so_the_payload_is_unchanged() -> None:
    """The guarantee the flag was introduced with: an existing consumer sees no new bytes."""
    for mode in ("5pt", "dense"):
        off = _bridge_stub(mode, publish=False)._mc_snapshot(1.0, (0.0, None, None, None), True)
        assert not (set(off) & _GHOST_KEYS), f"{mode} leaks ghost keys with the flag off"


def test_dense_keeps_its_own_command_arrays_alongside_the_ghost_ones() -> None:
    """The ghost array must never REPLACE the dense command array. They mean different things
    ("what the ghost draws" vs "what the policy is commanded") and conflating them is exactly
    the confusion the ghost exists to dispel — so both spellings must be present."""
    on = _bridge_stub("dense", publish=True)._mc_snapshot(1.0, (0.0, None, None, None), True)
    assert "joint_pos" in on and "joint_vel" in on and "mc_qpos" in on
    assert on["mc_qpos"] == on["joint_pos"]


def test_five_point_payload_carries_the_whole_body_for_the_viewer() -> None:
    """Both readouts must hand back the 29-DoF body the command was sliced off — joints, their
    names, AND the root that places them — or the viewer can only draw spheres. Names ride
    EVERY frame so a late-joining consumer cannot index-align by accident."""
    src = Path(_VIVE / "mc_teleop_source.py").read_text()
    body = src.split("def _five_point_locked", 1)[1].split("def _dense_locked", 1)[0]
    assert '"joint_pos"' in body, "5pt readout no longer carries the joints"
    assert '"joint_names"' in body, "5pt readout no longer carries the joint NAMES"
    assert '"root_pos"' in body and '"root_quat"' in body, "5pt readout lost the ghost root"
    dense = src.split("def _dense_locked", 1)[1].split("\n    # -- telemetry", 1)[0]
    assert '"root_pos"' in dense and '"root_quat"' in dense, "dense readout lost the ghost root"
    # The bridge keeps the FLAG; the payload keys live in the shared `wire_payload`, which
    # the trackerless replay driver also calls (one definition of the wire format).
    bridge = Path(_VIVE / "vive_teleop_bridge.py").read_text()
    assert 'publish_qpos=bool(getattr(self._args, "publish_mc_qpos", False))' in bridge, \
        "flag guard gone"
    payload = src.split("def wire_payload", 1)[1].split("\nclass ", 1)[0]
    for k in sorted(_GHOST_KEYS):
        assert f'"{k}"' in payload, f"wire_payload no longer emits {k}"


def test_the_launcher_pins_viser_to_loopback() -> None:
    """The page carries live telemetry from a physical robot and has no auth. viser's OWN
    default is 0.0.0.0, so the default here must be overridden, not inherited."""
    viz = Path(_VIVE / "mc_teleop_gate_viz.py").read_text()
    assert 'default="127.0.0.1"' in viz, "viser bind is not pinned to loopback"
    assert "host=args.host" in viz, "the --host arg is parsed but not passed to ViserServer"
    sh = Path(_VIVE).parents[2].parents[1] / "force_deploy/4_pose_match_viz.sh"
    if sh.is_file():
        assert "mc_teleop_gate_viz.py" in sh.read_text(), "launcher must invoke the FILE, not -m"


def test_gap_lines_are_re_added_not_mutated() -> None:
    """viser's `LineSegmentsHandle` has NO settable `points` (only
    name/position/scale/visible/wxyz/remove). Assigning `handle.points = ...` therefore sets
    an unused Python attribute and raises NOTHING — the gap line freezes at its first
    position while the sphere it is supposed to connect keeps moving. A frozen line that
    looks live is the worst possible failure for a tool whose whole job is showing a
    distance, so the draw loop must RE-ADD under the same name.
    """
    src = Path(_VIVE / "mc_teleop_gate_viz.py").read_text()
    # Comment lines are skipped on purpose: the module DOCUMENTS the trap, and the
    # explanation quotes the bad assignment. Only real code must be clean.
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    offenders = [ln.strip() for ln in code if ".points = " in ln or ".points=" in ln]
    assert not offenders, (
        "LineSegmentsHandle has no settable `points` — assigning it silently freezes the "
        f"gap line. Re-add the node under the same name instead. Found: {offenders}"
    )
    # And the re-add must really be inside the per-frame loop, not just present somewhere.
    loop = src.split("while True:", 1)[1]
    assert "add_line_segments(" in loop, "gap lines are not re-added per frame"


def test_repo_paths_used_by_this_suite_exist() -> None:
    """Guards the path arithmetic above: a moved file would otherwise make the source-content
    assertions vacuously pass (`"x" in ""`) rather than fail."""
    assert (_VIVE / "mc_gate_reference.py").is_file()
    assert (_VIVE / "mc_teleop_gate_viz.py").is_file()
    assert (_VIVE / "mc_teleop_source.py").is_file()
    assert (_VIVE / "vive_teleop_bridge.py").is_file()
    assert os.path.isdir(_VIVE)


# ---------------------------------------------------------------------------
# the BROWSER-side completion ghost (5pt + dense)
#
# A light-purple translucent body holding the whole completion, drawn next to the solid robot
# the low-level controller is driving. It exists to separate two failures that look identical
# in the sim -- "the completion never asked for a step" vs "it asked and the controller did
# not take it" -- and the three things below are what keep it honest. All are source-level:
# the TS runs in a browser under WASM MuJoCo, so a unit test cannot execute it, but each of
# these was a real way to get it wrong and each is greppable.
# ---------------------------------------------------------------------------

_TEMPLATE = Path(__file__).resolve().parents[1] / "wbt_realtime_eval" / "template"


def _ghost_method(name: str) -> str:
    """One method body out of WbtTrackingCommand.ts, by name."""
    src = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    assert f"private {name}(" in src or f"{name}(" in src, f"{name} is gone from the term"
    return src.split(f"{name}(", 1)[1].split("\n  private ", 1)[0]


def test_the_ghost_never_writes_the_arrays_the_POLICY_reads() -> None:
    """THE invariant that keeps this a viewer instead of an intervention.

    The parent's ghost is posed from `refJointPos/refRootPos/refRootQuat[refIdx]` -- and those
    same three arrays ARE the policy's reference: `getCommand()` and every motion_ref_* obs
    read them. Moving the parent's ghost by overwriting them would silently change the
    policy's input, so that a tool built to explain the robot's behaviour would be causing it.
    Hence a SECOND ghost with its own MjData, and hence this test: not one assignment into
    those arrays anywhere in the term.
    """
    import re

    src = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    bad = re.findall(r"this\.ref(?:JointPos|RootPos|RootQuat)\s*(?:\[[^\]]*\])?\s*=[^=]", src)
    assert not bad, f"the term now WRITES the policy's reference arrays: {bad}"
    ghost = _ghost_method("updateMcGhost")
    assert "this.mcGhostData" in ghost, "the ghost no longer uses its own MjData"
    assert "refRootPos" in ghost and "refRootQuat" in ghost, \
        "the ghost must still READ the reference root -- it is the frame it composes onto"


def test_the_ghost_places_joints_BY_NAME_not_by_index() -> None:
    """MC's joint order and the render model's are both real orders and both 29 long, so an
    index-aligned handoff scrambles the ghost into a plausible-looking pose. Name-keyed extras
    make an unknown joint simply not move (loud) instead of taking someone else's angle."""
    obs = (_TEMPLATE / "WbtObservations.ts").read_text()
    cmd = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    assert "wbt_mcghost__q__${names[i]}" in obs, "the producer no longer keys joints by name"
    assert "wbt_mcghost__q__${j.name}" in cmd, "the consumer no longer reads joints by name"
    ghost = _ghost_method("initMcGhost")
    assert "jnt_qposadr" in ghost and "jnt(j).name" in ghost, \
        "the ghost must resolve qpos addresses from the RENDER model's own joint names"
    # A ball joint owns four qpos slots; a scalar angle written into the first is a corrupt
    # quaternion, so it must be skipped rather than indexed.
    assert "type === 1) continue" in ghost, "ball joints are no longer skipped"


def test_the_ghost_needs_the_root_and_says_so_when_the_flag_is_off() -> None:
    """Two failure modes that both draw a body and are both wrong.

    1. No root -> the joints flex in place, so the STEP (the entire point) is invisible. The
       four wire keys are therefore required TOGETHER; a fresh root on stale joints, or the
       reverse, is a pose that never existed and reads as a completion artifact.
    2. Bridge started without `--publish-mc-qpos` -> nothing arrives. That must be said out
       loud, naming the flag, exactly as vive/mc_teleop_gate_viz.py does -- and said from the
       store, the only place that can tell it apart from "this preset has no MC stream".
    """
    obs = (_TEMPLATE / "WbtObservations.ts").read_text()
    for k in ("mc_qpos", "mc_joint_names", "mc_root_pos", "mc_root_quat"):
        assert k in obs, f"the store no longer parses {k}"
    # One conjunctive guard, so the four cannot be latched independently.
    guard = obs.split("if (Array.isArray(msg.mc_qpos)", 1)[1].split(") {", 1)[0]
    for k in ("mc_joint_names", "mc_root_pos", "mc_root_quat"):
        assert k in guard, f"{k} is no longer required alongside mc_qpos"
    assert "--publish-mc-qpos" in obs, "the missing-flag warning no longer names the flag"
    # `__seq` is the consumer's gate and must be written LAST or a half-written body latches.
    pub = obs.split("private publishGhostExtras(", 1)[1].split("\n  private ", 1)[0]
    assert pub.index("wbt_mcghost__q__") < pub.index("'wbt_mcghost__seq'"), \
        "the seq gate must be written AFTER the body it gates"


def test_the_ghost_hides_when_the_stream_goes_stale() -> None:
    """THE teleop-specific failure. The operator is steering AT this mesh, so a frozen one
    turns their own tracking error into a phantom policy failure -- a dead bridge would read
    as "the policy stopped following". It must therefore go AWAY, not sit there looking
    authoritative, and it must use the browser's own monotonic clock: the bridge can be on a
    different host, and clock skew must not be able to blank (or un-blank) the target.
    """
    cmd = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    obs = (_TEMPLATE / "WbtObservations.ts").read_text()
    assert "MC_GHOST_STALE_MS = 500" in cmd, "the freshness budget is gone or changed silently"
    assert "wbt_mcghost__rx" in obs and "performance.now()" in obs, \
        "the producer no longer stamps a receive time"
    ghost = _ghost_method("updateMcGhost")
    assert "performance.now() - getPolicyExtra('wbt_mcghost__rx'" in ghost, \
        "the consumer no longer compares against its OWN clock"
    stale = ghost.split("MC_GHOST_STALE_MS)", 1)[1].split("return;", 1)[0]
    assert "root.visible = false" in stale, "a stale ghost is still drawn"


def test_a_degenerate_root_quaternion_cannot_reach_mujoco() -> None:
    """`Number(x) || 0` -- the pattern every other field on this wire uses -- turns a garbage
    quaternion into all-zeros, which is not a rotation. MuJoCo answers with NaN world poses,
    and a NaN written into a THREE transform poisons the whole subtree's matrix, so one bad
    frame reads as the SIM crashing. Normalized at the producer, identity if degenerate, and
    non-finite body poses are skipped at the consumer as a second line."""
    obs = (_TEMPLATE / "WbtObservations.ts").read_text()
    pub = obs.split("private publishGhostExtras(", 1)[1].split("\n  private ", 1)[0]
    assert "Math.hypot(qw, qx, qy, qz)" in pub, "the root quaternion is no longer normalized"
    assert "qw = 1.0" in pub, "a degenerate quaternion no longer falls back to identity"
    ghost = _ghost_method("updateMcGhost")
    assert "Number.isFinite(p.x)" in ghost, "non-finite body poses are no longer skipped"


def test_the_ghost_is_light_purple_and_not_confusable_with_what_shares_its_screen() -> None:
    """It is the GT the operator steers at, so its colour is load-bearing: it shares a screen
    with the parent ghost's pale GREEN, the solid robot, and the SATURATED purple F^cmd force
    arrow -- a different quantity entirely. Pinned as one named constant so the material and
    any future legend cannot drift apart."""
    import re

    cmd = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    m = re.search(r"MC_GHOST_RGB[^=]*=\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]", cmd)
    assert m, "MC_GHOST_RGB is gone -- the ghost colour is no longer a named constant"
    r, g, b = (float(x) for x in m.groups())
    assert b > r > g, f"({r},{g},{b}) is not a purple (want blue > red > green)"
    assert min(r, g, b) > 0.4, f"({r},{g},{b}) is not LIGHT -- it must read as pale lavender"
    # Distinct from the saturated F^cmd arrow purple, which is a force and not a pose.
    main_py = (Path(__file__).resolve().parents[1] / "wbt_realtime_eval" / "main.py").read_text()
    arrow = re.search(r"_ARROW_RGB_CMD\s*=\s*\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]", main_py)
    assert arrow, "the F^cmd arrow colour moved; re-check the ghost is still distinct from it"
    ar, ag, ab = (float(x) for x in arrow.groups())
    assert abs(r - ar) + abs(g - ag) + abs(b - ab) > 0.3, \
        "the ghost is too close to the F^cmd arrow's purple to tell apart"
    # And the material really uses the constant rather than a second hardcoded triple.
    assert "MC_GHOST_RGB[0]" in cmd, "the material no longer reads the named constant"


def test_the_ghost_hides_with_the_reference_it_is_and_frees_its_wasm_data() -> None:
    """It IS the reference, so it hides with the existing reference toggle -- an operator who
    hides one ghost and keeps the other reads the leftover as policy behaviour. And its MjData
    is a WASM allocation: a live preset switch that leaks it leaks a whole model."""
    ghost = _ghost_method("updateMcGhost")
    assert "this.referenceVisible" in ghost, "the ghost no longer honours the reference toggle"
    disp = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    disp = disp.split("override dispose()", 1)[1].split("override async setSelectedMotion", 1)[0]
    assert "this.mcGhostData?.delete?.()" in disp, "the ghost's MjData is leaked on dispose"
    assert "this.mcGhostRoot" in disp, "the ghost meshes are left in the scene on dispose"
