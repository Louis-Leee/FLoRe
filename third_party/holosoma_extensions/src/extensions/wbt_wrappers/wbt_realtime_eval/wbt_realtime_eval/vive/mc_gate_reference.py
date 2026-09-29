"""The pose the ABSOLUTE pre-arm gate wants, exported so an operator can SEE it.

WHY THIS EXISTS
===============
`holosoma_wbt._vive_arm_pose_ok` refuses to arm ABSOLUTE teleop until the operator's live
points are within tolerance of the robot's HELD REFERENCE. Neither quantity is visible
anywhere: the refusal printed distances, so an operator could tell they were far without
ever learning *from what*. On the rig that cost a whole session — the reference turned out
to be 0.44 m STRAIGHT AHEAD of the pelvis and 0.10 m above it (forearms horizontal), while
the operator was holding the pucks up at chest height, 0.9 m away.

So this module answers exactly one question — "what am I supposed to match?" — as data, for
`vive/mc_teleop_gate_viz.py` to draw.

WHY IT IS A SEPARATE PROCESS FROM THE VIEWER
===========================================
Two envs, and neither can be merged into the other:

  * the reference is holosoma's (``default_dof_angles``, ``_READY_POSE_ARM_JOINTS``,
    ``VR_5POINT_*``, and the FK itself) so it can only be computed in **hssim**;
  * **hssim has no viser** and it is the interpreter that flies a physical G1 — adding a
    render stack to it is not a trade worth making.

The reference is STATIC (it is a fixed joint vector, not a stream), so the boundary costs
nothing: dump once to JSON, draw from the JSON. That is why this is a dump and not a
service.

THE FK IS NOT REIMPLEMENTED HERE
================================
``reference_keypoints`` calls ``HolosomaWBTPolicy._compute_vr_fk`` — the *same* method the
gate calls — on a minimally-populated instance. A second copy of "offset in the body frame,
then into the pelvis frame" is precisely the thing that drifts silently: it would render a
plausible target that the gate disagrees with, which is worse than showing nothing.
``test_gate_reference_matches_the_policys_own_fk`` pins the equality, and building the
instance by hand means a future refactor that adds a dependency to ``_compute_vr_fk``
fails HERE (loudly, in a test) instead of on the rig.

Usage (hssim)::

    python -m wbt_realtime_eval.vive.mc_gate_reference --mode 5pt --out /tmp/gate_ref.json

``HOLOSOMA_INIT_READY_POSE`` is honoured exactly as the policy honours it, because it is
what SELECTS the reference: on (the launcher's default) the hold is the ready pose; off, the
hold is the motion clip's frame 0 and this module refuses rather than drawing a pose the
policy is not holding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

#: Modes this can describe. `dense` has no keypoints — its gate is joint-space — so the
#: dump carries the joint vector and an empty keypoint list rather than pretending.
MODES = ("5pt", "3pt", "2pt", "dense")


def _holosoma_yam_urdf() -> Path:
    """Locate the yam URDF through the INSTALLED holosoma package.

    Not from the ONNX's ``robot_urdf_path`` metadata: that records the training host's path
    (a path on the training cluster) and does not exist here. Not from a hardcoded sibling
    checkout either — the hssim env's editable install resolves holosoma to a *different*
    clone than this one, and the URDF must come from whichever clone the POLICY imports, or
    the drawn reference is a different robot from the gated one.
    """
    import holosoma

    p = Path(holosoma.__file__).parent / "data" / "robots" / "g1" / "g1_29dof_yam.urdf"
    if not p.is_file():
        raise FileNotFoundError(f"yam URDF not found under the imported holosoma package: {p}")
    return p


def ready_pose_enabled() -> bool:
    """Whether the policy's frozen hold is the READY POSE (vs the clip's frame 0).

    Mirrors `_maybe_apply_init_ready_pose`'s env parsing. The `auto` case there resolves
    from `headless_sim_eval`, which is a task-config field this module has no access to;
    on the rig the launcher sets the var explicitly, and auto means "on for a real robot",
    so auto is treated as ON and said out loud in the dump.
    """
    env = os.environ.get("HOLOSOMA_INIT_READY_POSE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    return True  # explicit on, or `auto` -> on for the real-robot path


def ready_pose_qpos() -> tuple[np.ndarray, list[str]]:
    """(29 joint angles, dof_names) for the ready pose the frozen hold uses.

    Built the way `_maybe_apply_init_ready_pose` builds it — `default_dof_angles` with the
    14 arm joints zeroed, resolved BY NAME — and reading the SAME two constants, so it
    cannot describe a pose the policy does not hold.
    """
    from wbt_wrappers_inference.config_values.robot_config import g1_29dof_wbt
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    names = list(g1_29dof_wbt.dof_names)
    q = np.asarray(g1_29dof_wbt.default_dof_angles, dtype=np.float64).reshape(-1).copy()
    if q.shape[0] != len(names):
        raise ValueError(f"default_dof_angles is {q.shape[0]} long but dof_names is {len(names)}")
    for n in HolosomaWBTPolicy._READY_POSE_ARM_JOINTS:
        q[names.index(n)] = 0.0
    return q, names


def reference_keypoints(qpos, mode: str = "5pt") -> tuple[np.ndarray, list[str]]:
    """(N,3) pelvis-frame keypoints for ``qpos``, via the POLICY's own FK.

    See the module docstring: this deliberately drives `HolosomaWBTPolicy._compute_vr_fk`
    rather than restating it, so the drawn target and the gated target are the same number
    by construction.
    """
    from holosoma_inference.policies.wbt_utils import PinocchioRobot

    from wbt_wrappers_inference.config_values.robot_config import (
        VR_2POINT_BODIES,
        VR_3POINT_BODIES,
        VR_5POINT_BODIES,
        g1_29dof_wbt,
    )
    from wbt_wrappers_inference.policies.holosoma_wbt import (
        _VR_2POINT_OFFSETS,
        _VR_3POINT_OFFSETS_TCP,
        _VR_5POINT_OFFSETS_TCP,
        HolosomaWBTPolicy,
    )

    bodies, offsets = {
        "5pt": (VR_5POINT_BODIES, _VR_5POINT_OFFSETS_TCP),
        "3pt": (VR_3POINT_BODIES, _VR_3POINT_OFFSETS_TCP),
        "2pt": (VR_2POINT_BODIES, _VR_2POINT_OFFSETS),
    }[mode]

    p = object.__new__(HolosomaWBTPolicy)
    p.pinocchio_robot = PinocchioRobot(g1_29dof_wbt, _holosoma_yam_urdf().read_text())
    model = p.pinocchio_robot.robot_model
    p._vr_frame_ids = [model.getFrameId(b) for b in bodies]
    # The anchor cancels out of a pelvis-LOCAL readout (the FK subtracts it again), so any
    # upright base works; 0.78 m is only there to keep the numbers readable if printed.
    p._ref_pos_xyz_t = np.array([[0.0, 0.0, 0.78]], dtype=np.float32)
    p._ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    p._vr_orn_6d = False

    pos, _ = p._compute_vr_fk(np.asarray(qpos, dtype=np.float32).reshape(-1), offsets)
    return np.asarray(pos, dtype=np.float64).reshape(-1, 3), list(bodies)


def build(mode: str = "5pt") -> dict:
    """The whole dump as a plain dict (JSON-ready)."""
    from wbt_wrappers_inference.policies.holosoma_wbt import (
        _VR_5POINT_OPERATOR_PLACED,
        _VR_DENSE_DERIVED_JOINT_KEYS,
        _VR_POINT_LABELS,
    )

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not ready_pose_enabled():
        raise SystemExit(
            "HOLOSOMA_INIT_READY_POSE is OFF, so the policy's frozen hold is the motion "
            "clip's frame 0, not the ready pose. This dump would draw a target the policy "
            "is not holding, which is worse than drawing nothing. Re-run the policy with "
            "HOLOSOMA_INIT_READY_POSE=1 (the launcher's default), or read the hold off the "
            "clip yourself."
        )

    q, names = ready_pose_qpos()
    out: dict = {
        "mode": mode,
        "source": "ready_pose (default_dof_angles with the 14 arm joints zeroed)",
        "dof_names": names,
        "qpos": q.tolist(),
        "arm_tol_m": float(os.environ.get("HOLOSOMA_VIVE_ARM_TOL", "0.07")),
        "arm_tol_rad": float(os.environ.get("HOLOSOMA_VIVE_ARM_TOL_RAD", "0.20")),
    }
    if mode == "dense":
        out["gated_joints"] = [
            n for n in names if not any(k in n for k in _VR_DENSE_DERIVED_JOINT_KEYS)
        ]
        out["keypoints"] = []
        out["point_labels"] = []
        out["n_gated_points"] = 0
        return out

    kp, bodies = reference_keypoints(q, mode)
    n_gate = _VR_5POINT_OPERATOR_PLACED if mode == "5pt" else kp.shape[0]
    out["keypoints"] = kp.tolist()
    out["keypoint_bodies"] = bodies
    out["point_labels"] = list(_VR_POINT_LABELS[: kp.shape[0]])
    out["n_gated_points"] = int(min(n_gate, kp.shape[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--mode", default="5pt", choices=MODES)
    ap.add_argument("--out", default="", help="write JSON here (default: stdout)")
    args = ap.parse_args()

    doc = build(args.mode)
    text = json.dumps(doc, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[gate-ref] wrote {args.out} (mode={args.mode})")
        for lbl, p in zip(doc.get("point_labels", []), doc.get("keypoints", [])):
            gated = doc["point_labels"].index(lbl) < doc["n_gated_points"]
            print(
                f"[gate-ref]   {lbl:12s} x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}"
                f"   {'GATED' if gated else 'not gated (completion-inferred)'}"
            )
    else:
        print(text)


if __name__ == "__main__":
    main()
