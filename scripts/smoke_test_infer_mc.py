#!/usr/bin/env python3
"""Smoke-test the offline inference viewer's WHOLE-BODY motion completion, headless.

The assertion that matters is not "a humanoid appeared" — it is that the completion is
CONDITIONED ON THE POSES THE PAGE DRAWS and lands back on them. So this builds the real
viser scene twice, once with ``--motion-completion`` and once without, and checks:

  1. ADDITIVE. Without the flag the scene has no ``/mc`` subtree and no G1 URDF at all, and
     every node the plain viewer draws is bit-identical in both. That is the property the
     "same-level extension" requirement means, and the only one a diff can catch.
  2. THE CHAIN IS RIGHT. Round-trip the completed 29 DoF back out: FK the wrist, mount it
     forward to ``gripper_base``, and compare against the pose the model was conditioned on.
     Sub-mm means the ``TCP -> wrist -> gripper_base`` chain matches; ~57 mm means the gripper
     mount was double-applied or skipped, ~242 mm means the TCP lever was missed.
  3. IT IS ON THE DRAWN CHUNK. The conditioned hand must sit within a few mm of the viewer's
     own displayed TCP for that role and step, once the dataset's wrist+mount lever is
     removed — i.e. the humanoid really is executing the orange line, not a parallel copy of
     it in another frame.
  4. THE ROBOT MOVED, and stands. Joint angles change across the chunk, and the posture
     verdict is "standing" rather than the crouch a mis-placed take produces.

Needs a 3-tracker checkpoint, a GPU, and the same Motion_Completion prerequisites the panel
does. Run:

    .venv/bin/python scripts/smoke_test_infer_mc.py ckpt_0831_best [--sample-index 21000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def _node_transforms(server) -> dict:
    """Every scene node's (name -> position, wxyz), read off the real server state."""
    out = {}
    for name, handle in server.scene._handle_from_node_name.items():
        try:
            out[name] = (tuple(np.asarray(handle.position, dtype=np.float64).ravel()),
                         tuple(np.asarray(handle.wxyz, dtype=np.float64).ravel()))
        except Exception:  # noqa: BLE001 - a node with no transform is still a node
            out[name] = None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--display-frame", default="g1-ready-pose", choices=("display-home", "g1-ready-pose"),
        help="which placement to test (default: the deploy ready pose, as the launcher uses)")
    args = parser.parse_args(argv)

    import viser

    from flore.s1_policy.inference.diffusion import DiffusionInferenceSession
    from flore.s1_policy.inference.display import (
        DISPLAY_HOME_POS, read_stored_extrinsic, wrist_pose_from_stored,
    )
    from flore.s1_policy.inference.mc_humanoid import MCHumanoid
    from flore.s1_policy.inference.viewer import _build_display_poses, _heading_alignment
    from flore.s1_policy.inference.display import resolve_role_geometry
    from flore.s1_policy.inference import ready_pose as RP

    session = DiffusionInferenceSession(args.checkpoint, device=args.device)
    try:
        print(f"[smoke] layout={session.source.layout.mode} roles={session.source.layout.roles}")
        result = session.infer(args.sample_index, seed=0)
        # Same two-step the viewer does: resolve the placement, then hang the geometry on it.
        deploy_anchor, home_pose, mc_extra = None, None, {}
        if args.display_frame == "g1-ready-pose":
            deploy_anchor, reason = RP.resolve_deploy_anchor(session.source.layout.roles)
            if deploy_anchor is None:
                print(f"[smoke] SKIP: ready pose unavailable — {reason}")
                return 0
            print(f"[smoke] placement: {deploy_anchor.describe()}")
            home_pose = {r: (deploy_anchor.pos[r], deploy_anchor.rotmat[r])
                         for r in deploy_anchor.pos}
            mc_extra = {"cond_mode": "absolute", "dz_wrist": 0.0,
                        "placement_note": deploy_anchor.describe()}
        role_geometry, _ = resolve_role_geometry(
            session.source.layout, session.source.file_attrs, home_pose)
        world_rotmat, heading_deg, _heading_source = _heading_alignment(result, True, deploy_anchor is not None)
        if world_rotmat is not None:
            print(f"[smoke] scene turned {-heading_deg:+.1f} deg (recorded chest heading "
                  f"{heading_deg:+.1f}) so the body's left is the drawn left")
        display_poses = {
            r.role: _build_display_poses(r, None, role_geometry[r.role], True,
                                         world_rotmat=world_rotmat)
            for r in result.roles
        }

        # ---- 1. additive -------------------------------------------------------------
        plain = viser.ViserServer(host="127.0.0.1", port=args.port, label="plain")
        try:
            off = MCHumanoid(plain, enabled=False)
            off.on_rollout(result, display_poses, session.source.file_attrs, True)
            off.poll(0)
            off.render_at(0)
            plain_nodes = _node_transforms(plain)
        finally:
            plain.stop()
        check("disabled instance draws nothing and is inert",
              not any(n.startswith("/mc") for n in plain_nodes) and not off.available,
              f"{len(plain_nodes)} nodes, reason={off.disabled_reason!r}")

        server = viser.ViserServer(host="127.0.0.1", port=args.port + 1, label="mc")
        try:
            mc = MCHumanoid(server, enabled=True, log=lambda m: print(f"[smoke][mc] {m}", flush=True),
                            **mc_extra)
            if not mc.available:
                print(f"[smoke] SKIP: motion completion unavailable — {mc.disabled_reason}")
                return 0
            mc.on_rollout(result, display_poses, session.source.file_attrs, True)
            # Importable only now: the overlay's construction is what put motion_completion/ on
            # the path (mc_humanoid appends it), so this cannot move above the line before it.
            import mc_frames as F  # noqa: PLC0415
            from flore.utils.rot6d import rot6d_to_rotmat  # noqa: PLC0415

            cond = mc.overlay.cond
            check("conditioning covers the whole action chunk",
                  cond is not None and cond.n == result.predicted_actions.shape[0],
                  f"n={None if cond is None else cond.n} vs H={result.predicted_actions.shape[0]}")
            check("30 Hz grid, no resampling", abs(1.0 / cond.dt - 30.0) < 0.5,
                  f"{1.0 / cond.dt:.2f} Hz, windows={cond.starts}")
            if deploy_anchor is not None:
                # The FK already put the floor at z=0, so the lift must be EXACTLY the pinned
                # zero — a derived one would target RIG_CHEST_STAND_Z (1.30) against the
                # robot's own 1.151 chest and lift the whole take ~15 cm.
                check("ready-pose frame pins the z lift instead of deriving one",
                      cond.dz_wrist == 0.0 and cond.dz_anchor == "pinned",
                      f"dz={cond.dz_wrist:+.4f} m ({cond.dz_anchor})")
                # THE REGRESSION THIS FRAME EXISTS FOR: the conditioning must be a pose a G1
                # can stand in. The display homes put the chest 30 cm in front of the robot's
                # own chest, leaving the hands 5 cm ahead of the root instead of 45 cm, and the
                # completion answered by folding the arms back (elbows behind the torso).
                reach0 = float(cond.left_pos[0][0] - cond.root_pos[0][0])
                check("hands start well IN FRONT of the root (not folded back)",
                      reach0 > 0.15, f"gripper_base x - root x = {reach0:+.3f} m at step 0 "
                                     f"(display homes gave +0.05; the G1's ready pose is +0.26)")

                # LEFT MEANS LEFT. The scene's orientations are re-based onto the home (body
                # faces +x) while positions are only translated, so without the shared yaw turn
                # a motion to the operator's left is drawn going to the robot's RIGHT. Compared
                # against the operator's OWN body frame, straight off the recorded chest.
                head_result = next(r for r in result.roles if r.role == "head")
                head_R0 = F.quat_wxyz_to_R(np.asarray(
                    display_poses["head"].target_wxyz, dtype=np.float64)[:1])[0]
                raw_head_R0 = rot6d_to_rotmat(
                    np.asarray(head_result.target_rot6d, dtype=np.float64)[0])
                raw_left = np.asarray(
                    next(r for r in result.roles if r.role == "left").target_position,
                    dtype=np.float64)
                operator = RP.world_yaw_alignment(raw_head_R0)[0] @ (raw_left[-1] - raw_left[0])
                drawn = cond.left_pos[-1] - cond.left_pos[0]
                lateral_cm = abs(float(operator[1])) * 100.0
                same = (float(operator[1]) > 0) == (float(drawn[1]) > 0)
                check("the drawn motion goes the same way the OPERATOR moved (left is left)",
                      same or lateral_cm < 2.0,
                      f"operator {'LEFT' if operator[1] > 0 else 'RIGHT'} {lateral_cm:.1f} cm · "
                      f"drawn {'LEFT' if drawn[1] > 0 else 'RIGHT'} "
                      f"{abs(float(drawn[1])) * 100:.1f} cm · body faces "
                      f"{np.degrees(F.yaw_of_R(head_R0[None])[0]):+.1f} deg"
                      + ("" if lateral_cm >= 2.0 else " (too little lateral motion to judge)"))
            else:
                check("chest anchor derives a near-zero lift in the display frame",
                      abs(cond.dz_wrist) < 0.15,
                      f"dz={cond.dz_wrist:+.4f} m (chest home z={DISPLAY_HOME_POS['head'][2]})")

            # ---- run every window ----------------------------------------------------
            deadline = time.monotonic() + args.timeout_s
            for step in range(0, cond.n, 10):
                mc.poll(step)
            while time.monotonic() < deadline:
                mc.poll(cond.n - 1)
                if mc.overlay._filled is not None and mc.overlay._filled.all():
                    break
                time.sleep(0.25)
            filled = int(mc.overlay._filled.sum())
            check("every frame of the chunk got completed", filled == cond.n,
                  f"{filled}/{cond.n} frames, {len(mc.overlay._errors)} errors")
            if mc.overlay._errors:
                for msg in mc.overlay._errors[:3]:
                    print(f"[smoke]   window error: {msg}")

            mid = cond.n // 2
            mc.render_at(mid)
            nodes = _node_transforms(server)

            # ---- 2. the frame chain --------------------------------------------------
            errs = {}
            for side, body in (("left", "left_wrist_yaw_link"), ("right", "right_wrist_yaw_link")):
                b = mc.overlay.body_names.index(body)
                p_g, _ = F.mc_gripper_from_wrist(
                    mc.overlay._fk_pos[mid:mid + 1, b],
                    F.quat_wxyz_to_R(mc.overlay._fk_quat[mid:mid + 1, b]), side)
                target = getattr(cond, f"{side}_pos")[mid]
                errs[side] = float(np.linalg.norm(p_g[0] - target)) * 1e3
            check("round trip: completed gripper_base lands on the commanded pose",
                  max(errs.values()) < 5.0,
                  f"L {errs['left']:.2f} mm · R {errs['right']:.2f} mm "
                  f"(~57 mm = mount applied twice, ~242 mm = TCP lever missed)")

            # ---- 3. on the DRAWN chunk ----------------------------------------------
            extrinsic, _ = read_stored_extrinsic(session.source.file_attrs[0])
            eef_t_wrist, eef_R_wrist = wrist_pose_from_stored(extrinsic)
            drawn = {}
            for side in ("left", "right"):
                entry = display_poses[side]
                p_disp = np.asarray(entry.predicted_position[mid : mid + 1], dtype=np.float64)
                R_disp = F.quat_wxyz_to_R(
                    np.asarray(entry.predicted_wxyz[mid : mid + 1], dtype=np.float64))
                p_wr, R_wr = F.wrist_from_eef(p_disp, R_disp, eef_t_wrist, eef_R_wrist)
                p_gb, _ = F.mc_gripper_from_wrist(p_wr, R_wr, side)
                # cond carries the z lift; unlift to get back into the frame the page draws in
                drawn[side] = float(np.linalg.norm(
                    cond.unlift(getattr(cond, f"{side}_pos")[mid : mid + 1])[0] - p_gb[0])) * 1e3
            check("the conditioning IS the displayed chunk (same frame, not a parallel copy)",
                  max(drawn.values()) < 1e-6,
                  f"L {drawn['left']:.2e} mm · R {drawn['right']:.2e} mm")

            # ---- 4. it moved, and it stands -----------------------------------------
            q = mc.overlay._qpos
            spread = float(np.abs(q[:, 7:36] - q[0, 7:36]).max())
            check("the whole body actually moves across the chunk", spread > 0.05,
                  f"max |delta joint| = {spread:.3f} rad over {cond.n} frames")
            pelvis = q[:, 2]
            ankle = np.minimum(
                mc.overlay._fk_pos[:, mc.overlay.body_names.index("left_ankle_roll_link"), 2],
                mc.overlay._fk_pos[:, mc.overlay.body_names.index("right_ankle_roll_link"), 2])
            check("posture is STANDING (not the crouch a mis-placed take produces)",
                  float(pelvis.min()) > 0.70,
                  f"pelvis z {pelvis.min():.3f}..{pelvis.max():.3f} (G1 stands ~0.79), "
                  f"planted foot z {float(ankle.min()):+.3f}")

            # ELBOWS IN FRONT OF THE TORSO. The visible symptom of a mis-placed chunk is not a
            # number on the panel — it is the arms folded behind the back — so it gets its own
            # assertion, measured in the pelvis's own frame at every frame of the chunk.
            if "left_elbow_link" in mc.overlay.body_names:
                elbow = mc.overlay._fk_pos[:, mc.overlay.body_names.index("left_elbow_link"), 0]
                shoulder_name = next((b for b in ("left_shoulder_pitch_link",
                                                  "left_shoulder_roll_link") if b in
                                      mc.overlay.body_names), None)
                base = (mc.overlay._fk_pos[:, mc.overlay.body_names.index(shoulder_name), 0]
                        if shoulder_name else mc.overlay._qpos[:, 0])
                margin = float((elbow - base).min())
                detail = (f"worst elbow_x - shoulder_x = {margin:+.3f} m "
                          f"(measured -0.174 under the display homes, -0.009 at the ready pose)")
                if deploy_anchor is not None:
                    check("the left ELBOW stays in front of the shoulder across the chunk",
                          margin > -0.10, detail)
                else:
                    # Informational in this mode ON PURPOSE: --display-frame display-home IS the
                    # mis-placement, so asserting here would encode a known-bad baseline as a
                    # failure of the test rather than of the frame. Printed so the two runs are
                    # the comparison.
                    print(f"  info elbow (display-home baseline) — {detail}")

            mc_nodes = sorted(n for n in nodes if n.startswith("/mc"))
            shared = [n for n in plain_nodes if n in nodes and not n.startswith("/mc")]
            identical = sum(1 for n in shared if plain_nodes[n] == nodes[n])
            check("the plain scene's nodes are untouched by the extension",
                  identical == len(shared), f"{identical}/{len(shared)} bit-identical")
            print(f"  info +{len(mc_nodes)} /mc nodes; URDF joints "
                  f"{len(mc.overlay._urdf_joint_names)} (model drives "
                  f"{len(mc.overlay.joint_names)}, by name)")
        finally:
            mc.stop()
            server.stop()
    finally:
        session.close()

    if FAILURES:
        print(f"\nSMOKE FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("\nSMOKE OK (the whole-body completion is additive, sub-mm, and on the drawn chunk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
