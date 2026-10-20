#!/usr/bin/env python3
"""Smoke-test the WHOLE-BODY deploy pipeline end to end, with NO robot and NO sensor bridges.

Covers BOTH wire families off one completion — ``--low-level dense`` (29 joints) and
``--low-level 5pt`` (five keypoints) — because they share every step up to the readout, so
running one and trusting the other would leave the differing half untested.

``tests/test_hl_dense.py`` pins everything that can be checked on a CPU with no model. This is
the other half: the real checkpoint, the real motion-completion backend, and the real wire —
i.e. the questions that only a GPU and the g1_mosyn env can answer.

  1. THE CHUNK LANDS ON THE ANCHOR. Seeded at the deploy ready pose, step 0 of every role
     decodes back onto that pose (an anchor-relative chunk is t0-inclusive), so pressing Predict
     with the robot at its ready pose asks it to stay there. A few cm here means the anchor and
     the decode disagree and the arm would FLY on the first step.
  2. THE FRAME CHAIN IS INTACT. Round-trip the completion's own FK back out to ``gripper_base``
     and compare against what it was asked for: sub-mm means ``TCP -> wrist_yaw_link -> MC
     gripper_base`` lines up, ~57 mm means the gripper mount was double-applied, ~242 mm means
     the TCP lever was dropped.
  3. IT STANDS, AND IT IS THE PLACEMENT THAT DECIDES THAT. The pelvis lands in the G1's standing
     band and the elbows stay in front of the shoulders — the visible symptom of a mis-placed
     chunk, which is why it gets its own assertion rather than living in a comment.
  4. THE DENSE WIRE IS SAFE AND SMOOTH. Step 0 is a no-op under the clutch, no frame exceeds the
     low level's per-joint slew budget, the frame is finite, and ``ViveLiveSource``'s three
     fields are present at the right widths.
  5. THE 5pt WIRE STARTS ON THE GATE'S OWN REFERENCE. Read out against the anchor's pelvis,
     step 0's three GATED keypoints land within the low level's 0.07 m tolerance of
     ``_compute_vr_fk(ready pose)`` — which is why that family needs no clutch — the two derived
     FEET stay inside the 0.30 m "is it even standing" warning, and every quaternion is on the
     ``+w`` branch the policy was trained on.

Needs a 3-tracker checkpoint, a GPU, and the same Motion_Completion prerequisites the offline
panel does. Run:

    .venv/bin/python scripts/smoke_test_hl_dense.py ckpt_0831_final
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILURES: list[str] = []

#: The low level's own per-joint rate limit (``HOLOSOMA_VIVE_MAX_JOINT_STEP``, rad/tick at
#: 50 Hz). A completion step is 1/30 s = 1.67 ticks, so a step bigger than ~2x this is asking
#: for motion the slew will refuse to deliver — the command then lags the plan silently.
LOW_LEVEL_SLEW_RAD_PER_TICK = 0.05
LOW_LEVEL_HZ = 50.0


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--algo", default="diffusion", choices=("diffusion", "flow", "act"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clutch", default="joint", choices=("joint", "none"))
    args = parser.parse_args(argv)

    import hl_serving_mc as M

    from flore.client.s0_driver import S0Driver
    from flore.s1_policy.data.layout import ACTION_DT
    from flore.s1_policy.inference import ready_pose as RP
    from flore.s1_policy.serving import hl_dense
    from flore.s1_policy.serving.hl_online import LivePlanner, load_hl_model
    from flore.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES
    from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

    loaded = load_hl_model(args.checkpoint, args.algo, device=args.device, max_cameras=2)
    print(f"[smoke] layout={loaded.layout.mode} roles={loaded.layout.roles} "
          f"cameras={list(loaded.image_keys)} force={loaded.layout.force_roles}")
    check("the checkpoint is a 3-tracker one (both hands AND the chest)",
          tuple(loaded.layout.roles) == M.REQUIRED_ROLES, str(loaded.layout.roles))
    if FAILURES:
        return 1
    eef_t_wrist, eef_R_wrist = M._resolve_extrinsic(args.checkpoint)

    anchor, why = RP.resolve_deploy_anchor(loaded.layout.roles)
    if anchor is None:
        print(f"[smoke] SKIP: no deploy ready pose — {why}")
        return 0
    print(f"[smoke] {anchor.describe()}")

    planner = LivePlanner(loaded, obs_hz=1.0 / ACTION_DT)
    planner.seed_anchor(
        np.stack([anchor.pos[r] for r in loaded.layout.roles]),
        np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in loaded.layout.roles]))
    rng = np.random.default_rng(args.seed)
    for cam in range(loaded.num_cameras):
        planner.push_image(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), camera=cam)
    if loaded.layout.force_hist_channels:
        planner.push_force(np.zeros(loaded.layout.force_hist_channels))
    planner.push_proprio_from_anchor()

    completer = hl_dense.ChunkCompleter(log=lambda m: print(f"[smoke][mc] {m}", flush=True))
    if completer.disabled_reason:
        print(f"[smoke] SKIP: motion completion unavailable — {completer.disabled_reason}")
        return 0
    if not completer.start():
        print(f"[smoke] SKIP: backend failed to start — {completer.disabled_reason}")
        return 0
    try:
        chunk = planner.infer_chunk(seed=args.seed, force_enabled=False, gate_n=1.5)
        decoded = {rc.role: rc for rc in chunk.roles}

        # ---- 1. the chunk lands on the anchor ------------------------------------
        worst = max(float(np.linalg.norm(decoded[r].position[0] - anchor.pos[r]))
                    for r in loaded.layout.roles)
        check("step 0 decodes back onto the ready pose (no fly on the first step)",
              worst < 5e-3,
              f"worst role {worst * 1e3:.2f} mm; travel " + " ".join(
                  f"{r}={np.linalg.norm(decoded[r].position[-1] - decoded[r].position[0]) * 100:.1f}cm"
                  for r in loaded.layout.roles))

        cond = completer.build_cond(
            left_pos=decoded["left"].position,
            left_rotmat=rot6d_to_rotmat(decoded["left"].rot6d),
            right_pos=decoded["right"].position,
            right_rotmat=rot6d_to_rotmat(decoded["right"].rot6d),
            head_pos=decoded["head"].position,
            head_rotmat=rot6d_to_rotmat(decoded["head"].rot6d),
            eef_t_wrist=eef_t_wrist, eef_R_wrist=eef_R_wrist, dt=ACTION_DT,
            grip_width={r: decoded[r].gripper_m for r in loaded.layout.gripper_roles
                        if decoded[r].gripper_m is not None})
        print(f"[smoke] cond: {cond.describe()}")
        check("the lift is PINNED, not derived", cond.dz_wrist == 0.0 and cond.dz_anchor == "pinned",
              f"dz={cond.dz_wrist:+.4f} ({cond.dz_anchor}) — deriving one would target "
              "RIG_CHEST_STAND_Z 1.30 against this pose's own 1.151 and lift the chunk ~15 cm")
        for warning in cond.warnings():
            print(f"[smoke]   cond warning: {warning}")

        completed = completer.complete(cond)
        print(f"[smoke] completed: {completed.describe()}")
        check("every frame of the chunk got completed", completed.complete,
              f"{int(completed.valid.sum())}/{completed.n} frames, {completed.windows} window(s)")

        # ---- 2. the frame chain --------------------------------------------------
        err_mm = completed.wrist_error_m() * 1e3
        check("round trip: the completed gripper_base lands on the commanded pose",
              err_mm < 5.0, f"{err_mm:.2f} mm (~57 mm = mount applied twice, "
                            "~242 mm = TCP lever missed)")

        # ---- 3. it stands, with its elbows in front ------------------------------
        pelvis = completed.root_pos[:, 2]
        check("posture is STANDING (not the crouch a mis-placed chunk produces)",
              float(pelvis.min()) > 0.70, f"pelvis z {pelvis.min():.3f}..{pelvis.max():.3f}")
        if "left_elbow_link" in completed.body_names:
            elbow = completed.fk_pos[:, completed.body_names.index("left_elbow_link"), 0]
            shoulder_name = next((b for b in ("left_shoulder_pitch_link",
                                              "left_shoulder_roll_link")
                                  if b in completed.body_names), None)
            base = (completed.fk_pos[:, completed.body_names.index(shoulder_name), 0]
                    if shoulder_name else completed.root_pos[:, 0])
            margin = float((elbow - base).min())
            check("the left ELBOW stays in front of the shoulder across the chunk",
                  margin > -0.10,
                  f"worst elbow_x - shoulder_x = {margin:+.3f} m (the display-home "
                  "mis-placement measured -0.174)")

        # ---- 4. the wire ---------------------------------------------------------
        ready_mc = hl_dense.ready_pose_joint_targets(
            anchor.dof, G1_29DOF_SDK_DOF_NAMES, completer.joint_names)
        clutch = hl_dense.JointClutch(ready_mc, mode=args.clutch)
        published = clutch.apply(completed.joint_pos)
        qvel = hl_dense.joint_velocity(published, ACTION_DT)
        arm = [i for i, n in enumerate(completer.joint_names)
               if any(k in n for k in ("shoulder", "elbow", "wrist"))]
        raw_gap = float(np.abs(completed.joint_pos[0] - ready_mc)[arm].max())
        step0 = float(np.abs(published[0] - ready_mc).max())
        if args.clutch == "joint":
            check("step 0 is a NO-OP on every joint (the clutch absorbed the null space)",
                  step0 < 1e-12,
                  f"max |dq| vs the ready pose = {step0:.2e} rad, while the completion's own "
                  f"arm offset is {raw_gap:.3f} rad ({np.degrees(raw_gap):.1f} deg) against a "
                  "0.20 rad pre-arm gate")
        else:
            print(f"  info --clutch none: step 0 is {step0:.3f} rad from the ready pose "
                  f"(arms {raw_gap:.3f} rad); HOLOSOMA_VIVE_ARM_TOL_RAD is 0.20")
        per_step = float(np.abs(np.diff(published, axis=0)).max())
        budget = LOW_LEVEL_SLEW_RAD_PER_TICK * LOW_LEVEL_HZ * ACTION_DT
        check("no step asks for more than the low level's per-joint slew can deliver",
              per_step <= budget,
              f"worst {per_step:.4f} rad/step vs a {budget:.4f} rad budget "
              f"({LOW_LEVEL_SLEW_RAD_PER_TICK} rad/tick x {LOW_LEVEL_HZ:g} Hz x {ACTION_DT:.4f} s)")
        safe = [hl_dense.is_safe_dense_command(published[i], qvel[i])
                for i in range(published.shape[0])]
        check("every published frame passes the non-finite / sanity gate",
              all(ok for ok, _ in safe),
              next((r for ok, r in safe if not ok), "all finite and in range"))

        driver = S0Driver(port=8999)   # never started; send_dense only fills the latch
        driver.send_dense(published[0], qvel[0], completer.joint_names)
        msg = json.loads(driver._latest)
        check("the wire is a dense frame ViveLiveSource can read",
              msg["mode"] == "dense"
              and len(msg["joint_pos"]) == len(msg["joint_vel"]) == len(msg["joint_names"]) == 29
              and "f_cmd" not in msg,
              f"mode={msg['mode']} widths="
              f"{len(msg['joint_pos'])}/{len(msg['joint_vel'])}/{len(msg['joint_names'])} "
              f"names[0]={msg['joint_names'][0]}")

        # ---- 5. the OTHER wire off the same completion: 5pt keypoints -------------
        # The 5pt family's claim is a NUMBER, and this is where it can be checked: read out
        # against the anchor's own pelvis, step 0 of the completed chunk must land on
        # `_compute_vr_fk(ready pose)` — the very reference the low level's keypoint pre-arm
        # gate compares against — so the gate closes with no clutch on this side.
        readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, float(anchor.pelvis_z)))
        ready_pos, ready_orn = hl_dense.ready_pose_five_point(anchor, readout)
        five = hl_dense.FivePointWire(readout, ready_pos, ready_orn)
        kp_pos, kp_orn = readout.five_point(completed)
        check("the 5pt readout covers the whole chunk at the right widths",
              kp_pos.shape == (completed.n, 15) and kp_orn.shape == (completed.n, 20),
              f"pos {kp_pos.shape} orn {kp_orn.shape}")
        gaps = hl_dense.keypoint_gaps_m(kp_pos[0], ready_pos)
        gated = float(gaps[:hl_dense.FIVE_POINT_GATED].max())
        check("5pt step 0 lands on the low level's own keypoint gate reference",
              gated <= hl_dense.KEYPOINT_GATE_TOL_M,
              "per-point " + ", ".join(f"{lbl} {g * 1e3:.1f} mm"
                                       for lbl, g in zip(hl_dense.FIVE_POINT_LABELS, gaps))
              + f" (tol {hl_dense.KEYPOINT_GATE_TOL_M * 1e3:.0f} mm on the first "
                f"{hl_dense.FIVE_POINT_GATED}; the FEET are the completion's inference and are "
                "warned, not gated)")
        feet = float(gaps[hl_dense.FIVE_POINT_GATED:].max())
        check("the completion's FEET are a differently-stanced body, not a folded one",
              feet <= hl_dense.FIVE_POINT_DERIVED_WARN_M,
              f"worst foot {feet * 1e3:.0f} mm vs the "
              f"{hl_dense.FIVE_POINT_DERIVED_WARN_M * 1e3:.0f} mm warn threshold (the ready "
              "pose is a 38 deg crouch, so ~60-100 mm is expected and correct)")
        kp_safe = [hl_dense.is_safe_keypoint_command(kp_pos[i], kp_orn[i])
                   for i in range(kp_pos.shape[0])]
        check("every 5pt frame passes the non-finite / unit-quat / reach gate",
              all(ok for ok, _ in kp_safe),
              next((r for ok, r in kp_safe if not ok), "all finite, unit and in reach"))
        driver.send_5pt(kp_pos[0], kp_orn[0])
        msg5 = json.loads(driver._latest)
        check("the wire is a 5pt frame ViveLiveSource can read",
              msg5["mode"] == "5pt" and (len(msg5["pos"]), len(msg5["orn"])) == (15, 20)
              and "f_cmd" not in msg5,
              f"mode={msg5['mode']} widths={len(msg5['pos'])}/{len(msg5['orn'])}")
        w = float(min(np.asarray(msg5["orn"]).reshape(5, 4)[:, 3]))
        check("every 5pt quaternion is on the +w branch the policy was trained on",
              w >= 0.0, f"min scalar part {w:+.4f} (a -q is ~20 sigma OOD, and one chest pose "
                        "near the 180 deg seam flips all five at once)")
        _ok5, _note5 = five.gate_report(five.frame({"pos": kp_pos, "orn": kp_orn}, 0))
        print(f"  info 5pt {_note5}")
        print(f"  info completion {completed.ms:.0f} ms for {completed.n} steps "
              f"({completed.windows} window(s)); the chunk is the FUTURE, so none of the "
              "teleop path's 667 ms lag applies")

        # ---- 6. THE POST-COMPLETION COMMAND ANCHOR, on a real chunk ----------------------
        # `--post-anchor command` carries the last EXECUTED step forward instead of re-reading
        # every chunk off the fixed ready pose. Measured here on both wires because the thing
        # that can go wrong is arithmetic on real numbers: step 0 must stay a no-op relative to
        # the CARRIED base (not to the ready pose), and the leak must actually bind.
        print("\n[smoke] 6. the post-completion command anchor (--post-anchor command)")
        post = hl_dense.PostAnchorConfig(mode="command", eps_joint_rad=0.15, eps_m=0.03)
        carried = hl_dense.JointClutch(ready_mc, mode="joint", post=post)
        first = carried.apply(completed.joint_pos)
        carried.commit(first[16])                     # 16 = the default --exec-steps
        base_after = carried.base.copy()
        second = carried.apply(completed.joint_pos)   # the SAME chunk, re-published
        check("dense: step 0 of the next chunk lands on the CARRIED base, not the ready pose",
              float(np.abs(second[0] - base_after).max()) < 1e-12,
              f"|step0 - carried base| = {float(np.abs(second[0] - base_after).max()):.2e} rad, "
              f"and that base is {float(np.abs(base_after - ready_mc).max()):.3f} rad from the "
              "ready pose (which is the accumulated command the mode exists to keep)")
        # The robot is HOLDING the ready pose here (no rig attached), so `ready_mc` is exactly
        # what RecordTap would report — which makes this the worst case for the leak: the whole
        # accumulated command is a lead over a robot that has not moved at all.
        label = carried.sync(ready_mc, 0.0)
        pre = carried.post_stats()["max_lead_rad"]
        lead = float(np.abs(carried.base - ready_mc).max())
        check("dense: the carried lead is bounded by the leak, written back into the state",
              lead <= post.eps_joint_rad + 1e-9,
              f"{label} → post-leak lead {lead:.3f} rad (pre-leak {pre:.3f}, bound "
              f"{post.eps_joint_rad:g})")
        carried.commit(first[16])
        carried.sync(ready_mc, 8.0)                   # in contact: decay, do not advance
        check("dense: contact DECAYS the carry toward the robot instead of advancing it",
              carried.post_stats()["n_contact_pullback"] == 1
              and float(np.abs(carried.base - ready_mc).max()) <= post.eps_joint_rad + 1e-9,
              f"lead after the 8 N tick {float(np.abs(carried.base - ready_mc).max()):.3f} rad "
              "(a command that keeps advancing into a blocked hand is a press, not a reach)")

        kp_clutch = hl_dense.KeypointClutch(ready_pos, ready_orn, post=post)
        kp_five = hl_dense.FivePointWire(readout, ready_pos, ready_orn, clutch=kp_clutch,
                                         measured_frame=lambda _rf: (ready_pos, ready_orn))
        kc = kp_clutch.n_carried
        p1, o1 = kp_clutch.apply(kp_pos, kp_orn)
        # THE FEET MUST PASS THROUGH. Carrying them re-bases the completion's stance onto the
        # ready pose's 38 deg CROUCH — measured 74.7 mm of foot shift on the first chunk, i.e. a
        # leg command nobody asked for. This is the measurement that set `n_carried`.
        foot_shift = float(hl_dense.keypoint_gaps_m(
            p1.reshape(-1, 3 * 5)[:, 3 * kc:], kp_pos[:, 3 * kc:]).max())
        check("5pt: the two FEET are republished absolutely, never re-based",
              foot_shift == 0.0,
              f"foot shift {foot_shift * 1e3:.1f} mm; carrying them instead would move the "
              f"stance {float(hl_dense.keypoint_gaps_m(kp_pos[0, 3 * kc:], ready_pos[3 * kc:]).max()) * 1e3:.1f} mm "
              "onto the ready crouch")
        # The FIRST chunk's base is the ready frame, so step 0 of the carried points lands ON the
        # gate's own reference. That REMOVES a first-executed-step jump rather than adding one:
        # under mode=off step 0 is the readout's own landing, which is 0.7-3.5 mm on the hands but
        # tens of mm on the TORSO (only pelvis XY+yaw are conditioned, so the completion's torso
        # height and pitch are its own inference). The wire idles at the ready frame, so with
        # `command` the first executed step is a bit-exact no-op on all three gated points.
        snap = float(hl_dense.keypoint_gaps_m(p1[0, :3 * kc], kp_pos[0, :3 * kc]).max())
        check("5pt: the first chunk lands step 0 exactly ON the gate reference",
              float(hl_dense.keypoint_gaps_m(p1[0, :3 * kc], ready_pos[:3 * kc]).max()) < 1e-12
              and snap <= hl_dense.KEYPOINT_GATE_TOL_M,
              f"step 0 now sits ON the ready frame; the first step no longer jumps "
              f"{snap * 1e3:.1f} mm (worst gated point, vs the {hl_dense.KEYPOINT_GATE_TOL_M * 1e3:.0f} mm "
              "gate tol). All three gated points are carried TOGETHER — carrying the hands but "
              "not the torso would distort the hand-vs-chest geometry the completion produced.")
        kp_five.commit((p1[16], o1[16]))
        p2, o2 = kp_clutch.apply(kp_pos, kp_orn)
        step0 = float(hl_dense.keypoint_gaps_m(
            p2[0, :3 * kc], kp_clutch.base[0][:3 * kc]).max())
        check("5pt: step 0 of the next chunk lands on the CARRIED keypoint frame",
              step0 < 1e-12,
              f"worst |step0 - carried base| = {step0:.2e} m, carried "
              f"{kp_clutch.drift_m() * 1e3:.1f} mm from the ready frame")
        moved = float(hl_dense.keypoint_gaps_m(p2[-1] - p2[0], p1[-1] - p1[0]).max())
        check("5pt: the carry offsets the chunk without changing its SHAPE",
              moved < 1e-12,
              f"worst change in the within-chunk displacement = {moved:.2e} m (a carry that "
              "altered the path would be re-planning, not re-basing)")
        label5 = kp_five.sync(None, 0.0)
        gaps5 = hl_dense.keypoint_gaps_m(kp_clutch.base[0], ready_pos)[:kc]
        check("5pt: the leak binds the carried frame to the measured one",
              float(gaps5.max()) <= post.eps_m + 1e-9,
              f"{label5} → worst carried lead {float(gaps5.max()) * 1e3:.1f} mm "
              f"(bound {post.eps_m * 1e3:.0f} mm)")
    finally:
        completer.stop()

    if FAILURES:
        print(f"\nSMOKE FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("\nSMOKE OK (the chunk lands on the ready pose, completes sub-mm, stands, and the "
          "wire is a safe dense frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
