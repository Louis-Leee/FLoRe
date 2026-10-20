#!/usr/bin/env python3
"""Smoke-test the WHOLE-BODY (``diffusion_umi_wholebody``) deploy path — NO robot, NO bridges.

``tests/test_hl_serving_wholebody.py`` pins everything checkable on a CPU with no model. This is
the other half: the REAL checkpoint and the REAL wire, i.e. the questions only a trained model can
answer. It is the wholebody twin of ``scripts/smoke_test_hl_dense.py``, minus the whole
motion-completion section — there is nothing to complete.

  1. THE LAYOUT IS THE WHOLEBODY ONE, and the joint order is the SDK's. A 43-D action with an
     empty pose block, 29 joints named exactly as ``live_robot.G1_29DOF_SDK_DOF_NAMES`` — which
     is what makes ``DenseWire``'s by-name remap the identity rather than a silent permutation.
  2. THE CHUNK MEANS WHAT THE CHECKPOINT SAYS IT MEANS (``wb_joint_target``). For ``absolute``,
     step 0 of the raw chunk is compared against BOTH the ready pose and the dataset's own
     episode-start seed: it should be near the SEED and far from the ready pose — that asymmetry is
     the whole reason the joint clutch is the default, and if step 0 were already at the ready pose
     the chunk would have been relativized somewhere it should not have been. For ``delta`` the
     claim inverts: step 0 must be ~0 rad, i.e. the chunk is joint MOTION, and a step 0 sitting at
     a posture instead means an absolute model is being served as a delta one.
  3. THE WIRE IS SAFE AND SMOOTH. Step 0 is a bit-exact no-op under the clutch, no frame exceeds
     the low level's per-joint slew budget, every frame is finite, and ``ViveLiveSource``'s three
     dense fields are present at the right widths.
  4. THE LEGS ARE REPORTED, AND ``--legs hold`` REALLY HOLDS THEM. The chunk's own leg excursion
     is printed (this is the number that decides whether ``--legs hold`` is wanted for a first
     run) and the held wire is asserted to move the 17 waist+arm joints while leaving the 12 leg
     joints bit-exact.
  5. THE PROPRIO ROUND-TRIPS. ``commit_chunk_row`` -> ``push_proprio_joints`` -> ``snapshot_obs``
     puts the committed POSTURE back in the state vector's joint slot, and the gripper columns stay
     where the layout says. For ``absolute`` that is the chunk row verbatim; for ``delta`` it is
     ``latch + row``, because the row is motion. A drift here is the proprio silently leaving the
     model's posture space one chunk at a time, which nothing downstream would notice.

Needs a wholebody checkpoint and (for a fast run) a GPU. Run:

    .venv/bin/python scripts/smoke_test_hl_wholebody.py checkpoints/ckpt_0831_wb
    .venv/bin/python scripts/smoke_test_hl_wholebody.py <ckpt> --device cpu --seed 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILURES: list[str] = []

#: The low level's own per-joint rate limit (``HOLOSOMA_VIVE_MAX_JOINT_STEP``, rad/tick at 50 Hz).
#: A chunk step is 1/30 s = 1.67 ticks, so a step bigger than ~2x this asks for motion the slew
#: will refuse to deliver — the command then lags the plan SILENTLY.
LOW_LEVEL_SLEW_RAD_PER_TICK = 0.05
LOW_LEVEL_HZ = 50.0
#: The low level's joint-space pre-arm tolerance (``HOLOSOMA_VIVE_ARM_TOL_RAD``).
PRE_ARM_TOL_RAD = 0.20


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--algo", default=None, choices=(None, "diffusion", "flow", "act"),
                        help="default: read from the checkpoint's own config.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clutch", default="joint", choices=("joint", "none"))
    args = parser.parse_args(argv)

    import json

    import hl_serving as HL

    from flore.s1_policy.data.layout import ACTION_DT, WB_JOINT_NAMES
    from flore.s1_policy.inference import ready_pose as RP
    from flore.s1_policy.serving import hl_dense, hl_wholebody
    from flore.s1_policy.serving.hl_online import load_hl_model

    # The SDK names come via hl_wholebody, which re-exports them for exactly this reason: that
    # module asserts at IMPORT that they equal WB_JOINT_NAMES, so reading them from there is
    # reading the side that already refused a mismatch.

    algo = args.algo
    if algo is None:
        algo = str(json.load(open(os.path.join(args.checkpoint, "config.json"))).get("algo"))
    loaded = load_hl_model(args.checkpoint, algo, device=args.device, max_cameras=2)
    print(f"[smoke] layout={loaded.layout.mode} action_dim={loaded.layout.action_dim} "
          f"state_dim={loaded.layout.state_dim} cameras={list(loaded.image_keys)} "
          f"force={loaded.layout.force_roles} grip={loaded.layout.gripper_roles}")

    # ---- 1. the layout ---------------------------------------------------------
    check("the checkpoint is a WHOLEBODY one (a 29-joint action block)",
          loaded.layout.is_wholebody, str(loaded.layout.mode))
    if FAILURES:
        return 1
    check("no pose dims: nothing was relativized, so nothing is un-relativized here",
          loaded.layout.pose_dims() == (), f"pose_dims={loaded.layout.pose_dims()}")
    check("the joint order IS the SDK's, so DenseWire's by-name remap is the identity",
          tuple(WB_JOINT_NAMES) == tuple(hl_wholebody.G1_29DOF_SDK_DOF_NAMES))

    anchor, why = RP.resolve_deploy_anchor(("left", "right", "head"))
    if anchor is None or anchor.dof is None:
        print(f"[smoke] SKIP: no deploy ready pose — {why}")
        return 0
    print(f"[smoke] {anchor.describe()}")
    ready = np.asarray(anchor.dof, dtype=np.float64)

    seed_zarr = HL._checkpoint_zarr(args.checkpoint)
    seed_joints, seed_note = hl_wholebody.state_seed_from_zarr(seed_zarr)
    print(f"[smoke] {seed_note}")

    target = str(loaded.wb_joint_target)
    print(f"[smoke] wb_joint_target={target!r} (from the checkpoint; not a flag)")
    planner = hl_wholebody.WholeBodyPlanner(
        loaded, obs_hz=1.0 / ACTION_DT, state_source="command", seed_joints=seed_joints,
        joint_target=target)
    rng = np.random.default_rng(args.seed)
    for cam in range(loaded.num_cameras):
        planner.push_image(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), camera=cam)
    if loaded.layout.force_hist_channels:
        planner.push_force(np.zeros(loaded.layout.force_hist_channels))
    for role in loaded.layout.gripper_roles:
        planner.push_gripper(role, 0.09)
    planner.push_proprio_joints()

    chunk = planner.infer_chunk(seed=args.seed, force_enabled=False, gate_n=1.5)

    # ---- 2. the chunk means what wb_joint_target says --------------------------
    gap_ready = float(np.abs(chunk.joints[0] - ready).max())
    gap_seed = float(np.abs(chunk.joints[0] - seed_joints).max())
    if target == "delta":
        # The claim INVERTS here: row 0 is the reference the rest of the chunk is measured from,
        # so it must be ~0. A posture at step 0 means an `absolute` model is being served as a
        # delta one -- shape-identical, and it would publish that posture's motion as if it were
        # motion from the robot's own base.
        zero = float(np.abs(chunk.joints[0]).max())
        print(f"[smoke] |chunk step 0| = {zero:.4f} rad (delta target: row 0 is the reference)")
        check("step 0 of a DELTA chunk is ~zero, i.e. the chunk really is joint MOTION",
              zero < 0.05,
              f"worst |q[0]| {zero:.4f} rad -- an absolute chunk would sit near the "
              f"dataset seed ({gap_seed:.3f} rad away) instead")
        print("[smoke]   -> the clutch's q - q[0] is offset-invariant, so this chunk publishes "
              "the SAME wire an absolute one would. --wb-clutch none is refused for it.")
    else:
        print(f"[smoke] |chunk step 0 - ready pose| = {gap_ready:.3f} rad "
              f"(worst {WB_JOINT_NAMES[int(np.argmax(np.abs(chunk.joints[0] - ready)))]}) · "
              f"|chunk step 0 - dataset seed| = {gap_seed:.3f} rad")
        check("step 0 is nearer the DATASET's posture anchor than the ROBOT's ready pose "
              "(so the chunk was NOT relativized against the robot)",
              gap_seed < gap_ready or gap_ready < 0.05,
              f"seed {gap_seed:.3f} rad vs ready {gap_ready:.3f} rad")
        if gap_ready > PRE_ARM_TOL_RAD:
            print(f"[smoke]   -> this is exactly why --wb-clutch joint is the default: published "
                  f"absolutely, step 0 would be {gap_ready:.3f} rad against a "
                  f"{PRE_ARM_TOL_RAD:g} rad pre-arm gate.")
    travel = float(np.abs(chunk.joints[-1] - chunk.joints[0]).max())
    print(f"[smoke] chunk travel over {chunk.n} steps: worst joint {np.degrees(travel):.1f}° "
          f"({WB_JOINT_NAMES[int(np.argmax(np.abs(chunk.joints[-1] - chunk.joints[0])))]})")

    # ---- 3. the wire is safe and smooth ---------------------------------------
    ready_wire = hl_dense.ready_pose_joint_targets(
        anchor.dof, hl_wholebody.G1_29DOF_SDK_DOF_NAMES, WB_JOINT_NAMES)
    clutch = hl_dense.JointClutch(ready_wire, mode=args.clutch)
    try:
        wire = hl_wholebody.WholeBodyWire(clutch, ready_wire, legs="predict",
                                          joint_target=target)
    except ValueError as exc:
        # The only reachable case is --clutch none against a delta checkpoint, which the deploy
        # script refuses the same way. Report it as the refusal it is, not as a crash.
        print(f"[smoke] REFUSED (correctly): {exc}")
        return 1 if FAILURES else 0
    plan = wire.plan(chunk)
    published = np.asarray(plan["published"], dtype=np.float64)
    if args.clutch == "joint":
        check("step 0 is a BIT-EXACT no-op on the arm (the pre-arm gate closes by construction)",
              np.array_equal(published[0], clutch.base),
              f"worst {float(np.abs(published[0] - clutch.base).max()):.2e} rad")
    step_budget = LOW_LEVEL_SLEW_RAD_PER_TICK * (LOW_LEVEL_HZ * ACTION_DT)
    worst_step = float(np.abs(np.diff(published, axis=0)).max()) if published.shape[0] > 1 else 0.0
    check("no frame exceeds the low level's per-joint slew budget",
          worst_step <= step_budget,
          f"worst per-step |dq| {worst_step:.4f} rad vs budget {step_budget:.4f} "
          f"({100.0 * worst_step / step_budget:.0f}%)")
    check("every published frame is finite and inside the sanity bound",
          all(hl_dense.is_safe_dense_command(*wire.frame(plan, i))[0]
              for i in range(int(plan["n"]))))
    frame0 = wire.frame(plan, 0)
    check("the dense frame is (29 pos, 29 vel) — ViveLiveSource's own widths",
          len(frame0) == 2 and frame0[0].size == 29 and frame0[1].size == 29,
          f"{[np.size(a) for a in frame0]}")
    check("row 0's reference velocity is ZERO (there is no earlier frame in the chunk)",
          float(np.abs(plan["qvel"][0]).max()) == 0.0)

    # ---- 4. the legs ----------------------------------------------------------
    leg_rad, leg_name = hl_wholebody.leg_excursion_rad(chunk.joints)
    print(f"[smoke] LEG excursion within the chunk: {np.degrees(leg_rad):.1f}° ({leg_name}). "
          f"With --legs predict this is a real leg command.")
    held = hl_wholebody.WholeBodyWire(
        hl_dense.JointClutch(ready_wire, mode="joint"), ready_wire, legs="hold",
        joint_target=target).plan(chunk)
    legs = np.asarray(hl_wholebody.LEG_JOINT_INDICES, dtype=int)
    arms = np.array([i for i in range(29) if i not in set(legs.tolist())], dtype=int)
    check("--legs hold leaves the 12 leg joints BIT-EXACT for the whole chunk",
          np.array_equal(held["published"][:, legs],
                         np.broadcast_to(held["published"][0, legs], (int(held["n"]), legs.size))))
    check("--legs hold still moves the 17 waist + arm joints",
          float(np.abs(held["published"][:, arms] - held["published"][0, arms]).max()) > 1e-9,
          f"worst {float(np.abs(held['published'][:, arms] - held['published'][0, arms]).max()):.4f} rad")

    # ---- 5. the proprio round-trip -------------------------------------------
    # `commit_chunk_row`, not `set_joint_target`: the row is a POSTURE for `absolute` and MOTION
    # for `delta`, and the latch must end up holding a posture either way. Capture the latch
    # BEFORE committing, since the delta expectation is relative to it.
    slot = loaded.layout.joint_state_slot
    obs_before, _ap, _ar = planner.snapshot_obs()
    latch_before = np.asarray(obs_before.state[-1, slot], dtype=np.float64).copy()
    want = (chunk.joints[-1] if target == "absolute" else latch_before + chunk.joints[-1])
    planner.commit_chunk_row(chunk.joints[-1])
    planner.push_proprio_joints()
    obs, _ap, _ar = planner.snapshot_obs()
    check(f"the COMMITTED {target} row comes back in the state's joint slot as a POSTURE",
          np.allclose(obs.state[-1, slot], want, atol=1e-12, rtol=0),
          f"worst {float(np.abs(obs.state[-1, slot] - want).max()):.2e} rad")
    check("a written proprio latch is flagged MEASURED (not the seed's UNMEASURED)",
          bool(obs.state_valid[-1, slot].all()))
    for role in loaded.layout.gripper_roles:
        col = loaded.layout.state_slots()[role]["gripper"]
        check(f"the {role} gripper column ({col}) carries the pushed width",
              abs(float(obs.state[-1, col]) - 0.09 / 0.095) < 1e-6,
              f"{float(obs.state[-1, col]):.4f}")

    print()
    if FAILURES:
        print(f"[smoke] {len(FAILURES)} FAILURE(S): " + "; ".join(FAILURES))
        return 1
    print("[smoke] all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
