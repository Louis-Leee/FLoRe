#!/usr/bin/env python3
"""Smoke-test the FIVE-POINT (``diffusion_umi_5pt``) deploy path — NO robot, NO bridges.

``tests/test_hl_serving_5pt.py`` pins everything checkable on a CPU with no model. This is the
other half: the REAL checkpoint, the REAL decode anchor and the REAL wire — the questions only a
trained model can answer. Sibling of ``scripts/smoke_test_hl_wholebody.py``, minus the whole
motion-completion section (there is nothing to complete) and plus the display IK.

  1. THE LAYOUT IS A FIVE-POINT ONE, and the wire's slot order is holosoma's. Five roles, 59-D
     action, a GEOMETRIC state, and ``WIRE_ROLES`` in ``FIVE_POINT_BODIES`` order — which is what
     makes the gate's per-point labels and the published vector describe the same points.

  2. THE DATA'S ROW 0 **IS** ITS ANCHOR, AND THE MODEL'S IS NOT. The raw target chunk's step 0 must
     be an identity delta for all five roles (that is the converter's property, and if it were not,
     every chunk would be relativized against a frame it does not start at). The PREDICTION's step
     0 is then measured against the low level's own pre-arm gate reference: on the shipped
     checkpoint it lands 59-63 mm out against a 70 mm tolerance, which is the whole reason the
     keypoint clutch is the DEFAULT here and ``off`` on the completion path.

  3. THE CLUTCH MAKES STEP 0 EXACT. Under ``--post-anchor command`` the published step 0 must equal
     the ready frame bit-for-bit on every CARRIED point, and the gate report must say PASS.

  4. THE WIRE IS SAFE AND SMOOTH. Every frame is finite, every quaternion is unit with ``w >= 0``
     (the policy was trained on the ``+w`` branch only — an antipode is ~20 sigma OOD and reads on
     the rig as the arms snapping), the widths are 15/20, and the per-step keypoint slew is printed
     so a chunk asking for motion the low level will not deliver is visible rather than silent.

  5. THE OBSERVATION IS THE ANCHOR'S OWN GEOMETRY, and it round-trips. ``commit_role`` ->
     ``push_proprio_geom`` -> ``snapshot_obs`` must put the committed poses' geometry back in the
     state vector, and it must equal ``wb_geom.geom_from_bodies`` on those same poses — the
     function training used — rather than a re-derivation of it.

  6. THE DISPLAY IK CLOSES ON THE CHUNK. The per-limb solver must reproduce the ready pose to
     ~0 mm (that is the levers/chains/root cross-check) and then reach the predicted five points;
     the residual is printed per role. It is DISPLAY ONLY, so a large residual is reported and not
     a failure — except at the ready pose, where a non-zero value means the IK and the FK disagree.

A KEYPOINT-SUBSET checkpoint (``left_only`` — the plain ``diffusion_umi`` preset — or
``both_hands*``) runs the SAME six sections: the wire slots it does not predict are HELD at the
ready pose (``hl_5pt.held_anchor_poses``), section 2 additionally pins every held slot to the gate
reference at 0.000 mm, and section 5 checks the ``wrt_start`` observation instead of the geometry.

Needs a checkpoint and (for a fast run) a GPU. Run::

    .venv/bin/python scripts/smoke_test_hl_5pt.py checkpoints/umi_5pt/h50_best_overall
    .venv/bin/python scripts/smoke_test_hl_5pt.py <ckpt> --device cpu --post-anchor off
    .venv/bin/python scripts/smoke_test_hl_5pt.py checkpoints/umi_5pt_wipe_board/h50   # left_only
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

#: The low level's keypoint pre-arm tolerance (``HOLOSOMA_VIVE_ARM_TOL``, metres) — quoted so this
#: script can say "this WOULD arm" without importing the submodule.
PRE_ARM_TOL_M = 0.07


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
    parser.add_argument("--sample-index", type=int, default=0,
                        help="which HLDataset anchor the observation comes from")
    parser.add_argument("--post-anchor", default="command", choices=("off", "command"))
    parser.add_argument("--carry", default="all", choices=("all", "upper"))
    parser.add_argument("--no-ik", dest="ik", action="store_false",
                        help="skip section 6 (no yourdfpy / no submodule)")
    parser.set_defaults(ik=True)
    args = parser.parse_args(argv)

    raw = json.load(open(os.path.join(args.checkpoint, "config.json")))
    algo = args.algo or str(raw.get("algo") or "diffusion")
    horizon = raw.get({"diffusion": "diffusion", "flow": "model", "act": "act"}[algo], {}).get(
        "action_horizon")
    if isinstance(horizon, int) and horizon != 50:
        # Same contract as the launchers: layout.py resolves the pin at IMPORT, so it has to be set
        # before the first fuji import below.
        os.environ["FUJI_ACTION_HORIZON"] = str(horizon)

    from fuji.s1_policy.data import wb_geom
    from fuji.s1_policy.data.layout import get_layout
    from fuji.s1_policy.inference import ready_pose as RP
    from fuji.s1_policy.serving import hl_5pt, hl_dense
    from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

    print(f"\n[1] LAYOUT — {args.checkpoint}")
    mode = str(raw.get("hl_data", {}).get("mode") or "")
    try:
        layout = get_layout(mode)
        family = hl_5pt.check_layout(layout)
    except ValueError as exc:
        check("hl_data.mode is servable on the 5pt wire", False, f"{mode}: {exc}")
        return 1
    held = hl_5pt.held_roles(layout)
    check("hl_data.mode is servable on the 5pt wire", True, f"{mode} -> family {family}")
    if family == "fivepoint":
        check("five roles, in the wire's slot order", tuple(layout.roles) == hl_5pt.WIRE_ROLES,
              str(tuple(layout.roles)))
        check("the state is GEOMETRIC, the action is not wholebody",
              layout.state_is_geometric and not layout.is_wholebody,
              f"state_geom={list(layout.state_geom)} state_dim={layout.state_dim}")
        check("every wire body has a role", set(hl_5pt.GEOM_BODY_ROLE) <= set(RP.ROLE_BODY.values()),
              str(sorted(hl_5pt.GEOM_BODY_ROLE)))
    else:
        check("the predicted roles are a subset of the wire's five",
              set(layout.roles) <= set(hl_5pt.WIRE_ROLES),
              f"predicted {list(layout.roles)}, HELD {list(held)}")
        check("the state is wrt_start (the base LivePlanner's), the action is not wholebody",
              not layout.state_is_geometric and not layout.is_wholebody,
              f"state_dim={layout.state_dim} action_dim={layout.action_dim}")

    print("\n[2] THE ANCHOR, AND WHOSE ROW 0 IS IT")
    # Over the WIRE's five, whatever the layout predicts: the idle frame and the held slots need
    # every one, and the planner is seeded with the layout's own subset of the same anchor.
    anchor, why = RP.resolve_deploy_anchor(hl_5pt.WIRE_ROLES)
    check("the ready pose resolves for all five wire roles", anchor is not None, why)
    if anchor is None:
        return 1
    print(f"       {anchor.describe()}")
    readout = hl_dense.KeypointReadout(np.array([0.0, 0.0, float(anchor.pelvis_z)]))
    ready_pos, ready_orn = hl_dense.ready_pose_five_point(anchor, readout)
    held_poses = hl_5pt.held_anchor_poses(anchor, layout)
    check("every held slot has a ready-pose pose", set(held_poses) == set(held), str(sorted(held_poses)))

    session = _open_session(args.checkpoint, algo, args.device)
    try:
        res = session.infer(args.sample_index, seed=args.seed)
        tgt0 = max(float(np.linalg.norm(res.target_actions[0][layout.slots(r).pos]))
                   for r in layout.roles)
        check(f"the DATA's row 0 is an identity delta ({len(layout.roles)} predicted role(s))",
              tgt0 < 1e-9, f"worst |dp| {tgt0 * 1e3:.6f} mm — the converter's own property")

        ap = np.stack([anchor.pos[r] for r in layout.roles])
        ar = np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in layout.roles])
        from fuji.s1_policy.serving.hl_online import decode_chunk

        chunk = decode_chunk(res.predicted_actions, layout, ap, ar,
                             force_enabled=False, gate_n=0.0)
        # The wire is built HERE (section 3 reuses it): its `all_rows` is the one fill that puts the
        # held slots beside the predicted ones, so the absolute readout below is what `off` publishes.
        post = hl_dense.PostAnchorConfig(mode=args.post_anchor)
        clutch = hl_5pt.build_clutch(ready_pos, ready_orn, post=post, carry=args.carry)
        wire = hl_5pt.FivePointDirectWire(readout, ready_pos, ready_orn, clutch=clutch,
                                          held=held_poses)
        abs_pos, _abs_orn = readout.all_from_decoded(wire.all_rows(chunk))
        gaps_off = hl_dense.keypoint_gaps_m(abs_pos[0], ready_pos)
        print("       step-0 gap vs the gate reference, ABSOLUTE (mm): "
              + ", ".join(f"{lab} {g * 1e3:.1f}"
                          for lab, g in zip(hl_dense.FIVE_POINT_LABELS, gaps_off)))
        predicted_idx = [k for k, r in enumerate(hl_5pt.WIRE_ROLES) if r in layout.roles]
        held_idx = [k for k, r in enumerate(hl_5pt.WIRE_ROLES) if r in held]
        pred_gated = [k for k in predicted_idx if k < hl_dense.FIVE_POINT_GATED]
        if pred_gated:
            worst_pred = float(gaps_off[pred_gated].max())
            check("the MODEL's row 0 is NOT the anchor (so a clutch is warranted)", worst_pred > 1e-4,
                  f"worst predicted gated {worst_pred * 1e3:.1f} mm vs a "
                  f"{PRE_ARM_TOL_M * 1e3:.0f} mm tolerance")
        if held_idx:
            check(f"every HELD slot lands ON the gate reference ({len(held_idx)} slot(s))",
                  float(gaps_off[held_idx].max()) < 1e-9,
                  f"worst held {gaps_off[held_idx].max() * 1e6:.3f} µm — a held row IS the ready pose")
            steps_held = max(float(np.linalg.norm(np.diff(abs_pos[:, 3 * k:3 * k + 3], axis=0)))
                             for k in held_idx) if abs_pos.shape[0] > 1 else 0.0
            check("a held slot never moves within the chunk", steps_held < 1e-12,
                  f"worst step {steps_held * 1e6:.3f} µm")

        print(f"\n[3] THE CLUTCH — --post-anchor {args.post_anchor} --carry {args.carry}")
        clutch.sync(None)                        # first chunk: base = the READY frame
        plan = wire.plan(chunk)
        f0 = wire.frame(plan, 0)
        gaps0 = hl_dense.keypoint_gaps_m(f0[0], ready_pos)
        k = clutch.n_carried
        if post.carrying:
            check(f"published step 0 is bit-exact on the {k} CARRIED point(s)",
                  float(gaps0[:k].max()) == 0.0, f"worst {gaps0[:k].max() * 1e6:.3f} µm")
        ok_gate, note = wire.gate_report(f0)
        check("the gate report says PASS", ok_gate, note)
        print(f"       {wire.status(f0)}")

        print("\n[4] THE WIRE")
        pos, orn = plan["pos"], plan["orn"]
        check("widths are 15 positions + 20 quats", pos.shape[1] == 15 and orn.shape[1] == 20,
              f"{pos.shape} / {orn.shape}")
        check("every frame is finite", bool(np.isfinite(pos).all() and np.isfinite(orn).all()))
        q = orn.reshape(-1, 4)
        check("every quaternion is unit", float(np.abs(np.linalg.norm(q, axis=1) - 1.0).max()) < 1e-9)
        check("every quaternion has w >= 0 (the trained branch)", bool((q[:, 3] >= 0.0).all()),
              "an antipode is ~20 sigma OOD for this low level")
        for i in range(pos.shape[0]):
            ok, reason = hl_dense.is_safe_keypoint_command(pos[i], orn[i], n_points=5)
            if not ok:
                check(f"step {i} passes the publish gate", False, reason)
                break
        else:
            check("every step passes the publish gate", True, f"{pos.shape[0]} step(s)")
        span = hl_5pt.chunk_span(pos, hl_dense.FIVE_POINT_LABELS)
        print(f"       {span.describe()}")
        if held:
            still = [k for k, r in enumerate(hl_5pt.WIRE_ROLES) if r in held]
            check("the published HELD slots are still (excursion 0)",
                  float(span.excursion_mm[still].max()) < 1e-9,
                  f"worst {span.excursion_mm[still].max():.6f} mm")
        # The (6,) f_cmd for THIS checkpoint's hands: a held hand is a hard zero, never absent.
        wrench_rows = {r: (None if chunk.role(r).wrench_pred is None else chunk.role(r).wrench_pred[0])
                       for r in layout.force_roles}
        fcmd = hl_5pt.force_command(wrench_rows, 0.0)
        check("f_cmd is None or both hands wide", fcmd is None or fcmd.shape == (6,),
              "None" if fcmd is None else str(np.round(fcmd, 2).tolist()))
        for hand in hl_5pt.FORCE_HANDS:
            if hand not in layout.force_roles and fcmd is not None:
                k = hl_5pt.FORCE_HANDS.index(hand)
                check(f"the {hand} hand's f_cmd is a hard zero (not predicted)",
                      float(np.abs(fcmd[3 * k:3 * k + 3]).max()) == 0.0)
        print("       per-point reach / worst step (mm): "
              + ", ".join(f"{lab} {e:.0f}/{s:.0f}" for lab, e, s
                          in zip(hl_dense.FIVE_POINT_LABELS, span.excursion_mm, span.slew_mm)))
        # ⚠️ The per-point maximum above is NOT what the low level caps. `_slew_vec` takes one L2
        # over the whole (15,) vector, so the five points share one pool and the concatenated step
        # is the quantity that decides clamping. Printed together because the pair is the diagnosis:
        # per-point says WHICH point is fast, concatenated says whether the wire will deliver it.
        print(f"       all-point step {span.step_mm_total:.1f} mm vs budget "
              f"{span.budget_mm_per_step:.2f} mm/step -> ratio {span.slew_ratio:.1f}x "
              f"[{span.budget_source}]")
        # The cap-independent reading. A speed transfers between tasks; a ratio does not.
        print(f"       peak speed {span.peak_speed_ms:.2f} m/s"
              + ("  ⚠️ NOT a demonstrated motion — chunk DISCONTINUITY"
                 if span.implausible else "  (plausible demonstration speed)"))

        print("\n[5] THE OBSERVATION ROUND-TRIP")
        # The REAL `load_hl_model`, not a stand-in: this section is about the planner's own
        # observation assembly, and a hand-built LoadedHLModel would be a second definition of the
        # contract under test. It costs a second 673 MB load, which is the right price for a smoke
        # test that must exercise the path the rig runs.
        from fuji.s1_policy.serving.hl_online import load_hl_model

        loaded = load_hl_model(args.checkpoint, algo, device=args.device,
                               max_cameras=max(1, len(tuple(session.source.image_keys))))
        planner = hl_5pt.build_planner(loaded, obs_hz=30.0)
        check("the planner family follows the layout",
              type(planner).__name__ == ("FivePointPlanner" if family == "fivepoint"
                                         else "SubsetKeypointPlanner"), type(planner).__name__)
        planner.seed_anchor(ap, ar)
        names, state_ready, written = planner.state_readout()
        print("       at the ready pose: "
              + " ".join(f"{n}={v:+.4f}" for n, v in zip(names, state_ready)))
        last = int(pos.shape[0]) - 1
        if family == "fivepoint":
            check("the un-commanded state is flagged UNMEASURED", not written)
            for rc in chunk.roles:
                planner.commit_role(rc.role, rc.position[last], rc.rot6d[last])
            planner.push_proprio()
            obs, got_ap, _got_ar = planner.snapshot_obs()
            # The SAME function training used, on the SAME bodies, in the same frame.
            want = wb_geom.geom_from_bodies(
                {body: chunk.role(role).position[last]
                 for body, role in hl_5pt.GEOM_BODY_ROLE.items()},
                {body: rot6d_to_rotmat(chunk.role(role).rot6d[last])
                 for body, role in hl_5pt.GEOM_BODY_ROLE.items()},
                tuple(layout.state_geom))
            got = np.concatenate([obs.state[-1, layout.state_geom_slots()[b]]
                                  for b in layout.state_geom])
            check("the committed poses' geometry is in the state",
                  float(np.abs(got - want).max()) < 1e-12, f"worst {np.abs(got - want).max():.3e}")
            check("the anchor advanced to the last executed step",
                  float(np.abs(got_ap[0] - chunk.role("left").position[last]).max()) < 1e-12)
            _geom2, written2 = planner.geom_state()
            check("a commit clears the UNMEASURED flag", written2)
        else:
            from fuji.s1_policy.data.transforms import rot6d_relative

            ident = rotmat_to_rot6d(np.eye(3))
            for ri, role in enumerate(layout.roles):
                sl = layout.state_slots()[role]["wrt_start_rot6d"]
                check(f"{role}: the un-commanded wrt_start state is the IDENTITY (a training "
                      f"episode's frame 0)", float(np.abs(state_ready[sl] - ident).max()) < 1e-12)
            for rc in chunk.roles:
                planner.commit_role(rc.role, rc.position[last], rc.rot6d[last])
            planner.push_proprio()
            obs, got_ap, _got_ar = planner.snapshot_obs()
            for ri, role in enumerate(layout.roles):
                sl = layout.state_slots()[role]["wrt_start_rot6d"]
                want = rot6d_relative(chunk.role(role).rot6d[last], ar[ri])
                check(f"{role}: the committed rotation is in the state as R_start^T R_now",
                      float(np.abs(obs.state[-1, sl] - want).max()) < 1e-12,
                      f"worst {np.abs(obs.state[-1, sl] - want).max():.3e}")
            _names2, state2, _w2 = planner.state_readout()
            check("state_readout agrees with snapshot_obs",
                  float(np.abs(state2 - obs.state[-1]).max()) < 1e-12)
            check("the anchor advanced to the last executed step",
                  float(np.abs(got_ap[0] - chunk.role(layout.roles[0]).position[last]).max()) < 1e-12)

        if args.ik:
            print("\n[6] THE DISPLAY IK (violet body)")
            from fuji.s1_policy.inference.keypoint_ik import IK_ROLES, KeypointIK

            ik, why = KeypointIK.load()
            check("the IK loads", ik is not None, why)
            if ik is not None:
                mm, deg = ik.check_round_trip(anchor.dof)
                check("it reproduces the READY POSE exactly", mm < 1e-6,
                      f"{mm:.6f} mm / {deg:.6f} deg — the levers/chains/root cross-check")
                # ALL FIVE roles — the held ones from the wire's own ready-pose rows.
                poses = {role: (row["position"], rot6d_to_rotmat(row["rot6d"]))
                         for role, row in wire.all_rows(chunk).items()}
                out = ik.solve_chunk(poses, seed_dof=anchor.dof)
                print(f"       {out.describe()} over {out.n} step(s)")
                print("       per-role worst (mm): "
                      + ", ".join(f"{r} {v:.1f}" for r, v
                                  in zip(IK_ROLES, out.residual_mm.max(axis=0))))
                print("       DISPLAY ONLY — the low level solves its own body from the wire.")

        print("\n[7] REACH COMPENSATION on the HANDS (--reach-ki; default OFF) — against THIS chunk")
        from fuji.s1_policy.serving.hl_reach import ReachConfig

        hand_roles = tuple(r for r in layout.roles if r in hl_5pt.HandReach.HANDS)
        frames = [wire.frame(plan, i) for i in range(plan["n"])]
        off = hl_5pt.HandReach(ReachConfig(), layout.roles, force_roles=layout.force_roles)
        same = all(off.apply(f, measured=frames[0], armed=True, force_by_hand={}) is f for f in frames)
        check("--reach-ki 0 hands back the SAME frame object on every step (byte-identical wire)", same)
        on = hl_5pt.HandReach(ReachConfig(ki=0.05), layout.roles, force_roles=layout.force_roles)
        check(f"ki 0.05 compensates exactly the predicted hands {list(hand_roles)}",
              on.enabled and on.roles == hand_roles, f"roles {on.roles}, unprotected {on.unprotected}")
        if on.enabled:
            # A robot that NEVER MOVES (measured = step 0 for the whole chunk) under a hand the model
            # walks: the gate must open, the residual must grow along the hand's motion and stop at
            # the clamp, and nothing but the compensated hands may change by a single bit.
            stuck = frames[0]
            pubs = [on.apply(f, measured=stuck, armed=True,
                             force_by_hand={r: 0.0 for r in layout.force_roles}) for f in frames]
            hand_slots = [slice(3 * hl_5pt.WIRE_ROLES.index(r), 3 * hl_5pt.WIRE_ROLES.index(r) + 3)
                          for r in hand_roles]
            others = np.ones(15, dtype=bool)
            for sl in hand_slots:
                others[sl] = False
            untouched = all(np.array_equal(p[0][others], f[0][others]) and p[1] is f[1]
                            for p, f in zip(pubs, frames))
            check("torso, both feet and any held slot are bit-identical with compensation ON", untouched)
            extra = np.array([[np.linalg.norm(p[0][sl] - f[0][sl]) for sl in hand_slots]
                              for p, f in zip(pubs, frames)])
            check("the extra reach never exceeds the 3 cm clamp on any hand at any step",
                  float(extra.max()) <= 0.03 + 1e-12, f"worst {extra.max() * 1e3:.1f} mm")
            res = on.residual_mm()
            print(f"       stuck-robot rehearsal: residual after {plan['n']} steps "
                  + ", ".join(f"{r} {v:.1f} mm" for r, v in res.items())
                  + f" · {on.describe()}")
            # Un-armed: identity, and the residual is gone — step 0 stays ON the gate reference.
            f_unarmed = on.apply(frames[0], measured=stuck, armed=False, force_by_hand={})
            check("un-armed hands back the SAME object and drops the residual",
                  f_unarmed is frames[0] and all(v == 0.0 for v in on.residual_mm().values()))
    finally:
        session.close()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


def _open_session(checkpoint: str, algo: str, device: str):
    """The offline inference session for ``algo`` — the one place the three differ."""
    if algo == "diffusion":
        from fuji.s1_policy.inference.diffusion import DiffusionInferenceSession as S
    elif algo == "flow":
        from fuji.s1_policy.inference.flow import FlowInferenceSession as S
    else:
        from fuji.s1_policy.inference.act import ACTInferenceSession as S
    return S(checkpoint, device=device)



if __name__ == "__main__":
    raise SystemExit(main())
