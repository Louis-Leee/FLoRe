#!/usr/bin/env python3
"""Time the motion-completion post-processes on the DEPLOY path, per chunk and per step.

The question this answers: **can the real robot afford foot IK?** The planner
(``scripts/hl_serving_mc.py``) completes a whole 50-step chunk between one Predict and the
first published step, so the completion's cost is not amortised over the chunk — it is
latency the operator waits through, and (once running) it is latency that must fit inside the
motion the previous chunk still has left to execute.

The budget, stated once:

  * a chunk is ``H`` steps at 30 Hz = ``H/30`` s of motion;
  * the planner re-plans every ``--exec-steps`` (default 16) steps = **533 ms**;
  * the sampler runs first, so the completion gets ``533 ms − sample_ms``.

Anything past that and the plan arrives after the steps it was supposed to replace: the drip
runs off the end of the old chunk and the robot holds still mid-task.

WHAT IS ABLATED. Only the post-processes, and only one at a time. The chunk is predicted
**once** with the real checkpoint and every arm re-completes those SAME conditioning frames,
so the model, the frames and the contact pattern are identical across the table and the only
difference is the pass under test:

  * ``wrist IK`` — the per-window JAX pass that pulls the wrists onto the commanded poses.
    Cheap, and it is what makes the 0.8 mm hand accuracy the deploy contract quotes.
  * ``foot IK`` — the causal ``correct_g1_qpos_inertialize`` leg pass: locks a stance foot's
    XY and re-solves the legs with ``ik_iters`` Adam steps of differentiable FK. Cost is
    linear in iterations, so the sweep is over iterations.

Both accuracy columns are reported alongside the cost, because a pass that is free and
changes nothing is not a saving and a pass that is expensive and moves the HANDS is not an
option. Run:

    .venv/bin/python scripts/ablate_mc_foot_ik.py ckpt_0831_final --reps 3
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Arms, in the order they are printed. ``None`` iterations = leave the backend's own default.
ARMS: tuple[tuple[str, bool, str, int | None], ...] = (
    ("raw (no post-process)", False, "none", None),
    ("wrist IK only  ← deploy default", True, "none", None),
    ("wrist + foot IK  40 iters", True, "inertialize", 40),
    ("wrist + foot IK  80 iters", True, "inertialize", 80),
    ("wrist + foot IK 150 iters (MC default)", True, "inertialize", 150),
)


def _ankle_slide(chunk, body_names) -> float:
    """Median planar ankle speed (m/s) — what the foot pass exists to reduce."""
    idx = [body_names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
    v = np.concatenate([np.linalg.norm(np.diff(chunk.fk_pos[:, k, :2], axis=0), axis=-1)
                        * 30.0 for k in idx])
    return float(np.median(v))


#: Joint-name substrings for the two halves of the body the clutch publishes. The split
#: matters because the foot pass only rewrites the LEGS, and under the joint clutch a leg
#: rewrite is a leg COMMAND on a gantry-suspended robot.
_LEG_KEYS = ("hip", "knee", "ankle")


def _travel(chunk, want_legs: bool) -> tuple[float, float]:
    """``(max |q - q[0]|, worst per-step |dq|)`` over the leg or the non-leg joints."""
    cols = [i for i, n in enumerate(chunk.joint_names)
            if any(k in n for k in _LEG_KEYS) == want_legs]
    q = chunk.joint_pos[:, cols]
    return (float(np.abs(q - q[0]).max()), float(np.abs(np.diff(q, axis=0)).max()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("--algo", default="diffusion", choices=("diffusion", "flow", "act"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=3, help="timed repetitions per arm (median)")
    p.add_argument("--exec-steps", type=int, default=16,
                   help="the planner's re-plan cadence, for the duty column")
    p.add_argument("--arms", default="", help="comma-separated substrings; default = all")
    args = p.parse_args(argv)
    arms = [a for a in ARMS
            if not args.arms or any(s.strip().lower() in a[0].lower()
                                    for s in args.arms.split(",") if s.strip())]
    if not arms:
        print(f"[ablate] --arms {args.arms!r} matched none of "
              f"{[a[0] for a in ARMS]}")
        return 2

    import hl_serving_mc as M

    from fuji.s1_policy.data.layout import ACTION_DT
    from fuji.s1_policy.inference import ready_pose as RP
    from fuji.s1_policy.serving import hl_dense
    from fuji.s1_policy.serving.hl_online import LivePlanner, load_hl_model
    from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

    loaded = load_hl_model(args.checkpoint, args.algo, device=args.device, max_cameras=2)
    if tuple(loaded.layout.roles) != M.REQUIRED_ROLES:
        print(f"[ablate] not a 3-tracker checkpoint: roles={loaded.layout.roles}")
        return 1
    eef_t_wrist, eef_R_wrist = M._resolve_extrinsic(args.checkpoint)
    anchor, why = RP.resolve_deploy_anchor(loaded.layout.roles)
    if anchor is None:
        print(f"[ablate] no deploy ready pose — {why}")
        return 1

    # ---- ONE prediction, reused by every arm -------------------------------------
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

    t0 = time.perf_counter()
    chunk = planner.infer_chunk(seed=args.seed, force_enabled=False, gate_n=1.5)
    sample_cold_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    planner.infer_chunk(seed=args.seed, force_enabled=False, gate_n=1.5)
    sample_warm_ms = (time.perf_counter() - t0) * 1e3
    decoded = {rc.role: rc for rc in chunk.roles}
    H = int(decoded["left"].position.shape[0])
    print(f"[ablate] chunk H={H} @ {1.0 / ACTION_DT:g} Hz = {H * ACTION_DT * 1e3:.0f} ms of "
          f"motion · sampler {sample_cold_ms:.0f} ms cold / {sample_warm_ms:.0f} ms warm")

    completer = hl_dense.ChunkCompleter(log=lambda m: print(f"[ablate][mc] {m}", flush=True))
    if completer.disabled_reason or not completer.start():
        print(f"[ablate] motion completion unavailable — {completer.disabled_reason}")
        return 1

    try:
        cond = completer.build_cond(
            left_pos=decoded["left"].position,
            left_rotmat=rot6d_to_rotmat(decoded["left"].rot6d),
            right_pos=decoded["right"].position,
            right_rotmat=rot6d_to_rotmat(decoded["right"].rot6d),
            head_pos=decoded["head"].position,
            head_rotmat=rot6d_to_rotmat(decoded["head"].rot6d),
            eef_t_wrist=eef_t_wrist, eef_R_wrist=eef_R_wrist, dt=ACTION_DT,
            cond_mode="absolute", dz_wrist=0.0)
        print(f"[ablate] windows={len(cond.starts)} starts={list(cond.starts)}  "
              f"reps={args.reps}")

        rows = []
        for label, wrist_ik, foot_fix, iters in arms:
            completer.set_post_process(wrist_ik=wrist_ik, foot_fix=foot_fix,
                                       foot_ik_iters=iters)
            done, samples = None, []
            for rep in range(args.reps + 1):          # rep 0 = warm-up (JIT, cache)
                done = completer.complete(cond)
                if rep:
                    samples.append(done)
            ms = statistics.median(d.ms for d in samples)
            stages = {k: statistics.median(d.stage_ms()[k] for d in samples)
                      for k in ("flow", "decode", "ik", "foot", "transport")}
            # Per-window, from the slowest rep's own report: the two windows are NOT equal
            # (window 1's foot pass re-examines its look-back, so it covers more frames).
            per_window = [tuple(round(float(t.get(k, 0.0) or 0.0), 1)
                                for k in ("t_total_ms", "t_foot_ms"))
                          for t in samples[-1].timings]
            rows.append({
                "label": label, "ms": ms, **stages,
                "per_step": ms / H,
                "wrist_mm": done.wrist_error_m() * 1e3,
                "slide": _ankle_slide(done, list(done.body_names)),
                "pelvis": (float(done.root_pos[:, 2].min()), float(done.root_pos[:, 2].max())),
                "worst_dq": float(np.abs(np.diff(done.joint_pos, axis=0)).max()),
                "leg": _travel(done, True), "arm": _travel(done, False),
                "per_window": per_window,
            })
            print(f"  {label:<40} {ms:8.1f} ms   foot {stages['foot']:8.1f}   "
                  f"wrist {rows[-1]['wrist_mm']:6.2f} mm   slide {rows[-1]['slide']:.4f}   "
                  f"per-window total/foot {per_window}")
    finally:
        completer.stop()

    # ---- the table ---------------------------------------------------------------
    replan_ms = args.exec_steps / (1.0 / ACTION_DT) * 1e3
    budget_ms = replan_ms - sample_warm_ms
    print(f"\n=== per-CHUNK cost (H={H}, {len(cond.starts)} windows, median of {args.reps}) ===")
    print(f"{'arm':<40} {'chunk':>9} {'/step':>8} {'flow':>8} {'wristIK':>8} {'footIK':>9} "
          f"{'duty':>7}")
    for r in rows:
        print(f"{r['label']:<40} {r['ms']:8.1f}ms {r['per_step']:7.2f}ms {r['flow']:7.1f}ms "
              f"{r['ik']:7.1f}ms {r['foot']:8.1f}ms {100 * r['ms'] / budget_ms:6.0f}%")
    print(f"\nbudget: re-plan every {args.exec_steps} steps = {replan_ms:.0f} ms, minus the "
          f"{sample_warm_ms:.0f} ms sampler = {budget_ms:.0f} ms for the completion "
          f"(duty >100% = the plan lands late).")

    print("\n=== what it buys / costs ===")
    print(f"{'arm':<40} {'wrist err':>10} {'ankle slide':>12} {'pelvis z':>16} {'worst dq':>10}")
    for r in rows:
        print(f"{r['label']:<40} {r['wrist_mm']:9.2f}mm {r['slide']:11.4f} "
              f"{r['pelvis'][0]:7.3f}..{r['pelvis'][1]:<7.3f} {r['worst_dq']:9.4f}")

    # Under the joint clutch the published command IS this travel, so a pass that rewrites the
    # legs is a pass that commands the legs — on a gantry-suspended robot that is the column
    # to read before the millisecond ones.
    print("\n=== what the CLUTCH would publish (travel from step 0, rad) ===")
    print(f"{'arm':<40} {'legs max':>9} {'legs/step':>10} {'rest max':>9} {'rest/step':>10}")
    for r in rows:
        print(f"{r['label']:<40} {r['leg'][0]:8.4f} {r['leg'][1]:9.4f} "
              f"{r['arm'][0]:8.4f} {r['arm'][1]:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
