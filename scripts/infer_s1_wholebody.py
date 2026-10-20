#!/usr/bin/env python3
"""OFFLINE inference + viser page for a ``diffusion_umi_wholebody`` checkpoint (29 G1 joints).

The wholebody sibling of ``infer_s1_diffusion.py`` / ``infer_s1_flow.py`` / ``infer_s1_act.py``,
and separate from all three for a structural reason rather than a stylistic one: those pages draw
TRAJECTORIES (a blue target polyline, an orange prediction, gripper clusters, force arrows on a
path) and this layout has no position or rotation columns at all — ``layout.pose_dims()`` is
empty. What a wholebody chunk carries is a POSTURE per step, so the page draws the robot::

    🟪  violet body  = the policy's 29 predicted joint angles at the selected action step
    ⬜  pale body    = the dataset's own (masked) target joints at that step
        charts       = per-joint error in degrees, both grippers, both wrenches

One session serves all three algorithm blocks (``diffusion`` / ``flow`` / ``act``) because
``wholebody`` is a LAYOUT and the algorithm is a field in ``config.json`` — read from there by
default, so neither has to be typed.

    ./scripts/run_infer_s1_wholebody.sh checkpoints/ckpt_0831_wb            # the launcher
    uv run scripts/infer_s1_wholebody.py <ckpt> --self-test                 # headless JSON
    uv run scripts/infer_s1_wholebody.py <ckpt> --sample-index 42 --port 8088

⚠️ NOT a deploy preview. These are ABSOLUTE joints in the dataset's ``cond_mode="home"`` posture
space; deploy publishes ``base + (q[i] - q[0])`` through a joint clutch and never this absolute
pose (measured 0.514 rad apart at the knee from the ready-pose crouch). See
``docs/RUNBOOK_hl_serving_wholebody.md`` and ``scripts/hl_serving_wholebody.py``.
"""

from __future__ import annotations

import argparse
import json

from flore.s1_policy.inference.wholebody import (
    ALGORITHMS,
    WholeBodyInferenceSession,
    wholebody_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint",
                        help="checkpoint dir with config.json + model.safetensors "
                             "(hl_data.mode must be 'wholebody')")
    parser.add_argument("--algo", default=None, choices=list(ALGORITHMS),
                        help="which algorithm block the checkpoint carries. Default: read from "
                             "its own config.json — 'wholebody' is a LAYOUT, not an algorithm, "
                             "so this normally never has to be passed.")
    parser.add_argument("--zarr-path", action="append", default=None,
                        help="dataset path override; repeat in the checkpoint's original order")
    parser.add_argument("--device", default="cuda", help="torch device (cuda, cuda:1, cpu)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="sampler steps; default comes from the checkpoint (ACT pins 1)")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="initial exact HLDataset sample index")
    parser.add_argument("--seed", type=int, default=0, help="sampling noise seed")
    parser.add_argument("--host", default="127.0.0.1", help="viser bind host")
    parser.add_argument("--port", type=int, default=8088, help="viser port")
    parser.add_argument("--yam-urdf", default=None,
                        help="URDF for the two drawn bodies (default: this repo's "
                             "third_party/UMI_collection/robot_assets/g1_yam/g1_29dof_yam.urdf). "
                             "Absent, the charts and metrics still work and the page says so.")
    parser.add_argument("--self-test", action="store_true",
                        help="run ONE headless rollout, print JSON, exit — imports no viser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = WholeBodyInferenceSession(
        args.checkpoint,
        algorithm=args.algo,
        zarr_paths=None if args.zarr_path is None else tuple(args.zarr_path),
        device=args.device,
        num_steps=args.num_steps,
    )
    try:
        if args.self_test:
            print(json.dumps(wholebody_summary(session.infer(args.sample_index, seed=args.seed)),
                             indent=2))
            return 0
        from flore.s1_policy.inference.wholebody_viewer import run_wholebody_viewer

        run_wholebody_viewer(session, host=args.host, port=args.port,
                             sample_index=args.sample_index, seed=args.seed,
                             urdf_path=args.yam_urdf)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
