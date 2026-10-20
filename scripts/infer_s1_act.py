#!/usr/bin/env python3
"""Run deterministic offline inference and viser visualization for a DP-ACT checkpoint.

The selected index is an exact HLDataset sample from the checkpoint's train/validation
split. The viewer compares the native-layout prediction with that sample's masked target
on the fixed t0-inclusive 30 Hz action grid.

ACT is a single-pass CVAE: there is no sampler loop, so there is no ``--num-steps`` here
(the sibling flow/diffusion CLIs expose one) and ``--seed`` cannot change the output
unless the checkpoint was trained with ``act.sample_latent_at_inference``. Both are
deliberate: a flag that silently did nothing would be worse than its absence.
"""

from __future__ import annotations

import argparse
import json

from flore.s1_policy.inference.act import ACTInferenceSession
from flore.s1_policy.inference import keypoint_humanoid, mc_humanoid, ready_pose
from flore.s1_policy.inference.common import result_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="checkpoint directory containing config.json and model.safetensors")
    parser.add_argument(
        "--zarr-path",
        action="append",
        default=None,
        help="dataset path override; repeat in the checkpoint's original file order",
    )
    parser.add_argument("--device", default="cuda", help="torch device, for example cuda, cuda:1, or cpu")
    parser.add_argument("--sample-index", type=int, default=0, help="initial exact HLDataset sample index")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="latent seed. NO EFFECT unless the checkpoint set act.sample_latent_at_inference "
             "(the default decodes from the prior MEAN z=0, which is deterministic)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="viser bind host")
    parser.add_argument("--port", type=int, default=8089, help="viser port")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run one headless rollout, print JSON, and exit without importing viser",
    )
    # Opt-in whole-body motion completion. Defined in ONE place for all three
    # entry points (flore.s1_policy.inference.mc_humanoid) because these flags decide
    # what a MODEL is fed, and three drifting copies would make two runs of "the same"
    # completion incomparable. Absent the flag nothing about the page changes.
    mc_humanoid.add_cli_args(parser)
    # Its `fivepoint_geom_*` sibling: five predicted keypoints -> a G1 body by per-limb IK, no
    # second model. Defined here for the same reason the completion's flags are — one place, so
    # two runs of "the same" body cannot differ by a flag default.
    keypoint_humanoid.add_cli_args(parser)
    # --display-frame lives with it: it decides WHERE the chunk is placed, which is
    # the completion's single biggest input, and --motion-completion implies the
    # ready-pose frame (see ready_pose.options_from_args).
    ready_pose.add_cli_args(parser)
    args = parser.parse_args(argv)

    session = ACTInferenceSession(
        args.checkpoint,
        zarr_paths=None if args.zarr_path is None else tuple(args.zarr_path),
        device=args.device,
    )
    try:
        if args.self_test:
            result = session.infer(args.sample_index, seed=args.seed)
            print(json.dumps(result_summary(result), indent=2))
            return 0
        from flore.s1_policy.inference.viewer import run_viewer

        run_viewer(
            session,
            host=args.host,
            port=args.port,
            sample_index=args.sample_index,
            seed=args.seed,
            motion_completion=mc_humanoid.options_from_args(args),
            keypoint_ik=keypoint_humanoid.options_from_args(args),
            **ready_pose.options_from_args(args),
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
