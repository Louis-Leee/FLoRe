#!/usr/bin/env python3
"""Run the pure-holosoma HolosomaWBTPolicy headless over a reference NPZ.

Equivalent to::

    python -m wbt_wrappers_inference.run_policy \\
        inference:g1-29dof-holosoma-wbt \\
        --task.model-path /path/to/model.onnx \\
        --task.ref-motion-path /path/to/ref_motion.npz \\
        --task.headless-sim-eval --task.headless-max-frames N

but without tyro CLI parsing (the pinned bazel-env tyro lacks
`CascadeSubcommandArgs`), so we construct the InferenceConfig directly
from the shipped preset and override the two task fields we need.

Produces a debug-log NPZ per HOLOSOMA_DEBUG_LOG (the policy's own
instrumentation) containing per-frame action / dof_pos / timestep.
That stream is exactly what three_panel_eval --holosoma-reference
consumes.
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import os
import platform
import sys
from pathlib import Path


def _setup_environment() -> None:
    """Runtime-only env tweaks called from ``main()``.

    Kept out of module scope so importing this file as a library (tests,
    downstream tooling) does not mutate the caller's process: we set
    ``MUJOCO_GL`` for offscreen rendering and monkey-patch
    ``tyro.conf`` to paper over a name change
    (``CascadeSubcommandArgs`` → ``ConsolidateSubcommandArgs``) in the
    bazel-pinned tyro release.

    This monkey-patch survives for the lifetime of the process; do NOT
    import this module from libraries that expect a pristine tyro.conf.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

    import tyro.conf as _tyro_conf

    if not hasattr(_tyro_conf, "CascadeSubcommandArgs") and hasattr(_tyro_conf, "ConsolidateSubcommandArgs"):
        _tyro_conf.CascadeSubcommandArgs = _tyro_conf.ConsolidateSubcommandArgs


def main() -> int:
    _setup_environment()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="g1-29dof-holosoma-wbt")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--ref-motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Where the policy's internal debug-log NPZ is saved.")
    p.add_argument("--max-frames", type=int, default=0, help="0 = until ref-motion ends (no hard cap).")
    p.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="MJCF scene XML. Defaults to the shipped holosoma core scene_g1_29dof_wbt_plane.xml.",
    )
    args = p.parse_args()

    # Activate the policy's internal debug capture.
    os.environ["HOLOSOMA_DEBUG_LOG"] = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Resolve the preset ────────────────────────────────────────────
    from holosoma_inference.config.config_values.inference import DEFAULTS as CORE_DEFAULTS
    from wbt_wrappers_inference.config_values.inference import DEFAULTS as EXT_DEFAULTS

    presets = {**CORE_DEFAULTS, **EXT_DEFAULTS}
    if args.preset not in presets:
        print(f"error: preset {args.preset!r} not in available: {sorted(presets)}", file=sys.stderr)
        return 2
    inf_cfg = presets[args.preset]

    # ── Override task fields ──────────────────────────────────────────
    # headless_output is required even though we only care about the
    # debug log — the factory writes an eval video there (MUJOCO_GL
    # fallback may skip rendering but the field must be non-empty).
    headless_out = args.out.with_suffix(".headless.mp4")
    task_overrides = dict(
        model_path=str(args.onnx),
        ref_motion_path=str(args.ref_motion),
        headless_sim_eval=True,
        headless_output=str(headless_out),
        headless_debug_log=str(args.out),
    )
    if args.max_frames > 0:
        task_overrides["headless_max_frames"] = int(args.max_frames)
    if args.xml is not None:
        task_overrides["headless_xml"] = str(args.xml)
    # MuJoCo backend — don't touch real hardware.
    os.environ["HOLOSOMA_ROBOT_BACKEND"] = "mujoco"
    os.environ["HOLOSOMA_MUJOCO_REAL_TIME"] = "0"

    inf_cfg = _dc.replace(inf_cfg, task=_dc.replace(inf_cfg.task, **task_overrides))

    # ── Build policy + drive it ───────────────────────────────────────
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy
    from wbt_wrappers_inference.utils.interface import _create_headless_policy

    print(f"Constructing HolosomaWBTPolicy (preset={args.preset}) ...", flush=True)
    policy = _create_headless_policy(HolosomaWBTPolicy, inf_cfg)

    # Drive the policy via the same _run_headless_eval run_policy uses
    # for --task.headless-sim-eval. It runs exactly max_frames ticks
    # and writes the debug-log NPZ as part of its pipeline.
    from wbt_wrappers_inference.run_policy import _run_headless_eval

    print("Running headless eval ...", flush=True)
    try:
        _run_headless_eval(policy, policy.config, policy._headless_sim)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            policy._save_debug_log()
        except Exception as exc:
            print(f"warning: debug-log flush failed: {exc!r}", file=sys.stderr)

    if not args.out.is_file():
        print(f"error: expected debug log at {args.out} not produced", file=sys.stderr)
        return 3
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
