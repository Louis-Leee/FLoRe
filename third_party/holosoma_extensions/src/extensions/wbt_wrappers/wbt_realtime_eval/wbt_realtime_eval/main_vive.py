"""Live VIVE-teleop build entry (the parallel live path).

Builds an mjswan bundle for the ``g1-29dof-holosoma-2pt-wbt-live`` preset, whose
``vr_2point_pos`` / ``vr_2point_orn`` obs terms are fed LIVE from two VIVE
trackers over a WebSocket (served by ``wbt_realtime_eval.vive.vive_teleop_bridge``)
instead of the baked motion table. Every other obs term is identical to the
replay path: ``command_lower_body`` / ``motion_ref_ori_b`` come from the injected
``idle_default`` standing clip, and ``projected_gravity`` / ``base_ang_vel`` /
``joint_pos`` / ``joint_vel`` / ``actions`` from live sim state.

This reuses ``main.setup_builder`` wholesale (ONNX strip, yam scene auto-derive,
normalizer extras, idle_default bake) — the ONLY additions are (1) the live
preset and (2) per-term config (``vive_ws_url`` + the default-pose ``home``
fallback) injected via ``setup_builder(per_key_params=...)``. The replay
``wbt-realtime-eval`` entry is completely unaffected.

Usage (build + launch the viz)::

    wbt-vive-teleop \\
        --model-path logs/WBT-Holosoma/8k8unjkw/model_18500.onnx \\
        --motion-dir ~/data/sonic_manipulation_test \\
        --vive-ws-url ws://localhost:8770

Then start the tracker bridge in a separate shell (it holds the pysurvive
singleton)::

    wbt-vive-teleop-bridge --left-serial LHR-xxxx --right-serial LHR-yyyy

``--motion-dir`` is still required: the live path needs at least one motion
(``idle_default`` is auto-injected) to source the non-VR obs terms. Any motions
there just populate the picker; the default ``idle_default`` clip is what holds
the lower body standing while the live hands drive the arms.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from loguru import logger

import mjswan  # noqa: F401  (ensures mjswan import side effects match main.py)

# Import the submodule explicitly: `wbt_realtime_eval.__init__` re-exports the
# `main` FUNCTION under that name, which would shadow `from wbt_realtime_eval
# import main` (giving the function, not the module). `setup_builder` lives in
# the main module.
from wbt_realtime_eval.main import _read_onnx_metadata, setup_builder
from wbt_realtime_eval.vive import load_home_anchor

_LIVE_PRESET = "g1-29dof-holosoma-2pt-wbt-live"
# The 588-D force variant: live VR hands PLUS the F^cmd/F^ext sliders. Selected
# by the explicit --force flag (NOT auto-detected: obs dim 588 is shared by the
# force preset and the 3pt-global preset, so the dim alone cannot disambiguate
# the term layout — see presets._HOLOSOMA_2PT_FORCE_WBT_LIVE_TERMS).
_LIVE_FORCE_PRESET = "g1-29dof-holosoma-2pt-force-wbt-live"


def validate_home_anchor_matches_meta(meta: dict[str, str], *, source: str) -> None:
    """``_validate_home_anchor_matches_checkpoint`` with the ONNX metadata already
    read, so a caller that needs several metadata-driven checks pays the (multi-
    hundred-MB) ORT load once. ``source`` names the checkpoint in the error.

    Split out for :mod:`main_vive3` (the 3pt live entry), which runs this plus its
    own actor-term check. The pose being checked is the HANDS' default pose, which
    the 2pt and 3pt anchors share by construction (the 3pt anchor is the same bake
    FK on the same default_qpos), so both entries want exactly this comparison."""
    anchor = load_home_anchor()
    anchor_qpos = anchor.get("default_qpos")
    anchor_dof = anchor.get("dof_names")
    if not anchor_qpos or not anchor_dof:
        logger.warning("home_anchor.json lacks default_qpos/dof_names; skipping pose check.")
        return
    exp = json.loads(meta.get("experiment_config", "{}"))
    angles = exp.get("robot", {}).get("init_state", {}).get("default_joint_angles", {})
    if not angles:
        logger.warning("ONNX has no default_joint_angles; skipping HOME-anchor pose check.")
        return
    ckpt_qpos = np.array([float(angles.get(n, 0.0)) for n in anchor_dof], dtype=np.float64)
    if not np.allclose(np.asarray(anchor_qpos, dtype=np.float64), ckpt_qpos, atol=1e-3):
        raise ValueError(
            f"home_anchor.json (computed for run {anchor.get('source_run')}/"
            f"{anchor.get('source_model')}) has a default pose that differs from "
            f"{source}'s default_joint_angles (max abs diff "
            f"{float(np.max(np.abs(np.asarray(anchor_qpos) - ckpt_qpos))):.4f} rad). The live "
            f"HOME / calibration target would be wrong. Recompute vive/home_anchor.json for "
            f"this checkpoint, or pass the matching checkpoint."
        )
    logger.info("HOME anchor default pose matches the checkpoint (within 1e-3 rad).")


def _validate_home_anchor_matches_checkpoint(model_path: Path) -> None:
    """Fail loudly if home_anchor.json was computed for a DIFFERENT default pose
    than the supplied checkpoint.

    The HOME anchor (live calibration target + pre-first-message fallback) is
    keyed to the robot's default-pose FK geometry, so it stays valid across
    checkpoints of the SAME task. But a different-task checkpoint can have a
    different ``default_joint_angles`` → the live hands would calibrate/fall back
    to the wrong pose, silently offsetting or collapsing the arms. We compare the
    committed anchor's ``default_qpos`` against the ONNX-baked
    ``default_joint_angles`` and refuse the build on a mismatch (consistent with
    the package's other fail-loud checks: normalizer / dof_names / scene)."""
    validate_home_anchor_matches_meta(_read_onnx_metadata(model_path), source="this checkpoint")


def _checkpoint_force_cmd_hand_frame(model_path: Path) -> bool | None:
    """Read whether the checkpoint's ACTOR was trained with hand-frame F^cmd.

    Holosoma bakes the full training config into the ONNX ``experiment_config``
    metadata. The actor's force obs term carries ``params={"hand_frame": True}``
    for a ``--F_on_hand_frame`` run and ``params={}`` (pelvis) otherwise; the
    critic's term is always pelvis (privileged value function in a stable frame),
    so we read the ACTOR group specifically. Returns the trained flag, or None if
    the metadata has no force_cmd_local actor term (e.g. a non-force checkpoint)."""
    meta = _read_onnx_metadata(model_path)
    exp = json.loads(meta.get("experiment_config", "{}"))
    actor_terms = exp.get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
    for name, term in actor_terms.items():
        # The actor force term is named obs3_force_cmd_local; match on the func to
        # be robust to the obsN_ numeric prefix.
        if "force_cmd_local" in str(name) or str(term.get("func", "")).endswith(":force_cmd_local"):
            return bool((term.get("params") or {}).get("hand_frame", False))
    return None


def _validate_force_cmd_frame_matches_checkpoint(model_path: Path, *, force: bool, force_cmd_hand_frame: bool) -> None:
    """Fail loudly if the operator's frame choice contradicts the checkpoint.

    A force checkpoint trained with hand-frame F^cmd fed pelvis-frame F^cmd at
    deploy (or vice-versa) is NOT a build error — the dims match — but the actor
    sees an out-of-distribution obs, silently degrading tracking. We cross-check
    the trained ``hand_frame`` flag (from ONNX ``experiment_config``) against the
    selected mode, consistent with the package's other fail-loud guards
    (normalizer / dof_names / HOME pose). Only meaningful under ``--force`` (the
    plain-live preset has no force_cmd_local term)."""
    trained = _checkpoint_force_cmd_hand_frame(model_path)
    if not force:
        # Non-force build: warn (don't fail) if a force checkpoint was passed
        # without --force, since its force_cmd_local channel would be dropped.
        if trained is not None:
            logger.warning(
                "Checkpoint encodes a force_cmd_local actor term (a force checkpoint) but "
                "--force was not passed; building the plain 564-D live preset would DROP the "
                "force channel and mismatch the ONNX obs dim (the normalizer-length check will "
                "then fail). Pass --force for this checkpoint."
            )
        return
    if trained is None:
        logger.warning(
            "ONNX experiment_config has no force_cmd_local actor term; cannot verify the "
            "F^cmd frame. Proceeding with the selected mode "
            f"({'hand' if force_cmd_hand_frame else 'pelvis'})."
        )
        return
    if trained != force_cmd_hand_frame:
        selected = "hand" if force_cmd_hand_frame else "pelvis"
        trained_name = "hand" if trained else "pelvis"
        fix = "drop --pelvis-frame" if trained else "add --pelvis-frame"
        raise ValueError(
            f"F^cmd frame mismatch: this checkpoint was trained with the actor's force_cmd_local "
            f"in the {trained_name} frame, but the build selected the {selected} frame. Feeding the "
            f"actor an out-of-frame F^cmd is out-of-distribution and silently degrades tracking. "
            f"To match the checkpoint, {fix}."
        )
    logger.info(f"F^cmd frame matches the checkpoint (actor trained in the {'hand' if trained else 'pelvis'} frame).")


def _live_per_key_params(ws_url: str) -> dict[str, dict]:
    """Per-term config for the live VR obs classes: the WebSocket URL the
    browser connects to, plus the default-pose HOME fallback each class returns
    until the first live frame arrives (NEVER zeros — zeros collapse the arms).

    HOME is the flattened [L, R] default-pose 2pt target from
    ``vive/home_anchor.json`` (pos 6, orn rot6d 12), byte-for-byte matching what
    the bridge calibrates onto and emits.
    """
    anchor = load_home_anchor()
    home_pos = [v for row in anchor["home_pos_pelvis"] for v in row]  # 6
    home_orn = [v for row in anchor["home_orn_rot6d"] for v in row]  # 12
    return {
        "vr_2point_pos_live": {"vive_ws_url": ws_url, "home": home_pos},
        "vr_2point_orn_live": {"vive_ws_url": ws_url, "home": home_orn},
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="wbt-vive-teleop")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--motion-dir",
        required=True,
        type=Path,
        help="Reference motions (still required — supplies the non-VR obs terms "
        "via the injected idle_default clip; any clips here populate the picker).",
    )
    parser.add_argument("--output-dir", default=Path("dist/wbt_realtime_eval_vive"), type=Path)
    parser.add_argument(
        "--vive-ws-url",
        default="ws://localhost:8770",
        help="WebSocket URL of the VIVE bridge the browser connects to "
        "(must match wbt-vive-teleop-bridge --host/--port).",
    )
    parser.add_argument(
        "--scene-xml",
        default=None,
        type=Path,
        help="MuJoCo render scene (absolute or repo-root-relative). If omitted, "
        "auto-derived from the ONNX robot_urdf_path (yam checkpoint -> yam scene).",
    )
    parser.add_argument("--no-launch", action="store_true", help="Build only; don't start the server.")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Use the 588-D force preset (g1-29dof-holosoma-2pt-force-wbt-live): "
            "adds the 12 F^cmd/F^ext sliders and feeds the hand-frame F^cmd obs ON "
            "TOP of live VR. Requires a 588-D force checkpoint (e.g. the "
            "--F_on_hand_frame run 09wvlz16). Without it, the default 564-D "
            "non-force live bundle is built byte-identically to before."
        ),
    )
    parser.add_argument(
        "--pelvis-frame",
        action="store_true",
        help=(
            "With --force, feed F^cmd in the PELVIS frame instead of the hand "
            "frame. Only for a pelvis-frame force checkpoint; the default under "
            "--force is hand-frame, matching the --F_on_hand_frame run 09wvlz16. "
            "Ignored without --force (force_cmd_local isn't in the plain-live obs)."
        ),
    )
    parser.add_argument(
        "--show-control-point",
        action="store_true",
        help=(
            "Draw an orange sphere at the exact point each tracker steers — the "
            "policy's vr_2point CONTACT point (wrist_yaw + R(wrist) @ the 0.18 m "
            "VR lever), where the live hands are calibrated. Pure viz (no obs / "
            "physics / command change); opt-in, byte-identical when off."
        ),
    )
    args = parser.parse_args()

    # Explicit-flag preset selection (never obs-dim auto-detect — 588 is shared by
    # the force and 3pt-global layouts). Default (no --force) keeps the plain-live
    # path byte-identical to before: same preset, no force_cmd_hand_frame kwarg.
    preset = _LIVE_FORCE_PRESET if args.force else _LIVE_PRESET
    # Frame of the F^cmd obs (force preset only). 09wvlz16 is a HANDFRAME
    # checkpoint, so --force defaults to hand-frame; --pelvis-frame overrides.
    force_cmd_hand_frame = args.force and not args.pelvis_frame
    if args.pelvis_frame and not args.force:
        logger.warning("--pelvis-frame has no effect without --force (the plain-live preset has no F^cmd term).")

    logger.info(
        f"Live VIVE teleop build: preset={preset} ws={args.vive_ws_url}"
        + (f" (force sliders, F^cmd frame={'hand' if force_cmd_hand_frame else 'pelvis'})" if args.force else "")
    )
    # Refuse to build if the committed HOME anchor doesn't match this checkpoint's
    # default pose (would silently mis-target the live hands). Pose-keyed and
    # frame/force-agnostic, so it guards the 588-D force checkpoint too.
    _validate_home_anchor_matches_checkpoint(args.model_path.resolve())
    # Refuse to build if the selected F^cmd frame contradicts what the checkpoint
    # was trained on (hand vs pelvis) — a silent OOD obs otherwise. Reads the
    # trained hand_frame flag from ONNX experiment_config.
    _validate_force_cmd_frame_matches_checkpoint(
        args.model_path.resolve(), force=args.force, force_cmd_hand_frame=force_cmd_hand_frame
    )
    builder = setup_builder(
        preset=preset,
        model_path=args.model_path.resolve(),
        motion_dir=args.motion_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        scene_xml=args.scene_xml,
        per_key_params=_live_per_key_params(args.vive_ws_url),
        force_cmd_hand_frame=force_cmd_hand_frame,
        show_control_point=args.show_control_point,
    )
    app = builder.build(output_dir=str(args.output_dir.resolve()))
    logger.info(
        "Built. Start the tracker bridge in another shell: "
        "`wbt-vive-teleop-bridge --left-serial <LHR> --right-serial <LHR>` "
        "(hold the calibration pose ~3s), then watch the browser."
    )
    if args.no_launch or os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
