#!/usr/bin/env python3
"""Policy Runner with TML/BM WBT wrapper support.

This is a self-contained run_policy that mirrors the holosoma_inference
run_policy logic but uses the wrapper's own policy classes and configs.

Usage:
    python -m wbt_wrappers_inference.run_policy inference:tml-wbt \
        --task.model-path path/to/tml_policy.onnx \
        --task.ref-motion-path path/to/ref_motion.npz

    python -m wbt_wrappers_inference.run_policy inference:bm-wbt \
        --task.model-path path/to/bm_policy.onnx
"""

from __future__ import annotations

import sys
import traceback
from importlib.metadata import entry_points

import tyro
from loguru import logger
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.misc import restore_terminal_settings

from wbt_wrappers_inference.config_values.inference import DEFAULTS as EXT_DEFAULTS
from wbt_wrappers_inference.policies import BMWholeBodyTrackingPolicy, TMLCHIPPolicy, TMLWholeBodyTrackingPolicy
from wbt_wrappers_inference.policies.holosoma_maskedmimic import HolosomaMaskedMimicPolicy
from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy
from wbt_wrappers_inference.utils.interface import register_interface, _create_headless_policy

_POLICY_TYPE_MAP = {
    "locomotion": LocomotionPolicy,
    "wbt": WholeBodyTrackingPolicy,
    "tml_wbt": TMLWholeBodyTrackingPolicy,
    "tml_chip": TMLCHIPPolicy,
    "bm_wbt": BMWholeBodyTrackingPolicy,
    "holosoma_wbt": HolosomaWBTPolicy,
    "holosoma_maskedmimic": HolosomaMaskedMimicPolicy,
}


def _select_policy_class(config: InferenceConfig):
    """Select policy class from explicit policy_type or observation heuristic."""
    policy_type = getattr(config.task, "policy_type", None)
    if policy_type and policy_type in _POLICY_TYPE_MAP:
        return _POLICY_TYPE_MAP[policy_type]

    # Fall back to observation-based selection (for TaskConfig without policy_type field)
    actor_obs = config.observation.obs_dict.get("actor_obs", [])
    if "tml_command" in actor_obs:
        return TMLWholeBodyTrackingPolicy
    if "command" in actor_obs and "motion_ref_ori_b" in actor_obs:
        return BMWholeBodyTrackingPolicy
    if "motion_command" in actor_obs:
        return WholeBodyTrackingPolicy
    return LocomotionPolicy


def _build_annotated_config():
    """Build Tyro annotated config combining core and extension presets."""
    from holosoma_inference.config.config_values.inference import DEFAULTS as CORE_DEFAULTS

    all_defaults = {}
    all_defaults.update(CORE_DEFAULTS)
    for ep in entry_points(group="holosoma.config.inference"):
        if ep.name not in all_defaults:
            all_defaults[ep.name] = ep.load()
    all_defaults.update(EXT_DEFAULTS)

    return Annotated[
        InferenceConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"inference:{k}": v for k, v in all_defaults.items()}
            )
        ),
    ]


def _dump_config(config: InferenceConfig):
    """Print full config for debugging comparison."""
    logger.debug("--- BEGIN CONFIG DUMP ---")
    for section_name, section in [("robot", config.robot), ("observation", config.observation), ("task", config.task)]:
        logger.debug(f"[{section_name}]")
        for k in sorted(section.__dataclass_fields__):
            logger.debug(f"  {k} = {getattr(section, k)!r}")
    logger.debug("--- END CONFIG DUMP ---")


def _run_headless_eval(policy, config: InferenceConfig, mj_sim: dict):
    """Run headless eval loop: policy.policy_action() + rendering + metrics + video."""
    import json
    import os

    import numpy as np

    from wbt_wrappers_inference.utils.eval_utils import (
        BASE_AZIMUTH,
        MetricsLogger,
        _add_hand_force_arrows,
        _add_mm_body_markers,
        _add_vr_axis_markers,
        _get_vr_positions_and_rotations_from_mujoco,
        _make_renderer,
        _yaw,
        build_metrics_output,
        compose_and_save,
        extract_model_label,
        get_ref_marker_config,
        parse_force_local_env,
        render_reference,
    )

    task = config.task
    mj_model = mj_sim["model"]
    mj_data = mj_sim["data"]
    sim_params = mj_sim["sim_params"]
    metadata = mj_sim["metadata"]
    xml_path = mj_sim["xml_path"]
    ref_motion_path = mj_sim["ref_motion_path"]

    onnx_path, exp_config, meta_map = metadata
    obs_mode = sim_params["obs_mode"]
    model_label = extract_model_label(onnx_path, exp_config, meta_map)
    motion_label = os.path.splitext(os.path.basename(ref_motion_path))[0]
    # Pass the checkpoint's actor-obs terms so the 3pt HAND markers follow the geometry the
    # policy actually trained on: `vr_3point_pos_tcp` ⇒ the 0.242 m force TCP (the
    # TCP-unified exp:g1-wbt-3pt-hybrid-unified[-anneal]), else the 0.18 m VR lever.
    _actor_terms = (exp_config or {}).get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
    vr_bodies, vr_off, vr_viz_offsets, mm_body_names = get_ref_marker_config(obs_mode, _actor_terms)

    # When --task.headless-no-video is set we skip ALL rendering (reference +
    # policy frames + compose) and emit metrics JSON only. This is ~4x faster
    # and is the mode used for the force-benchmark sweeps; keep video for
    # spot-check clips.
    no_video = getattr(task, "headless_no_video", False)

    # Force-arrow overlay: read the SAME env vars the obs builder (F_cmd) and the
    # MuJoCo interface (F_ext) consume, so the drawn arrows are the exact vectors
    # in play. None when unset/all-zero → no glyph. Only needed when rendering.
    force_cmd_pelvis = None if no_video else parse_force_local_env("HOLOSOMA_FORCE_CMD_LOCAL")
    force_ext_pelvis = None if no_video else parse_force_local_env("HOLOSOMA_FORCE_EXT_LOCAL")

    # Render reference (left side) — pass the same replayed terrain OBJ as the
    # policy side (NOT the raw terrain NPY), so reference and policy render on
    # IDENTICAL obstacle placements. With a terrain NPY input,
    # _prepare_headless_replay_assets samples per-env obstacles and writes a
    # replayed OBJ — policy uses it, so reference must too.
    if no_video:
        ref_frames = []
        logger.info("Rendering disabled (--task.headless-no-video): metrics-only run")
    else:
        logger.info("Rendering reference...")
        ref_frames = render_reference(
            ref_motion_path,
            xml_path,
            task.headless_max_frames,
            vr_body_names=vr_bodies,
            vr_offsets=vr_off,
            vr_viz_offsets=vr_viz_offsets,
            mm_body_names=mm_body_names,
            fixed_camera=task.headless_fixed_camera,
            cam_distance=task.headless_cam_distance,
            terrain_obj=mj_sim.get("terrain_obj") or None,
        )
        logger.info(f"  {len(ref_frames)} reference frames")

    # Metrics logger
    ref = mj_sim["ref_npz"]
    ml = MetricsLogger(ref, mj_model)

    # Debug log
    if task.headless_debug_log:
        from collections import defaultdict

        policy._debug_log = defaultdict(list)

    # obs[0] sees motion[1] (training advances motion before _compute_observations).
    # `preload_motion_frame` primes motion_command/ref buffers at frame 1 AND seeds
    # _ext_frame_idx so policy_action picks up from there. Setting _ext_frame_idx
    # alone leaves the buffers at motion[0] — silent tracking regression.
    if hasattr(policy, "preload_motion_frame"):
        policy.preload_motion_frame(1)
    else:
        policy._ext_frame_idx = 1

    renderer = cam = None
    if not no_video:
        renderer, cam = _make_renderer(
            mj_model, fixed_camera=task.headless_fixed_camera, cam_distance=task.headless_cam_distance
        )
    init_yaw = _yaw(mj_data.qpos)

    # --- Simulation loop ---
    effective_max = ml.motion_length if task.headless_max_frames <= 0 else task.headless_max_frames
    frames = []
    n_steps = 0

    logger.info("Running policy...")
    for ticker in range(effective_max):
        mid = min(ticker, ml.motion_length - 1)

        # Metrics BEFORE action
        ml.log_frame(mid, mj_data)
        n_steps += 1

        # Render (pre-action state) — skipped entirely in no-video mode.
        if not no_video:
            if not task.headless_fixed_camera:
                cam.azimuth = BASE_AZIMUTH + np.degrees(_yaw(mj_data.qpos) - init_yaw)
            renderer.update_scene(mj_data, camera=cam)
            if vr_bodies:
                pos, rot = _get_vr_positions_and_rotations_from_mujoco(mj_model, mj_data, vr_bodies, vr_viz_offsets)
                _add_vr_axis_markers(renderer._scene, pos, rot)
            if mm_body_names:
                _add_mm_body_markers(renderer._scene, mj_model, mj_data, mm_body_names)
            # Force arrows LAST so they sit on top of the body/VR markers.
            _add_hand_force_arrows(
                renderer._scene, mj_model, mj_data, force_cmd_pelvis=force_cmd_pelvis, force_ext_pelvis=force_ext_pelvis
            )
            frames.append(renderer.render().copy())

        if mj_data.qpos[2] < 0.3:
            ml.mark_fallen()
            break

        # Inference + actuation AFTER metrics/render
        policy.policy_action()

    if renderer is not None:
        renderer.close()
    # In no-video mode `frames` is empty; survival/metrics use the step count.
    n_logged = len(frames) if not no_video else n_steps
    logger.info(f"  {n_logged} policy frames")

    # Finalize metrics
    metrics = ml.finalize(n_logged)
    if "summary" in metrics:
        s = metrics["summary"]
        logger.info(
            f"  Metrics: joint_err={s['mean_joint_angle_error_deg']:.1f}\u00b0 "
            f"body_err={s['mean_body_pos_error_cm']:.1f}cm "
            f"root_err={s['mean_root_pos_error_cm']:.1f}cm "
            f"survival={s['survival_ratio']:.0%}"
        )
    if "force_summary" in metrics and metrics["force_summary"]:
        fsx = metrics["force_summary"]
        if "force_realization_err_median_N_right" in fsx:
            logger.info(
                f"  Force(R): drift={fsx.get('force_drift_median_cm_right', 0):.2f}cm "
                f"e_F={fsx.get('force_realization_err_median_N_right', 0):.2f}N "
                f"(cmd |F|={fsx.get('force_cmd_mag_N_right', 0):.1f}N, "
                f"exp_drift={fsx.get('expected_drift_cm_right', 0):.2f}cm)"
            )

    # Debug log
    if task.headless_debug_log and hasattr(policy, "_debug_log") and policy._debug_log:
        save_dict = {k: np.array(v) for k, v in policy._debug_log.items() if v}
        np.savez(task.headless_debug_log, **save_dict)
        logger.info(f"Debug log: {task.headless_debug_log}")

    # Compose side-by-side video (skipped in no-video mode).
    output = task.headless_output
    if not no_video:
        ok = compose_and_save(
            ref_frames, frames, output, model_label=model_label or "", motion_label=motion_label, metrics=metrics
        )
        if ok:
            logger.info(f"Saved: {output}")
        else:
            logger.error("Failed to create video")
            sys.exit(1)

    # Save metrics JSON
    metrics_path = task.headless_metrics_output or os.path.splitext(output)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(build_metrics_output(metrics, motion_label, model_label), f, indent=2)
    logger.info(f"Metrics: {metrics_path}")


def run_policy(config: InferenceConfig):
    """Run policy with automatic policy class selection."""
    logger.info("Starting Policy...")
    logger.info(f"Robot: {config.robot.robot_type}")
    logger.info(f"Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"RL Rate: {config.task.rl_rate} Hz")
    logger.info(f"Model path: {config.task.model_path}")

    _dump_config(config)

    headless = config.task.headless_sim_eval

    policy = None
    try:
        policy_class = _select_policy_class(config)
        logger.info(f"Using {policy_class.__name__} (from {policy_class.__module__})")

        if headless:
            # TODO: should fix this to remove this if-branch which can potential cause divergence
            policy = _create_headless_policy(policy_class, config)
        else:
            policy = policy_class(config=config)
        policy = register_interface(policy)

        logger.info("Policy initialized successfully!")

        if headless:
            _run_headless_eval(policy, config, policy._headless_sim)
        else:
            policy.run()

        logger.info("Policy execution completed!")

    except Exception as e:
        logger.error(f"Error running policy: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if policy is not None and hasattr(policy.interface, "shutdown"):
            policy.interface.shutdown()
        restore_terminal_settings()


def main():
    # Try tyro CLI first (full config validation and --help support).
    # If tyro fails — common in Docker where the holosoma-inference image may
    # have incompatible or older config dataclass types that don't match the
    # extension's configs — fall back to a manual argv parser that only needs
    # the extension preset defaults (EXT_DEFAULTS) and --task.* overrides.
    # This makes the policy runnable even when the Docker image is out of date.
    try:
        annotated_config = _build_annotated_config()
        config = tyro.cli(annotated_config, config=TYRO_CONFIG)
    except (Exception, SystemExit) as e:
        logger.warning(f"tyro CLI failed ({e}), falling back to manual parser")
        config = _build_config_from_argv()
    run_policy(config)


def _build_config_from_argv():
    """Fallback: construct InferenceConfig from sys.argv without tyro CLI parser.

    Supports the same invocation as the tyro path:
        run_policy.py inference:<preset> --task.model-path /path --task.ref-motion-path /path

    Only --task.* overrides are supported (the most common use case for
    sim2real deployment). Robot and observation configs come from the preset.
    Type coercion is inferred from the preset's dataclass field types.
    """
    from dataclasses import replace

    args = sys.argv[1:]
    # Find the inference preset name (first positional arg like "inference:g1-wbt")
    preset_name = None
    remaining = []
    for a in args:
        if a.startswith("inference:"):
            preset_name = a.split(":", 1)[1]
        else:
            remaining.append(a)

    if not preset_name:
        logger.error("Usage: run_policy inference:<preset> [--task.field value ...]")
        sys.exit(1)

    if preset_name not in EXT_DEFAULTS:
        logger.error(f"Unknown preset: {preset_name}. Available: {list(EXT_DEFAULTS.keys())}")
        sys.exit(1)

    config = EXT_DEFAULTS[preset_name]

    # Parse --section.field value pairs
    _BOOL_TRUE = {"true", "1", "yes"}
    _BOOL_FALSE = {"false", "0", "no"}
    _BOOL_VALS = _BOOL_TRUE | _BOOL_FALSE
    i = 0
    task_overrides = {}
    while i < len(remaining):
        key = remaining[i]
        if not key.startswith("--"):
            i += 1
            continue
        parts = key[2:].split(".", 1)
        if len(parts) != 2 or parts[0] != "task":
            i += 1
            continue
        field = parts[1].replace("-", "_")
        field_type = type(getattr(config.task, field, ""))
        if field_type is bool:
            next_val = remaining[i + 1].lower() if i + 1 < len(remaining) else None
            if next_val in _BOOL_VALS:
                task_overrides[field] = next_val in _BOOL_TRUE
                i += 2
            else:
                task_overrides[field] = True
                i += 1
        elif i + 1 < len(remaining):
            val = remaining[i + 1]
            if field_type is int:
                task_overrides[field] = int(val)
            elif field_type is float:
                task_overrides[field] = float(val)
            else:
                task_overrides[field] = val
            i += 2
        else:
            i += 1

    if task_overrides:
        config = replace(config, task=replace(config.task, **task_overrides))
        logger.info(f"Config overrides: {task_overrides}")

    logger.info(f"Using preset: {preset_name} (fallback parser)")
    return config


if __name__ == "__main__":
    main()
