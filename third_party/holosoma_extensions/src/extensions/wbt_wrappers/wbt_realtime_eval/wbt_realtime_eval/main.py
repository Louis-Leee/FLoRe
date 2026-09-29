"""Realtime WBT eval entry point.

Build a static mjswan bundle for one ONNX model + the reference motions, then
launch the local server. Supports the holosoma-*-wbt family of inference
presets (see :mod:`wbt_realtime_eval.presets`). Pick the preset that matches
the trained checkpoint; if it doesn't match the ONNX obs dim, the build
refuses with a clear message instead of producing a bundle that silently
runs the wrong policy.

The CLI mirrors ``training_runs/batch_headless_eval.sh``::

    wbt-realtime-eval \
        --preset g1-29dof-holosoma-2pt-wbt \
        --model-path /path/to/model.onnx \
        --motion-dir /path/to/test_motions/

so a single Python entry point handles every preset that the headless eval
script handles, avoiding duplicate argument plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg

from holosoma_inference.config.config_values.robot import g1_29dof

from wbt_realtime_eval import obs_registry
from wbt_realtime_eval.bake_motions import bake_motion_dir
from wbt_realtime_eval.presets import Preset, get_preset
from wbt_realtime_eval.urdf_inertia import align_spec_inertials_to_urdf

_REPO_ROOT = Path(__file__).resolve().parents[5]
_HOLOSOMA_SCENES_DIR = _REPO_ROOT / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/scenes"
# The default render scene (old G1 rubber hand). Kept as the fallback so existing
# bundles for non-yam checkpoints render exactly as before.
_HOLOSOMA_XML = _HOLOSOMA_SCENES_DIR / "scene_g1_29dof_wbt_plane.xml"

# Map a trained-checkpoint URDF stem (from the ONNX ``robot_urdf_path`` metadata)
# to the MuJoCo render scene whose hand geometry matches it. This EXTENDS the
# headless eval's scene-selection (``batch_headless_eval.sh --xml`` /
# ``resolve_xml``, which REQUIRE an explicit ``--xml`` for non-default hands)
# with an auto-derive so a yam checkpoint "just works" with no flag. The ONNX
# ``robot_urdf`` metadata only ever feeds Pinocchio FK (which is
# hand-geometry-INDEPENDENT — verified FK-byte-identical across the rubber-hand
# and gripper URDFs), so the MuJoCo render scene must be chosen separately. A
# checkpoint trained with ``--robot.asset.urdf-file g1/g1_29dof_yam.urdf``
# therefore needs the yam scene here or the browser would render the OLD rubber
# hand on a yam policy. (The Pika gripper has been retired; pass --scene-xml for
# any legacy pika checkpoint.)
_URDF_STEM_TO_SCENE: dict[str, str] = {
    "g1_29dof": "scene_g1_29dof_wbt_plane.xml",
    "g1_29dof_yam": "scene_g1_29dof_wbt_plane_yam.xml",
}


def _decode_meta_str(meta: dict[str, str], key: str) -> str:
    """Read a string ONNX-metadata value, JSON-decoding it if it was stored
    JSON-encoded (holosoma writes metadata via ``json.dumps``, so a path comes
    back quote-wrapped). Matches how ``dof_names`` / ``experiment_config`` are
    decoded elsewhere in this module; robust to a bare (un-encoded) value too."""
    raw = meta.get(key, "")
    if not raw:
        return ""
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip().strip('"')
    return decoded if isinstance(decoded, str) else raw


def _resolve_scene_xml(scene_xml_arg: Path | None, meta: dict[str, str]) -> Path:
    """Pick the MuJoCo render scene for this checkpoint.

    Priority:

    1. An explicit ``--scene-xml`` (absolute, or repo-root-relative) — highest
       priority, lets the user point at any scene.
    2. Auto-derive from the ONNX ``robot_urdf_path`` metadata: map the trained
       URDF's stem (e.g. ``g1_29dof_yam``) to its sibling scene via
       ``_URDF_STEM_TO_SCENE`` so a yam checkpoint "just works" with no flag.
       If the stem IS mapped but the mapped scene file is missing on disk, this
       HARD-FAILS (raises) rather than silently rendering the wrong hand — the
       whole point of the auto-derive is that a yam policy gets the yam mesh.
    3. Fall back to the default old-rubber-hand scene (``_HOLOSOMA_XML``) for an
       UNMAPPED / absent URDF stem — the backward-compatible default for any
       checkpoint we can't classify (a warning names ``--scene-xml`` to override).

    Only the RENDER mesh / sim inertia differ between scenes; the 29-DOF
    articulation, body/joint names, and the force contact point
    (wrist_yaw → rubber_hand at ``0.0815, ±0.003`` on ``*_wrist_yaw_link``) are
    identical across the variants, so nothing else in the bundle changes.
    """
    if scene_xml_arg is not None:
        candidate = scene_xml_arg if scene_xml_arg.is_absolute() else (_REPO_ROOT / scene_xml_arg)
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"--scene-xml not found: {candidate}")
        logger.info(f"Render scene (explicit --scene-xml): {candidate}")
        return candidate

    urdf_path = _decode_meta_str(meta, "robot_urdf_path")
    if urdf_path:
        stem = Path(urdf_path).stem
        scene_name = _URDF_STEM_TO_SCENE.get(stem)
        if scene_name is not None:
            candidate = _HOLOSOMA_SCENES_DIR / scene_name
            if not candidate.is_file():
                # The checkpoint's hand IS known, so falling back to the default
                # scene would render the WRONG hand silently. Fail loudly instead.
                raise FileNotFoundError(
                    f"ONNX robot_urdf_path stem '{stem}' maps to render scene "
                    f"'{scene_name}', but it is missing at {candidate}. This scene "
                    f"ships in the holosoma submodule — ensure it is checked out / "
                    f"committed, or pass an explicit --scene-xml. Refusing to render "
                    f"the wrong hand mesh on a '{stem}' policy."
                )
            logger.info(f"Render scene auto-derived from ONNX robot_urdf_path (stem '{stem}'): {candidate.name}")
            return candidate
        logger.warning(
            f"No render scene mapped for ONNX URDF stem '{stem}'; using the "
            f"default old-rubber-hand scene. Pass --scene-xml to override."
        )

    logger.info(f"Render scene (default): {_HOLOSOMA_XML.name}")
    return _HOLOSOMA_XML


def _align_scene_inertia_to_checkpoint(spec: mujoco.MjSpec, meta: dict[str, str], scene_path: Path) -> None:
    """Align ``spec``'s body inertials to the checkpoint's baked ``robot_urdf``.

    Uses the URDF TEXT carried in the ONNX metadata (not a path on disk) so the
    browser payload matches the exact asset that run trained with, even if
    ``thirdparty/`` has moved since. If the checkpoint carries no ``robot_urdf``
    (a pre-metadata export) we warn and leave the scene as authored rather than
    guessing from an on-disk URDF that may not be the trained one.
    """
    urdf_text = _decode_meta_str(meta, "robot_urdf")
    if not urdf_text:
        logger.warning(
            f"Checkpoint carries no 'robot_urdf' metadata, so {scene_path.name}'s "
            f"own <inertial> values are kept. If this is the yam scene its wrists "
            f"are ~0.7 kg light vs training and the arms will ride high."
        )
        return
    urdf_name = _decode_meta_str(meta, "robot_urdf_path") or "<baked robot_urdf>"
    logger.info(f"Aligning {scene_path.name} inertials to the checkpoint's URDF ({Path(urdf_name).name}).")
    align_spec_inertials_to_urdf(spec, urdf_text)


def _read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    return dict(meta)


def _parse_kp_kd(meta: dict[str, str]) -> tuple[list[float], list[float]]:
    for key in ("kp", "kd"):
        if key not in meta:
            raise ValueError(
                f"ONNX metadata missing required '{key}' field. "
                "Expected a holosoma-trained checkpoint with kp/kd metadata."
            )
    kp = [float(x) for x in meta["kp"].strip("[]").split(",")]
    kd = [float(x) for x in meta["kd"].strip("[]").split(",")]
    if len(kp) != 29 or len(kd) != 29:
        raise ValueError(f"Expected 29 kp/kd, got kp={len(kp)} kd={len(kd)}")
    return kp, kd


def _parse_action_scale(meta: dict[str, str]) -> list[float]:
    if "action_scale" not in meta:
        raise ValueError("ONNX metadata missing required 'action_scale' field. Expected a holosoma-trained checkpoint.")
    try:
        raw = json.loads(meta["action_scale"])
    except json.JSONDecodeError as exc:
        snippet = meta["action_scale"][:80]
        raise ValueError(f"Failed to parse 'action_scale' metadata: {snippet!r}") from exc
    if isinstance(raw, (int, float)):
        result = [float(raw)] * 29
    elif len(raw) == 1:
        result = [float(raw[0])] * 29
    else:
        result = [float(x) for x in raw]
    if len(result) != 29:
        raise ValueError(f"Expected 29 action_scale entries, got {len(result)}")
    return result


def _read_body_names(npz_path: Path) -> list[str]:
    with np.load(npz_path) as d:
        if "body_names" not in d.files:
            raise ValueError(f"Motion {npz_path} missing 'body_names'")
        return [n.decode() if isinstance(n, bytes) else str(n) for n in d["body_names"].tolist()]


def _write_policy_config(
    path: Path, components: list[dict], *, onnx_filename: str, in_keys: list[str], out_keys: list[str]
) -> None:
    """Write the mjswan runtime config shape needed by the WBT ONNX.

    Each obs class maintains its own per-term 4-frame history internally
    (mirrors the training-time per-term stacking in
    ``HolosomaWBTPolicy._build_observation``), so the GROUP-level
    ``history_steps`` MUST be 1 here. Otherwise mjswan's PolicyRunner stacks
    a second time, producing the wrong layout.

    `components` is the list of obs term entries in policy order. Each entry
    needs at minimum `{"name": "<TsClassName>"}`; extra keys (e.g.
    `default_joint_angles` for WbtJointPos) are forwarded to the obs class
    constructor via ``ObservationConfig``.

    ``in_keys`` / ``out_keys`` are read from the ONNX session — different
    presets use different I/O names (e.g. holosoma-2pt-wbt uses ``policy`` /
    ``action``; holosoma-wbt uses ``obs`` / ``actions`` plus extra outputs
    for joint diagnostics).  We forward whatever the model exposes so this
    file generalizes to every checkpoint without per-preset hard-coding.

    The OBS GROUP NAME must match the primary ONNX input name. mjswan's
    PolicyRunner builds an obs dict keyed by group name and the runtime then
    assigns ``input[group_name] = obs[group_name]`` to the ONNX session — so
    if the group is ``"policy"`` but the ONNX expects ``"obs"``, the policy
    never receives any input and the humanoid runs with stale (zero-or-init)
    actions, manifesting as "robot falls limp on entry". (See
    ``third_party/mjswan/src/mjswan/template/src/core/engine/runtime.ts`` —
    the ``runOnnxInference`` loop assigns by key.)
    """
    obs_group = next((k for k in in_keys if k != "time_step"), "policy")
    payload = {
        "onnx": {"path": onnx_filename, "meta": {"in_keys": in_keys, "out_keys": out_keys}},
        "observations": {obs_group: {"history_steps": 1, "components": components}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _parse_normalizer(meta: dict[str, str]) -> tuple[list[float], list[float]]:
    """Read the policy's baked normalizer mean/std from ONNX metadata."""
    if "normalizer_mean" not in meta or "normalizer_std" not in meta:
        raise ValueError(
            "ONNX metadata missing normalizer_mean/normalizer_std. The browser "
            "runtime's normalizer pre-clip will silently degrade tracking; refusing "
            "to build."
        )
    mean = [float(x) for x in meta["normalizer_mean"].split(",")]
    std = [float(x) for x in meta["normalizer_std"].split(",")]
    if len(mean) != len(std):
        raise ValueError(f"normalizer mean/std length mismatch: {len(mean)} vs {len(std)}")
    return mean, std


def _parse_default_joint_angles(meta: dict[str, str], dof_names: list[str]) -> list[float]:
    """Read the trained policy's default_joint_angles from ONNX experiment_config.

    Falls back to g1_29dof.default_dof_angles only if the metadata path is
    absent. Order matches ``dof_names`` (the trained joint order).
    """
    exp = json.loads(meta.get("experiment_config", "{}"))
    angles_dict = exp.get("robot", {}).get("init_state", {}).get("default_joint_angles", {})
    if not angles_dict:
        logger.warning("No default_joint_angles in experiment_config; using g1_29dof fallback")
        return list(g1_29dof.default_dof_angles)
    return [float(angles_dict.get(name, 0.0)) for name in dof_names]


# Substrings identifying the ARM joints (both hands: shoulder p/r/y, elbow,
# wrist r/p/y = 14 of the 29 DOFs). Everything else (legs, waist) is untouched.
_ARM_JOINT_SUBSTRINGS = ("shoulder", "elbow", "wrist")


def _arm_zeroed_reference_pose(default_qpos: list[float], dof_names: list[str]) -> list[float]:
    """Return a COPY of ``default_qpos`` with every ARM joint set to 0 rad, for the
    ``idle_default`` REFERENCE clip only.

    "Arm joints at 0" = shoulders/elbows/wrists neutral, i.e. upper arm vertical /
    forearm horizontal (the URDF zero pose), which the user wants ``idle_default``
    to command instead of the trained slightly-raised default (shoulder ±0.2,
    elbow 0.6). Legs + waist keep their trained default angles so the stance /
    balance is unchanged.

    CRITICAL — this is the REFERENCE pose ONLY. It must NOT replace the trained
    ``default_qpos`` used for (a) the ``WbtJointPos`` obs offset (q − default) and
    (b) the policy action offset (``use_default_offset=True``); shifting either of
    those away from the trained values would corrupt the obs/action and topple the
    robot (same failure class as the obs-layout fall-over). The arm reference flows
    into the policy ONLY through the ``idle_default`` clip's VR 2-point wrist
    targets (``command_lower_body`` is lower-body-only and carries no arm dof), so
    the tracking policy is simply commanded to hold its hands at the arm-zero FK
    location — exactly the intended behavior.
    """
    out = list(default_qpos)
    zeroed = []
    for i, name in enumerate(dof_names):
        if any(s in name for s in _ARM_JOINT_SUBSTRINGS):
            out[i] = 0.0
            zeroed.append(name)
    logger.info(
        f"idle_default reference pose: zeroed {len(zeroed)} arm joint(s) {zeroed} (legs/waist keep trained default)."
    )
    return out


_DEFAULT_PRESET = "g1-29dof-holosoma-2pt-wbt"


# Slider keys consumed by WbtForceCmdLocal in
# wbt_realtime_eval/template/WbtObservations.ts. Order MUST match the obs
# class's KEYS table (left x/y/z then right x/y/z) — that's the same order
# the training-time force_cmd_local obs flattens into.
_FORCE_SLIDER_KEYS_LEFT = ("wbt_force_cmd_local__l_fx", "wbt_force_cmd_local__l_fy", "wbt_force_cmd_local__l_fz")
_FORCE_SLIDER_KEYS_RIGHT = ("wbt_force_cmd_local__r_fx", "wbt_force_cmd_local__r_fy", "wbt_force_cmd_local__r_fz")

# F^ext slider keys: the REAL external disturbance force injected into the
# browser physics (xfrc_applied) — distinct from F^cmd which the policy only
# OBSERVES. For the PLAIN g1-29dof-holosoma-2pt-force-wbt preset these use a
# different prefix from the F^cmd keys so no obs class reads them (F^ext is
# critic-only there — the actor is trained blind to it). NOTE (2026-07-11): the
# g1-29dof-holosoma-2pt-force-ext-wbt preset (run fmotwon3) DELIBERATELY feeds
# force_ext_local into the actor via the WbtForceExtLocal obs class, which reads
# these SAME keys — so for that preset the F^ext slider drives BOTH the xfrc
# disturbance and the obs channel (matching force_tracking.py:1447-1474). Same
# [L_xyz, R_xyz] order as F^cmd.
_FORCE_EXT_SLIDER_KEYS_LEFT = ("wbt_force_ext_local__l_fx", "wbt_force_ext_local__l_fy", "wbt_force_ext_local__l_fz")
_FORCE_EXT_SLIDER_KEYS_RIGHT = ("wbt_force_ext_local__r_fx", "wbt_force_ext_local__r_fy", "wbt_force_ext_local__r_fz")

# Arrow colors (RGB 0-1), matching the benchmark force-arrow videos
# (eval_utils.FORCE_ARROW_RGBA_*): orange = F^ext (real physics), purple =
# F^cmd (obs-only intent). Used as fixed arrow colors (no magnitude lerp).
_ARROW_RGB_CMD = [0.62, 0.20, 0.92]  # purple
_ARROW_RGB_EXT = [1.0, 0.45, 0.0]  # orange
# F^meas arrow (hybrid preset only): cyan, distinct from purple F^cmd + orange
# F^ext. This is what the wrist actually FEELS (the simulated F/T reading).
_ARROW_RGB_MEAS = [0.0, 0.75, 1.0]  # cyan

# Control-point marker color (RGB 0-1): orange sphere drawn at the exact point
# the VR tracker steers — the policy's vr_2point CONTACT point
# (wrist_yaw_link origin + R(wrist) @ VR_2POINT_OFFSETS), NOT the force/rubber-hand
# point. See _vr_control_point_extras.
_CONTROL_POINT_RGB = [1.0, 0.55, 0.0]  # orange
# Hybrid TCP marker color (RGB 0-1): WHITE sphere at the TCP (tcp_offset_x, 0, 0)
# — the point the causal wall / F^ext / F^meas / servo all act on. Distinct from
# the orange VR control-point (0.18 m lever) so the two are not confused: the VR
# point is where POSITION is tracked; the TCP is where FORCE acts. For hybrid the
# force viz is what matters, so we mark the TCP, not the VR point.
_HYBRID_TCP_MARKER_RGB = [1.0, 1.0, 1.0]  # white

# vr_3point TRACKING-point marker colors (RGB 0-1), one hue per tracked point so
# left hand / right hand / head read apart at a glance. These mark the THREE points
# the 3pt policy actually tracks (vr_3point CONTACT points = body origin + R(body) @
# VR_3POINT_OFFSETS): left_wrist_yaw +0.18, right_wrist_yaw +0.18, torso_link +0.35
# (≈ the head/HMD point). Distinct from the hybrid TCP white marker (force point,
# 0.242) and the orange 2pt control-point — this set is the POSITION-tracking target.
# Each point gets a REAL (solid, on the live robot) and a REFERENCE (translucent, on
# the ghost) sphere in the runtime; the color below is shared by the pair so "target
# vs actual" reads as light-vs-solid of the same hue. Order matches VR_3POINT_BODIES.
_VR_3POINT_MARKER_RGB = [
    [0.0, 0.85, 0.9],  # left hand  — cyan
    [0.95, 0.2, 0.75],  # right hand — magenta
    [1.0, 0.85, 0.0],  # head       — yellow
]

# vr_5point TRACKING-point marker colors — the 3pt hues VERBATIM (so a 5pt bundle reads
# the same as a 3pt one for the hands/head) plus two GREENS for the feet. Green is unused
# by every other marker/arrow in the scene (cyan F^meas is an ARROW, and the cyan hand
# sphere is at chest height, never at the ankle), so the feet cannot be confused with a
# hand. Left/right feet differ in brightness the same way nothing else does, letting a
# viewer tell which foot is off without reading the panel. Order = VR_5POINT_BODIES.
_VR_5POINT_MARKER_RGB = [
    *_VR_3POINT_MARKER_RGB,  # left hand cyan, right hand magenta, head yellow
    [0.2, 0.95, 0.35],  # left foot  — bright green
    [0.1, 0.55, 0.2],  # right foot — dark green
]

# TCP offset along wrist_yaw +x (m) for the hybrid force viz (wall / F^ext / F^meas
# / Jᵀ servo / white marker all anchor here). This MUST match what TRAINING uses:
# hybrid.py TCP_OFFSET_X = 0.242 (the fingertip GRASP MIDPOINT — the two-jaw
# clamping-face center), anchored at the wrist_yaw_link origin + (0.242, 0, 0). The
# force reward / F/T sensor / Jᵀ servo all FK this exact point, so the shipped
# checkpoints (fmotwon3 / 31wei7by / 6y8314vq) learned a TCP at 0.242.
#
# Geometry chain along wrist_yaw +x (FK-verified on g1_29dof_yam.urdf, the SAME file
# training loads): FT sensor 0.082 → gripper base 0.139 (= 0.082 + 0.0565 palm
# standoff) → finger-link ORIGINS (finger ROOTS) 0.1931 → fingertip grasp midpoint
# 0.242 (finger root + 0.0489 m fingertip reach). 0.1931 is the finger ROOT, NOT the
# TCP; 0.242 is the fingertip clamping center the policy actually tracks.
_TCP_OFFSET_X = 0.242

# Commanded-force onset for wall contact (N). The wall is seeded plane_dist ahead
# of the rest TCP along the push direction; the UniFP setpoint shifts toward it by
# Δx = F_cmd / stiffness_K, so contact (penetration > 0) begins when |F_cmd| reaches
# stiffness_K · plane_dist. Inverting that, we place the wall at
#   plane_dist = WALL_CONTACT_ONSET_N / stiffness_K
# so the hand starts touching at ~5 N of commanded force in ANY ±xyz direction (the
# wall normal is always −unit(F_cmd), so onset is direction-independent). At K=200
# this is 0.025 m, comfortably inside the training-sampled plane_dist range
# [0.02, 0.09] m, so the demo wall stays physically representative.
_WALL_CONTACT_ONSET_N = 5.0


def _force_preset_extras() -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (sliders, force_viz_bodies, force_ext_bodies) for the
    holosoma-2pt-force-wbt preset.

    Twelve live sliders in four groups: per-hand F^cmd (obs-only intent) and
    per-hand F^ext (real injected disturbance), each L/R x [Fx, Fy, Fz] in raw
    Newtons, pelvis frame. HandForceConfig defaults pinned to keep the live UI
    consistent with the training cfg in
    src/extensions/wbt_wrappers/wbt_training/config_values/force_tracking.py:
      f_cmd_max = f_ext_max = 25 N (slider range +/- this)
      body_names = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
      body_offsets_b = ((0.0815, +0.003, 0), (0.0815, -0.003, 0))  # yam gripper

    Three consumers of the same slider keys:
      * F^cmd keys  -> WbtForceCmdLocal obs class (x0.04) AND the purple arrow.
      * F^ext keys  -> the runtime xfrc_applied injection (force_ext_bodies)
                       AND the orange arrow. NEVER an obs term.
    """
    f_max = 25.0
    hands = [
        ("L", "left_wrist_yaw_link", [0.0815, +0.003, 0.0], _FORCE_SLIDER_KEYS_LEFT, _FORCE_EXT_SLIDER_KEYS_LEFT),
        ("R", "right_wrist_yaw_link", [0.0815, -0.003, 0.0], _FORCE_SLIDER_KEYS_RIGHT, _FORCE_EXT_SLIDER_KEYS_RIGHT),
    ]

    sliders: list[dict] = []
    # F^cmd sliders first (the obs channel), then F^ext (the physics channel),
    # so the panel reads cmd-above-ext per hand.
    for side, _body, _off, cmd_keys, _ext_keys in hands:
        grp = "Left" if side == "L" else "Right"
        for axis, key in zip("xyz", cmd_keys, strict=True):
            sliders.append(
                {
                    "label": f"{side} F{axis}",
                    "key": key,
                    "min": -f_max,
                    "max": f_max,
                    "default": 0.0,
                    "group": f"{grp} hand F^cmd  (intent, observed) N",
                }
            )
    for side, _body, _off, _cmd_keys, ext_keys in hands:
        grp = "Left" if side == "L" else "Right"
        for axis, key in zip("xyz", ext_keys, strict=True):
            sliders.append(
                {
                    "label": f"{side} F{axis}",
                    "key": key,
                    "min": -f_max,
                    "max": f_max,
                    "default": 0.0,
                    "group": f"{grp} hand F^ext  (real disturbance) N",
                }
            )

    # Per-hand arrow viz: TWO arrows each — orange F^ext + purple F^cmd, both
    # anchored at the rubber-hand point. WbtTrackingCommand reads slider_keys
    # (purple, F^cmd) and ext_slider_keys (orange, F^ext) from each entry.
    force_viz_bodies = [
        {
            "body_name": body,
            "offset_b": off,
            "slider_keys": list(cmd_keys),
            "ext_slider_keys": list(ext_keys),
            "arrow_color_cmd": _ARROW_RGB_CMD,
            "arrow_color_ext": _ARROW_RGB_EXT,
            "f_max_n": f_max,
        }
        for _side, body, off, cmd_keys, ext_keys in hands
    ]

    # The runtime physics patch reads this to inject F^ext into xfrc_applied at
    # the rubber-hand point (with the moment arm). Mirrors headless Eval-3.
    force_ext_bodies = [
        {"body_name": body, "offset_b": off, "slider_keys": list(ext_keys), "f_max_n": f_max}
        for _side, body, off, _cmd_keys, ext_keys in hands
    ]
    return sliders, force_viz_bodies, force_ext_bodies


# Per-hand arm joints the Jᵀ servo drives (yam URDF, matching hybrid.py
# _resolve_jacobian_indices arm_joint_names). Order = shoulder p/r/y, elbow,
# wrist r/p/y. The mjswan-fork runtime resolves each to its qpos/qvel/ctrl slot.
_HYBRID_ARM_JOINTS = {
    "left_wrist_yaw_link": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_wrist_yaw_link": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}


def _parse_hybrid_cfg(meta: dict[str, str]) -> dict | None:
    """Pull the HybridContactConfig out of the ONNX ``experiment_config`` metadata.

    A hybrid checkpoint (trained via the core PPO exporter) serializes the whole
    ExperimentConfig, so the HybridContactConfig lives nested under a command /
    reward / termination term's params. We walk the JSON for the first dict that
    carries the hybrid signature fields (``stiffness_K`` + ``servo_enabled`` +
    ``k_env_min``) — robust to exactly which term it hangs off. Returns None if
    absent (a non-hybrid checkpoint), so the caller falls back to no hybrid extras.
    """
    raw = meta.get("experiment_config", "")
    if not raw:
        return None
    try:
        exp = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    signature = {"stiffness_K", "servo_enabled", "k_env_min"}

    def _walk(obj: object, sig: set[str]) -> dict | None:
        if isinstance(obj, dict):
            if sig.issubset(obj.keys()):
                return obj
            for v in obj.values():
                found = _walk(v, sig)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = _walk(v, sig)
                if found is not None:
                    return found
        return None

    found = _walk(exp, signature)
    if found is None:
        return None

    # Gripper-mass DR range, if the run trained with it (training-side
    # tcp_precision.g1_29dof_wbt_randomization_hand_mass). It lives in the RANDOMIZATION
    # config, not in the HybridContactConfig, so we harvest it here — where the full
    # ExperimentConfig is still in scope — and stash it alongside. Purely informational:
    # the browser simulates ONE nominal robot, this only stops the UI from presenting a
    # nominal m_distal as if it were exact. Copy so the walked dict isn't mutated in place.
    mass_dr = _walk(exp, {"enable_link_mass", "link_mass_range"})
    if mass_dr is not None and mass_dr.get("enable_link_mass"):
        rng = mass_dr.get("link_mass_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            found = {**found, "_link_mass_scale_range": [float(rng[0]), float(rng[1])]}
    return found


def _distal_mass_range(cfg: dict, nominal: float) -> list[float] | None:
    """The [min, max] m_distal (kg) the checkpoint's gripper-mass DR spanned, or None.

    `_parse_hybrid_cfg` stashes the DR's link-mass SCALE range as `_link_mass_scale_range`.
    Only the ft_sensor+rubber_hand part of the merged wrist body scales — the fingers are
    separate prismatic bodies — so the m_distal swing is NOT simply `scale × nominal`;
    it mirrors training's `tcp_precision.distal_mass_from_body_mass`. Constants duplicated
    (not imported) because wbt_realtime_eval must not depend on wbt_training; they are
    pinned against the URDF on the training side by
    test_tcp_precision.py::test_gripper_mass_constants_match_the_urdf.
    """
    rng = cfg.get("_link_mass_scale_range")
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        return None
    scaled_kg, fingers_kg = 1.357992, 0.1420084  # g1_29dof_yam.urdf, per hand
    lo, hi = (float(s) * scaled_kg + fingers_kg for s in (rng[0], rng[1]))
    if lo > hi:
        lo, hi = hi, lo
    # A range that does not bracket the nominal means the constants drifted from the URDF
    # the checkpoint trained on; report nothing rather than a misleading band.
    return [lo, hi] if lo <= nominal <= hi else None


def _checkpoint_actor_obs_terms(meta: dict[str, str]) -> list[str] | None:
    """Return the actor-obs term KEYS the checkpoint was trained with, in config order.

    Read from the serialized ``experiment_config``
    (``observation.groups.actor_obs.terms``), whose keys are the zero-padded
    ``obs00_…`` slot names. Returns None when the metadata is absent or reshaped so
    callers can degrade to a warning rather than blocking a build.
    """
    raw = meta.get("experiment_config", "")
    if not raw:
        return None
    try:
        exp = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    try:
        terms = exp["observation"]["groups"]["actor_obs"]["terms"]
    except (KeyError, TypeError):
        return None
    return list(terms) if isinstance(terms, dict) else None


def _assert_checkpoint_vr_geometry(meta: dict[str, str], preset_cfg: Preset) -> None:
    """Refuse to build when the checkpoint's HAND TRACKING GEOMETRY disagrees with the
    preset's.

    This closes the one hole the obs-dim check cannot: the TCP-unified local twins
    (``vr_3point_pos_tcp``, hands at 0.242 m) and the pre-2026-08-03 unified twins
    (``vr_3point_pos``, hands at the 0.18 m VR lever) are BOTH 170D → ONNX ``[1, 680]``.
    So loading a pre-TCP checkpoint (``m3s0pw9i`` / ``4o9zbs7i``) into the TCP preset
    passes every existing guard, builds cleanly, and then feeds the policy a hand target
    6.2 cm from the one it trained on — the arms just reach to the wrong place, with
    nothing in the UI saying why. The geometry is recorded in each checkpoint's own obs
    TERM NAME, so we compare names rather than trusting the run id.

    Degrades to a warning (never blocks) when the checkpoint predates term-name metadata
    or its ``experiment_config`` is reshaped — an unreadable config is not evidence of a
    mismatch. Only a POSITIVE disagreement raises.
    """
    trained = _checkpoint_actor_obs_terms(meta)
    if trained is None:
        logger.warning(
            "Could not read actor_obs term names from the ONNX experiment_config; "
            "skipping the hand-geometry cross-check. If this is a 3pt unified checkpoint, "
            "confirm by hand that it was trained with the same tracking point as preset "
            f"'{preset_cfg.name}'."
        )
        return

    # Compare the POSITION-slot spelling on both sides. The preset carries the bare term
    # name ("vr_3point_pos_tcp"); the checkpoint carries the zero-padded slot
    # ("obs01_vr_3point_pos_tcp"), hence the suffix match. Checking _tcp FIRST matters:
    # "vr_3point_pos" is a prefix of "vr_3point_pos_tcp", so while `endswith` on the full
    # slot name distinguishes them today, ordering makes the intent explicit and keeps a
    # future suffixed spelling (…_pos_tcp_v2) from silently reading as the 0.18 lever.
    def _pos_slot(keys: list[str]) -> str | None:
        # `vr_5point_pos_tcp` FIRST for the same prefix reason spelled out above: it is not
        # a suffix of the 3pt spellings, so ordering is not strictly load-bearing today, but
        # keeping most-specific-first makes the intent explicit. Adding the 5pt spelling
        # cannot change any EXISTING preset's outcome — a 5pt checkpoint in a 3pt preset (or
        # vice versa) is 736D vs 680D and is already refused by the obs-dim check upstream;
        # this only upgrades the message from "dim mismatch" to "geometry mismatch" if the
        # widths ever coincide.
        for candidate in ("vr_5point_pos_tcp", "vr_3point_pos_tcp", "vr_3point_pos"):
            if any(k.endswith(candidate) for k in keys):
                return candidate
        return None

    want = _pos_slot(list(preset_cfg.obs_keys))
    got = _pos_slot(trained)
    if want is None or got is None or want == got:
        return

    _GEOM = {
        "vr_3point_pos_tcp": "0.242 m fingertip TCP (position+force UNIFIED)",
        "vr_3point_pos": "0.18 m VR lever",
        "vr_5point_pos_tcp": "5 points (2 hands @ the 0.242 m TCP + head + both feet)",
    }
    raise ValueError(
        f"Checkpoint / preset hand-geometry MISMATCH. Preset '{preset_cfg.name}' tracks "
        f"the hands at the {_GEOM[want]} (obs slot '{want}'), but this checkpoint was "
        f"TRAINED with '{got}' — the {_GEOM[got]}. Both layouts are "
        f"{preset_cfg.total_dim}D, so the obs-dim check cannot catch this: the bundle "
        "would build fine and then command the hands ~6.2 cm from where the policy "
        "learned them.\n"
        "  * Pre-2026-08-03 unified runs (m3s0pw9i R2 / 4o9zbs7i R1) are 0.18 m — view "
        "them by checking out a commit before 9d16291.\n"
        "  * For the TCP-unified presets use a run trained AFTER 9d16291 "
        "(e.g. pfl96opy, WBT-Hybrid3ptUniAnneal-tcp-*)."
    )


def _assert_checkpoint_tracking_family(meta: dict[str, str], preset_cfg: Preset) -> None:
    """Refuse to build when the checkpoint's TRACKING FRONT-END is a different KIND from the
    preset's — DENSE 29-DOF joint targets vs VR KEYPOINTS.

    This closes the second hole the obs-dim check cannot, and it is not hypothetical:
    ``g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt`` (183D, ``motion_command``) and
    ``g1-29dof-holosoma-3pt-hybrid-global-vel[-layerA]-wbt`` (183D, ``vr_3point_pos`` +
    the root trio) are BOTH ``[1, 732]``. Both are hybrid presets, so both build, and the
    two layouts share not a single term position — the robot would be fed 24D of lower-body
    command where it expects the first 24 of a 58D joint-target vector, and fall.

    Complements :func:`_assert_checkpoint_vr_geometry`, which compares hand GEOMETRY within
    the keypoint family; this compares the family itself. Same degradation policy: an
    unreadable ``experiment_config`` warns (there it is not evidence of a mismatch), and
    only a POSITIVE disagreement raises.

    Cannot change any pre-2026-08-27 build's outcome: it fires only when the widths already
    agree (it runs after the normalizer dim check), and every other width in the registry
    where one side is dense — 628 / 640 / 664 — is unique to a single preset.
    """
    trained = _checkpoint_actor_obs_terms(meta)
    if trained is None:
        # _assert_checkpoint_vr_geometry already warned about the same unreadable metadata;
        # don't log the identical complaint twice.
        return

    def _is_dense(keys: list[str]) -> bool:
        """True when the tracking front-end is the 58D full-joint-target vector.

        Both spellings are accepted because the two sides name the same term differently:
        the browser preset's key is ``command`` while a checkpoint's slot is
        ``obs00_motion_command`` (holosoma's own term name). Requiring the ABSENCE of any
        ``vr_``/``command_lower_body`` slot is what makes this a family test rather than a
        term test — a hypothetical future preset carrying both would be neither, and is
        deliberately left ungated instead of guessed at.
        """
        has_dense = any(k == "command" or k.endswith("motion_command") for k in keys)
        has_keypoints = any("vr_" in k or k.endswith("command_lower_body") for k in keys)
        return has_dense and not has_keypoints

    def _is_keypoint(keys: list[str]) -> bool:
        return any("vr_" in k for k in keys)

    preset_keys = list(preset_cfg.obs_keys)
    want_dense, got_dense = _is_dense(preset_keys), _is_dense(trained)
    want_kp, got_kp = _is_keypoint(preset_keys), _is_keypoint(trained)
    # Only a clear dense-vs-keypoint disagreement raises; anything ambiguous on either side
    # falls through untouched.
    if not ((want_dense and got_kp) or (want_kp and got_dense)):
        return

    _KIND = {True: "DENSE 29-DOF joint targets (motion_command, 58D)", False: "VR KEYPOINTS"}
    raise ValueError(
        f"Checkpoint / preset TRACKING-FAMILY mismatch. Preset '{preset_cfg.name}' tracks "
        f"{_KIND[want_dense]}, but this checkpoint was TRAINED on {_KIND[got_dense]} "
        f"(its obs slots: {trained}). Both layouts are {preset_cfg.total_dim}D, so the "
        "obs-dim check cannot catch this — the bundle would build and then feed the policy "
        "a completely different obs vector.\n"
        "  * DENSE 183D is exp:g1-wbt-hybrid-unified-anneal (run 7r54008a) -> preset "
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt.\n"
        "  * KEYPOINT 183D is exp:g1-wbt-3pt-hybrid-global-vel -> preset "
        "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt."
    )


def _hybrid_preset_extras(
    hybrid_cfg: dict | None,
    *,
    servo_enabled_override: bool | None = None,
    servo_observe_only_override: bool | None = None,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Returns (sliders, hybrid_extras, force_viz_bodies, force_ext_bodies) for the
    hybrid presets.

    * ``sliders`` — the SAME 12 F^cmd/F^ext sliders as the force preset (F^cmd is
      the commanded compliance target the servo tracks; F^ext is an ADDITIVE manual
      disturbance on top of the causal virtual wall).
    * ``hybrid_extras`` — the ``extras.hybrid`` block the mjswan-fork runtime reads
      to reconstruct the causal spring wall + F/T sensor + Jᵀ servo. Every physical
      constant is read from the checkpoint's HybridContactConfig (via
      :func:`_parse_hybrid_cfg`) so the viz self-configures per retrain; missing
      fields fall back to the training defaults baked into the runtime parser.
    * ``force_viz_bodies`` — per-hand arrow config: purple F^cmd (pelvis frame),
      orange F^ext (world), cyan F^meas (world). All anchored at the TCP
      (tcp_offset_x, 0, 0) — the SINGLE hybrid contact anchor (NOT the rubber-hand
      offset), matching hybrid.py _tcp_offsets_b.
    * ``force_ext_bodies`` — the runtime's ``applyExternalForces`` xfrc-injection
      config, TCP-anchored, driven by the F^ext slider keys. REQUIRED: the hybrid
      substep loop calls ``applyExternalForces()`` every substep, and that method
      is a no-op unless ``extras.force_ext_bodies`` is populated — so without this
      the 6 F^ext sliders would be dead and the orange arrow would draw a force
      that is never applied. F^ext here is ADDITIVE on top of the causal wall (the
      wall reads the F^cmd keys, not these), and the wrist xfrc rows are re-cleared
      each substep, so there is no double-injection.
    """
    cfg = hybrid_cfg or {}

    def _num(key: str, default: float) -> float:
        v = cfg.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # servo_enabled: from the checkpoint, unless the preset name forces it (the
    # -layerA preset renders servo-OFF regardless of what the ckpt baked).
    servo_enabled = bool(cfg.get("servo_enabled", False))
    if servo_enabled_override is not None:
        servo_enabled = servo_enabled_override
    # servo_observe_only (C1 unified): from the checkpoint, unless the preset name
    # forces it (the -unified presets deploy observe-only regardless — R2 bakes the
    # flag, but R1's terminal α=0 lives only in curriculum runtime state, never in
    # the serialized cfg, so the name override is what keeps the "neither twin
    # injects at deploy" contract for the anneal checkpoint).
    servo_observe_only = bool(cfg.get("servo_observe_only", False))
    if servo_observe_only_override is not None:
        servo_observe_only = servo_observe_only_override
    # Observe-only is a CONJUNCTION in the runtime: `servoOn = observeOnly ? servoEnabled
    # : <live toggle>` (runtime.ts). So observe_only=True with servo_enabled=False routes
    # to zeroHybridServoState, which publishes I = Δx = 0 — silently pinning 12 of the
    # unified actor's 170 obs dims to zero (48 of 680 with history) while contact_phase
    # keeps working, so nothing LOOKS broken. That combination is never intentional: both
    # unified twins train with servo_enabled=True (only the INJECTION is suppressed), and
    # a `-layerA` preset never sets observe_only. It is reachable when
    # _parse_hybrid_cfg returns None / a signature-less dict (unparseable or reshaped
    # experiment_config): the name override still forces observe_only, but servo_enabled
    # falls back to False. Repair it here rather than trusting the checkpoint.
    if servo_observe_only and not servo_enabled:
        if servo_enabled_override is False:
            # A caller explicitly demanded servo-OFF *and* observe-only. No preset does
            # this (layerA and unified names are disjoint), so fail loudly instead of
            # silently picking one — a future dispatch bug should not reach the browser.
            raise ValueError(
                "Contradictory hybrid overrides: servo_enabled_override=False with "
                "servo_observe_only=True would publish I=Δx=0 into a unified actor's "
                "obs. layerA (servo-off) and unified (observe-only) are mutually "
                "exclusive preset families."
            )
        logger.warning(
            "Hybrid cfg had servo_enabled=False but this is an observe-only (C1 unified) "
            "preset; forcing servo_enabled=True so the runtime keeps publishing the wound "
            "I/Δx the actor obs consume. (Δq is still never injected — that is what "
            "servo_observe_only controls.)"
        )
        servo_enabled = True

    f_cmd_max = _num("f_cmd_max", 25.0)
    # TCP = the training fingertip grasp midpoint (0.242 m, hybrid.py TCP_OFFSET_X).
    # Pinned to _TCP_OFFSET_X so the viz matches what the shipped checkpoints learned
    # (the checkpoint also bakes 0.242; we pin the constant rather than read it so a
    # future retrain under a changed URDF stays a one-line edit here). See _TCP_OFFSET_X.
    tcp_offset_x = _TCP_OFFSET_X
    k_env_min = _num("k_env_min", 100.0)
    k_env_max = _num("k_env_max", 800.0)
    # Demo K_env: geometric mean of the training log-uniform range [100, 800].
    k_env_demo = (k_env_min * k_env_max) ** 0.5
    stiffness_k = _num("stiffness_K", 200.0)

    # Place the wall so contact starts at ~WALL_CONTACT_ONSET_N of commanded force
    # (plane_dist = onset / K), then clamp into the training-sampled plane_dist
    # range so the demo stays physically representative. Direction-independent (the
    # wall normal is always −unit(F_cmd)), so ±xyz all begin contact at the same
    # |F_cmd|. See _WALL_CONTACT_ONSET_N.
    _pd_min = _num("plane_dist_min", 0.02)
    _pd_max = _num("plane_dist_max", 0.09)
    plane_dist = _WALL_CONTACT_ONSET_N / stiffness_k if stiffness_k > 0 else 0.5 * (_pd_min + _pd_max)
    plane_dist = min(max(plane_dist, _pd_min), _pd_max)

    hands = [
        ("L", "left_wrist_yaw_link", _FORCE_SLIDER_KEYS_LEFT, _FORCE_EXT_SLIDER_KEYS_LEFT),
        ("R", "right_wrist_yaw_link", _FORCE_SLIDER_KEYS_RIGHT, _FORCE_EXT_SLIDER_KEYS_RIGHT),
    ]

    # 12 sliders: F^cmd (intent, observed + servo-tracked) then F^ext (manual
    # disturbance on top of the causal wall), same layout as the force preset.
    sliders: list[dict] = []
    for side, _body, cmd_keys, _ext_keys in hands:
        grp = "Left" if side == "L" else "Right"
        for axis, key in zip("xyz", cmd_keys, strict=True):
            sliders.append(
                {
                    "label": f"{side} F{axis}",
                    "key": key,
                    "min": -f_cmd_max,
                    "max": f_cmd_max,
                    "default": 0.0,
                    "group": f"{grp} hand F^cmd  (intent, observed, servo target) N",
                }
            )
    for side, _body, _cmd_keys, ext_keys in hands:
        grp = "Left" if side == "L" else "Right"
        for axis, key in zip("xyz", ext_keys, strict=True):
            sliders.append(
                {
                    "label": f"{side} F{axis}",
                    "key": key,
                    "min": -f_cmd_max,
                    "max": f_cmd_max,
                    "default": 0.0,
                    "group": f"{grp} hand F^ext  (extra manual disturbance) N",
                }
            )

    hybrid_extras = {
        "servo_enabled": servo_enabled,
        # C1 unified: integrator publishes I/Δx (reference generator) but the runtime
        # NEVER injects Δq into ctrl. Soft contact_phase thresholds self-configure
        # from the checkpoint like every other physical constant.
        "servo_observe_only": servo_observe_only,
        "contact_phase_theta_on": _num("contact_phase_theta_on", 2.0),
        "contact_phase_theta_off": _num("contact_phase_theta_off", 1.5),
        "contact_phase_tau_s": _num("contact_phase_tau_s", 0.05),
        # Post-reset suppression window (ENV steps). Training forces the hysteresis OFF
        # for the first few steps of each episode so the reset teleport's inertial spike
        # in F^meas (m·a ≫ theta_on) can't slam the soft label to 1 in free space; the
        # browser needs the same guard because a_wrist is finite-differenced from TCP
        # position, so a motion-switch pose JUMP reads as enormous acceleration.
        "contact_phase_warmup_steps": _num("contact_phase_warmup_steps", 5.0),
        "stiffness_K": stiffness_k,
        "f_cmd_max": f_cmd_max,
        "k_env": k_env_demo,
        "plane_dist": plane_dist,
        "max_penetration_m": _num("max_penetration_m", 0.08),
        "max_force_n": _num("max_force_n", 35.0),
        # NOMINAL distal mass. With gripper-mass DR on (WBT_HAND_MASS_DR=1, training-side
        # tcp_precision.g1_29dof_wbt_randomization_hand_mass) each sim env gets its OWN
        # m_distal in [0.8, 1.5]×, so a single number is no longer "the" value the policy
        # trained against — it is the centre of a range. The browser runs one nominal robot,
        # so using the nominal here is right; what would be wrong is presenting it as exact.
        # `distal_mass_is_nominal` lets the UI label it (see distal_mass_range_kg below).
        "distal_mass_kg": _num("distal_mass_kg", 1.5),
        "distal_mass_is_nominal": bool(cfg.get("sync_distal_mass_from_sim", False)),
        # Present only when the checkpoint trained under mass DR; the UI can show the swing
        # the policy is robust to instead of implying a single exact payload.
        "distal_mass_range_kg": _distal_mass_range(cfg, _num("distal_mass_kg", 1.5)),
        "sensor_noise_std": 0.0,  # clean demo arrows (training uses 0.5)
        "servo_gain_k": _num("servo_gain_k", 0.05),
        "integrator_gain_ki": _num("integrator_gain_ki", 0.5),
        "integrator_max": _num("integrator_max", 40.0),
        "servo_dq_max": _num("servo_dq_max", 0.15),
        "tcp_offset_x": tcp_offset_x,
        "hands": [
            {"body_name": body, "cmd_slider_keys": list(cmd_keys), "arm_joints": _HYBRID_ARM_JOINTS[body]}
            for _side, body, cmd_keys, _ext_keys in hands
        ],
    }

    # Per-hand three-arrow viz, all anchored at the TCP (tcp_offset_x, 0, 0).
    tcp_off = [tcp_offset_x, 0.0, 0.0]
    meas_keys_by_side = {
        "L": ("wbt_force_meas__l_fx", "wbt_force_meas__l_fy", "wbt_force_meas__l_fz"),
        "R": ("wbt_force_meas__r_fx", "wbt_force_meas__r_fy", "wbt_force_meas__r_fz"),
    }
    force_viz_bodies = [
        {
            "body_name": body,
            "offset_b": tcp_off,
            "slider_keys": list(cmd_keys),  # purple F^cmd (pelvis frame)
            "ext_slider_keys": list(ext_keys),  # orange F^ext (pelvis frame)
            "meas_slider_keys": list(meas_keys_by_side[side]),  # cyan F^meas (WORLD frame)
            "arrow_color_cmd": _ARROW_RGB_CMD,
            "arrow_color_ext": _ARROW_RGB_EXT,
            "arrow_color_meas": _ARROW_RGB_MEAS,
            "meas_frame": "world",  # F^meas is world-native (wall normal is world-fixed)
            "f_max_n": f_cmd_max,
        }
        for side, body, cmd_keys, ext_keys in hands
    ]

    # F^ext xfrc-injection config (TCP-anchored, ext keys). The hybrid runtime
    # calls applyExternalForces() every substep, which no-ops unless this is
    # present — so this is what makes the F^ext sliders actually push (additive on
    # top of the causal wall; wrist xfrc is re-cleared each substep → no double-inject).
    force_ext_bodies = [
        {"body_name": body, "offset_b": tcp_off, "slider_keys": list(ext_keys), "f_max_n": f_cmd_max}
        for _side, body, _cmd_keys, ext_keys in hands
    ]
    return sliders, hybrid_extras, force_viz_bodies, force_ext_bodies


# NOTE (merge 2026-08-27): there is deliberately NO per-term config for the
# ``contact_phase`` obs class. An earlier version of the live-3pt work shipped a TS
# class that recomputed φ from the F^meas bus and needed the checkpoint's hysteresis /
# LPF / warmup constants here. hybrid_control's `WbtContactPhase` is the version that
# survived: it reads the runtime's own ``wbt_cphase__{l,r}``, which the spring-wall sim
# recomputes every 5 ms SUBSTEP (with a post-reset warmup self-armed off prevTcpW) —
# strictly better than a 20 ms control-step re-derivation. The four constants still
# reach that sim, via `hybrid_extras` above.


# Ball colours (RGB 0-1) for the LIVE 3pt commanded-target triads, one per tracked
# point. The AXES are always R/G/B (that is what the operator is reading); the ball
# only has to keep the three targets tellable apart when they overlap on screen.
# Chosen clear of the white achieved-TCP sphere and of the arrow palette.
_VR3_TARGET_BALL_RGB = {
    "l": [1.0, 0.20, 0.80],  # magenta — left TCP target
    "r": [1.0, 0.85, 0.10],  # yellow  — right TCP target
    "h": [0.20, 1.00, 0.60],  # spring green — head target
}


def _vr3_live_target_triad_extras() -> dict:
    """Config for the RGB axis triad drawn at each LIVE 3pt COMMANDED target pose.

    The three points are the policy's own tracked bodies, in obs order: left hand
    TCP, right hand TCP, head. ``WbtVr3PointPosTcpLive`` / ``WbtVr3PointOrnLive``
    publish the pelvis-frame pose they fed onto the policy-extras bus under
    ``wbt_vr3_tgt__{l,r,h}_*``; ``WbtTrackingCommand`` composes it with the live
    pelvis world pose and draws the triad there.

    Pure viz — no obs, no physics, no command change. This is the correspondence
    check the teleop needs: the triad is where the policy was TOLD to put the
    point, and the white TCP sphere (drawn by the hybrid branch) is where the
    robot actually got it, so a frame/axis error shows up as a constant rotation
    between the two rather than as vaguely wrong tracking.
    """
    return {
        "extras_prefix": "wbt_vr3_tgt",
        "ok_key": "wbt_vr3_live__ok",
        "axis_len": 0.12,
        "ball_radius": 0.018,
        "points": [
            {"tag": "l", "label": "L TCP target", "color": _VR3_TARGET_BALL_RGB["l"]},
            {"tag": "r", "label": "R TCP target", "color": _VR3_TARGET_BALL_RGB["r"]},
            {"tag": "h", "label": "HEAD target", "color": _VR3_TARGET_BALL_RGB["h"]},
        ],
    }


def _vr_control_point_extras() -> list[dict]:
    """Per-hand config for the orange CONTROL-POINT marker — the exact point the
    VIVE tracker steers.

    The 2pt policy does NOT track the wrist origin; it tracks the ``vr_2point``
    CONTACT point ``wrist_yaw_link_origin + R(wrist) @ VR_2POINT_OFFSETS[k]``,
    a ~0.18 m lever forward of each wrist (the live bridge calibrates the
    operator's hands onto exactly this point). WbtTrackingCommand reads
    ``vr_control_point_bodies`` from its motion-command params and draws one
    small orange sphere at that point per hand each render frame.

    This is the SAME body + swizzle math the force triad uses, but with the VR
    offset (``0.18, ∓0.025``), which is DIFFERENT from the force/rubber-hand
    offset (``0.0815, ±0.003`` — note the larger reach AND the flipped y sign),
    so it must carry its own offset rather than reuse the force one.

    Pure viz: no obs term, no physics, no command output. Sourced from the same
    single-source-of-truth (``VR_2POINT_OFFSETS`` / ``VR_2POINT_BODIES``) the
    bake + the live bridge use, so the marker lands exactly where the policy
    obs point is.
    """
    from wbt_wrappers_inference.config_values.robot_config import VR_2POINT_BODIES, VR_2POINT_OFFSETS

    return [
        {"body_name": str(body), "offset_b": [float(o) for o in offset], "color": _CONTROL_POINT_RGB, "radius": 0.025}
        for body, offset in zip(VR_2POINT_BODIES, VR_2POINT_OFFSETS, strict=True)
    ]


def _vr_tracking_point_markers_extras(*, tcp_unified: bool = False) -> list[dict]:
    """Per-point config for the vr_3point TRACKING-point markers — the THREE points
    the 3pt policy actually tracks (left hand, right hand, head).

    The 3pt policy tracks ``vr_3point`` CONTACT points
    ``body_origin + R(body) @ VR_3POINT_OFFSETS[k]`` for
    ``VR_3POINT_BODIES = (left_wrist_yaw_link, right_wrist_yaw_link, torso_link)``
    with offsets ``(0.18, ∓0.025, 0)`` for the hands and ``(0, 0, 0.35)`` for the
    head. This is the SAME body + swizzle math the 2pt control-point / force triad
    use, sourced from the single-source-of-truth (``VR_3POINT_OFFSETS`` /
    ``VR_3POINT_BODIES``) the bake + training FK use, so the marker lands exactly on
    the policy's obs point.

    ``tcp_unified=True`` (obs slot ``vr_3point_pos_tcp``, i.e.
    ``exp:g1-wbt-3pt-hybrid-unified[-anneal]``) swaps in ``VR_3POINT_OFFSETS_TCP``:
    both HAND points move onto the 0.242 m fingertip TCP — the same anchor as the
    white force marker — so on those presets the cyan/magenta position spheres are
    CO-LOCATED with the white force sphere instead of sitting 6.2 cm behind it. That
    co-location is the visual confirmation that position and force share one point.

    Each entry carries ``real_color`` (solid sphere on the LIVE robot) and
    ``ref_color`` (translucent sphere on the reference ghost) — the SAME hue, so a
    point's "commanded target vs actual" reads as light-vs-solid. The runtime
    (WbtTrackingCommand) draws both spheres per point and ties the reference sphere's
    visibility to the ghost's ``referenceVisible`` toggle.

    Pure viz: no obs term, no physics, no command output. Emitted ONLY for presets
    whose actor obs carries ``vr_3point_pos`` (see the dispatch in build_bundle), so
    every non-3pt preset stays byte-identical.
    """
    from wbt_wrappers_inference.config_values.robot_config import (
        VR_3POINT_BODIES,
        VR_3POINT_OFFSETS,
        VR_3POINT_OFFSETS_TCP,
    )

    offsets = VR_3POINT_OFFSETS_TCP if tcp_unified else VR_3POINT_OFFSETS
    return [
        {
            "body_name": str(body),
            "offset_b": [float(o) for o in offset],
            "real_color": _VR_3POINT_MARKER_RGB[k],
            "ref_color": _VR_3POINT_MARKER_RGB[k],
            "radius": 0.03,
        }
        for k, (body, offset) in enumerate(zip(VR_3POINT_BODIES, offsets, strict=True))
    ]


def _vr_5point_tracking_point_markers_extras() -> list[dict]:
    """Per-point config for the vr_5point TRACKING-point markers — the FIVE points
    ``exp:g1-wbt-5pt-hybrid-unified-anneal`` tracks: left hand, right hand, head, left
    foot, right foot.

    Structurally the 3pt TCP-unified marker list with two more entries: the hands stay on
    the 0.242 m fingertip TCP (so they remain CO-LOCATED with the white force sphere — the
    unified-geometry tell), the head stays at ``torso_link + 0.35 z``, and the feet are the
    bare ``*_ankle_roll_link`` origins (no sole lever — see ``VR_5POINT_OFFSETS_TCP``).

    Bodies + offsets come from the same single-source-of-truth the bake reads
    (``VR_5POINT_BODIES`` / ``VR_5POINT_OFFSETS_TCP``), so a marker cannot drift from the
    obs point it is supposed to annotate. Real/reference colors share a hue per point, as
    in the 3pt version, so "commanded target vs actual" reads as light-vs-solid.

    Emitted through the SAME ``vr_tracking_point_markers`` config key the 3pt markers use —
    ``WbtTrackingCommand`` parses that key as an arbitrary-length list, so five entries need
    no TS change. Pure viz: no obs term, no physics, no command output. Gated on
    ``vr_5point_pos_tcp`` in build_bundle, so every other preset stays byte-identical.
    """
    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP

    return [
        {
            "body_name": str(body),
            "offset_b": [float(o) for o in offset],
            "real_color": _VR_5POINT_MARKER_RGB[k],
            "ref_color": _VR_5POINT_MARKER_RGB[k],
            "radius": 0.03,
        }
        for k, (body, offset) in enumerate(zip(VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP, strict=True))
    ]


def _strip_policy_onnx(model_path: Path, out_path: Path) -> Path:
    """Strip a trained checkpoint's ONNX to the ``obs -> action`` subgraph the
    browser actually runs, dropping the baked motion-replay tables.

    Holosoma checkpoints export the policy MLP *and* the entire training motion
    library: extra outputs (``joint_pos`` / ``joint_vel`` / ``ref_pos_xyz`` /
    ``ref_quat_xyzw``) fed by ``Gather`` ops over multi-million-row ``Constant``
    tables indexed by a ``time_step`` input. For ``model_26500`` that is ~746 MB
    of a 783 MB file. The browser runtime
    (``runtime.ts``::``runOnnxInference``) reads ONLY ``outKeys[0]`` (the
    action) and feeds ``time_step = 0``; it never touches the replay outputs.
    But onnxruntime-web still downloads and allocates the WHOLE graph, which on
    a 783 MB model stalls / OOMs the tab — the scene-load promise rejects, is
    caught, and retries every ~9 s (perpetual "loading scene").

    Extracting the ``obs -> actions`` subgraph yields a ~37 MB model whose
    ``actions`` are BIT-IDENTICAL to the full model (verified ``max|Δ| = 0`` over
    random obs). ``onnx.utils.extract_model`` discards ``metadata_props`` — which
    the build reads (kp/kd/action_scale/normalizer/dof_names/experiment_config)
    — so we re-attach them to the stripped graph.

    Generic, not checkpoint-specific: the kept input is the single
    non-``time_step`` input and the kept output is ``outputs[0]`` (the action,
    matching the runtime's ``outKeys[0]`` contract). Returns ``model_path``
    unchanged when the model is already minimal (one non-``time_step`` input,
    one output), so a re-exported lean checkpoint is a no-op.
    """
    from onnx import utils as onnx_utils

    model = onnx.load(str(model_path), load_external_data=False)
    in_names = [i.name for i in model.graph.input]
    out_names = [o.name for o in model.graph.output]
    primary = [n for n in in_names if n != "time_step"]
    if len(primary) != 1:
        raise ValueError(
            f"Expected exactly one non-time_step ONNX input, got {primary}. "
            "Update wbt_realtime_eval.main if a multi-input policy needs to be supported."
        )
    if not out_names:
        raise ValueError(f"ONNX {model_path} has no outputs.")
    action_out = out_names[0]

    if in_names == primary and out_names == [action_out]:
        logger.info(f"ONNX already minimal (in={in_names}, out={out_names}); no strip needed.")
        return model_path

    logger.info(
        f"Stripping ONNX {model_path.name}: inputs {in_names} -> {primary}, "
        f"outputs {out_names} -> [{action_out}] (drops baked motion-replay tables "
        "the browser never reads)."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_utils.extract_model(str(model_path), str(out_path), input_names=primary, output_names=[action_out])
    # extract_model drops metadata_props; the build reads kp/kd/etc, so re-attach.
    stripped = onnx.load(str(out_path), load_external_data=False)
    del stripped.metadata_props[:]
    stripped.metadata_props.extend(model.metadata_props)
    onnx.save(stripped, str(out_path))

    orig_mb = model_path.stat().st_size / 1e6
    new_mb = out_path.stat().st_size / 1e6
    logger.info(f"  ONNX {orig_mb:.0f} MB -> {new_mb:.0f} MB (browser-ready policy graph)")
    return out_path


def _onnx_io_keys(model_path: Path) -> tuple[list[str], list[str]]:
    """Return (in_keys, out_keys) directly from the ONNX session.

    Both lists preserve the ONNX model's declared input/output order, because
    mjswan's runtime maps ``inKeys[i] -> session.inputNames[i]`` (and the
    output side similarly) by INDEX, not by name — see
    ``third_party/mjswan/src/mjswan/template/src/core/policy/OnnxModule.ts``,
    ``runInference``. Reordering keys here would silently feed the obs tensor
    into the ``time_step`` slot.
    """
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_keys = [i.name for i in sess.get_inputs()]
    out_keys = [o.name for o in sess.get_outputs()]
    primary = [n for n in in_keys if n != "time_step"]
    if len(primary) != 1:
        raise ValueError(
            f"Expected exactly one non-time_step ONNX input, got {primary}. "
            "Update wbt_realtime_eval.main if a multi-input policy needs to be supported."
        )
    return in_keys, out_keys


def _ground_contact_root_z(
    scene_path: Path, default_qpos: list[float], dof_names: list[str], *, fallback_z: float = 0.78
) -> float:
    """Pelvis height (m) that rests the default standing pose's feet ON the floor.

    The injected ``idle_default`` motion holds the ONNX default joint angles
    (a slightly crouched stance, knees ~0.67 rad). A FIXED pelvis z (the old
    hard-coded 0.78) leaves the feet floating ~2.4 cm, because the pelvis→sole
    distance depends on that crouch, not on a constant. We instead compile the
    actual render scene, set the default pose at an arbitrary pelvis z, run FK,
    find the lowest robot collision geom, and lower the pelvis by exactly that
    clearance so the lowest geom sits at z=0 (the floor plane). Geometry-/hand-
    independent — works for the rubber-hand and the yam scene alike.

    Falls back to ``fallback_z`` (with a warning) if the scene can't be probed.
    """
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        qpos = np.zeros(model.nq)
        qpos[3] = 1.0  # identity root quat (wxyz)
        for name, angle in zip(dof_names, default_qpos, strict=True):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            qpos[model.jnt_qposadr[jid]] = angle
        probe_z = 1.0  # arbitrary; we measure clearance and subtract it
        qpos[2] = probe_z
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        # Lowest extent of every robot geom (skip the floor). For primitive geoms
        # subtract the z half-size so foot capsules' undersides count; meshes use
        # their frame origin (conservative — feet are primitives here anyway).
        zmin = np.inf
        for gi in range(model.ngeom):
            if gi == floor_gid:
                continue
            gz = float(data.geom_xpos[gi][2])
            gtype = model.geom_type[gi]
            sub = 0.0
            if gtype in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX):
                sz = model.geom_size[gi]
                sub = float(sz[2]) if sz[2] > 0 else float(sz[0])
            zmin = min(zmin, gz - sub)
        if not np.isfinite(zmin):
            raise ValueError("no robot geoms found to measure ground clearance")
        # At probe_z the lowest geom is at zmin; drop the pelvis by that clearance.
        root_z = probe_z - zmin
        logger.info(
            f"idle_default pelvis z resolved to {root_z:.4f} m so the default "
            f"stance's feet rest on the floor (was a fixed {fallback_z})."
        )
        return root_z
    except Exception as exc:  # noqa: BLE001 — robustness: never fail the build on this
        logger.warning(
            f"Could not FK-resolve idle_default ground height from {scene_path} "
            f"({exc}); falling back to fixed root_z={fallback_z}."
        )
        return fallback_z


def _bake_idle_default(
    *,
    default_qpos: list[float],
    dof_names: list[str],
    body_names: list[str],
    baked_dir: Path,
    anchor_body_name: str = "pelvis",
    root_z: float = 0.78,
) -> Path:
    """Write a 1-frame default-pose SOURCE NPZ, then bake it so the injected
    ``idle_default`` motion carries REAL FK tables (vr_pos/orn, lower-body,
    ref-root, force_hand_ref) instead of zero-fills.

    Why not ``add_synthetic_default_pose_motion``: its bake-less NPZ makes
    ``WbtTrackingCommand`` zero-fill ``vr_pos_local`` / ``vr_orn_local``, which
    for the FORCE/2pt/3pt presets commands BOTH hands to the pelvis ORIGIN
    (0,0,0) — collapsing the arms instead of holding the URDF hand-raised
    default pose (the default-pose hands sit ~0.35 m out at pelvis-local
    [+0.22, ±0.22, -0.15]). Baking the same 1-frame pose through ``bake_motion``
    (the exact FK real motions use) makes ``idle_default`` HOLD the default
    pose, and populates ``force_hand_ref_local`` so the live force-error metric
    reads ~0 cm when the policy holds the pose at F=0.

    The 7-DOF root prefix on ``joint_pos`` is REQUIRED — ``bake_motion`` rejects
    DOF-only ``joint_pos`` (it expects ``[root_xyz(3), root_quat_wxyz(4),
    dof(N)]``). T=1 is safe: zero ``joint_vel`` → zero ref/lower-body velocity,
    no finite-diff. ``body_lin_vel_w`` / ``body_ang_vel_w`` are REQUIRED keys —
    mjswan's parent ``TrackingCommand.loadMotion`` throws "missing
    body_lin_vel_w" without them.
    """
    from wbt_realtime_eval.bake_motions import bake_motion

    n_dof = len(dof_names)
    n_bodies = len(body_names)
    root_pos = np.array([0.0, 0.0, root_z], dtype=np.float32)
    root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
    dof = np.asarray(default_qpos, dtype=np.float32)
    # 7-prefixed joint_pos: [root_xyz(3), root_quat_wxyz(4), dof(N)] -> (1, 7+N).
    joint_pos = np.concatenate([root_pos, root_quat_wxyz, dof])[None, :].astype(np.float32)
    # joint_vel WITH the 6-prefix [root_lin(3), root_ang(3), dof(N)] -> (1, 6+N), all zero.
    joint_vel = np.zeros((1, 6 + n_dof), dtype=np.float32)
    # Park every body at the pelvis pose. _resolve_root_pose only reads the
    # pelvis index by name; the wrist FK comes from Pinocchio on the DOFs, so
    # the parked body positions are never trusted for the VR tables.
    body_pos_w = np.tile(root_pos, (1, n_bodies, 1)).astype(np.float32)
    body_quat_w = np.tile(root_quat_wxyz, (1, n_bodies, 1)).astype(np.float32)
    body_lin_vel_w = np.zeros((1, n_bodies, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((1, n_bodies, 3), dtype=np.float32)

    # Reserved source stem so it can never collide with a user-supplied clip.
    src_path = baked_dir / "_idle_default_source.npz"
    out_path = baked_dir / "idle_default.npz"
    np.savez(
        src_path,
        fps=np.asarray(50.0, dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        body_names=np.asarray(list(body_names), dtype="U"),
        joint_names=np.asarray(list(dof_names), dtype="U"),
    )
    bake_motion(src_path, out_path, anchor_body_name=anchor_body_name)
    return out_path


# Wrist bodies (left, right) the hand-frame F^cmd obs is rotated into. Match
# HandForceConfig.body_names in
# src/extensions/wbt_wrappers/wbt_training/config_values/force_tracking.py and the
# inference Pinocchio frames (_VR_2POINT_BODIES) — the rubber_hand->wrist_yaw
# fixed joint is rpy=0 so wrist_yaw orientation == the contact-point orientation.
_FORCE_CMD_HAND_FRAME_WRIST_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


def setup_builder(
    *,
    model_path: Path,
    motion_dir: Path,
    output_dir: Path,
    preset: str = _DEFAULT_PRESET,
    scene_xml: Path | None = None,
    per_key_params: dict[str, dict] | None = None,
    force_cmd_hand_frame: bool = False,
    show_control_point: bool = False,
    align_inertia: bool = True,
) -> mjswan.Builder:
    # Fail fast (before the motion bake + ONNX strip) if the holosoma data dir /
    # default scene is missing entirely. The FINAL scene is resolved after the
    # ONNX metadata is read (it may be the yam scene), but the default scene
    # must exist regardless, so this is a cheap up-front sanity check.
    if not _HOLOSOMA_XML.is_file():
        raise FileNotFoundError(f"Holosoma default scene XML not found: {_HOLOSOMA_XML}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion dir not found: {motion_dir}")
    if not any(motion_dir.glob("*.npz")):
        raise FileNotFoundError(f"No *.npz found in {motion_dir}")

    preset_cfg: Preset = get_preset(preset)
    logger.info(
        f"Preset '{preset_cfg.name}': {len(preset_cfg.obs_keys)} terms, "
        f"{preset_cfg.per_frame_dim}D/frame x {preset_cfg.history_length} = {preset_cfg.total_dim}D"
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Bake to a SIBLING dir, not inside output_dir — `builder.build(output_dir)`
    # wipes output_dir before writing the bundle, which would delete the bakes
    # before they're copied into the policy assets.
    baked_dir = output_dir.parent / f"{output_dir.name}_baked"
    baked_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baking motions: {motion_dir} -> {baked_dir}")
    baked_paths = bake_motion_dir(motion_dir, baked_dir)
    logger.info(f"  baked {len(baked_paths)} motion(s)")
    if not baked_paths:
        raise RuntimeError(f"bake_motion_dir produced no outputs from {motion_dir}")

    body_names = _read_body_names(baked_paths[0])
    logger.info(f"  body_names from {baked_paths[0].name}: {len(body_names)} entries")
    for npz in baked_paths[1:]:
        other = _read_body_names(npz)
        if other != body_names:
            raise ValueError(
                f"Motion {npz.name} has inconsistent body_names "
                f"(expected {len(body_names)} entries from {baked_paths[0].name}, "
                f"got {len(other)}). All motions must share the same body ordering."
            )

    # Strip the checkpoint to the obs->action subgraph the browser runs, BEFORE
    # reading metadata / I/O keys / loading the graph — so the config's in_keys /
    # out_keys, the loaded policy, and the served ONNX all describe the SAME lean
    # graph. The strip is a no-op for an already-minimal model and re-attaches
    # metadata_props (kp/kd/action_scale/normalizer/dof_names) that
    # extract_model would otherwise drop. Written to the SIBLING baked_dir so it
    # survives the output_dir wipe in builder.build().
    model_path = _strip_policy_onnx(model_path, baked_dir / "policy_stripped.onnx")

    meta = _read_onnx_metadata(model_path)
    # Pick the MuJoCo render scene (old rubber hand vs yam gripper). Resolved
    # from --scene-xml or auto-derived from the ONNX robot_urdf_path metadata,
    # defaulting to the old-rubber-hand scene for backward compatibility. Only
    # the render mesh / sim inertia differ; the 29-DOF articulation, body/joint
    # names, and the force contact point are identical across variants.
    scene_path = _resolve_scene_xml(scene_xml, meta)
    if not scene_path.is_file():
        raise FileNotFoundError(f"Holosoma scene XML not found: {scene_path}")
    # Re-derive the render scene's fused body inertials from the URDF BAKED INTO
    # THIS CHECKPOINT, so the browser's simulated payload equals the training
    # payload. The hand-maintained MJCF carries a STALE lumped <inertial> for the
    # wrists (0.879804 kg vs the URDF's 1.584577 — 0.705 kg light per hand, from
    # the 2026-07-10 rubber_hand mass bump the MJCF never followed), which makes
    # the arms ride HIGH here on a policy that is actually correct. No-op for a
    # scene that already agrees. See urdf_inertia for the parallel-axis math and
    # why we mutate the spec instead of editing gitignored thirdparty/ MJCF.
    scene_spec = mujoco.MjSpec.from_file(str(scene_path))
    if align_inertia:
        _align_scene_inertia_to_checkpoint(scene_spec, meta, scene_path)
    else:
        logger.warning(
            "--no-align-inertia: keeping the render scene's own <inertial> values. "
            "The yam scene's wrists are 0.705 kg light vs the training URDF, so the "
            "arms will ride high; use this only to reproduce a pre-2026-08-05 bundle."
        )
    # ONNX metadata's `dof_names` is the training joint order; the browser must
    # use the same order for the policy obs to land in the right positions.
    onnx_dof_names = json.loads(meta.get("dof_names", "[]")) if "dof_names" in meta else []
    if onnx_dof_names and onnx_dof_names != list(g1_29dof.dof_names):
        raise ValueError(
            f"ONNX dof_names mismatch with g1_29dof.dof_names. ONNX: {onnx_dof_names}; "
            f"config: {list(g1_29dof.dof_names)}"
        )
    dof_names = list(g1_29dof.dof_names)
    # Use ONNX-baked default_joint_angles for the joint_pos obs offset (matches
    # what the policy was trained with, regardless of the static config).
    default_qpos = _parse_default_joint_angles(meta, dof_names)
    kp, kd = _parse_kp_kd(meta)
    action_scale = _parse_action_scale(meta)
    norm_mean, norm_std = _parse_normalizer(meta)

    in_keys, out_keys = _onnx_io_keys(model_path)
    expected_dim = preset_cfg.total_dim
    if len(norm_mean) != expected_dim:
        raise ValueError(
            f"Preset '{preset_cfg.name}' expects obs dim {expected_dim} but the ONNX "
            f"normalizer is length {len(norm_mean)}. Either pick the right --preset "
            f"or re-export the checkpoint."
        )
    logger.info(f"ONNX I/O: in={in_keys} out={out_keys}")
    _assert_checkpoint_vr_geometry(meta, preset_cfg)
    # Dense-vs-keypoint family check. Separate from the geometry check above because it
    # answers a different question, and load-bearing for the 732D dense/global-vel collision.
    _assert_checkpoint_tracking_family(meta, preset_cfg)

    obs_funcs = obs_registry.register()
    per_key_params = per_key_params or {}
    components: list[dict] = []
    for key in preset_cfg.obs_keys:
        entry: dict = {"name": obs_funcs[key].ts_name}
        # WbtJointPos needs the ONNX-baked default_joint_angles to subtract
        # in computeCurrent; pass them via the obs term's config object.
        if key == "joint_pos":
            entry["default_joint_angles"] = default_qpos
        # Per-term extra config (e.g. the live VR terms get vive_ws_url + the
        # default-pose HOME fallback). main_vive.py supplies these; the existing
        # replay path passes nothing, so this is a no-op there.
        if key in per_key_params:
            entry.update(per_key_params[key])
        # Hand-frame F^cmd ablation (--F_on_hand_frame checkpoints): tell the
        # WbtForceCmdLocal obs term to rotate the pelvis-frame slider Newtons
        # into each wrist (wrist_yaw) frame, matching what such a checkpoint was
        # trained on. OPT-IN only — when force_cmd_hand_frame is False we add
        # NOTHING, so the components dict (and thus every existing pelvis bundle)
        # is byte-identical to before. Affects ONLY the actor's observation; the
        # F^ext physics and the purple F^cmd debug arrow stay pelvis-frame.
        if key == "force_cmd_local" and force_cmd_hand_frame:
            entry["hand_frame"] = True
            entry["wrist_body_names"] = list(_FORCE_CMD_HAND_FRAME_WRIST_BODIES)
        components.append(entry)
    # Always append the metrics tap last so the panel runs every step. It has
    # size==0 so it doesn't change the obs vector.
    components.append({"name": obs_funcs["_metrics_tap"].ts_name})

    onnx_filename = f"{preset_cfg.name.replace('-', '_')}.onnx"
    policy_config_path = output_dir.parent / f"{output_dir.name}_policy_config.json"
    _write_policy_config(
        policy_config_path, components, onnx_filename=onnx_filename, in_keys=in_keys, out_keys=out_keys
    )

    builder = mjswan.Builder()
    project = builder.add_project(name=f"WBT Realtime Eval [{preset_cfg.name}]")
    scene = project.add_scene(name="G1 29dof", spec=scene_spec)
    scene.set_viewer_config(mjswan.ViewerConfig(lookat=(0.0, 0.0, 0.8), distance=3.0, elevation=-15.0, azimuth=135.0))

    stiffness = dict(zip(dof_names, kp, strict=True))
    damping = dict(zip(dof_names, kd, strict=True))

    # Per-preset extras: force preset gets six sliders + per-hand viz config.
    # Other presets get neither, which exercises the no-op path on the mjswan
    # side and keeps their bundles unchanged.
    extras: dict[str, object] = {
        "normalizer_mean": norm_mean,
        "normalizer_std": norm_std,
        "normalizer_clip": 3.0,
        "normalizer_eps": 1e-2,
        "default_joint_angles": default_qpos,
        "preset": preset_cfg.name,
    }
    motion_command_params: dict[str, object] = {}
    # Preset dispatch on the OBS LAYOUT (not the preset NAME):
    #   * HYBRID (force_meas_local present) — the causal spring wall + F/T sensor
    #     + Jᵀ servo path. Emits extras.hybrid (the runtime reconstructs the wall +
    #     servo), the 12 F^cmd/F^ext sliders, and the 3-arrow (purple F^cmd, orange
    #     F^ext, cyan F^meas) TCP-anchored viz. F^ext is NOT injected via
    #     force_ext_bodies here (it is an ADDITIVE manual disturbance the runtime
    #     already handles inside the hybrid loop via applyExternalForces reading the
    #     same slider keys — see the hybrid branch of executeSimulationSteps).
    #   * FORCE (force_cmd_local present, no force_meas_local) — the shipped UniFP
    #     path: 12 sliders + force_ext_bodies xfrc injection + purple/orange arrows.
    # Every non-force/non-hybrid bundle stays byte-identical (guarded by
    # test_force_extras_gate_is_obs_key_driven).
    if "force_meas_local" in preset_cfg.obs_keys:
        # servo_enabled: from the checkpoint, but force OFF for the -layerA preset
        # name so run 6y8314vq renders as Layer-A regardless of its baked flag.
        servo_override = False if "layera" in preset_cfg.name.lower() else None
        # servo_observe_only: force TRUE for the -unified preset names (C1 single-
        # policy). R2 bakes servo_observe_only=True anyway, but R1 (-unified-anneal)
        # bakes the FULL-servo cfg — its terminal α=0 is curriculum runtime state,
        # not serialized — so the name override is what upholds "neither unified twin
        # injects a servo at deploy". NOT the layerA pattern: servo_enabled stays
        # true so the runtime keeps publishing the wound I/Δx the policy needs.
        observe_only_override = True if "unified" in preset_cfg.name.lower() else None
        hybrid_cfg = _parse_hybrid_cfg(meta)
        if hybrid_cfg is None:
            # f-string, not printf args — loguru formats with str.format (see the note at
            # the vr_3point marker log below); "%r" would print literally.
            logger.warning(
                f"Hybrid preset {preset_cfg.name!r} but no HybridContactConfig found in ONNX "
                "experiment_config; using training defaults for the wall/servo constants."
            )
        hy_sliders, hy_extras, hy_viz, hy_ext_bodies = _hybrid_preset_extras(
            hybrid_cfg, servo_enabled_override=servo_override, servo_observe_only_override=observe_only_override
        )
        extras["sliders"] = hy_sliders
        extras["hybrid"] = hy_extras
        # The runtime's per-substep applyExternalForces() reads force_ext_bodies to
        # inject the MANUAL F^ext slider disturbance (additive on top of the causal
        # wall). Without this the F^ext sliders are dead + the orange arrow lies.
        extras["force_ext_bodies"] = hy_ext_bodies
        # WbtTrackingCommand draws the per-hand triad + THREE arrows (purple F^cmd,
        # orange F^ext, cyan F^meas) + the spring wall + the servo-phase badge.
        motion_command_params["force_viz_bodies"] = hy_viz
        motion_command_params["hybrid_viz"] = {
            "servo_enabled": hy_extras["servo_enabled"],
            # C1 unified: the UI must not claim a Jᵀ servo is actuating when it is not.
            # With this true, WbtTrackingCommand replaces the "Force servo (Layer B)"
            # CHECKBOX with a read-only observe-only notice (the runtime ignores the
            # toggle in this mode — flipping it would only mislead) and the badge reads
            # "UNIFIED POLICY (…)" instead of "FORCE SERVO / POSITION TRACKING (servo
            # off)". Absent/false for every pre-unified preset → their UI is unchanged.
            "servo_observe_only": hy_extras["servo_observe_only"],
            "tcp_offset_x": hy_extras["tcp_offset_x"],
            "max_penetration_m": hy_extras["max_penetration_m"],
            "plane_dist": hy_extras["plane_dist"],
            "hands": [
                {"body_name": h["body_name"], "cmd_slider_keys": h["cmd_slider_keys"]} for h in hy_extras["hands"]
            ],
        }
        # Mark the TCP (the force contact point) with a small WHITE sphere — the
        # SAME point the arrows/wall/F^meas act on (tcp_offset_x, 0, 0). Reuses the
        # vr_control_point_bodies sphere-drawing mechanism but anchored at the TCP,
        # NOT the 0.18 m VR-tracked point (which would sit ~6 cm off the arrows and
        # read as "not at the tap center"). This SUPERSEDES the --show-control-point
        # VR sphere for hybrid (gated out below), so there is exactly one marker.
        _tcp_off = [float(hy_extras["tcp_offset_x"]), 0.0, 0.0]
        motion_command_params["vr_control_point_bodies"] = [
            {"body_name": h["body_name"], "offset_b": _tcp_off, "color": _HYBRID_TCP_MARKER_RGB, "radius": 0.02}
            for h in hy_extras["hands"]
        ]
        if hy_extras["servo_observe_only"]:
            # f-string, NOT loguru's {}-format args: the message names the extras keys
            # wbt_cphase__l / __r, and any literal brace in a loguru message with args
            # is parsed as a format placeholder (a "wbt_cphase__{l,r}" spelling raises
            # KeyError: 'l,r' at build time).
            logger.info(
                "Hybrid UNIFIED preset (C1 single-policy): causal spring wall + F/T "
                "sensor (F^meas) + contact-gated integrator all LIVE, but the Jᵀ servo "
                "is OBSERVE-ONLY — I/Δx are published into the actor obs (reference "
                "generator) and Δq_arm is NEVER injected; the policy is the sole "
                "executor. 12 sliders, 3-arrow TCP viz, soft contact_phase "
                f"(theta_on={hy_extras['contact_phase_theta_on']} / "
                f"theta_off={hy_extras['contact_phase_theta_off']} N, "
                f"tau={hy_extras['contact_phase_tau_s']} s) -> wbt_cphase__l / __r."
            )
        else:
            logger.info(
                "Hybrid preset: causal spring wall + F/T sensor (F^meas) + Jᵀ servo "
                f"(servo_enabled={hy_extras['servo_enabled']}), 12 sliders, 3-arrow TCP viz "
                "(purple F^cmd / orange F^ext / cyan F^meas), wall + servo-phase badge."
            )
    elif "force_cmd_local" in preset_cfg.obs_keys:
        force_sliders, force_viz_bodies, force_ext_bodies = _force_preset_extras()
        extras["sliders"] = force_sliders
        # The runtime patch (applyExternalForces) reads `force_ext_bodies` from
        # the policy extras to inject F^ext into xfrc_applied at the rubber-hand
        # point each control step. F^ext is NEVER an obs term — it only affects
        # physics (and the orange arrow).
        extras["force_ext_bodies"] = force_ext_bodies
        # WbtTrackingCommand reads `force_viz_bodies` from its config to draw
        # the per-hand triad + the two force arrows (purple F^cmd, orange F^ext)
        # each render frame.
        motion_command_params["force_viz_bodies"] = force_viz_bodies
        logger.info(
            "Force preset: 12 live sliders (6 F^cmd obs + 6 F^ext physics), "
            "2 per-hand visualizers (purple F^cmd + orange F^ext arrows), "
            "F^ext xfrc injection wired up"
        )

    # LIVE 3pt teleop: draw an RGB axis triad at each of the three COMMANDED target
    # poses the bridge is feeding (left TCP, right TCP, head). Gated on the LIVE 3pt
    # obs term, so it appears for exactly the teleop preset and nothing else — every
    # replay and 2pt bundle keeps its motion_command_params byte-identical. Always on
    # for that preset (not behind a flag): reading the commanded-vs-achieved
    # correspondence IS the reason to run the sim teleop.
    if "vr_3point_pos_tcp_live" in preset_cfg.obs_keys:
        motion_command_params["vr_live_target_triads"] = _vr3_live_target_triad_extras()
        logger.info(
            "Live 3pt teleop: RGB axis triad + ball at each commanded target "
            "(magenta L TCP / yellow R TCP / green head); dims when the bridge is "
            "stale. The white spheres are the ACHIEVED TCPs — compare the two."
        )

    # Opt-in orange CONTROL-POINT marker: one sphere per hand at the policy's
    # vr_2point contact point (where the VIVE tracker actually steers, at the 0.18 m
    # lever). Pure viz (no obs / physics / command change), gated on the 2pt obs term.
    # SKIPPED for hybrid presets: they already draw a WHITE marker at the TCP (the
    # force contact point) above — drawing the 0.18 m VR sphere too would put a
    # second marker ~6 cm off the arrows and read as "not at the tap center".
    _is_hybrid = "force_meas_local" in preset_cfg.obs_keys
    if show_control_point and _is_hybrid:
        logger.info(
            "Control-point marker: hybrid preset already marks the TCP (white sphere); skipping the VR-point sphere."
        )
    elif show_control_point and ("vr_2point_pos" in preset_cfg.obs_keys or "vr_2point_pos_live" in preset_cfg.obs_keys):
        motion_command_params["vr_control_point_bodies"] = _vr_control_point_extras()
        logger.info("Control-point marker: orange sphere per hand at the vr_2point contact point (pure viz).")
    elif show_control_point:
        logger.warning(
            f"show_control_point requested but preset {preset_cfg.name!r} has no vr_2point term; skipping the marker."
        )

    # vr_3point TRACKING-point markers: always-on (NOT opt-in) small spheres at the
    # THREE points the 3pt policy tracks — left hand, right hand, head. Drawn on BOTH
    # the live robot (solid) and the reference ghost (translucent, same hue) so
    # "target vs actual" reads at a glance. Gated on vr_3point_pos so every non-3pt
    # preset (2pt / force / global / maskmimic) stays byte-identical. This is a NEW
    # key, independent of vr_control_point_bodies (the hybrid TCP white ball above),
    # so the two coexist on a 3pt-hybrid preset without conflict.
    #
    # Both position-slot spellings gate this: `vr_3point_pos` (0.18 m VR lever) and
    # `vr_3point_pos_tcp` (0.242 m force TCP, exp:g1-wbt-3pt-hybrid-unified[-anneal]).
    # The marker offsets FOLLOW the active slot so the sphere always sits on the point
    # the loaded policy actually tracks — an exact-membership check on the old name alone
    # would have silently dropped the markers for the TCP-unified presets.
    # The LIVE spellings gate these too. The markers ride the ROBOT (and the ghost), not
    # the command, so they are the ACHIEVED keypoints — which is precisely the
    # commanded-vs-achieved read a teleop session is for. Dropping them on a `_live` preset
    # would remove the one visual that says whether the policy got there.
    _keys = {k.removesuffix("_live") for k in preset_cfg.obs_keys}
    _vr3_tcp = "vr_3point_pos_tcp" in _keys
    if _vr3_tcp or "vr_3point_pos" in _keys:
        motion_command_params["vr_tracking_point_markers"] = _vr_tracking_point_markers_extras(tcp_unified=_vr3_tcp)
        # f-string, NOT loguru's %s/{}-style args: loguru formats with str.format, so a
        # printf-style "%s" placeholder is emitted LITERALLY (observed: the build log read
        # "...(translucent)%s." and the TCP confirmation never appeared — precisely the line
        # that tells the operator which hand geometry is live).
        _geom_note = (
            " — hands on the TCP-UNIFIED 0.242 m point (co-located with the white force marker)" if _vr3_tcp else ""
        )
        logger.info(
            "vr_3point tracking-point markers: 3 spheres (left hand / right hand / head) "
            f"on BOTH the live robot (solid) and the reference ghost (translucent){_geom_note}."
        )

    # vr_5point TRACKING-point markers (exp:g1-wbt-5pt-hybrid-unified-anneal): the same
    # always-on mechanism with FIVE spheres — the 3pt hues for hands/head plus two greens
    # for the feet. `elif`-style exclusivity is structural, not enforced here: a preset
    # carries EITHER a vr_3point_pos* slot OR vr_5point_pos_tcp, never both (widths differ:
    # 9 vs 15), so the two branches cannot both write vr_tracking_point_markers. Gated on
    # the 5pt slot, so every 2pt/3pt/dense/global preset stays byte-identical.
    if "vr_5point_pos_tcp" in _keys:
        motion_command_params["vr_tracking_point_markers"] = _vr_5point_tracking_point_markers_extras()
        # f-string, NOT loguru {}-args — same reason as the 3pt log above.
        logger.info(
            "vr_5point tracking-point markers: 5 spheres (left hand / right hand / head / "
            "left foot / right foot) on BOTH the live robot (solid) and the reference ghost "
            "(translucent) — hands on the TCP-UNIFIED 0.242 m point (co-located with the "
            "white force marker), feet at the bare ankle_roll origins."
        )

    policy = scene.add_policy(
        name=preset_cfg.name.replace("-", "_"),
        policy=onnx.load(str(model_path), load_external_data=True),
        config_path=str(policy_config_path),
        commands={"motion": mjswan.CommandTermConfig(term_name="WbtTrackingCommand", params=motion_command_params)},
        actions={
            "joint_pos": JointPositionActionCfg(
                actuator_names=(".*",),
                scale=action_scale,
                use_default_offset=True,
                stiffness=stiffness,
                damping=damping,
            )
        },
        policy_joint_names=dof_names,
        default_joint_pos=default_qpos,
        # Extras: surfaced to the browser runtime as `config.extras`. Used by
        # the patched runtime.ts for normalizer pre-clip, to rehydrate the
        # ONNX-baked default_joint_angles on the JS side, and (for the force
        # preset) to drive the SliderPanel. See
        # third_party/mjswan/src/mjswan/template/src/core/engine/runtime.ts
        # (search for `onnxNormMean`).
        extras=extras,
        default=True,
    )

    # Always inject an "idle_default" motion BEFORE the baked motions. This
    # guarantees every bundle has at least one motion in the picker (so an
    # empty/exotic --motion-dir doesn't strand the user with a blank dropdown)
    # and gives them an explicit "hold the default pose" choice — ideal for a
    # clean F^ext step-response test. Unlike the old bake-less synthetic clip
    # (which zero-filled VR targets and collapsed the arms), this is a real
    # 1-frame bake, so it HOLDS a well-posed reference and its
    # force_hand_ref_local drives the live hand-error metric.
    #
    # REFERENCE pose = trained default with the ARM joints zeroed (upper arm
    # vertical / forearm horizontal), per the user's request. Legs + waist keep the
    # trained default so the stance is unchanged. This arm-zeroed pose is used ONLY
    # for the idle_default clip + its ground-clearance FK — NOT for the obs/action
    # default offsets (which stay = default_qpos, the trained values), so the
    # policy obs/action contract is untouched and the robot does not topple.
    idle_ref_qpos = _arm_zeroed_reference_pose(default_qpos, dof_names)
    idle_default_is_default = not any(npz.stem == "idle_stand_1" for npz in baked_paths)
    # FK-resolve the pelvis height for the SAME render scene so the reference
    # stance's feet rest on the floor (instead of a fixed z that floats them). Uses
    # the arm-zeroed reference pose so the clip's own pose sits on the ground (arm
    # angles don't affect foot height, but keep the FK consistent with the clip).
    idle_root_z = _ground_contact_root_z(scene_path, idle_ref_qpos, dof_names)
    idle_npz = _bake_idle_default(
        default_qpos=idle_ref_qpos,
        dof_names=dof_names,
        body_names=body_names,
        baked_dir=baked_dir,
        anchor_body_name="pelvis",
        root_z=idle_root_z,
    )
    policy.add_motion(
        name="idle_default",
        source=str(idle_npz),
        fps=50.0,
        anchor_body_name="pelvis",
        body_names=tuple(body_names),
        dataset_joint_names=dof_names,
        default=idle_default_is_default,
        loop=False,
    )

    # Register the baked motions; the runtime spreads them into the command
    # term's `motions` array automatically (see runtime.ts:644-648).
    have_idle_stand = any(npz.stem == "idle_stand_1" for npz in baked_paths)
    for npz in baked_paths:
        # Skip a user-supplied `idle_default` source — the injected one above
        # owns that name (avoids a duplicate-name / bundle-filename collision).
        if npz.stem == "idle_default":
            logger.warning(
                "Source motion 'idle_default' shadows the injected default-pose clip; skipping the user copy."
            )
            continue
        # If the user supplied an `idle_stand_1`, keep its historical role
        # as the default so existing bundles render the same as before.
        # Otherwise the injected idle_default holds the default slot.
        is_default = have_idle_stand and npz.stem == "idle_stand_1"
        policy.add_motion(
            name=npz.stem,
            source=str(npz),
            fps=50.0,
            anchor_body_name="pelvis",
            body_names=tuple(body_names),
            dataset_joint_names=dof_names,
            default=is_default,
            loop=False,
        )

    return builder


def main() -> None:
    parser = argparse.ArgumentParser(prog="wbt-realtime-eval")
    parser.add_argument(
        "--preset",
        default=_DEFAULT_PRESET,
        help=(
            "Inference preset name (matches batch_headless_eval.sh --preset). "
            f"Default: {_DEFAULT_PRESET}. See wbt_realtime_eval.presets.PRESETS for "
            "the full list."
        ),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("dist/wbt_realtime_eval"), type=Path)
    parser.add_argument(
        "--scene-xml",
        default=None,
        type=Path,
        help=(
            "MuJoCo render scene XML (absolute or repo-root-relative). Mirrors "
            "batch_headless_eval.sh --xml. If omitted, the scene is auto-derived "
            "from the ONNX robot_urdf_path metadata (e.g. a yam-gripper-trained "
            "checkpoint picks scene_g1_29dof_wbt_plane_yam.xml), falling "
            "back to the default old-rubber-hand scene. Pass this to force a "
            "specific scene."
        ),
    )
    parser.add_argument("--no-launch", action="store_true", help="Build only; don't start local HTTP server.")
    parser.add_argument(
        "--force-cmd-hand-frame",
        action="store_true",
        help=(
            "Feed the actor F^cmd in each HAND (wrist_yaw) frame instead of the "
            "pelvis frame — for a --F_on_hand_frame checkpoint. Mirrors the "
            "headless eval's --force-cmd-hand-frame (HOLOSOMA_FORCE_CMD_HAND_FRAME=1). "
            "Opt-in; without it the bundle is byte-identical to the pelvis-frame "
            "build. Rotates ONLY the actor's observation; the F^ext physics and the "
            "purple F^cmd debug arrow stay pelvis-frame."
        ),
    )
    parser.add_argument(
        "--show-control-point",
        action="store_true",
        help=(
            "Draw an orange sphere at the policy's vr_2point CONTACT point per "
            "hand (wrist_yaw + R(wrist) @ the 0.18 m VR lever) — the point the "
            "tracker steers. Pure viz; opt-in, byte-identical when off. Only the "
            "2pt presets have a vr_2point term."
        ),
    )
    parser.add_argument(
        "--no-align-inertia",
        dest="align_inertia",
        action="store_false",
        help=(
            "Keep the render scene's authored <inertial> values instead of "
            "re-deriving them from the checkpoint's baked URDF. Default is to "
            "align, because the yam scene's fused wrist inertial is STALE "
            "(0.879804 kg vs the URDF's 1.584577 — 0.705 kg light per hand), which "
            "makes the arms ride high on an otherwise-correct policy. Use this only "
            "to reproduce a pre-2026-08-05 bundle."
        ),
    )
    args = parser.parse_args()

    builder = setup_builder(
        preset=args.preset,
        model_path=args.model_path.resolve(),
        motion_dir=args.motion_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        scene_xml=args.scene_xml,
        force_cmd_hand_frame=args.force_cmd_hand_frame,
        show_control_point=args.show_control_point,
        align_inertia=args.align_inertia,
    )
    app = builder.build(output_dir=str(args.output_dir.resolve()))
    if args.no_launch or os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
