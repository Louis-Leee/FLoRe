"""Pre-bake VR forward-kinematics + lower-body command tables for browser eval.

The browser-side eval for any holosoma-*-wbt preset needs a handful of
per-frame derived tables that are too expensive (or too dependency-heavy) to
compute in the browser:

* VR 2pt and 3pt body poses (FK on the URDF)
* lower-body joint position/velocity slices
* the world-frame anchor pose/velocity of the reference motion

Computing those in JS would require shipping Pinocchio + the URDF and
reimplementing the dof reorder; instead we pre-compute everything here once
at bundle build time and merge the result back into a copy of the source
motion NPZ.

mjswan's ``policy.add_motion(source=path)`` only bundles a single NPZ per motion
into the static-site dist, so we keep everything in one file.  The merged NPZ is
browser-shaped: ``joint_pos`` / ``joint_vel`` are stripped to the 29 MuJoCo-order
DOFs that mjswan's ``TrackingCommand`` writes into hinge joints, while
``source_joint_pos`` / ``source_joint_vel`` preserve the original root-prefixed
Holosoma arrays for diagnostics.  The merged file is written to a build-time
output dir; the source dataset under ``wbt_training/motions/`` is never
modified.

The FK math mirrors :func:`wbt_wrappers_inference.policies.holosoma_wbt
.HolosomaWBTPolicy._compute_vr_fk` (see lines ~541-584 in that module) so the
runtime and the bake produce numerically equivalent VR commands.

Added keys (per motion)::

    vr_pos_local   (T, 6)  float32 — 2pt left/right wrist xyz in anchor frame
    vr_orn_local   (T, 12) float32 — 2pt left/right wrist 6D rot rep (first two
                                     columns of the rotation matrix, per wrist)
    vr3_pos_local  (T, 9)  float32 — 3pt body xyz in anchor frame
    vr3_pos_tcp_local (T, 9) float32 — as vr3_pos_local but with both HAND points on the
                                     0.242 m fingertip TCP (the force anchor) instead of
                                     the 0.18 m VR-tracker lever — the TCP-unified
                                     geometry of exp:g1-wbt-3pt-hybrid-unified[-anneal]
    vr3_orn_local  (T, 12) float32 — 3pt body xyzw quat
    vr3_orn6d_local (T, 18) float32 — 3pt body 6D rot rep (first two columns of the
                                     rotation matrix, per body) — for the 6D-orn 3pt
                                     preset (exp:g1-wbt-3pt-hybrid-global-vel)
    vr5_pos_tcp_local (T, 15) float32 — vr3_pos_tcp_local with BOTH FEET
                                     (``*_ankle_roll_link``, bare origin) appended, i.e.
                                     the 5-point tracking set of
                                     exp:g1-wbt-5pt-hybrid-unified-anneal
    vr5_orn_local  (T, 20) float32 — vr3_orn_local with the two feet appended (xyzw quat
                                     per body, 5 x 4)
    lower_body_jp  (T, 12) float32 — dof_pos[LOWER_BODY_INDICES]
    lower_body_jv  (T, 12) float32 — dof_vel[LOWER_BODY_INDICES] (zeros if missing)
    ref_root_pos_w (T, 3)  float32 — anchor body world position
    ref_root_vel_w (T, 3)  float32 — anchor body world linear velocity (finite-diff if missing)
    force_hand_ref_local (T, 6) float32 — reference rubber-hand xyz [L, R] in the
                                     pelvis frame, FORCE offset (0.0815, ±0.003);
                                     drives the browser's live per-hand force error
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

# Repo root: <root>/src/extensions/wbt_wrappers/wbt_realtime_eval/
#                                       wbt_realtime_eval/bake_motions.py
# parents[0] inner pkg -> [1] outer pkg dir -> [2] wbt_wrappers ->
# [3] extensions -> [4] src -> [5] repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_URDF = _REPO_ROOT / "src/extensions/wbt_wrappers/wbt_wrappers_inference/urdf/g1_29dof.urdf"


def _load_pinocchio_robot():
    """Build a :class:`PinocchioRobot` for g1_29dof using the on-disk URDF.

    We use ``_g1_29dof_wbt_wrapper_robot`` from the inference config values so
    that ``robot.dof_names`` and ``robot.motion["body_name_ref"]`` match what
    the runtime policy uses.  If ``motion`` is ever ``None`` we fall back to a
    pelvis anchor — the bake itself doesn't use ``ref_body_frame_id``, but
    ``PinocchioRobot.__init__`` requires a valid body name.
    """
    from holosoma_inference.policies.wbt_utils import PinocchioRobot
    from wbt_wrappers_inference.config_values.inference import _g1_29dof_wbt_wrapper_robot

    robot_cfg = _g1_29dof_wbt_wrapper_robot
    if not robot_cfg.motion or not robot_cfg.motion.get("body_name_ref"):
        robot_cfg = dataclasses.replace(robot_cfg, motion={"body_name_ref": ["pelvis"]})

    if not _DEFAULT_URDF.is_file():
        raise FileNotFoundError(f"URDF not found at {_DEFAULT_URDF}")

    urdf_text = _DEFAULT_URDF.read_text()
    return PinocchioRobot(robot_cfg, urdf_text)


def _resolve_root_pose(
    data: dict[str, np.ndarray], anchor_body_name: str
) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
    """Extract per-frame floating-root xyz + wxyz quaternion from the motion NPZ.

    Prefers ``body_pos_w[:, body_idx]`` / ``body_quat_w[:, body_idx]`` where
    ``body_idx`` is looked up by name from ``body_names``.  Falls back to body
    index 0 with a warning when ``body_names`` is missing or doesn't contain
    ``anchor_body_name``.  As a last resort (no ``body_pos_w``/``body_quat_w``
    arrays at all) reads ``joint_pos[:, :3]`` + ``joint_pos[:, 3:7]``.
    """
    if "body_pos_w" in data and "body_quat_w" in data:
        body_idx = 0
        if "body_names" in data:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in data["body_names"].tolist()]
            try:
                body_idx = names.index(anchor_body_name)
            except ValueError:
                # Fall back to body 0 with a warning, since some motions don't include the anchor explicitly
                logger.warning(f"anchor body {anchor_body_name!r} not in body_names; falling back to index 0")
        root_pos = np.asarray(data["body_pos_w"][:, body_idx], dtype=np.float32)
        root_quat_wxyz = np.asarray(data["body_quat_w"][:, body_idx], dtype=np.float32)
        return root_pos, root_quat_wxyz

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    if joint_pos.shape[1] < 7:
        raise ValueError("joint_pos must include 7-dof root prefix when body_pos_w/body_quat_w are absent")
    root_pos = joint_pos[:, :3].astype(np.float32)
    root_quat_wxyz = joint_pos[:, 3:7].astype(np.float32)
    return root_pos, root_quat_wxyz


def _resolve_dof_pos_vel(
    data: dict[str, np.ndarray], num_dofs: int
) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
    """Extract per-frame DOF positions and velocities (29-vector, full DOF set).

    ``joint_pos`` may be (T, num_dofs+7) [root_xyz(3) + root_quat_wxyz(4) +
    dof(N)] or (T, num_dofs) [bare DOF only].  ``joint_vel`` may be (T,
    num_dofs+6) [root_lin(3) + root_ang(3) + dof(N)] or (T, num_dofs).  The
    velocity check mirrors the runtime split at
    ``holosoma_wbt.py:472`` (``raw_vel[6:] if raw_vel.shape[0] > num_dofs``).
    If ``joint_vel`` is missing we return zeros so downstream consumers always
    have a 29-vector (full DOF set).
    """
    joint_pos = np.asarray(data["joint_pos"])
    if joint_pos.shape[1] >= num_dofs + 7:
        dof_pos = joint_pos[:, 7 : 7 + num_dofs].astype(np.float32)
    elif joint_pos.shape[1] == num_dofs:
        dof_pos = joint_pos.astype(np.float32)
    else:
        raise ValueError(f"joint_pos has {joint_pos.shape[1]} cols, expected {num_dofs} or {num_dofs + 7}")

    if "joint_vel" in data:
        joint_vel = np.asarray(data["joint_vel"])
        if joint_vel.ndim != 2:
            raise ValueError(f"joint_vel must be 2D, got shape {joint_vel.shape}")
        if joint_vel.shape[1] > num_dofs:
            # Has [root_lin(3), root_ang(3), dof(N)] prefix
            dof_vel = joint_vel[:, 6 : 6 + num_dofs].astype(np.float32)
        elif joint_vel.shape[1] == num_dofs:
            dof_vel = joint_vel.astype(np.float32)
        else:
            raise ValueError(f"joint_vel has {joint_vel.shape[1]} cols, expected {num_dofs} or {num_dofs + 6}")
    else:
        dof_vel = np.zeros_like(dof_pos)
    return dof_pos, dof_vel


#: Semantic version of the baked tables' CONTENTS, stamped into every output as
#: ``bake_format_version`` and checked by :func:`_is_current_browser_bake`.
#:
#: BUMP THIS whenever a change alters what an existing key CONTAINS rather than adding a new
#: key. ``bake_motion_dir`` reuses any output whose mtime beats its source, and the schema
#: check is key-based, so without a version stamp a corrected bake is silently skipped and
#: the stale numbers ship — the rebuild even logs "baked N motion(s)".
#:
#: 1 — pre-2026-08-30 (unstamped; treated as version 0 and always re-baked).
#: 2 — 2026-08-30: quaternion orientation tables (``vr3_orn_local``, ``vr5_orn_local``)
#:     canonicalized to the ``w >= 0`` double-cover branch, the one the quaternion-orn
#:     presets trained on. Values only; every key and width is unchanged.
_BAKE_FORMAT_VERSION = 2


def _is_current_browser_bake(npz_path: Path, num_dofs: int = 29) -> bool:
    """Return True when an existing baked NPZ matches the browser schema.

    The schema includes the always-present 2pt VR tables (pre-existing) plus
    the 3pt VR tables and the world-frame ref-root pose/vel arrays added
    when the bake was extended to support every holosoma-*-wbt preset.
    Older bakes that lack the new keys return False so callers re-run.
    """
    try:
        with np.load(npz_path) as data:
            required = {
                "source_joint_pos",
                "source_joint_vel",
                "joint_pos",
                "joint_vel",
                "vr_pos_local",
                "vr_orn_local",
                "vr3_pos_local",
                "vr3_orn_local",
                # The 6D 3pt orientation table (exp:g1-wbt-3pt-hybrid-global-vel). REQUIRED
                # so a pre-6D bake is re-run instead of silently reused: the browser's
                # width guard only throws on WRONG width, not ABSENCE, so a missing key
                # would feed 18 zeros into the vr_3point_orn_6d obs slot (OOD orientation).
                "vr3_orn6d_local",
                # TCP-unified 3pt hand point (exp:g1-wbt-3pt-hybrid-unified[-anneal],
                # 2026-08-03). REQUIRED for the same reason as vr3_orn6d_local: the TS
                # accessor falls back to the 0.18 m table when absent, which would train-
                # deploy-mismatch by 6.2 cm silently. Listing it here forces a re-bake.
                "vr3_pos_tcp_local",
                # 5-point tables (exp:g1-wbt-5pt-hybrid-unified-anneal, 2026-08-27). Same
                # reason again: their TS accessors zero-fill when absent, which would drive
                # all five tracked points to the pelvis origin. Listing them forces a
                # re-bake of any pre-2026-08-27 NPZ that is otherwise mtime-current.
                "vr5_pos_tcp_local",
                "vr5_orn_local",
                "lower_body_jp",
                "lower_body_jv",
                "ref_root_pos_w",
                "ref_root_vel_w",
                # Semantic version stamp — see the version gate below. Absent in every
                # pre-2026-08-30 bake, so listing it here forces those to be re-baked.
                "bake_format_version",
            }
            if not required.issubset(set(data.files)):
                return False
            # SEMANTIC version gate, on top of the key-presence check above. Every earlier
            # extension to this bake happened to ADD a key, so key presence was enough to
            # invalidate a stale cache. A change to what an EXISTING key CONTAINS is
            # invisible to that test — and `bake_motion_dir` otherwise reuses any output
            # whose mtime beats its source, so a corrected bake would be silently skipped
            # and the old numbers shipped. (Exactly what happened when the quaternion
            # double-cover canonicalization landed: rebuild reported "baked 2 motion(s)"
            # and re-emitted the unfixed tables.) Bump _BAKE_FORMAT_VERSION whenever the
            # VALUES change meaning; a missing key reads as version 0.
            if int(np.asarray(data["bake_format_version"]).reshape(-1)[0]) != _BAKE_FORMAT_VERSION:
                return False
            return (
                data["joint_pos"].ndim == 2
                and data["joint_vel"].ndim == 2
                and data["joint_pos"].shape[1] == num_dofs
                and data["joint_vel"].shape[1] == num_dofs
            )
    except Exception:
        return False


def bake_motion(npz_in: Path, npz_out: Path, *, anchor_body_name: str = "pelvis") -> None:
    """Read source motion, compute VR FK + lower-body tables, write a MERGED NPZ.

    The output NPZ contains all source keys except that ``joint_pos`` and
    ``joint_vel`` are replaced with 29-DOF browser-compatible arrays.  Original
    root-prefixed arrays are kept as ``source_joint_pos`` / ``source_joint_vel``.
    This way ``policy.add_motion(source=npz_out)`` (mjswan) bundles one file
    per motion that has everything the policy + ``WbtTrackingCommand`` need —
    no sibling asset, no custom mjswan internals.

    Mirrors the in-policy FK at
    ``wbt_wrappers_inference/policies/holosoma_wbt.py:541-584`` so the browser
    can replay the exact same VR commands the runtime policy would consume.

    Parameters
    ----------
    npz_in
        Source reference-motion NPZ.
    npz_out
        Destination merged NPZ path.  Must NOT point at the source dataset —
        callers should pick a build-time output dir.
    anchor_body_name
        Name of the body to use as the anchor (root) frame.  Looked up against
        ``data["body_names"]`` when present; falls back to body index 0 if the
        name isn't found or ``body_names`` is missing.
    """
    import pinocchio as pin
    from holosoma_inference.utils.math.quat import (
        matrix_from_quat,
        quat_inverse,
        quat_mul,
        quat_rotate_inverse,
        wxyz_to_xyzw,
        xyzw_to_wxyz,
    )
    from wbt_wrappers_inference.config_values.robot_config import (
        LOWER_BODY_INDICES,
        VR_2POINT_BODIES,
        VR_2POINT_OFFSETS,
        VR_3POINT_BODIES,
        VR_3POINT_OFFSETS,
        VR_3POINT_OFFSETS_TCP,
        VR_5POINT_BODIES,
        VR_5POINT_FEET_BODIES,
        VR_5POINT_OFFSETS_TCP,
    )

    npz_in = Path(npz_in)
    npz_out = Path(npz_out)
    if not npz_in.is_file():
        raise FileNotFoundError(f"Reference motion NPZ not found: {npz_in}")
    if not _DEFAULT_URDF.is_file():
        raise FileNotFoundError(f"URDF not found at {_DEFAULT_URDF}")

    pin_robot = _load_pinocchio_robot()
    model = pin_robot.robot_model
    pin_data = pin_robot.robot_data

    def _resolve_frame_ids(body_names: list[str]) -> list[int]:
        ids: list[int] = []
        for body_name in body_names:
            fid = model.getFrameId(body_name)
            if fid >= model.nframes:
                raise ValueError(
                    f"VR body '{body_name}' not found in URDF at {_DEFAULT_URDF}; FK output would be invalid"
                )
            ids.append(fid)
        return ids

    frame_ids_2pt = _resolve_frame_ids(VR_2POINT_BODIES)
    frame_ids_3pt = _resolve_frame_ids(VR_3POINT_BODIES)
    # 5-point = the 3pt set + both FEET. Only the FEET get their own FK pass: the 5pt
    # tables are built by CONCATENATING the foot columns onto the 3pt-TCP / 3pt-quat
    # tables, which makes "slots [0:9]/[0:12] are the 3pt vectors" true by construction
    # instead of by a comment (and costs 2 extra frames of FK, not 5).
    frame_ids_5pt_feet = _resolve_frame_ids(list(VR_5POINT_FEET_BODIES))

    offsets_2pt = np.asarray(VR_2POINT_OFFSETS, dtype=np.float32)  # (2, 3)
    offsets_3pt = np.asarray(VR_3POINT_OFFSETS, dtype=np.float32)  # (3, 3)
    # TCP-unified 3pt offsets: both HAND points at the 0.242 m fingertip TCP (the force
    # anchor), head unchanged — exp:g1-wbt-3pt-hybrid-unified[-anneal], user 2026-08-03.
    # Same bodies + same FK as offsets_3pt, so only the offset vector differs.
    offsets_3pt_tcp = np.asarray(VR_3POINT_OFFSETS_TCP, dtype=np.float32)  # (3, 3)
    # Foot offsets for the 5-point set: the TAIL of VR_5POINT_OFFSETS_TCP, i.e. the two
    # bare (0,0,0) origins. Sliced from the shared constant rather than re-typed as zeros
    # so that if a sole offset is ever introduced there, the bake follows automatically.
    offsets_5pt_feet = np.asarray(VR_5POINT_OFFSETS_TCP[len(VR_3POINT_BODIES) :], dtype=np.float32)  # (2, 3)
    # Rubber-hand (force contact point) FK offsets in each wrist_yaw_link LOCAL
    # frame, [left, right]. These DIFFER from VR_2POINT_OFFSETS (0.18, ±0.025):
    # the force metric uses the physical contact point, not the VR tracker. Must
    # mirror eval_utils.FORCE_HAND_OFFSETS_B byte-for-byte (left y=+0.003, right
    # y=-0.003) — the per-hand y sign even FLIPS vs the VR offsets, so reusing
    # offsets_2pt here would silently produce a plausible-but-wrong drift number.
    force_offsets = np.array(
        [[0.0815, +0.003, 0.0], [0.0815, -0.003, 0.0]], dtype=np.float32
    )  # (2,3) L,R  yam: wrist→ft(0.072)→hand(0.0095)
    lower_idx = np.asarray(LOWER_BODY_INDICES, dtype=np.int64)  # (12,)

    num_dofs = int(getattr(getattr(pin_robot, "robot_model", None), "nq", 0))
    # Prefer the static config's 29 dof count; the urdf model's nq has the
    # extra 7 floating-base coords so subtract.
    dof_count_from_model = max(num_dofs - 7, 0)
    if dof_count_from_model:
        num_dofs = dof_count_from_model
    else:
        num_dofs = 29

    src = np.load(npz_in)
    required = {"joint_pos"}
    missing = required - set(src.files)
    if missing:
        src.close()
        raise ValueError(f"Motion NPZ {npz_in} missing required keys: {sorted(missing)}")
    joint_pos_arr = np.asarray(src["joint_pos"])
    if joint_pos_arr.ndim != 2 or joint_pos_arr.shape[1] < 7 + num_dofs:
        src.close()
        raise ValueError(
            f"joint_pos has shape {joint_pos_arr.shape}; expected (T, >={7 + num_dofs}) for {num_dofs}-DOF robot"
        )
    data = {k: src[k] for k in src.files}
    src.close()

    dof_pos, dof_vel = _resolve_dof_pos_vel(data, num_dofs)
    root_pos, root_quat_wxyz = _resolve_root_pose(data, anchor_body_name)

    T = dof_pos.shape[0]
    if root_pos.shape[0] != T:
        raise ValueError(f"Frame count mismatch: dof_pos has {T} frames but root_pos has {root_pos.shape[0]}")

    real2pin = pin_robot.real2pinocchio_index

    # 2pt orientation is the 12D continuous 6D rotation rep (2 wrists x 6); 3pt
    # stays the 12D raw xyzw quat (3 bodies x 4). The 6D rep matches the
    # retrained exp:g1-wbt-2pt* obs and avoids the antipodal quaternion bug.
    vr2_pos_local = np.zeros((T, len(VR_2POINT_BODIES) * 3), dtype=np.float32)
    vr2_orn_local = np.zeros((T, len(VR_2POINT_BODIES) * 6), dtype=np.float32)
    vr3_pos_local = np.zeros((T, len(VR_3POINT_BODIES) * 3), dtype=np.float32)
    vr3_orn_local = np.zeros((T, len(VR_3POINT_BODIES) * 4), dtype=np.float32)
    # 3pt orientation as the 6D rotation rep (3 bodies x 6 = 18D), for the 6D-orn 3pt
    # preset (exp:g1-wbt-3pt-hybrid-global-vel). Baked ALONGSIDE the quat vr3_orn_local
    # (which stays for exp:g1-wbt-3pt-hybrid); the browser picks the table by obs term
    # name (vr_3point_orn_6d vs vr_3point_orn). Same _fk_into helper with orn_is_6d=True.
    vr3_orn6d_local = np.zeros((T, len(VR_3POINT_BODIES) * 6), dtype=np.float32)
    _vr3_pos_scratch = np.zeros((T, len(VR_3POINT_BODIES) * 3), dtype=np.float32)  # discarded (== vr3_pos_local)
    # TCP-unified 3pt POSITION table (hands at 0.242, head unchanged) for
    # exp:g1-wbt-3pt-hybrid-unified[-anneal]. Baked ALONGSIDE vr3_pos_local (which stays
    # for every 0.18 m preset); the browser picks the table by obs term name
    # (vr_3point_pos_tcp vs vr_3point_pos). The orn buffer is scratch — the orientation
    # is offset-independent (it is the wrist LINK frame), so vr3_orn_local is reused.
    vr3_pos_tcp_local = np.zeros((T, len(VR_3POINT_BODIES) * 3), dtype=np.float32)
    _vr3_orn_tcp_scratch = np.zeros((T, len(VR_3POINT_BODIES) * 4), dtype=np.float32)  # discarded
    # 5-point FOOT columns only (exp:g1-wbt-5pt-hybrid-unified-anneal). The full 15D/20D
    # tables are hstacked from the 3pt tables + these below, so the 3pt prefix is exact.
    _vr5_feet_pos = np.zeros((T, len(VR_5POINT_FEET_BODIES) * 3), dtype=np.float32)
    _vr5_feet_orn = np.zeros((T, len(VR_5POINT_FEET_BODIES) * 4), dtype=np.float32)
    # Reference rubber-hand position in the reference PELVIS frame, [L_xyz, R_xyz]
    # (T, 6), using the FORCE offsets above. Drives the browser's live per-hand
    # force-tracking error (raw drift + Eq.2 residual) — see eval_utils._log_force_frame
    # ref_hand_rel. The scratch orn buffer is discarded (we only need positions).
    force_hand_ref_local = np.zeros((T, len(VR_2POINT_BODIES) * 3), dtype=np.float32)
    force_orn_scratch = np.zeros((T, len(VR_2POINT_BODIES) * 6), dtype=np.float32)

    def _fk_into(
        t: int,
        anchor_pos: "NDArray[np.float32]",
        anchor_quat_wxyz: "NDArray[np.float32]",
        body_frame_ids: list[int],
        body_offsets: "NDArray[np.float32]",
        out_pos: "NDArray[np.float32]",
        out_orn: "NDArray[np.float32]",
        orn_is_6d: bool,
    ) -> None:
        orn_stride = 6 if orn_is_6d else 4
        for k, fid in enumerate(body_frame_ids):
            frame_pose = pin_data.oMf[fid]
            body_pos_w = np.asarray(frame_pose.translation, dtype=np.float32)
            body_quat_xyzw = np.asarray(pin.Quaternion(frame_pose.rotation).coeffs(), dtype=np.float32)
            body_quat_wxyz = xyzw_to_wxyz(body_quat_xyzw.reshape(1, 4))

            offset = body_offsets[k].reshape(1, 3)
            offset_world = quat_rotate_inverse(quat_inverse(body_quat_wxyz), offset).reshape(3)
            vr_pos_w = body_pos_w + offset_world

            diff = (vr_pos_w - anchor_pos).reshape(1, 3)
            local_pos = quat_rotate_inverse(anchor_quat_wxyz, diff).reshape(3).astype(np.float32)

            local_quat_wxyz = quat_mul(quat_inverse(anchor_quat_wxyz), body_quat_wxyz)
            if orn_is_6d:
                # 6D rot rep from the WXYZ quat (matrix_from_quat expects wxyz),
                # first two columns flattened C-order [r00,r01,r10,r11,r20,r21].
                # Bit-for-bit identical to training's vr_2point_local_orn_target
                # and to runtime _compute_vr_fk's 2pt branch.
                local_orn = matrix_from_quat(local_quat_wxyz)[..., :2].reshape(-1).astype(np.float32)
            else:
                local_orn = wxyz_to_xyzw(local_quat_wxyz).reshape(4).astype(np.float32)
                # CANONICALIZE THE DOUBLE COVER TO w >= 0. `q` and `-q` are the same
                # rotation, but a network sees raw numbers, and the quaternion-orn presets
                # (`vr_3point_orn`, `vr_5point_orn`) were trained on data whose local quat
                # is ALWAYS on the +w branch — the checkpoints' own normalizer says so
                # unambiguously (na475b60 `vr_5point_orn` w: mean +0.73 hands / +0.98 head
                # / +0.96 feet, std 0.17/0.09/0.11, i.e. unimodal and nowhere near 0).
                #
                # Nothing upstream guarantees that branch here. Both inputs to
                # `conj(anchor) * body` come from representatives chosen independently:
                # the anchor is whatever the motion NPZ stored, the body is Eigen's
                # `pin.Quaternion(R).coeffs()`. On a take whose pelvis sits near the 180°
                # seam (|w| ~ 0.1 — exactly what UMI's world_align Rz180 produces) the
                # product lands on -identity, w ~ -0.99, for EVERY frame and EVERY body.
                # Measured on the two motion-completion takes: 100% of frames flipped,
                # max |z| vs the training normalizer 20.8 -> 1.58 once canonicalized.
                #
                # This is the same double-cover trap the `-global-vel` family escaped by
                # switching to the 6D rep (see observation.py `_vr_local_orn_target`'s note
                # on "the observed left-wrist blow-up"). The unified family still ships the
                # raw quaternion, so it has to be canonicalized at every producer instead.
                #
                # IDEMPOTENT AND A NO-OP ON ALREADY-CORRECT DATA: a table that was already
                # +w (e.g. the builder's `idle_default`, 0% flips) is unchanged bit-for-bit,
                # which is why this cannot regress any preset that was working.
                if local_orn[3] < 0.0:
                    local_orn = -local_orn

            out_pos[t, k * 3 : (k + 1) * 3] = local_pos
            out_orn[t, k * orn_stride : (k + 1) * orn_stride] = local_orn

    for t in range(T):
        wxyz = root_quat_wxyz[t]
        ref_quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
        ref_pos = root_pos[t].astype(np.float32)

        joint_pos_pin = dof_pos[t][real2pin]
        q = np.concatenate([ref_pos, ref_quat_xyzw, joint_pos_pin]).astype(np.float64)
        pin.framesForwardKinematics(model, pin_data, q)

        anchor_quat_wxyz = wxyz.reshape(1, 4).astype(np.float32)

        _fk_into(t, ref_pos, anchor_quat_wxyz, frame_ids_2pt, offsets_2pt, vr2_pos_local, vr2_orn_local, orn_is_6d=True)
        _fk_into(
            t, ref_pos, anchor_quat_wxyz, frame_ids_3pt, offsets_3pt, vr3_pos_local, vr3_orn_local, orn_is_6d=False
        )
        # 6D-orn 3pt table: same 3pt frames/offsets, orn as rot6d (pos buffer is a scratch
        # dup of vr3_pos_local — identical positions, discarded). 18D orn out.
        _fk_into(
            t, ref_pos, anchor_quat_wxyz, frame_ids_3pt, offsets_3pt, _vr3_pos_scratch, vr3_orn6d_local, orn_is_6d=True
        )
        # TCP-unified 3pt positions: SAME frames, offsets_3pt_tcp instead of offsets_3pt
        # (hands 0.18 -> 0.242, head unchanged). orn out is scratch.
        _fk_into(
            t,
            ref_pos,
            anchor_quat_wxyz,
            frame_ids_3pt,
            offsets_3pt_tcp,
            vr3_pos_tcp_local,
            _vr3_orn_tcp_scratch,
            orn_is_6d=False,
        )
        # 5-point FOOT columns: the two `*_ankle_roll_link` frames at their bare origins,
        # same anchor + same math as every other table. Positions AND orientations are both
        # kept here (unlike the TCP pass, whose orn is a scratch dup) — the 5pt actor obs
        # consumes both.
        _fk_into(
            t,
            ref_pos,
            anchor_quat_wxyz,
            frame_ids_5pt_feet,
            offsets_5pt_feet,
            _vr5_feet_pos,
            _vr5_feet_orn,
            orn_is_6d=False,
        )
        # Rubber-hand reference position in the reference PELVIS frame, FORCE offset
        # (0.0815, ±0.003). Same wrist_yaw frames as 2pt VR, different offset.
        # orn output is scratch (force metric only needs position).
        _fk_into(
            t,
            ref_pos,
            anchor_quat_wxyz,
            frame_ids_2pt,
            force_offsets,
            force_hand_ref_local,
            force_orn_scratch,
            orn_is_6d=True,
        )

    # Fancy column-indexing (`arr[:, [list]]`) on a C-contiguous (T, 29) source
    # returns an F-contiguous (T, 12) view; np.savez then writes
    # `fortran_order=True` in the .npy header. mjswan's browser-side parser
    # rejects fortran-ordered arrays (silently, via try/catch in loadNpz),
    # leaving the key missing at WbtTrackingCommand.setSelectedMotion and
    # surfacing as: `Motion ... missing baked key 'lower_body_jp'`. Force
    # C-contiguous before saving so the .npy header stays C-order.
    lower_body_jp = np.ascontiguousarray(dof_pos[:, lower_idx], dtype=np.float32)
    lower_body_jv = np.ascontiguousarray(dof_vel[:, lower_idx], dtype=np.float32)

    # World-frame anchor pose (already extracted as root_pos / root_quat_wxyz);
    # for motion_ref_pos_b we just need the position. Velocity uses the source
    # joint_vel root prefix when present, else finite-diff at the source NPZ's
    # native fps (read from the `fps` key when present; 50 Hz is the
    # holosoma-default rl_rate so it's the right fallback when fps is absent).
    ref_root_pos_w = np.ascontiguousarray(root_pos, dtype=np.float32)
    if "joint_vel" in data:
        joint_vel_arr = np.asarray(data["joint_vel"])
        if joint_vel_arr.ndim == 2 and joint_vel_arr.shape[1] >= 3 + num_dofs:
            ref_root_vel_w = np.ascontiguousarray(joint_vel_arr[:, :3], dtype=np.float32)
        else:
            ref_root_vel_w = _finite_diff_vel(ref_root_pos_w, fps=_motion_fps(data))
    else:
        ref_root_vel_w = _finite_diff_vel(ref_root_pos_w, fps=_motion_fps(data))

    # Merge: keep the source metadata/body trajectories intact, but convert
    # joint_pos/joint_vel to the root-free 29-DOF layout that mjswan expects.
    # Holosoma NPZs store joint_pos as [root_xyz(3), root_quat(4), dof(29)] and
    # joint_vel as [root_lin(3), root_ang(3), dof(29)]. TrackingCommand writes
    # joint_pos[i] directly to the i-th named hinge; leaving the root prefix in
    # place makes both the live robot and the reference ghost use nonsense
    # joint angles. Preserve the original arrays under source_* for debugging.
    merged: dict[str, np.ndarray] = {key: np.asarray(data[key]) for key in data}
    merged["source_joint_pos"] = np.asarray(data["joint_pos"])
    if "joint_vel" in data:
        merged["source_joint_vel"] = np.asarray(data["joint_vel"])
    merged["joint_pos"] = np.ascontiguousarray(dof_pos, dtype=np.float32)
    merged["joint_vel"] = np.ascontiguousarray(dof_vel, dtype=np.float32)
    merged["vr_pos_local"] = vr2_pos_local
    merged["vr_orn_local"] = vr2_orn_local
    merged["vr3_pos_local"] = vr3_pos_local
    merged["vr3_pos_tcp_local"] = vr3_pos_tcp_local
    merged["vr3_orn_local"] = vr3_orn_local
    merged["vr3_orn6d_local"] = vr3_orn6d_local
    # 5-point tables = the 3pt-TCP position / 3pt quaternion tables with the two foot
    # blocks appended, matching VR_5POINT_BODIES = [*VR_3POINT_BODIES, *feet]. hstack of
    # two C-contiguous (T, k) float32 arrays is C-contiguous, but pin it anyway: a
    # fortran-ordered .npy header is silently dropped by mjswan's browser parser (see the
    # lower_body_jp note above), which for these keys would zero-fill all five points.
    merged["vr5_pos_tcp_local"] = np.ascontiguousarray(np.hstack([vr3_pos_tcp_local, _vr5_feet_pos]), dtype=np.float32)
    merged["vr5_orn_local"] = np.ascontiguousarray(np.hstack([vr3_orn_local, _vr5_feet_orn]), dtype=np.float32)
    # Cheap structural pin: the widths must equal 5 bodies x (3, 4). A drift here (e.g. a
    # 6th tracked point added upstream without touching the TS width) would otherwise
    # surface as a browser-side "width != 15" throw at motion-select time.
    assert merged["vr5_pos_tcp_local"].shape[1] == len(VR_5POINT_BODIES) * 3
    assert merged["vr5_orn_local"].shape[1] == len(VR_5POINT_BODIES) * 4
    merged["lower_body_jp"] = lower_body_jp
    merged["lower_body_jv"] = lower_body_jv
    merged["ref_root_pos_w"] = ref_root_pos_w
    merged["ref_root_vel_w"] = ref_root_vel_w
    # Semantic version of the table CONTENTS — see _BAKE_FORMAT_VERSION. This is what lets
    # a values-only fix invalidate an mtime-current cache.
    merged["bake_format_version"] = np.asarray(_BAKE_FORMAT_VERSION, dtype=np.int32)
    # Reference rubber-hand position, pelvis-frame, [L_xyz, R_xyz] (T, 6). Force
    # offset (0.0815, ±0.003). ascontiguousarray to dodge the fortran-order
    # .npy-header trap documented above (line ~360).
    merged["force_hand_ref_local"] = np.ascontiguousarray(force_hand_ref_local, dtype=np.float32)

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_out, **merged)


def _finite_diff_vel(positions: "NDArray[np.float32]", fps: float) -> "NDArray[np.float32]":
    """Forward finite-diff at ``fps`` Hz with the last frame replicated. Fallback
    when the source NPZ doesn't carry a root linear velocity prefix."""
    if positions.shape[0] < 2:
        return np.zeros_like(positions)
    diff = np.diff(positions, axis=0) * fps
    last = diff[-1:].copy()
    return np.ascontiguousarray(np.concatenate([diff, last], axis=0), dtype=np.float32)


def _motion_fps(data: dict[str, np.ndarray]) -> float:
    """Read the per-motion fps from the NPZ when present, else fall back to
    50 Hz (the holosoma-default ``rl_rate`` and the rate every shipped motion
    in ``wbt_training/motions/`` is recorded at)."""
    raw = data.get("fps")
    if raw is None:
        return 50.0
    arr = np.asarray(raw)
    if arr.size == 0:
        return 50.0
    val = float(arr.flat[0])
    return val if val > 0.0 else 50.0


def bake_motion_dir(in_dir: Path, out_dir: Path, *, anchor_body_name: str = "pelvis") -> list[Path]:
    """Bake every ``*.npz`` in ``in_dir`` into ``<stem>.npz`` in ``out_dir``.

    The output uses the SAME stem as the source — no ``.baked.`` suffix — so
    mjswan's ``policy.add_motion(source=...)`` can bundle the merged NPZ
    directly.  Skips a re-bake when the output already exists and is strictly
    newer than the input.  Returns the sorted list of output paths (whether
    freshly baked or skipped).

    Refuses to run when ``out_dir`` resolves to the same directory as
    ``in_dir`` — overwriting source motions with merged copies would
    silently mutate the training dataset.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {in_dir}")
    if out_dir.resolve() == in_dir.resolve():
        raise ValueError(f"out_dir must differ from in_dir to avoid clobbering source motions: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for src in sorted(in_dir.glob("*.npz")):
        dst = out_dir / src.name  # SAME stem now — no ".baked." suffix.
        if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime and _is_current_browser_bake(dst):
            outputs.append(dst)
            continue
        bake_motion(src, dst, anchor_body_name=anchor_body_name)
        outputs.append(dst)
    return sorted(outputs)


def _main() -> None:  # pragma: no cover - simple CLI shim
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="reference motion NPZ or directory")
    parser.add_argument("output", type=Path, help="output baked NPZ or directory")
    parser.add_argument(
        "--anchor-body-name", default="pelvis", help="anchor body name to look up in body_names (default: pelvis)"
    )
    args = parser.parse_args()

    if args.input.is_dir():
        outs = bake_motion_dir(args.input, args.output, anchor_body_name=args.anchor_body_name)
        print(f"Baked {len(outs)} motions -> {args.output}")
    else:
        bake_motion(args.input, args.output, anchor_body_name=args.anchor_body_name)
        print(f"Baked 1 motion -> {args.output}")


if __name__ == "__main__":  # pragma: no cover
    _main()
