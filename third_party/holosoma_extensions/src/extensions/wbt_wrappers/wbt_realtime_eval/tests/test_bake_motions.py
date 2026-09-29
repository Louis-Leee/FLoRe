"""Numerical smoke tests for bake_motions."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
# _HERE = tests/  ; parents[0] = wbt_realtime_eval/  ; parents[1] = wbt_wrappers/
_MOTIONS_DIR = _HERE.parents[1] / "wbt_training" / "motions"
_FIRST_MOTION = _MOTIONS_DIR / "idle_stand_1.npz"


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_idle_stand_shapes(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "idle_stand_1.npz"
    bake_motion(_FIRST_MOTION, out)

    with np.load(out) as d:
        assert set(d.files) >= {
            "vr_pos_local",
            "vr_orn_local",
            "lower_body_jp",
            "lower_body_jv",
            "force_hand_ref_local",
        }
        T = d["vr_pos_local"].shape[0]
        assert d["vr_pos_local"].shape == (T, 6)
        # vr_orn_local is the 12D continuous 6D rotation rep (2 wrists x 6 =
        # first two columns of the rotation matrix), NOT the old 8D xyzw quat.
        assert d["vr_orn_local"].shape == (T, 12)
        assert d["lower_body_jp"].shape == (T, 12)
        assert d["lower_body_jv"].shape == (T, 12)
        # force_hand_ref_local: rubber-hand ref pos [L_xyz, R_xyz], pelvis frame.
        assert d["force_hand_ref_local"].shape == (T, 6)
        # Each 6D rot rep is the first two columns of a rotation matrix,
        # C-order flattened from (3,2): [r00, r01, r10, r11, r20, r21]. So
        # column 0 = strided indices [0,2,4], column 1 = [1,3,5]; both unit
        # length and mutually orthogonal.
        for k in range(2):
            block = d["vr_orn_local"][:, k * 6 : (k + 1) * 6]
            col0 = block[:, [0, 2, 4]]
            col1 = block[:, [1, 3, 5]]
            np.testing.assert_allclose(np.linalg.norm(col0, axis=1), 1.0, atol=1e-3)
            np.testing.assert_allclose(np.linalg.norm(col1, axis=1), 1.0, atol=1e-3)
            # Columns of a rotation matrix are orthonormal.
            np.testing.assert_allclose(np.sum(col0 * col1, axis=1), 0.0, atol=1e-3)

    # Critical for browser parsing: every baked array must be C-contiguous
    # (NOT fortran_order=True in the .npy header). mjswan's loadNpz throws
    # "Fortran-ordered arrays not supported" for the latter, silently dropping
    # the key — surfaces to the user as `missing baked key 'lower_body_jp'`.
    # Re-read the raw .npy from inside the zip and assert the header.
    import zipfile

    with zipfile.ZipFile(out) as z:
        for key in (
            "joint_pos",
            "joint_vel",
            "vr_pos_local",
            "vr_orn_local",
            "lower_body_jp",
            "lower_body_jv",
            "force_hand_ref_local",
        ):
            with z.open(f"{key}.npy") as f:
                raw = f.read()
            major = raw[6]
            if major == 1:
                hlen = raw[8] | (raw[9] << 8)
                hstart = 10
            else:
                hlen = raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24)
                hstart = 12
            header = raw[hstart : hstart + hlen].decode()
            assert "'fortran_order': False" in header, (
                f"Baked key {key!r} stored with fortran_order=True. Header: {header!r}"
            )


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_vr3_orn6d_is_18d_and_consistent_with_quaternion(tmp_path):
    """The 6D 3pt orientation table (vr3_orn6d_local, for exp:g1-wbt-3pt-hybrid-global-vel)
    is (T, 18) = 3 bodies x 6D rot rep, with orthonormal columns, and encodes the SAME
    rotation as the existing quaternion vr3_orn_local (T, 12) — the quat table is untouched."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "idle_stand_1.npz"
    bake_motion(_FIRST_MOTION, out)

    with np.load(out) as d:
        assert "vr3_orn6d_local" in d.files
        T = d["vr3_pos_local"].shape[0]
        orn6d = d["vr3_orn6d_local"]
        quat = d["vr3_orn_local"]
        assert orn6d.shape == (T, 18)  # 3 bodies x 6
        assert quat.shape == (T, 12)  # 3 bodies x 4 (xyzw) — UNCHANGED
        # Per-body 6D block: first two columns of R, C-order [r00,r01,r10,r11,r20,r21].
        # col0 = [0,2,4], col1 = [1,3,5]; both unit-norm and mutually orthogonal.
        for k in range(3):
            block = orn6d[:, k * 6 : (k + 1) * 6]
            col0 = block[:, [0, 2, 4]]
            col1 = block[:, [1, 3, 5]]
            np.testing.assert_allclose(np.linalg.norm(col0, axis=1), 1.0, atol=1e-3)
            np.testing.assert_allclose(np.linalg.norm(col1, axis=1), 1.0, atol=1e-3)
            np.testing.assert_allclose(np.sum(col0 * col1, axis=1), 0.0, atol=1e-3)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_preserves_source_metadata_and_raw_joint_arrays(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "idle_stand_1.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(_FIRST_MOTION) as src, np.load(out) as dst:
        for key in src.files:
            assert key in dst.files, f"merged NPZ dropped {key}"
            if key not in {"joint_pos", "joint_vel"}:
                np.testing.assert_array_equal(np.asarray(src[key]), np.asarray(dst[key]))

        np.testing.assert_array_equal(np.asarray(src["joint_pos"]), np.asarray(dst["source_joint_pos"]))
        np.testing.assert_array_equal(np.asarray(src["joint_vel"]), np.asarray(dst["source_joint_vel"]))
        for key in ("vr_pos_local", "vr_orn_local", "lower_body_jp", "lower_body_jv"):
            assert key in dst.files


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_strips_root_prefix_for_mjswan_tracking_command(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "idle_stand_1.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(_FIRST_MOTION) as src, np.load(out) as dst:
        assert dst["joint_pos"].shape == (src["joint_pos"].shape[0], 29)
        assert dst["joint_vel"].shape == (src["joint_vel"].shape[0], 29)
        np.testing.assert_allclose(dst["joint_pos"], src["joint_pos"][:, 7:36], atol=1e-6)
        np.testing.assert_allclose(dst["joint_vel"], src["joint_vel"][:, 6:35], atol=1e-6)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_motion_dir_skips_when_output_newer(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion_dir

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    # Copy a single motion as fixture
    src = _FIRST_MOTION
    dst = in_dir / src.name
    dst.write_bytes(src.read_bytes())

    outs1 = bake_motion_dir(in_dir, out_dir)
    assert len(outs1) == 1
    assert outs1[0].exists()
    # Output uses the same stem as the source — no ".baked." suffix.
    assert outs1[0].name == src.name
    mtime_first = outs1[0].stat().st_mtime

    # Second invocation should skip (output newer than input)
    outs2 = bake_motion_dir(in_dir, out_dir)
    assert outs2 == outs1
    assert outs2[0].stat().st_mtime == mtime_first


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_motion_dir_rebakes_stale_browser_schema(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion_dir

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    src = _FIRST_MOTION
    input_motion = in_dir / src.name
    stale_output = out_dir / src.name
    input_motion.write_bytes(src.read_bytes())
    stale_output.write_bytes(src.read_bytes())  # root-prefixed old browser schema
    newer = input_motion.stat().st_mtime + 10.0
    os.utime(stale_output, (newer, newer))

    outs = bake_motion_dir(in_dir, out_dir)
    assert outs == [stale_output]
    with np.load(stale_output) as d:
        assert d["joint_pos"].shape[1] == 29
        assert d["joint_vel"].shape[1] == 29
        assert "source_joint_pos" in d.files


def test_bake_motion_dir_refuses_inplace(tmp_path):
    from wbt_realtime_eval.bake_motions import bake_motion_dir

    with pytest.raises(ValueError, match="must differ"):
        bake_motion_dir(tmp_path, tmp_path)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_motion_dir_rebakes_when_only_tcp_table_is_missing(tmp_path):
    """A bake that is otherwise CURRENT but lacks ``vr3_pos_tcp_local`` must be re-run.

    This is the silent-6.2 cm path: `getVr3PosTcpLocal()` FALLS BACK to the 0.18 m
    `vr3PosLocal` table when the TCP key is absent, so a pre-2026-08-03 bake reused against
    a TCP-unified checkpoint would feed the policy hand targets 6.2 cm from its training
    geometry — with the right widths, no error, and nothing visible in the UI. The only
    thing standing between that and a wrong bundle is this key being in
    `_is_current_browser_bake`'s REQUIRED set, so pin it directly by deleting JUST that key
    from an otherwise-valid bake."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import _is_current_browser_bake, bake_motion, bake_motion_dir

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    input_motion = in_dir / _FIRST_MOTION.name
    input_motion.write_bytes(_FIRST_MOTION.read_bytes())

    # A genuinely current bake is accepted...
    good = out_dir / _FIRST_MOTION.name
    bake_motion(input_motion, good)
    assert _is_current_browser_bake(good), "a freshly baked NPZ must count as current"

    # ...but strip ONLY vr3_pos_tcp_local and it must be rejected as stale.
    with np.load(good) as d:
        assert "vr3_pos_tcp_local" in d.files
        pruned = {k: d[k] for k in d.files if k != "vr3_pos_tcp_local"}
    np.savez(good, **pruned)
    newer = input_motion.stat().st_mtime + 10.0
    os.utime(good, (newer, newer))
    assert not _is_current_browser_bake(good), (
        "a bake lacking vr3_pos_tcp_local must NOT count as current — otherwise the mtime "
        "check alone would let it be silently reused, and the TS accessor would quietly "
        "serve the 0.18 m table to a 0.242 m policy."
    )

    # bake_motion_dir must therefore actually re-bake it (mtime is newer, so ONLY the
    # schema check can trigger this) and restore the key.
    outs = bake_motion_dir(in_dir, out_dir)
    assert outs == [good]
    with np.load(good) as d:
        assert "vr3_pos_tcp_local" in d.files, "re-bake must restore the TCP table"
        assert d["vr3_pos_tcp_local"].shape[1] == 9


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_vr3_tcp_table_uses_the_tcp_offsets(tmp_path):
    """The baked ``vr3_pos_tcp_local`` must be the 3pt FK evaluated at the TCP offsets, and
    must DIFFER from the 0.18 m table by exactly the hand lever change.

    Guards the geometry itself rather than just the key's presence: a bake that computed the
    TCP table with the wrong offsets (or aliased it to the 0.18 m one) would still have the
    right key, right width, and pass every other test here."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_OFFSETS, VR_3POINT_OFFSETS_TCP

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "baked.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(out) as d:
        tcp = d["vr3_pos_tcp_local"]
        base = d["vr3_pos_local"]
        orn = d["vr3_orn_local"]
    assert tcp.shape == base.shape

    # HEAD point (index 2) is unchanged by the TCP swap, so those 3 columns must be EQUAL.
    np.testing.assert_allclose(tcp[:, 6:9], base[:, 6:9], atol=1e-6)
    # The two HAND points MUST differ — an aliased/miscomputed table would be identical.
    assert not np.allclose(tcp[:, 0:6], base[:, 0:6], atol=1e-4), (
        "the TCP hand points must differ from the 0.18 m ones; identical columns mean the "
        "bake aliased vr3_pos_local instead of re-running FK at the TCP offsets."
    )

    # Quantitative: the per-hand displacement equals R(wrist) @ (tcp_offset - vr_offset),
    # so its NORM equals |tcp_offset - vr_offset| at every frame (rotation is isometric).
    # vr3_orn_local is xyzw per body; the offsets are in the wrist LOCAL frame.
    for hand in (0, 1):
        delta_local = np.asarray(VR_3POINT_OFFSETS_TCP[hand], dtype=np.float64) - np.asarray(
            VR_3POINT_OFFSETS[hand], dtype=np.float64
        )
        expected_norm = float(np.linalg.norm(delta_local))
        got = np.linalg.norm(tcp[:, 3 * hand : 3 * hand + 3] - base[:, 3 * hand : 3 * hand + 3], axis=1)
        np.testing.assert_allclose(got, expected_norm, atol=2e-5, err_msg=f"hand {hand} lever change norm")
        # Sanity-check the magnitude actually is the documented ~6.7 cm 3-D lever change
        # (0.062 along x plus dropping the 0.025 lateral term).
        assert 0.05 < expected_norm < 0.08, expected_norm

    # And the DIRECTION must follow the live wrist orientation, not a fixed world axis:
    # rotate delta_local by each frame's wrist quaternion and compare to the observed shift.
    from holosoma_inference.utils.math.quat import matrix_from_quat

    for hand in (0, 1):
        delta_local = np.asarray(VR_3POINT_OFFSETS_TCP[hand], dtype=np.float32) - np.asarray(
            VR_3POINT_OFFSETS[hand], dtype=np.float32
        )
        q_xyzw = orn[:, 4 * hand : 4 * hand + 4]
        q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
        rot = np.asarray(matrix_from_quat(_as_tensor(q_wxyz)))  # (T, 3, 3)
        expected = rot @ delta_local
        observed = tcp[:, 3 * hand : 3 * hand + 3] - base[:, 3 * hand : 3 * hand + 3]
        np.testing.assert_allclose(observed, expected, atol=2e-5, err_msg=f"hand {hand} lever direction")


def _as_tensor(arr):
    """numpy -> torch, for the holosoma_inference quat helpers (which are torch-based)."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_matches_runtime_compute_vr_fk(tmp_path):
    """Verify bake output matches an in-test FK helper that mirrors the runtime
    ``HolosomaWBTPolicy._compute_vr_fk`` math (holosoma_wbt.py:541-584).

    Rather than instantiating a full ``HolosomaWBTPolicy`` (which requires
    hardware init, ONNX checkpoint, etc.), we copy the FK math into a small
    helper here and assert numerical equivalence on frame 0.
    """
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    import pinocchio as pin
    from holosoma_inference.utils.math.quat import (
        matrix_from_quat,
        quat_inverse,
        quat_mul,
        quat_rotate_inverse,
        wxyz_to_xyzw,
        xyzw_to_wxyz,
    )
    from wbt_wrappers_inference.config_values.robot_config import VR_2POINT_BODIES, VR_2POINT_OFFSETS

    from wbt_realtime_eval.bake_motions import _load_pinocchio_robot, bake_motion

    # Run the bake.
    out = tmp_path / "idle_stand_1.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(out) as d:
        bake_pos = d["vr_pos_local"][0].copy()
        bake_orn = d["vr_orn_local"][0].copy()

    # Reproduce the runtime FK by hand for frame 0.
    with np.load(_FIRST_MOTION) as src:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in src["body_names"].tolist()]
        body_idx = names.index("pelvis")
        ref_pos = np.asarray(src["body_pos_w"][0, body_idx], dtype=np.float32)
        ref_quat_wxyz = np.asarray(src["body_quat_w"][0, body_idx], dtype=np.float32)
        joint_pos_full = np.asarray(src["joint_pos"][0], dtype=np.float32)

    pin_robot = _load_pinocchio_robot()
    model = pin_robot.robot_model
    data = pin_robot.robot_data
    real2pin = pin_robot.real2pinocchio_index
    num_dofs = max(int(model.nq) - 7, 0) or 29

    dof_pos = joint_pos_full[7 : 7 + num_dofs]
    ref_quat_xyzw = wxyz_to_xyzw(ref_quat_wxyz.reshape(1, 4)).reshape(4).astype(np.float32)
    joint_pos_pin = dof_pos[real2pin]
    q = np.concatenate([ref_pos, ref_quat_xyzw, joint_pos_pin]).astype(np.float64)
    pin.framesForwardKinematics(model, data, q)

    anchor_pos = ref_pos.copy()
    anchor_quat_wxyz = xyzw_to_wxyz(ref_quat_xyzw.reshape(1, 4))

    offsets = np.asarray(VR_2POINT_OFFSETS, dtype=np.float32)
    frame_ids = [model.getFrameId(name) for name in VR_2POINT_BODIES]

    vr_pos_local_list: list[np.ndarray] = []
    vr_orn_local_list: list[np.ndarray] = []
    for i, fid in enumerate(frame_ids):
        frame_pose = data.oMf[fid]
        body_pos_w = np.asarray(frame_pose.translation, dtype=np.float32)
        body_quat_xyzw = np.asarray(pin.Quaternion(frame_pose.rotation).coeffs(), dtype=np.float32)
        body_quat_wxyz = xyzw_to_wxyz(body_quat_xyzw.reshape(1, 4))

        offset = offsets[i].reshape(1, 3)
        offset_world = quat_rotate_inverse(quat_inverse(body_quat_wxyz), offset).reshape(3)
        vr_pos_w = body_pos_w + offset_world

        diff = (vr_pos_w - anchor_pos).reshape(1, 3)
        local_pos = quat_rotate_inverse(anchor_quat_wxyz, diff).reshape(3)
        vr_pos_local_list.append(local_pos)

        local_quat_wxyz = quat_mul(quat_inverse(anchor_quat_wxyz), body_quat_wxyz)
        # 6D continuous rotation rep: first two columns of the rotation matrix,
        # C-order [r00,r01,r10,r11,r20,r21] — matches the bake's rot6d branch.
        local_orn = matrix_from_quat(local_quat_wxyz)[..., :2].reshape(-1).astype(np.float32)
        vr_orn_local_list.append(local_orn)

    runtime_pos = np.concatenate(vr_pos_local_list).astype(np.float32)
    runtime_orn = np.concatenate(vr_orn_local_list).astype(np.float32)

    np.testing.assert_allclose(bake_pos, runtime_pos, atol=1e-4)
    np.testing.assert_allclose(bake_orn, runtime_orn, atol=1e-4)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_lower_body_matches_runtime(tmp_path):
    """Verify ``lower_body_jp`` / ``lower_body_jv`` match what the runtime
    ``HolosomaWBTPolicy._compute_obs_terms`` produces for ``command_lower_body``.

    Runtime (holosoma_wbt.py:642-647)::
        ref_jp = self._motion_command_t[0, :29]   # MuJoCo-ordered dof_pos
        ref_jv = self._motion_command_t[0, 29:]   # MuJoCo-ordered dof_vel
        terms[name] = concat([ref_jp[LOWER_BODY_INDICES], ref_jv[LOWER_BODY_INDICES]])

    where ``LOWER_BODY_INDICES = [ISAACLAB_TO_MUJOCO_DOF[i] for i in range(12)]``
    -- pick the 12 MuJoCo-order indices that correspond to Isaac-Lab lower body.

    The bake must produce the same per-frame slice; otherwise the policy sees
    wrong joints in command_lower_body and tracking fails silently.
    """
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_wrappers_inference.config_values.robot_config import LOWER_BODY_INDICES

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "idle_stand_1.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(out) as d:
        bake_jp = np.asarray(d["lower_body_jp"])
        bake_jv = np.asarray(d["lower_body_jv"])

    with np.load(_FIRST_MOTION) as src:
        # MuJoCo-order dof slices from the source NPZ.
        runtime_jp = np.asarray(src["joint_pos"][:, 7:36], dtype=np.float32)
        runtime_jv = np.asarray(src["joint_vel"][:, 6:35], dtype=np.float32)

    runtime_lower_jp = runtime_jp[:, LOWER_BODY_INDICES]
    runtime_lower_jv = runtime_jv[:, LOWER_BODY_INDICES]

    np.testing.assert_allclose(bake_jp, runtime_lower_jp, atol=1e-6)
    np.testing.assert_allclose(bake_jv, runtime_lower_jv, atol=1e-6)


def _negate_anchor_quat(src: Path, dst: Path, anchor: str = "pelvis") -> None:
    """Copy a motion NPZ, replacing the ANCHOR body's quaternion with ``-q``.

    Physically the identical motion: ``q`` and ``-q`` are the same rotation, so every body
    pose, every joint angle and every keypoint position is untouched. Only the stored
    double-cover REPRESENTATIVE of the anchor changes — which is exactly the difference
    between a take whose pelvis sits near the 180 deg seam and one that does not (UMI's
    world_align Rz180 puts |w| ~ 0.1, where either representative is a hair away).
    """
    with np.load(src, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in data["body_names"].tolist()]
    idx = names.index(anchor)
    data["body_quat_w"] = data["body_quat_w"].copy()
    data["body_quat_w"][:, idx] *= -1.0
    # joint_pos carries the root prefix [xyz(3), wxyz(4), dof(29)] and IS the FK anchor for
    # the fallback path; keep the two consistent so this stays one motion, not two.
    if data["joint_pos"].shape[1] >= 7:
        data["joint_pos"] = data["joint_pos"].copy()
        data["joint_pos"][:, 3:7] *= -1.0
    np.savez(dst, **data)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_quaternion_orn_is_invariant_to_the_source_double_cover(tmp_path):
    """Negating the anchor's stored quaternion must not change the baked orn tables.

    THE BUG THIS PINS (2026-08-30). The quaternion orn tables are
    ``conj(q_anchor) * q_body``. Both inputs' double-cover representatives are chosen
    independently upstream — the anchor by whatever wrote the motion NPZ, the body by
    Eigen's ``pin.Quaternion(R).coeffs()`` — so the product's SIGN is not determined by the
    motion at all. On the two motion-completion takes (pelvis |w| ~ 0.1, courtesy of UMI's
    world_align Rz180) it landed on -identity for 100 % of frames and all five bodies:
    ``w ~ -0.99`` where na475b60 trained on ``w`` mean +0.73/+0.98/+0.96, std <= 0.17.
    That is a ~20-sigma observation, and it showed up as the arms snapping into a nonsense
    pose the instant a clip was selected — while ``idle_default`` (already +w) looked fine,
    and the dense preset (no orn term at all) was unaffected.

    Canonicalizing to ``w >= 0`` makes the tables a function of the MOTION rather than of
    an arbitrary upstream sign choice, which is what this asserts. Positions must be
    bit-identical either way too — they never depended on the sign.
    """
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion

    plus = tmp_path / "plus.npz"
    minus_src = tmp_path / "minus_src.npz"
    minus = tmp_path / "minus.npz"
    _negate_anchor_quat(_FIRST_MOTION, minus_src)
    bake_motion(_FIRST_MOTION, plus)
    bake_motion(minus_src, minus)

    with np.load(plus) as a, np.load(minus) as b:
        for key in ("vr3_orn_local", "vr5_orn_local"):
            qa, qb = a[key], b[key]
            n = qa.shape[1] // 4
            wa = qa.reshape(-1, n, 4)[..., 3]
            assert (wa >= 0.0).all(), f"{key}: bake emitted a w<0 quaternion"
            np.testing.assert_array_equal(
                qa, qb, err_msg=f"{key} changed when only the SOURCE's double-cover sign flipped"
            )
        # Positions were never sign-dependent; assert it rather than assume it.
        for key in ("vr3_pos_tcp_local", "vr5_pos_tcp_local"):
            np.testing.assert_allclose(a[key], b[key], atol=1e-6)
        # The 6D rep has no double cover, so it was already invariant — keep that true.
        np.testing.assert_allclose(a["vr3_orn6d_local"], b["vr3_orn6d_local"], atol=1e-6)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_canonicalization_preserves_the_rotation_it_encodes(tmp_path):
    """``w >= 0`` must be reached by negating the WHOLE quaternion, not by flipping w alone.

    Flipping only ``w`` would also satisfy a naive `w >= 0` assertion while silently
    mirroring the rotation, so the sign fix is checked against the rotation it must still
    represent: the baked quat, converted to a matrix, has to match the matrix built from
    the source NPZ's own ``body_quat_w`` (which the bake never sees for the body frames --
    it re-derives them through pinocchio FK).
    """
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import bake_motion
    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_BODIES

    out = tmp_path / "baked.npz"
    bake_motion(_FIRST_MOTION, out)

    def R_of_wxyz(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    with np.load(_FIRST_MOTION, allow_pickle=True) as s, np.load(out) as b:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in s["body_names"].tolist()]
        pel = names.index("pelvis")
        q5 = b["vr5_orn_local"].reshape(-1, 5, 4)
        for t in (0, q5.shape[0] // 2, q5.shape[0] - 1):
            R_pel = R_of_wxyz(s["body_quat_w"][t, pel].astype(np.float64))
            for k, body in enumerate(VR_5POINT_BODIES):
                R_body = R_of_wxyz(s["body_quat_w"][t, names.index(body)].astype(np.float64))
                want = R_pel.T @ R_body
                x, y, z, w = q5[t, k].astype(np.float64)
                got = R_of_wxyz(np.array([w, x, y, z]))
                np.testing.assert_allclose(
                    got, want, atol=2e-3,
                    err_msg=f"t={t} {body}: canonicalized quat no longer encodes the source rotation",
                )


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_motion_dir_rebakes_when_only_the_format_VERSION_is_stale(tmp_path):
    """A values-only fix must invalidate an mtime-current cache.

    Every earlier change to this bake happened to ADD a key, so the key-presence schema
    check was enough. The quaternion canonicalization changed what EXISTING keys contain,
    which that check cannot see — and `bake_motion_dir` reuses any output newer than its
    source. The observed consequence: rebuilding after the fix logged "baked 2 motion(s)"
    and re-shipped the unfixed tables. `bake_format_version` closes that, and this test is
    what keeps it closed.
    """
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    pytest.importorskip("wbt_wrappers_inference")

    from wbt_realtime_eval.bake_motions import _BAKE_FORMAT_VERSION, bake_motion, bake_motion_dir

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    src = in_dir / _FIRST_MOTION.name
    src.write_bytes(_FIRST_MOTION.read_bytes())

    # A FULLY CURRENT bake (all keys, right widths) whose only defect is an old version.
    good = out_dir / _FIRST_MOTION.name
    bake_motion(src, good)
    with np.load(good, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    assert int(data["bake_format_version"]) == _BAKE_FORMAT_VERSION
    data["bake_format_version"] = np.asarray(_BAKE_FORMAT_VERSION - 1, dtype=np.int32)
    sentinel = np.full_like(data["vr5_orn_local"], 0.5)
    data["vr5_orn_local"] = sentinel  # detectably wrong values, right key and width
    np.savez(good, **data)
    newer = src.stat().st_mtime + 10.0
    os.utime(good, (newer, newer))

    outs = bake_motion_dir(in_dir, out_dir)
    assert outs == [good]
    with np.load(good) as d:
        assert int(d["bake_format_version"]) == _BAKE_FORMAT_VERSION
        assert not np.allclose(d["vr5_orn_local"], sentinel), "stale values survived the re-bake"

    # ...and an output that IS current is still skipped (the gate must not force a re-bake
    # every time, which would make every build pay the FK cost).
    mtime = good.stat().st_mtime
    assert bake_motion_dir(in_dir, out_dir) == [good]
    assert good.stat().st_mtime == mtime
