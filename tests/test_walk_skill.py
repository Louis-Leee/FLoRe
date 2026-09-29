"""The WALK SKILL (``fuji/s1_policy/serving/walk_skill.py``) and its wiring into the 5pt page.

Every claim the module's docstring makes that can be checked on a CPU with no clip, no model and
no robot is checked here against a SYNTHETIC holosoma-shaped NPZ; the bundled walk-with-box feet
are checked directly, and pinned equal to a fresh extraction of the source NPZ / to mjswan's own
bake of it when those files are present (skipped otherwise). The wiring tests are source-level over ``scripts/hl_serving_5pt.py`` for the same
reason ``test_hl_serving_mc_wiring.py`` is: the page is one closure over six sockets and a GPU.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fuji.s1_policy.serving import hl_dense, walk_skill as W
from fuji.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES as SDK

_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _ROOT / "scripts" / "hl_serving_5pt.py"
_REAL_CLIP = Path(os.path.expanduser("~/data/walk_clip")) / W.DEFAULT_CLIP
_BAKED = (_ROOT / "third_party" / "holosoma_extensions" / "dist" / "viz_5pt_walk_baked"
          / W.DEFAULT_CLIP)

FPS = 50.0
DT = 1.0 / FPS
#: A ready-like wire frame: distinct upper values so a slot mix-up is a value error, not a shape one.
READY_POS = np.array([[0.44, 0.15, 0.10], [0.44, -0.15, 0.10], [0.0, 0.0, 0.39],
                      [0.0, 0.12, -0.72], [0.0, -0.12, -0.72]])
READY_ORN = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1))
READY = (READY_POS.reshape(-1), READY_ORN.reshape(-1))


def _xyzw_to_wxyz(q):
    q = np.asarray(q)
    return q[..., [3, 0, 1, 2]]


def _joint_pattern(T):
    """A KNOWN 29-joint channel: joint k oscillates about 0.1*k with its own amplitude and phase, with
    the analytic derivative as the velocity — so an extraction, a resample and a speed scaling can
    each be checked against arithmetic rather than against themselves."""
    t = np.arange(T) * DT
    k = np.arange(29)
    q = 0.1 * k[None, :] + 0.2 * np.sin(2 * np.pi * 0.5 * t[:, None] + 0.3 * k[None, :])
    qd = 0.2 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t[:, None] + 0.3 * k[None, :])
    return q, qd


def _synthetic(tmp_path, *, T=80, moving_feet=True, names=None, drop_key=None, joints=False,
               joint_names=None):
    """A holosoma-shaped NPZ: a pelvis that translates AND rotates, feet at KNOWN body-frame poses.

    ``joints=True`` also writes the 29-joint channel of :func:`_joint_pattern` under the SDK joint
    names (root prefix included, as the real files have it); the default keeps the legacy
    duplicate-name stub, i.e. a clip whose joint channel is UNUSABLE and loads feet-only.

    Returns ``(path, expected_local_pos (T,2,3), expected_local_quat_xyzw (T,2,4))``."""
    t = np.arange(T) * DT
    # Pelvis: walks along +x with a bob, yaws 40 deg over the clip and pitches ±5 deg.
    p_pelvis = np.stack([0.8 * t, 0.05 * np.sin(2 * np.pi * t), 0.75 + 0.02 * np.cos(4 * np.pi * t)], 1)
    r_pelvis = Rotation.from_euler("ZYX", np.stack([np.deg2rad(40.0) * t / t[-1],
                                                    np.deg2rad(5.0) * np.sin(2 * np.pi * t),
                                                    np.zeros(T)], 1))
    # Feet in the PELVIS frame: a stride (swinging fore-aft, lifting) or a fixed stance.
    if moving_feet:
        swing = 0.25 * np.sin(2 * np.pi * 1.0 * t)
        lift = np.clip(0.08 * np.sin(2 * np.pi * 1.0 * t), 0.0, None)
        lp = np.stack([swing, 0.12 + 0 * t, -0.72 + lift], 1)
        rp = np.stack([-swing, -0.12 + 0 * t, -0.72 + np.clip(-0.08 * np.sin(2 * np.pi * t), 0, None)], 1)
        lq = Rotation.from_euler("ZYX", np.stack([0 * t, 0.3 * np.sin(2 * np.pi * t), 0 * t], 1))
        rq = Rotation.from_euler("ZYX", np.stack([0 * t, -0.3 * np.sin(2 * np.pi * t), 0 * t], 1))
    else:
        lp = np.tile([0.0, 0.12, -0.72], (T, 1))
        rp = np.tile([0.0, -0.12, -0.72], (T, 1))
        lq = Rotation.from_quat(np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)))
        rq = Rotation.from_quat(np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)))
    exp_pos = np.stack([lp, rp], 1)
    exp_quat = W.canonical_xyzw(np.stack([lq.as_quat(), rq.as_quat()], 1))
    # World poses = pelvis (x) local.
    l_w = p_pelvis + r_pelvis.apply(lp)
    r_w = p_pelvis + r_pelvis.apply(rp)
    lq_w = (r_pelvis * lq).as_quat()
    rq_w = (r_pelvis * rq).as_quat()
    # Deliberately hand the extractor a NON-canonical world quaternion for the right foot on every
    # other frame (q and -q), so the w>=0 canonicalisation is exercised rather than assumed.
    rq_w[::2] *= -1.0
    names = names or ["pelvis", "left_hip_pitch_link", "left_ankle_roll_link", "torso_link",
                      "right_ankle_roll_link", "right_wrist_yaw_link"]
    n = len(names)
    body_pos = np.zeros((T, n, 3), np.float32)
    body_quat = np.zeros((T, n, 4), np.float32)
    body_quat[..., 0] = 1.0
    if "pelvis" in names:
        i = names.index("pelvis")
        body_pos[:, i] = p_pelvis
        body_quat[:, i] = _xyzw_to_wxyz(r_pelvis.as_quat())
    if "left_ankle_roll_link" in names:
        i = names.index("left_ankle_roll_link")
        body_pos[:, i] = l_w
        body_quat[:, i] = _xyzw_to_wxyz(lq_w)
    if "right_ankle_roll_link" in names:
        i = names.index("right_ankle_roll_link")
        body_pos[:, i] = r_w
        body_quat[:, i] = _xyzw_to_wxyz(rq_w)
    arrays = {"fps": np.int64(50), "joint_pos": np.zeros((T, 36), np.float32),
              "body_pos_w": body_pos, "body_quat_w": body_quat,
              "body_names": np.array(names), "joint_names": np.array(["j"] * 29)}
    if joints:
        q, qd = _joint_pattern(T)
        jp = np.zeros((T, 36), np.float64)
        jp[:, 7:] = q
        jv = np.zeros((T, 35), np.float64)
        jv[:, 6:] = qd
        arrays.update({"joint_pos": jp, "joint_vel": jv,
                       "joint_names": np.array(list(joint_names or SDK))})
    if drop_key:
        arrays.pop(drop_key)
    path = tmp_path / "synthetic_walk.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path, exp_pos, exp_quat


def _all_steps(player, *, max_steps=5000, stop_at=None):
    """Drive a player to ``done``; returns ``(feet (n,2,3), quats (n,2,4), phases)``."""
    pos, quat, phases = [], [], []
    for i in range(max_steps):
        p, q, ph = player.step()
        pos.append(p)
        quat.append(q)
        phases.append(ph)
        if stop_at is not None and i == stop_at:
            player.request_stop()
        if player.done:
            break
    return np.array(pos), np.array(quat), phases


def _pool_steps(feet):
    """Per-step L2 over the CONCATENATED (6,) foot-position vector — the low level's own metric."""
    d = np.diff(feet.reshape(len(feet), -1), axis=0)
    return np.linalg.norm(d, axis=-1)


# ---------------------------------------------------------------------------
# extraction: the frame is the low level's own
# ---------------------------------------------------------------------------
def test_extract_recovers_the_body_frame_feet_under_a_moving_rotating_pelvis(tmp_path):
    path, exp_pos, exp_quat = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    assert clip.n_frames == 80 and clip.fps == 50.0
    # float32 storage of world poses ~1 m from the origin: ~1e-7 relative, so 1 µm is generous.
    assert np.abs(clip.foot_pos - exp_pos).max() < 1e-6
    # Quaternions: same rotation, and on the canonical branch even though the input flipped sign.
    dots = np.abs(np.sum(clip.foot_quat * exp_quat, axis=-1))
    assert np.all(dots > 1.0 - 1e-6)
    assert np.all(clip.foot_quat[..., 3] >= 0.0)
    assert np.allclose(np.linalg.norm(clip.foot_quat, axis=-1), 1.0, atol=1e-6)


def test_a_fixed_stance_extracts_as_a_constant_whatever_the_pelvis_does(tmp_path):
    path, exp_pos, _ = _synthetic(tmp_path, moving_feet=False)
    clip = W.load_walk_clip(path)
    assert np.abs(clip.foot_pos - clip.foot_pos[0]).max() < 1e-6
    st = clip.stats()
    assert st.max_step_m < 1e-6 and st.max_orn_step < 1e-5
    # ... while the pelvis genuinely turned: the yaw report is the pelvis's, not the feet's.
    assert 35.0 < st.yaw_net_deg < 45.0


def test_extract_refuses_a_clip_without_an_ankle_by_name(tmp_path):
    path, _, _ = _synthetic(tmp_path, names=["pelvis", "left_ankle_roll_link", "torso_link"])
    with pytest.raises(ValueError, match="right_ankle_roll_link"):
        W.load_walk_clip(path)


def test_extract_reads_the_pelvis_by_name_not_by_index(tmp_path):
    # Pelvis is body 3 here; the result must be identical to the body-0 layout.
    names = ["torso_link", "left_ankle_roll_link", "right_ankle_roll_link", "pelvis"]
    path, exp_pos, _ = _synthetic(tmp_path, names=names)
    clip = W.load_walk_clip(path)
    assert np.abs(clip.foot_pos - exp_pos).max() < 1e-6


def test_load_refuses_a_non_holosoma_npz_by_name(tmp_path):
    path, _, _ = _synthetic(tmp_path, drop_key="body_quat_w")
    with pytest.raises(ValueError, match="body_quat_w"):
        W.load_walk_clip(path)
    with pytest.raises(FileNotFoundError):
        W.load_walk_clip(tmp_path / "nope.npz")


# ---------------------------------------------------------------------------
# the extracted (bundled) format
# ---------------------------------------------------------------------------
def test_save_feet_clip_round_trips_bit_exactly_and_keeps_the_provenance(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    out = W.save_feet_clip(clip, tmp_path / "feet" / "x.npz", source_sha256="abc123")
    back = W.load_walk_clip(out)
    assert (back.foot_pos == clip.foot_pos).all() and (back.foot_quat == clip.foot_quat).all()
    assert (back.root_yaw_deg == clip.root_yaw_deg).all() and back.fps == clip.fps
    assert back.name == clip.name            # the SOURCE's name, not the bundle file's
    assert back.source == str(out)
    with np.load(out, allow_pickle=True) as z:
        assert str(z["schema"]) == W.FEET_SCHEMA and str(z["source_sha256"]) == "abc123"
        assert list(z["foot_bodies"]) == list(W.FOOT_BODIES) and str(z["pelvis_body"]) == W.PELVIS_BODY
    assert out.stat().st_size < 120_000      # the feet only — not a copy of the source


def test_load_refuses_a_bundle_whose_feet_are_not_the_wires(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    out = W.save_feet_clip(clip, tmp_path / "x.npz")
    with np.load(out, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d["foot_bodies"] = np.array(["right_ankle_roll_link", "left_ankle_roll_link"])   # swapped
    np.savez(tmp_path / "swapped.npz", **d)
    with pytest.raises(ValueError, match="foot_bodies"):
        W.load_walk_clip(tmp_path / "swapped.npz")
    d["foot_bodies"] = np.array(list(W.FOOT_BODIES))
    d["schema"] = np.array("fuji.walk_feet.v0")
    np.savez(tmp_path / "old.npz", **d)
    with pytest.raises(ValueError, match="schema"):
        W.load_walk_clip(tmp_path / "old.npz")


def test_the_bundled_clip_is_the_walk_with_box_feet_and_needs_no_flag():
    """The page ships the skill: the asset is inside the package, loads through the same reader, and
    IS the clip the user named (its name, frame count and mid-stride frame 0 pin it)."""
    assert W.BUNDLED_CLIP.is_file(), W.BUNDLED_CLIP
    assert W.BUNDLED_CLIP.parent.name == "assets" and W.BUNDLED_CLIP.parent.parent.name == "serving"
    clip = W.load_default_clip()
    assert clip.name == W.DEFAULT_CLIP and clip.fps == 50.0 and clip.n_frames == 320
    assert np.allclose(clip.foot_pos[0, 0], [0.0703, 0.0402, -0.7133], atol=1e-4)
    assert np.allclose(clip.foot_pos[0, 1], [-0.2423, -0.1242, -0.6796], atol=1e-4)
    assert np.all(clip.foot_quat[..., 3] >= 0.0)
    # And since 2026-09-07 its 29 joints by name — the legs that ride with the feet as `ref_joints`.
    assert clip.has_joints and clip.legs_note == ""
    assert clip.joint_names == tuple(SDK) and clip.joint_pos.shape == (320, 29) == clip.joint_vel.shape
    assert 0.0 < np.degrees(clip.joint_pos[:, SDK.index("left_knee_joint")].max()) < 90.0
    assert "legs: 29 reference joints" in clip.stats().describe(feet_slewed=False)
    with np.load(W.BUNDLED_CLIP, allow_pickle=True) as z:
        assert len(str(z["source_sha256"])) == 64     # provenance travels with the feet
        assert {"joint_names", "joint_pos", "joint_vel"} <= set(z.files)


@pytest.mark.skipif(not _REAL_CLIP.is_file(), reason="the source walk clip is not on this box")
def test_the_bundle_equals_a_fresh_extraction_of_the_source_bit_for_bit():
    import hashlib

    fresh = W.load_walk_clip(_REAL_CLIP)
    bundled = W.load_default_clip()
    assert (fresh.foot_pos == bundled.foot_pos).all() and (fresh.foot_quat == bundled.foot_quat).all()
    assert (fresh.root_yaw_deg == bundled.root_yaw_deg).all()
    assert fresh.joint_names == bundled.joint_names
    assert (fresh.joint_pos == bundled.joint_pos).all() and (fresh.joint_vel == bundled.joint_vel).all()
    # ...and the joints ARE the source's own joint_pos[:, 7:] / joint_vel[:, 6:], untouched.
    with np.load(_REAL_CLIP) as src:
        assert (np.asarray(src["joint_pos"], dtype=np.float64)[:, 7:] == bundled.joint_pos).all()
        assert (np.asarray(src["joint_vel"], dtype=np.float64)[:, 6:] == bundled.joint_vel).all()
    with np.load(W.BUNDLED_CLIP, allow_pickle=True) as z:
        assert str(z["source_sha256"]) == hashlib.sha256(_REAL_CLIP.read_bytes()).hexdigest()


def test_the_foot_slots_are_the_wires_last_two_bodies_in_wire_order():
    assert W.FOOT_BODIES == ("left_ankle_roll_link", "right_ankle_roll_link")
    assert W.FOOT_BODIES == hl_dense.FIVE_POINT_BODIES[hl_dense.FIVE_POINT_GATED:]
    assert W.N_UPPER == hl_dense.FIVE_POINT_GATED == 3 and W.N_POINTS == 5


def test_the_tick_rate_matches_hl_5pt():
    from fuji.s1_policy.serving import hl_5pt

    assert W.LOW_LEVEL_HZ == hl_5pt.LOW_LEVEL_HZ


# ---------------------------------------------------------------------------
# quaternions
# ---------------------------------------------------------------------------
def test_slerp_takes_the_short_path_across_the_double_cover_and_lands_on_w_positive():
    q0 = np.array([[0.0, 0.0, 0.0, 1.0]])
    r1 = Rotation.from_euler("z", 30, degrees=True).as_quat()
    q1 = -r1[None]                       # the antipode: same rotation, opposite sign
    mid = W.slerp_xyzw(q0, q1, 0.5)
    ang = np.degrees(W.quat_angle_rad(q0, mid))
    assert abs(ang[0] - 15.0) < 1e-6      # halfway along the 30 deg arc, not the 330 deg one
    assert mid[0, 3] >= 0.0 and abs(np.linalg.norm(mid[0]) - 1.0) < 1e-9
    assert np.allclose(W.slerp_xyzw(q0, q1, 0.0), q0)
    assert np.allclose(np.abs(W.slerp_xyzw(q0, q1, 1.0)), np.abs(r1))


def test_frame_interpolates_between_two_frames(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    p, q = clip.frame(10.5)
    assert np.allclose(p, 0.5 * (clip.foot_pos[10] + clip.foot_pos[11]))
    half = np.degrees(W.quat_angle_rad(clip.foot_quat[10], q))
    full = np.degrees(W.quat_angle_rad(clip.foot_quat[10], clip.foot_quat[11]))
    assert np.allclose(half, full / 2.0, atol=1e-6)
    p_end, _ = clip.frame(1e9)             # clamped, never out of range
    assert np.allclose(p_end, clip.foot_pos[-1])


# ---------------------------------------------------------------------------
# composing the wire frame
# ---------------------------------------------------------------------------
def test_compose_holds_the_upper_three_slots_byte_for_byte_and_replaces_the_feet():
    fp = np.array([[0.1, 0.2, -0.7], [-0.1, -0.2, -0.71]])
    fq = Rotation.from_euler("y", np.array([[10.0], [-10.0]]), degrees=True).as_quat()
    pos, orn = compose = W.compose_frame(READY, fp, fq)
    assert pos.shape == (15,) and orn.shape == (20,)
    assert np.array_equal(pos.reshape(5, 3)[:3], READY_POS[:3])
    assert np.array_equal(orn.reshape(5, 4)[:3], READY_ORN[:3])
    assert np.allclose(pos.reshape(5, 3)[3:], fp)
    assert np.allclose(np.abs(orn.reshape(5, 4)[3:]), np.abs(fq))
    ok, why = hl_dense.is_safe_keypoint_command(*compose)
    assert ok, why
    back_p, back_q = W.feet_from_frame(compose)
    assert np.allclose(back_p, fp)
    assert np.all(back_q[:, 3] >= 0.0)


def test_compose_refuses_a_non_finite_foot():
    fp = np.array([[0.1, 0.2, np.nan], [-0.1, -0.2, -0.71]])
    with pytest.raises(ValueError, match="non-finite"):
        W.compose_frame(READY, fp, READY_ORN[3:])


# ---------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------
def _player(clip, **kw):
    sp, sq = W.feet_from_frame(READY)
    kw.setdefault("step_dt", DT)
    kw.setdefault("ramp_speed_m_s", 0.25)
    kw.setdefault("ramp_min_s", 0.5)
    return W.WalkPlayer(clip, start_pos=sp, start_quat=sq, **kw)


def test_the_player_leads_in_from_the_start_feet_to_frame_zero_within_the_rate_bound(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=False)
    sp, _ = W.feet_from_frame(READY)
    feet, quats, phases = _all_steps(pl)
    n_in = phases.index("clip")
    assert n_in == pl.lead_in_steps >= int(0.5 / DT)
    # Step 0 leaves the start feet by at most the bound; the ramp's last step IS frame 0.
    pool = _pool_steps(np.concatenate([sp[None], feet[:n_in]]))
    assert pool.max() <= 0.25 * DT + 1e-9
    assert np.allclose(feet[n_in - 1], clip.foot_pos[0])
    assert np.all(quats[..., 3] >= 0.0)
    assert np.allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-9)


def test_the_clip_phase_is_the_clips_own_feet_verbatim(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=False, speed=1.0)
    feet, _, phases = _all_steps(pl)
    idx = [i for i, ph in enumerate(phases) if ph == "clip"]
    # At 1x and publish == fps every clip step is an integer frame: frames 1..T-1.
    assert len(idx) == clip.n_frames - 1
    assert np.allclose(feet[idx], clip.foot_pos[1:])


def test_no_loop_plays_once_then_ramps_back_to_the_start_feet_and_ends(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=False)
    sp, sq = W.feet_from_frame(READY)
    feet, quats, phases = _all_steps(pl)
    assert pl.done and phases[-1] == "lead_out" and "wrap" not in phases
    assert np.allclose(feet[-1], sp) and np.allclose(np.abs(quats[-1]), np.abs(sq))
    out = [i for i, ph in enumerate(phases) if ph == "lead_out"]
    pool = _pool_steps(feet[out[0] - 1:])
    assert pool.max() <= 0.25 * DT + 1e-9
    # Once done, further steps hold rather than raise.
    p, q, ph = pl.step()
    assert ph == "done" and np.allclose(p, sp)


def test_request_stop_mid_clip_ramps_home_from_where_the_feet_are(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=True)
    sp, _ = W.feet_from_frame(READY)
    feet, _, phases = _all_steps(pl, stop_at=pl.lead_in_steps + 20)
    k = phases.index("lead_out")
    assert phases[k - 1] == "clip"
    # The lead-out starts from the LAST clip feet (continuity), not from frame 0 or the ready feet.
    assert _pool_steps(feet[k - 1:k + 1])[0] <= 0.25 * DT + 1e-9
    assert np.allclose(feet[-1], sp) and pl.done
    pl.request_stop()                     # idempotent once stopping/done
    assert pl.done


def test_loop_wraps_through_a_rate_bounded_blend_rather_than_a_cut(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=True)
    n_clip = clip.n_frames - 1
    feet, _, phases = _all_steps(pl, stop_at=pl.lead_in_steps + 2 * n_clip + 60)
    assert "wrap" in phases and pl.laps >= 1
    wrap = np.array([ph == "wrap" for ph in phases])
    pool = _pool_steps(feet)
    assert pool[wrap[1:]].max() <= 0.25 * DT + 1e-9
    # The wrap lands ON frame 0 and the next clip step is frame 1 — no frame is skipped or repeated.
    first_wrap_end = max(i for i, ph in enumerate(phases) if ph == "wrap" and phases[i + 1] == "clip")
    assert np.allclose(feet[first_wrap_end], clip.foot_pos[0])
    assert np.allclose(feet[first_wrap_end + 1], clip.foot_pos[1])


def test_speed_scales_the_phase_advance_and_is_clamped(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    slow = _player(clip, loop=False, speed=0.5)
    fast = _player(clip, loop=False, speed=2.0)
    _, _, ph_s = _all_steps(slow)
    _, _, ph_f = _all_steps(fast)
    assert ph_s.count("clip") == 2 * (clip.n_frames - 1)
    assert ph_f.count("clip") == (clip.n_frames - 1) // 2
    pl = _player(clip)
    pl.set_speed(100.0)
    assert pl.speed == W.SPEED_MAX
    pl.set_speed(float("nan"))
    assert pl.speed == W.SPEED_MAX          # ignored, not applied
    pl.set_speed(0.0)
    assert pl.speed == W.SPEED_MIN


def test_a_ramp_bounds_the_concatenated_pool_not_each_foot(tmp_path):
    # Both feet 0.1 m away in the same direction: a per-foot bound would allow 5 mm each = 7.07 mm
    # of pool per step; the bound has to hold on the (6,) vector the low level actually caps.
    path, _, _ = _synthetic(tmp_path, moving_feet=False)
    clip = W.load_walk_clip(path)
    sp, sq = W.feet_from_frame(READY)
    far = clip.foot_pos[0] + np.array([0.1, 0.0, 0.0])
    pl = W.WalkPlayer(clip, start_pos=far, start_quat=sq, step_dt=DT, loop=False,
                      ramp_speed_m_s=0.25, ramp_min_s=0.0)
    feet, _, phases = _all_steps(pl)
    n_in = phases.index("clip")
    pool = _pool_steps(np.concatenate([far[None], feet[:n_in]]))
    assert pool.max() <= 0.25 * DT + 1e-9
    assert n_in == int(np.ceil(np.sqrt(2) * 0.1 / (0.25 * DT)))


def test_every_composed_frame_of_a_full_run_passes_the_wires_gate(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=True)
    for i in range(600):
        p, q, _ = pl.step()
        frame = W.compose_frame(READY, p, q)
        ok, why = hl_dense.is_safe_keypoint_command(*frame)
        assert ok, why
        assert np.array_equal(frame[0][:9], READY[0][:9]) and np.array_equal(frame[1][:12], READY[1][:12])
        if i == 400:
            pl.request_stop()
        if pl.done:
            break
    assert pl.done


# ---------------------------------------------------------------------------
# the report: the cap is stated, both ratios, at the selected speed
# ---------------------------------------------------------------------------
def test_stats_score_the_shared_pool_step_against_the_cap_and_flag_the_clamp(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    st = W.load_walk_clip(path).stats()
    # A 0.5 m stride at 1 Hz peaks near 1.6 m/s per foot = ~31 mm/tick per foot.
    assert st.peak_foot_speed_ms > 1.0
    assert st.max_step_m >= st.per_foot_step_m.max()          # pool >= any single foot
    # Scored as a low level that COVERS the feet reports itself (feet_slewed=True): the clamp is a
    # warning. (The exempt / not-yet-reported readings have their own test below.)
    line = st.describe(speed=1.0, budget_m_per_tick=0.005, budget_source="pre-flight", feet_slewed=True)
    assert "⚠️" in line and "CLAMPED" in line and "pre-flight" in line and "5.0 mm/tick" in line
    # Slower halves the ratio; a generous cap removes the position flag but the orientation one is
    # scored SEPARATELY against its own cap.
    half = st.describe(speed=0.5, budget_m_per_tick=0.005, feet_slewed=True)
    r_full = float(re.search(r"([\d.]+)x \(mean\)", line).group(1))
    r_half = float(re.search(r"([\d.]+)x \(mean\)", half).group(1))
    assert abs(r_half - r_full / 2.0) < 0.15
    loose = st.describe(speed=1.0, budget_m_per_tick=1.0, orn_cap_per_tick=1.0, feet_slewed=True)
    assert "CLAMPED" not in loose and "ORIENTATION" not in loose
    only_orn = st.describe(speed=1.0, budget_m_per_tick=1.0, orn_cap_per_tick=1e-4, feet_slewed=True)
    assert "ORIENTATION cap" in only_orn and "CLAMPED" not in only_orn


def test_describe_reads_the_feet_exemption_off_the_robot_rather_than_assuming_it(tmp_path):
    """Three readings of one fact. Reported EXEMPT: the demand is printed and no ratio is a warning.
    Reported COVERED: the clamp is a warning that names the fix (relaunch with SLEW_FEET=0). Not yet
    reported: the ratio is printed conditionally, because nothing has been observed."""
    path, _, _ = _synthetic(tmp_path)
    st = W.load_walk_clip(path).stats()
    exempt = st.describe(speed=1.0, budget_m_per_tick=0.005, feet_slewed=False)
    assert "EXEMPT" in exempt and "⚠️" not in exempt and "x (mean)" not in exempt
    covered = st.describe(speed=1.0, budget_m_per_tick=0.005, feet_slewed=True)
    assert "⚠️" in covered and "CLAMPED" in covered and "SLEW_FEET=0" in covered
    unknown = st.describe(speed=1.0, budget_m_per_tick=0.005, feet_slewed=None)
    assert "IF the low level exempts" in unknown and "⚠️" not in unknown and "x (mean)" in unknown
    # The absolute demand (cap-independent) is identical in all three.
    for line in (exempt, covered, unknown):
        assert re.search(r"feet demand mean \d+ / p95 \d+ / max \d+ mm per tick", line)


def test_stats_report_the_wrap_gap_and_the_lead_in_gap(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    st = clip.stats()
    assert np.allclose(st.wrap_gap_m, np.linalg.norm(clip.foot_pos[-1] - clip.foot_pos[0], axis=-1))
    sp, _ = W.feet_from_frame(READY)
    assert np.allclose(st.lead_in_gap_m(sp), np.linalg.norm(sp - clip.foot_pos[0], axis=-1))
    assert st.duration_s == pytest.approx(clip.n_frames / 50.0)


# ---------------------------------------------------------------------------
# the real clip and mjswan's own bake of it (skipped when either is absent)
# ---------------------------------------------------------------------------
def test_the_walk_with_box_clip_is_mid_stride_and_far_over_the_cap_it_is_exempt_from():
    """The bundled clip's own numbers — the ones the low-level exemption exists for."""
    clip = W.load_default_clip()
    st = clip.stats()
    assert clip.fps == 50.0 and clip.n_frames == 320
    # Frame 0: right foot ~24 cm behind the left — the reason the feet are NOT per-point re-based.
    assert clip.foot_pos[0, 0, 0] - clip.foot_pos[0, 1, 0] > 0.25
    # 46 mm/tick mean, 95 max: 9x / 19x the 5 mm cap — slewed, this is a crawl, not a walk.
    assert 0.040 < st.mean_step_m < 0.055 and st.max_step_m > 0.08
    assert st.peak_foot_speed_ms > 4.0
    assert st.mean_step_m / 0.005 > 8.0 and st.p95_orn_step / W.LAUNCHER_ORN_CAP_PER_TICK > 4.0
    assert "CLAMPED" in st.describe(speed=1.0, budget_m_per_tick=0.005, feet_slewed=True)
    assert "EXEMPT" in st.describe(speed=1.0, budget_m_per_tick=0.005, feet_slewed=False)
    # A straight walk with the sway of a carried box: net heading small, no lap around the room.
    assert abs(st.yaw_net_deg) < 30.0 and st.yaw_range_deg < 60.0


@pytest.mark.skipif(not _BAKED.is_file(), reason="needs mjswan's baked table of the clip")
def test_the_extraction_matches_mjswans_own_bake_to_a_micron():
    clip = W.load_default_clip()   # the bundle is pinned equal to the source's extraction above
    with np.load(_BAKED, allow_pickle=True) as baked:
        pos = np.asarray(baked["vr5_pos_tcp_local"], dtype=np.float64).reshape(-1, 5, 3)[:, 3:]
        orn = np.asarray(baked["vr5_orn_local"], dtype=np.float64).reshape(-1, 5, 4)[:, 3:]
    assert np.abs(clip.foot_pos - pos).max() < 1e-5          # 10 µm (float32 bake)
    assert np.abs(clip.foot_quat - orn).max() < 1e-5


# ---------------------------------------------------------------------------
# wiring into the page (source-level, like the sibling wiring tests)
# ---------------------------------------------------------------------------
_SRC = _PAGE.read_text()


_WALK_MARK = "# ---------------- the WALK SKILL (extension) ----------------"
_LEGS_MARK = "# ---------------- the LEG OVERRIDE (extension) ----------------"
_REC_MARK = "# ---------------- the RUN RECORDER (extension) ----------------"


def _walk_block() -> str:
    """The 🚶 walk: from its marker to the LEG OVERRIDE's (the block that follows it)."""
    return _SRC[_SRC.index(_WALK_MARK):_SRC.index(_LEGS_MARK)]


def _legs_block() -> str:
    """The LEG OVERRIDE: from its marker to the RUN RECORDER's (the block that follows it)."""
    return _SRC[_SRC.index(_LEGS_MARK):_SRC.index(_REC_MARK)]


def test_the_walk_panel_is_one_self_contained_block_after_the_reviewed_page():
    """No flag gates it any more (the clip is bundled), so the extension claim is structural: the
    whole skill is ONE block placed after the reviewed controls, and nothing above that block reads
    a name it defines — the page above it is byte-for-byte the page reviewed before the skill."""
    block = _walk_block()
    start = _SRC.index("# ---------------- the WALK SKILL (extension) ----------------")
    page_before = _SRC[_SRC.index("def _run_viser("):start]
    for name in ("walk_running", "walk_ctl", "walk_paths", "_run_walk", "b_walk", "g_walk", "g_wdemand",
                 "_walk_demand", "_draw_walk"):
        assert name not in page_before, name
    assert "walk_running = threading.Event()" in block
    assert block.count("add_folder(") == 1 and "add_dropdown" not in block   # ONE clip, no picker
    assert "--walk-dir" not in _SRC and "walk_clips" not in _SRC
    assert re.search(r'add_argument\("--walk-clip", default=str\(walk_skill\.BUNDLED_CLIP\)', _SRC)
    assert "set_defaults(walk_loop=True)" in _SRC
    # The chunk-scoring path reads the low level's REPORTED coverage, not an assumption.
    # Three sites score a chunk — the policy Predict, the recorded-episode REPLAY (2026-09-19) and the
    # stream — and every one of them reads the low level's REPORTED coverage, never an assumption.
    assert _SRC.count("covered=wire.slew.slew_mask") == 3
    assert "feet_slewed=wire.slew.feet_slewed" in block


def test_the_walk_publishes_through_the_reviewed_wire_path_and_owns_the_page():
    run = _walk_block()
    run = run[run.index("def _run_walk()"):run.index("@b_walk.on_click")]
    # Same three calls as `_drip`, in the same order: gate+send, record the wire, advance the base —
    # each carrying the LEGS (`ref_joints=legs`, None when the checkbox is off) beside the feet.
    assert (run.index("_send(frame, fcmd if on else None, ref_joints=legs)")
            < run.index("_set_wire(frame, fcmd, ref_joints=legs)") < run.index("wire.commit(frame)"))
    # Owns the page: `busy` (Predict/Run) and `dripping` (the heartbeat, both ▶▶ cadences).
    assert "busy.acquire(blocking=False)" in run and "dripping.set()" in run
    assert "busy.release()" in run and "dripping.clear()" in run and "walk_running.clear()" in run
    # The feet bypass the PER-POINT clutch and the planner is untouched (see the module docstring).
    assert "clutch.apply" not in run and "wire.plan(" not in run and "commit_role" not in run
    # STOP freezes within one step; the toggle ramps back.
    assert "if stop.is_set():" in run and "player.request_stop()" in run
    # The cap is measured while the wire is saturated.
    assert "wire.slew.observe(_record_frame())" in run


def test_the_main_loop_refuses_predict_run_reset_while_a_walk_is_active():
    assert re.search(r'if walk_running\.is_set\(\) and \(flags\["predict"\] or flags\["run"\] or '
                     r'flags\["reset"\]\)', _SRC)


def test_the_toggle_starts_on_its_own_thread_and_stops_by_request():
    block = _walk_block()
    # The body is `_walk_toggle()` (also the `k` hotkey's, tty_keys); the button is a one-line call to it.
    handler = block[block.index("def _walk_toggle() -> str:"):block.index("@s_wspeed.on_update")]
    assert 'walk_ctl["stop_req"] = True' in handler
    assert 'threading.Thread(target=_run_walk, daemon=True, name="walk")' in handler
    button = handler[handler.index("@b_walk.on_click"):]
    assert "_walk_toggle()" in button and "stop_req" not in button


def test_the_parser_defaults_point_at_the_bundled_clip_with_no_flag():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_walk_wiring", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_walk_wiring"] = mod
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(["--checkpoint", "x"])
    assert not hasattr(args, "walk_dir")
    assert args.walk_clip == str(W.BUNDLED_CLIP) and Path(args.walk_clip).is_file()
    assert args.walk_loop is True and args.walk_speed == 1.0
    assert args.walk_hz == W.LOW_LEVEL_HZ and args.walk_ramp_speed == 0.25
    args2 = mod.build_parser().parse_args(["--checkpoint", "x", "--walk-clip", "/tmp/x.npz",
                                           "--no-walk-loop", "--walk-speed", "0.5"])
    assert args2.walk_clip == "/tmp/x.npz" and args2.walk_loop is False and args2.walk_speed == 0.5
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--checkpoint", "x", "--walk-dir", "/tmp/x"])


# ---------------------------------------------------------------------------
# the LEGS: the clip's 29 joints ride with its feet (ref_joints)
# ---------------------------------------------------------------------------
READY_Q = np.zeros(29)
READY_Q[[0, 6]] = -0.312      # hip pitch
READY_Q[[3, 9]] = 0.669       # knee
READY_Q[[4, 10]] = -0.363     # ankle pitch


def test_extract_joints_reads_the_29_joints_by_name_from_the_root_prefixed_arrays(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    q, qd = _joint_pattern(80)
    assert clip.has_joints and clip.legs_note == ""
    assert clip.joint_names == tuple(SDK)
    assert np.allclose(clip.joint_pos, q) and np.allclose(clip.joint_vel, qd)
    assert clip.stats().n_joints == 29 and clip.stats().peak_joint_speed_rad_s == pytest.approx(np.abs(qd).max())
    # Sampling: linear on both arrays, velocity scaled by the playback speed.
    got = clip.joints(10.5, speed=0.5)
    assert np.allclose(got[0], 0.5 * (q[10] + q[11])) and np.allclose(got[1], 0.25 * (qd[10] + qd[11]))
    assert np.allclose(clip.joints(3.0, speed=2.0)[1], 2.0 * qd[3])


def test_a_clip_with_an_unusable_joint_channel_loads_feet_only_with_a_note(tmp_path):
    good_path, exp_pos, _ = _synthetic(tmp_path, joints=True)
    good = W.load_walk_clip(good_path)
    # 1. the legacy stub: 29 copies of the name "j" (the pre-2026-09-07 synthetic fixture)
    path, _, _ = _synthetic(tmp_path, joints=False)
    clip = W.load_walk_clip(path)
    assert not clip.has_joints and "distinct" in clip.legs_note and clip.joints(0.0) is None
    assert np.array_equal(clip.foot_pos, good.foot_pos), "the feet do not depend on the joint channel"
    assert "feet only" in clip.describe() and "legs: NONE" in clip.stats().describe(feet_slewed=False)
    # 2. a missing joint_vel is DERIVED, not refused (the bake does the same)
    path2, _, _ = _synthetic(tmp_path / "b", joints=True)
    with np.load(path2, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.pop("joint_vel")
    np.savez(tmp_path / "b" / "novel.npz", **arrays)
    clip2 = W.load_walk_clip(tmp_path / "b" / "novel.npz")
    q, qd = _joint_pattern(80)
    assert clip2.has_joints and np.allclose(clip2.joint_vel[5:-5], qd[5:-5], atol=0.02)
    # 3. a wrong-width joint_pos: feet only, said so
    arrays["joint_pos"] = arrays["joint_pos"][:, :20]
    np.savez(tmp_path / "b" / "narrow.npz", **arrays)
    clip3 = W.load_walk_clip(tmp_path / "b" / "narrow.npz")
    assert not clip3.has_joints and "neither (T, 29) nor (T, 36)" in clip3.legs_note
    assert np.array_equal(clip3.foot_pos, exp_pos.astype(np.float64)) or np.allclose(clip3.foot_pos, exp_pos, atol=1e-5)


def test_save_feet_clip_round_trips_the_joints_and_an_old_bundle_loads_feet_only(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    out = W.save_feet_clip(clip, tmp_path / "with_legs.npz", source_sha256="abc")
    back = W.load_walk_clip(out)
    assert back.has_joints and back.joint_names == clip.joint_names
    assert (back.joint_pos == clip.joint_pos).all() and (back.joint_vel == clip.joint_vel).all()
    # A bundle written before the legs existed has no joint arrays: it still loads, feet only.
    with np.load(out, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files if k not in ("joint_names", "joint_pos", "joint_vel", "joints_note")}
    np.savez(tmp_path / "old.npz", **arrays)
    old = W.load_walk_clip(tmp_path / "old.npz")
    assert not old.has_joints and "before the legs existed" in old.legs_note
    assert (old.foot_pos == clip.foot_pos).all()


def _player(clip, **kw):
    kw.setdefault("start_pos", READY_POS[3:])
    kw.setdefault("start_quat", READY_ORN[3:])
    kw.setdefault("step_dt", DT)
    kw.setdefault("loop", False)
    return W.WalkPlayer(clip, **kw)


def test_the_player_legs_follow_the_clip_by_name_and_scale_velocity_with_speed(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    # The player is asked for the legs in a SHUFFLED order relative to the file: by name, not index.
    rng = np.random.default_rng(7)
    perm = rng.permutation(29)
    names = tuple(SDK[i] for i in perm)
    player = _player(clip, start_joints=READY_Q[perm], joint_names=names, speed=0.5)
    assert player.legs_enabled and player.legs_note == ""
    q, qd = _joint_pattern(80)
    # Run through the lead-in, then compare the first clip-phase legs with the clip's own row.
    while True:
        _p, _q, ph = player.step()
        if ph == "clip":
            break
    names_out, q_out, qd_out = player.legs
    assert names_out == names
    f = player.clip_frame
    ref = clip.joints(f, speed=0.5)
    assert np.allclose(q_out, ref[0][perm]) and np.allclose(qd_out, ref[1][perm])
    # Half speed: the velocity is half the file's at the same phase (the targets are unchanged).
    full = clip.joints(f, speed=1.0)
    assert np.allclose(q_out, full[0][perm]) and np.allclose(qd_out, 0.5 * full[1][perm])
    assert np.abs(qd).max() > 0.1   # the pattern's velocity is not trivially zero
    # The feet are the ones a legs-less player publishes: the legs change nothing on the feet.
    twin = _player(clip, speed=0.5)
    feet_a, _, _ = _all_steps(_player(clip, start_joints=READY_Q[perm], joint_names=names, speed=0.5))
    feet_b, _, _ = _all_steps(twin)
    assert np.array_equal(feet_a, feet_b)


def test_the_player_legs_ramp_from_the_start_joints_and_back_with_a_consistent_velocity(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    player = _player(clip, start_joints=READY_Q, joint_names=SDK)
    n = player.lead_in_steps
    q0 = clip.joints(0.0)[0]
    assert player.lead_in_joint_gap_rad == pytest.approx(np.abs(q0 - READY_Q).max())
    _all = []
    while not player.done:
        _p, _q, ph = player.step()
        names, q, qd = player.legs
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(qd)) and names == tuple(SDK)
        _all.append((ph, q.copy(), qd.copy()))
    # Lead-in: the lerp from the start joints to frame 0, at the lerp's own rate.
    ph, q1, qd1 = _all[0]
    assert ph == "lead_in"
    assert np.allclose(q1, READY_Q + (q0 - READY_Q) / n)
    assert np.allclose(qd1, (q0 - READY_Q) / (n * DT))
    # The last lead-in step lands ON frame 0's joints; the first clip step is the clip's own.
    last_in = max(i for i, (p, _, _) in enumerate(_all) if p == "lead_in")
    assert np.allclose(_all[last_in][1], q0)
    # Lead-out ends on the start joints at REST — the value the low level's frozen clip then holds.
    ph_end, q_end, qd_end = _all[-1]
    assert ph_end == "lead_out" and np.allclose(q_end, READY_Q) and np.all(qd_end == 0.0)
    # After `done`, the legs stay there (a caller over-running by a step publishes a hold).
    player.step()
    assert np.allclose(player.legs[1], READY_Q) and np.all(player.legs[2] == 0.0)


def test_legs_are_off_without_start_joints_or_a_joint_channel_and_the_feet_are_unchanged(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    off = _player(clip)                                   # no start joints handed over
    assert not off.legs_enabled and off.legs is None and "no start joints" in off.legs_note
    feet_off, _, _ = _all_steps(off)
    feet_on, _, _ = _all_steps(_player(clip, start_joints=READY_Q, joint_names=SDK))
    assert np.array_equal(feet_off, feet_on)
    legacy, _, _ = _synthetic(tmp_path / "legacy", joints=False)
    p2 = _player(W.load_walk_clip(legacy), start_joints=READY_Q, joint_names=SDK)
    assert not p2.legs_enabled and p2.legs is None and "distinct" in p2.legs_note
    # A clip that names its joints differently from the player's order: refused by NAME, not an error.
    odd, _, _ = _synthetic(tmp_path / "odd", joints=True, joint_names=["x%d" % i for i in range(29)])
    p3 = _player(W.load_walk_clip(odd), start_joints=READY_Q, joint_names=SDK)
    assert not p3.legs_enabled and "names no joint" in p3.legs_note


def test_the_player_refuses_malformed_start_joints_by_value(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    with pytest.raises(ValueError, match="29 finite"):
        _player(clip, start_joints=np.zeros(12), joint_names=SDK)
    with pytest.raises(ValueError, match="29 finite"):
        _player(clip, start_joints=np.full(29, np.nan), joint_names=SDK)
    with pytest.raises(ValueError, match="29 distinct"):
        _player(clip, start_joints=READY_Q, joint_names=("a",) * 29)


def test_the_bundled_clips_lead_in_from_the_ready_legs_is_what_the_page_will_publish():
    """The real clip against the real ready pose: the numbers the robot sees on the first press."""
    clip = W.load_default_clip()
    from fuji.s1_policy.inference import ready_pose as RP

    anchor, why = RP.resolve_deploy_anchor(("left", "right", "head"))
    if anchor is None or anchor.dof is None:
        pytest.skip(f"no deploy anchor on this box: {why}")
    player = _player(clip, start_joints=anchor.dof, joint_names=SDK)
    assert player.legs_enabled
    # The knee opens from the 38 deg ready crouch toward the clip's frame 0 — a real gap, ramped.
    assert 0.3 < player.lead_in_joint_gap_rad < 1.5
    _p, _q, _ph = player.step()
    _names, q1, qd1 = player.legs
    assert np.abs(q1 - anchor.dof).max() < 0.05, "the first published legs are within 3 deg of the held pose"
    assert np.abs(qd1).max() * player.lead_in_steps * DT == pytest.approx(player.lead_in_joint_gap_rad, rel=1e-6)


# ---------------------------------------------------------------------------
# wiring of the legs into the page (source-level)
# ---------------------------------------------------------------------------
def test_the_legs_ride_only_the_walks_frames_and_the_heartbeat_repeats_them():
    block = _walk_block()
    run = block[block.index("def _run_walk()"):block.index("@b_walk.on_click")]
    # The walk constructs the player from the pose the low level HOLDS, by the SDK names.
    assert "start_joints=(anchor.dof if legs_on else None)" in run
    assert "joint_names=(G1_29DOF_SDK_DOF_NAMES if legs_on else None)" in run
    assert 'legs = player.legs if walk_ctl["legs"] else None' in run
    # A finished walk drops the field (the ramp landed on the ready legs = the frozen clip).
    assert 'last_wire["ref_joints"] = None' in run
    # The checkbox lives in the walk folder; the flag defaults ON.
    assert 'c_wlegs = server.gui.add_checkbox(' in block and '"Send the clip\'s LEGS (ref_joints)"' in block
    assert "set_defaults(walk_legs=True)" in _SRC and '"--no-walk-legs"' in _SRC
    # Outside the walk: `_drip` / the stream / Reset store the wire WITHOUT legs (default None) —
    # the only other `ref_joints=` spellings are the two seams' signatures and the heartbeat's replay.
    page_before = _SRC[_SRC.index("def _run_viser("):_SRC.index(_WALK_MARK)]
    assert page_before.count("ref_joints=") == 4, page_before.count("ref_joints=")   # _set_wire def, _send def+call, heartbeat
    assert "_send(frame, fc if on else None, ref_joints=rj)" in page_before
    assert "_set_wire(pub, fcmd)" in page_before and "_set_wire(wire.idle(), None)" in _SRC
    # STOP keeps the frozen legs with the frozen feet (one consistent body); it only drops F^cmd.
    stop = _SRC[_SRC.index("def _stop_now() -> str:"):_SRC.index("@b_auto.on_click")]   # the STOP body
    assert 'last_wire["fcmd"] = None' in stop and "ref_joints" not in stop
    # What the LOW LEVEL says about the legs is read off RecordTap, next to what the page sends.
    assert "rframe.ref_joints_live" in block and "g_wlegs.value = _legs_status()" in _SRC


def test_the_parser_defaults_send_the_legs_and_the_flag_turns_them_off():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_walk_legs_wiring", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_walk_legs_wiring"] = mod
    spec.loader.exec_module(mod)
    assert mod.build_parser().parse_args(["--checkpoint", "x"]).walk_legs is True
    assert mod.build_parser().parse_args(["--checkpoint", "x", "--no-walk-legs"]).walk_legs is False


# ---------------------------------------------------------------------------
# the LEG OVERRIDE: the same feet + legs under a RUNNING planner
# ---------------------------------------------------------------------------
def _moving_home(step_m=0.001):
    """A home that walks +x by ``step_m`` every time it is read — the planner's foot target drifting."""
    state = {"n": 0}

    def home():
        state["n"] += 1
        return READY_POS[3:] + np.array([[step_m * state["n"], 0.0, 0.0]] * 2), READY_ORN[3:]

    return home, state


def test_the_lead_out_pursues_a_moving_home_within_the_bound_and_lands_exactly(tmp_path):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    home, st = _moving_home(0.001)
    pl = W.WalkPlayer(clip, start_pos=READY_POS[3:], start_quat=READY_ORN[3:], step_dt=DT, loop=True,
                      start_joints=READY_Q, joint_names=SDK, home_provider=home)
    for _ in range(60):
        pl.step()                       # lead-in, then well into the clip: the feet are far from home
    pl.request_stop()
    assert pl.phase == "lead_out" and isinstance(pl._ramp, W._Pursuit)
    assert "pursuing" in pl.progress()
    feet = [pl.cur_pos.copy()]
    phases = []
    while not pl.done:
        p, _q, ph = pl.step()
        feet.append(p)
        phases.append(ph)
        assert len(feet) < 2000
    steps = _pool_steps(np.array(feet))
    assert steps.max() <= 0.25 * DT + 1e-9                      # the same 5 mm/tick bound as the ramps
    assert set(phases) == {"lead_out"}
    # Landed EXACTLY on the home as it was at the last step (it kept moving underneath the pursuit).
    p_home, q_home = READY_POS[3:] + np.array([[0.001 * st["n"], 0.0, 0.0]] * 2), READY_ORN[3:]
    assert np.allclose(pl.cur_pos, p_home, atol=0.0) and np.allclose(pl.cur_quat, q_home)
    # The legs landed on the START (ready) joints, at rest — what the low level's frozen clip holds.
    names, q, qd = pl.legs
    assert names == tuple(SDK) and np.allclose(q, READY_Q) and np.all(qd == 0.0)


def test_the_pursuit_closes_position_and_angle_together_and_the_last_step_is_inside_the_bound():
    p0 = READY_POS[3:]
    q0 = READY_ORN[3:]
    p1 = p0 + np.array([[0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])          # 20 mm on one foot
    q1 = np.tile(Rotation.from_euler("z", 30.0, degrees=True).as_quat(), (2, 1))   # 30 deg on both
    pur = W._Pursuit(p0, q0, lambda: (p1, q1), step_dt=DT, speed_m_s=0.25, rate_rad_s=np.deg2rad(45.0),
                     j0=np.zeros(29), j1=np.ones(29))
    pos, quat, k = [], [], 0
    while not pur.done:
        p, q = pur.step()
        pos.append(p)
        quat.append(q)
        k += 1
        assert k < 1000
    # 30 deg at 0.9 deg/tick is the binding constraint (34 ticks), not 20 mm at 5 mm/tick (4).
    assert 33 <= k <= 35
    assert np.allclose(pos[-1], p1) and np.allclose(np.abs(quat[-1]), np.abs(q1))
    steps = _pool_steps(np.array(pos))
    assert steps.max() <= 0.25 * DT + 1e-9
    angs = [W.quat_angle_rad(a, b).max() for a, b in zip(quat[:-1], quat[1:])]
    assert max(angs) <= np.deg2rad(45.0) * DT + 1e-6      # slerp round-off, not a rate violation
    # The joints closed the same fraction each step and landed with the feet, then are at rest.
    q, qd = pur.joints()
    assert np.allclose(q, 1.0)


def test_without_a_home_provider_the_lead_out_is_still_the_fixed_ramp(tmp_path):
    path, _, _ = _synthetic(tmp_path)
    clip = W.load_walk_clip(path)
    pl = _player(clip, loop=True)
    for _ in range(40):
        pl.step()
    pl.request_stop()
    assert isinstance(pl._ramp, W._Ramp) and pl.home_provider is None
    pl2 = _player(clip, loop=False)
    while pl2.phase != "lead_out":
        pl2.step()
    assert isinstance(pl2._ramp, W._Ramp)


def _override(tmp_path, *, legs=True, home=None):
    path, _, _ = _synthetic(tmp_path, joints=True)
    clip = W.load_walk_clip(path)
    pl = W.WalkPlayer(clip, start_pos=READY_POS[3:], start_quat=READY_ORN[3:], step_dt=DT, loop=True,
                      start_joints=READY_Q, joint_names=SDK,
                      home_provider=home or (lambda: (READY_POS[3:], READY_ORN[3:])))
    return W.LegOverride(pl, legs_enabled=lambda: legs), pl


def test_the_override_replaces_only_the_two_foot_slots_and_rides_the_legs(tmp_path):
    ov, pl = _override(tmp_path)
    for _ in range(30):
        ov.tick()
    policy = (READY_POS.reshape(-1) + 0.123, READY_ORN.reshape(-1))     # the planner's own frame
    (pos, orn), legs = ov.apply(policy, None)
    assert np.array_equal(pos[:9], policy[0][:9]) and np.array_equal(orn[:12], policy[1][:12])
    fp, fq, cur_legs = ov.current()
    assert np.array_equal(pos[9:], fp.reshape(-1)) and np.allclose(orn[12:], fq.reshape(-1))
    assert legs is not None and legs[0] == tuple(SDK) and np.array_equal(legs[1], cur_legs[1])
    assert ov.n_frames == 1 and ov.sending_legs
    # Legs unticked: the caller's own ref_joints pass through (None for every non-walk frame).
    ov2, _ = _override(tmp_path, legs=False)
    ov2.tick()
    (_p, _o), legs2 = ov2.apply(policy, "callers-own")
    assert legs2 == "callers-own" and not ov2.sending_legs


def test_the_override_holds_under_stop_with_the_legs_at_rest_but_a_hand_back_proceeds(tmp_path):
    ov, pl = _override(tmp_path)
    for _ in range(40):
        ov.tick()
    fp0, _fq0, legs0 = ov.current()
    assert np.any(legs0[2] != 0.0)                # mid-clip the reference legs MOVE
    for _ in range(10):
        assert ov.tick(hold=True) == "clip"
    fp1, _fq1, legs1 = ov.current()
    assert np.array_equal(fp1, fp0) and ov.n_held == 10 and ov.n_ticks == 40
    assert np.all(legs1[2] == 0.0) and np.array_equal(legs1[1], legs0[1])   # frozen, at rest
    ov.request_release()
    assert ov.stopping
    n = 0
    while not ov.done:
        ov.tick(hold=True)                          # STOP still latched: the hand-back proceeds anyway
        n += 1
        assert n < 1000
    fp2, fq2, legs2 = ov.current()
    assert np.allclose(fp2, READY_POS[3:]) and np.allclose(np.abs(fq2), np.abs(READY_ORN[3:]))
    assert np.allclose(legs2[1], READY_Q) and np.all(legs2[2] == 0.0)
    assert ov.tick() == "done" and "done" in ov.describe()


def test_the_frame_counter_advances_per_substituted_frame_so_the_page_can_wait_for_the_landing(tmp_path):
    """The page holds the hook until ONE more frame has been substituted after the landing; that
    signal is :attr:`LegOverride.n_frames`, so it must advance per `apply` and be readable at once."""
    ov, pl = _override(tmp_path)
    ov.tick()
    frame = (READY_POS.reshape(-1), READY_ORN.reshape(-1))
    assert ov.n_frames == 0
    for i in range(1, 4):
        ov.apply(frame, None)
        assert ov.n_frames == i
    ov.request_release()
    while not ov.done:
        ov.tick()
    n0 = ov.n_frames
    (pos, _orn), _legs = ov.apply(frame, None)          # the LANDED feet, published after `done`
    assert ov.n_frames == n0 + 1
    assert np.allclose(pos[9:].reshape(2, 3), READY_POS[3:], atol=0.0)


def test_the_override_hands_back_onto_the_planners_moving_feet(tmp_path):
    home, st = _moving_home(0.0005)
    ov, pl = _override(tmp_path, home=home)
    for _ in range(50):
        ov.tick()
    ov.request_release()
    while not ov.done:
        ov.tick()
    fp, _fq, _l = ov.current()
    assert np.allclose(fp, READY_POS[3:] + np.array([[0.0005 * st["n"], 0.0, 0.0]] * 2), atol=0.0)


# ---------------------------------------------------------------------------
# wiring of the override into the page (source-level)
# ---------------------------------------------------------------------------
def test_the_leg_override_is_one_block_after_the_walk_and_the_reviewed_page_has_only_the_hook_slot():
    block = _legs_block()
    page_before = _SRC[_SRC.index("def _run_viser("):_SRC.index(_WALK_MARK)]
    for name in ("legs_engaged", "legs_ctl", "_run_legs", "_legs_toggle", "b_legs", "g_legs", "_policy_feet",
                 "_armed_now", "LegOverride", "externally_driven"):
        assert name not in page_before, name
        assert name in block, name
    # The reviewed page contributes exactly one thing: the hook slot, run by `_send` before the gate.
    assert 'wire_hooks: dict = {"legs": None}' in page_before
    send = page_before[page_before.index("def _send(frame, fcmd, ref_joints=None) -> bool:"):
                       page_before.index("def _heartbeat() -> None:")]
    assert (send.index('hook = wire_hooks["legs"]') < send.index("frame, ref_joints = hook.apply(frame, ref_joints)")
            < send.index("ok, reason = wire.publish(s0, frame, fcmd, ref_joints=ref_joints)"))
    assert block.count("add_folder(") == 1
    assert _walk_block().count("add_folder(") == 1     # the walk folder is still the walk's alone


def test_the_override_is_a_substitution_not_a_publisher_and_does_not_own_the_page():
    run = _legs_block()
    run = run[run.index("def _run_legs()"):run.index("def _legs_toggle()")]
    # Never publishes, never records the wire, never advances the base, never plans: the planner does.
    for forbidden in ("_send(", "_set_wire(", "wire.commit(", "wire.plan(", "commit_role", "clutch.apply",
                      "busy.acquire", "dripping.set", "stop.clear()"):
        assert forbidden not in run, forbidden
    # Tell the clutch, THEN install the hook; uninstall the hook FIRST, then release the clutch.
    assert run.index("wire.externally_driven = _FOOT_SLOTS") < run.index('wire_hooks["legs"] = override')
    fin = run[run.index("finally:"):]
    assert fin.index('wire_hooks["legs"] = None') < fin.index("wire.externally_driven = frozenset()")
    assert "legs_engaged.clear()" in fin
    # The clip is the same player the 🚶 walk uses, handed the planner's LIVE feet as its home, and
    # the SAME live controls (Speed / Loop / Legs) — no second set.
    assert "home_provider=_policy_feet" in run
    assert 'player.set_speed(walk_ctl["speed"])' in run and 'player.set_loop(walk_ctl["loop"])' in run
    assert 'legs_enabled=lambda: bool(walk_ctl["legs"])' in run
    assert "start_joints=(anchor.dof if legs_on else None)" in run
    # STOP freezes the gait in place; a hand-back proceeds; a DISARM hands back on its own.
    assert "override.tick(hold=stop.is_set())" in run
    # The landed frame is PUBLISHED before the hook comes off (bounded), so the seam is 0 mm.
    assert "while override.n_frames == n0 and (time.monotonic() - t_land) < 0.25:" in run
    assert "override.request_release()" in run and "unarmed_since" in run
    assert "_armed_now() is False" in run and "refused — the low level is NOT ARMED" in run


def test_the_override_and_the_walk_refuse_each_other_and_reset_is_refused_while_engaged():
    walk = _walk_block()
    toggle = walk[walk.index("def _walk_toggle() -> str:"):walk.index("@b_walk.on_click")]
    assert 'if wire_hooks["legs"] is not None:' in toggle and "press l" in toggle
    legs = _legs_block()
    lt = legs[legs.index("def _legs_toggle() -> str:"):legs.index("@b_legs.on_click")]
    assert "if walk_running.is_set():" in lt and "press k" in lt
    assert 'legs_ctl["release_req"] = True' in lt
    assert 'threading.Thread(target=_run_legs, daemon=True, name="legs")' in lt
    # The main loop: Reset refused while engaged; Predict / Run deliberately NOT (that is the point).
    assert re.search(r'if legs_engaged\.is_set\(\) and flags\["reset"\]:', _SRC)
    assert not re.search(r'legs_engaged\.is_set\(\) and \(flags\["predict"\]', _SRC)
    # The walk's `legs` line reads the hook too, so "SENDING" is true during an override.
    assert 'sending = sending or (hook is not None and hook.sending_legs)' in walk
