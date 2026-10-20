"""Contract tests for :class:`flore.s1_policy.data.hl_dataset.HLDataset`.

Builds synthetic ``.zarr.zip`` files carrying the HL-v2 ``hl_*`` schema (no camera
hardware, no real session needed), then pins the properties the trained policy depends
on: the relative/absolute round-trip serving relies on, per-hand local frames, force
passthrough, mask semantics, and loud failure on any contract mismatch.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from flore.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HORIZON,
    ACTION_HZ,
    FORCE_HIST_LEN,
    FORCE_HIST_SPAN_S,
    get_layout,
)

zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# Synthetic HL zarr builder
# ---------------------------------------------------------------------------
def _write_hl_zarr(
    path,
    *,
    # 2 episodes x (one full chunk + 20 interior anchors). Sized from ACTION_HORIZON
    # rather than a literal 40: an episode SHORTER than ACTION_SPAN_S has no non-tail
    # steps at all, which silently turns every tail-vs-glitch assertion into a tautology.
    n_frames=2 * (ACTION_HORIZON + 20),
    roles=("left",),
    force_roles=("left",),
    gripper_roles=("left",),
    primary_role="left",
    attrs_override=None,
    omit_keys=(),
    n_episodes=2,
    seed=0,
    image_hw=8,
    store_horizon=None,
    n_cameras=1,
    invalid_force_history_roles=(),
):
    """Write a minimal but SCHEMA-FAITHFUL HL-v2 zarr.

    Poses are absolute and distinct per role/frame/step so a mis-slice or a
    role-crossing bug shows up as a value mismatch rather than a shape error.
    Images are a plain uint8 array (no JpegXl) — the dataset only divides by 255.

    ``store_horizon`` writes a chunk WIDER than this build trains on (and stamps the
    matching attrs at the same dt), which is the shipped-dataset case once
    ``FLORE_ACTION_HORIZON`` shortens the horizon: an H=50 file feeding an H=30 run. Default
    ``None`` == exactly ``ACTION_HORIZON``, so every existing fixture is unchanged.
    """
    rng = np.random.default_rng(seed)
    H, L = int(store_horizon or ACTION_HORIZON), FORCE_HIST_LEN
    store = zarr.MemoryStore()
    root = zarr.group(store=store)
    data = root.require_group("data")
    meta = root.require_group("meta")

    # cameraN_rgb for N in [0, n_cameras). Each camera draws its OWN random pixels, so a
    # dataset that double-feeds camera0 (or transposes the camera/history axes) shows up as
    # a value mismatch instead of a silently plausible tensor.
    for cam in range(int(n_cameras)):
        data.create_dataset(
            f"camera{cam}_rgb",
            data=rng.integers(0, 256, size=(n_frames, image_hw, image_hw, 3), dtype=np.uint8),
            overwrite=True,
        )
    # Camera spacing == ACTION_DT (the v3 grid: the camera publishes at ACTION_HZ, so the
    # obs and action grids coincide). Previously a literal 0.0841 s (the old ~11.9 Hz
    # camera): with ACTION_SPAN_S now 1.67 s, a 20-frame episode at 0.0841 s spans only
    # 1.6 s, i.e. SHORTER than one chunk — so every step of every anchor is tail padding
    # and the tail-vs-glitch tests could not distinguish the two. Deriving it keeps each
    # fixture episode long enough to hold a full chunk plus a real interior.
    data.create_dataset("episode_timestamps",
                        data=np.arange(n_frames, dtype=np.float64) * ACTION_DT,
                        overwrite=True)

    role_slots = {"left": 0.0, "right": 100.0, "head": 200.0}
    for role in roles:
        base = role_slots[role]
        # Absolute anchor pose on the camera grid (robot0_ = primary, robot1_ = other).
        prefix = "robot0" if role == primary_role else "robot1"
        if role == "head":
            pos_key, rot_key, grip_key = "head_pos", "head_rot_axis_angle", None
        else:
            pos_key = f"{prefix}_eef_pos"
            rot_key = f"{prefix}_eef_rot_axis_angle"
            grip_key = f"{prefix}_gripper_width"
        anchor_pos = base + np.stack(
            [np.arange(n_frames) * 0.01, np.arange(n_frames) * 0.02, np.arange(n_frames) * 0.03], axis=1
        )
        anchor_rot = np.stack([np.full(n_frames, 0.1), np.full(n_frames, 0.2), np.arange(n_frames) * 0.005], axis=1)
        data.create_dataset(pos_key, data=anchor_pos.astype(np.float32), overwrite=True)
        data.create_dataset(rot_key, data=anchor_rot.astype(np.float32), overwrite=True)
        if grip_key is not None and role in gripper_roles:
            # Metres; 0.095 is full open, so 0.0475 normalizes to 0.5.
            data.create_dataset(grip_key, data=np.full((n_frames, 1), 0.0475, np.float32), overwrite=True)

        # HL future pose targets: the anchor plus a per-step increment, so the
        # relative chunk is a known ramp. T0-INCLUSIVE grid (DECISION-T0): step k lands
        # at t0 + k*dt, so step 0 is ON the anchor (increment 0 -> identity in
        # anchor-relative space), exactly as the converter now emits.
        steps = np.arange(0, H)[None, :, None] * 0.001
        pose = np.zeros((n_frames, H, 6), dtype=np.float32)
        pose[:, :, :3] = anchor_pos[:, None, :] + steps
        pose[:, :, 3:] = anchor_rot[:, None, :] + steps * 0.5
        data.create_dataset(f"hl_{role}_action_pose", data=pose, overwrite=True)
        data.create_dataset(f"hl_{role}_action_valid", data=np.ones((n_frames, H), np.float32), overwrite=True)
        if role in gripper_roles:
            data.create_dataset(
                f"hl_{role}_action_gripper", data=np.full((n_frames, H), 0.0475, np.float32), overwrite=True
            )
            data.create_dataset(
                f"hl_{role}_action_gripper_valid", data=np.ones((n_frames, H), np.float32), overwrite=True
            )
        if role in force_roles:
            # Force values encode (role, frame, step, channel) so any cross-wiring shows.
            f = base + (
                np.arange(n_frames)[:, None, None] * 1.0
                + np.arange(H)[None, :, None] * 0.01
                + np.arange(6)[None, None, :] * 0.0001
            )
            data.create_dataset(f"hl_{role}_action_force", data=f.astype(np.float32), overwrite=True)
            data.create_dataset(
                f"hl_{role}_action_force_valid", data=np.ones((n_frames, H), np.float32), overwrite=True
            )
            fh = base + (
                np.arange(n_frames)[:, None, None] * 1.0
                + np.arange(L)[None, :, None] * 0.1
                + np.arange(6)[None, None, :] * 0.0001
            )
            data.create_dataset(f"hl_{role}_force_history", data=fh.astype(np.float32), overwrite=True)
            # `invalid_force_history_roles` marks a hand whose wrench window was never a
            # measurement (dead sensor / permanent gap) while the OTHER hand's stays valid —
            # the case a single shared validity bool cannot express.
            hv = 0.0 if role in invalid_force_history_roles else 1.0
            data.create_dataset(
                f"hl_{role}_force_history_valid", data=np.full((n_frames, 1), hv, np.float32),
                overwrite=True
            )

    ends = np.linspace(n_frames // n_episodes, n_frames, n_episodes).astype(np.int64)
    meta.create_dataset("episode_ends", data=ends, overwrite=True)

    a = {
        "schema": "umi_replay_buffer_v1",
        "robot0_primary_role": primary_role,
        "tracking_mode": "3pt",
        "hands_mode": "left",
        "hl_windows": True,
        # DERIVED from the layout, never literals: the loader's _check_file gates these
        # against the same constants, so a hardcoded 1.0/16.0 here makes every fixture
        # unloadable the moment the contract is re-pinned (which is exactly what
        # happened on the v3 30 Hz / 50-step move). A test that needs a MISMATCHED value
        # passes it via attrs_override, so the deliberate cases stay explicit.
        "hl_force_hist_len": FORCE_HIST_LEN,
        "hl_force_hist_span_s": FORCE_HIST_SPAN_S,
        # Stamped from the STORED width, so a store_horizon fixture stays self-consistent:
        # the span scales with the count at an unchanged dt, which is exactly the relation
        # _check_file verifies before it allows a prefix.
        "hl_action_horizon": H,
        "hl_action_span_s": H * ACTION_DT,
        "hl_action_hz": ACTION_HZ,
        "hl_action_first_step": "t0",  # t0-inclusive grid (DECISION-T0)
        "hl_pose_frame": "absolute_aligned_world",
        "hl_force_frame": "bota_sensor",
        "n_cameras": int(n_cameras),
    }
    a.update(attrs_override or {})
    for k, v in a.items():
        root.attrs[k] = v

    for k in omit_keys:
        if k in data:
            del data[k]

    with zarr.ZipStore(str(path), mode="w") as zs:
        zarr.copy_store(store, zs)
    return str(path)


@pytest.fixture()
def left_zarr(tmp_path):
    return _write_hl_zarr(tmp_path / "left.zarr.zip")


@pytest.fixture()
def three_zarr(tmp_path):
    return _write_hl_zarr(
        tmp_path / "three.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        attrs_override={"hands_mode": "both_head"},
    )


@pytest.fixture()
def dual_zarr(tmp_path):
    """The diffusion_umi_3tracker rig: 3 pucks, 2 cameras, 2 F/T sensors, 2 grippers."""
    return _write_hl_zarr(
        tmp_path / "dual.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left", "right"),
        gripper_roles=("left", "right"),
        n_cameras=2,
        attrs_override={"hands_mode": "both_head_dual"},
    )


def _ds(path, **kw):
    from flore.s1_policy.data.hl_dataset import HLDataset

    return HLDataset(path, **kw)


def test_source_fingerprint_is_full_file_sha256(left_zarr):
    from flore.s1_policy.data.hl_dataset import zip_content_fingerprint

    expected = hashlib.sha256()
    with open(left_zarr, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            expected.update(chunk)
    assert zip_content_fingerprint(left_zarr) == expected.hexdigest()


# ---------------------------------------------------------------------------
# Shapes and the observation contract
# ---------------------------------------------------------------------------
def test_sample_shapes_left_only(left_zarr):
    """The v3 DEFAULT sample: the observation is the single anchor frame t0, so there is
    NO leading To axis on image/state, and the force input is one bin."""
    ds = _ds(left_zarr)  # To=1 default (v3)
    s = ds[5]
    assert s["image"].shape == (3, 8, 8)
    assert s["state"].shape == (7,)  # wrt_start_rot6d + gripper
    assert s["state_valid"].shape == (7,)
    assert s["force_history"].shape == (FORCE_HIST_LEN, 6)  # (1, 6) on v3
    assert s["action_chunk"].shape == (ACTION_HORIZON, 16)  # (50, 16) on v3
    assert s["action_mask"].shape == (ACTION_HORIZON, 16)
    assert s["anchor_pos"].shape == (1, 3) and s["anchor_rot6d"].shape == (1, 6)
    assert s["action_chunk"].dtype == torch.float32
    assert s["action_mask"].dtype == torch.bool
    ds.close()


def test_single_frame_obs_collapses_the_history_axis(left_zarr):
    """To=1 (the v3 default) has no leading history axis, and its VALUES equal the t0
    frame (last slot) of a To=2 sample — so raising To only PREPENDS older frames.

    The To=2 dataset needs glitch_pre_s >= the obs look-back; on the v3 grid the default
    pre-window is one action step (0.033 s), which is shorter than 0.25 s. Passing it
    explicitly is what the guard in HLDataset demands, and exercising that path here is
    the point: a To>1 run must widen the glitch window to match its look-back."""
    ds1 = _ds(left_zarr, obs_history_len=1)
    ds2 = _ds(left_zarr, obs_history_len=2, glitch_pre_s=0.25)
    s1, s2 = ds1[5], ds2[5]
    assert s1["image"].shape == (3, 8, 8)
    assert s1["state"].shape == (7,)
    assert s1["state_valid"].shape == (7,)
    assert s2["image"].shape == (2, 3, 8, 8)
    assert s2["state"].shape == (2, 7)
    # To=1 image/state == the t0 (last) frame of the To=2 sample.
    assert torch.allclose(s1["image"], s2["image"][-1])
    assert torch.allclose(s1["state"], s2["state"][-1])
    ds1.close()
    ds2.close()


def test_collate_hl_batch_splits_history_into_pseudo_cameras():
    """collate_hl_batch is the seam between the (To,C,H,W) sample and the model's
    To-pseudo-camera global_cond. Pin: To frames -> To 'images' entries (oldest-first,
    matching images[t] to the sample's frame t), To matching img_masks, and (B,To,SD)
    state; To==1 -> length-1 lists."""
    from flore.s1_policy.training.trainer import collate_hl_batch

    def _sample(To):
        img = (
            torch.arange(To * 3 * 4 * 4, dtype=torch.float32).reshape(To, 3, 4, 4)
            if To > 1
            else torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        )
        return {
            "image": img,
            "state": torch.randn(To, 7) if To > 1 else torch.randn(7),
            "state_valid": torch.ones(To, 7, dtype=torch.bool) if To > 1 else torch.ones(7, dtype=torch.bool),
            "force_history": torch.randn(FORCE_HIST_LEN, 6),
            "force_history_valid": torch.tensor(True),
            "action_chunk": torch.randn(ACTION_HORIZON, 16),
            "action_mask": torch.ones(ACTION_HORIZON, 16, dtype=torch.bool),
        }

    # To=2: two pseudo-cameras, oldest-first, matching the per-frame images.
    samples = [_sample(2) for _ in range(3)]
    b = collate_hl_batch(samples)
    assert len(b["images"]) == 2 and len(b["img_masks"]) == 2
    assert all(im.shape == (3, 3, 4, 4) for im in b["images"])
    assert b["state"].shape == (3, 2, 7) and b["state_valid"].shape == (3, 2, 7)
    for t in range(2):  # images[t] must be sample frame t (oldest-first)
        assert torch.allclose(b["images"][t][0], samples[0]["image"][t])

    # To=1: length-1 lists, single-frame state.
    b1 = collate_hl_batch([_sample(1) for _ in range(3)])
    assert len(b1["images"]) == 1 and len(b1["img_masks"]) == 1
    assert b1["state"].shape == (3, 7)


def test_wrt_start_is_orientation_relative_to_episode_start(tmp_path):
    """DECISION-WRT0. state's wrt_start_rot6d is the keypoint's CURRENT orientation in
    its EPISODE-START frame (R_start^T R_now) — never the absolute orientation. At the
    episode's first frame it is the identity; deeper in it is the accumulated rotation.
    Frame-invariant to the world origin, which is what makes it OOD-safe to feed."""
    from flore.s1_policy.data.transforms import (
        IDENTITY_ROT6D,
        rot6d_relative,
        rot6d_to_rotmat,
        rotvec_to_rot6d,
    )

    p = _write_hl_zarr(tmp_path / "wrt.zarr.zip", n_frames=60, n_episodes=2)  # ends [30,60]
    ds = _ds(p, obs_history_len=1, drop_glitch_windows=False)
    lay = get_layout("left_only")
    wsl = lay.state_slots()["left"]["wrt_start_rot6d"]

    # Anchor at the FIRST frame of episode 1 (frame 30): wrt_start == identity.
    i0 = next(i for i, (_, f) in enumerate(ds._index) if f == 30)
    ws0 = ds[i0]["state"][wsl].numpy()
    assert np.allclose(ws0, IDENTITY_ROT6D, atol=1e-6), ws0

    # A later frame in episode 1: equals R_start^T R_now computed from the raw rot arrays,
    # and the reference is EP1's start (frame 30), not the file's frame 0 (per-episode).
    root, store = ds._open(p)
    rot = np.asarray(root["data"]["robot0_eef_rot_axis_angle"][:], dtype=np.float64)
    store.close()
    i1 = next(i for i, (_, f) in enumerate(ds._index) if f == 45)
    ws1 = ds[i1]["state"][wsl].numpy()
    expect = rot6d_relative(rotvec_to_rot6d(rot[45]), rotvec_to_rot6d(rot[30]))
    assert np.allclose(ws1, expect, atol=1e-6)
    # It is a proper rotation (rot6d lifts to an orthonormal matrix).
    R = rot6d_to_rotmat(ws1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    ds.close()


def test_wrt_start_routes_per_role_for_head_and_right_primary(tmp_path):
    """Every keypoint's wrt_start must come from ITS OWN rot array at the episode start,
    not the primary's. Covers the head (a body reference: wrt_start-only, no gripper) in
    a 3-role file, and a right-PRIMARY file where the right hand lives in robot0_*.
    The left-only value test above cannot catch a role-crossing bug on head/right."""
    from flore.s1_policy.data.transforms import rot6d_relative, rotvec_to_rot6d

    # 3-role file: assert head + right wrt_start equal rot6d_relative(cur, start) read
    # from each role's own rot key (head_rot_axis_angle / robot1_eef_rot_axis_angle).
    p = _write_hl_zarr(
        tmp_path / "3.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        attrs_override={"hands_mode": "both_head"},
        n_frames=60,
        n_episodes=1,
    )
    ds = _ds(p, mode="both_hands_head", obs_history_len=1, drop_glitch_windows=False)
    lay = get_layout("both_hands_head")
    ss = lay.state_slots()
    root, store = ds._open(p)
    rots = {
        "right": np.asarray(root["data"]["robot1_eef_rot_axis_angle"][:], np.float64),
        "head": np.asarray(root["data"]["head_rot_axis_angle"][:], np.float64),
    }
    store.close()
    i = next(k for k, (_, f) in enumerate(ds._index) if f == 40)
    s = ds[i]
    for role in ("right", "head"):
        got = s["state"][ss[role]["wrt_start_rot6d"]].numpy()
        expect = rot6d_relative(rotvec_to_rot6d(rots[role][40]), rotvec_to_rot6d(rots[role][0]))
        assert np.allclose(got, expect, atol=1e-6), role
    ds.close()

    # Right-PRIMARY file: the right hand is in robot0_*, so its wrt_start must read robot0_.
    pr = _write_hl_zarr(
        tmp_path / "rp.zarr.zip",
        roles=("right",),
        force_roles=("right",),
        gripper_roles=("right",),
        primary_role="right",
        n_frames=60,
        n_episodes=1,
        attrs_override={"hands_mode": "right", "robot0_primary_role": "right"},
    )
    dsr = _ds(pr, mode="right_only", obs_history_len=1, drop_glitch_windows=False)
    assert dsr._keys[0]["roles"]["right"]["anchor_rot"] == "robot0_eef_rot_axis_angle"
    rss = get_layout("right_only").state_slots()
    rootr, storer = dsr._open(pr)
    rr = np.asarray(rootr["data"]["robot0_eef_rot_axis_angle"][:], np.float64)
    storer.close()
    j = next(k for k, (_, f) in enumerate(dsr._index) if f == 40)
    gotr = dsr[j]["state"][rss["right"]["wrt_start_rot6d"]].numpy()
    expr = rot6d_relative(rotvec_to_rot6d(rr[40]), rotvec_to_rot6d(rr[0]))
    assert np.allclose(gotr, expr, atol=1e-6)
    dsr.close()


def test_wrt_start_reference_glitch_invalidates_the_channel(tmp_path):
    """A glitch at the EPISODE-START frame poisons the wrt_start reference for the whole
    episode; those anchors keep their pose sample but get state_valid=False on the
    wrt_start slot (and the gripper slot stays valid)."""
    p = _write_hl_zarr(tmp_path / "sr.zarr.zip", n_frames=60, n_episodes=2)  # ends [30,60]
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    pg = np.zeros((60, 1), np.float32)
    pg[30] = 1.0  # glitch at ep1's FIRST frame (the ref)
    root["data"].create_dataset("robot0_pose_glitch", data=pg, overwrite=True)
    store.close()

    lay = get_layout("left_only")
    wsl = lay.state_slots()["left"]["wrt_start_rot6d"]
    gcol = lay.state_slots()["left"]["gripper"]
    # Disable glitch-window drop so the ep1 anchors survive to be inspected.
    ds = _ds(p, obs_history_len=1, drop_glitch_windows=False)
    # A deep ep1 anchor (frame 45): its ref is frame 30 (glitched) -> wrt_start invalid.
    i1 = next(k for k, (_, f) in enumerate(ds._index) if f == 45)
    s1 = ds[i1]
    assert not s1["state_valid"][wsl].any(), "wrt_start must be flagged when ep-start ref is glitched"
    assert bool(s1["state_valid"][gcol]), "gripper slot is unaffected by a wrt_start-ref glitch"
    # An ep0 anchor (ref frame 0, clean) keeps wrt_start valid.
    i0 = next(k for k, (_, f) in enumerate(ds._index) if f == 15)
    assert ds[i0]["state_valid"][wsl].all()
    ds.close()


def test_obs_history_is_time_anchored_and_clamps_at_episode_start(tmp_path):
    """The To obs frames are nearest-camera-frame at t0 - k*dt, WITHIN the episode; the
    look-back never crosses into the previous demonstration and clamps to the episode's
    first frame (UMI start-pad).

    The expected frame index is DERIVED from the grid (dt == ACTION_DT), not a literal:
    a fixed 0.25 s was 3 frames on the old ~11.9 Hz grid and is 7.5 on the v3 30 Hz one,
    so a hardcoded '42' silently asserted the wrong thing rather than failing."""
    p = _write_hl_zarr(tmp_path / "th.zarr.zip", n_frames=60, n_episodes=2)  # ends [30,60]
    back = 3
    obs_dt_s = back * ACTION_DT
    ds = _ds(p, obs_history_len=2, obs_history_dt_s=obs_dt_s, drop_glitch_windows=False)
    # A deep-in-episode anchor: the older frame is `back` frames back, same episode.
    hf = ds._history_frames(0, 45)
    assert hf[-1] == 45 and hf[0] == 45 - back, hf.tolist()
    # The FIRST frame of episode 1 (frame 30): the look-back clamps to 30 (start-pad),
    # never reaching into episode 0.
    hf0 = ds._history_frames(0, 30)
    assert hf0.tolist() == [30, 30], hf0.tolist()
    ds.close()


def test_observation_contains_no_absolute_pose(left_zarr):
    """The v2 decision: absolute EE POSITION is NOT an input. state carries only
    wrt_start_rot6d (orientation relative to episode start — frame-invariant, OOD-safe)
    plus gripper; never the absolute world position that made the pre-v2 deploy
    conditioning OOD. The anchor (absolute pos+rot) is returned for SERVING only."""
    ds = _ds(left_zarr)
    s = ds[5]
    lay = get_layout("left_only")
    assert s["state"].shape[-1] == lay.state_dim  # wrt_start_rot6d + gripper, no pos
    assert "wrt_start_rot6d" in lay.state_slots()["left"]
    # The anchor is returned for SERVING, but is not part of any model input tensor.
    assert set(s) == {
        "image",
        "state",
        "state_valid",
        "force_history",
        "force_history_valid",
        "action_chunk",
        "action_mask",
        "anchor_pos",
        "anchor_rot6d",
    }
    ds.close()


def test_state_is_normalized_gripper_opening(left_zarr):
    ds = _ds(left_zarr)
    gcol = get_layout("left_only").state_slots()["left"]["gripper"]
    # 0.0475 m of a 0.095 m full-open range -> 0.5, at every obs-history frame.
    s = ds[3]["state"]  # (To, 7)
    assert torch.allclose(s[gcol], torch.tensor(0.5), atol=1e-6)
    ds.close()


def test_image_normalized_to_unit_range(left_zarr):
    ds = _ds(left_zarr)
    img = ds[7]["image"]
    assert img.min() >= -1.0 and img.max() <= 1.0
    ds.close()


def test_three_role_shapes_and_dims(three_zarr):
    ds = _ds(three_zarr, mode="both_hands_head")
    s = ds[5]
    lay = get_layout("both_hands_head")
    assert s["action_chunk"].shape == (ACTION_HORIZON, lay.action_dim) == (ACTION_HORIZON, 41)
    assert s["anchor_pos"].shape == (3, 3)
    # Only the left hand has an F/T sensor on this rig.
    assert s["force_history"].shape == (FORCE_HIST_LEN, 6)
    # State (To=1, the v3 single-frame obs): wrt_start_rot6d for L+R+head (6 each) +
    # one left gripper = 19. No leading To axis.
    assert s["state"].shape == (19,) == (lay.state_dim,)
    ds.close()


# ---------------------------------------------------------------------------
# Frames: the round-trip serving depends on
# ---------------------------------------------------------------------------
def test_relative_to_absolute_roundtrip_is_exact(left_zarr):
    """THE load-bearing invariant. Serving un-relativizes the model's chunk against
    the anchor; if this is not an exact inverse, every deployed target is wrong."""
    from flore.s1_policy.data.transforms import delta_ee_to_absolute, rotvec_to_rot6d

    ds = _ds(left_zarr)
    root, store = ds._open(left_zarr)
    gt = np.asarray(root["data"]["hl_left_action_pose"][:], dtype=np.float64)
    store.close()

    for idx in (0, 4, 17, 33):
        s = ds[idx]
        _, frame = ds._index[idx]
        ac = s["action_chunk"].numpy().astype(np.float64)
        abs_pos, abs_rot6d = delta_ee_to_absolute(
            ac[:, 0:3],
            ac[:, 3:9],
            s["anchor_pos"][0].numpy().astype(np.float64),
            s["anchor_rot6d"][0].numpy().astype(np.float64),
        )
        assert np.allclose(abs_pos, gt[frame, :, :3], atol=1e-5), idx
        assert np.allclose(abs_rot6d, rotvec_to_rot6d(gt[frame, :, 3:]), atol=1e-5), idx
    ds.close()


def test_each_hand_is_in_its_own_local_frame(three_zarr):
    """Per-hand local frames (NOT one shared left-hand frame): the left and right
    anchors are 100 m apart in world space, yet both relative chunks are small."""
    ds = _ds(three_zarr, mode="both_hands_head")
    s = ds[10]
    lay = get_layout("both_hands_head")
    world_gap = np.linalg.norm(s["anchor_pos"][1].numpy() - s["anchor_pos"][0].numpy())
    assert world_gap > 50.0
    for role in ("left", "right", "head"):
        rel = s["action_chunk"][:, lay.slots(role).pos].numpy()
        assert np.abs(rel).max() < 0.1, f"{role} not relativized to its OWN anchor"
    ds.close()


def test_first_step_is_the_identity_and_later_steps_are_not(left_zarr):
    """T0-INCLUSIVE grid (DECISION-T0, UMI-on-Air parity): step 0 lands ON the anchor,
    so its anchor-relative POSE is the identity (zero translation), while every later
    step carries real motion. This is the deliberate one-dt shift back from the pre-v2
    grid, which excluded t0 and started at t0+dt."""
    ds = _ds(left_zarr)
    ac = ds[5]["action_chunk"].numpy()
    lay = get_layout("left_only")
    assert np.abs(ac[0, lay.slots("left").pos]).max() < 1e-6, "step 0 pose must be identity"
    assert np.abs(ac[1, lay.slots("left").pos]).max() > 1e-6, "step 1 must carry motion"
    ds.close()


def test_force_and_gripper_pass_through_unrelativized(left_zarr):
    """Force is a wrench and gripper an opening — neither is an SE(3) pose, so the
    anchor-relative transform must not touch their columns."""
    ds = _ds(left_zarr)
    root, store = ds._open(left_zarr)
    gt_f = np.asarray(root["data"]["hl_left_action_force"][:], dtype=np.float64)
    store.close()
    lay = get_layout("left_only")
    s = ds[9]
    _, frame = ds._index[9]
    assert np.allclose(s["action_chunk"][:, lay.slots("left").force].numpy(), gt_f[frame], atol=1e-3)
    # Gripper is ABSOLUTE (normalized), identical at every step here.
    grip = s["action_chunk"][:, lay.slots("left").gripper].numpy()
    assert np.allclose(grip, 0.5, atol=1e-6)
    ds.close()


def test_force_history_is_read_from_the_right_frame(left_zarr):
    ds = _ds(left_zarr)
    root, store = ds._open(left_zarr)
    gt = np.asarray(root["data"]["hl_left_force_history"][:], dtype=np.float64)
    store.close()
    for idx in (0, 11, 30):
        _, frame = ds._index[idx]
        assert np.allclose(ds[idx]["force_history"].numpy(), gt[frame], atol=1e-3), idx
    ds.close()


def test_multi_hand_force_channels_do_not_cross(tmp_path, monkeypatch):
    """FORWARD-COMPATIBILITY. This rig has one F/T sensor, so no shipped mode declares
    two. When a second is added, the history channels must stay per-role in keypoint
    order — a crossed slice would train on the wrong hand's contact. Simulate that
    future rig by declaring both hands, and assert the channel routing."""
    from flore.s1_policy.data import layout as layout_mod

    modes = dict(layout_mod._V2_MODES)
    modes["both_hands"] = (("left", "right"), ("left", "right"), ("left", "right"))
    monkeypatch.setattr(layout_mod, "_V2_MODES", modes)

    lay = get_layout("both_hands")
    assert lay.force_roles == ("left", "right")
    # state = wrt_start_rot6d(6) per hand + gripper(1) per hand = 14 (both hands gripped).
    assert lay.force_hist_channels == 12 and lay.state_dim == 14
    assert lay.unmeasured_dims() == ()  # everything reserved is now measured

    p = _write_hl_zarr(
        tmp_path / "twoft.zarr.zip",
        roles=("left", "right"),
        force_roles=("left", "right"),
        gripper_roles=("left", "right"),
        attrs_override={"hands_mode": "both"},
    )
    ds = _ds(p, mode="both_hands")
    s = ds[6]
    assert s["force_history"].shape == (FORCE_HIST_LEN, 12)
    left_ch = s["force_history"][:, 0:6].numpy()
    right_ch = s["force_history"][:, 6:12].numpy()
    # The builder offsets the right hand's values by +100, so they must land in 6:12.
    assert right_ch.min() - left_ch.max() > 50.0
    # Both gripper observations are populated, in keypoint-order state slots (To=1).
    assert s["state"].shape == (lay.state_dim,) == (14,)
    sslots = lay.state_slots()
    for role in ("left", "right"):
        assert torch.allclose(s["state"][sslots[role]["gripper"]], torch.tensor(0.5), atol=1e-6)
    ds.close()


def test_reserved_but_unmeasured_columns_are_zero_and_masked(three_zarr):
    """On the REAL rig the right hand has no gripper and no F/T. Its reserved columns
    must be exactly zero AND masked False — never supervised toward a fabricated
    constant, which would teach the policy that the right hand always feels nothing."""
    ds = _ds(three_zarr, mode="both_hands_head")
    lay = get_layout("both_hands_head")
    s = ds[8]
    unmeasured = list(lay.unmeasured_dims())
    assert unmeasured, "expected the right hand's gripper+force to be unmeasured"
    assert np.allclose(s["action_chunk"][:, unmeasured].numpy(), 0.0)
    assert not s["action_mask"][:, unmeasured].any()
    # Everything measured IS supervised here (the synthetic file is all-valid).
    assert s["action_mask"][:, list(lay.measured_dims())].all()
    ds.close()


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def test_mask_is_all_true_when_everything_is_valid(left_zarr):
    ds = _ds(left_zarr)
    assert ds[5]["action_mask"].all()
    assert ds[5]["force_history_valid"].item() is True
    ds.close()


def test_invalid_step_masks_that_whole_step(tmp_path):
    p = _write_hl_zarr(tmp_path / "m.zarr.zip")
    # Invalidate the last 4 steps of frame 5 (a mid-episode anchor).
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    v = np.asarray(root["data"]["hl_left_action_valid"][:])
    v[5, -4:] = 0.0
    root["data"]["hl_left_action_valid"][:] = v
    store.close()

    # This test isolates STEP masking in __getitem__; disable the glitch-window sample
    # drop, which (correctly) treats these non-tail invalid steps as a glitch and would
    # remove frame 5 from the index entirely — that behaviour is covered separately.
    ds = _ds(p, drop_glitch_windows=False)
    idx = next(i for i, (_, f) in enumerate(ds._index) if f == 5)
    mask = ds[idx]["action_mask"].numpy()
    assert not mask[-4:].any(), "invalid steps must mask EVERY dim of the step"
    assert mask[:-4].all()
    ds.close()


def test_force_columns_need_both_step_and_force_validity(tmp_path):
    """Force validity is strictly stronger than step validity: a step inside the
    episode whose wrench bin is unmeasured/tare-dirty must mask ONLY the force
    columns, leaving pose and gripper supervised."""
    p = _write_hl_zarr(tmp_path / "f.zarr.zip")
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    fv = np.asarray(root["data"]["hl_left_action_force_valid"][:])
    fv[7, 3:6] = 0.0
    root["data"]["hl_left_action_force_valid"][:] = fv
    store.close()

    ds = _ds(p)
    lay = get_layout("left_only")
    idx = next(i for i, (_, f) in enumerate(ds._index) if f == 7)
    mask = ds[idx]["action_mask"].numpy()
    assert not mask[3:6, lay.slots("left").force].any()
    assert mask[3:6, lay.slots("left").pos].all()
    assert mask[3:6, lay.slots("left").gripper].all()
    ds.close()


def test_invalid_force_history_is_flagged_and_optionally_dropped(tmp_path):
    p = _write_hl_zarr(tmp_path / "h.zarr.zip", n_frames=40)
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    hv = np.asarray(root["data"]["hl_left_force_history_valid"][:])
    hv[:10] = 0.0
    root["data"]["hl_left_force_history_valid"][:] = hv
    store.close()

    # Default: kept, but flagged, so pose/gripper supervision is not thrown away.
    ds = _ds(p)
    assert len(ds) == 40
    idx = next(i for i, (_, f) in enumerate(ds._index) if f == 3)
    assert ds[idx]["force_history_valid"].item() is False
    ds.close()

    # Opt-in: dropped from the index entirely.
    ds2 = _ds(p, drop_invalid_force_history=True)
    assert len(ds2) == 30
    assert all(f >= 10 for _, f in ds2._index)
    ds2.close()


# ---------------------------------------------------------------------------
# Glitch-window sample dropping
# ---------------------------------------------------------------------------
def _episode_ends(path):
    """The file's own ``meta/episode_ends``, so a test never hardcodes the geometry."""
    store = zarr.ZipStore(path, mode="r")
    try:
        return [int(x) for x in zarr.group(store=store)["meta"]["episode_ends"][:]]
    finally:
        store.close()


def _set_valid(path, frame, steps, value=0.0, key="hl_left_action_valid"):
    """Set ``data[key][frame, steps] = value`` in an existing zarr.zip."""
    store = zarr.ZipStore(path, mode="a")
    root = zarr.group(store=store)
    v = np.asarray(root["data"][key][:])
    v[frame, steps] = value
    root["data"][key][:] = v
    store.close()


def test_clean_file_drops_nothing_including_tails(left_zarr):
    """A file whose only invalid steps are legitimate episode-tail padding must lose
    NO anchors — the tail is not a glitch. (_write_hl_zarr is all-valid, but the real
    converter always tail-invalidates the trailing ACTION_SPAN_S; simulate that here.)

    Episode geometry is READ from the fixture rather than assumed: the frame counts move
    with ACTION_HORIZON, and hardcoding them made this assert on tail-only frames whose
    steps were ALL padding — which is true for any filter and so tested nothing."""
    ep_end = _episode_ends(left_zarr)[0]  # first episode's exclusive end
    # For frame f, step k ends at (f + k)*dt, so steps with f + k >= ep_end are padding.
    # Invalidate exactly that suffix on the last 3 frames of the episode.
    for f in range(ep_end - 3, ep_end):
        _set_valid(left_zarr, f, slice(ep_end - f, None))
    ds_on = _ds(left_zarr, drop_glitch_windows=True)
    ds_off = _ds(left_zarr, drop_glitch_windows=False)
    assert len(ds_on) == len(ds_off), "tail padding must not be dropped as a glitch"
    ds_on.close()
    ds_off.close()


def test_midepisode_glitch_drops_the_pm1s_window(tmp_path):
    """A NON-tail invalid step (a glitch) drops every anchor whose history/action window
    straddles it, and ONLY those.

    All frame arithmetic is in units of the grid (dt == ACTION_DT): the drop reach is
    ACTION_SPAN_S of FUTURE window, so an anchor up to ACTION_HORIZON frames BEFORE the
    glitch still sees it. Sized so far-away anchors genuinely exist on both sides."""
    reach = ACTION_HORIZON            # frames of future window == ACTION_SPAN_S / dt
    n_frames = 4 * reach
    glitch_frame = 2 * reach
    p = _write_hl_zarr(tmp_path / "g.zarr.zip", n_frames=n_frames, n_episodes=1)
    _set_valid(p, glitch_frame, slice(0, 4))  # first few steps invalid, mid-episode

    ds_off = _ds(p, drop_glitch_windows=False)
    ds_on = _ds(p, drop_glitch_windows=True)
    assert len(ds_on) < len(ds_off)
    kept = {f for _, f in ds_on._index}
    # The glitch sits at ~glitch_frame*dt (its first few steps). Every anchor whose
    # future window reaches it must be gone...
    assert glitch_frame not in kept
    for f in range(glitch_frame - reach + 4, glitch_frame + 1):
        assert f not in kept, f
    # ...and anchors far outside the reach on BOTH sides must survive. The pre-window is
    # only FORCE_HIST_SPAN_S (one step on v3), so frames just after the glitch return.
    assert 0 in kept, "anchors long before the glitch must survive"
    assert n_frames - 1 in kept, "anchors long after the glitch must survive"
    ds_off.close()
    ds_on.close()


def test_pose_glitch_flag_frames_are_dropped(tmp_path):
    """An UNBRIDGED excursion is flagged in robot0_pose_glitch (not necessarily in
    action_valid). Those frames must also drop their surrounding windows."""
    p = _write_hl_zarr(tmp_path / "pg.zarr.zip", n_frames=60, n_episodes=1)
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    pg = np.zeros((60, 1), np.float32)
    pg[30] = 1.0
    root["data"].create_dataset("robot0_pose_glitch", data=pg, overwrite=True)
    store.close()

    ds_on = _ds(p, drop_glitch_windows=True)
    ds_off = _ds(p, drop_glitch_windows=False)
    assert len(ds_on) < len(ds_off)
    assert 30 not in {f for _, f in ds_on._index}
    ds_on.close()
    ds_off.close()


def test_per_role_pose_glitch_masks_are_unioned_for_the_roles_the_layout_reads(tmp_path):
    """Since 2026-09-11 the converter writes ``<role>_pose_glitch`` per emitted tracker
    role next to the primary-only ``robot0_pose_glitch``. The window filter must take the
    UNION over the roles the layout reads — and ONLY those: a ``right_pose_glitch`` frame in
    a left_only file is a role this layout never sees, so it must not drop anything."""
    p = _write_hl_zarr(tmp_path / "pr.zarr.zip", n_frames=60, n_episodes=1)
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    lg = np.zeros((60, 1), np.float32)
    lg[20] = 1.0                                     # a LEFT-role mask (no robot0 one)
    rg = np.zeros((60, 1), np.float32)
    rg[40] = 1.0                                     # a RIGHT-role mask: not read by left_only
    root["data"].create_dataset("left_pose_glitch", data=lg, overwrite=True)
    root["data"].create_dataset("right_pose_glitch", data=rg, overwrite=True)
    store.close()

    ds_on = _ds(p, drop_glitch_windows=True)
    ds_off = _ds(p, drop_glitch_windows=False)
    kept = {f for _, f in ds_on._index}
    assert 20 not in kept, "a left_pose_glitch frame must drop its anchor like robot0_pose_glitch does"
    assert 40 in kept, "right_pose_glitch is a role left_only never reads; it must not drop anchors"
    assert len(ds_on) < len(ds_off)
    ds_on.close()
    ds_off.close()


def test_glitch_drop_is_per_episode(tmp_path):
    """episode_timestamps is RELATIVE (restarts each episode). A glitch at local index k
    in episode 0 must not drop the anchor at the SAME local index k in episode 1.

    This is the sole guard on the per-episode-scan invariant, so the test must FAIL if
    that scoping is removed. That requires production-faithful timestamps: _write_hl_zarr
    writes a GLOBAL monotone episode_timestamps, under which the two frames are 2.5 s
    apart and survive regardless of scoping (the test would pass for the wrong reason).
    We overwrite it to restart at ~0 each episode, so the glitch frame and the ep1 frame
    share a relative time and only per-episode isolation keeps the ep1 frame."""
    p = _write_hl_zarr(tmp_path / "pe.zarr.zip", n_frames=60, n_episodes=2)  # ends [30,60]
    # Make episode_timestamps relative-per-episode (each episode restarts at 0), as
    # production zarrs are — 30 frames per episode at ~11.9 Hz.
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    rel = np.concatenate([np.arange(30), np.arange(30)]).astype(np.float64) * 0.0841
    root["data"]["episode_timestamps"][:] = rel
    store.close()

    local_k = 15
    _set_valid(p, local_k, slice(0, 4))  # glitch at local index 15 of ep0
    ds = _ds(p, drop_glitch_windows=True)
    kept = {f for _, f in ds._index}
    assert local_k not in kept, "the glitch anchor in ep0 is dropped"
    # Same LOCAL index in ep1 (frame 30 + 15) shares the glitch's relative time; only the
    # per-episode scan keeps it. A single-episode (unscoped) scan would drop it too.
    assert (30 + local_k) in kept, "a clean anchor at the same relative time in ep1 must survive"
    ds.close()


def test_glitch_drop_localizes_to_the_file_with_the_glitch(tmp_path):
    """Across multiple files, only the file carrying the glitch loses anchors."""
    a = _write_hl_zarr(tmp_path / "a.zarr.zip", n_frames=60, n_episodes=1, seed=1)
    b = _write_hl_zarr(tmp_path / "b.zarr.zip", n_frames=60, n_episodes=1, seed=2)
    _set_valid(a, 30, slice(0, 4))  # glitch in file a only
    ds = _ds([a, b], drop_glitch_windows=True)
    per_file = {0: 0, 1: 0}
    for fi, _ in ds._index:
        per_file[fi] += 1
    assert per_file[0] < 60, "file a (glitched) must lose anchors"
    assert per_file[1] == 60, "file b (clean) must keep every anchor"
    ds.close()


def test_whole_episode_wipeout_warns(tmp_path, caplog):
    """A demonstration losing ALL its anchors is louder than the per-file total; it must
    emit a warning. Use a 2-episode file so the dataset is still non-empty."""
    import logging as _logging

    p = _write_hl_zarr(tmp_path / "wipe.zarr.zip", n_frames=60, n_episodes=2)  # ends [30,60]
    # Flag EVERY frame of episode 0 via robot0_pose_glitch so its whole window drops.
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    pg = np.zeros((60, 1), np.float32)
    pg[0:30] = 1.0
    root["data"].create_dataset("robot0_pose_glitch", data=pg, overwrite=True)
    store.close()

    with caplog.at_level(_logging.WARNING, logger="flore.s1_policy.data.hl_dataset"):
        ds = _ds(p, drop_glitch_windows=True)
    assert any("lost ALL" in r.message for r in caplog.records), "expected a wipeout warning"
    # Episode 1 still contributes, so the dataset is not empty.
    assert len(ds) > 0
    assert all(f >= 30 for _, f in ds._index), "only ep1 anchors survive"
    ds.close()


def test_glitch_at_episode_boundary_does_not_leak(tmp_path):
    """A glitch on the LAST frame of ep0 must drop no ep1 anchor (boundary isolation)."""
    p = _write_hl_zarr(tmp_path / "bnd.zarr.zip", n_frames=60, n_episodes=2)  # ends [30,60]
    _set_valid(p, 29, slice(0, 4))  # last frame of ep0
    ds = _ds(p, drop_glitch_windows=True)
    kept = {f for _, f in ds._index}
    assert all(f in kept for f in range(30, 60)), "no ep1 anchor may be dropped by an ep0-tail glitch"
    ds.close()


def test_history_side_and_future_side_glitches_both_drop(tmp_path):
    """The drop window is symmetric: a glitch near the START of an episode removes
    LATER anchors (their history reaches back to it); a glitch near the END removes
    EARLIER anchors (their future reaches forward to it)."""
    # Near-start glitch: earlier-anchor FUTURE side.
    p1 = _write_hl_zarr(tmp_path / "hs1.zarr.zip", n_frames=60, n_episodes=1)
    _set_valid(p1, 3, slice(0, 4))
    ds1 = _ds(p1, drop_glitch_windows=True)
    kept1 = {f for _, f in ds1._index}
    assert 0 not in kept1, "an anchor whose future window reaches a near-start glitch must drop"
    assert 59 in kept1, "a far anchor must survive"
    ds1.close()
    # Near-end glitch: later-anchor HISTORY side.
    p2 = _write_hl_zarr(tmp_path / "hs2.zarr.zip", n_frames=60, n_episodes=1)
    _set_valid(p2, 56, slice(0, 4))
    ds2 = _ds(p2, drop_glitch_windows=True)
    kept2 = {f for _, f in ds2._index}
    assert 59 not in kept2, "an anchor whose history window reaches a near-end glitch must drop"
    assert 0 in kept2, "a far anchor must survive"
    ds2.close()


def test_force_only_invalidity_does_not_drop_the_pose_sample(tmp_path):
    """A masked FORCE step is a sensor/tare issue, not a pose glitch. It must NOT drop
    the (still pose-trainable) sample — only the per-step force mask handles it."""
    p = _write_hl_zarr(tmp_path / "fo.zarr.zip", n_frames=60, n_episodes=1)
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    fv = np.asarray(root["data"]["hl_left_action_force_valid"][:])
    fv[30, :] = 0.0  # whole future force window invalid, pose fine
    root["data"]["hl_left_action_force_valid"][:] = fv
    store.close()

    ds_on = _ds(p, drop_glitch_windows=True)
    ds_off = _ds(p, drop_glitch_windows=False)
    assert len(ds_on) == len(ds_off), "force-only invalidity must not drop a pose sample"
    ds_on.close()
    ds_off.close()


# ---------------------------------------------------------------------------
# Loud failure on contract mismatches
# ---------------------------------------------------------------------------
def test_missing_hl_windows_attr_raises_with_a_fix_hint(tmp_path):
    p = _write_hl_zarr(tmp_path / "nohl.zarr.zip", attrs_override={"hl_windows": False})
    with pytest.raises(ValueError, match="hl_windows"):
        _ds(p)


def test_grid_mismatch_raises(tmp_path):
    """A different history length or horizon mis-times every label; it must never be
    silently accepted just because the array shapes happen to broadcast."""
    p = _write_hl_zarr(tmp_path / "g1.zarr.zip", attrs_override={"hl_force_hist_len": 4})
    with pytest.raises(ValueError, match="hl_force_hist_len"):
        _ds(p)
    p2 = _write_hl_zarr(tmp_path / "g2.zarr.zip", attrs_override={"hl_action_horizon": 8})
    with pytest.raises(ValueError, match="hl_action_horizon"):
        _ds(p2)
    p3 = _write_hl_zarr(tmp_path / "g3.zarr.zip", attrs_override={"hl_action_span_s": 2.0})
    with pytest.raises(ValueError, match="hl_action_span_s"):
        _ds(p3)
    p4 = _write_hl_zarr(tmp_path / "g4.zarr.zip", attrs_override={"hl_action_hz": 20.0})
    with pytest.raises(ValueError, match="hl_action_hz"):
        _ds(p4)
    p5 = _write_hl_zarr(tmp_path / "g5.zarr.zip", attrs_override={"hl_force_hist_span_s": None})
    with pytest.raises(ValueError, match="hl_force_hist_span_s"):
        _ds(p5)
    p6 = _write_hl_zarr(tmp_path / "g6.zarr.zip", attrs_override={"hl_action_horizon": 16.9})
    with pytest.raises(ValueError, match="hl_action_horizon"):
        _ds(p6)
    p7 = _write_hl_zarr(tmp_path / "g7.zarr.zip", attrs_override={"hl_action_hz": np.nan})
    with pytest.raises(ValueError, match="hl_action_hz"):
        _ds(p7)


def test_longer_stored_horizon_is_prefix_truncated(tmp_path):
    """A file storing MORE steps than this build trains on is loaded as its own prefix.

    This is what lets one shipped dataset feed an H=50 and an H=30 run
    (``FLORE_ACTION_HORIZON``) with no re-convert, so the two runs differ only in the
    horizon. Asserted on VALUES, not just shape: the truncation must take the FIRST rows
    (step k stays at t0 + k*dt), because taking any other slice would silently re-phase
    every label while every shape still matched.
    """
    wide = ACTION_HORIZON + 17
    p = _write_hl_zarr(tmp_path / "wide.zarr.zip", store_horizon=wide)
    narrow = _write_hl_zarr(tmp_path / "exact.zarr.zip")
    ds_w, ds_n = _ds(p), _ds(narrow)
    s_w, s_n = ds_w[3], ds_n[3]
    assert s_w["action_chunk"].shape == s_n["action_chunk"].shape == (ACTION_HORIZON, ds_w.layout.action_dim)
    assert s_w["action_mask"].shape == s_n["action_mask"].shape
    # The two fixtures build the SAME ramp per step index, so a correct prefix makes the
    # wide file's chunk identical to the exactly-sized one's.
    torch.testing.assert_close(s_w["action_chunk"], s_n["action_chunk"])
    assert torch.equal(s_w["action_mask"], s_n["action_mask"])


def test_shorter_stored_horizon_is_refused(tmp_path):
    """The prefix relation runs ONE way: a file with fewer steps cannot be extended."""
    p = _write_hl_zarr(tmp_path / "short.zarr.zip", store_horizon=ACTION_HORIZON - 5)
    with pytest.raises(ValueError, match="at least"):
        _ds(p)


def test_prefix_requires_an_identical_step_dt(tmp_path):
    """A longer horizon at a DIFFERENT dt is not a superset — it is a rescaled grid.

    The dangerous shape this catches: ``hl_action_hz`` stamped 30 (so the exact rate check
    passes) while the file's own span/horizon says 20 Hz. Checking the count alone would
    accept it and train on labels stretched 1.5x, which is invisible in every other attr.
    """
    wide = ACTION_HORIZON + 10
    p = _write_hl_zarr(
        tmp_path / "slowgrid.zarr.zip",
        store_horizon=wide,
        attrs_override={"hl_action_span_s": wide * ACTION_DT * 1.5},
    )
    with pytest.raises(ValueError, match="step dt"):
        _ds(p)


def test_stale_action_grid_phase_is_refused(tmp_path):
    """DECISION-T0. A file from the pre-v2 t0-EXCLUDED grid carries identical
    horizon/span/hz attrs — the one-dt shift is invisible in the values, so the loader
    must key on hl_action_first_step. A non-'t0' value AND a null/absent value (a
    pre-provenance file that could be either grid) are both refused rather than silently
    trained on the wrong phase."""
    stale = _write_hl_zarr(tmp_path / "stale.zarr.zip", attrs_override={"hl_action_first_step": "t0+dt"})
    with pytest.raises(ValueError, match="hl_action_first_step"):
        _ds(stale)
    absent = _write_hl_zarr(tmp_path / "absent.zarr.zip", attrs_override={"hl_action_first_step": None})
    with pytest.raises(ValueError, match="hl_action_first_step"):
        _ds(absent)


def test_requesting_a_role_the_file_lacks_raises(left_zarr):
    """A left-only file must NOT quietly serve a both_hands request with zeros. Caught
    by the hands_mode reconciliation (the file says --hands left)."""
    with pytest.raises(ValueError, match="converted with --hands"):
        _ds(left_zarr, mode="both_hands")


def test_missing_role_arrays_raise_even_without_a_hands_mode_attr(tmp_path):
    """The array check is the backstop for a file with no hands_mode provenance (an
    early HL conversion): a mode whose arrays are absent must still be rejected."""
    p = _write_hl_zarr(tmp_path / "nomode.zarr.zip", attrs_override={"hands_mode": None})
    with pytest.raises(ValueError, match="hl_right_action_pose|absent"):
        _ds(p, mode="both_hands")


def test_a_wider_file_does_not_silently_train_a_narrower_mode(three_zarr):
    """The one-directional gap: a 3-role file loaded as left_only used to succeed and
    silently discard every hl_right_*/hl_head_* column — half the demonstration thrown
    away, with only a log line to show it."""
    with pytest.raises(ValueError, match="converted with --hands 'both_head'"):
        _ds(three_zarr, mode="left_only")


def test_force_frame_must_be_the_sensor_frame(tmp_path):
    """The force columns pass through un-relativized on the strength of this attr. If a
    converter starts rotating the wrench, masks stay valid and values stay plausible —
    invisible in training, visible only as a policy pushing the wrong way."""
    p = _write_hl_zarr(tmp_path / "wf.zarr.zip", attrs_override={"hl_force_frame": "world_aligned"})
    with pytest.raises(ValueError, match="hl_force_frame"):
        _ds(p)


def test_episode_forward_poses_carries_each_frames_own_measured_wrench(left_zarr):
    """The playback tail's ``wrench`` is step 0 of each frame's OWN action chunk.

    A viewer sweeps this tail to draw the recorded contact force frame by frame, so row i must
    be frame ``start + i``'s measurement — not the anchor's whole chunk re-read, which is the
    off-by-a-dimension that would make every playback arrow show the SAME force. The fixture
    encodes force as ``frame*1.0 + step*0.01 + channel*0.0001``, so the expected value pins
    frame, step AND channel simultaneously rather than just a shape.
    """
    ds = _ds(left_zarr)
    try:
        start = 5
        times_s, per_role = ds.episode_forward_poses(0, start)
        entry = per_role["left"]
        wrench = entry["wrench"]
        assert wrench.shape == (times_s.shape[0], 6)
        assert wrench.shape[0] == entry["position"].shape[0]  # same tail length as the poses
        frames = start + np.arange(wrench.shape[0])
        expected = frames[:, None] * 1.0 + 0.0 * 0.01 + np.arange(6)[None, :] * 0.0001
        np.testing.assert_allclose(wrench, expected, atol=2e-4)
        assert entry["wrench_valid"].shape == (wrench.shape[0],)
        assert entry["wrench_valid"].dtype == np.bool_
        assert entry["wrench_valid"].all()
    finally:
        ds.close()


def test_episode_forward_poses_reports_no_wrench_for_a_force_less_role(tmp_path):
    """A role with no F/T sensor yields ``None``, never a plausible zero-filled array.

    Zeros would be indistinguishable from "measured, no contact" and would draw a hidden arrow
    that silently claims the sensor exists — the same reserve-but-mask discipline the action
    columns follow.
    """
    p = _write_hl_zarr(
        tmp_path / "noft.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        attrs_override={"hands_mode": "both_head"},
    )
    ds = _ds(p, mode="both_hands_head")
    try:
        _, per_role = ds.episode_forward_poses(0, 3)
        assert per_role["left"]["wrench"] is not None
        for role in ("right", "head"):
            assert per_role[role]["wrench"] is None
            assert per_role[role]["wrench_valid"] is None
    finally:
        ds.close()


def test_gripper_validity_mask_is_required_not_optional(tmp_path):
    """Treating the mask as optional-with-a-fallback silently reintroduces the bug the
    converter's always-emit fix removed: a dead-encoder take is zero-filled, and falling
    back to pose validity would supervise 16 steps toward a fabricated 'fully closed'."""
    p = _write_hl_zarr(tmp_path / "nogv.zarr.zip", omit_keys=("hl_left_action_gripper_valid",))
    with pytest.raises(ValueError, match="hl_left_action_gripper_valid"):
        _ds(p)


def test_observed_gripper_key_is_required(tmp_path):
    """0.0 is a LEGAL gripper value (fully closed), so an absent observation key must
    raise rather than degrade to it."""
    p = _write_hl_zarr(tmp_path / "nogo.zarr.zip", omit_keys=("robot0_gripper_width",))
    with pytest.raises(ValueError, match="robot0_gripper_width"):
        _ds(p)


def test_dead_encoder_take_is_zero_filled_and_masked(tmp_path):
    """The converter's dead-encoder path: zeros + valid=0. The gripper column must be
    unsupervised, while pose stays trainable."""
    p = _write_hl_zarr(tmp_path / "dead.zarr.zip")
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    root["data"]["hl_left_action_gripper"][:] = np.zeros(root["data"]["hl_left_action_gripper"].shape, np.float32)
    root["data"]["hl_left_action_gripper_valid"][:] = np.zeros(root["data"]["hl_left_action_gripper_valid"].shape, np.float32)
    store.close()

    ds = _ds(p)
    lay = get_layout("left_only")
    s = ds[5]
    assert not s["action_mask"][:, lay.slots("left").gripper].any()
    assert s["action_mask"][:, lay.slots("left").pos].all()
    ds.close()


def test_missing_required_array_raises(tmp_path):
    p = _write_hl_zarr(tmp_path / "miss.zarr.zip", omit_keys=("hl_left_action_force_valid",))
    with pytest.raises(ValueError, match="hl_left_action_force_valid"):
        _ds(p)


def test_non_absolute_pose_frame_raises(tmp_path):
    p = _write_hl_zarr(tmp_path / "rel.zarr.zip", attrs_override={"hl_pose_frame": "anchor_relative"})
    with pytest.raises(ValueError, match="hl_pose_frame"):
        _ds(p)


def test_legacy32_mode_is_rejected(left_zarr):
    with pytest.raises(ValueError, match="legacy32"):
        _ds(left_zarr, mode="legacy32")


def test_unknown_mode_raises(left_zarr):
    with pytest.raises(ValueError, match="unknown layout mode"):
        _ds(left_zarr, mode="left")


def test_all_frames_dropped_raises_rather_than_yielding_an_empty_dataset(tmp_path):
    p = _write_hl_zarr(tmp_path / "empty.zarr.zip")
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    root["data"]["hl_left_force_history_valid"][:] = np.zeros(root["data"]["hl_left_force_history_valid"].shape, np.float32)
    store.close()
    with pytest.raises(ValueError, match="no usable anchors"):
        _ds(p, drop_invalid_force_history=True)


# ---------------------------------------------------------------------------
# Primary-role indirection and worker safety
# ---------------------------------------------------------------------------
def test_right_primary_file_routes_robot0_to_the_right_hand(tmp_path):
    """A right-primary session puts the right hand in robot0_*. The anchor must be
    read from robot0_ for 'right' — reading robot1_ would silently anchor the chunk
    on the wrong hand."""
    p = _write_hl_zarr(
        tmp_path / "rp.zarr.zip",
        roles=("right",),
        force_roles=("right",),
        gripper_roles=("right",),
        primary_role="right",
        attrs_override={"hands_mode": "right", "robot0_primary_role": "right"},
    )
    ds = _ds(p, mode="right_only")
    assert ds._keys[0]["roles"]["right"]["anchor_pos"] == "robot0_eef_pos"
    s = ds[4]
    assert s["action_chunk"].shape == (ACTION_HORIZON, 16)
    assert np.abs(s["action_chunk"][:, 0:3].numpy()).max() < 0.1
    ds.close()


def test_pickles_without_live_handles(left_zarr):
    """DataLoader workers pickle the dataset; a live ZipStore handle must not travel
    (a shared/inherited fd corrupts concurrent reads)."""
    import pickle

    ds = _ds(left_zarr)
    _ = ds[0]  # force a handle open
    assert ds._roots, "expected an open handle before pickling"
    state = pickle.loads(pickle.dumps(ds))
    assert state._roots == {} and state._stores == {} and state._handle_pid is None
    assert np.allclose(state[0]["action_chunk"].numpy(), ds[0]["action_chunk"].numpy())
    ds.close()


def test_lazy_worker_open_checks_file_identity_without_rehashing(left_zarr, monkeypatch):
    import pickle

    from flore.s1_policy.data import hl_dataset as dataset_module

    ds = pickle.loads(pickle.dumps(_ds(left_zarr)))

    def forbidden_rehash(_path):
        raise AssertionError("worker open must not hash the full ZIP again")

    monkeypatch.setattr(dataset_module, "zip_content_fingerprint", forbidden_rehash)
    assert ds[0]["action_chunk"].shape == (ACTION_HORIZON, 16)
    ds.close()


def test_multiple_files_are_concatenated(tmp_path):
    a = _write_hl_zarr(tmp_path / "a.zarr.zip", n_frames=20, seed=1)
    b = _write_hl_zarr(tmp_path / "b.zarr.zip", n_frames=30, seed=2)
    ds = _ds([a, b])
    assert len(ds) == 50
    assert {fi for fi, _ in ds._index} == {0, 1}
    ds.close()


def test_getitem_reads_only_single_frames(left_zarr, monkeypatch):
    """Guard against the pre-v2 whole-column read pattern (`data[key][:]`), which for
    an (T,16,6) array is megabytes per sample. Every zarr read in __getitem__ must be
    scalar-indexed."""
    ds = _ds(left_zarr)
    root = ds._root(0)
    arr = root["data"]["hl_left_action_pose"]
    seen = []
    orig = type(arr).__getitem__

    def spy(self, key):
        seen.append(key)
        return orig(self, key)

    monkeypatch.setattr(type(arr), "__getitem__", spy)
    _ = ds[5]
    assert seen, "expected at least one read"
    assert all(isinstance(k, (int, np.integer)) for k in seen), seen
    ds.close()


def test_gripper_width_reference_is_asserted_not_assumed(tmp_path):
    """REGRESSION. The converter records gripper_width_open_m specifically so the loader
    can ASSERT the scale it normalizes by. A per-session calibration with a different
    width_open_m (0.085 vs 0.095 — the recorder's own mock default) silently rescales
    every gripper label by 12% and moves UMI's 0.08 m grasp threshold from 0.94 to 0.84
    in normalized space, with nothing to show it."""
    from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M

    # A file converted against the SAME width loads.
    ok = _write_hl_zarr(tmp_path / "gw_ok.zarr.zip", attrs_override={"gripper_width_open_m": GRIPPER_WIDTH_REF_M})
    ds = _ds(ok)
    ds.close()

    # A mismatched one is refused, with the scale error in the message.
    bad = _write_hl_zarr(tmp_path / "gw_bad.zarr.zip", attrs_override={"gripper_width_open_m": 0.085})
    with pytest.raises(ValueError, match="full-open gripper width"):
        _ds(bad)
    nonfinite = _write_hl_zarr(tmp_path / "gw_nan.zarr.zip", attrs_override={"gripper_width_open_m": np.nan})
    with pytest.raises(ValueError, match="full-open gripper width"):
        _ds(nonfinite)

    # A pre-provenance file (no attr) still loads — absence is not a contradiction.
    none = _write_hl_zarr(tmp_path / "gw_none.zarr.zip", attrs_override={"gripper_width_open_m": None})
    ds2 = _ds(none)
    ds2.close()


def test_fabricated_gripper_observation_is_flagged_not_fed(tmp_path):
    """REGRESSION. The gripper TARGET was masked but the OBSERVATION was not.
    robot0_gripper_width has no mask of its own and the converter ZERO-FILLS it when the
    rig calibration is absent (a stderr warning only) — and 0.0 is a legal width (fully
    closed), so a fabricated observation is invisible in the values."""
    p = _write_hl_zarr(tmp_path / "nocal.zarr.zip", attrs_override={"gripper_width_open_m": None})
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    # Exactly the converter's no-calibration path: width zeroed, targets zeroed + invalid.
    root["data"]["robot0_gripper_width"][:] = np.zeros(root["data"]["robot0_gripper_width"].shape, np.float32)
    root["data"]["hl_left_action_gripper"][:] = np.zeros(root["data"]["hl_left_action_gripper"].shape, np.float32)
    root["data"]["hl_left_action_gripper_valid"][:] = np.zeros(root["data"]["hl_left_action_gripper_valid"].shape, np.float32)
    store.close()

    ds = _ds(p)
    lay = get_layout("left_only")
    gcol = lay.state_slots()["left"]["gripper"]
    s = ds[5]
    # The target stays masked (that already worked)...
    assert not s["action_mask"][:, lay.slots("left").gripper].any()
    # ...and the OBSERVATION's GRIPPER slot is now flagged (every obs frame) rather than
    # fed as a real 0.0 measurement. wrt_start_rot6d is still a real measurement, so only
    # the gripper column of state_valid is False.
    assert not s["state_valid"][gcol].any()
    assert torch.all(s["state"][gcol] == 0.0)
    assert s["state_valid"][lay.state_slots()["left"]["wrt_start_rot6d"]].all()
    # Pose is still trainable — the anchor is fine, only the gripper is missing.
    assert s["action_mask"][:, lay.slots("left").pos].all()
    ds.close()


def test_valid_gripper_observation_is_fed_and_flagged_true(left_zarr):
    ds = _ds(left_zarr)
    s = ds[5]
    gcol = get_layout("left_only").state_slots()["left"]["gripper"]
    assert s["state_valid"].all()
    assert torch.allclose(s["state"][gcol], torch.tensor(0.5), atol=1e-6)
    ds.close()


def test_gripper_history_validity_is_checked_per_frame(tmp_path):
    """Validity is resolved PER OBSERVATION FRAME, not once for the sample.

    Explicitly To=2 (with the matching wider glitch window): the property under test only
    exists when there is more than one obs frame, so it must not silently collapse to the
    v3 To=1 default. Sample 5's history frame is 5 - round(0.25/ACTION_DT) frames back."""
    p = _write_hl_zarr(tmp_path / "history_valid.zarr.zip")
    # Look-back expressed as a FRAME COUNT, so the anchor is always far enough into the
    # episode (a fixed 0.25 s is 7.5 frames at 30 Hz and would underflow frame 5).
    obs_dt_s = 2 * ACTION_DT
    anchor_frame = 5
    ds = _ds(p, obs_history_len=2, obs_history_dt_s=obs_dt_s, glitch_pre_s=obs_dt_s)
    # ASK the dataset which frame it resolves as the older obs slot rather than
    # recomputing it: the resolver picks the nearest frame at-or-before t0 - dt, and with
    # dt an exact multiple of the grid that target lands ON a frame boundary, where
    # floating-point decides between it and its predecessor. Recomputing here would just
    # duplicate that coin-flip and assert against the wrong frame half the time.
    hist_frame = int(ds._history_frames(0, anchor_frame)[0])
    ds.close()
    assert hist_frame < anchor_frame, "look-back must reach a strictly earlier frame"
    store = zarr.ZipStore(p, mode="a")
    root = zarr.group(store=store)
    valid = root["data"]["hl_left_action_gripper_valid"][:]
    valid[hist_frame] = 0.0  # the OLDER obs frame; sample 5's own t0 frame stays valid.
    root["data"]["hl_left_action_gripper_valid"][:] = valid
    store.close()

    ds = _ds(p, obs_history_len=2, obs_history_dt_s=obs_dt_s, glitch_pre_s=obs_dt_s)
    gcol = get_layout("left_only").state_slots()["left"]["gripper"]
    sample = ds[anchor_frame]
    assert sample["state_valid"][:, gcol].tolist() == [False, True]
    assert sample["state"][0, gcol] == 0.0
    assert sample["state"][1, gcol] == pytest.approx(0.5)
    ds.close()


def test_null_hands_mode_still_catches_extra_roles(tmp_path):
    """REGRESSION. hands_mode must tolerate None (pre-provenance files predate it), but a
    null used to make the mode gate VACUOUS: a file converted with --hl-windows and no
    --hands emits every tracker on disk and stamped None, so EVERY mode was accepted and
    silently discarded the rest. The role-set backstop catches it."""
    p = _write_hl_zarr(
        tmp_path / "nullmode.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        attrs_override={"hands_mode": None},
    )
    with pytest.raises(ValueError, match="would be silently discarded"):
        _ds(p, mode="left_only")
    # The mode that consumes every present role is fine.
    ds = _ds(p, mode="both_hands_head")
    ds.close()
