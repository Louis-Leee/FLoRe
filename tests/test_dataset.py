"""UMIDataset: builds tiny synthetic .zarr.zip ReplayBuffers (1pt and 3pt schemas)
in tmp_path and checks sample shapes, masks, and the absolute->relative anchor math."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from flore.s1_policy.data.umi_dataset import ACTION_DIM, UMIDataset

H, W = 224, 224
ACTION_HORIZON = 8


def _write_zarr(path: str, episodes: list[dict], primary_role: str | None = None) -> None:
    """Write a minimal UMI ReplayBuffer .zarr.zip from a list of per-episode dicts.

    ``primary_role`` stamps ``root.attrs["robot0_primary_role"]`` when given (the
    LEFT-canonical routing key); omitted => legacy zarr (loader defaults to 'right').
    """
    store = zarr.MemoryStore()
    root = zarr.group(store=store)
    data = root.require_group("data")
    meta = root.require_group("meta")

    keys = [k for k in episodes[0] if k != "camera0_rgb"]
    total_T = sum(ep["camera0_rgb"].shape[0] for ep in episodes)

    for key in keys:
        sample = episodes[0][key]
        data.create_dataset(
            key,
            shape=(total_T,) + sample.shape[1:],
            chunks=(min(total_T, 256),) + sample.shape[1:],
            dtype=sample.dtype,
            overwrite=True,
        )
    cam = data.create_dataset(
        "camera0_rgb", shape=(total_T, H, W, 3), chunks=(1, H, W, 3), dtype=np.uint8, overwrite=True
    )

    ends, cursor = [], 0
    for ep in episodes:
        T = ep["camera0_rgb"].shape[0]
        for key in keys:
            data[key][cursor : cursor + T] = ep[key]
        cam[cursor : cursor + T] = ep["camera0_rgb"]
        cursor += T
        ends.append(cursor)
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64), overwrite=True)
    if primary_role is not None:
        root.attrs["robot0_primary_role"] = primary_role

    with zarr.ZipStore(path, mode="w") as zip_store:
        zarr.copy_store(store, zip_store)


def _episode_1pt(T: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "robot0_eef_pos": rng.normal(0, 0.1, (T, 3)).astype(np.float32),
        "robot0_eef_rot_axis_angle": rng.normal(0, 0.3, (T, 3)).astype(np.float32),
        "robot0_gripper_width": rng.uniform(0, 0.095, (T, 1)).astype(np.float32),
        "episode_timestamps": np.arange(T, dtype=np.float64),
        "camera0_rgb": rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
    }


def _episode_3pt(T: int, seed: int = 0) -> dict:
    ep = _episode_1pt(T, seed)
    rng = np.random.default_rng(seed + 100)
    ep.update(
        {
            "robot1_eef_pos": rng.normal(0, 0.1, (T, 3)).astype(np.float32),
            "robot1_eef_rot_axis_angle": rng.normal(0, 0.3, (T, 3)).astype(np.float32),
            "robot1_gripper_width": rng.uniform(0, 0.095, (T, 1)).astype(np.float32),
            "head_pos": rng.normal(0, 0.1, (T, 3)).astype(np.float32),
            "head_rot_axis_angle": rng.normal(0, 0.3, (T, 3)).astype(np.float32),
        }
    )
    return ep


@pytest.fixture
def zarr_1pt(tmp_path):
    path = str(tmp_path / "s1pt.zarr.zip")
    _write_zarr(path, [_episode_1pt(20, 0), _episode_1pt(15, 1)])
    return path


@pytest.fixture
def zarr_3pt(tmp_path):
    path = str(tmp_path / "s3pt.zarr.zip")
    _write_zarr(path, [_episode_3pt(20, 2)])
    return path


def _episode_left_canonical_3pt(T: int, seed: int = 0) -> dict:
    """LEFT-canonical 3pt as the modern converter writes it: robot0 = LEFT hand
    (WITH gripper), robot1 = RIGHT hand (NO gripper stream — single physical gripper
    on the primary hand), head. Note the absence of robot1_gripper_width."""
    rng = np.random.default_rng(seed)
    return {
        "robot0_eef_pos": rng.normal(0, 0.1, (T, 3)).astype(np.float32),
        "robot0_eef_rot_axis_angle": rng.normal(0, 0.3, (T, 3)).astype(np.float32),
        "robot0_gripper_width": rng.uniform(0, 0.095, (T, 1)).astype(np.float32),
        "robot1_eef_pos": rng.normal(0, 0.1, (T, 3)).astype(np.float32),
        "robot1_eef_rot_axis_angle": rng.normal(0, 0.3, (T, 3)).astype(np.float32),
        "head_pos": rng.normal(0, 0.1, (T, 3)).astype(np.float32),
        "head_rot_axis_angle": rng.normal(0, 0.3, (T, 3)).astype(np.float32),
        "episode_timestamps": np.arange(T, dtype=np.float64),
        "camera0_rgb": rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
    }


@pytest.fixture
def zarr_left_1pt(tmp_path):
    """LEFT-canonical 1pt: robot0 = LEFT hand, attr routes it into the LEFT slot."""
    path = str(tmp_path / "left1pt.zarr.zip")
    _write_zarr(path, [_episode_1pt(20, 5)], primary_role="left")
    return path


@pytest.fixture
def zarr_left_3pt(tmp_path):
    path = str(tmp_path / "left3pt.zarr.zip")
    _write_zarr(path, [_episode_left_canonical_3pt(20, 6)], primary_role="left")
    return path


def test_left_canonical_1pt_routes_robot0_to_left_slot(zarr_left_1pt):
    """A LEFT-canonical 1pt zarr (robot0=left) must fill the LEFT block [0:10],
    NOT the RIGHT block — the opposite of a legacy 1pt zarr."""
    ds = UMIDataset(zarr_left_1pt, action_horizon=ACTION_HORIZON)
    mask = ds[0]["action_dim_mask"].numpy()
    expected = np.zeros(ACTION_DIM, dtype=bool)
    expected[0:10] = True  # left block real
    np.testing.assert_array_equal(mask, expected)
    # action[0] relative-identity on the LEFT block.
    a0 = ds[0]["action_chunk"][0].numpy()
    np.testing.assert_allclose(a0[0:3], 0.0, atol=1e-5)
    np.testing.assert_allclose(a0[3:9], [1, 0, 0, 0, 1, 0], atol=1e-5)


def test_left_canonical_3pt_slots_and_absent_right_gripper(zarr_left_3pt):
    """LEFT-canonical 3pt: left [0:10] real (with gripper), right [10:19] real but
    right-gripper slot 19 FALSE (no physical right gripper), head [20:29] real."""
    ds = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON)
    mask = ds[0]["action_dim_mask"].numpy()
    expected = np.zeros(ACTION_DIM, dtype=bool)
    expected[0:10] = True    # left pos+rot+gripper
    expected[10:19] = True   # right pos+rot (no gripper)
    expected[20:29] = True   # head pos+rot
    np.testing.assert_array_equal(mask, expected)
    assert not mask[19], "right-gripper slot must be masked False (no right gripper)"
    assert mask[9], "left-gripper slot must be real"


def test_len_equals_total_frames(zarr_1pt):
    ds = UMIDataset(zarr_1pt, action_horizon=ACTION_HORIZON)
    assert len(ds) == 35  # 20 + 15


def test_1pt_sample_shapes_and_masks(zarr_1pt):
    ds = UMIDataset(zarr_1pt, action_horizon=ACTION_HORIZON)
    s = ds[0]
    assert s["image"].shape == (3, H, W)
    assert s["state"].shape == (ACTION_DIM,)
    assert s["action_chunk"].shape == (ACTION_HORIZON, ACTION_DIM)
    assert s["action_is_pad"].shape == (ACTION_HORIZON,)
    assert s["action_dim_mask"].shape == (ACTION_DIM,)
    assert s["action_dim_mask"].dtype == torch.bool

    # 1pt: only the RIGHT block [10:20] is real; left/head + final pad are False.
    mask = s["action_dim_mask"].numpy()
    expected = np.zeros(ACTION_DIM, dtype=bool)
    expected[10:20] = True
    np.testing.assert_array_equal(mask, expected)


def test_3pt_sample_masks(zarr_3pt):
    ds = UMIDataset(zarr_3pt, action_horizon=ACTION_HORIZON)
    s = ds[0]
    mask = s["action_dim_mask"].numpy()
    expected = np.ones(ACTION_DIM, dtype=bool)
    expected[29:] = False  # 3 zero-pad dims
    np.testing.assert_array_equal(mask, expected)


def test_action_is_pad_flags_episode_end(zarr_1pt):
    ds = UMIDataset(zarr_1pt, action_horizon=ACTION_HORIZON)
    # Last frame of episode 0 is index 19; anchor there -> all future frames clamp/pad.
    s = ds[19]
    pad = s["action_is_pad"].numpy()
    # action[0] is the anchor frame itself (not pad); the rest run past ep end.
    assert not pad[0]
    assert pad[1:].all()


def test_first_action_row_is_near_identity(zarr_1pt):
    """action[0] is the anchor pose relative to itself: ~zero pos, identity rot6d."""
    ds = UMIDataset(zarr_1pt, action_horizon=ACTION_HORIZON)
    s = ds[0]
    a0 = s["action_chunk"][0].numpy()
    # right block: pos [10:13] ~ 0, rot6d [13:19] ~ identity [1,0,0,0,1,0]
    np.testing.assert_allclose(a0[10:13], 0.0, atol=1e-5)
    np.testing.assert_allclose(a0[13:19], [1, 0, 0, 0, 1, 0], atol=1e-5)


def test_relative_action_roundtrips_to_absolute(zarr_1pt):
    """The stored relative chunk + anchor state recovers the absolute poses (GCL SE(3))."""
    from flore.s1_policy.data.transforms import delta_ee_to_absolute, rotvec_to_rot6d

    ds = UMIDataset(zarr_1pt, action_horizon=ACTION_HORIZON)
    s = ds[0]
    state = s["state"].numpy()
    chunk = s["action_chunk"].numpy()
    ref_pos, ref_rot6d = state[10:13], state[13:19]

    abs_pos, abs_rot6d = delta_ee_to_absolute(chunk[:, 10:13], chunk[:, 13:19], ref_pos, ref_rot6d)

    # Recompute the on-disk absolute right-hand poses for frames [0, H).
    root = zarr.group(store=zarr.ZipStore(zarr_1pt, mode="r"))
    pos = np.asarray(root["data"]["robot0_eef_pos"][:ACTION_HORIZON], dtype=np.float64)
    rot6d = rotvec_to_rot6d(np.asarray(root["data"]["robot0_eef_rot_axis_angle"][:ACTION_HORIZON], dtype=np.float64))

    np.testing.assert_allclose(abs_pos, pos, atol=1e-5)
    np.testing.assert_allclose(abs_rot6d, rot6d, atol=1e-5)


def test_multi_file_dataset(zarr_1pt, zarr_3pt):
    ds = UMIDataset([zarr_1pt, zarr_3pt], action_horizon=ACTION_HORIZON)
    assert len(ds) == 35 + 20
    # A sample from the 3pt file has full masks; from the 1pt file, right-only.
    last = ds[len(ds) - 1]
    assert last["action_dim_mask"].numpy()[0:9].all()  # left block real in 3pt


# ---------------------------------------------------------------------------
# Reference-frame ablation (per_keypoint | left | right | head | world)
# ---------------------------------------------------------------------------
def test_ref_frame_default_is_byte_identical(zarr_left_3pt):
    """ref_frame='per_keypoint' (default) must produce EXACTLY today's tensors."""
    a = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON)[3]
    b = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, ref_frame="per_keypoint")[3]
    for k in ("image", "state", "action_chunk", "action_is_pad", "action_dim_mask"):
        np.testing.assert_array_equal(a[k].numpy(), b[k].numpy())


def test_ref_frame_left_makes_left_state_identity(zarr_left_3pt):
    """Under ref_frame='left', the LEFT keypoint's own state is exact identity
    (pos 0, rot6d [1,0,0,0,1,0]); the RIGHT keypoint's state is NOT identity
    (it encodes right-in-left-frame configuration)."""
    ds = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, ref_frame="left")
    s = ds[3]["state"].numpy()
    np.testing.assert_allclose(s[0:3], 0.0, atol=1e-6)              # left pos
    np.testing.assert_allclose(s[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)  # left rot6d
    assert not np.allclose(s[10:13], 0.0), "right pos must encode right-in-left frame"


def test_ref_frame_left_action_reference_row_identity(zarr_left_3pt):
    """Under a shared frame, action[0] is ~identity ONLY for the reference keypoint
    (left); non-reference keypoints' action[0] equals their (non-identity) state."""
    ds = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, ref_frame="left")
    s = ds[3]
    a0 = s["action_chunk"][0].numpy()
    state = s["state"].numpy()
    # Left (reference) row-0 is ~identity.
    np.testing.assert_allclose(a0[0:3], 0.0, atol=1e-5)
    np.testing.assert_allclose(a0[3:9], [1, 0, 0, 0, 1, 0], atol=1e-5)
    # Right row-0 == right state (both are right-in-left-frame at the anchor), NOT identity.
    np.testing.assert_allclose(a0[10:13], state[10:13], atol=1e-5)
    np.testing.assert_allclose(a0[13:19], state[13:19], atol=1e-5)
    assert not np.allclose(a0[13:19], [1, 0, 0, 0, 1, 0], atol=1e-3)


@pytest.mark.parametrize("ref_frame", ["left", "right", "head", "world"])
def test_ref_frame_roundtrips_to_absolute(zarr_left_3pt, ref_frame):
    """The shared-frame chunk + the base recovered from the raw absolute anchor state
    reconstructs the original absolute per-keypoint trajectory (SE(3) round-trip)."""
    from flore.s1_policy.data.transforms import (
        delta_ee_to_absolute, resolve_base, rotvec_to_rot6d,
    )

    # Absolute anchor state (per_keypoint state IS absolute) for base resolution.
    abs_state = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON,
                           ref_frame="per_keypoint")[0]["state"].numpy()
    ds = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, ref_frame=ref_frame)
    chunk = ds[0]["action_chunk"].numpy()

    if ref_frame == "world":
        base_pos, base_rot6d = np.zeros(3), np.array([1.0, 0, 0, 0, 1, 0])
    else:
        base_pos, base_rot6d = resolve_base(abs_state, ref_frame)

    root = zarr.group(store=zarr.ZipStore(zarr_left_3pt, mode="r"))
    specs = {  # zarr key prefix -> 32-dim (pos_slice, rot_slice)
        "robot0": (slice(0, 3), slice(3, 9)),   # left
        "robot1": (slice(10, 13), slice(13, 19)),  # right
    }
    for prefix, (pos_sl, rot_sl) in specs.items():
        abs_pos, abs_rot6d = delta_ee_to_absolute(
            chunk[:, pos_sl], chunk[:, rot_sl], base_pos, base_rot6d)
        want_pos = np.asarray(root["data"][f"{prefix}_eef_pos"][:ACTION_HORIZON], dtype=np.float64)
        want_rot = rotvec_to_rot6d(np.asarray(
            root["data"][f"{prefix}_eef_rot_axis_angle"][:ACTION_HORIZON], dtype=np.float64))
        np.testing.assert_allclose(abs_pos, want_pos, atol=1e-5)
        np.testing.assert_allclose(abs_rot6d, want_rot, atol=1e-5)


def test_ref_frame_left_missing_keypoint_raises(zarr_left_1pt):
    """ref_frame='right' on a 1pt (left-only) zarr must fail loudly, not read a zero slot."""
    with pytest.raises(ValueError, match="absent"):
        UMIDataset(zarr_left_1pt, action_horizon=ACTION_HORIZON, ref_frame="right")


# ---------------------------------------------------------------------------
# Observation-history window (To frames ending at the anchor)
# ---------------------------------------------------------------------------
def test_history_shapes(zarr_left_3pt):
    """To>1 yields (To,C,H,W) image and (To,32) state; action chunk is unchanged."""
    To = 3
    ds = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, obs_history_len=To,
                    obs_history_stride=2)
    s = ds[10]
    assert s["image"].shape == (To, 3, H, W)
    assert s["state"].shape == (To, ACTION_DIM)
    assert s["action_chunk"].shape == (ACTION_HORIZON, ACTION_DIM)
    # Last history frame == the anchor: its state row matches the To=1 anchor state.
    s1 = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON)[10]
    np.testing.assert_allclose(s["state"].numpy()[-1], s1["state"].numpy(), atol=1e-6)


def test_history_clamps_at_episode_start(zarr_left_3pt):
    """Near the episode start the window edge-clamps to ep_start (repeats the earliest
    frame) — never crossing into a previous episode. anchor=0 => all To frames equal."""
    To = 4
    ds = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, obs_history_len=To,
                    obs_history_stride=1)
    s = ds[0]  # anchor at the very first frame of the episode
    st = s["state"].numpy()
    for t in range(To):
        np.testing.assert_allclose(st[t], st[0], atol=1e-6)  # all clamped to frame 0


def test_history_default_len_one_is_single_frame(zarr_left_3pt):
    """obs_history_len=1 (default) keeps the (C,H,W)/(32,) single-frame contract."""
    s = UMIDataset(zarr_left_3pt, action_horizon=ACTION_HORIZON, obs_history_len=1)[5]
    assert s["image"].shape == (3, H, W)
    assert s["state"].shape == (ACTION_DIM,)
