"""Contract tests for ``diffusion_umi_wholebody`` — the WHOLE-BODY sibling of
``diffusion_umi_3tracker`` (layout ``wholebody``).

Same rig, same two cameras, same two Botas, same two grippers. What changes is the ACTION: 29
absolute G1 joint angles in place of three keypoint poses, produced once at data-preparation
time by motion completion (``UMI_collection/motion_completion/mc_to_hl_wholebody.py``) instead
of at 30 Hz on the robot.

Everything pinned here fails SILENTLY otherwise — that is the selection criterion:

* the joints are ABSOLUTE, and an accidental anchor-relative transform still produces a
  plausible 43-dim chunk (it would just command the wrong posture);
* ``pose_dims()`` is empty for this layout, and ``mask[..., []].all(-1)`` is ``True`` while a
  norm over 0 columns is exactly ``0.0`` — so the naive metric arithmetic reports a PERFECT
  0.00 cm position error for a policy with no position columns at all;
* the chest owns no keypoint block here but IS the completion's root conditioning, so a scan
  over ``layout.roles`` silently stops dropping chest-glitch windows;
* the wholebody file is a strict superset of the dual one, so nothing about its SHAPE says
  whether a loader picked up the extra arrays or ignored them;
* any permutation of 29 joint angles is a valid robot pose, so a leg/arm swap trains happily.

The 3-tracker preset must stay untouched, so the no-regression direction is asserted too: the
same augmented file loaded as ``both_hands_head_dual`` must be tensor-for-tensor identical to
the un-augmented file.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from flore.s1_policy.data.layout import (
    ACTION_HORIZON,
    FORCE_DIM,
    FORCE_HIST_LEN,
    JOINT_DIM,
    WB_JOINT_NAMES,
    WB_JOINT_TARGETS,
    get_layout,
    hands_flag,
    tracker_roles_required,
)
from flore.s1_policy.training.config import HLDataConfig, TrainConfig

# Reuse the synthetic-zarr BUILDER rather than a second copy of it, for the reason
# test_diffusion_umi_3tracker gives: a drifting second schema writer is how a loader test
# starts passing against data no converter produces.
from tests.test_hl_dataset import _ds, _write_hl_zarr

MODE = "wholebody"
DUAL_MODE = "both_hands_head_dual"
CAMS = ("camera0_rgb", "camera1_rgb")

# The EXPORTER, imported from the submodule rather than reimplemented, so the arrays this test
# feeds the loader are the ones the real tool writes. Same pattern (and the same
# sys.path-into-the-submodule trick) as tests/test_hl_contract.py, which imports the converter.
_UMI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "third_party", "UMI_collection")
for _sub in ("motion_completion", "replay", "processing", "viz"):
    _p = os.path.join(_UMI, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
wb_export = pytest.importorskip(
    "mc_to_hl_wholebody",
    reason="third_party/UMI_collection is not checked out (git submodule update --init)")


# ---------------------------------------------------------------------------
# Fixtures: a dual zarr, and the same file with wb_* arrays added by the exporter
# ---------------------------------------------------------------------------
def _synthetic_qpos(n: int, seed: int) -> np.ndarray:
    """``(n, 36)`` fake completion output: root pose + 29 joints, distinct per frame/joint.

    Every value encodes ``(frame, joint)`` so a mis-slice, a transposed axis or a
    role-crossing bug shows up as a value mismatch rather than a shape error — the same
    discipline ``_write_hl_zarr`` applies to the pose/force arrays.
    """
    rng = np.random.default_rng(seed)
    qpos = np.zeros((n, 36), dtype=np.float64)
    qpos[:, :3] = rng.normal(scale=0.01, size=(n, 3))          # pelvis xyz
    qpos[:, 3] = 1.0                                            # pelvis quat wxyz
    qpos[:, 7:36] = (np.arange(n, dtype=np.float64)[:, None] * 0.001
                     + np.arange(JOINT_DIM, dtype=np.float64)[None, :] * 0.1)
    return qpos


class _ExportArgs:
    """The attribute subset ``_write_superset`` reads off an argparse Namespace."""

    def __init__(self, cond_mode="home"):
        self.cond_mode = cond_mode
        self.dz_anchor = "chest"
        self.dz_wrist = None
        self.no_wrist_ik = False
        self.foot_ik = False


def _augment(src: str, out: str, *, cond_mode="home", seed=7,
             episodes=None) -> tuple[str, dict[str, np.ndarray]]:
    """Add ``wb_*`` arrays to ``src`` using the EXPORTER's own assembly + writer.

    Returns ``(out_path, arrays)`` so a test can assert against exactly what was written.
    """
    import zarr

    from flore.s1_policy.data.hl_dataset import _register_image_codecs

    _register_image_codecs()
    with zarr.ZipStore(src, mode="r") as store:
        root = zarr.group(store=store)
        attrs = dict(root.attrs)
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        ts = np.asarray(root["data"]["episode_timestamps"][:], dtype=np.float64)
    starts = np.concatenate([[0], ends[:-1]])
    # Re-pin the exporter's grid from the file, exactly as the real tool's main() does.
    grid = wb_export.core.set_grid(int(attrs["hl_force_hist_len"]),
                                   int(attrs["hl_action_horizon"]),
                                   float(attrs["hl_action_hz"]),
                                   float(attrs["hl_force_hist_span_s"]))
    eps = range(len(ends)) if episodes is None else episodes
    completions = {int(e): _synthetic_qpos(int(ends[e]) - int(starts[e]), seed + int(e))
                   for e in eps}
    arrays = wb_export.build_wholebody_arrays(completions, starts, ends, ts)
    wb_export._write_superset(src, out, _ExportArgs(cond_mode), grid, arrays,
                              [{"episode": int(e)} for e in eps], overwrite=True)
    return out, arrays



def _patch_zarr(path: str, *, arrays: dict | None = None, attrs: dict | None = None) -> str:
    """Rewrite ``path`` in place with some arrays/attrs replaced.

    A ``ZipStore(mode="a")`` cannot be used for this: it APPENDS a second member under the
    same name and zarr then reads whichever the zipfile index resolves first, so the "patch"
    silently does nothing (the observed symptom is a ``UserWarning: Duplicate name`` and a
    test that passes for the wrong reason). Copying through a MemoryStore and rewriting the
    archive is the only way to actually change a value.
    """
    import zarr

    from flore.s1_policy.data.hl_dataset import _register_image_codecs

    _register_image_codecs()
    mem = zarr.MemoryStore()
    with zarr.ZipStore(path, mode="r") as src:
        zarr.copy_store(src, mem)
    root = zarr.group(store=mem)
    for key, value in (arrays or {}).items():
        root["data"].create_dataset(key, data=value, overwrite=True)
    root.attrs.update(attrs or {})
    os.remove(path)
    with zarr.ZipStore(path, mode="w") as dst:
        zarr.copy_store(mem, dst)
    return path


def _read_array(path: str, key: str) -> np.ndarray:
    import zarr

    from flore.s1_policy.data.hl_dataset import _register_image_codecs

    _register_image_codecs()
    with zarr.ZipStore(path, mode="r") as store:
        return np.asarray(zarr.group(store=store)["data"][key][:])


@pytest.fixture()
def dual_zarr(tmp_path):
    """The 3-tracker rig: 3 pucks, 2 cameras, 2 F/T sensors, 2 grippers. No wb_* arrays."""
    return _write_hl_zarr(
        tmp_path / "dual.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left", "right"),
        gripper_roles=("left", "right"),
        n_cameras=2,
        attrs_override={"hands_mode": "both_head_dual"},
    )


@pytest.fixture()
def wb_pair(tmp_path, dual_zarr):
    """``(dual_path, wb_path, arrays)`` — the same recording before and after augmentation."""
    out, arrays = _augment(dual_zarr, str(tmp_path / "dual_wb.zarr.zip"))
    return dual_zarr, out, arrays


def _wb_ds(path, **kw):
    kw.setdefault("mode", MODE)
    kw.setdefault("image_keys", CAMS)
    return _ds(path, **kw)


class TinyEncoder(nn.Module):
    """Stand-in for the timm ViT: same (N,C,H,W) -> (N, num_features) contract, no download."""

    num_features = 32

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, self.num_features, kernel_size=8, stride=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).mean(-1)


def _train_config(**hl_kw) -> TrainConfig:
    kw = dict(mode=MODE, image_keys=CAMS, norm_stats_path="stats.json")
    kw.update(hl_kw)
    return TrainConfig(algo="diffusion", pipeline="hl", hl_data=HLDataConfig(**kw))


# ---------------------------------------------------------------------------
# One joint order, one key table — across two repos
# ---------------------------------------------------------------------------
def test_joint_names_match_the_sdk_order():
    """The 29 columns must be holosoma's dof_names order in all THREE places that name it.

    A permutation of 29 joint angles is a valid pose, so a swap has no numeric tell: it trains
    to a healthy loss and renders a robot doing something else. FLoRe mirrors the order twice
    (the layout for training, live_robot for the RecordTap remap) and the exporter mirrors it
    from holosoma_motion; nothing but this test connects them.
    """
    from flore.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES

    assert WB_JOINT_NAMES == G1_29DOF_SDK_DOF_NAMES
    assert WB_JOINT_NAMES == wb_export.WB_JOINT_NAMES
    assert len(WB_JOINT_NAMES) == JOINT_DIM == 29
    assert len(set(WB_JOINT_NAMES)) == JOINT_DIM        # no duplicate name


def test_exporter_and_loader_agree_on_the_array_names(wb_pair):
    """Every ``wb_*`` key the loader reads must be one the exporter writes."""
    _dual, wb, _arrays = wb_pair
    ds = _wb_ds(wb)
    try:
        read = set(ds._keys[0]["wb"].values())
        assert read <= set(wb_export.WB_KEYS), read - set(wb_export.WB_KEYS)
        # wb_root_pose is written but deliberately NOT read (no low level has a root channel).
        assert "wb_root_pose" in set(wb_export.WB_KEYS) - read
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Widths: every one DERIVED, so a disagreement is impossible by construction
# ---------------------------------------------------------------------------
def test_config_pushes_every_width_from_the_layout():
    cfg = _train_config()
    m = cfg.active_model
    lay = get_layout(MODE)
    assert (m.action_dim, lay.action_dim) == (JOINT_DIM + 2 * (1 + FORCE_DIM), 43)
    assert (m.state_dim, lay.state_dim) == (JOINT_DIM + 2, 31)
    assert m.action_horizon == ACTION_HORIZON
    assert m.layout_mode == MODE
    assert m.num_cameras == 2
    assert m.force_hist_channels == 2 * FORCE_DIM == 12
    assert m.force_hist_len == FORCE_HIST_LEN


def test_the_layout_has_no_pose_and_no_head_block():
    """The three structural departures, stated as assertions rather than comments."""
    lay = get_layout(MODE)
    assert lay.is_wholebody
    assert lay.pose_dims() == ()                 # nothing is relativized
    assert lay.joint_dims() == tuple(range(JOINT_DIM))
    assert lay.roles == ("left", "right")        # the two grip+wrench blocks
    assert "head" not in lay.roles
    # ...but the mode still needs the chest tracker, and shares the dual conversion.
    assert tracker_roles_required(MODE) == ("left", "right", "head")
    assert hands_flag(MODE) == hands_flag(DUAL_MODE) == "both_head_dual"
    # Everything is supervised: no reserved-but-unmeasured bookkeeping in this mode.
    assert lay.unmeasured_dims() == () and lay.pad_dims() == ()


def test_state_is_joint_space_not_wrt_start():
    lay = get_layout(MODE)
    assert lay.joint_state_slot == slice(0, JOINT_DIM)
    slots = lay.state_slots()
    assert slots == {"left": {"gripper": JOINT_DIM}, "right": {"gripper": JOINT_DIM + 1}}
    # The key a keypoint-mode consumer reaches for is ABSENT rather than 0-width, so an
    # un-taught consumer raises KeyError on the name instead of writing nothing.
    assert "wrt_start_rot6d" not in slots["left"]
    assert "wrt_start_rot6d" in get_layout(DUAL_MODE).state_slots()["left"]


# ---------------------------------------------------------------------------
# The sample: absolute joints, and the identity that ties action to state
# ---------------------------------------------------------------------------
def test_the_action_is_the_absolute_joints_verbatim(wb_pair):
    """No anchor, no delta: ``action[:, :29]`` IS what the exporter wrote.

    An accidental ``absolute_to_delta_ee`` here would still yield a 43-dim chunk with a sane
    loss — and would command the wrong posture at deploy, where ``JointClutch`` already takes
    the within-chunk difference itself.
    """
    _dual, wb, arrays = wb_pair
    ds = _wb_ds(wb)
    try:
        idx = 0
        fi, frame = ds._index[idx]
        assert fi == 0
        s = ds[idx]
        want = arrays["wb_action_joints"][frame][:ACTION_HORIZON]
        got = s["action_chunk"][:, :JOINT_DIM].numpy()
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
        # Step 0 lands ON t0, so it must equal the anchor frame's own state joints (both are
        # the same instant). That identity is what says the two arrays came from one
        # completion, and it is exactly what --check prints on real data.
        np.testing.assert_allclose(got[0], arrays["wb_state_joints"][frame],
                                   rtol=0, atol=1e-5)
        np.testing.assert_allclose(s["state"][:JOINT_DIM].numpy(),
                                   arrays["wb_state_joints"][frame], rtol=0, atol=1e-6)
    finally:
        ds.close()


def test_sample_shapes_and_the_gripper_state_columns(wb_pair):
    _dual, wb, _arrays = wb_pair
    ds = _wb_ds(wb)
    try:
        s = ds[0]
        assert s["action_chunk"].shape == (ACTION_HORIZON, 43)
        assert s["action_mask"].shape == (ACTION_HORIZON, 43)
        assert s["state"].shape == (31,)
        assert s["state_valid"].shape == (31,)
        assert s["force_history"].shape == (FORCE_HIST_LEN, 12)
        assert s["force_history_valid"].shape == (2,)     # per SENSOR, not shared
        assert s["image"].shape[0] == 2                    # two cameras
        # 0.0475 m of 0.095 m full open == 0.5, in BOTH gripper state columns.
        np.testing.assert_allclose(s["state"][JOINT_DIM:].numpy(), [0.5, 0.5], atol=1e-6)
    finally:
        ds.close()


def test_both_hands_force_and_gripper_are_supervised_and_not_crossed(wb_pair):
    """The 7-dim hand blocks keep the dual preset's values, just at the new offsets."""
    _dual, wb, _arrays = wb_pair
    lay = get_layout(MODE)
    ds = _wb_ds(wb)
    try:
        fi, frame = ds._index[0]
        s = ds[0]
        a, m = s["action_chunk"].numpy(), s["action_mask"].numpy()
        # _write_hl_zarr encodes force as base + frame + 0.01*step + 0.0001*channel, with
        # base 0 for LEFT and 100 for RIGHT — so a crossed hand is a 100.0 error.
        for role, base in (("left", 0.0), ("right", 100.0)):
            f = lay.slots(role).force
            want = base + frame + 0.01 * np.arange(ACTION_HORIZON)[:, None] \
                + 0.0001 * np.arange(FORCE_DIM)[None, :]
            np.testing.assert_allclose(a[:, f], want, rtol=0, atol=1e-3)
            assert m[:, f].any(), role
            assert m[:, lay.slots(role).gripper].any(), role
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Masks: every tracker gates the joints, and the joints gate everything else
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_role", ["left", "right", "head"])
def test_the_joint_mask_ands_every_tracker_including_the_chest(tmp_path, bad_role):
    """A dropout on ANY puck invalidates the joint columns — the CHEST above all.

    The completion is conditioned on all three, so an interpolated puck means fabricated
    joints. The chest owns no keypoint block in this layout, so a scan over ``layout.roles``
    would miss it entirely and supervise 29 of 43 columns on invented posture.
    """
    src = _write_hl_zarr(
        tmp_path / f"bad_{bad_role}.zarr.zip",
        roles=("left", "right", "head"), force_roles=("left", "right"),
        gripper_roles=("left", "right"), n_cameras=2,
        attrs_override={"hands_mode": "both_head_dual"},
    )
    # Invalidate ONE future step of every anchor for one role, then read a sample back. Done
    # after the fixture is written so only this array changes.
    bad_step = 3
    v = _read_array(src, f"hl_{bad_role}_action_valid").astype(np.float32)
    v[:, bad_step] = 0.0
    _patch_zarr(src, arrays={f"hl_{bad_role}_action_valid": v})
    assert _read_array(src, f"hl_{bad_role}_action_valid")[0, bad_step] == 0.0
    wb, _arrays = _augment(src, str(tmp_path / "bad_wb.zarr.zip"))
    lay = get_layout(MODE)
    # drop_glitch_windows would remove every anchor (the corruption is on all of them), so it
    # is off here: this test is about the per-STEP mask, and the window drop has its own test.
    ds = _wb_ds(wb, drop_glitch_windows=False)
    try:
        m = ds[0]["action_mask"].numpy()
        assert not m[bad_step, list(lay.joint_dims())].any(), bad_role
        assert m[bad_step + 1, list(lay.joint_dims())].all(), bad_role
        # ...and the gripper/force columns inherit it: a target the BODY cannot reach is not a
        # useful gripper or force label either (the keypoint modes apply the same asymmetry
        # with the pose mask).
        for role in ("left", "right"):
            assert not m[bad_step, lay.slots(role).gripper]
            assert not m[bad_step, lay.slots(role).force].any()
    finally:
        ds.close()


def test_a_chest_glitch_drops_the_whole_window(tmp_path):
    """``drop_glitch_windows`` must scan the CHEST, which owns no role in this layout.

    Per-step masking cannot repair this: the completion's ROOT conditioning is the chest, so a
    chest excursion corrupts the posture of every step in the window, not just the flagged one.
    """
    src = _write_hl_zarr(
        tmp_path / "chest.zarr.zip",
        roles=("left", "right", "head"), force_roles=("left", "right"),
        gripper_roles=("left", "right"), n_cameras=2,
        attrs_override={"hands_mode": "both_head_dual"},
    )
    v = _read_array(src, "hl_head_action_valid").astype(np.float32)
    # A mid-EPISODE corruption on the first episode only, so the second episode's anchors
    # survive and "everything was dropped" cannot pass this test by accident.
    v[30, :] = 0.0
    _patch_zarr(src, arrays={"hl_head_action_valid": v})
    wb, _arrays = _augment(src, str(tmp_path / "chest_wb.zarr.zip"))
    kept = _ds(wb, mode=MODE, image_keys=CAMS, drop_glitch_windows=True)
    allv = _ds(wb, mode=MODE, image_keys=CAMS, drop_glitch_windows=False)
    try:
        assert len(kept) < len(allv)
        # Every dropped anchor is in episode 0 (where the chest glitch is).
        dropped = set(allv._index) - set(kept._index)
        assert dropped and all(f < int(allv._episode_ends_by_file[0][0]) for _fi, f in dropped)
    finally:
        kept.close()
        allv.close()


def test_an_uncompleted_tail_is_not_supervised(tmp_path, dual_zarr):
    """``--episodes 0`` leaves episode 1 present but valid=0, never trained toward 0 rad.

    0 rad is a LEGAL posture (the 29-joints-zero pose the ``home`` anchor starts from), so the
    zero-filled rows cannot say they are fabricated — only the mask can.
    """
    wb, arrays = _augment(dual_zarr, str(tmp_path / "ep0_wb.zarr.zip"), episodes=[0])
    ds = _wb_ds(wb, drop_glitch_windows=False)
    try:
        ends = ds._episode_ends_by_file[0]
        ep1_frames = [i for i, (_fi, f) in enumerate(ds._index) if f >= ends[0]]
        assert ep1_frames
        lay = get_layout(MODE)
        s = ds[ep1_frames[0]]
        assert not s["action_mask"].numpy()[:, list(lay.joint_dims())].any()
        assert not s["state_valid"].numpy()[:JOINT_DIM].any()
        assert float(np.abs(arrays["wb_state_joints"][ends[0]]).max()) == 0.0
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# NO REGRESSION: the 3-tracker preset over the same file
# ---------------------------------------------------------------------------
def test_the_dual_preset_is_unchanged_by_augmentation(wb_pair):
    """The augmented file loaded as ``both_hands_head_dual`` == the original, tensor for tensor.

    This is what makes the wholebody export additive rather than a fork: one zarr, two presets,
    and no re-conversion of the camera streams. Nothing about the SHAPES would reveal a
    regression here, since the wb_* arrays are simply extra keys.
    """
    dual, wb, _arrays = wb_pair
    a = _ds(dual, mode=DUAL_MODE, image_keys=CAMS)
    b = _ds(wb, mode=DUAL_MODE, image_keys=CAMS)
    try:
        assert len(a) == len(b) and a._index == b._index
        assert a.anchor_indices_fingerprint == b.anchor_indices_fingerprint
        for idx in (0, len(a) // 2, len(a) - 1):
            sa, sb = a[idx], b[idx]
            assert sa.keys() == sb.keys()
            for key in sa:
                assert torch.equal(sa[key], sb[key]), f"{key} changed at sample {idx}"
    finally:
        a.close()
        b.close()


def test_a_plain_dual_zarr_is_refused_for_wholebody_by_name(dual_zarr):
    """A file with no ``wb_*`` arrays must name the exporter, not fail on a missing key."""
    with pytest.raises(ValueError, match=r"wb_windows"):
        _wb_ds(dual_zarr)
    with pytest.raises(ValueError, match=r"run_mc_to_hl_wholebody"):
        _wb_ds(dual_zarr)


def test_a_left_only_file_is_still_refused_for_wholebody(tmp_path):
    """The hands_mode gate is unchanged: wholebody needs the dual conversion."""
    left = _write_hl_zarr(tmp_path / "left.zarr.zip")
    with pytest.raises(ValueError, match=r"--hands"):
        _wb_ds(left)


# ---------------------------------------------------------------------------
# The gates that exist because the values cannot tell you
# ---------------------------------------------------------------------------
def test_two_completion_anchors_in_one_run_are_refused(tmp_path, dual_zarr):
    """A ``home`` shard and an ``absolute`` shard are two posture spaces at one width."""
    a, _ = _augment(dual_zarr, str(tmp_path / "home.zarr.zip"), cond_mode="home")
    b, _ = _augment(dual_zarr, str(tmp_path / "abs.zarr.zip"), cond_mode="absolute", seed=99)
    with pytest.raises(ValueError, match=r"wb_cond_mode"):
        _ds([a, b], mode=MODE, image_keys=CAMS)
    # ...and each on its own is fine, so the refusal is about the MIX.
    for path in (a, b):
        ds = _wb_ds(path)
        ds.close()


def test_the_anchor_is_recorded_in_the_split_provenance(wb_pair):
    """Norm stats fitted over one anchor must not compare equal to another's.

    ``dataset_knobs`` is compared for EXACT equality by ``trainer.validate_split_provenance``,
    and the two anchors' widths are identical — so this key is the only thing that catches it.
    """
    _dual, wb, _arrays = wb_pair
    ds = _wb_ds(wb)
    try:
        knobs = ds.split_provenance(1)["dataset_knobs"]
        assert knobs["wb_cond_mode"] == "home"
        assert knobs["mode"] == MODE
    finally:
        ds.close()
    # A keypoint run's provenance must stay byte-identical to what existing stats files carry.
    dual_ds = _ds(_dual, mode=DUAL_MODE, image_keys=CAMS)
    try:
        assert "wb_cond_mode" not in dual_ds.split_provenance(1)["dataset_knobs"]
    finally:
        dual_ds.close()


def test_a_permuted_joint_order_is_refused(tmp_path, dual_zarr):
    """Swapping two joint names must be a loud refusal — the VALUES cannot say."""
    wb, _arrays = _augment(dual_zarr, str(tmp_path / "perm.zarr.zip"))
    names = list(WB_JOINT_NAMES)
    names[3], names[9] = names[9], names[3]        # left_knee <-> right_knee
    _patch_zarr(wb, attrs={"wb_joint_names": names})
    with pytest.raises(ValueError, match=r"wb_joint_names"):
        _wb_ds(wb)


@pytest.mark.parametrize("attr,value", [
    ("wb_joints", 23),
    ("wb_action_first_step", "t0+dt"),
    ("wb_action_hz", 16.0),
])
def test_a_wb_grid_mismatch_is_refused(tmp_path, dual_zarr, attr, value):
    wb, _arrays = _augment(dual_zarr, str(tmp_path / f"grid_{attr}.zarr.zip"))
    _patch_zarr(wb, attrs={attr: value})
    with pytest.raises(ValueError, match=attr):
        _wb_ds(wb)


# ---------------------------------------------------------------------------
# Metrics: the empty-pose trap
# ---------------------------------------------------------------------------
def test_metrics_report_joint_degrees_and_never_a_fabricated_position():
    """``physical_trajectory_sums`` must OMIT the pose keys, not report a perfect 0.00 cm.

    ``mask[..., []].all(axis=-1)`` is True everywhere and a norm over 0 columns is exactly
    0.0, so the un-guarded arithmetic logs ``val/sample_position_cm = 0.00`` for a policy that
    predicts no position at all — a metric that says the run is perfect.
    """
    from flore.s1_policy.training.metrics import physical_trajectory_sums

    lay = get_layout(MODE)
    pred = np.zeros((2, ACTION_HORIZON, lay.action_dim))
    target = np.zeros_like(pred)
    target[..., :JOINT_DIM] = np.radians(2.0)          # 2 deg off on every joint
    mask = np.ones_like(pred, dtype=bool)
    sums = physical_trajectory_sums(pred, target, mask, lay)
    assert "position_cm" not in sums and "rotation_geodesic_deg" not in sums
    num, den = sums["joint_deg"]
    assert den == 2 * ACTION_HORIZON
    assert num / den == pytest.approx(2.0, abs=1e-9)
    # ...while the keypoint layout still reports the pose keys and no joint key.
    kp = get_layout(DUAL_MODE)
    kp_sums = physical_trajectory_sums(
        np.zeros((1, ACTION_HORIZON, kp.action_dim)),
        np.zeros((1, ACTION_HORIZON, kp.action_dim)),
        np.ones((1, ACTION_HORIZON, kp.action_dim), dtype=bool), kp)
    assert "position_cm" in kp_sums and "joint_deg" not in kp_sums


def test_metric_groups_log_joints_and_no_position():
    groups = get_layout(MODE).metric_group_dims()
    assert set(groups) == {"joints", "left", "right", "grip", "force"}
    assert groups["joints"] == tuple(range(JOINT_DIM))
    assert "pos" not in groups and "rot" not in groups


# ---------------------------------------------------------------------------
# The model: 43 dims out, two force encoder groups, a learned state token
# ---------------------------------------------------------------------------
def _model(cfg_diffusion):
    from flore.s1_policy.model.dp_diffusion import DPDiffusionModel

    cfg_diffusion.pretrained = False
    return DPDiffusionModel(cfg_diffusion, encoder=TinyEncoder())


def _fake_batch(lay, b: int = 2) -> dict:
    return {
        "images": [torch.randn(b, 3, 224, 224), torch.randn(b, 3, 224, 224)],
        "img_masks": [torch.ones(b, dtype=torch.bool)] * 2,
        "state": torch.randn(b, lay.state_dim),
        "force_history": torch.randn(b, FORCE_HIST_LEN, lay.force_hist_channels),
        "force_history_valid": torch.ones(b, 2, dtype=torch.bool),
        "actions": torch.randn(b, ACTION_HORIZON, lay.action_dim),
        "action_mask": torch.ones(b, ACTION_HORIZON, lay.action_dim, dtype=torch.bool),
    }


def test_model_trains_and_samples_the_full_43_dim_chunk():
    lay = get_layout(MODE)
    cfg = _train_config()
    model = _model(cfg.diffusion)
    batch = _fake_batch(lay)
    out = model(batch)
    assert torch.isfinite(out["loss"])
    # Per-group metrics must name the joints, and not a position group.
    names = set(out["metrics"])
    for want in ("diffusion_loss/joints", "diffusion_loss/left", "diffusion_loss/right",
                 "diffusion_loss/grip", "diffusion_loss/force"):
        assert want in names, (want, sorted(names))
    assert "diffusion_loss/pos" not in names and "diffusion_loss/head" not in names
    with torch.no_grad():
        chunk = model.sample_actions(batch)
    assert tuple(chunk.shape) == (2, ACTION_HORIZON, 43)
    assert torch.isfinite(chunk).all()


def test_the_state_encoder_is_a_learned_projection_when_asked():
    """The Psi0-shaped option: one Linear over the padded state, not a zero-pad slot.

    pi0 does exactly ``nn.Linear(action_dim, width)`` on the proprioceptive state; the preset
    turns this on because 29 raw radians beside 768-dim ViT features is a scale mismatch a
    learned projection absorbs. Both paths must build, so the ablation is one flag.
    """
    plain = _model(_train_config().diffusion)
    assert plain.state_proj is None

    cfg = _train_config()
    cfg.diffusion.state_projection = True
    proj = _model(cfg.diffusion)
    assert isinstance(proj.state_proj, nn.Linear)
    assert proj.state_proj.in_features == get_layout(MODE).action_dim
    assert torch.isfinite(proj(_fake_batch(get_layout(MODE)))["loss"])


def test_both_force_groups_are_built_and_receive_gradient():
    model = _model(_train_config().diffusion)
    assert model.force_encoder is not None
    model(_fake_batch(get_layout(MODE)))["loss"].backward()
    grads = dict(model.force_encoder.named_parameters())
    # Group 0 is the LEFT Bota, extra_groups[0] the RIGHT: if only group 0 were wired the
    # right wrench would never reach global_cond and the run would look entirely healthy.
    for name in ("conv.1.weight", "extra_groups.0.conv.1.weight"):
        g = grads[name].grad
        assert g is not None and float(g.norm()) > 0.0, name


# ---------------------------------------------------------------------------
# Serving / offline inference cross-load
# ---------------------------------------------------------------------------
def test_a_wholebody_checkpoint_cannot_be_served_as_the_dual_preset():
    """``layout_mode`` is the only discriminator, so it must be pinned unconditionally."""
    from flore.s1_policy.inference.common import validate_model_layout

    cfg = _train_config().diffusion
    hl_data = {"mode": MODE, "obs_history_len": 1, "image_keys": list(CAMS)}
    validate_model_layout(cfg, hl_data, get_layout(MODE), num_cameras=2)
    with pytest.raises(ValueError, match=r"layout_mode"):
        validate_model_layout(cfg, hl_data, get_layout(DUAL_MODE), num_cameras=2)


# ---------------------------------------------------------------------------
# The exporter's own window math
# ---------------------------------------------------------------------------
def test_joint_windows_are_time_anchored_not_frame_indexed():
    """A DROPPED camera frame must shift no label — the query is a TIME, not an index.

    On a uniform grid the two forms agree, which is why this needs a non-uniform fixture: with
    one frame missing, "joints at frame+k" is a different instant from "joints at t0+k*dt" for
    the whole rest of the episode.
    """
    from flore.s1_policy.data.layout import ACTION_DT

    wb_export.core.set_grid(FORCE_HIST_LEN, ACTION_HORIZON, 1.0 / ACTION_DT)
    n = ACTION_HORIZON + 20
    ts = np.arange(n, dtype=np.float64) * ACTION_DT
    ts[10:] += ACTION_DT                                    # frame 10 dropped
    # A joint trajectory that is an exact linear function of TIME, so the correct answer is
    # computable in closed form and an index-based read is off by exactly one dt.
    joints = np.tile((ts * 3.0)[:, None], (1, JOINT_DIM))
    act, valid, sv = wb_export.joint_windows(joints, ts, n)
    assert act.shape == (n, ACTION_HORIZON, JOINT_DIM)
    assert sv.shape == (n, 1) and bool(sv.all())
    for frame in (0, 9, 11, 30):
        want = (ts[frame] + ACTION_DT * np.arange(ACTION_HORIZON)) * 3.0
        ok = valid[frame]
        np.testing.assert_allclose(act[frame, ok, 0], want[ok], rtol=0, atol=1e-9)
    # The tail past the episode end is CLAMPED in value and INVALID in the mask, never trained.
    assert not valid[-1, 1:].any()


def test_a_fully_completed_episode_supervises_every_non_tail_step():
    """No interior step may be dropped when the completion covered every frame.

    THE regression this guards: ``future_step_valid``'s interior-coverage bin defaults to ONE
    action step, and here the source grid IS the query grid — so each bin holds exactly one
    sample and the float error in ``t0 + k*dt`` decides whether it lands inside. Measured with
    the default, anchors lost 1-3 of 50 steps at random interior positions, the glitch-window
    filter read those as tracker dropouts, and **93% of all anchors were dropped** from a
    perfectly clean dataset. Nothing about that is visible in a value: the joints were right,
    there were simply far fewer samples than there should have been.
    """
    from flore.s1_policy.data.layout import ACTION_DT

    wb_export.core.set_grid(FORCE_HIST_LEN, ACTION_HORIZON, 1.0 / ACTION_DT)
    n = ACTION_HORIZON + 20
    ts = np.arange(n, dtype=np.float64) * ACTION_DT
    joints = np.tile(np.arange(n, dtype=np.float64)[:, None], (1, JOINT_DIM))
    _act, valid, _sv = wb_export.joint_windows(joints, ts, n)
    # A step is tail padding iff its own time runs past the episode's last frame; everything
    # else must be supervised. Derived, not a magic count.
    times = ts[:, None] + ACTION_DT * np.arange(ACTION_HORIZON)[None, :]
    want = times <= ts[-1] + 1e-9
    np.testing.assert_array_equal(valid, want)
    assert bool(valid[0].all())            # the first anchor's whole chunk is interior


def test_joint_windows_flag_the_uncompleted_prefix_boundary():
    from flore.s1_policy.data.layout import ACTION_DT

    wb_export.core.set_grid(FORCE_HIST_LEN, ACTION_HORIZON, 1.0 / ACTION_DT)
    n = ACTION_HORIZON + 20
    ts = np.arange(n, dtype=np.float64) * ACTION_DT
    joints = np.tile(np.arange(n, dtype=np.float64)[:, None], (1, JOINT_DIM))
    n_cov = 40
    _act, valid, sv = wb_export.joint_windows(joints[:n_cov], ts, n_cov)
    assert bool(sv[:n_cov].all()) and not bool(sv[n_cov:].any())
    # Anchor 0's steps beyond the covered prefix must be invalid, not endpoint-held.
    assert valid[0, :n_cov].all()
    assert not valid[0, n_cov:].any()


# ---------------------------------------------------------------------------
# wb_joint_target: the absolute-vs-delta A/B on the 29 joint TARGET columns
# ---------------------------------------------------------------------------
def test_the_delta_target_is_exactly_the_absolute_one_minus_row_zero(wb_pair):
    """``delta`` is ``q[k] - q[0]`` of the SAME file, and its row 0 is exactly zero.

    Pinned against the absolute dataset rather than against the raw array, because the property
    that matters is that the two targets are one subtraction apart: that is what makes the
    deploy-side ``JointClutch`` (``base + (q[i] - q[0])``, offset-invariant) publish the same
    wire for both, and it is what a future change to the joint block would break silently.
    """
    _dual, wb, arrays = wb_pair
    absolute = _wb_ds(wb)
    delta = _wb_ds(wb, wb_joint_target="delta")
    try:
        assert absolute.wb_joint_target == "absolute"     # the default is unchanged
        fi, frame = delta._index[0]
        assert fi == 0
        a = absolute[0]["action_chunk"][:, :JOINT_DIM].numpy()
        d = delta[0]["action_chunk"][:, :JOINT_DIM].numpy()
        np.testing.assert_allclose(d, a - a[0], rtol=0, atol=1e-9)
        # Row 0 is EXACTLY zero, not merely small: it is the reference the whole chunk is
        # measured from, and step 0 landing on the clutch base is what makes the first executed
        # step a no-op on the robot.
        np.testing.assert_array_equal(d[0], np.zeros(JOINT_DIM))
        # The STATE stays ABSOLUTE. A delta state relative to itself would be identically zero,
        # i.e. no proprioception at all -- so the posture is not thrown away, only moved out of
        # the target. This is also what lets serving reconstruct a posture by integration.
        np.testing.assert_allclose(delta[0]["state"][:JOINT_DIM].numpy(),
                                   arrays["wb_state_joints"][frame], rtol=0, atol=1e-6)
    finally:
        absolute.close()
        delta.close()


def test_the_delta_target_changes_no_other_column(wb_pair):
    """Only ``[0:29]`` moves. The grippers and both wrenches are absolute in both modes."""
    _dual, wb, _arrays = wb_pair
    absolute = _wb_ds(wb)
    delta = _wb_ds(wb, wb_joint_target="delta")
    try:
        a = absolute[0]
        d = delta[0]
        np.testing.assert_array_equal(a["action_chunk"][:, JOINT_DIM:].numpy(),
                                      d["action_chunk"][:, JOINT_DIM:].numpy())
        np.testing.assert_array_equal(a["action_mask"][:, JOINT_DIM:].numpy(),
                                      d["action_mask"][:, JOINT_DIM:].numpy())
        np.testing.assert_array_equal(a["state"].numpy(), d["state"].numpy())
        np.testing.assert_array_equal(a["force_history"].numpy(), d["force_history"].numpy())
    finally:
        absolute.close()
        delta.close()


def test_an_uncovered_anchor_step_unsupervises_the_whole_delta_chunk(tmp_path, dual_zarr):
    """Row 0 is the chunk's reference, so an invalid row 0 poisons every row -- mask them all.

    ``wb_action_joints`` is ZERO-FILLED where the completion never reached and 0 rad is a legal
    posture, so the values cannot say the reference is fabricated. In ``absolute`` mode each row
    stands alone and only that row is masked; in ``delta`` mode the offset propagates, so the
    joint mask must AND in row 0's own validity. Nothing about the failure is visible in a shape.

    ``drop_glitch_windows=False`` on purpose: with the production default ON, an anchor whose t0
    step is uncovered is dropped outright and this mask never has to hold. That makes it a SECOND
    line rather than the only one — which is exactly why it needs a test, since the first line is
    a knob (``--hl-data.no-drop-glitch-windows``, and every ablation that sets it).
    """
    out, arrays = _augment(dual_zarr, str(tmp_path / "row0_bad_wb.zarr.zip"))
    valid = np.asarray(arrays["wb_action_valid"], dtype=np.float32)
    frame = 0
    assert bool(valid[frame].all())          # the fixture's first anchor starts fully covered
    valid[frame, 0] = 0.0                    # ... now its REFERENCE step is not
    _patch_zarr(out, arrays={"wb_action_valid": valid})

    absolute = _wb_ds(out, drop_glitch_windows=False)
    delta = _wb_ds(out, drop_glitch_windows=False, wb_joint_target="delta")
    try:
        idx = next(i for i, (fi, fr) in enumerate(absolute._index) if fi == 0 and fr == frame)
        a_mask = absolute[idx]["action_mask"][:, :JOINT_DIM].numpy()
        d_mask = delta[idx]["action_mask"][:, :JOINT_DIM].numpy()
        # absolute: step 0 alone is unsupervised, the rest of the chunk is still a label.
        assert not a_mask[0].any() and a_mask[1:].any()
        # delta: nothing in the chunk is a label, because everything is measured from step 0.
        assert not d_mask.any()
        # The gripper/force columns are NOT joint-anchored, so they keep the absolute mode's
        # per-step masking in both -- a row-0 gap must not discard a measured wrench.
        np.testing.assert_array_equal(absolute[idx]["action_mask"][:, JOINT_DIM:].numpy(),
                                      delta[idx]["action_mask"][:, JOINT_DIM:].numpy())
    finally:
        absolute.close()
        delta.close()


def test_the_joint_target_is_in_the_exact_equality_knobs_only_when_not_default(wb_pair):
    """``delta`` is recorded; ``absolute`` is NOT, so every stats file fitted before the knob
    existed keeps comparing equal. Same compatibility rule ``image_keys`` follows."""
    _dual, wb, _arrays = wb_pair
    absolute = _wb_ds(wb)
    delta = _wb_ds(wb, wb_joint_target="delta")
    try:
        assert "wb_joint_target" not in absolute.split_provenance(1)["dataset_knobs"]
        assert delta.split_provenance(1)["dataset_knobs"]["wb_joint_target"] == "delta"
    finally:
        absolute.close()
        delta.close()


@pytest.mark.parametrize("target", ["Delta", "relative", ""])
def test_an_unknown_joint_target_is_refused(wb_pair, target):
    _dual, wb, _arrays = wb_pair
    with pytest.raises(ValueError, match=r"wb_joint_target must be one of"):
        _wb_ds(wb, wb_joint_target=target)


def test_a_keypoint_layout_refuses_a_joint_target(dual_zarr):
    """There is no joint block to describe, so honouring it would be a silent no-op -- exactly
    the shape of bug where a launcher passes the knob to the wrong preset and believes it."""
    with pytest.raises(ValueError, match=r"needs a whole-body layout"):
        _ds(dual_zarr, mode="both_hands_head_dual", image_keys=CAMS, wb_joint_target="delta")


def test_the_config_refuses_a_joint_target_off_wholebody_and_a_typo():
    """Checked in TrainConfig too, because this is the failure the operator sees FIRST: the
    dataloader's refusal arrives after every model is built and W&B is open."""
    assert WB_JOINT_TARGETS == ("absolute", "delta")
    with pytest.raises(ValueError, match=r"wb_joint_target must be one of"):
        _train_config(wb_joint_target="detla")
    with pytest.raises(ValueError, match=r"needs --hl-data.mode wholebody"):
        TrainConfig(algo="diffusion", pipeline="hl",
                    hl_data=HLDataConfig(mode="both_hands_head_dual", image_keys=CAMS,
                                         norm_stats_path="stats.json",
                                         wb_joint_target="delta"))
    # And the default must still build the shipped preset.
    assert _train_config().hl_data.wb_joint_target == "absolute"
