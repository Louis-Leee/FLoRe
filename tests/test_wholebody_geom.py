"""Contract tests for the ``wholebody_geom_*`` family — the SAME 43-dim whole-body action with
a GEOMETRIC observation in place of the 29 raw joint angles.

Why these particular things are pinned: every one of them fails silently otherwise.

* The five whole-body modes share a byte-identical ACTION layout and differ ONLY in ``state``.
  So a crossed load is shape-correct in the action and wrong in the observation, and only
  ``layout_mode`` says so. ``wholebody`` must additionally be UNCHANGED by this family's
  arrival — its 31-dim state is what four shipped checkpoints recorded.
* ``lp_x``/``lp_y`` are "left TCP relative to the PELVIS" computed as ``p_L[:2]``, which is only
  correct because FK's frame origin IS the pelvis. Written as a bare index, that identity is
  invisible; a test is the only thing that keeps someone from "fixing" it into a subtraction of
  the wrong vector.
* The block widths live in ``layout.WB_STATE_BLOCK_DIM`` while the COLUMN ORDER lives in
  ``wb_geom.GEOM_BLOCK_NAMES``. Two tables, and a disagreement would slice a real feature into
  the wrong state column at full loss weight.
* An uncovered frame's geometry is ZERO, and zero is a plausible posture (both hands at the
  pelvis) — so it must be masked by ``state_valid``, exactly as the raw joints are.
* The jitter is applied by a WRAPPER on the train subset. If it ever leaked onto the shared
  dataset it would jitter the validation objective and the norm-stats fit, which no shape and no
  loss curve would reveal, and the sigma must land on the geometric columns only.

The FK itself is not re-tested here (``fuji/s1_policy/inference/ready_pose.py`` owns it and
cross-checks against holosoma's own bake). These tests inject a STUB whose ``bodies()`` returns
known poses, so they run with no URDF, no yourdfpy and no submodule — the same reason the
model tests use ``TinyEncoder`` instead of downloading a ViT.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fuji.s1_policy.data import wb_geom
from fuji.s1_policy.data.hl_dataset import StateNoise
from fuji.s1_policy.data.layout import (
    GRIPPER_DIM,
    JOINT_DIM,
    WB_STATE_BLOCK_DIM,
    get_layout,
    hands_flag,
    tracker_roles_required,
)
from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

from tests.test_diffusion_umi_wholebody import _augment, _wb_ds, dual_zarr  # noqa: F401

CAMS = ("camera0_rgb", "camera1_rgb")
GEOM_MODES = ("wholebody_geom_xy_rpy", "wholebody_geom_xy", "wholebody_geom_rpy",
              "wholebody_geom_none")
WB_MODES = ("wholebody", *GEOM_MODES)


class _FKStub:
    """Stand-in for ``KeypointFK``: ``bodies(q)`` returns poses DERIVED FROM q.

    Derived, not constant, so a test can tell "the dataset FK'd the right frame's joints" from
    "the dataset FK'd something". Frame f's left TCP is at ``(f, 10f, 100f)`` and its rotation is
    a yaw of ``f`` radians scaled down; the right TCP trails it by a fixed offset.
    """

    @staticmethod
    def _q(dof) -> float:
        return float(np.asarray(dof, dtype=np.float64).reshape(-1)[0])

    def bodies(self, dof, *, lift_z: float = 0.0):
        v = self._q(dof)
        pos = {wb_geom.LEFT_BODY: np.array([v, 10.0 * v, 100.0 * v]),
               wb_geom.RIGHT_BODY: np.array([v - 0.25, 10.0 * v - 0.5, 100.0 * v + 0.75])}
        rot = {wb_geom.LEFT_BODY: Rotation.from_euler("xyz", [0.1 * v, 0.2 * v, 0.3 * v]).as_matrix(),
               wb_geom.RIGHT_BODY: Rotation.from_euler("xyz", [-0.4 * v, 0.5 * v, -0.6 * v]).as_matrix()}
        return pos, rot


# ---------------------------------------------------------------------------
# The layout: one action, five states
# ---------------------------------------------------------------------------
def test_the_shipped_wholebody_state_is_unchanged():
    """31 = 29 joints + 2 grippers, with the joints FIRST. Four checkpoints recorded this."""
    lay = get_layout("wholebody")
    assert lay.state_dim == JOINT_DIM + 2 * GRIPPER_DIM == 31
    assert lay.state_geom == ("joints",)
    assert lay.joint_state_slot == slice(0, JOINT_DIM)
    assert lay.state_slots() == {"left": {"gripper": 29}, "right": {"gripper": 30}}


@pytest.mark.parametrize("mode,state_dim,geom", [
    ("wholebody", 31, ("joints",)),
    ("wholebody_geom_xy_rpy", 12, ("rel_xy", "rpy")),
    ("wholebody_geom_xy", 6, ("rel_xy",)),
    ("wholebody_geom_rpy", 8, ("rpy",)),
    ("wholebody_geom_none", 2, ()),
])
def test_state_widths_and_composition(mode, state_dim, geom):
    lay = get_layout(mode)
    assert lay.state_geom == geom
    assert lay.state_dim == state_dim
    # The geometric blocks are contiguous from 0, and the grippers come after all of them.
    slots = lay.state_geom_slots()
    assert list(slots) == list(geom)
    cur = 0
    for block in geom:
        assert slots[block] == slice(cur, cur + WB_STATE_BLOCK_DIM[block])
        cur = slots[block].stop
    assert lay.state_slots()["left"]["gripper"] == cur
    assert lay.state_slots()["right"]["gripper"] == cur + 1
    # No whole-body mode carries the keypoint modes' orientation channel.
    assert all("wrt_start_rot6d" not in entry for entry in lay.state_slots().values())


def test_every_whole_body_mode_shares_one_action_layout():
    """The ablation is on the OBSERVATION alone: the action must be identical, column for column.

    Checked as the whole set of action-side accessors rather than just ``action_dim``, because
    matching widths are exactly what makes a crossed load silent."""
    ref = get_layout("wholebody")
    for mode in GEOM_MODES:
        lay = get_layout(mode)
        assert lay.action_dim == ref.action_dim == 43, mode
        assert lay.keypoints == ref.keypoints, mode
        assert lay.joints == ref.joints == slice(0, JOINT_DIM), mode
        assert lay.joint_dims() == ref.joint_dims(), mode
        assert lay.pose_dims() == ref.pose_dims() == (), mode
        assert lay.force_dims() == ref.force_dims(), mode
        assert lay.gripper_dims() == ref.gripper_dims(), mode
        assert lay.measured_dims() == ref.measured_dims(), mode
        assert lay.unmeasured_dims() == ref.unmeasured_dims() == (), mode
        assert lay.metric_group_dims() == ref.metric_group_dims(), mode
        assert lay.force_hist_channels == ref.force_hist_channels, mode


@pytest.mark.parametrize("mode", WB_MODES)
def test_every_whole_body_mode_reads_the_same_conversion(mode):
    """One zarr trains all five: same --hands flag, same three required trackers.

    The chest is required even though no mode owns a chest BLOCK — it is the completion's root
    conditioning, so its glitch corrupts all 29 joint columns."""
    assert hands_flag(mode) == "both_head_dual"
    assert tracker_roles_required(mode) == ("left", "right", "head")


def test_the_widths_table_and_the_column_order_table_agree():
    """``layout.WB_STATE_BLOCK_DIM`` owns widths, ``wb_geom.GEOM_BLOCK_NAMES`` owns order.

    They are in different modules on purpose (the layout must not import scipy or reach into
    ``inference/``), so the one thing that could drift — the width — is pinned here. A
    disagreement would slice a real feature into the wrong state column at full loss weight."""
    for block, names in wb_geom.GEOM_BLOCK_NAMES.items():
        assert WB_STATE_BLOCK_DIM[block] == len(names), block
    # Every FK block is declared in the widths table, and `joints` is deliberately NOT an FK one.
    assert set(wb_geom.FK_BLOCKS) < set(WB_STATE_BLOCK_DIM)
    assert "joints" not in wb_geom.FK_BLOCKS


@pytest.mark.parametrize("bad", [("rel_xy", "rel_xy"), ("joints", "joints"), ("nope",)])
def test_a_bad_state_block_tuple_is_refused(bad):
    from fuji.s1_policy.data.layout import _build_wholebody

    with pytest.raises(ValueError):
        _build_wholebody("x", state_geom=bad)


# ---------------------------------------------------------------------------
# The features themselves
# ---------------------------------------------------------------------------
def test_rel_xy_is_hand_to_hand_then_hand_to_PELVIS():
    """``lp_*`` is ``p_L[:2]`` because the FK frame's ORIGIN is the pelvis.

    That identity is the whole reason the second pair needs no second body, and it is invisible
    in the code. If someone ever subtracts a real pelvis position here, this fails."""
    p_l = np.array([0.44, 0.15, 0.95])
    p_r = np.array([0.43, -0.15, 0.96])
    pos = {wb_geom.LEFT_BODY: p_l, wb_geom.RIGHT_BODY: p_r}
    rot = {b: np.eye(3) for b in (wb_geom.LEFT_BODY, wb_geom.RIGHT_BODY)}
    feats = wb_geom.geom_from_bodies(pos, rot, ("rel_xy",))
    assert feats.shape == (4,)
    np.testing.assert_allclose(feats[:2], (p_l - p_r)[:2])
    np.testing.assert_allclose(feats[2:], p_l[:2])
    # ...and it really is (p_L - p_pelvis)[:2], the pelvis being the origin of this frame.
    np.testing.assert_allclose(feats[2:], (p_l - np.zeros(3))[:2])
    # z is deliberately absent from both pairs: 4 columns, not 6.
    assert 0.95 not in feats and 0.96 not in feats


def test_rpy_is_each_hands_euler_in_order_and_carries_no_pelvis():
    want_l = np.array([0.11, -0.22, 0.33])
    want_r = np.array([-0.44, 0.05, 0.66])
    pos = {b: np.zeros(3) for b in (wb_geom.LEFT_BODY, wb_geom.RIGHT_BODY)}
    rot = {wb_geom.LEFT_BODY: Rotation.from_euler("xyz", want_l).as_matrix(),
           wb_geom.RIGHT_BODY: Rotation.from_euler("xyz", want_r).as_matrix()}
    feats = wb_geom.geom_from_bodies(pos, rot, ("rpy",))
    assert feats.shape == (6,)
    np.testing.assert_allclose(feats[:3], want_l, atol=1e-12)
    np.testing.assert_allclose(feats[3:], want_r, atol=1e-12)
    assert wb_geom.RPY_NAMES == ("left_roll", "left_pitch", "left_yaw",
                                 "right_roll", "right_pitch", "right_yaw")


def test_blocks_concatenate_in_the_order_asked():
    pos = {wb_geom.LEFT_BODY: np.array([1.0, 2.0, 3.0]),
           wb_geom.RIGHT_BODY: np.array([0.0, 0.0, 0.0])}
    rot = {b: np.eye(3) for b in (wb_geom.LEFT_BODY, wb_geom.RIGHT_BODY)}
    both = wb_geom.geom_from_bodies(pos, rot, ("rel_xy", "rpy"))
    np.testing.assert_allclose(both[:4], wb_geom.geom_from_bodies(pos, rot, ("rel_xy",)))
    np.testing.assert_allclose(both[4:], wb_geom.geom_from_bodies(pos, rot, ("rpy",)))


def test_geom_series_leaves_uncovered_frames_zero():
    """An uncovered frame holds the exporter's zero-fill, and 0 rad is a legal posture.

    FK'ing it would invent a pose for a frame that has none, so those rows must stay zero —
    ``__getitem__`` is what then marks them ``state_valid=False``."""
    q = np.tile(np.arange(1.0, 6.0)[:, None], (1, JOINT_DIM))
    covered = np.array([True, False, True, False, True])
    out = wb_geom.geom_series(_FKStub(), q, covered, ("rel_xy", "rpy"), log_every=0)
    assert out.shape == (5, 10)
    assert np.all(out[~covered] == 0.0)
    assert np.all(np.abs(out[covered]).sum(axis=1) > 0)
    # The value at a covered frame is that frame's OWN joints, not a neighbour's.
    np.testing.assert_allclose(out[2], wb_geom.geom_from_joints(_FKStub(), q[2], ("rel_xy", "rpy")))


def test_a_wrapping_euler_channel_warns(caplog):
    """RPY is used deliberately and is only safe while the angles stay small (measured max 41.5
    deg on merged3). A task that reaches past the wrap must SAY so — the 2*pi jump is something
    the loss would chase with nothing in the shapes to reveal it."""

    class _Wrapped(_FKStub):
        def bodies(self, dof, *, lift_z: float = 0.0):
            pos, rot = super().bodies(dof, lift_z=lift_z)
            rot[wb_geom.LEFT_BODY] = Rotation.from_euler("xyz", [3.0, 0.0, 0.0]).as_matrix()
            return pos, rot

    with caplog.at_level("WARNING"):
        wb_geom.geom_series(_Wrapped(), np.ones((2, JOINT_DIM)), np.ones(2, bool), ("rpy",),
                            log_every=0)
    assert any("left_roll" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The jitter
# ---------------------------------------------------------------------------
def test_noise_sigma_lands_on_the_geometric_columns_only():
    lay = get_layout("wholebody_geom_xy_rpy")
    sigma = wb_geom.noise_sigma(lay, xy_m=0.01, rpy_rad=0.05)
    assert sigma.shape == (lay.state_dim,)
    np.testing.assert_allclose(sigma[lay.state_geom_slots()["rel_xy"]], 0.01)
    np.testing.assert_allclose(sigma[lay.state_geom_slots()["rpy"]], 0.05)
    # The grippers are a direct encoder measurement with no posture gap to absorb, and they are
    # the one state channel the action also predicts. Exactly zero.
    for role in ("left", "right"):
        assert sigma[lay.state_slots()[role]["gripper"]] == 0.0


@pytest.mark.parametrize("mode", ("wholebody", "wholebody_geom_none", "both_hands_head_dual"))
def test_noise_sigma_is_empty_for_a_mode_with_no_geometry(mode):
    sigma = wb_geom.noise_sigma(get_layout(mode), xy_m=0.01, rpy_rad=0.05)
    assert not np.any(sigma), mode


def test_state_noise_jitters_the_geometry_and_not_the_grippers():
    lay = get_layout("wholebody_geom_xy_rpy")
    sigma = wb_geom.noise_sigma(lay, xy_m=0.02, rpy_rad=0.1)
    base_state = torch.arange(lay.state_dim, dtype=torch.float32)

    class _Base(torch.utils.data.Dataset):
        def __len__(self):
            return 4096

        def __getitem__(self, idx):
            return {"state": base_state.clone(), "tag": idx}

    noisy = StateNoise(_Base(), sigma)
    torch.manual_seed(0)
    drawn = torch.stack([noisy[i]["state"] for i in range(4096)])
    delta = drawn - base_state
    grip = [lay.state_slots()[r]["gripper"] for r in ("left", "right")]
    assert torch.all(delta[:, grip] == 0.0)          # exactly, not approximately
    geom = [d for d in range(lay.state_dim) if d not in grip]
    assert torch.all(delta[:, geom].std(0) > 0)
    # Zero-mean, and at the requested scale per block (loose bounds: 4096 draws).
    np.testing.assert_allclose(delta[:, geom].mean(0).numpy(), 0.0, atol=0.01)
    np.testing.assert_allclose(delta[:, geom].std(0).numpy(),
                               np.asarray(sigma)[geom], rtol=0.1)
    assert noisy[7]["tag"] == 7                       # the rest of the sample passes through


def test_state_noise_refuses_a_no_op_and_a_width_mismatch():
    lay = get_layout("wholebody_geom_xy_rpy")

    class _Base(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {"state": torch.zeros(lay.state_dim)}

    # An all-zero sigma would be a silent no-op — i.e. "the flags never reached the layout".
    with pytest.raises(ValueError, match="no-op"):
        StateNoise(_Base(), np.zeros(lay.state_dim))
    with pytest.raises(ValueError, match="different layout"):
        StateNoise(_Base(), np.full(lay.state_dim + 3, 0.01))[0]


@pytest.mark.parametrize("mode,kw", [
    ("wholebody", {"state_noise_xy_m": 0.01}),
    ("wholebody", {"state_noise_rpy_deg": 2.0}),
    ("wholebody_geom_xy", {"state_noise_rpy_deg": 2.0}),
    ("wholebody_geom_rpy", {"state_noise_xy_m": 0.01}),
    ("wholebody_geom_none", {"state_noise_xy_m": 0.01}),
    ("both_hands_head_dual", {"state_noise_xy_m": 0.01}),
])
def test_a_sigma_the_mode_cannot_carry_is_refused(mode, kw):
    """Refused, not ignored: "the ablation ran without its noise" is invisible in the loss, in
    the checkpoint's widths and in W&B."""
    with pytest.raises(ValueError, match="state_noise"):
        TrainConfig(algo="diffusion", pipeline="hl",
                    hl_data=HLDataConfig(mode=mode, norm_stats_path="s.json", **kw))


@pytest.mark.parametrize("kw", [{"state_noise_xy_m": -0.01}, {"state_noise_rpy_deg": -1.0},
                                {"state_noise_xy_m": float("nan")}])
def test_a_negative_or_non_finite_sigma_is_refused(kw):
    with pytest.raises(ValueError, match="state_noise"):
        TrainConfig(algo="diffusion", pipeline="hl",
                    hl_data=HLDataConfig(mode="wholebody_geom_xy_rpy",
                                         norm_stats_path="s.json", **kw))


# ---------------------------------------------------------------------------
# End to end through HLDataset, with the FK stubbed out
# ---------------------------------------------------------------------------
@pytest.fixture()
def wb_zarr(tmp_path, dual_zarr):  # noqa: F811
    out, arrays = _augment(dual_zarr, str(tmp_path / "dual_wb.zarr.zip"))
    return out, arrays


@pytest.fixture()
def stub_fk(monkeypatch):
    """Inject ``_FKStub`` so the dataset needs no URDF, yourdfpy or submodule."""
    monkeypatch.setattr("fuji.s1_policy.data.hl_dataset.load_geom_fk",
                        lambda **kw: (_FKStub(), ""))


@pytest.mark.parametrize("mode,state_dim", [("wholebody_geom_xy_rpy", 12),
                                            ("wholebody_geom_xy", 6),
                                            ("wholebody_geom_rpy", 8),
                                            ("wholebody_geom_none", 2)])
def test_the_sample_carries_the_geometry_and_the_same_43_dim_action(wb_zarr, stub_fk, mode,
                                                                   state_dim):
    path, arrays = wb_zarr
    ds = _wb_ds(path, mode=mode)
    sample = ds[0]
    lay = get_layout(mode)
    assert sample["state"].shape == (state_dim,)
    assert sample["action_chunk"].shape[-1] == 43
    # The action is untouched by the state choice: still the exporter's own absolute joints.
    fi, frame = ds._index[0]
    np.testing.assert_allclose(sample["action_chunk"].numpy()[:, :JOINT_DIM],
                               arrays["wb_action_joints"][frame][:sample["action_chunk"].shape[0]],
                               atol=1e-5)
    # ...and the geometry is the FK of THIS frame's own state joints, block by block.
    want = wb_geom.geom_from_joints(_FKStub(), arrays["wb_state_joints"][frame],
                                   tuple(b for b in lay.state_geom if b in wb_geom.FK_BLOCKS))
    got = sample["state"].numpy()[:want.size]
    np.testing.assert_allclose(got, want, atol=1e-5)
    ds.close()


def test_an_uncovered_anchor_frame_masks_the_geometry(tmp_path, dual_zarr, stub_fk):  # noqa: F811
    """Zero geometry is "both hands at the pelvis" — plausible, and fabricated. It must be
    flagged, exactly as the raw joints are."""
    out, _ = _augment(dual_zarr, str(tmp_path / "wb.zarr.zip"))
    from tests.test_diffusion_umi_wholebody import _patch_zarr, _read_array

    sv = _read_array(out, "wb_state_valid").copy()
    sv[:] = 0.0
    patched = _patch_zarr(out, arrays={"wb_state_valid": sv})
    ds = _wb_ds(patched, mode="wholebody_geom_xy_rpy", drop_glitch_windows=False)
    sample = ds[0]
    lay = get_layout("wholebody_geom_xy_rpy")
    valid = sample["state_valid"].numpy()
    for block, slot in lay.state_geom_slots().items():
        assert not valid[slot].any(), block
    assert np.all(sample["state"].numpy()[:10] == 0.0)
    ds.close()


def test_a_missing_fk_is_refused_by_name_not_zero_filled(wb_zarr, monkeypatch):
    """The geometry IS this mode's proprioception, so an unavailable FK must stop the run. A
    zero block would be a plausible posture that no shape, mask or metric would flag."""
    path, _ = wb_zarr
    monkeypatch.setattr("fuji.s1_policy.data.hl_dataset.load_geom_fk",
                        lambda **kw: (None, "yourdfpy is unavailable (test)"))
    with pytest.raises(ValueError, match="yourdfpy is unavailable"):
        _wb_ds(path, mode="wholebody_geom_xy_rpy")
    # `wholebody` and `wholebody_geom_none` need no FK, and must still load.
    for mode in ("wholebody", "wholebody_geom_none"):
        ds = _wb_ds(path, mode=mode)
        assert ds[0]["state"].shape == (get_layout(mode).state_dim,)
        ds.close()


def test_the_dataset_knobs_distinguish_the_modes(wb_zarr, stub_fk):
    """`mode` is in the EXACT-equality provenance, so a norm-stats file fitted for one state
    composition cannot be reused for another — the widths differ, but so do the UNITS, and the
    trainer's shape assertion alone would pass for two modes of equal width."""
    path, _ = wb_zarr
    seen = {}
    for mode in WB_MODES:
        ds = _wb_ds(path, mode=mode)
        seen[mode] = ds.split_provenance(1)["dataset_knobs"]["mode"]
        ds.close()
    assert seen == {m: m for m in WB_MODES}
