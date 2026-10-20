"""Contract tests for ``diffusion_umi_3tracker`` — the 3-tracker / 2-camera / 2-F/T /
2-gripper extension of the ``diffusion umi`` preset (layout ``both_hands_head_dual``).

Every check here pins something that would otherwise fail SILENTLY — with correct shapes
and a healthy-looking loss — because the extension is deliberately shaped so that nothing
about the tensor geometry announces a mistake:

* the action vector is 41 dims in BOTH ``both_hands_head`` and ``both_hands_head_dual``,
  so only the mask distinguishes "the right hand has no sensor" from "it has one";
* the two cameras arrive flattened into one list alongside any history frames, so a
  transposed order is a valid tensor;
* the two wrenches are stacked into one 12-channel window, so a mis-split routes half of
  one hand's wrench into the other hand's encoder at the right width.

The left-only preset must stay untouched, so the "no regression" direction is asserted
too — new mode, new ``--hands`` value, byte-identical single-camera provenance, and a
force-encoder ``state_dict`` that still matches the shipped checkpoints.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flore.s1_policy.data.layout import ACTION_HORIZON, FORCE_DIM, FORCE_HIST_LEN, get_layout
from flore.s1_policy.model.force_encoder import ForceHistoryEncoder
from flore.s1_policy.training.config import HLDataConfig, TrainConfig
from flore.s1_policy.training.trainer import collate_hl_batch

# Reuse the synthetic-zarr BUILDER rather than a second copy of it: a drifting second schema
# writer is how a loader test starts passing against data no converter produces. The fixtures
# are re-declared here (not imported) because importing a pytest fixture shadows the name and
# ruff flags every use of it as F811.
from tests.test_hl_dataset import _ds, _write_hl_zarr

MODE = "both_hands_head_dual"
CAMS = ("camera0_rgb", "camera1_rgb")


@pytest.fixture()
def dual_zarr(tmp_path):
    """The 3-tracker rig: 3 pucks, 2 cameras, 2 F/T sensors, 2 grippers."""
    return _write_hl_zarr(
        tmp_path / "dual.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left", "right"),
        gripper_roles=("left", "right"),
        n_cameras=2,
        attrs_override={"hands_mode": "both_head_dual"},
    )


@pytest.fixture()
def left_zarr(tmp_path):
    """The shipped single-hand, single-camera contract, for the no-regression direction."""
    return _write_hl_zarr(tmp_path / "left.zarr.zip")


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


def _model(cfg_diffusion):
    from flore.s1_policy.model.dp_diffusion import DPDiffusionModel

    cfg_diffusion.pretrained = False
    return DPDiffusionModel(cfg_diffusion, encoder=TinyEncoder())


# ---------------------------------------------------------------------------
# Widths: every one is DERIVED, so a disagreement is impossible by construction
# ---------------------------------------------------------------------------
def test_config_pushes_every_width_from_the_layout_and_the_camera_list():
    """action/state/force/camera widths come from the layout + image_keys, never the CLI.

    ``num_cameras`` is the newly pushed one and it is the load-bearing case: the vision
    encoder's contribution to ``global_cond`` is ``num_cameras * To * features``, so a
    ``num_cameras`` that disagreed with the number of keys the dataset stacks would build a
    model of the wrong width. It would be caught — the decoder's cond projection refuses —
    but the message would read as a model-config typo rather than a data/model mismatch.
    """
    d = _train_config().diffusion
    assert (d.action_dim, d.action_horizon) == (41, 50)
    assert d.state_dim == 20
    assert d.num_cameras == 2
    assert d.force_hist_channels == 12
    assert d.layout_mode == MODE
    # ...and one camera stays one camera.
    assert _train_config(image_keys=()).diffusion.num_cameras == 1
    assert _train_config(image_keys=("camera0_rgb",)).diffusion.num_cameras == 1


def test_resolved_image_keys_is_the_single_rule_both_sides_read():
    """``num_cameras`` and the dataset's stack must come from ONE resolver, or a run can
    build a 2-camera model over a 1-camera stack (and vice versa)."""
    assert HLDataConfig().resolved_image_keys() == ("camera0_rgb",)
    assert HLDataConfig(image_key="camX_rgb").resolved_image_keys() == ("camX_rgb",)
    assert HLDataConfig(image_keys=CAMS).resolved_image_keys() == CAMS
    # image_keys WINS when both are set — otherwise the pushed num_cameras (from the list)
    # and the dataset's first key (from image_key) could describe different rigs.
    cfg = HLDataConfig(image_key="ignored_rgb", image_keys=CAMS)
    assert cfg.resolved_image_keys() == CAMS


# ---------------------------------------------------------------------------
# Dataset: what the two extra cameras and the second sensor actually produce
# ---------------------------------------------------------------------------
def test_dataset_emits_two_distinct_camera_frames_in_key_order(dual_zarr):
    """``image`` is ``(n_cams, C, H, W)`` in image_keys ORDER, and the two entries are
    different pictures. A dataset that read camera0 twice would produce the right shape and
    a model that trains fine on half its conditioning."""
    ds = _ds(dual_zarr, mode=MODE, image_keys=CAMS, require_episode_ends=True)
    try:
        s = ds[len(ds) // 2]
        assert tuple(s["image"].shape) == (2, 3, 8, 8)
        assert not torch.allclose(s["image"][0], s["image"][1])
        # ORDER: entry i must be image_keys[i]. Compared against a reversed-key dataset,
        # because the only way to be sure of the order is to swap it and see it swap.
        rev = _ds(dual_zarr, mode=MODE, image_keys=CAMS[::-1], require_episode_ends=True)
        try:
            r = rev[len(ds) // 2]
            assert torch.allclose(s["image"][0], r["image"][1])
            assert torch.allclose(s["image"][1], r["image"][0])
        finally:
            rev.close()
    finally:
        ds.close()


def test_single_camera_sample_is_bit_identical_to_the_pre_multicamera_contract(left_zarr):
    """The left-only preset must be untouched: one camera still collapses the leading axis
    to ``(C, H, W)`` and every tensor matches what ``image_key`` alone produced."""
    a = _ds(left_zarr, mode="left_only", require_episode_ends=True)
    b = _ds(left_zarr, mode="left_only", image_keys=("camera0_rgb",), require_episode_ends=True)
    try:
        assert len(a) == len(b)
        sa, sb = a[3], b[3]
        assert tuple(sa["image"].shape) == (3, 8, 8)      # NO leading camera axis
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k
        # force_history_valid stays a SCALAR for a one-sensor layout.
        assert sa["force_history_valid"].ndim == 0
        # ...and the provenance dict gains no key, so every existing norm_stats.json is
        # still accepted by trainer.validate_split_provenance (exact dict equality).
        knobs = a.split_provenance(1)["dataset_knobs"]
        assert "image_keys" not in knobs
        assert knobs["image_key"] == "camera0_rgb"
    finally:
        a.close()
        b.close()


def test_multi_camera_provenance_records_the_full_list(dual_zarr):
    """With >1 camera the list IS recorded, so a 2-camera run cannot be fed norm stats
    fitted over 1 camera (the widths would match; only the anchors' images differ)."""
    ds = _ds(dual_zarr, mode=MODE, image_keys=CAMS, require_episode_ends=True)
    try:
        knobs = ds.split_provenance(1)["dataset_knobs"]
        assert knobs["image_keys"] == list(CAMS)
        assert knobs["image_key"] == CAMS[0]
    finally:
        ds.close()


def test_a_missing_second_camera_is_named_not_silently_dropped(left_zarr):
    with pytest.raises(ValueError, match=r"missing image key\(s\) \['camera1_rgb'\]"):
        _ds(left_zarr, mode="left_only", image_keys=CAMS)


def test_a_duplicated_camera_key_is_refused(dual_zarr):
    """Two entries reading the same array would double-feed one view while the model
    believed it had two — invisible in every shape."""
    with pytest.raises(ValueError, match="duplicates"):
        _ds(dual_zarr, mode=MODE, image_keys=("camera0_rgb", "camera0_rgb"))


def test_both_hands_force_and_gripper_are_supervised_and_not_crossed(dual_zarr):
    """The 7 columns the left-only modes masked off (right grip + right force) are now
    trained, and each hand's values land in ITS OWN block.

    The fixture encodes ``base + frame + step + channel`` with base 0 for left and 100 for
    right, so a crossed block is a ~100 value error rather than a plausible number.
    """
    lay = get_layout(MODE)
    ds = _ds(dual_zarr, mode=MODE, image_keys=CAMS, require_episode_ends=True)
    try:
        s = ds[5]
        act = s["action_chunk"].numpy()
        mask = s["action_mask"].numpy()
        assert act.shape == (50, 41) and mask.shape == (50, 41)
        # Nothing is reserved-but-unmeasured any more.
        assert lay.unmeasured_dims() == ()
        for role, base in (("left", 0.0), ("right", 100.0)):
            sl = lay.slots(role)
            assert mask[:, sl.gripper].all(), role
            assert mask[:, sl.force].all(), role
            f = act[:, sl.force]
            # base + frame*1.0 dominates; left is O(1), right is O(100).
            assert abs(f.min() - base) < 50.0, (role, f.min())
        # The head has neither, in both modes.
        head = lay.slots("head")
        assert head.gripper is None and head.force is None
        # force_history: 12 channels, left block then right block, in force_roles order.
        fh = s["force_history"].numpy()
        assert fh.shape == (1, 12)
        assert fh[:, :FORCE_DIM].max() < 50.0        # left ~ O(1)
        assert fh[:, FORCE_DIM:].min() > 50.0        # right ~ O(100)
    finally:
        ds.close()


def test_force_history_valid_is_per_sensor(dual_zarr, tmp_path):
    """One hand's unmeasured window must not discard the other hand's measured one.

    With a single shared bool the encoder replaces the WHOLE 12-channel encoding with its
    learned "no data" embedding, so a right-hand dropout would delete every left-hand force
    observation on those anchors — silently, since a stale hold and a real zero are
    numerically identical after a free-space tare.
    """
    ds = _ds(dual_zarr, mode=MODE, image_keys=CAMS, require_episode_ends=True)
    try:
        v = ds[7]["force_history_valid"]
        assert tuple(v.shape) == (2,) and bool(v.all())
    finally:
        ds.close()

    # Now the same rig with ONLY the right sensor's window unmeasured.
    path = _write_hl_zarr(
        tmp_path / "right_ft_dead.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left", "right"),
        gripper_roles=("left", "right"),
        n_cameras=2,
        invalid_force_history_roles=("right",),
        attrs_override={"hands_mode": "both_head_dual"},
    )
    ds2 = _ds(path, mode=MODE, image_keys=CAMS, require_episode_ends=True)
    try:
        v = ds2[7]["force_history_valid"].numpy()
        assert v.tolist() == [True, False], v
        # ...and the anchor is STILL trainable (drop_invalid_force_history defaults False),
        # which is the point: half a measurement is not none.
        assert len(ds2) > 0
    finally:
        ds2.close()


# ---------------------------------------------------------------------------
# Collate + model
# ---------------------------------------------------------------------------
def test_collate_splits_the_camera_axis_into_the_models_images_list(dual_zarr):
    ds = _ds(dual_zarr, mode=MODE, image_keys=CAMS, require_episode_ends=True)
    try:
        batch = collate_hl_batch([ds[i] for i in range(4)])
        assert len(batch["images"]) == 2
        assert all(tuple(t.shape) == (4, 3, 8, 8) for t in batch["images"])
        assert len(batch["img_masks"]) == 2
        assert tuple(batch["force_history_valid"].shape) == (4, 2)
        assert tuple(batch["actions"].shape) == (4, 50, 41)
        assert tuple(batch["state"].shape) == (4, 20)
        # Entry i of the list must still be camera i (collate must not reorder).
        for i in range(2):
            assert torch.equal(batch["images"][i][0], ds[0]["image"][i])
    finally:
        ds.close()


def test_model_builds_one_force_encoder_group_per_declared_sensor():
    """The group count is DERIVED from ``layout_mode``, not configured, so 12 channels can
    never be built as one ungatable group."""
    m = _model(_train_config().diffusion)
    assert m.force_encoder.groups == 2
    assert m.force_encoder.group_channels == FORCE_DIM
    assert m.force_encoder.total_out_dim == 2 * m.config.force_enc_out_dim
    # The single-sensor modes still build exactly one.
    single = _train_config(mode="both_hands_head", image_keys=("camera0_rgb",)).diffusion
    assert _model(single).force_encoder.groups == 1


def test_both_force_groups_receive_gradient():
    """If only group 0 were wired, the RIGHT wrench would never reach ``global_cond`` and
    the run would look completely healthy (the force COLUMNS are still supervised, so
    ``diffusion_loss/force`` would just be worse than it should be)."""
    m = _model(_train_config().diffusion)
    b = _fake_batch()
    m(b)["loss"].backward()
    grads = dict(m.force_encoder.named_parameters())
    for name in ("conv.1.weight", "extra_groups.0.conv.1.weight"):
        g = grads[name].grad
        assert g is not None and float(g.norm()) > 0.0, name


def test_per_group_metrics_report_both_hands_force_and_grip():
    """A dual run must LOG the right hand's contribution or it is invisible on the
    dashboard while the loss optimizes it."""
    m = _model(_train_config().diffusion)
    out = m(_fake_batch())
    names = set(out["metrics"])
    for want in ("diffusion_loss/left", "diffusion_loss/right", "diffusion_loss/head",
                 "diffusion_loss/grip", "diffusion_loss/force"):
        assert want in names, (want, sorted(names))


def test_sampling_returns_the_full_41_dim_chunk():
    m = _model(_train_config().diffusion)
    with torch.no_grad():
        a = m.sample_actions(_fake_batch())
    assert tuple(a.shape) == (2, 50, 41)
    assert torch.isfinite(a).all()


def _fake_batch(b: int = 2) -> dict:
    return {
        "images": [torch.randn(b, 3, 224, 224), torch.randn(b, 3, 224, 224)],
        "img_masks": [torch.ones(b, dtype=torch.bool)] * 2,
        "state": torch.randn(b, 20),
        "state_valid": torch.ones(b, 20, dtype=torch.bool),
        "force_history": torch.randn(b, 1, 12),
        "force_history_valid": torch.tensor([[True, True], [True, False]]),
        "actions": torch.randn(b, 50, 41),
        "action_mask": torch.ones(b, 50, 41, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Force encoder: the shipped checkpoints must keep loading
# ---------------------------------------------------------------------------
def test_one_group_state_dict_is_unchanged_and_group_zero_is_name_compatible():
    """``groups=1`` must be byte-compatible with the shipped single-hand checkpoints, and a
    two-group encoder's LEFT branch must keep those same names — otherwise the left hand's
    trained force weights are not even a candidate initialization."""
    one = ForceHistoryEncoder(hist_len=1, channels=6)
    assert sorted(one.state_dict()) == [
        "conv.1.bias", "conv.1.weight", "conv.4.bias", "conv.4.weight",
        "invalid_embed", "readout.bias", "readout.weight",
    ]
    two = ForceHistoryEncoder(hist_len=1, channels=12, groups=2)
    missing, unexpected = two.load_state_dict(one.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("extra_groups.") for k in missing), missing


def test_a_group_reads_only_its_own_channels():
    """A mis-split would route half of one hand's wrench into the other's encoder at
    exactly the right width."""
    enc = ForceHistoryEncoder(hist_len=1, channels=12, groups=2)
    base = torch.zeros(1, 1, 12)
    left_only = base.clone()
    left_only[0, 0, :6] = 1.0
    right_only = base.clone()
    right_only[0, 0, 6:] = 1.0
    with torch.no_grad():
        o_base, o_l, o_r = enc(base), enc(left_only), enc(right_only)
    # Changing the LEFT channels moves only the first out_dim slice, and vice versa.
    assert not torch.allclose(o_l[:, :64], o_base[:, :64])
    assert torch.allclose(o_l[:, 64:], o_base[:, 64:])
    assert torch.allclose(o_r[:, :64], o_base[:, :64])
    assert not torch.allclose(o_r[:, 64:], o_base[:, 64:])


def test_uneven_channel_split_is_refused():
    with pytest.raises(ValueError, match="not divisible"):
        ForceHistoryEncoder(hist_len=1, channels=12, groups=5)


def test_a_wrong_width_validity_flag_is_refused_not_broadcast():
    enc = ForceHistoryEncoder(hist_len=1, channels=12, groups=2)
    with pytest.raises(ValueError, match=r"must be \(B,\) or \(B, 2\)"):
        enc(torch.randn(3, 1, 12), torch.ones(3, 3, dtype=torch.bool))


# ---------------------------------------------------------------------------
# Cross-mode refusals: the two 41-dim modes must never be confused
# ---------------------------------------------------------------------------
def test_a_dual_file_cannot_be_loaded_as_the_single_sensor_mode(dual_zarr):
    """Both modes have 41 action columns and the same three pucks, so ONLY the stamped
    hands_mode distinguishes them. Loading a dual file as ``both_hands_head`` would drop
    both extra sensors AND read the head's state block one column early."""
    with pytest.raises(ValueError, match=r"--hands 'both_head_dual' but loading as"):
        _ds(dual_zarr, mode="both_hands_head", image_keys=CAMS)


def test_a_single_sensor_file_cannot_be_loaded_as_the_dual_mode(tmp_path):
    path = _write_hl_zarr(
        tmp_path / "three.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        n_cameras=2,
        attrs_override={"hands_mode": "both_head"},
    )
    with pytest.raises(ValueError, match=r"--hands 'both_head' but loading as "
                                            r"mode='both_hands_head_dual'"):
        _ds(path, mode=MODE, image_keys=CAMS)


def _dual_hl_data(path) -> dict:
    """The ``hl_data`` block a diffusion_umi_3tracker checkpoint carries."""
    return {
        "zarr_paths": [str(path)],
        "mode": MODE,
        "image_key": CAMS[0],
        "image_keys": list(CAMS),
        "image_mean": 0.5,
        "image_std": 0.5,
        "obs_history_len": 1,
        "obs_history_dt_s": 0.25,
        "drop_invalid_force_history": False,
        "drop_glitch_windows": True,
        "norm_stats_path": "",
        "val_last_episodes": 0,
    }


def test_offline_inference_carries_both_cameras_into_the_observation(dual_zarr):
    """The offline session must feed the model AS MANY views as it was trained on.

    This is the direction that fails silently: handing a 2-camera checkpoint only
    ``camera0_rgb`` produces a shape-correct observation and a plausible-looking rollout,
    with the second ViT tower reading a repeat of the first. So the assertion is on the
    COUNT of decoded conditioning images, not merely on the source loading."""
    from flore.s1_policy.inference.common import HLSampleSource

    source = HLSampleSource(_dual_hl_data(dual_zarr))
    try:
        assert source.image_keys == CAMS
        assert source.dataset.image_keys == CAMS
        _, sample = source.get(0)
        # camera-major, one history frame -> exactly 2 images, in image_keys order
        assert sample["image"].shape[0] == len(CAMS)
    finally:
        source.close()


def test_serving_still_refuses_a_multi_camera_checkpoint_by_name():
    """Offline scoring and SERVING are separate milestones, and only serving is blocked:
    the live planner subscribes to one D405, so its num_cameras pin must stay explicit
    rather than following the checkpoint the way the dataset-backed tools now do."""
    from flore.s1_policy.data.layout import get_layout as _get_layout
    from flore.s1_policy.inference.common import validate_model_layout
    from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig

    layout = _get_layout(MODE)
    hl_data = _dual_hl_data("unused.zarr.zip")
    config = DPDiffusionConfig(
        layout_mode=MODE,
        action_dim=layout.action_dim,
        action_horizon=ACTION_HORIZON,
        state_dim=layout.state_dim,
        obs_history_len=1,
        num_cameras=len(CAMS),
        force_hist_channels=layout.force_input_channels,
        force_hist_len=FORCE_HIST_LEN,
    )
    # dataset-backed default: derived from the checkpoint, so it is ACCEPTED
    validate_model_layout(config, hl_data, layout)
    # serving pins 1 and must refuse, naming the second view as the blocker
    with pytest.raises(ValueError, match=r"num_cameras=2 disagrees .*plumbed"):
        validate_model_layout(config, hl_data, layout, num_cameras=1)


def test_a_pre_multicamera_checkpoint_still_loads_its_hl_data():
    """``image_keys`` is OPTIONAL in the allow-list: it did not exist when every shipped
    checkpoint was written, and a bidirectional membership check would refuse all of them
    for lacking a key they could not have had."""
    from flore.s1_policy.inference.common import HLSampleSource

    hl_data = {
        "zarr_paths": [],            # forces a later, DIFFERENT error
        "mode": "left_only",
        "image_key": "camera0_rgb",
        "image_mean": 0.5,
        "image_std": 0.5,
        "obs_history_len": 1,
        "obs_history_dt_s": 0.25,
        "drop_invalid_force_history": False,
        "drop_glitch_windows": True,
        "norm_stats_path": "",
        "val_last_episodes": 4,
    }
    with pytest.raises(ValueError, match="no HL zarr paths"):
        HLSampleSource(hl_data)
