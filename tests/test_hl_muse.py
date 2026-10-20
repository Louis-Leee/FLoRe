"""Pure tests for the ``--algo muse`` shim: the wire codec, the native-layout mapping, the
relative-pose convention shared with UMI, and the planner's extra observation ring.

No uvaforce env, no UVA model, no subprocess: everything here is the FLoRe half, which is
exactly the half that can silently disagree with the backend. The one thing that cannot be
tested without the other env — that ``sample_tokens`` returns what we think — is guarded at
runtime instead (shape checks in the backend, ``native_chunk_from_uva``'s width check here).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flore.s1_policy.data.layout import ACTION_DT, FORCE_DIM, get_layout
from flore.s1_policy.data.transforms import (
    GRIPPER_WIDTH_REF_M,
    absolute_to_delta_ee,
    delta_ee_to_absolute,
)
from flore.s1_policy.serving.hl_muse import (
    MUSE_ALGO,
    MUSE_EEF_POS_REL_KEY,
    MUSE_EEF_ROT6D_REL_KEY,
    MUSE_IMAGE_MEAN,
    MUSE_IMAGE_STD,
    UVA_ACTION_DIM,
    MuseConfig,
    MuseLivePlanner,
    MuseMeta,
    MuseRemoteModel,
    native_chunk_from_uva,
)
from flore.s1_policy.serving.hl_online import LoadedHLModel, decode_chunk
from flore.utils import muse_wire as wire
from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

LAYOUT = get_layout("left_only")


# ---------------------------------------------------------------------------
# Wire codec — the ONE piece both interpreters must agree on byte for byte.
# ---------------------------------------------------------------------------
def test_wire_roundtrips_meta_and_arrays():
    line = wire.pack(wire.TAG_CHUNK, {"kind": "live", "seed": 7},
                     action_rel=np.arange(12, dtype=np.float64).reshape(4, 3),
                     force_pred=np.zeros((4, 6), dtype=np.float32))
    tag, meta, arrays = wire.unpack(line)
    assert tag == wire.TAG_CHUNK
    assert meta == {"kind": "live", "seed": 7}
    np.testing.assert_array_equal(arrays["action_rel"], np.arange(12).reshape(4, 3))
    assert arrays["force_pred"].dtype == np.float32  # dtypes survive; no silent upcast


def test_wire_request_lines_are_untagged_and_distinguishable():
    """A request has NO prefix and a reply does. That is what lets the frontend tell protocol
    lines apart from the model's own stdout chatter with a cheap prefix test."""
    req = wire.pack("", {"kind": wire.KIND_ZARR, "episode": 3})
    assert not wire.is_message(req)
    assert wire.unpack(req)[0] == ""
    assert wire.is_message(wire.pack(wire.TAG_READY, {}))
    for chatter in ("", "  ", "Loading VAE...", "[hydra] compose", "0.123"):
        assert not wire.is_message(chatter)


def test_wire_rejects_malformed_lines_rather_than_half_decoding():
    with pytest.raises(ValueError, match="empty"):
        wire.unpack("   ")
    with pytest.raises(ValueError, match="no payload"):
        wire.unpack(wire.LINE_PREFIX + "ready")
    with pytest.raises(ValueError, match="base64"):
        wire.unpack(wire.LINE_PREFIX + "ready not-base64!!")
    with pytest.raises(ValueError, match="reserved"):
        wire.pack("x", {}, **{wire.META_KEY: np.zeros(1)})


# ---------------------------------------------------------------------------
# The relative-pose convention. hl_muse's whole design rests on FLoRe's
# absolute_to_delta_ee being the SAME operation as UMI's convert_pose_mat_rep('relative'),
# and on both spelling rot6d as the first two ROWS. Pin it numerically.
# ---------------------------------------------------------------------------
_UVA_ROOT = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "third_party" / "unified_video_action"
)


def _load_by_path(name: str, relative: str):
    """Import one UPSTREAM module by file path, bypassing its package.

    By path, not ``import unified_video_action...``: those packages' ``__init__`` chains pull in
    torch 2.7 / hydra / pytorch3d, none of which exist in FLoRe's ``.venv``. The two modules we
    want are numpy(+scipy)-only, and testing against the REAL upstream functions rather than an
    inlined transcription is the whole point — a transcription cannot catch upstream changing.
    """
    import importlib.util

    path = _UVA_ROOT / relative
    if not path.is_file():
        pytest.skip(f"submodule file not present: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mat(pos: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot6d_to_rotmat(rot6d)
    out[:3, 3] = pos
    return out


def test_flore_relative_equals_umi_convert_pose_mat_rep():
    """FLoRe's ``absolute_to_delta_ee`` == UMI's ``convert_pose_mat_rep(..., 'relative')``.

    This is the load-bearing assumption of the whole shim: the backend returns UVA's relative
    chunk untouched and FLoRe's ``decode_chunk`` un-relativizes it. If the two conventions
    differed, every shape check would still pass and the arm would go somewhere else.
    """
    pose_util = _load_by_path("umi_pose_util", "umi/common/pose_util.py")
    repr_util = _load_by_path(
        "uva_pose_repr_util", "unified_video_action/common/pose_repr_util.py")

    rng = np.random.default_rng(0)
    for _ in range(20):
        ref_pos = rng.normal(size=3)
        ref_rot = rotmat_to_rot6d(_random_rotation(rng))
        pos = rng.normal(size=(5, 3))
        rot = np.stack([rotmat_to_rot6d(_random_rotation(rng)) for _ in range(5)])

        flore_pos, flore_rot = absolute_to_delta_ee(pos, rot, ref_pos, ref_rot)

        base = _mat(ref_pos, ref_rot)
        poses = np.stack([_mat(pos[i], rot[i]) for i in range(5)])
        umi10d = pose_util.mat_to_pose10d(
            repr_util.convert_pose_mat_rep(
                poses, base_pose_mat=base, pose_rep="relative", backward=False))

        np.testing.assert_allclose(flore_pos, umi10d[:, :3], atol=1e-12)
        np.testing.assert_allclose(flore_rot, umi10d[:, 3:9], atol=1e-12)

        # ...and the inverse: UMI's backward pass == FLoRe's delta_ee_to_absolute.
        back = pose_util.mat_to_pose10d(
            repr_util.convert_pose_mat_rep(
                pose_util.pose10d_to_mat(umi10d), base_pose_mat=base,
                pose_rep="relative", backward=True))
        flore_back_pos, flore_back_rot = delta_ee_to_absolute(
            flore_pos, flore_rot, ref_pos, ref_rot)
        np.testing.assert_allclose(flore_back_pos, back[:, :3], atol=1e-10)
        np.testing.assert_allclose(flore_back_rot, back[:, 3:9], atol=1e-10)
        np.testing.assert_allclose(flore_back_pos, pos, atol=1e-10)


def test_rot6d_is_the_first_two_rows_not_columns():
    """The classic trap. If either side used COLUMNS, every rotation would be transposed and
    the arm would rotate the wrong way while every shape check still passed."""
    rng = np.random.default_rng(1)
    r = _random_rotation(rng)
    np.testing.assert_allclose(rotmat_to_rot6d(r), r[:2, :].reshape(6), atol=1e-12)
    assert not np.allclose(rotmat_to_rot6d(r), r[:, :2].T.reshape(6), atol=1e-6)


def _random_rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# ---------------------------------------------------------------------------
# UVA (H,10) + (H,6) -> FLoRe native (H,16)
# ---------------------------------------------------------------------------
def test_native_chunk_maps_every_uva_column_to_its_slot():
    rng = np.random.default_rng(2)
    horizon = 32
    action = rng.normal(size=(horizon, UVA_ACTION_DIM))
    action[:, 9] = np.linspace(0.02, 0.09, horizon)  # gripper, METRES
    force = rng.normal(size=(horizon, FORCE_DIM)) * 5.0

    native = native_chunk_from_uva(action, force, LAYOUT, predict_force=True)
    slots = LAYOUT.slots("left")
    assert native.shape == (horizon, LAYOUT.action_dim)
    np.testing.assert_allclose(native[:, slots.pos], action[:, 0:3])
    np.testing.assert_allclose(native[:, slots.rot6d], action[:, 3:9])
    np.testing.assert_allclose(native[:, slots.force], force)
    # The gripper is the ONE converted channel: UVA trains metres, the native layout carries a
    # fraction of GRIPPER_WIDTH_REF_M. decode_chunk multiplies it back, so this must round trip.
    np.testing.assert_allclose(native[:, slots.gripper], action[:, 9] / GRIPPER_WIDTH_REF_M)


def test_native_chunk_gripper_round_trips_through_decode_chunk():
    horizon = 8
    widths = np.linspace(0.044, 0.094, horizon)
    action = np.zeros((horizon, UVA_ACTION_DIM))
    action[:, 3:9] = rotmat_to_rot6d(np.eye(3))
    action[:, 9] = widths
    native = native_chunk_from_uva(action, np.zeros((horizon, FORCE_DIM)), LAYOUT,
                                  predict_force=True)
    result = decode_chunk(native, LAYOUT, np.zeros((1, 3)),
                          np.tile(rotmat_to_rot6d(np.eye(3)), (1, 1)),
                          force_enabled=False, gate_n=0.0)
    np.testing.assert_allclose(result.role("left").gripper_m, widths, atol=1e-12)


def test_native_chunk_position_round_trips_through_decode_chunk():
    """UVA's relative action decoded by FLoRe's decode_chunk must land where UMI's own
    ``backward=True`` un-relativization would put it."""
    rng = np.random.default_rng(3)
    horizon = 6
    anchor_pos = rng.normal(size=3)
    anchor_rot = rotmat_to_rot6d(_random_rotation(rng))
    abs_pos = rng.normal(size=(horizon, 3))
    abs_rot = np.stack([rotmat_to_rot6d(_random_rotation(rng)) for _ in range(horizon)])
    rel_pos, rel_rot = absolute_to_delta_ee(abs_pos, abs_rot, anchor_pos, anchor_rot)

    action = np.concatenate([rel_pos, rel_rot, np.zeros((horizon, 1))], axis=1)
    native = native_chunk_from_uva(action, np.zeros((horizon, FORCE_DIM)), LAYOUT,
                                  predict_force=True)
    result = decode_chunk(native, LAYOUT, anchor_pos[None, :], anchor_rot[None, :],
                          force_enabled=False, gate_n=0.0)
    np.testing.assert_allclose(result.role("left").position, abs_pos, atol=1e-10)
    np.testing.assert_allclose(result.role("left").rot6d, abs_rot, atol=1e-10)
    np.testing.assert_allclose(result.action_times_s, np.arange(horizon) * ACTION_DT)


def test_native_chunk_zeroes_force_when_the_checkpoint_has_no_force_head():
    """predict_force=False must leave EXACT zeros, so gate_force_command can never pass it —
    a pure-position UVA checkpoint is pure position whatever the viser switch says."""
    action = np.zeros((4, UVA_ACTION_DIM))
    native = native_chunk_from_uva(action, None, LAYOUT, predict_force=False)
    assert np.count_nonzero(native[:, LAYOUT.slots("left").force]) == 0
    from flore.s1_policy.serving.hl_online import gate_force_command

    gated = gate_force_command(native[:, LAYOUT.slots("left").force],
                               force_enabled=True, gate_n=0.0)
    assert np.count_nonzero(gated) == 0


def test_native_chunk_refuses_a_wrong_width_or_missing_wrench():
    with pytest.raises(ValueError, match="UVA action chunk"):
        native_chunk_from_uva(np.zeros((4, 9)), None, LAYOUT, predict_force=False)
    with pytest.raises(ValueError, match="predict_force=True"):
        native_chunk_from_uva(np.zeros((4, UVA_ACTION_DIM)), np.zeros((3, FORCE_DIM)),
                              LAYOUT, predict_force=True)
    with pytest.raises(ValueError, match="predict_force=True"):
        native_chunk_from_uva(np.zeros((4, UVA_ACTION_DIM)), None, LAYOUT, predict_force=True)


# ---------------------------------------------------------------------------
# MuseMeta: the derived image stride, and the refusal that protects it.
# ---------------------------------------------------------------------------
def _meta_payload(img_indices=(19, 23, 27, 31, 39, 47, 55, 63), n_cond=4, **over):
    meta = {
        "num_actions": 32, "action_dim": 10, "force_dim": 6, "history_len": 32,
        "n_img_cond": n_cond, "force_history_length": 8,
        "obs_pose_repr": "relative", "action_pose_repr": "relative",
        "use_force": True, "predict_force": True, "n_action_steps": 8,
        "down_sample_steps": 1, "checkpoint": "/tmp/x.slim.pt", "random_init": False,
    }
    meta.update(over)
    arrays = {
        "img_indices": np.asarray(img_indices, dtype=np.int64),
        "ft_scale": np.asarray([15.0, 15.0, 15.0, 1.5, 1.5, 1.5]),
        "episode_frame_nums": np.asarray([300, 280], dtype=np.int64),
    }
    return meta, arrays


def test_meta_derives_the_image_stride_and_obs_dt():
    meta = MuseMeta.from_meta(*_meta_payload())
    assert meta.img_step_stride == 4
    # 4 steps at 30 Hz = 133.3 ms; hl_serving multiplies this by --obs-hz to get its ingest
    # stride, which is what makes the ring's To frames land on the model's own comb.
    assert meta.obs_history_dt_s == pytest.approx(4 * ACTION_DT)
    assert meta.n_img_cond == 4 and meta.num_actions == 32


def test_meta_refuses_an_unevenly_spaced_image_comb():
    """The live ingest is a FIXED-stride ring; an uneven comb cannot be reproduced from it, and
    the failure would be silent (right shapes, frames at the wrong times)."""
    with pytest.raises(ValueError, match="not evenly spaced"):
        MuseMeta.from_meta(*_meta_payload(img_indices=(10, 23, 27, 31, 39, 47, 55, 63)))


# ---------------------------------------------------------------------------
# MuseRemoteModel: the refusals that keep a missing observation from being guessed at.
# ---------------------------------------------------------------------------
class _FakeBackend:
    """Records the last request and replies with a canned chunk. No subprocess."""

    def __init__(self, meta: MuseMeta, horizon: int = 32):
        self.meta = meta
        self.horizon = horizon
        self.last: dict | None = None

    def request(self, meta, *, timeout_s=None, **arrays):
        self.last = {"meta": meta, "arrays": arrays}
        action = np.zeros((self.horizon, UVA_ACTION_DIM))
        action[:, 3:9] = rotmat_to_rot6d(np.eye(3))
        return {}, {"action_rel": action, "force_pred": np.zeros((self.horizon, FORCE_DIM))}


def _batch(to: int = 4, fh: int = 8):
    return {
        "images": [torch.zeros(1, 3, 224, 224) for _ in range(to)],
        "state": torch.zeros(1, to, LAYOUT.state_dim),
        "state_valid": torch.ones(1, to, LAYOUT.state_dim, dtype=torch.bool),
        "force_history": torch.zeros(1, fh, LAYOUT.force_hist_channels),
        "force_history_valid": torch.tensor([True]),
        MUSE_EEF_POS_REL_KEY: np.zeros((to, 3)),
        MUSE_EEF_ROT6D_REL_KEY: np.tile(rotmat_to_rot6d(np.eye(3)), (to, 1)),
    }


def test_remote_model_refuses_a_batch_without_the_pose_history():
    """A recorded deploy take has no absolute EE position in it, so re-inferring one with
    --algo muse must SAY so rather than feed a zero-motion history — which is a perfectly
    plausible observation and would produce a confident, subtly wrong chunk."""
    meta = MuseMeta.from_meta(*_meta_payload())
    model = MuseRemoteModel(_FakeBackend(meta), meta, LAYOUT)
    batch = _batch()
    del batch[MUSE_EEF_POS_REL_KEY]
    with pytest.raises(KeyError, match="no absolute EE position"):
        model.sample_actions(batch)


def test_remote_model_refuses_a_sampler_step_count():
    meta = MuseMeta.from_meta(*_meta_payload())
    model = MuseRemoteModel(_FakeBackend(meta), meta, LAYOUT)
    with pytest.raises(ValueError, match="one pass|num_steps must be 1"):
        model.sample_actions(_batch(), num_steps=10)


def test_remote_model_sends_images_in_0_1_space_and_gripper_in_metres():
    """The wire carries 0-1 CHW images (what UVA's process_data expects) and METRES for the
    jaw (what the zarr column holds), converted from the ring's [-1,1] / fraction forms."""
    meta = MuseMeta.from_meta(*_meta_payload())
    backend = _FakeBackend(meta)
    model = MuseRemoteModel(backend, meta, LAYOUT)
    batch = _batch()
    # A mid-grey frame in the ring's normalized space, and a half-open jaw.
    for im in batch["images"]:
        im.fill_(0.25)
    grip_col = LAYOUT.state_slots()["left"]["gripper"]
    batch["state"][0, :, grip_col] = 0.5

    out = model.sample_actions(batch, generator=torch.Generator().manual_seed(11))
    assert out.shape == (1, 32, LAYOUT.action_dim)
    sent = backend.last["arrays"]
    np.testing.assert_allclose(sent["images"], 0.25 * MUSE_IMAGE_STD + MUSE_IMAGE_MEAN, atol=1e-6)
    np.testing.assert_allclose(sent["grip_m"], 0.5 * GRIPPER_WIDTH_REF_M)
    # The seed rides the generator, so sample_chunk's seed contract carries over unchanged.
    assert backend.last["meta"]["seed"] == 11
    assert backend.last["meta"]["kind"] == wire.KIND_LIVE


def test_remote_model_refuses_a_history_of_the_wrong_length():
    meta = MuseMeta.from_meta(*_meta_payload())
    model = MuseRemoteModel(_FakeBackend(meta), meta, LAYOUT)
    with pytest.raises(ValueError, match="conditioning frames"):
        model.sample_actions(_batch(to=2))


# ---------------------------------------------------------------------------
# MuseLivePlanner: the extra ring, its alignment, and the anchor-relative math.
# ---------------------------------------------------------------------------
def _loaded(meta: MuseMeta, backend) -> LoadedHLModel:
    return LoadedHLModel(
        algorithm=MUSE_ALGO,
        model=MuseRemoteModel(backend, meta, LAYOUT),
        config=MuseConfig(action_horizon=meta.num_actions,
                          force_hist_len=meta.force_history_length,
                          action_dim=LAYOUT.action_dim, state_dim=LAYOUT.state_dim),
        layout=LAYOUT,
        norm_stats={},
        device=torch.device("cpu"),
        num_steps=1,
        obs_history_len=meta.n_img_cond,
        obs_history_dt_s=meta.obs_history_dt_s,
        image_mean=MUSE_IMAGE_MEAN,
        image_std=MUSE_IMAGE_STD,
        image_key="camera0_rgb",
        force_hist_len=meta.force_history_length,
    )


def _planner(horizon: int = 32):
    meta = MuseMeta.from_meta(*_meta_payload())
    backend = _FakeBackend(meta, horizon=horizon)
    return MuseLivePlanner(_loaded(meta, backend), obs_hz=30.0), backend


def test_planner_refuses_a_non_muse_checkpoint():
    """The local algorithms must keep using LivePlanner: this subclass overrides the proprio
    push and adds a ring they never read."""
    meta = MuseMeta.from_meta(*_meta_payload())
    loaded = _loaded(meta, _FakeBackend(meta))
    flow = LoadedHLModel(**{**loaded.__dict__, "algorithm": "flow"})
    with pytest.raises(ValueError, match="MuseLivePlanner is for"):
        MuseLivePlanner(flow)


def test_planner_rings_stay_the_same_length_and_aligned():
    """push_proprio_from_anchor writes THREE rings under one lock. If they ever diverge in
    length, row i of the position history stops describing the same frame as row i of the
    rotation history — a silent pose/rotation skew."""
    planner, _ = _planner()
    to = planner.m.obs_history_len
    for k in range(3 * to):
        planner.commit_role("left", np.array([0.01 * k, 0.0, 0.0]), rotmat_to_rot6d(np.eye(3)))
        planner.push_proprio_from_anchor()
        assert len(planner._m_pos) == len(planner._buf_rot6d) == len(planner._buf_grip) <= to
    # Oldest-first: the last row is the newest commit.
    np.testing.assert_allclose(planner._m_pos[-1], [0.01 * (3 * to - 1), 0.0, 0.0])
    np.testing.assert_allclose(planner._m_pos[0], [0.01 * (2 * to), 0.0, 0.0])


def test_seed_anchor_reseeds_the_position_ring_too():
    planner, _ = _planner()
    planner.commit_role("left", np.array([9.0, 9.0, 9.0]), rotmat_to_rot6d(np.eye(3)))
    planner.push_proprio_from_anchor()
    seed_pos = np.array([[0.3, -0.1, 0.2]])
    seed_rot = np.tile(rotmat_to_rot6d(np.eye(3)), (1, 1))
    planner.seed_anchor(seed_pos, seed_rot)
    assert len(planner._m_pos) == planner.m.obs_history_len
    for row in planner._m_pos:
        np.testing.assert_allclose(row, seed_pos[0])
    planner.reset_start_pose()
    for row in planner._m_pos:
        np.testing.assert_allclose(row, seed_pos[0])


def test_snapshot_muse_obs_is_anchor_relative_and_recovers_the_rotation_history():
    """The two derived arrays must equal a direct absolute_to_delta_ee against the SNAPSHOTTED
    anchor. The rotation one is the interesting case: it is reconstructed from the wrt_start
    state channel (R_anchor^T R_t = (R_start^T R_anchor)^T (R_start^T R_t)) rather than stored,
    so an error there would be invisible in every shape check."""
    rng = np.random.default_rng(5)
    planner, _ = _planner()
    to = planner.m.obs_history_len

    start_pos = rng.normal(size=3)
    start_rot = rotmat_to_rot6d(_random_rotation(rng))
    planner.seed_anchor(start_pos[None, :], start_rot[None, :])

    poses = []
    for _ in range(to):
        pos = rng.normal(size=3)
        rot = rotmat_to_rot6d(_random_rotation(rng))
        poses.append((pos, rot))
        planner.commit_role("left", pos, rot)
        planner.push_proprio_from_anchor()

    obs, anchor_pos, anchor_rot6d, pos_rel, rot6d_rel = planner.snapshot_muse_obs()
    abs_pos = np.stack([p for p, _ in poses])
    abs_rot = np.stack([r for _, r in poses])
    want_pos, want_rot = absolute_to_delta_ee(abs_pos, abs_rot, anchor_pos[0], anchor_rot6d[0])
    np.testing.assert_allclose(pos_rel, want_pos, atol=1e-10)
    np.testing.assert_allclose(rot6d_rel, want_rot, atol=1e-10)
    # The newest row IS the anchor (push_proprio_from_anchor logged it), so it relativizes to
    # the identity pose — the property "obs base == action base" made visible.
    np.testing.assert_allclose(pos_rel[-1], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rot6d_rel[-1], rotmat_to_rot6d(np.eye(3)), atol=1e-10)
    assert obs.state.shape == (to, LAYOUT.state_dim)
    assert obs.force_history.shape == (planner.m.force_hist_len, LAYOUT.force_hist_channels)


def test_infer_chunk_decodes_against_the_snapshotted_anchor():
    """End to end on the FLoRe half: a canned identity-relative chunk must decode to the anchor
    pose held for every step, and the capture dict must carry what the recorder reads."""
    planner, backend = _planner(horizon=32)
    anchor = np.array([[0.42, 0.11, -0.07]])
    rot = np.tile(rotmat_to_rot6d(_random_rotation(np.random.default_rng(6))), (1, 1))
    planner.seed_anchor(anchor, rot)
    planner.push_proprio_from_anchor()

    capture: dict = {}
    result = planner.infer_chunk(seed=3, force_enabled=False, gate_n=0.0, capture=capture)
    rc = result.role("left")
    assert rc.position.shape == (32, 3)
    np.testing.assert_allclose(rc.position, np.tile(anchor, (32, 1)), atol=1e-12)
    np.testing.assert_allclose(rc.rot6d, np.tile(rot, (32, 1)), atol=1e-10)
    assert set(capture) == {"obs", "anchor_pos", "anchor_rot6d", "prediction", "seed", "num_steps"}
    assert capture["seed"] == 3 and capture["num_steps"] == 1
    assert capture["prediction"].shape == (32, LAYOUT.action_dim)
    assert backend.last["meta"]["seed"] == 3


def test_infer_chunk_refuses_a_sampler_step_count():
    planner, _ = _planner()
    planner.push_proprio_from_anchor()
    with pytest.raises(ValueError, match="one pass"):
        planner.infer_chunk(seed=0, force_enabled=False, gate_n=0.0, num_steps=4)


def test_muse_planner_does_not_enable_temporal_ensembling():
    """A UVA sample is ~0.6-2 s, so a per-step re-query at 30 Hz is not on the table; the page's
    checkbox must read as unavailable rather than silently doing single-chunk work."""
    planner, _ = _planner()
    assert planner.temporal_ensemble_enabled is False
    with pytest.raises(RuntimeError, match="does not use temporal"):
        planner.infer_step_ensembled(seed=0, force_enabled=False, gate_n=0.0)
