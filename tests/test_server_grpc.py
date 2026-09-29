"""In-process gRPC tests for FujiPolicyService.

Spins the REAL server on a random localhost port with a FAKE ChunkPolicy (no
model, no GPU, no checkpoint), then drives Predict / PredictChunk / Reset /
GetPolicyInfo through a real grpc channel and asserts shapes + reset behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

import grpc
from fuji.s1_policy.serving.proto import fuji_policy_pb2 as pb2
from fuji.s1_policy.serving.proto import fuji_policy_pb2_grpc as pb2_grpc
from fuji.s1_policy.serving.server import build_server, encode_image_jpeg, make_servicer

ACTION_DIM = 32
ACTION_HORIZON = 8


class FakeChunkPolicy:
    """Deterministic fake: chunk k has value (k*100 + i) on every dim."""

    def __init__(self, chunk_size: int = ACTION_HORIZON, action_dim: int = ACTION_DIM):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.call_count = 0
        self.reset_count = 0
        self.image_res = 224
        self.history_len = 0
        self.config: dict = {"image_res": 224}

    def predict_chunk(self, images, state, prompt=None, steps_executed=0, delay_estimate=0, prev_actions=None):
        chunk_idx = self.call_count
        self.call_count += 1
        chunk = np.empty((1, self.chunk_size, self.action_dim), dtype=np.float32)
        for i in range(self.chunk_size):
            chunk[0, i, :] = chunk_idx * 100 + i
        return chunk

    def reset(self) -> None:
        self.reset_count += 1


def _request(state_dim: int = ACTION_DIM) -> pb2.PredictRequest:
    rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    state = np.arange(state_dim, dtype=np.float32)
    return pb2.PredictRequest(state=state.tolist(), image=encode_image_jpeg(rgb), prompt="test")


@pytest.fixture(params=[True, False], ids=["rtc", "sync"])
def channel_and_fake(request):
    """Yield (grpc channel, fake policy) for both RTC and sync drip modes."""
    fake = FakeChunkPolicy()
    servicer = make_servicer(
        fake,
        rtc=request.param,
        s_min=2,
        delay_estimate=2,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        checkpoint="fake-ckpt",
        act_timeout_secs=5.0,
    )
    server, port = build_server(servicer, host="127.0.0.1", port=0)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=5.0)
    try:
        yield channel, fake
    finally:
        channel.close()
        server.stop(grace=0)


def test_predict_returns_single_action(channel_and_fake):
    channel, _fake = channel_and_fake
    stub = pb2_grpc.FujiPolicyServiceStub(channel)
    resp = stub.Predict(_request(), timeout=10.0)
    assert len(resp.action) == ACTION_DIM
    # First action of the first chunk is all-zeros (chunk_idx=0, step 0).
    np.testing.assert_array_equal(np.asarray(resp.action), np.zeros(ACTION_DIM, dtype=np.float32))


def test_predict_drip_feeds_in_order_sync():
    """Sync (--no-rtc) mode is strictly open-loop: the first chunk is drip-fed
    0..H-1 with no replan interruption. (RTC mode legitimately replans mid-chunk,
    so this strict-order property is sync-only.)"""
    fake = FakeChunkPolicy()
    servicer = make_servicer(fake, rtc=False, action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON, checkpoint="fake")
    server, port = build_server(servicer, host="127.0.0.1", port=0)
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=5.0)
    try:
        stub = pb2_grpc.FujiPolicyServiceStub(channel)
        first_vals = [stub.Predict(_request(), timeout=10.0).action[0] for _ in range(ACTION_HORIZON)]
        # Within the first chunk the leading dim increments 0..H-1.
        assert first_vals == [float(i) for i in range(ACTION_HORIZON)]
    finally:
        channel.close()
        server.stop(grace=0)


def test_predict_chunk_returns_full_chunk(channel_and_fake):
    channel, _fake = channel_and_fake
    stub = pb2_grpc.FujiPolicyServiceStub(channel)
    resp = stub.PredictChunk(_request(), timeout=10.0)
    assert resp.horizon == ACTION_HORIZON
    assert resp.action_dim == ACTION_DIM
    assert len(resp.actions) == ACTION_HORIZON * ACTION_DIM
    chunk = np.asarray(resp.actions, dtype=np.float32).reshape(resp.horizon, resp.action_dim)
    # Some chunk k: row i is all (k*100 + i).
    for i in range(ACTION_HORIZON):
        assert len(set(chunk[i].tolist())) == 1  # constant across dims
        assert chunk[i, 0] - chunk[0, 0] == i  # rows increment by 1


def test_get_policy_info(channel_and_fake):
    channel, _fake = channel_and_fake
    stub = pb2_grpc.FujiPolicyServiceStub(channel)
    info = stub.GetPolicyInfo(pb2.PolicyInfoRequest(), timeout=10.0)
    assert info.action_dim == ACTION_DIM
    assert info.action_horizon == ACTION_HORIZON
    assert info.checkpoint == "fake-ckpt"
    assert info.s_min == 2
    assert info.delay_estimate == 2


def test_reset_clears_state_and_calls_policy_reset(channel_and_fake):
    channel, fake = channel_and_fake
    stub = pb2_grpc.FujiPolicyServiceStub(channel)
    # Consume the whole first chunk.
    for _ in range(ACTION_HORIZON):
        stub.Predict(_request(), timeout=10.0)
    resp = stub.Reset(pb2.ResetRequest(), timeout=10.0)
    assert resp.ok is True
    assert fake.reset_count >= 1
    # After reset the next Predict starts a fresh episode and still returns an action.
    resp = stub.Predict(_request(), timeout=10.0)
    assert len(resp.action) == ACTION_DIM


def test_predict_rejects_empty_state(channel_and_fake):
    channel, _fake = channel_and_fake
    stub = pb2_grpc.FujiPolicyServiceStub(channel)
    rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    bad = pb2.PredictRequest(state=[], image=encode_image_jpeg(rgb), prompt="")
    with pytest.raises(grpc.RpcError) as exc:
        stub.Predict(bad, timeout=10.0)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# Un-relativize: server converts anchor-relative chunk -> absolute (the fix for
# the train/deploy reference-frame blocker the e2e review caught).
# ---------------------------------------------------------------------------
def test_chunk_to_absolute_inverts_training_transform():
    """DPFlowChunkPolicy._chunk_to_absolute must be the exact inverse of the
    training transform (umi_dataset's absolute_to_delta_ee against the anchor):
    encode known absolute targets as an anchor-relative chunk, un-relativize,
    recover the targets. All H rows share the ONE anchor (obs state)."""
    from scipy.spatial.transform import Rotation

    from fuji.s1_policy.data.transforms import absolute_to_delta_ee
    from fuji.s1_policy.data.umi_dataset import _HEAD, _LEFT, _RIGHT
    from fuji.s1_policy.serving.server import DPFlowChunkPolicy
    from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_quat_xyzw, rotmat_to_rot6d

    rng = np.random.default_rng(0)
    H = 5
    # Anchor (obs) pose: random per keypoint.
    anchor = np.zeros(ACTION_DIM, dtype=np.float64)
    abs_pos = {}
    abs_rot = {}
    for name, (pos_sl, rot_sl, _grip) in (("L", _LEFT), ("R", _RIGHT), ("H", _HEAD)):
        ap = rng.normal(size=3)
        aR = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        anchor[pos_sl] = ap
        anchor[rot_sl] = rotmat_to_rot6d(aR)
        # Known absolute TARGET trajectory (H rows) per keypoint.
        tp = rng.normal(size=(H, 3))
        tR = Rotation.random(H, random_state=rng.integers(1 << 30)).as_matrix()
        abs_pos[name] = tp
        abs_rot[name] = tR

    # Encode the chunk exactly as training does: relative to the anchor.
    chunk = np.zeros((1, H, ACTION_DIM), dtype=np.float32)
    for name, (pos_sl, rot_sl, _grip) in (("L", _LEFT), ("R", _RIGHT), ("H", _HEAD)):
        rel_pos, rel_rot6d = absolute_to_delta_ee(
            abs_pos[name],
            np.stack([rotmat_to_rot6d(R) for R in abs_rot[name]]),
            anchor[pos_sl],
            anchor[rot_sl],
        )
        chunk[0, :, pos_sl] = rel_pos
        chunk[0, :, rot_sl] = rel_rot6d

    # Un-relativize (server side) — no model needed, call the method directly.
    policy = DPFlowChunkPolicy.__new__(DPFlowChunkPolicy)  # bypass __init__ (no model)
    out = policy._chunk_to_absolute(chunk, anchor.astype(np.float32))

    # Every row must recover the known absolute target.
    for name, (pos_sl, rot_sl, _grip) in (("L", _LEFT), ("R", _RIGHT), ("H", _HEAD)):
        np.testing.assert_allclose(out[0, :, pos_sl], abs_pos[name], atol=1e-4)
        for t in range(H):
            q = rotmat_to_quat_xyzw(rot6d_to_rotmat(out[0, t, rot_sl]))
            q_exp = Rotation.from_matrix(abs_rot[name][t]).as_quat()
            assert abs(np.dot(q, q_exp)) > 0.999, f"{name} row {t} rot mismatch"


def _write_min_checkpoint(ckpt_dir, data_cfg: dict, model_cfg: dict | None = None) -> None:
    """Minimal checkpoint dir with just a config.json (no weights) — enough to reach
    the serving guardrail before any model is built."""
    import json

    (ckpt_dir / "config.json").write_text(
        json.dumps({"model": model_cfg or {}, "data": data_cfg}))


def test_serving_rejects_shared_ref_frame_checkpoint(tmp_path):
    """A shared-frame model must be REFUSED at load (serving hardcodes per_keypoint
    un-relativization, so deploying it would silently corrupt absolute poses)."""
    from fuji.s1_policy.serving.server import load_dpflow_chunk_policy

    _write_min_checkpoint(tmp_path, {"ref_frame": "left"})
    with pytest.raises(NotImplementedError, match="ref_frame"):
        load_dpflow_chunk_policy(str(tmp_path), device="cpu")


def test_serving_rejects_history_checkpoint(tmp_path):
    """An observation-history model must be REFUSED at load (serving builds a
    single-frame obs and would mis-size global_cond). obs_history_len is baked under
    MODEL (TrainConfig pushes it there for both pipelines) — HLDataConfig has no such
    field, so a real HL config.json carries it under `model`, NOT `data`. The guard must
    read model-first, so this test writes the REAL shape."""
    from fuji.s1_policy.serving.server import load_dpflow_chunk_policy

    _write_min_checkpoint(tmp_path, {"ref_frame": "per_keypoint"},
                          model_cfg={"obs_history_len": 2})
    with pytest.raises(NotImplementedError, match="obs_history_len"):
        load_dpflow_chunk_policy(str(tmp_path), device="cpu")


def test_serving_rejects_hl_layout_checkpoint(tmp_path):
    """An HL-v2 checkpoint (non-empty model.layout_mode / a force encoder) must be
    REFUSED at load even at To=1: serving still implements the pre-v2 32-dim layout
    (_chunk_to_absolute hardcodes it, predict_chunk builds no force_history / wrt_start
    state), so deploying it would crash opaquely or emit wrong targets."""
    from fuji.s1_policy.serving.server import load_dpflow_chunk_policy

    _write_min_checkpoint(tmp_path, {"ref_frame": "per_keypoint"},
                          model_cfg={"layout_mode": "left_only", "force_hist_channels": 6})
    with pytest.raises(NotImplementedError, match="HL-v2"):
        load_dpflow_chunk_policy(str(tmp_path), device="cpu")


def test_serving_rejects_diffusion_checkpoint(tmp_path):
    """A diffusion checkpoint must be REFUSED at load. For algo='diffusion' the trained
    config lives under raw['diffusion'] while raw['model'] holds flow DEFAULTS (empty
    layout_mode, force 0, obs_history 1) — so the HL/history/force guards would all read
    those defaults and PASS, silently building a garbage 32-dim flow model. The algo guard
    must fire FIRST on the real (mis-serialized) shape."""
    import json

    from fuji.s1_policy.serving.server import load_dpflow_chunk_policy

    # The realistic shape a diffusion TrainConfig serializes: flow defaults under `model`,
    # the true config under `diffusion`, algo at top level.
    (tmp_path / "config.json").write_text(json.dumps({
        "algo": "diffusion",
        "model": {"layout_mode": "", "force_hist_channels": 0, "obs_history_len": 1, "action_dim": 32},
        "diffusion": {"layout_mode": "left_only", "force_hist_channels": 6, "obs_history_len": 2, "action_dim": 16},
        "data": {"ref_frame": "per_keypoint"},
    }))
    with pytest.raises(NotImplementedError, match="algo='diffusion'|algo=.diffusion."):
        load_dpflow_chunk_policy(str(tmp_path), device="cpu")


# ---------------------------------------------------------------------------
# Regression: predict_chunk must feed _chunk_to_absolute a 1-D (AD,) anchor.
#
# The real drip callers (SyncChunkDripPolicy.get_action, RTC _inference_loop) call
# predict_chunk(images[None], state[None], ...), so `state` arrives as (1, AD). Before the
# fix, predict_chunk forwarded that (1, AD) straight into _chunk_to_absolute, whose
# per-keypoint anchor[slice(3, 9)] indexing needs a 1-D (AD,) vector — the (1, AD) sliced
# axis 0 and delta_ee_to_absolute's einsum raised ValueError on the FIRST real inference.
# The unit test above misses this: it calls _chunk_to_absolute directly with a 1-D anchor,
# and the gRPC tests use FakeChunkPolicy which skips _chunk_to_absolute entirely. This test
# drives the REAL DPFlowChunkPolicy.predict_chunk (with a fake model) through the production
# drip path so an anchor-rank regression fails loudly.
# ---------------------------------------------------------------------------
class _FakeModel:
    """Stands in for DPFlowModel: sample_actions returns a zero anchor-relative chunk
    of shape (B, H, AD), so predict_chunk exercises the un-relativize path without a GPU."""

    def __init__(self, horizon: int, action_dim: int):
        self._h = horizon
        self._ad = action_dim

    def sample_actions(self, batch_dict, num_steps=None, steps_executed=0, delay_estimate=0):
        import torch

        state = batch_dict["state"]
        b = state.shape[0] if hasattr(state, "shape") and state.ndim >= 1 else 1
        return torch.zeros(b, self._h, self._ad)

    def reset(self):
        pass


def _real_chunk_policy(horizon=ACTION_HORIZON, action_dim=ACTION_DIM):
    """A DPFlowChunkPolicy with a fake model and NO norm stats (so state/actions pass
    through un-normalized) — real predict_chunk + real _chunk_to_absolute, no GPU."""
    from fuji.s1_policy.serving.server import DPFlowChunkPolicy

    return DPFlowChunkPolicy(
        _FakeModel(horizon, action_dim),
        norm_stats=None,
        device="cpu",
        action_dim=action_dim,
        action_horizon=horizon,
    )


def test_predict_chunk_accepts_batched_anchor_from_drip():
    """Regression for the (1, AD)-anchor crash: DPFlowChunkPolicy.predict_chunk must
    accept the (num_cameras, H, W, C) images + 1-D (AD,) state that the drip path passes
    as images[None] / state[None], and return a finite (B, H, AD) absolute chunk."""
    policy = _real_chunk_policy()
    images = np.zeros((1, 224, 224, 3), dtype=np.uint8)  # (num_cameras=1, H, W, C)
    state = np.arange(ACTION_DIM, dtype=np.float32) * 0.01
    # Exactly what SyncChunkDripPolicy.get_action / RTC do: add the batch dim.
    chunk = policy.predict_chunk(images[None], state[None])
    assert chunk.shape == (1, ACTION_HORIZON, ACTION_DIM)
    assert np.isfinite(chunk).all()


def test_sync_drip_get_action_survives_real_unrelativize():
    """End-to-end through the REAL production drip wrapper: SyncChunkDripPolicy.get_action
    must not crash in _chunk_to_absolute (the (1, AD) anchor path) and must drip finite
    (AD,) actions across a chunk boundary."""
    from fuji.s1_policy.serving.server import SyncChunkDripPolicy

    drip = SyncChunkDripPolicy(_real_chunk_policy())
    images = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    state = np.arange(ACTION_DIM, dtype=np.float32) * 0.01
    for step in range(ACTION_HORIZON + 2):  # cross a chunk boundary -> forces a re-predict
        action, obs_idx = drip.get_action(images, state)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (ACTION_DIM,), f"step {step}: bad shape {action.shape}"
        assert np.isfinite(action).all(), f"step {step}: non-finite action"


def test_rtc_get_action_survives_real_unrelativize():
    """Same guard on the RTC path: the background inference thread calls predict_chunk with
    state[None]; get_action must return a finite (AD,) action once the first chunk lands."""
    from fuji.s1_policy.rtc import RealTimeChunkingPolicy

    rtc = RealTimeChunkingPolicy(_real_chunk_policy(), s_min=2, delay_estimate=2)
    try:
        images = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        state = np.arange(ACTION_DIM, dtype=np.float32) * 0.01
        action, _obs_idx = rtc.get_action(images, state, timeout_secs=10.0)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape == (ACTION_DIM,)
        assert np.isfinite(action).all()
    finally:
        rtc.stop()


def test_resize_center_crop_matches_training_converter():
    """Regression for the naive-stretch camera bug: the client's _resize_center_crop_224
    must be byte-identical to the training converter's resize+center-crop geometry
    (compute_crop_resize + INTER_AREA), so deploy frames match the trained distribution."""
    import cv2

    from fuji.client.main import _resize_center_crop_224

    rng = np.random.default_rng(0)
    for in_w, in_h in [(640, 480), (640, 360), (480, 270), (1280, 720), (224, 224)]:
        img = (rng.random((in_h, in_w, 3)) * 255).astype(np.uint8)
        mine = _resize_center_crop_224(img)
        # Ground-truth converter geometry (mirrors umi_convert_core.compute_crop_resize):
        scale = max(224 / in_w, 224 / in_h)
        rw = max(int(np.ceil(in_w * scale)), 224)
        rh = max(int(np.ceil(in_h * scale)), 224)
        cx, cy = (rw - 224) // 2, (rh - 224) // 2
        gt = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)[cy : cy + 224, cx : cx + 224]
        assert mine.shape == (224, 224, 3)
        assert np.array_equal(mine, gt), f"{in_w}x{in_h}: resize not byte-identical to converter"
