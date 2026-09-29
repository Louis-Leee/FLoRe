"""Offline HL-v2 Flow/Diffusion checkpoint inference and viewer contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from fuji.s1_policy.data.layout import ACTION_HORIZON, ACTION_HZ
import torch
from torch import nn

from tests.test_hl_dataset import _write_hl_zarr

pytest.importorskip("zarr")


class _StubEncoder(nn.Module):
    """Small image encoder that accepts the synthetic fixture's 8x8 images."""

    def __init__(self, num_features: int = 8) -> None:
        super().__init__()
        self.num_features = num_features
        self.projection = nn.Conv2d(3, num_features, kernel_size=1)

    def forward(self, image):
        return self.projection(image).mean(dim=(-1, -2))


class _ACTStubBackbone(nn.Module):
    """ACT's trunk contract, not the ViT one: ``(N,C,H,W) -> (features, pos)``.

    ACT consumes a spatial GRID (its transformer cross-attends to every cell), so a stub
    that pooled to one vector would test a different architecture. Keeps the 8x8 fixture
    image's resolution so the grid has 64 cells.
    """

    def __init__(self, hidden_dim: int, num_channels: int = 8) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Conv2d(3, num_channels, kernel_size=1)

    def forward(self, image):
        features = self.projection(image)
        pos = torch.zeros(1, self.hidden_dim, features.shape[2], features.shape[3],
                          dtype=features.dtype, device=features.device)
        return features, pos


@dataclass
class _Bundle:
    algorithm: str
    checkpoint: Path
    zarr_path: str
    session_type: type
    module: object
    model_type: type


def _refresh_model_hash(checkpoint: Path) -> None:
    weights = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["_provenance"]["model_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))


def _make_bundle(
    tmp_path: Path,
    monkeypatch,
    algorithm: str,
    *,
    production: bool = False,
    history_len: int | None = None,
) -> _Bundle:
    from fuji.s1_policy.data.hl_dataset import HLDataset
    from fuji.s1_policy.data.norm_stats import NormStats, save_norm_stats
    from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel
    from fuji.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig
    from fuji.s1_policy.training.trainer import save_checkpoint

    zarr_path = _write_hl_zarr(
        tmp_path / f"{algorithm}.zarr.zip",
        n_frames=8,
        n_episodes=2,
        image_hw=224 if production else 8,
    )
    obs_history_len = history_len if history_len is not None else (2 if production else 1)
    hl_data = HLDataConfig(
        zarr_paths=(zarr_path,),
        mode="left_only",
        obs_history_len=obs_history_len,
        drop_glitch_windows=False,
        val_last_episodes=1,
        norm_stats_path="fitted_norm_stats.json",
    )
    shared = dict(
        vit="vit_ti",
        pretrained=True,
        diffusion_net="dit",
        diffusion_step_embed_dim=16,
        model_channels=32,
        num_blocks=1,
        n_heads=4,
        mlp_mult=2,
        filled_state_dim=8,
        force_enc_width=8,
        force_enc_out_dim=8,
        num_draws=1,
    )
    if algorithm == "flow":
        model_config = DPFlowConfig(**shared, inference_steps=2)
        config = TrainConfig(algo="flow", pipeline="hl", model=model_config, hl_data=hl_data, ema=False)
        model_type = DPFlowModel
        model = model_type(
            replace(config.model, pretrained=False),
            encoder=None if production else _StubEncoder(),
        )
        import fuji.s1_policy.inference.flow as module

        from fuji.s1_policy.inference.flow import FlowInferenceSession as session_type

        if not production:
            monkeypatch.setattr(module, "DPFlowModel", lambda loaded: model_type(loaded, encoder=_StubEncoder()))
    elif algorithm == "act":
        from fuji.s1_policy.model.dp_act import DPACTConfig, DPACTModel

        # ACT does not share the DiT/UNet knobs (no decoder channels, no zero-pad state
        # slot), so it takes its own tiny-config kwargs rather than `shared`. hidden_dim is
        # small but must stay EVEN and divisible by nheads (its 2D sine embedding splits
        # y/x), which TrainConfig now asserts.
        act_config = DPACTConfig(
            hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1, dec_layers=1,
            vae_enc_layers=1, latent_dim=4, force_enc_width=8, force_enc_out_dim=8,
            pretrained=True, backbone="resnet18",
        )
        config = TrainConfig(algo="act", pipeline="hl", act=act_config, hl_data=hl_data, ema=False)
        model_type = DPACTModel
        model = model_type(
            replace(config.act, pretrained=False),
            encoder=None if production else _ACTStubBackbone(config.act.hidden_dim),
        )
        import fuji.s1_policy.inference.act as module

        from fuji.s1_policy.inference.act import ACTInferenceSession as session_type

        if not production:
            monkeypatch.setattr(
                module, "DPACTModel",
                lambda loaded: model_type(loaded, encoder=_ACTStubBackbone(loaded.hidden_dim)))
    else:
        if production:
            shared.update(diffusion_net="unet", down_dims=(16, 32), n_groups=4)
        diffusion_config = DPDiffusionConfig(
            **shared,
            num_train_timesteps=8,
            num_inference_steps=2,
            input_pertub=0.0,
            clip_sample=True,
        )
        config = TrainConfig(algo="diffusion", pipeline="hl", diffusion=diffusion_config, hl_data=hl_data, ema=False)
        model_type = DPDiffusionModel
        model = model_type(
            replace(config.diffusion, pretrained=False),
            encoder=None if production else _StubEncoder(),
        )
        import fuji.s1_policy.inference.diffusion as module

        from fuji.s1_policy.inference.diffusion import DiffusionInferenceSession as session_type

        if not production:
            monkeypatch.setattr(module, "DPDiffusionModel", lambda loaded: model_type(loaded, encoder=_StubEncoder()))

    dataset = HLDataset(
        zarr_path,
        mode="left_only",
        obs_history_len=obs_history_len,
        drop_glitch_windows=False,
        require_episode_ends=True,
    )
    split = dataset.split_provenance(1)
    dataset.close()
    stats = {
        "actions": NormStats(q01=np.arange(16) * 0.1 - 1.0, q99=np.arange(16) * 0.2 + 1.0),
        "state": NormStats(q01=np.arange(7) * -0.2 - 1.0, q99=np.arange(7) * 0.1 + 1.0),
        "force_history": NormStats(q01=np.arange(6) - 20.0, q99=np.arange(6) + 20.0),
    }
    norm_path = tmp_path / f"{algorithm}_norm.json"
    save_norm_stats(stats, norm_path, meta=split)
    norm_text = norm_path.read_text()
    provenance = {
        "split": split,
        "norm_stats_meta": split,
        "norm_stats_sha256": hashlib.sha256(norm_text.encode()).hexdigest(),
        "global_step_semantics": "completed_optimizer_updates",
    }
    checkpoint = save_checkpoint(
        model,
        config,
        step=1,
        output_dir=tmp_path / f"{algorithm}_checkpoints",
        provenance=provenance,
        norm_stats_text=norm_text,
    )
    return _Bundle(algorithm, checkpoint, zarr_path, session_type, module, model_type)


@pytest.mark.parametrize("algorithm", ["flow", "diffusion", "act"])
def test_checkpoint_session_rollout_is_reproducible(tmp_path, monkeypatch, algorithm):
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm)
    # ACT is a SINGLE-PASS CVAE: num_steps is structurally 1 (its session and its model both
    # refuse anything else), and with z pinned at the prior mean the seed cannot change the
    # output. So the "different seed differs" assertion below is inverted for ACT — that
    # determinism is the property worth pinning, not a limitation to route around.
    steps = 1 if algorithm == "act" else 2
    seed_matters = algorithm != "act"
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu",
                                 num_steps=steps)
    try:
        first = session.infer(0, seed=17)
        second = session.infer(0, seed=17)
        different = session.infer(0, seed=18)
    finally:
        session.close()
    with pytest.raises(RuntimeError, match="is closed"):
        session.infer(0, seed=17)
    session.close()  # Idempotent cleanup is safe for CLI finally blocks.

    assert first.algorithm == algorithm
    assert first.layout_mode == "left_only"
    assert first.sample.split == "train"
    assert first.sample.file_frame == 0 and first.sample.episode_frame == 0
    assert first.predicted_actions.shape == (ACTION_HORIZON, 16)
    assert np.isfinite(first.predicted_actions).all()
    # Step k lands at k*ACTION_DT, t0-inclusive: 0, 33.3 ms, ... on the v3 30 Hz grid.
    # allclose, not array_equal: 1/30 is not representable in binary floating point, so
    # `k * ACTION_DT` (what the session computes) and `k / ACTION_HZ` differ by up to
    # 1 ulp on some k. Asserting bit equality would pin one spelling of the arithmetic.
    np.testing.assert_allclose(first.action_times_s,
                               np.arange(ACTION_HORIZON) / ACTION_HZ, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first.predicted_actions, second.predicted_actions)
    if seed_matters:
        assert not np.array_equal(first.predicted_actions, different.predicted_actions)
    else:
        np.testing.assert_array_equal(first.predicted_actions, different.predicted_actions)
    assert len(first.conditioning_images) == 1
    assert first.conditioning_images[0].shape == (8, 8, 3)
    assert first.roles[0].predicted_wrench.shape == (ACTION_HORIZON, 6)
    with pytest.raises(ValueError):
        first.predicted_actions.setflags(write=True)


@pytest.mark.parametrize("algorithm", ["flow", "diffusion", "act"])
def test_production_model_constructor_loads_checkpoint(tmp_path, monkeypatch, algorithm):
    """Exercise the real timm constructor; Diffusion also uses its production UNet path."""
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm, production=True)
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    try:
        result = session.infer(0, seed=3, num_steps=1)
    finally:
        session.close()
    assert result.predicted_actions.shape == (ACTION_HORIZON, 16)
    assert len(result.conditioning_images) == 2
    assert result.conditioning_images[0].shape == (224, 224, 3)

    root = Path(__file__).resolve().parents[1]
    script = f"infer_s1_{algorithm}.py"
    # ACT's CLI deliberately has NO --num-steps: it is a single-pass CVAE, and a flag that
    # silently did nothing would be worse than its absence.
    step_args = [] if algorithm == "act" else ["--num-steps", "1"]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / script),
            str(bundle.checkpoint),
            "--zarr-path",
            bundle.zarr_path,
            "--device",
            "cpu",
            *step_args,
            "--self-test",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["algorithm"] == algorithm


@pytest.mark.parametrize("algorithm", ["flow", "diffusion", "act"])
def test_normalization_has_numerical_oracle(tmp_path, monkeypatch, algorithm):
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm, history_len=2)
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    _, raw_sample = session.source.get(3)
    captured = {}

    def fake_sample(_model, batch, **_kwargs):
        captured["state"] = batch["state"].detach().cpu().numpy()
        captured["force_history"] = batch["force_history"].detach().cpu().numpy()
        captured["images"] = [image.detach().cpu().numpy() for image in batch["images"]]
        # (B, ACTION_HORIZON, action_dim): the horizon must track the layout, or the
        # session's own shape check rejects the stub before the oracle is compared.
        return torch.zeros((1, ACTION_HORIZON, 16), device=batch["state"].device)

    session.model.sample_actions = types.MethodType(fake_sample, session.model)
    try:
        result = session.infer(3, seed=0)
    finally:
        session.close()

    def normalize(raw, stats):
        scale = np.maximum(stats.q99 - stats.q01, 0.02) + 1e-6
        return (raw - stats.q01) / scale * 2.0 - 1.0

    expected_state = normalize(raw_sample["state"].unsqueeze(0).numpy(), session.norm_stats["state"])
    expected_force = normalize(raw_sample["force_history"].unsqueeze(0).numpy(), session.norm_stats["force_history"])
    action_stats = session.norm_stats["actions"]
    action_scale = np.maximum(action_stats.q99 - action_stats.q01, 0.02) + 1e-6
    expected_actions = np.broadcast_to(action_stats.q01 + 0.5 * action_scale, (ACTION_HORIZON, 16))
    np.testing.assert_allclose(captured["state"], expected_state, atol=1e-7)
    np.testing.assert_allclose(captured["force_history"], expected_force, atol=1e-7)
    np.testing.assert_allclose(result.predicted_actions, expected_actions, atol=1e-7)
    image = raw_sample["image"].numpy()
    assert len(captured["images"]) == image.shape[0]
    for index, observed in enumerate(captured["images"]):
        np.testing.assert_array_equal(observed[0], image[index])
    assert not np.array_equal(captured["images"][0], captured["images"][-1])


@pytest.mark.parametrize("algorithm", ["flow", "diffusion", "act"])
def test_session_suppresses_pretrained_download(tmp_path, monkeypatch, algorithm):
    """A local checkpoint must NEVER trigger a network weight download.

    ACT matters here as much as the ViT presets: its ResNet trunk defaults to
    ``pretrained=True`` (upstream relies on ImageNet weights), so ``dataclass_config``
    forcing ``pretrained=False`` is the only thing standing between a serving restore and
    a torchvision download.
    """
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm)
    seen = []

    def factory(config):
        seen.append(config.pretrained)
        # ACT's trunk contract is (features, pos), not a pooled CLS vector.
        stub = _ACTStubBackbone(config.hidden_dim) if algorithm == "act" else _StubEncoder()
        return bundle.model_type(config, encoder=stub)

    model_name = {"flow": "DPFlowModel", "diffusion": "DPDiffusionModel", "act": "DPACTModel"}[algorithm]
    monkeypatch.setattr(bundle.module, model_name, factory)
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    session.close()
    assert seen == [False]


def test_norm_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    norm_path = bundle.checkpoint / "norm_stats.json"
    norm_path.write_text(norm_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_norm_metadata_tampering_is_rejected_after_rehash(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    norm_path = bundle.checkpoint / "norm_stats.json"
    payload = json.loads(norm_path.read_text())
    payload["_meta"]["strategy"] = "tampered"
    norm_text = json.dumps(payload)
    norm_path.write_text(norm_text)
    config_path = bundle.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["_provenance"]["norm_stats_sha256"] = hashlib.sha256(norm_text.encode()).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="metadata differs"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_reversed_normalization_quantiles_are_rejected(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    norm_path = bundle.checkpoint / "norm_stats.json"
    payload = json.loads(norm_path.read_text())
    payload["actions"]["q99"][0] = payload["actions"]["q01"][0] - 1.0
    norm_text = json.dumps(payload)
    norm_path.write_text(norm_text)
    config_path = bundle.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["_provenance"]["norm_stats_sha256"] = hashlib.sha256(norm_text.encode()).hexdigest()
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="q99 values below q01"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_changed_dataset_content_is_rejected(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "diffusion")
    changed = _write_hl_zarr(tmp_path / "changed.zarr.zip", n_frames=8, n_episodes=2, seed=99)
    with pytest.raises(ValueError, match="source_content_fingerprint"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(changed,), device="cpu")


def test_dataset_replacement_after_session_construction_is_rejected(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    replacement = _write_hl_zarr(tmp_path / "replacement.zarr.zip", n_frames=8, n_episodes=2, seed=99)
    Path(replacement).replace(bundle.zarr_path)
    try:
        with pytest.raises(ValueError, match="changed after HLDataset indexed"):
            session.infer(0, seed=0)
    finally:
        session.close()


def test_missing_model_config_field_is_rejected(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "diffusion")
    config_path = bundle.checkpoint / "config.json"
    raw = json.loads(config_path.read_text())
    del raw["diffusion"]["num_inference_steps"]
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing DPDiffusionConfig fields"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision", "mystery-float", "unsupported floating-point precision"),
        ("filled_state_dim", 4, "would truncate"),
        ("inference_steps", 2.5, "num_steps must be an integer"),
    ],
)
def test_malformed_model_precision_or_state_width_is_rejected(tmp_path, monkeypatch, field, value, message):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    config_path = bundle.checkpoint / "config.json"
    raw = json.loads(config_path.read_text())
    raw["model"][field] = value
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_diffusion_steps_cannot_exceed_training_schedule(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "diffusion")
    with pytest.raises(ValueError, match=r"num_steps must be in \[1, 8\]"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu", num_steps=9)
    with pytest.raises(ValueError, match="num_steps must be an integer"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu", num_steps=2.5)

    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    try:
        with pytest.raises(ValueError, match=r"num_steps must be in \[1, 8\]"):
            session.infer(0, num_steps=9)
    finally:
        session.close()


def test_diffusion_rejects_nonzero_scheduler_offset(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "diffusion")
    config_path = bundle.checkpoint / "config.json"
    raw = json.loads(config_path.read_text())
    raw["diffusion"]["steps_offset"] = 1
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="steps_offset must be 0"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_hl_data_config_must_be_complete(tmp_path, monkeypatch, mutation):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    config_path = bundle.checkpoint / "config.json"
    raw = json.loads(config_path.read_text())
    if mutation == "missing":
        del raw["hl_data"]["mode"]
    else:
        raw["hl_data"]["future_knob"] = True
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="HL data fields"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_strict_weight_loading_rejects_missing_tensor(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    weights_path = bundle.checkpoint / "model.safetensors"
    state = load_file(str(weights_path))
    state.pop(next(iter(state)))
    save_file(state, str(weights_path))
    _refresh_model_hash(bundle.checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_model_hash_rejects_same_shape_weight_swap(tmp_path, monkeypatch):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _make_bundle(first_root, monkeypatch, "flow")
    second = _make_bundle(second_root, monkeypatch, "flow")
    shutil.copyfile(second.checkpoint / "model.safetensors", first.checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="model.safetensors hash mismatch"):
        first.session_type(first.checkpoint, zarr_paths=(first.zarr_path,), device="cpu")


def test_weight_hash_and_deserialization_use_the_same_bytes(tmp_path, monkeypatch):
    import safetensors.torch

    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    weights_path = bundle.checkpoint / "model.safetensors"
    expected_payload = weights_path.read_bytes()
    real_load = safetensors.torch.load
    observed = []

    def replace_after_read(payload):
        observed.append(payload)
        weights_path.write_bytes(b"replacement after verified read")
        return real_load(payload)

    monkeypatch.setattr(safetensors.torch, "load", replace_after_read)
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    session.close()
    assert observed == [expected_payload]


def test_weight_dtype_must_match_checkpoint_config(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    weights_path = bundle.checkpoint / "model.safetensors"
    state = load_file(str(weights_path))
    name = next(name for name, tensor in state.items() if tensor.is_floating_point())
    state[name] = state[name].half()
    save_file(state, str(weights_path))
    _refresh_model_hash(bundle.checkpoint)
    with pytest.raises(ValueError, match="tensor dtype mismatch"):
        bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")


def test_masked_segments_do_not_bridge_invalid_steps():
    from fuji.s1_policy.inference.viewer import _segments

    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    valid = np.array([True, True, False, True, True])
    segments = _segments(points, valid)
    assert segments.shape == (2, 2, 3)
    np.testing.assert_array_equal(segments[0], points[:2])
    np.testing.assert_array_equal(segments[1], points[3:])


def test_viewer_rotation_and_sample_index_guards():
    import socket

    from fuji.s1_policy.inference.viewer import (
        _checked_gui_integer,
        _checked_port,
        _checked_sample_index,
        _require_port_available,
        _wxyz,
    )

    assert _wxyz(np.zeros(6)) == (1.0, 0.0, 0.0, 0.0)
    assert _wxyz(np.ones(6)) == (1.0, 0.0, 0.0, 0.0)
    assert _checked_sample_index(2, 3) == 2
    with pytest.raises(IndexError, match="outside"):
        _checked_sample_index(3, 3)
    assert _checked_port(8086) == 8086
    with pytest.raises(ValueError, match="port must be"):
        _checked_port(70000)
    with pytest.raises(ValueError, match="port must be"):
        _checked_port(65535)
    assert _checked_gui_integer(2.0, "steps") == 2
    assert _checked_gui_integer("9007199254740993", "seed") == 9007199254740993
    with pytest.raises(ValueError, match="must be an integer"):
        _checked_gui_integer(2.5, "steps")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(RuntimeError, match="port .* is unavailable"):
            _require_port_available("127.0.0.1", occupied.getsockname()[1])


def test_conditioning_image_round_trip_is_exact():
    from fuji.s1_policy.inference.common import conditioning_images

    pixels = np.array([[[1, 127, 254], [255, 3, 128]]], dtype=np.uint8)
    normalized = (pixels.astype(np.float32) / 255.0 - 0.5) / 0.5
    sample = {"image": torch.from_numpy(np.transpose(normalized, (2, 0, 1)))}
    (recovered,) = conditioning_images(sample, mean=0.5, std=0.5)
    np.testing.assert_array_equal(recovered, pixels)


def test_native_bimanual_head_layout_and_masks(tmp_path):
    from fuji.s1_policy.inference.common import HLSampleSource, build_rollout_result

    zarr_path = _write_hl_zarr(
        tmp_path / "three.zarr.zip",
        n_frames=8,
        n_episodes=2,
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        attrs_override={"hands_mode": "both_head"},
    )
    hl_data = {
        "zarr_paths": [zarr_path],
        "mode": "both_hands_head",
        "image_key": "camera0_rgb",
        "image_mean": 0.5,
        "image_std": 0.5,
        "obs_history_len": 2,
        "obs_history_dt_s": 0.25,
        "drop_invalid_force_history": False,
        "drop_glitch_windows": False,
        "norm_stats_path": "",
        "val_last_episodes": 1,
    }
    source = HLSampleSource(hl_data)
    ref, original = source.get(0)
    sample = dict(original)
    sample["action_mask"] = original["action_mask"].clone()
    layout = source.layout
    sample["action_mask"][3, layout.slots("left").pos] = False
    sample["action_mask"][4, layout.slots("left").rot6d] = False
    sample["action_mask"][5, layout.slots("right").pos] = False
    result = build_rollout_result(
        algorithm="flow",
        num_steps=1,
        seed=0,
        ref=ref,
        sample=sample,
        predicted_actions=sample["action_chunk"].numpy(),
        source=source,
    )
    source.close()

    assert result.predicted_actions.shape == (ACTION_HORIZON, 41)
    assert tuple(role.role for role in result.roles) == ("left", "right", "head")
    assert not result.roles[0].pose_valid[3]
    assert not result.roles[0].pose_valid[4]
    assert not result.roles[0].position_valid[3].all()
    assert result.roles[0].position_valid[4].all()
    assert result.roles[0].rotation_valid[3].all()
    assert not result.roles[0].rotation_valid[4].all()
    assert result.roles[0].pose_valid[5]
    assert result.roles[1].pose_valid[3] and result.roles[1].pose_valid[4]
    assert not result.roles[1].pose_valid[5]
    assert result.roles[2].pose_valid.all()
    for role, base in zip(result.roles, (0.0, 100.0, 200.0), strict=True):
        np.testing.assert_allclose(role.target_position[0], np.full(3, base), atol=1e-5)
        np.testing.assert_allclose(role.target_position[1], np.full(3, base + 0.001), atol=1e-5)
    assert result.roles[0].predicted_gripper_m is not None
    assert result.roles[0].predicted_wrench is not None
    assert result.roles[1].predicted_gripper_m is None
    assert result.roles[1].predicted_wrench is None
    assert result.roles[2].predicted_gripper_m is None
    assert result.roles[2].predicted_wrench is None


def test_episode_forward_positions_connect_to_target_and_stop_at_episode_end(tmp_path):
    """The green rest-of-episode path is the raw recording from the sampled frame to its
    episode end, in the SAME absolute frame as the blue target line (row 0 == target
    step 0), and it never crosses an episode boundary."""
    from fuji.s1_policy.inference.common import HLSampleSource, build_rollout_result

    # n_frames=8, n_episodes=2 -> episode_ends [4, 8]; drop_glitch_windows off so sample
    # index i is anchor (fi=0, frame=i). Left base 0: anchor_pos[f] = [f*.01, f*.02, f*.03].
    zarr_path = _write_hl_zarr(tmp_path / "forward.zarr.zip", n_frames=8, n_episodes=2)
    hl_data = {
        "zarr_paths": [zarr_path],
        "mode": "left_only",
        "image_key": "camera0_rgb",
        "image_mean": 0.5,
        "image_std": 0.5,
        "obs_history_len": 1,
        "obs_history_dt_s": 0.25,
        "drop_invalid_force_history": False,
        "drop_glitch_windows": False,
        "norm_stats_path": "",
        "val_last_episodes": 1,
    }
    source = HLSampleSource(hl_data)
    try:
        ref, sample = source.get(1)  # episode 0, file_frame 1
        assert ref.file_frame == 1 and ref.file_index == 0
        forward = source.episode_forward_positions(ref)
        left = forward["left"]
        # Frames 1..3 (episode 0 ends at 4): three rows, stops before the episode boundary.
        expected = np.stack(
            [np.arange(1, 4) * 0.01, np.arange(1, 4) * 0.02, np.arange(1, 4) * 0.03], axis=1
        )
        np.testing.assert_allclose(left, expected, atol=1e-6)

        # Row 0 coincides with the decoded target chunk's step-0 target (same world frame).
        result = build_rollout_result(
            algorithm="flow",
            num_steps=1,
            seed=0,
            ref=ref,
            sample=sample,
            predicted_actions=sample["action_chunk"].numpy(),
            source=source,
        )
        np.testing.assert_allclose(left[0], result.roles[0].target_position[0], atol=1e-6)

        # Last frame of an episode yields a single point (no drawable segment).
        last_ref, _ = source.get(3)  # episode 0's final frame
        assert source.episode_forward_positions(last_ref)["left"].shape == (1, 3)
    finally:
        source.close()


def test_episode_forward_positions_route_per_role(tmp_path):
    """Each role reads its OWN camera-grid position array (no cross-wiring)."""
    from fuji.s1_policy.data.hl_dataset import HLDataset

    zarr_path = _write_hl_zarr(
        tmp_path / "three_forward.zarr.zip",
        n_frames=8,
        n_episodes=2,
        roles=("left", "right", "head"),
        force_roles=("left",),
        gripper_roles=("left",),
        attrs_override={"hands_mode": "both_head"},
    )
    dataset = HLDataset(zarr_path, mode="both_hands_head", obs_history_len=1, drop_glitch_windows=False)
    try:
        forward = dataset.episode_forward_positions(0, 0)  # episode 0, frame 0
        assert set(forward) == {"left", "right", "head"}
        for role, base in (("left", 0.0), ("right", 100.0), ("head", 200.0)):
            assert forward[role].shape == (4, 3)
            np.testing.assert_allclose(forward[role][0], np.full(3, base), atol=1e-6)
            np.testing.assert_allclose(forward[role][1], base + np.array([0.01, 0.02, 0.03]), atol=1e-6)
        with pytest.raises(IndexError, match="file_frame"):
            dataset.episode_forward_positions(0, 8)
        with pytest.raises(IndexError, match="file_index"):
            dataset.episode_forward_positions(5, 0)
    finally:
        dataset.close()


def test_physical_metrics_have_expected_units(tmp_path):
    from fuji.s1_policy.inference.common import HLSampleSource, build_rollout_result

    zarr_path = _write_hl_zarr(tmp_path / "metrics.zarr.zip", n_frames=8, n_episodes=2)
    hl_data = {
        "zarr_paths": [zarr_path],
        "mode": "left_only",
        "image_key": "camera0_rgb",
        "image_mean": 0.5,
        "image_std": 0.5,
        "obs_history_len": 1,
        "obs_history_dt_s": 0.25,
        "drop_invalid_force_history": False,
        "drop_glitch_windows": False,
        "norm_stats_path": "",
        "val_last_episodes": 1,
    }
    source = HLSampleSource(hl_data)
    ref, sample = source.get(0)
    prediction = sample["action_chunk"].numpy().copy()
    slots = source.layout.slots("left")
    prediction[:, slots.pos.start] += 0.01
    prediction[:, slots.gripper] += 0.1
    prediction[:, slots.force.start] += 2.0
    prediction[:, slots.force.start + 3] += 0.5
    result = build_rollout_result(
        algorithm="flow",
        num_steps=1,
        seed=0,
        ref=ref,
        sample=sample,
        predicted_actions=prediction,
        source=source,
    )
    source.close()
    assert result.metrics["position_cm"] == pytest.approx(1.0, abs=1e-5)
    assert result.metrics["gripper_mm"] == pytest.approx(9.5, abs=1e-5)
    assert result.metrics["force_n"] == pytest.approx(2.0, abs=1e-5)
    assert result.metrics["torque_nm"] == pytest.approx(0.5, abs=1e-5)


def test_degenerate_predicted_rotation_is_rejected(tmp_path):
    from fuji.s1_policy.inference.common import HLSampleSource, build_rollout_result

    zarr_path = _write_hl_zarr(tmp_path / "degenerate.zarr.zip", n_frames=8, n_episodes=2)
    hl_data = {
        "zarr_paths": [zarr_path],
        "mode": "left_only",
        "image_key": "camera0_rgb",
        "image_mean": 0.5,
        "image_std": 0.5,
        "obs_history_len": 1,
        "obs_history_dt_s": 0.25,
        "drop_invalid_force_history": False,
        "drop_glitch_windows": False,
        "norm_stats_path": "",
        "val_last_episodes": 1,
    }
    source = HLSampleSource(hl_data)
    ref, sample = source.get(0)
    prediction = sample["action_chunk"].numpy().copy()
    prediction[:, source.layout.slots("left").rot6d] = 0.0
    try:
        with pytest.raises(FloatingPointError, match="rot6d is degenerate"):
            build_rollout_result(
                algorithm="flow",
                num_steps=1,
                seed=0,
                ref=ref,
                sample=sample,
                predicted_actions=prediction,
                source=source,
            )
    finally:
        source.close()


def test_masked_svg_chart_splits_target_runs():
    from fuji.s1_policy.inference.viewer import _chart_svg

    times = np.arange(5, dtype=np.float64) / 16.0
    values = np.arange(15, dtype=np.float64).reshape(5, 3)
    valid = np.ones((5, 3), dtype=bool)
    valid[2] = False
    svg = _chart_svg("position", ("x", "y", "z"), times, values, values, valid, ("red", "green", "blue"))
    assert svg.count('data-kind="target" data-axis="0"') == 2
    assert svg.count('data-kind="target"') == 6
    assert "nan" not in svg.lower()
    assert "overflow-x:auto" in svg and "min-width:720px" in svg

    all_invalid = _chart_svg(
        "position", ("x", "y", "z"), times, values, values, np.zeros_like(valid), ("red", "green", "blue")
    )
    assert 'data-kind="target"' not in all_invalid


def test_the_three_algorithm_imports_are_isolated():
    """Importing one algorithm's session must not initialize another's model.

    For ACT this is not merely tidiness: its module pulls in ``torchvision.models`` and can
    construct an ImageNet-pretrained ResNet, so leaking it into a flow-only serving process
    would make that process pay for a vision-model import it never uses. And in the other
    direction, an ACT deploy must not drag in ``diffusers``.
    """
    root = Path(__file__).resolve().parents[1]
    checks = (
        ("fuji.s1_policy.inference.flow", "fuji.s1_policy.model.dp_diffusion"),
        ("fuji.s1_policy.inference.flow", "fuji.s1_policy.model.dp_act"),
        ("fuji.s1_policy.inference.diffusion", "fuji.s1_policy.model.dp_flow"),
        ("fuji.s1_policy.inference.diffusion", "fuji.s1_policy.model.dp_act"),
        ("fuji.s1_policy.inference.act", "fuji.s1_policy.model.dp_flow"),
        ("fuji.s1_policy.inference.act", "fuji.s1_policy.model.dp_diffusion"),
        ("fuji.s1_policy.inference.act", "diffusers"),
    )
    for imported, forbidden in checks:
        code = f"import sys; import {imported}; assert {forbidden!r} not in sys.modules, sorted(sys.modules)"
        subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_lazy_model_exports_support_introspection_and_cache():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import fuji.s1_policy.model as model; "
        "assert set(model.__all__) <= set(dir(model)); "
        "assert 'fuji.s1_policy.model.dp_flow' not in sys.modules; "
        "value = model.DPFlowConfig; "
        "assert model.__dict__['DPFlowConfig'] is value; "
        "assert 'fuji.s1_policy.model.dp_diffusion' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_inference_imports_no_third_party_or_viser_runtime():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import pathlib, sys; import fuji.s1_policy.inference.flow; "
        "import fuji.s1_policy.inference.diffusion; "
        "bad = {name: str(getattr(module, '__file__', '')) for name, module in sys.modules.items() "
        "if '/third_party/' in str(getattr(module, '__file__', '')).replace('\\\\', '/')}; "
        "assert not bad, bad; assert 'viser' not in sys.modules; assert 'mink' not in sys.modules; "
        "assert 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


@pytest.mark.parametrize(
    ("algorithm", "script_module", "session_name"),
    [
        ("flow", "scripts.infer_s1_flow", "FlowInferenceSession"),
        ("diffusion", "scripts.infer_s1_diffusion", "DiffusionInferenceSession"),
        ("act", "scripts.infer_s1_act", "ACTInferenceSession"),
    ],
)
def test_cli_self_test_is_headless_and_closes(tmp_path, monkeypatch, capsys, algorithm, script_module, session_name):
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm)
    real_session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cpu")
    result = real_session.infer(0, seed=5)
    real_session.close()
    state = {"closed": False}

    # ACT's CLI has no --num-steps (single-pass CVAE), so its session is constructed
    # WITHOUT that kwarg. Asserting `"num_steps" not in kwargs` rather than skipping the
    # check keeps the flag's absence a tested contract: if someone adds a no-op
    # --num-steps to infer_s1_act.py, this fails.
    expects_steps = algorithm != "act"

    class FakeSession:
        def __init__(self, *args, **kwargs):
            assert args == ("unused",)
            assert kwargs["device"] == "cpu"
            assert kwargs["zarr_paths"] == ("one.zarr.zip", "two.zarr.zip")
            if expects_steps:
                assert kwargs["num_steps"] == 7
            else:
                assert "num_steps" not in kwargs

        def infer(self, sample_index, *, seed):
            assert sample_index == 0 and seed == 5
            return result

        def close(self):
            state["closed"] = True

    module = importlib.import_module(script_module)
    monkeypatch.setattr(module, session_name, FakeSession)
    exit_code = module.main(
        [
            "unused",
            "--zarr-path",
            "one.zarr.zip",
            "--zarr-path",
            "two.zarr.zip",
            "--device",
            "cpu",
            *(["--num-steps", "7"] if expects_steps else []),
            "--seed",
            "5",
            "--self-test",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["algorithm"] == algorithm and output["finite"] is True
    assert state["closed"] is True
    assert "viser" not in sys.modules


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("algorithm", ["flow", "diffusion", "act"])
def test_cuda_seed_is_bit_exact(tmp_path, monkeypatch, algorithm):
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm)
    session = bundle.session_type(bundle.checkpoint, zarr_paths=(bundle.zarr_path,), device="cuda")
    # ACT: single-pass, so 1 step, and z=0 makes the seed inert (checked below).
    steps = 1 if algorithm == "act" else 2
    try:
        first = session.infer(0, seed=23, num_steps=steps)
        second = session.infer(0, seed=23, num_steps=steps)
        different = session.infer(0, seed=24, num_steps=steps)
        if algorithm == "diffusion":
            assert any(
                isinstance(value, torch.Tensor) and value.device.type == "cuda"
                for value in vars(session.model.noise_scheduler).values()
            )
    finally:
        session.close()
    if algorithm == "diffusion":
        assert all(
            not isinstance(value, torch.Tensor) or value.device.type == "cpu"
            for value in vars(session.model.noise_scheduler).values()
        )
    np.testing.assert_array_equal(first.predicted_actions, second.predicted_actions)
    if algorithm == "act":
        # z is the prior MEAN, so the seed is inert — the rollout is bit-identical across
        # seeds on CUDA too. That determinism is ACT's deploy property, not a shortfall.
        np.testing.assert_array_equal(first.predicted_actions, different.predicted_actions)
    else:
        assert not np.array_equal(first.predicted_actions, different.predicted_actions)
