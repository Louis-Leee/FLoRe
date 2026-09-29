"""Fast tests for episode validation, deterministic evaluation, and exact resume."""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_hl_dataset import _write_hl_zarr

zarr = pytest.importorskip("zarr")


class _DummyModel(torch.nn.Module):
    def __init__(self, _config=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_weights = []
        self._last_actions_valid = False

    def forward(self, batch):
        self.seen_weights.append(float(self.weight.detach()))
        error = torch.rand_like(batch["actions"]) * self.weight
        mask = batch.get("action_mask", torch.ones_like(error, dtype=torch.bool)).float()
        num = (error.square() * mask).sum()
        den = mask.sum()
        sums = {
            "overall": (num, den),
            "action_nonforce": (num, den),
            "pos": (num, den),
            "rot": (num, den),
            "grip": (num, den),
            "force": (num, den),
        }
        return {"loss": num / den.clamp(min=1), "metrics": {}, "metric_sums": sums}

    def reset(self):
        self._last_actions_valid = False

    def sample_actions(self, batch, generator=None, **_kwargs):
        return (
            torch.randn(
                batch["actions"].shape,
                device=batch["actions"].device,
                generator=generator,
            )
            * self.weight
        )


class _FakeEMA:
    def __init__(self, shadow: float):
        self.shadow = shadow
        self.saved = None
        self.copy_count = 0

    def store(self, parameters):
        self.saved = [p.detach().clone() for p in parameters]

    def copy_to(self, parameters):
        self.copy_count += 1
        for parameter in parameters:
            parameter.data.fill_(self.shadow)

    def restore(self, parameters):
        for parameter, saved in zip(parameters, self.saved):
            parameter.data.copy_(saved)


def _config(path, tmp_path, **kwargs):
    from fuji.s1_policy.training.config import HLDataConfig, TrainConfig

    data = HLDataConfig(
        zarr_paths=(path,),
        val_last_episodes=kwargs.pop("val_last_episodes", 1),
        drop_glitch_windows=False,
        obs_history_len=1,
        norm_stats_path=kwargs.pop("norm_stats_path", ""),
    )
    return TrainConfig(
        hl_data=data,
        batch_size=kwargs.pop("batch_size", 3),
        num_workers=0,
        device="cpu",
        bf16=False,
        ema=kwargs.pop("ema", False),
        max_steps=kwargs.pop("max_steps", 4),
        output_dir=str(tmp_path / "out"),
        eval_seeds=kwargs.pop("eval_seeds", (0, 1)),
        **kwargs,
    )


def test_clean99_episode_split_has_no_frame_leakage(tmp_path):
    from fuji.s1_policy.data.hl_dataset import HLDataset

    path = _write_hl_zarr(tmp_path / "clean99.zarr.zip", n_frames=99, n_episodes=99)
    dataset = HLDataset(path, drop_glitch_windows=False, require_episode_ends=True)
    train, val = dataset.split_indices(4)
    assert dataset.episode_count == 99
    assert dataset.split_provenance(4)["val_episode_ids"] == [95, 96, 97, 98]
    assert set(dataset.sample_episode_ids[i] for i in train) == set(range(95))
    assert set(dataset.sample_episode_ids[i] for i in val) == {95, 96, 97, 98}
    assert set(train).isdisjoint(val)
    provenance = dataset.split_provenance(4)
    assert provenance["transform_contract_version"] == 1
    assert provenance["train_anchor_count"] == 95
    assert provenance["val_anchor_count"] == 4
    assert len(provenance["anchor_indices_fingerprint"]) == 64
    assert len(provenance["source_content_fingerprint"]) == 64
    assert all(len(entry) == 2 for entry in dataset._index)  # historical tuple API
    dataset.close()


def test_multi_file_episode_ids_are_global_and_unambiguous(tmp_path):
    from fuji.s1_policy.data.hl_dataset import HLDataset

    a = _write_hl_zarr(tmp_path / "a.zarr.zip", n_frames=6, n_episodes=2)
    b = _write_hl_zarr(tmp_path / "b.zarr.zip", n_frames=9, n_episodes=3)
    dataset = HLDataset([a, b], drop_glitch_windows=False, require_episode_ends=True)
    assert dataset.episode_identities == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    assert set(dataset.sample_episode_ids) == set(range(5))
    train, val = dataset.split_indices(2)
    assert set(dataset.sample_episode_ids[i] for i in val) == {3, 4}
    assert not set(dataset.sample_episode_ids[i] for i in train) & {3, 4}
    dataset.close()


@pytest.mark.parametrize("invalid", ["missing", "non_monotonic"])
def test_validation_refuses_missing_or_invalid_episode_ends(tmp_path, invalid):
    from fuji.s1_policy.data.hl_dataset import HLDataset

    path = _write_hl_zarr(tmp_path / f"{invalid}.zarr.zip", n_frames=12, n_episodes=3)
    if invalid == "missing":
        rewritten = tmp_path / "without_ends.zarr.zip"
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(rewritten, "w") as target:
            for item in source.infolist():
                if not item.filename.startswith("meta/episode_ends/"):
                    target.writestr(item, source.read(item.filename))
        path = str(rewritten)
    else:
        store = zarr.ZipStore(path, mode="a")
        root = zarr.group(store=store)
        root["meta"]["episode_ends"][:] = np.array([4, 3, 12], dtype=np.int64)
        store.close()
    with pytest.raises(ValueError, match="episode_ends"):
        HLDataset(path, drop_glitch_windows=False, require_episode_ends=True)


def test_norm_fit_uses_only_train_episodes_and_records_provenance(tmp_path, monkeypatch):
    from scripts import fit_hl_norm_stats

    path = _write_hl_zarr(tmp_path / "stats.zarr.zip", n_frames=12, n_episodes=6)
    output = tmp_path / "stats.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_hl_norm_stats.py",
            "--zarr-paths",
            path,
            "--out",
            str(output),
            "--val-last-episodes",
            "2",
            "--keep-glitch-windows",
            "--obs-history-len",
            "1",
        ],
    )
    fit_hl_norm_stats.main()
    payload = json.loads(output.read_text())
    meta = payload["_meta"]
    assert meta["total_episodes"] == 6
    assert meta["train_episode_ids"] == [0, 1, 2, 3]
    assert meta["val_episode_ids"] == [4, 5]
    assert len(meta["episode_ends_fingerprint"]) == 64
    assert meta["fit_train_anchor_count"] == 8
    assert meta["dataset_knobs"]["obs_history_len"] == 1
    assert meta["dataset_knobs"]["drop_invalid_force_history"] is False
    # Builder force values encode frame ID; held-out frames 8..11 cannot affect q99.
    assert max(payload["force_history"]["q99"]) < 8.0


def test_split_provenance_mismatch_is_rejected():
    from fuji.s1_policy.training.trainer import validate_split_provenance

    actual = {
        "transform_contract_version": 1,
        "strategy": "last_global_flattened_episodes",
        "val_last_episodes": 4,
        "total_episodes": 99,
        "train_episode_ids": list(range(95)),
        "val_episode_ids": [95, 96, 97, 98],
        "episode_identities": [[0, i] for i in range(99)],
        "episode_ends_fingerprint": "a" * 64,
        "anchor_indices_fingerprint": "b" * 64,
        "source_content_fingerprint": "c" * 64,
        "train_anchor_count": 95,
        "val_anchor_count": 4,
        "dataset_knobs": {"mode": "left_only"},
    }
    validate_split_provenance(dict(actual), actual)
    stale = dict(actual)
    stale["val_episode_ids"] = [94, 95, 96, 97]
    with pytest.raises(ValueError, match="leakage"):
        validate_split_provenance(stale, actual)


@pytest.mark.parametrize("mutation", ["q99_shape", "nonfinite", "reversed"])
def test_training_rejects_malformed_normalization_before_use(mutation):
    from fuji.s1_policy.data.norm_stats import NormStats
    from fuji.s1_policy.training.trainer import Trainer

    q01 = np.zeros(3, dtype=np.float64)
    q99 = np.ones(3, dtype=np.float64)
    if mutation == "q99_shape":
        q99 = np.ones(1, dtype=np.float64)
        message = "q01/q99 shapes"
    elif mutation == "nonfinite":
        q99[0] = np.inf
        message = "non-finite"
    else:
        q99[0] = -1.0
        message = "below q01"
    trainer = object.__new__(Trainer)
    trainer.norm_stats = {"actions": NormStats(q01=q01, q99=q99)}
    with pytest.raises(ValueError, match=message):
        trainer._check_norm_stats({"actions": torch.zeros(2, 4, 3)})


def test_deploy_sampler_step_config_is_validated():
    from fuji.s1_policy.model.base import FlowMatchingVLABase
    from fuji.s1_policy.model.diffusion_base import DiffusionVLABase
    from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from fuji.s1_policy.model.dp_flow import DPFlowConfig
    from fuji.s1_policy.training.config import TrainConfig

    with pytest.raises(ValueError, match="inference_steps must be a positive integer"):
        TrainConfig(model=DPFlowConfig(inference_steps=0))
    with pytest.raises(ValueError, match="must be an integer in"):
        TrainConfig(
            algo="diffusion",
            diffusion=DPDiffusionConfig(num_train_timesteps=8, num_inference_steps=9, clip_sample=False),
        )
    with pytest.raises(ValueError, match="positive integer"):
        TrainConfig(model=DPFlowConfig(inference_steps=2.5))
    with pytest.raises(ValueError, match="steps_offset must be 0"):
        TrainConfig(algo="diffusion", diffusion=DPDiffusionConfig(steps_offset=1, clip_sample=False))
    flow = TrainConfig(model=DPFlowConfig(inference_steps=np.int64(2)))
    diffusion = TrainConfig(
        algo="diffusion",
        diffusion=DPDiffusionConfig(
            num_train_timesteps=np.int64(8),
            num_inference_steps=np.int64(2),
            clip_sample=False,
        ),
    )
    assert type(flow.model.inference_steps) is int
    assert type(diffusion.diffusion.num_train_timesteps) is int
    assert type(diffusion.diffusion.num_inference_steps) is int
    with pytest.raises(ValueError, match="unsupported floating-point precision"):
        FlowMatchingVLABase._parse_dtype(torch.int64)
    with pytest.raises(ValueError, match="unsupported floating-point precision"):
        DiffusionVLABase._parse_dtype(torch.int64)


def test_train_and_val_loader_semantics_include_validation_tail(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "loader.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    trainer = trainer_module.Trainer(_config(path, tmp_path, batch_size=3))
    train_loader, val_loader = trainer.build_dataloaders()
    assert train_loader.drop_last is True
    assert val_loader.drop_last is False
    assert [batch["actions"].shape[0] for batch in val_loader] == [3, 1]
    assert trainer.train_sampler is train_loader.sampler
    trainer.dataset.close()


def test_deterministic_eval_is_rng_isolated_and_uses_then_restores_ema(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "eval.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    trainer = trainer_module.Trainer(_config(path, tmp_path))
    trainer.build_dataloaders()
    trainer.ema = _FakeEMA(shadow=3.0)
    before = torch.get_rng_state().clone()
    first = trainer.evaluate()
    assert torch.equal(torch.get_rng_state(), before)
    second = trainer.evaluate()
    assert first == second
    assert trainer.ema.copy_count == 2
    assert trainer.model.seen_weights and set(trainer.model.seen_weights) == {3.0}
    assert trainer.model.weight.item() == pytest.approx(1.0)
    trainer.dataset.close()


def test_metric_accumulator_uses_sum_and_count_not_batch_means():
    from fuji.s1_policy.training.metrics import MetricAccumulator

    aggregate = MetricAccumulator()
    aggregate.add({"overall": (torch.tensor(10.0), torch.tensor(10.0))})
    aggregate.add({"overall": (torch.tensor(9.0), torch.tensor(1.0)), "force": (torch.tensor(0.0), torch.tensor(0.0))})
    means = aggregate.means()
    assert means["overall"] == pytest.approx(19.0 / 11.0)
    assert math.isnan(means["force"])


def test_physical_metrics_are_mask_aware_and_use_physical_units():
    from fuji.s1_policy.data.layout import get_layout
    from fuji.s1_policy.training.metrics import physical_trajectory_sums
    from fuji.utils.rot6d import rotmat_to_rot6d

    layout = get_layout("left_only")
    target = np.zeros((1, 1, layout.action_dim), dtype=np.float64)
    pred = target.copy()
    slots = layout.slots("left")
    identity = np.eye(3)
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target[..., slots.rot6d] = rotmat_to_rot6d(identity)
    pred[..., slots.rot6d] = rotmat_to_rot6d(quarter_turn)
    pred[..., slots.pos] = [0.01, 0.0, 0.0]
    pred[..., slots.gripper] = 0.1
    pred[..., slots.force] = [3.0, 4.0, 0.0, 0.0, 0.0, 2.0]
    mask = np.ones_like(target, dtype=bool)
    sums = physical_trajectory_sums(pred, target, mask, layout)
    assert sums["position_cm"] == pytest.approx((1.0, 1.0))
    assert sums["rotation_geodesic_deg"] == pytest.approx((90.0, 1.0))
    assert sums["gripper_mm"] == pytest.approx((9.5, 1.0))
    assert sums["force_n"] == pytest.approx((5.0, 1.0))
    assert sums["torque_nm"] == pytest.approx((2.0, 1.0))
    mask[..., slots.force] = False
    masked = physical_trajectory_sums(pred, target, mask, layout)
    assert masked["force_n"][1] == 0 and masked["torque_nm"][1] == 0


def test_best_selection_and_metric_names():
    from fuji.s1_policy.training.trainer import is_improvement, metric_log

    assert is_improvement(1.0, math.inf)
    assert not is_improvement(math.nan, 1.0)
    assert not is_improvement(2.0, 1.0)
    logs = metric_log("val", {"overall": 2.0, "force": 3.0}, "flow_loss")
    assert logs["val/objective"] == 2.0
    assert logs["val/flow_loss"] == 2.0
    assert logs["val/flow_loss/force"] == 3.0


def test_epoch_sampler_mid_epoch_resume_preserves_remaining_order():
    from fuji.s1_policy.training.trainer import EpochShuffleSampler

    full = EpochShuffleSampler(11, seed=7)
    full.set_epoch(3)
    order = list(full)
    resumed = EpochShuffleSampler(11, seed=7)
    resumed.load_state_dict({"size": 11, "seed": 7, "epoch": 3, "offset": 6})
    assert list(resumed) == order[6:]


def test_resume_restores_completed_step_optimizer_scheduler_and_sampler(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "resume.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(path, tmp_path, batch_size=2)
    trainer = trainer_module.Trainer(config)
    trainer.build_dataloaders()
    trainer.optimizer.zero_grad()
    trainer.model.weight.square().backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.global_step = 1
    trainer.epoch = 0
    trainer.batch_in_epoch = 2
    trainer.best_metrics = {"overall": 0.5, "force": 0.75}
    trainer._wandb_run_id = "resume-run-id"
    checkpoint = trainer._save_resume_checkpoint()
    expected_next_random = torch.rand(4)
    saved = torch.load(checkpoint / "training_state.pt", weights_only=False)
    assert saved["global_step"] == 1
    assert saved["global_step_semantics"] == "completed_optimizer_updates"
    assert saved["sampler"]["offset"] == 4
    assert saved["train_generator_state"].device.type == "cpu"
    assert saved["rng"]["torch_cpu"].device.type == "cpu"

    resumed_config = _config(path, tmp_path, batch_size=2, resume_from=str(checkpoint))
    resumed = trainer_module.Trainer(resumed_config)
    assert torch.allclose(torch.rand(4), expected_next_random)
    resumed.build_dataloaders()
    assert resumed.global_step == 1
    assert resumed.epoch == 0 and resumed.batch_in_epoch == 2
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    assert resumed.best_metrics == trainer.best_metrics
    assert resumed._wandb_run_id == "resume-run-id"
    assert (
        list(resumed.train_sampler)
        == list(trainer_module.EpochShuffleSampler(len(resumed.train_loader.dataset), resumed.config.seed))[4:]
    )  # epoch 0 remainder
    trainer.dataset.close()
    resumed.dataset.close()


def test_version_three_resume_requires_split_provenance(tmp_path, monkeypatch):
    from dataclasses import replace

    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "missing_provenance.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(path, tmp_path, batch_size=2)
    trainer = trainer_module.Trainer(config)
    trainer.build_dataloaders()
    checkpoint = trainer._save_resume_checkpoint()
    trainer.dataset.close()
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, weights_only=False)
    state.pop("split_provenance")
    torch.save(state, state_path)

    with pytest.raises(ValueError, match="missing split_provenance"):
        trainer_module.Trainer(replace(config, resume_from=str(checkpoint)))


def test_epoch_training_writes_resume_and_local_bests(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "train.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(
        path,
        tmp_path,
        batch_size=3,
        max_steps=3,
        eval_seeds=(0,),
        log_every=1,
        save_every=0,
        sample_every_epochs=0,
        artifact_every_epochs=0,
    )
    trainer = trainer_module.Trainer(config)
    trainer.train()
    output = tmp_path / "out"
    assert trainer.global_step == 3
    assert (output / "step_00000003" / "training_state.pt").exists()
    assert (output / "best_overall" / "model.safetensors").exists()
    assert (output / "best_force" / "model.safetensors").exists()
    checkpoint_config = json.loads((output / "step_00000003" / "config.json").read_text())
    split = checkpoint_config["_provenance"]["split"]
    assert split["total_episodes"] == 3 and split["val_episode_ids"] == [2]
    assert checkpoint_config["_provenance"]["global_step_completed_updates"] == 3
    manifest = json.loads((output / "best_manifest.json").read_text())
    assert manifest["selection"] == "deterministic_validation_objective"
    assert set(manifest["records"]) == {"overall", "force"}
    assert all(len(record["model_sha256"]) == 64 for record in manifest["records"].values())
    saved = torch.load(output / "step_00000003" / "training_state.pt", weights_only=False)
    assert saved["version"] == 3
    assert saved["best_metrics"] == trainer.best_metrics
    assert set(saved["best_records"]) == {"overall", "force"}
    trainer.dataset.close()

    from dataclasses import replace

    (output / "best_overall" / "config.json").unlink()
    with pytest.raises(ValueError, match="deploy checkpoint is incomplete"):
        trainer_module.Trainer(replace(config, resume_from=str(output / "step_00000003")))


def test_resume_restores_ema_shadow_and_live_weights(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "ema_resume.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(path, tmp_path, batch_size=2, ema=True)
    trainer = trainer_module.Trainer(config)
    trainer.build_dataloaders()
    trainer.model.weight.data.fill_(4.0)
    trainer.ema.step(trainer.model.parameters())
    shadow = [value.clone() for value in trainer.ema.shadow_params]
    checkpoint = trainer._save_resume_checkpoint()

    resumed = trainer_module.Trainer(_config(path, tmp_path, batch_size=2, ema=True, resume_from=str(checkpoint)))
    assert resumed.model.weight.item() == pytest.approx(4.0)
    for got, expected in zip(resumed.ema.shadow_params, shadow):
        assert torch.equal(got, expected)
    trainer.dataset.close()


def test_resume_rejects_changed_norm_stats_and_uses_bundled_copy(tmp_path, monkeypatch):
    from fuji.s1_policy.data.hl_dataset import HLDataset
    from fuji.s1_policy.data.norm_stats import NormStats, save_norm_stats
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "norm_resume.zarr.zip", n_frames=10, n_episodes=3)
    probe = HLDataset(path, drop_glitch_windows=False, obs_history_len=1)
    layout = probe.layout
    meta = probe.split_provenance(1)
    probe.close()
    stats_path = tmp_path / "norm_stats.json"
    save_norm_stats(
        {
            "actions": NormStats(np.zeros(layout.action_dim), np.ones(layout.action_dim)),
            "state": NormStats(np.zeros(layout.state_dim), np.ones(layout.state_dim)),
            "force_history": NormStats(np.zeros(6), np.ones(6)),
        },
        stats_path,
        meta=meta,
    )

    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(path, tmp_path, batch_size=2, norm_stats_path=str(stats_path))
    trainer = trainer_module.Trainer(config)
    trainer.build_dataloaders()
    frozen_payload = stats_path.read_text()
    payload = json.loads(stats_path.read_text())
    payload["actions"]["q01"][0] = 123.0
    stats_path.write_text(json.dumps(payload))
    checkpoint = trainer._save_resume_checkpoint()
    trainer.dataset.close()
    assert (checkpoint / "norm_stats.json").read_text() == frozen_payload

    with pytest.raises(ValueError, match="normalization statistics differ"):
        trainer_module.Trainer(
            _config(
                path,
                tmp_path,
                batch_size=2,
                norm_stats_path=str(stats_path),
                resume_from=str(checkpoint),
            )
        )

    stats_path.unlink()
    resumed = trainer_module.Trainer(
        _config(
            path,
            tmp_path,
            batch_size=2,
            norm_stats_path=str(stats_path),
            resume_from=str(checkpoint),
        )
    )
    assert Path(resumed._norm_stats_path) == checkpoint / "norm_stats.json"
    model_hash = json.loads((checkpoint / "config.json").read_text())["_provenance"]["model_sha256"]
    resumed._validate_deploy_checkpoint(checkpoint, model_hash)
    (checkpoint / "norm_stats.json").unlink()
    with pytest.raises(ValueError, match="missing its recorded normalization payload"):
        resumed._validate_deploy_checkpoint(checkpoint, model_hash)


def test_save_checkpoint_removes_stale_bundled_norm_stats(tmp_path):
    from fuji.s1_policy.training.config import TrainConfig
    from fuji.s1_policy.training.trainer import save_checkpoint

    model = _DummyModel()
    config = TrainConfig(ema=False)
    checkpoint = save_checkpoint(
        model,
        config,
        1,
        tmp_path,
        checkpoint_name="reused",
        norm_stats_text='{"frozen": true}',
    )
    assert (checkpoint / "norm_stats.json").exists()

    save_checkpoint(model, config, 2, tmp_path, checkpoint_name="reused")
    assert not (checkpoint / "norm_stats.json").exists()


def test_incomplete_norm_stats_are_rejected(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "incomplete_norm.zarr.zip", n_frames=10, n_episodes=3)
    stats_path = tmp_path / "incomplete.json"
    stats_path.write_text(
        json.dumps(
            {
                "actions": {"q01": [0.0] * 16, "q99": [1.0] * 16},
            }
        )
    )
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    with pytest.raises(ValueError, match="missing state"):
        trainer_module.Trainer(
            _config(
                path,
                tmp_path,
                batch_size=2,
                norm_stats_path=str(stats_path),
            )
        )


def test_terminal_resumed_partial_epoch_keeps_absolute_batch_index(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "partial_epoch.zarr.zip", n_frames=16, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(
        path,
        tmp_path,
        batch_size=2,
        max_steps=3,
        eval_seeds=(0,),
        save_every=0,
        log_every=1,
        sample_every_epochs=0,
        artifact_every_epochs=0,
    )
    trainer = trainer_module.Trainer(config)
    trainer.build_dataloaders()
    trainer.global_step = 2
    trainer.batch_in_epoch = 2
    checkpoint = trainer._save_resume_checkpoint()
    trainer.dataset.close()

    resumed = trainer_module.Trainer(
        _config(
            path,
            tmp_path,
            batch_size=2,
            max_steps=3,
            eval_seeds=(0,),
            save_every=0,
            log_every=1,
            sample_every_epochs=0,
            artifact_every_epochs=0,
            resume_from=str(checkpoint),
        )
    )
    resumed.train()
    assert resumed.epoch == 0
    assert resumed.batch_in_epoch == 3
    saved = torch.load(
        tmp_path / "out" / "step_00000003" / "training_state.pt",
        weights_only=False,
    )
    assert saved["epoch"] == 0
    assert saved["batch_in_epoch"] == 3


class _FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = dict(metadata or {})
        self.dirs: list[tuple[str, str | None]] = []
        self.files: list[tuple[str, str | None]] = []

    def add_dir(self, path, name=None):
        self.dirs.append((str(path), name))

    def add_file(self, path, name=None):
        self.files.append((str(path), name))


class _FakeWandbRun:
    def __init__(self, run_id="fakerun"):
        self.id = run_id
        self.logged: list[tuple[_FakeArtifact, list[str]]] = []

    def log(self, *_args, **_kwargs):
        pass

    def log_artifact(self, artifact, aliases=None):
        self.logged.append((artifact, list(aliases or [])))


def _attach_fake_wandb(trainer, run_id="fakerun"):
    """Install a minimal W&B double: only Artifact + log_artifact are exercised."""
    import types

    run = _FakeWandbRun(run_id)
    trainer._wandb = types.SimpleNamespace(Artifact=_FakeArtifact, finish=lambda: None)
    trainer._wandb_run = run
    return run


def test_epoch_artifact_uploads_on_cadence_with_addressable_name(tmp_path, monkeypatch):
    """Every Nth epoch must publish its own named artifact, not just a rolling version."""
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "epoch_artifact.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    # 6 train anchors / batch 3 => TWO optimizer steps per epoch, so the epoch number and
    # the step number differ. That is what makes the artifact NAME assertion meaningful:
    # naming it after global_step would produce a different string.
    config = _config(
        path,
        tmp_path,
        batch_size=3,
        max_steps=4,
        eval_seeds=(0,),
        log_every=1,
        save_every=0,
        sample_every_epochs=0,
        artifact_every_epochs=0,
        epoch_artifact_every_epochs=2,
    )
    trainer = trainer_module.Trainer(config)
    run = _attach_fake_wandb(trainer)
    trainer.train()

    epoch_uploads = [(a, al) for a, al in run.logged if a.type == "s1-epoch"]
    assert [a.metadata["epoch"] for a, _ in epoch_uploads] == [2]
    first, aliases = epoch_uploads[0]
    # Epoch 2 is reached after 4 optimizer steps: the name must track the EPOCH.
    assert first.metadata["global_step_completed_updates"] == 4
    assert first.name == "fakerun-epoch_002"
    assert aliases == ["epoch-2", "latest-epoch"]
    # Deploy payload only, flat: another machine needs the weights + the config that
    # names the layout/mode. training_state.pt is deliberately excluded (see
    # _EPOCH_ARTIFACT_FILES) -- it is ~80% of the bytes and only a resume wants it.
    checkpoint_dir = tmp_path / "out" / "step_00000004"
    assert first.dirs == []
    assert first.files == [
        (str(checkpoint_dir / "model.safetensors"), "model.safetensors"),
        (str(checkpoint_dir / "config.json"), "config.json"),
    ]
    assert first.metadata["contents"] == "deploy_only"
    assert first.metadata["algo"] == "flow"
    assert (checkpoint_dir / "training_state.pt").is_file(), "local resume state must still exist"
    trainer.dataset.close()


def test_coinciding_cadences_save_the_checkpoint_once(tmp_path, monkeypatch):
    """A shared save: writing the same multi-GB dir twice per epoch is pure waste."""
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "both_cadences.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(
        path,
        tmp_path,
        batch_size=6,
        max_steps=2,
        eval_seeds=(0,),
        log_every=1,
        save_every=0,
        sample_every_epochs=0,
        artifact_every_epochs=2,
        epoch_artifact_every_epochs=2,
    )
    trainer = trainer_module.Trainer(config)
    run = _attach_fake_wandb(trainer)
    saves = []
    original = trainer._save_resume_checkpoint

    def counting_save():
        result = original()
        saves.append(result)
        return result

    monkeypatch.setattr(trainer, "_save_resume_checkpoint", counting_save)
    trainer.train()

    assert len(saves) == 1, f"expected one shared save at the coinciding epoch, got {saves}"
    types_logged = [a.type for a, _ in run.logged]
    assert types_logged.count("s1-resume") == 1
    assert types_logged.count("s1-epoch") == 1
    # Both artifacts must read from the SAME directory that was written once: the
    # resume one takes the whole dir, the epoch one only its deploy files.
    resume_dir = next(a.dirs[0][0] for a, _ in run.logged if a.type == "s1-resume")
    epoch_files = next(a.files for a, _ in run.logged if a.type == "s1-epoch")
    assert resume_dir == str(saves[0])
    assert {Path(p).parent for p, _ in epoch_files} == {saves[0]}
    trainer.dataset.close()


def test_epoch_artifact_failure_does_not_stop_training(tmp_path, monkeypatch):
    """W&B is best-effort: a broken upload must never lose the local run."""
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "artifact_failure.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(
        path,
        tmp_path,
        batch_size=6,
        max_steps=2,
        eval_seeds=(0,),
        log_every=1,
        save_every=0,
        sample_every_epochs=0,
        artifact_every_epochs=0,
        epoch_artifact_every_epochs=2,
    )
    trainer = trainer_module.Trainer(config)
    run = _attach_fake_wandb(trainer)

    def exploding_log_artifact(_artifact, aliases=None):
        raise RuntimeError("network down")

    run.log_artifact = exploding_log_artifact
    trainer.train()

    assert trainer.global_step == 2
    assert (tmp_path / "out" / "step_00000002" / "training_state.pt").is_file()
    trainer.dataset.close()


def test_epoch_artifact_cadence_writes_nothing_extra_without_wandb(tmp_path, monkeypatch):
    """With W&B off the cadence has no consumer, so it must not multiply local saves."""
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "no_wandb.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    # 4 steps at a 2-epoch cadence means epoch 2 is a MID-run cadence hit, whose save
    # would land on its own step_00000002 dir -- distinguishable from the terminal save.
    config = _config(
        path,
        tmp_path,
        batch_size=6,
        max_steps=4,
        eval_seeds=(0,),
        log_every=1,
        save_every=0,
        sample_every_epochs=0,
        artifact_every_epochs=0,
        epoch_artifact_every_epochs=2,
    )
    trainer = trainer_module.Trainer(config)
    assert trainer._wandb_run is None
    saves = []
    original = trainer._save_resume_checkpoint

    def counting_save():
        result = original()
        saves.append(result.name)
        return result

    monkeypatch.setattr(trainer, "_save_resume_checkpoint", counting_save)
    trainer.train()

    # Exactly one save: the terminal one. The epoch-2 cadence must not have fired.
    assert saves == ["step_00000004"], saves
    steps = sorted(p.name for p in (tmp_path / "out").glob("step_*"))
    assert steps == ["step_00000004"]
    trainer.dataset.close()


def test_flow_sampler_generator_is_reproducible():
    from tests.test_model_forward import _batch, _tiny_model

    model = _tiny_model()
    model.eval()
    batch = _batch()
    first = model.sample_actions(batch, num_steps=3, generator=torch.Generator().manual_seed(42))
    second = model.sample_actions(batch, num_steps=3, generator=torch.Generator().manual_seed(42))
    assert torch.allclose(first, second)


# --------------------------------------------------------------------------------------------
# FINETUNE (--init-from): another checkpoint's WEIGHTS, and deliberately nothing else.
# --------------------------------------------------------------------------------------------


def _deploy_checkpoint(tmp_path, monkeypatch, weight: float, **cfg_kwargs):
    """Train-and-save one step so a real deploy dir (weights + config + _provenance) exists."""
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "base.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    config = _config(path, tmp_path, batch_size=2, **cfg_kwargs)
    trainer = trainer_module.Trainer(config)
    trainer.build_dataloaders()
    trainer.model.weight.data.fill_(weight)
    trainer.global_step = 7
    checkpoint = trainer._save_resume_checkpoint()
    trainer.dataset.close()
    return path, checkpoint


def test_init_from_loads_weights_only_and_restarts_the_run(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    _, checkpoint = _deploy_checkpoint(tmp_path, monkeypatch, weight=3.5)
    # A DIFFERENT dataset: the entire point of a finetune, and what --resume-from refuses.
    other = _write_hl_zarr(tmp_path / "supplement.zarr.zip", n_frames=12, n_episodes=4)
    finetune = trainer_module.Trainer(_config(other, tmp_path, batch_size=2, init_from=str(checkpoint)))
    finetune.build_dataloaders()

    assert finetune.model.weight.item() == pytest.approx(3.5)   # the weights came across
    assert finetune.global_step == 0                            # ...and nothing else did
    assert finetune.epoch == 0 and finetune.batch_in_epoch == 0
    assert finetune._resume_split_provenance is None
    # The saved provenance names where the weights came from -- the only record, since
    # `init_from` is elided from every config comparison.
    assert finetune._checkpoint_provenance()["init_from"] == Path(checkpoint).as_posix()
    finetune.dataset.close()


def test_init_from_seeds_the_ema_shadow_from_the_loaded_weights(tmp_path, monkeypatch):
    """EMAModel CLONES at construction, so a load placed after it would shadow random init."""
    from fuji.s1_policy.training import trainer as trainer_module

    # The BASE runs with EMA off deliberately: save_checkpoint writes model.safetensors from the
    # EMA whenever it is enabled, so an EMA base that never stepped would ship its random-init
    # shadow and the assertion below could not tell a seeded shadow from an unseeded one.
    _, checkpoint = _deploy_checkpoint(tmp_path, monkeypatch, weight=2.25, ema=False)
    other = _write_hl_zarr(tmp_path / "supplement.zarr.zip", n_frames=12, n_episodes=4)
    finetune = trainer_module.Trainer(
        _config(other, tmp_path, batch_size=2, ema=True, init_from=str(checkpoint)))
    assert finetune.ema is not None
    assert finetune.model.weight.item() == pytest.approx(2.25)
    for shadow in finetune.ema.shadow_params:
        assert shadow.item() == pytest.approx(2.25)


def test_init_from_and_resume_from_together_are_refused():
    from fuji.s1_policy.training.config import TrainConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(resume_from="a", init_from="b")


def test_init_from_refuses_a_checkpoint_that_is_not_a_deploy_dir(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    path = _write_hl_zarr(tmp_path / "x.zarr.zip", n_frames=10, n_episodes=3)
    monkeypatch.setattr(trainer_module, "DPFlowModel", _DummyModel)
    with pytest.raises(FileNotFoundError, match="not a deploy checkpoint dir"):
        trainer_module.Trainer(_config(path, tmp_path, init_from=str(tmp_path / "nope")))


def test_init_from_refuses_a_different_layout_mode(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    _, checkpoint = _deploy_checkpoint(tmp_path, monkeypatch, weight=1.0)
    # Rewrite the SAVED config's layout: same widths on this dummy model, different meaning.
    config_path = Path(checkpoint) / "config.json"
    raw = json.loads(config_path.read_text())
    raw["hl_data"]["mode"] = "both_hands"
    config_path.write_text(json.dumps(raw))
    other = _write_hl_zarr(tmp_path / "supplement.zarr.zip", n_frames=12, n_episodes=4)
    with pytest.raises(ValueError, match="hl_data.mode"):
        trainer_module.Trainer(_config(other, tmp_path, init_from=str(checkpoint)))


def test_init_from_refuses_a_different_active_algo_block(tmp_path, monkeypatch):
    from fuji.s1_policy.training import trainer as trainer_module

    _, checkpoint = _deploy_checkpoint(tmp_path, monkeypatch, weight=1.0)
    config_path = Path(checkpoint) / "config.json"
    raw = json.loads(config_path.read_text())
    raw["model"]["action_horizon"] = int(raw["model"]["action_horizon"]) + 5
    config_path.write_text(json.dumps(raw))
    other = _write_hl_zarr(tmp_path / "supplement.zarr.zip", n_frames=12, n_episodes=4)
    with pytest.raises(ValueError, match="action_horizon"):
        trainer_module.Trainer(_config(other, tmp_path, init_from=str(checkpoint)))


def test_validate_init_norm_stats_ignores_the_split_but_not_the_transform():
    from fuji.s1_policy.training.trainer import validate_init_norm_stats

    base = {
        "total_episodes": 232,
        "episode_ends_fingerprint": "aaa",
        "dataset_knobs": {"zarr_paths": ["base.zarr.zip"], "mode": "fivepoint_geom_xy3_rpy",
                          "wb_cond_mode": "home", "action_horizon": 100},
    }
    # A different episode set with the same transform: accepted, because that IS the finetune.
    supplement = {
        "total_episodes": 44,
        "episode_ends_fingerprint": "bbb",
        "dataset_knobs": {"zarr_paths": ["sup.zarr.zip"], "mode": "fivepoint_geom_xy3_rpy",
                          "wb_cond_mode": "home", "action_horizon": 100},
    }
    validate_init_norm_stats(base, supplement)

    # The export PLACEMENT decides what an absolute column means: refused by name.
    crossed = json.loads(json.dumps(supplement))
    crossed["dataset_knobs"]["wb_cond_mode"] = "absolute"
    with pytest.raises(ValueError, match="wb_cond_mode"):
        validate_init_norm_stats(base, crossed)

    # So does the horizon, whose quantiles differ at an identical width.
    crossed = json.loads(json.dumps(supplement))
    crossed["dataset_knobs"]["action_horizon"] = 50
    with pytest.raises(ValueError, match="action_horizon"):
        validate_init_norm_stats(base, crossed)

    with pytest.raises(ValueError, match="records no dataset_knobs"):
        validate_init_norm_stats({}, supplement)


def test_norm_stats_shift_reports_the_relative_range_change():
    from fuji.s1_policy.training.trainer import norm_stats_shift

    def stats(q99):
        return json.dumps({
            "actions": {"q01": [0.0, 0.0], "q99": q99, "mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "state": {"q01": [0.0], "q99": [1.0], "mean": [0.0], "std": [1.0]},
        })

    shift = norm_stats_shift(stats([1.0, 2.0]), stats([1.1, 2.0]))
    assert shift["actions"] == pytest.approx(0.10)
    assert shift["state"] == pytest.approx(0.0)
    # A zero-width (padded / unsupervised) column is skipped rather than dividing by zero.
    assert norm_stats_shift(stats([0.0, 2.0]), stats([0.0, 3.0]))["actions"] == pytest.approx(0.5)
