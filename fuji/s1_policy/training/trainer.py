"""Single-GPU S1 training with episode validation and exact resume."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import random
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from fuji.s1_policy.data.norm_stats import (
    normalize_quantile,
    parse_norm_stats,
    parse_norm_stats_meta,
    unnormalize_quantile,
)
from fuji.s1_policy.data.umi_dataset import UMIDataset
from fuji.s1_policy.model.dp_act import DPACTModel
from fuji.s1_policy.model.dp_diffusion import DPDiffusionModel
from fuji.s1_policy.model.dp_flow import DPFlowModel
from fuji.s1_policy.training.config import TrainConfig
from fuji.s1_policy.training.distributed import (
    all_reduce_sum,
    any_rank_flagged,
    barrier,
    broadcast_flag,
    destroy_process_group,
    get_dist_info,
    init_process_group,
    wrap_model,
)
from fuji.s1_policy.training.metrics import physical_trajectory_sums

logger = logging.getLogger(__name__)

S1Model = DPFlowModel | DPDiffusionModel | DPACTModel

#: ``algo`` -> (model class NAME, ``TrainConfig`` config attribute, W&B loss-key namespace).
#: ONE table instead of an if/elif chain whose ``else`` was a live trap: it built a
#: ``DPFlowModel`` from ``config.model`` for ANY unrecognized algo and logged it as
#: ``flow_loss``, so a new algorithm silently trained the wrong architecture on
#: never-configured defaults. A missing key now raises.
#:
#: The class is held by NAME and resolved from this module's globals at construction time,
#: NOT as a captured class object. Two reasons, and the first is load-bearing: tests
#: substitute a tiny stand-in with ``monkeypatch.setattr(trainer_module, "DPFlowModel",
#: ...)``, which a table of captured references would silently bypass — the suite would then
#: build real ViT/ResNet models (downloading weights) while appearing to test a stub.
_ALGO_MODELS: dict[str, tuple[str, str, str]] = {
    "flow": ("DPFlowModel", "model", "flow_loss"),
    "diffusion": ("DPDiffusionModel", "diffusion", "diffusion_loss"),
    "act": ("DPACTModel", "act", "act_loss"),
}

#: Algorithms whose DDP path is implemented AND covered by a cross-process gradient-parity
#: test. Membership is a claim about EVIDENCE, not about whether the code happens to run:
#: ``tests/test_ddp_training.py`` (act) and ``tests/test_ddp_diffusion.py`` (diffusion) each
#: prove a 2-rank step reproduces the single-process gradient, that every parameter receives
#: one (so ``find_unused_parameters=False`` is sound), and that no buffer drifts between
#: ranks.
#:
#: ``flow`` is deliberately ABSENT even though it would probably run. Its training forward
#: draws a per-sample RTC delay (``apply_rtc_time_masking``) that EXCLUDES a variable-length
#: prefix from the loss, so the per-rank valid-element count varies by construction rather
#: than incidentally — and DDP averages per-rank MEANS (see ``train_step``). Nothing has
#: measured how far that pushes the reduced gradient from the global masked mean, and a
#: plausible-looking multi-GPU run of an unverified preset is worse than a refusal.
_DDP_ALGOS = frozenset({"act", "diffusion"})


def collate_umi_batch(samples: list[dict]) -> dict:
    """Collate legacy UMI samples into the model batch contract."""
    img0 = samples[0]["image"]
    batch_size = len(samples)
    if img0.dim() == 4:
        images = [torch.stack([s["image"][t] for s in samples]) for t in range(img0.shape[0])]
    else:
        images = [torch.stack([s["image"] for s in samples])]
    return {
        "images": images,
        "img_masks": [torch.ones(batch_size, dtype=torch.bool) for _ in images],
        "state": torch.stack([s["state"] for s in samples]),
        "actions": torch.stack([s["action_chunk"] for s in samples]),
        "action_is_pad": torch.stack([s["action_is_pad"] for s in samples]),
        "action_dim_mask": torch.stack([s["action_dim_mask"] for s in samples]),
    }


def collate_hl_batch(samples: list[dict]) -> dict:
    """Collate HL-v2 samples, preserving the per-step/per-dimension action mask."""
    batch_size = len(samples)
    img0 = samples[0]["image"]
    if img0.dim() == 4:
        images = [torch.stack([s["image"][t] for s in samples]) for t in range(img0.shape[0])]
    else:
        images = [torch.stack([s["image"] for s in samples])]
    return {
        "images": images,
        "img_masks": [torch.ones(batch_size, dtype=torch.bool) for _ in images],
        "state": torch.stack([s["state"] for s in samples]),
        "state_valid": torch.stack([s["state_valid"] for s in samples]),
        "force_history": torch.stack([s["force_history"] for s in samples]),
        "force_history_valid": torch.stack([s["force_history_valid"] for s in samples]),
        "actions": torch.stack([s["action_chunk"] for s in samples]),
        "action_mask": torch.stack([s["action_mask"] for s in samples]),
    }


def cosine_warmup_lr(step: int, warmup: int, max_steps: int, min_ratio: float) -> float:
    """Multiplicative LR factor: linear warmup then cosine decay to ``min_ratio``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


class EpochShuffleSampler(Sampler[int]):
    """Deterministic epoch permutation with a sample offset for mid-epoch resume.

    Under DDP (``world_size > 1``) every rank builds the SAME seeded global permutation and
    then takes a strided slice of it, so the ranks partition each epoch without
    communicating. Two properties are load-bearing:

    * **Equal shard lengths.** The permutation is truncated to a multiple of
      ``world_size`` before striding, so every rank yields exactly ``size // world_size``
      indices and therefore runs an identical number of batches. Unequal counts are the
      classic silent DDP deadlock: the rank with fewer batches leaves the loop and never
      reaches the last gradient all-reduce, so the others block until the NCCL timeout.
      Truncation drops at most ``world_size - 1`` samples, and a different tail each epoch
      (the permutation is reseeded per epoch), so nothing is systematically excluded.
    * **``offset`` counts SHARD-LOCAL samples.** It is derived from ``batch_in_epoch``,
      which counts optimizer updates — a global quantity identical on every rank.
    """

    def __init__(self, size: int, seed: int, rank: int = 0, world_size: int = 1) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside [0, {world_size})")
        self.size = int(size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        #: Samples this rank sees per epoch, after the equalizing truncation.
        self.shard_size = self.size // self.world_size
        self.epoch = 0
        self.offset = 0

    def set_epoch(self, epoch: int, offset: int = 0) -> None:
        self.epoch = int(epoch)
        self.offset = int(offset)
        # Bounded by the SHARD, not the dataset: an offset past the shard end would
        # silently yield an empty rank while its peers still had batches to run.
        if not 0 <= self.offset <= self.shard_size:
            raise ValueError(f"sampler offset {self.offset} is outside [0, {self.shard_size}]")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        if self.world_size > 1:
            usable = self.shard_size * self.world_size
            order = order[:usable][self.rank :: self.world_size]
        return iter(order[self.offset :])

    def __len__(self) -> int:
        return max(0, self.shard_size - self.offset)

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "offset": self.offset,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["size"]) != self.size or int(state["seed"]) != self.seed:
            raise ValueError("resume sampler does not match the current train subset/seed")
        # The shard math depends on world_size, so resuming a 4-GPU run on 2 GPUs would
        # replay a DIFFERENT sample stream while claiming exact resume. Absent key = 1,
        # so every checkpoint written before DDP existed still resumes.
        saved_world = int(state.get("world_size", 1))
        if saved_world != self.world_size:
            raise ValueError(
                f"resume checkpoint was written with world_size={saved_world} but this run has "
                f"{self.world_size}; the per-rank shard is derived from it, so the sample "
                f"stream would differ. Relaunch on {saved_world} process(es)."
            )
        self.set_epoch(int(state["epoch"]), int(state["offset"]))


def _align_inactive_algo_blocks(saved: dict, current: dict) -> tuple[dict, dict]:
    """Drop an INACTIVE algorithm's config block when only one side has it.

    Resume and deploy-checkpoint validation compare the whole serialized ``TrainConfig`` by
    exact dict equality — deliberately, so "the original command plus only ``--resume-from``"
    is enforced. But adding a new algorithm adds a KEY, and a checkpoint written before that
    algorithm existed simply has no such key. Without this, shipping a new preset silently
    breaks resume for every checkpoint already on disk (verified: adding ``act`` made a
    pre-existing FLOW resume fail with "resume config differs", the only difference being
    the added key).

    Only an INACTIVE block is elided, and only when one side lacks it entirely:

    * the ACTIVE algorithm's block is always compared in full — it is the one that
      determines the architecture, so a difference there must still refuse;
    * ``algo`` itself is compared first, so this can never bridge two runs of different
      algorithms;
    * a key present on BOTH sides is compared normally, so a real edit to an inactive block
      still refuses (it would change the checkpoint's serialized config).

    Returns the two dicts to compare, leaving the callers' originals untouched.
    """
    from fuji.s1_policy.training.config import ALGO_CONFIG_FIELD

    if saved.get("algo") != current.get("algo"):
        return saved, current  # different algorithms: let the full comparison refuse
    active = ALGO_CONFIG_FIELD.get(str(current.get("algo", "")))
    saved, current = dict(saved), dict(current)
    for algo, field_name in ALGO_CONFIG_FIELD.items():
        if field_name == active:
            continue  # never elide the block that defines the model being trained
        if (field_name in saved) != (field_name in current):
            saved.pop(field_name, None)
            current.pop(field_name, None)
            logger.info(
                "resume/deploy config comparison: ignoring the inactive %r block, absent from "
                "one side (a checkpoint predating that algorithm)", algo)
    return saved, current


#: Every ``TrainConfig`` field a FINETUNE (``--init-from``) must not change, because changing it
#: changes what the loaded weights mean. Not an allowlist of what may change — the data, the
#: schedule, the augmentation and the output paths are all expected to move, and enumerating
#: those would make every future field default to "must match" for no reason.
#:
#: ``load_state_dict(strict=True)`` already refuses any tensor-SHAPE difference, so this list
#: exists for the shape-COMPATIBLE ones, which are the dangerous half:
#:
#: * ``hl_data.mode`` — two layouts can share a width and mean different columns.
#: * ``hl_data.wb_joint_target`` — ``absolute`` vs ``delta`` is the same 43 columns offset by
#:   ``q[0]``; nothing about the tensors says which.
#: * ``hl_data.image_mean`` / ``image_std`` — the pixels the ViT was trained on.
#: * ``hl_data.obs_history_len`` / ``obs_history_dt_s`` — which frames the stack holds.
#:
#: ``algo`` plus the ACTIVE algorithm block cover the architecture and the horizon; the inactive
#: blocks are elided by :func:`_align_inactive_algo_blocks` exactly as they are for resume.
_INIT_ARCH_HL_KEYS = (
    "mode",
    "image_key",
    "image_keys",
    "image_mean",
    "image_std",
    "obs_history_len",
    "obs_history_dt_s",
    "wb_joint_target",
)

#: The ``dataset_knobs`` entry a finetune is EXPECTED to change. Everything else in that dict
#: describes the transform rather than the episodes, so it must still agree.
_INIT_KNOB_EXEMPT = ("zarr_paths",)


def validate_init_config(saved: dict, current: dict) -> None:
    """Refuse ``--init-from`` weights whose architecture or input meaning differs.

    ``saved`` is the init checkpoint's serialized ``TrainConfig`` (its ``config.json`` minus
    ``_provenance``), ``current`` the run about to start. Only :data:`_INIT_ARCH_HL_KEYS`,
    ``algo``, ``pipeline`` and the active algorithm block are compared — see the module note
    above for why this is a denylist of meanings rather than an allowlist of knobs.
    """
    from fuji.s1_policy.training.config import ALGO_CONFIG_FIELD

    for key in ("algo", "pipeline"):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"--init-from checkpoint has {key}={saved.get(key)!r} but this run is "
                f"{current.get(key)!r}: the weights are a different model, not a starting point."
            )
    active = ALGO_CONFIG_FIELD[str(current["algo"])]
    saved_block = saved.get(active) or {}
    current_block = current.get(active) or {}
    if saved_block != current_block:
        differing = sorted(
            k for k in set(saved_block) | set(current_block)
            if saved_block.get(k) != current_block.get(k)
        )
        detail = ", ".join(f"{k}: {saved_block.get(k)!r} -> {current_block.get(k)!r}" for k in differing)
        raise ValueError(
            f"--init-from checkpoint's {active!r} block differs from this run's ({detail}). That "
            f"block defines the architecture and the action horizon, so the weights would either "
            f"fail to load or mean something else. Match it, or train from scratch."
        )
    saved_hl = saved.get("hl_data") or {}
    current_hl = current.get("hl_data") or {}
    for key in _INIT_ARCH_HL_KEYS:
        if saved_hl.get(key) != current_hl.get(key):
            raise ValueError(
                f"--init-from checkpoint has hl_data.{key}={saved_hl.get(key)!r} but this run is "
                f"{current_hl.get(key)!r}. That changes what the loaded weights' inputs or output "
                f"columns MEAN at an unchanged width, which nothing downstream would notice."
            )


def validate_init_norm_stats(saved_meta: dict, actual: dict) -> None:
    """Refuse a finetune whose data goes through a DIFFERENT transform than the weights saw.

    The finetune fits its own statistics over its own episodes, so the split fingerprints,
    counts and ids in ``actual`` are expected to differ from the init checkpoint's — that is the
    whole operation. What may NOT differ is the ``dataset_knobs`` that decide what a column is:
    ``mode``, ``wb_cond_mode`` (the export placement), ``action_horizon``, the camera list, the
    image normalization and the glitch/force knobs that shape which anchors exist.

    ``zarr_paths`` is exempt for the same reason ``fuji.s1_policy.inference.common`` exempts it:
    a path is not a property of the data.
    """
    saved_knobs = dict((saved_meta or {}).get("dataset_knobs") or {})
    actual_knobs = dict((actual or {}).get("dataset_knobs") or {})
    if not saved_knobs:
        raise ValueError(
            "--init-from checkpoint's norm_stats.json records no dataset_knobs; it predates the "
            "provenance block, so nothing can confirm the finetune data goes through the same "
            "transform. Re-fit that checkpoint's stats, or train from scratch."
        )
    for key in _INIT_KNOB_EXEMPT:
        saved_knobs.pop(key, None)
        actual_knobs.pop(key, None)
    if saved_knobs != actual_knobs:
        differing = sorted(
            k for k in set(saved_knobs) | set(actual_knobs) if saved_knobs.get(k) != actual_knobs.get(k)
        )
        detail = ", ".join(f"{k}: {saved_knobs.get(k)!r} -> {actual_knobs.get(k)!r}" for k in differing)
        raise ValueError(
            f"--init-from data goes through a different transform than the weights were trained "
            f"on ({detail}). These knobs decide what an action or state COLUMN is, so a finetune "
            f"across them is not a finetune."
        )


def norm_stats_shift(saved_text: str, current_text: str) -> dict[str, float]:
    """Per-key max relative change in the normalization RANGE, ``saved`` -> ``current``.

    A finetune bundles its own statistics, so every normalized input and target shifts a little
    against what the weights learned. This quantifies it so the shift is on the log rather than
    implicit: ``(q99 - q01)`` per column, compared elementwise, largest relative change per key.
    Columns whose saved range is zero (a padded or unsupervised slot) are skipped — they carry no
    signal and would otherwise divide by zero.
    """
    saved = parse_norm_stats(saved_text)
    current = parse_norm_stats(current_text)
    out: dict[str, float] = {}
    for key in sorted(saved.keys() & current.keys()):
        a, b = saved[key], current[key]
        span_a = np.asarray(a.q99, dtype=np.float64) - np.asarray(a.q01, dtype=np.float64)
        span_b = np.asarray(b.q99, dtype=np.float64) - np.asarray(b.q01, dtype=np.float64)
        if span_a.shape != span_b.shape:
            out[key] = float("inf")
            continue
        live = np.abs(span_a) > 0.0
        if not live.any():
            continue
        out[key] = float(np.max(np.abs(span_b[live] - span_a[live]) / np.abs(span_a[live])))
    return out


def _move_batch(batch: dict, device: torch.device, non_blocking: bool = False) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, list):
            out[key] = [tensor.to(device, non_blocking=non_blocking) for tensor in value]
        elif torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=non_blocking)
        else:
            out[key] = value
    return out


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_safetensors(state: dict[str, torch.Tensor], path: Path) -> None:
    import safetensors.torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        safetensors.torch.save_file(state, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def ema_weights(model: torch.nn.Module, ema):
    """Temporarily install EMA parameters and always restore the live parameters."""
    if ema is None:
        yield
        return
    ema.store(model.parameters())
    try:
        ema.copy_to(model.parameters())
        yield
    finally:
        ema.restore(model.parameters())


@contextmanager
def isolated_rng(seed: int):
    """Seed Python/NumPy/CPU/CUDA RNGs and restore all of them on exit."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def validate_split_provenance(meta: dict, actual: dict) -> None:
    """Refuse norm stats/checkpoints fitted against a different episode split."""
    required = (
        "transform_contract_version",
        "strategy",
        "val_last_episodes",
        "total_episodes",
        "train_episode_ids",
        "val_episode_ids",
        "episode_identities",
        "episode_ends_fingerprint",
        "anchor_indices_fingerprint",
        "source_content_fingerprint",
        "train_anchor_count",
        "val_anchor_count",
        "dataset_knobs",
    )
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(
            "split provenance is incomplete (missing "
            + ", ".join(missing)
            + "); re-fit norm stats with the current fit_hl_norm_stats.py"
        )
    for key in required:
        if meta[key] != actual[key]:
            raise ValueError(
                f"split provenance mismatch for {key}: stats/checkpoint has {meta[key]!r}, "
                f"current dataset has {actual[key]!r}. Refuse to risk validation leakage."
            )


def save_checkpoint(
    model: S1Model,
    config: TrainConfig,
    step: int,
    output_dir: str | Path,
    norm_stats_path: str = "",
    ema=None,
    *,
    training_state: dict | None = None,
    provenance: dict | None = None,
    checkpoint_name: str | None = None,
    norm_stats_text: str | None = None,
) -> Path:
    """Atomically save deploy weights and optional complete resume state.

    ``model.safetensors`` retains deploy semantics (EMA when enabled). The optional
    ``training_state.pt`` always carries live weights and all state needed to resume.
    ``step`` is the number of optimizer updates already completed.
    """
    ckpt_dir = Path(output_dir) / (checkpoint_name or f"step_{step:08d}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.ema) != (ema is not None):
        raise ValueError(
            f"config.ema={config.ema!r} disagrees with the checkpoint EMA object "
            f"({'present' if ema is not None else 'absent'})"
        )
    with ema_weights(model, ema):
        _atomic_safetensors(model.state_dict(), ckpt_dir / "model.safetensors")

    payload = asdict(config)
    payload["_provenance"] = dict(provenance or {})
    payload["_provenance"]["global_step_completed_updates"] = int(step)
    payload["_provenance"]["deploy_weights"] = "ema" if ema is not None else "live"
    payload["_provenance"]["model_sha256"] = _file_sha256(ckpt_dir / "model.safetensors")
    _atomic_text(ckpt_dir / "config.json", json.dumps(payload, indent=2, default=str))
    bundled_norm_path = ckpt_dir / "norm_stats.json"
    wrote_norm_stats = False
    if norm_stats_text is not None:
        _atomic_text(bundled_norm_path, norm_stats_text)
        wrote_norm_stats = True
    elif norm_stats_path:
        norm_path = Path(norm_stats_path)
        if norm_path.exists():
            _atomic_text(bundled_norm_path, norm_path.read_text())
            wrote_norm_stats = True
    if not wrote_norm_stats:
        bundled_norm_path.unlink(missing_ok=True)
    if training_state is not None:
        _atomic_torch_save(training_state, ckpt_dir / "training_state.pt")
    logger.info("Saved %s checkpoint to %s", "resume+deploy" if training_state else "deploy", ckpt_dir)
    return ckpt_dir


def _accumulate_tensor_sums(storage: dict, values: dict) -> None:
    for name, (num, den) in values.items():
        if name not in storage:
            storage[name] = [num.detach().to(torch.float64), den.detach().to(torch.float64)]
        else:
            storage[name][0] += num.detach().to(torch.float64)
            storage[name][1] += den.detach().to(torch.float64)


def _tensor_sum_means(storage: dict) -> dict[str, float]:
    out = {}
    for name, (num, den) in storage.items():
        n, d = float(num.cpu()), float(den.cpu())
        out[name] = n / d if d > 0 else math.nan
    return out


def metric_log(prefix: str, means: dict[str, float], loss_key: str) -> dict[str, float]:
    """Unified W&B names plus namespaced compatibility with existing loss keys."""
    out = {f"{prefix}/{name}": value for name, value in means.items()}
    if "overall" in means:
        out[f"{prefix}/objective"] = means["overall"]
        out[f"{prefix}/{loss_key}"] = means["overall"]
    for name, value in means.items():
        if name != "overall":
            out[f"{prefix}/{loss_key}/{name}"] = value
    return out


def is_improvement(value: float, best: float) -> bool:
    return math.isfinite(value) and value < best


class Trainer:
    """Own the model, episode loaders, validation, and complete resume state.

    Single-process by default. Launched under ``torchrun`` with ``WORLD_SIZE > 1`` it runs
    DDP data parallelism for the algorithms in :data:`_DDP_ALGOS` (``act``, ``diffusion``) —
    see :mod:`fuji.s1_policy.training.distributed` and
    ``docs/superpowers/specs/2026-08-10-act-multi-gpu-ddp-design.md``. Everything the
    single-GPU path does is unchanged when ``world_size == 1``.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # ---- distributed topology (no-op without torchrun) ----
        self.dist_info = get_dist_info()
        if self.dist_info.enabled:
            # Refuse rather than run unverified -- see _DDP_ALGOS for what "verified" means
            # here and why `flow` is not in it.
            if config.algo not in _DDP_ALGOS:
                raise ValueError(
                    f"multi-GPU (DDP) training is implemented and verified for algo in "
                    f"{sorted(_DDP_ALGOS)}, got algo={config.algo!r} under WORLD_SIZE="
                    f"{self.dist_info.world_size}. Run it without torchrun (single GPU), or "
                    f"use CUDA_VISIBLE_DEVICES to place several single-GPU runs side by side."
                )
            init_process_group(self.dist_info)

        if self.dist_info.enabled and torch.cuda.is_available() and config.device != "cpu":
            # Each rank owns exactly ONE card. Honouring a bare `--device cuda` here would
            # put every rank on cuda:0: N models on one card, and NCCL deadlocking against
            # itself. An EXPLICIT `--device cpu` is still honoured, though: it is the only way
            # to exercise the DDP path without a GPU, which is what the gloo-backed
            # Trainer-level tests in tests/test_ddp_diffusion.py do -- silently overriding it
            # made them allocate on cuda:0 and cuda:1 on any box that has cards.
            self.device = torch.device(f"cuda:{self.dist_info.local_rank}")
        else:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        try:
            model_name, config_field, self._loss_key = _ALGO_MODELS[config.algo]
        except KeyError:
            raise ValueError(
                f"no model registered for algo={config.algo!r}; expected one of "
                f"{sorted(_ALGO_MODELS)}") from None
        model_class = globals()[model_name]  # late-bound; see _ALGO_MODELS
        self.model = model_class(getattr(config, config_field)).to(self.device)
        # `self.model` STAYS UNWRAPPED for the whole run: state_dict(), EMA, gradient
        # clipping, saving and evaluation all go through it. `_fwd_model` is the DDP view
        # and is used ONLY for the training forward pass, which is what installs the
        # gradient-averaging hooks. Saving from the wrapper would prefix every key with
        # "module." -- and both scripts/eval_checkpoints_physical.py and
        # fuji.s1_policy.inference.common load with strict=True, so such a checkpoint is
        # written successfully and then refused by every reader.
        self._fwd_model = wrap_model(self.model, self.dist_info)

        # FINETUNE weights land HERE — before the EMA below, not after. `EMAModel` CLONES the
        # parameters it is handed, so an init-from load placed after it would leave the shadow
        # copy at random initialization; with decay 0.9999 a 2000-step finetune moves that
        # shadow ~18% of the way to the live weights, and `model.safetensors` is written FROM
        # the EMA whenever it is enabled. The run would train correctly and save noise.
        self._init_from: Path | None = None
        self._init_norm_stats_text: str | None = None
        if config.init_from:
            self._load_init_weights(config.init_from)

        # A model may prescribe its own parameter groups. ACT does: its ImageNet-pretrained
        # ResNet trunk trains at `act.lr_backbone` while the from-scratch transformer trains
        # at `--lr` (upstream detr/main.py:80-86). One shared LR either destroys the
        # pretrained features or starves the transformer. The flow/diffusion siblings define
        # no such hook, so their single-group AdamW is byte-identical to before.
        param_groups = self.model.param_groups(config.lr) if hasattr(self.model, "param_groups") \
            else self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay
        )
        # LambdaLR multiplies each group by the factor against THAT GROUP's own initial_lr
        # (verified: base_lrs == [1e-4, 1e-5] and a 0.5 factor yields [5e-5, 5e-6]), so the
        # warmup + cosine schedule applies to the trunk and the transformer alike and the
        # trunk/head LR RATIO is preserved for the whole run. A multi-group optimizer
        # therefore needs no scheduler change.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_warmup_lr(step, config.warmup_steps, config.max_steps, config.lr_min_ratio),
        )

        self.ema = None
        if config.ema:
            from diffusers.training_utils import EMAModel

            self.ema = EMAModel(
                self.model.parameters(), decay=config.ema_decay, use_ema_warmup=True, power=config.ema_warmup_power
            )
            self.ema.to(self.device)

        configured_norm_stats_path = (
            config.hl_data.norm_stats_path if config.pipeline == "hl" else config.data.norm_stats_path
        )
        self._norm_stats_path = configured_norm_stats_path
        if configured_norm_stats_path and not Path(configured_norm_stats_path).exists() and config.resume_from:
            resume_path = Path(config.resume_from)
            bundled = (resume_path if resume_path.is_dir() else resume_path.parent) / "norm_stats.json"
            if bundled.exists():
                self._norm_stats_path = str(bundled)
                logger.info("Using normalization statistics bundled with resume checkpoint: %s", bundled)
        norm_stats_bytes = Path(self._norm_stats_path).read_bytes() if self._norm_stats_path else None
        self._norm_stats_text = norm_stats_bytes.decode("utf-8") if norm_stats_bytes is not None else None
        self.norm_stats = parse_norm_stats(self._norm_stats_text) if self._norm_stats_text is not None else None
        if self.norm_stats is not None:
            missing = sorted({"actions", "state"} - self.norm_stats.keys())
            if missing:
                raise ValueError("normalization statistics are incomplete (missing " + ", ".join(missing) + ")")
        self._norm_stats_meta = (
            parse_norm_stats_meta(self._norm_stats_text) if self._norm_stats_text is not None else {}
        )
        self._norm_stats_sha256 = hashlib.sha256(norm_stats_bytes).hexdigest() if norm_stats_bytes is not None else None
        if self._init_from is not None:
            if self._init_norm_stats_text is not None and self._norm_stats_text is not None:
                validate_init_norm_stats(
                    parse_norm_stats_meta(self._init_norm_stats_text), self._norm_stats_meta)
            self._report_init_norm_stats_shift()

        self.use_bf16 = config.bf16 and self.device.type == "cuda"
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.best_metrics = {"overall": math.inf, "force": math.inf}
        self.best_records: dict[str, dict] = {}
        self._last_val_step = -1
        self._last_checkpoint_step = -1
        self._split_provenance: dict = {}
        self._resume_split_provenance: dict | None = None
        self._pending_rng_state: dict | None = None
        self._resume_sampler_state: dict | None = None
        self._train_generator = torch.Generator().manual_seed(config.seed)
        self._val_generator = torch.Generator().manual_seed(config.seed + 1)
        self.train_sampler: EpochShuffleSampler | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.dataset = None
        # Set by build_dataloaders. False here so an evaluate() before it (or a test driving
        # the loaders directly) never attempts a collective that would have no counterpart.
        self._val_is_sharded = False

        self._wandb = None
        self._wandb_run = None
        self._wandb_run_id = ""
        if config.resume_from:
            self._load_resume(config.resume_from)
        self._init_wandb()
        if self._pending_rng_state is not None:
            self._restore_rng_state(self._pending_rng_state)

    def _init_wandb(self) -> None:
        if not self.config.wandb:
            return
        # One run per job, not one per rank: N ranks calling wandb.init with the same id
        # produce interleaved duplicate histories. Non-main ranks keep _wandb_run None, so
        # every _wandb_log below is already a no-op for them.
        if not self.dist_info.is_main:
            return
        try:
            import wandb

            self._wandb = wandb
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or None,
                config=asdict(self.config),
                id=self._wandb_run_id or None,
                resume="allow" if self._wandb_run_id else None,
            )
            self._wandb_run_id = self._wandb_run.id
        except Exception as exc:  # W&B must never stop local training.
            logger.warning("W&B initialization failed; continuing locally: %s", exc)
            self._wandb = None
            self._wandb_run = None

    def _wandb_log(self, payload: dict, step: int | None = None) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(payload, step=self.global_step if step is None else step)
        except Exception as exc:
            logger.warning("W&B log failed; continuing locally: %s", exc)

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        cfg = self.config
        if cfg.pipeline == "hl":
            from fuji.s1_policy.data.hl_dataset import HLDataset

            hl = cfg.hl_data
            dataset = HLDataset(
                list(hl.zarr_paths),
                mode=hl.mode,
                image_key=hl.image_key,
                image_keys=hl.resolved_image_keys(),
                image_mean=hl.image_mean,
                image_std=hl.image_std,
                drop_invalid_force_history=hl.drop_invalid_force_history,
                drop_glitch_windows=hl.drop_glitch_windows,
                obs_history_len=hl.obs_history_len,
                obs_history_dt_s=hl.obs_history_dt_s,
                require_episode_ends=hl.val_last_episodes > 0,
                wb_joint_target=hl.wb_joint_target,
            )
            self.dataset = dataset
            train_indices, val_indices = dataset.split_indices(hl.val_last_episodes)
            self._split_provenance = dataset.split_provenance(hl.val_last_episodes)
            if self.norm_stats is not None:
                validate_split_provenance(self._norm_stats_meta, self._split_provenance)
            collate = collate_hl_batch
        else:
            from fuji.s1_policy.data.hl_dataset import zip_content_fingerprint

            dataset = UMIDataset(
                list(cfg.data.zarr_paths),
                action_horizon=cfg.data.action_horizon,
                image_key=cfg.data.image_key,
                image_mean=cfg.data.image_mean,
                image_std=cfg.data.image_std,
                ref_frame=cfg.data.ref_frame,
                obs_history_len=cfg.data.obs_history_len,
                obs_history_stride=cfg.data.obs_history_stride,
            )
            self.dataset = dataset
            train_indices, val_indices = list(range(len(dataset))), []
            source_fingerprints = [zip_content_fingerprint(path) for path in cfg.data.zarr_paths]
            source_payload = json.dumps(source_fingerprints, separators=(",", ":"), sort_keys=False).encode("utf-8")
            self._split_provenance = {
                "strategy": "legacy_all_anchors",
                "anchor_count": len(dataset),
                "source_content_fingerprint": hashlib.sha256(source_payload).hexdigest(),
                "dataset_knobs": {
                    "zarr_paths": [Path(path).as_posix() for path in cfg.data.zarr_paths],
                    "action_horizon": cfg.data.action_horizon,
                    "image_key": cfg.data.image_key,
                    "image_mean": cfg.data.image_mean,
                    "image_std": cfg.data.image_std,
                    "ref_frame": cfg.data.ref_frame,
                    "obs_history_len": cfg.data.obs_history_len,
                    "obs_history_stride": cfg.data.obs_history_stride,
                },
            }
            collate = collate_umi_batch

        if self._resume_split_provenance is not None:
            if cfg.pipeline == "hl":
                validate_split_provenance(self._resume_split_provenance, self._split_provenance)
            elif self._resume_split_provenance != self._split_provenance:
                raise ValueError(
                    "legacy dataset provenance differs from the resume checkpoint; refuse to continue on changed data"
                )

        # STATE JITTER, on the TRAIN subset only. Wrapping here rather than flagging the dataset
        # is what keeps the validation objective and the norm-stats fit clean by construction —
        # both go through the same HLDataset instance (see data.hl_dataset.StateNoise). It sits
        # OUTSIDE `Subset` so sample indices keep their meaning for `_split_provenance`.
        train_source: Dataset = dataset
        state_noise_sigma = cfg.hl_data.state_noise_sigma() if cfg.pipeline == "hl" else None
        if state_noise_sigma is not None:
            from fuji.s1_policy.data.hl_dataset import StateNoise

            train_source = StateNoise(dataset, state_noise_sigma)
            logger.info(
                "state jitter (TRAIN only): sigma per column %s (raw units, pre-normalization); "
                "validation and the norm-stats fit are unjittered",
                np.array2string(np.asarray(state_noise_sigma), precision=4, suppress_small=True))
        train_subset = Subset(train_source, train_indices)
        # Validation is sharded CONTIGUOUSLY across ranks and re-aggregated exactly (each
        # rank's (numerator, denominator) sums are all-reduced before the division in
        # `evaluate`), so the objective is identical to a single-process pass over the same
        # anchors. A strided shard would work equally well; contiguous keeps `val_indices`
        # order readable in a debugger. The remainder goes to the LAST ranks one each, so no
        # anchor is dropped -- unlike training, an uneven count is harmless here because
        # validation runs no collective per batch.
        # Sharding is skipped when it cannot give EVERY rank at least one anchor: a rank with
        # an empty shard would return early from `evaluate` and never reach the all-reduce,
        # hanging its peers. In that case all ranks evaluate the whole set (identical work,
        # identical answer) and `_val_is_sharded` stays False so no reduction is applied --
        # reducing duplicated sums would multiply the objective by world_size.
        self._val_is_sharded = False
        if val_indices and self.dist_info.enabled and len(val_indices) >= self.dist_info.world_size:
            world, rank = self.dist_info.world_size, self.dist_info.rank
            base, extra = divmod(len(val_indices), world)
            start = rank * base + min(rank, extra)
            stop = start + base + (1 if rank < extra else 0)
            val_indices = val_indices[start:stop]
            self._val_is_sharded = True
        val_subset = Subset(dataset, val_indices) if val_indices else None
        # Under DDP `batch_size` is PER RANK, so the guard must compare against this rank's
        # SHARD. Checking the whole subset would let a world size large enough to starve
        # each rank pass, and every rank would then silently run zero batches.
        world = self.dist_info.world_size
        shard_size = len(train_subset) // world
        if shard_size < cfg.batch_size:
            detail = (
                f" (per-rank shard of {len(train_subset)} across world_size={world})"
                if world > 1 else ""
            )
            raise ValueError(
                f"train subset has {shard_size} samples{detail}, smaller than "
                f"batch_size={cfg.batch_size} with drop_last=True"
            )
        self.train_sampler = EpochShuffleSampler(
            len(train_subset), cfg.seed, rank=self.dist_info.rank, world_size=world)
        offset = self.batch_in_epoch * cfg.batch_size
        self.train_sampler.set_epoch(self.epoch, offset)
        if self._resume_sampler_state is not None:
            self.train_sampler.load_state_dict(self._resume_sampler_state)
            expected = {"epoch": self.epoch, "offset": offset}
            if self.train_sampler.epoch != expected["epoch"] or self.train_sampler.offset != expected["offset"]:
                raise ValueError("resume sampler epoch/offset disagrees with epoch and batch_in_epoch")

        pin_memory = bool(cfg.pin_memory and self.device.type == "cuda")
        self.train_loader = DataLoader(
            train_subset,
            batch_size=cfg.batch_size,
            sampler=self.train_sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=pin_memory,
            generator=self._train_generator,
        )
        self.val_loader = None
        if val_subset is not None:
            self.val_loader = DataLoader(
                val_subset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=collate,
                drop_last=False,
                persistent_workers=cfg.num_workers > 0,
                pin_memory=pin_memory,
                generator=self._val_generator,
            )
        return self.train_loader, self.val_loader

    def build_dataloader(self) -> DataLoader:
        """Backward-compatible singular API returning the train loader."""
        return self.build_dataloaders()[0]

    def _check_norm_stats(self, batch: dict) -> None:
        for key in ("actions", "state", "force_history"):
            stats = self.norm_stats.get(key)
            if stats is None or key not in batch:
                continue
            expected_shape = (int(batch[key].shape[-1]),)
            if stats.q01.shape != expected_shape or stats.q99.shape != expected_shape:
                raise ValueError(
                    f"norm_stats['{key}'] q01/q99 shapes {stats.q01.shape}/{stats.q99.shape} "
                    f"do not match batch width {expected_shape}; "
                    "re-fit stats with the same layout and dataset knobs"
                )
            if not np.isfinite(stats.q01).all() or not np.isfinite(stats.q99).all():
                raise ValueError(f"norm_stats['{key}'] contains non-finite q01/q99 values")
            if np.any(stats.q99 < stats.q01):
                raise ValueError(f"norm_stats['{key}'] contains q99 values below q01")
        self._norm_checked = True

    def _apply_norm(self, batch: dict) -> dict:
        if self.norm_stats is None:
            return batch
        if not getattr(self, "_norm_checked", False):
            self._check_norm_stats(batch)
        for key in ("actions", "state", "force_history"):
            if key in self.norm_stats and key in batch:
                array = normalize_quantile(batch[key].numpy(), self.norm_stats[key])
                batch[key] = torch.from_numpy(array).to(batch[key].dtype)
        return batch

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16)

    def _abort_if_any_rank_nonfinite(self, local_bad: bool, what: str) -> None:
        """Raise on EVERY rank if ``local_bad`` holds on ANY rank.

        The message distinguishes "this rank saw it" from "a peer saw it", because under
        DDP the interesting question when the job dies is which rank's data was
        responsible — and the peers' own tensors are finite, so without this they would
        report a confusing clean value.
        """
        if not any_rank_flagged(local_bad):
            return
        where = "this rank" if local_bad else f"another rank (rank {self.dist_info.rank} is finite)"
        raise FloatingPointError(
            f"non-finite {what} at completed step {self.global_step}, detected on {where}"
        )

    def train_step(self, batch: dict) -> dict:
        """Complete one optimizer update; caller increments ``global_step`` afterward.

        Under DDP the reduced gradient is the MEAN OF THE PER-RANK LOSSES, and every S1 loss
        is a masked mean (``sum / valid_count``). So when the ranks hold different numbers of
        valid elements the reduction gives ``mean_r(S_r / C_r)`` where a single process over
        the concatenated batch would give ``(sum_r S_r) / (sum_r C_r)``. The two agree exactly
        when the counts agree, and diverge in proportion to the count IMBALANCE — on HL-v2
        that is a fraction of a percent (the counts differ only by temporal padding and force
        validity within a 32x50x16 mask), which is why this is documented rather than
        corrected with a per-step count all-reduce. ``tests/test_ddp_diffusion.py`` pins both
        halves: exact parity on equal counts, and the measured average-of-means behaviour on
        deliberately lopsided ones.
        """
        self.model.train()
        batch = self._apply_norm(batch)
        batch = _move_batch(batch, self.device, non_blocking=self.config.pin_memory and self.device.type == "cuda")
        batch["_skip_scalar_metrics"] = True
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            # The DDP view under torchrun, `self.model` itself otherwise. Forward through
            # the wrapper is what arms the gradient-averaging reducer for this backward.
            out = self._fwd_model(batch)
        loss = out["loss"]
        # A one-rank raise would leave its peers blocked at the NEXT collective until the
        # NCCL timeout, turning a legible crash into a hang that reads as slow training.
        # All-reducing the health flag makes every rank abort on the same step instead.
        self._abort_if_any_rank_nonfinite(
            not bool(torch.isfinite(loss.detach())), "training loss")
        loss.backward()
        # DDP averages gradients DURING backward, so clipping and the finite-check below
        # already act on the averaged gradient -- the single-GPU ordering stays correct.
        # Clipping reads `self.model.parameters()`, which are the same tensors DDP wraps.
        max_norm = self.config.grad_clip if self.config.grad_clip > 0 else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self._abort_if_any_rank_nonfinite(
            not bool(torch.isfinite(grad_norm)), "gradient norm")
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.step(self.model.parameters())
        return {
            "metric_sums": out["metric_sums"],
            "grad_norm": float(grad_norm.detach().cpu()),
            "batch_size": int(batch["actions"].shape[0]),
        }

    def evaluate(self) -> dict[str, float]:
        """Full deterministic validation objective, exactly aggregated over valid elements."""
        if self.val_loader is None:
            return {}
        was_training = self.model.training
        aggregate: dict = {}
        nonfinite_val = False
        try:
            with ema_weights(self.model, self.ema):
                self.model.eval()
                with torch.no_grad():
                    for seed in self.config.eval_seeds:
                        self._val_generator.manual_seed(int(seed))
                        with isolated_rng(int(seed)):
                            for batch in self.val_loader:
                                batch = self._apply_norm(batch)
                                batch = _move_batch(
                                    batch,
                                    self.device,
                                    non_blocking=self.config.pin_memory and self.device.type == "cuda",
                                )
                                batch["_skip_scalar_metrics"] = True
                                with self._autocast():
                                    out = self.model(batch)
                                # Recorded, NOT raised here: the ranks hold different numbers
                                # of validation batches, so an immediate raise on one rank is
                                # a collective its peers never make -- they would proceed to
                                # the all-reduce below and block until the NCCL timeout. The
                                # flag is reduced once, after the loop, where every rank
                                # participates exactly once.
                                if not bool(torch.isfinite(out["loss"].detach())):
                                    nonfinite_val = True
                                _accumulate_tensor_sums(aggregate, out["metric_sums"])
        finally:
            self.model.train(was_training)
        # Before the reduction below, and unconditionally on every rank, so a non-finite
        # validation loss anywhere fails the whole job instead of deadlocking it. Single
        # process: `any_rank_flagged` is the identity, so the raise is exactly as before.
        self._abort_if_any_rank_nonfinite(nonfinite_val, "validation loss")
        if self._val_is_sharded:
            # Reduce the (numerator, denominator) pairs BEFORE dividing, so the result is
            # the exact global ratio rather than a mean of per-rank means -- the same reason
            # `_accumulate_tensor_sums` keeps sums instead of batch means. Sorted keys keep
            # the collective order identical on every rank; iterating a dict built from a
            # ragged final batch could otherwise diverge and deadlock.
            #
            # This relies on the metric KEY SET being identical across ranks, which holds
            # because `masked_error_sums` derives its keys from `group_dims` (layout/config,
            # not data) plus the action width -- so every rank produces the same names even
            # from different samples. If a future metric key ever became sample-dependent
            # (e.g. a group emitted only when some condition occurs in the batch), sorting
            # would NOT save it: ranks would reduce different numbers of tensors and hang.
            # Such a change needs a key-set all-gather here first.
            for name in sorted(aggregate):
                num, den = aggregate[name]
                aggregate[name] = [all_reduce_sum(num), all_reduce_sum(den)]
        return _tensor_sum_means(aggregate)

    def _action_to_physical(self, actions: torch.Tensor) -> np.ndarray:
        array = actions.detach().float().cpu().numpy()
        if self.norm_stats is not None and "actions" in self.norm_stats:
            array = unnormalize_quantile(array, self.norm_stats["actions"])
        return array

    @staticmethod
    def _slice_batch(batch: dict, count: int) -> dict:
        out = {}
        for key, value in batch.items():
            if isinstance(value, list):
                out[key] = [tensor[:count] for tensor in value]
            elif torch.is_tensor(value):
                out[key] = value[:count]
            else:
                out[key] = value
        return out

    def run_sample_diagnostics(self) -> dict[str, float]:
        """Sample fixed validation anchors and report physical trajectory errors."""
        if self.val_loader is None or self.config.sample_diagnostic_batches <= 0:
            return {}
        from fuji.s1_policy.data.layout import get_layout

        layout = get_layout(self.config.hl_data.mode)
        # Keyed off `physical_trajectory_sums` itself rather than a literal list, because WHICH
        # metrics exist follows the layout: a whole-body run reports `joint_deg` and no
        # position/rotation at all, and a hardcoded set would KeyError on the new name (or,
        # worse, keep reporting a 0.00 cm position error for a mode with no position columns).
        totals: dict[str, list[float]] = {}
        rows = []
        sample_count = 0
        was_training = self.model.training
        cache = getattr(self.model, "_last_actions", None)
        cache = cache.detach().clone() if cache is not None else None
        cache_valid = getattr(self.model, "_last_actions_valid", False)
        try:
            with ema_weights(self.model, self.ema), isolated_rng(self.config.diagnostic_seed):
                self.model.eval()
                self._val_generator.manual_seed(self.config.diagnostic_seed)
                with torch.no_grad():
                    for batch_index, raw_batch in enumerate(self.val_loader):
                        if batch_index >= self.config.sample_diagnostic_batches:
                            break
                        remaining = self.config.sample_diagnostic_samples - sample_count
                        if remaining <= 0:
                            break
                        raw_batch = self._slice_batch(raw_batch, min(remaining, raw_batch["actions"].shape[0]))
                        target = raw_batch["actions"].clone()
                        mask = raw_batch["action_mask"].clone()
                        batch = self._apply_norm(raw_batch)
                        batch = _move_batch(batch, self.device)
                        self.model.reset()
                        generator = torch.Generator().manual_seed(self.config.diagnostic_seed + batch_index)
                        pred = self.model.sample_actions(batch, generator=generator)
                        pred_physical = self._action_to_physical(pred)
                        if not np.isfinite(pred_physical).all():
                            raise FloatingPointError(
                                f"non-finite sampled trajectory at completed step {self.global_step}"
                            )
                        target_physical = target.numpy()
                        sums = physical_trajectory_sums(pred_physical, target_physical, mask, layout)
                        for name, (num, den) in sums.items():
                            entry = totals.setdefault(name, [0.0, 0.0])
                            entry[0] += num
                            entry[1] += den
                        for local in range(pred_physical.shape[0]):
                            for action_step in range(pred_physical.shape[1]):
                                rows.append(
                                    [
                                        sample_count + local,
                                        action_step,
                                        target_physical[local, action_step].tolist(),
                                        pred_physical[local, action_step].tolist(),
                                        mask[local, action_step].tolist(),
                                    ]
                                )
                        sample_count += pred_physical.shape[0]
        finally:
            if cache is not None:
                self.model._last_actions = cache
                self.model._last_actions_valid = cache_valid
            self.model.train(was_training)

        metrics = {f"val/sample_{name}": (num / den if den > 0 else math.nan) for name, (num, den) in totals.items()}
        logger.info("sample diagnostics at step %d: %s", self.global_step, metrics)
        wandb_payload: dict[str, object] = dict(metrics)
        if self._wandb is not None and self._wandb_run is not None:
            try:
                wandb_payload["val/trajectory"] = self._wandb.Table(
                    columns=["sample", "action_step", "ground_truth", "prediction", "valid_mask"], data=rows
                )
                first = [row for row in rows if row[0] == 0]
                if first:
                    ground_truth = np.asarray([row[2] for row in first])
                    prediction = np.asarray([row[3] for row in first])
                    xs = list(range(len(first)))
                    # WHOLE-BODY: one plot for the 29 joint channels, in place of the per-role
                    # position plots (that layout has no position columns). Named by JOINT so a
                    # limb that tracks badly is identifiable without counting indices.
                    joint_dims = layout.joint_dims()
                    if joint_dims:
                        from fuji.s1_policy.data.layout import WB_JOINT_NAMES

                        joint_keys = [f"gt_{WB_JOINT_NAMES[d]}" for d in joint_dims]
                        joint_keys += [f"pred_{WB_JOINT_NAMES[d]}" for d in joint_dims]
                        joint_ys = [ground_truth[:, d].tolist() for d in joint_dims]
                        joint_ys += [prediction[:, d].tolist() for d in joint_dims]
                        wandb_payload["val/trajectory_joints"] = self._wandb.plot.line_series(
                            xs=xs, ys=joint_ys, keys=joint_keys,
                            title="29-DoF joint trajectory (rad)", xname="action step"
                        )
                    for role in layout.roles:
                        slots = layout.slots(role)
                        pos_dims = list(range(slots.pos.start, slots.pos.stop))
                        # Guarded for the same reason `physical_trajectory_sums` guards: a
                        # whole-body role's pose slices are deliberately 0-width, and an empty
                        # `ys` is a plot of nothing rather than an error.
                        if pos_dims:
                            pos_keys = [f"gt_{axis}" for axis in "xyz"] + [f"pred_{axis}" for axis in "xyz"]
                            pos_ys = [ground_truth[:, d].tolist() for d in pos_dims]
                            pos_ys += [prediction[:, d].tolist() for d in pos_dims]
                            wandb_payload[f"val/trajectory_{role}_position"] = self._wandb.plot.line_series(
                                xs=xs, ys=pos_ys, keys=pos_keys,
                                title=f"{role} position trajectory", xname="action step"
                            )
                        if slots.gripper is not None:
                            d = slots.gripper
                            wandb_payload[f"val/trajectory_{role}_gripper"] = self._wandb.plot.line_series(
                                xs=xs,
                                ys=[ground_truth[:, d].tolist(), prediction[:, d].tolist()],
                                keys=["ground_truth", "prediction"],
                                title=f"{role} gripper trajectory",
                                xname="action step",
                            )
                        if slots.force is not None:
                            force_dims = list(range(slots.force.start, slots.force.stop))
                            labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
                            force_keys = [f"gt_{label}" for label in labels]
                            force_keys += [f"pred_{label}" for label in labels]
                            force_ys = [ground_truth[:, d].tolist() for d in force_dims]
                            force_ys += [prediction[:, d].tolist() for d in force_dims]
                            wandb_payload[f"val/trajectory_{role}_wrench"] = self._wandb.plot.line_series(
                                xs=xs,
                                ys=force_ys,
                                keys=force_keys,
                                title=f"{role} wrench trajectory",
                                xname="action step",
                            )
            except Exception as exc:
                logger.warning("W&B trajectory plot creation failed: %s", exc)
        self._wandb_log(wandb_payload)
        return metrics

    def _monitor_mask_counts(self, batch: dict, storage: dict[str, list[float]]) -> None:
        if "action_mask" not in batch:
            return
        mask = batch["action_mask"]
        groups = self.dataset.layout.metric_group_dims()
        groups["action"] = tuple(range(mask.shape[-1]))
        # "joints" is present only for the whole-body layout and "pos"/"rot" only for the
        # keypoint ones; the loop already skips a group the layout omits, so one list covers
        # both without a mode branch.
        for name in ("action", "joints", "pos", "rot", "grip", "force"):
            dims = groups.get(name, ())
            if not dims:
                continue
            selected = mask[..., list(dims)]
            value = storage.setdefault(name, [0.0, 0.0])
            value[0] += float(selected.sum())
            value[1] += float(selected.numel())
        invalid = (~batch["force_history_valid"].bool()).sum()
        value = storage.setdefault("invalid_force_history", [0.0, 0.0])
        value[0] += float(invalid)
        value[1] += float(batch["force_history_valid"].numel())

    def _parameter_norm(self) -> float:
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            total += parameter.detach().float().square().sum()
        return float(total.sqrt().cpu())

    def _checkpoint_provenance(self) -> dict:
        out = {
            "split": self._split_provenance,
            "norm_stats_meta": self._norm_stats_meta,
            "norm_stats_sha256": self._norm_stats_sha256,
            "global_step_semantics": "completed_optimizer_updates",
        }
        # Recorded ONLY for a finetune, so every from-scratch checkpoint's provenance stays
        # byte-identical to what it was before this field existed. It lives here rather than in
        # the config because `init_from` is elided from every config comparison (see
        # _load_resume) -- this is the one place that still says where the weights came from.
        if self._init_from is not None:
            out["init_from"] = self._init_from.as_posix()
        return out

    def _capture_rng_state(self) -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    #: Stride between per-rank RNG streams (see :meth:`_decorrelate_rng_across_ranks`). A
    #: large odd constant, so ranks land far apart in seed space instead of on neighbouring
    #: seeds.
    _RANK_STREAM_STRIDE = 1_000_003

    def _decorrelate_rng_across_ranks(self) -> None:
        """Give each rank its OWN stochastic stream. No-op single-process and on rank 0.

        Every rank seeds identically in ``__init__`` (``config.seed``) so all replicas build
        the same weights without relying on DDP's construction-time broadcast. Left that way,
        the per-step stochastic draws would be identical on every rank too — and for the
        models that have them, that quietly cancels most of the benefit of adding GPUs:

        * **diffusion**: ``_sample_noise`` and ``_sample_timesteps`` are drawn per ROW, but
          from an identical stream, so world_size=8 x batch=32 covers 32 distinct
          ``(epsilon, t)`` pairs, each shared by 8 different action chunks — not 256. The
          estimator stays unbiased (each row's ``t`` is still uniform), but the ``(epsilon, t)``
          component of the gradient variance shrinks by 32, not by the advertised 256. That
          is the term diffusion training is dominated by.
        * **image augmentation** (``ImageAugment``, on in ``RECIPE=umi``): torchvision's
          ``RandomCrop``/``ColorJitter`` draw from the global CPU RNG, so all ranks would
          apply the SAME crop offset and the SAME jitter to their different frames.
        * **ACT**: the CVAE reparameterization ``randn_like(mu)`` is likewise shared.

        Nothing requires cross-rank RNG identity after construction: parameters stay in
        lockstep because DDP reduces gradients (bit-identically on every rank), not because
        the ranks draw the same numbers. ``evaluate`` and ``run_sample_diagnostics`` set an
        explicit seed via :func:`isolated_rng` and restore afterwards, so validation stays
        deterministic and this offset never reaches it.

        Rank 0 is deliberately UNTOUCHED, which buys two properties: a ``world_size=1`` run is
        bit-identical to before this existed, and rank 0 of an N-GPU run draws the same
        sequence a 1-GPU run of the same command would.

        The seed is derived by HASHING the current RNG state rather than from
        ``config.seed + rank``. Reading the state does not consume it (so rank 0's stream is
        genuinely untouched), and it makes the offset depend on where the stream actually is
        — so a resumed run does not replay the noise sequence it used at step 0, which a
        fixed ``config.seed + rank`` would.

        Caveat, deliberate: only rank 0's RNG state rides in ``training_state.pt`` (only rank
        0 writes it), so on resume every rank re-derives its stream from that one state. The
        streams are then deterministic and rank-distinct but NOT the continuation of what each
        rank held before the interruption. Exact resume of the DATA order, optimizer, EMA and
        schedule is unaffected; only which i.i.d. draws come next changes, and one i.i.d.
        sequence is as good as another. A ``world_size=1`` resume remains bit-exact.
        """
        if not self.dist_info.enabled or self.dist_info.rank == 0:
            return
        digest = hashlib.blake2b(
            torch.get_rng_state().numpy().tobytes(), digest_size=8
        ).digest()
        seed = (int.from_bytes(digest, "little")
                + self.dist_info.rank * self._RANK_STREAM_STRIDE) % (2**63 - 1)
        random.seed(seed)
        np.random.seed(seed % 2**32)  # numpy's legacy seeder is 32-bit
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("rank %d drew its own RNG stream (seed %d)", self.dist_info.rank, seed)

    def _training_state(self) -> dict:
        if not self._split_provenance:
            raise RuntimeError("build_dataloaders() must establish split provenance before saving resume state")
        sampler_state = self.train_sampler.state_dict() if self.train_sampler is not None else None
        if sampler_state is not None:
            sampler_state = dict(sampler_state)
            sampler_state["epoch"] = self.epoch
            sampler_state["offset"] = self.batch_in_epoch * self.config.batch_size
        return {
            "version": 3,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "rng": self._capture_rng_state(),
            "train_generator_state": self._train_generator.get_state(),
            "sampler": sampler_state,
            "best_metrics": self.best_metrics,
            "best_records": self.best_records,
            "wandb_run_id": self._wandb_run_id,
            "split_provenance": self._split_provenance,
            "norm_stats_sha256": self._norm_stats_sha256,
            "global_step_semantics": "completed_optimizer_updates",
            # Recorded HERE and not in TrainConfig on purpose: the resume path compares the
            # whole serialized config for exact equality, so a new config field would make
            # every checkpoint already on disk unresumable. The sampler enforces the match
            # (see EpochShuffleSampler.load_state_dict).
            "world_size": self.dist_info.world_size,
        }

    def _save_resume_checkpoint(self) -> Path:
        """Write the rank-0 checkpoint; every rank updates its bookkeeping and waits.

        Only rank 0 touches the filesystem — N ranks writing byte-identical multi-GB
        directories to one path would race on the same temp-then-rename and waste N-1x the
        I/O. All ranks still advance ``_last_checkpoint_step`` so the "did we already save at
        this step" branches in ``train``/``_complete_epoch`` stay in lockstep, and all ranks
        meet at the barrier so none races ahead while rank 0 is still writing.
        """
        path = Path(self.config.output_dir) / f"step_{self.global_step:08d}"
        if self.dist_info.is_main:
            path = save_checkpoint(
                self.model,
                self.config,
                self.global_step,
                self.config.output_dir,
                self._norm_stats_path,
                self.ema,
                training_state=self._training_state(),
                provenance=self._checkpoint_provenance(),
                norm_stats_text=self._norm_stats_text,
            )
        self._last_checkpoint_step = self.global_step
        barrier()
        return path

    def _load_init_weights(self, init_from: str) -> None:
        """Start this run from another checkpoint's WEIGHTS and nothing else.

        A finetune, not a resume: no optimizer, no scheduler, no step count, no RNG, no data
        provenance. See :data:`_INIT_ARCH_HL_KEYS` for what is checked and
        ``TrainConfig.init_from`` for why the rest deliberately is not.
        """
        from fuji.s1_policy.inference.common import strict_load_weights

        path = Path(init_from)
        config_path = path / "config.json"
        if not config_path.is_file() or not (path / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"--init-from {path} is not a deploy checkpoint dir (needs model.safetensors "
                f"and config.json; a step_NNNNNNNN/ directory is one)"
            )
        raw = json.loads(config_path.read_text())
        provenance = raw.get("_provenance") or {}
        expected_hash = provenance.get("model_sha256")
        if not expected_hash:
            raise ValueError(f"--init-from {path}/config.json has no _provenance.model_sha256")
        saved_config = {k: v for k, v in raw.items() if k != "_provenance"}
        current_config = json.loads(json.dumps(asdict(self.config), default=str))
        saved_config, current_config = _align_inactive_algo_blocks(saved_config, current_config)
        validate_init_config(saved_config, current_config)
        # strict=True: every tensor must be present and shaped as this config constructs it, so a
        # width the checks above cannot see still refuses rather than partially loading.
        strict_load_weights(self.model, path, expected_hash)
        self._init_from = path
        bundled_stats = path / "norm_stats.json"
        # Kept for the startup shift report only. The run's OWN stats are the ones it trains and
        # bundles -- reusing these values would make the finetuned checkpoint's provenance claim
        # a split it was not fitted on, which is what every reader compares.
        self._init_norm_stats_text = bundled_stats.read_text() if bundled_stats.is_file() else None
        logger.info(
            "FINETUNE: loaded weights from %s (sha256 %s...), fresh optimizer/scheduler/EMA, "
            "global_step back to 0", path, str(expected_hash)[:12],
        )

    def _report_init_norm_stats_shift(self) -> None:
        """Log how far this run's normalization moved from the one the init weights learned."""
        if self._init_from is None or self._norm_stats_text is None:
            return
        if self._init_norm_stats_text is None:
            logger.warning(
                "FINETUNE: --init-from %s bundles no norm_stats.json, so the shift between its "
                "normalization and this run's cannot be reported", self._init_from)
            return
        shift = norm_stats_shift(self._init_norm_stats_text, self._norm_stats_text)
        logger.info(
            "FINETUNE normalization shift (max relative change in q99-q01 per key): %s",
            ", ".join(f"{k} {v:+.1%}" for k, v in shift.items()) or "(no comparable keys)")
        worst = max(shift.values(), default=0.0)
        if worst > 0.5:
            logger.warning(
                "FINETUNE: normalization moved by up to %.0f%%. The loaded weights were trained "
                "against the OTHER range, so a short finetune spends its budget re-learning the "
                "scale. Check that both datasets are the same task and the same export placement.",
                100.0 * worst)

    def _load_resume(self, resume_from: str) -> None:
        path = Path(resume_from)
        state_path = path / "training_state.pt" if path.is_dir() else path
        if not state_path.exists():
            raise FileNotFoundError(f"resume state not found: {state_path}")
        # RNG and DataLoader generator states are required to stay CPU ByteTensors.
        # Model/optimizer/EMA loaders move their own tensors onto parameter devices.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("version") != 3:
            raise ValueError("resume checkpoint version is unsupported; start from a version-3 state")
        if state.get("global_step_semantics") != "completed_optimizer_updates":
            raise ValueError("resume checkpoint has unknown/off-by-one global_step semantics")
        resume_split_provenance = state.get("split_provenance")
        if not isinstance(resume_split_provenance, dict) or not resume_split_provenance:
            raise ValueError("version-3 resume checkpoint is missing split_provenance")
        saved_config = dict(state.get("config") or {})
        current_config = asdict(self.config)
        # Both flags name HOW this process was started, not what it trains, and both are elided
        # for the same reason: `--resume-from` is the one flag the resume command adds, and
        # `--init-from` is the one it must DROP (they are mutually exclusive, so a resume of a
        # finetune carries no --init-from and would otherwise differ from the saved config on
        # exactly that key). Eliding it also keeps every checkpoint written before this field
        # existed resumable, the compatibility rule _align_inactive_algo_blocks documents.
        for _cfg in (saved_config, current_config):
            _cfg.pop("resume_from", None)
            _cfg.pop("init_from", None)
        saved_config, current_config = _align_inactive_algo_blocks(saved_config, current_config)
        if saved_config != current_config:
            raise ValueError(
                "resume config differs from the saved training config; exact resume requires "
                "the original command plus only --resume-from"
            )
        if state.get("norm_stats_sha256") != self._norm_stats_sha256:
            raise ValueError(
                "resume normalization statistics differ from the saved file; use the exact "
                "original file or the norm_stats.json bundled with the checkpoint"
            )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if (state.get("ema") is None) != (self.ema is None):
            raise ValueError("resume checkpoint EMA setting differs from the current config")
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
            self.ema.to(self.device)
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.batch_in_epoch = int(state["batch_in_epoch"])
        self.best_metrics = dict(state.get("best_metrics", self.best_metrics))
        self.best_records = dict(state.get("best_records", {}))
        self._wandb_run_id = str(state.get("wandb_run_id", ""))
        self._resume_split_provenance = resume_split_provenance
        self._pending_rng_state = state["rng"]
        self._train_generator.set_state(state["train_generator_state"].cpu())
        self._resume_sampler_state = state.get("sampler")
        self._restore_bundled_bests(state_path.parent)
        logger.info(
            "Loaded resume state %s: completed_updates=%d epoch=%d batch_in_epoch=%d",
            state_path,
            self.global_step,
            self.epoch,
            self.batch_in_epoch,
        )

    def _validate_deploy_checkpoint(self, directory: Path, expected_hash: str) -> None:
        model_path = directory / "model.safetensors"
        config_path = directory / "config.json"
        if not model_path.is_file() or not config_path.is_file():
            raise ValueError(f"deploy checkpoint is incomplete: {directory}")
        if _file_sha256(model_path) != expected_hash:
            raise ValueError(f"{directory} does not match its recorded model SHA-256")
        try:
            raw_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"deploy checkpoint has an unreadable config: {directory}") from exc
        if not isinstance(raw_config, dict):
            raise ValueError(f"deploy checkpoint config is not an object: {directory}")
        provenance = raw_config.get("_provenance")
        if not isinstance(provenance, dict) or provenance.get("model_sha256") != expected_hash:
            raise ValueError(f"deploy checkpoint config has inconsistent model provenance: {directory}")

        saved_config = dict(raw_config)
        saved_config.pop("_provenance", None)
        current_config = json.loads(json.dumps(asdict(self.config), default=str))
        for _cfg in (saved_config, current_config):  # see _load_resume for why both flags go
            _cfg.pop("resume_from", None)
            _cfg.pop("init_from", None)
        saved_config, current_config = _align_inactive_algo_blocks(saved_config, current_config)
        if saved_config != current_config:
            raise ValueError(f"deploy checkpoint config differs from the resume config: {directory}")
        expected_split = self._resume_split_provenance or self._split_provenance
        if provenance.get("split") != expected_split:
            raise ValueError(f"deploy checkpoint split provenance differs from the resume state: {directory}")
        if provenance.get("norm_stats_sha256") != self._norm_stats_sha256:
            raise ValueError(f"deploy checkpoint normalization provenance differs from the resume state: {directory}")

        norm_path = directory / "norm_stats.json"
        if self._norm_stats_sha256 is None:
            if norm_path.exists():
                raise ValueError(f"unnormalized deploy checkpoint has a stale norm_stats.json: {directory}")
        elif not norm_path.is_file() or _file_sha256(norm_path) != self._norm_stats_sha256:
            raise ValueError(f"deploy checkpoint is missing its recorded normalization payload: {directory}")

    def _restore_bundled_bests(self, checkpoint_dir: Path) -> None:
        """Restore best deploy files included in a downloaded resume artifact."""
        output_dir = Path(self.config.output_dir)
        for name, record in self.best_records.items():
            source = checkpoint_dir / f"best_{name}"
            destination = output_dir / f"best_{name}"
            expected_hash = record.get("model_sha256")
            if not expected_hash:
                raise ValueError(f"resume best record {name!r} has no model SHA-256")
            source_model = source / "model.safetensors"
            destination_model = destination / "model.safetensors"
            if destination_model.is_file():
                self._validate_deploy_checkpoint(destination, expected_hash)
                continue
            if destination.exists():
                raise ValueError(f"existing best checkpoint directory is incomplete: {destination}")
            if source_model.is_file():
                self._validate_deploy_checkpoint(source, expected_hash)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
            else:
                raise FileNotFoundError(
                    f"best checkpoint {name!r} is recorded but absent from both "
                    f"{destination} and resume bundle {source}"
                )
        source_manifest = checkpoint_dir / "best_manifest.json"
        destination_manifest = output_dir / "best_manifest.json"
        if source_manifest.is_file() and not destination_manifest.exists():
            destination_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_manifest, destination_manifest)

    def _write_best_manifest(self) -> None:
        payload = {
            "selection": "deterministic_validation_objective",
            "eval_seeds": list(self.config.eval_seeds),
            "records": self.best_records,
        }
        _atomic_text(Path(self.config.output_dir) / "best_manifest.json", json.dumps(payload, indent=2, allow_nan=True))

    def _update_best(self, val_means: dict[str, float]) -> None:
        """Record best-so-far metrics on every rank; write the files on rank 0 only.

        The METRICS are all-reduced already (``evaluate``), so every rank reaches the same
        improvement verdict and ``best_metrics`` stays consistent for the next resume state.
        Only rank 0 writes ``best_*/`` and the manifest, and only it can compute the file
        hash, so ``best_records`` -- which feeds the resume state's integrity check -- is
        maintained on rank 0 alone.
        """
        for name in ("overall", "force"):
            value = val_means.get(name, math.nan)
            if not is_improvement(value, self.best_metrics[name]):
                continue
            self.best_metrics[name] = value
            if not self.dist_info.is_main:
                continue
            checkpoint_name = f"best_{name}"
            path = save_checkpoint(
                self.model,
                self.config,
                self.global_step,
                self.config.output_dir,
                self._norm_stats_path,
                self.ema,
                provenance=self._checkpoint_provenance(),
                checkpoint_name=checkpoint_name,
                norm_stats_text=self._norm_stats_text,
            )
            self.best_records[name] = {
                "metric": value,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "path": str(path),
                "model_sha256": _file_sha256(path / "model.safetensors"),
            }
            self._write_best_manifest()

    def _upload_resume_artifact(self, checkpoint: Path) -> None:
        if self._wandb_run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(
                f"{self._wandb_run.id}-resume",
                type="s1-resume",
                metadata={"global_step": self.global_step, "epoch": self.epoch},
            )
            artifact.add_dir(str(checkpoint))
            for name in self.best_records:
                best_dir = Path(self.config.output_dir) / f"best_{name}"
                if best_dir.is_dir():
                    artifact.add_dir(str(best_dir), name=f"best_{name}")
            manifest = Path(self.config.output_dir) / "best_manifest.json"
            if manifest.is_file():
                artifact.add_file(str(manifest))
            self._wandb_run.log_artifact(artifact, aliases=["latest-resume", f"epoch-{self.epoch}"])
        except Exception as exc:
            logger.warning("W&B resume artifact upload failed; local checkpoint is intact: %s", exc)

    # Deploy payload only. training_state.pt is ~80% of a checkpoint's bytes (optimizer +
    # EMA shadow + RNG), and pushing it every cadence hit would move tens of GB per run to
    # W&B for a file only a RESUME needs -- and the rolling "-resume" artifact already
    # carries that. An epoch artifact is for inspecting/deploying that epoch elsewhere.
    _EPOCH_ARTIFACT_FILES = ("model.safetensors", "config.json", "norm_stats.json")

    def _upload_epoch_artifact(self, checkpoint: Path) -> None:
        """Publish this epoch's deploy weights under its own immutable artifact name.

        The rolling ``-resume`` artifact keeps every upload as a new version of ONE
        artifact, so pulling "epoch 30" from another machine means knowing which version
        that was. A per-epoch name (``<run-id>-epoch_030``) is addressable directly.
        """
        if self._wandb_run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(
                f"{self._wandb_run.id}-epoch_{self.epoch:03d}",
                type="s1-epoch",
                metadata={
                    "global_step_completed_updates": self.global_step,
                    "epoch": self.epoch,
                    "algo": self.config.algo,
                    "checkpoint": checkpoint.name,
                    "contents": "deploy_only",
                },
            )
            for name in self._EPOCH_ARTIFACT_FILES:
                # norm_stats.json is absent for an unnormalized run; the other two always
                # exist because save_checkpoint wrote them before returning.
                member = checkpoint / name
                if member.is_file():
                    artifact.add_file(str(member), name=name)
            self._wandb_run.log_artifact(artifact, aliases=[f"epoch-{self.epoch}", "latest-epoch"])
            logger.info("Queued W&B epoch artifact for epoch %d from %s", self.epoch, checkpoint)
        except Exception as exc:
            logger.warning("W&B epoch artifact upload failed; local checkpoint is intact: %s", exc)

    def _upload_best_artifact(self) -> None:
        if self._wandb_run is None or self._wandb is None or not self.best_records:
            return
        try:
            artifact = self._wandb.Artifact(f"{self._wandb_run.id}-best", type="s1-best")
            for name in self.best_records:
                best_dir = Path(self.config.output_dir) / f"best_{name}"
                if best_dir.is_dir():
                    artifact.add_dir(str(best_dir), name=f"best_{name}")
            manifest = Path(self.config.output_dir) / "best_manifest.json"
            if manifest.exists():
                artifact.add_file(str(manifest))
            self._wandb_run.log_artifact(artifact, aliases=["best"])
        except Exception as exc:
            logger.warning("W&B best artifact upload failed; local best checkpoints are intact: %s", exc)

    def _finish_wandb(self) -> None:
        if self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception as exc:
            logger.warning("wandb.finish failed: %s", exc)

    def _validate_and_record(self) -> dict[str, float]:
        val_means = self.evaluate()
        if not val_means:
            return {}
        self._last_val_step = self.global_step
        logs = metric_log("val", val_means, self._loss_key)
        logs["val/epoch"] = self.epoch
        self._wandb_log(logs)
        logger.info(
            "validation epoch=%d step=%d objective=%.6f force=%s",
            self.epoch,
            self.global_step,
            val_means.get("overall", math.nan),
            f"{val_means['force']:.6f}" if math.isfinite(val_means.get("force", math.nan)) else "nan",
        )
        self._update_best(val_means)
        return val_means

    def _complete_epoch(self) -> None:
        self.epoch += 1
        self.batch_in_epoch = 0
        # A checkpoint written after the last batch predates this canonical epoch state.
        self._last_checkpoint_step = -1
        self._validate_and_record()
        if self.config.sample_every_epochs > 0 and self.epoch % self.config.sample_every_epochs == 0:
            # 4 samples, and its only consumer is a W&B table/plot. Running it on every rank
            # would duplicate the work and log nothing extra.
            if self.dist_info.is_main:
                self.run_sample_diagnostics()
        resume_due = self.config.artifact_every_epochs > 0 and self.epoch % self.config.artifact_every_epochs == 0
        # This cadence exists only to publish an epoch to W&B, so with no live run there is
        # nothing to publish -- do not write an extra multi-GB checkpoint no one can fetch.
        # `_wandb_run` is rank-0-only, so the answer is BROADCAST: the save inside this
        # branch contains a barrier, and if the ranks disagreed about entering it the peers
        # would never arrive and rank 0 would block there until the NCCL timeout.
        #
        # Evaluated UNCONDITIONALLY rather than as the tail of an `and` chain: `and`
        # short-circuits, so folding a collective into one makes the number of collectives
        # depend on the preceding terms. Those terms are config-derived (identical on every
        # rank) so today it is safe, but a future per-rank term anywhere earlier in the chain
        # would silently turn this into a deadlock. Cheap insurance -- one scalar per epoch.
        wandb_live = broadcast_flag(self._wandb_run is not None)
        epoch_due = (
            self.config.epoch_artifact_every_epochs > 0
            and self.epoch % self.config.epoch_artifact_every_epochs == 0
            and wandb_live
        )
        if resume_due or epoch_due:
            # One save serves both uploads: a second call would rewrite the same
            # multi-GB directory, and W&B reads the files lazily during upload.
            checkpoint = self._save_resume_checkpoint()
            if resume_due:
                self._upload_resume_artifact(checkpoint)
            if epoch_due:
                self._upload_epoch_artifact(checkpoint)

    def _close_data(self) -> None:
        for loader in (self.train_loader, self.val_loader):
            iterator = getattr(loader, "_iterator", None) if loader is not None else None
            if iterator is not None:
                try:
                    iterator._shutdown_workers()
                except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                    logger.warning("DataLoader worker cleanup failed: %s", exc)
        close = getattr(self.dataset, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                logger.warning("Dataset cleanup failed: %s", exc)

    def train(self) -> None:
        try:
            train_loader, _ = self.build_dataloaders()
        except Exception:
            self._close_data()
            self._finish_wandb()
            # No collective has run yet at this point, so this raise cannot deadlock its
            # peers -- torchrun reaps the whole job. Tear the group down anyway: leaving it
            # initialized keeps this rank's NCCL communicator alive while the process unwinds,
            # which can hold the GPU and delay the next launch.
            destroy_process_group()
            raise
        cfg = self.config
        if self._pending_rng_state is not None:
            self._restore_rng_state(self._pending_rng_state)
        # AFTER the restore, and in exactly one place: applying it twice would compose two
        # offsets, and applying it before the restore would be undone by it.
        self._decorrelate_rng_across_ranks()
        logger.info(
            "Starting training: completed_updates=%d max_steps=%d epoch=%d batch_in_epoch=%d device=%s bf16=%s",
            self.global_step,
            cfg.max_steps,
            self.epoch,
            self.batch_in_epoch,
            self.device,
            self.use_bf16,
        )
        if self.dist_info.enabled:
            world = self.dist_info.world_size
            # Stated loudly because --batch-size is PER RANK (the torchrun convention), so
            # the optimization problem differs from a single-GPU run of the same command:
            # fewer, larger updates per epoch. LR is deliberately NOT auto-scaled.
            logger.info(
                "DDP: world_size=%d rank=%d | batch_size=%d per rank -> EFFECTIVE BATCH %d | "
                "%d updates/epoch (single-GPU would be %d) | lr=%.3g NOT auto-scaled",
                world, self.dist_info.rank, cfg.batch_size, cfg.batch_size * world,
                self.train_sampler.shard_size // cfg.batch_size,
                len(train_loader.dataset) // cfg.batch_size, cfg.lr,
            )

        metric_window: dict = {}
        mask_window: dict[str, list[float]] = {}
        grad_norm_sum = 0.0
        data_wait_sum = 0.0
        step_time_sum = 0.0
        sample_sum = 0
        window_steps = 0
        cuda_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        try:
            while self.global_step < cfg.max_steps:
                offset = self.batch_in_epoch * cfg.batch_size
                self.train_sampler.set_epoch(self.epoch, offset)
                self._train_generator.manual_seed(cfg.seed + self.epoch)
                iterator = iter(train_loader)
                while self.global_step < cfg.max_steps:
                    wait_start = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    data_wait_sum += time.perf_counter() - wait_start
                    self._monitor_mask_counts(batch, mask_window)
                    if self.device.type == "cuda":
                        step_start = torch.cuda.Event(enable_timing=True)
                        step_end = torch.cuda.Event(enable_timing=True)
                        stream = torch.cuda.current_stream(self.device)
                        step_start.record(stream)
                    else:
                        step_start = time.perf_counter()
                    result = self.train_step(batch)
                    if self.device.type == "cuda":
                        step_end.record(stream)
                        cuda_step_events.append((step_start, step_end))
                    else:
                        step_time_sum += time.perf_counter() - step_start

                    # Step semantics: increment immediately after the completed update.
                    self.global_step += 1
                    self.batch_in_epoch += 1
                    _accumulate_tensor_sums(metric_window, result["metric_sums"])
                    grad_norm_sum += result["grad_norm"]
                    sample_sum += result["batch_size"]
                    window_steps += 1

                    if self.global_step % cfg.log_every == 0:
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                            step_time_sum += sum(start.elapsed_time(end) / 1000.0 for start, end in cuda_step_events)
                        means = _tensor_sum_means(metric_window)
                        logs = metric_log("train", means, self._loss_key)
                        logs.update(
                            {
                                # Group 0 is the main (non-backbone) group by construction in
                                # every param_groups() implementation, so train/lr keeps its
                                # historical meaning. A model with a second group also logs
                                # it, rather than leaving the trunk's schedule unobservable.
                                "train/lr": self.scheduler.get_last_lr()[0],
                                **({f"train/lr_group{i}": v
                                    for i, v in enumerate(self.scheduler.get_last_lr())}
                                   if len(self.optimizer.param_groups) > 1 else {}),
                                "train/grad_norm_preclip": grad_norm_sum / max(1, window_steps),
                                "train/data_wait_s": data_wait_sum / max(1, window_steps),
                                "train/step_time_s": step_time_sum / max(1, window_steps),
                                "train/samples_per_s": sample_sum / max(data_wait_sum + step_time_sum, 1e-9),
                                "train/epoch": self.epoch,
                            }
                        )
                        for name, (num, den) in mask_window.items():
                            logs[f"train/mask/{name}_ratio"] = num / den if den else math.nan
                        log_index = self.global_step // cfg.log_every
                        if cfg.param_norm_every > 0 and log_index % cfg.param_norm_every == 0:
                            logs["train/parameter_norm"] = self._parameter_norm()
                        if self.device.type == "cuda":
                            logs["train/gpu_max_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                            logs["train/gpu_max_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
                            torch.cuda.reset_peak_memory_stats(self.device)
                        self._wandb_log(logs)
                        logger.info(
                            "step %d train objective %.6f lr %.2e samples/s %.1f",
                            self.global_step,
                            means.get("overall", math.nan),
                            self.scheduler.get_last_lr()[0],
                            logs["train/samples_per_s"],
                        )
                        metric_window = {}
                        mask_window = {}
                        grad_norm_sum = data_wait_sum = step_time_sum = 0.0
                        sample_sum = window_steps = 0
                        cuda_step_events = []

                    if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
                        self._save_resume_checkpoint()

                if self.global_step >= cfg.max_steps:
                    break
                self._complete_epoch()

            # If max_steps lands exactly on an epoch boundary, record the completed
            # epoch rather than saving a state that would resume into an empty tail.
            # Per RANK: with drop_last=True each rank runs shard_size // batch_size batches,
            # and batch_in_epoch counts this rank's completed updates. Using the full dataset
            # length here would understate progress by world_size and re-run _complete_epoch.
            batches_per_epoch = (self.train_sampler.shard_size if self.train_sampler is not None
                                 else len(train_loader.dataset)) // cfg.batch_size
            if self.batch_in_epoch >= batches_per_epoch:
                self._complete_epoch()
            # A short smoke run or an interrupted schedule may end mid-epoch. Always do
            # one final deterministic validation so every completed run has best models.
            if self.val_loader is not None and self._last_val_step != self.global_step:
                self._validate_and_record()
                # Persist the updated best metrics/records even when a periodic checkpoint
                # was already written at this same optimizer step before final validation.
                self._last_checkpoint_step = -1
            if self._last_checkpoint_step != self.global_step:
                self._save_resume_checkpoint()
            self._upload_best_artifact()
            logger.info("Training complete at %d completed optimizer updates", self.global_step)
        finally:
            self._close_data()
            self._finish_wandb()
            # Last: an abort on one rank leaves the others in a collective, and tearing the
            # group down releases them to fail too instead of waiting out the NCCL timeout.
            destroy_process_group()
