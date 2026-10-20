"""Shared HL-v2 sample and result handling for offline S1 inference.

This module owns only algorithm-neutral behavior: reconstructing the exact training
dataset, selecting a real dataset anchor, applying the checkpoint's normalization,
and translating native-layout action chunks back into absolute trajectories. Flow
and diffusion checkpoint/model loading remain in independent sibling modules.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from flore.s1_policy.data.hl_dataset import HLDataset
from flore.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HORIZON,
    ACTION_HORIZON_ENV,
    DROP_FORCE_ENV,
    DROP_FORCE_INPUT_ENV,
    FORCE_HIST_LEN,
    ActionLayout,
    get_layout,
)
from flore.s1_policy.data.norm_stats import NormStats, normalize_quantile, parse_norm_stats, parse_norm_stats_meta
from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M, delta_ee_to_absolute
from flore.utils.dataset_paths import resolve_dataset_paths
from flore.utils.rot6d import rot6d_to_rotmat


@dataclass(frozen=True)
class SampleRef:
    """Stable identity of one exact, usable :class:`HLDataset` anchor."""

    dataset_index: int
    file_index: int
    source_path: str
    global_episode_id: int
    file_episode_id: int
    file_frame: int
    episode_frame: int
    split: str


@dataclass(frozen=True)
class RoleTrajectory:
    """Physical prediction and target for one native-layout role."""

    role: str
    predicted_position: np.ndarray
    predicted_rot6d: np.ndarray
    target_position: np.ndarray
    target_rot6d: np.ndarray
    position_valid: np.ndarray
    rotation_valid: np.ndarray
    pose_valid: np.ndarray
    predicted_gripper_m: np.ndarray | None = None
    target_gripper_m: np.ndarray | None = None
    gripper_valid: np.ndarray | None = None
    predicted_wrench: np.ndarray | None = None
    target_wrench: np.ndarray | None = None
    wrench_valid: np.ndarray | None = None


@dataclass(frozen=True)
class RolloutResult:
    """One deterministic offline rollout against one exact training sample."""

    algorithm: str
    layout_mode: str
    seed: int
    num_steps: int
    sample: SampleRef
    action_times_s: np.ndarray
    predicted_actions: np.ndarray
    target_actions: np.ndarray
    action_mask: np.ndarray
    state_valid: np.ndarray
    force_history_valid: bool
    conditioning_images: tuple[np.ndarray, ...]
    roles: tuple[RoleTrajectory, ...]
    metrics: Mapping[str, float]
    #: Per-F/T-sensor force-window validity, aligned with ``layout.force_roles``. ``()`` on a
    #: single-sensor layout, where ``force_history_valid`` above already says everything.
    #: A dual-Bota layout gates each sensor independently (they tare independently, so one
    #: hand's window can be masked while the other's is fine), and the scalar above is the
    #: conjunction — so it is the pessimistic summary, not a claim about either sensor.
    force_history_valid_per_sensor: tuple[bool, ...] = ()


def _readonly(array: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    # bytes owns an immutable buffer, so callers cannot undo this with setflags(write=True).
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _logical_split_provenance(actual: dict, logical_paths: tuple[str, ...]) -> dict:
    """Replace machine-local resolved paths while retaining every content identity."""
    out = copy.deepcopy(actual)
    out["dataset_knobs"]["zarr_paths"] = [Path(path).as_posix() for path in logical_paths]
    return out


class HLSampleSource:
    """Exact training dataset plus stable episode/frame identities for its anchors."""

    def __init__(self, hl_data: Mapping[str, Any], zarr_paths: tuple[str, ...] | None = None) -> None:
        expected_fields = {
            "zarr_paths",
            "mode",
            "image_key",
            "image_mean",
            "image_std",
            "obs_history_len",
            "obs_history_dt_s",
            "drop_invalid_force_history",
            "drop_glitch_windows",
            "norm_stats_path",
            "val_last_episodes",
        }
        # Fields this class TOLERATES but does not require. The membership test is
        # bidirectional on purpose — an unknown field means the checkpoint carries a knob this
        # code does not honour, a missing one means it predates a knob this code needs — so a
        # field added to HLDataConfig LATER must be listed here, or every checkpoint written
        # before it is refused for lacking a key it could not have had. `image_keys` (the
        # multi-camera list, default empty) is exactly that case.
        # `state_noise_*` are TRAINING-ONLY augmentation: `StateNoise` wraps the train subset,
        # so an inference session must read the clean state — which is why they are tolerated
        # rather than honoured here. They are still recorded in the checkpoint (an ablation cell
        # has to be identifiable after the fact), so they must be listed or every geom-mode
        # checkpoint would be refused for carrying a knob offline inference correctly ignores.
        optional_fields = {"image_keys", "wb_joint_target",
                           "state_noise_xy_m", "state_noise_rpy_deg"}
        unknown = sorted(set(hl_data) - expected_fields - optional_fields)
        missing = sorted(expected_fields - set(hl_data))
        if unknown:
            raise ValueError(f"checkpoint has unsupported HL data fields: {unknown}")
        if missing:
            raise ValueError(f"checkpoint is missing HL data fields: {missing}")
        configured_paths = tuple(str(path) for path in hl_data.get("zarr_paths", ()))
        # The checkpoint's recorded paths are the TRAINING machine's; relocate them to this one
        # (basename match in the local dataset dir) when they are absent. Content identity is
        # still checked by assert_provenance, so this cannot swap in a different dataset.
        resolved_paths = resolve_dataset_paths(
            (str(path) for path in (zarr_paths or configured_paths)), log=print
        )
        if not resolved_paths:
            raise ValueError("checkpoint has no HL zarr paths; pass an explicit zarr path override")
        if zarr_paths is not None and configured_paths and len(resolved_paths) != len(configured_paths):
            raise ValueError(
                "zarr override count must preserve checkpoint file order: "
                f"expected {len(configured_paths)}, got {len(resolved_paths)}"
            )
        logical_paths = configured_paths or resolved_paths

        self.layout = get_layout(str(hl_data.get("mode", "left_only")))
        if self.layout.legacy:
            raise ValueError("offline HL inference does not support the legacy32 layout")
        # MULTI-CAMERA is CARRIED, not dropped. `image_keys` is the full ordered camera list
        # the checkpoint was trained on (absent on every pre-diffusion_umi_3tracker
        # checkpoint, hence the allow-list above); it is handed to HLDataset verbatim so the
        # observation this class rebuilds has the SAME number of views the weights expect.
        # Passing only `image_key` would feed a 2-camera model half its conditioning at
        # exactly the right shapes and with no error -- which is why this used to refuse the
        # checkpoint outright. `validate_model_layout` still pins the count against the
        # weights, so a dataset missing camera1 fails by name rather than being padded.
        #
        # ⚠️ The ONE thing that stays single-camera is the recorded-episode PLAYBACK helpers
        # below (`episode_image` -> HLDataset.episode_image -> self.image_key): they show the
        # demonstration to a human, they are not model input, and camera0 is the wrist view
        # the whole viewer vocabulary is built around. Nothing that feeds the model reads
        # them.
        cam_keys = tuple(str(k) for k in (hl_data.get("image_keys") or ()))
        self.image_keys = cam_keys or (str(hl_data.get("image_key", "camera0_rgb")),)
        self.val_last_episodes = int(hl_data.get("val_last_episodes", 0))
        self.image_mean = float(hl_data.get("image_mean", 0.5))
        self.image_std = float(hl_data.get("image_std", 0.5))
        # CARRIED for the same reason `image_keys` is: it decides what the checkpoint's 29 joint
        # target columns MEAN, and the width is identical either way, so reconstructing the
        # dataset without it would score a delta-trained model against absolute labels (or the
        # reverse) with plausible-looking numbers. Default `absolute` = every checkpoint written
        # before the knob existed. Read by `build_wholebody_result`, which un-relativizes so the
        # drawn bodies stay postures.
        self.wb_joint_target = str(hl_data.get("wb_joint_target", "absolute"))
        self.dataset = HLDataset(
            list(resolved_paths),
            mode=self.layout.mode,
            image_key=str(hl_data.get("image_key", "camera0_rgb")),
            image_keys=cam_keys or None,
            image_mean=self.image_mean,
            image_std=self.image_std,
            drop_invalid_force_history=bool(hl_data.get("drop_invalid_force_history", False)),
            drop_glitch_windows=bool(hl_data.get("drop_glitch_windows", True)),
            obs_history_len=int(hl_data.get("obs_history_len", 2)),
            obs_history_dt_s=float(hl_data.get("obs_history_dt_s", 0.25)),
            require_episode_ends=self.val_last_episodes > 0,
            wb_joint_target=self.wb_joint_target,
        )
        self.resolved_paths = resolved_paths
        self.logical_paths = logical_paths
        actual_split = self.dataset.split_provenance(self.val_last_episodes)
        self.split_provenance = _logical_split_provenance(actual_split, logical_paths)

        first_val = self.dataset.episode_count - self.val_last_episodes
        refs: list[SampleRef] = []
        for dataset_index, ((file_index, frame), global_episode_id) in enumerate(
            zip(self.dataset._index, self.dataset.sample_episode_ids, strict=True)
        ):
            identity_file, file_episode_id = self.dataset.episode_identities[global_episode_id]
            if identity_file != file_index:
                raise RuntimeError("HLDataset episode identity disagrees with its anchor index")
            ends = self.dataset._episode_ends_by_file[file_index]
            episode_start = 0 if file_episode_id == 0 else int(ends[file_episode_id - 1])
            refs.append(
                SampleRef(
                    dataset_index=dataset_index,
                    file_index=file_index,
                    source_path=resolved_paths[file_index],
                    global_episode_id=int(global_episode_id),
                    file_episode_id=int(file_episode_id),
                    file_frame=int(frame),
                    episode_frame=int(frame) - episode_start,
                    split="validation" if global_episode_id >= first_val else "train",
                )
            )
        self.refs = tuple(refs)

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, dataset_index: int) -> tuple[SampleRef, dict]:
        index = int(dataset_index)
        if not 0 <= index < len(self):
            raise IndexError(f"sample index {index} is outside [0, {len(self) - 1}]")
        return self.refs[index], self.dataset[index]

    @property
    def file_attrs(self) -> tuple[Mapping[str, Any], ...]:
        """Root attrs of each source zarr, in this dataset's file order.

        Read straight off :attr:`HLDataset.file_attrs`. The viewer needs these to place its
        display cluster from the dataset's OWN ``T_tracker_tcp`` extrinsic (which differs
        between datasets), so this deliberately exposes provenance rather than letting a
        display layer hardcode a rig constant."""
        return self.dataset.file_attrs

    def episode_forward_positions(self, ref: SampleRef) -> dict[str, np.ndarray]:
        """Recorded absolute EE positions from this sample's frame to its episode end.

        One ``(N, 3)`` array per layout role, in the SAME absolute aligned-world frame as
        the rollout's ``target_position`` (a viewer draws it continuous with the target
        line — its first row equals the target chunk's step-0 target). See
        :meth:`HLDataset.episode_forward_positions`; the tail is the raw recording and is
        deliberately not glitch-masked."""
        return self.dataset.episode_forward_positions(ref.file_index, ref.file_frame)

    def episode_forward_poses(self, ref: SampleRef) -> tuple[np.ndarray, dict[str, dict]]:
        """Recorded absolute poses (+ gripper width) + camera times from this sample's
        frame to its episode end, for PLAYING the recorded demonstration forward.

        ``(times_s, per_role)`` — see :meth:`HLDataset.episode_forward_poses`. Row 0 of
        every role is the sample's own anchor (step-0 target), so a viewer rebases the tail
        against the SAME reference it uses for the action chunk. Each force role also carries
        its per-frame measured ``wrench``/``wrench_valid`` (still in the bota_sensor frame), so
        playback can draw the recorded contact force alongside the swept pose. Same
        raw-recording caveat as :meth:`episode_forward_positions`."""
        return self.dataset.episode_forward_poses(ref.file_index, ref.file_frame)

    def episode_image(self, ref: SampleRef, frame_offset: int) -> np.ndarray:
        """Raw ``(H, W, 3)`` uint8 camera frame ``frame_offset`` steps into this sample's
        episode tail (0 == the sample's own observation frame ``t0``).

        The undecorated recording, for showing the observation as the demonstration plays.
        ``frame_offset`` is clamped into ``[0, tail_len)`` by the dataset's own bounds check
        via the absolute frame ``ref.file_frame + frame_offset``."""
        return self.dataset.episode_image(ref.file_index, ref.file_frame + int(frame_offset))

    def close(self) -> None:
        self.dataset.close()


_SPLIT_KEYS = (
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


def validate_split_provenance(expected: Mapping[str, Any], actual: Mapping[str, Any], label: str) -> None:
    """Validate all split/content fields; only workstation-local path spelling may differ."""
    missing = [key for key in _SPLIT_KEYS if key not in expected]
    if missing:
        raise ValueError(f"{label} split provenance is incomplete (missing {', '.join(missing)})")
    expected_copy = copy.deepcopy(dict(expected))
    actual_copy = copy.deepcopy(dict(actual))
    expected_copy["dataset_knobs"].pop("zarr_paths", None)
    actual_copy["dataset_knobs"].pop("zarr_paths", None)
    for key in _SPLIT_KEYS:
        if expected_copy[key] != actual_copy.get(key):
            raise ValueError(
                f"{label} split provenance mismatch for {key}: "
                f"checkpoint has {expected_copy[key]!r}, dataset has {actual_copy.get(key)!r}"
            )


def read_checkpoint_config(checkpoint_dir: str | Path, expected_algorithm: str) -> tuple[Path, dict]:
    checkpoint = Path(checkpoint_dir)
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"{checkpoint} must contain config.json and model.safetensors")
    raw = json.loads(config_path.read_text())
    if raw.get("algo") != expected_algorithm:
        raise ValueError(f"checkpoint algo={raw.get('algo')!r}; expected {expected_algorithm!r}")
    if raw.get("pipeline") != "hl":
        raise ValueError(f"offline inference requires pipeline='hl', got {raw.get('pipeline')!r}")
    if not isinstance(raw.get("hl_data"), dict):
        raise ValueError("checkpoint config is missing its hl_data block")
    provenance = raw.get("_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint is missing _provenance")
    expected_hash = provenance.get("model_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("checkpoint is missing a valid _provenance.model_sha256")
    try:
        int(expected_hash, 16)
    except ValueError as exc:
        raise ValueError("checkpoint _provenance.model_sha256 is not hexadecimal") from exc
    if not isinstance(raw.get("ema"), bool):
        raise ValueError(f"checkpoint config.ema must be a boolean, got {raw.get('ema')!r}")
    expected_kind = "ema" if raw.get("ema") is True else "live"
    if provenance.get("deploy_weights") != expected_kind:
        raise ValueError(
            f"checkpoint deploy_weights={provenance.get('deploy_weights')!r} disagrees with "
            f"config.ema={raw.get('ema')!r}"
        )
    return checkpoint, raw


#: Fields a checkpoint written BEFORE they existed may legitimately omit, in which case the
#: dataclass default applies.
#:
#: The missing-field check below is otherwise deliberately strict, because a defaulted
#: dimension or scheduler setting means serving a model whose weights say one thing and
#: whose config says another — silently. These specific fields are exempt because they are
#: TRAIN-TIME ONLY and provably cannot change a single inference-time tensor:
#:
#:   * ``image_aug*`` — :class:`~flore.s1_policy.model.umi_parity.ImageAugment` returns its
#:     input unchanged unless ``self.training``, and inference always runs under ``eval()``.
#:     It also holds no parameters, so it adds no key to ``model.safetensors``.
#:   * ``encoder_lr_scale`` — read only by ``Model.param_groups`` when building AdamW.
#:
#: Anything else stays refused. Do not extend this set for a field that any eval-mode code
#: path reads.
_OPTIONAL_MODEL_CONFIG_FIELDS = frozenset({
    "image_aug",
    "image_aug_crop_ratio",
    "image_aug_brightness",
    "image_aug_contrast",
    "image_aug_saturation",
    "image_aug_hue",
    "encoder_lr_scale",
})

#: ``dataset_image_mean`` / ``dataset_image_std`` are train-time-only on the flow and
#: diffusion configs (they only invert the affine inside ``ImageAugment``), but ACT reads
#: them AT INFERENCE to re-normalize to ImageNet statistics before its frozen-BatchNorm
#: trunk — defaulting them there would feed the trunk out-of-distribution input, which is
#: precisely the bug the re-normalization exists to prevent. So they are exempt only on a
#: config that does NOT declare ``imagenet_renormalize``. Keyed on that field's presence
#: rather than on a class name, so the exemption tracks the actual reason.
_OPTIONAL_IMAGE_AFFINE_FIELDS = frozenset({"dataset_image_mean", "dataset_image_std"})


def dataclass_config(config_type, values: Mapping[str, Any]):
    """Reconstruct a saved model dataclass without silently dropping unknown fields."""
    fields = {field.name for field in dataclasses.fields(config_type)}
    unknown = sorted(set(values) - fields)
    if unknown:
        raise ValueError(f"checkpoint has unsupported {config_type.__name__} fields: {unknown}")
    optional = _OPTIONAL_MODEL_CONFIG_FIELDS
    if "imagenet_renormalize" not in fields:
        optional = optional | _OPTIONAL_IMAGE_AFFINE_FIELDS
    missing = sorted(fields - set(values) - optional)
    if missing:
        raise ValueError(f"checkpoint is missing {config_type.__name__} fields: {missing}")
    kwargs = {key: tuple(value) if isinstance(value, list) else value for key, value in values.items()}
    config = config_type(**kwargs)
    config.pretrained = False  # A local checkpoint must never trigger a network download.
    return config


def validate_model_layout(config, hl_data: Mapping[str, Any], layout: ActionLayout,
                          *, grid_from_checkpoint: bool = False,
                          num_cameras: int | None = None) -> None:
    """Check a checkpoint's config against the layout (and this build's grid).

    ``num_cameras`` — how many views the CALLER's observation carries. ``None`` (the default)
    derives it from ``hl_data.image_keys``, which is right for every DATASET-BACKED consumer:
    :class:`HLSampleSource` rebuilds the training observation, so it has exactly as many views
    as the checkpoint was trained on. A caller that assembles its own observation must pass
    its real count, and :func:`flore.s1_policy.serving.hl_online.load_hl_model` passes ``1``
    because the live planner subscribes to one D405 — that is what refuses a 2-camera
    checkpoint at the robot instead of feeding it half its conditioning.

    ``grid_from_checkpoint`` — set it ONLY if the caller consumes the checkpoint's own
    ``force_hist_len`` rather than :data:`FORCE_HIST_LEN`. It exempts that one field from the
    build-contract pin, because a dataset-free consumer (the live planner: see
    :class:`flore.s1_policy.serving.hl_online.LoadedHLModel`) has nothing for the constant to
    disagree WITH, and pinning it there would refuse an older but perfectly serviceable model
    the moment the contract moves — v2's 8 bins -> v3's 1 is exactly that move. The
    dataset-backed inference sessions leave it False: their ``HLSampleSource`` feeds the model
    from a zarr whose own attrs are pinned to the same constant, so a checkpoint that disagrees
    would be fed windows of the wrong length.

    ``action_horizon`` stays pinned either way — ``scripts/hl_serving.py`` sizes its
    execute-N-of-S choices and clamps from the constant, so a checkpoint with another horizon
    needs more than a derived window before it can be served.
    """
    expected = {
        "layout_mode": layout.mode,
        "action_dim": layout.action_dim,
        "action_horizon": ACTION_HORIZON,
        "state_dim": layout.state_dim,
        "obs_history_len": int(hl_data.get("obs_history_len", 2)),
        # What the CALLER's observation actually carries -- see the `num_cameras` arg. Derived
        # from the checkpoint's own camera list for the dataset-backed consumers (they rebuild
        # the training observation, cameras included) and pinned to 1 by the live planner,
        # which subscribes to one D405. Getting this wrong in the permissive direction is the
        # dangerous one: a 2-camera model handed one view is shape-correct and silent.
        "num_cameras": (int(num_cameras) if num_cameras is not None
                        else max(1, len(tuple(hl_data.get("image_keys") or ())))),
        # What the MODEL consumes, not what the dataset emitted: the two differ under the
        # force-output-only ablation, and this gate exists to stop a model built without a
        # force encoder from being handed a process that thinks it has one.
        "force_hist_channels": layout.force_input_channels,
    }
    if not grid_from_checkpoint:
        # `force_input_channels` again, for the same reason the channel pin above uses it: a
        # model built with NO force encoder reads no window at all, so there is no length for
        # the dataset's window to disagree WITH, and the build's bin count is not a contract
        # it is party to. BOTH force-less ablations make training record `force_hist_len=0`
        # (config.py: `FORCE_HIST_LEN if layout.force_input_channels else 0`), so pinning the
        # constant here refused every DATASET-BACKED consumer -- infer_s1_*.py,
        # reinfer_hl_take.py, eval_checkpoints_physical.py -- for a checkpoint the live
        # planner loads fine, and with no hint attached to say why.
        #
        # This masks nothing: `force_hist_channels` is checked BEFORE this key, so a channel
        # disagreement (process expects force, checkpoint has none, or the reverse) still
        # fires first with its own hint, and the (len, channels) coherence check below still
        # refuses the pair going incoherent. By the time this line is compared, the two sides
        # already agree on WHETHER force is consumed; it only ever asks HOW MUCH.
        expected["force_hist_len"] = FORCE_HIST_LEN if layout.force_input_channels else 0
    for field, wanted in expected.items():
        got = getattr(config, field)
        if got != wanted:
            # The force-channel mismatch has exactly one cause in practice and a one-word fix,
            # so say it. Without this the operator reads "0 disagrees with 6" at the robot and
            # has no way to know a per-process env var decides it.
            hint = ""
            if field == "action_horizon":
                # Same reason as the force-channel hint below: a per-process env var decides
                # this, and "30 disagrees with 50" gives the operator no way to know that.
                # scripts/run_hl_serving.sh derives the pin from the checkpoint, so this
                # message is for the consumers that have no launcher (reinfer_hl_take.py,
                # infer_s1_*.py, eval_checkpoints_physical.py).
                hint = (f" -- the horizon is a per-process pin; export "
                        f"{ACTION_HORIZON_ENV}={got!r} for this process (a SHORTER horizon is a "
                        f"strict prefix of the stored chunk at the same dt, so no re-convert is "
                        f"needed; a LONGER one does need re-converted data)")
            if field == "num_cameras":
                hint = (f" -- this consumer builds a {wanted}-camera observation. The offline"
                        f" dataset-backed tools (infer_s1_*.py) take the count from the"
                        f" checkpoint's own hl_data.image_keys and so follow a"
                        f" diffusion_umi_3tracker checkpoint automatically; SERVING and"
                        f" reinfer_hl_take pin 1, because the live rig subscribes to one D405"
                        f" and a recorded deploy take stored one image -- a {got}-camera"
                        f" checkpoint cannot be served until the second view is plumbed"
                        f" through hl_online/hl_take")
            if field == "force_hist_channels":
                if not got and wanted:
                    # TWO ablations produce a force-encoder-less checkpoint and they are not
                    # interchangeable, so name both: DROP_FORCE also stops supervising the
                    # force COLUMNS, DROP_FORCE_INPUT keeps predicting them. The action mask
                    # in the checkpoint's own config tells them apart if in doubt.
                    hint = (f" -- the checkpoint was built with NO force encoder. If it is the "
                            f"force-output-only ablation (it still predicts the force columns), "
                            f"export {DROP_FORCE_INPUT_ENV}=1; if it is PURE POSITION (force "
                            f"columns unsupervised too), export {DROP_FORCE_ENV}=1")
                elif got and not wanted:
                    hint = (f" -- this process has {DROP_FORCE_ENV} or "
                            f"{DROP_FORCE_INPUT_ENV} set but the checkpoint WAS trained with a "
                            f"force encoder; unset it")
            raise ValueError(
                f"checkpoint {field}={got!r} disagrees with HL layout/data value {wanted!r}{hint}")
    # A force-less checkpoint must not claim a window, and vice versa — the channel width is
    # still pinned above, so this only catches the (len, channels) pair going incoherent, which
    # `grid_from_checkpoint` would otherwise let through as len=0 with 6 channels.
    if bool(config.force_hist_channels) != bool(config.force_hist_len):
        raise ValueError(
            f"checkpoint force_hist_channels={config.force_hist_channels} and "
            f"force_hist_len={config.force_hist_len} disagree about whether force is present")
    # The zero-pad truncation check applies ONLY to the flat-global_cond models (Flow and
    # Diffusion), whose ``_pad_state`` right-pads state into a ``filled_state_dim`` slot and
    # would silently truncate on a negative F.pad width. ACT has no such slot — state enters
    # as a transformer TOKEN through its own Linear — so ``DPACTConfig`` declares
    # ``state_projection=True`` and carries no ``filled_state_dim`` at all. Defaulting the
    # getattr to True is the safe polarity: it SKIPS the check for a config with no zero-pad
    # path, whereas defaulting to False would fall through and AttributeError on the very
    # next attribute.
    if not getattr(config, "state_projection", True) and config.filled_state_dim < layout.state_dim:
        raise ValueError(
            f"checkpoint filled_state_dim={config.filled_state_dim} would truncate layout state_dim={layout.state_dim}"
        )


def load_norm_bundle(raw: Mapping[str, Any], checkpoint: Path, source: HLSampleSource) -> dict[str, NormStats]:
    """Load and content-bind the bundled normalization payload to config and dataset."""
    provenance = raw.get("_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint is missing _provenance")
    split = provenance.get("split")
    if not isinstance(split, dict):
        raise ValueError("checkpoint is missing _provenance.split")
    validate_split_provenance(split, source.split_provenance, "checkpoint")

    configured = bool(raw["hl_data"].get("norm_stats_path"))
    norm_path = checkpoint / "norm_stats.json"
    expected_hash = provenance.get("norm_stats_sha256")
    expected_meta = provenance.get("norm_stats_meta")
    if not configured:
        if expected_hash is not None or norm_path.exists():
            raise ValueError("checkpoint bundles normalization but hl_data.norm_stats_path is empty")
        return {}
    if not norm_path.is_file():
        raise FileNotFoundError("normalized checkpoint is missing bundled norm_stats.json")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("normalized checkpoint is missing _provenance.norm_stats_sha256")
    if not isinstance(expected_meta, dict):
        raise ValueError("normalized checkpoint is missing _provenance.norm_stats_meta")

    payload = norm_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"norm_stats.json hash mismatch: expected {expected_hash}, got {actual_hash}")
    text = payload.decode("utf-8")
    actual_meta = parse_norm_stats_meta(text)
    if actual_meta != expected_meta:
        raise ValueError("norm_stats.json metadata differs from checkpoint provenance")
    validate_split_provenance(actual_meta, source.split_provenance, "normalization")
    stats = parse_norm_stats(text)
    missing = sorted({"actions", "state"} - stats.keys())
    if missing:
        raise ValueError(f"normalization statistics are missing {', '.join(missing)}")
    expected_widths = {
        "actions": source.layout.action_dim,
        "state": source.layout.state_dim,
        "force_history": source.layout.force_hist_channels,
    }
    for key, norm in stats.items():
        if key not in expected_widths:
            continue
        expected_shape = (expected_widths[key],)
        if norm.q01.shape != expected_shape or norm.q99.shape != expected_shape:
            raise ValueError(
                f"norm_stats[{key!r}] q01/q99 shapes {norm.q01.shape}/{norm.q99.shape} != {expected_shape}"
            )
        if not np.isfinite(norm.q01).all() or not np.isfinite(norm.q99).all():
            raise ValueError(f"norm_stats[{key!r}] contains non-finite q01/q99 values")
        if np.any(norm.q99 < norm.q01):
            raise ValueError(f"norm_stats[{key!r}] contains q99 values below q01")
    return stats


def strict_load_weights(model: torch.nn.Module, checkpoint: Path, expected_hash: str) -> None:
    from safetensors.torch import load

    payload = (checkpoint / "model.safetensors").read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"model.safetensors hash mismatch: expected {expected_hash}, got {actual_hash}")
    state = load(payload)
    expected = model.state_dict()
    for name in state.keys() & expected.keys():
        if state[name].dtype != expected[name].dtype:
            raise ValueError(
                f"model tensor dtype mismatch for {name}: checkpoint has {state[name].dtype}, "
                f"config constructs {expected[name].dtype}"
            )
    model.load_state_dict(state, strict=True)


def configure_deterministic_inference(device: torch.device) -> None:
    """Make a fresh CUDA inference process fail rather than use nondeterministic kernels."""
    if device.type != "cuda":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_inference_device(device: str | torch.device) -> torch.device:
    """Resolve bare ``cuda`` once so later current-device changes cannot split a session."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {str(device)!r} requested but CUDA is unavailable")
    if resolved.index is None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def validate_sampler_steps(value: Any, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"num_steps must be an integer, got {value!r}")
    steps = int(value)
    if steps < 1 or (maximum is not None and steps > maximum):
        requirement = f"in [1, {maximum}]" if maximum is not None else "positive"
        raise ValueError(f"num_steps must be {requirement}, got {steps}")
    return steps


def prepare_model_batch(
    sample: Mapping[str, torch.Tensor],
    norm_stats: Mapping[str, NormStats],
    device: torch.device,
) -> dict:
    """Collate one HL sample and normalize only the inputs used during sampling."""
    image = sample["image"]
    if image.dim() == 4:
        images = [image[index].unsqueeze(0) for index in range(image.shape[0])]
    else:
        images = [image.unsqueeze(0)]
    batch = {
        "images": images,
        "img_masks": [torch.ones(1, dtype=torch.bool) for _ in images],
        "state": sample["state"].unsqueeze(0),
        "state_valid": sample["state_valid"].unsqueeze(0),
        "force_history": sample["force_history"].unsqueeze(0),
        # unsqueeze, NOT reshape(1): HLDataset emits a SCALAR for a one-sensor layout and an
        # (n_ft,) vector for more (`both_hands_head_dual` has two Botas), and the force
        # encoder accepts exactly (B,) or (B, groups) — refusing anything else so it cannot
        # gate the wrong sensor. Unsqueezing produces (1,) and (1, n_ft) respectively, which
        # is also byte-for-byte what the trainer's `torch.stack` collate yields at B=1;
        # reshape(1) hard-assumed one sensor and raised on the dual layout.
        "force_history_valid": sample["force_history_valid"].unsqueeze(0),
    }
    for key in ("state", "force_history"):
        if key in norm_stats:
            normalized = normalize_quantile(batch[key].numpy(), norm_stats[key])
            batch[key] = torch.from_numpy(normalized).to(batch[key].dtype)
    return {
        key: ([tensor.to(device) for tensor in value] if isinstance(value, list) else value.to(device))
        for key, value in batch.items()
    }


def conditioning_images(sample: Mapping[str, torch.Tensor], mean: float, std: float) -> tuple[np.ndarray, ...]:
    image = sample["image"].detach().cpu().numpy()
    if image.ndim == 3:
        image = image[None]
    rgb = np.transpose(image, (0, 2, 3, 1))
    rgb = np.rint(np.clip((rgb * std + mean) * 255.0, 0.0, 255.0)).astype(np.uint8)
    return tuple(_readonly(frame) for frame in rgb)


def _valid_rot6d(rot6d: np.ndarray) -> np.ndarray:
    rows = np.asarray(rot6d, dtype=np.float64).reshape(-1, 2, 3)
    first_norm = np.linalg.norm(rows[:, 0], axis=-1)
    first = rows[:, 0] / np.maximum(first_norm[:, None], 1e-12)
    second_residual = rows[:, 1] - np.sum(rows[:, 1] * first, axis=-1, keepdims=True) * first
    return (first_norm > 1e-6) & (np.linalg.norm(second_residual, axis=-1) > 1e-6)


def _physical_metrics(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray, layout: ActionLayout
) -> Mapping[str, float]:
    totals = {
        "position_cm": [0.0, 0.0],
        "rotation_geodesic_deg": [0.0, 0.0],
        "gripper_mm": [0.0, 0.0],
        "force_n": [0.0, 0.0],
        "torque_nm": [0.0, 0.0],
    }

    def add(name: str, values: np.ndarray, valid: np.ndarray) -> None:
        if valid.any():
            totals[name][0] += float(values[valid].sum())
            totals[name][1] += float(valid.sum())

    for role in layout.roles:
        slots = layout.slots(role)
        pos = np.arange(slots.pos.start, slots.pos.stop)
        valid = mask[:, pos].all(axis=-1)
        add("position_cm", np.linalg.norm(pred[:, pos] - target[:, pos], axis=-1) * 100.0, valid)

        rot = np.arange(slots.rot6d.start, slots.rot6d.stop)
        valid = mask[:, rot].all(axis=-1)
        pred_rot = rot6d_to_rotmat(pred[:, rot])
        target_rot = rot6d_to_rotmat(target[:, rot])
        relative = np.einsum("...ij,...kj->...ik", pred_rot, target_rot)
        cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        add("rotation_geodesic_deg", np.degrees(np.arccos(cosine)), valid)

        if slots.gripper is not None:
            add(
                "gripper_mm",
                np.abs(pred[:, slots.gripper] - target[:, slots.gripper]) * GRIPPER_WIDTH_REF_M * 1000.0,
                mask[:, slots.gripper],
            )
        if slots.force is not None:
            force = np.arange(slots.force.start, slots.force.start + 3)
            torque = np.arange(slots.force.start + 3, slots.force.stop)
            add("force_n", np.linalg.norm(pred[:, force] - target[:, force], axis=-1), mask[:, force].all(axis=-1))
            add(
                "torque_nm",
                np.linalg.norm(pred[:, torque] - target[:, torque], axis=-1),
                mask[:, torque].all(axis=-1),
            )
    return MappingProxyType({name: values[0] / values[1] if values[1] else math.nan for name, values in totals.items()})


def build_rollout_result(
    *,
    algorithm: str,
    num_steps: int,
    seed: int,
    ref: SampleRef,
    sample: Mapping[str, torch.Tensor],
    predicted_actions: np.ndarray,
    source: HLSampleSource,
) -> RolloutResult:
    layout = source.layout
    # A WHOLEBODY layout has no pose columns at all (`pose_dims()` is empty), so everything below
    # is undefined for it: `_valid_rot6d` would reshape a (H, 0) slice to (-1, 2, 3) and
    # `delta_ee_to_absolute` would un-relativize nothing against an anchor that does not exist.
    # Refused BY NAME here rather than left to crash inside the rot6d reshape, because the failure
    # would name a rot6d array on a layout that has no rotations — which points at the wrong thing.
    if layout.is_wholebody:
        raise ValueError(
            f"layout {layout.mode!r} predicts 29 JOINT angles, not keypoint poses, so it has no "
            f"position/rotation columns for this builder to un-relativize. Use "
            f"scripts/infer_s1_wholebody.py (flore.s1_policy.inference.wholebody) instead.")
    prediction = np.asarray(predicted_actions, dtype=np.float64)
    target = sample["action_chunk"].detach().cpu().numpy().astype(np.float64)
    mask = sample["action_mask"].detach().cpu().numpy().astype(bool)
    if prediction.shape != target.shape:
        raise ValueError(f"sampled actions have shape {prediction.shape}; expected {target.shape}")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("sampled trajectory contains non-finite values")

    anchor_pos = sample["anchor_pos"].detach().cpu().numpy()
    anchor_rot6d = sample["anchor_rot6d"].detach().cpu().numpy()
    # (1,) for one Bota, (n_ft,) for more -- reshape(-1) covers both without asking which.
    force_valid_flags = np.atleast_1d(
        sample["force_history_valid"].detach().cpu().numpy()).astype(bool).reshape(-1)
    if force_valid_flags.size not in (1, len(layout.force_roles)) and layout.force_roles:
        raise ValueError(
            f"sample carries {force_valid_flags.size} force-window validity flags but the "
            f"{layout.mode} layout declares {len(layout.force_roles)} F/T role(s) "
            f"{list(layout.force_roles)} -- a width mismatch would attribute one hand's "
            f"gating to the other")
    roles: list[RoleTrajectory] = []
    for role_index, role in enumerate(layout.roles):
        slots = layout.slots(role)
        predicted_rotation_valid = _valid_rot6d(prediction[:, slots.rot6d])
        if not predicted_rotation_valid.all():
            bad_steps = np.flatnonzero(~predicted_rotation_valid).tolist()
            raise FloatingPointError(f"sampled {role} rot6d is degenerate at action steps {bad_steps}")
        target_rotation_valid = _valid_rot6d(target[:, slots.rot6d])
        supervised_rotation = mask[:, slots.rot6d].all(axis=-1)
        if np.any(supervised_rotation & ~target_rotation_valid):
            bad_steps = np.flatnonzero(supervised_rotation & ~target_rotation_valid).tolist()
            raise ValueError(f"dataset {role} rot6d is degenerate at valid action steps {bad_steps}")
        pred_pos, pred_rot = delta_ee_to_absolute(
            prediction[:, slots.pos], prediction[:, slots.rot6d], anchor_pos[role_index], anchor_rot6d[role_index]
        )
        target_pos, target_rot = delta_ee_to_absolute(
            target[:, slots.pos], target[:, slots.rot6d], anchor_pos[role_index], anchor_rot6d[role_index]
        )
        position_valid = mask[:, slots.pos]
        rotation_valid = mask[:, slots.rot6d]
        pose_valid = position_valid.all(axis=-1) & rotation_valid.all(axis=-1)

        pred_gripper = target_gripper = gripper_valid = None
        if slots.gripper is not None and role in layout.gripper_roles:
            pred_gripper = prediction[:, slots.gripper] * GRIPPER_WIDTH_REF_M
            target_gripper = target[:, slots.gripper] * GRIPPER_WIDTH_REF_M
            gripper_valid = mask[:, slots.gripper]

        pred_wrench = target_wrench = wrench_valid = None
        if slots.force is not None and role in layout.force_roles:
            pred_wrench = prediction[:, slots.force]
            target_wrench = target[:, slots.force]
            wrench_valid = mask[:, slots.force]

        roles.append(
            RoleTrajectory(
                role=role,
                predicted_position=_readonly(pred_pos, np.float64),
                predicted_rot6d=_readonly(pred_rot, np.float64),
                target_position=_readonly(target_pos, np.float64),
                target_rot6d=_readonly(target_rot, np.float64),
                position_valid=_readonly(position_valid, bool),
                rotation_valid=_readonly(rotation_valid, bool),
                pose_valid=_readonly(pose_valid, bool),
                predicted_gripper_m=None if pred_gripper is None else _readonly(pred_gripper, np.float64),
                target_gripper_m=None if target_gripper is None else _readonly(target_gripper, np.float64),
                gripper_valid=None if gripper_valid is None else _readonly(gripper_valid, bool),
                predicted_wrench=None if pred_wrench is None else _readonly(pred_wrench, np.float64),
                target_wrench=None if target_wrench is None else _readonly(target_wrench, np.float64),
                wrench_valid=None if wrench_valid is None else _readonly(wrench_valid, bool),
            )
        )

    return RolloutResult(
        algorithm=algorithm,
        layout_mode=layout.mode,
        seed=int(seed),
        num_steps=int(num_steps),
        sample=ref,
        action_times_s=_readonly(np.arange(ACTION_HORIZON, dtype=np.float64) * ACTION_DT),
        predicted_actions=_readonly(prediction, np.float64),
        target_actions=_readonly(target, np.float64),
        action_mask=_readonly(mask, bool),
        state_valid=_readonly(sample["state_valid"].detach().cpu().numpy(), bool),
        # `.item()` raised on the dual layout, whose sample carries one flag PER Bota. The
        # scalar becomes the conjunction (pessimistic: "every force input this model reads is
        # good") and the per-sensor flags are kept beside it, so a half-gated dual sample is
        # reportable rather than collapsed into one unattributable False.
        force_history_valid=bool(force_valid_flags.all()),
        force_history_valid_per_sensor=tuple(bool(flag) for flag in force_valid_flags)
        if force_valid_flags.size > 1 else (),
        conditioning_images=conditioning_images(sample, source.image_mean, source.image_std),
        roles=tuple(roles),
        metrics=_physical_metrics(prediction, target, mask, layout),
    )


def result_summary(result: RolloutResult) -> dict[str, Any]:
    """JSON-friendly summary used by both headless CLI smoke tests."""
    return {
        "algorithm": result.algorithm,
        "layout": result.layout_mode,
        "sample_index": result.sample.dataset_index,
        "episode": result.sample.global_episode_id,
        "episode_frame": result.sample.episode_frame,
        "split": result.sample.split,
        "seed": result.seed,
        "num_steps": result.num_steps,
        "shape": list(result.predicted_actions.shape),
        "finite": bool(np.isfinite(result.predicted_actions).all()),
        "state_valid": bool(result.state_valid.all()),
        "force_history_valid": result.force_history_valid,
        "metrics": {key: (float(value) if math.isfinite(value) else None) for key, value in result.metrics.items()},
    }
