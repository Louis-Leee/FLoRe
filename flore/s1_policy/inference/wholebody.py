"""OFFLINE inference for ``diffusion_umi_wholebody``: a 29-joint rollout against a real sample.

The offline half of the wholebody preset, and a SIBLING of
:mod:`flore.s1_policy.inference.diffusion` / ``flow`` / ``act` rather than a flag on them — for
exactly one structural reason, stated as a shape:

    keypoint layout   action = [pos3 rot6d6 grip1 force6] x roles   ->  pose_dims() is 41 wide
    wholebody layout  action = [joints29][grip1 force6][grip1 force6] -> pose_dims() is EMPTY

Everything in :func:`~flore.s1_policy.inference.common.build_rollout_result` is built on that
first line: it calls ``_valid_rot6d(prediction[:, slots.rot6d])`` (a ``(H, 0)`` reshape to
``(-1, 2, 3)``) and ``delta_ee_to_absolute(prediction[:, slots.pos], ...)`` per role. Neither is
meaningful for a layout whose poses are 0-width slices, and ``viewer.py``'s entire vocabulary —
gripper clusters, display frames, blue/orange trajectory lines, episode playback — is
positions and rotations. So this module owns the wholebody result and
:mod:`flore.s1_policy.inference.wholebody_viewer` owns the wholebody picture, and neither existing
file changes.

WHAT IS REUSED
==============
Every loader seam, verbatim: :class:`~flore.s1_policy.inference.common.HLSampleSource` rebuilds
the exact training ``HLDataset`` (it is layout-driven and already wholebody-aware),
``dataclass_config`` / ``validate_model_layout`` / ``load_norm_bundle`` /
``prepare_model_batch`` / ``strict_load_weights`` are the same strict, sha256-checked,
provenance-bound path the three keypoint sessions use. So a wholebody rollout is exactly as
reproducible, and a checkpoint/dataset mismatch is refused by exactly the same code.

THE ONE METRIC TRAP, and why this file states it twice
=====================================================
``mask[..., []].all(axis=-1)`` is ``True`` everywhere and ``np.linalg.norm`` over 0 columns is
exactly ``0.0``. So the un-guarded keypoint arithmetic does not fail on this layout — it reports a
**perfect 0.00 cm position error** for a policy with no position columns at all. That already bit
the training metrics (``training/metrics.physical_trajectory_sums`` now omits the pose keys here
rather than reporting them), and :func:`wholebody_metrics` below is the offline twin: it reports
``joint_deg`` and never a ``position_cm`` it cannot compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from flore.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HORIZON,
    WB_JOINT_NAMES,
    ActionLayout,
)
from flore.s1_policy.data.norm_stats import unnormalize_quantile
from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M
from flore.s1_policy.inference.common import (
    HLSampleSource,
    SampleRef,
    conditioning_images,
    configure_deterministic_inference,
    dataclass_config,
    load_norm_bundle,
    prepare_model_batch,
    read_checkpoint_config,
    resolve_inference_device,
    strict_load_weights,
    validate_model_layout,
    validate_sampler_steps,
)

#: The algorithms a wholebody checkpoint can carry. Same three blocks as every other HL-v2
#: checkpoint — ``wholebody`` is a LAYOUT, not a fourth algorithm — so this session serves all
#: three off one dispatch instead of three near-identical files.
ALGORITHMS = ("diffusion", "flow", "act")


@dataclass(frozen=True)
class WholeBodyRollout:
    """One deterministic offline wholebody rollout against one exact training sample.

    ``predicted_joints`` / ``target_joints`` are ``(H, 29)`` ABSOLUTE radians in
    :data:`~flore.s1_policy.data.layout.WB_JOINT_NAMES` order — and so are
    ``predicted_actions`` / ``target_actions``, whose joint block is the same thing. For an
    ``absolute`` checkpoint (the default) no transform has been applied and none should be: the
    layout declares no pose dims, so there is no anchor to un-relativize against and the numbers
    in the chunk are the numbers the ``dense`` low level would be handed.

    For a ``delta`` checkpoint (:data:`~flore.s1_policy.data.layout.WB_JOINT_TARGETS`) exactly ONE
    transform has been applied by :func:`build_wholebody_result`: the observation's own absolute
    joints, added back to BOTH arrays. That is what keeps this class's contract single-valued —
    every field here is a posture, whatever the checkpoint predicts. Every reported metric is
    bit-identical either way, since a shared offset cancels in every difference.
    """

    algorithm: str
    layout_mode: str
    seed: int
    num_steps: int
    sample: SampleRef
    joint_names: tuple[str, ...]
    action_times_s: np.ndarray               # (H,)
    predicted_joints: np.ndarray             # (H, 29) rad
    target_joints: np.ndarray                # (H, 29) rad
    joint_valid: np.ndarray                  # (H, 29) bool
    #: role -> (H,) metres. Empty when the layout declares no gripper for that role.
    predicted_gripper_m: Mapping[str, np.ndarray]
    target_gripper_m: Mapping[str, np.ndarray]
    gripper_valid: Mapping[str, np.ndarray]
    #: role -> (H, 6) physical N / N·m in the raw bota_sensor frame.
    predicted_wrench: Mapping[str, np.ndarray]
    target_wrench: Mapping[str, np.ndarray]
    wrench_valid: Mapping[str, np.ndarray]
    #: ``(To, 31)`` — ALWAYS 2-D, even at ``To == 1``, where ``HLDataset`` emits a bare
    #: ``(31,)`` (``hl_dataset.py:1516``) because the model accepts both. Normalized in the
    #: builder so a consumer indexing ``state[-1]`` gets the newest FRAME rather than the last
    #: COLUMN of it — that silent 31-vs-1 confusion is exactly what the current v3 grid
    #: (``obs_history_len=1``) would produce on every single sample.
    state: np.ndarray                        # (To, 31) the observed proprio, raw
    state_valid: np.ndarray                  # (To, 31) bool
    force_history: np.ndarray                # (FH, 12) raw sensor-frame
    force_history_valid: bool
    force_history_valid_per_sensor: tuple[bool, ...]
    conditioning_images: tuple[np.ndarray, ...]
    metrics: Mapping[str, float]
    predicted_actions: np.ndarray            # (H, 43) the raw chunk, for the record
    target_actions: np.ndarray               # (H, 43)
    action_mask: np.ndarray                  # (H, 43) bool

    @property
    def horizon(self) -> int:
        return int(self.predicted_joints.shape[0])

    def joint_error_deg(self) -> np.ndarray:
        """``(H, 29)`` absolute per-joint error in DEGREES, NaN where unsupervised.

        NaN rather than 0 for the masked entries: 0 is a legitimate error and the chart must not
        draw an unsupervised tail as a perfect one (which is the same trap
        :func:`wholebody_metrics` guards, one level down)."""
        err = np.degrees(np.abs(self.predicted_joints - self.target_joints))
        return np.where(self.joint_valid, err, np.nan)


def _readonly(array: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def wholebody_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                      layout: ActionLayout) -> Mapping[str, float]:
    """Physical metrics for a wholebody chunk: ``joint_deg``, ``gripper_mm``, force, torque.

    NO ``position_cm`` AND NO ``rotation_geodesic_deg``, and their absence is the point. The
    keypoint version of this function derives them from ``slots.pos``/``slots.rot6d``, which are
    0-width here; ``mask[:, []].all(-1)`` is True and a norm over 0 columns is 0.0, so running it
    unchanged reports a **perfect 0.00 cm** position error for a policy that predicts no
    position. Omitting the key means a consumer that reads it gets a ``KeyError`` naming what it
    asked for, instead of a number that looks like a triumph.

    ``joint_deg`` is the MEAN absolute per-joint error over the 29 joints, in degrees — the same
    definition the trainer logs (``training/metrics.physical_trajectory_sums``), so an offline
    number and a W&B curve are directly comparable.
    """
    totals: dict[str, list[float]] = {
        "joint_deg": [0.0, 0.0],
        "gripper_mm": [0.0, 0.0],
        "force_n": [0.0, 0.0],
        "torque_nm": [0.0, 0.0],
    }

    def add(name: str, values: np.ndarray, valid: np.ndarray) -> None:
        if valid.any():
            totals[name][0] += float(values[valid].sum())
            totals[name][1] += float(valid.sum())

    joint_dims = np.asarray(layout.joint_dims(), dtype=int)
    if joint_dims.size:
        add("joint_deg",
            np.degrees(np.abs(pred[:, joint_dims] - target[:, joint_dims])).mean(axis=-1),
            mask[:, joint_dims].all(axis=-1))
    for role in layout.roles:
        slots = layout.slots(role)
        if slots.gripper is not None:
            add("gripper_mm",
                np.abs(pred[:, slots.gripper] - target[:, slots.gripper])
                * GRIPPER_WIDTH_REF_M * 1000.0,
                mask[:, slots.gripper])
        if slots.force is not None:
            force = np.arange(slots.force.start, slots.force.start + 3)
            torque = np.arange(slots.force.start + 3, slots.force.stop)
            add("force_n", np.linalg.norm(pred[:, force] - target[:, force], axis=-1),
                mask[:, force].all(axis=-1))
            add("torque_nm", np.linalg.norm(pred[:, torque] - target[:, torque], axis=-1),
                mask[:, torque].all(axis=-1))
    return MappingProxyType({
        name: (values[0] / values[1] if values[1] else math.nan)
        for name, values in totals.items()
    })


def build_wholebody_result(
    *,
    algorithm: str,
    num_steps: int,
    seed: int,
    ref: SampleRef,
    sample: Mapping[str, torch.Tensor],
    predicted_actions: np.ndarray,
    source: HLSampleSource,
) -> WholeBodyRollout:
    """Assemble a :class:`WholeBodyRollout`. The wholebody twin of ``build_rollout_result``."""
    layout = source.layout
    if not layout.is_wholebody or layout.joints is None:
        raise ValueError(
            f"build_wholebody_result needs a wholebody layout, got {layout.mode!r}. A keypoint "
            "checkpoint has no joint block; use infer_s1_{diffusion,flow,act}.py.")
    prediction = np.asarray(predicted_actions, dtype=np.float64)
    target = sample["action_chunk"].detach().cpu().numpy().astype(np.float64)
    mask = sample["action_mask"].detach().cpu().numpy().astype(bool)
    if prediction.shape != target.shape:
        raise ValueError(f"sampled actions have shape {prediction.shape}; expected {target.shape}")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("sampled trajectory contains non-finite values")

    jsl = layout.joints
    if getattr(source, "wb_joint_target", "absolute") == "delta":
        # A DELTA checkpoint's joint columns are `q[k] - q[0]`, so drawing them straight would
        # put both G1 bodies at the 29-joints-zero pose plus a few centidegrees of motion -- and
        # this page's whole content is two POSTURES (there are no position columns to draw a
        # trajectory from). Add the observation's own absolute joints back, ONCE, here: every
        # consumer below -- the two bodies, `joint_error_deg`, `wholebody_metrics`, the named
        # worst joint -- then works on absolute radians with no branch of its own.
        #
        # The metrics are unchanged by this, which is the point: both arrays get the SAME
        # offset, so every difference (and therefore every reported number) is identical to
        # scoring the deltas directly. It only fixes what is DRAWN.
        anchor = sample["state"].detach().cpu().numpy().astype(np.float64).reshape(
            -1, layout.state_dim)[-1, layout.joint_state_slot]
        prediction = prediction.copy()
        target = target.copy()
        prediction[:, jsl] += anchor
        target[:, jsl] += anchor
    pred_joints = prediction[:, jsl]
    target_joints = target[:, jsl]
    joint_valid = mask[:, jsl]

    pred_grip: dict[str, np.ndarray] = {}
    tgt_grip: dict[str, np.ndarray] = {}
    grip_valid: dict[str, np.ndarray] = {}
    pred_wrench: dict[str, np.ndarray] = {}
    tgt_wrench: dict[str, np.ndarray] = {}
    wrench_valid: dict[str, np.ndarray] = {}
    for role in layout.roles:
        slots = layout.slots(role)
        if slots.gripper is not None and role in layout.gripper_roles:
            pred_grip[role] = _readonly(prediction[:, slots.gripper] * GRIPPER_WIDTH_REF_M)
            tgt_grip[role] = _readonly(target[:, slots.gripper] * GRIPPER_WIDTH_REF_M)
            grip_valid[role] = _readonly(mask[:, slots.gripper], bool)
        if slots.force is not None and role in layout.force_roles:
            pred_wrench[role] = _readonly(prediction[:, slots.force])
            tgt_wrench[role] = _readonly(target[:, slots.force])
            wrench_valid[role] = _readonly(mask[:, slots.force], bool)

    # (1,) for one Bota, (n_ft,) for two -- reshape(-1) covers both. Same contract as
    # build_rollout_result: the scalar is the conjunction, the per-sensor tuple is the detail.
    force_flags = np.atleast_1d(
        sample["force_history_valid"].detach().cpu().numpy()).astype(bool).reshape(-1)
    if layout.force_roles and force_flags.size not in (1, len(layout.force_roles)):
        raise ValueError(
            f"sample carries {force_flags.size} force-window validity flags but the "
            f"{layout.mode} layout declares {len(layout.force_roles)} F/T role(s) — a width "
            "mismatch would attribute one hand's gating to the other")

    return WholeBodyRollout(
        algorithm=str(algorithm),
        layout_mode=layout.mode,
        seed=int(seed),
        num_steps=int(num_steps),
        sample=ref,
        joint_names=tuple(WB_JOINT_NAMES),
        action_times_s=_readonly(np.arange(ACTION_HORIZON, dtype=np.float64) * ACTION_DT),
        predicted_joints=_readonly(pred_joints, np.float64),
        target_joints=_readonly(target_joints, np.float64),
        joint_valid=_readonly(joint_valid, bool),
        predicted_gripper_m=MappingProxyType(pred_grip),
        target_gripper_m=MappingProxyType(tgt_grip),
        gripper_valid=MappingProxyType(grip_valid),
        predicted_wrench=MappingProxyType(pred_wrench),
        target_wrench=MappingProxyType(tgt_wrench),
        wrench_valid=MappingProxyType(wrench_valid),
        # reshape(-1, state_dim), not atleast_2d: at To == 1 the dataset hands back a bare
        # (state_dim,) and atleast_2d would give (1, state_dim) too — but only by luck of which
        # axis it prepends. Naming the width makes the intent "To frames of state_dim" explicit
        # and turns any other shape into a loud reshape error.
        state=_readonly(sample["state"].detach().cpu().numpy().reshape(-1, layout.state_dim),
                        np.float64),
        state_valid=_readonly(
            sample["state_valid"].detach().cpu().numpy().reshape(-1, layout.state_dim), bool),
        force_history=_readonly(sample["force_history"].detach().cpu().numpy(), np.float64),
        force_history_valid=bool(force_flags.all()),
        force_history_valid_per_sensor=(tuple(bool(f) for f in force_flags)
                                        if force_flags.size > 1 else ()),
        conditioning_images=conditioning_images(sample, source.image_mean, source.image_std),
        metrics=wholebody_metrics(prediction, target, mask, layout),
        predicted_actions=_readonly(prediction, np.float64),
        target_actions=_readonly(target, np.float64),
        action_mask=_readonly(mask, bool),
    )


def wholebody_summary(result: WholeBodyRollout) -> dict[str, Any]:
    """JSON-friendly summary for ``--self-test``. Mirrors ``common.result_summary``.

    ``worst_joint`` is named rather than indexed: 29 columns is enough that "index 19 is bad"
    costs a lookup, and the whole reason the joint names are pinned across three places is so a
    diagnosis can say ``left_wrist_roll_joint``.
    """
    err = result.joint_error_deg()
    per_joint = np.nanmean(err, axis=0) if np.isfinite(err).any() else np.full(err.shape[1], np.nan)
    worst = int(np.nanargmax(per_joint)) if np.isfinite(per_joint).any() else -1
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
        "supervised_steps": int(result.joint_valid.all(axis=-1).sum()),
        "state_valid": bool(result.state_valid.all()),
        "force_history_valid": result.force_history_valid,
        "worst_joint": None if worst < 0 else result.joint_names[worst],
        "worst_joint_deg": None if worst < 0 else float(per_joint[worst]),
        "metrics": {k: (float(v) if math.isfinite(v) else None)
                    for k, v in result.metrics.items()},
    }


def _build(algorithm: str, config_block: Mapping[str, Any]):
    """``(config, model)`` for one algorithm block. Explicit, so an unknown algo cannot fall
    through into another algorithm's config class (see ``hl_online._algo_config_key`` for the
    binary-collapse bug that motivates spelling all three)."""
    if algorithm == "diffusion":
        from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel

        config = dataclass_config(DPDiffusionConfig, config_block)
        return config, DPDiffusionModel(config)
    if algorithm == "flow":
        from flore.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

        config = dataclass_config(DPFlowConfig, config_block)
        return config, DPFlowModel(config)
    if algorithm == "act":
        from flore.s1_policy.model.dp_act import DPACTConfig, DPACTModel

        config = dataclass_config(DPACTConfig, config_block)
        return config, DPACTModel(config)
    raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}")


def _algo_block(algorithm: str) -> str:
    from flore.s1_policy.training.config import ALGO_CONFIG_FIELD

    return ALGO_CONFIG_FIELD[algorithm]


class WholeBodyInferenceSession:
    """A strict, deterministic offline inference session for one ``wholebody`` checkpoint.

    Serves all three algorithm blocks: the layout is what makes this a separate session, and the
    algorithm is a field in the checkpoint. ``algorithm=None`` reads it from ``config.json``,
    which is what every launcher does — the operator thinks in presets, not blocks.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        algorithm: str | None = None,
        zarr_paths: tuple[str, ...] | None = None,
        device: str = "cuda",
        num_steps: int | None = None,
    ) -> None:
        import json

        resolved_algo = algorithm
        if resolved_algo is None:
            try:
                resolved_algo = str(json.loads(
                    (Path(checkpoint_dir) / "config.json").read_text()).get("algo"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"could not read algo from {checkpoint_dir}/config.json: {exc}") from exc
        if resolved_algo not in ALGORITHMS:
            raise ValueError(
                f"checkpoint algo={resolved_algo!r} is not one of {ALGORITHMS}. muse is not "
                "served here: its backend runs in another interpreter (see "
                "docs/RUNBOOK_muse_hl_serving.md).")
        checkpoint, raw = read_checkpoint_config(checkpoint_dir, resolved_algo)
        source = HLSampleSource(raw["hl_data"], zarr_paths=zarr_paths)
        try:
            if not source.layout.is_wholebody:
                raise ValueError(
                    f"checkpoint layout is {source.layout.mode!r}, not a wholebody one. This "
                    f"session predicts 29 JOINT angles; a keypoint checkpoint predicts poses — "
                    f"view it with scripts/infer_s1_{resolved_algo}.py instead.")
            config_values = raw.get(_algo_block(resolved_algo))
            if not isinstance(config_values, dict):
                raise ValueError(
                    f"{resolved_algo} checkpoint is missing its {_algo_block(resolved_algo)!r} "
                    "config block")
            config, model = _build(resolved_algo, config_values)
            validate_model_layout(config, raw["hl_data"], source.layout)
            norm_stats = load_norm_bundle(raw, checkpoint, source)
            if getattr(config, "clip_sample", False) and "actions" not in norm_stats:
                raise ValueError("clip_sample=True requires bundled action normalization")

            resolved_device = resolve_inference_device(device)
            configure_deterministic_inference(resolved_device)
            strict_load_weights(model, checkpoint, raw["_provenance"]["model_sha256"])
            model.to(resolved_device).eval()
        except Exception:
            source.close()
            raise

        self.algorithm = resolved_algo
        self.checkpoint = checkpoint
        self.raw_config = raw
        self.config = config
        self.source = source
        self.norm_stats = norm_stats
        self.device = resolved_device
        self.model = model
        self._closed = False
        # Same per-algorithm ceilings the three keypoint sessions declare: DDIM cannot exceed its
        # training timesteps, ACT has no sampler loop at all, flow is unbounded.
        if resolved_algo == "diffusion":
            self.max_num_steps: int | None = int(config.num_train_timesteps)
            default_steps = int(config.num_inference_steps)
        elif resolved_algo == "act":
            self.max_num_steps = 1
            default_steps = 1
        else:
            self.max_num_steps = None
            default_steps = int(config.inference_steps)
        try:
            self.num_steps = validate_sampler_steps(
                default_steps if num_steps is None else num_steps, self.max_num_steps)
        except ValueError:
            self.close()
            raise

    def infer(self, sample_index: int, *, seed: int = 0,
              num_steps: int | None = None) -> WholeBodyRollout:
        if self._closed:
            raise RuntimeError("WholeBodyInferenceSession is closed")
        steps = (self.num_steps if num_steps is None
                 else validate_sampler_steps(num_steps, self.max_num_steps))
        ref, sample = self.source.get(sample_index)
        batch = prepare_model_batch(sample, self.norm_stats, self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.model.reset()
        with torch.inference_mode():
            sampled = self.model.sample_actions(batch, num_steps=steps, generator=generator)
        prediction = sampled[0].detach().float().cpu().numpy()
        if "actions" in self.norm_stats:
            prediction = unnormalize_quantile(prediction, self.norm_stats["actions"])
        return build_wholebody_result(
            algorithm=self.algorithm,
            num_steps=steps,
            seed=seed,
            ref=ref,
            sample=sample,
            predicted_actions=np.asarray(prediction),
            source=self.source,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.source.close()
        if self.device.type == "cuda":
            self.model.to("cpu")
            scheduler = getattr(self.model, "noise_scheduler", None)
            if scheduler is not None:
                for name, value in vars(scheduler).items():
                    if isinstance(value, torch.Tensor):
                        setattr(scheduler, name, value.cpu())

    def __enter__(self) -> "WholeBodyInferenceSession":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


__all__ = [
    "ALGORITHMS",
    "WholeBodyInferenceSession",
    "WholeBodyRollout",
    "build_wholebody_result",
    "wholebody_metrics",
    "wholebody_summary",
]
