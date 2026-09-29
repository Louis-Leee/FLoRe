"""ONNX export utilities for WBT policies.

Ported from TML-BeyondMimic-Generalist ``whole_body_tracking.utils.exporter``
to produce ONNX files compatible with ``wbt_wrappers_inference``:

* **Generalist export** (``export_generalist_policy_as_onnx``) -- obs-only
  input, actions-only output.  Used by ``TMLWholeBodyTrackingPolicy``.
* **Motion-embedded export** (``export_motion_policy_as_onnx``) -- obs +
  time_step inputs, actions + full motion outputs.  Used by
  ``BMWholeBodyTrackingPolicy``.
* **Metadata attachment** (``attach_onnx_metadata``) -- writes
  ``default_joint_pos``, ``action_scale``, ``joint_names``,
  ``anchor_body_name``, ``body_names``, etc. as ONNX custom metadata
  strings consumable by both inference wrappers.
"""

from __future__ import annotations

import os
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_to_csv(arr: list, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr)


# ---------------------------------------------------------------------------
# Generalist exporter (obs -> actions)
# ---------------------------------------------------------------------------


class _OnnxGeneralistPolicyExporter(torch.nn.Module):
    """Export a generalist policy: ``obs`` in, ``actions`` out.

    This matches the ONNX format expected by
    ``wbt_wrappers_inference.policies.tml_wbt.TMLWholeBodyTrackingPolicy``.
    """

    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = actor_critic.actor
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs, time_step):
        return (self.actor(self.normalizer(obs)),)

    def export(self, path: str, filename: str = "policy.onnx"):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=["actions"],
            dynamic_axes={},
        )


# ---------------------------------------------------------------------------
# Motion-embedded exporter (obs + time_step -> actions + motion data)
# ---------------------------------------------------------------------------


class _OnnxMotionPolicyExporter(torch.nn.Module):
    """Export a motion-embedded policy with reference tensors baked in.

    This matches the ONNX format expected by
    ``wbt_wrappers_inference.policies.bm_wbt.BMWholeBodyTrackingPolicy``.
    """

    def __init__(self, env, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = actor_critic.actor
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = torch.nn.Identity()

        cmd = env.command_manager.get_term("motion")
        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, obs, time_step):
        ts = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(obs)),
            self.joint_pos[ts],
            self.joint_vel[ts],
            self.body_pos_w[ts],
            self.body_quat_w[ts],
            self.body_lin_vel_w[ts],
            self.body_ang_vel_w[ts],
        )

    def export(self, path: str, filename: str = "policy.onnx"):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


# ---------------------------------------------------------------------------
# Public export helpers
# ---------------------------------------------------------------------------


def export_generalist_policy_as_onnx(
    actor_critic, *, path: str, normalizer=None, filename: str = "policy.onnx", verbose: bool = False
) -> str:
    """Export a generalist (obs-only) ONNX policy for TML inference.

    Returns the full path to the exported file.
    """
    os.makedirs(path, exist_ok=True)
    exporter = _OnnxGeneralistPolicyExporter(actor_critic, normalizer, verbose)
    exporter.export(path, filename)
    return os.path.join(path, filename)


def export_motion_policy_as_onnx(
    env, actor_critic, *, path: str, normalizer=None, filename: str = "policy.onnx", verbose: bool = False
) -> str:
    """Export a motion-embedded ONNX policy for BM inference.

    ``env`` must be a live ``ManagerBasedRLEnv`` with a ``motion`` command
    term so that reference tensors can be baked into the model.

    Returns the full path to the exported file.
    """
    os.makedirs(path, exist_ok=True)
    exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    exporter.export(path, filename)
    return os.path.join(path, filename)


# ---------------------------------------------------------------------------
# Metadata attachment
# ---------------------------------------------------------------------------


def build_metadata_from_env(env, run_path: str = "") -> dict[str, Any]:
    """Extract ONNX metadata fields from a live Isaac Lab environment.

    The returned dict can be passed to :func:`attach_onnx_metadata`.
    """
    robot_data = env.scene["robot"].data
    motion_cmd = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")

    return {
        "run_path": run_path,
        "joint_names": robot_data.joint_names,
        "joint_stiffness": robot_data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot_data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": robot_data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": action_term._scale[0].cpu().tolist(),
        "anchor_body_name": motion_cmd.cfg.anchor_body_name,
        "body_names": motion_cmd.cfg.body_names,
    }


def attach_onnx_metadata(onnx_path: str, metadata: dict[str, Any]) -> None:
    """Write key-value metadata into an existing ONNX file.

    Each value is stored as a comma-separated string (for lists) or a plain
    string, matching the format consumed by both
    ``TMLWholeBodyTrackingPolicy`` and ``BMWholeBodyTrackingPolicy``.
    """
    import onnx

    model = onnx.load(onnx_path)
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = _list_to_csv(value) if isinstance(value, list) else str(value)
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)
