"""Utilities for loading policy and motion assets from Weights & Biases."""

from __future__ import annotations

import copy
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import onnx


@dataclass
class PtOnnxExportContext:
    """Reusable mjlab export state for repeated PT->ONNX conversion."""

    env: Any
    runner: Any
    joint_names: list[str]
    default_joint_pos: list[float]
    encoder_bias: list[float]

    def close(self) -> None:
        self.env.close()


def resolve_wandb_run_path(
    *,
    wandb_run_path: str | None = None,
    run_id: str | None = None,
    entity: str | None = None,
    project: str | None = None,
) -> str:
    """Resolve either a fully qualified run path or ``run_id`` shorthand."""
    if wandb_run_path:
        return wandb_run_path
    if run_id and entity and project:
        return f"{entity}/{project}/{run_id}"
    raise ValueError(
        "Provide either wandb_run_path='entity/project/run_id' or "
        "run_id together with entity and project."
    )


def resolve_wandb_artifact_path(
    wandb_artifact_path: str,
) -> tuple[str, str, str]:
    """Resolve a W&B artifact reference to ``(artifact_name, type, file_path)``.

    Accepts either:
    - a fully-qualified artifact name like ``entity/project/name:v0``
    - a W&B artifact URL like
      ``https://wandb.ai/entity/project/artifacts/motions/name/v0/files/motion.npz``
    """
    parsed = urlparse(wandb_artifact_path)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.strip("/")
        match = re.match(
            r"^(?P<entity>[^/]+)/(?P<project>[^/]+)/artifacts/"
            r"(?P<artifact_type>[^/]+)/(?P<name>[^/]+)/(?P<version>[^/]+)"
            r"(?:/files/(?P<file_path>.+))?$",
            path,
        )
        if match is None:
            raise ValueError(
                "Unsupported W&B artifact URL format. "
                "Expected something like "
                "'https://wandb.ai/entity/project/artifacts/motions/name/v0/files/motion.npz'."
            )
        artifact_name = (
            f"{match.group('entity')}/{match.group('project')}/"
            f"{match.group('name')}:{match.group('version')}"
        )
        artifact_type = match.group("artifact_type")
        file_path = match.group("file_path") or "motion.npz"
        return artifact_name, artifact_type, file_path

    return wandb_artifact_path, "motions", "motion.npz"


def _extract_required_capacity(message: str, name: str) -> int | None:
    match = re.search(rf"{name} overflow \({name} must be >= (\d+)\)", message)
    if match is None:
        return None
    return int(match.group(1))


def _next_capacity(required: int) -> int:
    slack = max(32, required // 8)
    return required + slack


def create_pt_onnx_export_context(
    task_id: str, *, env_cfg: Any | None = None
) -> PtOnnxExportContext:
    """Create a reusable mjlab export context for PT->ONNX conversion.

    Builds the mjlab environment and runner once so that multiple
    checkpoints can be loaded and exported without repeated startup cost.
    Extracts core policy metadata (joint names, default joint positions,
    encoder bias) from the action manager.
    """
    try:
        import mjlab.tasks  # noqa: F401 — populates the task registry
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    except ImportError as e:
        raise ImportError(
            "mjlab and torch are required for only_latest=False. "
            "Install them with: pip install mjlab torch"
        ) from e

    env_cfg = (
        copy.deepcopy(env_cfg)
        if env_cfg is not None
        else load_env_cfg(task_id, play=True)
    )
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task_id)

    while True:
        try:
            env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
            break
        except ValueError as exc:
            message = str(exc)
            required_nconmax = _extract_required_capacity(message, "nconmax")
            required_njmax = _extract_required_capacity(message, "njmax")
            if required_nconmax is None and required_njmax is None:
                raise
            if required_nconmax is not None:
                env_cfg.sim.nconmax = _next_capacity(required_nconmax)
            if required_njmax is not None:
                env_cfg.sim.njmax = _next_capacity(required_njmax)

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cpu")
    except Exception:
        wrapped_env.close()
        raise

    # Extract core policy metadata from the action manager.
    joint_names: list[str] = []
    default_joint_pos: list[float] = []
    encoder_bias: list[float] = []

    inner_env = wrapped_env.env if hasattr(wrapped_env, "env") else wrapped_env

    # Joint names, default positions, and encoder bias from the action manager.
    action_mgr = getattr(inner_env, "action_manager", None)
    if action_mgr is not None:
        for term_name in action_mgr.active_terms:
            term = action_mgr.get_term(term_name)
            if not hasattr(term, "target_names"):
                continue
            entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
            prefix = f"{entity_name}/" if entity_name else ""
            joint_names = [f"{prefix}{n}" for n in term.target_names]
            if hasattr(term, "offset") and term.offset is not None:
                offset = term.offset
                if hasattr(offset, "tolist"):
                    flat = offset.flatten().tolist()
                elif hasattr(offset, "__iter__"):
                    flat = list(offset)
                else:
                    flat = [float(offset)] * len(joint_names)
                default_joint_pos = flat[: len(joint_names)]
            if entity_name:
                entity = inner_env.scene[entity_name]
                bias = entity.data.encoder_bias
                if hasattr(bias, "detach"):
                    bias = bias.detach()
                if hasattr(bias, "cpu"):
                    bias = bias.cpu()
                if hasattr(term.target_ids, "detach"):
                    target_ids = term.target_ids.detach()
                else:
                    target_ids = term.target_ids
                if hasattr(target_ids, "cpu"):
                    target_ids = target_ids.cpu()
                if hasattr(target_ids, "tolist"):
                    target_indices = target_ids.tolist()
                else:
                    target_indices = list(target_ids)
                if hasattr(bias, "__getitem__"):
                    selected_bias = bias[0, target_indices]
                    if hasattr(selected_bias, "tolist"):
                        encoder_bias = selected_bias.tolist()
                    else:
                        encoder_bias = list(selected_bias)
            break

    return PtOnnxExportContext(
        env=wrapped_env,
        runner=runner,
        joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        encoder_bias=encoder_bias,
    )


def fetch_onnx_from_wandb_run(run_path: str) -> tuple[str, onnx.ModelProto]:
    """Download the latest ONNX policy file from a W&B run.

    Finds the most recently updated ``.onnx`` file attached to the run and
    loads it into memory as an :class:`onnx.ModelProto`.  The policy name is
    the filename with its extension removed (e.g. ``"2026-02-25_04-30-08.onnx"``
    becomes ``"2026-02-25_04-30-08"``).

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.

    Returns:
        A ``(policy_name, onnx_model)`` tuple for the latest ``.onnx`` file.

    Raises:
        ValueError: If no ``.onnx`` files are found in the run.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]

    if not onnx_files:
        raise ValueError(f"No .onnx files found in W&B run: {run_path}")

    latest = max(onnx_files, key=lambda f: f.updated_at)

    with tempfile.TemporaryDirectory() as tmp_dir:
        latest.download(root=tmp_dir, replace=True)
        local_path = Path(tmp_dir) / latest.name
        name = local_path.stem
        model = onnx.load(str(local_path))

    return name, model


def fetch_motion_npz_from_wandb_run(run_path: str) -> tuple[str, bytes]:
    """Download the ``motion.npz`` artifact used or logged by a W&B run."""
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
    if artifact is None:
        raise ValueError(f"No motion artifact found in W&B run: {run_path}")

    artifact_name = artifact.name.split("/")[-1]
    motion_name = artifact_name.split(":", 1)[0] or "motion"

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(artifact.download(root=tmp_dir))
        motion_path = root / "motion.npz"
        if not motion_path.exists():
            raise ValueError(
                f"Motion artifact for run '{run_path}' did not contain motion.npz"
            )
        return motion_name, motion_path.read_bytes()


def fetch_motion_npz_from_wandb_artifact(
    wandb_artifact_path: str,
) -> tuple[str, bytes]:
    """Download ``motion.npz`` directly from a W&B motion artifact."""
    import wandb

    artifact_name, artifact_type, file_path = resolve_wandb_artifact_path(
        wandb_artifact_path
    )

    api = wandb.Api()
    artifact = api.artifact(artifact_name, type=artifact_type)
    motion_name = artifact_name.split("/")[-1].split(":", 1)[0] or "motion"

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(artifact.download(root=tmp_dir))
        motion_path = root / file_path
        if not motion_path.exists():
            raise ValueError(
                f"Motion artifact '{artifact_name}' did not contain '{file_path}'"
            )
        return motion_name, motion_path.read_bytes()


def fetch_pt_onnx_from_wandb_run(
    run_path: str,
    task_id: str,
    export_context: PtOnnxExportContext,
) -> list[tuple[str, onnx.ModelProto]]:
    """Download all ``model_*.pt`` checkpoints from a W&B run and convert each to ONNX.

    Checkpoints are sorted by training step before conversion, so the returned
    list is ordered from earliest to latest (e.g. ``model_0``, ``model_50``, …).

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.
        task_id: mjlab task identifier (e.g. ``"go2_flat"``).
        export_context: Pre-built export context. Reused across all checkpoints
            and **not** closed by this function.

    Returns:
        List of ``(policy_name, onnx_model)`` tuples sorted by training step.

    Raises:
        ImportError: If ``mjlab`` or ``torch`` are not installed.
        ValueError: If no ``model_*.pt`` files are found in the run.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    pt_files = [f for f in run.files() if re.match(r"^model_\d+\.pt$", f.name)]

    if not pt_files:
        raise ValueError(f"No model_*.pt files found in W&B run: {run_path}")

    # Sort by training step number (model_0.pt < model_50.pt < model_100.pt, …)
    pt_files.sort(key=lambda f: int(re.search(r"\d+", f.name).group()))  # type: ignore[union-attr]

    results: list[tuple[str, onnx.ModelProto]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for wandb_file in pt_files:
            wandb_file.download(root=tmp_dir, replace=True)
            pt_path = tmp_path / wandb_file.name
            name = pt_path.stem  # e.g. "model_0", "model_50"
            onnx_filename = f"{name}.onnx"

            export_context.runner.load(
                str(pt_path),
                load_cfg={"actor": True},
                strict=True,
                map_location="cpu",
            )
            export_context.runner.export_policy_to_onnx(tmp_dir, onnx_filename)

            model = onnx.load(str(tmp_path / onnx_filename))
            results.append((name, model))

    return results


__all__ = [
    "fetch_motion_npz_from_wandb_artifact",
    "fetch_motion_npz_from_wandb_run",
    "fetch_onnx_from_wandb_run",
    "resolve_wandb_artifact_path",
    "resolve_wandb_run_path",
]
