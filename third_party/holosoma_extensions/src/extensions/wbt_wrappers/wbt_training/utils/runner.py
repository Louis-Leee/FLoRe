"""Custom RSL-RL runner that exports ONNX with metadata on every checkpoint save.

Ported from TML-BeyondMimic-Generalist ``MotionOnPolicyRunner``.  Extends the
standard ``OnPolicyRunner`` to automatically:

1. Export a generalist ONNX policy (for ``TMLWholeBodyTrackingPolicy``).
2. Attach deployment metadata (``default_joint_pos``, ``action_scale``, ...).
3. Optionally upload the artifact to W&B.

Usage with the holosoma training loop is **optional** -- the runner is
provided for workflows that call RSL-RL directly rather than through
``holosoma.train_agent.train()``.
"""

from __future__ import annotations

import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from wbt_training.utils.exporter import (
    attach_onnx_metadata,
    build_metadata_from_env,
    export_generalist_policy_as_onnx,
    export_motion_policy_as_onnx,
)


class WBTOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner that exports TML-compatible ONNX on each save.

    Parameters
    ----------
    env : VecEnv
        Vectorised environment (RSL-RL wrapper around Isaac Lab).
    train_cfg : dict
        RSL-RL training configuration dictionary.
    log_dir : str | None
        Directory for logs and checkpoints.
    device : str
        Torch device string.
    export_mode : str
        ``"generalist"`` for TML (obs-only) or ``"motion"`` for BM
        (motion-embedded).  Defaults to ``"generalist"``.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        export_mode: str = "generalist",
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.export_mode = export_mode

    def save(self, path: str, infos=None):
        """Save checkpoint and export ONNX with deployment metadata."""
        super().save(path, infos)

        # Derive export directory and filename from the checkpoint path.
        # RSL-RL passes ``path`` as e.g. ``<log_dir>/model_1000.pt``.
        policy_dir = path.rsplit("model", 1)[0]
        onnx_filename = os.path.basename(policy_dir.rstrip("/")) + ".onnx"

        if self.export_mode == "motion":
            onnx_path = export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                path=policy_dir,
                normalizer=self.obs_normalizer,
                filename=onnx_filename,
            )
        else:
            onnx_path = export_generalist_policy_as_onnx(
                self.alg.policy, path=policy_dir, normalizer=self.obs_normalizer, filename=onnx_filename
            )

        run_name = ""
        try:
            import wandb

            if wandb.run is not None:
                run_name = wandb.run.name or ""
        except ImportError:
            pass

        metadata = build_metadata_from_env(self.env.unwrapped, run_path=run_name)
        attach_onnx_metadata(onnx_path, metadata)

        # Upload to W&B if logging is active.
        if self.logger_type in ("wandb",) and not getattr(self, "disable_logs", False):
            try:
                import wandb

                wandb.save(onnx_path, base_path=os.path.dirname(policy_dir))
            except Exception:
                pass
