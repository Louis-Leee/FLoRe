"""Train G1 29-DOF whole-body tracking agents.

Usage::

    python -m wbt_training.train_agent \\
        exp:g1-wbt \\
        simulator:isaacsim \\
        --training.headless True

Set ``LOGURU_LEVEL=INFO`` to restore verbose logging (default is WARNING to
reduce simulator log noise).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time

# Suppress verbose simulator / library logs so training output is readable.
if "LOGURU_LEVEL" not in os.environ:
    os.environ["LOGURU_LEVEL"] = "WARNING"


# ---------------------------------------------------------------------------
# S3 checkpoint sync — runs inside the training process for reliability.
# Activated when CKPT_S3_DIR env var is set.
# ---------------------------------------------------------------------------
_s3_sync: "S3SyncManager | None" = None


class S3SyncManager:
    """Sync training logs/checkpoints to S3 after each save and periodically.

    Verifies S3 accessibility on init. If the bucket is unreachable (e.g. no
    credentials, no network), sync calls become no-ops with a warning.
    """

    def __init__(self, local_dir: str, s3_dir: str, interval_sec: int = 600):
        self.local_dir = local_dir
        self.s3_dir = s3_dir
        self.interval_sec = interval_sec
        self._last_sync_time = 0.0
        self._proc: subprocess.Popen | None = None
        self._disabled = False
        self._consecutive_failures = 0

        # Verify aws CLI exists and S3 path is accessible
        if not self._check_s3_access():
            print(
                "[S3-SYNC] WARNING: S3 not accessible, sync disabled. Will retry on each checkpoint save.",
                file=sys.stderr,
            )
            self._disabled = True
        else:
            print(f"[S3-SYNC] Init: {local_dir} -> {s3_dir} (interval {interval_sec}s)", file=sys.stderr)

    def _check_s3_access(self) -> bool:
        """Verify aws CLI works and the S3 bucket/prefix is reachable."""
        # Extract bucket from s3://bucket/prefix/...
        parts = self.s3_dir.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", f"s3://{bucket}/", "--max-items", "1"], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return True
            print(f"[S3-SYNC] S3 access check failed: {result.stderr[:200]}", file=sys.stderr)
            return False
        except FileNotFoundError:
            print("[S3-SYNC] aws CLI not found", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print("[S3-SYNC] S3 access check timed out", file=sys.stderr)
            return False

    def maybe_sync(self, force: bool = False) -> None:
        # Re-check access if previously disabled (credentials may have been mounted late)
        if self._disabled:
            if force and self._check_s3_access():
                print("[S3-SYNC] S3 now accessible, re-enabling sync", file=sys.stderr)
                self._disabled = False
            else:
                return

        now = time.time()
        if not force and (now - self._last_sync_time) < self.interval_sec:
            return
        # Wait for previous sync if forcing, otherwise skip if still running
        if self._proc is not None and self._proc.poll() is None:
            if force:
                self._proc.wait()
            else:
                return
        if self._proc is not None and self._proc.returncode != 0:
            self._consecutive_failures += 1
            print(
                f"[S3-SYNC] Previous sync failed (#{self._consecutive_failures}, rc={self._proc.returncode})",
                file=sys.stderr,
            )
            if self._consecutive_failures >= 5:
                print("[S3-SYNC] Too many failures, re-checking S3 access...", file=sys.stderr)
                if not self._check_s3_access():
                    print("[S3-SYNC] S3 unreachable, disabling sync", file=sys.stderr)
                    self._disabled = True
                    return
                self._consecutive_failures = 0
        else:
            self._consecutive_failures = 0

        self._last_sync_time = now
        cmd = ["aws", "s3", "sync", f"{self.local_dir}/", f"{self.s3_dir}/", "--exclude", "*.lock", "--quiet"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[S3-SYNC] {time.strftime('%H:%M:%S')}: syncing to {self.s3_dir}", file=sys.stderr)
        except FileNotFoundError:
            print("[S3-SYNC] aws CLI not found, disabling sync", file=sys.stderr)
            self._disabled = True

    def final_sync(self) -> None:
        if self._disabled:
            if not self._check_s3_access():
                print("[S3-SYNC] S3 still not accessible, skipping final sync", file=sys.stderr)
                return
            self._disabled = False
        if self._proc is not None:
            self._proc.wait()
        cmd = ["aws", "s3", "sync", f"{self.local_dir}/", f"{self.s3_dir}/", "--exclude", "*.lock", "--quiet"]
        print(f"[S3-SYNC] {time.strftime('%H:%M:%S')}: final sync...", file=sys.stderr)
        try:
            result = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
            if result.returncode == 0:
                print("[S3-SYNC] Final sync complete", file=sys.stderr)
            else:
                print(f"[S3-SYNC] Final sync error: {result.stderr[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"[S3-SYNC] Final sync failed: {e}", file=sys.stderr)


def _patch_logging_helper() -> None:
    """Replace the rich Panel logging with a compact tqdm-compatible output.

    Core holosoma's ``LoggingHelper.post_epoch_logging`` prints a full
    ``rich.Live(Panel(...))`` every iteration, which floods the terminal and
    breaks tqdm progress bars (used by FastSAC).  This patch keeps
    tensorboard / wandb logging intact but replaces the console output with
    a single-line status update.
    """
    from holosoma.agents.modules.logging_utils import LoggingHelper

    _original = LoggingHelper.post_epoch_logging

    def _compact_post_epoch_logging(
        self,
        it: int,
        loss_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        # --- bookkeeping (same as original) ---
        self.tot_timesteps += self.num_steps_per_env * self.num_envs * self.num_gpus
        self.tot_time += self.collection_time + self.learn_time
        _iteration_time = self.collection_time + self.learn_time  # noqa: F841

        ep_string, ep_scalars_to_log = self._log_episode_info()
        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        fps = int(
            self.num_steps_per_env * self.num_envs * self.num_gpus / (self.collection_time + self.learn_time + 1e-8)
        )

        # --- tensorboard + wandb (same as original) ---
        self._logging_to_writer(
            it=it,
            loss_dict=loss_dict,
            extra_log_dicts=extra_log_dicts,
            env_log_dict=env_log_dict,
            fps=fps,
            ep_scalars_to_log=ep_scalars_to_log,
        )

        # --- compact console line instead of rich Panel ---
        if self.is_main_process:
            parts = [f"iter {it}/{self.num_learning_iterations}"]
            if len(self.rewbuffer) > 0:
                parts.append(f"rew {statistics.mean(self.rewbuffer):.2f}")
            if len(self.lenbuffer) > 0:
                parts.append(f"ep_len {statistics.mean(self.lenbuffer):.1f}")
            for k, v in loss_dict.items():
                parts.append(f"{k} {v:.4f}")
            eta = self.tot_time / (it + 1) * (self.num_learning_iterations - it)
            parts.append(f"fps {fps}")
            parts.append(f"ETA {eta:.0f}s")

            try:
                import tqdm

                tqdm.tqdm.write(" | ".join(parts))
            except ImportError:
                print(" | ".join(parts), file=sys.stderr)

        # --- cleanup (same as original) ---
        self.ep_infos.clear()
        self.raw_ep_infos.clear()
        self.learn_time = 0.0
        self.collection_time = 0.0

    LoggingHelper.post_epoch_logging = _compact_post_epoch_logging


_patch_logging_helper()


def _attach_metadata_to_onnx(env, onnx_path: str) -> None:
    """Attach TML-compatible metadata to an exported ONNX checkpoint."""
    try:
        from wbt_training.utils.exporter import attach_onnx_metadata, build_metadata_from_env

        metadata = build_metadata_from_env(env)
        attach_onnx_metadata(onnx_path, metadata)
    except Exception as e:
        print(f"[WARN] Failed to attach ONNX metadata to {onnx_path}: {e}", file=sys.stderr)


def _attach_normalizer_to_onnx(normalizer, onnx_path: str) -> None:
    """Bake normalizer mean/std into ONNX metadata so inference needs only the .onnx file.

    The normalizer computation is already in the ONNX graph (via ActorWrapper).
    This stores the raw stats as metadata so inference can do pre-clipping
    without loading the .pt checkpoint separately.
    """
    try:
        import onnx

        model = onnx.load(onnx_path)
        # Support both EmpiricalNormalization (_mean/_std) and RunningMeanStd (running_mean/running_var)
        if hasattr(normalizer, "_mean"):
            mean = normalizer._mean.detach().cpu().numpy().flatten()
            std = normalizer._std.detach().cpu().numpy().flatten()
        elif hasattr(normalizer, "running_mean"):
            import numpy as np

            mean = normalizer.running_mean.detach().cpu().numpy().flatten()
            var = normalizer.running_var.detach().cpu().numpy().flatten()
            std = np.sqrt(var + 1e-5)
        else:
            print(f"[WARN] Unknown normalizer type {type(normalizer)}, skipping ONNX embed", file=sys.stderr)
            return

        for key, arr in [("normalizer_mean", mean), ("normalizer_std", std)]:
            entry = onnx.StringStringEntryProto()
            entry.key = key
            entry.value = ",".join(f"{x:.8f}" for x in arr)
            model.metadata_props.append(entry)
        onnx.save(model, onnx_path)
    except Exception as e:
        print(f"[WARN] Failed to attach normalizer to {onnx_path}: {e}", file=sys.stderr)


def _patch_progress_bars() -> None:
    """Add tqdm progress bars to PPO (matching FastSAC) and label both with algo name."""
    import tqdm as _tqdm

    # --- PPO: wrap learn() with a progress bar ---
    from holosoma.agents.ppo.ppo import PPO

    _original_ppo_learn = PPO.learn

    def _ppo_learn_with_pbar(self):
        """PPO.learn() with a tqdm progress bar showing 'PPO'."""
        global _s3_sync
        self._train_mode()
        obs_dict = self.env.reset_all()

        if self.config.init_at_random_ep_len:
            import torch

            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        if self.is_main_process and _s3_sync is None:
            s3_dir = os.environ.get("CKPT_S3_DIR", "")
            if s3_dir:
                interval = int(os.environ.get("SYNC_INTERVAL_SEC", "600"))
                _s3_sync = S3SyncManager(self.log_dir, s3_dir, interval)

        pbar = _tqdm.tqdm(
            total=self.config.num_learning_iterations, initial=self.current_learning_iteration, desc="PPO"
        )

        for it in range(
            self.current_learning_iteration, self.current_learning_iteration + self.config.num_learning_iterations
        ):
            self.current_learning_iteration = it

            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                onnx_path = os.path.join(self.log_dir, f"model_{it:05d}.onnx")
                self.export(onnx_file_path=onnx_path)
                _attach_metadata_to_onnx(self.env, onnx_path)
                if self.empirical_normalization:
                    _attach_normalizer_to_onnx(self.actor_obs_normalizer, onnx_path)
                if _s3_sync is not None:
                    _s3_sync.maybe_sync(force=True)

            if _s3_sync is not None:
                _s3_sync.maybe_sync()

            pbar.update(1)

        pbar.close()

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            onnx_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx")
            self.export(onnx_file_path=onnx_path)
            _attach_metadata_to_onnx(self.env, onnx_path)
            if self.empirical_normalization:
                _attach_normalizer_to_onnx(self.actor_obs_normalizer, onnx_path)
            if _s3_sync is not None:
                _s3_sync.final_sync()

    PPO.learn = _ppo_learn_with_pbar

    # --- FastSAC: patch tqdm.tqdm to inject desc="FastSAC" ---
    from holosoma.agents.fast_sac import fast_sac_agent

    _original_tqdm = _tqdm.tqdm

    class _LabeledTqdm(_original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("desc", "FastSAC")
            super().__init__(*args, **kwargs)

    fast_sac_agent.tqdm.tqdm = _LabeledTqdm


_patch_progress_bars()


def _patch_deploy_onnx_export() -> None:
    """Force WBT training to export the DEPLOYMENT (motion-embedded) ONNX, not the
    38 MB actor-only "browser" ONNX.

    Background: which format ``model_*.onnx`` gets is decided in holosoma core's
    ``PPO.export`` by ``if command_manager.get_state("motion_command") is not None``:
    present -> ``export_motion_and_policy_as_onnx`` (obs+time_step in; actions +
    joint_pos/joint_vel/ref_pos_xyz/ref_quat_xyzw out; the full motion library baked
    in -> ~784 MB, what the real-robot ``bm_wbt`` deploy stack consumes); absent ->
    ``export_policy_as_onnx`` (actor_obs -> action only, ~38 MB, browser viz).

    Some holosoma builds the training cluster has run take the actor-only branch, so
    WBT runs uploaded 38 MB ONNX that the deploy stack can't use (missing time_step +
    the reference outputs). This override makes the extensions layer OWN the decision:
    when a ``motion_command`` term exists (every WBT experiment), ALWAYS export the
    motion-embedded deployment ONNX, independent of the pinned holosoma version. When
    it is absent (a non-WBT experiment), fall back to holosoma's original ``export``
    so plain locomotion / other tasks are unaffected.

    The metadata + wandb-upload tail is a faithful mirror of holosoma's ``PPO.export``
    (same helpers, same keys), so nothing downstream drifts; only the export-format
    branch is forced.
    """
    from holosoma.agents.ppo.ppo import PPO
    from holosoma.utils.inference_helpers import (
        attach_onnx_metadata,
        export_motion_and_policy_as_onnx,
        get_command_ranges_from_env,
        get_control_gains_from_config,
        get_urdf_text_from_robot_config,
    )

    # Idempotency guard: if this patch is somehow applied twice (e.g. a test that
    # reloads this module), don't capture the ALREADY-wrapped export as the
    # "original" — that would make the non-WBT fallback recurse. A no-op on the
    # normal single-import path; pure hardening.
    if getattr(PPO.export, "_wbt_deploy_patched", False):
        return
    _original_ppo_export = PPO.export

    def _ppo_export_deploy(self, onnx_file_path: str) -> None:
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            # Non-WBT experiment (no motion library to bake) -> keep holosoma's
            # original behavior verbatim.
            _original_ppo_export(self, onnx_file_path)
            return

        was_training = self.actor.training
        self._eval_mode()

        # FORCE the 784 MB motion-embedded deploy export (obs+time_step ->
        # actions + joint_pos/joint_vel/ref_pos_xyz/ref_quat_xyzw).
        export_motion_and_policy_as_onnx(self.actor_onnx_wrapper, motion_command, onnx_file_path, self.device)

        # Metadata tail: byte-for-byte the same keys/helpers as holosoma PPO.export,
        # so deploy metadata (dof_names/kp/kd/action_scale/command_ranges/robot_urdf/
        # experiment_config/iteration/normalizer_*) is identical to a native deploy export.
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        action_scales = getattr(self.env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=metadata)

        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self._train_mode()

    _ppo_export_deploy._wbt_deploy_patched = True
    PPO.export = _ppo_export_deploy


_patch_deploy_onnx_export()


def _patch_bad_tracking_z_only() -> None:
    """Patch BadTracking.bad_motion_body_pos to check Z-component only.

    Holosoma's default uses full 3D distance for the end-effector body
    position termination check.  TML-BeyondMimic-Generalist uses Z-only
    (``bad_motion_body_pos_z_only``).  Because our motion data is converted
    with MuJoCo FK (which can differ from IsaacSim FK in the XY plane),
    the full 3D check causes immediate termination.  Matching TML's Z-only
    check fixes this.
    """
    from holosoma.managers.termination.terms.wbt import BadTracking
    from holosoma.utils.safe_torch_import import torch

    def _bad_motion_body_pos_z_only(self, motion_command) -> torch.Tensor:
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, 2] - motion_command.robot_body_pos_w[:, body_idx, 2]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    BadTracking.bad_motion_body_pos = _bad_motion_body_pos_z_only


_patch_bad_tracking_z_only()


def _patch_ppo_actor_positive_std() -> None:
    """Ensure PPOActor std is always positive via softplus.

    With multi-GPU training (DDP), gradient all-reduce can produce NaN or
    extremely large gradients that push the raw std parameter negative or NaN.
    ``torch.clamp`` cannot recover from NaN.  Using ``softplus`` on the raw
    parameter guarantees a strictly positive std regardless of parameter value.
    """
    from holosoma.agents.modules.ppo_modules import PPOActor
    from holosoma.utils.safe_torch_import import torch

    _original_update_distribution = PPOActor.update_distribution

    def _safe_update_distribution(self, observations):
        mean = self.actor_module(observations)
        # softplus guarantees positive output; nan_to_num recovers from NaN
        # (multi-GPU all-reduce can propagate NaN from any single GPU)
        safe_std = torch.nan_to_num(self.std, nan=1.0)
        positive_std = torch.nn.functional.softplus(safe_std)
        if self.min_noise_std:
            positive_std = torch.clamp(positive_std, min=self.min_noise_std)
        mean = torch.nan_to_num(mean, nan=0.0)
        from torch.distributions import Normal

        self.distribution = Normal(mean, mean * 0.0 + positive_std)

    PPOActor.update_distribution = _safe_update_distribution


_patch_ppo_actor_positive_std()


def _patch_nan_safe_gradient_reduce() -> None:
    """Make multi-GPU gradient all-reduce NaN-safe.

    If any single GPU produces NaN gradients (e.g. from numerical overflow in
    the critic or unusual observation values), the all_reduce(SUM) propagates
    NaN to ALL GPUs, corrupting every model parameter after the optimizer step.

    This patch:
    1. Replaces NaN/Inf in gradients with zeros before all_reduce
    2. Checks for NaN after all_reduce and replaces with zeros
    """
    from holosoma.agents.ppo.ppo import PPO
    from holosoma.utils.safe_torch_import import torch

    _original_reduce = PPO._reduce_parameters

    def _nan_safe_reduce_parameters(self):
        grads = [
            param.grad.view(-1)
            for model in [self.actor, self.critic]
            for param in model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)

        # Replace NaN/Inf with zeros before all_reduce (GPU-only, no sync)
        all_grads = torch.nan_to_num(all_grads, nan=0.0, posinf=0.0, neginf=0.0)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Replace NaN/Inf after all_reduce (GPU-only, no sync)
        all_grads = torch.nan_to_num(all_grads, nan=0.0, posinf=0.0, neginf=0.0)

        offset = 0
        for model in [self.actor, self.critic]:
            for param in model.parameters():
                if param.grad is not None:
                    numel = param.numel()
                    param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                    offset += numel

    PPO._reduce_parameters = _nan_safe_reduce_parameters


_patch_nan_safe_gradient_reduce()


import tyro  # noqa: E402

from holosoma.train_agent import train  # noqa: E402
from holosoma.utils.sim_utils import setup_simulator_imports  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402
from wbt_training.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402


def _stagger_multi_gpu_init() -> None:
    """Lightweight stagger for multi-GPU IsaacSim init.

    Each rank delays by (local_rank * INIT_STAGGER_SEC) seconds before starting.
    This avoids all 8 ranks hitting the GPU simultaneously during AppLauncher init.

    With the num_envs-per-rank fix in setup_isaaclab_launcher (each rank only
    allocates its share of envs), the per-rank init is light enough that a short
    stagger (3-5s) suffices. No sequential waiting needed.

    Set INIT_STAGGER_SEC=0 to disable staggering entirely.
    Set INIT_MODE=sequential to fall back to the old sentinel-based sequential init.
    """
    import time

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        return

    init_mode = os.environ.get("INIT_MODE", "stagger")

    if init_mode == "sequential":
        _sequential_multi_gpu_init()
        return

    stagger_sec = int(os.environ.get("INIT_STAGGER_SEC", "5"))
    delay = local_rank * stagger_sec

    if delay > 0:
        print(f"[INIT] Rank {local_rank}/{world_size}: staggering {delay}s before IsaacSim init", file=sys.stderr)
        time.sleep(delay)

    print(f"[INIT] Rank {local_rank}/{world_size}: starting IsaacSim init", file=sys.stderr)


def _sequential_multi_gpu_init() -> None:
    """Sequential IsaacSim init across GPUs using file-based signaling (fallback).

    Each rank waits for the previous rank to write a sentinel file before starting.
    Rank 0 starts immediately. Use INIT_MODE=sequential to enable this mode.
    """
    import time

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        return

    sentinel_dir = os.environ.get("INIT_SENTINEL_DIR", "/tmp/wbt_init_sentinels")
    os.makedirs(sentinel_dir, exist_ok=True)

    max_wait = int(os.environ.get("INIT_MAX_WAIT_SEC", "3600"))

    if local_rank > 0:
        prev_sentinel = os.path.join(sentinel_dir, f"rank_{local_rank - 1}_done")
        print(f"[SEQ-INIT] Rank {local_rank}/{world_size}: waiting for rank {local_rank - 1}...", file=sys.stderr)
        waited = 0
        while not os.path.exists(prev_sentinel):
            time.sleep(5)
            waited += 5
            if waited > max_wait:
                print(f"[SEQ-INIT] Rank {local_rank}: TIMEOUT after {max_wait}s. Proceeding.", file=sys.stderr)
                break
            if waited % 60 == 0:
                print(f"[SEQ-INIT] Rank {local_rank}: still waiting ({waited}s)...", file=sys.stderr)
        print(f"[SEQ-INIT] Rank {local_rank}: starting IsaacSim init", file=sys.stderr)
    else:
        for f in os.listdir(sentinel_dir):
            if f.startswith("rank_") and f.endswith("_done"):
                os.remove(os.path.join(sentinel_dir, f))
        print("[SEQ-INIT] Rank 0: starting IsaacSim init (first)", file=sys.stderr)


def _write_init_sentinel() -> None:
    """Write sentinel file to signal that this rank's IsaacSim init is done."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    sentinel_dir = os.environ.get("INIT_SENTINEL_DIR", "/tmp/wbt_init_sentinels")
    os.makedirs(sentinel_dir, exist_ok=True)
    sentinel_path = os.path.join(sentinel_dir, f"rank_{local_rank}_done")
    with open(sentinel_path, "w") as f:
        f.write(f"rank {local_rank} IsaacSim init done\n")
    print(f"[SEQ-INIT] Rank {local_rank}: IsaacSim init done, wrote {sentinel_path}", file=sys.stderr)


def _patch_train_sequential_init() -> None:
    """Patch init_sim_imports to write sentinel AFTER IsaacSim init.

    Only needed for INIT_MODE=sequential. Writes sentinel between
    init_sim_imports() and configure_multi_gpu() so the next rank can proceed.
    """
    init_mode = os.environ.get("INIT_MODE", "stagger")
    if init_mode != "sequential":
        return

    import holosoma.train_agent as holo_train_mod

    _orig_init_sim_imports = holo_train_mod.init_sim_imports

    def _init_sim_imports_with_sentinel(tyro_config):
        result = _orig_init_sim_imports(tyro_config)
        _write_init_sentinel()
        return result

    holo_train_mod.init_sim_imports = _init_sim_imports_with_sentinel


def _pop_f_on_hand_frame_flag() -> bool:
    """Pop the bare store-true `--F_on_hand_frame` token from sys.argv.

    This is the `--F_on_hand_frame` ablation switch (see force_cmd_local's
    docstring). We consume it BEFORE tyro.cli() so tyro never sees the unknown
    flag, then apply it to the parsed config in main(). Absent flag ⇒ untouched
    old behavior, byte-for-byte. Every rank parses argv independently and sees
    the same flag, which is correct (the obs encoding must match across ranks).
    Accepts the literal `--F_on_hand_frame`; a trailing `True`/`False` value is
    also honored if present (e.g. `--F_on_hand_frame True`).
    """
    token = "--F_on_hand_frame"
    if token not in sys.argv:
        return False
    idx = sys.argv.index(token)
    enabled = True
    # Optionally consume a following bool literal so both `--F_on_hand_frame`
    # and `--F_on_hand_frame True/False` work.
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in ("True", "False", "true", "false"):
        enabled = sys.argv[idx + 1] in ("True", "true")
        del sys.argv[idx + 1]
    del sys.argv[idx]
    return enabled


def _apply_f_on_hand_frame(tyro_cfg, enabled: bool):
    """Return tyro_cfg with the ACTOR's force_cmd_local obs term in the hand frame.

    Returns the cfg UNCHANGED (same object) when the flag is off or the preset has
    no force_cmd_local actor term, so the flag is harmless on non-force experiments.
    `ExperimentConfig` is a frozen dataclass, so we rebuild it with
    `dataclasses.replace(observation=...)` rather than mutating in place. The CRITIC
    term and all physics/reward/command config are untouched — strict actor-only,
    pelvis-frame sampling/application. See force_cmd_local's docstring.
    """
    if not enabled:
        return tyro_cfg
    import dataclasses

    from wbt_training.config_values.force_tracking import with_force_cmd_hand_frame

    obs = getattr(tyro_cfg, "observation", None)
    if obs is None or not getattr(obs, "groups", None):
        print("[F_on_hand_frame] WARNING: config has no observation groups; flag ignored.", file=sys.stderr)
        return tyro_cfg
    new_obs = with_force_cmd_hand_frame(obs)
    if new_obs is obs:
        print(
            "[F_on_hand_frame] WARNING: no force_cmd_local actor term in this preset "
            "(only g1-wbt-2pt-force has one); flag had no effect.",
            file=sys.stderr,
        )
        return tyro_cfg
    print("[F_on_hand_frame] ENABLED: actor force_cmd_local obs encoded in the HAND frame.", file=sys.stderr)
    return dataclasses.replace(tyro_cfg, observation=new_obs)


def _pop_f_on_sensor_frame_flag() -> bool:
    """Pop the bare store-true `--F_on_sensor_frame` token from sys.argv.

    Sensor-frame counterpart of `--F_on_hand_frame` (see force_cmd_local's
    sensor-frame docstring). Same argv-pop-before-tyro discipline; accepts a bare
    flag or a trailing True/False literal. Mutually exclusive with
    `--F_on_hand_frame` (enforced in main()).
    """
    token = "--F_on_sensor_frame"
    if token not in sys.argv:
        return False
    idx = sys.argv.index(token)
    enabled = True
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1] in ("True", "False", "true", "false"):
        enabled = sys.argv[idx + 1] in ("True", "true")
        del sys.argv[idx + 1]
    del sys.argv[idx]
    return enabled


def _apply_f_on_sensor_frame(tyro_cfg, enabled: bool):
    """Return tyro_cfg with the ACTOR's force_cmd_local obs term in the sensor frame.

    Sensor-frame counterpart of `_apply_f_on_hand_frame`. Returns the cfg UNCHANGED
    when off or when the preset has no force_cmd_local actor term. Strict actor-only,
    pelvis-frame sampling/application; see force_cmd_local's sensor-frame docstring.
    """
    if not enabled:
        return tyro_cfg
    import dataclasses

    from wbt_training.config_values.force_tracking import with_force_cmd_sensor_frame

    obs = getattr(tyro_cfg, "observation", None)
    if obs is None or not getattr(obs, "groups", None):
        print("[F_on_sensor_frame] WARNING: config has no observation groups; flag ignored.", file=sys.stderr)
        return tyro_cfg
    new_obs = with_force_cmd_sensor_frame(obs)
    if new_obs is obs:
        print(
            "[F_on_sensor_frame] WARNING: no force_cmd_local actor term in this preset "
            "(only force presets have one); flag had no effect.",
            file=sys.stderr,
        )
        return tyro_cfg
    print("[F_on_sensor_frame] ENABLED: actor force_cmd_local obs encoded in the FT-SENSOR frame.", file=sys.stderr)
    return dataclasses.replace(tyro_cfg, observation=new_obs)


def main() -> None:
    """Launch training with G1 WBT presets."""
    _stagger_multi_gpu_init()
    _f_on_hand_frame = _pop_f_on_hand_frame_flag()
    _f_on_sensor_frame = _pop_f_on_sensor_frame_flag()
    if _f_on_hand_frame and _f_on_sensor_frame:
        raise SystemExit(
            "--F_on_hand_frame and --F_on_sensor_frame are mutually exclusive "
            "(both re-encode the actor's force_cmd_local obs; pick one)."
        )
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    tyro_cfg = _apply_f_on_hand_frame(tyro_cfg, _f_on_hand_frame)
    tyro_cfg = _apply_f_on_sensor_frame(tyro_cfg, _f_on_sensor_frame)
    setup_simulator_imports(tyro_cfg)
    _patch_train_sequential_init()

    # Profiling: set PROFILE=1 to enable timing instrumentation
    if os.environ.get("PROFILE", "") == "1":
        from wbt_training.profiling_patch import apply_profiling_patches, dump_profiling_report

        warmup = int(os.environ.get("PROFILE_WARMUP", "20"))
        niters = int(os.environ.get("PROFILE_ITERS", "30"))
        apply_profiling_patches(warmup=warmup, profile_iters=niters)

        import atexit

        report_path = os.environ.get(
            "PROFILE_OUTPUT", f"/tmp/profiling_report_rank{os.environ.get('LOCAL_RANK', '0')}.json"
        )
        atexit.register(dump_profiling_report, report_path)

    train(tyro_cfg)


if __name__ == "__main__":
    main()
