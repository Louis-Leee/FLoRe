"""Profiling instrumentation for WBT training.

Monkey-patches timing stubs into the training loop to measure per-component
latency with CUDA synchronization. Outputs a detailed JSON report.

Usage: call ``apply_profiling_patches(warmup=20, profile_iters=30)`` before
``agent.learn()``.  After training, call ``dump_profiling_report(path)`` to
write results.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Timer infrastructure
# ---------------------------------------------------------------------------

_timings: dict[str, list[float]] = defaultdict(list)
_profiling_active = False
_warmup_iters = 0
_profile_iters = 0
_current_iter = [0]  # mutable counter


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def _timed(name: str):
    """Context manager that records wall-clock time with CUDA sync."""
    if not _profiling_active:
        yield
        return
    _cuda_sync()
    t0 = time.perf_counter()
    yield
    _cuda_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _timings[name].append(elapsed_ms)


def _wrap_method(cls, method_name: str, timer_name: str):
    """Wrap an existing method with timing instrumentation."""
    original = getattr(cls, method_name)

    def wrapper(self, *args, **kwargs):
        with _timed(timer_name):
            return original(self, *args, **kwargs)

    wrapper.__name__ = method_name
    wrapper._original = original
    setattr(cls, method_name, wrapper)


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


def apply_profiling_patches(warmup: int = 20, profile_iters: int = 30) -> None:
    """Instrument the training loop for profiling.

    Args:
        warmup: Number of PPO iterations to skip before recording.
        profile_iters: Number of PPO iterations to profile (then stop recording).
    """
    global _warmup_iters, _profile_iters, _profiling_active
    _warmup_iters = warmup
    _profile_iters = profile_iters
    _current_iter[0] = 0

    # -- 1. Patch PPO.learn() to track iteration count and toggle profiling --
    from holosoma.agents.ppo.ppo import PPO

    _original_learn = PPO.learn

    def _profiled_learn(self):
        # We need to intercept the per-iteration loop. The loop is inside learn().
        # Instead, we patch the rollout + training steps.
        return _original_learn(self)

    # -- 2. Patch the per-iteration callbacks in the main loop --
    # PPO.learn calls _rollout_step and _training_step in a loop.
    # We wrap _rollout_step and _training_step to manage profiling state.

    _original_rollout = PPO._rollout_step
    _original_training = PPO._training_step

    def _profiled_rollout_step(self, obs_dict):
        global _profiling_active
        it = _current_iter[0]
        _profiling_active = _warmup_iters <= it < _warmup_iters + _profile_iters

        with _timed("ppo/rollout_total"):
            result = _original_rollout(self, obs_dict)

        return result

    def _profiled_training_step(self):
        global _profiling_active
        it = _current_iter[0]
        _profiling_active = _warmup_iters <= it < _warmup_iters + _profile_iters

        with _timed("ppo/training_total"):
            result = _original_training(self)

        _current_iter[0] += 1

        # Print progress
        if _profiling_active:
            n = it - _warmup_iters + 1
            print(f"[PROFILE] iter {it} ({n}/{_profile_iters})", flush=True)
        elif it == _warmup_iters + _profile_iters:
            print(f"[PROFILE] Profiling complete. {_profile_iters} iters recorded.", flush=True)
            _profiling_active = False
            # Dump report immediately (atexit unreliable with IsaacSim shutdown)
            report_path = os.environ.get(
                "PROFILE_OUTPUT", f"/tmp/profiling_report_rank{os.environ.get('LOCAL_RANK', '0')}.json"
            )
            dump_profiling_report(report_path)

        return result

    PPO._rollout_step = _profiled_rollout_step
    PPO._training_step = _profiled_training_step

    # -- 3. Patch env.step() sub-components --
    from holosoma.envs.base_task.base_task import BaseTask

    _original_pre_physics = BaseTask._pre_physics_step
    _original_physics = BaseTask._physics_step
    _original_post_physics = BaseTask._post_physics_step
    _original_refresh_sim = BaseTask._refresh_sim_tensors
    _original_update_tasks = BaseTask._update_tasks_callback
    _original_check_term = BaseTask._check_termination
    _original_compute_reward = BaseTask._compute_reward
    _original_compute_obs = BaseTask._compute_observations
    _original_compute_final_obs = BaseTask._compute_final_observations
    _original_reset_envs = BaseTask.reset_envs_idx
    _original_clip_obs = BaseTask._clip_observations

    def _prof_pre_physics(self, actions):
        with _timed("env/pre_physics_step"):
            return _original_pre_physics(self, actions)

    def _prof_physics(self):
        with _timed("env/physics_step"):
            return _original_physics(self)

    def _prof_refresh_sim(self):
        with _timed("env/refresh_sim_tensors"):
            return _original_refresh_sim(self)

    def _prof_update_tasks(self):
        with _timed("env/update_tasks"):
            return _original_update_tasks(self)

    def _prof_check_term(self):
        with _timed("env/check_termination"):
            return _original_check_term(self)

    def _prof_compute_reward(self):
        with _timed("env/compute_reward"):
            return _original_compute_reward(self)

    def _prof_compute_obs(self):
        with _timed("env/compute_observations"):
            return _original_compute_obs(self)

    def _prof_compute_final_obs(self):
        with _timed("env/compute_final_observations"):
            return _original_compute_final_obs(self)

    def _prof_reset_envs(self, env_ids):
        with _timed("env/reset_envs"):
            return _original_reset_envs(self, env_ids)

    def _prof_clip_obs(self):
        with _timed("env/clip_observations"):
            return _original_clip_obs(self)

    # We also want a fine-grained _post_physics_step to see the total
    def _prof_post_physics(self):
        with _timed("env/post_physics_total"):
            return _original_post_physics(self)

    BaseTask._pre_physics_step = _prof_pre_physics
    BaseTask._physics_step = _prof_physics
    BaseTask._post_physics_step = _prof_post_physics
    BaseTask._refresh_sim_tensors = _prof_refresh_sim
    BaseTask._update_tasks_callback = _prof_update_tasks
    BaseTask._check_termination = _prof_check_term
    BaseTask._compute_reward = _prof_compute_reward
    BaseTask._compute_observations = _prof_compute_obs
    BaseTask._compute_final_observations = _prof_compute_final_obs
    BaseTask.reset_envs_idx = _prof_reset_envs
    BaseTask._clip_observations = _prof_clip_obs

    # -- 4. Patch PPO internal components --
    # Actor inference, critic inference, gradient reduce
    _original_update_algo = PPO._update_algo_step

    def _prof_update_algo(self, minibatch, gpu_loss_accum):
        with _timed("ppo/update_algo_step"):
            return _original_update_algo(self, minibatch, gpu_loss_accum)

    PPO._update_algo_step = _prof_update_algo

    # Gradient reduce
    _original_reduce = PPO._reduce_parameters

    def _prof_reduce(self):
        with _timed("ppo/gradient_reduce"):
            return _original_reduce(self)

    PPO._reduce_parameters = _prof_reduce

    # -- 5. Patch MotionCommand.step() --
    from holosoma.managers.command.terms.wbt import MotionCommand

    _original_motion_step = MotionCommand.step

    def _prof_motion_step(self):
        with _timed("command/motion_command_step"):
            return _original_motion_step(self)

    MotionCommand.step = _prof_motion_step

    # -- 6. Patch observation manager compute_group for per-group timing --
    from holosoma.managers.observation.manager import ObservationManager

    _original_compute_group = ObservationManager.compute_group

    def _prof_compute_group(self, group_name, *args, **kwargs):
        with _timed(f"obs_group/{group_name}"):
            return _original_compute_group(self, group_name, *args, **kwargs)

    ObservationManager.compute_group = _prof_compute_group

    # -- 7. Patch reward manager for per-term timing --
    from holosoma.managers.reward.manager import RewardManager

    _original_reward_compute = RewardManager.compute

    def _prof_reward_compute(self, dt):
        with _timed("reward/total"):
            return _original_reward_compute(self, dt)

    RewardManager.compute = _prof_reward_compute

    print(f"[PROFILE] Patches applied. Warmup={warmup}, profile_iters={profile_iters}", flush=True)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def dump_profiling_report(path: str) -> dict[str, Any]:
    """Write profiling results to a JSON file and return the summary dict."""
    import statistics as stat

    report = {}
    for name, times in sorted(_timings.items()):
        if not times:
            continue
        report[name] = {
            "count": len(times),
            "mean_ms": round(stat.mean(times), 3),
            "median_ms": round(stat.median(times), 3),
            "std_ms": round(stat.stdev(times), 3) if len(times) > 1 else 0.0,
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "total_ms": round(sum(times), 3),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 3) if len(times) >= 20 else None,
        }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Component':<45} {'Mean(ms)':>10} {'Median':>10} {'P95':>10} {'Count':>8}")
    print("-" * 90)
    for name in sorted(report, key=lambda k: -report[k]["mean_ms"]):
        r = report[name]
        p95 = f"{r['p95_ms']:.1f}" if r["p95_ms"] is not None else "n/a"
        print(f"{name:<45} {r['mean_ms']:>10.2f} {r['median_ms']:>10.2f} {p95:>10} {r['count']:>8}")
    print("=" * 90)
    print(f"Report saved to: {path}")

    return report
