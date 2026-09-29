"""Multi-motion dataset management: combining motion/terrain NPZs, WandB registry
pulling, and teacher model wrapping.

Uses
``holosoma.utils.wandb_registry`` instead of ``whole_body_tracking.utils.wandb_helpers``.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Motion / terrain combining
# ---------------------------------------------------------------------------

# Shared temp directory for combined files
_TMP_DIR = Path(tempfile.mkdtemp(prefix="holosoma_motions_"))


def combine_motions(input_files: List[str | Path]) -> Path:
    """Concatenate multiple motion NPZ files into one.

    Uses a deterministic hash of input filenames for caching.
    """
    data_list = []
    for f in input_files:
        with np.load(f) as data:
            d = {key: data[key] for key in data.files}
            if "motion_ends" not in d:
                d["motion_ends"] = np.zeros((d["joint_pos"].shape[0],), dtype=bool)
                d["motion_ends"][-1] = True
            data_list.append(d)

    # Keys that are scalar metadata — keep from first file, don't concatenate.
    # - fps: scalar
    # - joint_names / body_names: per-motion metadata, identical across files
    #   in the same registry. Converted holosoma NPZs ship these fields;
    #   ref artifact NPZs do not.
    _SCALAR_KEYS = {"fps", "joint_names", "body_names"}

    combined = {}
    for key in data_list[0].keys():
        if key in _SCALAR_KEYS:
            combined[key] = data_list[0][key]
        else:
            combined[key] = np.concatenate([d[key] for d in data_list], axis=0)

    combined_file_name = hashlib.sha256(",".join(str(f) for f in input_files).encode()).hexdigest()[:16]
    out_path = _TMP_DIR / f"{combined_file_name}_motions.npz"
    np.savez(out_path, **combined)
    return out_path


def combine_terrains(input_files: List[str | Path]) -> Path:
    """Concatenate multiple terrain NPZ files into one."""
    obj_list: list[np.ndarray] = []
    obj_count = 0
    obj_count_list = [0]

    for f in input_files:
        data = np.load(f)
        if np.prod(data.shape) == 0:
            obj_list.append(np.zeros((0, 10, 10)))
            obj_count_list.append(obj_count)
        else:
            obj_list.append(data)
            obj_count_list.append(obj_count + data.shape[0])
            obj_count += data.shape[0]

    # Determine num_variant from first non-empty entry
    num_variant = 10
    for arr in obj_list:
        if arr.shape[0] != 0:
            num_variant = arr.shape[1]
            break
    for i in range(len(obj_list)):
        if obj_list[i].shape[0] == 0:
            obj_list[i] = np.zeros((0, num_variant, 10))

    combined_obj_list = np.concatenate(obj_list, axis=0)
    obj_count_arr = np.array(obj_count_list)

    combined_file_name = hashlib.sha256(",".join(str(f) for f in input_files).encode()).hexdigest()[:16]
    out_path = _TMP_DIR / f"{combined_file_name}_terrains.npz"
    np.savez(out_path, obj_list=combined_obj_list, obj_count_list=obj_count_arr)
    return out_path


def add_motion_idx_from_motion_counts(motion_file: Path, motion_counts: list[int]) -> None:
    """Add ``motion_idxs`` array to motion file based on per-registry motion counts."""
    with np.load(motion_file) as data:
        T = data["joint_pos"].shape[0]
        motion_ends = data["motion_ends"]
        motion_ends_list = np.nonzero(motion_ends)[0]
        motion_idxs_cumsum = np.cumsum(np.array(motion_counts))

        motion_idxs = np.searchsorted(motion_ends_list, np.arange(T), side="left")
        motion_idxs = np.searchsorted(motion_idxs_cumsum, motion_idxs, side="right")

        new_motion_file = motion_file.parent / f"{motion_file.stem}.npz"
        np.savez(new_motion_file, **dict(data), motion_idxs=motion_idxs)


# ---------------------------------------------------------------------------
# WandB registry helpers
# ---------------------------------------------------------------------------


def pull_from_wandb_registry(registry_list: str | List[str]) -> Tuple[list[int], Path, Path]:
    """Download and combine motions/terrains from WandB registry names.

    Parameters
    ----------
    registry_list : str or list[str]
        Comma-separated registry names or a list.

    Returns
    -------
    tuple
        ``(motion_counts, motion_file, terrain_file)``
    """
    from holosoma.utils.wandb_registry import download_motion_and_terrain, resolve_tag

    if isinstance(registry_list, str):
        registry_list = registry_list.split(",")

    motion_files: list[Path] = []
    terrain_files: list[Path] = []
    motion_counts: list[int] = []

    for registry_name in registry_list:
        registry_name = resolve_tag(registry_name)
        motion_file, terrain_file = download_motion_and_terrain(registry_name)
        motion_files += motion_file if motion_file else []
        terrain_files += terrain_file if terrain_file else []
        motion_counts.append(len(motion_file) if motion_file else 0)

    motion_file = combine_motions(motion_files)
    terrain_file = combine_terrains(terrain_files)

    add_motion_idx_from_motion_counts(motion_file, motion_counts)
    return motion_counts, motion_file, terrain_file


def pull_paired_from_wandb_registry(registry_list: str | List[str]) -> Tuple[list[int], Path, Path, list[Path]]:
    """Download paired (motion, terrain) data from WandB registry.

    Each WandB artifact is expected to contain paired files:

    * **Motion** files: any file whose name contains ``"motion"``
      (typically ``motion*.npz``).
    * **Terrain** files: any file whose name contains ``"terrain"``
      (typically ``terrain*.npy`` for the new-format obstacle-pose arrays,
      or ``terrain.obj`` for legacy OBJ mesh artifacts).

    The function separates terrain files by extension:

    * ``.npy`` terrain files are obstacle-pose arrays — they are combined
      via :func:`combine_terrains` into a single NPZ that
      :func:`~holosoma.utils.obstacle_helpers.add_onpath_obstacle` can
      consume (``obj_list`` + ``obj_count_list``).
    * ``.obj`` terrain files are mesh files — they are **not** combined
      but returned as-is in a separate list.  The caller can pass the
      first OBJ path directly to
      ``--terrain.terrain_term.obj_file_path``.

    Parameters
    ----------
    registry_list : str or list[str]
        Comma-separated WandB registry names (e.g.
        ``"server/wandb-registry-terrains-motions/walk:latest"``),
        or a Python list of such names.

    Returns
    -------
    motion_counts : list[int]
        Number of motion files contributed by each registry entry.
    motion_file : Path
        Combined motion NPZ (with ``motion_idxs`` added).
    terrain_npy_file : Path
        Combined terrain obstacle NPZ (from ``.npy`` files).
        Will be an empty-data NPZ if no ``.npy`` terrain files were found.
    terrain_obj_files : list[Path]
        Raw ``.obj`` terrain mesh files (empty list if none found).
    """
    from holosoma.utils.wandb_registry import download_motion_and_terrain, resolve_tag

    if isinstance(registry_list, str):
        registry_list = registry_list.split(",")

    motion_files: list[Path] = []
    terrain_npy_files: list[Path] = []
    terrain_obj_files: list[Path] = []
    motion_counts: list[int] = []

    for registry_name in registry_list:
        registry_name = resolve_tag(registry_name)
        motion_paths, terrain_paths = download_motion_and_terrain(registry_name)

        if motion_paths:
            motion_files += motion_paths
            motion_counts.append(len(motion_paths))
        else:
            motion_counts.append(0)

        if terrain_paths:
            for tp in terrain_paths:
                if tp.suffix.lower() == ".obj":
                    terrain_obj_files.append(tp)
                else:
                    terrain_npy_files.append(tp)

    # Combine motion NPZ files
    motion_file = combine_motions(motion_files)
    add_motion_idx_from_motion_counts(motion_file, motion_counts)

    # Combine terrain NPY files (obstacle pose arrays) if any exist
    if terrain_npy_files:
        terrain_npy_file = combine_terrains(terrain_npy_files)
    else:
        # Create an empty terrain NPZ so downstream code doesn't break
        empty_file = _TMP_DIR / "empty_terrains.npz"
        np.savez(empty_file, obj_list=np.zeros((0, 10, 10)), obj_count_list=np.array([0] * (len(registry_list) + 1)))
        terrain_npy_file = empty_file

    return motion_counts, motion_file, terrain_npy_file, terrain_obj_files


def pull_from_wandb_path(
    wandb_path: str | List[str], override_registry_name: str | None = None
) -> Tuple[list[Path], list[int], Path, Path]:
    """Pull checkpoints and associated motions/terrains from WandB run paths.

    Parameters
    ----------
    wandb_path : str or list[str]
        Comma-separated WandB run paths or a list.
    override_registry_name : str | None
        If provided, fetch motion/terrain from these registries instead of
        the runs' linked artifacts.

    Returns
    -------
    tuple
        ``(resume_paths, motion_counts, motion_file, terrain_file)``
    """
    from holosoma.utils.wandb_registry import (
        download_checkpoint,
        download_checkpoint_motion_and_terrain,
        download_motion_and_terrain,
        resolve_tag,
    )

    if isinstance(wandb_path, str):
        wandb_path = wandb_path.split(",")

    resume_paths: list[Path] = []
    motion_files: list[Path] = []
    terrain_files: list[Path] = []
    motion_counts: list[int] = []

    if not override_registry_name:
        for path in wandb_path:
            resume_path, motion_file, terrain_file = download_checkpoint_motion_and_terrain(path)
            resume_paths.append(resume_path)
            motion_files += motion_file if motion_file else []
            terrain_files += terrain_file if terrain_file else []
            motion_counts.append(len(motion_file) if motion_file else 0)
    else:
        override_list = (
            override_registry_name.split(",") if isinstance(override_registry_name, str) else override_registry_name
        )
        for path in wandb_path:
            resume_path = download_checkpoint(path)
            resume_paths.append(resume_path)
        for registry_name in override_list:
            registry_name = resolve_tag(registry_name)
            motion_file, terrain_file = download_motion_and_terrain(registry_name)
            motion_files += motion_file if motion_file else []
            terrain_files += terrain_file if terrain_file else []
            motion_counts.append(len(motion_file) if motion_file else 0)

    # Combining is optional — a registry entry with a single motion or no terrain
    # still produces a usable motion_file. Log and continue on failure.
    motion_file_combined = Path("")
    terrain_file_combined = Path("")
    try:
        motion_file_combined = combine_motions(motion_files)
        terrain_file_combined = combine_terrains(terrain_files)
    except Exception as e:
        logger.warning(f"combine_motions/combine_terrains failed: {e}")

    try:
        add_motion_idx_from_motion_counts(motion_file_combined, motion_counts)
    except Exception as e:
        logger.warning(f"add_motion_idx_from_motion_counts failed: {e}")

    return resume_paths, motion_counts, motion_file_combined, terrain_file_combined


# ---------------------------------------------------------------------------
# Policy wrappers (pure nn.Module)
# ---------------------------------------------------------------------------


class PolicyWrapper(torch.nn.Module):
    """Wrap a teacher policy (actor or student) for uniform inference."""

    def __init__(self, policy: torch.nn.Module, normalizer: torch.nn.Module | None = None):
        super().__init__()
        self.is_recurrent = getattr(policy, "is_recurrent", False)
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "teacher"):
            self.actor = copy.deepcopy(policy.teacher)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm  # type: ignore[assignment]

        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()
        self.device = next(self.actor.parameters()).device

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))


class MultiTeacher(torch.nn.Module):
    """Route observations to the correct teacher model based on motion index (first obs dim)."""

    def __init__(self, teacher_models: list[PolicyWrapper]):
        super().__init__()
        self.device = teacher_models[0].device
        self.teacher_models = teacher_models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x[:, 0].long()
        outputs = self.teacher_models[0](x)
        for i in range(1, len(self.teacher_models)):
            mask = idx == i
            outputs[mask] = self.teacher_models[i](x[mask])
        mask = idx == -1
        outputs[mask] = 0.0
        return outputs


# ---------------------------------------------------------------------------
# Checkpoint adaptation
# ---------------------------------------------------------------------------


def wrap_model_dict(state_old: dict, state_new: dict) -> dict:
    """Adapt a teacher checkpoint's keys to match a student-teacher model."""
    # Detect MyHolosoma PPO-format checkpoint (holosoma.agents.ppo.ppo.PPO saves
    # actor weights under "actor_model_state_dict" with "actor_module.module.*"
    # keys due to DataParallel wrapping). Flatten to the rsl_rl-style "actor.N.X"
    # form that the rest of this function expects. Mirrors the inverse transform
    # at src/holosoma/holosoma/agents/ppo/ppo.py:_map_external_policy_state_dict.
    if isinstance(state_old.get("actor_model_state_dict"), dict):
        ppo_actor_sd = state_old["actor_model_state_dict"]
        flat: dict = {}
        for k, v in ppo_actor_sd.items():
            if k == "std":
                flat["std"] = v
            elif k.startswith("actor_module.module."):
                flat["actor." + k.removeprefix("actor_module.module.")] = v
        state_old = {"model_state_dict": flat}

    # Support both wrapped checkpoints (with "model_state_dict") and raw state dicts
    has_wrapper = "model_state_dict" in state_old
    if has_wrapper:
        flat_old = state_old["model_state_dict"]
    else:
        flat_old = state_old
    flat_new = state_new

    def _replace_actor_in_keys(flat: dict, old_key: str, new_key: str) -> dict:
        new_flat = {}
        for k, v in flat.items():
            parts = k.split(".")
            parts = [new_key if p == old_key else p for p in parts]
            new_flat[".".join(parts)] = v
        return new_flat

    flat_old = _replace_actor_in_keys(flat_old, "actor", "teacher")
    shared = set(flat_old.keys()) & set(flat_new.keys())

    mismatches = []
    for k in sorted(shared):
        if "critic" in k:
            continue
        v_old = flat_old[k]
        v_new = flat_new[k]
        if isinstance(v_old, torch.Tensor) and isinstance(v_new, torch.Tensor):
            if v_old.shape != v_new.shape:
                padded_tensor = torch.zeros_like(v_new)
                if v_old.dim() == 2:
                    padded_tensor[: v_old.shape[0], -v_old.shape[1] :] = v_old
                elif v_old.dim() == 1:
                    padded_tensor[: v_old.shape[0]] = v_old
                else:
                    # Fall back: copy the overlap prefix along each dim
                    slices = tuple(slice(0, s) for s in v_old.shape)
                    padded_tensor[slices] = v_old
                flat_old[k] = padded_tensor
                mismatches.append((k, v_old.shape, v_new.shape))

    if mismatches:
        for k, shape_old, shape_new in mismatches:
            print(f"Shape mismatch at key '{k}': old shape {shape_old}, new shape {shape_new}")

    flat_old = _replace_actor_in_keys(flat_old, "teacher", "actor")
    if has_wrapper:
        state_old["model_state_dict"] = flat_old
    else:
        state_old = {"model_state_dict": flat_old}
    return state_old


def wrap_model_dict_from_file(path_old: Path, state_new: dict) -> str:
    """Load a checkpoint, adapt it, save a ``*_wrapped.pt`` copy and return the path."""
    try:
        state_old = torch.load(str(path_old), map_location="cpu", weights_only=False)
    except Exception:
        # Checkpoint may be a TorchScript (jit.save) model — extract state_dict
        jit_model = torch.jit.load(str(path_old), map_location="cpu")
        state_old = jit_model.state_dict()
    state_old = wrap_model_dict(state_old, state_new)
    wrapped_path = str(path_old).replace(".pt", "_wrapped.pt")
    torch.save(state_old, wrapped_path)
    return wrapped_path
