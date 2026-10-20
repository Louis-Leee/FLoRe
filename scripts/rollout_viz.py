#!/usr/bin/env python3
"""Flow-matching INFERENCE visualization in viser — recorded zarr vs. DP-Flow rollout.

The training-side counterpart of ``UMI_collection/replay/replay_raw_viz.py``: instead
of replaying what a tracker recorded, this loads a converted UMI ``.zarr.zip`` episode,
lets you stop on ANY frame, and — on a "Rollout" click — runs the trained DP-Flow
policy on that frame (plus the observation history / reference frame the checkpoint was
trained with) to produce ONE action chunk. It then draws BOTH trajectories in one G1
scene so you can eyeball the policy against the ground truth:

  * the RECORDED zarr trajectory (left hand + right hand + head/torso) as a BLUE
    GRADIENT (oldest dark -> newest light), because the collected data is by
    construction a sensible trajectory;
  * the ROLLOUT trajectory (the model's absolute EE targets for the same keypoints)
    in ORANGE, starting from the chosen frame;
  * the G1 humanoid animated through the SAME mink IK the collection viz uses
    (``ThreePointIK``), pre-solved once per trajectory so scrubbing never calls mink;
  * gripper open/close WIDTH (left + right) as readouts + charts;
  * the recorded ``robot0_ft_wrench`` force/torque as arrows on the primary wrist +
    scrolling charts (mirrors ``run_replay_viz.sh``).

ONE rollout = ONE H-step chunk (no chained re-inference): to inspect a longer horizon,
move the start frame later and roll out again. This deliberately avoids feeding the
policy its own imagined future images.

CRITICAL — this tool does NOT use ``serving/server.py::load_dpflow_chunk_policy``:
that loader hard-refuses ``ref_frame != per_keypoint``, ``obs_history_len > 1``, and every
HL-v2 / diffusion checkpoint (the exact things we want to visualize). Instead it builds
the model directly and replicates the dataset preprocessing + un-relativization itself.

Two model families are supported, auto-detected from the checkpoint's ``config.json``:

* **Legacy 32-dim flow** (``algo`` absent/``flow`` with an empty ``layout_mode``): a
  ``DPFlowModel`` fed by ``UMIDataset``-style preprocessing (``resolve_base`` +
  ``delta_ee_to_absolute``), honoring whatever ref_frame / obs_history_len it declares.
* **HL-v2** (non-empty ``layout_mode``): a ``DPFlowModel`` (``algo=flow``) OR a
  ``DPDiffusionModel`` (``algo=diffusion``) fed by the REAL :class:`HLDataset` (so the
  observation — To image frames, ``wrt_start_rot6d`` state, the 8-bin force history — is
  bit-identical to training), then the per-hand ``[pos rot6d gripper force]`` output is
  un-relativized against each hand's own ``t0`` anchor and mapped into the 32-dim
  ``chunk_abs`` the IK + rendering downstream already consume. Diffusion samples via its
  DDIM loop, flow via its Euler ODE — the tool only calls ``sample_actions``.

With ``--checkpoint`` empty it random-inits the legacy model from
``configs/dp_flow_3pt.py`` so the pipeline can be exercised before a real checkpoint
exists (the rollout is then meaningless — a wiring check only).

ISOLATION: reads only already-committed files. It NEVER writes zarr/checkpoints, never
touches ``~/UMI_collection`` code (the mink IK + yam URDF are referenced read-only via
PYTHONPATH / path), and runs in FLoRe's own ``.venv`` (viz deps are the optional ``viz``
extra) — the training env is unaffected.

Run (run_rollout_viz.sh sets PYTHONPATH for the IK import):
    PYTHONPATH=third_party/UMI_collection/third_party/humi_ik:third_party/UMI_collection/viz \\
        .venv/bin/python scripts/rollout_viz.py /tmp/session_3pt_ft.zarr.zip \\
        --checkpoint checkpoints/dp_flow_3pt/step_00100000
    # no checkpoint -> random-init (pipeline check only):
    ... scripts/rollout_viz.py /tmp/session_3pt_ft.zarr.zip
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLORE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UMI = os.path.join(_FLORE_ROOT, "third_party", "UMI_collection")
# The mink IK stack + the reusable viser Timeline/wrench/chart helpers are vendored in
# UMI_collection (read-only). Add them to the path here as a fallback so the script also
# runs when launched directly (run_rollout_viz.sh sets these too).
for _p in (
    os.path.join(_UMI, "third_party", "humi_ik"),  # humi_data_collection.ikumi
    os.path.join(_UMI, "viz"),                       # g1_three_point_ik / g1_right_hand_ik
    os.path.join(_UMI, "replay"),                     # replay_common (Timeline, WrenchArrows, charts)
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _FLORE_ROOT not in sys.path:
    sys.path.insert(0, _FLORE_ROOT)

import viser  # noqa: E402
from viser import transforms as vtf  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402

import replay_common as rc  # noqa: E402  (UMI_collection/replay: Timeline + helpers)

from flore.s1_policy.data.transforms import (  # noqa: E402
    delta_ee_to_absolute,
    gripper_norm_to_width,
    gripper_width_to_norm,
    resolve_base,
    rotvec_to_rot6d,
)
from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_quat_xyzw, rotmat_to_rot6d  # noqa: E402

_DEFAULT_YAM = os.path.join(_UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
_JAW_OPEN_M = 0.0475  # URDF jaw prismatic upper limit (matches collection_viz / replay)
_GRASP_THRESH_M = 0.08  # UMI's hard-coded grasp threshold, for the readout label

# 32-dim keypoint slots, kept in sync with flore.s1_policy.data.umi_dataset:
#   name: (pos_slice, rot6d_slice, gripper_index_or_None)
_LEFT = (slice(0, 3), slice(3, 9), 9)
_RIGHT = (slice(10, 13), slice(13, 19), 19)
_HEAD = (slice(20, 23), slice(23, 29), None)
ACTION_DIM = 32
#: role -> the 32-dim (pos, rot6d, gripper) slots above. Used ONLY by the HL-v2 path to
#: map an HL model's per-hand output block into the fixed 32-dim ``chunk_abs`` the IK +
#: rendering downstream already consumes, so that machinery is reused unchanged.
_ROLE_TO_32SLOT = {"left": _LEFT, "right": _RIGHT, "head": _HEAD}

# Recorded trajectory: BLUE gradient (dark early -> light late). Rollout: ORANGE.
_RECORDED_BLUE_DARK = np.array([25, 40, 110], dtype=np.float64)
_RECORDED_BLUE_LIGHT = np.array([120, 190, 255], dtype=np.float64)
_ROLLOUT_ORANGE = (245, 150, 40)

# Which keypoints get a trajectory line, and the IK role each maps to.
_KEYPOINTS = (("left", _LEFT), ("right", _RIGHT), ("head", _HEAD))

# Robot frame each IK role's trajectory line is READ FROM (mirrors g1_three_point_ik
# _ROLE_FRAMES). The trajectory polylines are drawn from these SOLVED frame positions (in
# the robot/MJCF world), NOT the raw zarr world coords — so a line always sits exactly on
# the rendered robot part (the hand_site the IK targets / the torso for the head).
_SITE_FRAMES = {
    "left": ("left_hand_site", "site"),
    "right": ("right_hand_site", "site"),
    "head": ("torso_link", "body"),
}


# ---------------------------------------------------------------------------
# Zarr episode loader (mirrors UMIDataset's on-disk contract, read-only)
# ---------------------------------------------------------------------------
class Episode:
    """One episode of a converted UMI ``.zarr.zip``, sliced on episode boundaries.

    Holds the raw per-keypoint absolute poses + gripper widths + F/T + RGB for the
    chosen episode, plus the LEFT-canonical routing (which zarr key feeds which 32-dim
    slot) resolved from ``robot0_primary_role``. The model preprocessing and the
    recorded-trajectory rendering both read from here."""

    def __init__(self, zarr_path: str, episode: int):
        import zarr
        from imagecodecs.numcodecs import register_codecs

        register_codecs()  # camera0_rgb is JpegXl; decode-on-read needs this registered
        self.path = zarr_path
        self.g = zarr.open(zarr_path, "r")
        self.attrs = dict(self.g.attrs)
        ends = np.asarray(self.g["meta/episode_ends"][:], dtype=np.int64)
        if ends.size == 0:
            raise SystemExit(f"[rollout-viz] no episodes in {zarr_path}")
        starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        self.n_ep = int(ends.size)
        self.ep = int(np.clip(episode, 0, self.n_ep - 1))
        self._a, self._b = int(starts[self.ep]), int(ends[self.ep])
        self.ep_lengths = [int(e - s) for s, e in zip(starts, ends)]

        data = self.g["data"]
        self.data_keys = set(data.keys())
        self.image_key = "camera0_rgb"
        self.primary_role = str(self.attrs.get("robot0_primary_role", "right"))
        self.tracking_mode = str(self.attrs.get("tracking_mode", "?"))
        self.has_left = "robot1_eef_pos" in self.data_keys
        self.has_head = "head_pos" in self.data_keys
        self.has_ft = "robot0_ft_wrench" in self.data_keys

        # LEFT-canonical zarr-key -> 32-slot routing (reuse the dataset's own logic so
        # this tool can never drift from training).
        from flore.s1_policy.data.umi_dataset import UMIDataset

        self.specs = UMIDataset._build_specs(
            self.primary_role, self.has_left, self.has_head, self.data_keys)
        # The keypoint whose recorded pose drives the IK primary arm + camera + F/T.
        # LEFT-canonical -> 'left'; legacy single-right -> 'right'.
        self.primary_kp = "left" if self.primary_role == "left" else "right"

        self.T = self._b - self._a
        # Timestamps are RELATIVE per episode (DECISION-A) — each starts at 0.
        self.ts = self._slice("episode_timestamps").reshape(-1).astype(np.float64)

        # Absolute per-keypoint pose arrays over the whole episode (world frame), keyed
        # by keypoint name: {name: (pos[T,3], rot6d[T,6], rotvec[T,3], grip_norm[T]|None)}.
        self.kp: dict[str, tuple] = {}
        for name, keys, (_pos_sl, _rot_sl, grip_i) in self.specs:
            pos_key, rot_key, grip_key = keys
            pos = self._slice(pos_key).astype(np.float64)          # (T, 3)
            rotvec = self._slice(rot_key).astype(np.float64)       # (T, 3)
            rot6d = rotvec_to_rot6d(rotvec)                        # (T, 6)
            grip_norm = None
            if grip_i is not None and grip_key is not None:
                gw = self._slice(grip_key).reshape(-1).astype(np.float64)  # metres
                grip_norm = gripper_width_to_norm(gw)              # [0, 1]
            self.kp[name] = (pos, rot6d, rotvec, grip_norm)

        self.ft = self._slice("robot0_ft_wrench").astype(np.float64) if self.has_ft else None

    def _slice(self, key: str) -> np.ndarray:
        return np.asarray(self.g[f"data/{key}"][self._a:self._b])

    def image_at(self, i: int) -> np.ndarray:
        """Decoded RGB (H, W, 3) uint8 at episode-local frame ``i``."""
        return np.asarray(self.g[f"data/{self.image_key}"][self._a + int(i)])

    def gripper_width_m(self, name: str, i: int) -> float | None:
        """Recorded gripper width (metres) for keypoint ``name`` at frame ``i``."""
        grip_norm = self.kp[name][3]
        if grip_norm is None:
            return None
        return float(gripper_norm_to_width(grip_norm[i]))


# ---------------------------------------------------------------------------
# Model construction (real checkpoint OR random-init fallback)
# ---------------------------------------------------------------------------
class LoadedModel:
    """A ready-to-infer ``DPFlowModel`` plus the data knobs the checkpoint was trained
    with (ref_frame / obs_history_len / stride / image norm / norm_stats).

    The knobs come from the checkpoint's ``config.json`` when present; with no
    checkpoint they fall back to ``configs/dp_flow_3pt.py`` (random-init) so the whole
    pipeline can be smoke-tested before training finishes."""

    def __init__(self, checkpoint: str, device: str, num_denoising_steps: int | None):
        import torch

        from flore.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

        self.torch = torch
        self.device = device
        self.random_init = not checkpoint
        self.norm_stats: dict = {}
        self.image_mean = 0.5
        self.image_std = 0.5
        self.ref_frame = "per_keypoint"
        self.obs_history_len = 1
        self.obs_history_stride = 1

        if self.random_init:
            ref = self._config_from_reference()  # full TrainConfig
            cfg = ref.model
            # Take the data knobs from the reference config's DataConfig (not hardcoded),
            # so editing configs/dp_flow_3pt.py's DataConfig (To/stride/ref_frame/image
            # norm) is honored here too — cfg.obs_history_len is already synced to
            # data.obs_history_len by TrainConfig.__post_init__.
            self.ref_frame = str(ref.data.ref_frame)
            self.obs_history_len = int(ref.data.obs_history_len)
            self.obs_history_stride = int(ref.data.obs_history_stride)
            self.image_mean = float(ref.data.image_mean)
            self.image_std = float(ref.data.image_std)
            print("[rollout-viz] no --checkpoint: RANDOM-INIT model from configs/dp_flow_3pt.py "
                  f"(ref_frame={self.ref_frame}, obs_history_len={self.obs_history_len}; "
                  "pipeline check only — rollout is meaningless).", flush=True)
            model = DPFlowModel(cfg)
        else:
            import dataclasses
            import json

            ckpt_dir = Path(checkpoint)
            raw = json.loads((ckpt_dir / "config.json").read_text())
            model_cfg_raw = raw.get("model", raw)
            data_cfg_raw = raw.get("data", {})
            self.ref_frame = str(data_cfg_raw.get("ref_frame", "per_keypoint"))
            # obs_history_len is authoritative under MODEL (TrainConfig pushes it there
            # for both pipelines; HLDataConfig has no such field, so an HL config.json
            # carries it only under `model`). Read model-first — the reverse would
            # resolve an HL history checkpoint to To=1 and mis-size the model.
            self.obs_history_len = int(
                model_cfg_raw.get("obs_history_len", data_cfg_raw.get("obs_history_len", 1)))
            self.obs_history_stride = int(data_cfg_raw.get("obs_history_stride", 1))
            self.image_mean = float(data_cfg_raw.get("image_mean", 0.5))
            self.image_std = float(data_cfg_raw.get("image_std", 0.5))

            field_types = {f.name for f in dataclasses.fields(DPFlowConfig)}
            kwargs = {}
            for k, v in model_cfg_raw.items():
                if k not in field_types:
                    continue
                kwargs[k] = tuple(v) if isinstance(v, list) else v
            cfg = DPFlowConfig(**kwargs)
            # The model must be sized for the SAME To it was trained with.
            cfg.obs_history_len = self.obs_history_len
            model = DPFlowModel(cfg)
            model.load_checkpoint(str(ckpt_dir))
            print(f"[rollout-viz] loaded checkpoint {ckpt_dir} "
                  f"(ref_frame={self.ref_frame}, obs_history_len={self.obs_history_len}).", flush=True)

            ns_path = ckpt_dir / "norm_stats.json"
            if ns_path.exists():
                from flore.s1_policy.data.norm_stats import load_norm_stats

                self.norm_stats = load_norm_stats(ns_path)

        model.to(device).eval()
        self.model = model
        self.cfg = cfg
        self.action_horizon = int(cfg.action_horizon)
        self.num_denoising_steps = num_denoising_steps or int(cfg.inference_steps)
        # Action-stats key. Match the TRAINER exactly: it normalizes actions ONLY under
        # the literal key "actions" (trainer._apply_norm), so we must too — probing
        # "action_chunk"/"action" would unnormalize an output the model was trained to
        # emit RAW, injecting a large spurious offset into the orange path. Warn loudly if
        # a misleading-but-unused action key is present so the mismatch is visible.
        self.action_key = "actions"
        if self.norm_stats and "actions" not in self.norm_stats:
            stray = [k for k in ("action_chunk", "action") if k in self.norm_stats]
            if stray:
                print(f"[rollout-viz] WARNING: norm_stats has {stray} but not 'actions'; the "
                      f"trainer only normalizes the 'actions' key, so actions are treated as "
                      f"UNNORMALIZED (matching training). Re-key your norm_stats to 'actions' "
                      f"if that is wrong.", flush=True)

    @staticmethod
    def _config_from_reference():
        """Build the full TrainConfig from configs/dp_flow_3pt.py (import by path — configs
        is not a package). Returns the whole config so both the model AND the data knobs
        (ref_frame / obs_history_len / stride / image norm) can be read from it."""
        import importlib.util

        cfg_path = os.path.join(_FLORE_ROOT, "configs", "dp_flow_3pt.py")
        spec = importlib.util.spec_from_file_location("dp_flow_3pt_cfg", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build()


# ---------------------------------------------------------------------------
# Preprocess one anchor frame -> batch_dict, sample, un-relativize (train contract)
# ---------------------------------------------------------------------------
def build_batch_and_sample(lm: LoadedModel, ep: Episode, anchor: int) -> np.ndarray:
    """Replicate ``UMIDataset.__getitem__`` for a single anchor frame, run
    ``sample_actions``, and un-relativize to ABSOLUTE 32-dim poses.

    Returns ``chunk_abs`` of shape (H, 32): per-keypoint absolute pos+rot6d in the world
    frame, gripper dims kept as normalized [0,1] (grip is absolute in every ref_frame).
    Honors the checkpoint's ref_frame + obs_history_len (bypassing the serving guard)."""
    torch = lm.torch
    To = lm.obs_history_len
    stride = lm.obs_history_stride

    # Observation window: causal, ENDS at the anchor, clamped to the episode start so it
    # never crosses the boundary (== UMIDataset). obs_frames[-1] == anchor.
    obs_frames = np.clip(anchor - stride * np.arange(To - 1, -1, -1), 0, ep.T - 1)

    # Assemble the 32-dim absolute anchor state (last obs frame) for resolve_base, and
    # the per-keypoint obs-window arrays for the state history.
    anchor_abs = np.zeros(ACTION_DIM, dtype=np.float64)
    obs_state = np.zeros((To, ACTION_DIM), dtype=np.float64)
    per_kp_obs = {}  # name -> (pos_sl, rot_sl, grip_i, obs_pos[To,3], obs_rot6d[To,6], obs_grip[To]|None)
    for name, _keys, (pos_sl, rot_sl, grip_i) in ep.specs:
        pos, rot6d, _rotvec, grip_norm = ep.kp[name]
        obs_pos = pos[obs_frames]           # (To, 3)
        obs_rot6d = rot6d[obs_frames]       # (To, 6)
        obs_grip = grip_norm[obs_frames] if grip_norm is not None else None
        per_kp_obs[name] = (pos_sl, rot_sl, grip_i, obs_pos, obs_rot6d, obs_grip)
        anchor_abs[pos_sl] = obs_pos[-1]
        anchor_abs[rot_sl] = obs_rot6d[-1]

    # Shared base for left/right/head/world; None for per_keypoint (own-anchor per kp).
    shared = lm.ref_frame != "per_keypoint"
    base_pos, base_rot6d = (resolve_base(anchor_abs, lm.ref_frame) if shared else (None, None))

    # State history (To, 32) in the model's frame. per_keypoint: absolute obs poses.
    from flore.s1_policy.data.transforms import absolute_to_delta_ee

    for name, (pos_sl, rot_sl, grip_i, obs_pos, obs_rot6d, obs_grip) in per_kp_obs.items():
        if shared:
            s_pos, s_rot6d = absolute_to_delta_ee(obs_pos, obs_rot6d, base_pos, base_rot6d)
        else:
            s_pos, s_rot6d = obs_pos, obs_rot6d
        obs_state[:, pos_sl] = s_pos
        obs_state[:, rot_sl] = s_rot6d
        if grip_i is not None and obs_grip is not None:
            obs_state[:, grip_i] = obs_grip

    # Normalize state exactly like the trainer (quantile stats when present).
    state_in = obs_state.copy()
    if "state" in lm.norm_stats:
        from flore.s1_policy.data.norm_stats import normalize_quantile

        state_in = normalize_quantile(state_in, lm.norm_stats["state"])
    if To == 1:
        state_tensor = torch.from_numpy(state_in[0]).float()[None]   # (1, 32)
    else:
        state_tensor = torch.from_numpy(state_in).float()[None]      # (1, To, 32)

    # Images: To frames of the single camera, normalized like the dataset, as To list
    # entries (To pseudo-cameras) — matches collate_umi_batch's history layout.
    images = []
    for f in obs_frames:
        img = ep.image_at(int(f)).astype(np.float32) / 255.0
        img = (img - lm.image_mean) / lm.image_std
        images.append(np.transpose(img, (2, 0, 1)))  # (C, H, W)
    image_tensors = [torch.from_numpy(im).float()[None].to(lm.device) for im in images]  # each (1,C,H,W)
    img_masks = [torch.ones(1, dtype=torch.bool, device=lm.device) for _ in image_tensors]

    batch_dict = {
        "images": image_tensors,
        "img_masks": img_masks,
        "state": state_tensor.to(lm.device),
    }

    lm.model.reset()  # clear any RTC prefix from a previous rollout — fresh chunk
    with torch.no_grad():
        out = lm.model.sample_actions(batch_dict, num_steps=lm.num_denoising_steps)
    chunk = out.detach().float().cpu().numpy()[0]  # (H, 32) normalized, model frame

    # Unnormalize actions (inverse of the trainer's quantile normalize).
    if lm.action_key in lm.norm_stats:
        from flore.s1_policy.data.norm_stats import unnormalize_quantile

        chunk = unnormalize_quantile(chunk, lm.norm_stats[lm.action_key])

    # Un-relativize to ABSOLUTE world poses. per_keypoint: each keypoint vs its OWN
    # anchor (obs[-1]); shared frames: every keypoint vs the single shared base.
    chunk_abs = chunk.copy()
    for name, (pos_sl, rot_sl, grip_i, obs_pos, obs_rot6d, _obs_grip) in per_kp_obs.items():
        if shared:
            ref_pos, ref_rot6d = base_pos, base_rot6d
        else:
            ref_pos, ref_rot6d = obs_pos[-1], obs_rot6d[-1]
        abs_pos, abs_rot6d = delta_ee_to_absolute(
            rel_pos=chunk[:, pos_sl], rel_rot6d=chunk[:, rot_sl],
            ref_pos=ref_pos, ref_rot6d=ref_rot6d)
        chunk_abs[:, pos_sl] = abs_pos
        chunk_abs[:, rot_sl] = abs_rot6d
    # Grip + pad dims pass through unchanged (absolute in every frame mode).
    return chunk_abs


# ---------------------------------------------------------------------------
# HL-v2 path: detect, load (flow OR diffusion), and sample via the real HLDataset
# ---------------------------------------------------------------------------
def _detect_hl_checkpoint(checkpoint: str) -> tuple[bool, str, str]:
    """Inspect a checkpoint's ``config.json`` -> ``(is_hl, algo, layout_mode)``.

    A checkpoint is HL-v2 iff the ACTIVE model config (``diffusion`` block when
    ``algo=diffusion``, else ``model``) has a non-empty ``layout_mode``. ``algo`` defaults
    to ``flow`` (pre-algo-switch checkpoints). An empty ``--checkpoint`` is never HL (the
    random-init legacy path). Reads only the JSON, so it is cheap and import-free."""
    if not checkpoint:
        return False, "flow", ""
    import json

    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    algo = str(raw.get("algo", "flow") or "flow")
    active = raw.get(_algo_config_key(algo), raw) or {}
    layout_mode = str(active.get("layout_mode", "") or "")
    return bool(layout_mode), algo, layout_mode


def _algo_config_key(algo: str) -> str:
    """The ``config.json`` block holding ``algo``'s REAL model config.

    Delegates to :data:`flore.s1_policy.training.config.ALGO_CONFIG_FIELD` so this tool
    cannot drift from the trainer's own mapping. It used to be
    ``"diffusion" if algo == "diffusion" else "model"``, which read the FLOW block for any
    third algorithm — and because a flow-default block has ``layout_mode=""``, an HL
    checkpoint was then misclassified as pre-v2 and loaded through the legacy 32-dim
    ``DPFlowModel`` path, whose ``load_checkpoint`` uses ``strict=False`` and logs
    mismatched keys as WARNINGS. That is a silent garbage load, so an unknown algo raises
    here instead.
    """
    from flore.s1_policy.training.config import ALGO_CONFIG_FIELD

    try:
        return ALGO_CONFIG_FIELD[algo]
    except KeyError:
        raise SystemExit(
            f"checkpoint declares algo={algo!r}, which this tool does not know how to read "
            f"(known: {sorted(ALGO_CONFIG_FIELD)}). Refusing rather than guessing a config "
            f"block — reading the wrong one silently loads a differently-shaped model."
        ) from None


class HLLoadedModel:
    """A ready-to-infer HL-v2 model (flow OR diffusion) + the HLDataset that builds its
    training-identical observation.

    Unlike :class:`LoadedModel` (legacy 32-dim, which replicates ``UMIDataset``
    preprocessing by hand), this REUSES the real :class:`HLDataset` for the observation —
    the To image frames, the ``wrt_start_rot6d`` state, and the 8-bin causal force
    history are then guaranteed bit-identical to what the model trained on, and can never
    drift from the dataset. It samples one action chunk and maps the per-hand
    ``[pos rot6d gripper force]`` blocks into the fixed 32-dim ``chunk_abs`` the IK +
    rendering downstream already consume."""

    def __init__(self, checkpoint: str, algo: str, layout_mode: str, zarr_path: str,
                 device: str, num_denoising_steps: int | None):
        import dataclasses
        import json

        import torch

        from flore.s1_policy.data.hl_dataset import HLDataset
        from flore.s1_policy.data.layout import get_layout
        from flore.s1_policy.data.norm_stats import load_norm_stats

        self.torch = torch
        self.device = device
        self.random_init = False  # HL path always needs a real checkpoint
        self.algo = algo
        self.layout_mode = layout_mode
        self.layout = get_layout(layout_mode)
        # HL is per-keypoint by contract (each hand relative to its OWN anchor); exposed
        # so main()'s shared GUI + guards read it uniformly with the legacy LoadedModel.
        self.ref_frame = "per_keypoint"

        ckpt_dir = Path(checkpoint)
        raw = json.loads((ckpt_dir / "config.json").read_text())
        model_cfg_raw = raw.get(_algo_config_key(algo), raw)

        if algo == "diffusion":
            from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig as Cfg
            from flore.s1_policy.model.dp_diffusion import DPDiffusionModel as Model
        elif algo == "act":
            from flore.s1_policy.model.dp_act import DPACTConfig as Cfg
            from flore.s1_policy.model.dp_act import DPACTModel as Model
        else:
            from flore.s1_policy.model.dp_flow import DPFlowConfig as Cfg
            from flore.s1_policy.model.dp_flow import DPFlowModel as Model

        field_types = {f.name for f in dataclasses.fields(Cfg)}
        kwargs = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in model_cfg_raw.items() if k in field_types}
        cfg = Cfg(**kwargs)
        # A LOCAL CHECKPOINT MUST NEVER TRIGGER A NETWORK DOWNLOAD. inference/common.py's
        # dataclass_config does this for the offline/serving paths; this viz builds its
        # config by hand and so must do it too. It mattered little while every preset used
        # timm (the local weights overwrote whatever was fetched), but ACT's ResNet trunk
        # defaults to pretrained=True, so without this line every launch hits torchvision.
        cfg.pretrained = False
        self.cfg = cfg
        self.obs_history_len = int(getattr(cfg, "obs_history_len", 2))
        self.image_mean = float(model_cfg_raw.get("image_mean", 0.5))
        self.image_std = float(model_cfg_raw.get("image_std", 0.5))
        # HL image norm is a DATA knob (hl_data.image_mean/std), not on the model config;
        # read it from the hl_data block when present, else the [-1,1] default.
        hl_data_raw = raw.get("hl_data", {}) or {}
        self.image_mean = float(hl_data_raw.get("image_mean", self.image_mean))
        self.image_std = float(hl_data_raw.get("image_std", self.image_std))
        self.obs_history_dt_s = float(hl_data_raw.get("obs_history_dt_s", 0.25))

        model = Model(cfg)
        model.load_checkpoint(str(ckpt_dir))
        model.to(device).eval()
        self.model = model
        self.action_horizon = int(cfg.action_horizon)
        # Diffusion: DDIM steps; flow: Euler steps; ACT: structurally 1 (a CVAE decodes the
        # whole chunk in one forward pass, and its sample_actions REFUSES any other value).
        # ACT is pinned here rather than left to the getattr chain because its config does
        # carry an `inference_steps` field (== 1), so a CLI --num-denoising-steps override
        # would otherwise reach the sampler and raise mid-rollout.
        if algo == "act":
            self.num_denoising_steps = 1
        else:
            default_steps = int(getattr(cfg, "num_inference_steps", getattr(cfg, "inference_steps", 16)))
            self.num_denoising_steps = num_denoising_steps or default_steps

        self.norm_stats: dict = {}
        ns_path = ckpt_dir / "norm_stats.json"
        if ns_path.exists():
            self.norm_stats = load_norm_stats(ns_path)

        # The dataset IS the preprocessing. drop_glitch_windows=False so its index is a
        # 1:1 map from anchor frame to (file, frame) — the viz lets you roll out from ANY
        # frame, including ones a training run would drop, and clamps invalid targets via
        # the mask anyway. obs history knobs come from the checkpoint.
        print(f"[rollout-viz] HL-v2 {algo} checkpoint {ckpt_dir} "
              f"(mode={layout_mode}, To={self.obs_history_len}, "
              f"denoise={self.num_denoising_steps}).", flush=True)
        self.ds = HLDataset(
            [zarr_path], mode=layout_mode, image_mean=self.image_mean, image_std=self.image_std,
            drop_glitch_windows=False, drop_invalid_force_history=False,
            obs_history_len=self.obs_history_len, obs_history_dt_s=self.obs_history_dt_s)
        # Global frame -> dataset index. With drop_glitch_windows=False this is identity
        # per file, but build the map explicitly so a future filtered index still resolves.
        self._frame_to_idx = {frame: i for i, (fi, frame) in enumerate(self.ds._index) if fi == 0}

    def close(self) -> None:
        try:
            self.ds.close()
        except Exception:  # noqa: BLE001 — best-effort
            pass


def build_hl_batch_and_sample(hlm: "HLLoadedModel", ep: "Episode", anchor: int) -> np.ndarray:
    """Sample one HL-v2 action chunk at ``anchor`` and return it as a 32-dim ``chunk_abs``.

    ``anchor`` is EPISODE-LOCAL (0..ep.T-1, the Start-frame slider's range), matching every
    other ``ep``-driven quantity (blue trajectory, IK calib, camera frustum). ``HLDataset``,
    however, indexes the WHOLE file with GLOBAL frame numbers, so the lookup MUST offset by
    the episode's start frame ``ep._a`` — otherwise ``--episode > 0`` would silently sample
    a DIFFERENT episode's frame (global frame ``anchor`` lands in episode 0), producing an
    orange rollout computed from the wrong episode's observation but drawn against this
    episode's arm. (``--episode 0`` has ``ep._a == 0`` so local == global — which is exactly
    why the single-episode smoke test did not catch this.)

    Uses the real :class:`HLDataset` sample at that global frame (training-identical
    observation), normalizes state + force_history exactly as the trainer does (only
    state+force are model INPUTS), samples, un-normalizes the action, and un-relativizes
    each hand's ``[pos, rot6d]`` block against that hand's own ``t0`` anchor
    (``anchor_pos``/``anchor_rot6d`` from the dataset — the per-keypoint frame the HL
    contract uses). The per-hand gripper (normalized [0,1]) is written into the 32-dim
    gripper slot; force columns are not rendered. Returns ``(H, 32)``; roles absent from
    this mode leave their 32-dim slots at zero."""
    import torch

    from flore.s1_policy.data.norm_stats import normalize_quantile, unnormalize_quantile
    from flore.s1_policy.data.transforms import delta_ee_to_absolute
    from flore.s1_policy.training.trainer import collate_hl_batch

    global_frame = int(ep._a) + int(anchor)  # episode-local -> global file frame
    idx = hlm._frame_to_idx.get(global_frame)
    if idx is None:  # frame filtered out of the index — nearest available GLOBAL frame
        idx = min(hlm._frame_to_idx.values(),
                  key=lambda i: abs(hlm.ds._index[i][1] - global_frame))
    sample = hlm.ds[idx]
    batch = collate_hl_batch([sample])  # B == 1

    # Normalize the MODEL INPUTS the way the trainer does. Actions are the target, never
    # fed in, so they are NOT normalized here (they are the thing we sample + un-normalize).
    for key in ("state", "force_history"):
        if key in hlm.norm_stats and key in batch:
            arr = batch[key].cpu().numpy()
            batch[key] = torch.from_numpy(normalize_quantile(arr, hlm.norm_stats[key])).to(batch[key].dtype)
    batch = {k: ([t.to(hlm.device) for t in v] if isinstance(v, list)
                 else (v.to(hlm.device) if torch.is_tensor(v) else v))
             for k, v in batch.items()}

    hlm.model.reset()  # fresh chunk, clear any RTC prefix (no-op for the diffusion sampler)
    with torch.no_grad():
        out = hlm.model.sample_actions(batch, num_steps=hlm.num_denoising_steps)
    chunk = out.detach().float().cpu().numpy()[0]  # (H, action_dim) normalized, model frame

    if "actions" in hlm.norm_stats:
        chunk = unnormalize_quantile(chunk, hlm.norm_stats["actions"])

    # Un-relativize each hand into the 32-dim chunk_abs. anchor_pos/anchor_rot6d are
    # per-role, in layout keypoint order (roles == hlm.layout.roles).
    lay = hlm.layout
    H = chunk.shape[0]
    chunk_abs = np.zeros((H, ACTION_DIM), dtype=np.float64)
    anchor_pos = sample["anchor_pos"].numpy()      # (n_roles, 3)
    anchor_rot6d = sample["anchor_rot6d"].numpy()  # (n_roles, 6)
    for ri, role in enumerate(lay.roles):
        slots = lay.slots(role)                    # HL 16-dim block slices
        dst = _ROLE_TO_32SLOT.get(role)
        if dst is None:
            continue
        pos_sl32, rot_sl32, grip_i32 = dst
        abs_pos, abs_rot6d = delta_ee_to_absolute(
            rel_pos=chunk[:, slots.pos], rel_rot6d=chunk[:, slots.rot6d],
            ref_pos=anchor_pos[ri], ref_rot6d=anchor_rot6d[ri])
        chunk_abs[:, pos_sl32] = abs_pos
        chunk_abs[:, rot_sl32] = abs_rot6d
        # Gripper: normalized [0,1], absolute (frame-invariant) — clip like the legacy path.
        if slots.gripper is not None and grip_i32 is not None:
            chunk_abs[:, grip_i32] = np.clip(chunk[:, slots.gripper], 0.0, 1.0)
    return chunk_abs


# ---------------------------------------------------------------------------
# IK: pre-solve a keypoint trajectory (absolute EE poses) into yam configs
# ---------------------------------------------------------------------------
def _rot6d_to_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """(6,) rot6d -> (4,) wxyz quaternion for the IK tracker pose."""
    q_xyzw = rotmat_to_quat_xyzw(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def tcp_to_tracker(pos: np.ndarray, rot6d: np.ndarray, inv_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover per-frame TRACKER poses from stored TCP poses for a HAND keypoint.

    The converter stored ``world_T_tcp = world_T_tracker @ T_tracker_tcp`` for the hands
    (``convert_to_umi.apply_tcp_extrinsic``; 0.21 m along wrist +x). The home-anchored IK
    must be driven from the TRACKER pose (exactly like ``replay_raw_viz`` drives it from
    the raw tracker, and ``replay_lerobot_viz.recover_tracker_poses`` re-derives it):
    ``world_T_tracker = world_T_tcp @ inv(T_tracker_tcp)``. Feeding TCP poses straight to
    the IK injects a spurious ``(R_cur - R_ref)·[0.21,0,0]`` lever-arm translation on any
    wrist rotation (~0.30 m at 90°). ``inv_ext`` is the precomputed 4x4 inverse.

    Returns (pos[K,3], rot6d[K,6]) in the tracker frame. Head keypoints get NO extrinsic
    (a body reference, not an EE) so they must NOT be passed here."""
    pos = np.asarray(pos, dtype=np.float64)
    rot6d = np.asarray(rot6d, dtype=np.float64)
    K = pos.shape[0]
    world_T_tcp = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
    world_T_tcp[:, :3, :3] = rot6d_to_rotmat(rot6d)
    world_T_tcp[:, :3, 3] = pos
    world_T_tracker = world_T_tcp @ inv_ext[None]
    tr_pos = world_T_tracker[:, :3, 3]
    tr_rot6d = rotmat_to_rot6d(world_T_tracker[:, :3, :3])
    return tr_pos, tr_rot6d


def presolve_trajectory(ik, mapper, active_roles, kp_poses, prim_role,
                        prim_grip_norm, other_role, other_grip_norm,
                        inv_ext, hand_roles, calib_ref=None, seed_q=None):
    """Solve the multi-role IK over a trajectory and cache the yam config per frame.

    ``kp_poses`` = {role: (pos[K,3], rot6d[K,6])} for the ACTIVE roles: absolute-world
    **TCP** poses for the hands (as stored in the zarr / predicted by the model) and raw
    body poses for the head. HAND poses are first recovered to the TRACKER frame via
    ``inv_ext`` (the inverse tracker->TCP extrinsic) — the home-anchored IK must be driven
    from the tracker pose, exactly like ``replay_raw_viz`` / ``replay_lerobot_viz`` do;
    feeding raw TCP poses would inject a lever-arm translation on wrist rotations. Head
    keypoints (``role`` not in ``hand_roles``) get NO extrinsic.

    Calibration: home-anchored, so the tracker frame-0 pose maps to the robot home pose.
    ``calib_ref`` overrides the reference used for the home anchor (default: this
    trajectory's own frame 0). The RECORDED pass uses its own frame 0; the ROLLOUT pass
    passes the RECORDED tracker poses at the anchor frame as ``calib_ref``, so both share
    ONE calibration — the rollout arm continues from exactly where the recorded arm was at
    the anchor, and the orange line branches off the blue line (directly comparable, no
    jump on a Showing switch). The jaw of each hand comes from that hand's gripper series.

    Returns (cfgs[K, n_yam], prim_wrist_pos[K,3], prim_wrist_wxyz[K,4], site_paths, q_seq)
    where the wrist frames follow the PRIMARY hand (camera + F/T mount), ``site_paths`` =
    {role: (K,3)} are the SOLVED robot-frame site positions per role — the trajectory
    polylines are drawn from these (so a line always sits on the rendered robot part), NOT
    from the raw zarr world coords — and ``q_seq`` (K, nq) is the solved HuMI configuration
    per frame (so a later rollout can SEED its solver from the recorded arm's pose at the
    anchor, making the branch tight from step 0 instead of creeping out from home).

    ``seed_q``: if given, the solver configuration is set to it AFTER recalibrate() (which
    resets to home) and BEFORE the first step — used by the rollout to continue from the
    recorded arm's actual anchor configuration."""
    roles = list(active_roles)
    K = len(next(iter(kp_poses.values()))[0])
    n_yam = len(mapper._yam_names)
    cfgs = np.zeros((K, n_yam))
    prim_wrist_pos = np.zeros((K, 3))
    prim_wrist_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (K, 1)).astype(float)
    site_paths = {role: np.zeros((K, 3)) for role in roles}
    q_seq = np.zeros((K, ik._solver.configuration.q.shape[0]))
    prim_wrist_link = f"{prim_role}_wrist_yaw_link"

    # Convert each role's trajectory into the IK (tracker) frame ONCE: hands via the
    # inverse TCP extrinsic, head passed through. Cache (pos[K,3], wxyz[K,4]) per role.
    ik_poses = {}
    for role in roles:
        pos, rot6d = kp_poses[role]
        if role in hand_roles:
            pos, rot6d = tcp_to_tracker(pos, rot6d, inv_ext)
        wxyz = np.stack([_rot6d_to_wxyz(rot6d[i]) for i in range(K)], axis=0)
        ik_poses[role] = (np.asarray(pos, dtype=float), wxyz)

    def pose_at(role, i):
        pos, wxyz = ik_poses[role]
        return pos[i], wxyz[i]

    # Home-anchor calibration: this trajectory's own frame 0, unless a shared reference was
    # supplied (rollout reuses the recorded frame-0 tracker poses).
    ref = calib_ref if calib_ref is not None else {role: pose_at(role, 0) for role in roles}
    ik.recalibrate(ref)
    if seed_q is not None:
        # Continue from the recorded arm's solved anchor pose (recalibrate reset to home;
        # calibration flag persists through update()).
        ik._solver.configuration.update(q=np.asarray(seed_q, dtype=float))
    for i in range(K):
        q = ik.step({role: pose_at(role, i) for role in roles})
        # Jaw metres per hand: primary hand always has a gripper; the other hand only if
        # its normalized series is present (usually absent — single physical gripper).
        jaw_kw = {}
        prim_jaw = (float(prim_grip_norm[i]) * _JAW_OPEN_M) if prim_grip_norm is not None else 0.0
        jaw_kw["left_jaw_m" if prim_role == "left" else "right_jaw_m"] = prim_jaw
        if other_role is not None and other_grip_norm is not None:
            other_jaw = float(other_grip_norm[i]) * _JAW_OPEN_M
            jaw_kw["left_jaw_m" if other_role == "left" else "right_jaw_m"] = other_jaw
        cfgs[i] = mapper.to_yam(q, **jaw_kw)
        q_seq[i] = q
        conf = ik._solver.configuration
        wrist = conf.get_transform_frame_to_world(frame_name=prim_wrist_link, frame_type="body")
        prim_wrist_pos[i] = wrist.translation()
        prim_wrist_wxyz[i] = wrist.rotation().wxyz
        # Solved robot-frame site position per role -> the trajectory polyline points.
        for role in roles:
            fn, ft = _SITE_FRAMES[role]
            site_paths[role][i] = conf.get_transform_frame_to_world(
                frame_name=fn, frame_type=ft).translation()
    return cfgs, prim_wrist_pos, prim_wrist_wxyz, site_paths, q_seq


def _tracker_poses_at(ep, active_roles, hand_roles, inv_ext, frame_idx):
    """Recorded TRACKER poses {role: (pos[3], wxyz[4])} at one episode frame — the shared
    home-anchor reference for the rollout IK (so it continues from the recorded arm)."""
    ref = {}
    for role in active_roles:
        pos = ep.kp[role][0][frame_idx:frame_idx + 1]     # (1,3) TCP/world
        rot6d = ep.kp[role][1][frame_idx:frame_idx + 1]   # (1,6)
        if role in hand_roles:
            pos, rot6d = tcp_to_tracker(pos, rot6d, inv_ext)
        ref[role] = (np.asarray(pos[0], dtype=float), _rot6d_to_wxyz(rot6d[0]))
    return ref


# ---------------------------------------------------------------------------
# Gradient / solid trajectory line helpers
# ---------------------------------------------------------------------------
def _blue_gradient_colors(n: int) -> np.ndarray:
    """(n, 3) uint8 blue ramp, dark (oldest) -> light (newest)."""
    if n <= 1:
        return _RECORDED_BLUE_LIGHT[None].astype(np.uint8)
    t = np.linspace(0.0, 1.0, n)[:, None]
    ramp = (_RECORDED_BLUE_DARK[None] * (1 - t) + _RECORDED_BLUE_LIGHT[None] * t)
    return np.clip(ramp, 0, 255).astype(np.uint8)


def draw_gradient_path(server, name: str, pts: np.ndarray, line_width: float = 3.0):
    """Draw a poly-line through ``pts`` (P,3) as a BLUE GRADIENT via add_line_segments
    ((N,2,3) points + (N,2,3) colors — one segment per consecutive pair). Returns the
    scene handle (its ``.visible`` toggles the whole line)."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return server.scene.add_line_segments(name, points=np.zeros((1, 2, 3), np.float32),
                                              colors=np.zeros((1, 2, 3), np.uint8), visible=False)
    seg_pts = np.stack([pts[:-1], pts[1:]], axis=1)  # (N, 2, 3)
    vcols = _blue_gradient_colors(pts.shape[0])       # (P, 3)
    seg_cols = np.stack([vcols[:-1], vcols[1:]], axis=1)  # (N, 2, 3)
    return server.scene.add_line_segments(name, points=seg_pts, colors=seg_cols, line_width=line_width)


def draw_solid_path(server, name: str, pts: np.ndarray, color, line_width: float = 3.0):
    """Draw a solid-color poly-line (used for the ORANGE rollout). Returns the handle."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return server.scene.add_line_segments(name, points=np.zeros((1, 2, 3), np.float32),
                                              colors=np.zeros((1, 2, 3), np.uint8), visible=False)
    seg_pts = np.stack([pts[:-1], pts[1:]], axis=1)
    col = np.array(color, dtype=np.uint8)
    seg_cols = np.broadcast_to(col, (seg_pts.shape[0], 2, 3)).copy()
    return server.scene.add_line_segments(name, points=seg_pts, colors=seg_cols, line_width=line_width)


def _load_inv_tcp_extrinsic(zarr_path: str, ep: "Episode", override: str) -> np.ndarray:
    """Return inv(T_tracker_tcp) as a 4x4, to recover tracker poses from stored TCP poses.

    Resolution order: (1) an explicit --tcp-extrinsic JSON override, (2) the repo-default
    ``third_party/UMI_collection/robot_assets/calibration/tcp_extrinsic.json`` (the SAME
    file + ``T_tracker_tcp`` key the converter's ``load_tcp_extrinsic`` reads). If the zarr
    reports ``tcp_calibrated=False`` the converter used an identity extrinsic (TCP ==
    tracker), so identity is returned and the recovery is a no-op. If calibration is
    claimed but no file resolves, we WARN and fall back to identity (better a small
    lever-arm artifact than a crash)."""
    import json

    if not bool(ep.attrs.get("tcp_calibrated", False)):
        return np.eye(4)
    path = override or os.path.join(
        _UMI, "robot_assets", "calibration", "tcp_extrinsic.json")
    if not os.path.exists(path):
        print(f"[rollout-viz] WARNING: tcp_calibrated=True but no extrinsic file at {path}; "
              "driving IK from raw TCP poses (wrist rotations will show a ~0.2 m lever-arm "
              "offset). Pass --tcp-extrinsic to fix.", flush=True)
        return np.eye(4)
    d = json.load(open(path))
    T = np.asarray(d.get("T_tracker_tcp", np.eye(4).tolist()), dtype=np.float64)
    print(f"[rollout-viz] tracker->TCP extrinsic from {os.path.relpath(path, _FLORE_ROOT)} "
          f"(translation {np.round(T[:3, 3], 3)} m).", flush=True)
    return np.linalg.inv(T)


# ---------------------------------------------------------------------------
def _make_ik(active_roles, config_name, reach_scale, yam_urdf):
    """Build the multi-role IK + yam mapper for the given active roles.

    ThreePointIK requires the active roles to be a prefix of (left, right, head) with
    'left' first. All modern (LEFT-canonical) zarrs satisfy this. A legacy right-primary
    zarr would need RightHandIK; that path is out of scope here (the 3pt training data is
    LEFT-canonical) and is reported as unsupported rather than mis-driven."""
    from g1_right_hand_ik import YamConfigMapper, yam_actuated_joint_names
    from g1_three_point_ik import ThreePointIK

    ik = ThreePointIK(config_name=config_name, reach_scale=reach_scale, active_roles=tuple(active_roles))
    mapper = YamConfigMapper(ik._humi_joint_names, yam_actuated_joint_names(yam_urdf))
    return ik, mapper


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("zarr_path", help="a converted UMI .zarr.zip (from convert_to_umi.py)")
    p.add_argument("--checkpoint", default="",
                   help="checkpoint dir (model.safetensors + config.json [+ norm_stats.json]); "
                        "empty = random-init from configs/dp_flow_3pt.py (pipeline check only)")
    p.add_argument("--episode", type=int, default=0, help="episode index to load")
    p.add_argument("--device", default="cuda", help="torch device (cuda|cpu); falls back to cpu")
    p.add_argument("--num-denoising-steps", type=int, default=None,
                   help="flow-matching Euler steps (default: model config inference_steps)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8086)
    p.add_argument("--yam-urdf", default=_DEFAULT_YAM)
    p.add_argument("--tcp-extrinsic", default="",
                   help="tcp_extrinsic.json override (T_tracker_tcp); default = the repo "
                        "calibration file the converter used. Only used when the zarr is "
                        "tcp_calibrated.")
    p.add_argument("--ik-config", default="proposal")
    p.add_argument("--reach-scale", type=float, default=1.0)
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04))
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0))
    p.add_argument("--cam-fov-deg", type=float, default=58.0)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.exists(args.zarr_path):
        raise SystemExit(f"[rollout-viz] not found: {args.zarr_path}")

    # --- resolve device (cpu fallback) ---
    import torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[rollout-viz] CUDA unavailable; falling back to cpu.", flush=True)
        device = "cpu"

    # --- load episode + model ---
    ep = Episode(args.zarr_path, args.episode)
    print(f"[rollout-viz] {os.path.basename(args.zarr_path)}: episode {ep.ep}/{ep.n_ep - 1}, "
          f"{ep.T} frames, mode={ep.tracking_mode}, primary={ep.primary_role}, "
          f"keypoints={[n for n, _ in ep.kp.items()]}", flush=True)
    if ep.primary_role != "left":
        raise SystemExit(
            "[rollout-viz] this tool drives the humanoid via ThreePointIK (LEFT-canonical, "
            f"'left' primary), but the zarr is primary_role={ep.primary_role!r} (legacy). "
            "Re-record/convert LEFT-canonical, or extend this tool with RightHandIK.")

    # Auto-detect the checkpoint family. An HL-v2 checkpoint (non-empty layout_mode, flow
    # OR diffusion) uses the real HLDataset for its observation + a per-hand 16-dim action;
    # a legacy/empty checkpoint uses the 32-dim UMIDataset-style path.
    is_hl, hl_algo, hl_mode = _detect_hl_checkpoint(args.checkpoint)
    if is_hl:
        # HL modes name TRACKERS; this zarr must carry them. left_only needs only 'left'
        # (== primary), which the primary_role=='left' guard above already ensured.
        lm = HLLoadedModel(args.checkpoint, hl_algo, hl_mode, args.zarr_path, device,
                           args.num_denoising_steps)
    else:
        lm = LoadedModel(args.checkpoint, device, args.num_denoising_steps)
    if lm.obs_history_len > ep.T:
        raise SystemExit(f"[rollout-viz] obs_history_len={lm.obs_history_len} exceeds episode "
                         f"length {ep.T}; pick a longer episode.")

    # A shared reference frame reads its base pose out of that keypoint's 32-dim slot, so
    # the keypoint MUST exist in this zarr — otherwise resolve_base reads an all-zero
    # (absent) slot and hands a degenerate zero-rotation base to every SE(3) transform,
    # SILENTLY collapsing the whole rollout (and the state) to the origin. The training
    # dataset guards this same case loudly (UMIDataset.__init__); replicate it here since
    # this tool bypasses both the dataset and the serving loader.
    if lm.ref_frame in ("left", "right", "head") and lm.ref_frame not in ep.kp:
        raise SystemExit(
            f"[rollout-viz] checkpoint ref_frame={lm.ref_frame!r} but keypoint "
            f"{lm.ref_frame!r} is absent in this zarr (present: {sorted(ep.kp)}). Cannot "
            f"anchor a shared frame on a missing keypoint — load an episode/zarr that has "
            f"it, or use a per_keypoint/world checkpoint.")

    # Active IK roles = the recorded keypoints, as a prefix of (left, right, head).
    present = {name for name, _ in ep.kp.items()}
    active_roles = tuple(r for r in ("left", "right", "head") if r in present)
    prim_role = ep.primary_kp
    other_hand_role = "right" if prim_role == "left" else "left"
    other_hand_role = other_hand_role if other_hand_role in present else None
    # Hands ('left'/'right') carry the tracker->TCP extrinsic in the zarr; 'head' does not.
    hand_roles = {r for r in active_roles if r in ("left", "right")}

    # Recover the tracker->TCP extrinsic so the IK is driven from tracker poses (the zarr
    # stores TCP poses for the hands). Uses the SAME loader + repo-default the converter
    # used; identity when the zarr wasn't TCP-calibrated (then TCP == tracker, no-op).
    inv_ext = _load_inv_tcp_extrinsic(args.zarr_path, ep, args.tcp_extrinsic)

    print(f"[rollout-viz] building IK ({args.ik_config}, roles={active_roles})…", flush=True)
    ik, mapper = _make_ik(active_roles, args.ik_config, args.reach_scale, args.yam_urdf)

    # --- pre-solve the RECORDED trajectory IK once (whole episode) ---
    def kp_poses_recorded(frames: np.ndarray):
        return {r: (ep.kp[r][0][frames], ep.kp[r][1][frames]) for r in active_roles}

    prim_grip_rec = ep.kp[prim_role][3]
    other_grip_rec = ep.kp[other_hand_role][3] if other_hand_role else None
    all_frames = np.arange(ep.T)
    print(f"[rollout-viz] pre-solving RECORDED IK over {ep.T} frames…", flush=True)
    rec_cfgs, rec_wrist_pos, rec_wrist_wxyz, rec_paths, rec_q_seq = presolve_trajectory(
        ik, mapper, active_roles, kp_poses_recorded(all_frames),
        prim_role, prim_grip_rec, other_hand_role, other_grip_rec, inv_ext, hand_roles)
    print("[rollout-viz] recorded IK done.", flush=True)

    # Recorded per-keypoint paths for the blue-gradient lines are the SOLVED robot-frame
    # site positions (rec_paths from the IK), NOT raw zarr world coords — so each blue line
    # sits exactly on the rendered robot hand/torso it drove.

    # --- scene ---
    server = viser.ViserServer(host=args.host, port=args.port, label="FLoRe rollout viz")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.25)

    base_xyz = np.asarray(ik.home_q()[0:3], dtype=float).copy()
    server.scene.add_frame("/robot", show_axes=False, position=tuple(base_xyz))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")

    # Camera frustum + F/T arrows ride the primary wrist (like the raw replay tool).
    follow = server.scene.add_frame("/follow", show_axes=True, axes_length=0.08,
                                    axes_radius=0.004, origin_radius=0.01)
    rpy = np.asarray(args.cam_rpy, float)
    cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
    server.scene.add_frame("/follow/cam", show_axes=False,
                           position=tuple(np.asarray(args.cam_offset, float)),
                           wxyz=tuple(cam_rot.wxyz))
    frustum = server.scene.add_camera_frustum(
        "/follow/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=1.0, scale=0.12,
        color=(40, 200, 255))
    wrench = rc.WrenchArrows(server, name="/follow/wrench") if ep.has_ft else None

    # Draw the recorded blue-gradient paths (static). Keep handles for visibility toggling.
    # Only paths with >=2 points are "real"; a degenerate (<2 pt) path is drawn hidden and
    # must NOT be force-shown by the toggle loop below (so keep just the real ones).
    rec_path_handles = {r: draw_gradient_path(server, f"/recorded/{r}", rec_paths[r])
                        for r in active_roles if rec_paths[r].shape[0] >= 2}
    roll_path_handles: dict = {}

    # --- rollout state (mutable; produced on a Rollout click, consumed on the loop) ---
    # source: which trajectory the timeline currently shows ("recorded"|"rollout").
    # req_anchor / pending / pending_source: cross-thread REQUESTS set by GUI callbacks
    # (never read back as the result); applied on the loop thread in before_render.
    # shown_anchor: the anchor the CURRENTLY-CACHED rollout was computed at — written only
    # on the loop thread, so render_at reads a value consistent with cfgs/paths/ts. Kept
    # separate from req_anchor so a re-click mid-compute can't stomp the shown result.
    roll: dict = {
        "cfgs": None, "wrist_pos": None, "wrist_wxyz": None, "ts": None,
        "paths": None, "prim_grip": None, "other_grip": None,
        "req_anchor": 0, "shown_anchor": 0, "pending": False,
        "source": "recorded", "pending_source": None,
    }

    # --- GUI ---
    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")
    with gui.add_folder("Dataset"):
        gui.add_text("File", initial_value=os.path.basename(args.zarr_path), disabled=True)
        gui.add_text("Episode", initial_value=f"{ep.ep} / {ep.n_ep - 1}  ({ep.T} fr)", disabled=True)
        gui.add_text("Mode", initial_value=f"{ep.tracking_mode} (primary={ep.primary_role})", disabled=True)
    with gui.add_folder("Model"):
        gui.add_text("Checkpoint",
                     initial_value=("RANDOM-INIT" if lm.random_init else os.path.basename(args.checkpoint.rstrip("/"))),
                     disabled=True)
        gui.add_text("ref_frame", initial_value=lm.ref_frame, disabled=True)
        gui.add_text("obs_history_len", initial_value=str(lm.obs_history_len), disabled=True)
        gui.add_text("horizon / steps",
                     initial_value=f"H={lm.action_horizon}  denoise={lm.num_denoising_steps}", disabled=True)
    with gui.add_folder("Rollout"):
        start_slider = gui.add_slider("Start frame", min=0, max=max(ep.T - 1, 0), step=1, initial_value=0)
        rollout_btn = gui.add_button("▶ Rollout from start frame")
        show_dd = gui.add_dropdown("Showing", ("recorded", "rollout"), initial_value="recorded")
        roll_status = gui.add_text("Status", initial_value="(no rollout yet)", disabled=True)
    with gui.add_folder("Frame readout"):
        rd_t = gui.add_text("t (s, rel)", initial_value="0.000", disabled=True)
        rd_src = gui.add_text("trajectory", initial_value="recorded", disabled=True)
        rd_prim_pos = gui.add_text(f"{prim_role} xyz", initial_value="-", disabled=True)
        rd_prim_grip = gui.add_text(f"{prim_role} gripper", initial_value="-", disabled=True)
        if other_hand_role is not None:
            rd_other_grip = gui.add_text(f"{other_hand_role} gripper", initial_value="-", disabled=True)
        else:
            rd_other_grip = None
        rd_force = gui.add_text("|F| (N)", initial_value="-", disabled=True)
        rd_torque = gui.add_text("|T| (Nm)", initial_value="-", disabled=True)
    with gui.add_folder("View"):
        show_cam = gui.add_checkbox("Show camera (recorded obs)", initial_value=True)
        show_wrench = gui.add_checkbox("Show F/T arrows", initial_value=bool(ep.has_ft))
        show_recorded = gui.add_checkbox("Show recorded (blue) paths", initial_value=True)
        show_rollout = gui.add_checkbox("Show rollout (orange) path", initial_value=True)
    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, f"{prim_role} gripper width (m)", "width",
                                          color_idx=1, y_range=(0.0, max(_JAW_OPEN_M * 2.2, 0.1)))
        prim_grip_m_rec = (gripper_norm_to_width(prim_grip_rec) if prim_grip_rec is not None
                           else np.zeros(ep.T))
        rc.set_full_single(grip_chart, ep.ts, prim_grip_m_rec)
        if ep.has_ft:
            force_chart = rc.make_xyz_chart(gui, "Force (N)", ("Fx", "Fy", "Fz"))
            torque_chart = rc.make_xyz_chart(gui, "Torque (Nm)", ("Tx", "Ty", "Tz"))
            rc.set_full_xyz(force_chart, ep.ts, ep.ft[:, 0:3])
            rc.set_full_xyz(torque_chart, ep.ts, ep.ft[:, 3:6])

    # ------------------------------------------------------------------
    # Rollout click -> flag only (GUI thread). The loop thread runs the (slow) model +
    # IK solve so the render loop / scene is never mutated from a callback thread.
    # ------------------------------------------------------------------
    @rollout_btn.on_click
    def _(_e) -> None:
        # Callback thread: record the REQUEST only (req_anchor is callback-owned; the loop
        # thread never writes it back, so a re-click during a slow compute is honored, not
        # clobbered). The actual compute runs in before_render on the loop thread.
        req = int(start_slider.value)
        roll["req_anchor"] = req
        roll["pending"] = True
        roll_status.value = f"rolling out from frame {req}…"

    # Set True by the loop thread around a programmatic show_dd.value write so the
    # synchronous on_update echo (viser runs value-setter callbacks inline) is ignored —
    # same discipline as Timeline._set_slider's _suppress.
    _dd_suppress = {"on": False}

    @show_dd.on_update
    def _(_e) -> None:
        # GUI thread (or a suppressed loop-thread echo): only request the switch. The loop
        # thread retargets the timeline (a series swap must not race an in-flight render).
        if _dd_suppress["on"]:
            return
        roll["pending_source"] = str(show_dd.value)

    def _set_dropdown(value: str) -> None:
        """Programmatically set the dropdown WITHOUT triggering a pending_source echo."""
        _dd_suppress["on"] = True
        try:
            show_dd.value = value
        finally:
            _dd_suppress["on"] = False

    def _retarget_timeline(source: str) -> None:
        """Point the shared Timeline at the series for ``source`` (loop thread only).

        Recorded uses the episode's own relative timestamps (T frames); rollout uses a
        0..(H-1) axis at the recorded median dt (H frames). Keeps the scrubber length in
        lockstep with whichever trajectory ``render_at`` will index. A no-op (no series
        swap, no cursor reset) when the source doesn't actually change."""
        want = "rollout" if (source == "rollout" and roll["cfgs"] is not None) else "recorded"
        if want == roll["source"]:
            # Already showing this trajectory; keep the dropdown honest (e.g. user picked
            # 'rollout' with none computed -> snap it back to 'recorded') and don't reset.
            _set_dropdown(want)
            return
        roll["source"] = want
        timeline.set_series(roll["ts"] if want == "rollout" else ep.ts)
        _set_dropdown(want)

    def compute_rollout(anchor: int) -> None:
        """Slow path (loop thread): sample the chunk, un-relativize, pre-solve the
        rollout IK, cache everything for the timeline."""
        # HL-v2 samples via the real HLDataset + per-hand un-relativization; the legacy
        # path replicates UMIDataset preprocessing. Both return a (H, 32) chunk_abs so the
        # IK + rendering below is identical.
        if is_hl:
            chunk_abs = build_hl_batch_and_sample(lm, ep, anchor)  # (H, 32) absolute
        else:
            chunk_abs = build_batch_and_sample(lm, ep, anchor)  # (H, 32) absolute
        H = chunk_abs.shape[0]
        # Per-keypoint rollout TCP/world poses (fed to IK); the ORANGE line is drawn from
        # the SOLVED site positions returned below, not these raw coords.
        kp_poses = {}
        for name, slot in _KEYPOINTS:
            if name not in present:
                continue
            pos_sl, rot_sl, _grip_i = slot
            kp_poses[name] = (chunk_abs[:, pos_sl], chunk_abs[:, rot_sl])
        # Gripper (normalized) per hand, clipped to [0,1] (model can over/undershoot).
        prim_grip = np.clip(chunk_abs[:, _LEFT[2] if prim_role == "left" else _RIGHT[2]], 0.0, 1.0)
        other_grip = None
        if other_hand_role is not None and other_grip_rec is not None:
            og_i = _LEFT[2] if other_hand_role == "left" else _RIGHT[2]
            other_grip = np.clip(chunk_abs[:, og_i], 0.0, 1.0)

        # Share the recorded run's EXACT calibration: use the RECORDED tracker poses at
        # FRAME 0 as the home anchor (the same reference the recorded presolve used). The
        # home-anchored IK maps target = home + (tracker - ref), so with a common ref both
        # runs live in one mapping: the model's step-0 pose (~the recorded anchor pose)
        # lands the orange line right where the blue line is at the anchor — a true branch,
        # no teleport on a 'Showing' switch. (Calibrating at the anchor instead would map
        # the anchor->home and collapse orange[0] to the home pose — wrong.) Seeding the
        # solver from the recorded arm's SOLVED config at the anchor makes the branch tight
        # from step 0 (else the velocity-capped IK creeps out from home over ~40 frames).
        calib_ref = _tracker_poses_at(ep, active_roles, hand_roles, inv_ext, 0)
        cfgs, wpos, wwxyz, paths, _q = presolve_trajectory(
            ik, mapper, active_roles, {r: kp_poses[r] for r in active_roles},
            prim_role, prim_grip, other_hand_role, other_grip, inv_ext, hand_roles,
            calib_ref=calib_ref, seed_q=rec_q_seq[anchor])
        # Rollout uses its own 0..(H-1) relative time axis (dt from the recorded grid).
        # Force "recorded" first so _retarget_timeline("rollout") always sees a real
        # source change and swaps the series (guards the case where a previous rollout
        # already left source=="rollout").
        dt = float(np.median(np.diff(ep.ts))) if ep.T > 1 else 0.1
        roll["source"] = "recorded"
        # Single bulk update AFTER all slow work, on the loop thread: render_at (same
        # thread) can only observe the fully-updated result, never a torn cfgs/ts/anchor.
        # shown_anchor (not the callback-owned req_anchor) records what this chunk used.
        roll.update(cfgs=cfgs, wrist_pos=wpos, wrist_wxyz=wwxyz, paths=paths,
                    prim_grip=prim_grip, other_grip=other_grip, shown_anchor=anchor,
                    ts=np.arange(H) * dt)
        # Draw the orange rollout path for every keypoint (keep only real >=2-pt handles
        # so the toggle loop never force-shows a degenerate placeholder line).
        for name in paths:
            if paths[name].shape[0] < 2:
                draw_solid_path(server, f"/rollout/{name}", paths[name], _ROLLOUT_ORANGE)
                continue
            roll_path_handles[name] = draw_solid_path(
                server, f"/rollout/{name}", paths[name], _ROLLOUT_ORANGE)
            roll_path_handles[name].visible = show_rollout.value
        # Auto-switch the view to the rollout + retarget the timeline to H frames. This
        # runs on the loop thread (before_render), so touching the timeline is safe.
        _retarget_timeline("rollout")
        roll_status.value = (f"rollout @ frame {anchor}: H={H} steps"
                             + ("  [RANDOM-INIT — not meaningful]" if lm.random_init else ""))

    # ------------------------------------------------------------------
    # Render: index either the recorded episode or the cached rollout chunk.
    # ------------------------------------------------------------------
    _last = {"cam_i": -1}

    def render_at(i: int) -> None:
        src = roll["source"]
        have_roll = roll["cfgs"] is not None
        use_roll = (src == "rollout") and have_roll

        if use_roll:
            cfgs, wpos, wwxyz, ts = roll["cfgs"], roll["wrist_pos"], roll["wrist_wxyz"], roll["ts"]
        else:
            cfgs, wpos, wwxyz, ts = rec_cfgs, rec_wrist_pos, rec_wrist_wxyz, ep.ts
        # Clamp to the active trajectory's own length (cfgs / paths / ts are all the same
        # length per source), so a momentary source/timeline desync can't index OOB.
        i = int(np.clip(i, 0, cfgs.shape[0] - 1))
        urdf.update_cfg(cfgs[i])
        follow.position = tuple(wpos[i])
        follow.wxyz = tuple(wwxyz[i])

        # Camera frustum: always the RECORDED observation image (the model conditions on
        # recorded frames; a rollout has no real image). Anchor for the rollout view is
        # the recorded frame closest to the rollout step. Uses shown_anchor (the anchor
        # the cached chunk was actually computed at — loop-thread-written), NOT the
        # callback-owned req_anchor, so a re-click never misaligns the frustum.
        rec_i = i if not use_roll else int(np.clip(roll["shown_anchor"] + i, 0, ep.T - 1))
        frustum.visible = show_cam.value
        if show_cam.value and rec_i != _last["cam_i"]:
            _last["cam_i"] = rec_i
            img = ep.image_at(rec_i)
            h, w = img.shape[:2]
            frustum.aspect = w / max(h, 1)
            frustum.image = np.asarray(img)

        # F/T arrows: the RECORDED wrench on the primary wrist. Shown only in the
        # recorded view — drawing recorded force on the PREDICTED wrist pose would be a
        # real signal on an imagined pose. The F/T CHARTS (always recorded) stay the
        # honest reference in both views; readouts below are labeled accordingly.
        if wrench is not None:
            ft_arrows_on = show_wrench.value and ep.has_ft and not use_roll
            wrench.set_visible(ft_arrows_on)
            if ft_arrows_on and ep.ft is not None:
                wrench.update(wpos[i], wwxyz[i], ep.ft[rec_i, 0:3], ep.ft[rec_i, 3:6])

        # Readouts. Use the ACTIVE trajectory's own time axis (index-safe: same length
        # as cfgs above), not timeline.times, which can momentarily lag a source switch.
        rd_t.value = f"{(ts[i] - ts[0]) if len(ts) else 0.0:.3f}"
        rd_src.value = "rollout (orange)" if use_roll else "recorded (blue)"
        if use_roll:
            prim_xyz = roll["paths"][prim_role][i]
            prim_g = roll["prim_grip"][i] if roll["prim_grip"] is not None else None
            other_g = roll["other_grip"][i] if roll["other_grip"] is not None else None
        else:
            prim_xyz = rec_paths[prim_role][i]
            prim_g = prim_grip_rec[i] if prim_grip_rec is not None else None
            other_g = other_grip_rec[i] if other_grip_rec is not None else None
        rd_prim_pos.value = np.array2string(np.asarray(prim_xyz), precision=3, suppress_small=True)
        if prim_g is not None:
            gw = float(gripper_norm_to_width(prim_g))
            rd_prim_grip.value = f"{gw * 1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'})"
        if rd_other_grip is not None and other_g is not None:
            gw = float(gripper_norm_to_width(other_g))
            rd_other_grip.value = f"{gw * 1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'})"
        if ep.ft is not None:
            rd_force.value = f"{np.linalg.norm(ep.ft[rec_i, 0:3]):.2f}"
            rd_torque.value = f"{np.linalg.norm(ep.ft[rec_i, 3:6]):.3f}"

    timeline = rc.Timeline(server, ep.ts, render_at, folder_label="Playback")

    # Last-seen checkbox values for the render_at-gated toggles (camera + F/T arrows).
    # Those two are applied INSIDE render_at, which the loop skips while paused on an
    # unchanged frame — so a toggle while paused would be dead until the next scrub. We
    # poll them here and request a one-shot re-render on change (mirrors replay_raw_viz's
    # poll_ft_mode). The blue/orange PATH toggles are immune (set directly every tick).
    _view_last = {"cam": bool(show_cam.value), "wrench": bool(show_wrench.value)}

    def before_render() -> None:
        # Apply path-visibility toggles via the cached scene handles (cheap; viser
        # ignores a set to the same value).
        for h in rec_path_handles.values():
            h.visible = bool(show_recorded.value)
        for h in roll_path_handles.values():
            h.visible = bool(show_rollout.value)
        # Force a repaint when a render_at-gated toggle changed while paused.
        if bool(show_cam.value) != _view_last["cam"] or bool(show_wrench.value) != _view_last["wrench"]:
            _view_last["cam"] = bool(show_cam.value)
            _view_last["wrench"] = bool(show_wrench.value)
            timeline.request_render()
        # Consume a pending rollout on the loop thread (never in the click callback). Read
        # the callback-owned req_anchor into a local BEFORE clearing pending, so a click
        # landing in this window can't make us compute the wrong (or a duplicate) anchor.
        if roll["pending"]:
            anchor = int(roll["req_anchor"])
            roll["pending"] = False
            try:
                compute_rollout(anchor)
            except Exception as e:  # noqa: BLE001 — surface, keep the loop alive
                roll_status.value = f"rollout failed: {e!r}"
                print(f"[rollout-viz] rollout error: {e!r}", flush=True)
        # Consume a pending "Showing" source switch (loop thread retargets the timeline
        # so the scrubber length matches the trajectory render_at will index).
        elif roll["pending_source"] is not None:
            src = roll["pending_source"]
            roll["pending_source"] = None
            _retarget_timeline(src)

    timeline.before_render = before_render
    print(f"[rollout-viz] http://{args.host}:{args.port}  — pick a Start frame, click Rollout, "
          f"toggle 'Showing' recorded/rollout", flush=True)
    try:
        timeline.run()
    finally:
        # Release the HL model's HLDataset ZipStore handle on exit (Ctrl-C or return).
        # Legacy LoadedModel holds no such handle, so guard on the method.
        if hasattr(lm, "close"):
            lm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
