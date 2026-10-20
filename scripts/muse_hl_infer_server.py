#!/usr/bin/env python3
"""MuSe-on-UVA HL-v2 model BACKEND — the torch/hydra half of ``--algo muse``.

Runs INSIDE the UVA model env (the ``uvaforce`` conda env: torch 2.7.1+cu128, hydra,
diffusers — and NO viser/mink). Its ONLY job is "one observation in, one action+force chunk
out". Every bit of rig logic — the pelvis frame, the TCP lever, the ``:8770`` wire, the force
gate, the recorder, reach compensation — stays in the FLoRe-side planner
(``scripts/hl_serving.py`` + ``flore/s1_policy/serving/hl_muse.py``), which runs in FLoRe's own
``.venv``. The two environments are mutually exclusive; that hard split is the whole reason
this is a subprocess talking over a pipe rather than an import.

Contrast with its two older siblings, both of which stay untouched:
  * ``muse_force_infer_server.py``  — offline rollout backend for ``uva_umi_force`` (16 steps).
  * ``muse_force_serving_server.py`` — a WHOLE online planner in the backend, with its own
    wire mapper and anchor bookkeeping.
This one is deliberately stateless and model-only, so the HL-v2 planner stays the single
owner of the deploy contract.

Wire protocol
-------------
:mod:`flore.utils.muse_wire` (numpy-only, imported from FLoRe's source tree so there is exactly
ONE copy of the codec). ``stdout`` is a PURE protocol channel: hydra, MAR and the VAE print a
lot, so ``sys.stdout`` is pointed at stderr at import time and protocol lines go to the
stashed real fd 1.

Two request kinds
-----------------
``live``  the FLoRe planner sends a FULLY RELATIVIZED observation assembled from the rig's
          sensor rings (anchor-relative pose history, ``wrt_start`` rotation, gripper metres,
          PHYSICAL wrench, 0-1 CHW images). We lay it into the length-``2H`` window the model
          was trained on, apply the ``÷ft_scale`` input pre-scale, and sample.
``zarr``  offline inference: we build the window from the TRAINING dataset at an explicit
          ``(episode, anchor)``, which is the eval path step for step, and additionally return
          the anchor's ABSOLUTE pose so the FLoRe side can cross-check that its own dataset's
          anchor agrees (that check is what validates the frame contract end to end).

Both go through the same ``prepare_data_predict_action`` + ``sample_tokens(task_mode=
'policy_model')`` + ``unnormalize_future_action`` path, so the only difference between them is
where the observation came from.

Window layout (read before changing anything about shapes)
----------------------------------------------------------
``H = model.policy.num_actions`` (32 for ``uva_hl_force``). The dataset window is ``2H`` rows
and ``process_data`` chunks it in half, so the HISTORY half is rows ``[0..H-1]`` == steps
``[-(H-1)..0]``. Then:
  * with ``different_history_freq``, the proprio streams are GATHERED with the first half of
    ``img_indices`` — positional slots inside that history half. For ``uva_hl_force`` those are
    ``[-12,-8,-4,0] + (H-1) = [19,23,27,31]``, i.e. 4 frames 133 ms apart ending at now.
  * the force encoder keeps the NEWEST ``force_history_length`` (8) rows of the history half.
  * images are ``2*n_cond`` frames; ``get_vae_latent`` does ``torch.chunk(x, 2, dim=2)`` and
    the action head conditions on the FIRST half only, so the conditioning frames must be the
    RECENT ones ending at now and the target half is inert filler. (Getting that backwards
    conditions the arm on the OLDEST frames and is invisible in every shape check — it was a
    real bug in the older serving backend.)

With no ``--checkpoint`` the model is random-initialised from the config: a wiring check only,
and the handshake says ``random_init`` so the FLoRe side can refuse to arm.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

# --- keep fd 1 (stdout) clean: model/hydra/VAE chatter must go to fd 2 --------
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLORE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UVA = os.path.join(_FLORE_ROOT, "third_party", "unified_video_action")

# FLoRe root on sys.path up front: the wire codec and the rot6d helpers live there and are
# pure numpy (flore/utils/__init__.py is empty), so this env can import them without pulling
# in torch-2.13 / grpc / zarr. Same trick muse_force_serving_server.py already relies on.
if _FLORE_ROOT not in sys.path:
    sys.path.insert(0, _FLORE_ROOT)

from flore.utils import muse_wire as wire  # noqa: E402
from flore.utils.rot6d import rotmat_to_rot6d, rotvec_to_rotmat  # noqa: E402


def _emit(tag: str, meta: dict | None = None, **arrays: np.ndarray) -> None:
    _REAL_STDOUT.write(wire.pack(tag, meta, **arrays) + "\n")
    _REAL_STDOUT.flush()


def _log(msg: str) -> None:
    print(f"[muse-hl] {msg}", file=sys.stderr, flush=True)


def _rotvec_to_rot6d(rotvec) -> np.ndarray:
    return rotmat_to_rot6d(rotvec_to_rotmat(np.asarray(rotvec, dtype=np.float64)))


class _ZeroNormalRNG:
    """A ``numpy.random.Generator`` proxy whose ``normal`` returns zeros.

    Used to disable the training dataset's additive start-pose noise for an inference window
    (see :meth:`Backend._neutralize_startpose_noise`). A proxy rather than a monkey-patch
    because ``Generator`` is a C type with read-only method attributes.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):  # every other draw (notably `choice`) is untouched
        return getattr(self._inner, name)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is not None:
            return np.zeros(size, dtype=np.float64)
        # No `size`: numpy's own shape comes from broadcasting loc/scale, and two scalars give
        # a plain float. Match both so a caller that indexes the result still works.
        shape = np.broadcast_shapes(np.shape(loc), np.shape(scale))
        return np.zeros(shape, dtype=np.float64) if shape else 0.0


# ===========================================================================
# Model + (optional) dataset
# ===========================================================================
class Backend:
    """Loads the UVA HL force policy and answers ``live`` / ``zarr`` requests.

    Reuses the SAME defensive load checks as ``muse_force_infer_server.py`` (EMA weights
    really populated, frozen VAE really restored, ``ft_scale`` consistent across the three
    config nodes that interpolate it), because each of them turns a silently-degraded model
    into a loud failure: ``load_payload`` forwards ``strict=False`` to ``load_state_dict``,
    which NEVER raises and whose ``_IncompatibleKeys`` result is discarded, so a renamed key
    would leave a real parameter at random init with no error at all.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str,
        config_name: str,
        dataset_root: str,
        zarr_path: str,
        model_size: str,
        force_fusion: str | None,
        need_dataset: bool,
    ) -> None:
        # Resolve user paths to ABSOLUTE *before* the chdir below: __init__ does
        # os.chdir(_UVA) (the UVA config/pretrained paths are relative to it), so a relative
        # --checkpoint would otherwise be looked up against _UVA and vanish.
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint)) if checkpoint else ""
        zarr_path = os.path.abspath(os.path.expanduser(zarr_path)) if zarr_path else ""
        if dataset_root:
            dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
        elif zarr_path:
            dataset_root = os.path.dirname(zarr_path)

        sys.path.insert(0, _UVA)
        os.chdir(_UVA)

        import hydra
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf, open_dict

        self.torch = torch
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        if device.startswith("cuda") and not torch.cuda.is_available():
            _log("CUDA unavailable; falling back to cpu.")
            device = "cpu"
        self.device = device
        self.random_init = not checkpoint
        self.need_dataset = bool(need_dataset)
        self._want_zarr = zarr_path
        self._force_invalid_n = 0

        overrides = [
            "task.dataset.dataloader_cfg.batch_size=1",
            "task.dataset.dataloader_cfg.num_workers=0",
            f"model.policy.autoregressive_model_params.model_size={model_size}",
        ]
        if dataset_root:
            overrides.append(f"task.dataset.dataset_root_dir={dataset_root}")
        if force_fusion:
            overrides.append(f"model.policy.force_fusion={force_fusion}")

        def pin_single_zarr(config) -> None:
            """Collapse ``dataset_configs`` to the ONE zarr the caller asked for.

            ``uva_hl_force`` co-trains over five sessions, so the default would open five zarr
            handles AND make ``datasets[0]`` — which every ``zarr`` request indexes — whichever
            member happened to be first. That is not merely wasteful: it silently answers with
            windows from a DIFFERENT session than the one the FLoRe side is comparing against,
            which the frontend then catches as an anchor mismatch. Pin it instead.
            """
            if not zarr_path:
                return
            with open_dict(config):
                config.task.dataset.dataset_configs = {
                    "serve": {
                        "zarr_path": os.path.basename(zarr_path),
                        "include_episode_num": -1,
                        "mask_mirror": False,
                    }
                }

        payload = None
        if self.random_init:
            overrides.append(
                "model.policy.autoregressive_model_params.pretrained_model_path=null")
            cfg_dir = pathlib.Path(_UVA) / "unified_video_action" / "config"
            with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides)
            pin_single_zarr(cfg)
            OmegaConf.resolve(cfg)
            self.cfg = cfg
            policy = hydra.utils.instantiate(
                cfg.model.policy, task_name=cfg.task.name,
                task_modes=cfg.task.task_modes,
                normalizer_type=cfg.task.dataset.normalizer_type,
                language_emb_model=cfg.task.dataset.language_emb_model)
            _log(f"no --checkpoint: RANDOM-INIT UVA model from {config_name}.yaml "
                 "(wiring check only — every chunk is meaningless).")
        else:
            from unified_video_action.workspace.train_unified_video_action_workspace import (  # noqa: E501
                TrainUnifiedVideoActionWorkspace,
            )

            payload = torch.load(open(checkpoint, "rb"), map_location="cpu", weights_only=False)
            cfg = payload["cfg"]
            with open_dict(cfg):
                if dataset_root:
                    cfg.task.dataset.dataset_root_dir = dataset_root
                cfg.task.dataset.dataloader_cfg.batch_size = 1
                cfg.task.dataset.dataloader_cfg.num_workers = 0
            pin_single_zarr(cfg)
            OmegaConf.resolve(cfg)
            self.cfg = cfg
            ws = TrainUnifiedVideoActionWorkspace(cfg)
            # strict=False is REQUIRED for the slim inference checkpoint (it drops the frozen
            # `vae_model.*` weights and the redundant non-EMA `model`). It validates NOTHING
            # on its own — _assert_weights_loaded below is what does.
            ws.load_payload(payload, strict=False)
            policy = ws.ema_model if ws.ema_model is not None else ws.model
            _log(f"loaded checkpoint {checkpoint}")

        policy.to(device).eval()
        self.policy = policy
        self.checkpoint = checkpoint

        if payload is not None:
            self._assert_weights_loaded(policy, payload)
            self._assert_vae_loaded()
        self._assert_ft_scale_consistent()

        # --- the grid this checkpoint speaks ---------------------------------
        pol = self.cfg.model.policy
        self.num_actions = int(pol.num_actions)
        self.action_dim = int(self.cfg.task.shape_meta.action.shape[0])
        self.force_dim = int(pol.get("force_dim", 6))
        self.use_force = bool(pol.get("use_force", False))
        self.predict_force = bool(pol.get("predict_force", False))
        self.force_history_length = int(pol.get("force_history_length", 16))
        self.n_action_steps = int(pol.get("n_action_steps", 1))
        self.obs_pose_repr = str(self.cfg.task.pose_repr.obs_pose_repr)
        self.action_pose_repr = str(self.cfg.task.pose_repr.action_pose_repr)
        self.down_sample_steps = int(self.cfg.task.dataset.down_sample_steps)
        self.ft_scale = np.asarray(
            list(self.cfg.task.get("ft_scale", [1.0] * self.force_dim)), dtype=np.float64)
        self.img_indices, self.n_img_cond = self._resolve_img_indices()
        # `2H`, not `task.action_horizon`: the two are equal in every shipped config, but the
        # thing that actually matters is that torch.chunk(nactions, 2) yields an H-row history
        # and an H-row future, and only `num_actions` states H.
        self.window_len = 2 * self.num_actions
        self.epoch, self.global_step = self._resolve_provenance(payload)

        # --- optional dataset (offline `zarr` requests only) ----------------
        self.lazy = None
        self.episode_starts = np.zeros(0, dtype=np.int64)
        self.episode_frame_nums = np.zeros(0, dtype=np.int64)
        self.train_episodes = np.zeros(0, dtype=np.int64)
        self.zarr_path = ""
        if self.need_dataset:
            self._build_dataset(hydra)

    # --- load-time guards -------------------------------------------------
    @staticmethod
    def _assert_weights_loaded(policy, payload) -> None:
        """Re-verify that the EMA weights really populated the model.

        ``base_workspace.load_payload`` discards ``load_state_dict``'s ``_IncompatibleKeys``
        and ``strict=False`` never raises, so without this a future retrained slim checkpoint
        with a renamed key would silently serve a partially random-init model.
        """
        weights = payload["state_dicts"].get("ema_model") or payload["state_dicts"]["model"]
        result = policy.load_state_dict(weights, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("vae_model")]
        if result.unexpected_keys or missing:
            raise RuntimeError(
                "checkpoint weights do not match the constructed model: "
                f"{len(result.unexpected_keys)} unexpected key(s) "
                f"{result.unexpected_keys[:5]}, {len(missing)} missing non-VAE key(s) "
                f"{missing[:5]}. Serving a partially random-init model would look healthy and "
                "produce garbage, so this is fatal.")

    def _assert_vae_loaded(self) -> None:
        """The frozen KL-16 VAE is restored from its own file at construction; if that file
        was absent, AutoencoderKL keeps a RANDOM VAE and every rollout is garbage with no
        error at all. Check the file the config points at actually exists."""
        path = str(self.cfg.model.policy.vae_model_params.autoencoder_path)
        resolved = path if os.path.isabs(path) else os.path.join(_UVA, path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"the frozen VAE checkpoint {resolved} is missing. Without it the model keeps "
                "a RANDOMLY INITIALISED VAE and every prediction is noise, silently.")

    def _assert_ft_scale_consistent(self) -> None:
        """``ft_scale`` appears in three config nodes (``task``, ``task.dataset``,
        ``model.policy``), all ``${task.ft_scale}`` interpolations. Serving multiplies the
        predicted wrench by ONE of them, so an override of a single node would silently
        desync physical force units from what the model was trained on."""
        seen = {}
        for name, node in (("task", self.cfg.task),
                           ("task.dataset", self.cfg.task.dataset),
                           ("model.policy", self.cfg.model.policy)):
            value = node.get("ft_scale", None)
            if value is not None:
                seen[name] = [float(v) for v in value]
        distinct = {tuple(v) for v in seen.values()}
        if len(distinct) > 1:
            raise RuntimeError(f"ft_scale disagrees across config nodes: {seen}")

    def _resolve_img_indices(self) -> tuple[np.ndarray, int]:
        """The image window's positional slots inside the length-``2H`` window, + how many of
        them the action head reads.

        Derived exactly as ``umi_lazy_dataset._process_source_data`` does
        (``include_indices / down_sample_steps + (H-1)``) so the live path and the dataset path
        gather the SAME frames. ``history_zero_offset`` used to be a hardcoded 15, correct only
        for H==16; deriving it is what makes H==32 land "now" at slot 31 instead of 15."""
        meta = self.cfg.task.dataset.source_data_meta
        entry = meta.get("camera0_rgb", None)
        if entry is None:
            raise RuntimeError("config has no camera0_rgb in task.dataset.source_data_meta")
        include = np.asarray([int(v) for v in entry.include_indices], dtype=np.int64)
        dss = max(1, self.down_sample_steps)
        offset = self.num_actions - 1
        idx = include // dss + offset
        n_img = int(idx.shape[0])
        if n_img % 2 != 0:
            raise RuntimeError(
                f"camera0_rgb has {n_img} include_indices; get_vae_latent splits the frame "
                "axis in half, so it must be even.")
        n_cond = n_img // 2
        if idx[:n_cond].min() < 0 or idx[:n_cond].max() >= self.num_actions:
            raise RuntimeError(
                f"conditioning image slots {idx[:n_cond].tolist()} fall outside the "
                f"[0, {self.num_actions}) history half — the proprio gather would wrap.")
        return idx, n_cond

    @staticmethod
    def _resolve_provenance(payload) -> tuple[int | None, int | None]:
        """``(epoch, global_step)`` out of the checkpoint's dill-encoded ``pickles``, or
        ``(None, None)``. Best-effort provenance for the page readout: a checkpoint whose
        pickles cannot be decoded must still serve."""
        if not payload:
            return None, None
        try:
            import dill

            pickles = payload.get("pickles", {}) or {}
            values = {k: dill.loads(v) for k, v in pickles.items()}
            epoch = values.get("epoch")
            step = values.get("global_step")
            return (int(epoch) if epoch is not None else None,
                    int(step) if step is not None else None)
        except Exception as exc:  # noqa: BLE001 - provenance is not load-bearing
            _log(f"could not decode checkpoint pickles ({type(exc).__name__}); "
                 "epoch/global_step unavailable.")
            return None, None

    def _build_dataset(self, hydra) -> None:
        """Instantiate the single lazy training dataset and put it in DEPLOY mode.

        The lazy dataset is the TRAINING dataset, so ``__getitem__`` is stochastic twice over —
        a live kornia augmentation stack on ``camera0_rgb`` and N(0, 0.05) noise added to
        ``robot0_demo_start_pose`` (which perturbs the ``wrt_start`` proprio the model
        conditions on). A deployed ``predict_action`` sees neither, so both are gated off; a
        scrubbed rollout must reflect the real policy, not a training-augmented one.
        """
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        lazy = dataset.datasets[0]
        if getattr(lazy, "random_img_sampling", False):
            raise RuntimeError(
                "random_img_sampling is True: which 4 frames condition the policy would be "
                "redrawn per call, so no offline rollout would be reproducible.")
        lazy.apply_augmentation_in_cpu = False
        self._neutralize_startpose_noise(lazy)
        if int(getattr(lazy, "down_sample_steps", 1)) != self.down_sample_steps:
            raise RuntimeError("dataset down_sample_steps disagrees with the config")
        self.dataset = dataset
        self.lazy = lazy
        self.zarr_path = str(lazy.zarr_path)
        if len(dataset.datasets) != 1 or (
                self._want_zarr and os.path.basename(self.zarr_path)
                != os.path.basename(self._want_zarr)):
            # pin_single_zarr should have collapsed the mix; if it did not, `datasets[0]` is a
            # DIFFERENT session than the caller asked about and every window would come from the
            # wrong recording. Loud, because the values would all look plausible.
            raise RuntimeError(
                f"expected exactly the requested zarr, got {len(dataset.datasets)} member(s) "
                f"with datasets[0] = {self.zarr_path!r} (asked for {self._want_zarr!r})")
        ends = np.asarray(lazy.episode_ends, dtype=np.int64)
        self.episode_starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        self.episode_frame_nums = (ends - self.episode_starts).astype(np.int64)
        # UVA holds out episodes by RATIO (`used_episode_indices` are the ones trained on), which
        # is a different split from FLoRe's last-N convention — so report it and let the frontend
        # say whether a scrubbed episode was in-distribution.
        used = sorted(int(i) for i in getattr(lazy, "used_episode_indices", []) or [])
        self.train_episodes = np.asarray(used, dtype=np.int64)
        _log(f"dataset ready: {self.zarr_path} ({len(self.episode_frame_nums)} episodes, "
             f"{len(used)} of them in UVA's train split)")

    @staticmethod
    def _neutralize_startpose_noise(lazy) -> None:
        """Zero the additive start-pose noise in ``_process_source_data`` for this instance.

        That code calls ``self.rng.normal(scale=[0.05]*6)``, which perturbs the
        ``robot0_eef_rot_axis_angle_wrt_start`` proprio the model conditions on (measured
        max|Δ| ~0.13). A deployed ``predict_action`` never sees it, so an inference window must
        not either.

        We cannot edit the submodule, and ``rng.normal`` cannot be monkey-patched —
        ``numpy.random.Generator`` is a C extension type whose method attributes are READ-ONLY
        (``AttributeError: attribute 'normal' is read-only``). So the whole ``rng`` is swapped
        for a proxy that forwards every other call: ``rng.choice`` still drives
        ``random_img_sampling`` (already off, and asserted off), so overriding exactly one
        method keeps this surgical rather than a blanket determinism hack.
        """
        rng = getattr(lazy, "rng", None)
        if rng is None:
            return
        lazy.rng = _ZeroNormalRNG(rng)

    # --- handshake --------------------------------------------------------
    def ready_message(self) -> tuple[dict, dict[str, np.ndarray]]:
        meta = {
            "num_actions": self.num_actions,
            "action_dim": self.action_dim,
            "force_dim": self.force_dim,
            "history_len": self.num_actions,
            "n_img_cond": self.n_img_cond,
            "force_history_length": self.force_history_length,
            "obs_pose_repr": self.obs_pose_repr,
            "action_pose_repr": self.action_pose_repr,
            "use_force": self.use_force,
            "predict_force": self.predict_force,
            "n_action_steps": self.n_action_steps,
            "down_sample_steps": self.down_sample_steps,
            "checkpoint": self.checkpoint,
            "random_init": self.random_init,
            "num_sampling_steps": str(
                self.cfg.model.policy.autoregressive_model_params.num_sampling_steps),
            "num_iter": int(self.cfg.model.policy.autoregressive_model_params.num_iter),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "zarr_path": self.zarr_path,
            # An offline caller must know which anchors produce a full, unpadded window:
            # valid_idx_min = max_history - history_padding, and the future half needs
            # `num_actions` frames past the anchor.
            "valid_anchor_min": max(
                0, (self.num_actions - 1) - int(
                    getattr(self.lazy, "history_padding_length", 0) or 0)),
            "valid_anchor_slack_end": self.num_actions,
        }
        arrays = {
            "img_indices": self.img_indices.astype(np.int64),
            "ft_scale": self.ft_scale.astype(np.float64),
            "episode_frame_nums": self.episode_frame_nums.astype(np.int64),
            "train_episodes": self.train_episodes.astype(np.int64),
        }
        return meta, arrays

    # --- window assembly (live) -------------------------------------------
    def _place_history(self, values: np.ndarray) -> np.ndarray:
        """``(n_cond, C)`` values at the conditioning slots -> a full ``(H, C)`` history half.

        The gather only ever reads ``img_indices[:n_cond]``, so the other rows are inert for
        this config — they are filled by holding the nearest earlier conditioning value
        (edge-clamped at the front) rather than left at zero, so a future config that reads a
        denser comb degrades to a step-held history instead of to fabricated zeros.
        """
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
        if vals.shape[0] != self.n_img_cond:
            raise ValueError(
                f"expected {self.n_img_cond} conditioning rows, got {vals.shape[0]}")
        slots = self.img_indices[: self.n_img_cond]  # ascending
        out = np.repeat(vals[0:1], self.num_actions, axis=0)
        for i, slot in enumerate(slots):
            out[int(slot):] = vals[i]  # hold this value from its own slot onward
        return out

    def _place_force(self, window: np.ndarray) -> np.ndarray:
        """``(L, 6)`` newest-last measured wrench -> a ``(H, 6)`` history half.

        The encoder keeps the NEWEST ``force_history_length`` rows, so the measured window is
        right-aligned and anything older is edge-clamped to its oldest sample — exactly the
        dataset's history padding at an episode start.
        """
        win = np.asarray(window, dtype=np.float64).reshape(-1, self.force_dim)
        need = self.force_history_length
        if win.shape[0] < 1:
            raise ValueError("force window is empty")
        if win.shape[0] > need:
            win = win[-need:]
        out = np.repeat(win[0:1], self.num_actions, axis=0)
        out[self.num_actions - win.shape[0]:] = win
        return out

    def _image_stack(self, images01: np.ndarray) -> np.ndarray:
        """``(n_cond,3,224,224)`` 0-1 CHW conditioning frames -> the full ``(2*n_cond,...)`` stack.

        Slots ``[0..n_cond-1]`` are the recent frames ENDING AT NOW (the half
        ``get_vae_latent`` hands the action head); the target half repeats the newest frame and
        is never read at ``task_mode='policy_model'``.
        """
        cond = np.asarray(images01, dtype=np.float32)
        if cond.shape[0] != self.n_img_cond:
            raise ValueError(
                f"expected {self.n_img_cond} conditioning images, got {cond.shape[0]}")
        newest = cond[-1:]
        target = np.repeat(newest, self.img_indices.shape[0] - self.n_img_cond, axis=0)
        return np.concatenate([cond, target], axis=0)

    def _obs_from_live(self, arrays: dict, force_valid: bool) -> dict:
        torch = self.torch

        def dup(a: np.ndarray) -> np.ndarray:
            """History half -> the length-2H window process_data chunks back in half."""
            return np.concatenate([a, a], axis=0)

        eef_pos = dup(self._place_history(arrays["eef_pos_rel"]))
        eef_rot = dup(self._place_history(arrays["eef_rot6d_rel"]))
        wrt_start = dup(self._place_history(arrays["eef_rot6d_wrt_start"]))
        grip = dup(self._place_history(arrays["grip_m"]))
        images = self._image_stack(arrays["images"])

        obs = {
            "camera0_rgb": torch.as_tensor(images).float()[None].to(self.device),
            "robot0_eef_pos": torch.as_tensor(eef_pos).float()[None].to(self.device),
            "robot0_eef_rot_axis_angle": torch.as_tensor(eef_rot).float()[None].to(self.device),
            "robot0_eef_rot_axis_angle_wrt_start":
                torch.as_tensor(wrt_start).float()[None].to(self.device),
            "robot0_gripper_width": torch.as_tensor(grip).float()[None].to(self.device),
            "img_indices": torch.as_tensor(
                self.img_indices.astype(np.float64))[None, :, None].float().to(self.device),
        }
        if self.use_force:
            wrench = dup(self._place_force(arrays["force"]))
            # ÷ ft_scale: the DATASET applies this pre-scale before any normalizer, so a live
            # observation must too or the force encoder sees ~15x its training distribution.
            # (The `zarr` path gets it from the dataset and must NOT double-apply it.)
            obs["robot0_ft_wrench"] = (
                torch.as_tensor(wrench / self.ft_scale).float()[None].to(self.device))
            if not force_valid:
                # Rate-limited: at a real F/T rate (~204 Hz over a 267 ms window, ~54 samples)
                # this should never fire, so it means a gap — but a per-request line would
                # bury the rest of the log if the bridge is flapping. UVA has no per-window
                # validity input (`force_allow_missing: false`), so the samples go in as-is
                # with the older bins edge-held; the FLoRe planner's own freshness gate is what
                # refuses a Predict on a DEAD bridge.
                self._force_invalid_n += 1
                if self._force_invalid_n in (1, 10) or self._force_invalid_n % 100 == 0:
                    _log(f"force window flagged INVALID by the planner "
                         f"({self._force_invalid_n} so far); fed as-is with older bins "
                         "edge-held.")
        return obs

    # --- inference --------------------------------------------------------
    def _sample(self, obs: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        from unified_video_action.eval.eval import prepare_data_predict_action
        from unified_video_action.utils.data_utils import resize_image, unnormalize_future_action

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        x = resize_image(self.cfg, {"obs": obs})
        _b, t_img, _c, _h, _w = x["obs"]["image"].size()
        # Only the SHAPE matters at task_mode='policy_model': forward_mae_encoder uses
        # `fake_action_latent` for the action stream unless task_mode=='dynamic_model', so
        # `nactions` is never read. Sized as training does (2H rows, chunked in half) so a
        # future mode that DOES read it gets a well-formed tensor rather than a short one.
        actions = torch.zeros(
            (1, self.window_len, self.action_dim), device=self.device, dtype=torch.float32)
        (_x, _real, _lat, cond, text_latents, hist_traj, traj, prop) = \
            prepare_data_predict_action(
                self.cfg, x, actions, self.policy, t_img, self.device, language_goal=None)

        amp = self.cfg.model.policy.autoregressive_model_params
        with torch.no_grad():
            _z, act_out, force_out = self.policy.model.sample_tokens(
                bsz=1, cond=cond, text_latents=text_latents,
                num_iter=amp.num_iter, cfg=amp.cfg, cfg_schedule=amp.cfg_schedule,
                temperature=amp.temperature, history_nactions=hist_traj,
                nactions=traj, proprioception_input=prop, task_mode="policy_model")
        act_out = unnormalize_future_action(
            normalizer=self.policy.normalizer,
            normalizer_type=self.policy.normalizer_type, actions=act_out)
        action_rel = np.asarray(
            act_out[0].detach().float().cpu().numpy(), dtype=np.float64)  # (H,10) relative
        if force_out is not None:
            force_pred = np.asarray(
                force_out[0].detach().float().cpu().numpy(), dtype=np.float64) * self.ft_scale
        else:
            force_pred = np.zeros((0, self.force_dim), dtype=np.float64)
        if action_rel.shape != (self.num_actions, self.action_dim):
            raise RuntimeError(
                f"model returned a {action_rel.shape} chunk; expected "
                f"({self.num_actions}, {self.action_dim})")
        return action_rel, force_pred

    def infer_live(self, meta: dict, arrays: dict) -> tuple[dict, dict]:
        obs = self._obs_from_live(arrays, bool(meta.get("force_valid", True)))
        action_rel, force_pred = self._sample(obs, int(meta.get("seed", 0)))
        return {"kind": wire.KIND_LIVE}, {"action_rel": action_rel, "force_pred": force_pred}

    def infer_zarr(self, meta: dict) -> tuple[dict, dict]:
        torch = self.torch
        if self.lazy is None:
            raise RuntimeError(
                "this backend was started without a dataset; pass --zarr-path to answer "
                f"{wire.KIND_ZARR!r} requests.")
        episode = int(meta["episode"])
        anchor = int(meta["anchor"])
        n_ep = len(self.episode_frame_nums)
        if not 0 <= episode < n_ep:
            raise ValueError(f"episode {episode} is outside [0, {n_ep - 1}]")
        length = int(self.episode_frame_nums[episode])
        lo = max(0, (self.num_actions - 1)
                 - int(getattr(self.lazy, "history_padding_length", 0) or 0))
        hi = length - self.num_actions - 1
        if not lo <= anchor <= hi:
            raise ValueError(
                f"anchor {anchor} is outside this checkpoint's usable window for episode "
                f"{episode} ({length} frames): legal anchors are [{lo}, {hi}]. Below {lo} the "
                f"{self.num_actions}-step history would be padded past what training allowed; "
                f"above {hi} the future half runs off the episode end.")

        # Build the window at (episode, anchor) by overriding the lazy dataset's index pool
        # for exactly one __getitem__, then restoring it.
        saved = self.lazy.index_pool
        self.lazy.index_pool = [(episode, anchor)]
        try:
            sample = self.lazy[0]
        finally:
            self.lazy.index_pool = saved

        obs = {k: torch.as_tensor(v)[None].to(self.device) for k, v in sample["obs"].items()}
        action_rel, force_pred = self._sample(obs, int(meta.get("seed", 0)))

        # The anchor's ABSOLUTE pose, straight off the raw zarr. The FLoRe side cross-checks it
        # against its OWN dataset's anchor for the same (episode, frame) — the two loaders
        # relativize independently, so an agreement here is what proves both are decoding in
        # the same absolute frame (a non-identity `world_align` on one side would show up as a
        # constant offset instead of as a subtly wrong rollout).
        store = self.lazy.zarr_store
        gstart = int(self.episode_starts[episode])
        frame = gstart + anchor
        anchor_pos = np.asarray(store["data/robot0_eef_pos"][frame], dtype=np.float64)
        anchor_rot6d = _rotvec_to_rot6d(store["data/robot0_eef_rot_axis_angle"][frame])
        return (
            {"kind": wire.KIND_ZARR, "episode": episode, "anchor": anchor,
             "episode_frames": length},
            {"action_rel": action_rel, "force_pred": force_pred,
             "anchor_pos": anchor_pos, "anchor_rot6d": anchor_rot6d},
        )


# ===========================================================================
# --print-meta: read the grid WITHOUT building the model
# ===========================================================================
def print_meta(checkpoint: str, config_name: str, key: str = "") -> int:
    """Print one JSON line describing the checkpoint's grid, then exit.

    The launchers need ``num_actions`` BEFORE FLoRe's python starts, because
    ``ACTION_HORIZON`` is resolved from ``FLORE_ACTION_HORIZON`` at import time. A UVA
    checkpoint keeps its config as a pickled OmegaConf inside the ``.pt``, so only this
    interpreter can read it — hence a probe subprocess rather than a local JSON read.

    Deliberately builds NO model and touches no GPU: this runs on every launch.
    """
    sys.path.insert(0, _UVA)
    os.chdir(_UVA)
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    if checkpoint:
        path = os.path.abspath(os.path.expanduser(checkpoint))
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            # mmap needs the new zipfile format; fall back to a full read.
            payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
    else:
        from hydra import compose, initialize_config_dir

        cfg_dir = pathlib.Path(_UVA) / "unified_video_action" / "config"
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cfg = compose(config_name=config_name,
                          overrides=["task.dataset.dataset_root_dir=/tmp"])
    OmegaConf.resolve(cfg)
    pol = cfg.model.policy
    dss = int(cfg.task.dataset.down_sample_steps)
    meta = {
        "num_actions": int(pol.num_actions),
        "action_dim": int(cfg.task.shape_meta.action.shape[0]),
        "force_history_length": int(pol.get("force_history_length", 16)),
        "use_force": bool(pol.get("use_force", False)),
        "predict_force": bool(pol.get("predict_force", False)),
        "down_sample_steps": dss,
        "ft_scale": [float(v) for v in cfg.task.get("ft_scale", [])],
    }
    if dss != 1:
        # Refused HERE, in the probe, so a launcher never even exports a horizon for a
        # checkpoint whose model step is several recorded frames: FLoRe's grid is one step per
        # frame, so every predicted step would be published at the wrong instant.
        raise SystemExit(
            f"checkpoint was trained with down_sample_steps={dss}; FLoRe's action grid is one "
            "step per recorded frame (ACTION_DT), so this checkpoint cannot be served here.")
    if key:
        if key not in meta:
            raise SystemExit(f"--print-meta-key {key!r} is not one of {sorted(meta)}")
        _REAL_STDOUT.write(f"{meta[key]}\n")
    else:
        _REAL_STDOUT.write(json.dumps(meta) + "\n")
    _REAL_STDOUT.flush()
    return 0


# ===========================================================================
def main(argv=None) -> int:
    raw_argv = argv if argv is not None else sys.argv[1:]
    # --help must land on the REAL stdout. sys.stdout is pointed at stderr at import time (the
    # protocol channel must stay clean of model chatter), and argparse writes help to
    # sys.stdout at call time — so without this, `--help | grep` returns nothing.
    if "-h" in raw_argv or "--help" in raw_argv:
        sys.stdout = _REAL_STDOUT
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default="",
                   help="UVA workspace .ckpt or exported .slim.pt. Omit for a RANDOM-INIT "
                        "wiring check (the handshake says so; never arm a robot on it).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--config-name", default="uva_hl_force",
                   help="hydra config used only when --checkpoint is omitted (a real "
                        "checkpoint carries its own cfg).")
    p.add_argument("--dataset-root", default="",
                   help="dir holding the zarr(s); defaults to --zarr-path's directory.")
    p.add_argument("--zarr-path", default="",
                   help=f"enable {wire.KIND_ZARR!r} requests over THIS zarr (offline "
                        "inference). Omitted => live requests only, and no zarr is opened.")
    p.add_argument("--model-size", default="mar_base")
    p.add_argument("--force-fusion", default=None, choices=["early", "late", "both"],
                   help="override the checkpoint's fusion mode (ablation only).")
    p.add_argument("--print-meta", action="store_true",
                   help="print one JSON line with the checkpoint's grid and exit (no model, "
                        "no GPU). Used by the launchers to derive FLORE_ACTION_HORIZON.")
    p.add_argument("--print-meta-key", default="",
                   help="like --print-meta but print ONLY this field's value, so a shell "
                        "launcher needs no JSON parsing (e.g. num_actions).")
    args = p.parse_args(raw_argv)

    if args.print_meta or args.print_meta_key:
        return print_meta(args.checkpoint, args.config_name, args.print_meta_key)

    try:
        backend = Backend(
            args.checkpoint,
            device=args.device,
            config_name=args.config_name,
            dataset_root=args.dataset_root,
            zarr_path=args.zarr_path,
            model_size=args.model_size,
            force_fusion=args.force_fusion,
            need_dataset=bool(args.zarr_path),
        )
    except Exception as exc:  # noqa: BLE001 - report the cause over the wire, then die
        import traceback

        traceback.print_exc()
        _emit(wire.TAG_ERROR, {"message": f"{type(exc).__name__}: {exc}"})
        return 2

    meta, arrays = backend.ready_message()
    _emit(wire.TAG_READY, meta, **arrays)
    _log("ready; waiting for requests on stdin")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            _tag, req_meta, req_arrays = wire.unpack(line)
            kind = str(req_meta.get("kind", ""))
            if kind == wire.KIND_LIVE:
                reply_meta, reply_arrays = backend.infer_live(req_meta, req_arrays)
            elif kind == wire.KIND_ZARR:
                reply_meta, reply_arrays = backend.infer_zarr(req_meta)
            elif kind == wire.KIND_PING:
                reply_meta, reply_arrays = backend.ready_message()
                _emit(wire.TAG_READY, reply_meta, **reply_arrays)
                continue
            else:
                raise ValueError(f"unknown request kind {kind!r}")
            _emit(wire.TAG_CHUNK, reply_meta, **reply_arrays)
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the backend
            import traceback

            traceback.print_exc()
            # Stay alive: a bad anchor or a malformed frame mid-bring-up must be reportable,
            # not fatal — losing the backend costs another ~60 s model load.
            _emit(wire.TAG_ERROR, {"message": f"{type(exc).__name__}: {exc}"})
    _log("stdin closed; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
