#!/usr/bin/env python3
"""MuSe-on-UVA inference BACKEND — the torch/hydra half of the force rollout viz.

This process runs INSIDE the UVA model environment (the `uvaforce` conda env:
torch 2.7.1+cu128, hydra, diffusers — and NO viser/mink). Its only job is to
turn a scrubbed episode+anchor into ONE UVA action/force chunk and hand it back
to the viser frontend (``scripts/muse_force_rollout_viz.py``), which runs in
FLoRe's own ``.venv`` (viser + mink + mujoco, and NO torch-2.7/hydra). The two
environments are mutually exclusive — that hard split is the whole reason this
is a separate process talking over a pipe instead of one script like the
flow-matching ``scripts/rollout_viz.py``.

Wire protocol (deliberately dependency-free — FLoRe's .venv has no zmq/msgpack,
but both envs have numpy):

  * stdout (fd 1) is a PURE protocol channel: one line per message,
    ``MUSEVIZ<tag> <base64(npz)>``. The model + hydra + VAE print a lot of
    noise; ALL of it is redirected to stderr so it can never corrupt fd 1.
  * A ``ready`` handshake message reports the resolved zarr path, episode
    layout, ft_scale, action horizon, obs history length and image-frame count,
    so the frontend reads the EXACT same zarr and never has to guess a shape.
  * Each request is a line on stdin: ``{"episode": e, "anchor": a}`` (JSON).
    The reply is a ``rollout`` npz with the RELATIVE action chunk, the physical
    (N/N·m) predicted wrench, and the anchor's absolute EE pose so the frontend
    can un-relativize with ``get_real_umi_action`` exactly as deployment does.

Correctness contract (all source-verified against ``unified_video_action/`` and
checked empirically, see docs/muse_on_uva_force_design.md + the memory note):

  * UVA-umi is single-robot, ``act_dim=10`` = the PRIMARY (LEFT) hand only
    (pos3 + rot6d + gripper). The right hand and head are NOT predicted.
  * The inference path replicates ``eval.py::test_action_l2`` step for step:
    build a dataset window, ``prepare_data_predict_action(eval=False)``,
    ``sample_tokens(task_mode="policy_model")`` → ``act_out (1,16,10)`` RELATIVE
    + ``force_out (1,16,6)`` in the ÷ft_scale space. We multiply force back by
    ft_scale so the frontend receives physical N / N·m (matching what
    ``predict_action`` returns to a real deployment).
  * The window is built at an ARBITRARY anchor by overriding the lazy dataset's
    ``index_pool`` to ``[(store_episode, anchor)]``.
  * DEPLOY FIDELITY / DETERMINISM: the lazy dataset is the TRAINING dataset, so
    its ``__getitem__`` would (a) run the kornia image-augmentation stack and
    (b) add N(0,0.05) start-pose noise — neither of which a deployed
    ``predict_action`` sees. Both are disabled for the rollout window in
    ``Backend.__init__`` (``apply_augmentation_in_cpu=False`` +
    ``_neutralize_startpose_noise``). ``sample_tokens`` still draws diffusion
    noise, so ``rollout`` seeds the torch RNG per ``(episode, anchor)`` — a given
    request reproduces exactly across clicks. (``random_img_sampling=False``
    only fixes WHICH 4 frames are used; it is asserted but is not what makes the
    rollout deterministic.)

With no ``--checkpoint`` the model is random-initialised from
``config/uva_umi_force.yaml`` (a wiring check only — the rollout is meaningless).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import sys

import numpy as np

# --- keep fd 1 (stdout) clean: the model/hydra/VAE chatter must go to fd 2 ----
# We stash the real stdout, then point sys.stdout at stderr for the rest of the
# process so every stray print() lands on stderr. Protocol messages are written
# to the stashed real stdout explicitly via _emit().
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLORE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UVA = os.path.join(_FLORE_ROOT, "third_party", "unified_video_action")


def _emit(tag: str, **arrays) -> None:
    """Write one protocol line: ``MUSEVIZ<tag> <base64(npz)>`` to real stdout.

    Arrays are packed with ``np.savez`` (any numpy dtypes/shapes, plus 0-d
    arrays wrapping JSON-able scalars/strings). Base64 keeps the payload on a
    single line so the frontend can read it with ``readline()``.
    """
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    _REAL_STDOUT.write(f"MUSEVIZ{tag} {b64}\n")
    _REAL_STDOUT.flush()


def _log(msg: str) -> None:
    print(f"[muse-infer] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Model + dataset construction (UVA env)
# ---------------------------------------------------------------------------
class Backend:
    """Holds the UVA policy + the (single) lazy dataset, and answers rollouts.

    The dataset is instantiated exactly as training/eval do (hydra), so the
    windowing, the ÷ft_scale pre-scale, the relative-pose obs and the
    pose-glitch window exclusion are all the real thing. Inference replicates
    ``eval.py::test_action_l2`` so the numbers match the validated eval metric.
    """

    def __init__(self, checkpoint: str, dataset_root: str, device: str,
                 model_size: str, force_fusion: str | None):
        # Resolve user paths to ABSOLUTE *before* the chdir below: __init__ does
        # os.chdir(_UVA) (the UVA config/pretrained paths are relative to it), so
        # a relative --checkpoint or --dataset-root passed on the CLI would
        # otherwise be looked up against _UVA and vanish. The launcher already
        # abspaths dataset_root; do both here so a bare `python backend.py` works.
        if checkpoint:
            checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        if dataset_root:
            dataset_root = os.path.abspath(os.path.expanduser(dataset_root))

        sys.path.insert(0, _UVA)
        os.chdir(_UVA)

        import hydra
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf, open_dict

        self.torch = torch
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        # --- resolve device (cpu fallback) ---
        if device.startswith("cuda") and not torch.cuda.is_available():
            _log("CUDA unavailable; falling back to cpu.")
            device = "cpu"
        self.device = device
        self.random_init = not checkpoint

        cfg_dir = pathlib.Path(_UVA) / "unified_video_action" / "config"
        overrides = [
            f"task.dataset.dataset_root_dir={dataset_root}",
            "task.dataset.dataloader_cfg.batch_size=1",
            "task.dataset.dataloader_cfg.num_workers=0",
            f"model.policy.autoregressive_model_params.model_size={model_size}",
        ]
        if force_fusion:
            overrides.append(f"model.policy.force_fusion={force_fusion}")

        if self.random_init:
            # No checkpoint: build straight from the config (fresh weights). The
            # pretrained MAR path is nulled so nothing tries to download.
            overrides.append(
                "model.policy.autoregressive_model_params.pretrained_model_path=null")
            with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
                cfg = compose(config_name="uva_umi_force", overrides=overrides)
            OmegaConf.resolve(cfg)
            self.cfg = cfg
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            policy = hydra.utils.instantiate(
                cfg.model.policy, task_name=cfg.task.name,
                task_modes=cfg.task.task_modes,
                normalizer_type=cfg.task.dataset.normalizer_type,
                language_emb_model=cfg.task.dataset.language_emb_model)
            _log("no --checkpoint: RANDOM-INIT UVA model from uva_umi_force.yaml "
                 "(pipeline check only — rollout is meaningless).")
        else:
            # Real checkpoint: rebuild the workspace from the pickled cfg, then
            # reuse the EMA weights (what deployment serves). We also re-point
            # the dataset_root and drop the dataloader worker count via the
            # SAME cfg so the dataset matches the checkpoint's task exactly.
            from unified_video_action.workspace.train_unified_video_action_workspace import (  # noqa: E501
                TrainUnifiedVideoActionWorkspace,
            )

            ckpt = pathlib.Path(checkpoint)
            payload = torch.load(ckpt.open("rb"), map_location="cpu", weights_only=False)
            cfg = payload["cfg"]
            with open_dict(cfg):
                cfg.task.dataset.dataset_root_dir = dataset_root
                cfg.task.dataset.dataloader_cfg.batch_size = 1
                cfg.task.dataset.dataloader_cfg.num_workers = 0
            OmegaConf.resolve(cfg)
            self.cfg = cfg
            ws = TrainUnifiedVideoActionWorkspace(cfg)
            # strict=False is REQUIRED for the slim inference checkpoint: it drops
            # the frozen `vae_model.*` weights (restored from kl16.ckpt at model
            # construction) and the redundant non-EMA `model` state. load_payload
            # forwards **kwargs straight to load_state_dict. NOTE strict=False
            # (unlike strict=True) NEVER raises — it silently collects missing +
            # unexpected keys — and load_payload discards that result, so this
            # call alone validates NOTHING. `_assert_weights_loaded` below
            # re-checks in-code that the EMA weights actually populated the model
            # (0 unexpected, 0 non-vae missing — verified against best-ckpt-slim).
            ws.load_payload(payload, strict=False)
            policy = ws.ema_model if ws.ema_model is not None else ws.model
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            _log(f"loaded checkpoint {ckpt}")

        policy.to(device).eval()
        self.policy = policy
        self.dataset = dataset
        # UmiMultiDataset -> the single underlying lazy dataset for the force zarr
        self.lazy = dataset.datasets[0]

        # --- put the dataset in EVAL/DEPLOY mode for the rollout window --------
        # The lazy dataset is the TRAINING dataset, so its __getitem__ is
        # stochastic in two ways that must NOT touch an inference window:
        #   (1) `apply_augmentation_in_cpu=True` runs a live kornia stack
        #       (RandomCrop/ColorJitter/RandomGrayscale/RandomGaussianBlur) on
        #       camera0_rgb — re-rolled every call (measured img max|Δ|≈0.9 on a
        #       0-1 image), and
        #   (2) `_process_source_data` adds N(0, 0.05) noise to
        #       robot0_demo_start_pose, perturbing the
        #       robot0_eef_rot_axis_angle_wrt_start proprio the model conditions
        #       on (measured wrt_start max|Δ|≈0.13).
        # A deployed `predict_action` sees neither. Both are gated OFF here so
        # the rollout reflects the real policy, not a training-augmented,
        # start-pose-jittered one. `random_img_sampling` (which 4 frames) is
        # already False for the force task; assert it so a future config change
        # can't silently reintroduce nondeterministic frame selection.
        assert not self.lazy.random_img_sampling, (
            "random_img_sampling must be False for a deterministic scrubbed "
            "rollout; the force task sets it False — got True.")
        # Disable image augmentation on the single lazy dataset (the multi-dataset
        # reads this flag off datasets[0], so set it there).
        self.lazy.apply_augmentation_in_cpu = False
        # Gate the HACK start-pose noise in umi_lazy_dataset._process_source_data.
        # It reads `self.rng.normal(scale=[0.05]*6)`; we can't edit that file, but
        # monkey-patching this instance's rng.normal to return zeros neutralises
        # ONLY the additive-noise call (rng.choice for random_img_sampling is
        # off). Deterministic + deployment-faithful.
        self._neutralize_startpose_noise(self.lazy)

        # --- defensive load validation ---------------------------------------
        # base_workspace.load_payload discards load_state_dict's _IncompatibleKeys,
        # and strict=False never raises, so a future retrained slim ckpt with a
        # renamed key could silently leave a real param at random init. Re-verify
        # in-code that the EMA weights actually populated `policy.model`: 0
        # unexpected keys and no MISSING key outside the intentionally-dropped
        # frozen `vae_model.*`.
        if not self.random_init:
            self._assert_weights_loaded(policy, payload)
            # The frozen VAE is restored from its own file at construction
            # (kl16.ckpt); if that file was absent AutoencoderKL silently keeps a
            # RANDOM VAE and every rollout is garbage with no error. Fail loudly.
            self._assert_vae_loaded()

        self.ft_scale = np.asarray(
            list(self.cfg.task.get("ft_scale", [1.0] * 6)), dtype=np.float64)
        # ft_scale is read from cfg.task.ft_scale here; eval.py reads
        # cfg.model.policy.ft_scale and the dataset divides by
        # cfg.task.dataset.ft_scale. All three are `${task.ft_scale}`
        # interpolations, but assert they agree so an override of one node can't
        # silently desync the physical-force units from the metric's scale.
        self._assert_ft_scale_consistent()
        self.action_dim = int(self.cfg.task.shape_meta.action.shape[0])  # 10
        self.action_horizon = int(self.cfg.task.action_horizon) // 2       # 16 future
        self.predict_force = bool(self.cfg.model.policy.get("predict_force", False))
        self.obs_pose_repr = str(self.cfg.task.pose_repr.obs_pose_repr)
        self.action_pose_repr = str(self.cfg.task.pose_repr.action_pose_repr)

        # zarr path the dataset actually opened (frontend must read the same one)
        self.zarr_path = str(self.lazy.zarr_path)
        # store-global episode layout (matches the raw zarr meta/episode_ends).
        # `episode_ends` is the clean ndarray; `episode_starts` is derived from it
        # (`episode_frame_nums` on the lazy dataset is a dict keyed by index, so
        # we build the arrays here rather than np.asarray it).
        ends = np.asarray(self.lazy.episode_ends, dtype=np.int64)
        self.episode_starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        self.episode_frame_nums = (ends - self.episode_starts).astype(np.int64)
        self.num_img_frames = self._infer_num_img_frames()

        # The recorded<->predicted frame mapping (backend fut_idx=a+1+i, and the
        # frontend's rec_i / force-overlay base / rollout time axis) all assume
        # ONE recorded frame per model step, which holds only when the dataset's
        # down_sample_steps == 1. The force task pins it to 1; assert so a
        # checkpoint trained with a different stride can't silently produce a
        # time-compressed (and therefore mis-aligned) overlay.
        self.down_sample_steps = int(getattr(self.lazy, "down_sample_steps", 1))
        assert self.down_sample_steps == 1, (
            "muse force viz assumes down_sample_steps==1 (1 recorded frame per "
            f"model step); checkpoint cfg has {self.down_sample_steps}. The "
            "recorded/predicted frame mapping would be time-compressed.")

        # store-global train/val labelling: the lazy dataset is the TRAIN split
        # (used_episode_ratio<1). `used_episode_indices` are the store-episode
        # indices actually trained on; the rest are the held-out val episodes the
        # monitored val_action_l2 metric was measured on. The frontend uses this
        # to warn when a scrubbed episode is in-distribution (train), so a tighter
        # -than-metric overlay isn't mistaken for generalization.
        used = sorted(int(i) for i in getattr(self.lazy, "used_episode_indices", []))
        n_store = len(self.episode_frame_nums)
        self.train_episodes = np.asarray(used, dtype=np.int64)
        self.val_episodes = np.asarray(
            [i for i in range(n_store) if i not in set(used)], dtype=np.int64)

    def _infer_num_img_frames(self) -> int:
        """How many image frames one window carries (8 for the umi 4+4 grid)."""
        for m in self.lazy.source_data_meta.values():
            if (m.name in self.lazy.output_data_meta
                    and self.lazy.output_data_meta[m.name].data_type == "image"):
                return len(m.include_indices)
        return 8

    # --- deploy-fidelity / load-validation helpers ------------------------
    @staticmethod
    def _neutralize_startpose_noise(lazy) -> None:
        """Zero ONLY the additive start-pose noise in the lazy dataset.

        ``umi_lazy_dataset._process_source_data`` does
        ``start_pose += self.rng.normal(scale=[0.05]*6, size=...)`` (a training
        HACK). We cannot edit that submodule file, so we wrap this instance's
        ``rng.normal`` to return zeros while leaving every other method of the
        Generator (``choice`` etc.) intact. With ``random_img_sampling=False``
        the only ``rng`` draw on the inference path is this ``normal`` call, so
        the window becomes deterministic AND matches a real deployment (which
        adds no start-pose noise)."""
        rng = lazy.rng

        class _NoNormalRng:
            def normal(self, *a, **k):
                size = k.get("size")
                if size is None and len(a) >= 3:
                    size = a[2]
                return np.zeros(size) if size is not None else 0.0

            def __getattr__(self, name):
                return getattr(rng, name)

        lazy.rng = _NoNormalRng()

    @staticmethod
    def _assert_weights_loaded(policy, payload) -> None:
        """Re-check that the EMA weights actually populated ``policy.model``.

        ``load_payload``/``strict=False`` swallow mismatches, so verify here:
        every checkpoint tensor key (minus the ``module.`` prefix) must exist in
        the model, and no NON-``vae_model.*`` model param may be absent from the
        checkpoint (the frozen VAE is intentionally dropped and restored from
        kl16.ckpt)."""
        import torch  # noqa: F401
        ckpt_keys = {k.replace("module.", "") for k in
                     payload["state_dicts"]["ema_model"].keys()}
        model_keys = set(policy.state_dict().keys())
        unexpected = sorted(ckpt_keys - model_keys)
        missing = sorted(k for k in (model_keys - ckpt_keys)
                         if not k.startswith("vae_model"))
        if unexpected or missing:
            raise RuntimeError(
                "checkpoint did not load cleanly into the policy — "
                f"{len(unexpected)} unexpected key(s) {unexpected[:5]}, "
                f"{len(missing)} non-vae MISSING key(s) {missing[:5]}. A "
                "renamed/extra tensor would leave a real param at random init; "
                "refusing to serve a partially-loaded model.")
        _log(f"weight-load check OK: {len(ckpt_keys)} ckpt tensors, "
             f"0 unexpected, 0 non-vae missing.")

    def _assert_vae_loaded(self) -> None:
        """Fail loudly if the frozen VAE file the slim ckpt relies on is absent.

        The 312 ``vae_model.*`` keys are dropped from the slim ckpt and restored
        from ``vae_model_params.autoencoder_path`` (kl16.ckpt) at construction.
        ``AutoencoderKL.__init__`` guards that restore with ``os.path.exists`` and
        SILENTLY keeps a random VAE if the file is missing — poisoning every
        rollout with no error. The path is relative and resolved against the
        ``os.chdir(_UVA)`` done in ``__init__``."""
        try:
            vp = self.cfg.model.policy.vae_model_params
            path = vp.get("autoencoder_path", None)
        except Exception:  # noqa: BLE001 — cfg shape differs → skip the guard
            return
        if not path:
            return
        # resolved against _UVA (we chdir'd there); accept absolute too.
        resolved = path if os.path.isabs(path) else os.path.join(_UVA, path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"frozen VAE weights not found at {resolved!r} "
                f"(cfg vae autoencoder_path={path!r}). The slim checkpoint drops "
                "vae_model.* and relies on this file; without it the VAE stays "
                "randomly initialised and every rollout is meaningless. Run the "
                "RUNBOOK pretrained-models download on this box.")

    def _assert_ft_scale_consistent(self) -> None:
        """Assert the three ft_scale config nodes agree (units == metric scale).

        Backend reads ``cfg.task.ft_scale``; eval reads
        ``cfg.model.policy.ft_scale``; the dataset divides by
        ``cfg.task.dataset.ft_scale``. All are ``${task.ft_scale}``
        interpolations for this config family, but an override of one alone
        would desync the overlay's physical units from what the head was scored
        against."""
        def _get(node, key):
            try:
                v = node.get(key, None)
            except Exception:  # noqa: BLE001
                return None
            return None if v is None else [round(float(x), 6) for x in v]
        task = _get(self.cfg.task, "ft_scale")
        pol = _get(self.cfg.model.policy, "ft_scale")
        dset = _get(self.cfg.task.dataset, "ft_scale")
        present = [v for v in (task, pol, dset) if v is not None]
        if present and any(v != present[0] for v in present):
            raise ValueError(
                "ft_scale mismatch across config nodes — "
                f"task={task}, model.policy={pol}, task.dataset={dset}. Physical "
                "force units would disagree with the metric's scale.")

    # --- one rollout ------------------------------------------------------
    def rollout(self, episode: int, anchor: int) -> dict:
        """Sample ONE UVA chunk at (store-episode ``episode``, frame ``anchor``).

        The action comes out of the model RELATIVE to the anchor EE pose
        (``action_pose_repr='relative'``). We un-relativize it HERE, in the UVA
        env, with the exact same ``get_real_umi_action`` a real deployment uses
        — so the frontend never re-derives the model's pose contract and can't
        drift from it. Verified: applied to the ground-truth action this
        recovers the recorded future world poses to machine precision.

        Returns a dict of numpy arrays for ``_emit``:
          * ``action_pos`` (H,3), ``action_rotvec`` (H,3): ABSOLUTE world EE
            pose of the predicted future left-hand chunk.
          * ``action_grip`` (H,): predicted gripper width (metres, as stored).
          * ``force_pred`` (H,6): predicted future wrench in PHYSICAL N/N·m
            (already multiplied by ft_scale), or an empty array if the model
            has no force head.
          * ``anchor_pos`` (3,), ``anchor_rotvec`` (3,): the anchor's ABSOLUTE
            EE pose (raw zarr) — the branch point the frontend shares with the
            recorded trajectory.
          * ``gt_force_future`` (H,6): recorded future wrench (physical), for a
            predicted-vs-ground-truth force overlay.
        """
        import torch
        from unified_video_action.utils.data_utils import (
            resize_image, unnormalize_future_action)
        from unified_video_action.eval.eval import prepare_data_predict_action
        # NOTE: we do NOT import umi.real_world.real_inference_util.get_real_umi_action
        # even though this reproduces it exactly. That module's top-level
        # `from ...rotation_transformer import RotationTransformer` pulls in
        # pytorch3d, which is not installed in this env (and is unused by the
        # un-relativization). `_unrelativize_action` below inlines the identical
        # math from the pytorch3d-free `pose_util` + `pose_repr_util`.

        e = int(np.clip(episode, 0, len(self.episode_frame_nums) - 1))
        length = int(self.episode_frame_nums[e])
        a = int(np.clip(anchor, 0, length - 1))

        # Seed the global torch RNG per (e, a) so a given request reproduces
        # exactly across clicks / process restarts. sample_tokens draws the
        # diffusion noise (act + force heads) at temperature<1 from the global
        # RNG; deployment is stochastic too, but for an inspection tool a stable
        # picture beats a jumping one. Image aug + start-pose noise are already
        # disabled in __init__, so the window itself is deterministic.
        torch.manual_seed(1_000_003 * e + a)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1_000_003 * e + a)

        # Build the window at (e, a) by overriding the lazy dataset's index pool
        # for exactly one __getitem__ call, then restoring it.
        saved = self.lazy.index_pool
        self.lazy.index_pool = [(e, a)]
        try:
            sample = self.lazy[0]
        finally:
            self.lazy.index_pool = saved

        obs = {k: torch.as_tensor(v)[None].to(self.device)
               for k, v in sample["obs"].items()}
        actions = torch.as_tensor(sample["action"])[None].to(self.device)
        x = {"obs": obs}
        x = resize_image(self.cfg, x)
        B, T, C, H, W = x["obs"]["image"].size()

        (_x, _real, _lat, c, text_latents, hist_traj, traj, prop) = \
            prepare_data_predict_action(
                self.cfg, x, actions, self.policy, T, self.device,
                language_goal=None)

        amp = self.cfg.model.policy.autoregressive_model_params
        with torch.no_grad():
            _z, act_out, force_out = self.policy.model.sample_tokens(
                bsz=B, cond=c, text_latents=text_latents,
                num_iter=amp.num_iter, cfg=amp.cfg, cfg_schedule=amp.cfg_schedule,
                temperature=amp.temperature, history_nactions=hist_traj,
                nactions=traj, proprioception_input=prop, task_mode="policy_model")

        # RELATIVE action, unnormalized (umi normalizer is 'none' → pass-through,
        # but call the official inverse anyway to stay contract-exact).
        act_out = unnormalize_future_action(
            normalizer=self.policy.normalizer,
            normalizer_type=self.policy.normalizer_type, actions=act_out)
        action_rel = act_out[0].detach().float().cpu().numpy()  # (H, 10)

        # Predicted future wrench back in physical units.
        if force_out is not None:
            force_pred = force_out[0].detach().float().cpu().numpy() * self.ft_scale
        else:
            force_pred = np.zeros((0, 6), dtype=np.float64)

        # Anchor absolute pose from the RAW zarr (the un-relativization base).
        g = self._zarr()
        gstart = int(self.episode_starts[e])
        anchor_pos = np.asarray(g["data/robot0_eef_pos"][gstart + a], dtype=np.float64)
        anchor_rotvec = np.asarray(
            g["data/robot0_eef_rot_axis_angle"][gstart + a], dtype=np.float64)

        # Un-relativize EXACTLY as deployment's get_real_umi_action does, inlined
        # from the pytorch3d-free helpers (see the import note above).
        action_pos, action_rotvec, action_grip = self._unrelativize_action(
            action_rel, anchor_pos, anchor_rotvec)

        # Recorded future wrench (physical N/N·m), edge-clamped at the episode end.
        H_f = force_pred.shape[0] if force_pred.size else self.action_horizon
        fut_idx = [min(a + 1 + i, length - 1) for i in range(H_f)]
        gt_force = np.asarray(
            [g["data/robot0_ft_wrench"][gstart + j] for j in fut_idx],
            dtype=np.float64)

        return dict(
            action_pos=action_pos, action_rotvec=action_rotvec,
            action_grip=action_grip,
            force_pred=force_pred.astype(np.float64),
            anchor_pos=anchor_pos, anchor_rotvec=anchor_rotvec,
            gt_force_future=gt_force,
            meta=np.asarray(json.dumps(
                {"episode": e, "anchor": a, "H": int(action_pos.shape[0]),
                 "random_init": bool(self.random_init),
                 "split": ("train" if e in set(self.train_episodes.tolist())
                           else "val")})),
        )

    def _unrelativize_action(self, action_rel: np.ndarray, anchor_pos: np.ndarray,
                             anchor_rotvec: np.ndarray):
        """Invert UVA's relative action to ABSOLUTE world poses (single robot).

        This is exactly ``get_real_umi_action(action, env_obs, 'relative')`` for
        one robot: the env-obs anchor is the LAST observed EE pose (our
        ``anchor_*``). For every future step: ``pose10d -> mat``, then
        ``convert_pose_mat_rep(..., base=anchor_mat, backward=True)`` which for
        ``'relative'`` is ``anchor_mat @ action_mat``; then ``mat -> pose``.
        Verified against ground truth to machine precision.

        ``action_rel`` is (H, 10) = pos(3) + rot6d(6) + gripper(1). Returns
        ``(pos[H,3], rotvec[H,3], grip[H])``.
        """
        from umi.common.pose_util import pose_to_mat, mat_to_pose, pose10d_to_mat
        from unified_video_action.common.pose_repr_util import convert_pose_mat_rep

        anchor_mat = pose_to_mat(np.concatenate([anchor_pos, anchor_rotvec]))
        action_pose10d = action_rel[..., :9]
        action_grip = np.asarray(action_rel[..., 9], dtype=np.float64)
        action_mat = convert_pose_mat_rep(
            pose10d_to_mat(action_pose10d), base_pose_mat=anchor_mat,
            pose_rep=self.action_pose_repr, backward=True)
        action_pose = mat_to_pose(action_mat)  # (H, 6) pos + rotvec
        return (np.asarray(action_pose[..., :3], dtype=np.float64),
                np.asarray(action_pose[..., 3:6], dtype=np.float64),
                action_grip)

    _zarr_handle = None

    def _zarr(self):
        if self._zarr_handle is None:
            import zarr
            from unified_video_action.codecs.imagecodecs_numcodecs import (
                register_codecs,
            )
            register_codecs()
            self._zarr_handle = zarr.open(self.zarr_path, "r")
        return self._zarr_handle


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-root", required=True,
                   help="dir containing the muse_force zarr (as the config expects)")
    p.add_argument("--checkpoint", default="",
                   help="UVA workspace checkpoint (.ckpt); empty = random-init")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-size", default="mar_base")
    p.add_argument("--force-fusion", default=None,
                   choices=[None, "early", "late", "both"],
                   help="override force_fusion (default: config = both)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    be = Backend(args.checkpoint, args.dataset_root, args.device,
                 args.model_size, args.force_fusion)

    _emit("ready",
          zarr_path=np.asarray(be.zarr_path),
          episode_starts=be.episode_starts,
          episode_frame_nums=be.episode_frame_nums,
          ft_scale=be.ft_scale,
          train_episodes=be.train_episodes,
          val_episodes=be.val_episodes,
          info=np.asarray(json.dumps({
              "action_dim": be.action_dim,
              "action_horizon": be.action_horizon,
              "num_img_frames": be.num_img_frames,
              "predict_force": be.predict_force,
              "obs_pose_repr": be.obs_pose_repr,
              "action_pose_repr": be.action_pose_repr,
              "down_sample_steps": be.down_sample_steps,
              "random_init": be.random_init,
          })))
    _log(f"ready: zarr={be.zarr_path} episodes={len(be.episode_frame_nums)} "
         f"H={be.action_horizon} predict_force={be.predict_force} "
         f"train/val={len(be.train_episodes)}/{len(be.val_episodes)}")

    # Request loop: one JSON object per stdin line.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _log(f"bad request line: {line!r}")
            continue
        if req.get("cmd") == "quit":
            break
        try:
            out = be.rollout(int(req["episode"]), int(req["anchor"]))
            _emit("rollout", **out)
        except Exception as e:  # noqa: BLE001 — report, keep serving
            import traceback
            _log("rollout error:\n" + traceback.format_exc())
            _emit("error", msg=np.asarray(f"{e!r}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
