"""Tyro dataclasses for S1 training (flow matching | diffusion | ACT).

``TrainConfig`` is the CLI root (``scripts/train_s1.py``); it nests one model config per
algorithm (:class:`DPFlowConfig`, :class:`DPDiffusionConfig`, :class:`DPACTConfig`) plus
the data configs. ``algo`` selects which model config is ACTIVE; see
:data:`ALGO_CONFIG_FIELD`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

from flore.s1_policy.model.dp_act import DPACTConfig
from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig
from flore.s1_policy.model.dp_flow import DPFlowConfig

# The model config IS the training-time model config. Aliased so callers can import
# a single name from the training package.
ModelConfig = DPFlowConfig

#: ``algo`` -> the ``TrainConfig`` attribute holding that algorithm's model config, which
#: is ALSO its key in the serialized ``config.json``. Every algorithm-dispatching site
#: reads this instead of re-deriving the mapping, because the failure mode of getting it
#: wrong is silent: for ``algo='diffusion'``, ``config.json['model']`` still holds FLOW
#: DEFAULTS, so a loader that guessed 'model' would build a default-shaped flow model and
#: ``load_state_dict(strict=False)`` a foreign checkpoint into it — mismatched keys logged
#: as warnings, garbage weights served. (See the ALGO GUARD in serving/server.py.)
ALGO_CONFIG_FIELD: dict[str, str] = {
    "flow": "model",
    "diffusion": "diffusion",
    "act": "act",
}

#: The algorithms ``TrainConfig`` accepts, derived from the mapping so the two can never
#: disagree. Tyro renders this in ``--help``.
ALGOS = tuple(ALGO_CONFIG_FIELD)


@dataclass
class HLDataConfig:
    """HL-v2 dataset settings (the ``hl_*`` fixed-dt window schema).

    Selected by ``TrainConfig.pipeline='hl'``. Mirrors
    :class:`flore.s1_policy.data.hl_dataset.HLDataset`'s ``__init__`` knobs, so a training
    setting can never exist here without a dataset counterpart. The one intentional
    asymmetry: ``glitch_pre_s`` / ``glitch_post_s`` are dataset-only — they are PINNED to
    the layout spans (``FORCE_HIST_SPAN_S`` / ``ACTION_SPAN_S``) and must never be shrunk
    below the real window, so exposing them as tunables would be a footgun. The window
    geometry (8 history bins over 1 s, 16 action steps over 1 s) is likewise NOT
    configurable here: it is contract, fixed in ``flore.s1_policy.data.layout`` and
    asserted against each zarr's attrs.
    """

    # One or more .zarr.zip paths written with convert_to_umi.py --hands ...
    zarr_paths: tuple[str, ...] = ()
    # Layout mode: left_only | right_only | both_hands | both_hands_head |
    # both_hands_head_dual (two F/T sensors + two grippers) | wholebody (the 29 G1 JOINT
    # ANGLES instead of keypoint poses -- needs a zarr carrying the wb_* arrays, written by
    # UMI_collection's motion_completion/mc_to_hl_wholebody.py). See layout.py.
    mode: str = "left_only"
    image_key: str = "camera0_rgb"
    # MULTI-CAMERA. Empty (default) == (image_key,), the single-camera contract. Set it to
    # every camera the model should see, in encoder order:
    #     --hl-data.image-keys camera0_rgb camera1_rgb
    # `model.num_cameras` is PUSHED from its length, so the two can never disagree. Only
    # the 3-tracker rig records a second view (camera1_rgb, the RIGHT wrist D405, paired
    # onto camera0's grid by the converter's causal frame selection).
    image_keys: tuple[str, ...] = ()
    image_mean: float = 0.5
    image_std: float = 0.5
    # Observation history (DECISION-OBS2, UMI-on-Air parity): To frames of image + state,
    # oldest-first, ending at t0. **1 (v3 default) is the single-frame contract** — the
    # observation is t0 alone, matching the 1-bin force input. 2 gives a short-horizon
    # velocity cue instead. obs_history_dt_s is the wall-clock spacing (time-anchored,
    # robust to a non-uniform camera grid) and is unused at To == 1. BAKED into the
    # checkpoint via model.obs_history_len; serving builds a matching To-frame buffer.
    obs_history_len: int = 1
    obs_history_dt_s: float = 0.25
    # Drop anchors whose force-history window is unmeasured, instead of keeping them
    # with force_history_valid=False (they remain usable for pose/gripper supervision).
    drop_invalid_force_history: bool = False
    # Drop anchors whose [t0-1s, t0+1s] sample window straddles a tracker glitch/dropout
    # (per-step masking can't repair a poisoned anchor). ON by default; see HLDataset.
    # Keep this identical to what fit_hl_norm_stats.py used, so the norm stats are fit
    # over the same anchors training sees.
    drop_glitch_windows: bool = True
    # Path to a norm_stats.json fitted with scripts/fit_hl_norm_stats.py on the SAME
    # mode. Empty = train unnormalized (raw metres / rot6d / newtons).
    norm_stats_path: str = ""
    # Episode-disjoint validation: global flattened episode IDs at the tail are held out.
    # clean99 therefore reserves 95,96,97,98. Set 0 only for a deliberate train-only run.
    val_last_episodes: int = 4
    # `wholebody` ONLY (refused elsewhere): what the 29 joint TARGET columns mean.
    #   absolute  the export's own radians, row 0 == state. The shipped default.
    #   delta     q[k] - q[0], anchor-relative joint MOTION; row 0 is exactly zero.
    # The state stays absolute either way, and the deploy-side JointClutch (base + q[i] - q[0])
    # is offset-invariant, so the published wire is the same function of the output in both
    # modes -- this is an A/B on what the network represents, not a new layout. See
    # layout.WB_JOINT_TARGETS. ⚠️ Norm stats must be re-fit at the same value: it lands in the
    # EXACT-equality dataset_knobs, so a crossed file is refused by name.
    wb_joint_target: str = "absolute"
    # `wholebody_geom_*` ONLY (refused on a mode with no matching state block): zero-mean
    # Gaussian jitter added to the GEOMETRIC state columns of every TRAINING sample, in RAW
    # units, before normalization. Validation and the norm-stats fit stay clean —
    # data.hl_dataset.StateNoise wraps the train subset only.
    #
    # WHAT IT IS FOR: the deployed features come from FK of the robot's own joints, and the
    # robot's posture is not the export's. Measured over the 159 valid episode starts of
    # session_2026.09.02_merged3 against the FK'd ready pose, the systematic offset is
    #     lr_dx 11 mm   lp_x 15 mm   left roll/pitch/yaw 1.3 / 4.2 / 2.2 deg
    # (the two `y` channels are 87-92 mm, a placement property of the export that noise cannot
    # fix -- see layout.py's mode table). Sigma should COVER the offsets it can, so a deployed
    # value lands ~1 sigma inside the training distribution rather than 3.
    #
    # DEFAULT 0.0, deliberately: it keeps every previously written TrainConfig byte-identical
    # (--resume-from does exact config equality), so the preset passes the values explicitly.
    # The launcher's default is 0.010 m / 2.0 deg.
    #
    # ⚠️ SCALE MATTERS MORE THAN THE HABIT SUGGESTS. These features are tight: over 7153
    # frames, lr_dx/lr_dy std is 2.6/2.2 cm and left roll/yaw std is 2.9/4.4 deg. So 2.5 cm and
    # 10-15 deg -- the magnitudes a proprioception-noise habit reaches for -- are 1.0x and
    # 3-4x the SIGNAL'S OWN std and would erase the channel while leaving the loss curve
    # looking healthy. Anything much above ~0.015 m / ~3 deg should be treated as an
    # "is this feature useful at all" upper bound, not as regularization.
    state_noise_xy_m: float = 0.0
    state_noise_rpy_deg: float = 0.0

    def state_noise_sigma(self):
        """``(state_dim,)`` per-column sigma, or ``None`` when no noise is configured.

        ONE resolver, read by both :meth:`_post_init_hl`'s validation and
        ``Trainer.build_dataloaders``, for the same reason ``resolved_image_keys`` is one: the
        check and the thing being checked must come from the same rule."""
        from flore.s1_policy.data.layout import get_layout
        from flore.s1_policy.data.wb_geom import noise_sigma

        if not (self.state_noise_xy_m or self.state_noise_rpy_deg):
            return None
        import numpy as np

        sigma = noise_sigma(get_layout(self.mode), xy_m=self.state_noise_xy_m,
                            rpy_rad=float(np.deg2rad(self.state_noise_rpy_deg)))
        return sigma if np.any(sigma > 0) else None

    def resolved_image_keys(self) -> tuple[str, ...]:
        """The camera list, in encoder order. ``image_keys`` if set, else ``(image_key,)``.

        ONE resolver, read by both ``_post_init_hl`` (which pushes ``num_cameras`` from its
        length) and ``Trainer.build_dataloaders`` (which hands it to ``HLDataset``), so the
        model's width and the dataset's stack can never come from different rules.
        """
        if self.image_keys:
            return tuple(self.image_keys)
        return (self.image_key,)


@dataclass
class DataConfig:
    """Legacy (pre-v2) dataset + normalization settings.

    Used by ``TrainConfig.pipeline='legacy'``, which keeps the already-trained 32-dim
    checkpoint reproducible. New runs should use :class:`HLDataConfig`.
    """

    # One or more .zarr.zip UMI ReplayBuffer paths.
    zarr_paths: tuple[str, ...] = ()
    action_horizon: int = 50
    image_key: str = "camera0_rgb"
    image_mean: float = 0.5
    image_std: float = 0.5
    # Path to a norm_stats.json (quantile stats). Empty = train without normalization
    # (raw metres/rot6d). When set, the trainer loads it and bakes it into the checkpoint.
    norm_stats_path: str = ""
    # Reference frame the state + action_chunk are expressed in. 'per_keypoint'
    # (default) = each keypoint relative to its OWN anchor (historical behavior).
    # 'left'/'right'/'head' = ONE shared SE(3) frame anchored on that keypoint
    # (HoMMI gripper-centric); 'world' = absolute poses. See UMIDataset.__init__.
    ref_frame: str = "per_keypoint"
    # Observation history: To frames per sample (ending at the anchor). 1 = single
    # frame (historical). obs_history_stride = camera-frame gap between them.
    obs_history_len: int = 1
    obs_history_stride: int = 1


@dataclass
class TrainConfig:
    """Root training config (tyro CLI entry)."""

    # Which learning algorithm (and therefore which model) to train:
    #   'flow'      — flow matching (DPFlowModel, the deployed default). Uses ``model``.
    #   'diffusion' — DDPM/DDIM diffusion (DPDiffusionModel), the equal-level sibling
    #                 modelled on UMI-on-Air. Uses ``diffusion``.
    #   'act'       — Action Chunking Transformer (DPACTModel), a single-pass CVAE ported
    #                 from third_party/act. Uses ``act``.
    # The model configs are separate dataclasses (not a union) so a flow run can never
    # silently pick up a diffusion-only knob (scheduler betas, input_pertub) or an ACT-only
    # one (kl_weight, lr_backbone) or vice versa, and each is BAKED INTO ITS CHECKPOINT
    # verbatim for reproducible serving. ``ALGO_CONFIG_FIELD`` is the single mapping from
    # algo to the field/JSON key.
    algo: str = "flow"

    # Which data pipeline to train on:
    #   'hl'     — the HL-v2 contract (image + gripper + 8 force bins -> 16 steps of
    #              pose/force/gripper on a 16 Hz grid). Uses ``hl_data``.
    #   'legacy' — the frozen pre-v2 32-dim pipeline, kept so the already-trained and
    #              deployed checkpoint stays reproducible. Uses ``data``.
    # The two configs are separate rather than one union so a legacy run can never
    # silently pick up an HL-only knob (or vice versa).
    pipeline: str = "hl"

    model: ModelConfig = field(default_factory=lambda: DPFlowConfig(action_horizon=50))
    #: Diffusion model config, used ONLY when ``algo='diffusion'``. Ignored (but harmless)
    #: for flow runs. The layout-derived dims are pushed into whichever config is active.
    diffusion: DPDiffusionConfig = field(default_factory=DPDiffusionConfig)
    #: ACT model config, used ONLY when ``algo='act'``. Same "ignored but harmless"
    #: contract as ``diffusion``: it is serialized into every checkpoint regardless, and
    #: an ``act`` run's real values live here while ``model``/``diffusion`` keep defaults.
    act: DPACTConfig = field(default_factory=DPACTConfig)
    hl_data: HLDataConfig = field(default_factory=HLDataConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # EMA (exponential moving average of weights). Materially improves diffusion sample
    # quality (UMI relies on it); flag-gated so flow runs can opt in too. Default: ON for
    # diffusion, OFF for flow and ACT — resolved in __post_init__ when left at None.
    # ACT defaults OFF because upstream has no EMA at all: it selects the best checkpoint
    # by validation loss (imitate_episodes.py:351-354) and its published success rates are
    # for raw weights. Opt in with --ema if you want it.
    ema: bool | None = None
    ema_decay: float = 0.9999
    ema_warmup_power: float = 0.6666666666666666  # diffusers EMAModel default (inv_gamma=1)

    # Optimization
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    max_steps: int = 100_000
    warmup_steps: int = 1_000
    # Cosine-decay floor as a fraction of peak lr.
    lr_min_ratio: float = 0.1

    # Precision / device
    bf16: bool = True
    device: str = "cuda"
    seed: int = 0

    # Train-time RTC. When set, overrides model.rtc_delay_range with this (min, max).
    rtc_delay_range: tuple[int, int] | None = None

    # Checkpointing
    output_dir: str = "checkpoints/dp_flow"
    save_every: int = 5_000
    log_every: int = 50
    resume_from: str = ""
    # FINETUNE: a DEPLOY checkpoint dir (model.safetensors + config.json + norm_stats.json)
    # whose WEIGHTS start this run. Nothing else is carried: fresh optimizer, fresh scheduler
    # (so `warmup_steps` warms up again), fresh EMA seeded FROM the loaded weights, and
    # `global_step` back at 0. That is what separates it from `resume_from`, which is an EXACT
    # continuation and therefore requires the same config, the same norm-stats file and the same
    # data down to the anchor fingerprints.
    #
    # WHAT IS CHECKED, and it is only the two things that make the weights mean anything:
    #   * the ARCHITECTURE and the meaning of its inputs/outputs -- `algo`, `pipeline` and the
    #     ACTIVE algorithm block, plus the HL knobs that move a width or a unit
    #     (Trainer._INIT_ARCH_KEYS). A shape mismatch would be caught by load_state_dict's
    #     strict=True anyway; this catches the shape-COMPATIBLE ones, of which
    #     `wb_joint_target` absolute-vs-delta is the live example.
    #   * the norm-stats TRANSFORM knobs (`mode`, `wb_cond_mode`, `action_horizon`, the camera
    #     list, ...), so a `home`-placement checkpoint cannot be finetuned on `absolute` data.
    # The DATA is deliberately not checked -- a different dataset is the entire point.
    #
    # ⚠️ The new run fits and bundles its OWN norm stats, so every input/output range shifts a
    # little relative to what the weights learned. That is reported at startup (the per-key max
    # relative q01/q99 change) rather than hidden: a few percent is the finetune absorbing a
    # slightly different distribution, and a large number means the two datasets do not
    # describe the same task and a short finetune is the wrong tool.
    init_from: str = ""
    pin_memory: bool = True

    # Deterministic full-validation objective and bounded sample diagnostics.
    eval_seeds: tuple[int, ...] = (0, 1, 2, 3)
    sample_every_epochs: int = 5
    sample_diagnostic_batches: int = 1
    sample_diagnostic_samples: int = 4
    diagnostic_seed: int = 0
    param_norm_every: int = 10

    # A resume-capable W&B artifact is uploaded at this epoch cadence. Local checkpoint
    # failures are fatal; W&B failures are warning-only.
    artifact_every_epochs: int = 50

    # A per-epoch W&B artifact named "<run-id>-epoch_<NNN>" is uploaded at this cadence so
    # a specific epoch can be pulled on another machine. Separate from
    # artifact_every_epochs, whose single rolling "-resume" artifact keeps only its latest
    # version aliased and is meant for resuming this run, not for picking an epoch.
    # Carries the DEPLOY files only (weights + config + norm stats, ~0.6 GB), not
    # training_state.pt -- see Trainer._EPOCH_ARTIFACT_FILES. 0 disables.
    epoch_artifact_every_epochs: int = 10

    # wandb (flag-gated; off by default)
    wandb: bool = False
    wandb_project: str = "flore-s1"
    wandb_run_name: str = ""

    @property
    def active_model(self) -> ModelConfig | DPDiffusionConfig | DPACTConfig:
        """The model config the run actually trains — ``model``, ``diffusion`` or ``act``.

        Everything that derives model dimensions from the layout (``_post_init_hl``) or
        that the trainer/serving reads back MUST go through this, so a mode/horizon push
        can never land on the inactive config while the active one keeps stale defaults.

        Resolved through :data:`ALGO_CONFIG_FIELD`, NOT a chain of ``if algo == ... else
        self.model``. That shape was a live trap: any algorithm the chain did not name fell
        into the ``else`` and silently returned the FLOW config, so ``_post_init_hl`` pushed
        every layout-derived dimension onto ``model`` while the algorithm's own block kept
        its defaults — and the checkpoint then shipped with a plausible-looking but
        never-configured block. An unknown algo now raises instead."""
        try:
            return getattr(self, ALGO_CONFIG_FIELD[self.algo])
        except KeyError:
            raise ValueError(f"algo must be one of {ALGOS}, got {self.algo!r}") from None

    def __post_init__(self) -> None:
        if self.algo not in ALGO_CONFIG_FIELD:
            raise ValueError(f"algo must be one of {ALGOS}, got {self.algo!r}")
        # Two DIFFERENT operations on the same slot, and picking silently would pick wrong:
        # --resume-from continues one run (its optimizer, its step count, its data), while
        # --init-from starts a new one from another run's weights. Whichever won, the other's
        # intent would be dropped without a word.
        if self.resume_from and self.init_from:
            raise ValueError(
                "resume_from and init_from are mutually exclusive: resume_from CONTINUES a run "
                "(same config, same norm stats, same data, optimizer and step count restored), "
                "init_from STARTS one from another checkpoint's weights (fresh optimizer, step 0, "
                "new data). Pick the one you mean."
            )
        if self.algo == "flow" and (
            isinstance(self.model.inference_steps, bool)
            or not isinstance(self.model.inference_steps, Integral)
            or self.model.inference_steps < 1
        ):
            raise ValueError("model.inference_steps must be a positive integer")
        if self.algo == "flow":
            self.model.inference_steps = int(self.model.inference_steps)
        if self.algo == "diffusion":
            train_steps = self.diffusion.num_train_timesteps
            inference_steps = self.diffusion.num_inference_steps
            if self.diffusion.steps_offset != 0:
                raise ValueError("diffusion.steps_offset must be 0")
            if isinstance(train_steps, bool) or not isinstance(train_steps, Integral) or train_steps < 1:
                raise ValueError("diffusion.num_train_timesteps must be a positive integer")
            if (
                isinstance(inference_steps, bool)
                or not isinstance(inference_steps, Integral)
                or not 1 <= inference_steps <= train_steps
            ):
                raise ValueError(
                    "diffusion.num_inference_steps must be an integer in [1, num_train_timesteps], "
                    f"got {inference_steps} and {train_steps}"
                )
            self.diffusion.num_train_timesteps = int(train_steps)
            self.diffusion.num_inference_steps = int(inference_steps)
        if self.algo == "act":
            self._post_init_act()

        # RTC is a flow-only feature: BOTH siblings' samplers are single-shot or plain, so
        # there is no iterative trajectory whose prefix could be frozen. Route the CLI-level
        # range into the flow model; reject it for anything else rather than silently
        # ignore. The refusal is keyed on `algo != "flow"` (open-world), NOT on a list of
        # the non-flow algos: with the old `if algo == "diffusion"` shape, `--rtc-delay-range
        # 0 5 --algo act` skipped the refusal and then wrote `self.model.rtc_delay_range`,
        # mutating the INACTIVE flow config while the ACT run silently got no RTC at all.
        if self.rtc_delay_range is not None:
            if self.algo != "flow":
                raise ValueError(
                    f"rtc_delay_range is set but algo={self.algo!r}: only the flow-matching "
                    f"sampler supports Real-Time Chunking (the diffusion sampler is plain and "
                    f"ACT decodes the whole chunk in one forward pass — neither has an "
                    f"iterative trajectory whose prefix could be frozen). Use algo='flow' for "
                    f"train-time RTC, or drop --rtc-delay-range."
                )
            self.model.rtc_delay_range = self.rtc_delay_range

        # EMA default: ON for diffusion (UMI relies on it), OFF for flow and ACT (upstream
        # ACT has none — it selects on validation loss). Explicit wins.
        if self.ema is None:
            self.ema = self.algo == "diffusion"

        if self.pipeline not in ("hl", "legacy"):
            raise ValueError(f"pipeline must be 'hl' or 'legacy', got {self.pipeline!r}")

        # The frozen legacy 32-dim pipeline exists only to reproduce the deployed FLOW
        # checkpoint; no other algorithm has a legacy checkpoint, and their configs lack the
        # legacy knobs. Refuse the nonsensical combination up front — keyed on `!= "flow"`
        # rather than naming diffusion, because with the old shape `--pipeline legacy --algo
        # act` sailed past this guard into `_post_init_legacy`, which hardcodes
        # `self.model.*` and would have configured the FLOW config for an ACT run.
        if self.pipeline == "legacy" and self.algo != "flow":
            raise ValueError(
                f"pipeline='legacy' is flow-only (it reproduces the deployed pre-v2 "
                f"checkpoint, and only the flow config carries the legacy knobs); use "
                f"pipeline='hl' with algo={self.algo!r}."
            )

        if self.pipeline == "hl":
            self._post_init_hl()
        else:
            self._post_init_legacy()

        # clip_sample <-> normalization coupling (diffusion only). DDIM's clip_sample=True
        # clamps the predicted x0 to [-1, 1] at EVERY reverse step, which is correct ONLY
        # when the actions were normalized to ~[-1, 1] (a norm_stats.json is set). Training
        # unnormalized (raw metres/newtons, |value| > 1) with clip_sample=True is a silent
        # trap: the DDPM loss is on epsilon so training looks healthy, but the sampler can
        # never emit any action component outside [-1, 1] — the force/pose output is
        # crushed at deploy with no error. The flow model has no such clipping, so this is
        # a diffusion-specific footgun; refuse the combination rather than ship crushed
        # actions. (norm_stats lives on the ACTIVE data config: hl_data for pipeline='hl'.)
        if self.algo == "diffusion" and getattr(self.diffusion, "clip_sample", False):
            ns_path = self.hl_data.norm_stats_path if self.pipeline == "hl" else self.data.norm_stats_path
            if not ns_path:
                raise ValueError(
                    "algo='diffusion' with diffusion.clip_sample=True requires a norm_stats "
                    "path (actions must be normalized to ~[-1, 1], or DDIM's per-step x0 "
                    "clamp crushes every out-of-range pose/force at sampling). Pass "
                    "--hl-data.norm-stats-path <stats.json>, or set --diffusion.no-clip-sample "
                    "to train unnormalized."
                )

    def _post_init_act(self) -> None:
        """Validate the ACT-only knobs, mirroring what the flow/diffusion branches check.

        ACT has no sampler loop, so ``inference_steps`` is structurally 1. It is refused
        rather than coerced: a run configured with ``--act.inference-steps 10`` was written
        by someone expecting an iterative sampler, and silently normalizing it to 1 would
        leave that expectation intact while the model behaved differently.
        """
        a = self.act
        if int(a.inference_steps) != 1:
            raise ValueError(
                f"act.inference_steps must be 1 (ACT decodes the whole chunk in ONE forward "
                f"pass — there is no sampler loop to spend steps on), got {a.inference_steps}")
        if not (a.kl_weight >= 0.0 and math.isfinite(a.kl_weight)):
            raise ValueError(f"act.kl_weight must be finite and >= 0, got {a.kl_weight!r}")
        if a.latent_dim < 1:
            raise ValueError(f"act.latent_dim must be >= 1, got {a.latent_dim}")
        if a.hidden_dim % 2 != 0:
            raise ValueError(
                f"act.hidden_dim must be even (the 2D sine position embedding splits it "
                f"y/x), got {a.hidden_dim}")
        if a.hidden_dim % a.nheads != 0:
            raise ValueError(
                f"act.hidden_dim={a.hidden_dim} must be divisible by act.nheads={a.nheads}")
        if not (a.lr_backbone >= 0.0 and math.isfinite(a.lr_backbone)):
            raise ValueError(f"act.lr_backbone must be finite and >= 0, got {a.lr_backbone!r}")
        if not (a.temporal_ensemble_k >= 0.0 and math.isfinite(a.temporal_ensemble_k)):
            raise ValueError(
                f"act.temporal_ensemble_k must be finite and >= 0, got {a.temporal_ensemble_k!r}")
        if not a.state_projection:
            # ACT has no zero-padded state slot: state enters as a transformer TOKEN through
            # its own Linear, so there is no `filled_state_dim` to size. Clearing this flag
            # would send validate_model_layout down the zero-pad branch and AttributeError on
            # a field that does not exist. Refuse with the reason instead.
            raise ValueError(
                "act.state_projection cannot be False: ACT always projects state through its "
                "own Linear into a transformer token (there is no zero-padded state slot, and "
                "hence no filled_state_dim to check against).")
        if a.imagenet_renormalize and a.dataset_image_std == 0.0:
            raise ValueError("act.dataset_image_std must be non-zero to invert the dataset "
                             "image normalization")
        # ACT's objective is an L1 over the RAW action vector, which mixes metres, unitless
        # rot6d, a [0,1] gripper fraction, and N/N·m. Unnormalized, the block shares of the
        # loss are pos 7.6% / rot 22.6% / grip 21.6% / force 43.3% / TORQUE 4.9% — i.e. the
        # torque columns are barely supervised at all, and no error surfaces (the loss looks
        # healthy, the model simply never learns them). Under quantile normalization every
        # block sits in ~[-1,1] and the shares are comparable. An MSE has the same issue but
        # the flow/diffusion siblings regress a velocity/epsilon target whose scale the
        # objective sets, so this is ACT-specific. Refuse rather than train a torque-blind
        # policy; --act.no-require-norm-stats is the escape hatch for a deliberate probe.
        if a.require_norm_stats and not self.hl_data.norm_stats_path:
            raise ValueError(
                "algo='act' requires a norm_stats path: its L1 objective is taken over the "
                "raw action vector, so unnormalized the torque columns receive only ~4.9% of "
                "the loss weight and are effectively unsupervised (silently — the loss still "
                "looks healthy). Pass --hl-data.norm-stats-path <stats.json> (fit it with "
                "scripts/fit_hl_norm_stats.py), or --act.no-require-norm-stats to override.")
        from flore.s1_policy.model.act.detr_backbone import RESNET_SPECS

        if a.backbone not in RESNET_SPECS:
            raise ValueError(
                f"act.backbone must be one of {sorted(RESNET_SPECS)}, got {a.backbone!r}")

    def _post_init_umi_parity(self, m) -> None:
        """Validate ``encoder_lr_scale`` on whichever model config is active.

        Only this one knob is checked here, and deliberately so: it is consumed by the
        OPTIMIZER (``Model.param_groups`` -> AdamW), so nothing downstream would catch a
        bad value — ``0`` would freeze the ViT silently and a negative would ascend its
        gradient. The ``image_aug_*`` knobs are instead validated by
        :class:`flore.s1_policy.model.umi_parity.ImageAugment`, which owns their meaning;
        re-checking them here would be two places to drift.

        ACT carries neither field (it has its own ``lr_backbone`` split), so this keys on
        the field's PRESENCE rather than on ``algo`` — an algorithm that grows the knob
        later gets the check for free instead of silently skipping it.
        """
        scale = getattr(m, "encoder_lr_scale", None)
        if scale is None:
            return
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError(f"{self.algo}.encoder_lr_scale must be a number, got {scale!r}")
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                f"{self.algo}.encoder_lr_scale must be finite and > 0 (it MULTIPLIES --lr for "
                f"the pretrained ViT; 0 would freeze it silently), got {scale!r}")

    def _post_init_hl(self) -> None:
        """Derive every model dimension from the LAYOUT, so a mode change cannot leave
        the model and the data disagreeing about what a column means.

        Pushes land on :attr:`active_model` (the flow OR the diffusion config, per
        ``algo``) — both dataclasses share these field names by design, so a single push
        path serves both models."""
        from flore.s1_policy.data.layout import (
            ACTION_HORIZON,
            FORCE_HIST_LEN,
            WB_JOINT_TARGETS,
            get_layout,
        )

        m = self.active_model
        layout = get_layout(self.hl_data.mode)  # raises on an unknown mode
        if self.hl_data.val_last_episodes < 0:
            raise ValueError("hl_data.val_last_episodes must be >= 0")
        # Checked HERE as well as in HLDataset, because this is the failure the operator sees
        # first: a typo'd `--hl-data.wb-joint-target detla` would otherwise build every model and
        # open W&B before the dataloader refused it.
        if self.hl_data.wb_joint_target not in WB_JOINT_TARGETS:
            raise ValueError(
                f"hl_data.wb_joint_target must be one of {WB_JOINT_TARGETS}, got "
                f"{self.hl_data.wb_joint_target!r}")
        if self.hl_data.wb_joint_target != "absolute" and not layout.is_wholebody:
            raise ValueError(
                f"hl_data.wb_joint_target={self.hl_data.wb_joint_target!r} needs "
                f"--hl-data.mode wholebody; mode={self.hl_data.mode!r} has no joint block.")
        # Refused rather than silently ignored, per block: a sigma the layout has nowhere to put
        # would make StateNoise a no-op, and "the ablation ran without its noise" is invisible
        # in the loss, in the checkpoint's widths and in W&B. Checked HERE (not only in the
        # dataset) so the operator sees it before N models are built and W&B opens.
        #
        # Each sigma names the blocks it can land on BY UNIT, not one block by name: `rel_xy`
        # (wholebody_geom_*) and `rel_xy3` (fivepoint_*) are both metres and both take the xy
        # sigma. Keeping the mapping here parallel to `wb_geom.noise_sigma`'s `per_block` is what
        # makes "the flag was accepted but jittered nothing" impossible in either direction --
        # that function refuses a block with no unit, and this refuses a unit with no block.
        geom_slots = layout.state_geom_slots()
        for field_name, value, blocks in (
            ("state_noise_xy_m", self.hl_data.state_noise_xy_m, ("rel_xy", "rel_xy3")),
            ("state_noise_rpy_deg", self.hl_data.state_noise_rpy_deg, ("rpy",)),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"hl_data.{field_name} must be finite and >= 0, got {value!r}")
            if value and not any(b in geom_slots for b in blocks):
                raise ValueError(
                    f"hl_data.{field_name}={value} jitters the "
                    f"{' / '.join(repr(b) for b in blocks)} state block(s), and "
                    f"mode={self.hl_data.mode!r} carries none of them (its state blocks are "
                    f"{list(geom_slots) or 'none'}). Train a mode that includes one, or leave "
                    f"the sigma at 0.")
        if not self.eval_seeds:
            raise ValueError("eval_seeds must contain at least one deterministic seed")
        if layout.legacy:
            raise ValueError(
                "pipeline='hl' cannot use the frozen 'legacy32' layout; "
                "use pipeline='legacy' for the pre-v2 checkpoint."
            )
        # PUSHED, not validated: the layout is the single source of truth, so a CLI
        # able to set these independently would be a footgun with no upside.
        m.action_dim = layout.action_dim
        m.action_horizon = ACTION_HORIZON
        m.state_dim = layout.state_dim
        # The zero-pad truncation check applies ONLY to the flat-global_cond models, whose
        # `_pad_state` right-pads state into a `filled_state_dim` slot and would silently
        # truncate via a negative F.pad width. ACT has no such slot (state enters as a
        # transformer TOKEN through its own Linear), so it declares state_projection=True
        # and has no filled_state_dim at all. Defaulting the getattr to True is the safe
        # polarity: it SKIPS the check for a config with no zero-pad path, where
        # `getattr(..., False)` would fall through and AttributeError on filled_state_dim.
        if not getattr(m, "state_projection", True) and m.filled_state_dim < m.state_dim:
            raise ValueError(
                f"filled_state_dim={m.filled_state_dim} would truncate HL "
                f"state_dim={m.state_dim}; increase it or enable state_projection"
            )
        # HL feeds obs_history_len frames (image + state), oldest-first, ending at t0
        # (DECISION-OBS2). Pushed so the model sizes global_cond for To frames; must be
        # >= 1. The dataset uses the same value, so a mismatch cannot arise.
        if self.hl_data.obs_history_len < 1:
            raise ValueError(f"hl_data.obs_history_len must be >= 1, got {self.hl_data.obs_history_len}")
        m.obs_history_len = self.hl_data.obs_history_len
        # Baked into the checkpoint: a bare 16-dim output is ambiguous between
        # left_only and right_only, so serving must be able to read the mode back. It also
        # carries the DECLARED hardware — both_hands_head and both_hands_head_dual have the
        # same 41 action columns and differ only in which of them are measured — and the
        # force encoder derives its per-sensor group count from it.
        m.layout_mode = self.hl_data.mode
        # PUSHED like every other width: the vision encoder's output size is
        # num_cameras * To * features, so a num_cameras that disagreed with the number of
        # image keys the dataset actually stacks would mis-size global_cond. The refusal
        # would be loud (the decoder's cond projection rejects the width) but would read as
        # a model-config typo rather than a data/model disagreement.
        m.num_cameras = len(self.hl_data.resolved_image_keys())

        # Force-history conditioning. The WIDTH comes from DECLARED hardware, not the
        # reserved columns (see layout.py "RESERVED vs DECLARED"): a rig with one F/T
        # sensor feeds 6 channels even in a bimanual mode that reserves 12 columns.
        # channels == 0 means no encoder is built.
        #
        # `force_input_channels`, NOT `force_hist_channels`: the two differ only under the
        # force-output-only ablation (FLORE_DROP_FORCE_INPUT), where the model must be built
        # WITHOUT a force encoder while the dataset keeps emitting the real 6-channel window
        # and the force action columns stay supervised. Pushing the dataset width here would
        # build the encoder and quietly defeat the ablation.
        m.force_hist_channels = layout.force_input_channels
        m.force_hist_len = FORCE_HIST_LEN if layout.force_input_channels else 0

        # Every model needs the dataset's own image affine, for two different reasons, and
        # in both cases it is PUSHED (never set independently) because a drift silently
        # feeds the backbone out-of-distribution input:
        #   * ACT INVERTS it before its ImageNet re-normalization — the exact failure the
        #     re-normalization exists to prevent.
        #   * flow/diffusion invert it inside ImageAugment before ColorJitter, which is only
        #     defined on [0,1]; a wrong affine makes every jitter strength mean something
        #     other than UMI's value.
        # All three configs carry these two field names, so one push serves them all.
        m.dataset_image_mean = float(self.hl_data.image_mean)
        m.dataset_image_std = float(self.hl_data.image_std)

        self._post_init_umi_parity(m)

    def _post_init_legacy(self) -> None:
        # Validate the reference-frame knob (dataset enum) up front.
        _ref_choices = ("per_keypoint", "left", "right", "head", "world")
        if self.data.ref_frame not in _ref_choices:
            raise ValueError(f"data.ref_frame must be one of {_ref_choices}, got {self.data.ref_frame!r}")

        # The model must know To to size global_cond (vision + state conditioning both
        # scale by To). The DATA config owns To; push it into the model so a mismatched
        # global_cond width can never be built.
        if self.data.obs_history_len < 1:
            raise ValueError(f"data.obs_history_len must be >= 1, got {self.data.obs_history_len}")
        self.model.obs_history_len = self.data.obs_history_len

        # The DATA config owns the horizon; the model must agree or the decoder emits a
        # chunk of a different length than the labels. This was previously NOT pushed
        # (unlike obs_history_len), so `--data.action-horizon 16` without a matching
        # `--model.action-horizon 16` silently trained a 50-step model on 16-step labels.
        if self.data.action_horizon != self.model.action_horizon:
            raise ValueError(
                f"data.action_horizon={self.data.action_horizon} != "
                f"model.action_horizon={self.model.action_horizon}. Pass BOTH "
                f"(--data.action-horizon N --model.action-horizon N)."
            )

        # Same push as the HL branch, from the legacy data config's own affine, so
        # ImageAugment can invert it. legacy is flow-only (guarded above).
        self.model.dataset_image_mean = float(self.data.image_mean)
        self.model.dataset_image_std = float(self.data.image_std)
        self._post_init_umi_parity(self.model)
