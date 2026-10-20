"""DP-Diffusion VLA: ViT encoder + Unet/DiT decoder with DDPM/DDIM diffusion.

The equal-level sibling of :class:`flore.s1_policy.model.dp_flow.DPFlowModel`. Same inputs
(1+ camera images, per-keypoint ``wrt_start`` state, an 8-bin causal force history), same
16-dim HL-v2 action output ``[pos3 rot6d6 grip1 force6]`` per hand, same decoders — but the
objective is discrete DDPM training / DDIM sampling (via ``diffusers``) instead of
flow-matching. Selected by ``TrainConfig.algo='diffusion'``.

Modelled on UMI-on-Air's ``DiffusionUnetTimmPolicy`` (DDIM, ε-prediction,
``squaredcos_cap_v2`` schedule, input perturbation, multi-sample-per-obs) but
REIMPLEMENTED against FLoRe's own encoder/decoder/dataset rather than importing any UMI
code, and extended so force is BOTH an input (the force-history encoder) AND an output
(the 6 force columns of every hand block) — UMI's force is input-only.

STANDALONE by design (user decision 2026-07-31): this module does NOT touch
``dp_flow.py`` and carries its OWN copy of the observation-encoding logic, so the deployed
flow-matching checkpoint is byte-for-byte unaffected. The leaf modules that are already
self-contained — :class:`ForceHistoryEncoder` and the two decoders — are IMPORTED and
shared (copying them would only invite drift).

Architecture (identical to DP-Flow up to the decoder's role)::

    Images (n_cams x To)  -> ViT -> CLS per frame -\
    state (To, SD)        -> zero-pad/proj --------  concat -> global_cond
    force_history (L, C)  -> ForceHistoryEncoder --/
                                                   |

``diffusion_umi_3tracker`` (layout ``both_hands_head_dual``) is this same model with the
widths the two-hand rig implies — nothing about the architecture changes:

======================  =============  ==============================================
                        left_only      both_hands_head_dual
======================  =============  ==============================================
cameras                 1 (768 feat)   **2** (1536) — both wrist D405s, one CLS each
state                   7              **20** — L/R [rot6d 6 + grip 1] + head [rot6d 6]
force in                (1, 6)         **(1, 12)** — 2 sensors, ONE ENCODER GROUP EACH
action out              50 x 16        50 x **41** — L/R [pos rot6d grip force], head pose
======================  =============  ==============================================

The two design choices worth stating, since neither is forced:

* **Two cameras are two TOKENS, not a 6-channel image.** They are different viewpoints (one
  per hand), the pretrained ViT is a 3-channel model, and separate CLS features let the
  decoder's conditioning attend to each hand's own view. The cost is the honest one: the ViT
  forward is the dominant term, so this is ~2x the encoder compute.
* **Each F/T sensor gets its own encoder group.** See
  :mod:`flore.s1_policy.model.force_encoder` — per-hand validity gating is only expressible
  that way, and the two wrenches are in different (per-sensor) frames.
    Decoder (ConditionalDiT1D | ConditionalUnet1D):
        input: x_t (noised actions), integer timestep;  condition: global_cond
        output: ε (or the scheduler's target)
        |  DDIM reverse loop (diffusers scheduler)
        v
    Actions [B, action_horizon, action_dim]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from flore.s1_policy.model.diffusion_base import DiffusionVLABase
from flore.s1_policy.model.dit import ConditionalDiT1D, ConditionalUnet1D

logger = logging.getLogger(__name__)


#: Shared with DP-Flow: timm backbone short-name -> full model id.
VIT_MODEL_MAP = {
    "vit_ti": "vit_tiny_patch16_224",
    "vit_s": "vit_small_patch16_224",
    "vit_b": "vit_base_patch16_clip_224.openai",
    "vit_l": "vit_large_patch14_clip_224.openai",
    "vit_h": "vit_huge_patch14_clip_224.laion2b",
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_l": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_g": "vit_giant_patch14_dinov2.lvd142m",
}


@dataclass
class DPDiffusionConfig:
    """Configuration for the DP-Diffusion VLA (ViT encoder + Unet/DiT decoder + DDIM).

    Field names shared with :class:`DPFlowConfig` are kept IDENTICAL so ``TrainConfig``'s
    layout-derived pushes (``action_dim``, ``state_dim``, ``obs_history_len``,
    ``force_hist_*``, ``layout_mode``) apply unchanged to either model.
    """

    # ViT backbone
    vit: str = "vit_b"
    pretrained: bool = True

    # Decoder selection: "unet" (UMI's default, FiLM) or "dit" (AdaLN + RoPE).
    # UMI's DiffusionUnetTimmPolicy uses the UNet; we default to it for numerical parity,
    # but the DiT works too (its per-position time is simply unused without RTC).
    diffusion_net: str = "unet"

    # Action space (pushed from the layout by TrainConfig for the HL pipeline).
    action_dim: int = 16
    action_horizon: int = 16
    state_dim: int = 7
    num_cameras: int = 1
    # Observation history: To frames per sample (>=1), oldest-first, ending at t0. To == 2
    # is UMI's 2-frame obs; matches DP-Flow's obs_history handling exactly.
    obs_history_len: int = 1

    # Shared decoder params
    diffusion_step_embed_dim: int = 128  # UMI uses 128 for the timm UNet policy

    # Unet-specific (UMI's values)
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True

    # DiT-specific (used only when diffusion_net == "dit")
    model_channels: int = 512
    num_blocks: int = 12
    n_heads: int = 16
    mlp_mult: int = 4
    pos_emb_type: str = "rope"
    max_seq_len: int = 64

    # ---- Diffusion process (diffusers DDIMScheduler), UMI's exact config ----
    num_train_timesteps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    #: Clip the predicted x0 to [-1, 1] each reverse step. CORRECT ONLY when actions are
    #: normalized to ~[-1, 1] (i.e. a norm_stats.json is set) — exactly UMI's regime. If you
    #: train unnormalized (raw metres / newtons), set this False or the sampler will crush
    #: every target into [-1, 1].
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    #: "epsilon" (predict the noise, UMI's default), "sample" (predict x0), or
    #: "v_prediction".
    prediction_type: str = "epsilon"
    #: Reverse-diffusion steps at inference (UMI: 16, well below num_train_timesteps=50).
    num_inference_steps: int = 16
    #: DDPM-IP input perturbation strength (0 disables). https://github.com/forever208/DDPM-IP
    input_pertub: float = 0.1
    #: UMI's train_diffusion_n_samples: K independent (noise, t) draws per encoder forward.
    #: Default 1 — UMI warns that raising it too far destabilizes diffusion training.
    num_draws: int = 1

    # Precision
    precision: str = "float32"

    # ---- UMI training-recipe parity (see flore.s1_policy.model.umi_parity) ----
    #: Train-time image augmentation (UMI's RandomCrop 0.95 + Resize + ColorJitter). OFF by
    #: default so an existing run stays byte-identical; the strengths below are UMI's
    #: values. Applied ONLY in train mode, unlike upstream (which also augments at val and
    #: at deploy) — otherwise a val curve would move for two reasons at once.
    image_aug: bool = False
    image_aug_crop_ratio: float = 0.95
    image_aug_brightness: float = 0.3
    image_aug_contrast: float = 0.4
    image_aug_saturation: float = 0.5
    image_aug_hue: float = 0.08
    #: LR multiplier for the pretrained ViT relative to ``--lr``. UMI uses 0.1 whenever the
    #: obs encoder is pretrained (train_diffusion_unet_image_workspace.py:62); 1.0 is
    #: FLoRe's historical single-LR behavior. Does NOT touch ``force_encoder`` (from scratch).
    encoder_lr_scale: float = 1.0
    #: The dataset's own image affine, needed to INVERT it before ColorJitter (which is only
    #: defined on [0,1]). PUSHED from hl_data/data by TrainConfig, never set independently —
    #: a drift here would make the jitter strengths mean something other than UMI's.
    dataset_image_mean: float = 0.5
    dataset_image_std: float = 0.5

    # State conditioning (same contract as DP-Flow).
    filled_state_dim: int = 64
    state_projection: bool = False
    state_proj_mult: int = 4

    # Force-history conditioning (HL-v2). A (force_hist_len, force_hist_channels) causal
    # wrench window encoded by ForceHistoryEncoder, appended LAST to global_cond.
    # force_hist_channels == 0 -> no force encoder (image + state only).
    force_hist_len: int = 0
    force_hist_channels: int = 0
    force_enc_width: int = 128
    force_enc_out_dim: int = 64

    #: HL-v2 layout mode, pushed in by TrainConfig and BAKED INTO THE CHECKPOINT so serving
    #: can slice the 16-dim output unambiguously (left_only vs right_only). Empty = no HL
    #: layout (per-group metrics are then skipped).
    layout_mode: str = ""


class DPDiffusionModel(DiffusionVLABase):
    """DP-Diffusion VLA: ViT encoder + configurable Unet/DiT decoder, DDPM/DDIM objective.

    Parameters
    ----------
    config : DPDiffusionConfig
    encoder : nn.Module, optional
        Pluggable image encoder that REPLACES the timm backbone (used in tests to avoid
        downloads). Must expose an int ``num_features`` and map ``(N, C, H, W)`` to either
        ``(N, num_features)`` or ``(N, tokens, num_features)`` (CLS token at index 0).
    """

    def __init__(self, config: DPDiffusionConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config

        dtype = self._parse_dtype(config.precision)

        if encoder is not None:
            self.encoder = encoder.to(dtype)
        else:
            import timm

            vit_timm_name = VIT_MODEL_MAP[config.vit]
            self.encoder = timm.create_model(
                model_name=vit_timm_name,
                pretrained=config.pretrained,
                global_pool="",
                num_classes=0,
                img_size=224,
            ).to(dtype)
        if not isinstance(getattr(self.encoder, "num_features", None), int):
            raise ValueError("encoder must expose an int `num_features` attribute")

        obs_hist = max(1, getattr(config, "obs_history_len", 1))
        # `enc_out_dim` below multiplies the two, and HLDataset emits the flattened list in
        # CAMERA-MAJOR, oldest-first-within-a-camera order — so the width is right for any
        # (num_cameras, To) pair. The refusal stands anyway: nothing has VERIFIED the two
        # axes together end to end (the v3 contract is To == 1, which is what
        # diffusion_umi_3tracker's 2 cameras run at), and a silently transposed 4-entry list
        # would train perfectly happily on cameras it believed were history frames.
        if config.num_cameras > 1 and obs_hist > 1:
            raise NotImplementedError(
                f"num_cameras={config.num_cameras} with obs_history_len={obs_hist} is not "
                f"supported: cameras and history frames are flattened into ONE images list, "
                f"and their interleaving is only pinned by a test at To == 1. Use 2 cameras "
                f"at To == 1 (the v3 contract), or 1 camera with history.")
        enc_out_dim = config.num_cameras * obs_hist * self.encoder.num_features
        # State contributes either filled_state_dim (zero-pad) or a learned projection
        # width per frame; _build_state_encoder (in the base) sets self.state_proj and
        # returns the PER-FRAME width, scaled by To here.
        state_cond_width = self._build_state_encoder() * obs_hist

        # Force-history encoder (HL-v2), appended LAST to global_cond.
        self.force_encoder: nn.Module | None = None
        force_width = 0
        if getattr(config, "force_hist_channels", 0) > 0:
            from flore.s1_policy.model.force_encoder import ForceHistoryEncoder

            self.force_encoder = ForceHistoryEncoder(
                hist_len=config.force_hist_len,
                channels=config.force_hist_channels,
                width=config.force_enc_width,
                out_dim=config.force_enc_out_dim,
                groups=self._force_groups(),
            ).to(dtype)
            # NOT `force_enc_out_dim`: with two F/T sensors the encoder holds one
            # independent stack per sensor and concatenates them, so it contributes
            # groups * out_dim. Reading the config field directly would under-size
            # global_cond by 64 and the decoder's cond projection would refuse — loudly,
            # but for a reason that reads like a config typo.
            force_width = self.force_encoder.total_out_dim

        global_cond_dim = enc_out_dim + state_cond_width + force_width

        if config.diffusion_net == "unet":
            self.decoder = ConditionalUnet1D(
                input_dim=config.action_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                down_dims=list(config.down_dims),
                kernel_size=config.kernel_size,
                n_groups=config.n_groups,
                cond_predict_scale=config.cond_predict_scale,
            ).to(dtype)
        elif config.diffusion_net == "dit":
            self.decoder = ConditionalDiT1D(
                input_dim=config.action_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                model_channels=config.model_channels,
                num_blocks=config.num_blocks,
                n_heads=config.n_heads,
                mlp_mult=config.mlp_mult,
                pos_emb_type=config.pos_emb_type,
                max_seq_len=config.max_seq_len,
            ).to(dtype)
        else:
            raise ValueError(f"Unknown diffusion_net: {config.diffusion_net!r}. Expected 'unet' or 'dit'.")

        self._build_noise_scheduler()

        # UMI-parity train-time augmentation. Built AFTER `self.to(dtype)` would be wrong
        # (it holds no parameters, but `.to()` on an nn.Sequential of transforms is a no-op
        # anyway), and it must exist before `_init_rtc_state` for a stable module order.
        from flore.s1_policy.model.umi_parity import build_image_augment

        self.image_aug = build_image_augment(config)

        self.to(dtype)
        self._init_rtc_state()

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        logger.info(
            "DPDiffusionModel: vit=%s, decoder=%s, scheduler=DDIM(T=%d,%s,%s), "
            "enc_params=%d, dec_params=%d, total=%d, image_aug=%s, encoder_lr_scale=%.3g",
            config.vit, config.diffusion_net, config.num_train_timesteps,
            config.beta_schedule, config.prediction_type,
            enc_params, dec_params, enc_params + dec_params,
            self.image_aug is not None, config.encoder_lr_scale,
        )

    def _force_groups(self) -> int:
        """How many INDEPENDENT F/T sensors ``force_hist_channels`` packs.

        DERIVED from ``layout_mode`` rather than carried as its own config field, for the
        same reason ``action_dim``/``state_dim`` are pushed from the layout: the group count
        and the channel width describe the same hardware, and two independently settable
        numbers can disagree (12 channels in 1 group trains a model that silently cannot
        gate either sensor's validity). ``layout_mode`` is already baked into every
        checkpoint, so serving re-derives the same number with nothing extra to store.

        Falls back to 1 when no layout is declared — the pre-HL/test path, where
        ``force_hist_channels`` is whatever the caller passed and one group is the only
        interpretation that cannot be wrong.
        """
        mode = getattr(self.config, "layout_mode", "") or ""
        if not mode:
            return 1
        from flore.s1_policy.data.layout import get_layout

        return max(1, len(get_layout(mode).force_roles))

    def param_groups(self, lr: float) -> list[dict] | object:
        """AdamW parameter groups implementing UMI's reduced encoder LR.

        The trainer calls this when a model defines it. At the default
        ``encoder_lr_scale=1.0`` it returns ``self.parameters()``, i.e. the same
        single-group AdamW as before this hook existed.
        """
        from flore.s1_policy.model.umi_parity import encoder_lr_param_groups

        return encoder_lr_param_groups(self, lr, float(self.config.encoder_lr_scale))

    # ------------------------------------------------------------------
    # Per-group metrics (logging only). Mirrors DP-Flow but keyed "diffusion_loss/".
    # ------------------------------------------------------------------
    def _metric_group_defs(self) -> dict[str, tuple[int, ...]]:
        """Group name -> action dims, resolved from the layout when layout_mode is set.

        No legacy fallback (the diffusion model has no pre-v2 checkpoint to preserve): with
        no layout_mode, per-group metrics are simply skipped.
        """
        mode = getattr(self.config, "layout_mode", "") or ""
        if not mode:
            return {}
        from flore.s1_policy.data.layout import get_layout

        return get_layout(mode).metric_group_dims()

    @torch.no_grad()
    def _extra_train_metrics(self, pred, target, action_is_pad, action_dim_mask, action_mask=None):
        """Per-group masked MSE (logging only), directly comparable to ``diffusion_loss``.

        Uses the SAME time-include mask (temporal-pad) AND the SAME per-dim
        ``action_dim_mask`` / ``action_mask`` composition as :meth:`_compute_diffusion_loss`,
        so each ``diffusion_loss/<group>`` is that same numerator/denominator ratio
        restricted to a group's dims. Vectorized to one GPU->CPU sync.
        """
        groups = self._metric_group_defs()
        if not groups:
            return {}

        sq_err = (pred - target) ** 2  # (B, T, D)

        include = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        if action_is_pad is not None:
            include = include & ~action_is_pad
        time_w = include.unsqueeze(-1).float()  # (B, T, 1)

        if action_dim_mask is not None:
            weight = time_w * action_dim_mask.unsqueeze(1).float()
        else:
            weight = time_w.expand_as(sq_err)
        if action_mask is not None:
            weight = weight * action_mask.to(weight.dtype)

        group_sel = getattr(self, "_metric_group_sel", None)
        if group_sel is None or group_sel.device != pred.device or group_sel.shape[1] != sq_err.shape[-1]:
            action_dim = sq_err.shape[-1]
            names = list(groups.keys())
            sel = torch.zeros(len(names), action_dim, dtype=torch.float32, device=pred.device)
            for gi, name in enumerate(names):
                idx = [d for d in groups[name] if d < action_dim]
                if idx:
                    sel[gi, idx] = 1.0
            self._metric_group_names = names
            self._metric_group_sel = sel
            group_sel = sel

        per_dim_num = (sq_err * weight).sum(dim=(0, 1))  # (D,)
        per_dim_den = weight.sum(dim=(0, 1))  # (D,)
        num = group_sel @ per_dim_num  # (G,)
        den = (group_sel @ per_dim_den).clamp(min=1.0)  # (G,)
        vals = (num / den).tolist()
        return {f"diffusion_loss/{name}": v for name, v in zip(self._metric_group_names, vals)}

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load weights from a safetensors checkpoint saved by the trainer."""
        from pathlib import Path

        import safetensors.torch

        model_path = Path(checkpoint_dir) / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"model.safetensors not found at {model_path}")
        state_dict = safetensors.torch.load_file(str(model_path), device="cpu")
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("DPDiffusionModel.load_checkpoint: missing keys: %s", missing)
        if unexpected:
            logger.warning("DPDiffusionModel.load_checkpoint: unexpected keys: %s", unexpected)
        logger.info("DPDiffusionModel: loaded checkpoint from %s", model_path)

    # ------------------------------------------------------------------
    # Observation encoding — OWN copy of DP-Flow's logic (standalone by design).
    # ------------------------------------------------------------------
    def _encode_vision(self, images: list[Tensor], img_masks: list[Tensor] | None) -> Tensor:
        """Encode a list of per-camera (or per-history-frame) image batches into flat
        vision features. Masked-out images contribute a zero feature. Returns
        ``(B, num_entries * num_features)``."""
        imgs = torch.stack(images, dim=1).to(self.dtype)  # (B, N, C, H, W)
        batch_size, num_imgs = imgs.shape[:2]
        img_shape = imgs.shape[2:]

        if img_masks is None:
            masks = torch.ones(batch_size, num_imgs, dtype=torch.bool, device=imgs.device)
        else:
            masks = torch.stack(img_masks, dim=1)

        flat_images = imgs.reshape(batch_size * num_imgs, *img_shape)
        flat_masks = masks.reshape(batch_size * num_imgs)

        valid_images = flat_images[flat_masks]
        if valid_images.numel() > 0:
            # Augment only the frames that actually reach the encoder (a no-op in eval
            # mode). Applied to the FLAT (N, C, H, W) batch, exactly where UMI's
            # TimmObsEncoder.forward applies its own transform stack.
            if self.image_aug is not None:
                valid_images = self.image_aug(valid_images)
            valid_features = self.encoder(valid_images)
            if valid_features.ndim == 3:
                valid_features = valid_features[:, 0, :]  # CLS token
            flat_features = torch.zeros(
                batch_size * num_imgs, valid_features.shape[-1], device=imgs.device, dtype=self.dtype)
            flat_features[flat_masks] = valid_features.to(self.dtype)
        else:
            flat_features = torch.zeros(
                batch_size * num_imgs, self.encoder.num_features, device=imgs.device, dtype=self.dtype)

        return flat_features.reshape(batch_size, -1)

    def _encode_global_cond(self, batch_dict: dict) -> Tensor:
        """Encode images + state (+ force history) into a flat conditioning vector.

        Same assembly order as DP-Flow (vision | state | force?), so the two models size and
        populate global_cond identically. FLoRe has no language, so there is no language
        slot (see docs/architecture.md)."""
        images = batch_dict["images"]
        img_masks = batch_dict.get("img_masks")
        state = batch_dict.get("state")

        vision_features = self._encode_vision(images, img_masks)
        batch_size = vision_features.shape[0]
        device = vision_features.device

        # Zero any observation entry the dataset flagged fabricated (state_valid False),
        # e.g. a dead-encoder gripper zero — 0.0 is a LEGAL width, so an ungated entry would
        # feed the model a made-up measurement.
        if state is not None:
            sv = batch_dict.get("state_valid")
            if sv is not None:
                state = state * sv.to(state.device, state.dtype)

        filled_state = self._project_state(state, batch_size, device)

        parts = [vision_features, filled_state]

        if self.force_encoder is not None:
            fh = batch_dict.get("force_history")
            if fh is None:
                raise KeyError(
                    "this model was built with a force encoder "
                    f"(force_hist_channels={self.config.force_hist_channels}) but the batch "
                    "has no 'force_history'. A silently zero-filled window would read as 'no "
                    "contact' — pass the real window or build the model without force.")
            fhv = batch_dict.get("force_history_valid")
            parts.append(self.force_encoder(
                fh.to(device),
                fhv.to(device) if fhv is not None else None).to(self.dtype))

        return torch.cat(parts, dim=1)
