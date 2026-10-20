"""DP-Flow VLA: ViT encoder + Unet/DiT decoder with flow matching.

Ported from GCL (``covariant/models/gam/gcl/policies/networks/vit_dit.py``). Strips
the GCL component registry, the matplotlib serving-debug plumbing (``_GlobalCondDebugger``),
and the language path entirely (FLoRe has no language — see docs/architecture.md). The
flow-matching and RTC logic lives in :class:`flore.s1_policy.model.base.FlowMatchingVLABase`.

Architecture::

    Images (1+ cameras)
        |  ViT encoder (timm, configurable backbone; or an injected encoder)
        v
    CLS token per camera → flatten → vision features
        |  concat with state (+ force history when a force encoder is built)
        v
    global_cond = [vision | state | force?]
        |
        v
    Decoder (ConditionalDiT1D [default] or ConditionalUnet1D)
        input: x_t, timestep;  condition: global_cond;  output: velocity v_t
        |  Euler ODE integration (+ RTC frozen prefix)
        v
    Actions [B, action_horizon, action_dim]

Configurable ViT backbones (via timm): vit_ti/vit_s/vit_b/vit_l/vit_h, dinov2_s/b/l/g.
Configurable decoders: "dit" (AdaLN + RoPE, DEFAULT, RTC-capable) or "unet" (FiLM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from flore.s1_policy.model.base import FlowMatchingVLABase
from flore.s1_policy.model.dit import ConditionalDiT1D, ConditionalUnet1D

logger = logging.getLogger(__name__)


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
class DPFlowConfig:
    """Configuration for the DP-Flow VLA (ViT encoder + Unet/DiT decoder).

    Plain (non-frozen) dataclass so tyro can build it from the CLI and the trainer
    can override fields. Defaults target the FLoRe 3-point setup: 32D action/state
    (29 real + 3 pad), 50-step horizon, single 224x224 camera.
    """

    # ViT backbone
    vit: str = "vit_b"
    pretrained: bool = True

    # Decoder selection: "dit" (default, RTC-capable) or "unet" (FiLM, no RTC).
    diffusion_net: str = "dit"

    # Action space
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32
    num_cameras: int = 1
    # Observation history: To frames per sample (>=1). The dataset yields To images
    # per camera and a (To, state_dim) state; the encoder treats the To frames as To
    # extra "pseudo-cameras" (concat their CLS features) and concatenates the To state
    # frames into global_cond. To == 1 (default) is the historical single-frame path.
    obs_history_len: int = 1

    # Shared decoder params
    diffusion_step_embed_dim: int = 256

    # Unet-specific
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True

    # DiT-specific
    model_channels: int = 512
    num_blocks: int = 12
    n_heads: int = 16
    mlp_mult: int = 4
    pos_emb_type: str = "rope"
    max_seq_len: int = 64

    # Inference
    inference_steps: int = 10

    # Flow matching (Beta(1.5, 1.0) time sampling)
    flow_beta_a: float = 1.5
    flow_beta_b: float = 1.0

    # RTC (Real-Time Chunking). (min_delay, max_delay) or None to disable train-time RTC.
    rtc_delay_range: tuple[int, int] | None = None

    # Multi-draw: sample K independent (noise, timestep) per example to amortize
    # the expensive encoder forward across multiple gradient signals.
    num_draws: int = 8

    # Precision
    precision: str = "float32"

    # ---- UMI training-recipe parity (see flore.s1_policy.model.umi_parity) ----
    # Field names are IDENTICAL to DPDiffusionConfig's, like every other shared knob here,
    # so TrainConfig's single push path serves both models.
    #: Train-time image augmentation (UMI's RandomCrop 0.95 + Resize + ColorJitter). OFF by
    #: default so the deployed flow checkpoint's recipe is unchanged. Train mode only.
    image_aug: bool = False
    image_aug_crop_ratio: float = 0.95
    image_aug_brightness: float = 0.3
    image_aug_contrast: float = 0.4
    image_aug_saturation: float = 0.5
    image_aug_hue: float = 0.08
    #: LR multiplier for the pretrained ViT relative to ``--lr`` (UMI: 0.1 when the obs
    #: encoder is pretrained). 1.0 is FLoRe's historical single-LR behavior. Excludes
    #: ``force_encoder`` (from scratch, keeps the full rate).
    encoder_lr_scale: float = 1.0
    #: The dataset's own image affine, needed to INVERT it before ColorJitter (only defined
    #: on [0,1]). PUSHED from the active data config by TrainConfig, never set by hand.
    dataset_image_mean: float = 0.5
    dataset_image_std: float = 0.5

    # State conditioning.
    filled_state_dim: int = 64
    # When True, learn nn.Linear(action_dim, filled_state_dim * state_proj_mult)
    # instead of zero-padding state into the cond vector.
    state_projection: bool = False
    state_proj_mult: int = 4  # projected width = filled_state_dim * mult

    # Force-history conditioning (HL-v2). The observation carries a (force_hist_len,
    # force_hist_channels) causal wrench window, encoded by a small causal Conv1d
    # stack (see flore/s1_policy/model/force_encoder.py) whose output is appended to
    # global_cond. A TIME SERIES, not a flat vector: the temporal profile of a contact
    # is the signal, and flattening it into `state` would also overflow
    # filled_state_dim and be silently truncated by _pad_state.
    #   force_hist_channels == 0 (default) -> no force encoder at all, and the model is
    #   byte-identical to the pre-force architecture.
    force_hist_len: int = 0
    force_hist_channels: int = 0
    force_enc_width: int = 128
    #: Width contributed to global_cond by the force encoder.
    force_enc_out_dim: int = 64

    #: HL-v2 layout mode (``left_only`` | ``right_only`` | ``both_hands`` |
    #: ``both_hands_head``), pushed in by ``TrainConfig`` and BAKED INTO THE
    #: CHECKPOINT. It is what tells serving how to slice this model's output: without
    #: it a 16-dim vector is ambiguous between left_only and right_only. Empty string
    #: = the frozen pre-v2 32-dim layout.
    layout_mode: str = ""


# Per-group action-dim index sets for per-group TRAINING METRICS ONLY (velocity-space
# masked MSE, logged not optimized). Mirrors the fixed 32-dim slot layout written by
# flore.s1_policy.data.umi_dataset (kept in sync by test_metric_groups_match_dataset_layout):
#   [0:3] L pos   [3:9] L rot6d   [9]  L grip
#   [10:13] R pos [13:19] R rot6d [19] R grip
#   [20:23] H pos [23:29] H rot6d [29:32] zero pad
# Groups overlap by design: keypoint groups (left/right/head) partition the real dims,
# and quantity groups (pos/rot/grip) re-partition the SAME dims a different way. The pad
# dims [29:32] belong to no group. Absent dims (e.g. a missing keypoint) are still zeroed
# out per-sample by action_dim_mask, so a group with no present dims logs 0.0.
_METRIC_GROUP_DIMS: dict[str, tuple[int, ...]] = {
    "left": tuple(range(0, 10)),
    "right": tuple(range(10, 20)),
    "head": tuple(range(20, 29)),
    "pos": (0, 1, 2, 10, 11, 12, 20, 21, 22),
    "rot": (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28),
    "grip": (9, 19),
}


class DPFlowModel(FlowMatchingVLABase):
    """DP-Flow VLA: ViT encoder + configurable Unet/DiT decoder.

    Uses flat global conditioning (no cross-attention to VLM features).

    Parameters
    ----------
    config : DPFlowConfig
    encoder : nn.Module, optional
        Pluggable image encoder. When provided it REPLACES the timm backbone
        (used in tests to avoid downloads). It must expose an int ``num_features``
        attribute and map a ``(N, C, H, W)`` image batch to either ``(N, num_features)``
        or ``(N, tokens, num_features)`` (CLS token at index 0 is used).
    """

    def __init__(self, config: DPFlowConfig, encoder: nn.Module | None = None) -> None:
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

        # With an observation history of To frames, each of the num_cameras cameras
        # contributes To CLS features (To "pseudo-cameras"), and the state cond carries
        # To frames — both scale global_cond by To. To == 1 is the single-frame default.
        obs_hist = max(1, getattr(config, "obs_history_len", 1))
        # collate_umi_batch flattens cameras and history frames into ONE images list, so
        # num_cameras and To are indistinguishable there — and the dataset yields a single
        # camera, so collate emits exactly To entries. Guard the ambiguous combination:
        # num_cameras > 1 with history would mis-size enc_out_dim vs what collate provides.
        if config.num_cameras > 1 and obs_hist > 1:
            raise NotImplementedError(
                f"num_cameras={config.num_cameras} with obs_history_len={obs_hist} is not "
                f"supported: collate_umi_batch flattens cameras and history into one images "
                f"list, so global_cond would be mis-sized. Use a single camera with history, "
                f"or add a (camera, frame) two-axis obs layout first.")
        enc_out_dim = config.num_cameras * obs_hist * self.encoder.num_features
        # State contributes either filled_state_dim (zero-pad) or a learned
        # projection width per frame; _build_state_encoder sets self.state_proj +
        # returns the PER-FRAME width, scaled by To here.
        state_cond_width = self._build_state_encoder() * obs_hist

        # Force-history encoder (HL-v2). Appended LAST to global_cond so every
        # pre-existing conditioning column keeps its index — a checkpoint trained
        # without force can be widened by appending columns rather than permuting.
        self.force_encoder: nn.Module | None = None
        force_width = 0
        if getattr(config, "force_hist_channels", 0) > 0:
            from flore.s1_policy.model.force_encoder import ForceHistoryEncoder

            self.force_encoder = ForceHistoryEncoder(
                hist_len=config.force_hist_len,
                channels=config.force_hist_channels,
                width=config.force_enc_width,
                out_dim=config.force_enc_out_dim,
            ).to(dtype)
            force_width = config.force_enc_out_dim

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

        # UMI-parity train-time augmentation (None unless config.image_aug). Holds no
        # parameters, so it changes neither model.safetensors nor DDP's reduction.
        from flore.s1_policy.model.umi_parity import build_image_augment

        self.image_aug = build_image_augment(config)

        self.to(dtype)
        self._init_rtc_state()

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        logger.info(
            "DPFlowModel: vit=%s, decoder=%s, enc_params=%d, dec_params=%d, total=%d, "
            "image_aug=%s, encoder_lr_scale=%.3g",
            config.vit,
            config.diffusion_net,
            enc_params,
            dec_params,
            enc_params + dec_params,
            self.image_aug is not None,
            config.encoder_lr_scale,
        )

    def param_groups(self, lr: float) -> list[dict] | object:
        """AdamW parameter groups implementing UMI's reduced encoder LR.

        At the default ``encoder_lr_scale=1.0`` this returns ``self.parameters()``, i.e.
        exactly the single-group AdamW the trainer built before this hook existed.
        """
        from flore.s1_policy.model.umi_parity import encoder_lr_param_groups

        return encoder_lr_param_groups(self, lr, float(self.config.encoder_lr_scale))

    def _metric_group_defs(self) -> dict[str, tuple[int, ...]]:
        """Group name -> action dims, for the per-group logging metrics.

        Resolved from :mod:`flore.s1_policy.data.layout` when ``config.layout_mode`` is
        set (the HL-v2 path, where the layout is the single source of truth and knows
        which groups even exist in this mode). Falls back to the frozen
        :data:`_METRIC_GROUP_DIMS` table otherwise, so the pre-v2 32-dim checkpoint's
        metrics are unchanged.
        """
        mode = getattr(self.config, "layout_mode", "") or ""
        if not mode:
            return _METRIC_GROUP_DIMS
        from flore.s1_policy.data.layout import get_layout

        return get_layout(mode).metric_group_dims()

    # MUST stay on THIS method, not on a helper: a base-class decorator does not
    # propagate to a subclass override, so without it every logged step builds an
    # autograd graph over the (B, T, D) squared-error intermediate that the final
    # .tolist() immediately discards. Verified by the requires_grad of that intermediate
    # with and without the decorator; the wasted peak memory scales with batch size.
    @torch.no_grad()
    def _extra_train_metrics(self, v_t, u_t, action_is_pad, prefix_mask, action_dim_mask,
                             action_mask=None):
        """Per-group velocity-space masked MSE, for logging only (no gradient).

        Splits the single ``flow_loss`` into ``flow_loss/<group>`` sub-losses over the
        keypoint blocks (left/right/head) and the physical-quantity blocks
        (pos/rot/grip). Uses the SAME time-include mask (temporal-pad AND RTC-prefix)
        AND the SAME per-dim ``action_dim_mask`` as :meth:`_compute_flow_loss`, so each
        sub-loss is directly comparable to ``flow_loss`` (it is the same numerator/
        denominator ratio restricted to that group's dims). Groups with zero included
        weight log 0.0.

        Vectorized to a single GPU->CPU sync (one ``.tolist()``) so per-step logging
        cost stays ~flat regardless of group count.
        """
        # Squared velocity error, shared across groups: (B, T, D).
        sq_err = (v_t - u_t) ** 2

        # Time-axis include mask (B, T), same convention as _compute_flow_loss.
        include = torch.ones(v_t.shape[:2], dtype=torch.bool, device=v_t.device)
        if action_is_pad is not None:
            include = include & ~action_is_pad
        if prefix_mask is not None:
            include = include & ~prefix_mask
        time_w = include.unsqueeze(-1).float()  # (B, T, 1)

        # Per-dim weight (B, T, D): time include mask AND action_dim_mask (if any).
        if action_dim_mask is not None:
            dim_w = action_dim_mask.unsqueeze(1).float()  # (B, 1, D)
            weight = time_w * dim_w
        else:
            weight = time_w.expand_as(sq_err)
        # Same composition order as _compute_flow_loss, so each group's sub-loss stays
        # directly comparable to the optimized flow_loss.
        if action_mask is not None:
            weight = weight * action_mask.to(weight.dtype)

        # Lazily cache a (G, D) boolean group-selector matrix on the right device.
        group_sel = getattr(self, "_metric_group_sel", None)
        if group_sel is None or group_sel.device != v_t.device or group_sel.shape[1] != sq_err.shape[-1]:
            action_dim = sq_err.shape[-1]
            groups = self._metric_group_defs()
            names = list(groups.keys())
            sel = torch.zeros(len(names), action_dim, dtype=torch.float32, device=v_t.device)
            for gi, name in enumerate(names):
                idx = [d for d in groups[name] if d < action_dim]
                if idx:
                    sel[gi, idx] = 1.0
            self._metric_group_names = names
            self._metric_group_sel = sel
            group_sel = sel

        # Per-group numerator/denominator via one matmul over the dim axis.
        # weighted_sq/weight are (B, T, D); collapse (B, T) then project onto groups.
        per_dim_num = (sq_err * weight).sum(dim=(0, 1))  # (D,)
        per_dim_den = weight.sum(dim=(0, 1))  # (D,)
        num = group_sel @ per_dim_num  # (G,)
        den = (group_sel @ per_dim_den).clamp(min=1.0)  # (G,)
        vals = (num / den).tolist()  # single sync
        return {f"flow_loss/{name}": v for name, v in zip(self._metric_group_names, vals)}

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
            logger.warning("DPFlowModel.load_checkpoint: missing keys: %s", missing)
        if unexpected:
            logger.warning("DPFlowModel.load_checkpoint: unexpected keys: %s", unexpected)
        logger.info("DPFlowModel: loaded checkpoint from %s", model_path)

    def _encode_vision(self, images: list[Tensor], img_masks: list[Tensor] | None) -> Tensor:
        """Encode a list of per-camera image batches into flat vision features.

        Masked-out images (img_masks[c][b] == False) contribute a zero feature so a
        missing camera never injects garbage. Returns (B, num_cameras * num_features).
        """
        imgs = torch.stack(images, dim=1).to(self.dtype)  # (B, num_cameras, C, H, W)
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
            # Augment only the frames that reach the encoder (no-op in eval mode). Applied
            # to the FLAT (N, C, H, W) batch, where UMI's TimmObsEncoder.forward applies
            # its own transform stack.
            if self.image_aug is not None:
                valid_images = self.image_aug(valid_images)
            valid_features = self.encoder(valid_images)
            if valid_features.ndim == 3:
                valid_features = valid_features[:, 0, :]  # CLS token
            flat_features = torch.zeros(
                batch_size * num_imgs,
                valid_features.shape[-1],
                device=imgs.device,
                dtype=self.dtype,
            )
            flat_features[flat_masks] = valid_features.to(self.dtype)
        else:
            flat_features = torch.zeros(
                batch_size * num_imgs,
                self.encoder.num_features,
                device=imgs.device,
                dtype=self.dtype,
            )

        return flat_features.reshape(batch_size, -1)

    def _encode_global_cond(self, batch_dict: dict) -> Tensor:
        """Encode images + state (+ force history) into a flat conditioning vector."""
        images = batch_dict["images"]
        img_masks = batch_dict.get("img_masks")
        state = batch_dict.get("state")

        vision_features = self._encode_vision(images, img_masks)
        batch_size = vision_features.shape[0]
        device = vision_features.device

        # Zero any observation entry the dataset flagged as fabricated (see HLDataset:
        # robot0_gripper_width has no mask of its own and is zero-filled when the rig
        # calibration is absent, and 0.0 is a LEGAL width — so an ungated entry feeds the
        # model a made-up measurement). Zeroing is the conservative default: it matches
        # the zero-pad the state slot already uses for absent dims.
        if state is not None:
            sv = batch_dict.get("state_valid")
            if sv is not None:
                state = state * sv.to(state.device, state.dtype)

        filled_state = self._project_state(state, batch_size, device)

        parts = [vision_features, filled_state]

        # Force history goes LAST, matching the width accounting in __init__ (and
        # keeping every older conditioning column at its original index).
        if self.force_encoder is not None:
            fh = batch_dict.get("force_history")
            if fh is None:
                raise KeyError(
                    "this model was built with a force encoder "
                    f"(force_hist_channels={self.config.force_hist_channels}) but the "
                    "batch has no 'force_history'. A silently zero-filled window would "
                    "read as 'no contact' — pass the real window or build the model "
                    "without force.")
            # Move BOTH or neither: the explicit .to(device) on the window documents
            # intent to accept a host-side batch (serving / eval / a notebook), and
            # leaving its companion behind fails exactly there — the trainer's
            # _move_batch already moves everything, so the bug is invisible in training.
            fhv = batch_dict.get("force_history_valid")
            parts.append(self.force_encoder(
                fh.to(device),
                fhv.to(device) if fhv is not None else None).to(self.dtype))

        return torch.cat(parts, dim=1)
