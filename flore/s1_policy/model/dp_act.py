"""DP-ACT: the Action Chunking Transformer as a third equal-level S1 preset.

The sibling of :class:`flore.s1_policy.model.dp_flow.DPFlowModel` and
:class:`flore.s1_policy.model.dp_diffusion.DPDiffusionModel`. Same inputs (ONE 224x224
camera frame at t0, the per-keypoint ``wrt_start_rot6d`` + gripper state, and the 1-bin
causal force window), same 16-dim-per-hand HL-v2 action output
``[pos3 rot6d6 grip1 force6]`` over 50 steps at 30 Hz, same masking contract, same
checkpoint/serving/inference plumbing. Selected by ``TrainConfig.algo='act'``.

REIMPLEMENTED against FLoRe's dataset/serving seams from ``third_party/act`` (Zhao et al.,
*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023), whose DETR
transformer and ResNet trunk are ported near-verbatim into
:mod:`flore.s1_policy.model.act`. Nothing is imported from ``third_party/act`` at runtime:
that tree is an unmodified upstream checkout (its own git repo, `IPython.embed` debug
hooks, ALOHA/mujoco imports, an `argparse` parser evaluated at model-build time), so
depending on it would drag a sim stack and a CLI parser into a serving process.

What ACT is, and why it is worth having next to the other two
------------------------------------------------------------
Flow and Diffusion are both GENERATIVE over the action chunk: they start from noise and
iterate (10 Euler steps / 16 DDIM steps). ACT is a **conditional VAE with a single
forward pass**::

    train:  q(z | state, actions) --KL--> N(0, I)          [the "style" encoder]
            decoder(image, state, force, z) --L1--> actions
    infer:  z := 0  (the prior mean), ONE forward pass -> the whole 50-step chunk

Two consequences that matter operationally:

* **Inference is deterministic and single-shot.** There is no sampler loop, so
  ``num_steps`` is structurally 1 and the seed changes nothing. This is not an
  optimization — it is what the architecture is. The refusals in
  :meth:`sample_actions` say so rather than accepting a value that would be ignored.
  Cheap enough (4.35 ms/chunk measured) that serving re-queries EVERY step and blends the
  overlapping predictions — upstream's ``--temporal_agg``, see
  :mod:`flore.s1_policy.model.act.temporal_ensemble` and ``config.temporal_ensemble``.
* **It cross-attends to a spatial feature GRID, not a pooled vector.** The ResNet trunk
  emits 7x7 = 49 tokens per frame at 224x224, and each of the 50 action queries attends
  over all of them. The DiT/UNet siblings condition on a single CLS vector per frame.
  That is the substantive architectural difference between this preset and those.

How FLoRe's observation reaches an architecture built for ALOHA
-------------------------------------------------------------
Upstream's decoder gets exactly two non-visual tokens: the latent ``z`` and a
proprioception token (14 joint angles). FLoRe has a third modality ALOHA has no sensor
for — the F/T wrench — so :class:`~flore.s1_policy.model.act.ACTTransformer` takes a LIST
of conditioning tokens and a matching learned position embedding. With force absent the
sequence is byte-identical to upstream; with force present ONE token is appended and the
visual token positions are untouched::

    [ z | state | force? ] ++ [ 49 visual tokens per (camera x history frame) ]
      |__ prepended, own learned positions __|   |__ 2D sine positions __|

The force token reuses the SAME :class:`ForceHistoryEncoder` the other two presets use
(a causal Conv1d stack with a learned "no data" embedding for an unmeasured window),
then a ``Linear`` to ``hidden_dim``. Sharing that leaf is deliberate: an invalid force
window must mean the same thing to all three models, and a second copy would drift.

Three deliberate deviations from upstream
-----------------------------------------
1. **The L1 loss divides by the VALID element count, not by ``numel()``.** Upstream is
   ``(all_l1 * ~is_pad.unsqueeze(-1)).mean()`` (policy.py:31) — the padded entries
   contribute 0 to the numerator but still count in the denominator. That makes the
   reconstruction term's magnitude depend on how much of the chunk is padding, which on
   this dataset varies per sample (only 83.7% of anchors have a full 50-step horizon;
   the mean is 46.1/50 valid steps), so it would silently reweight L1 against the KL
   term sample by sample. FLoRe's :func:`absolute_error_sums` divides by the true valid
   count, which is also what makes the logged number comparable across samples.
2. **``is_pad_head`` is not ported.** Upstream builds it and returns ``is_pad_hat``
   (detr_vae.py:53,138) but never puts it in any loss (policy.py:30-34) — it is dead
   weight. FLoRe knows each step's validity exactly, from the converter's own
   ``action_valid``; predicting it would be predicting an input.
3. **Masked action entries are ZEROED before the CVAE encoder sees them**, matching what
   both siblings already do before building their interpolant / noising. The style
   encoder attends across the whole chunk, so a stale-held value at a masked step would
   leak into ``z`` — and at inference ``z`` is 0, so the model would be conditioned at
   train time on information that does not exist at deploy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from flore.s1_policy.model.loss_metrics import absolute_error_sums

logger = logging.getLogger(__name__)

#: Which stacked decoder-layer activation the action head reads. **0 = the SHALLOWEST layer**,
#: which is upstream's ``hs[0]`` (``detr_vae.py:131``) and therefore load-bearing for
#: weight-for-weight parity — see :meth:`DPACTModel._decode`. It doubles as the truncation
#: depth (``decoder_layers = this + 1``), so the "read layer i" and "run i+1 layers" facts can
#: never drift apart. Pinned by ``tests/test_act_model.py``.
_READOUT_LAYER_INDEX = 0


def _sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """``(1, n_position, d_hid)`` fixed sinusoid table.

    Ported from ``detr_vae.py::get_sinusoid_encoding_table``, vectorized (upstream builds
    it with a Python list comprehension per position). Verified equal to upstream to
    float32 precision by ``tests/test_act_model.py``.
    """
    position = np.arange(n_position)[:, None]
    hid_j = np.arange(d_hid)[None, :]
    table = position / np.power(10000, 2 * (hid_j // 2) / d_hid)
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


@dataclass
class DPACTConfig:
    """Configuration for DP-ACT (ResNet trunk + DETR transformer + CVAE).

    Field names shared with :class:`DPFlowConfig` / :class:`DPDiffusionConfig` are kept
    IDENTICAL so ``TrainConfig``'s layout-derived pushes (``action_dim``, ``state_dim``,
    ``action_horizon``, ``obs_history_len``, ``force_hist_*``, ``layout_mode``) apply
    unchanged to any of the three, and so
    :func:`flore.s1_policy.inference.common.validate_model_layout` can check all three
    with one code path.
    """

    # ---- vision trunk (ACT's, not a ViT) ----
    #: ResNet variant; see :data:`flore.s1_policy.model.act.RESNET_SPECS`. ACT's default
    #: and only tested trunk is resnet18.
    backbone: str = "resnet18"
    #: Load ImageNet weights. ACT relies on a pretrained trunk (its FrozenBatchNorm keeps
    #: the ImageNet statistics), so this defaults ON exactly as upstream. Every checkpoint
    #: loader forces it False so a local restore never hits the network.
    pretrained: bool = True
    #: Replace the last stride with dilation (DETR's DC5). ACT never enables it; kept so
    #: the ported backbone's signature is complete and a 14x14 grid is reachable.
    dilation: bool = False
    #: Re-normalize the image to IMAGENET statistics before the ResNet, undoing the
    #: dataset's ``(x/255 - mean)/std``. **Load-bearing, default ON** — the trunk is
    #: ImageNet-pretrained with FROZEN BatchNorm, so its baked-in statistics only transfer
    #: if the input distribution matches, and FLoRe's shared [-1,1] dataset convention is
    #: measurably out of distribution for it (input std 2.23x too small; layer4 features at
    #: cosine 0.64 / relative L2 0.79 versus the correct path). Upstream ACT applies the
    #: ImageNet transform right before its backbone (``policy.py:20-22``); this reproduces
    #: it. See ResNetBackbone's docstring for why it lives here and not in the dataset.
    imagenet_renormalize: bool = True
    #: The dataset's own image affine, needed to INVERT it before applying ImageNet's. Must
    #: match ``hl_data.image_mean``/``image_std``; ``TrainConfig`` pushes them so they
    #: cannot drift, and they are baked into the checkpoint.
    dataset_image_mean: float = 0.5
    dataset_image_std: float = 0.5

    # ---- transformer (ACT's published hyperparameters) ----
    #: ACT's ``hidden_dim``. README trains with 512.
    hidden_dim: int = 512
    #: ACT's ``dim_feedforward``. README trains with 3200.
    dim_feedforward: int = 3200
    nheads: int = 8
    #: Decoder layers. ACT hardcodes 7 (imitate_episodes.py:55).
    dec_layers: int = 7
    #: Encoder layers. ACT hardcodes 4, and SHARES the count with the CVAE style encoder
    #: (detr_vae.py:217 notes the sharing) — see :attr:`vae_enc_layers`.
    enc_layers: int = 4
    dropout: float = 0.1
    pre_norm: bool = False
    #: Layers in the CVAE STYLE encoder (q(z|state, actions)). Upstream reuses
    #: ``enc_layers`` for this with a ``# TODO shared with VAE decoder`` comment; split
    #: out here so the two can be tuned independently. Leave equal to ``enc_layers`` for
    #: upstream parity.
    vae_enc_layers: int = 4

    # ---- CVAE ----
    #: Latent width. Upstream: 32, with a ``# TODO tune``.
    latent_dim: int = 32
    #: Refuse to train without a ``norm_stats`` path. Default ON: the L1 objective is taken
    #: over the raw action vector, which mixes metres / unitless rot6d / a [0,1] gripper
    #: fraction / N and N·m. Unnormalized, the measured block shares are pos 7.6%, rot
    #: 22.6%, grip 21.6%, force 43.3%, **torque 4.9%** — the torque columns end up
    #: effectively unsupervised, and nothing errors (the loss looks healthy). Enforced in
    #: ``TrainConfig._post_init_act``; the diffusion sibling has the analogous
    #: ``clip_sample`` refusal for the same class of silent trap.
    require_norm_stats: bool = True
    #: Weight on the KL term: ``loss = L1 + kl_weight * KL``. ACT's README uses 10.
    #: NOTE this interacts with deviation (1) in the module docstring — FLoRe's L1 is a
    #: true masked mean, so it is ~8% larger than upstream's numel-divided version on
    #: this dataset, making the effective KL pressure slightly LOWER at the same weight.
    kl_weight: float = 10.0
    #: Sample ``z ~ N(mu, sigma)`` from the PRIOR at inference instead of taking the prior
    #: MEAN (``z = 0``). Upstream is ``z = 0`` (detr_vae.py:113) — deterministic, which is
    #: what makes an ACT rollout reproducible without a seed. Turning this on makes
    #: ``sample_actions`` stochastic and the ``generator`` argument meaningful.
    sample_latent_at_inference: bool = False

    # ---- temporal ensembling (upstream's --temporal_agg) ----
    #: Re-query EVERY step and exponentially blend the overlapping chunk predictions
    #: (``imitate_episodes.py:216-259``). **Default ON**: ACT decodes from the prior mean
    #: with a deterministic decoder, so on an anchor where the demonstrations went two ways
    #: it must pick one mode or average them — and the future wipe direction in this dataset
    #: IS bimodal in the anchor frame (histogram trough/peak 0.054). Averaging a bimodal
    #: wipe stalls, which is the "jerky / pauses mid-episode" symptom ACT's tuning note
    #: names. Measured cost: 4.35 ms per chunk on this GPU against a 33.3 ms budget at
    #: 30 Hz — 7.7x headroom, so per-step re-query is affordable. BAKED INTO THE CHECKPOINT
    #: so serving inherits the intent rather than needing a matching CLI flag.
    temporal_ensemble: bool = True
    #: Exponential decay rate for the blend. Upstream's ``k = 0.01``
    #: (``imitate_episodes.py:255``). With ``k > 0`` the OLDEST surviving prediction gets the
    #: MOST weight — upstream's ordering, and deliberate: an older chunk saw a more distant
    #: observation but has been corroborated by every step since, so weighting it up is what
    #: smooths the executed path instead of chasing the newest, noisiest chunk. ``k = 0`` is
    #: a plain mean over the survivors.
    temporal_ensemble_k: float = 0.01
    #: Also feed the force window to the CVAE STYLE encoder. Default OFF, mirroring
    #: upstream, whose style encoder deliberately sees only proprioception + the action
    #: sequence (never the image): force is an OBSERVATION, and observations belong on the
    #: decoder side. Turning this on lets the latent explain force-driven variation
    #: directly, at the cost of a train/deploy asymmetry that ``z = 0`` must absorb.
    encoder_sees_force: bool = False

    # ---- action space (pushed from the layout by TrainConfig for the HL pipeline) ----
    action_dim: int = 16
    #: ACT's ``num_queries`` / ``chunk_size``: one learned query per predicted step.
    action_horizon: int = 50
    state_dim: int = 7
    num_cameras: int = 1
    #: Observation history: To frames, oldest-first, ending at t0. Folded into the visual
    #: token axis (each frame is one more "camera"), exactly as upstream folds cameras.
    obs_history_len: int = 1

    # ---- inference ----
    #: STRUCTURALLY 1: a CVAE decodes the whole chunk in one forward pass. Present so the
    #: field exists on all three configs (serving/inference read it generically); any
    #: other value is refused rather than silently ignored.
    inference_steps: int = 1

    # ---- optimization (ACT's two-LR param-group split) ----
    #: Learning rate for the VISION TRUNK only. ACT trains the backbone at its own,
    #: usually lower, rate (main.py:80-86 builds two AdamW param groups keyed on whether
    #: the parameter name contains "backbone"); upstream hardcodes 1e-5
    #: (imitate_episodes.py:51). Consumed via :meth:`param_groups`; the trainer's global
    #: ``--lr`` applies to everything else.
    lr_backbone: float = 1e-5

    # ---- precision ----
    precision: str = "float32"

    # ---- state conditioning ----
    #: ACT ALWAYS projects state through its own ``nn.Linear`` into ``hidden_dim`` — there
    #: is no zero-padded state slot to size, because state enters as a TOKEN, not as a
    #: slice of a flat conditioning vector. Frozen True so
    #: ``validate_model_layout``'s ``filled_state_dim`` truncation check (which only
    #: applies to the zero-pad path) is correctly skipped.
    state_projection: bool = True

    # ---- force-history conditioning (HL-v2) ----
    #: A ``(force_hist_len, force_hist_channels)`` causal wrench window, encoded by the
    #: SHARED :class:`ForceHistoryEncoder` and appended as ONE transformer token.
    #: ``force_hist_channels == 0`` -> no force encoder and no force token, leaving the
    #: token sequence byte-identical to upstream ACT.
    force_hist_len: int = 0
    force_hist_channels: int = 0
    force_enc_width: int = 128
    force_enc_out_dim: int = 64

    #: HL-v2 layout mode, pushed in by TrainConfig and BAKED INTO THE CHECKPOINT so
    #: serving can slice the output unambiguously (a bare 16-dim vector is ambiguous
    #: between left_only and right_only).
    layout_mode: str = ""


class DPACTModel(nn.Module):
    """DP-ACT: ResNet trunk + DETR transformer + CVAE, on FLoRe's HL-v2 contract.

    Deliberately NOT a subclass of ``FlowMatchingVLABase`` / ``DiffusionVLABase``: those
    bases own an objective built around a noise/time interpolant and a per-step sampler
    loop, neither of which a single-pass CVAE has. It implements the same *call contract*
    instead — ``forward(batch) -> {"loss", "metrics", "metric_sums", "aux"}`` and
    ``sample_actions(batch, ...) -> (B, H, action_dim)`` — which is what the trainer,
    the offline inference sessions and the live planner actually depend on.

    Parameters
    ----------
    config : DPACTConfig
    encoder : nn.Module, optional
        Pluggable image encoder that REPLACES the ResNet trunk (used in tests to avoid
        downloads). Must map ``(N, C, H, W)`` to ``(features, pos)`` with
        ``features`` ``(N, num_channels, H', W')`` and ``pos`` broadcastable to
        ``(1, hidden_dim, H', W')``, and expose an int ``num_channels``.
    """

    def __init__(self, config: DPACTConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        dtype = self._parse_dtype(config.precision)

        from flore.s1_policy.model.act import ACTTransformer, ResNetBackbone, TransformerEncoder
        from flore.s1_policy.model.act.detr_transformer import TransformerEncoderLayer

        obs_hist = max(1, int(config.obs_history_len))
        self._n_visual_groups = int(config.num_cameras) * obs_hist
        H = int(config.hidden_dim)

        # ---- vision trunk -> hidden_dim ----
        if encoder is not None:
            self.backbone = encoder
        else:
            self.backbone = ResNetBackbone(
                config.backbone, H, pretrained=config.pretrained, dilation=config.dilation,
                renormalize=config.imagenet_renormalize,
                dataset_mean=config.dataset_image_mean, dataset_std=config.dataset_image_std)
        if not isinstance(getattr(self.backbone, "num_channels", None), int):
            raise ValueError("ACT backbone must expose an int `num_channels` attribute")
        self.input_proj = nn.Conv2d(self.backbone.num_channels, H, kernel_size=1)

        # ---- decoder-side conditioning tokens ----
        # State: the To frames are FLATTENED into one token (mirroring how the DiT/UNet
        # siblings concatenate To state frames into global_cond), so raising To adds
        # information without adding a token whose learned position nothing was trained on.
        self._state_in_dim = obs_hist * int(config.state_dim)
        self.input_proj_robot_state = nn.Linear(self._state_in_dim, H)

        self.force_encoder: nn.Module | None = None
        self.force_token_proj: nn.Module | None = None
        if int(config.force_hist_channels) > 0:
            from flore.s1_policy.model.force_encoder import ForceHistoryEncoder

            self.force_encoder = ForceHistoryEncoder(
                hist_len=config.force_hist_len,
                channels=config.force_hist_channels,
                width=config.force_enc_width,
                out_dim=config.force_enc_out_dim,
            )
            self.force_token_proj = nn.Linear(int(config.force_enc_out_dim), H)

        # z, state, and (when present) force. ORDER IS THE CHECKPOINT CONTRACT: the
        # learned `additional_pos_embed` rows are positional, so inserting a token
        # anywhere but the end would re-map every later token's learned position.
        self._n_extra_tokens = 2 + (1 if self.force_encoder is not None else 0)
        self.additional_pos_embed = nn.Embedding(self._n_extra_tokens, H)

        self.transformer = ACTTransformer(
            d_model=H,
            nhead=config.nheads,
            num_encoder_layers=config.enc_layers,
            num_decoder_layers=config.dec_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            normalize_before=config.pre_norm,
            return_intermediate_dec=True,
        )
        self.query_embed = nn.Embedding(int(config.action_horizon), H)
        self.action_head = nn.Linear(H, int(config.action_dim))

        # ---- CVAE style encoder: q(z | state, actions[, force]) ----
        vae_layer = TransformerEncoderLayer(H, config.nheads, config.dim_feedforward,
                                            config.dropout, "relu", config.pre_norm)
        vae_norm = nn.LayerNorm(H) if config.pre_norm else None
        self.vae_encoder = TransformerEncoder(vae_layer, int(config.vae_enc_layers), vae_norm)
        self.cls_embed = nn.Embedding(1, H)
        self.encoder_action_proj = nn.Linear(int(config.action_dim), H)
        self.encoder_state_proj = nn.Linear(self._state_in_dim, H)
        self.encoder_force_proj: nn.Module | None = None
        # [CLS] + state (+ force?) + the action sequence.
        n_vae_prefix = 2 + (1 if (config.encoder_sees_force and self.force_encoder is not None) else 0)
        if config.encoder_sees_force and self.force_encoder is not None:
            self.encoder_force_proj = nn.Linear(int(config.force_enc_out_dim), H)
        self._n_vae_prefix = n_vae_prefix
        self.latent_dim = int(config.latent_dim)
        self.latent_proj = nn.Linear(H, self.latent_dim * 2)
        self.latent_out_proj = nn.Linear(self.latent_dim, H)
        self.register_buffer(
            "pos_table", _sinusoid_encoding_table(n_vae_prefix + int(config.action_horizon), H),
            persistent=False)

        self.to(dtype)
        self._init_rtc_state()

        bb_params = sum(p.numel() for p in self.backbone.parameters())
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "DPACTModel: backbone=%s hidden=%d ff=%d enc/dec=%d/%d queries=%d latent=%d "
            "kl_weight=%.3g force_token=%s backbone_params=%d total=%d",
            config.backbone, H, config.dim_feedforward, config.enc_layers, config.dec_layers,
            config.action_horizon, self.latent_dim, config.kl_weight,
            self.force_encoder is not None, bb_params, total,
        )

    # ------------------------------------------------------------------
    # dtype / small plumbing shared in spirit with the sibling bases
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dtype(dtype_str) -> torch.dtype:
        if isinstance(dtype_str, torch.dtype):
            if not dtype_str.is_floating_point:
                raise ValueError(f"unsupported floating-point precision {dtype_str}")
            return dtype_str
        dtype_str = dtype_str.replace("torch.", "").replace("bf16", "bfloat16").replace("fp32", "float32")
        dtype = getattr(torch, dtype_str, None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"unsupported floating-point precision {dtype_str!r}")
        return dtype

    @property
    def dtype(self) -> torch.dtype:
        return self.action_head.weight.dtype

    def _init_rtc_state(self) -> None:
        """Cache-shaped state so serving's ``reset()`` behaves like the siblings'.

        ACT has no Real-Time Chunking (there is no iterative sampler whose prefix could be
        frozen), but the planner calls ``reset()`` at every episode boundary and reads the
        cache in its diagnostics, so the attribute must exist and mean the same thing.
        """
        self.register_buffer(
            "_last_actions",
            torch.zeros(1, int(self.config.action_horizon), int(self.config.action_dim)),
            persistent=False,
        )
        self._last_actions_valid: bool = False

    def reset(self) -> None:
        """Drop the cached chunk at an episode boundary (mirrors both siblings)."""
        self._last_actions_valid = False

    def param_groups(self, lr: float) -> list[dict]:
        """AdamW parameter groups implementing ACT's two-LR split.

        Upstream builds exactly this (main.py:80-86): every parameter whose NAME contains
        ``"backbone"`` trains at ``lr_backbone``, everything else at ``lr``. The trunk is
        ImageNet-pretrained while the transformer is from scratch, so one shared LR either
        destroys the pretrained features or starves the transformer.

        The trainer calls this when a model defines it; the Flow/Diffusion siblings do
        not, so their single-group AdamW is untouched.
        """
        head, trunk = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (trunk if "backbone" in name else head).append(p)
        groups = [{"params": head, "lr": float(lr)}]
        if trunk:
            groups.append({"params": trunk, "lr": float(self.config.lr_backbone)})
        return groups

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
            logger.warning("DPACTModel.load_checkpoint: missing keys: %s", missing)
        if unexpected:
            logger.warning("DPACTModel.load_checkpoint: unexpected keys: %s", unexpected)
        logger.info("DPACTModel: loaded checkpoint from %s", model_path)

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _flat_state(self, batch_dict: dict, batch_size: int, device: torch.device) -> Tensor:
        """``(B, To * state_dim)``, with dataset-flagged-fabricated entries zeroed.

        Mirrors both siblings: ``state_valid`` False means the dataset FABRICATED that
        entry (e.g. a zero gripper width with no rig calibration), and 0.0 is a LEGAL
        width — so an ungated entry feeds the model a made-up measurement.
        """
        state = batch_dict.get("state")
        if state is None:
            return torch.zeros(batch_size, self._state_in_dim, device=device, dtype=self.dtype)
        state = state.to(device=device, dtype=self.dtype)
        sv = batch_dict.get("state_valid")
        if sv is not None:
            state = state * sv.to(device=device, dtype=state.dtype)
        flat = state.reshape(state.shape[0], -1)
        if flat.shape[1] != self._state_in_dim:
            raise ValueError(
                f"state flattens to width {flat.shape[1]} but this model was built for "
                f"{self._state_in_dim} (= obs_history_len * state_dim). A width mismatch "
                f"mis-routes every proprioception channel.")
        return flat

    def _force_feature(self, batch_dict: dict, device: torch.device) -> Tensor | None:
        """``(B, force_enc_out_dim)`` from the shared force encoder, or None."""
        if self.force_encoder is None:
            return None
        fh = batch_dict.get("force_history")
        if fh is None:
            raise KeyError(
                "this model was built with a force encoder "
                f"(force_hist_channels={self.config.force_hist_channels}) but the batch has "
                "no 'force_history'. A silently zero-filled window would read as 'no "
                "contact' — pass the real window or build the model without force.")
        fhv = batch_dict.get("force_history_valid")
        return self.force_encoder(
            fh.to(device), fhv.to(device) if fhv is not None else None).to(self.dtype)

    def _encode_vision(self, images: list[Tensor], img_masks: list[Tensor] | None
                       ) -> tuple[Tensor, Tensor]:
        """Per-camera/history feature grids, folded into the WIDTH axis.

        Returns ``(src, pos)``: ``(B, hidden_dim, H', W' * n_groups)`` and
        ``(1, hidden_dim, H', W' * n_groups)``. Folding into width is exactly upstream's
        ``torch.cat(all_cam_features, axis=3)`` (detr_vae.py:129) — the transformer sees a
        flat token set either way, and this keeps the 2D sine embedding meaningful within
        each group.

        A masked-out entry (``img_masks[i][b]`` False) contributes a ZERO feature map for
        that sample, matching the siblings' zero-feature convention, so a missing camera
        never injects garbage.
        """
        if not images:
            raise ValueError("ACT needs at least one camera frame")
        if len(images) != self._n_visual_groups:
            raise ValueError(
                f"got {len(images)} image entries but this model was built for "
                f"{self._n_visual_groups} (= num_cameras * obs_history_len); collate "
                f"flattens cameras and history into one list, so a mismatch mis-sizes "
                f"the visual token set.")
        feats, poss = [], []
        for index, img in enumerate(images):
            f, p = self.backbone(img.to(self.dtype))
            f = self.input_proj(f)
            if img_masks is not None:
                keep = img_masks[index].reshape(-1, 1, 1, 1).to(f.dtype)
                f = f * keep
            feats.append(f)
            poss.append(p.to(f.dtype))
        return torch.cat(feats, axis=3), torch.cat(poss, axis=3)

    # ------------------------------------------------------------------
    # CVAE style encoder
    # ------------------------------------------------------------------
    def _encode_latent(self, actions: Tensor, flat_state: Tensor, action_pad: Tensor | None,
                       force_feat: Tensor | None) -> tuple[Tensor, Tensor]:
        """``q(z | state, actions[, force])`` -> ``(mu, logvar)``, each ``(B, latent_dim)``.

        Ported from ``detr_vae.py:88-108``. ``action_pad`` is ``(B, T)`` True where the step
        carries no measurement; those keys are masked out of the attention. The CLS and
        state tokens are NEVER masked (upstream's ``cls_joint_is_pad``), which is also what
        keeps a fully-padded sample from producing NaN — with every key masked,
        ``MultiheadAttention`` softmaxes over nothing.
        """
        bs = actions.shape[0]
        action_embed = self.encoder_action_proj(actions)                    # (B, T, H)
        prefix = [self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1),      # (B, 1, H)
                  self.encoder_state_proj(flat_state).unsqueeze(1)]        # (B, 1, H)
        if self.encoder_force_proj is not None:
            if force_feat is None:
                raise RuntimeError("encoder_sees_force is set but no force feature was produced")
            prefix.append(self.encoder_force_proj(force_feat).unsqueeze(1))
        encoder_input = torch.cat([*prefix, action_embed], axis=1)          # (B, P+T, H)
        encoder_input = encoder_input.permute(1, 0, 2)                      # (P+T, B, H)

        key_padding_mask = None
        if action_pad is not None:
            unmasked = torch.zeros((bs, self._n_vae_prefix), dtype=torch.bool, device=actions.device)
            key_padding_mask = torch.cat([unmasked, action_pad], axis=1)    # (B, P+T)

        pos_embed = self.pos_table.clone().detach().permute(1, 0, 2).to(encoder_input.dtype)
        out = self.vae_encoder(encoder_input, pos=pos_embed,
                               src_key_padding_mask=key_padding_mask)
        latent_info = self.latent_proj(out[0])                              # CLS token only
        return latent_info[:, : self.latent_dim], latent_info[:, self.latent_dim :]

    @staticmethod
    def _kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        """Total KL to ``N(0, I)``: summed over latent dims, averaged over the batch.

        Ported from ``policy.py::kl_divergence`` (its ``total_kld``). Computed in float32
        regardless of autocast: ``logvar.exp()`` on a bf16 tensor loses precision exactly
        where the KL is small, which is where it does its work.
        """
        mu32, logvar32 = mu.float(), logvar.float()
        klds = -0.5 * (1 + logvar32 - mu32.pow(2) - logvar32.exp())
        return klds.sum(1).mean()

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_from_masks(action_mask: Tensor | None, action_is_pad: Tensor | None,
                        shape: tuple[int, int], device: torch.device) -> Tensor | None:
        """``(B, T)`` bool, True where a step carries NO measurement at all.

        Prefers the explicit ``action_is_pad``; otherwise derives it from the HL-v2
        per-(step, dim) ``action_mask`` as "no dim of this step is valid". A step that is
        valid for pose but masked for force is NOT padding — it still supervises the pose
        columns, and dropping it from the style encoder would discard a real measurement.
        """
        if action_is_pad is not None:
            return action_is_pad.to(device=device, dtype=torch.bool)
        if action_mask is not None:
            return ~(action_mask.to(device).bool().any(dim=-1))
        _ = shape
        return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, batch_dict: dict) -> dict:
        """Training forward pass: L1 reconstruction + KL, both masked.

        Returns ``{"loss", "metrics", "metric_sums", "aux"}`` — the same contract the two
        siblings return, so ``Trainer.train_step`` / ``evaluate`` need no special case.

        ``metric_sums`` are the **L1** (order-1) sums, so ``train/objective`` for an ACT
        run is the masked mean absolute action error in normalized units. It is therefore
        NOT comparable to a flow or diffusion ``objective`` (those are squared errors on a
        velocity / epsilon target); it IS comparable across ACT runs and against ACT's own
        validation, which is what best-checkpoint selection uses.
        """
        actions = batch_dict["actions"].to(self.dtype)
        device = actions.device
        B = actions.shape[0]
        # Upstream truncates defensively (`policy.py:24-25`: actions[:, :num_queries]); we
        # refuse instead, because a chunk of the wrong length means the data pipeline and
        # the model disagree about the action horizon and quietly training on a prefix would
        # hide that. Checked HERE so the message names action_horizon: without it a mismatch
        # surfaces as an opaque tensor-size error from inside the CVAE encoder's attention.
        if actions.shape[1] != int(self.config.action_horizon):
            raise ValueError(
                f"actions have {actions.shape[1]} steps but this model was built for "
                f"action_horizon={self.config.action_horizon}. The dataset and the model "
                f"disagree about the chunk length.")

        action_mask = batch_dict.get("action_mask")
        action_is_pad = batch_dict.get("action_is_pad")
        action_dim_mask = batch_dict.get("action_dim_mask")
        action_pad = self._pad_from_masks(action_mask, action_is_pad, actions.shape[:2], device)

        # Zero UNSUPERVISED entries before the style encoder sees them (deviation 3 in the
        # module docstring): the encoder attends across the whole chunk, so a stale-held
        # value at a masked step would leak into z — and z is 0 at deploy.
        if action_mask is not None:
            actions = actions * action_mask.to(device=device, dtype=actions.dtype)

        flat_state = self._flat_state(batch_dict, B, device)
        force_feat = self._force_feature(batch_dict, device)
        mu, logvar = self._encode_latent(actions, flat_state, action_pad, force_feat)
        # Reparameterize with the module's own RNG stream (torch.randn_like, not a
        # Variable-wrapped .normal_() as upstream does) so the trainer's seeded/isolated
        # RNG context governs it like every other stochastic op in this repo.
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)

        pred = self._decode(z.to(self.dtype), flat_state, force_feat, batch_dict)

        l1 = self._masked_l1(pred, actions, action_is_pad, action_dim_mask, action_mask)
        kl = self._kl_divergence(mu, logvar)
        loss = l1 + float(self.config.kl_weight) * kl.to(l1.dtype)

        metrics: dict[str, float] = {}
        if not batch_dict.get("_skip_scalar_metrics", False):
            metrics = {"act_loss": float(loss.item()), "l1": float(l1.item()), "kl": float(kl.item())}
            metrics.update(self._extra_train_metrics(
                pred, actions, action_is_pad, action_dim_mask, action_mask))
        group_defs = self._metric_group_defs()
        with torch.no_grad():
            metric_sums = absolute_error_sums(
                pred, actions,
                action_is_pad=action_is_pad,
                action_dim_mask=action_dim_mask,
                action_mask=action_mask,
                group_dims=group_defs,
            )
            # KL rides in metric_sums as a (sum, count) pair so the trainer's exact
            # sum/count aggregation reports it correctly over a whole validation pass
            # (a mean-of-batch-means would be wrong on a ragged final batch), and so
            # `val/act_loss/kl` exists next to the reconstruction groups.
            kl_sum = kl.detach().to(torch.float32) * B
            metric_sums["kl"] = (kl_sum, torch.as_tensor(float(B), device=kl_sum.device))

        return {
            "loss": loss,
            "metrics": metrics,
            "metric_sums": metric_sums,
            "aux": {"pred": pred, "mu": mu, "logvar": logvar, "z": z, "l1": l1, "kl": kl},
        }

    def _decode(self, z: Tensor, flat_state: Tensor, force_feat: Tensor | None,
                batch_dict: dict) -> Tensor:
        """The conditional decoder: ``(image, state, force?, z) -> (B, H, action_dim)``."""
        src, pos = self._encode_vision(batch_dict["images"], batch_dict.get("img_masks"))
        extra = [self.latent_out_proj(z), self.input_proj_robot_state(flat_state)]
        if self.force_token_proj is not None:
            if force_feat is None:
                raise RuntimeError("a force token was configured but no force feature was produced")
            extra.append(self.force_token_proj(force_feat).to(self.dtype))
        # Upstream takes hs[0] — the FIRST of the stacked decoder-layer activations
        # (detr_vae.py:131). With return_intermediate_dec=True and a final LayerNorm,
        # TransformerDecoder replaces the last entry with the normed output, so index 0 is
        # layer 1's normed activation. Kept as-is for weight-for-weight parity with the
        # reference; taking hs[-1] (the deepest layer) would be a different model.
        #
        # BECAUSE the readout is index 0, INFERENCE only needs to RUN one decoder layer: the
        # stack is sequential, so entry 0 cannot depend on layers 2..N. Upstream runs all 7 and
        # discards six of them. Measured on the 100k-step checkpoint (batch 1, CUDA):
        # 4.84 -> 3.01 ms per forward, **1.61x**, output BIT-IDENTICAL (max |Δ| = 0.0). That is
        # 5.5% of a CPU core reclaimed at the 30 Hz temporal-ensembling re-query rate, where
        # the forward is CPU-bound on ~1300 kernel launches, not GPU-bound.
        #
        # ⚠ A GRAD-ENABLED FORWARD MUST STILL RUN THE WHOLE STACK, and not because the numbers
        # would differ (they would not — those layers get exactly zero gradient either way,
        # which test_the_action_head_reads_the_FIRST_decoder_layer pins). It is DDP: a
        # parameter that never enters the forward is UNUSED, and the trainer builds
        # DistributedDataParallel(find_unused_parameters=False) — the fast reducer, which
        # raises "Expected to have finished reduction in the prior iteration" on step 2.
        # Truncating there would trade a real multi-GPU speedup for a decoder that was already
        # free (the wasted layers cost GPU time, not the CPU time this fixes); reclaiming it
        # too means find_unused_parameters=True and measuring what that costs — separate work.
        #
        # The gate is `torch.is_grad_enabled()`, NOT `self.training`. `self.training` looks like
        # the natural flag and is WRONG: a model can be trained in eval() mode, and ACT
        # routinely is — `tests/test_ddp_training.py` builds it with `.eval()` on purpose, to
        # remove dropout as a source of cross-rank divergence, and then takes optimizer steps.
        # Gating on the mode truncated that run's decoder and reproduced the exact DDP failure
        # above. "Is anyone building a graph over these parameters" is the real question, and
        # grad-enabled is precisely that: under no_grad there is no graph, so nothing can be
        # unused. Every inference entry point (`sample_actions`, both offline sessions, the
        # live planner) is already under `torch.no_grad()`, and so is validation — which
        # therefore gets the same speedup with bit-identical metrics.
        #
        # `_READOUT_LAYER_INDEX` is the single place the readout depth and the truncation depth
        # are tied together, so one cannot drift from the other. All `dec_layers` layers'
        # WEIGHTS are always built and loaded — the checkpoint's state_dict is untouched.
        dec_layers = None if torch.is_grad_enabled() else _READOUT_LAYER_INDEX + 1
        hs = self.transformer(src, self.query_embed.weight.to(self.dtype), pos,
                              extra_tokens=[t.to(self.dtype) for t in extra],
                              additional_pos_embed=self.additional_pos_embed.weight.to(self.dtype),
                              decoder_layers=dec_layers)
        return self.action_head(hs[_READOUT_LAYER_INDEX])

    def _masked_l1(self, pred: Tensor, target: Tensor, action_is_pad: Tensor | None,
                   action_dim_mask: Tensor | None, action_mask: Tensor | None) -> Tensor:
        """Masked mean absolute error — the reconstruction term actually optimized.

        Divides by the VALID element count (deviation 1 in the module docstring), so its
        magnitude does not drift with a sample's padding fraction.
        """
        sums = absolute_error_sums(
            pred, target,
            action_is_pad=action_is_pad,
            action_dim_mask=action_dim_mask,
            action_mask=action_mask,
        )["overall"]
        return sums[0] / sums[1].clamp(min=1.0)

    def _metric_group_defs(self) -> dict[str, tuple[int, ...]]:
        """Group name -> action dims, resolved from the layout when ``layout_mode`` is set.

        No legacy fallback (ACT has no pre-v2 checkpoint to preserve): with no
        ``layout_mode``, per-group metrics are simply skipped.
        """
        mode = getattr(self.config, "layout_mode", "") or ""
        if not mode:
            return {}
        from flore.s1_policy.data.layout import get_layout

        return get_layout(mode).metric_group_dims()

    @torch.no_grad()
    def _extra_train_metrics(self, pred: Tensor, target: Tensor, action_is_pad: Tensor | None,
                             action_dim_mask: Tensor | None, action_mask: Tensor | None = None
                             ) -> dict[str, float]:
        """Per-group masked L1 (logging only), directly comparable to the ``l1`` term.

        Same mask composition as :meth:`_masked_l1`, so each ``act_loss/<group>`` is that
        same numerator/denominator ratio restricted to a group's dims. Vectorized to one
        GPU->CPU sync, mirroring both siblings.
        """
        groups = self._metric_group_defs()
        if not groups:
            return {}
        abs_err = (pred - target).abs()

        include = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        if action_is_pad is not None:
            include = include & ~action_is_pad
        weight = include.unsqueeze(-1).float()
        if action_dim_mask is not None:
            weight = weight * action_dim_mask.unsqueeze(1).float()
        else:
            weight = weight.expand_as(abs_err)
        if action_mask is not None:
            weight = weight * action_mask.to(weight.dtype)

        group_sel = getattr(self, "_metric_group_sel", None)
        if group_sel is None or group_sel.device != pred.device or group_sel.shape[1] != abs_err.shape[-1]:
            action_dim = abs_err.shape[-1]
            names = list(groups.keys())
            sel = torch.zeros(len(names), action_dim, dtype=torch.float32, device=pred.device)
            for gi, name in enumerate(names):
                idx = [d for d in groups[name] if d < action_dim]
                if idx:
                    sel[gi, idx] = 1.0
            self._metric_group_names = names
            self._metric_group_sel = sel
            group_sel = sel

        per_dim_num = (abs_err * weight).sum(dim=(0, 1))
        per_dim_den = weight.sum(dim=(0, 1))
        num = group_sel @ per_dim_num
        den = (group_sel @ per_dim_den).clamp(min=1.0)
        vals = (num / den).tolist()
        return {f"act_loss/{name}": v for name, v in zip(self._metric_group_names, vals)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        batch_dict: dict,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        policy_smooth_lam: float = 0.0,
        **rtc_kwargs,
    ) -> Tensor:
        """Decode the whole chunk in ONE forward pass with ``z`` from the prior.

        ``num_steps`` must be None or 1: a CVAE has no sampler loop, so any other value
        is a caller assuming an iterative sampler that does not exist. Refused rather than
        ignored — silently accepting ``num_steps=16`` would let an operator believe they
        had traded compute for quality when nothing changed.

        ``generator`` is only consulted when ``config.sample_latent_at_inference`` is set;
        the default ``z = 0`` path is bit-deterministic and a seed cannot affect it. Any
        non-trivial ``rtc_kwargs`` is refused for the same reason the diffusion sampler
        refuses it: there is no frozen-prefix inpainting here.

        ⚠ **DETERMINISTIC ONLY IN ``eval()`` MODE.** ACT's transformer carries
        ``dropout=0.1`` (upstream's default, active in both encoder and decoder), so in
        ``train()`` mode two identical calls differ by O(1) in normalized units even with
        ``z = 0``. The flow/diffusion siblings hide this behind a seeded ``generator``;
        here the seed is not in play, so training mode is the whole story. All three
        callers already handle it (``load_hl_model`` and both inference sessions call
        ``.eval()``; ``Trainer.run_sample_diagnostics`` does too, and restores the previous
        mode) — this method does NOT flip the mode itself, because a sampler that silently
        mutates the module's training state would corrupt a mid-epoch diagnostic.
        Pinned by ``tests/test_act_model.py``.
        """
        if num_steps is not None and int(num_steps) != 1:
            raise ValueError(
                f"ACT decodes the full chunk in one forward pass; num_steps must be 1 or "
                f"None, got {num_steps}. (There is no iterative sampler to spend steps on.)")
        delay = rtc_kwargs.get("delay_estimate", 0)
        delay = delay.item() if isinstance(delay, torch.Tensor) else delay
        if delay:
            raise NotImplementedError(
                "ACT has no Real-Time Chunking (no iterative sampler whose prefix could be "
                f"frozen): got delay_estimate={delay}. Sample with delay_estimate=0.")
        steps_executed = rtc_kwargs.get("steps_executed", 0)
        steps_executed = steps_executed.item() if isinstance(steps_executed, torch.Tensor) else steps_executed
        if steps_executed:
            raise NotImplementedError(
                f"ACT ignores steps_executed (it is an RTC input): got {steps_executed}. "
                "The chunk is always decoded fresh from the current observation.")

        if not hasattr(self, "_last_actions"):
            self._init_rtc_state()

        images = batch_dict["images"]
        B = images[0].shape[0]
        device = images[0].device

        flat_state = self._flat_state(batch_dict, B, device)
        force_feat = self._force_feature(batch_dict, device)

        if self.config.sample_latent_at_inference:
            z = torch.randn((B, self.latent_dim), dtype=torch.float32, device=device,
                            generator=self._device_generator(generator, device))
        else:
            # Upstream's inference path: z = 0, the prior MEAN (detr_vae.py:113).
            z = torch.zeros((B, self.latent_dim), dtype=torch.float32, device=device)

        pred = self._decode(z.to(self.dtype), flat_state, force_feat, batch_dict)

        if policy_smooth_lam > 0.0:
            from flore.s1_policy.model._action_smoothing import smooth_chunk_in_place

            pred = smooth_chunk_in_place(pred, policy_smooth_lam)

        self._last_actions = pred.detach().clone()
        self._last_actions_valid = True
        return pred.float()

    @staticmethod
    def _device_generator(generator: torch.Generator | None, device: torch.device):
        """Relocate a CPU generator onto ``device`` so seeded GPU sampling works.

        Same helper the diffusion sampler carries and for the same reason: a caller doing
        reproducible sampling naturally builds ``torch.Generator()`` (CPU by default),
        which ``torch.randn(device='cuda', generator=...)`` rejects. The full device is
        compared, not just ``.type``, so a cuda:0 generator against a cuda:1 model
        relocates too.
        """
        if generator is None or generator.device == device:
            return generator
        seed = int(torch.randint(0, torch.iinfo(torch.int64).max, (),
                                 device=generator.device, generator=generator).item())
        return torch.Generator(device=device).manual_seed(seed)
