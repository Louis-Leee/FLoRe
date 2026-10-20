"""Online HL-v2 planner core: a standalone checkpoint loader + a live observation
builder + running-anchor un-relativization + a settable force-command gate.

This is the model-facing half of the online S0<->S1 planner (the viser page and the
sensor/S0 plumbing live in ``scripts/hl_serving.py``). It is a separate, dependency-
light module so the load-bearing math — obs assembly, anchor decode, and the force
gate — is unit-testable WITHOUT viser, a robot, or the sensor bridges.

Why not reuse ``FlowInferenceSession``/``DiffusionInferenceSession`` directly:
those wrap :class:`HLSampleSource`, which rebuilds the exact TRAINING ``HLDataset``
from the checkpoint's zarr paths and cryptographically validates the split/norm
provenance against the on-disk dataset. ``infer()`` then only accepts an integer
dataset index — there is NO live-observation entry point. An online planner has no
training zarr; it must construct the model batch itself from live sensor rings. So
we reuse the *loader seams* from :mod:`flore.s1_policy.inference.common`
(``read_checkpoint_config`` / ``dataclass_config`` / ``validate_model_layout`` /
``strict_load_weights``) and load ``norm_stats.json`` standalone via
:func:`flore.s1_policy.data.norm_stats.parse_norm_stats` — bypassing the dataset-
welded ``HLSampleSource`` entirely.

Frame + units contract (grounded in ``flore/s1_policy/data``):
  * The per-hand action block is ``[pos3 | rot6d6 | gripper1 | force6]`` (16 dims);
    the head is ``[pos3 | rot6d6]`` (9). Slices come from ``layout.slots(role)`` —
    never hardcode offsets.
  * Pose is decoded per-role against that role's OWN current-EE anchor
    (``per_keypoint``): ``delta_ee_to_absolute(pos_rel, rot6d_rel, anchor_pos,
    anchor_rot6d)`` (the exact inverse the dataset relativized with).
  * The gripper channel is a fraction of ``GRIPPER_WIDTH_REF_M`` (0.095 m).
  * Force is in the RAW Bota SENSOR frame, physical N/N·m — there is NO ``ft_scale``
    in HL-v2 (that constant belongs to the unrelated UVA/MuSe model). The whole
    action vector (force columns included) is quantile-normalized under
    ``norm_stats['actions']`` at train time and inverted by ``unnormalize_quantile``
    here, so the force6 slice comes back in physical units.
  * The observation carries NO absolute position and NO force in ``state`` — only
    ``wrt_start_rot6d`` (R_start^T R_now) per keypoint (+ gripper fraction for
    gripper roles). ``force_history`` is a separate (8, 6*n_force_hands) raw-wrench
    window with its own encoder + a validity flag.

The force gate (the user's two-level contract, 2026-08-01):
  * ``force_enabled`` master switch (default OFF): OFF => publish NO force command
    (all zeros) => the low level is a pure position controller.
  * ``gate_n`` threshold (settable live, default 6 N): when ON, a hand whose
    predicted ``||F||`` (Newtons only, the first 3 of the 6) does NOT exceed the
    gate is HARD-ZEROED, so below the gate the system is still pure position; above
    the gate the (raw sensor-frame) wrench passes through to be published. See
    :func:`gate_force_command`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from flore.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HORIZON,
    FORCE_DIM,
    FORCE_HIST_LEN,
    ActionLayout,
    get_layout,
)
from flore.s1_policy.data.norm_stats import (
    NormStats,
    normalize_quantile,
    parse_norm_stats,
    unnormalize_quantile,
)
from flore.s1_policy.data.transforms import (
    GRIPPER_WIDTH_REF_M,
    delta_ee_to_absolute,
    gripper_width_to_norm,
    rot6d_relative,
)
from flore.s1_policy.inference.common import (
    dataclass_config,
    read_checkpoint_config,
    resolve_inference_device,
    strict_load_weights,
    validate_model_layout,
)


#: Quantization tolerance when assigning a force sample to its causal bin, in units of ONE
#: BIN. See :meth:`LivePlanner._bin_force_history` for the measured failure it fixes (an
#: edge-aligned push rate dropping a bin and invalidating a fully-measured window).
_BIN_EPS_FRAC = 1e-9

#: Age past which a live-ingest ring entry is NOT the current observation, in seconds.
#: Why this has to exist at all: every ingest source degrades by REPEATING its last value,
#: never by going absent. A dead camera bridge keeps ``_WSSubscriber.latest()`` returning the
#: same frame; a dead Bota reader thread leaves the F/T publisher broadcasting a frozen
#: payload with a fresh ``seq``/``t``; a crashed ingest thread freezes every ring at once. In
#: all three the model keeps being fed a confident, plausible, WRONG observation, so the only
#: way to notice is to time-stamp the rings and check the age here.
#: 0.5 s = 15 missed frames on the 30 Hz obs grid — long enough to ride out a scheduling
#: hiccup, short enough that no chunk (50 steps = 1.67 s) executes on a frozen scene.
OBS_MAX_AGE_S = 0.5
#: Same, for the gripper latch. Looser because the jaw bridge is a slower, best-effort
#: publisher and its channel carries 0.4% of the prediction's sensitivity (measured), so a
#: brief gap is not worth refusing a Predict over — it only needs to stop being presented as
#: a measurement.
GRIP_MAX_AGE_S = 1.0


# ---------------------------------------------------------------------------
# Execute-N-of-S: the page's chunk-execution window, DERIVED from the horizon
# ---------------------------------------------------------------------------
#: The cadences a page offers, BEFORE filtering to the checkpoint's own horizon. At the v3
#: grid's 30 Hz these are 0.27 / 0.53 / 1.0 / 1.67 / 3.33 / 6.67 s of demonstration executed
#: per chunk. The ladder is deliberately fixed and horizon-INDEPENDENT so that an operator who
#: has learned "16 means about half a second" reads the same thing off an H=50 page and an
#: H=200 one; only which rungs are REACHABLE moves with the horizon.
EXEC_STEP_LADDER = (8, 16, 30, 50, 100, 200)


def exec_step_choices(horizon: int, ladder: Sequence[int] = EXEC_STEP_LADDER) -> tuple[int, ...]:
    """The execute-N-of-S choices a page may offer for a chunk of ``horizon`` steps.

    ONE implementation for every planner page, because the alternative is a hardcoded option
    tuple per script and those do not follow the checkpoint. A hardcoded ``("8","16","30","50")``
    on an H=100 or H=200 model silently caps the page at a THIRD of the chunk it just paid for:
    the drip clamps to ``min(selected, chunk_len)`` so nothing unsafe happens, the operator just
    cannot ask for the horizon they trained.

    ``horizon`` is safe to take from :data:`ACTION_HORIZON` at module scope even though that
    constant is resolved from ``FLORE_ACTION_HORIZON`` at IMPORT time: the horizon is PINNED
    against the checkpoint by :func:`flore.s1_policy.inference.common.validate_model_layout`
    (``action_horizon`` is checked whether or not the grid comes from the checkpoint), so a
    process whose constant disagrees with the model refuses to serve it at all rather than
    running with a mis-sized window. The launchers export the pin from the checkpoint's own
    ``config.json``, which is what makes the two agree in the first place.

    The horizon itself is ALWAYS a choice, so "execute the whole chunk" is always offered and
    the result can never be empty (a page with an empty dropdown is unusable, and an H below
    the smallest rung is a legitimate configuration).

    ``ladder`` exists for the left-only planner, which keeps its historical ``(16, 30)``: its
    ``--exec-steps`` default is 4, so adding an 8 rung would silently re-snap that reviewed rig
    default from 16 down to 8.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError(f"action horizon must be >= 1 step, got {horizon!r}")
    return tuple(sorted({int(n) for n in (*ladder, h) if 1 <= int(n) <= h}))


def snap_exec_steps(requested: int, choices: Sequence[int]) -> int:
    """The member of ``choices`` nearest ``requested`` — ties going to the SMALLER.

    A CLI flag is free-form (``--exec-steps 20``) while the page's dropdown is a fixed set, and
    viser does not reject an ``initial_value`` outside its options: it renders a blank selector
    whose ``.value`` is nonetheless the off-list number. So an unsnapped flag gives a page that
    shows nothing while the drip loop executes 20 steps — the control and the behaviour
    disagree, with no error anywhere. Snapping first makes the dropdown's displayed value the
    truth by construction.

    Ties break DOWNWARD because a shorter window re-plans sooner, which is the safer direction
    on every one of these pages: less open-loop motion per observation.
    """
    opts = [int(c) for c in choices]
    if not opts:
        raise ValueError("no execute-N-of-S choices to snap to")
    req = int(requested)
    return min(opts, key=lambda c: (abs(c - req), c))


# ---------------------------------------------------------------------------
# Observation encoding seams — ONE implementation, three consumers
# ---------------------------------------------------------------------------
# The live planner (below), the deploy RECORDER (flore.s1_policy.serving.hl_record) and the
# re-inference tool (scripts/reinfer_hl_take.py) must all agree, byte for byte, about what
# "the observation" is. They are three different processes reading three different sources
# (live sensor rings / a written take / a re-read take), so any duplicated normalization is
# a place where "what the model saw at deploy" and "what the model is fed on replay" can
# drift silently — and the whole point of the recorder is to make those two comparable.
def resize_center_crop_224(img: np.ndarray, out_w: int = 224, out_h: int = 224) -> np.ndarray:
    """Aspect-preserving resize + CENTER CROP to 224x224 (cover fit), channel-order agnostic.

    Train/deploy parity: the training converter crops with EXACT INTEGER geometry —
    ``scale = max(out/in_w, out/in_h)``, ``resize = ceil(in*scale)`` (NOT round) clamped to
    ``>= out``, then a floor-div center crop, ALWAYS ``cv2.INTER_AREA`` (see the UMI
    submodule's ``umi_convert_core.compute_crop_resize``). A naive ``cv2.resize(f, (224,224))``
    STRETCHES and uses INTER_LINEAR, feeding the policy an out-of-distribution crop.

    Lives here so the LIVE ingest and an OFFLINE re-inference on a recorded camera frame crop
    identically — a re-inference whose crop differs from the deploy one is measuring the crop,
    not the checkpoint.
    """
    import cv2

    in_h, in_w = img.shape[:2]
    scale = max(out_w / in_w, out_h / in_h)
    resize_w = max(int(np.ceil(in_w * scale)), out_w)
    resize_h = max(int(np.ceil(in_h * scale)), out_h)
    crop_x = (resize_w - out_w) // 2
    crop_y = (resize_h - out_h) // 2
    resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    return resized[crop_y:crop_y + out_h, crop_x:crop_x + out_w]


def normalize_image_chw(img_224_rgb_uint8: np.ndarray, mean: float, std: float) -> np.ndarray:
    """``(224,224,3)`` uint8 RGB → ``(3,224,224)`` float32 ``(x/255 - mean)/std``.

    Mirrors ``HLDataset._load_image``. Used by :meth:`LivePlanner.push_image` on the live
    path and by the re-inference tool on a recorded frame, so a recorded uint8 crop
    re-encodes to the exact tensor the model was fed at deploy time."""
    img = np.asarray(img_224_rgb_uint8, dtype=np.float32)
    norm = (img / 255.0 - float(mean)) / float(std)
    return np.transpose(norm, (2, 0, 1)).astype(np.float32)


def denormalize_image_hwc(img_chw_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    """The exact inverse of :func:`normalize_image_chw` → ``(224,224,3)`` uint8 RGB.

    Round-trips to the SAME uint8 bytes for any frame that came in through
    :func:`normalize_image_chw` (the affine is invertible and the rounding is symmetric),
    which is what lets the recorder store the model's real input rather than a re-derived
    approximation of it."""
    hwc = np.transpose(np.asarray(img_chw_norm, dtype=np.float64), (1, 2, 0))
    rgb = (hwc * float(std) + float(mean)) * 255.0
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


@dataclass(frozen=True)
class ObsSnapshot:
    """One assembled, UN-normalized HL observation — the model's input before scaling.

    This is the recorder's unit of "what the policy saw". It is deliberately the
    pre-normalization form: quantile normalization is a function of the checkpoint's own
    ``norm_stats``, so storing the raw physical values keeps a recorded take readable (and
    re-runnable against a DIFFERENT checkpoint) instead of baking one model's scaling into
    the data.

    ``images`` is ``(n_cam*To,3,224,224)`` float32 in the same ``[-1,1]``-ish normalized image
    space the ring holds (images are NOT quantile-normalized — only state and force are), so
    it is fed verbatim; :func:`denormalize_image_hwc` turns it back into storable uint8. The
    leading axis is CAMERA-MAJOR then oldest-first, matching ``HLDataset.__getitem__``.
    """

    images: np.ndarray  # (n_cam*To,3,224,224) float32, camera-major, oldest-first per camera
    state: np.ndarray  # (To,state_dim) float64, raw
    state_valid: np.ndarray  # (To,state_dim) bool
    force_history: np.ndarray  # (FH,ch) float64, raw sensor-frame N/N·m
    force_history_valid: bool


def batch_from_obs(loaded: "LoadedHLModel", obs: ObsSnapshot) -> dict:
    """Collate + normalize one :class:`ObsSnapshot` into the batch ``sample_actions`` takes.

    The ONLY place the live planner and the re-inference tool normalize an observation, so
    the two cannot disagree about scaling. State and force_history are quantile-normalized
    under the checkpoint's own stats; images and actions are not (matching training).
    """
    dev = loaded.device
    state_t = torch.from_numpy(np.asarray(obs.state, dtype=np.float64)).float()
    force_t = torch.from_numpy(np.asarray(obs.force_history, dtype=np.float64)).float()
    if "state" in loaded.norm_stats:
        state_t = torch.from_numpy(
            normalize_quantile(state_t.numpy(), loaded.norm_stats["state"])
        ).float()
    if "force_history" in loaded.norm_stats and int(loaded.layout.force_hist_channels) > 0:
        force_t = torch.from_numpy(
            normalize_quantile(force_t.numpy(), loaded.norm_stats["force_history"])
        ).float()
    images = np.asarray(obs.images, dtype=np.float32)
    return {
        "images": [torch.from_numpy(images[t]).float().unsqueeze(0).to(dev)
                   for t in range(images.shape[0])],
        "img_masks": [torch.ones(1, dtype=torch.bool).to(dev) for _ in range(images.shape[0])],
        "state": state_t.unsqueeze(0).to(dev),
        "state_valid": torch.from_numpy(
            np.asarray(obs.state_valid, dtype=bool)).unsqueeze(0).to(dev),
        "force_history": force_t.unsqueeze(0).to(dev),
        # (1,) for one Bota, (1, n_ft) for more — the two shapes ``ForceHistoryEncoder``
        # accepts, and the same pair the trainer's collate produces at B=1. It is BROADCAST
        # from one bool rather than gated per sensor, and that is a fact about the live rig,
        # not a shortcut: both Botas arrive in the SAME :8771 message and
        # ``_pick_force_channels`` is all-or-nothing across roles, so a dropout masks both
        # windows at once. (Offline is different — a recorded take gates each sensor
        # independently, because they tare independently; see
        # ``RolloutResult.force_history_valid_per_sensor``.)
        "force_history_valid": _force_valid_tensor(loaded, bool(obs.force_history_valid), dev),
    }


def _force_valid_tensor(loaded: "LoadedHLModel", valid: bool, dev) -> torch.Tensor:
    n_ft = int(loaded.n_force_hands)
    if n_ft > 1:
        return torch.full((1, n_ft), bool(valid), dtype=torch.bool).to(dev)
    return torch.tensor([bool(valid)], dtype=torch.bool).to(dev)


def sample_chunk(
    loaded: "LoadedHLModel", batch: Mapping[str, Any], *, seed: int, num_steps: int | None = None
) -> np.ndarray:
    """Run ONE sampler pass on an already-collated batch → ``(H, action_dim)`` float64.

    Unnormalized back to physical units and finite-checked, i.e. the same array
    :meth:`LivePlanner.infer_chunk` hands to :func:`decode_chunk`. Factored out so an
    offline re-inference reproduces the deploy sampler exactly (same seed semantics, same
    step resolution, same unnormalization) without going through the live rings.
    """
    if loaded.model is None:
        raise RuntimeError(
            "this LoadedHLModel has no policy (replay_only_model) — nothing to sample.")
    steps = loaded.num_steps if num_steps is None else _resolve_num_steps(
        loaded.algorithm, loaded.config, num_steps
    )
    generator = torch.Generator(device=loaded.device).manual_seed(int(seed))
    loaded.model.reset()
    with torch.inference_mode():
        sampled = loaded.model.sample_actions(dict(batch), num_steps=steps, generator=generator)
    prediction = sampled[0].detach().float().cpu().numpy()
    if "actions" in loaded.norm_stats:
        prediction = unnormalize_quantile(prediction, loaded.norm_stats["actions"])
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("sampled HL chunk contains non-finite values")
    return prediction


# ---------------------------------------------------------------------------
# Force-command gate (the user's two-level position<->force contract)
# ---------------------------------------------------------------------------
def gate_force_command(
    wrench: np.ndarray,
    *,
    force_enabled: bool,
    gate_n: float,
) -> np.ndarray:
    """Apply the master switch + per-hand force deadband to a predicted wrench.

    Parameters
    ----------
    wrench : (..., 6) predicted wrench(es) ``[Fx Fy Fz Tx Ty Tz]`` in physical N /
        N·m (raw Bota sensor frame). The leading axis is per-hand when >1D.
    force_enabled : master switch. When False the result is ALL ZEROS regardless of
        magnitude — the low level receives no force command and is a pure position
        controller.
    gate_n : threshold in Newtons on ``||F||`` (the force triple, torque excluded).
        A hand whose force magnitude is ``<= gate_n`` is HARD-ZEROED (still position);
        a hand exceeding it passes through UNCHANGED (both force AND torque). Must be
        finite and >= 0.

    Returns
    -------
    (..., 6) float64, same shape as ``wrench``: either the input rows (above gate,
    switch on) or exact zeros (below gate, or switch off).

    Notes
    -----
    The gate keys on the CLEAN commanded ``||F||``, not a noisy measurement, so the
    0..gate_n band is an exact pure-position region. Hard-zero (not a soft ramp) is
    deliberate: it guarantees the published command is EXACTLY 0 below the gate, which
    is what makes the low level's ``force_cmd_local`` / integrator / ``x_tgt_shift``
    all collapse to 0 (true position control). The threshold-crossing discontinuity is
    then bounded downstream by the low level's own 2 N/tick slew on the applied F^cmd.
    """
    w = np.asarray(wrench, dtype=np.float64)
    if w.shape[-1] != FORCE_DIM:
        raise ValueError(f"wrench last dim must be {FORCE_DIM}, got shape {w.shape}")
    if not (np.isfinite(gate_n) and gate_n >= 0.0):
        raise ValueError(f"gate_n must be finite and >= 0, got {gate_n!r}")
    out = np.zeros_like(w)
    if not force_enabled:
        return out
    # Per-hand force magnitude uses only the force triple [0:3], not torque.
    force_mag = np.linalg.norm(w[..., 0:3], axis=-1, keepdims=True)  # (..., 1)
    passes = force_mag > float(gate_n)  # strictly greater: exactly gate_n stays gated
    return np.where(passes, w, out)


# ---------------------------------------------------------------------------
# Standalone loaded model (no HLSampleSource, no training zarr)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedHLModel:
    """A loaded HL-v2 policy plus the metadata a live planner needs.

    Built by :func:`load_hl_model` from a checkpoint directory (config.json +
    model.safetensors + optional norm_stats.json). ``model`` is on ``device`` and in
    eval mode; ``norm_stats`` is the standalone-parsed normalization bundle (empty if
    the checkpoint bundled none). Everything downstream slices via ``layout``.
    """

    algorithm: str
    model: torch.nn.Module | None  # None ONLY for replay_only_model() (no policy loaded)
    config: Any
    layout: ActionLayout
    norm_stats: Mapping[str, NormStats]
    device: torch.device
    num_steps: int
    obs_history_len: int
    obs_history_dt_s: float
    image_mean: float
    image_std: float
    image_key: str
    #: EVERY camera this checkpoint was trained on, in the model's own order — the same
    #: ``hl_data.image_keys`` the offline session hands to ``HLDataset``. One entry for a
    #: single-camera checkpoint (the vast majority), two for ``diffusion_umi_3tracker``. The
    #: live planner keeps one image ring per entry and stacks them CAMERA-MAJOR, matching
    #: ``HLDataset.__getitem__``'s ``for key in image_keys for hf in hist_frames``.
    image_keys: tuple[str, ...] = ()
    #: Causal force bins THIS CHECKPOINT's encoder expects — read from its own config, not
    #: from :data:`FORCE_HIST_LEN`. A checkpoint is a self-contained artifact: if the repo
    #: contract later moves (v2's 8 bins -> v3's 1 is exactly that), an already-trained model
    #: still wants the window it was trained on, and the rig has not changed. Sizing the live
    #: window from the live constant instead would hand the encoder a shape it rejects.
    #: ``obs_history_len`` above is already checkpoint-derived for the same reason.
    force_hist_len: int = FORCE_HIST_LEN
    #: ``wholebody`` ONLY: what this checkpoint's 29 joint columns mean —
    #: :data:`~flore.s1_policy.data.layout.WB_JOINT_TARGETS`. ``"absolute"`` for every keypoint
    #: checkpoint (which has no joint block) and for every wholebody one written before the knob
    #: existed. Checkpoint-derived rather than a serving flag for the reason the whole dataclass
    #: exists: it is a property of the trained weights and the two are indistinguishable by shape,
    #: so a flag would let an operator declare it wrong and see a plausible chunk either way.
    wb_joint_target: str = "absolute"

    @property
    def n_force_hands(self) -> int:
        return len(self.layout.force_roles)

    @property
    def num_cameras(self) -> int:
        """How many live camera streams this checkpoint needs. Never 0."""
        return max(1, len(self.image_keys))

    @property
    def force_hist_span_s(self) -> float:
        """Look-back of this checkpoint's force window, in seconds.

        DERIVED as ``force_hist_len * ACTION_DT`` — the v3 contract's rule that each input bin
        is exactly one action step wide (:data:`FORCE_HIST_SPAN_S` is the same product). The
        span is NOT stored in the checkpoint, so a future contract that decouples the two (v2
        was 8 bins over 1.0 s, i.e. 125 ms bins, which is not ``8 * ACTION_DT`` at any action
        rate) must bake it in rather than let this derivation guess."""
        return int(self.force_hist_len) * ACTION_DT


def _build_model(algorithm: str, config_block: Mapping[str, Any]):
    """Construct the right (config, model) pair for the algorithm without importing
    the sibling's model class unnecessarily."""
    if algorithm == "flow":
        from flore.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

        config = dataclass_config(DPFlowConfig, config_block)
        return config, DPFlowModel(config)
    if algorithm == "diffusion":
        from flore.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel

        config = dataclass_config(DPDiffusionConfig, config_block)
        return config, DPDiffusionModel(config)
    if algorithm == "act":
        from flore.s1_policy.model.dp_act import DPACTConfig, DPACTModel

        config = dataclass_config(DPACTConfig, config_block)
        return config, DPACTModel(config)
    raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {sorted(_ALGO_CONFIG_KEY)}")


#: ``algorithm`` -> its ``config.json`` block. Mirrors the trainer's
#: ``ALGO_CONFIG_FIELD``; imported lazily in :func:`_algo_config_key` so this
#: serving module keeps no import-time dependency on the training package.
_ALGO_CONFIG_KEY = ("flow", "diffusion", "act")


def _algo_config_key(algorithm: str) -> str:
    """The ``config.json`` key holding ``algorithm``'s REAL model config.

    Was ``"model" if algorithm == "flow" else "diffusion"`` — a binary collapse that read
    the DIFFUSION block for any third algorithm. It failed safe only because
    :func:`_build_model` refused the algorithm a few lines later; the moment a third branch
    was added there, an ACT checkpoint would have had the diffusion block poured into
    ``DPACTConfig``, surfacing as a baffling "unsupported DPACTConfig fields" instead of
    the real problem. Delegated to the trainer's single mapping so the two cannot drift.
    """
    from flore.s1_policy.training.config import ALGO_CONFIG_FIELD

    try:
        return ALGO_CONFIG_FIELD[algorithm]
    except KeyError:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of {sorted(ALGO_CONFIG_FIELD)}"
        ) from None


def load_hl_model(
    checkpoint_dir: str | Path,
    algorithm: str,
    *,
    device: str = "cuda",
    num_steps: int | None = None,
    max_cameras: int = 1,
) -> LoadedHLModel:
    """Load one HL-v2 checkpoint for LIVE inference (no training dataset required).

    Reuses the inference package's strict loader seams (config validation, layout
    agreement, sha256-checked weight load) but loads normalization standalone and
    never touches :class:`HLSampleSource` (which would demand the training zarr).

    ``max_cameras`` is how many live camera streams the CALLER has actually subscribed to,
    and it is a hard ceiling, not a hint. See the note at the ``validate_model_layout`` call.
    """
    checkpoint, raw = read_checkpoint_config(checkpoint_dir, algorithm)
    config_block = raw.get(_algo_config_key(algorithm))
    if not isinstance(config_block, dict):
        raise ValueError(f"{algorithm} checkpoint is missing its config block")

    hl_data = raw["hl_data"]
    layout = get_layout(str(hl_data.get("mode", "left_only")))
    if layout.legacy:
        raise ValueError("online HL inference does not support the legacy32 layout")

    config, model = _build_model(algorithm, config_block)
    # grid_from_checkpoint: serving has NO dataset to agree with, so pinning the checkpoint's
    # force window to this build's FORCE_HIST_LEN would refuse a perfectly good older model
    # the moment the contract moves — and the planner below consumes the checkpoint's own
    # value, which is what makes that safe. The dataset-backed offline sessions keep the
    # strict pin (their HLSampleSource feeds the model from a zarr that must match).
    #
    # THE CAMERA COUNT IS THE CALLER'S CLAIM, CHECKED AGAINST THE CHECKPOINT'S NEED.
    #
    # The permissive direction is the dangerous one: a 2-camera model handed one view is
    # shape-correct and SILENT (the second encoder group just eats a grey frame), so it cannot
    # be left to a default derived from the checkpoint. It also cannot be pinned to 1 forever —
    # `diffusion_umi_3tracker` genuinely needs two, and this loader is how it reaches the rig.
    #
    # So the planner declares how many streams it has SUBSCRIBED to and this refuses by name if
    # the checkpoint wants more. `validate_model_layout` then gets the checkpoint's own count,
    # which is what the model was built with.
    image_keys = tuple(str(k) for k in (hl_data.get("image_keys") or ()))
    if not image_keys:
        image_keys = (str(hl_data.get("image_key", "camera0_rgb")),)
    if len(image_keys) > int(max_cameras):
        raise ValueError(
            f"checkpoint was trained on {len(image_keys)} cameras {list(image_keys)} but this "
            f"planner subscribed to {int(max_cameras)}. A missing view is SILENT (the encoder "
            f"group is fed a grey frame and the shapes still line up), so it is refused here. "
            f"Bring up the extra camera bridge and pass its URL — for the dual rig that is "
            f"camera R on :8777 (see scripts/run_hl_serving.sh).")
    validate_model_layout(config, hl_data, layout, grid_from_checkpoint=True,
                          num_cameras=len(image_keys))

    resolved_device = resolve_inference_device(device)
    strict_load_weights(model, checkpoint, raw["_provenance"]["model_sha256"])
    model.to(resolved_device).eval()

    # Standalone normalization: parse norm_stats.json directly (no provenance replay,
    # which is dataset-bound). Width-check the columns we actually use so a mismatched
    # norm bundle fails loudly rather than silently mis-scaling.
    norm_stats: dict[str, NormStats] = {}
    if bool(hl_data.get("norm_stats_path")):
        norm_path = checkpoint / "norm_stats.json"
        if not norm_path.is_file():
            raise FileNotFoundError("checkpoint declares normalization but norm_stats.json is missing")
        norm_stats = parse_norm_stats(norm_path.read_text())
        expected_widths = {
            "actions": layout.action_dim,
            "state": layout.state_dim,
            "force_history": layout.force_hist_channels,
        }
        for key, width in expected_widths.items():
            if key in norm_stats and norm_stats[key].q01.shape != (width,):
                raise ValueError(
                    f"norm_stats[{key!r}] width {norm_stats[key].q01.shape} != expected ({width},)"
                )

    resolved_steps = _resolve_num_steps(algorithm, config, num_steps)

    return LoadedHLModel(
        algorithm=algorithm,
        model=model,
        config=config,
        layout=layout,
        norm_stats=norm_stats,
        device=resolved_device,
        num_steps=resolved_steps,
        obs_history_len=int(hl_data.get("obs_history_len", 2)),
        obs_history_dt_s=float(hl_data.get("obs_history_dt_s", 0.25)),
        image_mean=float(hl_data.get("image_mean", 0.5)),
        image_std=float(hl_data.get("image_std", 0.5)),
        image_key=image_keys[0],
        image_keys=image_keys,
        force_hist_len=int(getattr(config, "force_hist_len", FORCE_HIST_LEN)),
        wb_joint_target=str(hl_data.get("wb_joint_target", "absolute")),
    )


def replay_only_model(
    layout_mode: str = "left_only",
    *,
    obs_history_len: int = 2,
    obs_history_dt_s: float = 0.25,
) -> LoadedHLModel:
    """A :class:`LoadedHLModel` with **no model** — the anchor/layout half only.

    RECORDED-EPISODE REPLAY needs none of the policy: the action chunk comes out of a zarr
    (:mod:`flore.s1_policy.serving.hl_replay`) and is decoded by :func:`decode_chunk` against
    the running anchor. It still needs the LAYOUT (to slice the action vector) and the anchor
    bookkeeping (:class:`LivePlanner`'s seed/commit/snapshot). Loading a checkpoint just to
    reach those would pin a GPU and several GB of VRAM to run a pure data playback.

    So this returns the same dataclass the real loader returns, with ``model=None`` — which
    lets ``scripts/hl_serving.py`` keep ONE code path for the wire, the force gate, the drip,
    and the clutch re-anchor. :meth:`LivePlanner.infer_chunk` refuses on ``model is None``
    rather than crashing deep in torch, so a replay-only process that somehow reached the
    model path fails loudly instead of half-working.

    ``device`` is CPU and ``norm_stats`` is empty on purpose: nothing here normalizes or
    runs tensors. ``image_mean``/``image_std`` keep their dataset defaults so
    :meth:`LivePlanner.push_image` stays well-defined if a caller pushes frames anyway.
    """
    layout = get_layout(str(layout_mode))
    if layout.legacy:
        raise ValueError("recorded replay does not support the legacy32 layout")
    return LoadedHLModel(
        algorithm="replay",  # not 'flow'/'diffusion': there is no sampler here
        model=None,
        config=None,
        layout=layout,
        norm_stats={},
        device=torch.device("cpu"),
        num_steps=0,
        obs_history_len=int(obs_history_len),
        obs_history_dt_s=float(obs_history_dt_s),
        image_mean=0.5,
        image_std=0.5,
        image_key="camera0_rgb",
        image_keys=("camera0_rgb",),
    )


def _resolve_num_steps(algorithm: str, config, num_steps: int | None) -> int:
    """Resolve the sampler-step count for one algorithm, honoring its own ceiling.

    The default lives under a DIFFERENT field name per algorithm, so this dispatches
    explicitly. It used to be ``if flow: inference_steps else: num_inference_steps`` — the
    ``else`` reached for ``num_inference_steps`` on ANY non-flow config, which a
    ``DPACTConfig`` does not have, surfacing as an ``AttributeError`` deep inside the
    loader rather than a statement about the algorithm.
    """
    if algorithm == "act":
        # ACT decodes the whole chunk in ONE forward pass; there is no sampler loop. Any
        # other value is refused rather than silently ignored, so an operator who typed
        # `--num-steps 16` learns that it buys nothing here instead of believing it did.
        steps = 1 if num_steps is None else int(num_steps)
        if steps != 1:
            raise ValueError(
                f"ACT decodes the whole chunk in one forward pass; num_steps must be 1, got "
                f"{steps}. (There is no iterative sampler to spend steps on.)")
        return 1
    if algorithm == "flow":
        default = int(config.inference_steps)
    elif algorithm == "diffusion":
        default = int(config.num_inference_steps)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {sorted(_ALGO_CONFIG_KEY)}")
    steps = default if num_steps is None else int(num_steps)
    if steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {steps}")
    if algorithm == "diffusion" and steps > int(config.num_train_timesteps):
        raise ValueError(f"diffusion num_steps {steps} exceeds num_train_timesteps {config.num_train_timesteps}")
    return steps


# ---------------------------------------------------------------------------
# Live observation + running-anchor planner
# ---------------------------------------------------------------------------
class LivePlanner:
    """Owns the running per-role EE anchor + the live sensor rings, and turns one
    model inference into an absolute per-role chunk plus a gated force command.

    All poses are tracked in whatever absolute frame the caller seeds the anchor in
    (the low level's pelvis frame at deploy). The obs is built EE-relative to the
    SNAPSHOTTED anchor so "obs base == action base" is structural. The anchor is only
    advanced by :meth:`commit_anchor`, which the server calls AFTER a chunk is really
    executed — so a preview (predict-only) never moves the model frame.

    This class is deliberately robot-agnostic: it produces per-role absolute pose /
    gripper / gated-wrench arrays. Mapping those to the vr_3point wire (the TCP lever,
    the LEFT slot, right/head hold) lives in the server, exactly as the muse planner
    keeps the wire lever out of the model seam.
    """

    def __init__(self, loaded: LoadedHLModel, obs_hz: float = 16.0):
        import threading

        self.m = loaded
        self.layout = loaded.layout
        self.roles = loaded.layout.roles
        self._lock = threading.Lock()
        self._obs_hz = float(obs_hz)

        To = int(loaded.obs_history_len)
        self._To = To
        n_roles = len(self.roles)

        # Running per-role anchor (absolute): current-EE pose the action decodes against.
        # Seeded to identity; the server seeds real ready-pose values via seed_anchor().
        self._anchor_pos = np.zeros((n_roles, 3), dtype=np.float64)
        self._anchor_rot6d = np.tile(_IDENTITY_ROT6D, (n_roles, 1)).astype(np.float64)
        # Latched episode-start rotation per role (for the wrt_start_rot6d state channel).
        self._start_rot6d = self._anchor_rot6d.copy()

        # Proprio rings (length To, oldest-first): per-role rotation + gripper fraction.
        self._buf_rot6d = [self._anchor_rot6d.copy() for _ in range(To)]
        self._buf_grip = [np.zeros(n_roles, dtype=np.float64) for _ in range(To)]
        # Image rings: ONE PER CAMERA, each of length To, oldest-first; each entry a
        # (3,224,224) float [-1,1]-normalized frame (mean/std applied at push, matching
        # HLDataset._load_image) or None if unfilled.
        #
        # Per camera rather than one interleaved ring because the two D405s are separate
        # bridges on separate ports arriving at their own rates: a shared ring would let a
        # fast camera evict the slow one's history and silently feed the model two views of
        # the SAME instant in different history slots.
        self._n_cam = int(loaded.num_cameras)
        self._buf_img: list[list[np.ndarray | None]] = [
            [None for _ in range(To)] for _ in range(self._n_cam)
        ]
        # Force-history: the training converter builds FORCE_HIST_LEN causal ANTI-ALIASED
        # bins tiling exactly FORCE_HIST_SPAN_S ending at t0, each bin the mean of the RAW
        # F/T samples whose timestamps fall in it (``past_bin_edges`` +
        # ``resample_wrench_uniform_past``). To reproduce that live, the ring is keyed on
        # SAMPLE TIME and evicted by AGE, and :meth:`_bin_force_history` bins by real
        # timestamps.
        #
        # ⚠ IT USED TO BE KEYED ON COUNT — ``ceil(FORCE_HIST_SPAN_S * obs_hz) + 2`` slots,
        # all of them averaged into the window. At the v2 grid (8 bins over 1 s, 16 slots at
        # obs_hz=16) the ``+2`` was 12% slack and harmless. On the v3 grid the span collapsed
        # to ONE 33.3 ms bin, so at obs_hz=30 the cap became ``1 + 2 = 3`` slots — a ring
        # spanning 100 ms whose mean was published as the 33.3 ms bin the model trained on,
        # i.e. a 3x-too-wide window. Age-based eviction cannot drift that way: the window is
        # FORCE_HIST_SPAN_S wide by construction at any push rate.
        # Shape from the CHECKPOINT (``loaded.force_hist_len``), not from FORCE_HIST_LEN: an
        # already-trained model wants the window it was trained on even after the repo contract
        # moves, and its ForceHistoryEncoder hard-rejects any other length. See
        # LoadedHLModel.force_hist_len.
        self._fh_len = max(1, int(loaded.force_hist_len))
        self._fh_ch = int(self.layout.force_hist_channels)
        self._fh_span_s = float(loaded.force_hist_span_s)
        # Hard length cap, a backstop against an unbounded push rate — NOT the window
        # definition (age is). Generous: 2 kHz of F/T over the span, floor 8 slots.
        self._fh_raw_cap = max(8, self._fh_len, int(np.ceil(self._fh_span_s * 2000.0)))
        # Empty (cold start = no measurement) so the window reads INVALID until real samples
        # arrive — the learned no-data embedding, not a fabricated zero force. Entries are
        # ``(t_monotonic, wrench_or_None)``, oldest-first.
        self._buf_force: list[tuple[float, np.ndarray | None]] = []

        # Latest gripper fraction per gripper-role (fed as proprio), keyed by role.
        self._grip_frac = {role: 0.0 for role in self.layout.gripper_roles}

        # ARRIVAL STAMPS for the rings that had none. The force ring carries a timestamp per
        # sample by construction; the image ring was a bare list and the gripper a bare float,
        # so "this frame is 12 s old" was not representable — the ingest thread simply re-pushed
        # the same bytes and every consumer read it as current. ``None`` = never written, which
        # is a DIFFERENT and worse state than stale: an unwritten gripper reads 0.0 m, which
        # normalizes to about -2.6 against this checkpoint's own quantiles (training band
        # [-1, 1]) and is fed to the model as a real measurement of a fully closed jaw.
        # See :meth:`obs_ages` and the gripper validity gate in :meth:`snapshot_obs`.
        # One stamp PER CAMERA, and :meth:`obs_ages` reports the OLDEST — the same reasoning
        # as the gripper roles below: one dead camera bridge in a 2-camera deploy must not be
        # masked by a live one, because a frozen view is fed to its encoder group as a real
        # observation of a scene that has moved on.
        self._t_img: list[float | None] = [None for _ in range(self._n_cam)]
        self._t_grip: dict[str, float | None] = {role: None for role in self.layout.gripper_roles}
        #: Consumer-tunable staleness thresholds (seconds); the serving launcher overrides
        #: these from ``--obs-max-age`` so one flag moves the page readout, the Predict gate
        #: and the gripper validity flag together.
        self.obs_max_age_s = float(OBS_MAX_AGE_S)
        self.grip_max_age_s = float(GRIP_MAX_AGE_S)

    # --- seeding / reset -------------------------------------------------
    def seed_anchor(self, anchor_pos: np.ndarray, anchor_rot6d: np.ndarray) -> None:
        """Seed the running anchor (and re-latch the start rotation + re-seed rings) to
        real absolute per-role ready poses. ``anchor_pos`` (n_roles,3), ``anchor_rot6d``
        (n_roles,6)."""
        ap = np.asarray(anchor_pos, dtype=np.float64)
        ar = np.asarray(anchor_rot6d, dtype=np.float64)
        n = len(self.roles)
        if ap.shape != (n, 3) or ar.shape != (n, 6):
            raise ValueError(f"seed_anchor expects ({n},3)/({n},6), got {ap.shape}/{ar.shape}")
        with self._lock:
            self._anchor_pos = ap.copy()
            self._anchor_rot6d = ar.copy()
            self._start_rot6d = ar.copy()
            self._buf_rot6d = [ar.copy() for _ in range(self._To)]
            self._buf_grip = [self._grip_frac_vec() for _ in range(self._To)]

    def reset_start_pose(self) -> None:
        """Re-latch the episode-start rotation = current anchor and re-seed the proprio
        rings (fresh episode; the wrt_start channel is now relative to here)."""
        with self._lock:
            self._start_rot6d = self._anchor_rot6d.copy()
            self._buf_rot6d = [self._anchor_rot6d.copy() for _ in range(self._To)]
            self._buf_grip = [self._grip_frac_vec() for _ in range(self._To)]

    def commit_anchor(self, anchor_pos: np.ndarray, anchor_rot6d: np.ndarray) -> None:
        """Advance the running anchor to an EXECUTED chunk's last step (per role)."""
        ap = np.asarray(anchor_pos, dtype=np.float64)
        ar = np.asarray(anchor_rot6d, dtype=np.float64)
        with self._lock:
            self._anchor_pos = ap.copy()
            self._anchor_rot6d = ar.copy()

    def snapshot_anchor(self) -> tuple[np.ndarray, np.ndarray]:
        """An atomic ``(anchor_pos, anchor_rot6d)`` copy of the running per-role anchor.

        The decode base WITHOUT assembling an observation. :meth:`build_batch` also returns
        it, but a caller that supplies its own action chunk (the recorded-episode replay
        source in :mod:`flore.s1_policy.serving.hl_replay`) needs only the anchor — and must
        not be forced to build an image/force observation it will never feed to a model.
        Same frame contract as :meth:`build_batch`: whatever absolute frame the caller
        seeded, which for the deploy planner is the MODEL/puck frame."""
        with self._lock:
            return self._anchor_pos.copy(), self._anchor_rot6d.copy()

    def commit_role(self, role: str, pos: np.ndarray, rot6d: np.ndarray) -> None:
        """Advance ONE role's anchor to an executed step, leaving the others untouched.

        Used by the deploy server after an execute-N-of-S chunk: only the predicted hand
        (LEFT) moved; the held roles (RIGHT/HEAD) stay at their seeded home anchor. The
        anchor lands on the LAST EXECUTED step so the next chunk branches from where the
        arm actually stopped (not the un-executed chunk end)."""
        ri = self.roles.index(role)
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        r = np.asarray(rot6d, dtype=np.float64).reshape(6)
        with self._lock:
            self._anchor_pos[ri] = p
            self._anchor_rot6d[ri] = r

    # --- live ingest -----------------------------------------------------
    def push_image(self, img_224_rgb_uint8: np.ndarray, t: float | None = None,
                   camera: int = 0) -> None:
        """Push a (224,224,3) uint8 RGB frame, normalized to (3,224,224) [-1,1] like
        ``HLDataset._load_image`` (img/255 - mean)/std.

        ``camera`` indexes :attr:`LoadedHLModel.image_keys` — 0 is the checkpoint's FIRST
        camera, which for every layout is ``camera0_rgb``. It is refused rather than clamped:
        pushing camera 1 into a single-camera planner would silently overwrite the only view
        the model has.

        ``t`` defaults to ``time.monotonic()`` (the push instant) and is what
        :meth:`obs_ages` reports; pass it explicitly only to drive a deterministic test.
        Stamped per push, NOT per distinct frame: a bridge that re-sends an identical JPEG is
        indistinguishable here, and detecting THAT is the recorder's ``seq`` dedup job."""
        index = int(camera)
        if not 0 <= index < self._n_cam:
            raise IndexError(
                f"camera {index} out of range: this checkpoint has {self._n_cam} camera(s) "
                f"{list(self.m.image_keys) or ['camera0_rgb']}")
        chw = normalize_image_chw(img_224_rgb_uint8, self.m.image_mean, self.m.image_std)
        ts = time.monotonic() if t is None else float(t)
        with self._lock:
            ring = self._buf_img[index]
            ring.append(chw)
            if len(ring) > self._To:
                ring.pop(0)
            self._t_img[index] = ts

    def push_force(self, wrench_channels: np.ndarray | None, t: float | None = None) -> None:
        """Push ONE RAW sensor-frame wrench sample, ``(force_hist_channels,)`` (per force
        hand, 6 each), stamped with the time it was MEASURED.

        Call this once per real F/T sample, at the sensor's own rate — NOT once per
        observation tick. The training bin is the mean of every raw sample inside
        FORCE_HIST_SPAN_S (~6.8 samples at the measured 204 Hz F/T rate), so a caller that
        pushes one decimated sample per 30 Hz tick hands the model a differently-filtered
        signal than it trained on. ``build_batch`` bins whatever arrives by TIMESTAMP, so
        over-pushing is free and under-pushing degrades gracefully.

        ``t`` defaults to ``time.monotonic()`` (the push instant); pass it explicitly only to
        replay a known cadence deterministically, and then keep the same clock for every push
        — the bin edges are differences of these values.

        ``None`` marks a dropout: no measured force at ``t`` -> any window containing it is
        invalid -> the encoder swaps in its learned no-data embedding. Validity is per WINDOW
        (matching the converter's per-window flag), so a dropout stops mattering once it ages
        out of the span."""
        ts = time.monotonic() if t is None else float(t)
        if wrench_channels is None:
            entry: tuple[float, np.ndarray | None] = (ts, None)
        else:
            w = np.asarray(wrench_channels, dtype=np.float64).reshape(-1)
            if w.shape[0] != self._fh_ch:
                raise ValueError(f"force bin must be ({self._fh_ch},), got {w.shape}")
            entry = (ts, w)
        with self._lock:
            self._buf_force.append(entry)
            # Evict by AGE against the NEWEST sample's time (not wall-now): the window is
            # defined relative to t0, and t0 is the newest measurement we have.
            t_new = self._buf_force[-1][0]
            cutoff = t_new - self._fh_span_s
            keep = [e for e in self._buf_force if e[0] > cutoff]
            # An out-of-order or backwards-jumping timestamp would otherwise silently empty
            # the ring; keep at least the newest sample so a monotonic-clock hiccup degrades
            # to "one sample in the window" rather than to "no force at all".
            self._buf_force = keep or [self._buf_force[-1]]
            if len(self._buf_force) > self._fh_raw_cap:
                del self._buf_force[: len(self._buf_force) - self._fh_raw_cap]

    def _bin_force_history(self, raw: list) -> tuple[np.ndarray, bool]:
        """Bin the timestamped raw wrench ring into the FORCE_HIST_LEN causal windows the
        model trained on. Returns ``(window, valid)``.

        ``raw`` is ``[(t, wrench_or_None)]``, oldest-first. ``t0`` is the NEWEST timestamp;
        bin ``j`` covers ``(t0 - span + j*dt, t0 - span + (j+1)*dt]`` and is the mean of the
        real samples inside it — the same half-open, anti-aliased, causal tiling as
        ``umi_convert_core.past_bin_edges`` + ``_bin_mean_with_interp_fallback``, so the last
        bin ends exactly AT t0 and no bin can contain a future sample.

        ``valid`` is False unless EVERY bin holds at least one real sample and NO dropout
        (``None``) falls inside the span — a partially-measured window must reach the encoder
        as "no data", not as a plausible-looking average. An empty bin in an otherwise valid
        window still edge-holds a neighbour so the returned array is never garbage, but the
        flag already tells the encoder to ignore it."""
        out = np.zeros((self._fh_len, self._fh_ch), dtype=np.float64)
        if not raw or self._fh_ch == 0:
            return out, False
        t0 = raw[-1][0]
        dt = self._fh_span_s / self._fh_len
        # Bin index from age, derived from the edges rather than from a slot position: bin j
        # is (t0-span+j*dt, t0-span+(j+1)*dt], so a sample of age a lands in
        # j = L-1-floor(a/dt). The sample AT t0 (a=0) lands in the LAST bin, and one exactly
        # on an interior edge lands in the EARLIER bin — the converter's `(lo, hi]`.
        #
        # _BIN_EPS_FRAC is load-bearing, not defensive dressing. A caller that pushes exactly
        # at the bin rate — a 30 Hz bridge, or a replay source stepping the action grid — puts
        # every sample ON an edge, where `age/dt` is a difference of accumulated floats and
        # lands just BELOW the integer: measured 2.9999999999998295 for the 3*dt-old sample at
        # L=8, so floor() returned 2, that sample was filed into the wrong bin, and the bin it
        # should have filled read EMPTY -> the whole window reported INVALID and the encoder
        # got its no-data embedding on a fully measured window. The nudge is 1e-9 of a BIN
        # (~33 ps here), so it can only reclassify a sample already indistinguishable from the
        # edge, while making the aligned case exact.
        bins: list[list[np.ndarray]] = [[] for _ in range(self._fh_len)]
        dropout = False
        for ts, w in raw:
            age = t0 - ts
            # STRICT here, with no epsilon: `push_force`'s eviction keeps exactly
            # `age < span` against this same t0, so loosening only this side would let the
            # binner see a sample the ring had already decided was out of the window.
            if age >= self._fh_span_s or age < 0.0:
                continue  # outside the causal span (eviction should have removed it)
            idx = int(np.clip(self._fh_len - 1 - int(np.floor(age / dt + _BIN_EPS_FRAC)),
                              0, self._fh_len - 1))
            if w is None:
                dropout = True
            else:
                bins[idx].append(w)
        valid = (not dropout) and all(len(b) > 0 for b in bins)
        last_good: np.ndarray | None = None
        for b in range(self._fh_len):
            if bins[b]:
                out[b] = np.mean(np.stack(bins[b]), axis=0)
                last_good = out[b]
            elif last_good is not None:
                out[b] = last_good  # edge-hold forward from the last populated bin
        # Back-fill leading empty bins with the first populated one.
        first_good = next((out[b] for b in range(self._fh_len) if bins[b]), None)
        if first_good is not None:
            for b in range(self._fh_len):
                if bins[b]:
                    break
                out[b] = first_good
        return out, valid

    def push_gripper(self, role: str, width_m: float, t: float | None = None) -> None:
        """Update the latest observed gripper opening (metres) for a gripper role.

        ``t`` defaults to the push instant (see :meth:`push_image`). Stamping is what lets
        :meth:`snapshot_obs` mark the channel UNMEASURED instead of presenting a stale or
        never-written width as a measurement."""
        if role not in self._grip_frac:
            return
        ts = time.monotonic() if t is None else float(t)
        with self._lock:
            self._grip_frac[role] = float(gripper_width_to_norm(np.asarray(width_m)))
            self._t_grip[role] = ts

    def obs_ages(self, t_now: float | None = None) -> dict[str, float | None]:
        """Age in seconds of each live-ingest channel, or ``None`` if never written.

        ``{"image": …, "force": …, "gripper": …}``, measured against ``time.monotonic()`` —
        WALL CLOCK, deliberately, unlike the force window's own binning, which is relative to
        its newest sample (the window must be span-wide whatever the push rate). That relative
        eviction is correct for building the bin and useless for detecting a dead bridge: a
        frozen ring re-bins to the same "valid" window forever. This method is the wall-clock
        companion that makes the freeze visible.

        Callers: the page's freshness readout, the pre-inference gate in ``hl_serving``, and
        the gripper validity flag in :meth:`snapshot_obs`."""
        now = time.monotonic() if t_now is None else float(t_now)
        with self._lock:
            stamps = [v for v in self._t_img if v is not None]
            # Oldest camera, and None until EVERY camera has published at least once.
            t_img = min(stamps) if len(stamps) == len(self._t_img) and stamps else None
            t_force = self._buf_force[-1][0] if self._buf_force else None
            t_grip = [v for v in self._t_grip.values() if v is not None]
            # Oldest of the gripper roles: one dead jaw bridge in a bimanual setup must not be
            # masked by a live one. Empty (no gripper role, or none ever written) -> None.
            t_grip_oldest = min(t_grip) if len(t_grip) == len(self._t_grip) and t_grip else None
        return {
            "image": None if t_img is None else now - t_img,
            "force": None if t_force is None else now - t_force,
            "gripper": None if t_grip_oldest is None else now - t_grip_oldest,
        }

    def push_proprio_from_anchor(self) -> None:
        """Append the CURRENT anchor rotation + latest gripper fractions to the proprio
        rings. In the last-target obs contract, the state at frame t is the last
        COMMANDED target (the anchor we hold/command), not a robot readback."""
        with self._lock:
            self._buf_rot6d.append(self._anchor_rot6d.copy())
            self._buf_grip.append(self._grip_frac_vec())
            for b in (self._buf_rot6d, self._buf_grip):
                if len(b) > self._To:
                    b.pop(0)

    def _grip_frac_vec(self) -> np.ndarray:
        """Per-role gripper fraction in role order (0 for non-gripper roles)."""
        return np.asarray(
            [self._grip_frac.get(role, 0.0) for role in self.roles], dtype=np.float64
        )

    # --- observation assembly (mirrors HLDataset.__getitem__ model-facing keys) ---
    def snapshot_obs(self) -> tuple[ObsSnapshot, np.ndarray, np.ndarray]:
        """Assemble the CURRENT observation, UN-normalized, + the anchor to decode against.

        Returns ``(obs, anchor_pos, anchor_rot6d)``. This is the whole of
        :meth:`build_batch` except the final normalize/collate step, split out so the deploy
        RECORDER can store exactly the observation the model is about to be fed — the same
        arrays, from the same atomic ring snapshot, not a second reconstruction that could
        differ by a tick. :func:`batch_from_obs` finishes the job.

        The obs mirrors ``HLDataset.__getitem__``:
          * state (To,state_dim): per keypoint ``[wrt_start_rot6d 6][gripper 1?]``, where
            wrt_start = R_start^T R_now for each history frame's rotation.
          * force_history (force_hist_len, 6*n_ft) raw sensor-frame + a validity flag.
          * images (To,3,224,224), oldest-first (the last is t0).
        """
        t_snapshot = time.monotonic()
        with self._lock:
            rot6d_hist = np.stack(self._buf_rot6d)  # (To, n_roles, 6)
            grip_hist = np.stack(self._buf_grip)  # (To, n_roles)
            imgs = [list(ring) for ring in self._buf_img]
            raw_force = list(self._buf_force)  # [(t, (ch,) or None)], oldest->newest
            anchor_pos = self._anchor_pos.copy()
            anchor_rot6d = self._anchor_rot6d.copy()
            start_rot6d = self._start_rot6d.copy()
            t_grip = dict(self._t_grip)  # in the SAME atomic snapshot as grip_hist

        # Bin the raw sensor-wrench ring into FORCE_HIST_LEN causal anti-aliased windows
        # tiling FORCE_HIST_SPAN_S ending at t0 (each bin = mean of the raw samples whose
        # TIMESTAMPS fall in it), matching the training converter's
        # resample_wrench_uniform_past. Validity comes back from the same pass, because it is
        # a property of the binned window (every bin measured, no dropout inside the span) —
        # deriving it from the ring separately is what let a 3x-too-wide window read "valid".
        force_win, fh_valid = self._bin_force_history(raw_force)  # (FORCE_HIST_LEN, ch), bool

        To = self._To
        state_slots = self.layout.state_slots()
        state = np.zeros((To, self.layout.state_dim), dtype=np.float64)
        state_valid = np.ones((To, self.layout.state_dim), dtype=bool)
        for ri, role in enumerate(self.roles):
            entry = state_slots[role]
            wrt_sl = entry["wrt_start_rot6d"]
            for t in range(To):
                state[t, wrt_sl] = rot6d_relative(rot6d_hist[t, ri], start_rot6d[ri])
            if "gripper" in entry:
                gcol = entry["gripper"]
                for t in range(To):
                    state[t, gcol] = grip_hist[t, ri]
                # UNMEASURED GRIPPER -> state_valid False, exactly as the training converter
                # flags a dead encoder (HLDataset marks the slot invalid rather than trusting a
                # fabricated width, and the model ZEROES invalid entries after normalization).
                # Deploy had no way to say this: `state_valid` was unconditionally all-True, so
                # a missing :8784 bridge fed 0.0 m -- about -2.6 normalized, two training spans
                # below q01 -- through as a real "jaw fully closed" reading, and a bridge that
                # died mid-run pinned the last width there forever. 0.0 is a LEGAL width, which
                # is precisely why its absence cannot be inferred from the value.
                t_g = t_grip.get(role)
                if t_g is None or (t_snapshot - t_g) > self.grip_max_age_s:
                    state_valid[:, gcol] = False

        # Images: oldest-first, edge-clamp the unfilled (cold-start) slots to the newest
        # real frame (mid-grey if none), matching the dataset's start-pad behavior.
        image = self._image_stack(imgs)  # (n_cam*To,3,224,224), camera-major

        obs = ObsSnapshot(
            images=image,
            state=state,
            state_valid=state_valid,
            force_history=force_win.astype(np.float64),
            force_history_valid=bool(fh_valid),
        )
        return obs, anchor_pos, anchor_rot6d

    def build_batch(self) -> dict:
        """The device-ready, normalized batch ``model.sample_actions`` consumes, plus the
        per-role anchor the caller must un-relativize the resulting chunk against.

        Returns ``(batch, anchor_pos, anchor_rot6d)``. Thin wrapper over
        :meth:`snapshot_obs` + :func:`batch_from_obs` so the recorded observation and the fed
        observation are the same object by construction.
        """
        obs, anchor_pos, anchor_rot6d = self.snapshot_obs()
        return batch_from_obs(self.m, obs), anchor_pos, anchor_rot6d

    def snapshot_display_images(self) -> list[np.ndarray]:
        """Return the CURRENT image stack as display-ready ``(224,224,3)`` uint8 RGB — the
        EXACT frames ``build_batch`` feeds the model, denormalized back from the stored
        ``(x/255 - mean)/std`` [-1,1] tensors. Used by the viser "conditioning images" panel
        so the operator sees what the model saw.

        ``n_cam*To`` entries in the model's own CAMERA-MAJOR order: camera 0's history
        oldest-first, then camera 1's. The panel labels them from
        :attr:`LoadedHLModel.image_keys`, so a swapped pair of bridges is visible rather
        than inferred."""
        with self._lock:
            imgs = [list(ring) for ring in self._buf_img]
        stack = self._image_stack(imgs)  # (n_cam*To,3,224,224) normalized, edge-clamped
        return [denormalize_image_hwc(chw, self.m.image_mean, self.m.image_std) for chw in stack]

    def _image_stack(self, rings: list[list[np.ndarray | None]]) -> np.ndarray:
        """``(n_cam*To, 3, 224, 224)`` float32, CAMERA-MAJOR then oldest-first.

        The order is the model's, not a convenience: ``HLDataset.__getitem__`` builds
        ``[self._load_image(data, key, hf) for key in self.image_keys for hf in hist_frames]``,
        i.e. all of camera 0's history, then all of camera 1's. Interleaving instead would
        hand a 2-camera checkpoint the right SHAPE with the views swapped into each other's
        encoder groups — shape-correct and silent, which is why it is spelled out here.

        Within one camera: cold/unfilled slots edge-clamp to that camera's own oldest real
        frame; a camera with no frame at all is mid-grey (normalized). Mirrors HLDataset's
        start-pad, per camera, so one late bridge does not grey out the other's history.
        """
        return np.concatenate([self._one_camera_stack(ring) for ring in rings], axis=0)

    def _one_camera_stack(self, imgs: list[np.ndarray | None]) -> np.ndarray:
        """(To,3,224,224) float32, oldest-first, for ONE camera's ring."""
        To = self._To
        real = [im for im in imgs if im is not None]
        if real:
            oldest = real[0]  # leading/missing slots edge-clamp to the oldest real frame
        else:
            grey = np.full((224, 224, 3), 128, dtype=np.float32)
            grey = np.transpose((grey / 255.0 - self.m.image_mean) / self.m.image_std, (2, 0, 1))
            oldest = grey.astype(np.float32)
        out = np.empty((To, 3, 224, 224), dtype=np.float32)
        # Right-align: the newest real frame lands in the last slot (== t0).
        window = imgs[-To:]
        window = [None] * (To - len(window)) + window
        # Left-pad missing leading slots with the oldest real frame; hold gaps at oldest.
        for i, im in enumerate(window):
            out[i] = im if im is not None else oldest
        # If the last slot is empty (no frames yet), the whole window is oldest/grey.
        return out

    # --- temporal ensembling (ACT only) ----------------------------------
    def _ensembler(self, role: str):
        """Lazily build (and cache) this role's :class:`TemporalEnsembler`, or None.

        Only for a checkpoint whose config asked for it (ACT's ``temporal_ensemble``); every
        other algorithm returns None and this whole path is inert. Built lazily so a
        flow/diffusion or replay-only planner never imports the ACT subpackage.
        """
        cfg = self.m.config
        if not getattr(cfg, "temporal_ensemble", False):
            return None
        cache = getattr(self, "_ens_cache", None)
        if cache is None:
            cache = self._ens_cache = {}
        ens = cache.get(role)
        if ens is None:
            from flore.s1_policy.model.act import TemporalEnsembler

            ens = cache[role] = TemporalEnsembler(
                int(cfg.action_horizon), k=float(getattr(cfg, "temporal_ensemble_k", 0.01)))
        return ens

    def reset_temporal_ensemble(self) -> None:
        """Drop every stored chunk prediction. Call when the INTENT is abandoned.

        That means: a fresh episode, ``Reset start pose``, or a ``Stop`` — anything after
        which the previously-predicted trajectory is no longer what we want to execute.

        NOT on the ordinary per-Predict clutch re-anchor. Blending happens in ABSOLUTE
        space, so a prediction made before the anchor moved is still expressed in the same
        world frame and stays perfectly valid; and "blend predictions made from earlier
        observations" is precisely what temporal ensembling IS. Resetting on every re-anchor
        would leave the ensembler permanently at one survivor and silently disable the
        feature while appearing to run it. (This is the payoff of blending absolutely rather
        than in the relative space upstream uses — the anchor is free to move.)"""
        for ens in (getattr(self, "_ens_cache", None) or {}).values():
            ens.reset()

    @property
    def temporal_ensemble_enabled(self) -> bool:
        """True when the loaded checkpoint asked for temporal ensembling."""
        return bool(getattr(self.m.config, "temporal_ensemble", False))

    # --- one inference + per-role un-relativize (NO anchor advance) ------
    def infer_chunk(
        self, *, seed: int, force_enabled: bool, gate_n: float, num_steps: int | None = None,
        capture: dict | None = None,
    ) -> "ChunkResult":
        """Run ONE model inference and decode it to absolute per-role poses + a gated
        force command. Does NOT advance the anchor (that is :meth:`commit_anchor`), so a
        preview is repeatable and never moves the robot.

        ``capture`` — an optional dict this fills in place with the exact inputs and raw
        output of THIS inference (``obs`` :class:`ObsSnapshot`, ``anchor_pos``,
        ``anchor_rot6d``, ``prediction``, ``seed``, ``num_steps``). The deploy recorder uses
        it to log what the policy actually saw and emitted without re-running the sampler or
        re-reading the rings a tick later. Purely an out-parameter: nothing in the inference
        path reads it, so leaving it None is byte-identical to before.
        """
        if self.m.model is None:
            # A replay-only planner (see replay_only_model) has NO policy. Refuse here rather
            # than let `None.sample_actions` surface as an opaque AttributeError from inside
            # the batch build — the caller reached the model path in a process that never
            # loaded one, which is a wiring bug, not a runtime condition.
            raise RuntimeError(
                "this LivePlanner was built by replay_only_model() and has no policy — "
                "only recorded-episode replay is available. Launch the planner with "
                "--algo/--checkpoint to run the model."
            )
        obs, anchor_pos, anchor_rot6d = self.snapshot_obs()
        batch = batch_from_obs(self.m, obs)
        prediction = sample_chunk(self.m, batch, seed=seed, num_steps=num_steps)
        if capture is not None:
            capture.update(
                obs=obs, anchor_pos=anchor_pos, anchor_rot6d=anchor_rot6d,
                prediction=prediction, seed=int(seed),
                num_steps=(self.m.num_steps if num_steps is None else int(num_steps)),
            )
        return decode_chunk(
            prediction,
            self.layout,
            anchor_pos,
            anchor_rot6d,
            force_enabled=force_enabled,
            gate_n=gate_n,
        )

    def infer_step_ensembled(
        self, *, seed: int, force_enabled: bool, gate_n: float, role: str = "left",
        capture: dict | None = None,
    ) -> dict:
        """Re-query the policy and return ONE temporally-ensembled step for ``role``.

        This is the per-step cadence upstream ACT uses with ``--temporal_agg``
        (``imitate_episodes.py:191-192`` sets ``query_frequency = 1``): every wall-step gets
        a fresh chunk from the CURRENT observation, and the value actually executed is the
        exponentially-weighted blend of every still-live prediction for this instant. A
        chunk boundary is then a smooth handover rather than a jump, and one badly-sampled
        chunk is outvoted by its neighbours.

        Blending happens in ABSOLUTE space, after :func:`decode_chunk` has un-relativized
        each chunk against the anchor that was live when it was predicted. That is
        load-bearing rather than incidental: FLoRe's chunks are ANCHOR-RELATIVE, so raw
        blending would mix frames — up to 82 cm of error at the H=50 window's oldest
        survivor at 0.5 m/s. See :mod:`flore.s1_policy.model.act.temporal_ensemble`.

        Returns ``{"position": (3,), "rot6d": (6,), "gripper_m": float|None,
        "wrench": (6,)|None, "n_blended": int}`` — the wrench is the UNGATED prediction
        blended, so the caller applies the live gate exactly as it does per-step for a
        non-ensembled chunk.

        ``capture`` is forwarded to :meth:`infer_chunk` — it collects THIS step's own
        observation and raw chunk (each ensembled step is a fresh inference, so each has its
        own), for the deploy recorder.

        Raises ``RuntimeError`` when the loaded checkpoint did not ask for ensembling, so a
        caller cannot silently get single-chunk behavior while believing otherwise.
        """
        ens = self._ensembler(role)
        if ens is None:
            raise RuntimeError(
                f"this checkpoint (algo={self.m.algorithm!r}) does not use temporal "
                f"ensembling; call infer_chunk() and drip its steps instead.")
        result = self.infer_chunk(seed=seed, force_enabled=force_enabled, gate_n=gate_n,
                                  capture=capture)
        rc = result.role(role)
        return ens.add(rc.position, rc.rot6d, rc.gripper_m, rc.wrench_pred)


@dataclass(frozen=True)
class RoleChunk:
    """One role's decoded absolute chunk over the action horizon."""

    role: str
    position: np.ndarray  # (H,3) absolute
    rot6d: np.ndarray  # (H,6) absolute
    gripper_m: np.ndarray | None  # (H,) metres, or None (no gripper on this role)
    wrench_pred: np.ndarray | None  # (H,6) physical N/N·m (ungated, for viz), or None
    wrench_cmd: np.ndarray | None  # (H,6) GATED command to publish, or None


@dataclass(frozen=True)
class ChunkResult:
    """A decoded chunk for every role + the chunk-end anchor to commit after execution."""

    roles: tuple[RoleChunk, ...]
    action_times_s: np.ndarray  # (H,)
    end_anchor_pos: np.ndarray  # (n_roles,3)
    end_anchor_rot6d: np.ndarray  # (n_roles,6)

    def role(self, name: str) -> RoleChunk:
        for rc in self.roles:
            if rc.role == name:
                return rc
        raise KeyError(f"chunk has no role {name!r}")


def decode_chunk(
    prediction: np.ndarray,
    layout: ActionLayout,
    anchor_pos: np.ndarray,
    anchor_rot6d: np.ndarray,
    *,
    force_enabled: bool,
    gate_n: float,
) -> ChunkResult:
    """Decode a normalized-then-unnormalized ``(H, action_dim)`` chunk into absolute
    per-role poses + a gated force command. Pure function (no model, no locks) so it is
    directly unit-testable. Mirrors ``build_rollout_result`` but tracks the gated F^cmd
    and returns the chunk-end anchor for :meth:`LivePlanner.commit_anchor`.

    The position delta is in the anchor's BODY frame (``p_abs = R_anchor @ p_rel + p_anchor``) —
    the frame ``HLDataset.__getitem__`` authors the training target in, so this is the only frame
    a policy's output may be decoded in. Recorded-episode replay goes through this SAME function
    for exactly that reason; see
    :func:`flore.s1_policy.serving.hl_replay._body_vs_world_transport` for why a world-frame
    variant was measured and rejected.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    H = pred.shape[0]
    roles_out: list[RoleChunk] = []
    end_pos = np.zeros((len(layout.roles), 3), dtype=np.float64)
    end_rot6d = np.zeros((len(layout.roles), 6), dtype=np.float64)
    for ri, role in enumerate(layout.roles):
        slots = layout.slots(role)
        abs_pos, abs_rot6d = delta_ee_to_absolute(
            pred[:, slots.pos], pred[:, slots.rot6d], anchor_pos[ri], anchor_rot6d[ri]
        )
        end_pos[ri] = abs_pos[-1]
        end_rot6d[ri] = abs_rot6d[-1]

        grip_m = None
        if slots.gripper is not None and role in layout.gripper_roles:
            grip_m = pred[:, slots.gripper] * GRIPPER_WIDTH_REF_M  # (H,)

        wrench_pred = wrench_cmd = None
        if slots.force is not None and role in layout.force_roles:
            wrench_pred = pred[:, slots.force].copy()  # (H,6) ungated physical, for viz
            wrench_cmd = gate_force_command(
                wrench_pred, force_enabled=force_enabled, gate_n=gate_n
            )

        roles_out.append(
            RoleChunk(
                role=role,
                position=abs_pos,
                rot6d=abs_rot6d,
                gripper_m=grip_m,
                wrench_pred=wrench_pred,
                wrench_cmd=wrench_cmd,
            )
        )
    return ChunkResult(
        roles=tuple(roles_out),
        action_times_s=np.arange(H, dtype=np.float64) * ACTION_DT,
        end_anchor_pos=end_pos,
        end_anchor_rot6d=end_rot6d,
    )


# rot6d of the identity rotation (first two rows of I3).
_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

__all__ = [
    "gate_force_command",
    "LoadedHLModel",
    "load_hl_model",
    "replay_only_model",
    "LivePlanner",
    "ObsSnapshot",
    "RoleChunk",
    "ChunkResult",
    "batch_from_obs",
    "decode_chunk",
    "denormalize_image_hwc",
    "normalize_image_chw",
    "resize_center_crop_224",
    "sample_chunk",
    "ACTION_HORIZON",
    "EXEC_STEP_LADDER",
    "exec_step_choices",
    "snap_exec_steps",
]
