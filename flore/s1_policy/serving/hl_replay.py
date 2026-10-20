"""Recorded-episode action source for the online HL-v2 planner — "replay the demo".

The online planner (:mod:`scripts.hl_serving`) normally gets its action chunk from the
policy. This module provides the SAME SHAPE of chunk read straight out of a converted
UMI ``.zarr.zip`` instead, so the operator can drive the rig with a RECORDED human
demonstration (one of the 94 clean94 episodes) through the identical
predict -> preview -> execute -> commit pipeline. That makes "what the demo did" and
"what the policy predicts" directly comparable ON THE RIG: same wire, same execute-N-of-S
selector, same force gate, same anchor commit.

THE LOAD-BEARING IDEA: return the chunk in the model's OWN output form — a
``(H, action_dim)`` vector of ANCHOR-RELATIVE deltas — not absolute recorded poses. Then
:func:`flore.s1_policy.serving.hl_online.decode_chunk` (the exact function the model path
uses) re-absolutizes it against the planner's LIVE anchor. Two consequences, both wanted:

  * **No new frame convention.** The recorded poses are absolute LIGHTHOUSE-WORLD puck
    poses; the rig servos in PELVIS frame. Nothing here ever tries to align those two
    worlds — the chunk is relative, so the world frame CANCELS exactly (a global world
    rotation conjugates both the anchor and the delta and drops out; the same reason the
    runbook notes a session's ``world_align`` is harmless for anchor-relative actions).
    The replayed motion is therefore reproduced from WHEREVER THE ARM ACTUALLY IS — the
    clutch semantic the planner already implements for the model.
  * **Faithful to training.** ``absolute_to_delta_ee`` against the frame's own recorded
    anchor is byte-identical to what ``HLDataset.__getitem__`` builds as ``action_chunk``
    (DECISION-T0: step 0 of ``hl_*_action_pose[f]`` IS the frame's own pose, so step 0
    relativizes to an exact zero delta). So this replays the policy's TARGET, i.e. the
    thing the model is trained to emit — not a re-derived approximation.

Reads the PRE-BAKED ``hl_*`` action arrays (already on the uniform 16 Hz ACTION grid with
their own validity masks), never the camera-grid poses — re-resampling here would silently
introduce a second, divergent interpolation of the same data.

Robot-free and viser-free on purpose: everything load-bearing (episode bounds, chunk
relativization, validity truncation, time-based cursor advance) is unit-testable without
a rig. The wire/lever/IK half stays in ``scripts/hl_serving.py``, exactly as the model
path keeps ``WireMapper`` out of :mod:`flore.s1_policy.serving.hl_online`.

THE SAME SOURCE FEEDS THE FIVE-POINT PAGE (``scripts/hl_serving_5pt.py``, 2026-09-19). There the
recorded chunk is decoded against the page's running anchor by the identical ``decode_chunk`` and
then flows through ``FivePointDirectWire`` (held slots filled from the ready pose), the clutch,
the live force gate (the RECORDED wrench becomes ``F^cmd`` through ``hl_5pt.force_command``, i.e.
``-F_meas`` gated on magnitude), the jaw state machine, STOP / ⏹ / the ceiling, and the ⏺ run
recorder — ONE ``if`` at the top of the page's ``_predict``. What this module adds for it:
per-step force / gripper validity on every chunk (:attr:`RecordedChunk.force_step_valid`), so an
unmeasured (pre-tare) wrench is never commanded and a dead-encoder jaw width (zero-filled = CLOSED)
is never sent (:func:`apply_step_validity`); a second camera view (:meth:`RecordedEpisodeSource.image_at`
``key=``); and the operator-facing helpers :func:`episode_options`, :func:`speed_report` and
:func:`transport_rotation_deg`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from flore.s1_policy.data.layout import (
    ACTION_DT,
    ACTION_HORIZON,
    ACTION_HZ,
    ACTION_SPAN_S,
    FORCE_HIST_SPAN_S,
    ActionLayout,
)
from flore.s1_policy.data.transforms import (
    GRIPPER_WIDTH_REF_M,
    absolute_to_delta_ee,
    gripper_width_to_norm,
    rotvec_to_rot6d,
)
from flore.utils.rot6d import rot6d_to_rotmat

#: The stored camera view every HL-v2 file carries; a dual-rig file adds ``camera1_rgb``.
DEFAULT_IMAGE_KEY = "camera0_rgb"

_IMAGE_CODECS_REGISTERED = False


def _body_vs_world_transport() -> None:
    """WHY THE REPLAYED DEMO LOOKS REORIENTED ON THE RIG, AND WHY THAT IS NOT A BUG.

    Documentation anchor (never called). The operator's recurring report is that the green
    recorded path in ``run_hl_replay_data.sh`` does not match the trajectory they saw in
    ``collect_data.sh`` / ``run_replay_viz.sh`` / the zarr replay. It does not, and the reason
    is physical rather than a coding error.

    ``chunk_at`` / ``episode_path_relative`` relativize into the REFERENCE'S OWN BODY frame
    (``p_rel = R_ref^T (p − p_ref)``), and ``hl_online.decode_chunk`` re-absolutizes against the
    robot's measured wrist. The net map on the demo is the rigid rotation
    ``R_transport = R_wrist_robot @ R_wrist_handheld^T``. Measured on clean94: **106.9° mean
    (98.5–119.0)**, det = +1. It is exactly the angle between how the human HELD the tool
    (gripper along −Y, level) and how the G1 holds it at its deploy ready pose (gripper forward,
    44° nose-down, 47.08° off identity) — it is not introduced by the code, and no change of
    transport can remove it while step 0 stays on the measured pose. Zeroing it would require
    a demo-compatible READY POSE (wrist +x ≈ (−0.023, −0.999, −0.033)), i.e. a hardware change.

    The obvious "fix" — a WORLD-frame delta (``p_rel = p − p_ref``, matching
    ``gripper_only_view.home_anchor``) — was implemented, measured, and REVERTED. All numbers on
    the real 94-episode clean94 set with the deploy anchor:

    * **Tool geometry destroyed.** Per-sample attack angle (path tangent vs the commanded wrist
      +x — "is the tool pointing where it is moving"): body reproduces the demo to **2e-14°**;
      world is off by **50.0° mean, 109.6° p95, 145.7° max**, with 65.8 % of 39 380 samples
      beyond 30°. For a wiping task that tool-relative geometry IS the task. (An earlier
      per-episode-MEAN comparison suggested only ~5.7° — that statistic cancels the error and
      is the wrong one; compare per sample.)
    * **Reachability/safety.** World drives 4/94 episodes (27, 28, 30, 31) to a commanded
      pelvis-frame X of **+0.026 m** — into the torso; body's minimum over all 94 is +0.100 m.
    * **Lighthouse dependence.** Body makes the recording's world frame cancel EXACTLY (a
      (37, 5, −9)° session re-alignment changes the command by 1.3e-08 m = float32 storage
      noise); world changes it by **0.132 m**, so clean94 (``world_align`` [0,0,180]) and
      clean99 ([1.44, 2.14, −170.86]) would command different motion for the same demo.
    * **Training parity.** Body IS ``HLDataset.__getitem__``'s ``action_chunk`` (pinned to
      1e-6), which is what makes "replay the demo" comparable to "run the policy". World
      differs by up to 0.033 m in the position column.
    * **Step 0 cannot arbitrate.** Both land exactly on the measured pose (0 m, 1.7e-06°), so
      neither is safer for ABSOLUTE-mode arming.

    Note the collection-viz chains are NOT a precedent for a wire that moves an arm: they are
    gripper-only renders against a fixed cosmetic home with no robot command, and
    ``display.py`` already documents that they do not preserve the path-vs-gripper angle either
    (measured 49.8° mean distortion — worse than the world variant).
    """


def _register_image_codecs() -> None:
    """Register the JpegXl numcodecs codec used for ``camera0_rgb``. Idempotent.

    Mirrors :func:`flore.s1_policy.data.hl_dataset._register_image_codecs`; must run before
    the first array-metadata load or the read raises ``codec not available``. Kept local so
    this module never imports the training dataset (which would drag in torch)."""
    global _IMAGE_CODECS_REGISTERED
    if _IMAGE_CODECS_REGISTERED:
        return
    from imagecodecs.numcodecs import register_codecs

    register_codecs()
    _IMAGE_CODECS_REGISTERED = True


@dataclass(frozen=True)
class RecordedChunk:
    """One recorded action chunk, in the model's own ``(H, action_dim)`` relative form.

    ``actions`` is what a PERFECT policy would have emitted at this frame: per role,
    ``[pos3 | rot6d6 | gripper1 | force6]`` with pose ANCHOR-RELATIVE (relative to the
    recorded frame's own pose), gripper as a fraction of ``GRIPPER_WIDTH_REF_M``, and force
    in raw physical sensor units — the same conventions
    :func:`flore.s1_policy.serving.hl_online.decode_chunk` expects.

    ``n_valid`` is how many LEADING steps are real recorded future (the rest of the
    episode is shorter than the horizon near its end, where the converter pads by holding
    the last pose). ``actions`` is already TRUNCATED to those steps, so ``actions.shape[0]
    == n_valid`` and an executor can never drip a padded hold as if it were motion.
    """

    episode: int
    local_frame: int  # the episode-local frame this chunk was read from (cursor identity)
    actions: np.ndarray  # (n_valid, action_dim) anchor-relative
    n_valid: int
    time_s: float  # this frame's time within the episode (relative camera clock)
    gripper_valid: bool
    force_valid: bool
    #: PER-STEP converter validity over the ``n_valid`` steps, ``role -> (n_valid,) bool``, for the
    #: roles that carry the channel (empty dict otherwise). ``gripper_valid`` / ``force_valid`` above
    #: are step 0's bit for the whole chunk; these are what :func:`apply_step_validity` blanks the
    #: DECODED rows with, so an unmeasured wrench is never put on the F^cmd wire and a zero-filled
    #: jaw width (a dead encoder — 0 m is CLOSED) is never commanded. Defaults keep every existing
    #: constructor call valid.
    force_step_valid: dict = dataclasses.field(default_factory=dict)
    gripper_step_valid: dict = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class EpisodePlan:
    """The WHOLE episode as ONE open-loop action chunk, anchor-relative to its start frame.

    The open-loop analogue of :class:`RecordedChunk`. Where :meth:`RecordedEpisodeSource.chunk_at`
    returns one ``ACTION_HORIZON``-long future read from a single frame — to be RE-ANCHORED
    against the live wrist every chunk — this returns the ENTIRE camera-grid trajectory from
    ``start_frame`` to the episode end (or the first glitch hole) as a single relative chunk,
    to be decoded ONCE against one seeded anchor and dripped straight through.

    WHY THIS EXISTS (the accumulation the per-chunk path has and this one does not). The
    sequential replay (:func:`scripts.hl_serving._replay_play_sequential`) re-syncs the decode
    anchor to the robot's MEASURED EE before every chunk, so each ~50-step arc is re-based on
    where the arm actually is, and the servo's steady-state lag is folded into the anchor and
    carried into the next chunk — a receding-horizon clutch, not one placement of the demo.
    This plan instead pins the demonstration to the arm's pose ONCE (a single clutch at the
    start) and commands the rest without correction, so the executed-vs-commanded error is a
    clean OPEN-LOOP number rather than one confounded by a reference that keeps being redrawn
    from the arm's own lagging position.

    ``actions`` is ``(n_frames, action_dim)`` in the SAME anchor-relative form as
    :attr:`RecordedChunk.actions` (per role ``[pos3|rot6d6|gripper1|force6]``, pose relative to
    ``start_frame``'s own recorded pose, gripper a fraction of ``GRIPPER_WIDTH_REF_M``, force
    raw physical), so it flows through the IDENTICAL
    :func:`flore.s1_policy.serving.hl_online.decode_chunk` the model and per-chunk replay use.

    THE POSE COLUMNS ARE THE CAMERA-GRID DEMONSTRATION, not the per-frame action-grid future.
    They are exactly :meth:`RecordedEpisodeSource.episode_path_relative` (``robot0_eef_pos`` /
    ``_rot_axis_angle`` relativized to ``start_frame``) — the same curve the green line draws
    and the trace reference is built from. That is deliberate and load-bearing for the error
    measurement: the per-frame ``hl_*_action_pose[f]`` is the converter's UNIFORM 30 Hz
    RESAMPLE of the future, which drifts from the raw camera poses by up to ~23 mm over a 50-step
    window on this data (the camera timestamps are not perfectly uniform). Commanding the camera
    grid means the open-loop command, the drawn path, and the reference the sheet differences
    against are ONE curve, so the reported error is pure servo/IK tracking error and not a
    resampling artefact. (Only step 0 of the two grids is identical — ``hl_*_action_pose[f, 0]``
    IS ``robot0_eef_pos[f]`` — which is why the per-chunk path, being one chunk, stays close to
    its reference while a whole-episode action-grid concatenation would not.)

    ``truncated`` is True when a tracker-glitch/masked hole cut the plan short of the episode
    end (on the 2026.08.10 set, 3 of 140 episodes; the rest span the whole remainder). An
    open-loop drip CANNOT seek over such a hole the way the sequential loop does, so it stops at
    the last clean frame rather than command a teleport across the excursion.
    """

    episode: int
    start_frame: int
    actions: np.ndarray  # (n_frames, action_dim) anchor-relative
    n_frames: int
    times_s: np.ndarray  # (n_frames,) episode-relative camera clock for the commanded frames
    truncated: bool
    gripper_valid: bool  # step-0 gripper validity AT start_frame (the first commanded step)
    force_valid: bool  # step-0 force validity AT start_frame
    #: Per commanded FRAME (each frame's own step-0 bit), ``role -> (n_frames,) bool`` — the same
    #: contract as :attr:`RecordedChunk.force_step_valid`, for :func:`apply_step_validity`.
    force_step_valid: dict = dataclasses.field(default_factory=dict)
    gripper_step_valid: dict = dataclasses.field(default_factory=dict)


class RecordedEpisodeSource:
    """Read-only view over a converted UMI ``.zarr.zip``'s pre-baked HL action chunks.

    Opens the archive lazily and reads ONE frame's chunk at a time (an
    ``(T, 16, 6)`` float32 pose array is ~15 MB per full column, so whole-column reads are
    avoided exactly as the training dataset avoids them).

    Parameters
    ----------
    zarr_path : the ``.zarr.zip`` the checkpoint was trained on (its ``hl_data.zarr_paths``).
    layout : the checkpoint's :class:`ActionLayout`. Every role it declares must be present
        in the file as ``hl_<role>_action_*``; a missing role is a hard error rather than a
        silently smaller replay.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        layout: ActionLayout,
        *,
        glitch_pre_s: float = FORCE_HIST_SPAN_S,
        glitch_post_s: float = ACTION_SPAN_S,
    ):
        import zarr

        _register_image_codecs()
        self.path = str(zarr_path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"replay zarr not found: {self.path}")
        self.layout = layout
        if layout.legacy:
            raise ValueError("recorded replay does not support the legacy32 layout")
        # Glitch-window span, defaulted to the TRAINING filter's values so the frames replay
        # offers are the same anchors the policy was trained on. Lazily-built cache.
        self.glitch_pre_s = float(glitch_pre_s)
        self.glitch_post_s = float(glitch_post_s)
        self._glitch_keep: np.ndarray | None = None
        #: ``[n_frames]`` bool: step 0 of every driven role is a real target — the first half of
        #: :meth:`frame_is_usable`, read ONCE off the same whole-column pass :meth:`_glitch_keep_mask`
        #: already makes (a per-frame read per role per call took ~1364 episodes x 3 reads on the
        #: 2026-09-17 wipe_board_all file to label the episode picker).
        self._step0_valid: np.ndarray | None = None

        self._root = zarr.open(self.path, "r")
        self.attrs = dict(self._root.attrs)
        self._data = self._root["data"]

        ends = np.asarray(self._root["meta/episode_ends"][:], dtype=np.int64).reshape(-1)
        if ends.size == 0 or np.any(np.diff(ends) <= 0) or int(ends[0]) <= 0:
            raise ValueError(f"{self.path}: meta/episode_ends is not a usable partition")
        self.ends = ends
        self.starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        self.n_episodes = int(ends.size)
        self.n_frames = int(ends[-1])

        # Per-role array names. The pre-baked hl_* arrays are ROLE-named (not robot0/robot1),
        # so unlike the camera-grid keys they need no primary_role indirection.
        self._keys: dict[str, dict[str, str]] = {}
        available = set(self._data.keys())
        for role in layout.roles:
            entry = {
                "pose": f"hl_{role}_action_pose",
                "valid": f"hl_{role}_action_valid",
            }
            if role in layout.gripper_roles:
                entry["gripper"] = f"hl_{role}_action_gripper"
                entry["gripper_valid"] = f"hl_{role}_action_gripper_valid"
            if role in layout.force_roles:
                entry["force"] = f"hl_{role}_action_force"
                entry["force_valid"] = f"hl_{role}_action_force_valid"
            missing = sorted(k for k in entry.values() if k not in available)
            if missing:
                raise ValueError(
                    f"{self.path}: missing HL action arrays {missing} required by layout "
                    f"{layout.mode!r}. Re-convert the session with --hl-windows, or point "
                    "--replay-zarr at the dataset the checkpoint was trained on."
                )
            self._keys[role] = entry

        self._check_contract()

        # Camera-grid absolute poses, for drawing the recorded path (role -> (pos, rot) keys).
        primary = str(self.attrs.get("robot0_primary_role", "left"))
        self._cam_keys = {
            role: keys[:2] for role, keys in layout.zarr_keys(primary).items()
        }
        # Camera-grid gripper width (role -> key or None) — the third element of the same
        # triple. Kept separate from the pose pair because the head role has no gripper, and
        # because only the trace/QC path reads it.
        self._cam_grip_keys = {
            role: keys[2] for role, keys in layout.zarr_keys(primary).items()
        }

        # episode_timestamps is RELATIVE per episode (DECISION-A: each restarts at 0), so a
        # per-episode slice is already an episode-local clock and never leaks a neighbour's.
        self._ts = np.asarray(self._data["episode_timestamps"][:], dtype=np.float64).reshape(-1)
        if self._ts.size != self.n_frames:
            raise ValueError(
                f"{self.path}: episode_timestamps has {self._ts.size} entries but "
                f"episode_ends declares {self.n_frames} frames"
            )

    def _check_contract(self) -> None:
        """Refuse a zarr whose provenance attrs disagree with what this module assumes.

        ``--replay-zarr`` takes a free-form path, and the datasets directory holds sessions
        that LOOK interchangeable but are not (e.g. clean99 has the pre-fix puck→TCP extrinsic
        and a different ``world_align``). The training loader gates all of this in
        ``HLDataset._check_file``; without the same gates here a mismatched file replays
        plausible-looking garbage. The checks mirror that method (same attrs, same tolerances)
        so a file accepted for replay is one the training pipeline would also accept.

        The load-bearing one is ``hl_action_first_step``: a pre-v2 file EXCLUDED t0 (its chunk
        started at t0+dt) with IDENTICAL horizon/span/hz attrs, so the one-dt shift is
        invisible in every other check AND in the values. Replaying one would put step 0 off
        the live anchor — an instantaneous commanded jump on the first dripped step of every
        chunk (up to ~30 mm on this data) — so it is a hard error, and an absent value is
        refused too rather than assumed."""
        attrs = self.attrs
        path = self.path
        if not attrs.get("hl_windows"):
            raise ValueError(
                f"{path}: no HL-v2 windows (attrs.hl_windows is falsy) — this file has no "
                "pre-baked action chunks to replay. Re-convert with --hands/--hl-windows."
            )
        # HORIZON is ``stored >= wanted``, not ``==`` — the same relaxation as
        # ``HLDataset._check_file``, and for the same reason: on a t0-inclusive grid at a
        # fixed dt a shorter chunk is a strict PREFIX of a longer one, so the first H of a
        # 50-step file ARE its own first H steps. Without this, serving an H=30 checkpoint
        # (FLORE_ACTION_HORIZON=30) could not replay the 50-step session it was trained on —
        # the replay panel would refuse the very file the checkpoint's own
        # ``hl_data.zarr_paths`` points at. ``chunk_at`` takes that prefix.
        #
        # The relation holds ONLY at an identical dt, so ``hl_action_span_s`` is no longer
        # compared to ``ACTION_SPAN_S`` (it scales with the STORED horizon, so a correct file
        # would now fail it); it is checked as span/horizon == ACTION_DT below, which pins the
        # rate from the file's own two numbers instead of trusting its advertised hz.
        stored_h = attrs.get("hl_action_horizon")
        if stored_h is None:
            raise ValueError(f"{path}: attrs.hl_action_horizon is required to verify the HL grid contract")
        try:
            stored_h_num = float(stored_h)
        except (TypeError, ValueError):
            stored_h_num = float("nan")
        if not np.isfinite(stored_h_num) or stored_h_num < float(ACTION_HORIZON):
            raise ValueError(
                f"{path}: attrs.hl_action_horizon={stored_h} but this build expects at least "
                f"{ACTION_HORIZON} action steps. A shorter stored chunk cannot be extended "
                f"(only truncated) — replay a session converted with --hl-action-horizon "
                f"{ACTION_HORIZON} or more, or lower FLORE_ACTION_HORIZON."
            )
        stored_span = attrs.get("hl_action_span_s")
        if stored_span is None:
            raise ValueError(f"{path}: attrs.hl_action_span_s is required to verify the HL grid contract")
        try:
            stored_dt = float(stored_span) / stored_h_num
        except (TypeError, ValueError, ZeroDivisionError):
            stored_dt = float("nan")
        if not np.isfinite(stored_dt) or abs(stored_dt - ACTION_DT) > 1e-9:
            raise ValueError(
                f"{path}: attrs.hl_action_span_s={stored_span} over "
                f"attrs.hl_action_horizon={stored_h} is a step dt of {stored_dt}, but this "
                f"build's ACTION_DT is {ACTION_DT}. The prefix relation that allows a shorter "
                f"horizon holds only at an IDENTICAL dt, and a mis-timed chunk would be dripped "
                f"against the wrong execution grid — re-convert with --hl-action-hz {ACTION_HZ:g}."
            )
        got_hz = attrs.get("hl_action_hz")
        if got_hz is None:
            raise ValueError(f"{path}: attrs.hl_action_hz is required to verify the HL grid contract")
        try:
            numeric_hz = float(got_hz)
        except (TypeError, ValueError):
            numeric_hz = float("nan")
        if not np.isfinite(numeric_hz) or abs(numeric_hz - float(ACTION_HZ)) > 1e-9:
            raise ValueError(
                f"{path}: attrs.hl_action_hz={got_hz} but this build expects {ACTION_HZ} Hz "
                "action grid. The replayed chunk would be mis-timed against the execution grid."
            )
        first_step = attrs.get("hl_action_first_step")
        if first_step != "t0":
            raise ValueError(
                f"{path}: attrs.hl_action_first_step={first_step!r}, but replay requires the "
                "t0-inclusive action grid ('t0') — it relativizes each chunk against its own "
                "step 0 on the strength of step 0 BEING the frame's pose. A pre-v2 file "
                "excluded t0 with identical horizon/span attrs, so the one-dt shift is "
                "invisible in the values and would place step 0 off the live anchor "
                "(an instantaneous commanded jump every chunk). Re-convert."
            )
        rig_open = attrs.get("gripper_width_open_m")
        if rig_open is not None and (
            not np.isfinite(float(rig_open)) or abs(float(rig_open) - GRIPPER_WIDTH_REF_M) > 1e-9
        ):
            raise ValueError(
                f"{path}: converted against a full-open gripper width of {rig_open} m, but "
                f"this build normalizes by GRIPPER_WIDTH_REF_M={GRIPPER_WIDTH_REF_M} m — every "
                f"replayed jaw command would be scaled by "
                f"{float(rig_open) / GRIPPER_WIDTH_REF_M:.4f}."
            )
        if attrs.get("hl_pose_frame") not in (None, "absolute_aligned_world"):
            raise ValueError(
                f"{path}: attrs.hl_pose_frame={attrs['hl_pose_frame']!r}; replay relativizes "
                "ABSOLUTE stored poses, so a pre-relativized file would be double-relativized."
            )
        if attrs.get("hl_force_frame") not in (None, "bota_sensor"):
            raise ValueError(
                f"{path}: attrs.hl_force_frame={attrs['hl_force_frame']!r}, but replay passes "
                "the force columns straight to the low level's F^cmd wire, which is only "
                "correct for the raw Bota sensor frame."
            )

    # -- episode geometry -------------------------------------------------
    def episode_lengths(self) -> list[int]:
        return [int(b - a) for a, b in zip(self.starts, self.ends, strict=True)]

    def frame_is_usable(self, episode: int, local_frame: int) -> bool:
        """Whether ``local_frame`` is safe to EXECUTE: step 0 is a real target AND the frame
        is not inside a tracker-glitch window.

        Two separate exclusions, both needed:

        * **Step 0 masked.** The converter masks step 0 of some frames, and such a frame has
          no executable action at all (:meth:`chunk_at` raises). On clean94, 8 of the 94
          episodes are masked at their FIRST frame and 5 more have masked frames mid-episode.
        * **Glitch window.** A masked frame protects the glitch's CENTRE but not its
          neighbours: the frames just after a tracker excursion are unmasked, yet their chunk
          still contains the excursion. Executing one commands a teleport — measured on
          clean94, up to a **153 mm step inside one 62.5 ms wire tick (2.45 m/s)**, ~3x the
          48 mm worst case over training-clean anchors, and deterministically reachable by
          pressing Play on episode 27. So this reuses the TRAINING filter's own exclusion
          (:meth:`~flore.s1_policy.data.hl_dataset.HLDataset._glitch_window_keep`, ported in
          :meth:`_glitch_keep_mask`) — the replayed frames are then exactly the anchors the
          policy was trained on, which is also what makes the demo-vs-policy comparison fair.
        """
        a, b = self.episode_bounds(episode)
        lf = int(local_frame)
        if not 0 <= lf < (b - a):
            return False
        # Both halves off the cached whole-column pass (`_glitch_keep_mask` fills `_step0_valid`
        # from the same `hl_<role>_action_valid[:, 0]` bits the per-frame read used to test).
        keep = self._glitch_keep_mask()
        assert self._step0_valid is not None
        return bool(self._step0_valid[a + lf]) and bool(keep[a + lf])

    def _glitch_keep_mask(self) -> np.ndarray:
        """``[n_frames]`` bool, False for frames inside a tracker-glitch window.

        A port of ``HLDataset._glitch_window_keep`` restricted to what this module needs (no
        torch, no Dataset). Like the original it is derived ENTIRELY from the converter's own
        ``hl_<role>_action_valid`` + ``robot0_pose_glitch``, re-thresholding nothing, so the
        replay exclusion can never disagree with the training one. Computed once and cached —
        it reads three whole columns, which is why it is not done per frame."""
        if self._glitch_keep is not None:
            return self._glitch_keep
        keep = np.ones(self.n_frames, dtype=bool)
        ts = self._ts
        # T0-INCLUSIVE grid (DECISION-T0): step k ends at t0 + k*dt and its bin spans
        # (t_k - dt, t_k]. Using the pre-v2 (k+1)*dt offset would shift every glitch
        # interval one dt late.
        step_end_off = ACTION_DT * np.arange(0, ACTION_HORIZON, dtype=np.float64)
        # Only POSE validity signals a corrupted pose; a masked force step is a sensor/tare
        # coverage issue and is handled by the force flags, not by excluding the frame.
        # ``[:, :ACTION_HORIZON]`` is REQUIRED, not tidiness: the stored width may exceed this
        # build's horizon, and `bad` below ANDs this against a (n_ep, ACTION_HORIZON) grid
        # derived from step_end_off — an un-truncated (n_ep, 50) column raises a broadcast
        # error against a 30-step grid.
        pose_valids = [
            np.asarray(self._data[self._keys[role]["valid"]][:], dtype=np.float32)[:, :ACTION_HORIZON] > 0.5
            for role in self.layout.roles
        ]
        # Step 0 real for EVERY driven role — the same bit `frame_is_usable` used to read per frame.
        step0 = np.ones(self.n_frames, dtype=bool)
        for pv in pose_valids:
            step0 &= pv[:, 0] if pv.shape[1] else np.zeros(self.n_frames, dtype=bool)
        self._step0_valid = step0
        glitch_frame = None
        if "robot0_pose_glitch" in self._data:
            glitch_frame = (
                np.asarray(self._data["robot0_pose_glitch"][:], dtype=np.float32).reshape(-1) > 0.5
            )
        for a, b in zip(self.starts, self.ends, strict=True):
            a, b = int(a), int(b)
            if b <= a:
                continue
            rt = ts[a:b]  # per-episode RELATIVE times (restart at ~0)
            ep_end = float(rt[-1])
            step_end = rt[:, None] + step_end_off[None, :]
            step_lo = step_end - ACTION_DT
            not_tail = step_end <= ep_end + 1e-9  # a step past ep_end is padding, not a glitch
            lo_list: list[np.ndarray] = []
            hi_list: list[np.ndarray] = []
            for pv in pose_valids:
                bad = (~pv[a:b]) & not_tail
                if bad.any():
                    lo_list.append(step_lo[bad])
                    hi_list.append(step_end[bad])
            if glitch_frame is not None:
                pg = glitch_frame[a:b]
                if pg.any():
                    lo_list.append(rt[pg])
                    hi_list.append(rt[pg])
            if not lo_list:
                continue
            bad_lo = np.concatenate(lo_list)
            bad_hi = np.concatenate(hi_list)
            win_lo = rt - self.glitch_pre_s
            win_hi = rt + self.glitch_post_s
            drop = (
                (bad_lo[None, :] <= win_hi[:, None]) & (bad_hi[None, :] >= win_lo[:, None])
            ).any(axis=1)
            keep[a:b] &= ~drop
        self._glitch_keep = keep
        return keep

    def next_usable_frame(self, episode: int, local_frame: int) -> int | None:
        """The first frame at/after ``local_frame`` that :meth:`chunk_at` accepts, else None.

        Lets a caller SKIP the converter's masked frames instead of failing on them: a seek
        lands on the nearest playable frame, and a sequential replay steps over a mid-episode
        hole rather than aborting the rest of the demonstration."""
        a, b = self.episode_bounds(episode)
        n = b - a
        for lf in range(max(int(local_frame), 0), n):
            if self.frame_is_usable(episode, lf):
                return lf
        return None

    def first_usable_frame(self, episode: int) -> int | None:
        """The first playable frame of ``episode`` (0 for a normal episode), else None."""
        return self.next_usable_frame(episode, 0)

    def first_usable_frames(self) -> list[int | None]:
        """:meth:`first_usable_frame` for every episode, in order — what an episode picker labels."""
        return [self.first_usable_frame(i) for i in range(self.n_episodes)]

    def episode_bounds(self, episode: int) -> tuple[int, int]:
        """``(start, end)`` absolute frame indices of ``episode``; ``end`` is exclusive."""
        ep = int(episode)
        if not 0 <= ep < self.n_episodes:
            raise IndexError(f"episode {ep} is outside [0, {self.n_episodes - 1}]")
        return int(self.starts[ep]), int(self.ends[ep])

    def episode_times(self, episode: int) -> np.ndarray:
        """The episode's relative camera clock, ``(N,)`` seconds starting at ~0."""
        a, b = self.episode_bounds(episode)
        return self._ts[a:b].copy()

    # -- the chunk --------------------------------------------------------
    def chunk_at(self, episode: int, local_frame: int) -> RecordedChunk:
        """The recorded action chunk at ``local_frame`` within ``episode``.

        Returns the chunk in the model's own anchor-relative ``(n_valid, action_dim)`` form
        (see :class:`RecordedChunk`), truncated to the LEADING run of steps the converter
        marked valid. Raises ``ValueError`` if not even step 0 is valid (a fully-masked
        frame, which the caller must skip rather than execute)."""
        a, b = self.episode_bounds(episode)
        n = b - a
        lf = int(local_frame)
        if not 0 <= lf < n:
            raise IndexError(f"local_frame {lf} is outside [0, {n - 1}] for episode {episode}")
        frame = a + lf

        actions = np.zeros((ACTION_HORIZON, self.layout.action_dim), dtype=np.float64)
        # The number of leading REAL future steps is the min over roles: a step is only
        # executable if every driven role has a real target for it.
        n_valid = ACTION_HORIZON
        gripper_valid = True
        force_valid = True
        force_steps: dict[str, np.ndarray] = {}
        grip_steps: dict[str, np.ndarray] = {}

        for role in self.layout.roles:
            k = self._keys[role]
            slots = self.layout.slots(role)

            # ``[:ACTION_HORIZON]`` on every read: the file may store a LONGER chunk than this
            # build executes (see _check_contract), and a shorter horizon is its strict prefix.
            # A no-op when the two agree.
            pose = np.asarray(self._data[k["pose"]][frame], dtype=np.float64)[:ACTION_HORIZON]
            if pose.shape != (ACTION_HORIZON, 6):
                raise ValueError(
                    f"{self.path}: {k['pose']}[{frame}] has shape {pose.shape}, "
                    f"expected ({ACTION_HORIZON}, 6)"
                )
            abs_pos = pose[:, :3]
            abs_rot6d = rotvec_to_rot6d(pose[:, 3:])
            # Relativize against THIS FRAME'S OWN recorded pose == step 0 of the chunk
            # (DECISION-T0), which makes step 0 an exact zero delta and reproduces the
            # training action_chunk. Using step 0 rather than a separate camera-grid read
            # keeps the anchor and the chunk from ever disagreeing by a frame.
            #
            # BODY-frame, deliberately — see :func:`_body_vs_world_transport` for the measured
            # reason this is NOT changed to a world-frame delta despite the demo replaying
            # visibly REORIENTED on the rig.
            rel_pos, rel_rot6d = absolute_to_delta_ee(
                abs_pos, abs_rot6d, abs_pos[0], abs_rot6d[0]
            )
            actions[:, slots.pos] = rel_pos
            actions[:, slots.rot6d] = rel_rot6d

            step_valid = (
                np.asarray(self._data[k["valid"]][frame], dtype=np.float64).reshape(-1)[:ACTION_HORIZON] > 0.5
            )
            n_valid = min(n_valid, _leading_true(step_valid))

            if "gripper" in k:
                gw_m = np.asarray(self._data[k["gripper"]][frame], dtype=np.float64).reshape(-1)[:ACTION_HORIZON]
                # The stored array is ABSOLUTE METRES (~0.043..0.094 on this rig), NOT a
                # fraction — so it must be normalized exactly as ``HLDataset.__getitem__``
                # does before it can sit in an action vector. Skipping this reads 0.094 m as
                # "9.4% open" and would command the jaw nearly SHUT on a fully-open frame
                # (caught by the vs-training-action_chunk check, which is why that check
                # compares the gripper column and not just the pose).
                actions[:, slots.gripper] = gripper_width_to_norm(gw_m)
                gv = (np.asarray(self._data[k["gripper_valid"]][frame], dtype=np.float64)
                      .reshape(-1)[:ACTION_HORIZON] > 0.5)
                gripper_valid = gripper_valid and bool(gv[0])
                grip_steps[role] = gv

            if "force" in k:
                fw = np.asarray(self._data[k["force"]][frame], dtype=np.float64)[:ACTION_HORIZON]  # (H, 6)
                actions[:, slots.force] = fw
                fv = (np.asarray(self._data[k["force_valid"]][frame], dtype=np.float64)
                      .reshape(-1)[:ACTION_HORIZON] > 0.5)
                force_valid = force_valid and bool(fv[0])
                force_steps[role] = fv

        if n_valid < 1:
            raise ValueError(
                f"episode {episode} frame {local_frame}: no valid recorded action step "
                "(fully-masked frame) — skip it rather than executing padded values"
            )
        if not np.isfinite(actions[:n_valid]).all():
            raise FloatingPointError(
                f"episode {episode} frame {local_frame}: recorded chunk has non-finite values"
            )
        return RecordedChunk(
            episode=int(episode),
            local_frame=lf,
            actions=actions[:n_valid].copy(),
            n_valid=int(n_valid),
            time_s=float(self._ts[frame]),
            gripper_valid=gripper_valid,
            force_valid=force_valid,
            force_step_valid={r: v[:n_valid].copy() for r, v in force_steps.items()},
            gripper_step_valid={r: v[:n_valid].copy() for r, v in grip_steps.items()},
        )

    # -- whole-episode open-loop plan -------------------------------------
    def episode_plan(self, episode: int, start_frame: int) -> EpisodePlan:
        """The whole episode from ``start_frame`` as ONE anchor-relative open-loop chunk.

        See :class:`EpisodePlan`. The POSE columns are the CAMERA-GRID trajectory
        (``episode_path_relative``, the same curve the green line + the trace reference draw),
        NOT a concatenation of per-frame action-grid chunks — the camera grid is the
        demonstration's own shape without the converter's uniform-resample drift, and using it
        makes the command, the drawn path, and the comparison reference one curve (see
        :class:`EpisodePlan`). Gripper/force for each commanded frame come from that frame's
        action step 0 — the recorded value AT the frame, the same source ``chunk_at`` reads — so
        an open-loop drip commands the same jaw/wrench sequence the per-chunk path would, minus
        the per-chunk re-anchor.

        The plan runs from ``start_frame`` to the LAST frame before the first unusable
        (glitch-window or step-0-masked) frame at/after it — :meth:`frame_is_usable`, the exact
        gate the sequential loop skips on, except an open-loop drip TRUNCATES there instead of
        seeking past it (it cannot: the frames between would be a commanded teleport across the
        excursion). Raises ``ValueError`` if ``start_frame`` itself is not usable (the caller
        must seek to a playable frame first, exactly as for :meth:`chunk_at`)."""
        a, b = self.episode_bounds(episode)
        n = b - a
        sf = int(start_frame)
        if not 0 <= sf < n:
            raise IndexError(f"start_frame {sf} is outside [0, {n - 1}] for episode {episode}")
        if not self.frame_is_usable(episode, sf):
            raise ValueError(
                f"episode {episode} start_frame {sf} is not usable (masked/glitch) — seek to a "
                "playable frame with next_usable_frame() before planning the episode"
            )
        # The contiguous run of usable frames starting at sf. An open-loop plan stops at the
        # first hole rather than skipping it (a skip would command the excursion between).
        end = sf + 1
        while end < n and self.frame_is_usable(episode, end):
            end += 1
        truncated = end < n
        n_cmd = end - sf

        actions = np.zeros((n_cmd, self.layout.action_dim), dtype=np.float64)
        gripper_valid = True
        force_valid = True
        force_steps: dict[str, np.ndarray] = {}
        grip_steps: dict[str, np.ndarray] = {}
        for role in self.layout.roles:
            slots = self.layout.slots(role)
            pos_key, rot_key = self._cam_keys[role]
            pos = np.asarray(self._data[pos_key][a + sf : a + end], dtype=np.float64).reshape(n_cmd, 3)
            rot6d = rotvec_to_rot6d(
                np.asarray(self._data[rot_key][a + sf : a + end], dtype=np.float64).reshape(n_cmd, 3)
            )
            # Relative to the START frame's own pose (one anchor for the whole episode), NOT to
            # each frame's own pose — that single-anchor transport is the whole point.
            rel_pos, rel_rot6d = absolute_to_delta_ee(pos, rot6d, pos[0], rot6d[0])
            actions[:, slots.pos] = rel_pos
            actions[:, slots.rot6d] = rel_rot6d

            k = self._keys[role]
            if "gripper" in k:
                # Each commanded frame's OWN recorded jaw (step 0 of its action window), the same
                # per-frame source chunk_at reads — normalized to the action-vector fraction.
                gw_m = np.asarray(
                    self._data[k["gripper"]][a + sf : a + end, 0], dtype=np.float64
                ).reshape(-1)
                actions[:, slots.gripper] = gripper_width_to_norm(gw_m)
                # Each commanded frame's own step-0 bit, the same column `chunk_at` reads at [0].
                gvs = (np.asarray(self._data[k["gripper_valid"]][a + sf : a + end, 0], dtype=np.float64)
                       .reshape(-1) > 0.5)
                gripper_valid = gripper_valid and bool(gvs[0])
                grip_steps[role] = gvs

            if "force" in k:
                fw = np.asarray(
                    self._data[k["force"]][a + sf : a + end, 0], dtype=np.float64
                ).reshape(n_cmd, 6)
                actions[:, slots.force] = fw
                fvs = (np.asarray(self._data[k["force_valid"]][a + sf : a + end, 0], dtype=np.float64)
                       .reshape(-1) > 0.5)
                force_valid = force_valid and bool(fvs[0])
                force_steps[role] = fvs

        if not np.isfinite(actions).all():
            raise FloatingPointError(
                f"episode {episode} plan from {sf}: recorded trajectory has non-finite values"
            )
        return EpisodePlan(
            episode=int(episode),
            start_frame=sf,
            actions=actions,
            n_frames=int(n_cmd),
            times_s=self._ts[a + sf : a + end].copy(),
            truncated=bool(truncated),
            gripper_valid=gripper_valid,
            force_valid=force_valid,
            force_step_valid=force_steps,
            gripper_step_valid=grip_steps,
        )

    # -- cursor advance ---------------------------------------------------
    def advance_frame(self, episode: int, local_frame: int, n_steps_executed: int) -> int:
        """The local frame reached after EXECUTING ``n_steps_executed`` action steps.

        Advance is by TIME, not by frame count. The action grid is 16 Hz (``ACTION_DT``)
        while this rig's camera grid is ~15 Hz, so stepping the cursor by
        ``n_steps_executed`` FRAMES would replay the episode at the wrong speed and let the
        cursor drift out of phase with the executed motion over a long episode. We instead
        land on the camera frame nearest the wall-clock time the executed steps consumed.

        THE OFF-BY-ONE THAT MATTERS: executing ``k`` steps advances the demonstration by
        ``(k-1) * ACTION_DT``, not ``k * ACTION_DT``. Step 0 of a chunk is the anchor frame's
        OWN pose (DECISION-T0 — an exact zero delta), so it commands no motion; the arm's
        realized demo time after a chunk is ``ts[lf] + (k-1)*ACTION_DT``, which is also where
        the executor commits its anchor (the last executed step). Advancing by ``k`` instead
        would leave a one-step gap that is never commanded, and — compounded by the snap onto
        the coarser camera grid — SPATIALLY TRUNCATES the replay: measured on clean94, the arm
        traced only ~70% of the demonstration's path length at the default 4-steps-per-chunk
        (~93% once corrected). A cursor-vs-wall-clock check cannot see this; it shows up only
        as commanded-path vs recorded-path.

        Returns a frame clamped to the episode's last index, so a caller can detect
        "reached the end" as ``advance_frame(...) == length - 1``."""
        a, b = self.episode_bounds(episode)
        n = b - a
        lf = int(local_frame)
        if not 0 <= lf < n:
            raise IndexError(f"local_frame {lf} is outside [0, {n - 1}] for episode {episode}")
        steps = int(n_steps_executed)
        if steps < 1:
            raise ValueError(f"n_steps_executed must be >= 1, got {steps}")
        ts = self._ts[a:b]
        # (steps - 1): see the off-by-one note above. A 1-step chunk therefore advances by 0 s
        # and the min-progress clamp below moves it one frame, which is the correct floor.
        target_t = ts[lf] + (steps - 1) * ACTION_DT
        # searchsorted gives the first frame at/after target_t; pick whichever neighbour is
        # closer in time so the cursor cannot systematically lag or lead the executed motion.
        j = int(np.searchsorted(ts, target_t, side="left"))
        if j >= n:
            return n - 1
        if j > 0 and abs(ts[j - 1] - target_t) <= abs(ts[j] - target_t):
            j -= 1
        # Always make progress: a degenerate/duplicated timestamp must not stall a replay.
        return max(j, min(lf + 1, n - 1))

    # -- recorded path, for drawing ---------------------------------------
    def episode_path_relative(
        self, episode: int, ref_local_frame: int, role: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """The WHOLE episode's recorded pose, expressed relative to one of its own frames.

        Returns ``(rel_pos[N,3], rel_rot6d[N,6])`` — the episode's camera-grid trajectory in
        the body frame of ``ref_local_frame``'s recorded pose. The caller transports it into
        the robot's world with the SAME ``delta_ee_to_absolute`` + live anchor it decodes the
        executed chunk with, so the drawn path and the commanded motion cannot disagree about
        the frame (the mistake that made the offline viewer's scene read tilted).

        Camera-grid (not action-grid) because this is the demonstration's shape for the eye,
        matching what ``replay_zarr_viz.py`` draws as its green GT line."""
        a, b = self.episode_bounds(episode)
        n = b - a
        rf = int(ref_local_frame)
        if not 0 <= rf < n:
            raise IndexError(f"ref_local_frame {rf} is outside [0, {n - 1}] for episode {episode}")
        if role not in self._cam_keys:
            raise KeyError(f"layout {self.layout.mode!r} has no role {role!r}")
        pos_key, rot_key = self._cam_keys[role]
        pos = np.asarray(self._data[pos_key][a:b], dtype=np.float64).reshape(n, 3)
        rot6d = rotvec_to_rot6d(np.asarray(self._data[rot_key][a:b], dtype=np.float64).reshape(n, 3))
        # The SAME body-frame form :meth:`chunk_at` uses — so the drawn path and the commanded
        # motion stay the same curve in the same frame. That agreement is the whole point of
        # this method: the green line must show where the arm is actually going, INCLUDING the
        # ready-pose reorientation (see :func:`_body_vs_world_transport`). Drawing a
        # world-oriented line over body-oriented motion would be a prettier lie.
        return absolute_to_delta_ee(pos, rot6d, pos[rf], rot6d[rf])

    def recorded_pose(self, episode: int, local_frame: int, role: str) -> tuple[np.ndarray, np.ndarray]:
        """The ABSOLUTE recorded camera-grid pose of ``role`` at one frame: ``(pos (3,), rot6d (6,))``
        in the file's aligned-world frame. For reporting (the transport angle against the decode
        anchor), never for commanding — the command is always the relative form."""
        a, b = self.episode_bounds(episode)
        lf = int(local_frame)
        if not 0 <= lf < (b - a):
            raise IndexError(f"local_frame {lf} is outside [0, {b - a - 1}] for episode {episode}")
        if role not in self._cam_keys:
            raise KeyError(f"layout {self.layout.mode!r} has no role {role!r}")
        pos_key, rot_key = self._cam_keys[role]
        pos = np.asarray(self._data[pos_key][a + lf], dtype=np.float64).reshape(3)
        rot6d = rotvec_to_rot6d(np.asarray(self._data[rot_key][a + lf], dtype=np.float64).reshape(1, 3))[0]
        return pos, np.asarray(rot6d, dtype=np.float64).reshape(6)

    def episode_gripper_m(self, episode: int, role: str = "left") -> np.ndarray | None:
        """The episode's recorded jaw width, ``(N,)`` METRES on the camera grid, or None if
        this dataset/role has no gripper column.

        Metres, NOT the normalized fraction :meth:`chunk_at` puts in the action vector: this
        feeds the comparison sheet's jaw panel, which is read against the physical 43–94 mm the
        operator sees on the rig. Camera-grid to match :meth:`episode_path_relative`, so one
        clock serves the whole drawn/plotted episode."""
        a, b = self.episode_bounds(episode)
        key = self._cam_grip_keys.get(role)
        if key is None or key not in self._data:
            return None
        return np.asarray(self._data[key][a:b], dtype=np.float64).reshape(-1)

    @property
    def image_keys(self) -> tuple[str, ...]:
        """The stored camera views, ``camera0_rgb`` first (then ``camera1_rgb`` on a dual-rig file)."""
        keys = sorted(k for k in self._data.keys() if k.startswith("camera") and k.endswith("_rgb"))
        return tuple(keys)

    def image_at(self, episode: int, local_frame: int, key: str = DEFAULT_IMAGE_KEY) -> np.ndarray:
        """The recorded ``(H, W, 3)`` uint8 camera frame — what the operator was seeing.

        ``key`` picks the view (``camera0_rgb`` = the LEFT wrist D405, the default every file has;
        ``camera1_rgb`` on a dual-rig file); a view the file does not carry is refused by name rather
        than answered with another camera's frame."""
        a, b = self.episode_bounds(episode)
        lf = int(local_frame)
        if not 0 <= lf < (b - a):
            raise IndexError(f"local_frame {lf} is outside [0, {b - a - 1}] for episode {episode}")
        if key not in self._data:
            raise KeyError(f"{self.path}: no camera view {key!r}; this file carries {list(self.image_keys)}")
        return np.asarray(self._data[key][a + lf], dtype=np.uint8)

    def close(self) -> None:
        store = getattr(self._root, "store", None)
        if store is not None and hasattr(store, "close"):
            store.close()


def _leading_true(flags: np.ndarray) -> int:
    """How many entries at the FRONT of ``flags`` are True (0 if the first is False).

    Deliberately the LEADING run, not the total count: a chunk is executed in order, so a
    validity hole at step k makes every later step unusable regardless of its own flag."""
    f = np.asarray(flags, dtype=bool).reshape(-1)
    false_at = np.flatnonzero(~f)
    return int(f.size) if false_at.size == 0 else int(false_at[0])


def episode_options(source: RecordedEpisodeSource, firsts: Sequence[int | None] | None = None) -> list[str]:
    """One label per episode for a picker: ``ep 12  (253 fr)``, plus ``⚠ starts at N`` when the
    converter masked the episode's first frames (the cursor auto-seeks there) and ``⚠ no playable
    frame`` when nothing in it can be executed. Zero-padded to the file's width so the labels sort
    and search as numbers (``ep 0012`` on a 1364-episode file)."""
    lens = source.episode_lengths()
    if firsts is None:
        firsts = source.first_usable_frames()
    width = max(1, len(str(max(source.n_episodes - 1, 0))))
    out = []
    for i, n in enumerate(lens):
        first = firsts[i]
        label = f"ep {i:0{width}d}  ({n} fr)"
        if first is None:
            label += "  ⚠ no playable frame"
        elif first != 0:
            label += f"  ⚠ starts at {first}"
        out.append(label)
    return out


def speed_report(source: RecordedEpisodeSource, episode: int, step_hz: float,
                 exec_choices: Sequence[int]) -> str:
    """How the replay's WALL CLOCK compares to the human's, per execute-N choice — one line.

    The demonstration sits on the CAMERA grid and executes on the ACTION grid at ``step_hz``; the
    cursor lands on the camera frame nearest the executed time (:meth:`RecordedEpisodeSource.advance_frame`),
    so the cumulative speed depends on how many steps each chunk executes. Walks ``episode`` the
    way the sequential loop does — skipping unplayable frames (they cost no execution time) and
    clamping each chunk to its own valid length — and reports ``(recorded / executed − 1)`` per
    choice (``+`` = the replay is FASTER than the human). On a 30 Hz camera grid against 30 Hz
    execution every entry is within a frame of 0 %; on the old ~15 Hz rig it was −7..−20 %.
    Returns ``""`` when the episode is too short to score."""
    times = source.episode_times(episode)
    if times.size <= 2:
        return ""
    cam_dt = float(np.median(np.diff(times)))
    if cam_dt <= 1e-6:
        return ""
    step_dt = 1.0 / max(float(step_hz), 1.0)
    n_ep = int(times.size)
    start = source.first_usable_frame(episode)
    if start is None:
        return ""
    parts = []
    for n in exec_choices:
        cur, steps, guard = int(start), 0, 0
        t_from = float(times[cur])
        while cur < n_ep - 1 and guard < 100000:
            guard += 1
            u = source.next_usable_frame(episode, cur)
            if u is None:
                break
            if u != cur:
                cur = u
                continue
            n_done = min(int(n), source.chunk_at(episode, cur).n_valid)
            cur = source.advance_frame(episode, cur, n_done)
            steps += n_done
        exec_s = steps * step_dt
        rec_s = float(times[min(cur, n_ep - 1)]) - t_from
        if exec_s > 1e-6 and rec_s > 1e-6:
            parts.append(f"exec={int(n)}: {(rec_s / exec_s - 1.0) * 100:+.0f}%")
    return (f"recorded camera grid ~{1.0 / cam_dt:.2f} Hz vs execution {float(step_hz):.2f} Hz → "
            f"wall-clock speed vs the human ({', '.join(parts)}; + = faster)")


def transport_rotation_deg(recorded_rot6d: np.ndarray, anchor_rot6d: np.ndarray) -> float:
    """The angle (degrees) by which the replayed demo is RE-ORIENTED on the robot.

    The chunk is anchor-relative in the BODY frame, so the net map on the demonstration is the
    rigid rotation ``R_anchor @ R_recorded^T`` — the angle between how the human held the tool at
    the reference frame and how the robot holds it at the decode anchor
    (:func:`_body_vs_world_transport`: 106.9° mean on clean94). Reported, never corrected: it is
    physical, and removing it would need a demo-compatible ready pose, not a transport change."""
    r_rec = rot6d_to_rotmat(np.asarray(recorded_rot6d, dtype=np.float64).reshape(6))
    r_anc = rot6d_to_rotmat(np.asarray(anchor_rot6d, dtype=np.float64).reshape(6))
    r = r_anc @ r_rec.T
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def apply_step_validity(result, force_step_valid: dict | None, gripper_step_valid: dict | None):
    """Blank the DECODED rows the converter marked unmeasured, so replay never commands them.

    ``result`` is a decoded chunk (``hl_online.ChunkResult``: a tuple of per-role ``RoleChunk`` with
    ``wrench_pred`` / ``wrench_cmd`` / ``gripper_m``), duck-typed so this module stays torch-free.
    Per role:

    * a FORCE step whose bit is False → ``wrench_pred`` and ``wrench_cmd`` rows set to ZERO. The
      stored value there is the pre-tare / uncovered bin, and a zero wrench is exactly what the
      page's live gate turns into "no F^cmd" for that step;
    * a GRIPPER step whose bit is False → FORWARD-FILLED from the last measured width (or the
      first later one). The stored value is zero-filled for a dead encoder and 0 m is a CLOSED jaw,
      so commanding it would shut the hand; holding the last measured width is what the jaw would
      do anyway. A role with NO measured step in the chunk gets ``gripper_m=None`` — "no jaw on
      this role" to every consumer (the drip skips it, the ghost draws it open).

    Returns ``(result, note)``: the SAME object when nothing had to change (so a fully-measured
    chunk — and every model chunk — is untouched, provably), else a rebuilt one; ``note`` counts
    what was blanked per role (``{"force": {role: n}, "gripper": {role: n}, "gripper_dropped": [..]}``).
    """
    note = {"force": {}, "gripper": {}, "gripper_dropped": []}
    fsv = force_step_valid or {}
    gsv = gripper_step_valid or {}
    new_roles = []
    changed = False
    for rc in result.roles:
        rc_new = rc
        fv = fsv.get(rc.role)
        if fv is not None and rc.wrench_pred is not None:
            bad = ~np.asarray(fv, dtype=bool).reshape(-1)
            n = int(rc.wrench_pred.shape[0])
            bad = bad[:n]
            if bad.any():
                wp = np.asarray(rc.wrench_pred, dtype=np.float64).copy()
                wp[: bad.size][bad] = 0.0
                wc = rc.wrench_cmd
                if wc is not None:
                    wc = np.asarray(wc, dtype=np.float64).copy()
                    wc[: bad.size][bad] = 0.0
                rc_new = dataclasses.replace(rc_new, wrench_pred=wp, wrench_cmd=wc)
                note["force"][rc.role] = int(bad.sum())
                changed = True
        gv = gsv.get(rc.role)
        if gv is not None and rc.gripper_m is not None:
            ok = np.asarray(gv, dtype=bool).reshape(-1)
            g = np.asarray(rc.gripper_m, dtype=np.float64).reshape(-1)
            n = min(int(g.shape[0]), int(ok.size))
            ok = np.concatenate([ok[:n], np.zeros(max(0, g.shape[0] - n), dtype=bool)])
            if not ok.all():
                if not ok.any():
                    rc_new = dataclasses.replace(rc_new, gripper_m=None)
                    note["gripper_dropped"].append(rc.role)
                else:
                    idx = np.flatnonzero(ok)
                    # forward-fill: each step takes the nearest measured step at/before it, and
                    # the steps before the first measured one take the first measured one.
                    prev = np.maximum.accumulate(np.where(ok, np.arange(g.size), -1))
                    prev = np.where(prev < 0, int(idx[0]), prev)
                    filled = g[prev]
                    rc_new = dataclasses.replace(rc_new, gripper_m=filled)
                    note["gripper"][rc.role] = int((~ok).sum())
                changed = True
        new_roles.append(rc_new)
    if not changed:
        return result, note
    return dataclasses.replace(result, roles=tuple(new_roles)), note


__all__ = [
    "DEFAULT_IMAGE_KEY",
    "EpisodePlan",
    "RecordedChunk",
    "RecordedEpisodeSource",
    "apply_step_validity",
    "episode_options",
    "speed_report",
    "transport_rotation_deg",
]
