"""The G1 WHOLE BODY in the offline inference viewer, driven by MOTION COMPLETION.

Opt-in (``infer_s1_diffusion.py --motion-completion``) and additive: without the flag this
module is constructed but starts no backend, adds no scene node and no GUI panel, and the
viewer is byte-for-byte the trajectory-and-gripper-cluster tool it has always been. Same
shape and the same contract as UMI Collection's ``replay/motion_completion_overlay.py``,
which this delegates to wholesale — see that module and
``third_party/UMI_collection/motion_completion/README.md``.

WHY IT IS NOT AN IK
===================
The viewer already draws each keypoint's gripper cluster from the predicted pose alone;
placing a humanoid there would need arms, and an arm IK answers "which joints put the hand
here", not "what does the ROBOT do". Motion completion answers the second question: it is a
flow-matching model over G1 whole-body motion conditioned on the same 3 points, so the legs,
torso and pelvis it produces are the ones the low-level controller would be handed. A rollout
drawn this way is therefore checkable as a *deployment*: the posture readout says whether the
predicted hand heights make the robot stand, crouch or go on tiptoe (they are a CONDITIONED
channel), and the keypoint markers are literally what ``select_keypoints`` would send.

⚠️ The completion is NOT part of the policy and NOT part of any metric on the page. The
Metrics panel, the charts and the blue/orange lines are untouched by this module; it is a
second, downstream model shown beside them.

WHICH POSES ARE FED
===================
Whatever the viewer is currently DRAWING — the display-frame poses out of
``_build_display_poses``, not a re-derivation. That is the property that makes the picture
readable: the conditioning triads the panel draws land on the FLoRe gripper clusters, so
"the robot's hands are on the orange line" is a coordinate check you can make by eye.

It also happens to place the take correctly for free. The default home-anchored display puts
the hands at ``DISPLAY_HOME_POS`` z=0.95 and the chest at z=1.30, and those ARE
``mc_frames.RIG_WRIST_STAND_Z`` / ``RIG_CHEST_STAND_Z`` (one dict, pinned equal to UMI's
``gripper_only_view.HOME_WRIST_POSE`` by a test). So ``cond_mode="absolute"`` with the chest
anchor derives ``dz ~= 0`` on its own: the humanoid stands on the grid AND its hands sit on
the drawn chunk, with no per-session lift to guess. That is why the default here is
``absolute`` where the zarr replay's is ``home`` — the replay has a real libsurvive origin to
bridge, this viewer's display frame has already done it.

With the home-anchor checkbox OFF the raw absolute poses are fed instead and the chest anchor
does a real lift, exactly as in the zarr replay. Both are honest; only the first is registered
with the lines.

Two documented consequences of borrowing the display frame, neither hidden:

* the display home is PER ROLE, so the operator's measured hand-to-hand and hand-to-chest
  offsets are replaced by the homes' own (0.36 m apart, chest 0.35 m above the hands). Only
  the CHANGE in their relative pose survives — the same trade the collection viz, both zarr
  replays and this viewer's own gripper cluster already make.
* the rebased rotations start at identity, so the completed robot always begins facing +x
  regardless of which way the operator was standing.

THE PLAY HEAD IS THE ACTION STEP
================================
The chunk is 50 steps on the same 30 Hz grid the model was trained at, so the viewer's
"Action step" slider is a play head in the MC sense with no resampling. ``window_starts(50,
30, 10)`` is ``[0, 20]`` — two windows per rollout, run in order as the slider reaches them,
on the overlay's worker thread.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from flore.s1_policy.data.layout import ACTION_DT
from flore.s1_policy.inference.display import read_stored_extrinsic, wrist_pose_from_stored

#: The three keypoints motion completion conditions on. Two hands AND the chest: the model's
#: root channel (pelvis XY + heading) comes from the chest, so a ``both_hands`` layout with no
#: head role cannot drive it and is refused by name rather than fed a fabricated root.
REQUIRED_ROLES = ("left", "right", "head")

#: Which series drives the humanoid. "prediction" is the point of the tool; "target" is the
#: control — the same completion over the DATASET's own masked chunk, so a posture that looks
#: wrong can be attributed to the policy or to the demonstration.
SERIES = ("prediction", "target")

_PANEL_MD = """\
**Whole-body motion completion** — the 3 predicted keypoints completed to a 29-DoF G1 motion,
one window at a time as you drag *Action step*.

| | what it is |
|---|---|
| 🤖 **G1** | the completion, in this rig's `g1_29dof_yam.urdf` (yam jaws + F/T) |
| 🔵 / 🩷 **triads + trails** | the L / R `gripper_base` pose the completion was CONDITIONED on |
| 🟠 **path + triad** | the root cond: chest XY + heading (the only chest channels used) |
| 🟡 **markers** | the keypoints a low-level policy would be handed for the selected mode |

The blue/pink triads are **input** and the robot is **output**: they must coincide, and
`wrist err` measures it. They are fed the poses this page DRAWS, so they should also sit on
the FLoRe gripper clusters — that agreement is the coordinate check.

⚠️ This is a SECOND model, downstream of the policy. It changes no number in Metrics and no
line in the scene; `series` below chooses whether it completes the **prediction** or the
dataset **target**.
"""


def _find_umi_replay_dir() -> str | None:
    """Locate UMI Collection's ``replay/`` directory (holds motion_completion_overlay.py).

    Same walk-up as :func:`~flore.s1_policy.inference.viewer._find_umi_viz_dir`; the overlay
    itself puts ``motion_completion/`` and ``viz/`` on the path when imported.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(root, "third_party", "UMI_collection", "replay")
        if os.path.isfile(os.path.join(candidate, "motion_completion_overlay.py")):
            return candidate
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    return None


class MCHumanoid:
    """The offline viewer's motion-completion humanoid. One instance per viewer.

    A thin adapter, deliberately: every line of the streaming schedule, the scene, the panel
    and the readouts is UMI Collection's ``MotionCompletionOverlay``. What lives here is the
    one thing FLoRe knows and it does not — how to turn a :class:`RolloutResult` plus the
    viewer's display-frame poses into ``CondEpisode`` conditioning.
    """

    def __init__(self, server, *, enabled: bool = False, urdf_path: str | None = None,
                 log=None, wrist_ik: bool = True, foot_ik: bool = True,
                 foot_ik_iters: int | None = None, series: str = "prediction",
                 cond_mode: str = "absolute", dz_wrist: float | None = None,
                 placement_note: str = "") -> None:
        self.server = server
        self.enabled = bool(enabled)
        self.log = log or (lambda m: print(f"[s1-inference][mc] {m}", flush=True))
        self.overlay = None
        self.disabled_reason = "" if self.enabled else "not requested (--motion-completion)"
        if str(series) not in SERIES:
            raise ValueError(f"series must be one of {SERIES}, got {series!r}")
        self._series = str(series)
        # A pinned z lift, used when the caller already KNOWS the placement — the
        # ``--display-frame g1-ready-pose`` viewer FK'd the floor to z=0, so there is nothing
        # for auto_dz_chest to derive and deriving it would lift the take ~15 cm off the robot.
        # The panel's own "pin dz" still wins (see :meth:`_build_cond`), so this is a default,
        # not a lock.
        self._dz_wrist = None if dz_wrist is None else float(dz_wrist)
        self._placement_note = str(placement_note)
        # Retained so on_rollout can rebuild after the series dropdown changes without the
        # caller having to re-hand them (the viewer only re-runs inference on demand).
        self._pending: dict[str, Any] | None = None
        if not self.enabled:
            return
        replay_dir = _find_umi_replay_dir()
        if replay_dir is None:
            self.disabled_reason = (
                "third_party/UMI_collection/replay not found (git submodule update --init "
                "third_party/UMI_collection)")
            self.log(f"DISABLED: {self.disabled_reason}")
            return
        # append, NOT insert(0): the UMI replay/viz dirs hold generically-named modules
        # (collection_viz, replay_common, test_*) which must not shadow same-named top-level
        # imports for the rest of this process. Same rule the gripper-mesh overlay follows.
        if replay_dir not in sys.path:
            sys.path.append(replay_dir)
        try:
            from motion_completion_overlay import MotionCompletionOverlay  # noqa: PLC0415

            self.overlay = MotionCompletionOverlay(
                server, enabled=True, urdf_path=urdf_path, log=self.log,
                wrist_ik=wrist_ik, foot_ik=foot_ik, foot_ik_iters=foot_ik_iters,
                init_cond_mode=cond_mode, panel_md=_PANEL_MD,
                folder_label="Motion completion (G1 whole body)")
        except Exception as exc:  # noqa: BLE001 - the humanoid is optional; never kill the page
            self.disabled_reason = f"motion completion unavailable ({exc!r})"
            self.log(f"DISABLED: {self.disabled_reason}")
            return
        if not self.overlay.available:
            self.disabled_reason = self.overlay.disabled_reason
            self.overlay = None

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return self.enabled and self.overlay is not None and not self.disabled_reason

    def build_gui(self, gui) -> None:
        """Add the panel. A no-op without the flag, so the page is unchanged."""
        if not self.enabled:
            return
        if not self.available:
            with gui.add_folder("Motion completion (G1 whole body)"):
                gui.add_text("status", initial_value="DISABLED", disabled=True)
                gui.add_text("reason", initial_value=self.disabled_reason[:110], disabled=True)
                gui.add_text("what it needs", disabled=True, initial_value=(
                    "a Motion_Completion checkout + its g1_mosyn conda env + a downloaded "
                    "checkpoint (MOTION_COMPLETION_{DIR,PY,CKPT})"))
            return
        self.overlay.build_gui(gui)
        # FLoRe's one addition to the panel: which of the two drawn chunks to complete. Placed
        # in the overlay's own "needs a re-run" folder semantics — it is a CONDITIONING
        # choice, so it takes effect on the next rollout or on "Re-run this episode".
        with gui.add_folder("Motion completion — series (needs a re-run)"):
            self._series_dd = gui.add_dropdown(
                "complete", SERIES, initial_value=self._series)
            gui.add_text("what changes", disabled=True, initial_value=(
                "prediction = the policy's chunk · target = the dataset's own masked chunk"))
            # WHERE the chunk was placed is the single biggest driver of the posture the
            # completion returns, and it is decided by --display-frame, not by this panel — so
            # it is stated here rather than left to be inferred from the dz readout.
            gui.add_text("placement", disabled=True, initial_value=(
                self._placement_note[:110] if self._placement_note else
                "display homes (uncalibrated) — pass --display-frame g1-ready-pose"))

            @self._series_dd.on_update
            def _(_e) -> None:
                # GUI callback thread: flag only. The loop thread rebuilds (repo rule).
                self._series = str(self._series_dd.value)
                if self.overlay is not None:
                    self.overlay._restart_requested = True

    # ------------------------------------------------------------------ driving
    def on_rollout(self, result, display_poses: dict, file_attrs, home_anchored: bool) -> None:
        """Re-condition the completion on a fresh rollout. Loop thread.

        ``display_poses`` is the viewer's own ``role -> _DisplayPoses`` map for the CURRENT
        display-frame decision — passed in rather than recomputed, so the humanoid cannot end
        up in a different frame than the lines it is drawn against (the failure this viewer's
        ``_rebuild_scene`` docstring already warns about for the gripper cluster).
        """
        if not self.available:
            return
        missing = [role for role in REQUIRED_ROLES if role not in display_poses]
        if missing:
            self.disabled_reason = (
                f"layout {result.layout_mode!r} has no {', '.join(missing)} role(s); motion "
                f"completion conditions on both hands AND the chest (the root channel). Use a "
                f"3-tracker checkpoint (both_hands_head / both_hands_head_dual).")
            self.log(f"DISABLED: {self.disabled_reason}")
            self.overlay = None
            return
        extrinsic, _note = read_stored_extrinsic(file_attrs[0] if file_attrs else None)
        eef_t_wrist, eef_R_wrist = wrist_pose_from_stored(extrinsic)
        grip = self._gripper_widths(result)
        self._pending = {
            "display_poses": display_poses,
            "eef_t_wrist": eef_t_wrist,
            "eef_R_wrist": eef_R_wrist,
            "grip": grip,
            "home_anchored": bool(home_anchored),
            "identity": (f"{result.algorithm} sample {result.sample.dataset_index} "
                         f"seed {result.seed}"),
        }
        self.overlay.clear()
        self.overlay.on_cond(self._build_cond)

    def _gripper_widths(self, result) -> dict[str, dict[str, np.ndarray]]:
        """``series -> role -> (H,) metres``, for the DRAWN jaws only.

        The completion has no gripper channel, so the jaws come from the chunk — which for
        this viewer is a MODEL OUTPUT, not a recording. A role with no gripper column keeps
        its jaws shut, the same as the zarr replay's behaviour on a hand with no encoder.
        """
        out: dict[str, dict[str, np.ndarray]] = {"prediction": {}, "target": {}}
        for role_result in result.roles:
            for series, values in (("prediction", role_result.predicted_gripper_m),
                                   ("target", role_result.target_gripper_m)):
                if values is not None:
                    out[series][role_result.role] = np.asarray(
                        values, dtype=np.float64).reshape(-1)
        return out

    def _build_cond(self, **knobs):
        """``CondEpisode`` for the selected series, in the viewer's display frame.

        Called BY the overlay with its live conditioning knobs, so this never reads the GUI.
        """
        from mc_episode import CondEpisode  # noqa: PLC0415
        import mc_frames as F  # noqa: PLC0415

        state = self._pending
        if state is None:
            raise RuntimeError("motion completion asked for conditioning before a rollout")
        poses = state["display_poses"]
        series = self._series

        def _pose(role: str) -> tuple[np.ndarray, np.ndarray]:
            entry = poses[role]
            if series == "prediction":
                position, wxyz = entry.predicted_position, entry.predicted_wxyz
            else:
                position, wxyz = entry.target_position, entry.target_wxyz
            # wxyz -> rotmat through MC's own converter: the display poses are stored as
            # quaternions (that is what viser is handed), and re-deriving the rotation from
            # rot6d here would bypass the display-frame rebase the whole point is to share.
            return (np.asarray(position, dtype=np.float64),
                    F.quat_wxyz_to_R(np.asarray(wxyz, dtype=np.float64)))

        left_p, left_R = _pose("left")
        right_p, right_R = _pose("right")
        head_p, head_R = _pose("head")
        warn = []
        if not state["home_anchored"]:
            warn.append(
                "the viewer's home-anchor is OFF, so the RAW absolute poses are being "
                "completed: the z anchor now performs a real libsurvive->floor lift and the "
                "robot will not stand on the grid. That is the fidelity view; tick "
                "'Home-anchored +Z up' to put the completion back on the drawn chunk.")
        knobs = dict(knobs)
        if self._dz_wrist is not None and knobs.get("dz_wrist") is None:
            # The caller FK'd the placement, so pin the lift instead of deriving one. Only when
            # the panel has not pinned its own: an operator sweeping dz must still win.
            knobs["dz_wrist"] = self._dz_wrist
        elif (self._dz_wrist is None and knobs.get("dz_wrist") is None
              and str(knobs.get("cond_mode", "")) == "absolute" and state["home_anchored"]):
            warn.append(
                "placement is the viewer's UNCALIBRATED display homes: they put the chest 30 cm "
                "in front of where the G1's chest actually is, leaving the hands only 5 cm "
                "ahead of the root instead of 45 cm, which folds the arms back (elbows behind "
                "the torso). Relaunch with --display-frame g1-ready-pose to place the chunk on "
                "the pose the robot holds after the operator presses A.")
        return CondEpisode.from_poses(
            left_eef_pos=left_p, left_eef_rotmat=left_R,
            right_eef_pos=right_p, right_eef_rotmat=right_R,
            head_pos=head_p, head_rotmat=head_R,
            eef_t_wrist=state["eef_t_wrist"], eef_R_wrist=state["eef_R_wrist"],
            dt=ACTION_DT,
            grip_width=state["grip"].get(series, {}),
            warn=warn,
            src={"flore_series": series, "flore_rollout": state["identity"],
                 "flore_home_anchored": state["home_anchored"],
                 "flore_placement": self._placement_note or "display homes"},
            **knobs,
        )

    def poll(self, action_step: int) -> bool:
        """Advance the streaming loop; True when the scene should be redrawn. Loop thread."""
        if not self.available:
            return False
        return bool(self.overlay.poll(int(action_step)))

    def render_at(self, action_step: int) -> None:
        """Draw the completion at this action step. Loop thread."""
        if not self.available:
            return
        self.overlay.render_at(int(action_step))

    def clear(self) -> None:
        if self.available:
            self.overlay.clear()

    def stop(self) -> None:
        if self.overlay is not None:
            self.overlay.stop()


def add_cli_args(parser) -> None:
    """Add the ``--motion-completion`` flag group to an ``infer_s1_*.py`` parser.

    One definition for all three entry points rather than three copies: the flags decide what
    a MODEL is fed, so a default that drifted between the flow / diffusion / ACT viewers would
    make two runs of "the same" completion incomparable.
    """
    group = parser.add_argument_group(
        "motion completion (whole-body G1, opt-in)",
        "Draw the 29-DoF G1 whole body that the Motion_Completion model produces from the 3 "
        "predicted keypoints, instead of no humanoid at all. Needs a Motion_Completion "
        "checkout, its g1_mosyn conda env and a downloaded checkpoint; a missing piece "
        "disables the panel with a reason rather than failing the launch. Requires a 3-tracker "
        "checkpoint (both hands AND the chest).")
    group.add_argument(
        "--motion-completion", action="store_true",
        help="enable the whole-body completion panel (default: off, page unchanged)")
    group.add_argument(
        "--mc-series", default="prediction", choices=SERIES,
        help="complete the policy's chunk or the dataset's own masked chunk (default: %(default)s)")
    group.add_argument(
        "--mc-cond-mode", default="absolute", choices=("absolute", "home"),
        help="absolute (default) lands the robot's hands ON the drawn chunk, since this "
             "viewer's display frame already sits at the rig's standing heights; home starts "
             "it at the G1's 29-joints-zero pose instead, offset from the chunk")
    group.add_argument("--mc-urdf", default=None, help="override the drawn G1 URDF")
    group.add_argument(
        "--mc-no-wrist-ik", action="store_true",
        help="skip the per-window wrist IK (~20 ms) that lands the commanded hand to 0.3 mm")
    group.add_argument(
        "--mc-no-foot-ik", action="store_true",
        help="skip the causal foot IK. It is ~2 s/window and the reason the humanoid trails "
             "the slider; without it the feet slide (0.031 vs 0.0036 m/s)")
    group.add_argument(
        "--mc-foot-ik-iters", type=int, default=None,
        help="foot-IK Adam steps. A CONVERGENCE THRESHOLD, not a dial: below ~80 the leg IK "
             "never reaches its targets and the pass buys nothing at 13x the cost")


def options_from_args(args) -> dict | None:
    """``MCHumanoid`` kwargs for :func:`run_viewer`, or ``None`` when the flag is absent."""
    if not getattr(args, "motion_completion", False):
        return None
    return {
        "enabled": True,
        "urdf_path": args.mc_urdf,
        "series": args.mc_series,
        "cond_mode": args.mc_cond_mode,
        "wrist_ik": not args.mc_no_wrist_ik,
        "foot_ik": not args.mc_no_foot_ik,
        "foot_ik_iters": args.mc_foot_ik_iters,
    }


__all__ = ["MCHumanoid", "REQUIRED_ROLES", "SERIES", "add_cli_args", "options_from_args"]
