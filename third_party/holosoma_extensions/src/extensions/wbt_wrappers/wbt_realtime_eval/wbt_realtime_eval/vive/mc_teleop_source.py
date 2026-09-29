"""Live 3-tracker VIVE stream -> Motion_Completion -> 29-DoF whole body -> 5pt / dense.

WHY. `exp:g1-wbt-5pt-hybrid-unified-anneal` wants FIVE tracked points (both hands, the
head, and BOTH FEET) and `exp:g1-wbt-hybrid-unified-anneal` wants all 29 joint targets.
The rig has THREE trackers. Motion completion is the bridge: it takes the 3pt take and
produces the complete 29-DoF motion, out of which the extra points are just an FK index.

    3 VIVE pucks ─┐
                  │  yaw-align (the bridge's own world cal)
                  ▼
        wrist / chest poses ──home-anchor──▶ MC gripper_base ──push──▶ [g1_mosyn backend]
                                                                            │ L1Flow
                                                                            │ + wrist IK
                                        qpos(29) + FK(30 bodies)  ◀─────────┘
                                                   │
                        mc_frames.select_keypoints ─┴─▶  5pt keypoints  ──▶ pelvis frame
                                                         or 29 joint targets

Everything about the frame chain is REUSED from `UMI_collection/motion_completion`, the
path the user verified with `./run_replay_zarr_viz.sh <take>.zarr.zip --motion-completion`.
This module adds no geometry of its own: `home_wrist_pose`, `home_anchor_traj`,
`mc_gripper_from_wrist`, `root_cond_from_chest` and `select_keypoints` are all called from
`mc_frames`, so a fix there lands here. The one thing that IS new is the online schedule.

---------------------------------------------------------------------------
THE LATENCY FLOOR, WHICH IS THE WHOLE STORY
---------------------------------------------------------------------------
The checkpoint is WINDOWED and its conditioning is DENSE over the window: window ``w``
covers frames ``[20w, 20w+30)`` and cannot run until frame ``20w+29`` has been observed.
It then returns all 30 of those frames (the first ``overlap=10`` refine what ``w-1``
already gave; the last 10 will be refined again by ``w+1``).

So if we command output frame ``f - D`` at wall frame ``f``, continuity requires the
frame to already exist. Between windows the newest output frame is pinned: for
``f in [20w+29, 20w+48]`` the newest is ``20w+29``. Requiring ``f - D <= 20w+29`` at the
worst case ``f = 20w+48`` gives

        D >= 19 frames = 633 ms at 30 Hz.

`D = 20` (667 ms, one stride) is the smallest clean value and the default here. Below it
the command STALLS between windows instead of running late — worse for a whole-body
controller than honest lag, so `lag_frames` warns rather than silently stuttering.
(`mc_frames.deployment_lag_frames` quotes 30 = one full window, the conservative bound
that also covers the not-yet-refined tail; 20 is what smooth playback actually needs.)

COMPUTE is not the constraint: ~46 ms per window (5pt) against the 667 ms of motion a
window settles, i.e. ~6.9% duty; dense is ~65 ms / 9.8%.

FOOT IK is the constraint, and is therefore OFF here (`foot_fix="none"`). Measured on the
same take with only the flag changed: 1937 ms/window p50 (3092 max) = 291% duty, and the
COMMAND LAG blows out from a constant 700 ms to 6100 ms p50 / 8867 ms max. The duty figure
is the diagnosis but the lag is the answer — no operator closes a loop on 8.9 s, and each
late window pushes the next one later. Note the INPUT stream still reads 0.997x real time
(the tracker feed is accepted on the wall clock); it is the OUTPUT that falls behind, which
is why this needed measuring rather than eyeballing. Not tunable either: below ~80 Adam
steps the leg IK never reaches its targets and the output is ~= the input, so the cheap
settings cost 13x and buy nothing (measured; see the motion_completion README).
Reproduce with `scripts/_mc_teleop_selftest.sh <take>.zarr.zip --foot-ik`.

---------------------------------------------------------------------------
FRAME CHAIN, per MC frame
---------------------------------------------------------------------------
1. Yaw-align the raw puck pose with the bridge's own world cal:
   ``p_yaw = Rz(world_yaw) @ p_raw``, ``q_yaw = qz(world_yaw) (x) q_raw``.
   This is the SAME Rz the bridge applies in `_hand_command`, so "operator +x is robot
   +x" means the same thing in both paths.
2. The puck -> wrist leg (`hand_lever`, see MC_HAND_LEVERS):
   * `none` (default) treats the yaw-aligned PUCK pose as the operator's wrist pose. That is
     exactly the bridge's own model (`_hand_command` maps puck displacement onto the robot
     wrist and composes the world rotation delta onto a home quat) -- so this module inherits
     that approximation rather than inventing a second, different lever. It is EXACT for a
     delta chain (`home`) in rotation and leaves 14-26 mm of rotating-lever residue.
   * `tcp` applies `mc_frames.wrist_from_puck`, the converter's own two legs composed, which
     makes the commanded pose bit-identical to the offline label chain. REQUIRED by
     `cond_mode="absolute"`, where the un-levered puck axes are 180 deg wrong.
3. PLACE the take (`cond_mode`, see MC_COND_MODES):
   * `home` (default) -- re-base onto MC's home with `home_anchor_traj`, whose home is
     `home_wrist_pose(root_p0, root_R0, side)` = the G1's 29-joints-ZERO wrist pose placed
     at the chest puck's frame-0 XY + heading, standing on the floor at PELVIS_STAND_Z.
     The only mode that needs no measured libsurvive-origin height at all.
   * `absolute` -- keep the operator's own yaw-aligned world pose and apply ONE rigid z
     lift, frozen at the calibration frame (`_resolve_dz_locked`; `dz_anchor="hand"` puts
     the frame-0 mounted gripper at RIG_WRIST_STAND_Z, so nothing is measured either).
     This preserves the OPERATOR's hand-to-hand geometry instead of substituting the
     robot's 0.297 m, and it is the placement `mc_to_hl_wholebody.py --cond-mode absolute`
     prepares `diffusion_umi_wholebody` labels in -- which is why a COLLECTION run wants
     it and a plain teleop session does not care.
4. Apply the gripper mount: `mc_gripper_from_wrist`. The model was trained with
   `wrist_body=gripper_base`, and `wrist_ik_jax` applies the mount internally, so the
   backend must be fed the GRIPPER pose, not the wrist pose.
5. Relabel the CHEST puck into the chest BODY frame: `chest_body_from_puck`. ⚠️ This is the
   ONE absolute puck rotation in the chain -- everything else is a calibration-relative
   delta, which cancels a constant mount, so the chest bracket's rotation does not cancel
   and must be removed explicitly. The raw puck's +x is the chest body's +y, i.e. the
   operator's LEFT: measured 90.09 deg of heading error, which turns every commanded hand
   displacement by -90 deg (operator FORWARD -> the robot's RIGHT). `head_frame="body"`
   skips it, for a caller whose third pose came out of a converted zarr and is already
   relabelled -- the relabel has ORDER 3, so applying it twice does not cancel.
6. Root cond from the CHEST: `root_cond_from_chest(head_p, head_R, dz=0)` -- position
   plus a YAW-ONLY rotation, because the model reads only XY + heading off that slot.

Output, per commanded frame:
  * 5pt   -> `select_keypoints(mode="5pt", hand_point="tcp")` then relativized into the
             PELVIS frame, which is the frame `holosoma_wbt._compute_vr_fk` produces:
             ``p_local = R(q_pelvis)^T (p - p_pelvis)``, ``q_local = q_pelvis^-1 (x) q``.
             ⚠️ WHICH pelvis is THE load-bearing choice in this module, and it is
             `mc_frames.pelvis_ref_pose` FROZEN at cond frame 0 -- the same frame the home
             wrist was placed in -- NOT the completion's per-frame root. The root cond IS
             the chest puck, so a moving readout frame cancels the operator's own body
             motion: measured, a 300 mm walk moved the commanded points 7.3 mm. Read
             `_readout_frame_locked` before touching it; `MC_REF_FRAME=live` restores the
             old behaviour for an A/B and says so loudly at startup.
  * dense -> the 29 joint angles remapped onto holosoma's joint order BY NAME, plus a
             finite-difference velocity. Never index-aligned: MC's `g1_fk_params.npz`
             order and the policy's `policy_joint_names` are both real orders and they
             are NOT the same.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# the model's window schedule (fixed by the checkpoint, not tunable here)
# ---------------------------------------------------------------------------
MC_RATE_HZ = 30.0
MC_WINDOW = 30
MC_OVERLAP = 10
MC_STRIDE = MC_WINDOW - MC_OVERLAP  # 20

#: Smallest lag that still plays back CONTINUOUSLY -- see the module docstring.
MC_MIN_SMOOTH_LAG_FRAMES = MC_STRIDE - 1  # 19

#: Default episode length handed to the backend. It only sizes the output ring and
#: enumerates the window starts; nothing is precomputed, so a generous value is free.
#: 30 Hz * 20 min. `reset()` starts a fresh episode, so a session is never truncated by
#: accident without a loud message.
MC_DEFAULT_EPISODE_FRAMES = int(30 * 60 * 20)

_DEFAULT_UMI_DIR = os.path.expanduser("~/FUJI/third_party/UMI_collection")

#: Which pelvis frame the 5pt keypoints are read out in. `frozen` (the default) is the fixed
#: reference pelvis captured at cond frame 0; `live` is the completion's own per-frame root,
#: which is what shipped before 2026-08-29 and what TRAINING sees. The full argument, with the
#: measured numbers, is in `MotionCompletionSource._readout_frame_locked` -- one place, because
#: this is the single decision that determines whether the operator's body motion reaches the
#: robot at all. Irrelevant to `--mode dense`: joint angles carry no pelvis.
MC_REF_FRAMES = ("frozen", "live")
MC_DEFAULT_REF_FRAME = "frozen"

#: What the caller's THIRD pose already is. `puck` = the raw libsurvive chest puck, so the
#: chest's rotation-only body relabel still has to be applied here (the live rig: there is no
#: converter in front of the bridge). `body` = an already-relabelled chest pose, which is what
#: a converted `.zarr.zip` stores and therefore what `mc_episode.CondEpisode` hands over — the
#: offline drivers pass this so the relabel is not applied twice. It has ORDER 3, so a double
#: application does NOT cancel; it lands 120 deg away. See `mc_frames.CHEST_RELABEL_R`.
MC_HEAD_FRAMES = ("puck", "body")
MC_DEFAULT_HEAD_FRAME = "puck"

#: WHERE the operator's take is placed in the robot's world -- the same decision
#: `mc_episode.CondEpisode.COND_MODES` makes offline, with the same two names, because it is
#: one decision and mixing the two spellings is how a session gets collected in one posture
#: space and trained in the other.
#:
#: `home` (the default, and what shipped) re-bases each wrist onto the G1's 29-joints-ZERO
#: pose placed at the chest's frame-0 XY + heading and transfers only the operator's CHANGE
#: from there. Nothing has to be guessed: the floor is the robot's own. The cost is that the
#: operator's hand-to-hand offset is REPLACED by the robot's 0.297 m.
#:
#: ⚠️ It is also the ONLY placement that STARTS WHERE THE ROBOT IS. Frame 0's hands land on
#: the G1's own zero-config TCP geometry, which IS `_compute_vr_fk` of the READY POSE to
#: 0.1 mm — so arming moves nothing. In `absolute`, frame 0 is wherever the operator's hands
#: happen to be: measured on a real take, 0.2-0.5 m from the ready pose per hand, with the
#: operator's 0.437 m hand separation against the robot's 0.297 m, so it CANNOT be made to
#: coincide by standing differently. That step is a lunge on a whole-body controller.
#:
#: `absolute` keeps the operator's own world poses and applies ONE rigid z lift (`dz_anchor`
#: below), so the hand-to-hand and hand-to-chest geometry is the OPERATOR's, and it needs no
#: anchor reference to reproduce offline. ⚠️ It REQUIRES `hand_lever="tcp"` and is refused
#: without it: absolute is precisely the mode in which the puck->wrist relabel does NOT
#: cancel, and un-levered it commands every hand orientation 180 deg wrong (measured; the rig
#: symptom is the arms thrashing the instant teleop is armed). See MC_HAND_LEVERS.
#:
#: WHICH ONE A COLLECTION RUN WANTS. Whichever the LABELS are prepared in — running the rig in
#: one placement while `mc_to_hl_wholebody.py` prepares the joint targets in the other puts
#: every recorded image beside a target from a different posture, and nothing downstream
#: compares them. For teleop that is `home` on both sides (`--cond-mode home`, which is also
#: that script's default, and no dz to copy because the lift is exactly 0). `absolute` is what
#: the HANDHELD-rig datasets (`0902_merged3`) were prepared in, so a teleop session collected
#: in `home` is a different posture space and cannot be merged with them — `attrs.wb_cond_mode`
#: is in FUJI's exact-equality `dataset_knobs`, so the two refuse to share norm stats.
#:
#: ⚠️ Either placement is safe on the wire only because DENSE publishes JOINT ANGLES, which are
#: pelvis-relative: the completion's root follows the operator's chest, so where the operator
#: physically stands in the libsurvive room never reaches the robot. A wire that published
#: absolute positions would command the operator's room coordinates.
MC_COND_MODES = ("home", "absolute")
MC_DEFAULT_COND_MODE = "home"

#: With `cond_mode="absolute"`, WHICH measurement fixes the one rigid z lift. Commanded hand
#: HEIGHT is a conditioned channel of the completion, so this is what decides the posture
#: (hands at 0.645 m -> a 0.451 m pelvis crouch; the standing band is 0.77-0.84).
#:   `hand`  -- put the frame-0 mounted gripper z at `mc_frames.RIG_WRIST_STAND_Z` (0.95).
#:              Assumes a hand height, which is exactly what makes it reproducible.
#:   `chest` -- put the chest puck at `RIG_CHEST_STAND_Z` (1.30). The chest's height is fixed
#:              by the operator STANDING rather than re-chosen every second, so offline this
#:              is the better anchor; online it is offered for the same A/B.
#: `mc_episode`'s third value, `train`, is deliberately NOT offered: it reproduces MC's own
#: training-distribution lift, which on this rig under-lifts and crouches the robot (measured
#: pelvis 0.45 m vs 0.79 standing). Offline that is a diagnostic; on a real robot it is a
#: posture nobody asked for.
MC_DZ_ANCHORS = ("hand", "chest")
MC_DEFAULT_DZ_ANCHOR = "hand"

#: Whether to apply the puck->wrist leg (step 2 of the frame chain) before the placement.
#:
#: `none` (the default, and what shipped) treats the yaw-aligned PUCK pose as the operator's
#: wrist — the bridge's own long-standing model. That is exact for a chain built out of
#: calibration-relative DELTAS, because the leg is a constant body-local transform and
#: cancels: `mc_frames.PUCK_R_WRIST @ PUCK_R_WRIST.T == I`. MEASURED on 802 frames of
#: `session_2026.09.02_22.38.14/take_000`, online-vs-offline commanded gripper pose:
#:
#:     cond_mode   rotation err        position err
#:     home          0.0 deg           14-26 mm      (the rotating-lever residue only)
#:     absolute    180.0 deg           79-86 mm      (nothing cancels)
#:
#: `tcp` applies `mc_frames.wrist_from_puck`, the converter's own two legs composed, so the
#: commanded pose becomes bit-identical to what `mc_episode` builds from the SAME motion out
#: of a converted zarr: running both real implementations over 240 frames of a recorded take
#: in `home` with the same anchor agrees to **0.0000 mm / 0.0000 deg** (and the leg itself is
#: 1.1e-16 m / 2.4e-06 deg against the converter chain). It is also the physically true
#: statement: the puck sits a constant 52.8 mm off the wrist, so without it an operator's
#: wrist TWIST swings the robot's hand ~5 cm.
#:
#: OFF by default because it changes the FEEL of a validated teleop path by up to 26 mm, and
#: ON is what a DATA-COLLECTION run wants: the offline label prep has no choice but to use the
#: true wrist (that is what the zarr stores), so `tcp` is what makes the joint targets beside
#: each recorded image the ones the robot was actually driven with.
MC_HAND_LEVERS = ("none", "tcp")
MC_DEFAULT_HAND_LEVER = "none"


def _load_mc_modules(umi_dir: str):
    """Import `mc_frames` / `mc_backend_client` out of the UMI_collection checkout.

    They are NOT installed in this env and never will be -- they belong to a peer repo
    with its own venv. Both are numpy-only precisely so either side can import the one
    copy instead of carrying two that drift, so a path import is the intended use. The
    modules are inserted under their own names because `mc_backend_client` does
    `import mc_wire`, i.e. it expects its own directory on `sys.path`.
    """
    pkg = os.path.join(umi_dir, "motion_completion")
    if not os.path.isdir(pkg):
        raise FileNotFoundError(
            f"no motion_completion/ under {umi_dir} -- set MC_UMI_DIR / --mc-umi-dir to the "
            f"UMI_collection checkout (it holds the frame chain this mode is built on)"
        )
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    out = []
    for name in ("mc_wire", "mc_frames", "mc_backend_client"):
        if name in sys.modules:
            out.append(sys.modules[name])
            continue
        spec = importlib.util.spec_from_file_location(name, os.path.join(pkg, f"{name}.py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {pkg}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        out.append(mod)
    return out[1], out[2]  # mc_frames, mc_backend_client


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def wire_payload(
    mode: str,
    got: dict,
    *,
    seq: int,
    t: float,
    orientation: bool = True,
    publish_qpos: bool = False,
) -> dict:
    """The `:8770` frame for a completion command — the ONE definition of that format.

    Both producers call this: the live bridge (`ViveTeleopBridge._mc_snapshot`) and the
    trackerless replay driver (`vive/mc_replay_bridge.py`). Two copies would defeat the
    point of the replay driver, which is to exercise the browser/robot consumers with the
    payload they will actually get — a sim fed a subtly different shape proves nothing.

    Args:
      mode: ``5pt`` or ``dense``.
      got: a `MotionCompletionSource.latest()` result.
      seq / t: the producer's sequence number and wall time.
      orientation: the bridge's orientation toggle, echoed for the consumer.
      publish_qpos: also ship the WHOLE completed body — 29 joint angles + their names +
        the completion's own root pose in the readout frame — for a VIEWER to draw. Purely
        additive and OFF by default, which keeps the payload byte-identical for every
        existing consumer. Valid in BOTH modes: what it feeds (a reference ghost of the
        completion) is exactly as useful against a dense policy as against a 5pt one.
    """
    if mode == "5pt":
        out = {
            "seq": seq,
            "t": t,
            "mode": "5pt",
            "pos": got["pos"],  # 15 = [L,R,head,Lfoot,Rfoot] xyz, pelvis frame
            "orn": got["orn"],  # 20 = 5x xyzw quaternion
            "orientation": orientation,
            "mc_frame": got["frame"],
            "mc_lag_s": got["lag_s"],
        }
    elif mode == "dense":
        out = {
            "seq": seq,
            "t": t,
            "mode": "dense",
            # 29 joint targets + velocities, in the COMPLETION's joint order. The consumer
            # MUST remap by name -- `joint_names` is shipped so it cannot index-align.
            "joint_pos": got["joint_pos"],
            "joint_vel": got["joint_vel"],
            "joint_names": got["joint_names"],
            "orientation": orientation,
            "mc_frame": got["frame"],
            "mc_lag_s": got["lag_s"],
        }
    else:
        raise ValueError(f"mode must be 5pt or dense, got {mode!r}")
    if publish_qpos:
        # `mc_joint_names` rides along every frame for the same reason the dense payload
        # ships `joint_names`: MC's joint order and the policy's are both real orders and
        # are NOT the same, so a consumer must never be able to index-align by accident —
        # and one that connects late gets a self-describing frame instead of having to
        # have seen the first one.
        #
        # `mc_root_pos`/`mc_root_quat` are what make the body a BODY: without a root, a
        # viewer can only flex the joints in place, so the one thing this exists to show —
        # the completion taking a STEP — is exactly the thing it could not draw. They are
        # in the readout frame, the same frame the 5pt keypoints are in, so the ghost and
        # the command are the same picture (`_root_in_readout_frame_locked`).
        #
        # The keys are the SAME FOUR IN BOTH MODES even though dense's `mc_qpos` duplicates
        # its own `joint_pos`. 29 floats at 50 Hz is nothing next to the alternative: one
        # array that means "what the ghost draws" in one mode and "what the policy is
        # commanded" in the other is precisely the conflation this ghost exists to dispel.
        out["mc_qpos"] = got["joint_pos"]
        out["mc_joint_names"] = got["joint_names"]
        out["mc_root_pos"] = got["root_pos"]
        out["mc_root_quat"] = got["root_quat"]
    return out


class MotionCompletionSource:
    """Streaming 3pt-trackers -> 29-DoF -> keypoints, on the model's own window loop.

    Thread model: `push()` is called from the bridge's pysurvive/main loop and never
    blocks on inference -- it only appends a cond frame and (cheaply) hands the window
    index to a worker. `_worker` runs windows in order, which the backend REQUIRES (it
    seeds window w from w-1's tail, and running out of order raises rather than silently
    stepping at the seam). `latest_*` reads the settled ring under the same lock.
    """

    def __init__(
        self,
        *,
        keypoint_mode: str = "5pt",
        hand_point: str = "tcp",
        lag_frames: int = MC_STRIDE,
        wrist_ik: bool = True,
        foot_fix: str = "none",
        ref_frame: str | None = None,
        head_frame: str = MC_DEFAULT_HEAD_FRAME,
        cond_mode: str | None = None,
        dz_anchor: str = MC_DEFAULT_DZ_ANCHOR,
        dz_wrist: float | None = None,
        hand_lever: str | None = None,
        umi_dir: str | None = None,
        episode_frames: int = MC_DEFAULT_EPISODE_FRAMES,
        log=print,
    ) -> None:
        self._umi_dir = os.path.abspath(umi_dir or os.environ.get("MC_UMI_DIR") or _DEFAULT_UMI_DIR)
        self._F, self._C = _load_mc_modules(self._umi_dir)
        if keypoint_mode not in ("5pt", "dense"):
            raise ValueError(f"keypoint_mode must be 5pt or dense, got {keypoint_mode!r}")
        self._kp_mode = keypoint_mode
        # Precedence: explicit argument > env > default. The env pin exists so ONE process can
        # be flipped without a launcher edit (the same reason FUJI_FORCE_HIST_LEN does), which
        # is what an A/B on the rig or in the browser sim needs; an unknown value RAISES rather
        # than falling back, because silently reading out in the other frame is invisible until
        # the operator notices the robot ignoring them.
        self._ref_frame = str(ref_frame or os.environ.get("MC_REF_FRAME") or MC_DEFAULT_REF_FRAME)
        if self._ref_frame not in MC_REF_FRAMES:
            raise ValueError(
                f"ref_frame must be one of {MC_REF_FRAMES}, got {self._ref_frame!r} "
                f"(MC_REF_FRAME={os.environ.get('MC_REF_FRAME')!r})"
            )
        # No env override on purpose: unlike `ref_frame` there is no defensible alternative
        # here — whether the caller's third pose is a raw puck or an already-relabelled chest
        # is a FACT about that caller, not a preference, so it is an argument and an unknown
        # value raises. The measured cost of getting it wrong is 90 deg (see CHEST_RELABEL_R).
        self._head_frame = str(head_frame)
        if self._head_frame not in MC_HEAD_FRAMES:
            raise ValueError(f"head_frame must be one of {MC_HEAD_FRAMES}, got {head_frame!r}")
        # Same precedence as `ref_frame` (argument > env > default) and for the same reason: a
        # COLLECTION session has to be run in the placement its labels will be prepared in,
        # and that is a per-process decision an operator makes at launch. `train` is rejected
        # by the membership check rather than silently mapped -- see MC_DZ_ANCHORS.
        self._cond_mode = str(cond_mode or os.environ.get("MC_COND_MODE") or MC_DEFAULT_COND_MODE)
        if self._cond_mode not in MC_COND_MODES:
            raise ValueError(
                f"cond_mode must be one of {MC_COND_MODES}, got {self._cond_mode!r} "
                f"(MC_COND_MODE={os.environ.get('MC_COND_MODE')!r})"
            )
        self._dz_anchor = str(dz_anchor or MC_DEFAULT_DZ_ANCHOR)
        if self._dz_anchor not in MC_DZ_ANCHORS:
            raise ValueError(f"dz_anchor must be one of {MC_DZ_ANCHORS}, got {dz_anchor!r}")
        # Same precedence again (argument > env > default). See MC_HAND_LEVERS.
        self._hand_lever = str(
            hand_lever or os.environ.get("MC_HAND_LEVER") or MC_DEFAULT_HAND_LEVER)
        if self._hand_lever not in MC_HAND_LEVERS:
            raise ValueError(
                f"hand_lever must be one of {MC_HAND_LEVERS}, got {self._hand_lever!r} "
                f"(MC_HAND_LEVER={os.environ.get('MC_HAND_LEVER')!r})"
            )
        # REFUSED, not warned. `absolute` keeps the operator's ABSOLUTE hand pose, so the
        # constant puck->wrist relabel no longer cancels the way it does in a delta chain, and
        # the un-levered pose is a 180-deg-wrong hand orientation at EVERY frame. Measured on
        # 802 frames of session_2026.09.02_22.38.14/take_000, online vs the offline chain fed
        # the same motion: 180.0 deg + 79-86 mm in `absolute`, 0.0 deg + 14-26 mm in `home`.
        # The rig symptom is the arms thrashing the moment teleop is armed, which reads as a
        # model problem rather than a frame one -- so this cannot be a warning.
        if self._cond_mode == "absolute" and self._hand_lever == "none":
            raise ValueError(
                "cond_mode='absolute' requires hand_lever='tcp' (MC_HAND_LEVER=tcp).\n"
                "  ABSOLUTE feeds the operator's own hand POSE, so the puck->wrist relabel "
                "does not cancel the way it does in HOME's calibration-relative delta: the "
                "raw puck's axes are the measured -y,-x,-z relabel of the wrist's, i.e. a "
                "180-deg rotation. Measured online-vs-offline on 802 real frames: "
                "absolute 180.0 deg / 79-86 mm, home 0.0 deg / 14-26 mm.\n"
                "  Either add MC_HAND_LEVER=tcp (mc_frames.wrist_from_puck, which makes the "
                "commanded pose bit-identical to what mc_to_hl_wholebody.py builds), or use "
                "cond_mode='home' -- which is also the only placement that starts ON the "
                "pre-arm gate's reference, so arming moves nothing."
            )
        # A PINNED lift, which is the only way to make the online and the offline completion
        # place the take identically. Offline `auto_dz_wrist` averages over the WHOLE episode
        # (`n_frames=0`); online there is exactly one frame to average -- the calibration
        # frame -- so the two auto values differ by the operator's own hand-height drift over
        # the take. The recipe is therefore: run once, read the `[mc] absolute placement` line,
        # then pass that number to BOTH this and `mc_to_hl_wholebody.py --dz-wrist`.
        self._dz_pin = None if dz_wrist is None else float(dz_wrist)
        self._hand_point = hand_point
        self._wrist_ik = bool(wrist_ik)
        self._foot_fix = str(foot_fix)
        self._n = int(episode_frames)
        self._log = log
        self._lag = int(lag_frames)
        if self._lag < MC_MIN_SMOOTH_LAG_FRAMES:
            self._log(
                f"[mc] WARNING lag_frames={self._lag} < {MC_MIN_SMOOTH_LAG_FRAMES}: the "
                f"command will STALL between windows rather than run late. The floor is a "
                f"CHECKPOINT property (window {MC_WINDOW}, stride {MC_STRIDE}, dense cond), "
                f"not a tuning knob -- see mc_teleop_source's module docstring."
            )

        self._backend = self._C.MCBackend(log=self._log)
        self._starts: list[int] = []
        self._lock = threading.Lock()
        # ORDERS BACKEND REQUESTS AGAINST reset(), and nothing else. `_drain_cond` sends its
        # batch with `_lock` RELEASED on purpose (a ~37 ms round trip under the state lock
        # stalls the tracker loop -- a 12.4 s take once took 42 s), which leaves a window where
        # a `reset()` can slip between the send and the `_sent` bookkeeping. Then
        # `start_episode` empties the backend's cond buffer while frames from the PREVIOUS
        # episode are still in flight into it, and the counts diverge by exactly the batch:
        # measured as `window 0 needs cond frames up to 30, only 29 pushed`, after which every
        # later window fails "streaming is in order" and the whole episode publishes nothing.
        # Reachable from the rig too -- a Recalibrate mid-motion is the same mid-stream reset.
        #
        # ALWAYS take this one BEFORE `_lock`; `push()` never takes it, so the tracker thread
        # still cannot be blocked behind inference.
        self._io_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._cond_q: deque[tuple[int, tuple]] = deque()
        self._pending_ev = threading.Event()

        # --- episode state (reset() clears all of it) ------------------------
        self._t0: float | None = None
        self._pushed = 0  # cond frames the frame clock has allocated
        self._sent = 0  # cond frames the WORKER has actually delivered to the backend
        self._anchor: dict | None = None  # frame-0 home anchors, per side
        self._raw0: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # The rigid z lift, frozen at cond frame 0 and applied to every hand AND to the root
        # cond. Exactly 0.0 in `home` mode, where the home anchor already puts the take in the
        # robot's world with the floor at z=0 -- so ONE code path does both placements.
        self._dz = 0.0
        self._next_window = 0
        # settled output ring, absolute-frame indexed
        self._fk_pos: np.ndarray | None = None
        self._fk_quat: np.ndarray | None = None
        self._joint_ang: np.ndarray | None = None
        self._root_pos: np.ndarray | None = None
        self._root_quat: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._newest = -1
        self._body_names: list[str] = []
        self._joint_names: list[str] = []
        self._win_ms: deque[float] = deque(maxlen=32)
        self._win_count = 0
        self._err_count = 0
        self._last_err = ""

    # -- lifecycle -----------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        paths, reason = self._C.resolve()
        return (not reason), reason

    def start(self) -> bool:
        ok, reason = self.available()
        if not ok:
            self._log(f"[mc] UNUSABLE: {reason}")
            return False
        if not self._backend.start():
            self._log("[mc] backend failed to start")
            return False
        self._body_names, self._joint_names = self._F.load_mc_names(self._C.resolve()[0]["mc_dir"])
        if len(self._body_names) != 30 or len(self._joint_names) != 29:
            raise RuntimeError(
                f"unexpected FK params: {len(self._body_names)} bodies / "
                f"{len(self._joint_names)} joints (want 30 / 29)"
            )
        self.reset()
        self._worker = threading.Thread(target=self._run_worker, name="mc-windows", daemon=True)
        self._worker.start()
        self._log(
            f"[mc] streaming completion ONLINE: mode={self._kp_mode} hand_point={self._hand_point} "
            f"wrist_ik={self._wrist_ik} foot_fix={self._foot_fix} lag={self._lag} frames "
            f"({self._lag / MC_RATE_HZ * 1e3:.0f} ms at {MC_RATE_HZ:g} Hz) "
            f"head_frame={self._head_frame} cond_mode={self._cond_mode} "
            f"hand_lever={self._hand_lever}"
            + (f" dz_anchor={self._dz_anchor}" if self._cond_mode == "absolute" else "")
            + (f" readout_frame={self._ref_frame}" if self._kp_mode == "5pt" else "")
        )
        if self._hand_lever == "tcp":
            self._log(
                "[mc] hand_lever TCP: the puck->wrist leg (mc_frames.wrist_from_puck, 52.8 mm "
                "of rotating lever) is applied before the placement, so the commanded gripper "
                "pose is bit-identical to what mc_to_hl_wholebody.py builds from the same "
                "motion out of a converted zarr. This is what a COLLECTION run wants. It "
                "moves the commanded hand by up to 26 mm vs the shipped `none`."
            )
        else:
            self._log(
                "[mc] hand_lever NONE: the yaw-aligned PUCK is fed where the WRIST belongs "
                "(the bridge's long-standing model). Exact in rotation for cond_mode=home "
                "because the leg cancels in the delta; leaves a 14-26 mm rotating-lever "
                "difference against the offline label chain. MC_HAND_LEVER=tcp removes it."
            )
        if self._cond_mode == "absolute":
            self._log(
                "[mc] cond_mode ABSOLUTE: the take keeps the OPERATOR's own hand-to-hand and "
                "hand-to-chest geometry and is placed by ONE rigid z lift, resolved at the "
                "calibration pose (printed as '[mc] absolute placement' below). This is the "
                "placement `mc_to_hl_wholebody.py --cond-mode absolute` prepares labels in, so "
                "it is what a diffusion_umi_wholebody COLLECTION run wants -- and it means the "
                "posture now depends on how high the operator holds their hands while "
                "calibrating, which `home` did not."
            )
        if self._kp_mode == "5pt" and self._cond_mode == "absolute":
            self._log(
                "[mc] ⚠️  absolute + 5pt: the keypoint wire publishes POSITIONS, and this "
                "placement keeps the operator's world XY. The readout is still the frozen "
                "reference pelvis (so the command stays pelvis-relative and bounded), but the "
                "dense wire -- joint angles, which carry no position at all -- is what this "
                "mode was measured for."
            )
        if self._kp_mode == "5pt" and self._ref_frame == "live":
            self._log(
                "[mc] ⚠️  readout frame LIVE: the keypoints are relative to the completion's "
                "OWN moving root, so the operator's body translation CANCELS (measured: a "
                "300 mm walk moves the commanded points 7.3 mm). This is the pre-2026-08-29 "
                "behaviour, kept for A/B only -- see _readout_frame_locked."
            )
        return True

    def stop(self) -> None:
        self._stop.set()
        self._pending_ev.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        self._backend.stop()

    def reset(self) -> None:
        """Start a fresh MC episode: new home anchor, empty cond, empty output.

        Called on (re)calibration. The home anchor IS frame 0, so a Recalibrate that did
        not reset here would keep re-basing onto a stale operator pose and the completed
        motion would jump by the operator's drift since the last calibration.

        Takes `_io_lock` first, so a cond batch already in flight lands BEFORE the new episode
        starts instead of into it — see the note on `_io_lock`. This is why a mid-stream reset
        can block for up to one backend round trip (~37 ms).
        """
        with self._io_lock, self._lock:
            self._backend.start_episode(
                self._n, wrist_ik=self._wrist_ik, foot_fix=self._foot_fix
            )
            self._starts = self._F.window_starts(self._n, MC_WINDOW, MC_OVERLAP)
            self._t0 = None
            self._pushed = 0
            self._sent = 0
            self._anchor = None
            self._raw0 = {}
            self._dz = 0.0  # re-derived at the new cond frame 0 (a Recalibrate re-places)
            self._next_window = 0
            self._newest = -1
            self._cond_q.clear()
            self._fk_pos = np.zeros((self._n, 30, 3), dtype=np.float32)
            self._fk_quat = np.zeros((self._n, 30, 4), dtype=np.float32)
            self._joint_ang = np.zeros((self._n, 29), dtype=np.float32)
            self._root_pos = np.zeros((self._n, 3), dtype=np.float32)
            self._root_quat = np.zeros((self._n, 4), dtype=np.float32)
            self._valid = np.zeros(self._n, dtype=bool)
        self._log(f"[mc] episode reset ({self._n} frames, {len(self._starts)} windows)")

    # -- ingest --------------------------------------------------------------
    def push(
        self,
        world_yaw: float,
        left: tuple[np.ndarray, np.ndarray],
        right: tuple[np.ndarray, np.ndarray],
        head: tuple[np.ndarray, np.ndarray],
        now: float | None = None,
    ) -> int:
        """Feed one tracker sample. Returns the MC frame index it landed on, or -1.

        The bridge's loop runs faster than MC's 30 Hz, so the MC frame clock is driven off
        WALL TIME (`frame = floor((now - t0) * 30)`) and the newest sample wins within a
        frame -- a zero-order hold, which is what a 50 Hz stream feeding a 30 Hz model
        has to do. Falling BEHIND (a gap > 1 frame, e.g. a tracker dropout) repeats the
        current sample to fill, because a hole in the cond buffer would silently shift
        every later frame's timestamp by the size of the hole.
        """
        now = time.time() if now is None else float(now)
        with self._lock:
            if self._fk_pos is None:
                return -1
            if self._t0 is None:
                self._t0 = now
            # Frames that SHOULD exist by now. ROUND, not floor-plus-epsilon:
            # `(t0 + i/30) - t0` for a unix-epoch t0 (~1.8e9) only resolves to ~2.4e-7 s
            # in float64, so an exactly-1/30-spaced caller lands at i +- 4e-6 FRAMES and a
            # floor drops the frame roughly a third of the time -- measured, 50 of 371
            # samples silently folded. Rounding needs the error under half a frame, which
            # 4e-6 clears by five orders of magnitude. It also makes the resample
            # NEAREST-neighbour rather than floor, which is the better choice for a 50 Hz
            # tracker feed into a 30 Hz model (lower mean timing error, +-16.7 ms bound).
            want = int(round((now - self._t0) * MC_RATE_HZ)) + 1
            if want <= self._pushed:
                return self._pushed - 1
            if self._pushed >= self._n:
                return -1
            n_new = min(want - self._pushed, self._n - self._pushed)
            grip = self._cond_frame_locked(world_yaw, left, right, head)
            if grip is None:
                return -1
            # HAND OFF, DO NOT SEND. `MCBackend` serialises every request on one lock and
            # a `run_window` round trip is ~37 ms (it ships 30x30 poses back over a pipe),
            # so sending from here would block the tracker loop behind inference -- and did:
            # a 12.4 s take took 42 s of wall clock before this. The worker owns all
            # backend I/O; this thread only does numpy.
            self._cond_q.append((n_new, grip))
            self._pushed += n_new
            self._pending_ev.set()
            return self._pushed - 1

    def _cond_frame_locked(self, world_yaw, left, right, head):
        """One conditioning frame, in the model's own frame. Caller holds the lock."""
        F = self._F
        cz, sz = np.cos(world_yaw), np.sin(world_yaw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        qz = np.array([np.cos(world_yaw / 2.0), 0.0, 0.0, np.sin(world_yaw / 2.0)], dtype=np.float64)

        def align(pq):
            p, q = pq
            p = Rz @ np.asarray(p, dtype=np.float64).reshape(3)
            q = quat_mul_wxyz(qz, np.asarray(q, dtype=np.float64).reshape(4))
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            return p, F.quat_wxyz_to_R(q[None, :])[0]

        Lp, LR = align(left)
        Rp, RR = align(right)
        Hp, HR = align(head)

        # STEP 2 of the frame chain: the puck is not the wrist. Applied HERE — before `_raw0`
        # is captured and before either placement — so `home`'s delta is taken between two
        # WRIST poses rather than two puck poses; the constant part cancels either way, the
        # ROTATING part (52.8 mm of lever) does not. The chest is deliberately excluded: it is
        # a body reference with no wrist and no TCP, and its own relabel is applied below.
        if self._hand_lever == "tcp":
            pw, Rw = F.wrist_from_puck(np.stack([Lp, Rp]), np.stack([LR, RR]))
            Lp, LR, Rp, RR = pw[0], Rw[0], pw[1], Rw[1]

        # THE CHEST BODY RELABEL. `root_cond_from_chest` reads this pose's HEADING, and the
        # heading is the only ABSOLUTE puck rotation anywhere in this chain -- every other
        # rotation is a calibration-relative delta, which cancels a constant mount. So the
        # chest puck's bracket rotation does NOT cancel here, and the raw puck's +x axis is
        # the chest body's +y (the operator's LEFT): measured 90.09 deg off, std 0.41 over 979
        # frames, against 0.20 deg for the relabelled pose. Getting it wrong turns every
        # commanded hand displacement by -90 deg -- operator FORWARD comes out as the robot's
        # RIGHT -- while leaving frame 0 exact, because the home and the readout frame share
        # the same wrong heading. That is why this was invisible until someone walked.
        # The relabel is body-local, so it commutes with the world-frame `Rz` above.
        HR_in = HR
        if self._head_frame == "puck":
            Hp, HR = F.chest_body_from_puck(Hp, HR)

        # Root cond: chest position + YAW-ONLY rotation (the model reads XY + heading).
        root_p, root_R = F.root_cond_from_chest(Hp[None, :], HR[None, :], dz=0.0)

        if self._anchor is None:
            # Frame 0: MC's home is the 29-joints-ZERO wrist pose placed at the chest's
            # XY + heading. Nothing here is a guessed origin height -- that is the whole
            # reason cond_mode="home" is the default.
            #
            # Built in BOTH cond modes even though `absolute` never re-bases a hand onto it:
            # `pelvis` is the 5pt readout frame and is a function of the root's XY + heading
            # only, which the z lift does not touch, so the two modes share it exactly.
            #
            # `pelvis` is the SAME frame those two homes are expressed in (they are literally
            # it plus the zero-config offset -- `home_wrist_pose` calls `pelvis_ref_pose`),
            # and it is the frozen frame the 5pt readout relativises into. Capturing it here
            # rather than recomputing it per frame is what makes it frozen: it is reset only
            # by `reset()`, i.e. only by a Recalibrate. See `_readout_frame_locked`.
            self._anchor = {
                "left": F.home_wrist_pose(root_p[0], root_R[0], "left"),
                "right": F.home_wrist_pose(root_p[0], root_R[0], "right"),
                "pelvis": F.pelvis_ref_pose(root_p[0], root_R[0]),
            }
            self._raw0 = {"left": (Lp.copy(), LR.copy()), "right": (Rp.copy(), RR.copy())}
            self._log(
                f"[mc] home anchor set: L={np.round(self._anchor['left'][0], 4).tolist()} "
                f"R={np.round(self._anchor['right'][0], 4).tolist()} "
                f"pelvis={np.round(self._anchor['pelvis'][0], 4).tolist()} "
                f"(pelvis stand z={F.PELVIS_STAND_Z:.5f}, readout frame={self._ref_frame})"
            )
            # The heading is EVIDENCE, not decoration: after the world-yaw autocal the
            # operator faces robot +x, so `body` must read ~0 deg. A `body` heading near
            # +-90 is the chest relabel missing or applied twice, and the only symptom
            # downstream is that the operator's motion comes out turned -- which is not
            # something an operator can attribute without this line.
            self._log(
                f"[mc] chest heading: in={np.degrees(float(F.yaw_of_R(HR_in))):+7.2f} deg "
                f"-> body={np.degrees(float(F.yaw_of_R(HR))):+7.2f} deg "
                f"(head_frame={self._head_frame}; body should be ~0 after the world-yaw "
                f"autocal -- see mc_frames.CHEST_RELABEL_R)"
            )
            if self._cond_mode == "absolute":
                self._dz = self._resolve_dz_locked(Lp, LR, Rp, RR, Hp)

        # The lift is z-only, so applying it to the root here (rather than passing dz into
        # `root_cond_from_chest` above) is the same number and keeps the home branch's
        # `home_wrist_pose(root_p[0], ...)` reading the UNLIFTED root it has always read.
        root_p = F.lift_z(root_p, self._dz)

        out = []
        for side, p_now, R_now in (("left", Lp, LR), ("right", Rp, RR)):
            if self._cond_mode == "home":
                p0, R0 = self._raw0[side]
                p_home, R_home = self._anchor[side]
                # Call the SHARED implementation on a 2-frame [frame0, now] stack and take
                # index 1, rather than restating p_home + (p - p0) / (R R0^T) R_home here.
                # The side R_home is multiplied on has been got wrong before (it reads as a
                # constant tilt, not an offset), so there is exactly one copy of that formula.
                p_w, R_w = F.home_anchor_traj(
                    np.stack([p0, p_now]), np.stack([R0, R_now]), p_home, R_home
                )
                p_w, R_w = p_w[1:], R_w[1:]
            else:
                # `absolute`: the operator's OWN yaw-aligned world pose, unrebased. This is
                # the one branch where frame 0 is not the robot's home -- the lift below is
                # the whole placement, and `mc_episode`'s absolute branch does exactly this
                # (mount, then `lift_z`), in that order, because the mount is a body-local
                # lever and a z translation does not commute with it.
                p_w, R_w = p_now[None, :], R_now[None, :]
            pg, Rg = F.mc_gripper_from_wrist(p_w, R_w, side)
            pg = F.lift_z(pg, self._dz)  # dz == 0.0 in home mode: one path, both placements
            out += [pg[0], F.R_to_quat_wxyz(Rg)[0]]
        out += [root_p[0], F.R_to_quat_wxyz(root_R)[0]]
        return tuple(out)

    def _resolve_dz_locked(self, Lp, LR, Rp, RR, Hp) -> float:
        """The ONE rigid z lift for this episode, from cond frame 0. Caller holds `_lock`.

        Mirrors `mc_episode.CondEpisode.from_traj`'s absolute branch — pinned value wins,
        else the anchor decides — with one unavoidable difference: offline that branch
        averages the anchor's z over the WHOLE episode (`n_frames=0`), and here the episode
        is one frame old. So the auto value is the CALIBRATION POSE's height, not the take's
        mean, and the two differ by however much the operator's hands (or chest) drifted.

        That is why the log line below is the loudest thing this module prints: pass the same
        number to `mc_to_hl_wholebody.py --dz-wrist` and the labels are prepared in exactly
        the posture the robot was driven in. Leave it to auto on both sides and they are
        close but not equal, which is a systematic offset between every recorded image and
        the joint target beside it.
        """
        F = self._F
        if self._dz_pin is not None:
            dz, how = float(self._dz_pin), "pinned"
        elif self._dz_anchor == "chest":
            dz, how = float(F.auto_dz_chest(Hp[None, :])), "chest puck -> RIG_CHEST_STAND_Z"
        else:
            # On the MOUNTED gripper height, not the wrist: gripper_base z is what the model
            # actually reads, and the mount is a rotating lever (up to ~0.1 m of z).
            gb = [
                F.mc_gripper_from_wrist(p[None, :], R[None, :], s)[0]
                for s, p, R in (("left", Lp, LR), ("right", Rp, RR))
            ]
            dz = float(F.auto_dz_wrist(gb[0], gb[1], target_z=F.RIG_WRIST_STAND_Z))
            how = "frame-0 gripper z -> RIG_WRIST_STAND_Z"
        self._log(
            f"[mc] absolute placement: dz={dz:+.4f} m ({how}; anchor={self._dz_anchor}"
            f"{', PINNED' if self._dz_pin is not None else ''}) => the recorded floor sits at "
            f"z={-dz:+.4f} m.\n"
            f"[mc]     ⚠️  PREPARE THE LABELS WITH THE SAME LIFT: "
            f"run_mc_to_hl_wholebody.sh --cond-mode absolute --dz-wrist {dz:.6f}\n"
            f"[mc]     (auto on both sides is NOT the same number -- offline averages the "
            f"whole episode, this averaged one calibration frame.)"
        )
        return dz

    # -- window worker -------------------------------------------------------
    def _run_worker(self) -> None:
        """Sole owner of backend I/O: drain the cond queue, then run whatever is ready.

        The order matters and is the reason this is one thread rather than two: a window
        may only run once its cond frames have actually REACHED the backend, so readiness
        is keyed on `self._sent` (delivered), never on `self._pushed` (queued).
        """
        while not self._stop.is_set():
            self._pending_ev.wait(timeout=0.2)
            self._pending_ev.clear()
            if self._stop.is_set():
                break
            try:
                self._drain_cond()
            except Exception as exc:  # noqa: BLE001 - a bad send must not kill teleop
                self._err_count += 1
                self._last_err = f"push: {type(exc).__name__}: {exc}"
                self._log(f"[mc] cond push FAILED ({self._last_err})")
                continue
            while not self._stop.is_set():
                with self._lock:
                    w = self._next_window
                    ready = w < len(self._starts) and self._starts[w] + MC_WINDOW <= self._sent
                    if ready:
                        self._next_window += 1
                if not ready:
                    break
                try:
                    res = self._backend.run_window(w)
                except Exception as exc:  # noqa: BLE001 - a bad window must not kill teleop
                    self._err_count += 1
                    self._last_err = f"{type(exc).__name__}: {exc}"
                    self._log(f"[mc] window {w} FAILED ({self._last_err}) -- holding last command")
                    continue
                self._absorb(res)

    def _drain_cond(self) -> None:
        """Send every queued cond frame as ONE backend push (the wire takes (k,3)/(k,4)).

        Batching is not just for speed: after any hiccup the queue holds a burst, and one
        round trip for the burst keeps the window schedule from falling further behind.
        No ZOH padding is ever synthesised here -- the backend WOULD pad (`_slice` repeats
        the last frame) but only off the DECLARED episode length, and its cond buffer is
        append-only, so a padded early run could not be rolled back and would poison the
        next window's conditioning.

        Holds `_io_lock` across the take-batch / send / count-it sequence so a concurrent
        `reset()` cannot land between them; `_lock` is still released for the send itself, so
        the tracker thread's `push()` is never blocked behind inference.
        """
        with self._io_lock:
            with self._lock:
                batch = list(self._cond_q)
                self._cond_q.clear()
            if not batch:
                return
            cols: list[list[np.ndarray]] = [[] for _ in range(6)]
            for n_new, grip in batch:
                for c in range(6):
                    cols[c].append(
                        np.repeat(np.asarray(grip[c], dtype=np.float64)[None, :], n_new, axis=0))
            stacked = [np.concatenate(c, axis=0) for c in cols]
            self._backend.push(
                left_pos=stacked[0], left_quat=stacked[1],
                right_pos=stacked[2], right_quat=stacked[3],
                root_pos=stacked[4], root_quat=stacked[5],
            )
            with self._lock:
                self._sent += int(stacked[0].shape[0])

    def _absorb(self, res: dict) -> None:
        start, n = int(res["start"]), int(res["n"])
        if n <= 0:
            return
        # `MCBackend._request` returns the reply META with the arrays nested under
        # "arrays" — the wire carries them as separate fields, not merged into the meta.
        arr = res.get("arrays") or {}
        fk_pos = np.asarray(arr["fk_pos"], dtype=np.float32)
        fk_quat = np.asarray(arr["fk_quat"], dtype=np.float32)
        ja = np.asarray(arr["joint_angles"], dtype=np.float32)
        rp = np.asarray(arr["root_pos"], dtype=np.float32)
        rq = np.asarray(arr["root_quat"], dtype=np.float32)
        with self._lock:
            if self._fk_pos is None:
                return
            hi = min(start + n, self._n)
            k = hi - start
            if k <= 0:
                return
            # Overwrite, do not skip: a window REFINES its first `overlap` frames (the
            # blend against what w-1 left) and its last `overlap` are provisional until
            # w+1 refines them. Later writes are always the better estimate.
            self._fk_pos[start:hi] = fk_pos[:k]
            self._fk_quat[start:hi] = fk_quat[:k]
            self._joint_ang[start:hi] = ja[:k]
            self._root_pos[start:hi] = rp[:k]
            self._root_quat[start:hi] = rq[:k]
            self._valid[start:hi] = True
            self._newest = max(self._newest, hi - 1)
            self._win_count += 1
            self._win_ms.append(float(res.get("t_total_ms", 0.0)))
            if not bool(res.get("ik", True)):
                self._log(f"[mc] window {res.get('window')} wrist IK reported ERR")

    # -- output --------------------------------------------------------------
    def _command_frame_locked(self, now: float) -> int:
        """The frame to command: `now - lag`, clamped to what has actually settled."""
        if self._t0 is None or self._newest < 0:
            return -1
        f = int((now - self._t0) * MC_RATE_HZ) - self._lag
        if f > self._newest:
            f = self._newest  # behind schedule: hold the newest rather than extrapolate
        if f < 0:
            return -1
        return f if bool(self._valid[f]) else -1

    def latest(self, now: float | None = None) -> dict | None:
        """The commanded whole-body slice, as the wire wants it. None until ready."""
        now = time.time() if now is None else float(now)
        with self._lock:
            f = self._command_frame_locked(now)
            if f < 0:
                return None
            lag_s = (now - self._t0) - f / MC_RATE_HZ
            if self._kp_mode == "dense":
                out = self._dense_locked(f)
            else:
                out = self._five_point_locked(f)
            out["frame"] = f
            out["lag_s"] = float(lag_s)
            return out

    def _readout_frame_locked(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """The pelvis frame the 5pt keypoints are expressed in: (p (3,), q_wxyz (4,)).

        `ref_frame="frozen"` (default) returns the FIXED reference pelvis captured at frame
        0 -- `mc_frames.pelvis_ref_pose`, the same frame `home_wrist_pose` placed the home
        in. `ref_frame="live"` returns the completion's OWN root at frame `f`.

        WHY FROZEN IS THE DEFAULT, measured on `session_2026.08.27_1208_v3` ep 0:

        * **`live` deletes the operator's body motion.** The root conditioning IS the chest
          puck, so the completion's root follows the operator (measured: a 0.30 m walk moved
          the model's root 0.2948 m). Reading the keypoints out in that same moving frame
          cancels the translation twice over: a 300 mm walk moved the commanded points by
          **7.3 mm**, i.e. 97.6% of the operator's motion went nowhere at all. That is the
          rig symptom "I hold the tracker and walk forward and the point does not move
          forward". In `frozen` the same walk moves them 286.5 mm.
        * **`frozen` starts exactly on the pre-arm gate's reference.** At frame 0 the hands
          land on the G1's 29-joints-ZERO TCP geometry -- `(0.4418, +-0.1487, 0.0952)`, which
          IS `_compute_vr_fk` of the READY POSE -- to **0.1 / 0.3 mm**. So an ABSOLUTE-mode
          operator standing in the calibration pose passes the gate BY CONSTRUCTION instead
          of having to guess a pose. Using the model's output root at frame 0 instead is
          25-30 mm off, because it carries ~5 deg of tilt and stands ~11 mm low.
        * **`live` lets the completion's POSTURE leak into the hand and foot command.** The
          model's root tilts up to 11 deg and its height varies 0.754-0.783 m over an
          episode; in a moving frame a crouch reads as "raise the hands" and, worse, as
          "lift the feet" -- a body-inferred channel steering the one command the operator
          cannot see. In a fixed frame a crouch is invisible to the keypoints, which is
          right: the completion's posture is its own inference, not the operator's intent.

        `live` is kept because it is what TRAINING sees: the clip's keypoints are relative
        to the clip's per-frame root, and the root's own motion reaches the policy through
        `motion_ref_pos_b` / `motion_ref_vel_b`. But during VIVE teleop the reference clip is
        FROZEN (`holosoma_wbt` zeroes `motion_ref_vel_b` for exactly that reason), so nothing
        carries the root motion and `live` throws it away rather than expressing it. Driving
        the reference root off the completion instead would be the in-distribution way to give
        the operator locomotion; it is a separate change, and it makes a physical G1 WALK.
        """
        F = self._F
        if self._ref_frame == "live":
            p = np.asarray(self._root_pos[f], dtype=np.float64)
            q = np.asarray(self._root_quat[f], dtype=np.float64)  # wxyz
            return p, q / max(float(np.linalg.norm(q)), 1e-12)
        # frozen: the anchor is set on cond frame 0 and never moves again.
        p, R = self._anchor["pelvis"]
        return np.asarray(p, dtype=np.float64), F.R_to_quat_wxyz(np.asarray(R)[None, ...])[0]

    def _root_in_readout_frame_locked(self, f: int) -> tuple[list, list]:
        """The completion's OWN pelvis, expressed in the READOUT frame — for a viewer only.

        Returns ``(pos (3,), quat wxyz (4,))``. This is the missing half of a whole-body
        ghost: `joint_pos` flexes the body, this places it. Drawn without a root, a
        completion that steps sideways looks like a completion that stands still — the one
        reading a viewer of this stream is most likely to want.

        WHICH frame, and why this one: the SAME `_readout_frame_locked` the 5pt keypoints
        use, so the ghost and the command are one picture. Two consequences worth knowing
        before reading a ghost:

        * in `frozen` (the default) this is IDENTITY at frame 0 and then carries the
          completion's full body motion — a 0.30 m walk moves it 0.2948 m (measured);
        * in `live` it is identity at EVERY frame. That is not a bug in this method, it is
          the measured content of `live`: reading out in the completion's own moving root
          cancels the body motion (97.6% of a walk, see `_readout_frame_locked`), so a
          `live` ghost standing still while the operator walks IS the finding.

        Never reaches a policy — it rides the wire only under `--publish-mc-qpos`.
        """
        F = self._F
        p_pelvis, q_pelvis = self._readout_frame_locked(f)
        R_pelvis_T = F.quat_wxyz_to_R(q_pelvis[None, :])[0].T
        p_root = np.asarray(self._root_pos[f], dtype=np.float64)
        q_root = np.asarray(self._root_quat[f], dtype=np.float64)  # wxyz
        q_root = q_root / max(float(np.linalg.norm(q_root)), 1e-12)
        q_rel = quat_mul_wxyz(quat_inv_wxyz(q_pelvis), q_root)
        q_rel = q_rel / max(float(np.linalg.norm(q_rel)), 1e-12)
        return (R_pelvis_T @ (p_root - p_pelvis)).tolist(), q_rel.tolist()

    def _five_point_locked(self, f: int) -> dict:
        """5 keypoints in the PELVIS frame, in the policy's slot order.

        `select_keypoints("5pt")` already returns (L hand, R hand, torso, L ankle, R
        ankle) with the TCP / +0.35 z levers -- byte-for-byte holosoma's
        VR_5POINT_BODIES + VR_5POINT_OFFSETS_TCP; `mc_frames`'
        test_keypoint_bodies_and_levers_are_holosomas re-reads robot_config.py with `ast`
        so a preset change there fails there. So this method does NOT re-pick bodies.

        WHICH pelvis frame is `_readout_frame_locked`'s call, and it is the whole ballgame --
        read its docstring before changing anything here.
        """
        F = self._F
        kp = F.select_keypoints(
            self._fk_pos[f : f + 1], self._fk_quat[f : f + 1], self._body_names,
            mode="5pt", hand_point=self._hand_point,
        )
        p_pelvis, q_pelvis = self._readout_frame_locked(f)
        R_pelvis_T = F.quat_wxyz_to_R(q_pelvis[None, :])[0].T
        q_inv = quat_inv_wxyz(q_pelvis)
        pos, orn = [], []
        for k in range(len(kp["names"])):
            pos.extend((R_pelvis_T @ (np.asarray(kp["pos"][0, k]) - p_pelvis)).tolist())
            q_local = quat_mul_wxyz(q_inv, np.asarray(kp["quat"][0, k], dtype=np.float64))
            q_local = q_local / max(float(np.linalg.norm(q_local)), 1e-12)
            # Canonicalize the double cover to w >= 0 BEFORE the wxyz->xyzw swap, i.e. on
            # `q_local[0]`. Same fix as bake_motions._fk_into and holosoma_wbt's FK branch:
            # `q` and `-q` are one rotation but two different obs vectors, and the 5pt policy
            # trained only on the +w branch (na475b60 normalizer: w mean +0.73 hands / +0.98
            # head / +0.96 feet). Nothing pins the branch here either -- `q_pelvis` is the
            # readout frame's stored representative and `kp["quat"]` comes from the
            # completion's FK -- so a chest pose near the 180 deg seam sends every keypoint
            # to -identity at once (~20 sigma), which reads as the arms snapping to a
            # nonsense pose. No-op when the data is already +w.
            if q_local[0] < 0.0:
                q_local = -q_local
            orn.extend(quat_wxyz_to_xyzw(q_local).tolist())
        # `joint_pos`/`joint_names`/`root_*` are the SAME 29-DoF body these 5 points were
        # picked off, carried along so a viewer can draw the whole completion instead of five
        # spheres. The 5pt policy never reads them (its obs is the keypoints); the bridge only
        # puts them on the wire under `--publish-mc-qpos`. Free here — `_joint_ang[f]` is
        # already in hand, and `select_keypoints` above just FK'd the same frame.
        root_pos, root_quat = self._root_in_readout_frame_locked(f)
        return {
            "pos": pos,
            "orn": orn,
            "names": list(kp["names"]),
            "joint_pos": np.asarray(self._joint_ang[f], dtype=np.float64).tolist(),
            "joint_names": list(self._joint_names),
            "root_pos": root_pos,
            "root_quat": root_quat,
        }

    def _dense_locked(self, f: int) -> dict:
        """29 joint targets + a finite-difference velocity, in MC's joint order.

        The NAME remap onto the policy's `policy_joint_names` is the CALLER's job (it is
        the side that knows the policy) -- `mc_frames.urdf_cfg_from_qpos` does it by name.
        `joint_names` is returned alongside so the caller cannot index-align by accident.
        """
        q = np.asarray(self._joint_ang[f], dtype=np.float64)
        if f > 0 and bool(self._valid[f - 1]):
            dq = (q - np.asarray(self._joint_ang[f - 1], dtype=np.float64)) * MC_RATE_HZ
        else:
            dq = np.zeros(29, dtype=np.float64)
        # `root_*` is VIEWER-ONLY here, exactly as in `_five_point_locked`. The dense command
        # is joint angles and carries no pelvis (that is the whole reason its readout frame is
        # irrelevant to the policy) — but a ghost of the completion still has to stand
        # somewhere, and a dense policy is if anything MORE interesting to compare against,
        # since a joint-space command cannot express a step at all.
        root_pos, root_quat = self._root_in_readout_frame_locked(f)
        return {"joint_pos": q.tolist(), "joint_vel": dq.tolist(),
                "joint_names": list(self._joint_names),
                "root_pos": root_pos, "root_quat": root_quat}

    # -- telemetry -----------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            ms = list(self._win_ms)
            return {
                "pushed": self._pushed,
                "sent": self._sent,
                "newest": self._newest,
                "windows": self._win_count,
                "errors": self._err_count,
                "last_error": self._last_err,
                "win_ms_p50": float(np.median(ms)) if ms else 0.0,
                "win_ms_max": float(np.max(ms)) if ms else 0.0,
                "duty_pct": (float(np.median(ms)) / (MC_STRIDE / MC_RATE_HZ * 1e3) * 100.0)
                if ms else 0.0,
                "lag_frames": self._lag,
                "lag_ms": self._lag / MC_RATE_HZ * 1e3,
                # The PLACEMENT, so a recorder can stamp it into the session it is writing
                # rather than an operator having to copy it out of a log line. `dz` is 0.0
                # until the calibration pose sets it (and exactly 0.0 forever in home mode).
                "cond_mode": self._cond_mode,
                "hand_lever": self._hand_lever,
                "dz_anchor": self._dz_anchor if self._cond_mode == "absolute" else "",
                "dz_wrist": float(self._dz),
                "dz_pinned": self._dz_pin is not None,
            }


__all__ = [
    "MC_COND_MODES",
    "MC_DEFAULT_COND_MODE",
    "MC_DEFAULT_DZ_ANCHOR",
    "MC_DEFAULT_EPISODE_FRAMES",
    "MC_DEFAULT_HAND_LEVER",
    "MC_DEFAULT_HEAD_FRAME",
    "MC_DZ_ANCHORS",
    "MC_HAND_LEVERS",
    "MC_DEFAULT_REF_FRAME",
    "MC_HEAD_FRAMES",
    "MC_MIN_SMOOTH_LAG_FRAMES",
    "MC_OVERLAP",
    "MC_RATE_HZ",
    "MC_REF_FRAMES",
    "MC_STRIDE",
    "MC_WINDOW",
    "MotionCompletionSource",
    "quat_inv_wxyz",
    "quat_mul_wxyz",
    "quat_wxyz_to_xyzw",
    "wire_payload",
]
