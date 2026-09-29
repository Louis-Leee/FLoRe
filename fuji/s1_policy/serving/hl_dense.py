"""MOTION COMPLETION between the S1 planner and a WHOLE-BODY low-level controller.

The 3-tracker planner (``diffusion_umi_3tracker``, layout ``both_hands_head_dual``) predicts
three end-effector poses. The whole-body policies want more than three points: the ``dense``
family (``g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt``, W&B ``7vxmnwo6``) is steered by
**all 29 joint targets**, the ``5pt`` family
(``g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt``, W&B ``tqnoooo6``) by **five keypoints**
— both hands, the torso, and both feet. Motion completion is the bridge for both, and it is
the SAME bridge the rig's VIVE teleop already uses (``vive/mc_teleop_source.py``), only fed by
a policy instead of by an operator::

    S1 chunk (anchor-relative)  --decode-->  L/R TCP + chest, absolute, 30 Hz
                                                     |
                                     Motion_Completion (this module)
                                                     |
                                    qpos(29) + its own FK
                                        /                  \\
              :8770 {"mode": "dense"}                   :8770 {"mode": "5pt"}
              29 joints, by NAME                        5 keypoints, PELVIS frame
              dense / wbt family                        5pt family

⚠️ The module NAME is historical — dense was built first. The completion is one computation;
which wire it is read out on is a fact about the ONNX being served (:data:`WIRE_MODES`, and
:class:`DenseWire` / :class:`FivePointWire`, which own one family's wire each). The low level's
width guard DROPS a frame of the wrong shape, so serving the wrong one looks exactly like a
robot holding its ready pose forever.

WHY THERE IS NO 667 ms LAG HERE
===============================
That floor is the headline number of the teleop path and it does **not** apply, because it is
a statement about not knowing the future. The completion conditions DENSELY over a 30-frame
window, so window ``w`` covering ``[20w, 20w+30)`` cannot run until frame ``20w+29`` has
been observed; a live operator has not moved yet, so the command must lag ``stride-1 = 19``
frames.

A planner has the opposite problem: the whole chunk is a PREDICTION, so all 50 frames exist
the instant the sample returns. Every window is immediately runnable, and the completion is
a one-shot batch — ``window_starts(50, 30, 10) == [0, 20]``, two windows, ~35-40 ms each
measured, run in order (the backend seeds ``w`` from ``w-1``'s tail and refuses out of
order). What the planner pays instead is a per-Predict LATENCY, once, on top of the sampler.

The consequence to keep in mind is the other side of the same coin: the completion is only
as good as the chunk. A re-plan discards the tail of the previous completion, exactly as it
discards the tail of the chunk itself.

THE BUDGET, AND WHY FOOT IK IS OFF
==================================
That per-Predict latency is not free time: the planner re-plans every ``--exec-steps`` (16)
steps, so at 30 Hz the sampler AND the completion must fit inside 533 ms or the new plan
lands after the steps it was meant to replace — the drip runs off the end of the old chunk
and the robot stalls mid-task. With a ~53 ms sampler that leaves **481 ms**.

Measured by ``scripts/ablate_mc_foot_ik.py`` on ``ckpt_0831_final``: one predicted chunk
(H=50, 2 windows), re-completed per arm, median of 3, reproduced across two runs to <3%::

    post-process                 chunk     /step   footIK    duty   ankle slide   wrist err
    raw (nothing)                 37 ms   0.75 ms    ---       8%       0.0100     10.65 mm
    wrist IK only  <- DEFAULT     80 ms   1.60 ms    ---      17%       0.0100      1.05 mm
    + foot IK  40 iters         1376 ms  27.5  ms   1300     286%       0.0092      0.96 mm
    + foot IK  80 iters         2679 ms  53.6  ms   2597     557%       0.0026      1.02 mm
    + foot IK 150 iters (MC's)  4887 ms  97.7  ms   4804    1017%       0.0005      0.86 mm

So wrist IK is the whole post-process budget: **41 ms** to take the commanded hand from
10.65 mm to ~1 mm, which is the accuracy the deploy contract quotes. Foot IK is 17-59x the
entire rest of the chunk and misses the budget by 2.9-10.2x at EVERY setting, including the
cheap one that removes only 8% of the ankle slide (it is a convergence threshold, not a dial
— see ``mc_infer_server.StreamingCompleter.FOOT_IK_ITERS``).

⚠️ And the cost is not only time. The pass rewrites the LEGS, and under
:class:`JointClutch` a leg rewrite is a leg COMMAND: worst per-step leg ``|dq|`` goes
**0.0061 -> 0.0493 rad**, 59% of the low level's 0.0833 rad/step budget, on a robot whose
legs are holding it up. Anti-slide is a rendering nicety here — the robot's own feet are
already on the floor, and the low level, not us, decides what its legs do.

It is still **switchable at any time** (``--mc-foot-ik inertialize``, or the page's
``Foot IK / anti-slide`` checkbox) because a static posture check genuinely wants it, and
because a measured cost that the operator can reproduce beats a knob that is not there. What
the switch buys is not symmetric between the families, so both the page and the log name it:
``dense`` publishes the rewritten legs as joint targets, while ``5pt`` only moves the two FOOT
keypoints — which pass through the clutch absolutely and are not gated (warned at 300 mm), so
there the pass cannot reach the 3 points the pre-arm gate checks at all.

THE PLACEMENT IS THE ROBOT, NOT A GUESS
=======================================
An HL-v2 chunk is anchor-relative, so the pose it is decoded against decides where the hands
ARE — and motion completion reads the chest as its ROOT channel, so that placement decides
the POSTURE (the offline viewer measured 174 mm of elbow behind the shoulder from getting it
wrong; see ``fuji/s1_policy/inference/ready_pose.py``). On the rig the anchor is not a
choice: it is where the robot is when the operator presses **A**, i.e.
``home_anchor.json::default_qpos`` with the 14 arm joints zeroed, FK'd through this rig's
URDF — whose floor is z=0 by construction. So:

* ``cond_mode="absolute"`` — keep the decoded world poses. ``home`` would re-base the chunk
  onto MC's own 29-joints-zero wrist pose, throwing away the robot's actual arm state.
* ``dz_wrist=0.0`` — PINNED. ``auto_dz_chest`` aims at ``RIG_CHEST_STAND_Z`` (1.30) while
  the ready pose's own chest keypoint sits at 1.151, so deriving a lift would raise the
  whole chunk ~15 cm off the robot it is supposed to be standing on.

BY NAME, ALWAYS
===============
MC's joint order (``g1_fk_params.npz``) and holosoma's ``dof_names`` are byte-identical
*today*, so an index-aligned remap passes every realistic test and scrambles the body the day
either changes. Everything here carries names: :func:`remap_by_name` is the one remap, the
wire ships ``joint_names`` beside the values (``ViveLiveSource`` DROPS a dense frame without
them), and :func:`ready_pose_joint_targets` reorders the SDK ready pose into MC's order the
same way.

DEGRADES, NEVER ABORTS
======================
Motion_Completion is a peer checkout with its own conda env and its own S3 checkpoint. A
missing piece yields a ``disabled_reason`` and the planner refuses to ENTER dense mode with
that reason printed — it never half-starts, because a dense low level with no wire holds the
ready pose forever and looks like a hung robot.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from fuji.s1_policy.inference.ready_pose import ROLE_BODY as _READY_ROLE_BODY
from fuji.s1_policy.serving import hl_anchor
from fuji.utils.rot6d import rot6d_to_rotmat

#: The completion's own grid and window schedule. Fixed by the checkpoint, not tunable:
#: ``mc_frames``/``mc_teleop_source`` spell the same three numbers and a test pins them equal.
MC_RATE_HZ = 30.0
MC_WINDOW = 30
MC_OVERLAP = 10
MC_STRIDE = MC_WINDOW - MC_OVERLAP

#: G1 actuated joints — the width of a dense ``:8770`` frame, and what
#: ``ViveLiveSource._parse_joint_vec`` validates.
G1_DOF = 29

#: The two low-level FAMILIES this module can drive off ONE completion, and therefore the
#: two ``:8770`` wire modes. Which one a deploy needs is a fact about the ONNX being served,
#: not a preference — ``ViveLiveSource`` drops a frame of the wrong shape, so a mismatch
#: shows up as a robot holding its ready pose forever.
WIRE_MODES = ("dense", "5pt")

#: The 5pt keypoint bodies, in the policy's slot order. Duplicated from
#: ``mc_frames.KEYPOINT_MODES["5pt"]`` (which is itself re-read out of holosoma's
#: ``robot_config.VR_5POINT_BODIES`` by a test in the submodule) because this module must
#: import with no ``third_party`` on the path; ``tests/test_hl_dense.py`` pins them equal.
FIVE_POINT_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link",
                     "left_ankle_roll_link", "right_ankle_roll_link")

#: Operator-facing labels in the same order, so a gate readout can say WHICH point is out
#: instead of printing five bare numbers. Same strings as holosoma's ``_VR_POINT_LABELS``.
FIVE_POINT_LABELS = ("LEFT hand", "RIGHT hand", "head/torso", "LEFT foot", "RIGHT foot")

#: The three points a 3-tracker policy actually PREDICTS, in wire-slot order — i.e. the layout
#: roles that :meth:`KeypointReadout.upper_from_decoded` reads out of a decoded chunk. Spelled
#: as the first three of :data:`FIVE_POINT_BODIES`' roles rather than re-typed, because the
#: prefix property is the whole reason the substitution is legal.
_UPPER_ROLES: tuple[str, ...] = ("left", "right", "head")

#: URDF body -> the LAYOUT ROLE that predicts it, for the whole five. The inverse of
#: ``ready_pose.ROLE_BODY`` restricted to :data:`FIVE_POINT_BODIES`, and DERIVED from it at import
#: rather than typed out, so "which role fills wire slot k" is answered by the one table the
#: decode anchor and the FK already share — the ``left_foot``/``right_foot`` roles exist in
#: exactly one place. Used by :meth:`KeypointReadout.all_from_decoded` (the ``diffusion_umi_5pt``
#: path, where the policy predicts every slot).
#:
#: The import is LAZY-ish only in the sense that it reaches sideways into ``inference/``:
#: ``ready_pose`` imports ``display``, both of which are pure numpy with no viser and no
#: submodule, so this adds nothing heavy to a serving process. Asserted complete below, because
#: a body with no role would silently drop a keypoint out of the readout.
_ROLE_BY_FIVE_POINT_BODY: dict[str, str] = {
    body: role for role, body in _READY_ROLE_BODY.items() if body in FIVE_POINT_BODIES}
if tuple(sorted(_ROLE_BY_FIVE_POINT_BODY)) != tuple(sorted(FIVE_POINT_BODIES)):
    raise ImportError(  # pragma: no cover - import guard
        f"ready_pose.ROLE_BODY does not name a layout role for every 5pt wire body: got "
        f"{sorted(_ROLE_BY_FIVE_POINT_BODY)}, want {sorted(FIVE_POINT_BODIES)}. A direct 5pt "
        f"wire reads its slots by role, so a missing one would publish a stale keypoint.")

#: Where the three GATED points on a 5pt wire come from.
#:
#: ``completion`` (the historical behaviour) reads all five out of the completion's FK.
#: ``policy`` publishes the three the policy predicted and takes only the feet from the
#: completion. They agree to the completion's hand tracking error, **0.84 mm** — so this is not
#: an accuracy choice, it is a question of whether the completion sits between the policy and
#: the three points the low level gates on.
UPPER_SOURCES: tuple[str, ...] = ("completion", "policy")

#: Where the two UNGATED foot points come from.
#:
#: ``completion`` (the default) reads them out of the same completion pass — the feet are then
#: consistent with the posture the predicted hands imply, which is the completion's real value
#: here (all five points describe ONE physically possible body).
#:
#: ``hold`` pins them to the READY POSE's own ankles for every step, which makes the completion
#: unnecessary for the wire entirely. Measured support: within a 30-frame window the completed
#: feet move only **1.9-4.4 mm**, holosoma gates 3 of 5 and only warns past 300 mm, and the low
#: level's absolute arm clutch snapshots ``held - wire`` per point at arm — so a CONSTANT foot
#: wire commands exactly the stance the robot was already holding.
#: ⚠️ What it gives up is measured too: the operator's chest yaw spans 10.7 deg median / 48.5 deg
#: p90 over 167 episodes, and a yaw that large genuinely re-places the feet (~0.12 m at 48 deg).
#: ``hold`` will not follow it. And the 5pt low level is a whole-body BALANCE controller reading
#: all five points, so a foot target inconsistent with the posture the hands imply is a
#: balance-relevant command in a way a hand target is not — that interaction is NOT measured on
#: hardware, which is why ``completion`` remains the default.
FOOT_SOURCES: tuple[str, ...] = ("completion", "hold")

#: How many of the five the low level's pre-arm gate actually CHECKS
#: (``holosoma_wbt._VR_5POINT_OPERATOR_PLACED``). The last two are the completion's inferred
#: FEET: nobody can aim them, and against the ready pose's 38 deg crouch they bottom out at
#: 0.10 / 0.06 m versus a 0.07 m tolerance, so gating them was a permanent lockout. They are
#: still compared and WARNED past :data:`FIVE_POINT_DERIVED_WARN_M`, because a check that is
#: silently skipped is indistinguishable from a check that passes.
FIVE_POINT_GATED = 3

#: The low level's own keypoint pre-arm tolerance (``HOLOSOMA_VIVE_ARM_TOL``, metres) and the
#: derived-point warning threshold (``_VR_DERIVED_GAP_WARN_M``). Quoted here only so the
#: planner can print "this WOULD pass the gate" before anything moves; the gate itself is the
#: low level's.
KEYPOINT_GATE_TOL_M = 0.07
FIVE_POINT_DERIVED_WARN_M = 0.30

#: Foot IK is OFF **by default**. Measured on THIS path (``scripts/ablate_mc_foot_ik.py``, table
#: in the module docstring): 1376 ms per chunk at its cheapest setting and 4887 ms at MC's own 150
#: iters, against a 481 ms re-plan budget — so the body PAUSES between chunks at every setting,
#: and the cheap one removes only 8% of the ankle slide. It can be turned on anyway, with the
#: numbers printed: ``--mc-foot-ik inertialize`` at launch, or the planner page's
#: ``Foot IK / anti-slide`` checkbox at any time (applied on the next chunk boundary — see
#: :meth:`ChunkCompleter.set_post_process`). Being a COMPLETION post-process it is upstream of
#: the wire split, so one switch serves both families; the consequence is not the same in both
#: (dense publishes the rewritten legs, 5pt only moves its two ungated feet).
DEFAULT_FOOT_FIX = "none"

#: Per-window backend timeout is ``mc_backend_client.REPLY_TIMEOUT_S`` (60 s); this is the
#: budget for the WHOLE chunk, so a wedged backend surfaces as a refused Predict rather than
#: a viser page that stops repainting.
CHUNK_TIMEOUT_S = 120.0


def _find_umi_dir(explicit: str | None = None) -> str | None:
    """Locate the UMI_collection checkout that holds ``motion_completion/``."""
    if explicit:
        candidate = os.path.abspath(explicit)
        return candidate if os.path.isdir(os.path.join(candidate, "motion_completion")) else None
    env = os.environ.get("MC_UMI_DIR")
    if env:
        candidate = os.path.abspath(env)
        if os.path.isdir(os.path.join(candidate, "motion_completion")):
            return candidate
    root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(root, "third_party", "UMI_collection")
        if os.path.isdir(os.path.join(candidate, "motion_completion")):
            return candidate
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    return None


def _load_mc_modules(umi_dir: str):
    """Import ``mc_frames`` / ``mc_episode`` / ``mc_backend_client`` off the peer checkout.

    They are not installed in this env and never will be — they belong to UMI_collection,
    which has its own venv. All three are numpy-only precisely so either side imports the ONE
    copy instead of carrying two that drift, so a path import is the intended use.

    ``sys.path.append``, never ``insert(0)``: that directory holds generically-named modules
    (``mc_wire``, ``mc_episode``) which must not shadow same-named top-level imports for the
    rest of this process. The modules themselves are loaded by explicit spec so the name they
    land under is ours; ``mc_backend_client`` does a bare ``import mc_wire``, which is why the
    directory has to be on the path as well.
    """
    pkg = os.path.join(umi_dir, "motion_completion")
    if pkg not in sys.path:
        sys.path.append(pkg)
    out = []
    for name in ("mc_wire", "mc_frames", "mc_episode", "mc_backend_client"):
        if name in sys.modules:
            out.append(sys.modules[name])
            continue
        spec = importlib.util.spec_from_file_location(name, os.path.join(pkg, f"{name}.py"))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name} from {pkg}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        out.append(module)
    return out[1], out[2], out[3]  # mc_frames, mc_episode, mc_backend_client


# ---------------------------------------------------------------------------
# by-name joint remap
# ---------------------------------------------------------------------------
def remap_by_name(values, src_names, dst_names) -> np.ndarray:
    """Reorder ``values`` from ``src_names`` order into ``dst_names`` order.

    The one remap in this module, and it RAISES on a name that has no source rather than
    filling a zero: a silently-zeroed joint is a limb that goes limp on a physical robot.
    """
    array = np.asarray(values, dtype=np.float64)
    src = [str(n) for n in src_names]
    dst = [str(n) for n in dst_names]
    if array.shape[-1] != len(src):
        raise ValueError(f"values width {array.shape[-1]} != len(src_names) {len(src)}")
    if len(set(src)) != len(src):
        raise ValueError("src_names has duplicates; a by-name remap would be ambiguous")
    index = {name: i for i, name in enumerate(src)}
    missing = [name for name in dst if name not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} target joint(s) absent from the source order: {missing[:6]}"
            f"{' …' if len(missing) > 6 else ''}. Refusing to remap — a missing joint would "
            "be published as 0 rad, which is a real command on a real robot.")
    return array[..., [index[name] for name in dst]]


def joint_velocity(joint_pos: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference joint velocity, same convention as ``mc_teleop_source._dense_locked``.

    Backward difference so every row uses only frames at or before it (the wire is consumed
    causally), with row 0 at zero because there is no earlier frame in the chunk — the dense
    obs carries ``dof_vel`` from the robot's own encoders anyway, so this term is a reference
    velocity, not a measurement.
    """
    q = np.asarray(joint_pos, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"joint_pos must be (H, dof), got {q.shape}")
    out = np.zeros_like(q)
    if q.shape[0] > 1 and dt > 0.0:
        out[1:] = (q[1:] - q[:-1]) / float(dt)
    return out


# ---------------------------------------------------------------------------
# the completed chunk
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompletedChunk:
    """One action chunk, completed to a 29-DoF whole-body motion.

    ``joint_pos`` / ``joint_vel`` are in **MC's** joint order and ``joint_names`` says so;
    they go on the wire together and the low level remaps by name. ``fk_pos``/``fk_quat`` and
    ``root_*`` are the drawn body — the same arrays the offline viewer's humanoid renders, so
    the planner's ghost and the offline page show the same completion.
    """

    joint_pos: np.ndarray            # (H, 29) rad, MC joint order
    joint_vel: np.ndarray            # (H, 29) rad/s
    joint_names: tuple[str, ...]     # 29, MC order — ships on the wire
    fk_pos: np.ndarray               # (H, 30, 3) world, floor z=0
    fk_quat: np.ndarray              # (H, 30, 4) wxyz
    body_names: tuple[str, ...]      # 30
    root_pos: np.ndarray             # (H, 3) pelvis
    root_quat: np.ndarray            # (H, 4) wxyz
    valid: np.ndarray                # (H,) bool — every frame a window actually wrote
    cond: object                     # the CondEpisode, for the round-trip check + readouts
    windows: int
    ms: float
    timings: tuple[dict, ...] = ()   # per-window backend stage timings, in window order

    @property
    def n(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def complete(self) -> bool:
        return bool(self.valid.all())

    def wrist_error_m(self) -> float:
        """Max ``gripper_base`` miss between what MC was ASKED for and what it produced.

        The one number that says the frame chain is intact end to end: sub-mm means
        ``TCP -> wrist_yaw_link -> MC gripper_base`` lines up, ~57 mm means the gripper mount
        was applied twice, ~242 mm means the TCP lever was dropped. Same check the offline
        smoke test asserts, computed here so the rig can show it live before anything moves.
        """
        frames, _, backend = _MODULES
        if frames is None:                                     # pragma: no cover - defensive
            return float("nan")
        worst = 0.0
        for side, body in (("left", "left_wrist_yaw_link"), ("right", "right_wrist_yaw_link")):
            if body not in self.body_names:                    # pragma: no cover - defensive
                return float("nan")
            column = self.body_names.index(body)
            got, _ = frames.mc_gripper_from_wrist(
                self.fk_pos[:, column], frames.quat_wxyz_to_R(self.fk_quat[:, column]), side)
            want = getattr(self.cond, f"{side}_pos")
            worst = max(worst, float(np.abs(got - want).max()))
        del backend
        return worst

    def describe(self) -> str:
        return (f"n={self.n} @ {MC_RATE_HZ:g} Hz  windows={self.windows}  "
                f"{self.ms:.0f} ms  wrist err {self.wrist_error_m() * 1e3:.2f} mm  "
                f"pelvis z {self.root_pos[:, 2].min():.3f}..{self.root_pos[:, 2].max():.3f}")

    def stage_ms(self) -> dict:
        """Backend cost split by stage, summed over the chunk's windows.

        ``flow`` is the completion transformer, ``ik`` the per-window JAX wrist IK, ``foot``
        the causal leg-IK pass — which is off by default and, when on, dominates everything
        else by an order of magnitude (see :class:`ChunkCompleter`). ``backend`` is the sum
        the backend itself measured; ``self.ms`` additionally carries the pipe round trips,
        so the gap between the two is the transport.
        """
        keys = ("t_flow_ms", "t_decode_ms", "t_ik_ms", "t_foot_ms", "t_total_ms")
        out = {k[2:-3]: float(sum(float(t.get(k, 0.0) or 0.0) for t in self.timings))
               for k in keys}
        out["backend"] = out.pop("total")
        out["transport"] = max(0.0, self.ms - out["backend"])
        return out


#: Cached module handles, so :meth:`CompletedChunk.wrist_error_m` can reach ``mc_frames``
#: without every caller threading it through. Set by :meth:`ChunkCompleter.start`.
_MODULES: tuple = (None, None, None)


class ChunkCompleter:
    """Motion completion for ONE predicted chunk at a time, synchronously.

    Deliberately not a streaming source: the planner hands over a finished chunk and wants
    all of it back before it publishes the first step, so there is nothing for a worker
    thread to overlap and a synchronous call keeps the "who mutated what" story legible (the
    same reasoning ``mc_backend_client`` states for being synchronous). The lock only serves
    the case where two viser callbacks race — the backend is not thread-safe.
    """

    def __init__(self, *, umi_dir: str | None = None, wrist_ik: bool = True,
                 foot_fix: str = DEFAULT_FOOT_FIX, foot_ik_iters: int | None = None,
                 foot_lookback: int | None = None, log=None) -> None:
        self.log = log or (lambda m: print(f"[hl-dense][mc] {m}", flush=True))
        self._wrist_ik = bool(wrist_ik)
        self._foot_fix = str(foot_fix)
        self._foot_ik_iters = None if foot_ik_iters is None else int(foot_ik_iters)
        self._foot_lookback = None if foot_lookback is None else int(foot_lookback)
        self._lock = threading.Lock()
        self._backend = None
        self._frames = None
        self._episode = None
        self.joint_names: tuple[str, ...] = ()
        self.body_names: tuple[str, ...] = ()
        self.contract: dict = {}
        self.disabled_reason = ""
        self._umi_dir = _find_umi_dir(umi_dir)
        if self._umi_dir is None:
            self.disabled_reason = (
                "no third_party/UMI_collection/motion_completion (git submodule update "
                "--init third_party/UMI_collection), and MC_UMI_DIR is unset")
            return
        try:
            frames, episode, client = _load_mc_modules(self._umi_dir)
        except Exception as exc:  # noqa: BLE001 - a peer checkout problem must be a reason
            self.disabled_reason = f"cannot import the motion-completion modules ({exc!r})"
            return
        self._frames, self._episode, self._client = frames, episode, client
        paths, reason = client.resolve()
        if reason:
            self.disabled_reason = reason
            return
        self._paths = paths

    # -- lifecycle -----------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._backend is not None and not self.disabled_reason

    def start(self) -> bool:
        """Spawn the backend (~30-60 s: a 104 MB ckpt + the JAX wrist-IK JIT). False + reason."""
        global _MODULES
        if self.disabled_reason:
            return False
        backend = self._client.MCBackend(log=self.log)
        if not backend.start():
            self.disabled_reason = backend.disabled_reason or "motion-completion backend failed to start"
            return False
        body_names, joint_names = self._frames.load_mc_names(self._paths["mc_dir"])
        if len(body_names) != 30 or len(joint_names) != G1_DOF:
            backend.stop()
            self.disabled_reason = (
                f"unexpected FK params: {len(body_names)} bodies / {len(joint_names)} joints "
                f"(want 30 / {G1_DOF})")
            return False
        self._backend = backend
        self.body_names = tuple(str(b) for b in body_names)
        self.joint_names = tuple(str(j) for j in joint_names)
        self.contract = dict(backend.contract)
        _MODULES = (self._frames, self._episode, self._client)
        window = int(self.contract.get("window", MC_WINDOW))
        overlap = int(self.contract.get("overlap", MC_OVERLAP))
        if (window, overlap) != (MC_WINDOW, MC_OVERLAP):
            # Not fatal — the schedule is read from the contract below — but every latency
            # number in this module's docstring is derived from 30/10, so say it.
            self.log(f"⚠️  backend window/overlap is {window}/{overlap}, not "
                     f"{MC_WINDOW}/{MC_OVERLAP}; the per-chunk cost will differ")
        self.log(f"ready: {len(self.joint_names)} joints, wrist_ik={self._wrist_ik}, "
                 f"foot_fix={self._foot_fix}"
                 + (f" ({self._foot_ik_iters} leg-IK iters)" if self._foot_ik_iters else "")
                 + "  (foot IK costs 17-59x the whole rest of the chunk — see FOOT IK)")
        return True

    def stop(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.stop()

    def set_post_process(self, *, wrist_ik: bool | None = None, foot_fix: str | None = None,
                         foot_ik_iters: int | None = None) -> None:
        """Change the post-processes for the NEXT chunk.

        Safe between chunks because each chunk is its own backend episode — the settings ride
        on ``start_episode`` — and NOT safe mid-chunk, where it would put two different bodies
        in one motion.

        ⚠️ **Not lock-protected, deliberately.** :meth:`complete` holds ``self._lock`` for the
        whole chunk (1.4-4.9 s with foot IK on), so taking that lock here would freeze a GUI
        callback for seconds. The contract is therefore on the CALLER: call it from the thread
        that calls :meth:`complete`, before the chunk starts. ``scripts/ablate_mc_foot_ik.py``
        does that between arms; ``scripts/hl_serving_mc.py``'s page checkbox does it at the top
        of its ``_predict`` (the switch only records a request; the predicting thread applies
        it), which is what makes a live foot-IK toggle land on a chunk boundary.

        ⚠️ ``foot_ik_iters`` is "set or CLEAR", not "leave alone": omitting it resets the
        override to the checkpoint's own 150, a 3.5x cost change. Pass it every time.
        """
        if wrist_ik is not None:
            self._wrist_ik = bool(wrist_ik)
        if foot_fix is not None:
            self._foot_fix = str(foot_fix)
        self._foot_ik_iters = None if foot_ik_iters is None else int(foot_ik_iters)

    # -- the one operation ---------------------------------------------------
    def build_cond(self, *, left_pos, left_rotmat, right_pos, right_rotmat,
                   head_pos, head_rotmat, eef_t_wrist, eef_R_wrist, dt: float,
                   grip_width: dict | None = None, cond_mode: str = "absolute",
                   dz_wrist: float | None = 0.0, src: dict | None = None):
        """The decoded chunk as a ``CondEpisode`` — the SAME frame chain the viewer uses.

        Split out from :meth:`complete` so the conditioning can be built and inspected with no
        backend at all (which is what the tests do, and what a ``--dry-run`` placement check
        would do): everything here is pure numpy in UMI's ``mc_episode``.
        """
        if self._episode is None:
            raise RuntimeError(f"motion completion is unavailable: {self.disabled_reason}")
        return self._episode.CondEpisode.from_poses(
            left_eef_pos=left_pos, left_eef_rotmat=left_rotmat,
            right_eef_pos=right_pos, right_eef_rotmat=right_rotmat,
            head_pos=head_pos, head_rotmat=head_rotmat,
            eef_t_wrist=eef_t_wrist, eef_R_wrist=eef_R_wrist,
            dt=float(dt), grip_width=dict(grip_width or {}),
            cond_mode=str(cond_mode), dz_wrist=dz_wrist,
            window=int(self.contract.get("window", MC_WINDOW)),
            overlap=int(self.contract.get("overlap", MC_OVERLAP)),
            src=dict(src or {}),
        )

    def complete(self, cond) -> CompletedChunk:
        """Run every window of ``cond`` in order and return the whole completed motion.

        Raises on a backend failure rather than returning a partial body: a chunk half in one
        posture and half in another is a real command, and the planner's answer to "the
        completion failed" is to refuse the Predict, not to publish something.
        """
        if not self.available:
            raise RuntimeError(f"motion completion is unavailable: {self.disabled_reason}")
        n = int(cond.n)
        t0 = time.monotonic()
        with self._lock:
            self._backend.start_episode(n, wrist_ik=self._wrist_ik, foot_fix=self._foot_fix,
                                       foot_ik_iters=self._foot_ik_iters,
                                       foot_lookback=self._foot_lookback)
            # ALL the cond frames up front, in ONE push. This is the whole difference from
            # teleop: the chunk is a prediction, so nothing has to be waited for and every
            # window below is immediately runnable.
            self._backend.push(
                left_pos=cond.left_pos, left_quat=cond.left_quat,
                right_pos=cond.right_pos, right_quat=cond.right_quat,
                root_pos=cond.root_pos, root_quat=cond.root_quat)
            joint = np.zeros((n, G1_DOF), dtype=np.float64)
            fk_pos = np.zeros((n, len(self.body_names), 3), dtype=np.float64)
            fk_quat = np.zeros((n, len(self.body_names), 4), dtype=np.float64)
            root_pos = np.zeros((n, 3), dtype=np.float64)
            root_quat = np.zeros((n, 4), dtype=np.float64)
            valid = np.zeros(n, dtype=bool)
            timings: list[dict] = []
            for window in range(len(cond.starts)):
                if time.monotonic() - t0 > CHUNK_TIMEOUT_S:
                    raise TimeoutError(
                        f"motion completion exceeded {CHUNK_TIMEOUT_S:.0f} s on window "
                        f"{window}/{len(cond.starts)} — the backend is wedged, not slow")
                result = self._backend.run_window(window)
                timings.append({k: v for k, v in result.items() if k.startswith("t_")
                                or k in ("window", "foot", "slide_before", "slide_after")})
                start, count = int(result["start"]), int(result["n"])
                arrays = result.get("arrays") or {}
                high = min(start + count, n)
                k = high - start
                if k <= 0:
                    continue
                # OVERWRITE, never skip: a window REFINES the first ``overlap`` frames w-1
                # already produced, and its own last ``overlap`` stay provisional until w+1.
                # The later write is always the better estimate.
                joint[start:high] = np.asarray(arrays["joint_angles"], dtype=np.float64)[:k]
                fk_pos[start:high] = np.asarray(arrays["fk_pos"], dtype=np.float64)[:k]
                fk_quat[start:high] = np.asarray(arrays["fk_quat"], dtype=np.float64)[:k]
                root_pos[start:high] = np.asarray(arrays["root_pos"], dtype=np.float64)[:k]
                root_quat[start:high] = np.asarray(arrays["root_quat"], dtype=np.float64)[:k]
                valid[start:high] = True
                if not bool(result.get("ik", True)):
                    self.log(f"⚠️  window {window}: the backend reported a wrist-IK error")
        if not valid.all():
            raise RuntimeError(
                f"motion completion covered {int(valid.sum())}/{n} frames — windows "
                f"{cond.starts} did not span the chunk. Refusing a partially-completed body.")
        if not np.all(np.isfinite(joint)):
            raise ValueError("motion completion returned non-finite joint angles")
        return CompletedChunk(
            joint_pos=joint, joint_vel=joint_velocity(joint, float(cond.dt)),
            joint_names=self.joint_names, fk_pos=fk_pos, fk_quat=fk_quat,
            body_names=self.body_names, root_pos=root_pos, root_quat=root_quat,
            valid=valid, cond=cond, windows=len(cond.starts),
            ms=(time.monotonic() - t0) * 1e3, timings=tuple(timings))


# ---------------------------------------------------------------------------
# the idle hold: the ready pose, in MC's joint order
# ---------------------------------------------------------------------------
def ready_pose_joint_targets(dof_sdk, sdk_names, mc_names) -> np.ndarray:
    """The pose the low level HOLDS after **A**, reordered into MC's joint order.

    THE PRE-ARM GATE IS WHY THIS EXISTS. In dense mode that gate is joint-space
    (``HOLOSOMA_VIVE_ARM_TOL_RAD``): it compares the broadcast wire against the 29 joints the
    robot is actually holding and REFUSES to arm if they disagree. So the planner's idle wire
    must be that same ready pose — ``home_anchor.json::default_qpos`` with the 14 arm joints
    zeroed — and it must be published in the order the wire's ``joint_names`` declares, which
    is MC's, so that the executed chunk and the idle hold are the same kind of frame.
    """
    dof = np.asarray(dof_sdk, dtype=np.float64).reshape(-1)
    if dof.size != G1_DOF:
        raise ValueError(f"expected {G1_DOF} SDK joint angles, got {dof.size}")
    return remap_by_name(dof, sdk_names, mc_names)


# ---------------------------------------------------------------------------
# the OTHER readout off the same completion: 5 keypoints in a FROZEN pelvis frame
# ---------------------------------------------------------------------------
def _quat_xyzw_scalar_positive(rotmat) -> np.ndarray:
    """``(..., 3, 3)`` -> ``(..., 4)`` xyzw quaternion with the SCALAR part ``>= 0``.

    THE CANONICALISATION IS NOT COSMETIC. ``q`` and ``-q`` are one rotation but two
    different observation vectors, and the 5pt policy was trained on the ``+w`` branch only
    (``na475b60``'s normalizer: w mean +0.73 hands / +0.98 head / +0.96 feet), so an antipode
    is ~20 sigma OOD and reads on the rig as the arms snapping to a nonsense pose. Nothing
    upstream pins the branch — scipy picks a representative per matrix — so a chest pose near
    the 180 deg seam would otherwise flip every keypoint at once. Same fix, same reason, as
    ``mc_teleop_source._five_point_locked`` and ``bake_motions._fk_into``.
    """
    from fuji.utils.rot6d import rotmat_to_quat_xyzw

    quat = np.asarray(rotmat_to_quat_xyzw(np.asarray(rotmat, dtype=np.float64)),
                      dtype=np.float64)
    # xyzw: the scalar is the LAST component. Flipping on the wrong index is a silent
    # 180 deg error, which is why this is spelled with a name rather than a bare [..., 3].
    return np.where(quat[..., 3:4] < 0.0, -quat, quat)


def relativize_keypoints(pos_world, rotmat_world, pelvis_pos, pelvis_rotmat=None):
    """World keypoint poses -> the ``(pos, orn)`` pair the keypoint wire carries.

    Args:
      pos_world: ``(H, n, 3)`` keypoint positions, lever already applied.
      rotmat_world: ``(H, n, 3, 3)`` the BODY rotations (a lever does not rotate a point,
        and holosoma's orientation term is the body's own orientation).
      pelvis_pos / pelvis_rotmat: the frame to express them in. ``None`` rotation = identity.
    Returns:
      ``((H, 3n) float64, (H, 4n) float64)`` — ``p_local = R_pelvis^T (p - p_pelvis)`` and
      ``q_local = q_pelvis^-1 (x) q_body`` as xyzw with a non-negative scalar, which is
      exactly ``holosoma_wbt._compute_vr_fk``'s ``vr_pos_local`` / ``vr_orn_local``.

    Pure numpy on purpose: this is the whole of the wire's maths, so it is testable without
    the motion-completion submodule, a GPU or a robot.
    """
    pos = np.asarray(pos_world, dtype=np.float64)
    rot = np.asarray(rotmat_world, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"pos_world must be (H, n, 3), got {pos.shape}")
    if rot.shape != pos.shape[:2] + (3, 3):
        raise ValueError(f"rotmat_world must be {pos.shape[:2] + (3, 3)}, got {rot.shape}")
    p_pelvis = np.asarray(pelvis_pos, dtype=np.float64).reshape(3)
    R_pelvis = (np.eye(3) if pelvis_rotmat is None
                else np.asarray(pelvis_rotmat, dtype=np.float64).reshape(3, 3))
    local_pos = np.einsum("ij,hnj->hni", R_pelvis.T, pos - p_pelvis)
    local_rot = np.einsum("ij,hnjk->hnik", R_pelvis.T, rot)
    orn = _quat_xyzw_scalar_positive(local_rot)
    n = pos.shape[1]
    return local_pos.reshape(-1, 3 * n), orn.reshape(-1, 4 * n)


class KeypointReadout:
    """Read a completed chunk out as 5 keypoints in a FROZEN pelvis frame.

    ⚠️ WHICH PELVIS IS THE WHOLE BALLGAME, and on a planner it is not the same choice the
    teleop bridge makes. Three facts decide it:

    1. The low level's keypoint pre-arm gate compares the wire against
       ``_compute_vr_fk(motion_command_t[0])`` — the FROZEN reference clip frame, which under
       ``HOLOSOMA_INIT_READY_POSE=1`` is the READY POSE. So the gate's reference is a
       CONSTANT expressed in the ready pose's own pelvis.
    2. This planner decodes its chunk against exactly that ready pose
       (``ready_pose.resolve_deploy_anchor``), whose world frame is the pelvis frame lifted
       to ``floor z = 0`` — i.e. pelvis at ``(0, 0, pelvis_z)``, rotation identity.
    3. So reading out against the ANCHOR's pelvis makes step 0 of every chunk land on
       ``_compute_vr_fk(ready pose)`` bit-for-bit, and the gate closes by construction with
       no clutch on this side at all.

    ``mc_teleop_source`` instead uses ``mc_frames.pelvis_ref_pose`` (``PELVIS_STAND_Z``
    = 0.79186) because its completion is placed by ``cond_mode="home"`` on the zero
    configuration. Borrowing that number here would shift every keypoint's z by **-35 mm**
    (0.79186 vs this rig's ready-pose 0.7567) — the same class of error as letting
    ``auto_dz_chest`` pick the lift, and just as invisible on the wire.

    What is NOT done here, deliberately: reading out in the completion's own per-frame root.
    That cancels the body motion twice over — measured on the teleop path, a 0.30 m walk moved
    the commanded points 7.3 mm, i.e. 97.6% of the operator's motion went nowhere.
    """

    def __init__(self, pelvis_pos, pelvis_rotmat=None, *, hand_point: str = "tcp") -> None:
        self.pelvis_pos = np.asarray(pelvis_pos, dtype=np.float64).reshape(3).copy()
        self.pelvis_rotmat = (None if pelvis_rotmat is None
                              else np.asarray(pelvis_rotmat, dtype=np.float64).reshape(3, 3))
        self.hand_point = str(hand_point)

    def describe(self) -> str:
        turn = "identity rotation" if self.pelvis_rotmat is None else "a non-identity rotation"
        return f"frozen pelvis {np.round(self.pelvis_pos, 4).tolist()}, {turn}"

    def five_point(self, completed: CompletedChunk) -> tuple[np.ndarray, np.ndarray]:
        """``(H, 15)`` positions + ``(H, 20)`` xyzw quats, in the policy's slot order.

        The bodies and their levers come from ``mc_frames.select_keypoints`` rather than from
        a table here, because that function is holosoma's contract quoted in ONE place (its
        own test re-reads ``robot_config.py`` with ``ast``), so a preset change there fails
        there instead of drifting silently into this wire.
        """
        frames, _, _ = _MODULES
        if frames is None:  # pragma: no cover - the planner always has a started completer
            raise RuntimeError("mc_frames is not loaded; start the ChunkCompleter first")
        got = frames.select_keypoints(
            completed.fk_pos, completed.fk_quat, list(completed.body_names),
            mode="5pt", hand_point=self.hand_point)
        names = tuple(str(n) for n in got["names"])
        if names != FIVE_POINT_BODIES:
            raise RuntimeError(
                f"select_keypoints('5pt') returned {names}, but this wire's slot order is "
                f"{FIVE_POINT_BODIES}. holosoma's VR_5POINT_BODIES changed; the labels and "
                "the gated-point count here must be revisited before publishing.")
        rot = frames.quat_wxyz_to_R(
            np.asarray(got["quat"], dtype=np.float64).reshape(-1, 4)
        ).reshape(got["quat"].shape[0], len(names), 3, 3)
        return relativize_keypoints(got["pos"], rot, self.pelvis_pos, self.pelvis_rotmat)

    def upper_from_decoded(self, decoded: dict) -> tuple[np.ndarray, np.ndarray]:
        """``(H, 9)`` positions + ``(H, 12)`` xyzw quats for the THREE PREDICTED points.

        The policy's own chunk, relativized into this readout's frozen pelvis frame instead of
        being read back out of the completion's FK. Used by
        ``FivePointWire(..., upper_source="policy")``.

        ⚠️ It is the SAME QUANTITY either way, to within the completion's tracking error, and
        that error is the argument for this path rather than against it: the completion answers
        with the conditioned hand within **0.84 mm** of what it was told, so nothing about the
        wire changes measurably — what changes is that a 0.84 mm discrepancy, a wrist-IK failure,
        or a future completion with a looser hand objective can no longer move the three points
        the low level actually GATES on. The feet still come from the completion (or are held);
        nothing else can produce them.

        The LEVERS line up by construction and that is worth stating, because it is the one
        thing that would be silent if it did not: a decoded hand pose is un-relativized against
        the anchor's own hand pose, which ``resolve_deploy_anchor`` already places on the 0.242 m
        fingertip TCP — the same point ``mc_frames.select_keypoints(hand_point="tcp")`` reads and
        the same one holosoma's ``VR_5POINT_OFFSETS_TCP`` declares. Likewise the chest: the
        anchor's head role is ``torso_link`` plus ``home_anchor.json``'s own chest lever, which
        is where the 5pt wire's third point lives.
        """
        missing = [r for r in _UPPER_ROLES if r not in decoded]
        if missing:
            raise ValueError(f"decoded chunk has no {missing} role(s); the upper three points "
                             f"of a 5pt wire are {list(_UPPER_ROLES)}")
        pos = np.stack([np.asarray(decoded[r]["position"], dtype=np.float64).reshape(-1, 3)
                        for r in _UPPER_ROLES], axis=1)                       # (H, 3, 3)
        rot = np.stack([rot6d_to_rotmat(np.asarray(decoded[r]["rot6d"], dtype=np.float64))
                        for r in _UPPER_ROLES], axis=1)                       # (H, 3, 3, 3)
        return relativize_keypoints(pos, rot, self.pelvis_pos, self.pelvis_rotmat)

    def all_from_decoded(self, decoded: dict) -> tuple[np.ndarray, np.ndarray]:
        """``(H, 15)`` positions + ``(H, 20)`` xyzw quats for ALL FIVE points, from the POLICY.

        The ``diffusion_umi_5pt`` path (:mod:`fuji.s1_policy.serving.hl_5pt`): that layout predicts
        every keypoint the wire carries, so there is no completion in the chain at all and this is
        the whole readout rather than an override of three slots.

        Same relativisation, same frozen pelvis and the SAME SLOT ORDER as
        :meth:`five_point` — enforced by reading :data:`FIVE_POINT_BODIES` and mapping each body
        back to the layout role that predicts it, instead of listing five role names here. A
        re-ordering of holosoma's ``VR_5POINT_BODIES`` then moves both readouts together, which is
        the property that matters: the gate, the labels and the published vector are all indexed
        by that order.

        The LEVERS line up by construction, exactly as :meth:`upper_from_decoded` documents for
        the three: a decoded pose is un-relativized against ``resolve_deploy_anchor``'s own pose
        for that role, which already carries the 0.242 m fingertip lever for a hand,
        ``home_anchor.json``'s chest lever for the torso, and NO lever for an ankle — matching
        ``mc_frames.select_keypoints(hand_point="tcp")`` and holosoma's
        ``VR_5POINT_OFFSETS_TCP`` slot for slot.
        """
        roles = [_ROLE_BY_FIVE_POINT_BODY[b] for b in FIVE_POINT_BODIES]
        missing = [r for r in roles if r not in decoded]
        if missing:
            raise ValueError(
                f"decoded chunk has no {missing} role(s); a DIRECT 5pt wire needs all five "
                f"{roles} — the policy predicts every point, so a missing one cannot be filled "
                f"in by a completion. Serve a fivepoint_geom_* checkpoint.")
        pos = np.stack([np.asarray(decoded[r]["position"], dtype=np.float64).reshape(-1, 3)
                        for r in roles], axis=1)                          # (H, 5, 3)
        rot = np.stack([rot6d_to_rotmat(np.asarray(decoded[r]["rot6d"], dtype=np.float64))
                        for r in roles], axis=1)                          # (H, 5, 3, 3)
        return relativize_keypoints(pos, rot, self.pelvis_pos, self.pelvis_rotmat)

    def from_bodies(self, pos_by_body: dict, rotmat_by_body: dict) -> tuple[np.ndarray, np.ndarray]:
        """ONE frame's ``(15,), (20,)`` from already-FK'd body poses, keyed by body name.

        The idle-hold path: the ready pose's five keypoints come from
        ``ready_pose.resolve_deploy_anchor``'s own URDF FK, not from a completion, because the
        wire has to hold the gate's reference before any chunk exists.
        """
        missing = [b for b in FIVE_POINT_BODIES if b not in pos_by_body or b not in rotmat_by_body]
        if missing:
            raise ValueError(f"no FK pose supplied for {missing}; the 5pt wire needs all "
                             f"{len(FIVE_POINT_BODIES)} bodies")
        pos = np.stack([np.asarray(pos_by_body[b], dtype=np.float64).reshape(3)
                        for b in FIVE_POINT_BODIES])[None, ...]
        rot = np.stack([np.asarray(rotmat_by_body[b], dtype=np.float64).reshape(3, 3)
                        for b in FIVE_POINT_BODIES])[None, ...]
        out_pos, out_orn = relativize_keypoints(pos, rot, self.pelvis_pos, self.pelvis_rotmat)
        return out_pos[0], out_orn[0]


def ready_pose_five_point(anchor, readout: KeypointReadout) -> tuple[np.ndarray, np.ndarray]:
    """The READY POSE as a 5pt wire frame — the idle hold, and the gate's own reference.

    ``anchor.pos``/``rotmat`` already carry each role's lever (0.242 m fingertip TCP for a
    hand, holosoma's own chest lever for the head) and ``anchor.foot_*`` the bare ankle
    origins, so this is a re-keying by body name plus the one relativisation.
    """
    role_body = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link",
                 "head": "torso_link"}
    foot_body = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
    pos, rot = {}, {}
    for role, body in role_body.items():
        if role not in anchor.pos:
            raise ValueError(
                f"the deploy anchor has no {role!r} role, so the 5pt wire cannot be built. "
                "This planner already refuses a checkpoint without all three roles.")
        pos[body], rot[body] = anchor.pos[role], anchor.rotmat[role]
    for side, body in foot_body.items():
        if side not in getattr(anchor, "foot_pos", {}):
            raise ValueError(
                f"the deploy anchor carries no {side} ankle pose, so the 5pt wire's FOOT "
                "keypoints cannot be built (ready_pose.resolve_deploy_anchor supplies them; "
                "an anchor constructed by hand does not).")
        pos[body], rot[body] = anchor.foot_pos[side], anchor.foot_rotmat[side]
    return readout.from_bodies(pos, rot)


def keypoint_gaps_m(pos_a, pos_b) -> np.ndarray:
    """Per-point Euclidean gap between two keypoint vectors — the gate's own metric."""
    a = np.asarray(pos_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(pos_b, dtype=np.float64).reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError(f"keypoint vectors disagree: {a.shape} vs {b.shape}")
    return np.linalg.norm(a - b, axis=1)


# ---------------------------------------------------------------------------
# the POST-COMPLETION command anchor
# ---------------------------------------------------------------------------
#: Where the next chunk's WIRE continues from, decided AFTER motion completion.
#:
#: ``off`` (default) is the previous behaviour of whichever wire is selected, byte for byte:
#: the dense clutch re-bases on the robot's MEASURED joints at every Predict, and the 5pt wire
#: publishes the readout absolutely with no base at all.
#:
#: ``command`` carries the LAST EXECUTED published step forward instead. This is
#: :mod:`fuji.s1_policy.serving.hl_anchor`'s ``mode="command"`` — PLAI (IndustReal, arXiv
#: 2305.17110) — moved DOWNSTREAM of the completion, and that position is the whole point:
#:
#:     3 tracker poses ──HL──▶ chunk ──MC──▶ 29 joints ──▶ [HERE] ──▶ 5pt / dense wire
#:
#: ⚠️ It CANNOT be applied where ``hl_serving`` applies it (to the decode anchor), because on
#: this path the decode anchor is also the completion's ROOT conditioning: moving it changes
#: the POSTURE the completion answers with, not just where the hands go, and the low level's
#: pre-arm gate references FK of that one fixed pose. So the integrator has to sit after the
#: completion, on the quantity actually published — 29 joint targets or 5 keypoint poses.
POST_ANCHOR_MODES = ("off", "command")

#: Hard ceiling on how far a carried JOINT command may lead the measured joints, whatever the
#: operator types. The low level's own pre-arm tolerance is 0.20 rad (``HOLOSOMA_VIVE_ARM_TOL_RAD``)
#: and its per-step slew budget is 0.0833 rad, so a lead past 0.30 rad is a wind-up that would
#: both fail a re-arm and take 4+ ticks to unwind.
MAX_POST_EPS_RAD = 0.30


@dataclass(frozen=True)
class PostAnchorConfig:
    """``mode="off"`` (default) leaves every published number exactly as it was.

    ``eps_joint_rad`` bounds the DENSE lead (joint space); ``eps_m`` / ``eps_rad`` bound the
    5pt lead (per keypoint, position and geodesic orientation). ``contact_n`` is the measured
    contact magnitude at or above which the carried anchor DECAYS back toward the robot instead
    of advancing, and ``decay`` is that pull-back factor per Predict — the same defaults, and
    the same reasoning, as :class:`~fuji.s1_policy.serving.hl_anchor.AnchorConfig`, because
    this is that mechanism in a different space rather than a new idea.
    """

    mode: str = "off"
    eps_joint_rad: float = 0.15
    eps_m: float = 0.03
    eps_rad: float = 0.175  # 10 deg
    contact_n: float = 1.5
    decay: float = 0.9

    def __post_init__(self) -> None:
        if self.mode not in POST_ANCHOR_MODES:
            raise ValueError(
                f"post-anchor mode must be one of {POST_ANCHOR_MODES}, got {self.mode!r}")
        if not (0.0 <= self.eps_joint_rad <= MAX_POST_EPS_RAD):
            raise ValueError(
                f"eps_joint_rad must be in [0, {MAX_POST_EPS_RAD}] rad, got {self.eps_joint_rad}. "
                "This bounds an un-asked-for lead on a physical robot; raise MAX_POST_EPS_RAD "
                "deliberately if you mean it.")
        if not (0.0 <= self.eps_m <= hl_anchor.MAX_EPS_M):
            raise ValueError(
                f"eps_m must be in [0, {hl_anchor.MAX_EPS_M}] m, got {self.eps_m}")
        if not (0.0 <= self.eps_rad <= hl_anchor.MAX_EPS_RAD):
            raise ValueError(
                f"eps_rad must be in [0, {hl_anchor.MAX_EPS_RAD}] rad, got {self.eps_rad}")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        if self.contact_n < 0.0:
            raise ValueError(f"contact_n must be >= 0, got {self.contact_n}")

    @property
    def carrying(self) -> bool:
        return self.mode == "command"

    def describe(self) -> str:
        if not self.carrying:
            return "off (re-base on measured at every Predict — the previous behaviour)"
        return (f"command (carry the last executed step; leak {self.eps_joint_rad:g} rad / "
                f"{self.eps_m * 1e3:.0f} mm / {np.degrees(self.eps_rad):.0f} deg, "
                f"decay above {self.contact_n:g} N)")


def _leak_scalar(carried: np.ndarray, measured: np.ndarray, eps: float) -> tuple[np.ndarray, bool]:
    """Bound ``carried``'s per-element lead over ``measured`` to ``eps``, written BACK.

    Per-element rather than on the norm because the quantity is 29 independent joints: a norm
    bound would let one joint wind up while the others sat still. ``eps == 0`` collapses to
    ``measured``, which is how a zero leak degrades to ``mode="off"`` instead of to unbounded
    integration.
    """
    d = np.asarray(carried, dtype=np.float64) - np.asarray(measured, dtype=np.float64)
    over = np.abs(d) > eps
    if not np.any(over):
        return np.asarray(carried, dtype=np.float64).copy(), False
    out = np.asarray(carried, dtype=np.float64).copy()
    out[over] = measured[over] + np.sign(d[over]) * eps
    return out, True


# ---------------------------------------------------------------------------
# the joint clutch
# ---------------------------------------------------------------------------
#: Clutch modes. ``joint`` (default) publishes the completion's CHANGE from a running base;
#: ``none`` publishes its absolute joint angles.
CLUTCH_MODES = ("joint", "none")


class JointClutch:
    """Publish the completion's joint MOTION from a running base, not its absolute pose.

    THE MEASUREMENT THAT MAKES THIS THE DEFAULT. A 7-DoF arm is redundant, so motion
    completion is free to pick any null-space configuration that puts the hand where it was
    told. Measured on ``ckpt_0831_final``, chunk seeded exactly at the deploy ready pose:

        commanded gripper pose vs the completion's own FK   0.84 mm     (the chain is right)
        completion's ARM joints at step 0 vs zeroed arms    0.40 rad    (23 deg, null space)
        completion's KNEE at step 0 vs the ready crouch     0.54 rad    (it stands up)
        per-step |dq| WITHIN the chunk                      0.015 rad   (0.46 rad/s — smooth)

    So the hand is in the right place and the elbow is 23 degrees somewhere else. Published
    absolutely, that offset is a real command: the pre-arm gate compares the wire against the
    held pose on the ARM joints at a 0.20 rad tolerance (``HOLOSOMA_VIVE_ARM_TOL_RAD``), so
    the arms would either refuse to arm or, having armed against a ready-pose idle wire, snap
    23 deg on the first executed step — a motion the policy never asked for, at the moment the
    robot starts moving.

    The chunk it came from is anchor-relative and re-anchored on the MEASURED pose every
    Predict, i.e. this planner already commands CHANGE rather than absolute position. This is
    the joint-space form of the same contract, and it says the honest thing: the completion's
    null-space choice is not a claim about the robot, only its change is.

    ``base`` advances to the last EXECUTED step within a chunk. What happens at the chunk
    BOUNDARY is :data:`POST_ANCHOR_MODES`: ``off`` (default) re-bases on the robot's measured
    joints, so no shortfall accumulates and no jump appears; ``command`` keeps the carried
    command, leak-bounded against those same measured joints, so the accumulated command is not
    thrown away either. The two costs are opposite and both are real — see that constant.
    """

    def __init__(self, base: np.ndarray, *, mode: str = "joint",
                 post: PostAnchorConfig | None = None) -> None:
        if str(mode) not in CLUTCH_MODES:
            raise ValueError(f"mode must be one of {CLUTCH_MODES}, got {mode!r}")
        self.mode = str(mode)
        self.post = post or PostAnchorConfig()
        self._base = np.asarray(base, dtype=np.float64).reshape(-1).copy()
        if self._base.size != G1_DOF:
            raise ValueError(f"base must be {G1_DOF} joint angles, got {self._base.size}")
        self._home = self._base.copy()
        #: False until a chunk has actually been published, so the FIRST Predict after arm or
        #: Reset bootstraps from measured however the mode is set — a carried command from
        #: before an operator intervention must never be branched from.
        self._carrying = False
        self._n_sync = 0
        self._n_carry = 0
        self._n_clamp = 0
        self._n_contact = 0
        self._max_lead_rad = 0.0

    @property
    def base(self) -> np.ndarray:
        return self._base.copy()

    def reset(self, base: np.ndarray | None = None) -> None:
        """Back to the ready pose, or to a measured readback. Drops any carried command."""
        self._base = (self._home.copy() if base is None
                      else np.asarray(base, dtype=np.float64).reshape(-1).copy())
        self._carrying = False

    def sync(self, measured: np.ndarray, force_n: float | None = None) -> str:
        """Decide the base for the NEXT chunk from a measured readback. Returns a label.

        ``measured`` is already in the wire's joint order. ``force_n`` is the larger measured
        contact magnitude in newtons, or None when unknown — **None does not cut out**, for the
        same reason ``hl_anchor.ChunkAnchor.resolve`` refuses to: a missing force reading must
        not silently disable a guard the operator believes is armed.
        """
        q = np.asarray(measured, dtype=np.float64).reshape(-1)
        if q.size != G1_DOF or not np.all(np.isfinite(q)):
            return "last published (no usable readback)"
        self._n_sync += 1
        if not self.post.carrying or not self._carrying:
            self._base = q.copy()
            return "MEASURED (RecordTap)"

        self._n_carry += 1
        carried = self._base.copy()
        if force_n is not None and float(force_n) >= self.post.contact_n:
            # CONTACT: unwind toward the robot rather than advance. Checked BEFORE the clamp so
            # a hand pressing at the limit still decays instead of sitting pinned there.
            self._n_contact += 1
            carried = q + (carried - q) * self.post.decay
        lead = float(np.abs(carried - q).max())
        self._max_lead_rad = max(self._max_lead_rad, lead)
        carried, clamped = _leak_scalar(carried, q, self.post.eps_joint_rad)
        if clamped:
            self._n_clamp += 1
        if not np.all(np.isfinite(carried)):
            # Never carry a non-finite base: every published step would inherit it and holosoma
            # serializes NaN unchecked.
            self._base = q.copy()
            self._carrying = False
            return "MEASURED (a non-finite carry was dropped)"
        self._base = carried
        return f"CARRIED command (lead {lead:.3f} rad{', clamped' if clamped else ''})"

    def post_stats(self) -> dict:
        """Counters for the rig log. ``max_lead_rad`` is pre-clamp, so it may exceed the eps."""
        return {"mode": self.post.mode, "eps_joint_rad": self.post.eps_joint_rad,
                "n_sync": self._n_sync, "n_carried": self._n_carry,
                "n_clamped": self._n_clamp, "n_contact_pullback": self._n_contact,
                "max_lead_rad": self._max_lead_rad, "carrying": self._carrying}

    def apply(self, chunk_joint_pos: np.ndarray) -> np.ndarray:
        """``(H, 29)`` completion → ``(H, 29)`` to publish. ``mode="none"`` is the identity.

        Step 0 lands EXACTLY on the base, which is what makes the first executed step a
        no-op on the arm rather than a jump.
        """
        q = np.asarray(chunk_joint_pos, dtype=np.float64)
        if self.mode == "none":
            return q.copy()
        if q.ndim != 2 or q.shape[1] != G1_DOF:
            raise ValueError(f"chunk_joint_pos must be (H, {G1_DOF}), got {q.shape}")
        return q - q[0] + self._base

    def commit(self, published: np.ndarray) -> None:
        """Advance the base to the LAST EXECUTED published step.

        This is the integrator in ``mode="command"``: it is what makes the carried anchor a
        command rather than a memory of one, which is why it advances to the last *executed*
        step and not to the chunk's end (an interrupted chunk's tail was never asked for).
        """
        q = np.asarray(published, dtype=np.float64).reshape(-1)
        if q.size == G1_DOF and np.all(np.isfinite(q)):
            self._base = q.copy()
            self._carrying = True

    def drift_rad(self) -> float:
        """How far the base has travelled from the ready pose, worst joint. For the readout."""
        return float(np.abs(self._base - self._home).max())


class KeypointClutch:
    """The 5pt twin of :class:`JointClutch`: publish the chunk's keypoint CHANGE from a base.

    Only meaningful with ``PostAnchorConfig(mode="command")``; with ``off`` :meth:`apply` is the
    identity and the wire publishes the absolute readout exactly as before.

    THE OFFSET IS A CONSTANT IN THE PELVIS FRAME, per keypoint, for the whole chunk::

        p_pub[i] = p_base + (p[i] - p[0])
        R_pub[i] = (R_base @ R[0]^T) @ R[i]

    so step 0 lands on the base bit-exactly and the offset ``E = R_base @ R[0]^T`` is a fixed
    pelvis-frame rotation — the exact analogue of the constant position offset, and the reason
    the rotation is composed on the LEFT. Composing on the right instead
    (``R_base @ R[0]^T @ R[i]`` read as a body-frame delta) makes the offset get conjugated by
    the chunk's own motion, so a hand that turns would also drift.

    ⚠️ ONLY THE OPERATOR-PLACED POINTS ARE CARRIED — the two FEET pass through ABSOLUTELY, and
    that is the same distinction holosoma itself draws (``_VR_5POINT_OPERATOR_PLACED = 3``: the
    feet are warned, never gated). It was measured, not assumed. Carrying all five re-bases the
    chunk's stance onto the READY POSE's feet, and the ready pose is a 38 deg CROUCH while the
    completion stands straighter: on ``ckpt_0831_final`` that shifted the whole foot trajectory
    **74.7 mm** on the very first chunk. The feet are the completion's own inference about
    stance, not something the planner predicts or the operator aims, so re-basing them is the
    "the null space is not a command" error of :class:`JointClutch` in a different space —
    except here it would be a LEG command. :data:`FIVE_POINT_GATED` is therefore the carried
    count, and the leak never has to bound a foot because a foot is never carried.

    All three GATED points are carried together, and that is deliberate: carrying the hands but
    not the torso would distort the hand-versus-chest geometry the completion produced — the same
    objection ``display.py`` records against a per-role rebase matrix.

    ⚠️ THE FIRST CHUNK IS STILL NOT BYTE-IDENTICAL TO ``off``, and the difference is a jump this
    REMOVES. Its base is the READY frame, so step 0 lands exactly on the low level's own gate
    reference; under ``off`` step 0 is the readout's own landing instead, which is 0.7-3.5 mm on
    the hands but **41 mm on the torso** (only pelvis XY and yaw are conditioned, so the
    completion's torso height and pitch are its own inference). The wire idles at that same ready
    frame, so ``command`` makes the first executed step a bit-exact no-op where ``off`` starts it
    with a 41 mm step. Same property :class:`JointClutch` gives the dense wire. ``off`` remains
    byte-identical to the previous behaviour.

    ⚠️ WHY THIS NEEDS A MEASURED REFERENCE AND THE DENSE ONE DOES NOT GET A FREE PASS EITHER.
    Pure integration is not safe: replayed over the recorded takes, an unclamped command anchor
    diverges from the measured pose to **224 mm** by the end of a take (``hl_anchor``'s module
    docstring). The bound has to be against where the robot IS, not against the ready pose —
    clamping to home would cap the task itself, since a real reach moves the hand ~48 cm. On the
    5pt wire "where the robot is" only exists as 29 encoder readings, so the caller supplies a
    measured keypoint frame FK'd from RecordTap (``ready_pose.KeypointFK``). **With no measured
    frame the anchor is FROZEN rather than advanced** — refusing to integrate blind is the whole
    reason this class asks for the reference instead of defaulting it.
    """

    def __init__(self, base_pos: np.ndarray, base_orn: np.ndarray, *,
                 post: PostAnchorConfig | None = None,
                 n_points: int = len(FIVE_POINT_BODIES),
                 n_carried: int = FIVE_POINT_GATED) -> None:
        self.post = post or PostAnchorConfig()
        self.n_points = int(n_points)
        #: How many leading points are CARRIED; the rest pass through absolutely. Defaults to
        #: the operator-placed count (see the class docstring — carrying the feet is a 74.7 mm
        #: leg command). ``n_carried == n_points`` is available but has to be asked for.
        self.n_carried = max(0, min(int(n_carried), self.n_points))
        self._home_pos = np.asarray(base_pos, dtype=np.float64).reshape(-1).copy()
        self._home_orn = np.asarray(base_orn, dtype=np.float64).reshape(-1).copy()
        if self._home_pos.size != 3 * self.n_points or self._home_orn.size != 4 * self.n_points:
            raise ValueError(
                f"base must be ({3 * self.n_points},) positions and ({4 * self.n_points},) "
                f"quats for {self.n_points} points, got {self._home_pos.shape} / "
                f"{self._home_orn.shape}")
        self._pos = self._home_pos.copy()
        self._orn = self._home_orn.copy()
        self._carrying = False
        self._n_sync = 0
        self._n_carry = 0
        self._n_frozen = 0
        self._n_clamp = 0
        self._n_contact = 0
        self._max_lead_m = 0.0

    @property
    def base(self) -> tuple:
        return (self._pos.copy(), self._orn.copy())

    def reset(self) -> None:
        """Back to the ready-pose frame. Drops any carried command."""
        self._pos = self._home_pos.copy()
        self._orn = self._home_orn.copy()
        self._carrying = False

    def sync(self, measured: tuple | None, force_n: float | None = None, *,
             hold_slots=()) -> str:
        """Decide the base for the NEXT chunk. ``measured`` is a ``(pos, orn)`` wire frame.

        ``hold_slots`` — wire slot indices whose base is NOT leak-bounded this sync (kept exactly as
        the last ``commit`` left it). For a slot whose WIRE is being driven by something other than
        this clutch — the leg override's two feet, which are the clip's while the planner's own foot
        target keeps being committed here — "the lead against measured" is not a lead at all: the
        measured feet are walking, and bounding the planner's stance to within ``eps_m`` of them at
        every Predict would drag that stance along the gait, so the hand-back would land in a
        mid-stride split stance. Empty (the default) is byte-identical to before.
        """
        if not self.post.carrying:
            # mode="off": the wire is absolute, so there is no base to maintain. Kept as the
            # ready frame rather than left stale, so a mid-session mode flip cannot inherit one.
            self._pos = self._home_pos.copy()
            self._orn = self._home_orn.copy()
            # "keypoint", not "5pt": this clutch is width-parameterized and also backs the 3pt
            # wire (fuji/s1_policy/serving/hl_offline.ThreePointWire), where a "5pt" readout in
            # the log reads as the wrong wire family having been built.
            return "n/a (a keypoint wire is read out against the ready pose, not re-based)"
        if not self._carrying:
            return "READY frame (nothing executed yet)"
        if measured is None:
            self._n_frozen += 1
            return ("FROZEN carry — no RecordTap keypoints to bound the lead against; "
                    "the anchor is held, not advanced")
        self._n_sync += 1
        self._n_carry += 1
        # Only the CARRIED points have a lead to bound: the feet are republished absolutely, so
        # clamping them would be clamping the completion's own stance against the ready crouch.
        k = self.n_carried
        m_pos = np.asarray(measured[0], dtype=np.float64).reshape(-1, 3)[:k]
        m_rot = _rotmats_from_quat_xyzw(measured[1], self.n_points)[:k]
        pos = self._pos.reshape(-1, 3)[:k].copy()
        rot = _rotmats_from_quat_xyzw(self._orn, self.n_points)[:k]

        if force_n is not None and float(force_n) >= self.post.contact_n:
            self._n_contact += 1
            pos = m_pos + (pos - m_pos) * self.post.decay
            rot = _rot_toward_batch(m_rot, rot, self.post.decay)

        lead = np.linalg.norm(pos - m_pos, axis=1)
        self._max_lead_m = max(self._max_lead_m, float(lead.max()))
        over = lead > self.post.eps_m
        if self.post.eps_m <= 0.0:
            pos = m_pos.copy()
            self._n_clamp += 1
        elif np.any(over):
            scale = np.where(over, self.post.eps_m / np.maximum(lead, 1e-12), 1.0)[:, None]
            pos = m_pos + (pos - m_pos) * scale
            self._n_clamp += 1
        rot, clamped = _clamp_rot_batch(m_rot, rot, self.post.eps_rad)
        if clamped:
            self._n_clamp += 1

        orn = _quat_xyzw_scalar_positive(rot).reshape(-1)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(orn))):
            self.reset()
            return "READY frame (a non-finite carry was dropped)"
        # Write back only the carried slice; the tail keeps whatever `commit` last stored, which
        # `apply` ignores anyway (the feet are republished from the chunk). A HELD slot keeps its
        # committed base too — its wire is not this clutch's to bound (see the docstring).
        held = sorted({int(i) for i in hold_slots if 0 <= int(i) < k})
        prev_pos = self._pos.reshape(-1, 3).copy()
        prev_orn = self._orn.reshape(-1, 4).copy()
        self._pos = prev_pos.copy()
        self._pos[:k] = pos
        self._orn = prev_orn.copy()
        self._orn[:k] = orn.reshape(-1, 4)
        for i in held:
            self._pos[i] = prev_pos[i]
            self._orn[i] = prev_orn[i]
        self._pos = self._pos.reshape(-1)
        self._orn = self._orn.reshape(-1)
        bounded = [i for i in range(k) if i not in held]
        lead_b = lead[bounded] if bounded else lead[:0]
        lead_note = f"lead {float(lead_b.max()) * 1e3:.0f} mm" if lead_b.size else "no bounded slot"
        return (f"CARRIED command ({lead_note}"
                f"{', clamped' if (clamped or np.any(over[bounded])) else ''}"
                f"{f', {len(held)} slot(s) HELD (externally driven)' if held else ''})")

    def apply(self, pos: np.ndarray, orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(H, 15) / (H, 20)`` readout -> what to publish. ``mode="off"`` is the identity."""
        p = np.asarray(pos, dtype=np.float64)
        q = np.asarray(orn, dtype=np.float64)
        if not self.post.carrying:
            return p.copy(), q.copy()
        if p.ndim != 2 or p.shape[1] != 3 * self.n_points:
            raise ValueError(f"pos must be (H, {3 * self.n_points}), got {p.shape}")
        if q.shape != (p.shape[0], 4 * self.n_points):
            raise ValueError(
                f"orn must be ({p.shape[0]}, {4 * self.n_points}), got {q.shape}")
        h, k = p.shape[0], self.n_carried
        # The carried slice only. Everything past `n_carried` (the FEET) is copied through
        # untouched — see the class docstring: re-basing the stance is a leg command.
        out_pos = p.copy().reshape(h, self.n_points, 3)
        out_pos[:, :k] += (self._pos.reshape(-1, 3)[:k] - out_pos[0, :k])
        rot = _rotmats_from_quat_xyzw(q.reshape(-1), self.n_points * h).reshape(
            h, self.n_points, 3, 3)
        out_rot = rot.copy()
        if k:
            base_rot = _rotmats_from_quat_xyzw(self._orn, self.n_points)[:k]
            # E = R_base @ R[0]^T, one fixed pelvis-frame rotation per carried keypoint.
            offset = np.einsum("nij,nkj->nik", base_rot, rot[0, :k])
            out_rot[:, :k] = np.einsum("nij,hnjk->hnik", offset, rot[:, :k])
        return (out_pos.reshape(h, -1),
                _quat_xyzw_scalar_positive(out_rot).reshape(h, -1))

    def commit(self, frame: tuple) -> None:
        """Advance the base to the LAST EXECUTED published step."""
        if not self.post.carrying:
            return
        p = np.asarray(frame[0], dtype=np.float64).reshape(-1)
        q = np.asarray(frame[1], dtype=np.float64).reshape(-1)
        if (p.size == 3 * self.n_points and q.size == 4 * self.n_points
                and np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            self._pos = p.copy()
            self._orn = q.copy()
            self._carrying = True

    def drift_m(self) -> float:
        """How far the base has travelled from the ready frame, worst CARRIED keypoint.

        The feet are excluded because they are not carried: they sit 69-91 mm from the ready
        crouch by construction, so including them would report a permanent 70 mm "drift" that
        no command produced.
        """
        if not self.n_carried:
            return 0.0
        gaps = keypoint_gaps_m(self._pos, self._home_pos)[:self.n_carried]
        return float(gaps.max())

    def post_stats(self) -> dict:
        return {"mode": self.post.mode, "eps_m": self.post.eps_m,
                "n_carried_points": self.n_carried,
                "n_sync": self._n_sync, "n_carried": self._n_carry,
                "n_frozen_no_reference": self._n_frozen, "n_clamped": self._n_clamp,
                "n_contact_pullback": self._n_contact, "max_lead_m": self._max_lead_m,
                "carrying": self._carrying}


def _rotmats_from_quat_xyzw(orn, n_points: int) -> np.ndarray:
    """``(4n,)`` xyzw -> ``(n, 3, 3)``. The inverse of :func:`_quat_xyzw_scalar_positive`."""
    from fuji.utils.rot6d import quat_xyzw_to_rotmat

    return quat_xyzw_to_rotmat(np.asarray(orn, dtype=np.float64).reshape(n_points, 4))


def _rot_toward_batch(base: np.ndarray, rot: np.ndarray, keep: float) -> np.ndarray:
    """``rot`` moved a fraction ``1 - keep`` back toward ``base``, per keypoint (geodesic)."""
    from fuji.utils.rot6d import rotmat_to_rotvec, rotvec_to_rotmat

    out = np.empty_like(rot)
    for k in range(rot.shape[0]):
        rv = rotmat_to_rotvec(base[k].T @ rot[k])
        out[k] = base[k] @ rotvec_to_rotmat(rv * float(keep))
    return out


def _clamp_rot_batch(base: np.ndarray, rot: np.ndarray,
                     eps_rad: float) -> tuple[np.ndarray, bool]:
    """Bound each keypoint's geodesic deviation from ``base`` to ``eps_rad``."""
    from fuji.utils.rot6d import rotmat_to_rotvec, rotvec_to_rotmat

    out = np.array(rot, dtype=np.float64, copy=True)
    any_clamped = False
    for k in range(rot.shape[0]):
        rv = rotmat_to_rotvec(base[k].T @ rot[k])
        ang = float(np.linalg.norm(rv))
        if eps_rad <= 0.0:
            out[k] = base[k]
            any_clamped = any_clamped or ang > 0.0
        elif ang > eps_rad:
            out[k] = base[k] @ rotvec_to_rotmat(rv * (eps_rad / ang))
            any_clamped = True
    return out, any_clamped


# ---------------------------------------------------------------------------
# what the low level is allowed to receive
# ---------------------------------------------------------------------------
def is_safe_dense_command(joint_pos, joint_vel=None, *, max_rad: float = 6.5) -> tuple[bool, str]:
    """``(ok, reason)`` for one dense wire frame. Non-finite must NEVER reach the wire.

    ``json.dumps`` serializes ``NaN`` unchecked and ``ViveLiveSource`` validates only WIDTH,
    so nothing downstream would catch it — the same hole ``fuji.client.is_safe_to_send`` plugs
    for the 3-point wire, restated for 29 joints. ``max_rad`` is a sanity bound, not a joint
    limit table: the G1's widest range is under ~2.9 rad from zero, so anything past 6.5 is a
    decode or units failure rather than an aggressive pose, and the low level's own per-joint
    slew (``HOLOSOMA_VIVE_MAX_JOINT_STEP``) is what makes a legal-but-large target safe.
    """
    q = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
    if q.size != G1_DOF:
        return False, f"joint_pos has {q.size} entries, want {G1_DOF}"
    if not np.all(np.isfinite(q)):
        return False, f"joint_pos is non-finite at index {int(np.argmax(~np.isfinite(q)))}"
    worst = float(np.abs(q).max())
    if worst > max_rad:
        return False, f"|joint target| {worst:.2f} rad exceeds the {max_rad:g} rad sanity bound"
    if joint_vel is not None:
        v = np.asarray(joint_vel, dtype=np.float64).reshape(-1)
        if v.size != G1_DOF:
            return False, f"joint_vel has {v.size} entries, want {G1_DOF}"
        if not np.all(np.isfinite(v)):
            return False, f"joint_vel is non-finite at index {int(np.argmax(~np.isfinite(v)))}"
    return True, ""


def is_safe_keypoint_command(pos, orn, *, n_points: int = 5,
                             max_reach_m: float = 1.5) -> tuple[bool, str]:
    """``(ok, reason)`` for one KEYPOINT wire frame. The 5pt twin of the dense gate.

    ``ViveLiveSource`` validates the ``(len(pos), len(orn))`` PAIR and nothing else — not
    finiteness, not unit norm — and ``json.dumps`` serializes ``NaN`` unchecked, so this is
    again the only place a bad frame can be stopped. Three checks:

    * WIDTH, because a 3pt-width frame would be silently accepted as a 3pt command;
    * FINITE, for the reason above;
    * a **unit-norm** quaternion, because the low level composes these with the arm clutch's
      anchor and a de-normalised quat there scales the commanded rotation rather than failing.

    ``max_reach_m`` is a sanity bound on a PELVIS-RELATIVE point, not a workspace model: the
    G1's hand reaches ~0.9 m and its ankle sits ~0.8 m below the pelvis, so 1.5 m is a decode
    or a frame error rather than an ambitious pose.
    """
    p = np.asarray(pos, dtype=np.float64).reshape(-1)
    q = np.asarray(orn, dtype=np.float64).reshape(-1)
    if p.size != 3 * n_points:
        return False, f"pos has {p.size} entries, want {3 * n_points} ({n_points} points)"
    if q.size != 4 * n_points:
        return False, f"orn has {q.size} entries, want {4 * n_points} ({n_points} quats)"
    if not np.all(np.isfinite(p)):
        return False, f"pos is non-finite at index {int(np.argmax(~np.isfinite(p)))}"
    if not np.all(np.isfinite(q)):
        return False, f"orn is non-finite at index {int(np.argmax(~np.isfinite(q)))}"
    worst = float(np.abs(p.reshape(-1, 3)).max())
    if worst > max_reach_m:
        return False, (f"|keypoint| {worst:.2f} m exceeds the {max_reach_m:g} m sanity bound "
                       "— the wire is pelvis-RELATIVE, so this is a frame error")
    norms = np.linalg.norm(q.reshape(-1, 4), axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-3):
        bad = int(np.argmax(np.abs(norms - 1.0)))
        return False, (f"orn quaternion {bad} ({FIVE_POINT_LABELS[bad]}) has norm "
                       f"{norms[bad]:.4f}, not 1")
    return True, ""


# ---------------------------------------------------------------------------
# the wire, one class per low-level FAMILY
# ---------------------------------------------------------------------------
# WHY POLYMORPHIC AND NOT `if mode == "5pt"` IN THE PLANNER. The planner touches the wire in
# eight places (the idle heartbeat, the idle re-sync, Predict, the drip, Stop, Reset, the
# status line, the safety gate) and every one of them is safety-relevant. Threading a family
# flag through all eight is how a frame ends up published through the wrong gate. Each class
# below owns "what a completed chunk becomes", "what idles", and "what may reach the socket"
# for exactly one family, so the planner's loop is family-agnostic and a new family is a new
# class rather than eight new branches.
class DenseWire:
    """``mode="dense"``: 29 absolute joint targets, for the ``dense`` / ``wbt`` family.

    Carries the :class:`JointClutch`, because publishing the completion's ABSOLUTE joints is
    a 0.40 rad arm command the policy never asked for (see that class).
    """

    mode = "dense"
    #: What the low level must have been launched as, quoted in the planner's banner.
    low_level_mode = "dense-unified-anneal"
    gate_note = "joint-space (HOLOSOMA_VIVE_ARM_TOL_RAD, 0.20 rad)"

    def __init__(self, joint_names, sdk_names, clutch: JointClutch, ready_mc: np.ndarray) -> None:
        self.joint_names = tuple(str(n) for n in joint_names)
        self.sdk_names = tuple(str(n) for n in sdk_names)
        self.clutch = clutch
        self._ready = np.asarray(ready_mc, dtype=np.float64).reshape(-1).copy()

    def idle(self) -> tuple:
        """The frame published before anything executes: the ready pose, at rest."""
        return (self._ready.copy(), np.zeros(G1_DOF))

    def measured(self, rframe) -> tuple | None:
        """The robot's MEASURED joints as a wire frame, for the un-armed idle re-sync.

        Velocity ZERO: the robot is holding, not moving, and a leftover reference velocity
        from the last executed step would tell the tracker otherwise.
        """
        if rframe is None or rframe.dof_pos is None:
            return None
        try:
            return (remap_by_name(rframe.dof_pos, self.sdk_names, self.joint_names),
                    np.zeros(G1_DOF))
        except Exception:  # noqa: BLE001 - a bad readback holds, never re-poses the robot
            return None

    def sync(self, rframe, force_n: float | None = None) -> str:
        """Decide the clutch base for the next chunk. Returns a readout label.

        Which of the two things this does is :data:`POST_ANCHOR_MODES` — the clutch owns that
        decision, so this stays one call whichever mode is armed.
        """
        frame = self.measured(rframe)
        if frame is None:
            return "last published"
        return self.clutch.sync(frame[0], force_n)

    def plan(self, completed: CompletedChunk, decoded: dict) -> dict:
        del decoded  # the dense wire is the completion's joints; the EE decode is only drawn
        published = self.clutch.apply(completed.joint_pos)
        return {"published": published, "qvel": joint_velocity(published, 1.0 / MC_RATE_HZ),
                "n": int(published.shape[0])}

    def frame(self, plan: dict, i: int) -> tuple:
        return (plan["published"][i], plan["qvel"][i])

    def publish(self, s0, frame: tuple, f_cmd) -> tuple[bool, str]:
        ok, reason = is_safe_dense_command(frame[0], frame[1])
        if not ok:
            return False, reason
        s0.send_dense(frame[0], frame[1], self.joint_names, f_cmd=f_cmd)
        return True, ""

    def commit(self, frame: tuple) -> None:
        self.clutch.commit(frame[0])

    def status(self, frame: tuple) -> str:
        worst = float(np.abs(frame[0] - self._ready).max())
        carry = ("" if not self.clutch.post.carrying
                 else f" · carried lead {self.clutch.post_stats()['max_lead_rad']:.3f} rad")
        return (f"dense {G1_DOF} joints · worst |dq| vs ready {worst:.3f} rad · "
                f"clutch drift {self.clutch.drift_rad():.3f} rad{carry}")


class FivePointWire:
    """``mode="5pt"``: five keypoint poses, for the ``5pt-*`` family.

    NO NULL-SPACE CLUTCH ON THIS SIDE, and that is a consequence rather than an omission. The
    dense wire needs one because the completion's null-space choice is a 0.40 rad absolute joint
    command; a keypoint wire carries no null space — it carries the very points the completion
    was told to hit (to 0.84 mm) plus the two feet it inferred. Read out against the ANCHOR's
    pelvis (:class:`KeypointReadout`), step 0 therefore lands on ``_compute_vr_fk(ready pose)``,
    which is exactly what the low level's pre-arm gate compares against — so the gate closes by
    construction, and the low level's own absolute arm clutch (``HOLOSOMA_VIVE_ABS_CLUTCH``,
    default on, per-point ``held - wire`` at arm) has nothing left to absorb.

    The idle hold is that same constant, and it is NEVER chased toward the measured pose: the
    gate's reference is the frozen reference clip, not the robot's encoders, so tracking
    measured would move the wire AWAY from what the gate wants. (The dense wire tracks because
    its own idle hold was the subject of two live bugs; the difference is stated in the runbook
    rather than smoothed over.)

    ⚠️ ``PostAnchorConfig(mode="command")`` is a SEPARATE question from the null space, and it
    does add a base here — :class:`KeypointClutch`, which carries the last executed keypoint
    frame forward instead of re-reading every chunk off the fixed ready pose. With ``mode="off"``
    (the default) that clutch is the identity and everything above holds unchanged.
    """

    mode = "5pt"
    low_level_mode = "5pt-unified-anneal"
    gate_note = (f"keypoint-space (HOLOSOMA_VIVE_ARM_TOL, {KEYPOINT_GATE_TOL_M:g} m) on "
                 f"{FIVE_POINT_GATED} of {len(FIVE_POINT_BODIES)} points — the FEET are the "
                 "completion's inference and are warned, not gated")

    def __init__(self, readout: KeypointReadout, ready_pos: np.ndarray,
                 ready_orn: np.ndarray, *, clutch: KeypointClutch | None = None,
                 measured_frame=None, upper_source: str = "completion",
                 foot_source: str = "completion") -> None:
        if upper_source not in UPPER_SOURCES:
            raise ValueError(f"upper_source must be one of {UPPER_SOURCES}, got "
                             f"{upper_source!r}")
        if foot_source not in FOOT_SOURCES:
            raise ValueError(f"foot_source must be one of {FOOT_SOURCES}, got {foot_source!r}")
        self.readout = readout
        self.upper_source = str(upper_source)
        self.foot_source = str(foot_source)
        self._ready_pos = np.asarray(ready_pos, dtype=np.float64).reshape(-1).copy()
        self._ready_orn = np.asarray(ready_orn, dtype=np.float64).reshape(-1).copy()
        self.clutch = clutch or KeypointClutch(self._ready_pos, self._ready_orn)
        #: ``rframe -> (pos, orn) | None``: the ROBOT's measured keypoints, FK'd from RecordTap,
        #: which is the only thing a carried command can be leak-bounded against. Injected
        #: rather than built here because the FK needs a URDF and this module must import
        #: without one. ``None`` ⇒ ``KeypointClutch.sync`` freezes instead of integrating blind.
        self._measured_frame = measured_frame

    @property
    def ready_frame(self) -> tuple:
        return (self._ready_pos.copy(), self._ready_orn.copy())

    def idle(self) -> tuple:
        return self.ready_frame

    def measured(self, rframe) -> tuple | None:
        del rframe   # see the class docstring: the IDLE hold's reference is a CONSTANT
        return None

    def measured_keypoints(self, rframe) -> tuple | None:
        """The ROBOT's measured ``(pos, orn)`` wire frame, FK'd from one RecordTap frame — or None.

        The same function the clutch's leak bound reads (``measured_frame`` at construction,
        cross-checked against the anchor at startup), exposed for a consumer that needs the
        measurement PER PUBLISHED STEP: the hands' reach compensation (``hl_5pt.HandReach``). It is
        a readback, never a hold reference — :meth:`measured` still returns None and the idle wire
        is still never chased. A failed or absent readback is None, not a guess.
        """
        if self._measured_frame is None or rframe is None:
            return None
        try:
            return self._measured_frame(rframe)
        except Exception:  # noqa: BLE001 - a bad readback is "unknown", never a guess
            return None

    @property
    def hold_slots(self) -> tuple[int, ...]:
        """Wire slots whose clutch base is NOT leak-bounded by :meth:`sync` — none on this wire. A
        subclass whose slots can be driven by something other than the clutch
        (``hl_5pt.FivePointDirectWire.externally_driven``, the leg override) names them here."""
        return ()

    def sync(self, rframe, force_n: float | None = None) -> str:
        measured = None
        if self.clutch.post.carrying and self._measured_frame is not None:
            try:
                measured = self._measured_frame(rframe)
            except Exception:  # noqa: BLE001 - a bad readback FREEZES the carry, never guesses
                measured = None
        return self.clutch.sync(measured, force_n, hold_slots=self.hold_slots)

    def plan(self, completed: CompletedChunk, decoded: dict) -> dict:
        """The five keypoints for this chunk, per :data:`UPPER_SOURCES` / :data:`FOOT_SOURCES`.

        The completion is run and read either way — the feet exist nowhere else, and under
        ``foot_source="hold"`` its numbers still reach the status line and the gate report so a
        held foot can be compared against what the completion would have said.
        """
        pos, orn = self.readout.five_point(completed)
        n_up = len(_UPPER_ROLES)
        if self.upper_source == "policy":
            up_pos, up_orn = self.readout.upper_from_decoded(decoded)
            if up_pos.shape[0] != pos.shape[0]:
                raise ValueError(
                    f"decoded chunk has {up_pos.shape[0]} steps but the completion returned "
                    f"{pos.shape[0]}; the two must be the same chunk")
            pos[:, :3 * n_up] = up_pos
            orn[:, :4 * n_up] = up_orn
        if self.foot_source == "hold":
            # A CONSTANT for every step, not a ramp: the low level's absolute arm clutch
            # snapshots `held - wire` per point at arm, so a foot wire that never moves commands
            # exactly the stance the robot was already holding. Broadcast from the ready frame
            # rather than from step 0 of the chunk, because step 0 is the ready pose only when
            # nothing is carried (`--post-anchor off`) and this must not depend on that.
            pos[:, 3 * n_up:] = self._ready_pos[3 * n_up:]
            orn[:, 4 * n_up:] = self._ready_orn[4 * n_up:]
        pos, orn = self.clutch.apply(pos, orn)
        return {"pos": pos, "orn": orn, "n": int(pos.shape[0])}

    def frame(self, plan: dict, i: int) -> tuple:
        return (plan["pos"][i], plan["orn"][i])

    def publish(self, s0, frame: tuple, f_cmd, *, ref_joints=None) -> tuple[bool, str]:
        ok, reason = is_safe_keypoint_command(frame[0], frame[1],
                                              n_points=len(FIVE_POINT_BODIES))
        if not ok:
            return False, reason
        if ref_joints is None:
            s0.send_5pt(frame[0], frame[1], f_cmd=f_cmd)   # every frame but the walk skill's
        else:
            # The walk skill's reference legs ride the SAME frame as its feet (S0Driver.send_5pt
            # validates the block; the low level places it by name). Keyword-only and None by
            # default so every other caller is byte-identical.
            s0.send_5pt(frame[0], frame[1], f_cmd=f_cmd, ref_joints=ref_joints)
        return True, ""

    def commit(self, frame: tuple) -> None:
        # mode="off": nothing accumulates, every chunk is read out in the same frozen frame.
        self.clutch.commit(frame)

    def gate_report(self, frame: tuple) -> tuple[bool, str]:
        """``(would the low level arm?, a per-point readout)`` for THIS frame.

        The planner prints this for a staged chunk's step 0 before anything moves. It is a
        PREDICTION of the low level's gate, computed against the same reference, not the gate
        itself — the robot's is the one that decides.
        """
        gaps = keypoint_gaps_m(frame[0], self._ready_pos)
        gated, derived = gaps[:FIVE_POINT_GATED], gaps[FIVE_POINT_GATED:]
        ok = bool(np.all(gated <= KEYPOINT_GATE_TOL_M))
        parts = [f"{FIVE_POINT_LABELS[i]} {gaps[i] * 1e3:.0f} mm" for i in range(gaps.size)]
        note = (f"gate {'PASS' if ok else 'REFUSE'} on {FIVE_POINT_GATED}/{gaps.size} "
                f"(tol {KEYPOINT_GATE_TOL_M * 1e3:.0f} mm): " + ", ".join(parts))
        if derived.size and float(derived.max()) > FIVE_POINT_DERIVED_WARN_M:
            note += (f" ⚠️ a FOOT is {derived.max() * 1e3:.0f} mm out (> "
                     f"{FIVE_POINT_DERIVED_WARN_M * 1e3:.0f} mm): the completion is not "
                     "standing")
        if self.clutch.post.carrying and self.clutch.drift_m() > 0.0:
            note += (" — post-anchor is CARRYING, so this compares against the ready pose the "
                     "gate wants and not against where the wire now is; it only decides ARMING, "
                     "which happens before anything is carried")
        return ok, note

    def status(self, frame: tuple) -> str:
        gaps = keypoint_gaps_m(frame[0], self._ready_pos)
        carry = ("" if not self.clutch.post.carrying
                 else f" · carried drift {self.clutch.drift_m() * 1e3:.0f} mm")
        # The two sources are on the status line because they are the one thing about this wire
        # that a number cannot reveal: `policy` and `completion` agree to 0.84 mm, and `hold`
        # looks like a completion that happens to be standing still.
        src = f" · upper={self.upper_source} feet={self.foot_source}"
        return (f"5pt keypoints · worst gated |dp| vs ready "
                f"{float(gaps[:FIVE_POINT_GATED].max()) * 1e3:.0f} mm · feet "
                f"{float(gaps[FIVE_POINT_GATED:].max()) * 1e3:.0f} mm (not gated){carry}{src}")


@dataclass
class DenseStats:
    """Rolling per-chunk completion cost, for the planner's status line."""

    ms: list[float] = field(default_factory=list)
    chunks: int = 0
    errors: int = 0
    last_error: str = ""

    def note(self, chunk: CompletedChunk) -> None:
        self.chunks += 1
        self.ms.append(float(chunk.ms))
        del self.ms[:-32]

    def note_error(self, exc: BaseException) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"

    def summary(self) -> str:
        # The error TEXT is appended in both branches, including the no-timing one: the case
        # that matters most is the FIRST Predict failing, and there are no timings then, so a
        # branch that dropped the reason would leave the operator with a bare "1 error(s)".
        tail = f" · {self.errors} error(s): {self.last_error}" if self.errors else ""
        if not self.ms:
            return "no completion yet" + tail
        return (f"{self.chunks} chunk(s), {np.median(self.ms):.0f} ms p50 / "
                f"{max(self.ms):.0f} ms max" + tail)


__all__ = [
    "CHUNK_TIMEOUT_S",
    "CLUTCH_MODES",
    "DEFAULT_FOOT_FIX",
    "FIVE_POINT_BODIES",
    "FIVE_POINT_DERIVED_WARN_M",
    "FIVE_POINT_GATED",
    "FIVE_POINT_LABELS",
    "G1_DOF",
    "KEYPOINT_GATE_TOL_M",
    "JointClutch",
    "KeypointClutch",
    "KeypointReadout",
    "MAX_POST_EPS_RAD",
    "MC_OVERLAP",
    "MC_RATE_HZ",
    "MC_STRIDE",
    "MC_WINDOW",
    "POST_ANCHOR_MODES",
    "PostAnchorConfig",
    "WIRE_MODES",
    "ChunkCompleter",
    "CompletedChunk",
    "DenseStats",
    "DenseWire",
    "FivePointWire",
    "is_safe_dense_command",
    "is_safe_keypoint_command",
    "joint_velocity",
    "keypoint_gaps_m",
    "ready_pose_five_point",
    "ready_pose_joint_targets",
    "relativize_keypoints",
    "remap_by_name",
]
