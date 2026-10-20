"""``diffusion_umi_wholebody`` at deploy: the policy IS the whole body, so nothing completes it.

The serving core of the 29-joint preset, and the shortest module in this family precisely
because of what it does NOT contain. Compare the sibling paths::

    diffusion_umi_3tracker  -->  3 EE chunks  --> ChunkCompleter (80 ms/chunk, 2 windows,
                                                  a SECOND model in a SECOND interpreter)
                                              --> 29 joints --> DenseWire --> :8770
    diffusion_umi_wholebody -->  29 joints ------------------------------> DenseWire --> :8770

Same low level (``MODE=dense-unified-anneal``, W&B ``7vxmnwo6``), same wire, same clutch, same
force gate, same gripper bridges. The completion is gone from the deploy path because it already
ran ONCE, offline, over the training zarr
(``third_party/UMI_collection/motion_completion/mc_to_hl_wholebody.py``) — so this planner has no
``g1_mosyn`` conda env to find, no backend subprocess, no 30-60 s JIT warm-up, and no
``--mc-foot-ik`` decision to make. What is left is a policy that emits the numbers the wire
wants.

WHAT THAT COSTS, STATED UP FRONT
================================
The completion's job did not vanish; it moved into the WEIGHTS. Three consequences:

1. **The absolute joints live in the DATASET's posture space, not the robot's.**
   ``attrs.wb_cond_mode="home"`` re-based every take onto the G1's 29-joints-zero pose, so the
   predicted ``q`` is a posture around THAT anchor. The pose the low level actually holds after
   **A** is a 38 deg crouch (``ready_pose.resolve_deploy_anchor``), and measured on
   ``session_2026.08.30_1411_v3_dual_clean32_wb`` the two are **0.514 rad apart at the knee**
   (mean over 30 episode starts; hip pitch 0.35-0.39, shoulder roll 0.32). Publishing the
   predicted joints ABSOLUTELY would command that gap on the first executed step.

   :class:`~flore.s1_policy.serving.hl_dense.JointClutch` is what makes it a non-issue, and it is
   the SAME object the motion-completion planner uses for the SAME reason (there it is a
   null-space offset, here an anchor offset — both are "the absolute pose is not a claim about
   the robot, only its change is"). ``base + (q[i] - q[0])`` makes step 0 a bit-exact no-op, so
   the robot keeps its own crouch and follows the predicted joint MOTION. This is why
   ``--wb-clutch none`` is not merely a tuning knob: it publishes a 0.5 rad step into a 0.20 rad
   pre-arm gate.

2. **The PROPRIOCEPTION has the same gap, and the clutch cannot fix an input.** The state is
   ``[joints 29][L grip][R grip]`` in the model's own posture space, and the exporter's own check
   proves ``action[:, 0] == state`` to 0.000e+00 rad — the state IS step 0 of the chunk. So the
   honest live state is the last COMMANDED target in model space, which is exactly the
   ``obs_mode=last_target`` contract the whole repo already runs on (``:8770`` is one-way; there
   is no S0 state publisher). See :data:`STATE_SOURCES`: ``command`` (default) is
   in-distribution by construction; ``measured`` feeds RecordTap's encoders straight in and is
   0.514 rad OOD forever, which is why it warns rather than defaults.

   That leaves ONE frame with no answer: the first Predict, before anything has been commanded.
   See :func:`state_seed_from_zarr` — the seed is measured off the training data's own episode
   starts (worst per-joint spread across 30 episodes: **0.090 rad**, i.e. a well-defined pose),
   with the ``home`` anchor's literal zeros as the no-zarr fallback.

3. **The LEGS are on the wire.** A keypoint preset commands 3 points and lets the low level pick
   a posture; this one commands the hips, knees and ankles too. That is the preset, not a bug —
   but it means a predicted crouch is a real leg command, so :data:`LEG_MODES` offers
   ``--legs hold`` (publish the 17 waist+arm joints, hold the 12 leg joints at the clutch base)
   for a first bring-up, and the planner prints the measured per-chunk leg excursion either way.

WHAT IS DELIBERATELY REUSED RATHER THAN FORKED
==============================================
:class:`~flore.s1_policy.serving.hl_dense.DenseWire` — verbatim, by subclass. All eight
safety-relevant touch points (``idle``/``measured``/``sync``/``frame``/``publish``/``commit``/
``status`` + the clutch) are INHERITED; only :meth:`WholeBodyWire.plan` differs, because there is
no ``CompletedChunk`` to read joints out of. ``WB_JOINT_NAMES`` is byte-identical to
``live_robot.G1_29DOF_SDK_DOF_NAMES`` (asserted at import), so the inherited by-name remap is the
identity here and stays a remap rather than becoming an assumption.

:class:`~flore.s1_policy.serving.hl_online.LivePlanner` — by subclass, for the image / force /
gripper rings, the age bookkeeping and the causal force binning, all of which are layout-neutral.
Only the STATE assembly and the DECODE are overridden, and both had to be: ``snapshot_obs``
reads ``state_slots()[role]["wrt_start_rot6d"]``, a key the wholebody layout deliberately does not
emit, and ``decode_chunk`` un-relativizes ``slots.pos``, which here is a 0-width slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flore.s1_policy.data.layout import (
    JOINT_DIM,
    WB_JOINT_NAMES,
    WB_JOINT_TARGETS,
    ActionLayout,
)
from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M
from flore.s1_policy.serving.hl_dense import (
    G1_DOF,
    DenseWire,
    JointClutch,
    joint_velocity,
    MC_RATE_HZ,
)
from flore.s1_policy.serving.hl_online import (
    LivePlanner,
    batch_from_obs,
    gate_force_command,
    sample_chunk,
)
from flore.s1_policy.serving.hl_online import ObsSnapshot

# The whole reuse story rests on this: the policy emits joints in WB_JOINT_NAMES order and the
# wire remaps by name into the SDK order. They happen to be the same order today, which makes the
# inherited remap the identity — assert it so a future reordering of either list surfaces here,
# at import, rather than as a scrambled body on a real robot.
from flore.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES

if tuple(WB_JOINT_NAMES) != tuple(G1_29DOF_SDK_DOF_NAMES):  # pragma: no cover - import guard
    raise ImportError(
        "WB_JOINT_NAMES (the wholebody action's joint order) no longer matches "
        "live_robot.G1_29DOF_SDK_DOF_NAMES. Both are real orders and neither is wrong, but the "
        "wholebody wire assumes the policy's order is the SDK's; fix the wire's remap before "
        "serving, because a by-index publish of a permuted body is silent.")


#: Where the live 29-joint proprioception comes from.
#:
#: ``command`` — the last COMMANDED target, in the MODEL's own posture space. Default, and the
#: only one that is in-distribution: the exporter proves ``action[:, 0] == state`` exactly, so
#: "the state is step 0 of the last chunk" reproduces training's own relationship. It is also
#: the repo-wide ``obs_mode=last_target`` contract (``:8770`` is one-way; holosoma publishes no
#: state), so this is the wholebody spelling of an existing rule rather than a new one.
#:
#: ``measured`` — RecordTap's encoder readings, straight in. Available because "what does the
#: policy do when told where the robot really is" is a legitimate experiment, and refused-by-
#: default because the answer is measured: the robot's held ready pose is 0.514 rad (knee) from
#: the model's own posture anchor, so every frame of every chunk is fed an observation two
#: training spans outside the state quantiles.
STATE_SOURCES = ("command", "measured")

#: What reaches the 12 leg joints.
#:
#: ``predict`` (default) — all 29 published. The faithful wire: the preset's whole point is that
#: the policy answers with a posture, and the hips/knees/ankles are where a posture lives.
#:
#: ``hold`` — the 12 legs are pinned to the clutch base (i.e. to the joints the robot is already
#: holding, under the default ``--post-anchor off``), and only the 17 waist + arm joints follow
#: the chunk. For a first bring-up where the arms are the thing under test and a predicted
#: crouch is an unwanted variable. It needs the joint clutch to mean anything (with
#: ``--wb-clutch none`` there is no base to hold AT), and the planner refuses that pair by name.
LEG_MODES = ("predict", "hold")

#: Substrings that identify a LEG joint in :data:`WB_JOINT_NAMES`. Matched by substring rather
#: than listed as 12 literals so the set tracks the name list instead of drifting from it; the
#: count is asserted below, which is what makes the loose match safe.
_LEG_TOKENS = ("hip", "knee", "ankle")

#: Indices of the 12 leg joints within :data:`WB_JOINT_NAMES` / the action's joint block.
LEG_JOINT_INDICES: tuple[int, ...] = tuple(
    i for i, name in enumerate(WB_JOINT_NAMES) if any(tok in name for tok in _LEG_TOKENS)
)
if len(LEG_JOINT_INDICES) != 12:  # pragma: no cover - import guard
    raise ImportError(
        f"expected 12 leg joints in WB_JOINT_NAMES, matched {len(LEG_JOINT_INDICES)} "
        f"({[WB_JOINT_NAMES[i] for i in LEG_JOINT_INDICES]}). --legs hold would gate the wrong "
        "set of joints, so this is refused rather than guessed.")

#: The 29-joints-zero pose, i.e. the literal anchor ``attrs.wb_cond_mode="home"`` re-bases onto.
#: The state seed of last resort — see :func:`state_seed_from_zarr`.
HOME_ANCHOR_JOINTS: np.ndarray = np.zeros(JOINT_DIM, dtype=np.float64)


# ---------------------------------------------------------------------------
# the decoded chunk
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WholeBodyChunk:
    """One sampled ``(H, 43)`` action chunk, split into the things that leave this process.

    ``joints`` is un-transformed: the layout declares no pose dims, so there is no anchor to
    un-relativize against and none is applied. That is the load-bearing difference from
    :func:`~flore.s1_policy.serving.hl_online.decode_chunk`, and it is why an accidental
    transform here would be undetectable — a relative 29-vector is a perfectly plausible
    absolute one.

    What the values MEAN is the checkpoint's ``wb_joint_target``
    (:data:`~flore.s1_policy.data.layout.WB_JOINT_TARGETS`), and this class deliberately does not
    carry it, because the consumer that matters cannot tell the difference: ``JointClutch``
    publishes ``base + (q[i] - q[0])``, which is invariant to a constant offset, so an
    ``absolute`` and a ``delta`` chunk produce the SAME wire. The two places the distinction is
    real are both outside this dataclass — ``--wb-clutch none`` (refused for ``delta``, since
    near-zero radians is the 29-joints-zero pose) and the ``command`` proprio latch (which must
    integrate a delta row instead of replacing with it; see
    :meth:`WholeBodyPlanner.commit_chunk_row`).
    """

    joints: np.ndarray                       # (H, 29) rad, WB_JOINT_NAMES order, ABSOLUTE
    gripper_m: dict[str, np.ndarray]         # role -> (H,) metres
    wrench_pred: dict[str, np.ndarray]       # role -> (H, 6) physical N/N·m, UNGATED (for viz)
    wrench_cmd: dict[str, np.ndarray]        # role -> (H, 6) GATED command to publish

    @property
    def n(self) -> int:
        return int(self.joints.shape[0])


def decode_wholebody_chunk(
    prediction: np.ndarray,
    layout: ActionLayout,
    *,
    force_enabled: bool,
    gate_n: float,
) -> WholeBodyChunk:
    """``(H, 43)`` sampled chunk -> :class:`WholeBodyChunk`. Pure; no locks, no model.

    The wholebody twin of :func:`~flore.s1_policy.serving.hl_online.decode_chunk`, and a separate
    function rather than a branch inside it because the two share no step: that one's whole body
    is ``delta_ee_to_absolute`` over ``slots.pos``/``slots.rot6d``, which here are 0-width
    slices. Only the gripper and force blocks are common, and those are three lines.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != layout.action_dim:
        raise ValueError(
            f"prediction must be (H, {layout.action_dim}) for layout {layout.mode!r}, "
            f"got {pred.shape}")
    if layout.joints is None:
        raise ValueError(
            f"layout {layout.mode!r} declares no joint block, so it cannot be decoded as a "
            "whole-body chunk. Serve a keypoint layout with scripts/hl_serving.py, or a "
            "3-tracker one through motion completion with scripts/hl_serving_mc.py.")
    joints = pred[:, layout.joints].copy()
    if joints.shape[1] != G1_DOF:
        raise ValueError(f"joint block is {joints.shape[1]} wide, want {G1_DOF}")

    gripper_m: dict[str, np.ndarray] = {}
    wrench_pred: dict[str, np.ndarray] = {}
    wrench_cmd: dict[str, np.ndarray] = {}
    for role in layout.roles:
        slots = layout.slots(role)
        if slots.gripper is not None and role in layout.gripper_roles:
            gripper_m[role] = pred[:, slots.gripper] * GRIPPER_WIDTH_REF_M
        if slots.force is not None and role in layout.force_roles:
            wrench_pred[role] = pred[:, slots.force].copy()
            wrench_cmd[role] = gate_force_command(
                wrench_pred[role], force_enabled=force_enabled, gate_n=gate_n)
    return WholeBodyChunk(joints=joints, gripper_m=gripper_m,
                          wrench_pred=wrench_pred, wrench_cmd=wrench_cmd)


def leg_excursion_rad(joints: np.ndarray) -> tuple[float, str]:
    """``(worst |q - q[0]| over the 12 LEG joints, its name)`` across a chunk.

    Printed every Predict. The legs hold the robot up and are the one channel a keypoint preset
    never commanded, so "how much leg did this chunk ask for" is the number that decides whether
    ``--legs hold`` is wanted — and it is a property of the chunk, not of a setting, so it is
    reported whichever mode is armed.
    """
    q = np.asarray(joints, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != G1_DOF or q.shape[0] == 0:
        return 0.0, ""
    legs = np.asarray(LEG_JOINT_INDICES, dtype=int)
    delta = np.abs(q[:, legs] - q[0, legs])
    flat = int(np.argmax(delta))
    return float(delta.flat[flat]), WB_JOINT_NAMES[legs[flat % legs.size]]


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------
class WholeBodyWire(DenseWire):
    """The ``dense`` wire, fed by the POLICY instead of by a completion.

    Subclasses :class:`~flore.s1_policy.serving.hl_dense.DenseWire` rather than copying it, so
    every safety-relevant method — the idle frame, the un-armed measured re-sync, the clutch
    ``sync``, ``frame``, the ``is_safe_dense_command`` gate inside ``publish``, ``commit`` and
    the status line — is the SAME code the motion-completion planner is reviewed against. One
    method is overridden (``plan``, because there is no ``CompletedChunk``) and one behaviour is
    added (``legs``).
    """

    def __init__(self, clutch: JointClutch, ready_joints: np.ndarray, *,
                 legs: str = "predict", joint_target: str = "absolute") -> None:
        if str(legs) not in LEG_MODES:
            raise ValueError(f"legs must be one of {LEG_MODES}, got {legs!r}")
        if str(joint_target) not in WB_JOINT_TARGETS:
            raise ValueError(
                f"joint_target must be one of {WB_JOINT_TARGETS}, got {joint_target!r}")
        if str(joint_target) == "delta" and clutch.mode != "joint":
            # THE one combination where the two targets are not interchangeable on the wire.
            # `clutch.apply` is `q - q[0] + base` for mode="joint", which is invariant to a
            # constant offset — so a delta chunk and an absolute one publish the SAME values and
            # nothing here needs to know which it got. mode="none" is the identity instead, and
            # publishing a delta chunk identically commands ~0 rad on all 29 joints: the
            # 29-joints-zero pose, i.e. a standing robot told to straighten every leg and drop
            # both arms, at the instant it arms. Refused by name rather than gated at runtime,
            # because the wire's own width/finiteness checks would all pass.
            raise ValueError(
                "--wb-clutch none cannot publish a wb_joint_target='delta' checkpoint: that "
                "chunk is joint MOTION (row 0 is ~0 rad), so publishing it absolutely commands "
                "the 29-joints-zero pose. Use --wb-clutch joint (the default), which adds the "
                "motion to the robot's own base, or serve an 'absolute' checkpoint.")
        if str(legs) == "hold" and clutch.mode != "joint":
            raise ValueError(
                "--legs hold needs the joint clutch: it pins the legs to the CLUTCH BASE (the "
                "joints the robot is holding), and with clutch mode 'none' there is no base to "
                "hold at — the legs would be pinned to the model's own absolute step-0 posture, "
                "which is 0.51 rad (knee) from the robot's ready crouch.")
        # Both name lists are WB_JOINT_NAMES: the policy's order IS the SDK's (asserted at
        # import), so the inherited by-name remap is the identity. Passing it explicitly keeps
        # `measured()`/`publish()` a by-NAME operation rather than an index-aligned one.
        super().__init__(WB_JOINT_NAMES, G1_29DOF_SDK_DOF_NAMES, clutch, ready_joints)
        self.legs = str(legs)
        #: Carried for the status line and the refusal above ONLY. `plan()` does not branch on it,
        #: which is the property worth stating: the clutch makes the two targets the same wire.
        self.joint_target = str(joint_target)

    @property
    def gate_note(self) -> str:  # type: ignore[override]
        base = "joint-space (HOLOSOMA_VIVE_ARM_TOL_RAD, 0.20 rad)"
        return base if self.legs == "predict" else base + " · LEGS HELD at the clutch base"

    def plan(self, chunk: WholeBodyChunk, decoded: dict | None = None) -> dict:
        """``WholeBodyChunk`` -> the per-step frames to publish.

        ``decoded`` is accepted and ignored to keep the signature interchangeable with
        ``DenseWire.plan(completed, decoded)``, so the planner's drip loop does not have to know
        which family produced the joints.
        """
        del decoded
        published = self.clutch.apply(np.asarray(chunk.joints, dtype=np.float64))
        if self.legs == "hold":
            # published[0] IS the clutch base under mode="joint" (apply() lands step 0 exactly
            # there), so this pins the legs to the pose the robot is holding for the whole chunk
            # — no leg command at all, rather than a smaller one.
            legs = np.asarray(LEG_JOINT_INDICES, dtype=int)
            published = published.copy()
            published[:, legs] = published[0, legs]
        return {"published": published,
                "qvel": joint_velocity(published, 1.0 / MC_RATE_HZ),
                "n": int(published.shape[0])}

    def status(self, frame: tuple) -> str:
        return super().status(frame) + ("" if self.legs == "predict" else " · legs HELD")


# ---------------------------------------------------------------------------
# the state seed (the ONE frame with nothing commanded yet)
# ---------------------------------------------------------------------------
def state_seed_from_zarr(zarr_path: str | None) -> tuple[np.ndarray, str]:
    """``(q29, note)``: the 29-joint proprio seed for the FIRST Predict, from training data.

    WHY THIS IS MEASURED AND NOT ASSUMED. Every later Predict sees the last commanded target, so
    the state is in-distribution by construction (:data:`STATE_SOURCES`). The first one has
    nothing commanded, and the robot's own encoders are the wrong answer — the pose it holds
    after **A** is 0.514 rad (knee) from the model's ``cond_mode="home"`` posture anchor. The
    right answer is where the model's own training episodes STARTED, which the dataset carries:
    ``wb_state_joints`` at each episode's first frame.

    Measured on ``session_2026.08.30_1411_v3_dual_clean32_wb`` (30 completed episodes), the worst
    per-joint standard deviation across those starts is **0.090 rad** — so "the pose a take
    begins in" is a real, tight pose and its mean is a legitimate seed, not an average of
    unrelated things.

    Falls back to :data:`HOME_ANCHOR_JOINTS` (the literal 29 zeros) with a note when the zarr is
    absent or carries no ``wb_*`` arrays, which is the honest degradation: the rig may not hold a
    600 MB training file. Never raises — a seed problem must not stop a bring-up, it must be
    printed.
    """
    if not zarr_path:
        return HOME_ANCHOR_JOINTS.copy(), (
            "the checkpoint declares no reachable training zarr; seeding proprio at the "
            "cond_mode=home anchor (29 zeros). Measured against that dataset's own episode "
            "starts this is up to ~0.32 rad off (shoulder roll), which biases the FIRST chunk "
            "only — every later chunk sees the last commanded target.")
    try:
        import zarr  # noqa: PLC0415 - only needed on this path

        with zarr.ZipStore(str(zarr_path), mode="r") as store:
            root = zarr.group(store=store)
            attrs = dict(root.attrs)
            if not attrs.get("wb_windows"):
                return HOME_ANCHOR_JOINTS.copy(), (
                    f"{zarr_path} carries no wb_* arrays (attrs.wb_windows absent), so it cannot "
                    "supply an episode-start pose; seeding at the cond_mode=home anchor "
                    "(29 zeros).")
            joints = np.asarray(root["data"]["wb_state_joints"][:], dtype=np.float64)
            valid = np.asarray(root["data"]["wb_state_valid"][:], dtype=np.float64).reshape(
                joints.shape[0], -1)[:, 0] > 0.5
            ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]]).astype(int)
        starts = starts[(starts >= 0) & (starts < joints.shape[0])]
        rows = joints[starts][valid[starts]]
        if rows.shape[0] == 0 or not np.all(np.isfinite(rows)):
            return HOME_ANCHOR_JOINTS.copy(), (
                f"{zarr_path} has no episode start with a valid wb_state_joints row; seeding at "
                "the cond_mode=home anchor (29 zeros).")
        seed = rows.mean(axis=0)
        spread = float(rows.std(axis=0).max()) if rows.shape[0] > 1 else 0.0
        return seed, (
            f"proprio seed = mean of {rows.shape[0]} episode-start wb_state_joints rows in "
            f"{zarr_path.rsplit('/', 1)[-1]} (worst per-joint spread {spread:.3f} rad, "
            f"cond_mode={attrs.get('wb_cond_mode')!r})")
    except Exception as exc:  # noqa: BLE001 - a seed problem is reported, never fatal
        return HOME_ANCHOR_JOINTS.copy(), (
            f"could not read an episode-start pose from {zarr_path} ({type(exc).__name__}: "
            f"{exc}); seeding at the cond_mode=home anchor (29 zeros).")


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------
class WholeBodyPlanner(LivePlanner):
    """:class:`LivePlanner` with a JOINT proprio ring instead of a per-role pose anchor.

    Inherited unchanged, because none of it is layout-specific: the per-camera image rings and
    their camera-major stacking, the age-keyed causal force binning, the gripper latch and its
    ``state_valid`` downgrade, ``obs_ages``. Overridden:

    * :meth:`snapshot_obs` — the wholebody state is ``[joints 29][L grip][R grip]``; the base
      class reads ``state_slots()[role]["wrt_start_rot6d"]``, a key this layout deliberately does
      not emit (see ``ActionLayout.state_slots``), so calling it would ``KeyError`` on the name.
    * :meth:`infer_chunk` — decodes through :func:`decode_wholebody_chunk`; the base class
      un-relativizes ``slots.pos``, which is 0-width here.

    The inherited anchor bookkeeping (``seed_anchor`` / ``commit_role`` /
    ``push_proprio_from_anchor``) is left in place but unused: a wholebody chunk is absolute, so
    there is no anchor to seed or advance. It is not overridden to raise, because the base
    class's own ``__init__`` builds those arrays and a caller reading ``snapshot_anchor()`` for a
    log line should get a harmless identity rather than an exception.
    """

    def __init__(self, loaded, obs_hz: float = 30.0, *,
                 state_source: str = "command", seed_joints: np.ndarray | None = None,
                 joint_target: str = "absolute") -> None:
        if str(state_source) not in STATE_SOURCES:
            raise ValueError(f"state_source must be one of {STATE_SOURCES}, got {state_source!r}")
        if str(joint_target) not in WB_JOINT_TARGETS:
            raise ValueError(
                f"joint_target must be one of {WB_JOINT_TARGETS}, got {joint_target!r}")
        if not loaded.layout.is_wholebody:
            raise ValueError(
                f"WholeBodyPlanner needs a wholebody layout, got {loaded.layout.mode!r}. A "
                "keypoint checkpoint has no joint block to predict; serve it with "
                "scripts/hl_serving.py (3 points) or scripts/hl_serving_mc.py (3 trackers -> "
                "motion completion).")
        super().__init__(loaded, obs_hz=obs_hz)
        self.state_source = str(state_source)
        #: What the checkpoint's 29 joint columns mean. Read from the checkpoint, never a flag:
        #: it is a property of the trained weights, and the two are indistinguishable by shape.
        self.joint_target = str(joint_target)
        self._joint_slot = loaded.layout.joint_state_slot
        seed = (HOME_ANCHOR_JOINTS if seed_joints is None
                else np.asarray(seed_joints, dtype=np.float64).reshape(-1))
        if seed.size != JOINT_DIM:
            raise ValueError(f"seed_joints must be {JOINT_DIM} angles, got {seed.size}")
        #: The LATCH: what "the current joints" are until something newer lands. Advanced by
        #: :meth:`set_joint_target` (command mode) or :meth:`push_measured_joints` (measured).
        self._q_latch = seed.astype(np.float64).copy()
        self._q_seeded = seed.astype(np.float64).copy()
        #: The ring the observation is built from, oldest-first, length To.
        self._buf_joints: list[np.ndarray] = [self._q_latch.copy() for _ in range(self._To)]
        #: Arrival stamp of the latch, for the same reason every other ring has one: a frozen
        #: readback degrades by REPEATING, so age is the only way to see it. Seeded ``None`` =
        #: never written, which the readout reports as such.
        self._t_joints: float | None = None

    # --- the joint latch -------------------------------------------------
    def set_joint_target(self, q29: np.ndarray) -> None:
        """Latch the last COMMANDED joints, in the MODEL's posture space (``command`` mode).

        Called with the last EXECUTED step of the chunk — the model's own output, NOT the
        clutched wire value. That distinction is the whole point: the wire is in the robot's
        space (``base + (q[i] - q[0])``) and the observation must be in the model's, or the
        proprio drifts by the anchor gap one chunk at a time.
        """
        q = np.asarray(q29, dtype=np.float64).reshape(-1)
        if q.size != JOINT_DIM or not np.all(np.isfinite(q)):
            return
        import time  # noqa: PLC0415 - matches the base class's per-push stamping

        with self._lock:
            self._q_latch = q.copy()
            self._t_joints = time.monotonic()

    def commit_chunk_row(self, row29: np.ndarray) -> None:
        """Advance the proprio latch by one row of the chunk that just executed.

        The ONE place ``wb_joint_target`` reaches the observation, and it exists as its own method
        rather than a branch inside :meth:`set_joint_target` because that one is also how a
        MEASURED readback and a Reset land — both absolute, both of which an integrator would
        corrupt:

        * ``absolute`` — the row IS the posture the model asked for; latch it. Unchanged from the
          shipped behaviour, byte for byte.
        * ``delta`` — the row is ``q[k] - q[0]``, motion relative to the observation this chunk was
          conditioned on. The posture it implies is ``latch + row``, so this integrates. It stays
          exactly "the last commanded target in the model's space", which is what
          :data:`STATE_SOURCES` ``command`` means — the arithmetic that reconstructs it differs
          because the target does.

        ⚠️ ``delta`` therefore ACCUMULATES prediction error across chunks where ``absolute`` reads
        the model's own claim afresh each time. That is the honest cost of the parameterization and
        the reason this method is where it happens: with ``--state-source measured`` the latch is
        the encoders and there is nothing to accumulate, so the trade is switchable at launch.
        """
        row = np.asarray(row29, dtype=np.float64).reshape(-1)
        if row.size != JOINT_DIM or not np.all(np.isfinite(row)):
            return
        if self.joint_target == "absolute":
            self.set_joint_target(row)
            return
        with self._lock:
            base = self._q_latch.copy()
        self.set_joint_target(base + row)

    def push_measured_joints(self, q29: np.ndarray) -> None:
        """Latch the robot's MEASURED joints (``measured`` mode). Ignored in ``command`` mode.

        Ignored rather than refused so the ingest loop can call it unconditionally: whether the
        readback is the observation is a launch-time declaration, and a loop that has to know
        would be a second place the decision lives.
        """
        if self.state_source != "measured":
            return
        self.set_joint_target(q29)

    def push_proprio_joints(self) -> None:
        """Append one observation frame: the joint latch AND the gripper fractions.

        The wholebody twin of ``push_proprio_from_anchor``, and it must advance BOTH rings for the
        same reason that one does — the state at frame t is the last commanded target, held until
        a newer one arrives, so every channel of it is filled at the obs rate from a latched value.

        ⚠️ Advancing only the joint ring was a real bug, and a silent one: ``_buf_grip`` then
        stayed at its ``__init__`` zeros forever, so both gripper columns read **0.0 m** — the
        CLOSED end of the jaw's travel, a perfectly legal width — for the whole session, while
        ``push_gripper`` kept updating a latch nothing ever read. Nothing downstream could notice,
        because a shut jaw is not an error. Caught by
        ``scripts/smoke_test_hl_wholebody.py``'s proprio round-trip.
        """
        with self._lock:
            self._buf_joints.append(self._q_latch.copy())
            self._buf_grip.append(self._grip_frac_vec())
            for buf in (self._buf_joints, self._buf_grip):
                while len(buf) > self._To:
                    buf.pop(0)

    def reset_joints(self, q29: np.ndarray | None = None) -> None:
        """Re-seed the latch and refill the ring (a fresh episode / Reset)."""
        q = (self._q_seeded if q29 is None
             else np.asarray(q29, dtype=np.float64).reshape(-1))
        if q.size != JOINT_DIM or not np.all(np.isfinite(q)):
            return
        with self._lock:
            self._q_latch = q.astype(np.float64).copy()
            self._buf_joints = [self._q_latch.copy() for _ in range(self._To)]
            self._t_joints = None

    def joint_state(self) -> tuple[np.ndarray, float | None]:
        """``(latched joints, age_s or None)`` for the page readout."""
        import time  # noqa: PLC0415

        with self._lock:
            q, t = self._q_latch.copy(), self._t_joints
        return q, None if t is None else time.monotonic() - t

    # --- observation -----------------------------------------------------
    def snapshot_obs(self) -> tuple[ObsSnapshot, np.ndarray, np.ndarray]:
        """The wholebody observation: ``[joints 29][L grip][R grip]`` + images + force window.

        Returns the base class's 3-tuple so :func:`batch_from_obs` and the deploy recorder keep
        working unchanged; the two anchor arrays come back as the (unused) identity anchor,
        because an absolute chunk has nothing to decode against and fabricating a pose here would
        invite a caller to use it.
        """
        import time  # noqa: PLC0415

        t_snapshot = time.monotonic()
        with self._lock:
            joint_hist = np.stack(self._buf_joints)          # (To, 29)
            grip_hist = np.stack(self._buf_grip)             # (To, n_roles)
            imgs = [list(ring) for ring in self._buf_img]
            raw_force = list(self._buf_force)
            t_grip = dict(self._t_grip)
            t_joints = self._t_joints
            anchor_pos = self._anchor_pos.copy()
            anchor_rot6d = self._anchor_rot6d.copy()

        force_win, fh_valid = self._bin_force_history(raw_force)

        To = self._To
        state = np.zeros((To, self.layout.state_dim), dtype=np.float64)
        state_valid = np.ones((To, self.layout.state_dim), dtype=bool)
        state[:, self._joint_slot] = joint_hist
        # A NEVER-WRITTEN latch is the seed, which is a real pose but not a MEASUREMENT of this
        # robot — flagged so the model's own no-data embedding handles it rather than being told
        # the seed is where the body is. A STALE latch is not flagged: in `command` mode the last
        # commanded target stays true for as long as nothing newer is commanded (that is what
        # `last_target` means), so age there is not staleness. In `measured` mode a dead RecordTap
        # does freeze it, which the page reports and the low-level watchdog acts on.
        if t_joints is None:
            state_valid[:, self._joint_slot] = False
        for ri, role in enumerate(self.roles):
            entry = self.layout.state_slots()[role]
            if "gripper" not in entry:
                continue
            gcol = entry["gripper"]
            state[:, gcol] = grip_hist[:, ri]
            t_g = t_grip.get(role)
            if t_g is None or (t_snapshot - t_g) > self.grip_max_age_s:
                state_valid[:, gcol] = False

        obs = ObsSnapshot(
            images=self._image_stack(imgs),
            state=state,
            state_valid=state_valid,
            force_history=force_win.astype(np.float64),
            force_history_valid=bool(fh_valid),
        )
        return obs, anchor_pos, anchor_rot6d

    # --- one inference ---------------------------------------------------
    def infer_chunk(self, *, seed: int, force_enabled: bool, gate_n: float,
                    num_steps: int | None = None,
                    capture: dict | None = None) -> WholeBodyChunk:  # type: ignore[override]
        """Sample one chunk and split it. Returns a :class:`WholeBodyChunk`, NOT a
        ``ChunkResult`` — the two carry different things (29 joints vs per-role poses) and a
        common supertype would only let a caller reach for a field the other cannot fill."""
        if self.m.model is None:
            raise RuntimeError(
                "this WholeBodyPlanner has no policy — launch with --checkpoint. (A "
                "replay-only wholebody planner is not implemented: a recorded dense take is 29 "
                "joints, not a wire pose, so it needs its own reader.)")
        obs, anchor_pos, anchor_rot6d = self.snapshot_obs()
        batch = batch_from_obs(self.m, obs)
        prediction = sample_chunk(self.m, batch, seed=seed, num_steps=num_steps)
        if capture is not None:
            capture.update(
                obs=obs, anchor_pos=anchor_pos, anchor_rot6d=anchor_rot6d,
                prediction=prediction, seed=int(seed),
                num_steps=(self.m.num_steps if num_steps is None else int(num_steps)),
            )
        return decode_wholebody_chunk(prediction, self.layout,
                                      force_enabled=force_enabled, gate_n=gate_n)


__all__ = [
    "HOME_ANCHOR_JOINTS",
    "LEG_JOINT_INDICES",
    "LEG_MODES",
    "STATE_SOURCES",
    "WholeBodyChunk",
    "WholeBodyPlanner",
    "WholeBodyWire",
    "decode_wholebody_chunk",
    "leg_excursion_rad",
    "state_seed_from_zarr",
]
