"""``diffusion_umi_5pt`` at deploy: five predicted keypoints straight onto the ``5pt`` low level.

The third corner of the whole-body deploy family, and the one with the shortest chain::

    diffusion_umi_3tracker  -> 3 EE poses  -> ChunkCompleter -> 5 keypoints -> FivePointWire
    diffusion_umi_wholebody -> 29 joints   ---------------------------------> DenseWire
    diffusion_umi_5pt       -> 5 keypoints ---------------------------------> FivePointWire

Same low level as the first (``MODE=5pt-unified-anneal``, W&B ``tqnoooo6``), same wire, same gate,
same force/gripper bridges. The completion is gone because the policy predicts every slot the wire
carries — no ``g1_mosyn`` conda env, no backend subprocess, no 30-60 s JIT, no ``--mc-foot-ik``
decision. What is left is a decode and a publish.

THE OPERATOR'S OWN SEQUENCE, AND WHERE EACH STEP LIVES
======================================================
    press A          the low level ramps to its NEUTRAL pose and holds it
                     == ``ready_pose.resolve_deploy_anchor``: home_anchor default_qpos with the
                        14 arm joints zeroed, FK'd through this rig's URDF, floor z = 0.
    Predict          the chunk is DECODED against exactly that pose, so every step is
                     "where the five points go, starting from where the robot is".
    on that basis    :class:`~fuji.s1_policy.serving.hl_dense.KeypointClutch` publishes
                     ``p_base + (p[i] - p[0])`` with ``base`` = the ready frame, so step 0 is a
                     bit-exact no-op and every later step is the chunk's own displacement.
    -> pelvis frame  :class:`~fuji.s1_policy.serving.hl_dense.KeypointReadout` expresses all five
                     in the ANCHOR's pelvis frame — 15 positions + 20 xyzw quats, which IS
                     ``holosoma_wbt._compute_vr_fk``'s ``vr_pos_local`` / ``vr_orn_local``.

⚠️ WHY THE CLUTCH IS ON BY DEFAULT HERE AND OFF ON THE COMPLETION PATH. Measured on
``diffusion_umi_5pt_0902_merged3_h200_umi_ddp4/best_overall``, 6 samples spread across the
dataset, chunks decoded against the ready pose and read out in its pelvis frame — the gap between
the wire's step 0 and the low level's own pre-arm gate reference:

    post-anchor        LEFT hand   RIGHT hand   torso     LEFT foot   RIGHT foot
    off (absolute)      59.4 mm     62.8 mm     9.3 mm     17.6 mm      9.2 mm
    command, carry 3     0.0         0.0        0.0        17.6         9.2
    command, carry 5     0.0         0.0        0.0         0.0         0.0
    the gate            <= 70 mm    <= 70 mm   <= 70 mm    warn 300    warn 300

The DATA's own row 0 is its anchor to **0.000 mm / 0.000 deg** for all five roles (the converter's
property, re-measured), so those 59-63 mm are the MODEL's step-0 error, not a frame mistake. That
is inside the 70 mm tolerance with 7 mm to spare — i.e. arming would work today and stop working
on a slightly worse sample, silently, as a refusal to arm. On the completion path the same number
is 0.7-3.1 mm because the completion is *told* to hit the decoded hand pose, which is why ``off``
is a sane default there and not here.

⚠️ AND WHY ALL FIVE POINTS ARE CARRIED, unlike the completion path. ``KeypointClutch``'s default
``n_carried=3`` exists because a COMPLETION's feet sit 69-91 mm from the ready crouch by
construction (it stands straighter), so re-basing them shifted the foot trajectory 74.7 mm — a leg
command nobody asked for. Here the feet are decoded against the ready pose's OWN ankles, so they
start 9-18 mm away, and carrying them costs exactly that as a rigid offset while buying them the
same measured leak bound the hands get. Both are available (:data:`CARRY_MODES`); ``all`` is the
default because an UNBOUNDED absolute foot command into a whole-body balance controller is the
worse of the two risks.

WHAT IS REUSED RATHER THAN FORKED
=================================
:class:`~fuji.s1_policy.serving.hl_dense.FivePointWire` — by subclass. All the safety-relevant
methods (``idle`` / ``measured`` / ``sync`` / ``frame`` / the ``is_safe_keypoint_command`` gate
inside ``publish`` / ``commit`` / ``gate_report``) are the SAME code the motion-completion planner
is reviewed against; only :meth:`FivePointDirectWire.plan` differs, because there is no
``CompletedChunk`` to read keypoints out of.

:class:`~fuji.s1_policy.serving.hl_online.LivePlanner` — by subclass, for the per-camera image
rings, the age bookkeeping, the causal force binning and the gripper latch, none of which is
layout-specific. Overridden: :meth:`FivePointPlanner.snapshot_obs`, because this layout's state is
a pelvis-frame GEOMETRY and the base class reads ``state_slots()[role]["wrt_start_rot6d"]``, a key
it deliberately does not emit.

⚠️ THE OBSERVATION IS THE ANCHOR, AND THAT IS EXACT RATHER THAN CLOSE. ``obs_mode=last_target``
means the state is the last commanded target; the running anchor IS that (``commit_role`` advances
it to the last executed step), and it is already expressed in the anchor world frame whose pelvis
rotation is IDENTITY and whose origin differs from the pelvis only by ``+pelvis_z`` on z. Both
blocks of this state are invariant to that: ``rel_xy3`` is three pairwise DIFFERENCES (translation
cancels) and ``rpy`` needs only the frame's rotation. So the geometry is computed from the anchor
poses directly, with no pelvis correction, and it equals what
:func:`~fuji.s1_policy.data.wb_geom.geom_from_bodies` was fed in training frame for frame — the
same function, on the same three bodies, in the same frame.

A KEYPOINT CHECKPOINT THAT PREDICTS FEWER THAN FIVE POINTS — THE HELD SLOTS
==========================================================================
``diffusion_umi`` (layout ``left_only``: ONE hand, 16-D action, a 7-D ``wrt_start`` state, one
camera, one Bota, one jaw) drives the SAME wire. The 5pt low level wants five poses on every
frame, so the four it does not predict are **HELD**: filled with the READY POSE's own pose for
that slot (``resolve_deploy_anchor`` over all of :data:`WIRE_ROLES`) on every step of every
chunk — which is exactly what ``hl_serving.py``'s ``WireMapper`` did for RIGHT + HEAD on the
3-point wire ("held at the ready-pose home for the whole run; their wire slots never move").
Three things follow, none of them a new mechanism:

* a held row is constant, so under ``--post-anchor command`` the clutch publishes
  ``p_base + 0`` — the base, i.e. the low level's own pre-arm gate reference on the first chunk
  and the last published frame after — and under ``off`` it publishes the ready readout. Either
  way the held points land ON the gate reference to 0 mm by construction, and a Predict never
  moves them;
* the observation is the base class's own ``wrt_start_rot6d`` + jaw state
  (:class:`SubsetKeypointPlanner`), because that is what such a checkpoint was trained on; the
  geometric planner is not involved;
* the wire's ``f_cmd`` is ALWAYS ``(6,)`` = ``[L, R]`` with a zero for a hand the policy does
  not predict (:func:`force_command`) — the low level accepts a ``(3,)`` LEFT-only vector too,
  but a partial publish would leave the other hand's command at whatever it last had.

The role SET decides everything and the two families are told apart by :func:`check_layout`,
never by a flag: ``fivepoint_geom_*`` (five roles, geometric state) and any ``_V2_MODES``
keypoint layout whose roles are a subset of the five (``left_only`` first of all, and
``both_hands`` / ``both_hands_head[_dual]`` by the same rule). What is refused by name is a
whole-body layout (29 joints, no keypoints) and a role the wire has no slot for.

⚠️ The LEFT role's decode is the same one the five-point path uses for its LEFT role: the chunk is
un-relativized against the ready pose's fingertip-TCP pose FK'd through the yam URDF, with NO
URDF<->model seam applied (the CURRENT converter convention stores the wrist-oriented fingertip
TCP; ``hl_serving.py`` derives the same identity seam from ``T_tracker_tcp`` and refuses a
pre-2026-08-06 dataset). A checkpoint trained on a pre-2026-08-06 zarr is therefore NOT servable
here either — re-convert, as that planner says.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from fuji.s1_policy.data import wb_geom
from fuji.s1_policy.data.layout import ActionLayout
from fuji.s1_policy.data.transforms import rot6d_relative
from fuji.s1_policy.inference.ready_pose import ROLE_BODY
from fuji.s1_policy.serving.hl_dense import (
    FIVE_POINT_BODIES,
    FIVE_POINT_GATED,
    KeypointClutch,
    KeypointReadout,
    PostAnchorConfig,
    FivePointWire,
    _clamp_rot_batch,
    _rotmats_from_quat_xyzw,
)
from fuji.s1_policy.serving.hl_online import (
    ChunkResult,
    LivePlanner,
    ObsSnapshot,
    batch_from_obs,
    decode_chunk,
    sample_chunk,
)
from fuji.s1_policy.serving.hl_reach import ReachCompensator, ReachConfig
from fuji.utils.force_frame import cmd_from_meas_force
from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

#: How many of the five keypoints the clutch CARRIES (re-bases so step 0 lands on the base).
#:
#: ``all`` (default) — all five. The feet here are decoded against the ready pose's own ankles, so
#: carrying them is a 9-18 mm rigid offset (measured) and it gives them the same leak bound against
#: the robot's measured ankles that the hands get. An uncarried foot is an ABSOLUTE command that
#: drifts with the decode anchor, into a low level that is a whole-body balance controller.
#:
#: ``upper`` — the three GATED points only; the feet pass through absolutely. This is
#: ``KeypointClutch``'s own default and the motion-completion path's behaviour, kept available
#: because it is the reviewed one — but its REASON does not hold here (that reason is a
#: completion standing 69-91 mm straighter than the ready crouch; see the module docstring).
CARRY_MODES = ("all", "upper")

#: The layout roles that fill the wire's five slots, in slot order. Derived from
#: :data:`FIVE_POINT_BODIES` through the same ``ROLE_BODY`` table the decode anchor uses, so the
#: order cannot drift from the vector the low level parses.
WIRE_ROLES: tuple[str, ...] = tuple(
    next(role for role, body in ROLE_BODY.items() if body == b) for b in FIVE_POINT_BODIES)

#: The three bodies :func:`~fuji.s1_policy.data.wb_geom.geom_from_bodies` reads, and the layout
#: role that carries each. Only the TRACKED three appear: neither ``rel_xy3`` nor ``rpy`` reads a
#: foot, which is why a five-point observation is buildable at deploy at all.
GEOM_BODY_ROLE: dict[str, str] = {ROLE_BODY["left"]: "left",
                                  ROLE_BODY["right"]: "right",
                                  ROLE_BODY["head"]: "head"}

#: The two HAND slots of the wire's ``f_cmd``, in the order the low level reads a ``(6,)`` vector
#: (``ViveLiveSource``: ``(3,)`` = LEFT or ``(6,)`` = ``[L, R]``). This wire ALWAYS publishes the
#: ``(6,)`` form, a zero for any hand the policy does not predict — see :func:`force_command`.
FORCE_HANDS: tuple[str, ...] = ("left", "right")

#: The two answers :func:`check_layout` can give for a servable layout.
LAYOUT_FAMILIES: tuple[str, ...] = ("fivepoint", "keypoint_subset")

#: The low level's per-tick position slew cap (``HOLOSOMA_VIVE_MAX_POS_STEP``, m per 50 Hz TICK),
#: as a PRE-FLIGHT ESTIMATE ONLY. :class:`SlewMonitor` supersedes it from RecordTap the moment the
#: low level is armed, because a constant here can only ever be a guess about another process.
#:
#: ⚠️ 0.005, NOT the submodule's 0.02 code default. ``3_run_policy_vive_force.sh`` — the launcher
#: every 5pt runbook T4 invokes — sets ``HOLOSOMA_VIVE_MAX_POS_STEP=0.005`` as an INLINE prefix on
#: its single ``run_policy`` call, so it applies to every MODE and an operator's exported env var
#: cannot override it. ``wbt_wrappers/README.md`` documents 0.005/0.02 as the deliberately
#: conservative "first hardware trial" pair, to be relaxed once position tracking looks clean —
#: i.e. this number is expected to CHANGE, which is exactly why nothing safety-relevant may depend
#: on the constant. A previous revision quoted the 0.02 default and under-reported every ratio 4x.
LOW_LEVEL_SLEW_M_PER_TICK = 0.005
LOW_LEVEL_HZ = 50.0

#: ⚠️ A CHUNK STEP IS NOT A TICK. The chunk plays at 30 Hz and the low level slews at 50 Hz, so one
#: chunk step is ``50/30 = 1.67`` ticks. Over this, the wire is CLAMPED and the command silently
#: LAGS the plan — the body keeps moving in the right direction and never arrives, which reads as a
#: sluggish policy rather than as a rate limit, and no gate catches it because every clamped frame
#: is a perfectly valid keypoint command.
SLEW_BUDGET_PER_STEP_M = LOW_LEVEL_SLEW_M_PER_TICK * (LOW_LEVEL_HZ / 30.0)

#: ⚠️ The cap is ONE L2 over the whole concatenated position vector, not a per-point allowance:
#: ``holosoma_wbt._slew_vec`` takes ``norm(target - prev)`` over the ``(15,)`` 5pt array and scales
#: the ENTIRE delta by ``max_step/dist`` when over. So the five points share one pool and each is
#: clamped by the same proportional factor. If ``k`` points move comparably the per-point allowance
#: is ``budget/sqrt(k)`` — 8.33 mm total is 3.73 mm/point when all five move. A per-point maximum
#: (which is what this module reported before) is therefore a LOWER BOUND on the demand by up to
#: ``sqrt(5)`` = 2.24x, on top of the 4x from the wrong constant.
SLEW_IS_SHARED_POOL = True

#: A hand speed above which a chunk step is not a demonstrated motion but a DISCONTINUITY in the
#: prediction. Deliberately a PHYSICAL bound and not a fraction of the slew budget: a speed is a
#: property of the task and transfers across checkpoints, datasets and any future launcher value,
#: whereas a budget ratio moves whenever someone edits the cap. Human reaching peaks near
#: 1-2 m/s; the shipped H=50 checkpoint's documented 182-285 mm per 30 Hz step is **5.5-8.6 m/s**,
#: which no operator produced — raising the cap to "fit" that would execute the jump instead of
#: clamping it, which is the wrong direction.
IMPLAUSIBLE_SPEED_MS = 2.0


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------
class FivePointDirectWire(FivePointWire):
    """The ``5pt`` wire fed by the POLICY, with no completion in the chain.

    Subclasses :class:`~fuji.s1_policy.serving.hl_dense.FivePointWire` rather than copying it, so
    every safety-relevant method is the code that path is reviewed against. Exactly one is
    overridden — ``plan``, because its argument is a decoded chunk and not a ``CompletedChunk`` —
    and two defaults are inverted with the measurements that justify them in the module docstring
    (the clutch is ON, and it carries all five points).
    """

    low_level_mode = "5pt-unified-anneal"

    def __init__(self, readout: KeypointReadout, ready_pos: np.ndarray, ready_orn: np.ndarray,
                 *, clutch: KeypointClutch | None = None, measured_frame=None,
                 held: dict | None = None) -> None:
        # upper/foot_source are the COMPLETION path's two questions and neither exists here:
        # there is no completion to read a point out of, and the feet are predicted (or HELD —
        # see `held` below, which is a third answer neither of those knobs can spell). Pinned to
        # "policy"/"completion" so the inherited `plan` can never be reached by accident — it is
        # overridden anyway, and the status line below states the real source.
        super().__init__(readout, ready_pos, ready_orn, clutch=clutch,
                         measured_frame=measured_frame, upper_source="policy",
                         foot_source="completion")
        #: ``role -> (pos (3,), rot6d (6,))`` in the ANCHOR world frame, for every wire slot the
        #: policy does NOT predict — the ready pose's own pose for that slot
        #: (:func:`held_anchor_poses`). Broadcast over the chunk by :func:`fill_held_roles`, so a
        #: held point is a constant row: the clutch publishes its base and a Predict never moves
        #: it. Empty for a five-point checkpoint, which is then byte-identical to before.
        self.held: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role, (pos, rot6d) in (held or {}).items():
            if role not in WIRE_ROLES:
                raise ValueError(f"held role {role!r} is not a wire slot {list(WIRE_ROLES)}")
            p = np.asarray(pos, dtype=np.float64).reshape(-1)
            r = np.asarray(rot6d, dtype=np.float64).reshape(-1)
            if p.shape != (3,) or r.shape != (6,) or not (np.isfinite(p).all() and np.isfinite(r).all()):
                raise ValueError(f"held pose for {role!r} must be a finite (3,) position and (6,) "
                                 f"rot6d, got {p.shape} / {r.shape}")
            self.held[role] = (p.copy(), r.copy())
        #: Measures the low level's real position cap off its own RecordTap stream. Lives on the
        #: wire because ``sync`` is already the one method handed every frame, so the observation
        #: costs no new plumbing and cannot fall out of step with the frames the clutch sees.
        self.slew = SlewMonitor()
        #: Wire slots whose FRAMES are currently written by something other than this wire's plan —
        #: the walk skill's LEG OVERRIDE puts the clip's feet on slots 3 and 4 of every published
        #: frame while the planner keeps committing its own foot target here. ``sync`` then skips
        #: the leak bound on them (``KeypointClutch.sync(hold_slots=)``): the measured feet are
        #: walking, and bounding the planner's stance against them would drag it along the gait.
        #: Empty (the default) is byte-identical to before. Set by the page, read here.
        self.externally_driven: frozenset[int] = frozenset()

    @property
    def hold_slots(self) -> tuple[int, ...]:
        return tuple(sorted(int(i) for i in self.externally_driven))

    def sync(self, rframe, force_n: float | None = None) -> str:
        # Observe BEFORE delegating: `sync` is called every loop iteration whatever the clutch is
        # doing, so this is the only hook that sees un-clamped and clamped ticks alike. The
        # measurement is read-only and must never influence what is published.
        self.slew.observe(rframe)
        return super().sync(rframe, force_n)

    @property
    def gate_note(self) -> str:
        base = super().gate_note
        return base.replace("the completion's inference", "the POLICY's own prediction")

    @property
    def held_roles(self) -> tuple[str, ...]:
        """The wire slots filled from the ready pose rather than the policy, in wire order."""
        return tuple(r for r in WIRE_ROLES if r in self.held)

    def all_rows(self, chunk: ChunkResult) -> dict:
        """Every wire role's ``{"position": (H, 3), "rot6d": (H, 6)}`` — predicted from the
        chunk, held from :attr:`held`; a role that is neither is refused by name
        (:func:`fill_held_roles`)."""
        rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in chunk.roles}
        if not rows:
            raise ValueError("the chunk has no roles at all")
        n = int(next(iter(rows.values()))["position"].shape[0])
        return fill_held_roles(rows, self.held, n)

    def decoded_positions(self, chunk: ChunkResult) -> np.ndarray:
        """``(H, 15)`` anchor-frame positions in wire order, held slots included — the input
        :func:`chunk_span` wants. No readout and no clutch: both differ from this by a rigid
        per-keypoint offset, which cancels in an excursion and in a per-step difference, so the
        span is the same number without touching the clutch off the publisher's thread."""
        rows = self.all_rows(chunk)
        return np.concatenate([np.asarray(rows[r]["position"], dtype=np.float64).reshape(-1, 3)
                               for r in WIRE_ROLES], axis=1)

    def plan(self, chunk: ChunkResult, decoded: dict | None = None) -> dict:
        """A decoded chunk -> the per-step frames to publish, held slots filled.

        ``chunk`` is a :class:`~fuji.s1_policy.serving.hl_online.ChunkResult` carrying the roles
        the POLICY predicts — all five for ``fivepoint_geom_*``, one for ``left_only``; every
        other wire slot comes from :attr:`held`. ``decoded`` is accepted and ignored so the
        signature stays interchangeable with ``FivePointWire.plan(completed, decoded)`` and the
        drip loop need not know which family produced the keypoints.
        """
        del decoded
        pos, orn = self.readout.all_from_decoded(self.all_rows(chunk))
        pos, orn = self.clutch.apply(pos, orn)
        return {"pos": pos, "orn": orn, "n": int(pos.shape[0])}

    def status(self, frame: tuple) -> str:
        # The inherited line ends in "upper=policy feet=completion", which is true of the parent's
        # sources and a lie about this wire: there is no completion. Replaced rather than appended.
        base = super().status(frame).split(" · upper=")[0]
        carried = self.clutch.n_carried
        held = self.held_roles
        held_note = f" · HELD at the ready pose: {','.join(held)}" if held else ""
        ext = self.hold_slots
        ext_note = (f" · slots {','.join(WIRE_ROLES[i] for i in ext)} EXTERNALLY DRIVEN (leg override)"
                    if ext else "")
        return (f"{base} · direct (no completion) · clutch "
                f"{'ON' if self.clutch.post.carrying else 'OFF'}, carrying {carried}/"
                f"{len(FIVE_POINT_BODIES)}{held_note}{ext_note}")


#: A RecordTap frame older than this is "no measurement" for the hands' reach compensation: the
#: residual is HELD (re-applied unchanged, not grown) — the same answer :class:`ReachCompensator`
#: gives a ``measured=None`` step. A stale readback under a moving command would otherwise read as
#: a stalled arm and be integrated.
REACH_MEASURED_MAX_AGE_S = 0.5
#: What a hand's contact force is taken to be when the rig HAS a Bota for it and the bridge did not
#: answer this step: a FINITE value above any gate, so ``ReachCompensator``'s contact cut-out engages
#: (it treats a non-finite force as "no information", which would let the residual grow against a
#: hand that may be pressing). ``--reach-contact-n 0`` still disables the cut-out, as documented.
SILENT_BOTA_FORCE_N = 1.0e6


class HandReach:
    """Reach compensation on the 5pt wire — the two HANDS only, one :class:`ReachCompensator` each.

    ``run_hl_serving.sh``'s ``--reach-ki`` ported to this wire, with the four things that had to be
    decided for five points made explicit:

    * **Hands only.** The feet are a STANCE the balance controller reads (never an aim), the torso is
      the completion's root channel on the sibling path, and a held slot is CONSTANT by contract —
      so only the hand roles the POLICY PREDICTS get a compensator, selected by name from
      :data:`FORCE_HANDS`. Every other slot passes through byte-identically.
    * **Published, never committed.** The compensated frame goes on the wire; the clutch base is
      advanced with the UNCOMPENSATED frame by the caller. Committing the compensated one would fold
      the residual into the base every chunk and the "at most ``clamp_m`` of extra reach" bound
      would become "``clamp_m`` MORE per chunk" — the wind-up the module exists to rule out.
    * **Un-armed ⇒ identity, and the residual is dropped.** A low level that is not armed does not
      follow the wire, so "commanded moved, measured did not" is guaranteed and would integrate to
      the clamp before the first Run; and the pre-arm gate compares the broadcast wire against its
      reference, so a residual on the wire before arming would move step 0 off the very reference
      ``--post-anchor command`` puts it on. The caller passes ``armed`` from RecordTap.
    * **A silent Bota reads as CONTACT.** A hand this rig has a Bota for (``force_roles``) whose
      reading did not arrive is passed :data:`SILENT_BOTA_FORCE_N` — the cut-out engages (hold +
      decay) rather than the compensator integrating against a hand that may be pressing. A hand with NO Bota at all
      is :attr:`unprotected` (named at startup) and follows the module's own ``None`` contract.

    ``apply`` with ``ki = 0``, or with the page toggle off, returns the SAME frame object: the
    publish path is then byte-identical to the one reviewed before this class existed.
    Thread-safe: one lock over the per-hand state, because the page's toggle (a viser callback)
    resets while the publisher compensates.
    """

    HANDS: tuple[str, ...] = FORCE_HANDS

    def __init__(self, config: ReachConfig, hand_roles, force_roles=()) -> None:
        self.config = config
        roles = tuple(str(r) for r in hand_roles if str(r) in self.HANDS)
        self._slots = {r: 3 * WIRE_ROLES.index(r) for r in roles}
        self.comps: dict[str, ReachCompensator] = {r: ReachCompensator(config) for r in roles}
        self.force_roles = tuple(str(r) for r in force_roles)
        self.active = True
        self.n_unarmed_resets = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """``ki > 0`` AND at least one predicted hand — the flag alone does not make it live."""
        return bool(self.config.enabled and self.comps)

    @property
    def live(self) -> bool:
        """Enabled and not paused by the page's toggle."""
        return self.enabled and bool(self.active)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.comps)

    @property
    def unprotected(self) -> tuple[str, ...]:
        """Compensated hands with NO F/T channel on this layout: their contact cut-out can never
        engage, so the operator is told to prefer a smaller clamp."""
        return tuple(r for r in self.comps if r not in self.force_roles)

    def reset(self) -> None:
        with self._lock:
            for comp in self.comps.values():
                comp.reset()

    def apply(self, frame: tuple, *, measured: tuple | None, armed: bool | None,
              force_by_hand: dict | None) -> tuple:
        """``(pos (15,), orn (20,))`` to publish, from the clutched model target.

        ``measured`` is the robot's ``(pos, orn)`` wire frame from ``FivePointWire.measured_keypoints``
        (None = no fresh readback: hold), ``armed`` is RecordTap's ``vive_armed`` (None = unknown),
        ``force_by_hand`` is ``role -> |F| N`` for the hands the rig read this step.
        """
        if not self.live:
            return frame
        pos = np.asarray(frame[0], dtype=np.float64).reshape(-1)
        if pos.size != 3 * len(WIRE_ROLES):
            raise ValueError(f"a 5pt frame has {3 * len(WIRE_ROLES)} positions, got {pos.size}")
        if armed is False:
            with self._lock:
                self.n_unarmed_resets += 1
            self.reset()
            return frame
        meas = None
        if measured is not None:
            meas = np.asarray(measured[0], dtype=np.float64).reshape(-1)
            if meas.size != pos.size:
                raise ValueError(f"measured frame has {meas.size} positions, wire has {pos.size}")
        out = pos.copy()
        forces = force_by_hand or {}
        with self._lock:
            for role, comp in self.comps.items():
                sl = slice(self._slots[role], self._slots[role] + 3)
                f_n = forces.get(role)
                if f_n is None and role in self.force_roles:
                    f_n = SILENT_BOTA_FORCE_N   # a Bota this rig reads did not answer: CONTACT, not free
                out[sl] = comp.compensate(out[sl], None if meas is None else meas[sl], f_n)
        return (out, frame[1])

    def residual_mm(self) -> dict[str, float]:
        with self._lock:
            return {r: c.residual_norm * 1e3 for r, c in self.comps.items()}

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "active": bool(self.active),
                    "n_unarmed_resets": int(self.n_unarmed_resets),
                    "hands": {r: c.stats() for r, c in self.comps.items()}}

    def describe(self) -> str:
        if not self.enabled:
            return "OFF (--reach-ki 0) — the wire is the clutched model target"
        if not self.active:
            return "paused (page toggle off; residual dropped)"
        parts = []
        with self._lock:
            for role, comp in self.comps.items():
                rf = comp.realized_fraction
                parts.append(f"{'L' if role == 'left' else 'R'} {comp.residual_norm * 1e3:.1f} mm"
                             + ("" if rf is None else f" (realized {rf:.2f})")
                             + (" ⚠ SATURATED" if comp.saturated else ""))
        return " · ".join(parts)


def held_roles(layout: ActionLayout) -> tuple[str, ...]:
    """The wire slots ``layout`` does NOT predict, in wire order — ``()`` for a five-point one."""
    return tuple(r for r in WIRE_ROLES if r not in layout.roles)


def fill_held_roles(rows: dict, held: dict, n: int) -> dict:
    """Complete a per-role row dict to every wire slot.

    ``rows`` is ``role -> {"position": (H, 3), "rot6d": (H, 6)}`` for the PREDICTED roles;
    ``held`` is ``role -> (pos (3,), rot6d (6,))`` for the slots the policy does not predict.
    Returns a NEW dict with all of :data:`WIRE_ROLES`: predicted rows pass through untouched (same
    arrays, not copies), a held role becomes ``n`` identical rows. A role that is neither predicted
    nor held is refused BY NAME — a five-point wire cannot be published from four points, and a
    zero-filled slot would be a perfectly valid-looking command to fold the arm onto the pelvis.
    """
    if int(n) < 1:
        raise ValueError(f"a chunk has at least one row, got n={n}")
    out = dict(rows)
    for role in WIRE_ROLES:
        if role in out:
            continue
        if role not in held:
            raise ValueError(
                f"wire slot {role!r} is neither predicted by this checkpoint nor held: the 5pt "
                f"wire needs all of {list(WIRE_ROLES)}. Predicted {sorted(rows)}, held "
                f"{sorted(held)}.")
        pos, rot6d = held[role]
        out[role] = {"position": np.tile(np.asarray(pos, dtype=np.float64).reshape(1, 3), (int(n), 1)),
                     "rot6d": np.tile(np.asarray(rot6d, dtype=np.float64).reshape(1, 6), (int(n), 1))}
    return out


def held_anchor_poses(anchor, layout: ActionLayout) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``role -> (pos, rot6d)`` of the READY POSE for every wire slot ``layout`` does not predict.

    ``anchor`` is a ``ready_pose.DeployAnchor`` resolved over ALL of :data:`WIRE_ROLES` (the
    planner resolves it that way whatever the layout, because the wire's idle frame needs every
    slot regardless). Refuses by name if a held role is missing from it, rather than letting the
    wire fall back to a zero pose.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role in held_roles(layout):
        if role not in getattr(anchor, "pos", {}) or role not in getattr(anchor, "rotmat", {}):
            raise ValueError(
                f"the deploy anchor has no {role!r} role, so the wire slot the policy does not "
                f"predict cannot be held. Resolve it over hl_5pt.WIRE_ROLES, not layout.roles.")
        out[role] = (np.asarray(anchor.pos[role], dtype=np.float64).reshape(3).copy(),
                     np.asarray(rotmat_to_rot6d(np.asarray(anchor.rotmat[role], dtype=np.float64)),
                                dtype=np.float64).reshape(6))
    return out


def force_command(rows_by_hand: dict, gate_n: float) -> np.ndarray | None:
    """Per-hand predicted wrench rows -> the ``(6,)`` sensor-frame ``f_cmd`` = ``[L, R]``, or None.

    The same two-level contract and the same sign flip in the same one place as every planner:
    the model predicts ``F_meas`` (what the Bota READ during the demo) and the low level closes
    ``e = F^cmd + F_meas -> 0``, so the command reproducing the demonstrated press is ``-F_meas``
    (:func:`fuji.utils.force_frame.cmd_from_meas_force`). Gated on the MAGNITUDE, which the flip
    preserves: at or below ``gate_n`` a hand's command is a hard zero.

    ``rows_by_hand`` is ``hand -> (>=3,) wrench row | None`` for the hands the policy predicts; a
    hand that is absent, ``None`` or non-finite publishes ZERO. The result is always both hands
    wide — ``ViveLiveSource`` accepts a ``(3,)`` LEFT-only vector too, but a partial publish would
    leave the other hand's command at whatever the low level last had, and a held hand must read
    as "no force", not "the previous force". ``None`` (no key on the wire at all) when no hand is
    live, so a pure-position step is byte-identical to before.
    """
    out, any_live = [], False
    for hand in FORCE_HANDS:
        row = rows_by_hand.get(hand)
        if row is None:
            out.append(np.zeros(3))
            continue
        f3 = np.asarray(row, dtype=np.float64).reshape(-1)[0:3]
        if f3.shape != (3,) or not np.all(np.isfinite(f3)) or float(np.linalg.norm(f3)) <= gate_n:
            out.append(np.zeros(3))
            continue
        out.append(np.asarray(cmd_from_meas_force(f3), dtype=np.float64).reshape(3))
        any_live = True
    return np.concatenate(out) if any_live else None


class ManualJawOverride:
    """The operator's ``Manual jaws`` press WINS over the policy's jaw command — per hand, until Reset.

    Until 2026-09-18 a press was REFUSED while the policy was driving that jaw: the policy's
    per-step ``target_frac`` would have overwritten it within 33 ms, and the operator would have
    read the press as a dead button. The rig wanted the opposite precedence — the press is the
    operator's interrupt on THAT HAND while the arm keeps following the chunk. So :meth:`take`
    latches the hand, both publish paths (the chunked drip and the stream) ask :meth:`intercept`
    before the jaw state machine and DROP the policy's width while it is latched, and nothing else
    releases it — not ⏹, not STOP, not a new ▶▶, not the jaw checkbox — only :meth:`release_all`
    from the Reset branch, which is also where the page opens the policy's jaws, so the planner
    takes the hand back OPEN from the ready pose exactly as on the first run.

    ``lock`` is re-entrant and shared with the callers: the press does ``take`` + send under it
    and the publish paths do ``intercept`` + state machine + send under it, so a press can never
    land between the policy's check and its send (the one window in which a 33 ms overwrite could
    still eat the press). A hand the policy does not predict is never blocked (there is nothing
    to block) — ``take`` still records it so the readout can say who last drove the jaw.
    """

    def __init__(self, policy_roles) -> None:
        self.policy_roles = tuple(policy_roles)
        self.lock = threading.RLock()
        self._held: dict[str, dict] = {}

    def take(self, role: str, frac: float) -> bool:
        """Latch ``role`` at ``frac`` (1 = open). True when this blocks a jaw the POLICY predicts."""
        with self.lock:
            st = self._held.setdefault(role, {"skipped": 0})
            st["frac"] = float(frac)
        return role in self.policy_roles

    def blocks(self, role: str) -> bool:
        """Is the policy's jaw command for ``role`` dropped right now? Pure; see :meth:`intercept`."""
        with self.lock:
            return role in self._held and role in self.policy_roles

    def intercept(self, role: str) -> bool:
        """:meth:`blocks`, counting the dropped command — the publish paths' one call per step."""
        with self.lock:
            if role not in self._held or role not in self.policy_roles:
                return False
            self._held[role]["skipped"] += 1
            return True

    @property
    def roles(self) -> tuple[str, ...]:
        """The POLICY jaws the operator currently holds, in ``policy_roles`` order."""
        with self.lock:
            return tuple(r for r in self.policy_roles if r in self._held)

    def describe(self, role: str) -> str | None:
        """Readout for the page — None when the operator has not touched ``role``."""
        with self.lock:
            st = self._held.get(role)
            if st is None:
                return None
            frac, skipped = float(st["frac"]), int(st["skipped"])
        if role not in self.policy_roles:
            return f"manual {frac * 100:.0f}% open"
        return (f"MANUAL OVERRIDE {frac * 100:.0f}% open · {skipped} policy cmd(s) ignored · "
                f"Reset hands it back")

    def release_all(self) -> tuple[str, ...]:
        """Reset: forget every press. Returns the POLICY jaws that were being held."""
        with self.lock:
            was = tuple(r for r in self.policy_roles if r in self._held)
            self._held.clear()
        return was


def build_clutch(ready_pos: np.ndarray, ready_orn: np.ndarray, *, post: PostAnchorConfig,
                 carry: str) -> KeypointClutch:
    """The clutch for a direct 5pt wire. ``carry`` is one of :data:`CARRY_MODES`."""
    if str(carry) not in CARRY_MODES:
        raise ValueError(f"carry must be one of {CARRY_MODES}, got {carry!r}")
    n_points = len(FIVE_POINT_BODIES)
    n_carried = n_points if str(carry) == "all" else FIVE_POINT_GATED
    return KeypointClutch(ready_pos, ready_orn, post=post, n_points=n_points,
                          n_carried=n_carried)


# ---------------------------------------------------------------------------
# the low level's rate cap, MEASURED rather than declared
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlewBudget:
    """The low level's per-tick position cap, carrying WHERE THE NUMBER CAME FROM.

    The provenance is not decoration. FUJI and the low level are separate processes, possibly on
    separate hosts, and the value is an inline prefix in a launcher script that no exported
    environment variable can override — so this process fundamentally cannot look it up. Every
    printed ratio therefore has to say whether it was scored against something observed or
    something assumed, or a stale assumption reads exactly like a measurement.
    """

    m_per_tick: float = LOW_LEVEL_SLEW_M_PER_TICK
    source: str = "assumed (launcher default, not yet confirmed by RecordTap)"
    n_ticks: int = 0

    @property
    def mm_per_step(self) -> float:
        return self.m_per_tick * (LOW_LEVEL_HZ / 30.0) * 1e3


class SlewMonitor:
    """Measures the low level's position cap from its own RecordTap stream.

    ``RecordFrame.vr_cmd_pos`` is the keypoint target the low level's policy actually saw, **after**
    ``_slew_vec`` applied the cap. So successive frames differ by at most the cap, and while the
    wire is demanding more than the cap that difference SATURATES at exactly the cap on every tick.
    Two facts fall out with no configuration at all:

    * ``max`` observed per-tick step is a hard LOWER BOUND on the cap;
    * a high :attr:`saturated_frac` means many ticks landed on the same ceiling, i.e. the bound is
      tight and the cap has been *measured* — and simultaneously that the wire is being CLAMPED
      right now, which is the operationally useful reading.

    This is why the answer to "will I remember to retune this per task" is that there is nothing to
    retune: the number is read off the robot. It only ever narrows the estimate, never widens it, so
    a quiet session leaves the conservative pre-flight value in place.

    ⚠️ Only ARMED ticks count. Un-armed, the low level is not applying the wire and its target is
    whatever the idle path holds, so a difference there measures nothing about the cap.

    ⚠️ Only the points the cap COVERS are differenced. ``HOLOSOMA_VIVE_SLEW_FEET=0`` (the launcher's
    default since the walk skill) exempts the two 5pt feet, and the low level says so on every
    frame (``RecordFrame.vr_cmd_slewed``). A whole-vector norm would then read a walking foot's
    46 mm/tick as the "ceiling" and void the upper points' genuine 5 mm plateau — so the norm runs
    over the covered rows only, :attr:`slew_mask` exposes what the robot reported, and a frame
    without the key (an older low level) is treated as all-covered, byte-identical to before.
    """

    #: Within this relative distance of the running max, a tick counts as sitting on the ceiling.
    PLATEAU_TOL = 0.02
    #: Below this many armed ticks the sample is too small to claim anything.
    MIN_TICKS = 25
    #: Largest ``seq`` gap between two observed frames that is still normalised into a PER-TICK
    #: step. RecordTap emits one frame per control tick, but a reader may miss a tick now and then
    #: (a 50 Hz publisher against a 50 Hz tap, e.g. the walk skill) or read it only at a chunk
    #: boundary (~27 ticks). A raw difference over k ticks is k times the cap — measured on the mock
    #: rig, ONE skipped tick read as a "10.0 mm/tick ceiling" and voided a genuine 5 mm plateau on
    #: 367 clamped ticks. Past this gap the target may have turned, so the displacement says nothing
    #: about a per-tick cap and the history is re-anchored, unmeasured.
    MAX_GAP_TICKS = 5

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._prev_seq: int | None = None
        self._max_step = 0.0
        self._n = 0
        self._n_plateau = 0
        #: The last per-point coverage the low level reported (``vr_cmd_slewed``), or ``None`` if
        #: it never said (no frame yet, or a low level from before the key existed).
        self._mask: np.ndarray | None = None

    @property
    def slew_mask(self) -> np.ndarray | None:
        """``(n,)`` bool — which wire points the low level's caps cover — as REPORTED by it on
        RecordTap, or ``None`` when it has not said (armed or not). Read-only for the page."""
        return None if self._mask is None else self._mask.copy()

    @property
    def feet_slewed(self) -> bool | None:
        """Do the caps cover the two FEET (the last two points)? ``None`` = not reported yet."""
        if self._mask is None or self._mask.shape[0] < 2:
            return None
        return bool(self._mask[-2:].any())

    def observe(self, rframe) -> None:
        """Feed one RecordTap frame. Cheap (one subtract + one norm); safe to call every loop."""
        if rframe is None or getattr(rframe, "vr_cmd_pos", None) is None:
            return
        pos = np.asarray(rframe.vr_cmd_pos, dtype=np.float64).reshape(-1, 3)
        mask = getattr(rframe, "vr_cmd_slewed", None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if mask.shape[0] == pos.shape[0]:
                self._mask = mask   # the robot's own statement, armed or not
            else:
                mask = None
        if not bool(getattr(rframe, "vive_armed", False)):
            # Un-armed: the wire is not being applied. Drop the history too, so the first armed
            # tick is not differenced against a stale idle target.
            self._prev, self._prev_seq = None, None
            return
        if mask is None:
            cur = pos.reshape(-1)
        else:
            if not mask.any():
                return   # nothing is capped: there is no ceiling to measure
            cur = pos[mask].reshape(-1)
        seq = int(getattr(rframe, "seq", -1))
        if self._prev is not None and self._prev.shape == cur.shape and seq != self._prev_seq:
            # Per TICK, not per observed frame — see MAX_GAP_TICKS. A gap of exactly 1 (the tap read
            # every tick) divides by 1 and is byte-identical to the previous behaviour.
            n_ticks = seq - self._prev_seq if self._prev_seq is not None else 1
            step = (float(np.linalg.norm(cur - self._prev)) / n_ticks
                    if 1 <= n_ticks <= self.MAX_GAP_TICKS else float("nan"))
            if np.isfinite(step) and step > 0.0:
                self._n += 1
                if step > self._max_step * (1.0 + self.PLATEAU_TOL):
                    self._max_step, self._n_plateau = step, 1   # new ceiling; old plateau is void
                elif step >= self._max_step * (1.0 - self.PLATEAU_TOL):
                    self._max_step = max(self._max_step, step)
                    self._n_plateau += 1
        self._prev, self._prev_seq = cur, seq

    @property
    def saturated_frac(self) -> float:
        """Fraction of armed ticks sitting on the observed ceiling. High ⇒ actively CLAMPED."""
        return (self._n_plateau / self._n) if self._n else 0.0

    @property
    def budget(self) -> SlewBudget:
        """The best available cap: measured when the evidence supports it, else the estimate.

        Requires both a big enough sample and a real plateau. Without the plateau the observation
        is only "the wire never asked for more than X", which bounds the cap from below but is not
        a measurement of it — reporting that as the budget would silently TIGHTEN the printed
        ratio on a slow take and invent a clamp that is not happening.
        """
        if self._n < self.MIN_TICKS or self._max_step <= 0.0:
            return SlewBudget(n_ticks=self._n)
        if self.saturated_frac < 0.10:
            return SlewBudget(source=(f"assumed — {self._n} armed ticks never reached a ceiling "
                                      f"(peak {self._max_step * 1e3:.1f} mm/tick, no clamping "
                                      f"observed)"),
                              n_ticks=self._n)
        return SlewBudget(m_per_tick=self._max_step,
                          source=(f"MEASURED from RecordTap — {self._n} armed ticks, "
                                  f"{self.saturated_frac * 100:.0f}% on the ceiling"),
                          n_ticks=self._n)

    def describe_mask(self) -> str:
        m = self._mask
        if m is None:
            return "coverage not reported (low level silent or pre-dates vr_cmd_slewed: assumed ALL points)"
        n_cov = int(m.sum())
        if n_cov == m.shape[0]:
            return f"covers ALL {m.shape[0]} points (SLEW_FEET=1)"
        if m.shape[0] >= 2 and not m[-2:].any() and m[:-2].all():
            return f"covers the upper {n_cov} points, FEET EXEMPT (SLEW_FEET=0)"
        return f"covers points {np.flatnonzero(m).tolist()} of {m.shape[0]}"

    def describe(self) -> str:
        b = self.budget
        if self._n == 0:
            return ("slew cap: no armed RecordTap ticks yet — using the pre-flight estimate "
                    f"{LOW_LEVEL_SLEW_M_PER_TICK * 1e3:.0f} mm/tick "
                    f"({SLEW_BUDGET_PER_STEP_M * 1e3:.1f} mm/step) · {self.describe_mask()}")
        return (f"slew cap {b.m_per_tick * 1e3:.1f} mm/tick ({b.mm_per_step:.1f} mm/step) · "
                f"{b.source} · {self.describe_mask()}")


# ---------------------------------------------------------------------------
# the chunk's own numbers, for the page
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkSpan:
    """How far a chunk asks each keypoint to travel, and how fast. Printed every Predict.

    A five-point wire's whole risk profile is "how big is the step and how big is the excursion",
    and neither is visible from the gate report (which only looks at step 0). Both are properties
    of the CHUNK and not of a setting, so they are reported whatever the clutch is doing.
    """

    excursion_mm: np.ndarray     # (5,) worst |p[i] - p[0]|
    slew_mm: np.ndarray          # (5,) worst |p[i+1] - p[i]| PER POINT — says WHICH point is fast
    #: Worst per-step L2 over the CAPPED points concatenated (all of them unless the low level
    #: reported an exemption — see ``covered``). This — not ``slew_mm.max()`` — is the quantity
    #: ``_slew_vec`` bounds (see :data:`SLEW_IS_SHARED_POOL`), so it is what decides whether the
    #: wire is clamped. ``>= slew_mm.max()`` over the covered points.
    step_mm_total: float
    labels: tuple[str, ...]
    #: The budget this span was scored against, m/tick, and where that number came from. Defaults
    #: to the pre-flight estimate; a live planner passes :attr:`SlewMonitor.budget` so the printed
    #: ratio is scored against the cap the robot is DEMONSTRABLY applying.
    budget_m_per_tick: float = LOW_LEVEL_SLEW_M_PER_TICK
    budget_source: str = "assumed (launcher default, not yet confirmed by RecordTap)"
    #: Which points the cap covers, as the low level REPORTED it (``SlewMonitor.slew_mask``); ``None``
    #: = not reported, scored over all points (the historical reading). With the feet exempt the
    #: ratio is about the upper three only — a foot's step is still in ``slew_mm`` / ``peak_speed_ms``
    #: (the cap-independent numbers), it just no longer predicts a clamp.
    covered: tuple[bool, ...] | None = None

    @property
    def budget_mm_per_step(self) -> float:
        """The concatenated allowance in ONE 30 Hz chunk step."""
        return self.budget_m_per_tick * (LOW_LEVEL_HZ / 30.0) * 1e3

    @property
    def slew_ratio(self) -> float:
        """Worst per-step demand as a multiple of what the low level will DELIVER.

        Over 1.0 the low level clamps and the published command LAGS the plan — silently, because
        the body still moves the right way and simply never arrives. This is the number that says
        "the chunk is too jerky for this wire", and no gate catches it: every clamped frame is a
        perfectly valid keypoint command.
        """
        budget = self.budget_mm_per_step
        return float(self.step_mm_total / budget) if budget > 0.0 else float("inf")

    @property
    def peak_speed_ms(self) -> float:
        """Worst single-point speed the chunk demands, m/s. **Cap-independent.**

        The one number worth carrying between tasks: it is a property of the motion, so it stays
        comparable across checkpoints, datasets and any future launcher value, while
        :attr:`slew_ratio` moves whenever the cap is edited.
        """
        return float(self.slew_mm.max() * 1e-3 * 30.0)

    @property
    def implausible(self) -> bool:
        """True when the chunk demands a speed no demonstration contains — i.e. a PREDICTION
        DISCONTINUITY rather than a fast task. Needs no budget and no per-task tuning, which is the
        whole point: raising the cap cannot make this false, and should not be attempted."""
        return self.peak_speed_ms > IMPLAUSIBLE_SPEED_MS

    def describe(self) -> str:
        worst = int(np.argmax(self.slew_mm))
        ratio = self.slew_ratio
        speed = self.peak_speed_ms
        note = ""
        if ratio > 1.0:
            note = (f" ⚠️ {ratio:.1f}x the low level's {self.budget_mm_per_step:.1f} mm/step "
                    f"[{self.budget_source}] — it will CLAMP and the command will lag the plan")
        if self.implausible:
            note += (f" ⚠️ {speed:.1f} m/s is NOT a demonstrated motion ("
                     f">{IMPLAUSIBLE_SPEED_MS:g} m/s) — treat as a chunk DISCONTINUITY, not as a "
                     f"reason to raise the cap")
        if self.covered is None or all(self.covered):
            pool = "all-point step"
        else:
            pool = (f"capped-point step ({'+'.join(lab for lab, c in zip(self.labels, self.covered) if c)}; "
                    f"{'+'.join(lab for lab, c in zip(self.labels, self.covered) if not c)} exempt)")
        return (f"chunk reach {self.excursion_mm.max():.0f} mm · worst step "
                f"{self.slew_mm[worst]:.0f} mm ({self.labels[worst]}) = {speed:.2f} m/s · "
                f"{pool} {self.step_mm_total:.0f} mm{note}")


def chunk_span(pos: np.ndarray, labels: tuple[str, ...], *,
               budget: "SlewBudget | None" = None, covered=None) -> ChunkSpan:
    """``(H, 3n)`` published positions -> per-point excursion and per-step slew, in mm.

    ``budget`` is the cap to score against; ``None`` uses the pre-flight estimate. Pass
    :attr:`SlewMonitor.budget` from a live planner so the ratio reflects the MEASURED cap.
    ``covered`` is :attr:`SlewMonitor.slew_mask` — the points the cap covers as the low level
    reported them; ``None`` (not reported) scores all points, byte-identical to before.
    """
    p = np.asarray(pos, dtype=np.float64)
    n = len(labels)
    b = budget or SlewBudget()
    cov = None
    if covered is not None:
        cov_arr = np.asarray(covered, dtype=bool).reshape(-1)
        if cov_arr.shape[0] == n:
            cov = tuple(bool(c) for c in cov_arr)
    if p.ndim != 2 or p.shape[1] != 3 * n or p.shape[0] == 0:
        return ChunkSpan(np.zeros(n), np.zeros(n), 0.0, labels,
                         b.m_per_tick, b.source, cov)
    p = p.reshape(-1, n, 3)
    exc = np.linalg.norm(p - p[0], axis=-1).max(axis=0) * 1e3
    if p.shape[0] > 1:
        d = np.diff(p, axis=0)                                  # (H-1, n, 3)
        slew = np.linalg.norm(d, axis=-1).max(axis=0) * 1e3     # per point
        # The concatenated norm per step over the CAPPED points: sqrt over BOTH the point and the
        # xyz axis, because the low level caps the flat delta of those points as one vector.
        dc = d if cov is None else d[:, np.asarray(cov, dtype=bool), :]
        total = (float(np.linalg.norm(dc.reshape(dc.shape[0], -1), axis=-1).max() * 1e3)
                 if dc.shape[1] > 0 else 0.0)
    else:
        slew, total = np.zeros(n), 0.0
    return ChunkSpan(exc, slew, total, labels, b.m_per_tick, b.source, cov)


# ---------------------------------------------------------------------------
# the sampler's noise seed
# ---------------------------------------------------------------------------
class SeedCounter:
    """Which noise seed the next sample uses — pinned, or the next one up.

    A diffusion chunk is a *draw* from a conditional, so the seed is a real input to a Predict and
    the only one the operator cannot see on the page. Two Predicts from a bit-identical observation
    differ by this and nothing else; measured on the ``push_cart`` H=100 five-point checkpoint with
    the observation held fixed, that is **0.6-1.9 mm at step 0 but 6.7-18.6 mm at the worst step**,
    against a 65-88 mm chunk reach — the same gesture on a visibly different path. For scale, iid
    camera noise at 0.5-2 LSB moves it 0.1-0.3 mm and 1%-of-range force noise 0.6-1.1 mm, so the
    seed dominates the two live sensor channels by an order of magnitude.

    ⚠️ This is a class rather than three ``if seed is None`` branches at the sample sites because a
    planner samples from more than one place (Predict, and the streaming cadence's own thread). A
    site that forgot the pin would look like the pin working while that path's chunks drifted.

    :meth:`reset` is what makes ``Reset to ready pose`` mean what it says. Restoring the observation
    is not enough on its own: ``FivePointPlanner.seed_anchor`` re-seeds the pose rings and clears
    ``_geom_written``, so the *question* returns exactly — and the answer still changed, because the
    sampler had moved on. Resetting the seed is safe by construction: it selects WHICH sample, and
    the clutch, the pre-arm gate, the slew cap and the publish check are identical for every one.
    Re-sampling stays one press away, since a plain Predict advances the counter as before.
    """

    def __init__(self, seed: int | None = None) -> None:
        #: The pinned value, and also what :meth:`reset` returns to. 0 unpinned, so the first
        #: ``next()`` yields 1 — the historical numbering, kept so a log from before this existed
        #: still names the same chunk.
        self.seed0 = 0 if seed is None else int(seed)
        self.fixed = seed is not None
        self.n = self.seed0

    def next(self) -> int:
        """The seed for the next sample. Advances unless pinned."""
        if self.fixed:
            return int(self.seed0)
        self.n += 1
        return int(self.n)

    def reset(self) -> None:
        """Back to the start, so the next sample repeats the first one. No-op when pinned."""
        self.n = self.seed0

    def describe(self) -> str:
        if self.fixed:
            return f"{self.seed0} (PINNED by --seed — one observation, one chunk)"
        return f"{self.n} · increments per sample (Reset returns it to {self.seed0})"


# ---------------------------------------------------------------------------
# the violet body's staging discipline
# ---------------------------------------------------------------------------
class PreviewGate:
    """Which solved body — if any — is allowed on screen right now.

    The violet body is produced on a worker thread and consumed on the page loop, so "the body you
    are looking at" and "the chunk that is staged" are two different facts that can disagree. This
    holds the one token that keeps them honest: a **generation**. A worker captures it when it
    starts and may publish only while it is still current, so *bumping* it retires a solve that is
    already running.

    ⚠️ This class exists because clearing the solved body alone was a LIVE BUG. `Reset to ready
    pose` parked the body and dropped the solution, but left the generation untouched — so a solve
    already in flight still passed its own check and republished the OLD chunk's body a moment
    later. The violet mesh came back to wherever the last take had ended and stayed there until the
    next Predict, which reads exactly as "Reset did nothing to the purple robot". Nothing
    downstream could catch it: the body is display-only, so no gate, wire or clutch sees it.

    The window is not narrow in practice. A solve is ~0.3 s for a 50-step chunk and several seconds
    for a 200-step one, against a 533 ms drip — so at H = 100/200 a Reset pressed after a Run lands
    mid-solve nearly every time, and at H = 50 it usually does not. That is why the bug looked
    horizon-dependent rather than like a race.

    :meth:`submit` retires the previous solve as a side effect, which is the second half of the
    same invariant: only one body can ever land, and it is the newest one asked for.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen = 0
        self._solved: object | None = None
        self._res: str | None = None
        self._pending = False

    def submit(self) -> int:
        """Open a new generation and return it. Any in-flight solve is now unpublishable."""
        with self._lock:
            self._gen += 1
            self._pending = True
            return self._gen

    def publish(self, gen: int, solved: object, res: str) -> bool:
        """Offer a finished solve. ``False`` (and nothing changes) if it has been retired."""
        with self._lock:
            if int(gen) != self._gen:
                return False
            self._solved = solved
            self._res = res
            self._pending = False
            return True

    def give_up(self, gen: int) -> None:
        """A solve failed. Clears ``pending`` without publishing, if it is still the current one."""
        with self._lock:
            if int(gen) == self._gen:
                self._pending = False

    def retire(self) -> None:
        """Drop the body AND make every outstanding solve unpublishable. Reset's half of the fix."""
        with self._lock:
            self._gen += 1
            self._solved = None
            self._res = None
            self._pending = False

    @property
    def solved(self) -> object | None:
        return self._solved

    @property
    def pending(self) -> bool:
        return self._pending

    def status(self) -> str:
        """What the page's IK readout should say.

        A body on screen is not necessarily the STAGED chunk's — a 200-step solve outlasts the
        drip, so during a sequential run the page is usually showing the previous chunk. Said out
        loud rather than letting a stale body read as the current plan.
        """
        with self._lock:
            if self._pending:
                return ("solving … (the body shown is the PREVIOUS chunk)"
                        if self._solved is not None else "solving …")
            return self._res or "no solve yet"


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------
def _leash_toward(pos: np.ndarray, rot: np.ndarray, m_pos: np.ndarray, m_rot: np.ndarray,
                  eps_m: float, eps_rad: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """The belief's leash ARITHMETIC, shared by both planner families: ``(pos, rot, worst lead m,
    clamped)``.

    Each point is moved along the straight line to its measurement until the lead is at most
    ``eps_m`` (``eps_m <= 0`` puts it ON the measurement), each rotation until the geodesic deviation
    is at most ``eps_rad`` — ``KeypointClutch.sync``'s own rule, applied to the ANCHOR instead of the
    base. Inside the bound the inputs come back untouched and ``clamped`` is False, so a healthy run
    is byte-identical. One function rather than two copies because the two planners had drifted:
    the five-point one had the leash and the subset one had a docstring saying it did not need it.
    """
    pos = np.asarray(pos, dtype=np.float64).copy()
    lead = np.linalg.norm(pos - m_pos, axis=1)
    worst = float(lead.max())
    clamped = False
    if eps_m <= 0.0:
        pos = m_pos.copy()
        clamped = bool(np.any(lead > 0.0))
    else:
        over = lead > eps_m
        if np.any(over):
            scale = np.where(over, eps_m / np.maximum(lead, 1e-12), 1.0)[:, None]
            pos = m_pos + (pos - m_pos) * scale
            clamped = True
    rot, rot_clamped = _clamp_rot_batch(m_rot, rot, eps_rad)
    return pos, rot, worst, bool(clamped or rot_clamped)


def wire_frame_to_anchor(measured, readout):
    """A ``(pos, orn)`` WIRE frame (pelvis-relative, ``KeypointReadout``'s output) -> the same
    keypoints in the planner's ANCHOR frame (floor z = 0, pelvis at ``readout.pelvis_pos``); ``None``
    passes through as ``None``.

    THE BUG THIS EXISTS FOR (mock rig, robot ON the ready pose, ``rotate_box_all`` 2026-09-18): the
    very first Predict logged ``belief re-based: the commanded anchor had run 757 mm ahead of the
    measured keypoints`` — exactly ``pelvis_z``. ``seed_anchor`` gets ``resolve_deploy_anchor``'s
    WORLD poses (hands z 0.852, chest 1.151, feet 0.035) while ``FivePointWire.measured_keypoints``
    returns the wire's PELVIS frame (the same points minus ``(0, 0, pelvis_z)``: 0.095 / 0.394 /
    -0.722). Compared raw, every role of BOTH planner families reads ``pelvis_z`` of lead before
    anything has moved, and the leash then drags the whole belief 0.7 m down on the first Predict.
    The startup FK cross-check could not see it: it compares two READOUTS, both pelvis-relative.

    This inverts ``relativize_keypoints`` for the position (``p = p_local + p_pelvis``) and leaves
    the orientation alone ONLY because the readout's pelvis rotation is the identity here — the
    anchor's own contract (``resolve_deploy_anchor``: pelvis at ``(0, 0, pelvis_z)``, identity). A
    rotated readout is refused by name rather than half-converted.
    """
    if measured is None:
        return None
    rotm = getattr(readout, "pelvis_rotmat", None)
    if rotm is not None and not np.allclose(np.asarray(rotm, dtype=np.float64), np.eye(3), atol=1e-9):
        raise ValueError(
            "wire_frame_to_anchor needs a readout with an identity pelvis rotation (the anchor's "
            "contract); converting a rotated wire frame back would also have to rotate the "
            "orientations, and no planner anchor is defined in such a frame.")
    origin = np.asarray(readout.pelvis_pos, dtype=np.float64).reshape(3)
    pos = np.asarray(measured[0], dtype=np.float64).reshape(-1, 3) + origin[None, :]
    return pos.reshape(-1).copy(), np.asarray(measured[1], dtype=np.float64).reshape(-1).copy()


class FivePointPlanner(LivePlanner):
    """:class:`LivePlanner` with a pelvis-frame GEOMETRIC observation instead of ``wrt_start``.

    Inherited unchanged, because none of it is layout-specific: the per-camera image rings and
    their camera-major stacking, the age-keyed causal force binning, the gripper latch and its
    ``state_valid`` downgrade, ``obs_ages``, ``seed_anchor``/``commit_role``. Overridden:
    :meth:`snapshot_obs` and :meth:`infer_chunk`.

    ⚠️ ``push_proprio_from_anchor`` (the base class's per-frame proprio push) is inherited and
    still correct, but its ``_buf_rot6d`` ring is NOT what this state reads: the geometry needs the
    anchor's POSITIONS as well as its rotations, so :meth:`push_proprio_geom` pushes both. The base
    method is left callable rather than raising because a caller that pushes the wrong one is
    caught immediately and loudly — the geometry ring would be empty and
    :meth:`snapshot_obs` refuses on that, rather than silently reading a stale frame.
    """

    def __init__(self, loaded, obs_hz: float = 30.0) -> None:
        layout = loaded.layout
        if not layout.state_is_geometric or layout.is_wholebody:
            raise ValueError(
                f"FivePointPlanner needs a keypoint layout with a GEOMETRIC state, got "
                f"{layout.mode!r}. Serve a fivepoint_geom_* checkpoint; a wholebody one goes "
                f"through scripts/hl_serving_wholebody.py and a wrt_start one through "
                f"scripts/hl_serving.py.")
        missing = [r for r in WIRE_ROLES if r not in layout.roles]
        if missing:
            raise ValueError(
                f"layout {layout.mode!r} has no {missing} role(s); a direct 5pt wire publishes all "
                f"five of {list(WIRE_ROLES)}.")
        super().__init__(loaded, obs_hz=obs_hz)
        self._geom_blocks = tuple(layout.state_geom)
        self._geom_slots = layout.state_geom_slots()
        #: The observation ring: one ``(n_roles, 3)`` position frame per history step, beside the
        #: base class's own rotation ring. Seeded from the anchor at construction so a first
        #: Predict before any push still has the ready pose rather than zeros.
        with self._lock:
            self._buf_pos: list[np.ndarray] = [self._anchor_pos.copy() for _ in range(self._To)]
        #: True once a real target has been commanded. Until then the ring holds the SEEDED
        #: anchor — a real pose, but not one this robot was told to go to — so the state is
        #: flagged invalid and the model's own no-data embedding handles it. Same discipline as
        #: ``hl_wholebody``'s ``_t_joints is None``.
        self._geom_written = False

    # --- proprio ---------------------------------------------------------
    def seed_anchor(self, anchor_pos: np.ndarray, anchor_rot6d: np.ndarray) -> None:
        super().seed_anchor(anchor_pos, anchor_rot6d)
        with self._lock:
            self._buf_pos = [self._anchor_pos.copy() for _ in range(self._To)]
            self._geom_written = False

    def push_proprio_geom(self) -> None:
        """Append one observation frame: the anchor's POSES and the latest gripper fractions.

        The five-point twin of ``push_proprio_from_anchor``, and it must advance all three rings
        for the reason that one's docstring gives: the state at frame t is the last commanded
        target, held until a newer one arrives, so every channel is filled at the obs rate from a
        latched value. Advancing only the pose rings would leave both gripper columns at their
        ``__init__`` zeros — the CLOSED end of a legal range, which nothing downstream can notice.
        """
        with self._lock:
            self._buf_pos.append(self._anchor_pos.copy())
            self._buf_rot6d.append(self._anchor_rot6d.copy())
            self._buf_grip.append(self._grip_frac_vec())
            for buf in (self._buf_pos, self._buf_rot6d, self._buf_grip):
                while len(buf) > self._To:
                    buf.pop(0)

    def commit_role(self, role: str, pos: np.ndarray, rot6d: np.ndarray) -> None:
        super().commit_role(role, pos, rot6d)
        # A commit IS the last commanded target landing, so the geometry becomes a measurement of
        # something this robot was actually told. Set here rather than in the drip loop so the
        # flag cannot disagree with the ring.
        self._geom_written = True

    def leak_anchor_toward(self, measured, *, eps_m: float,
                           eps_rad: float) -> tuple[float, bool]:
        """Bound the BELIEF against where the robot actually is. ``(worst lead m, clamped)``.

        THE BUG THIS FIXES, measured on the rig twice. ``obs_mode=last_target`` makes this
        planner's whole observation a function of ``_anchor_pos`` / ``_anchor_rot6d``, and
        :meth:`commit_role` advances those by the model's OWN executed step. The published wire is
        NOT unbounded in the same way: ``KeypointClutch.sync`` pulls the clutch base back to within
        ``eps_m`` of the measured keypoints at every Predict. So the two ran on different leashes —
        the wire stayed near the robot while the belief walked away from both, and the next chunk
        was conditioned on a hand position the arm had never reached. Measured: ``lh_dx`` 0.605 ->
        0.876 m over sixteen chunks, past the arm's own kinematic reach, ending with the body folded
        (a balance controller asked for an unreachable hand folds the torso and knees; ``kp`` is
        still full, nothing goes limp).

        THE FIX IS THE SAME LEASH, not a new idea and not "feed measured into the observation":
        each point's belief is moved back along the straight line to its measurement until the lead
        is at most ``eps_m``, and its rotation until the geodesic deviation is at most ``eps_rad`` —
        the identical arithmetic, and by default the identical numbers, that ``sync`` already
        applies to the base. Inside the bound NOTHING MOVES, so a healthy run is byte-identical and
        ``action[:, 0] == state`` still holds exactly; the correction engages only once the command
        has diverged further than the wire itself is allowed to, i.e. once that equality has already
        stopped describing this robot.

        ⚠️ Under ``--post-anchor command`` this does not change what a GIVEN chunk publishes. The
        clutch publishes ``p_base + (p[i] - p[0])`` and every ``p[i]`` is decoded against this
        anchor, so a shift of it cancels in the difference. What it changes is the OBSERVATION, and
        therefore which chunk gets sampled next — the loop, not the arithmetic. Under
        ``--post-anchor off`` the readout is absolute and the correction reaches the wire too, which
        is the direction that mode wants anyway.

        ``measured`` is a ``(pos, orn)`` five-slot frame IN THE ANCHOR FRAME —
        ``FivePointWire.measured_keypoints`` (the wire's pelvis frame, the same readback the clutch
        bounds against, FK'd through the same ``ready_pose.KeypointFK``) passed through
        :func:`wire_frame_to_anchor`, which adds the readout pelvis back. Raw, the two frames differ
        by ``pelvis_z`` = 757 mm on every role (measured 2026-09-18) and the leash fires on a robot
        that has not moved. ``None`` (no RecordTap) is a NO-OP and reported
        as such: the anchor is left where it is rather than corrected toward a guess, which is the
        rule ``sync`` already follows for the base ("FROZEN carry"). The rings are rewritten with the
        corrected anchor so the next :meth:`snapshot_obs` sees it whatever the push timing.
        """
        if measured is None:
            return 0.0, False
        n = len(self.roles)
        m_pos = np.asarray(measured[0], dtype=np.float64).reshape(-1, 3)[:n]
        m_rot = _rotmats_from_quat_xyzw(measured[1], n)[:n]
        if m_pos.shape[0] != n:
            raise ValueError(f"measured frame has {m_pos.shape[0]} points, need {n}")
        if not (np.all(np.isfinite(m_pos)) and np.all(np.isfinite(m_rot))):
            return 0.0, False
        with self._lock:
            pos, rot, worst, clamped = _leash_toward(
                self._anchor_pos.reshape(n, 3), rot6d_to_rotmat(self._anchor_rot6d.reshape(n, 6)),
                m_pos, m_rot, eps_m, eps_rad)
            if not clamped:
                return worst, False
            self._anchor_pos = pos.reshape(self._anchor_pos.shape).copy()
            self._anchor_rot6d = rotmat_to_rot6d(rot).reshape(self._anchor_rot6d.shape).copy()
            # The observation is read off the RINGS, not off the anchor, so a correction that only
            # touched the anchor would take effect at the next 30 Hz push instead of at this
            # Predict — a one-frame race that would read as "the bound sometimes does nothing".
            self._buf_pos = [self._anchor_pos.copy() for _ in self._buf_pos]
            self._buf_rot6d = [self._anchor_rot6d.copy() for _ in self._buf_rot6d]
        return worst, True

    def geom_state(self) -> tuple[np.ndarray, bool]:
        """``(the geometric feature vector at t0, is it a commanded target?)`` for the readout."""
        with self._lock:
            pos, rot6d, written = (self._anchor_pos.copy(), self._anchor_rot6d.copy(),
                                   self._geom_written)
        return self._geom_from_anchor(pos, rot6d), bool(written)

    def _geom_from_anchor(self, pos: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
        """One frame's geometry from a ``(n_roles, 3)`` / ``(n_roles, 6)`` anchor.

        The anchor world frame has an IDENTITY pelvis rotation and an origin that differs from the
        pelvis only by ``+pelvis_z`` on z (``resolve_deploy_anchor``'s contract). ``rel_xy3`` is
        three pairwise differences, so the translation cancels; ``rpy`` reads only the rotation.
        Hence no pelvis correction — the anchor poses ARE the bodies
        :func:`wb_geom.geom_from_bodies` wants, and passing them through the same function is what
        makes the deploy observation the training one rather than a re-derivation of it.
        """
        idx = {role: i for i, role in enumerate(self.roles)}
        rot = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))
        body_pos = {body: np.asarray(pos[idx[role]], dtype=np.float64).reshape(3)
                    for body, role in GEOM_BODY_ROLE.items()}
        body_rot = {body: rot[idx[role]] for body, role in GEOM_BODY_ROLE.items()}
        return wb_geom.geom_from_bodies(body_pos, body_rot, self._geom_blocks)

    # --- observation -----------------------------------------------------
    def snapshot_obs(self) -> tuple[ObsSnapshot, np.ndarray, np.ndarray]:
        """The five-point observation: ``[rel_xy3 6?][rpy 6?][L grip][R grip]`` + images + force.

        Returns the base class's 3-tuple so :func:`batch_from_obs` and the deploy recorder keep
        working unchanged, and the two anchor arrays are the REAL running anchor here (unlike the
        wholebody planner's identity placeholder) — this layout's chunk is anchor-relative and is
        decoded against exactly them.
        """
        import time  # noqa: PLC0415 - matches the base class's per-push stamping

        t_snapshot = time.monotonic()
        with self._lock:
            pos_hist = np.stack(self._buf_pos)            # (To, n_roles, 3)
            rot_hist = np.stack(self._buf_rot6d)          # (To, n_roles, 6)
            grip_hist = np.stack(self._buf_grip)          # (To, n_roles)
            imgs = [list(ring) for ring in self._buf_img]
            raw_force = list(self._buf_force)
            t_grip = dict(self._t_grip)
            anchor_pos = self._anchor_pos.copy()
            anchor_rot6d = self._anchor_rot6d.copy()
            written = self._geom_written

        force_win, fh_valid = self._bin_force_history(raw_force)

        To = self._To
        state = np.zeros((To, self.layout.state_dim), dtype=np.float64)
        state_valid = np.ones((To, self.layout.state_dim), dtype=bool)
        feats = np.stack([self._geom_from_anchor(pos_hist[t], rot_hist[t]) for t in range(To)])
        offset = 0
        for block in self._geom_blocks:
            sl = self._geom_slots[block]
            width = sl.stop - sl.start
            state[:, sl] = feats[:, offset:offset + width]
            offset += width
        if not written:
            # The SEED is a real pose but not a target this robot was given, so every geometric
            # column is flagged rather than presented as proprioception. The gripper columns are
            # judged separately below, on their own bridge's age.
            for block in self._geom_blocks:
                state_valid[:, self._geom_slots[block]] = False
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

    # --- the family-agnostic surface the page drives -----------------------
    def push_proprio(self) -> None:
        """The page's per-tick proprio push. Here: :meth:`push_proprio_geom`."""
        self.push_proprio_geom()

    def state_readout(self) -> tuple[tuple[str, ...], np.ndarray, bool]:
        """``(column names, the state row at t0, is it a commanded target?)`` — the whole state
        vector as the model sees it, geometry then the jaws, for the page and the startup log.
        The same surface :meth:`SubsetKeypointPlanner.state_readout` offers, so the page never
        branches on the family."""
        geom, written = self.geom_state()
        with self._lock:
            grips = [float(self._grip_frac.get(r, 0.0)) for r in self.layout.gripper_roles]
        return state_names(self.layout), np.concatenate([geom, np.asarray(grips)]), bool(written)

    # --- one inference ---------------------------------------------------
    def infer_chunk(self, *, seed: int, force_enabled: bool, gate_n: float,
                    num_steps: int | None = None,
                    capture: dict | None = None) -> ChunkResult:  # type: ignore[override]
        """Sample one chunk and un-relativize all five roles against the running anchor.

        Overridden only to keep the capture payload and the decode in one place; the decode itself
        is :func:`~fuji.s1_policy.serving.hl_online.decode_chunk` verbatim, which is already
        generic over ``layout.roles`` and therefore already correct for five.
        """
        if self.m.model is None:
            raise RuntimeError(
                "this FivePointPlanner has no policy — launch with --checkpoint.")
        obs, anchor_pos, anchor_rot6d = self.snapshot_obs()
        batch = batch_from_obs(self.m, obs)
        prediction = sample_chunk(self.m, batch, seed=seed, num_steps=num_steps)
        if capture is not None:
            capture.update(
                obs=obs, anchor_pos=anchor_pos, anchor_rot6d=anchor_rot6d,
                prediction=prediction, seed=int(seed),
                num_steps=(self.m.num_steps if num_steps is None else int(num_steps)),
            )
        return decode_chunk(prediction, self.layout, anchor_pos, anchor_rot6d,
                            force_enabled=force_enabled, gate_n=gate_n)


class SubsetKeypointPlanner(LivePlanner):
    """:class:`LivePlanner` for a ``wrt_start`` keypoint layout whose roles are a SUBSET of the
    wire's five — ``left_only`` first of all.

    Nothing is overridden: the base class already builds this family's observation (per role
    ``[wrt_start_rot6d 6][gripper 1?]``, the same assembly ``scripts/hl_serving.py`` serves
    ``left_only`` with), decodes every role it has against its own anchor, and commits per role.
    What this adds is the CONSTRUCTOR check — a geometric-state or whole-body layout is refused by
    name — and the two methods the five-point page drives (:meth:`push_proprio`,
    :meth:`state_readout`), so the page never asks which family it holds. The wire slots this
    layout does not predict are the WIRE's business (:attr:`FivePointDirectWire.held`), not the
    planner's: its anchor, rings and decode are exactly ``layout.roles`` wide.
    """

    def __init__(self, loaded, obs_hz: float = 30.0) -> None:
        layout = loaded.layout
        if layout.is_wholebody:
            raise ValueError(
                f"SubsetKeypointPlanner needs a KEYPOINT layout, got {layout.mode!r} (29 joints). "
                f"Serve it with scripts/hl_serving_wholebody.py.")
        if layout.state_is_geometric:
            raise ValueError(
                f"SubsetKeypointPlanner needs a wrt_start state, got {layout.mode!r} (geometric). "
                f"That is FivePointPlanner's layout; use hl_5pt.build_planner to pick.")
        extra = [r for r in layout.roles if r not in WIRE_ROLES]
        if extra:
            raise ValueError(
                f"layout {layout.mode!r} predicts role(s) {extra} the 5pt wire has no slot for "
                f"({list(WIRE_ROLES)}).")
        super().__init__(loaded, obs_hz=obs_hz)

    def push_proprio(self) -> None:
        """The page's per-tick proprio push. Here: the base class's rotation + jaw rings."""
        self.push_proprio_from_anchor()

    def leak_anchor_toward(self, measured, *, eps_m: float,
                           eps_rad: float) -> tuple[float, bool]:
        """The belief's leash, on the wire slots this layout PREDICTS. ``(worst lead m, clamped)``.

        THE SAME BUG AS THE FIVE-POINT ONE, found on the rig a day later with a ``left_only``
        checkpoint (``wipe_board_new`` h50, ⏺ run ``2026-09-14_14-16-01``): over 14 s the published
        LEFT rotation walked to 67° from the start while the robot's reached 47°, the position lead
        grew 67 → 130 mm and the published z sank 94 mm with the measured z unmoved — and the state
        read ``left_r6d0`` 0.625 → 0.065, 1.65 band widths past q01. This family's observation is
        ``R_start^T R_anchor`` and :meth:`commit_role` advances the anchor by the model's OWN executed
        step with nothing pulling it back, exactly the asymmetry
        :meth:`FivePointPlanner.leak_anchor_toward` documents — while this class's docstring said a
        ``wrt_start`` state "does not have this failure mode". It does: it has no POSITION columns,
        but its ROTATION is the anchor's, and that is what ran.

        Same arithmetic (:func:`_leash_toward`, the clutch's own rule). The only difference is
        INDEXING: ``measured`` is the whole five-slot wire frame (``FivePointWire.measured_keypoints``,
        converted into the ANCHOR frame by :func:`wire_frame_to_anchor` — the two differ by
        ``pelvis_z`` and the leash must compare in one frame) and this planner's roles are a subset of
        it, so each role reads its own ``WIRE_ROLES`` slot;
        the held slots are the wire's business and are never touched here. ``None`` or a non-finite
        readback is a NO-OP (the anchor is HELD, never corrected toward a guess). The rotation ring is
        rewritten so the next :meth:`snapshot_obs` reads the corrected anchor at THIS Predict; there is
        no position ring to rewrite because this state never reads a position.

        ⚠️ Under ``--post-anchor command`` a given chunk's published motion is unchanged (the clutch's
        ``p[i] - p[0]`` cancels an anchor shift); what changes is the observation the NEXT chunk is
        sampled from — which is where the runaway lived.
        """
        if measured is None:
            return 0.0, False
        n = len(self.roles)
        slots = [WIRE_ROLES.index(r) for r in self.roles]
        m_pos_all = np.asarray(measured[0], dtype=np.float64).reshape(-1, 3)
        if m_pos_all.shape[0] < max(slots) + 1:
            raise ValueError(
                f"measured frame has {m_pos_all.shape[0]} points, need wire slot(s) {slots}")
        m_rot_all = _rotmats_from_quat_xyzw(measured[1], m_pos_all.shape[0])
        m_pos, m_rot = m_pos_all[slots], m_rot_all[slots]
        if not (np.all(np.isfinite(m_pos)) and np.all(np.isfinite(m_rot))):
            return 0.0, False
        with self._lock:
            pos, rot, worst, clamped = _leash_toward(
                self._anchor_pos.reshape(n, 3), rot6d_to_rotmat(self._anchor_rot6d.reshape(n, 6)),
                m_pos, m_rot, eps_m, eps_rad)
            if not clamped:
                return worst, False
            self._anchor_pos = pos.reshape(self._anchor_pos.shape).copy()
            self._anchor_rot6d = rotmat_to_rot6d(rot).reshape(self._anchor_rot6d.shape).copy()
            self._buf_rot6d = [self._anchor_rot6d.copy() for _ in self._buf_rot6d]
        return worst, True

    def state_readout(self) -> tuple[tuple[str, ...], np.ndarray, bool]:
        """``(column names, the state row at t0, True)``.

        The row is re-derived from the anchor exactly as :meth:`LivePlanner.snapshot_obs` derives
        the newest history frame — ``R_start^T R_now`` per role, then the jaw fraction — so the
        readout and the model's input cannot disagree. The flag is always True: before anything is
        commanded ``R_now == R_start`` and the channel reads the IDENTITY, which is what every
        training episode's first frame looks like — a real, in-distribution value and not a seed
        to flag (unlike the geometric family's un-commanded ready pose).
        """
        with self._lock:
            rot = self._anchor_rot6d.copy()
            start = self._start_rot6d.copy()
            grips = self._grip_frac_vec()
        slots = self.layout.state_slots()
        vec = np.zeros(self.layout.state_dim, dtype=np.float64)
        for ri, role in enumerate(self.roles):
            entry = slots[role]
            vec[entry["wrt_start_rot6d"]] = rot6d_relative(rot[ri], start[ri])
            if "gripper" in entry:
                vec[entry["gripper"]] = grips[ri]
        return state_names(self.layout), vec, True


#: Suffix ``state_names`` gives every JAW column. Read back by
#: :meth:`StateRangeReport.worst_body_excursion` to leave the jaws out of the ``--ood-stop``
#: decision, so the two cannot drift apart.
GRIP_COLUMN_SUFFIX = "_grip"


@dataclass(frozen=True)
class StateRangeReport:
    """Where ONE observation row sits against the checkpoint's own training range.

    ``outside`` is ``(column, value, q01, q99, excess)`` per column that lies outside the
    ``[q01, q99]`` band ``norm_stats.json`` maps onto ``[-1, 1]``; ``excess`` is how far past the
    nearer edge, in units of the band's width (0.5 = half a band past it). ``wrapping`` is
    ``(column, |angle| in degrees)`` for every Euler column past ``wb_geom.RPY_WRAP_WARN_RAD``.
    ``checked`` is False when the checkpoint bundled no state statistics, so "nothing outside"
    cannot be read as "everything inside".
    """

    outside: tuple[tuple[str, float, float, float, float], ...]
    wrapping: tuple[tuple[str, float], ...]
    checked: bool
    n_columns: int

    @property
    def ok(self) -> bool:
        return self.checked and not self.outside and not self.wrapping

    def worst_body_excursion(self) -> tuple[str, float]:
        """The worst out-of-band BODY column and how far past its band it sits (band widths).

        The channel a runaway shows up in. ``("", 0.0)`` when every body column is inside — and
        ``checked=False`` also reads as 0.0, because "no statistics" is not evidence of divergence.

        THE JAWS ARE EXCLUDED, deliberately, and only from THIS number — they stay in
        :meth:`describe`'s report. ``<role>_grip`` is a commanded WIDTH: a jaw resting at the open
        stop lands on its training ``q99`` to float precision (the reason ``edge_tol`` exists at
        all), and the same value sits fractionally outside a narrower band on a checkpoint whose
        demonstrations never held it fully open. So a jaw can be marginally out of band on a
        perfectly healthy run, while a jaw cannot be the channel a diverging BODY appears in — the
        state's geometry blocks are. Gating a rollout on it would stop good runs and catch nothing.
        """
        worst_name, worst = "", 0.0
        if not self.checked:               # no statistics is not evidence of divergence
            return worst_name, worst
        for name, _value, _lo, _hi, excess in self.outside:
            if name.endswith(GRIP_COLUMN_SUFFIX):
                continue
            if float(excess) > worst:
                worst_name, worst = str(name), float(excess)
        return worst_name, worst

    def describe(self) -> str:
        if not self.checked:
            return "range unchecked (the checkpoint bundled no state norm stats)"
        parts: list[str] = []
        if self.outside:
            parts.append("OUTSIDE the training range: " + ", ".join(
                f"{name} {value:+.3f} (q01 {lo:+.3f} · q99 {hi:+.3f}, {excess:.2f} band past)"
                for name, value, lo, hi, excess in self.outside))
        if self.wrapping:
            parts.append("near the ±180° wrap: " + ", ".join(
                f"{name} {deg:.0f}°" for name, deg in self.wrapping))
        if not parts:
            return f"all {self.n_columns} state columns inside the training [q01, q99] range"
        return " · ".join(parts)


def state_range_report(names, state_row, stats, *, ignore=(), edge_tol: float = 0.01,
                       rpy_warn_rad: float = wb_geom.RPY_WRAP_WARN_RAD) -> StateRangeReport:
    """Score one observation row against the checkpoint's ``norm_stats['state']``.

    WHY THIS EXISTS. The first observation of every run is the READY POSE's geometry, and whether
    the policy has ever seen it is a property of the DATASET's placement (``attrs.wb_cond_mode``),
    not of the wire: ``absolute --dz-anchor hand`` files start each take where the operator stood,
    ``home`` files re-base each take onto the G1's 29-joints-zero pose. Measured with
    ``scripts/measure_5pt_state_gap.py`` on push_gantry (``home``) every ready-pose column is
    inside ``[q01, q99]``; a task whose operator held the hands elsewhere would not be, and the
    only symptom would be a plausible chunk from an extrapolated conditional — DDIM's
    ``clip_sample`` clamps the ACTION to the band, nothing clamps or reports the STATE. So the
    band is read off the checkpoint itself (every checkpoint bundles its ``norm_stats.json``),
    which makes the check hold for any task's checkpoint with nothing to remember per task.

    ``stats`` is the ``NormStats`` for ``"state"`` (``q01`` / ``q99`` arrays) or ``None``.
    ``ignore`` names columns to leave out of the band check — the page passes the two ``*_grip``
    columns at STARTUP, before any jaw bridge has reported, because until then they read 0.0 (the
    closed end of the range) and would flag every checkpoint whose training jaws were open; once the
    bridges are up the live readout checks all columns. Columns whose training band is degenerate (``q99 - q01 < 0.02``, the normalizer's own floor — a jaw
    that never moved, a difference that is constant by construction) are judged against that
    floor rather than against a zero-width band, exactly as ``normalize_quantile`` scales them.
    ``edge_tol`` (band widths, default 0.01 = 1%) is how far past q01/q99 a value may sit and still
    count as inside: q99 is where the training jaws READ when open, so a live jaw at the same
    stop lands on it to float precision and a strict comparison flagged every Predict with
    "left_grip +0.991 (q99 +0.991, 0.00 band past)" on the first rig run — a message that reads
    as extrapolation for a value the policy saw on every frame.
    """
    names = tuple(str(n) for n in names)
    row = np.asarray(state_row, dtype=np.float64).reshape(-1)
    if row.size != len(names):
        raise ValueError(f"{len(names)} column names for a {row.size}-wide state row")
    wrapping: list[tuple[str, float]] = []
    for name, value in zip(names, row):
        if name.endswith(("_roll", "_pitch", "_yaw")) and abs(float(value)) > float(rpy_warn_rad):
            wrapping.append((name, float(np.degrees(abs(value)))))
    if stats is None:
        return StateRangeReport((), tuple(wrapping), False, len(names) - len(set(ignore) & set(names)))
    q01 = np.asarray(stats.q01, dtype=np.float64).reshape(-1)
    q99 = np.asarray(stats.q99, dtype=np.float64).reshape(-1)
    if q01.size != row.size or q99.size != row.size:
        raise ValueError(f"norm stats are {q01.size}-wide but the state row is {row.size}-wide "
                         f"({len(names)} names) — this is not this checkpoint's norm_stats.json")
    from fuji.s1_policy.data.norm_stats import _QUANTILE_MIN_RANGE  # noqa: PLC0415

    width = np.maximum(q99 - q01, _QUANTILE_MIN_RANGE)
    skip = {str(n) for n in ignore}
    outside: list[tuple[str, float, float, float, float]] = []
    for j, name in enumerate(names):
        if name in skip:
            continue
        v = float(row[j])
        tol = float(edge_tol) * float(width[j])
        if v < q01[j] - tol:
            outside.append((name, v, float(q01[j]), float(q99[j]), float((q01[j] - v) / width[j])))
        elif v > q99[j] + tol:
            outside.append((name, v, float(q01[j]), float(q99[j]), float((v - q99[j]) / width[j])))
    return StateRangeReport(tuple(outside), tuple(wrapping), True, len(names) - len(skip & set(names)))


def state_names(layout: ActionLayout) -> tuple[str, ...]:
    """Column names of ONE state frame, in ``layout.state_slots()`` order — for the page.

    Geometric family: the block names then ``<role>_grip`` per jaw. ``wrt_start`` family: per
    role in keypoint order, ``<role>_r6d0..5`` then ``<role>_grip`` if it has a jaw.
    """
    names: list[str] = []
    if layout.state_is_geometric:
        for block in layout.state_geom:
            names.extend(wb_geom.GEOM_BLOCK_NAMES[block])
        names.extend(f"{role}{GRIP_COLUMN_SUFFIX}" for role in layout.gripper_roles)
        return tuple(names)
    for role in layout.roles:
        names.extend(f"{role}_r6d{i}" for i in range(6))
        if role in layout.gripper_roles:
            names.append(f"{role}{GRIP_COLUMN_SUFFIX}")
    return tuple(names)


def check_layout(layout: ActionLayout) -> str:
    """Which family of :data:`LAYOUT_FAMILIES` serves ``layout`` on the 5pt wire, or ``ValueError``.

    ``"fivepoint"`` — a ``fivepoint_geom_*`` checkpoint: five roles, geometric state, the policy
    predicts every slot. ``"keypoint_subset"`` — a ``wrt_start`` keypoint layout whose roles are a
    subset of the five (``left_only``, ``both_hands``, ``both_hands_head[_dual]``); the rest are
    HELD at the ready pose. Refused by name: a whole-body layout (no keypoints to publish), the
    legacy layout, a role the wire has no slot for, and a geometric-state layout with fewer than
    five roles (no such mode exists; the geometric planner reads three bodies and would need all
    five slots predicted).
    """
    if layout.is_wholebody:
        raise ValueError(
            f"layout {layout.mode!r} predicts 29 JOINT angles, not keypoints. Serve it with "
            f"scripts/hl_serving_wholebody.py.")
    if getattr(layout, "legacy", False):
        raise ValueError(f"layout {layout.mode!r} is the frozen pre-v2 layout and is not served.")
    extra = [r for r in layout.roles if r not in WIRE_ROLES]
    if extra:
        raise ValueError(
            f"layout {layout.mode!r} predicts role(s) {extra} the 5pt wire has no slot for; the "
            f"wire is {list(WIRE_ROLES)}.")
    if layout.state_is_geometric:
        missing = [r for r in WIRE_ROLES if r not in layout.roles]
        if missing:
            raise ValueError(
                f"layout {layout.mode!r} has a geometric state but no {missing} role(s); the "
                f"geometric planner needs all five of {list(WIRE_ROLES)} predicted.")
        return "fivepoint"
    return "keypoint_subset"


def build_planner(loaded, obs_hz: float = 30.0):
    """The planner for ``loaded.layout``: :class:`FivePointPlanner` for the geometric family,
    :class:`SubsetKeypointPlanner` for a ``wrt_start`` subset. Refuses everything else by name
    (:func:`check_layout`)."""
    family = check_layout(loaded.layout)
    if family == "fivepoint":
        return FivePointPlanner(loaded, obs_hz=obs_hz)
    return SubsetKeypointPlanner(loaded, obs_hz=obs_hz)


def validate_layout(layout: ActionLayout) -> str:
    """Refuse a layout this planner cannot serve, by name and with the alternative; return the
    family (:data:`LAYOUT_FAMILIES`) otherwise. ``SystemExit`` so a launcher's mistake is one
    line, not a traceback."""
    try:
        return check_layout(layout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


__all__ = [
    "CARRY_MODES",
    "GEOM_BODY_ROLE",
    "IMPLAUSIBLE_SPEED_MS",
    "LOW_LEVEL_HZ",
    "LOW_LEVEL_SLEW_M_PER_TICK",
    "SLEW_BUDGET_PER_STEP_M",
    "SLEW_IS_SHARED_POOL",
    "WIRE_ROLES",
    "FORCE_HANDS",
    "LAYOUT_FAMILIES",
    "ChunkSpan",
    "FivePointDirectWire",
    "FivePointPlanner",
    "ManualJawOverride",
    "PreviewGate",
    "SlewBudget",
    "SlewMonitor",
    "GRIP_COLUMN_SUFFIX",
    "StateRangeReport",
    "SubsetKeypointPlanner",
    "build_clutch",
    "build_planner",
    "check_layout",
    "chunk_span",
    "fill_held_roles",
    "force_command",
    "held_anchor_poses",
    "held_roles",
    "state_names",
    "state_range_report",
    "validate_layout",
    "wire_frame_to_anchor",
]
