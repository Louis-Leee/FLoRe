"""A WALK SKILL for the five-point wire: a reference clip's two FEET under the planner's own upper body.

``scripts/hl_serving_5pt.py`` publishes five pelvis-frame keypoints — ``[L hand, R hand, torso,
L foot, R foot]`` — and the ``5pt`` low level is a whole-body BALANCE controller that reads all
five. The policy predicts the feet as a STANCE (`docs/RUNBOOK_diffusion_umi_5pt.md`: the foot
channel is pelvis-frame with no root pose, so a demonstration where the operator walks produces no
walking foot target). This module supplies what the policy cannot: a GAIT for the two foot slots,
taken from a holosoma reference clip (``bones_seed_g1_holosoma_f32``, the same NPZ family the low
level was trained to track), while the three UPPER slots keep whatever the wire was last commanded.

THE FRAME IS THE LOW LEVEL'S OWN. ``holosoma_wbt._compute_vr_fk`` expresses every keypoint in the
reference ROOT's body frame — the full pelvis rotation, not a heading frame — and
``bake_motions._fk_into`` reproduces it for the browser. :func:`extract_feet_local` does the same
arithmetic on the clip's stored ``body_pos_w`` / ``body_quat_w``::

    p_local = R_pelvis(t)^T (p_ankle(t) - p_pelvis(t))       q_local = q_pelvis(t)^-1 (x) q_ankle(t)

with the ankle at the bare ``*_ankle_roll_link`` ORIGIN (no lever — holosoma's
``VR_5POINT_OFFSETS_TCP`` is zero there, and so is ``ready_pose.KeypointFK``'s) and the quaternion
canonicalised to ``w >= 0`` (the trained branch; see ``hl_dense._quat_xyzw_scalar_positive``).
Cross-checked against mjswan's own baked ``vr5_pos_tcp_local`` for the walk-with-box clip:
**0.0013 mm** on position, 6e-7 on the quaternion. So a frame published from here is, slot for
slot, what that clip's own tracking obs would have been — which is the only sense in which "play
the clip's feet" is a well-defined command on this wire.

⚠️ THE FEET GO ON THE WIRE ABSOLUTELY, NOT THROUGH THE PER-POINT CLUTCH — and that is measured, not a
preference. ``KeypointClutch.apply`` re-bases each point onto its OWN base, so two feet would each
get a different offset (``base_i - clip_i[0]``). The walk-with-box clip's frame 0 is MID-STRIDE: in
its pelvis frame the left ankle sits at ``(+0.070, +0.040, -0.713)`` and the right at
``(-0.242, -0.124, -0.680)``, 31 cm apart fore-aft, while the ready pose's ankles are
``(-0.0014, ±0.1185, -0.7217)``. Per-point re-basing would therefore shift the RIGHT foot's whole
trajectory 24 cm forward relative to the LEFT's for the entire walk — a gait with the inter-foot
geometry destroyed, which is the one thing a balance controller reads from the feet. Instead
:class:`WalkPlayer` RAMPS from the wire's current feet to the clip's frame 0 (position lerp +
quaternion slerp, rate-bounded), plays the clip's own pelvis-frame feet verbatim, and ramps back to
where the feet were when asked to stop. The clutch's BASE still follows every published frame
(the page calls ``wire.commit``), so a chunk predicted afterwards continues from where the walk
left the feet rather than jumping.

⚠️ THE LOW LEVEL'S SLEW CAPS DO NOT COVER THE FEET — BY A LAUNCHER DECISION THIS PAGE CAN ONLY READ.
``3_run_policy_vive_force.sh`` pins ``HOLOSOMA_VIVE_MAX_POS_STEP=0.005`` (one L2 over the covered
position entries per 50 Hz tick) and ``HOLOSOMA_VIVE_MAX_ORN_STEP=0.02`` (one L2 over the covered
quaternion entries) as INLINE prefixes on its ``run_policy`` call — an exported env var cannot
override them. The walk-with-box feet demand a mean of **46 mm/tick** (p95 67, peak 95) and up to
**0.32** of quaternion L2 per tick — 9x and 16x those caps — and under them a walk is not slow, it
is GONE: the feet crawl at 5 mm/tick toward a target that keeps running away (measured on a mock
low level at the cap: 287 mm mean / 601 mm max lag). So the caps now cover only the three UPPER
points: ``HOLOSOMA_VIVE_SLEW_FEET=0`` (``SLEW_FEET`` on the launcher, its default) makes
``holosoma_wbt._slew_keypoints`` slew the leading ``3x3`` / ``3x4`` entries as one pool and copy the
two feet from the target verbatim, and the low level says which rows it covered on every RecordTap
frame (``vr_cmd_slewed``, a per-point mask). ``hl_5pt.SlewMonitor`` measures the cap over the
covered rows only and exposes the mask, so :meth:`WalkStats.describe` can print "feet EXEMPT" as a
fact READ OFF THE ROBOT rather than assumed — and, if a low level without the knob is on the other
end (``SLEW_FEET=1``, or a build from before it existed), the clamp ratios come back. Nothing here
relaxes the caps on the hands or the torso: those remain the only continuous rate bound that exists
while armed (`docs/RUNBOOK_hl_serving_5pt.md` §5, "Why the cap must not simply be raised"), and the
exemption is worth exactly what it costs — a stale→fresh bridge recovery is no longer ramped on the
feet (arming is still zero-jump: the arm clutch re-bases all five points).

WHAT ELSE IS NAMED RATHER THAN FIXED. (1) The low level pins the robot's HEADING to its frozen
reference clip (``motion_ref_ori_b`` — none of the 5pt/dense/wbt presets carries
``motion_ref_pos_b``, so trackers steer POSTURE only). The bundled clip is a straight walk with the
sway of carrying a box (source heading net +17°, range 43° over 6.4 s; the pelvis covers 8.1 m of
path), so what the robot's heading does under a pinned reference is part of the experiment. (2)
The upper three points are held CONSTANT IN THE PELVIS FRAME, which is the only meaning "unchanged"
has on this wire — the hands travel with the body, and "the feet go back to where they were" on
the second press means back to the same PELVIS-frame stance, not to the spot on the floor where the
walk began (the wire has no world position; the robot stays where it walked to). (3) The arms
swinging with the gait in every training clip are NOT reproduced; whether the balance controller
walks with a static upper body is exactly the experiment this button exists to run.

THE LEGS RIDE WITH THE FEET (2026-09-07). The first robot run stepped IN PLACE: the two foot
keypoints said "swing" while the low level's ``command_lower_body`` obs term — the REFERENCE
motion's 12 leg joints + velocities, 24 of the 184 per-frame dims — was still read from its
FROZEN idle clip and said "stand" (knee 38 deg, hip -18 deg, zero velocity). In mjswan, where the
same policy walks these clips, that term comes from the clip itself (``bake_motions``'
``lower_body_jp/jv``). So a :class:`WalkClip` also carries the clip's 29 joint targets + velocities
BY NAME (``joint_pos`` / ``joint_vel`` / ``joint_names``, from the NPZ's own ``joint_pos[:, 7:]`` /
``joint_vel[:, 6:]``), and :class:`WalkPlayer` emits them beside the feet (:attr:`WalkPlayer.legs`):
the clip's own values in the clip phase (velocity scaled by the playback speed, as a resampled
motion's is), and during every ramp the lerp between the low level's READY legs and the clip's
frame-0 legs with the lerp's own derivative as the velocity — a pos/vel pair the policy can
believe. The page puts them on the wire as the 5pt frame's optional ``ref_joints``
(``S0Driver.send_5pt``), which the low level substitutes into ``command_lower_body`` for 5pt only
and while ARMED (``holosoma_wbt._resolve_ref_joints``; ``HOLOSOMA_VIVE_REF_JOINTS=0`` ignores it),
never slews (a slewed reference is the crawl the feet were exempted from) and reports on RecordTap
as ``ref_joints_live``. A frame without the field is the byte-identical frozen-clip path, which is
what every non-walk frame is. The joints are placed by NAME on both ends; a clip whose joint
channel is unusable (missing / duplicate names, wrong width) loads FEET-ONLY with ``legs_note`` set,
exactly the pre-2026-09-07 behaviour.

THE LEG OVERRIDE (2026-09-12): THE SAME FEET AND LEGS UNDER A RUNNING PLANNER. The walk above
STOPS the policy (it holds ``busy`` and ``dripping`` and freezes the three upper slots) — the rig
recipe is ``p`` (pause) → ``k`` (walk) → ``k`` → ``p`` (resume). :class:`LegOverride` is the other
composition the user asked for: the planner keeps rolling out (Predict / ▶▶ sequential / ▶▶ stream /
the heartbeat, untouched), and only the two FOOT slots of every frame it publishes are REPLACED by the
clip's feet — the legs (``ref_joints``) riding along — at the page's single publish seam (``_send``),
so nothing else in the page knows. It is a SUBSTITUTION, not a publisher: a 50 Hz clock advances the
SAME :class:`WalkPlayer` (lead-in from the wire's current feet → the clip → loop), and whoever
publishes next (the drip at 60 Hz, the stream, the heartbeat) reads the current feet + legs off it.
Two things the composition needed that the stop-the-policy walk did not:

* **The hand-back target MOVES.** ``k``'s lead-out ramps to a snapshot (the stance the walk started
  from). Under a running planner the policy's own foot target keeps being committed, so the lead-out
  is a PURSUIT (:class:`_Pursuit`): each tick moves toward the planner's LIVE foot target by at most
  the same 5 mm / 0.9° bound and lands on it exactly (the last step is within the bound by
  construction), so the moment the override lets go the wire's feet ARE the planner's and the next
  chunk step is continuous. ``WalkPlayer(home_provider=...)`` selects it; without the provider the
  lead-out is the reviewed fixed ramp, byte-identical.
* **The clutch must not chase the gait.** With ``--carry all`` the leak bound pulls every carried
  point's base to within ``eps_m`` of the MEASURED keypoint at each Predict — and the measured feet
  are WALKING, so the planner's foot base would be dragged along the gait and the hand-back would
  land in a mid-stride split stance. ``KeypointClutch.sync(hold_slots=...)`` skips the two foot
  slots while the wire has them (``FivePointDirectWire.externally_driven``); the planner's stance
  target keeps advancing only by its own committed deltas and is what the pursuit returns to.

STOP freezes the gait where it is (the frozen legs' velocity is zeroed — a held reference is at
rest) and the override stays ENGAGED; ▶▶ / Run resumes it with the planner; the toggle hands the
legs back (the pursuit runs even under a latched STOP — it is the operator's explicit, rate-bounded
request). It refuses to ENGAGE while the low level reports UN-ARMED and hands back on its own when
the low level DISARMS mid-gait, both for one reason: holosoma's absolute arm clutch snapshots
``held - wire`` PER POINT at arm, so arming into a mid-stride wire gives each foot a constant offset
for the rest of the session — the inter-foot geometry error the fixed ramp above exists to avoid.

THE CLIP IS BUNDLED. ``assets/walk_with_box_feet.npz`` is the extraction above written out once by
``scripts/extract_walk_feet.py`` (the two feet + the 29 joints, ~190 KB, float64, with the
source's name and sha256), so the page needs no ``--walk-dir`` and no copy of the 546 KB source
NPZ; the same :func:`load_walk_clip` still reads a raw holosoma NPZ, and the extraction is pinned
equal to the bundle by a test whenever the source is present.

Pure numpy + scipy: no model, no URDF, no robot — every claim above is testable on a CPU
(``tests/test_walk_skill.py``).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from flore.s1_policy.serving.hl_dense import FIVE_POINT_BODIES, FIVE_POINT_GATED, FIVE_POINT_LABELS

#: The two wire slots this skill drives, in slot order, and the URDF/NPZ bodies that carry them.
#: DERIVED from the wire's own table rather than re-typed: if holosoma's ``VR_5POINT_BODIES``
#: changes, the feet move with it here too.
FOOT_BODIES: tuple[str, ...] = FIVE_POINT_BODIES[FIVE_POINT_GATED:]
FOOT_LABELS: tuple[str, ...] = FIVE_POINT_LABELS[FIVE_POINT_GATED:]
#: The slots held at the wire's last command: the three GATED, operator/policy-placed points.
N_UPPER = FIVE_POINT_GATED
N_FEET = len(FOOT_BODIES)
N_POINTS = len(FIVE_POINT_BODIES)
#: The clip's root body. holosoma's motion NPZs store the pelvis as body 0 AND name it, and the
#: name is what is read — an index would silently pick the wrong body on a re-ordered export.
PELVIS_BODY = "pelvis"

#: The clip the user asked for by name — the SOURCE of the bundled feet.
DEFAULT_CLIP = "AMASS_g1_GMR8_ACCAD_Male2Walking_c3d_B20_-__Walk_with_box_stageii.npz"
#: Its two feet, extracted once (``scripts/extract_walk_feet.py``) and shipped with the package, so
#: the page has the skill with no flag and no external file. Read with :func:`load_walk_clip`.
BUNDLED_CLIP = Path(__file__).resolve().with_name("assets") / "walk_with_box_feet.npz"
#: Schema tag of the extracted format (`save_feet_clip`). Refused by name on a mismatch. The joint
#: arrays were ADDED under the same tag (2026-09-07) as OPTIONAL keys: a bundle without them loads
#: feet-only, and an older reader ignores them.
FEET_SCHEMA = "flore.walk_feet.v1"
#: The G1's actuated joints — the width of a clip's joint channel. A holosoma motion NPZ stores
#: ``joint_pos`` as ``[root xyz 3, root quat 4, joints 29]`` and ``joint_vel`` as ``[6, 29]``
#: (``holosoma_wbt._get_ext_motion_at_frame`` reads the same slices).
N_JOINTS = 29

#: The low level's ORIENTATION cap per 50 Hz tick — ``HOLOSOMA_VIVE_MAX_ORN_STEP``, one L2 over the
#: ``(20,)`` xyzw vector — as pinned INLINE by ``3_run_policy_vive_force.sh`` (NOT the submodule's
#: 0.05 code default). A pre-flight estimate only, and unlike the position cap it is not yet
#: measurable off RecordTap (the record frame carries no ``vr_cmd_orn``); ``hl_5pt`` has no
#: constant for it because a chunk's rotation demand was never the binding one. A walk's is.
LAUNCHER_ORN_CAP_PER_TICK = 0.02
#: The low level's tick rate. Same value as ``hl_5pt.LOW_LEVEL_HZ``; spelled here so this module
#: imports without ``hl_online`` (torch). Pinned equal by a test.
LOW_LEVEL_HZ = 50.0

#: Speed multiplier bounds for the page's slider and :meth:`WalkPlayer.set_speed`.
SPEED_MIN, SPEED_MAX = 0.05, 3.0


# ---------------------------------------------------------------------------
# quaternion helpers (xyzw, the wire's representation)
# ---------------------------------------------------------------------------
def canonical_xyzw(q: np.ndarray) -> np.ndarray:
    """``(..., 4)`` xyzw -> the same rotations with the SCALAR (last) component ``>= 0``.

    The wire's trained branch. Flipping on the wrong index is a silent 180 deg error, which is why
    this is a named function rather than a bare ``[..., 3]``."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., 3:4] < 0.0, -q, q)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [1, 2, 3, 0]]


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path slerp between two ``(..., 4)`` xyzw quaternion arrays at ``t`` in ``[0, 1]``.

    ``q1`` is flipped into ``q0``'s hemisphere first, so the path never crosses the double cover
    (the low level's own slew does the same before ITS lerp — ``_align_quat_signs``). The result is
    unit-norm and canonicalised to ``w >= 0``."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    t = float(np.clip(t, 0.0, 1.0))
    # Near-parallel: linear + renormalise (the sin(theta) denominator is ill-conditioned).
    lin = a + (b - a) * t
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe = sin_theta > 1e-6
    w0 = np.where(safe, np.sin((1.0 - t) * theta) / np.where(safe, sin_theta, 1.0), 1.0 - t)
    w1 = np.where(safe, np.sin(t * theta) / np.where(safe, sin_theta, 1.0), t)
    out = np.where(dot > 0.9995, lin, w0 * a + w1 * b)
    out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
    return canonical_xyzw(out)


def quat_angle_rad(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Geodesic angle between two ``(..., 4)`` xyzw arrays, per row."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


# ---------------------------------------------------------------------------
# the clip
# ---------------------------------------------------------------------------
def extract_feet_local(body_pos_w: np.ndarray, body_quat_wxyz: np.ndarray,
                       body_names) -> tuple[np.ndarray, np.ndarray]:
    """World body poses -> the two ankles in the PELVIS body frame, xyzw with ``w >= 0``.

    Args:
      body_pos_w: ``(T, n_bodies, 3)`` world positions of the link frames.
      body_quat_wxyz: ``(T, n_bodies, 4)`` world orientations, WXYZ (holosoma's NPZ convention).
      body_names: ``n_bodies`` names; must contain :data:`PELVIS_BODY` and both :data:`FOOT_BODIES`.
    Returns:
      ``(pos (T, 2, 3), quat (T, 2, 4))`` in :data:`FOOT_BODIES` order — exactly
      ``bake_motions._fk_into``'s arithmetic for the ``vr5`` foot columns (verified to 0.0013 mm).
    """
    names = [str(n) for n in body_names]
    missing = [b for b in (PELVIS_BODY, *FOOT_BODIES) if b not in names]
    if missing:
        raise ValueError(
            f"the clip has no body {missing}; a 5pt foot command needs {PELVIS_BODY!r} and both of "
            f"{list(FOOT_BODIES)} (got {len(names)} bodies: {names[:6]}...)")
    pos = np.asarray(body_pos_w, dtype=np.float64)
    quat = np.asarray(body_quat_wxyz, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3 or quat.shape != pos.shape[:2] + (4,):
        raise ValueError(f"body_pos_w must be (T, n, 3) and body_quat_w (T, n, 4); got {pos.shape} / "
                         f"{quat.shape}")
    ip = names.index(PELVIS_BODY)
    r_pelvis_inv = Rotation.from_quat(wxyz_to_xyzw(quat[:, ip])).inv()
    out_pos = np.empty((pos.shape[0], N_FEET, 3), dtype=np.float64)
    out_quat = np.empty((pos.shape[0], N_FEET, 4), dtype=np.float64)
    for k, body in enumerate(FOOT_BODIES):
        ib = names.index(body)
        out_pos[:, k] = r_pelvis_inv.apply(pos[:, ib] - pos[:, ip])
        out_quat[:, k] = canonical_xyzw((r_pelvis_inv * Rotation.from_quat(wxyz_to_xyzw(quat[:, ib]))
                                         ).as_quat())
    if not (np.all(np.isfinite(out_pos)) and np.all(np.isfinite(out_quat))):
        raise ValueError("the clip's foot poses are non-finite after relativisation")
    return out_pos, out_quat


def extract_joints(data, name: str = "clip") -> tuple[tuple[str, ...], np.ndarray | None,
                                                     np.ndarray | None, str]:
    """A holosoma motion NPZ's 29 joint targets + velocities BY NAME, or feet-only with a reason.

    Returns ``(joint_names, joint_pos (T, 29), joint_vel (T, 29), note)``; on any inconsistency
    ``((), None, None, why)`` — the clip still plays its FEET (the behaviour before the legs
    existed), and the page prints ``why``. Never raises for a missing channel: a clip's joints are
    a second, optional product of the same file, and a wrong joint ORDER sent to a robot is worse
    than no joints, which is why the names are the gate.
    """
    keys = set(data.files)
    if "joint_names" not in keys or "joint_pos" not in keys:
        return (), None, None, f"{name}: no joint_names/joint_pos channel — feet only"
    names = tuple(str(n) for n in np.asarray(data["joint_names"]).reshape(-1))
    if len(names) != N_JOINTS or len(set(names)) != N_JOINTS:
        return (), None, None, (f"{name}: joint_names is not {N_JOINTS} distinct names "
                                f"({len(names)} given, {len(set(names))} distinct) — feet only")
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] not in (N_JOINTS, N_JOINTS + 7):
        return (), None, None, f"{name}: joint_pos {q.shape} is neither (T, 29) nor (T, 36) — feet only"
    q = q[:, 7:] if q.shape[1] == N_JOINTS + 7 else q
    if "joint_vel" in keys:
        qd = np.asarray(data["joint_vel"], dtype=np.float64)
        if qd.ndim != 2 or qd.shape[1] not in (N_JOINTS, N_JOINTS + 6) or qd.shape[0] != q.shape[0]:
            return (), None, None, f"{name}: joint_vel {qd.shape} does not pair with joint_pos {q.shape} — feet only"
        qd = qd[:, 6:] if qd.shape[1] == N_JOINTS + 6 else qd
    else:
        qd = None
    if not np.all(np.isfinite(q)) or (qd is not None and not np.all(np.isfinite(qd))):
        return (), None, None, f"{name}: non-finite joint data — feet only"
    if qd is None:
        # No stored velocity: a central difference at the clip's own rate (the bake does the same).
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in keys else LOW_LEVEL_HZ
        qd = np.gradient(q, 1.0 / fps, axis=0)
    return names, np.ascontiguousarray(q), np.ascontiguousarray(qd), ""


@dataclass(frozen=True)
class WalkStats:
    """What a clip asks of the two foot slots, per low-level TICK, and what it would do on a loop.

    ``step_m`` is the FEET-ONLY concatenated per-frame L2 (over the ``(6,)`` vector of both ankle
    positions), because that is the quantity the low level's shared-pool slew cap bounds when the
    upper three points are held still (`hl_5pt.SLEW_IS_SHARED_POOL`). ``orn_step`` is the L2 over
    the ``(8,)`` vector of both xyzw quaternions, the orientation cap's own metric.
    """

    n_frames: int
    fps: float
    step_m: np.ndarray            # (T-1,) feet-only concatenated per-frame position step
    per_foot_step_m: np.ndarray   # (T-1, 2) per-foot per-frame step
    orn_step: np.ndarray          # (T-1,) L2 over both quats per frame
    wrap_gap_m: np.ndarray        # (2,) |frame[-1] - frame[0]| per foot — the loop discontinuity
    wrap_gap_deg: np.ndarray      # (2,) the same on orientation
    yaw_net_deg: float            # pelvis heading change over the clip (unwrapped)
    yaw_range_deg: float
    frame0_pos: np.ndarray        # (2, 3) the clip's first foot frame, pelvis frame
    n_joints: int = 0             # 29 when the clip carries its joint channel, else 0
    peak_joint_speed_rad_s: float = 0.0   # worst |joint_vel| over the clip (0 without joints)

    @property
    def duration_s(self) -> float:
        return float(self.n_frames / self.fps)

    @property
    def mean_step_m(self) -> float:
        return float(self.step_m.mean()) if self.step_m.size else 0.0

    @property
    def p95_step_m(self) -> float:
        return float(np.percentile(self.step_m, 95)) if self.step_m.size else 0.0

    @property
    def max_step_m(self) -> float:
        return float(self.step_m.max()) if self.step_m.size else 0.0

    @property
    def peak_foot_speed_ms(self) -> float:
        """Worst single-foot speed, m/s — cap-independent, the number to carry between clips."""
        return float(self.per_foot_step_m.max() * self.fps) if self.per_foot_step_m.size else 0.0

    @property
    def max_orn_step(self) -> float:
        return float(self.orn_step.max()) if self.orn_step.size else 0.0

    @property
    def p95_orn_step(self) -> float:
        return float(np.percentile(self.orn_step, 95)) if self.orn_step.size else 0.0

    def lead_in_gap_m(self, start_pos: np.ndarray) -> np.ndarray:
        """``(2,)`` how far each foot must ramp from ``start_pos`` to reach the clip's frame 0."""
        return np.linalg.norm(np.asarray(start_pos, dtype=np.float64).reshape(N_FEET, 3)
                              - self.frame0_pos, axis=-1)

    def describe(self, *, speed: float = 1.0, budget_m_per_tick: float | None = None,
                 budget_source: str = "", orn_cap_per_tick: float = LAUNCHER_ORN_CAP_PER_TICK,
                 publish_hz: float = LOW_LEVEL_HZ, feet_slewed: bool | None = None) -> str:
        """One line for the page/log, at the selected ``speed`` and against the given caps.

        At ``speed`` s published at ``publish_hz`` the clip advances ``s * fps / publish_hz`` frames
        per published step, so a per-frame demand scales by that factor per tick (exact for the
        mean, an upper bound for the max of a smooth motion). ``budget_m_per_tick`` is the position
        cap the page believes — the pre-flight estimate or the one ``SlewMonitor`` measured — and
        ``budget_source`` says which, because the two print identical-looking ratios.

        ``feet_slewed`` is whether the low level's caps COVER the feet: ``False`` = the robot said
        so on RecordTap (``vr_cmd_slewed``) and the demand is printed as a fact with no clamp ratio;
        ``True`` = it covers them and the ratios are warnings; ``None`` = not yet observed (nothing
        armed yet), so the ratios are printed against the launcher's expected exemption.
        """
        scale = float(speed) * self.fps / max(float(publish_hz), 1e-9)
        mean_mm, p95_mm, max_mm = (self.mean_step_m * scale * 1e3, self.p95_step_m * scale * 1e3,
                                   self.max_step_m * scale * 1e3)
        orn_p95, orn_max = self.p95_orn_step * scale, self.max_orn_step * scale
        txt = (f"{self.n_frames} frames @ {self.fps:g} fps = {self.duration_s:.1f} s at 1x "
               f"({self.duration_s / max(float(speed), 1e-9):.1f} s at {float(speed):g}x) · feet demand "
               f"mean {mean_mm:.0f} / p95 {p95_mm:.0f} / max {max_mm:.0f} mm per tick · "
               f"peak foot speed {self.peak_foot_speed_ms * float(speed):.2f} m/s · orn "
               f"p95 {orn_p95:.3f} / max {orn_max:.3f} per tick · heading net "
               f"{self.yaw_net_deg:+.0f}° (range {self.yaw_range_deg:.0f}°) · wrap gap "
               f"{self.wrap_gap_m[0] * 1e3:.0f}/{self.wrap_gap_m[1] * 1e3:.0f} mm")
        if self.n_joints:
            txt += (f" · legs: {self.n_joints} reference joints ride with the feet (ref_joints), "
                    f"peak {self.peak_joint_speed_rad_s * float(speed):.1f} rad/s")
        else:
            txt += " · legs: NONE in this clip (feet only)"
        if feet_slewed is False:
            # Read off the robot: the caps do not cover the feet, so no ratio is a warning.
            return txt + (" · feet EXEMPT from the low level's slew caps (RecordTap vr_cmd_slewed) — "
                          "published verbatim; the upper three points keep their cap")
        if budget_m_per_tick is not None and budget_m_per_tick > 0.0:
            ratio_mean = mean_mm * 1e-3 / budget_m_per_tick
            ratio_max = max_mm * 1e-3 / budget_m_per_tick
            src = f" [{budget_source}]" if budget_source else ""
            if ratio_mean > 1.0:
                if feet_slewed is None:
                    txt += (f" · {ratio_mean:.1f}x (mean) / {ratio_max:.1f}x (max) the "
                            f"{budget_m_per_tick * 1e3:.1f} mm/tick position cap{src} — fine IF the "
                            f"low level exempts the feet (SLEW_FEET=0, the launcher default; confirmed "
                            f"off RecordTap once armed), a 5 mm/tick crawl if it does not")
                else:
                    txt += (f" ⚠️ {ratio_mean:.1f}x (mean) / {ratio_max:.1f}x (max) the low level's "
                            f"{budget_m_per_tick * 1e3:.1f} mm/tick position cap{src} — this low level "
                            f"SLEWS the feet (RecordTap vr_cmd_slewed): the wire will be CLAMPED and "
                            f"the feet will LAG the clip; relaunch it with SLEW_FEET=0")
            else:
                txt += (f" · {ratio_max:.2f}x the {budget_m_per_tick * 1e3:.1f} mm/tick position "
                        f"cap{src} at the worst step")
        if orn_cap_per_tick > 0.0 and orn_p95 > orn_cap_per_tick:
            if feet_slewed is None:
                txt += (f" · {orn_p95 / orn_cap_per_tick:.1f}x (p95) the {orn_cap_per_tick:g}/tick "
                        f"orientation cap, same condition")
            else:
                txt += (f" ⚠️ {orn_p95 / orn_cap_per_tick:.1f}x (p95) the {orn_cap_per_tick:g}/tick "
                        f"ORIENTATION cap (HOLOSOMA_VIVE_MAX_ORN_STEP) — foot rotation will lag too")
        return txt


@dataclass(frozen=True)
class WalkClip:
    """A reference clip's two feet in the pelvis frame — and its 29 joints by name — ready to be
    sampled at any phase."""

    name: str
    fps: float
    foot_pos: np.ndarray     # (T, 2, 3)
    foot_quat: np.ndarray    # (T, 2, 4) xyzw, w >= 0
    root_yaw_deg: np.ndarray  # (T,) unwrapped pelvis heading, for the report only
    source: str = ""
    #: The clip's joint channel (see the module docstring): 29 names in the FILE's order and the
    #: ``(T, 29)`` targets / velocities in that order. Empty / None = feet only; ``legs_note`` says why.
    joint_names: tuple[str, ...] = ()
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    legs_note: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.foot_pos.shape[0])

    @property
    def has_joints(self) -> bool:
        return self.joint_pos is not None and self.joint_vel is not None and len(self.joint_names) == N_JOINTS

    def joints(self, phase_frames: float, *, speed: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
        """``(q (29,), qd (29,))`` at a FRACTIONAL frame index in the file's joint order, or None.

        Linear between the two bracketing frames on BOTH arrays, and the velocity scaled by
        ``speed``: a clip played at 0.5x is a motion whose joints move half as fast, and the pair
        the policy sees must say so (the low level zeroes a velocity it cannot vouch for; a wrong
        one is worse than none)."""
        if not self.has_joints:
            return None
        assert self.joint_pos is not None and self.joint_vel is not None
        last = self.n_frames - 1
        f = float(np.clip(phase_frames, 0.0, last))
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, last)
        a = f - i0
        if a <= 0.0 or i1 == i0:
            return self.joint_pos[i0].copy(), self.joint_vel[i0] * float(speed)
        q = self.joint_pos[i0] * (1.0 - a) + self.joint_pos[i1] * a
        qd = (self.joint_vel[i0] * (1.0 - a) + self.joint_vel[i1] * a) * float(speed)
        return q, qd

    @property
    def duration_s(self) -> float:
        return float(self.n_frames / self.fps)

    def frame(self, phase_frames: float) -> tuple[np.ndarray, np.ndarray]:
        """``(pos (2, 3), quat (2, 4))`` at a FRACTIONAL frame index, clamped to the clip.

        Linear on position, shortest-path slerp on orientation, between the two bracketing
        frames — so a speed that is not an integer number of frames per step, or a publish rate
        that is not the clip's fps, still produces a smooth command."""
        last = self.n_frames - 1
        f = float(np.clip(phase_frames, 0.0, last))
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, last)
        a = f - i0
        if a <= 0.0 or i1 == i0:
            return self.foot_pos[i0].copy(), self.foot_quat[i0].copy()
        pos = self.foot_pos[i0] * (1.0 - a) + self.foot_pos[i1] * a
        quat = slerp_xyzw(self.foot_quat[i0], self.foot_quat[i1], a)
        return pos, quat

    def stats(self) -> WalkStats:
        p = self.foot_pos
        q = self.foot_quat
        if self.n_frames > 1:
            d = np.diff(p, axis=0)                                            # (T-1, 2, 3)
            per_foot = np.linalg.norm(d, axis=-1)                              # (T-1, 2)
            step = np.linalg.norm(d.reshape(d.shape[0], -1), axis=-1)         # (T-1,)
            # Shortest-path quaternion difference per frame, then one L2 over both feet.
            dq = np.where(np.sum(q[1:] * q[:-1], axis=-1, keepdims=True) < 0.0, -q[1:], q[1:]) - q[:-1]
            orn = np.linalg.norm(dq.reshape(dq.shape[0], -1), axis=-1)
        else:
            per_foot = np.zeros((0, N_FEET))
            step = np.zeros(0)
            orn = np.zeros(0)
        wrap_m = np.linalg.norm(p[-1] - p[0], axis=-1)
        wrap_deg = np.degrees(quat_angle_rad(q[-1], q[0]))
        yaw = self.root_yaw_deg
        peak_qd = float(np.abs(self.joint_vel).max()) if self.has_joints and self.joint_vel.size else 0.0
        return WalkStats(n_frames=self.n_frames, fps=self.fps, step_m=step, per_foot_step_m=per_foot,
                         orn_step=orn, wrap_gap_m=wrap_m, wrap_gap_deg=wrap_deg,
                         yaw_net_deg=float(yaw[-1] - yaw[0]) if yaw.size else 0.0,
                         yaw_range_deg=float(yaw.max() - yaw.min()) if yaw.size else 0.0,
                         frame0_pos=p[0].copy(),
                         n_joints=N_JOINTS if self.has_joints else 0,
                         peak_joint_speed_rad_s=peak_qd)

    def describe(self) -> str:
        legs = (f"{N_JOINTS} joints by name" if self.has_joints
                else f"feet only ({self.legs_note or 'no joint channel'})")
        return f"{self.name}: {self.n_frames} frames @ {self.fps:g} fps ({self.duration_s:.1f} s) · {legs}"


def _clip_from_holosoma(p: Path, data) -> WalkClip:
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    names = [str(n) for n in np.asarray(data["body_names"]).reshape(-1)]
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{p.name}: fps={fps!r} is not a positive rate")
    if body_pos.shape[0] < 2:
        raise ValueError(f"{p.name}: a clip needs at least 2 frames, got {body_pos.shape[0]}")
    foot_pos, foot_quat = extract_feet_local(body_pos, body_quat, names)
    ip = names.index(PELVIS_BODY)
    yaw = Rotation.from_quat(wxyz_to_xyzw(body_quat[:, ip])).as_euler("ZYX")[:, 0]
    yaw = np.degrees(np.unwrap(yaw))
    jn, jq, jqd, note = extract_joints(data, p.name)
    if jq is not None and jq.shape[0] != foot_pos.shape[0]:
        jn, jq, jqd, note = (), None, None, (f"{p.name}: joint_pos has {jq.shape[0]} frames but the "
                                             f"bodies {foot_pos.shape[0]} — feet only")
    return WalkClip(name=p.name, fps=fps, foot_pos=foot_pos, foot_quat=foot_quat,
                    root_yaw_deg=yaw, source=str(p), joint_names=jn, joint_pos=jq, joint_vel=jqd,
                    legs_note=note)


def _clip_from_feet(p: Path, data) -> WalkClip:
    schema = str(np.asarray(data["schema"]).reshape(-1)[0])
    if schema != FEET_SCHEMA:
        raise ValueError(f"{p.name}: schema {schema!r} is not {FEET_SCHEMA!r}")
    bodies = tuple(str(b) for b in np.asarray(data["foot_bodies"]).reshape(-1))
    if bodies != tuple(FOOT_BODIES):
        raise ValueError(f"{p.name}: foot_bodies {bodies} are not the wire's {tuple(FOOT_BODIES)} — "
                         "re-extract with scripts/extract_walk_feet.py")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    foot_pos = np.asarray(data["foot_pos"], dtype=np.float64)
    foot_quat = np.asarray(data["foot_quat_xyzw"], dtype=np.float64)
    yaw = np.asarray(data["root_yaw_deg"], dtype=np.float64).reshape(-1)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{p.name}: fps={fps!r} is not a positive rate")
    if (foot_pos.ndim != 3 or foot_pos.shape[1:] != (N_FEET, 3) or foot_quat.shape != foot_pos.shape[:2] + (4,)
            or yaw.shape[0] != foot_pos.shape[0]):
        raise ValueError(f"{p.name}: foot_pos {foot_pos.shape} / foot_quat_xyzw {foot_quat.shape} / "
                         f"root_yaw_deg {yaw.shape} do not describe T frames of {N_FEET} feet")
    if foot_pos.shape[0] < 2:
        raise ValueError(f"{p.name}: a clip needs at least 2 frames, got {foot_pos.shape[0]}")
    if not (np.all(np.isfinite(foot_pos)) and np.all(np.isfinite(foot_quat)) and np.all(np.isfinite(yaw))):
        raise ValueError(f"{p.name}: non-finite foot data")
    if np.any(np.abs(np.linalg.norm(foot_quat, axis=-1) - 1.0) > 1e-6) or np.any(foot_quat[..., 3] < 0.0):
        raise ValueError(f"{p.name}: foot_quat_xyzw must be unit quaternions with w >= 0")
    source = str(np.asarray(data["source_name"]).reshape(-1)[0]) if "source_name" in data.files else p.name
    jn: tuple[str, ...] = ()
    jq = jqd = None
    note = f"{p.name}: extracted before the legs existed (no joint arrays) — feet only"
    if {"joint_names", "joint_pos", "joint_vel"} <= set(data.files):
        jn, jq, jqd, note = extract_joints(data, p.name)
        if jq is not None and jq.shape[0] != foot_pos.shape[0]:
            jn, jq, jqd, note = (), None, None, (f"{p.name}: joint_pos has {jq.shape[0]} frames but the "
                                                 f"feet {foot_pos.shape[0]} — feet only")
    return WalkClip(name=source, fps=fps, foot_pos=foot_pos, foot_quat=foot_quat, root_yaw_deg=yaw,
                    source=str(p), joint_names=jn, joint_pos=jq, joint_vel=jqd, legs_note=note)


def load_walk_clip(path: str | os.PathLike) -> WalkClip:
    """Load the two feet of a clip from EITHER a holosoma motion NPZ or an extracted feet NPZ.

    A raw ``bones_seed_g1_holosoma_f32`` file is relativised on load (the STORED ``body_pos_w`` /
    ``body_quat_w`` are read rather than FK'd from ``joint_pos``: the two agree for this family, and
    it keeps the module free of a URDF and a kinematics library). An extracted file
    (:func:`save_feet_clip`, e.g. :data:`BUNDLED_CLIP`) is validated against its schema tag and the
    wire's foot bodies. Anything else is refused BY NAME.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such clip: {p}")
    with np.load(p, allow_pickle=True) as data:
        keys = set(data.files)
        if {"schema", "foot_pos", "foot_quat_xyzw", "fps", "root_yaw_deg", "foot_bodies"} <= keys:
            return _clip_from_feet(p, data)
        need = {"fps", "body_pos_w", "body_quat_w", "body_names"}
        missing = sorted(need - keys)
        if missing:
            raise ValueError(
                f"{p.name} is neither a holosoma motion NPZ (missing {missing}) nor an extracted feet "
                f"NPZ ({FEET_SCHEMA}); has {sorted(keys)[:8]}...")
        return _clip_from_holosoma(p, data)


def save_feet_clip(clip: WalkClip, path: str | os.PathLike, *, source_sha256: str = "") -> Path:
    """Write a :class:`WalkClip` as an extracted feet NPZ — the two feet only, with provenance.

    Round-trips bit-exactly through :func:`load_walk_clip` (float64 arrays, no re-quantisation),
    which is what lets a test pin the bundle to a fresh extraction of the source.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        schema=np.array(FEET_SCHEMA),
        fps=np.array(float(clip.fps)),
        foot_pos=np.asarray(clip.foot_pos, dtype=np.float64),
        foot_quat_xyzw=np.asarray(clip.foot_quat, dtype=np.float64),
        root_yaw_deg=np.asarray(clip.root_yaw_deg, dtype=np.float64),
        foot_bodies=np.array(list(FOOT_BODIES)),
        pelvis_body=np.array(PELVIS_BODY),
        frame=np.array("pelvis body frame of the source clip (full rotation), ankle_roll_link origin, "
                       "xyzw with w >= 0 — holosoma_wbt._compute_vr_fk's own convention"),
        source_name=np.array(str(clip.name)),
        source_sha256=np.array(str(source_sha256)),
    )
    if clip.has_joints:
        # The joint channel, by name, in the FILE's order (the reader places it by name too).
        arrays["joint_names"] = np.array(list(clip.joint_names))
        arrays["joint_pos"] = np.asarray(clip.joint_pos, dtype=np.float64)
        arrays["joint_vel"] = np.asarray(clip.joint_vel, dtype=np.float64)
        arrays["joints_note"] = np.array("the source's own joint_pos[:, 7:] / joint_vel[:, 6:] (rad, rad/s), "
                                         "sent as the 5pt wire's ref_joints while the clip plays so the low "
                                         "level's command_lower_body describes the same gait as the feet")
    np.savez(p, **arrays)
    return p


def load_default_clip() -> WalkClip:
    """The bundled walk-with-box feet (:data:`BUNDLED_CLIP`)."""
    return load_walk_clip(BUNDLED_CLIP)


# ---------------------------------------------------------------------------
# composing a wire frame
# ---------------------------------------------------------------------------
def feet_from_frame(frame: tuple) -> tuple[np.ndarray, np.ndarray]:
    """A ``(pos (15,), orn (20,))`` wire frame -> its two feet, ``(pos (2, 3), quat (2, 4))``."""
    pos = np.asarray(frame[0], dtype=np.float64).reshape(N_POINTS, 3)
    orn = np.asarray(frame[1], dtype=np.float64).reshape(N_POINTS, 4)
    return pos[N_UPPER:].copy(), orn[N_UPPER:].copy()


def compose_frame(upper_frame: tuple, foot_pos: np.ndarray,
                  foot_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The wire frame to publish: ``upper_frame``'s first three slots UNCHANGED, the feet replaced.

    The three upper slots are copied byte for byte (position AND orientation), which is the whole
    contract of "the upper body command does not change while the walk runs"."""
    pos = np.asarray(upper_frame[0], dtype=np.float64).reshape(N_POINTS, 3).copy()
    orn = np.asarray(upper_frame[1], dtype=np.float64).reshape(N_POINTS, 4).copy()
    fp = np.asarray(foot_pos, dtype=np.float64).reshape(N_FEET, 3)
    fq = np.asarray(foot_quat, dtype=np.float64).reshape(N_FEET, 4)
    if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fq))):
        raise ValueError("non-finite foot pose handed to compose_frame")
    pos[N_UPPER:] = fp
    orn[N_UPPER:] = canonical_xyzw(fq / np.maximum(np.linalg.norm(fq, axis=-1, keepdims=True), 1e-12))
    return pos.reshape(-1), orn.reshape(-1)


# ---------------------------------------------------------------------------
# the player: lead-in -> clip (-> wrap -> clip ...) -> lead-out
# ---------------------------------------------------------------------------
class _Ramp:
    """A rate-bounded blend between two foot frames, one published step at a time — and, when
    both are given, between two 29-joint references alongside (the lerp, with the lerp's own
    derivative as the velocity, so the pos/vel pair stays consistent for the policy)."""

    def __init__(self, p0, q0, p1, q1, *, step_dt: float, speed_m_s: float, rate_rad_s: float,
                 min_s: float, j0=None, j1=None) -> None:
        self.p0 = np.asarray(p0, dtype=np.float64).reshape(N_FEET, 3)
        self.q0 = canonical_xyzw(np.asarray(q0, dtype=np.float64).reshape(N_FEET, 4))
        self.p1 = np.asarray(p1, dtype=np.float64).reshape(N_FEET, 3)
        self.q1 = canonical_xyzw(np.asarray(q1, dtype=np.float64).reshape(N_FEET, 4))
        self.j0 = None if j0 is None or j1 is None else np.asarray(j0, dtype=np.float64).reshape(N_JOINTS)
        self.j1 = None if self.j0 is None else np.asarray(j1, dtype=np.float64).reshape(N_JOINTS)
        self.step_dt = float(step_dt)
        # The CONCATENATED gap over both feet, because the low level's cap is one L2 over the whole
        # vector (`hl_5pt.SLEW_IS_SHARED_POOL`) — a per-foot bound would let two feet moving at
        # full rate exceed the pool by sqrt(2) (measured 6.4 mm/step against a 5 mm cap).
        gap = float(np.linalg.norm((self.p1 - self.p0).reshape(-1)))
        ang = float(quat_angle_rad(self.q0, self.q1).max())
        dur = max(float(min_s), gap / max(float(speed_m_s), 1e-9), ang / max(float(rate_rad_s), 1e-9))
        self.n = max(1, int(np.ceil(dur / max(float(step_dt), 1e-9))))
        self.k = 0
        self.gap_m = gap
        self.angle_rad = ang

    @property
    def done(self) -> bool:
        return self.k >= self.n

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        self.k = min(self.k + 1, self.n)
        a = self.k / self.n
        pos = self.p0 * (1.0 - a) + self.p1 * a
        quat = slerp_xyzw(self.q0, self.q1, a)
        return pos, quat

    def joints(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(q, qd)`` of the joint lerp at the CURRENT step (after :meth:`step`), or None."""
        if self.j0 is None or self.j1 is None:
            return None
        a = self.k / self.n
        q = self.j0 * (1.0 - a) + self.j1 * a
        qd = (self.j1 - self.j0) / (self.n * self.step_dt)
        return q, qd


class _Pursuit:
    """A rate-bounded approach to a target that MAY MOVE between steps.

    The leg override's lead-out: its home is the PLANNER'S LIVE foot target (re-read from
    ``target()`` every step), not the snapshot :class:`_Ramp` interpolates to. Each step closes the
    same fraction of the remaining position gap and the remaining angle — the smaller of what the
    two per-tick bounds allow — so both land together, and the LAST step lands EXACTLY on the target
    (a remaining gap inside the bound is closed whole), which is what makes the hand-back
    continuous: when :attr:`done` the wire's feet ARE the planner's. The legs (when given) close the
    same fraction of THEIR gap toward ``j1`` with the lerp's own derivative as the velocity, so a
    pos/vel pair the policy can believe lands with the feet. Exposes the same ``step`` / ``joints`` /
    ``done`` / ``k`` / ``n`` surface as :class:`_Ramp`; ``n`` is the ESTIMATED steps remaining
    (display only — the target may move)."""

    def __init__(self, p0, q0, target: Callable[[], tuple], *, step_dt: float, speed_m_s: float,
                 rate_rad_s: float, j0=None, j1=None) -> None:
        self.pos = np.asarray(p0, dtype=np.float64).reshape(N_FEET, 3).copy()
        self.quat = canonical_xyzw(np.asarray(q0, dtype=np.float64).reshape(N_FEET, 4))
        self.target = target
        self.j0 = None if j0 is None or j1 is None else np.asarray(j0, dtype=np.float64).reshape(N_JOINTS)
        self.j1 = None if self.j0 is None else np.asarray(j1, dtype=np.float64).reshape(N_JOINTS)
        self._q = None if self.j0 is None else self.j0.copy()
        self._qd = None if self.j0 is None else np.zeros(N_JOINTS)
        self.step_dt = float(step_dt)
        self.max_step_m = float(speed_m_s) * self.step_dt
        self.max_step_rad = float(rate_rad_s) * self.step_dt
        self.k = 0
        self._done = False
        p1, q1 = self._read_target()
        self.gap_m = float(np.linalg.norm((p1 - self.pos).reshape(-1)))
        self.angle_rad = float(quat_angle_rad(self.quat, q1).max())
        self.n = self._estimate(self.gap_m, self.angle_rad)

    def _read_target(self) -> tuple[np.ndarray, np.ndarray]:
        p1, q1 = self.target()
        p1 = np.asarray(p1, dtype=np.float64).reshape(N_FEET, 3)
        q1 = canonical_xyzw(np.asarray(q1, dtype=np.float64).reshape(N_FEET, 4))
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(q1))):
            raise ValueError("the pursuit's target returned a non-finite foot pose")
        return p1, q1

    def _estimate(self, gap: float, ang: float) -> int:
        return max(1, int(np.ceil(max(gap / max(self.max_step_m, 1e-12),
                                      ang / max(self.max_step_rad, 1e-12)))))

    @property
    def done(self) -> bool:
        return self._done

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        if self._done:
            return self.pos.copy(), self.quat.copy()
        p1, q1 = self._read_target()
        gap = float(np.linalg.norm((p1 - self.pos).reshape(-1)))
        ang = float(quat_angle_rad(self.quat, q1).max())
        a_pos = 1.0 if gap <= self.max_step_m else self.max_step_m / gap
        a_ang = 1.0 if ang <= self.max_step_rad else self.max_step_rad / ang
        a = min(a_pos, a_ang)
        self.k += 1
        if a >= 1.0:
            self.pos, self.quat = p1.copy(), q1.copy()          # land EXACTLY on the target
            self._done = True
        else:
            self.pos = self.pos + a * (p1 - self.pos)
            self.quat = canonical_xyzw(slerp_xyzw(self.quat, q1, a))
        if self._q is not None:
            dq = a * (self.j1 - self._q) if not self._done else (self.j1 - self._q)
            self._qd = dq / self.step_dt
            self._q = self._q + dq
        self.gap_m, self.angle_rad = gap, ang
        self.n = self.k + self._estimate(gap * (1.0 - a), ang * (1.0 - a))
        return self.pos.copy(), self.quat.copy()

    def joints(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._q is None or self._qd is None:
            return None
        return self._q.copy(), self._qd.copy()


class WalkPlayer:
    """Emits the two foot poses for each published step; the page composes and publishes them.

    Phases: ``lead_in`` (ramp from the wire's current feet to the clip's frame 0) -> ``clip`` ->
    on the clip's end either ``wrap`` (a ramp back to frame 0, then ``clip`` again — ``loop``) or
    ``lead_out`` (a ramp back to the START feet) -> ``done``. :meth:`request_stop` switches to
    ``lead_out`` from wherever the feet are; the caller's STOP simply stops calling :meth:`step`,
    which freezes the wire on the last frame (that is the low level's own STOP semantics too).
    With ``home_provider`` the lead-out PURSUES the provider's live target instead of ramping to
    the start snapshot (:class:`_Pursuit`; the leg override's hand-back to a running planner).

    Ramps are bounded by ``ramp_speed_m_s`` (default 0.25 m/s = 5 mm per 50 Hz tick, i.e. never
    over the low level's pinned cap by construction) and ``ramp_rate_deg_s``, with a floor of
    ``ramp_min_s`` so a tiny gap is still a few steps rather than one.

    The clip's phase advances by ``speed * fps * step_dt`` frames per call, so the same player is
    correct at any publish rate; ``speed``/``loop`` are re-read every step (the page's slider and
    checkbox are live).

    THE LEGS. Hand it ``start_joints`` (the 29 joints the low level HOLDS — the ready pose,
    ``DeployAnchor.dof``) in ``joint_names`` order and, if the clip carries its joint channel,
    :attr:`legs` is ``(joint_names, q, qd)`` for every step in THAT order: the clip's own joints
    (velocity scaled by the speed) in the clip phase, the lerp from/to the start joints during the
    lead-in / wrap / lead-out ramps. Placed by name — a clip lacking any of the 29 names leaves the
    legs OFF (:attr:`legs_enabled` False, :attr:`legs_note` says why) and the feet unchanged.
    """

    PHASES = ("lead_in", "clip", "wrap", "lead_out", "done")

    def __init__(self, clip: WalkClip, *, start_pos, start_quat, step_dt: float,
                 speed: float = 1.0, loop: bool = True, ramp_speed_m_s: float = 0.25,
                 ramp_rate_deg_s: float = 45.0, ramp_min_s: float = 0.5,
                 start_joints=None, joint_names=None,
                 home_provider: Callable[[], tuple] | None = None) -> None:
        if clip.n_frames < 2:
            raise ValueError("a walk clip needs at least 2 frames")
        self.clip = clip
        #: Where the lead-out returns to. ``None`` (the ``k`` walk): the START feet, a snapshot, via
        #: the fixed :class:`_Ramp`. A callable (the leg override): the PLANNER'S LIVE foot target
        #: ``() -> (pos (2, 3), quat (2, 4))``, pursued by :class:`_Pursuit` so the hand-back lands
        #: on wherever the policy's stance is by then. The lead-in and the wrap are unchanged.
        self.home_provider = home_provider
        self.step_dt = float(step_dt)
        if not (self.step_dt > 0.0):
            raise ValueError(f"step_dt must be positive, got {step_dt}")
        self.ramp_speed_m_s = float(ramp_speed_m_s)
        self.ramp_rate_rad_s = float(np.deg2rad(ramp_rate_deg_s))
        self.ramp_min_s = float(ramp_min_s)
        self.start_pos = np.asarray(start_pos, dtype=np.float64).reshape(N_FEET, 3).copy()
        self.start_quat = canonical_xyzw(np.asarray(start_quat, dtype=np.float64).reshape(N_FEET, 4))
        self._speed = 1.0
        self.set_speed(speed)
        self.loop = bool(loop)
        # --- the legs (optional; see the class docstring) ---
        self.joint_names: tuple[str, ...] = tuple(str(n) for n in (joint_names or ()))
        self.legs_note = ""
        self._perm: np.ndarray | None = None      # clip joint order -> joint_names order
        self.start_q: np.ndarray | None = None
        if start_joints is None:
            self.legs_note = "no start joints handed to the player — legs off"
        elif not clip.has_joints:
            self.legs_note = clip.legs_note or "the clip has no joint channel — legs off"
        elif len(self.joint_names) != N_JOINTS or len(set(self.joint_names)) != N_JOINTS:
            raise ValueError(f"joint_names must be {N_JOINTS} distinct names, got {len(self.joint_names)}")
        else:
            sq = np.asarray(start_joints, dtype=np.float64).reshape(-1)
            if sq.shape != (N_JOINTS,) or not np.all(np.isfinite(sq)):
                raise ValueError(f"start_joints must be {N_JOINTS} finite joint angles, got {sq.shape}")
            index = {n: i for i, n in enumerate(clip.joint_names)}
            missing = [n for n in self.joint_names if n not in index]
            if missing:
                self.legs_note = (f"the clip names no joint {missing[:4]}{'…' if len(missing) > 4 else ''} "
                                  f"— legs off (feet unchanged)")
            else:
                self._perm = np.array([index[n] for n in self.joint_names], dtype=np.int64)
                self.start_q = sq.copy()
        self.cur_q: np.ndarray | None = None if self.start_q is None else self.start_q.copy()
        self.cur_qd: np.ndarray | None = None if self.start_q is None else np.zeros(N_JOINTS)
        self.phase = "lead_in"
        self.phase_frames = 0.0
        self.laps = 0
        self.n_steps = 0
        self.cur_pos = self.start_pos.copy()
        self.cur_quat = self.start_quat.copy()
        p1, q1 = clip.frame(0.0)
        self._ramp: _Ramp | None = self._make_ramp(p1, q1, self._clip_q(0.0))
        self.lead_in_steps = self._ramp.n
        self.lead_in_joint_gap_rad = (0.0 if self._ramp.j0 is None
                                      else float(np.abs(self._ramp.j1 - self._ramp.j0).max()))

    # --- the legs -------------------------------------------------------------------------
    @property
    def legs_enabled(self) -> bool:
        return self._perm is not None

    @property
    def legs(self) -> tuple[tuple[str, ...], np.ndarray, np.ndarray] | None:
        """``(joint_names, q (29,), qd (29,))`` for the step last returned by :meth:`step`, in
        ``joint_names`` order — the 5pt wire's ``ref_joints`` — or None when the legs are off."""
        if self._perm is None or self.cur_q is None or self.cur_qd is None:
            return None
        return self.joint_names, self.cur_q.copy(), self.cur_qd.copy()

    def _clip_q(self, phase_frames: float) -> np.ndarray | None:
        """The clip's joint TARGETS at a phase, in ``joint_names`` order (a ramp endpoint)."""
        if self._perm is None:
            return None
        got = self.clip.joints(phase_frames, speed=self._speed)
        return None if got is None else got[0][self._perm]

    def _clip_joints(self, phase_frames: float) -> tuple[np.ndarray, np.ndarray] | None:
        if self._perm is None:
            return None
        got = self.clip.joints(phase_frames, speed=self._speed)
        return None if got is None else (got[0][self._perm], got[1][self._perm])

    # --- live settings --------------------------------------------------------
    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, value: float) -> None:
        v = float(value)
        if not np.isfinite(v):
            return
        self._speed = float(np.clip(v, SPEED_MIN, SPEED_MAX))

    def set_loop(self, value: bool) -> None:
        self.loop = bool(value)

    # --- control ----------------------------------------------------------------
    def request_stop(self) -> None:
        """Graceful stop: ramp the feet (and the legs) back to where they were when the walk started."""
        if self.phase in ("lead_out", "done"):
            return
        self.phase = "lead_out"
        self._ramp = self._make_lead_out()

    @property
    def stopping(self) -> bool:
        return self.phase in ("lead_out", "done")

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def clip_frame(self) -> float:
        return float(self.phase_frames)

    def progress(self) -> str:
        if self.phase == "clip":
            return (f"clip frame {self.phase_frames:.0f}/{self.clip.n_frames - 1}"
                    f"{f' · lap {self.laps + 1}' if self.loop else ''}")
        if self.phase in ("lead_in", "wrap", "lead_out") and self._ramp is not None:
            if isinstance(self._ramp, _Pursuit):
                return (f"{self.phase} (pursuing the planner's feet) {self._ramp.gap_m * 1e3:.0f} mm / "
                        f"{np.rad2deg(self._ramp.angle_rad):.1f}° to go, step {self._ramp.k}")
            return f"{self.phase} {self._ramp.k}/{self._ramp.n}"
        return self.phase

    # --- the step -----------------------------------------------------------------
    def step(self) -> tuple[np.ndarray, np.ndarray, str]:
        """Advance one published step. Returns ``(foot_pos (2, 3), foot_quat (2, 4), phase)``.

        Once ``done`` it keeps returning the final (start) feet, so a caller that over-runs by a
        step publishes a hold rather than raising."""
        if self.phase == "done":
            return self.cur_pos.copy(), self.cur_quat.copy(), self.phase
        self.n_steps += 1
        if self.phase in ("lead_in", "wrap", "lead_out"):
            assert self._ramp is not None
            pos, quat = self._ramp.step()
            joints = self._ramp.joints()
            phase_now = self.phase
            if self._ramp.done:
                if self.phase == "lead_out":
                    self.phase = "done"
                    if joints is not None:
                        joints = (joints[0], np.zeros(N_JOINTS))   # landed: a held reference is at rest
                else:
                    # lead_in / wrap both land ON frame 0 of the clip; the next step advances.
                    self.phase = "clip"
                    self.phase_frames = 0.0
                    if phase_now == "wrap":
                        self.laps += 1
                self._ramp = None
            self._set_current(pos, quat, joints)
            return pos.copy(), quat.copy(), phase_now
        # phase == "clip"
        self.phase_frames += self._speed * self.clip.fps * self.step_dt
        last = self.clip.n_frames - 1
        if self.phase_frames > last + 1e-9:
            # The clip has ended. Loop: ramp from where the feet ARE back to frame 0 (the wrap
            # gap is measured at 40-330 mm across the pulled clips, so a cut would be a jump).
            # No loop: ramp home.
            self.phase_frames = float(last)
            if self.loop:
                self.phase = "wrap"
                p1, q1 = self.clip.frame(0.0)
                j1 = self._clip_q(0.0)
                self._ramp = self._make_ramp(p1, q1, j1)
            else:
                self.phase = "lead_out"
                self._ramp = self._make_lead_out()
            pos, quat = self._ramp.step()
            joints = self._ramp.joints()
            phase_now = self.phase
            if self._ramp.done:
                if self.phase == "lead_out":
                    self.phase = "done"
                    if joints is not None:
                        joints = (joints[0], np.zeros(N_JOINTS))
                else:
                    self.phase = "clip"
                    self.phase_frames = 0.0
                    self.laps += 1
                self._ramp = None
            self._set_current(pos, quat, joints)
            return pos.copy(), quat.copy(), phase_now
        pos, quat = self.clip.frame(self.phase_frames)
        self._set_current(pos, quat, self._clip_joints(self.phase_frames))
        return pos.copy(), quat.copy(), "clip"

    # --- internals ----------------------------------------------------------------------
    def _make_ramp(self, p1, q1, j1=None) -> _Ramp:
        j0 = self.cur_q if (self._perm is not None and j1 is not None) else None
        return _Ramp(self.cur_pos, self.cur_quat, p1, q1, step_dt=self.step_dt,
                     speed_m_s=self.ramp_speed_m_s, rate_rad_s=self.ramp_rate_rad_s,
                     min_s=self.ramp_min_s, j0=j0, j1=None if j0 is None else j1)

    def _make_lead_out(self):
        """The lead-out: the fixed ramp to the START feet, or — with ``home_provider`` — a pursuit
        of the provider's live target. The legs return to the START joints (the ready legs) either way."""
        if self.home_provider is None:
            return self._make_ramp(self.start_pos, self.start_quat, self.start_q)
        j0 = self.cur_q if (self._perm is not None and self.start_q is not None) else None
        return _Pursuit(self.cur_pos, self.cur_quat, self.home_provider, step_dt=self.step_dt,
                        speed_m_s=self.ramp_speed_m_s, rate_rad_s=self.ramp_rate_rad_s,
                        j0=j0, j1=None if j0 is None else self.start_q)

    def _set_current(self, pos, quat, joints=None) -> None:
        self.cur_pos = np.asarray(pos, dtype=np.float64).reshape(N_FEET, 3).copy()
        self.cur_quat = canonical_xyzw(np.asarray(quat, dtype=np.float64).reshape(N_FEET, 4))
        if self._perm is not None and joints is not None:
            self.cur_q = np.asarray(joints[0], dtype=np.float64).reshape(N_JOINTS).copy()
            self.cur_qd = np.asarray(joints[1], dtype=np.float64).reshape(N_JOINTS).copy()



# ---------------------------------------------------------------------------
# the leg override: the same feet + legs, substituted under a RUNNING planner
# ---------------------------------------------------------------------------
class LegOverride:
    """The clip's feet + legs on the two FOOT slots of every frame the page publishes — under a
    planner that keeps running.

    A SUBSTITUTION at the page's single publish seam, not a publisher (module docstring, "THE LEG
    OVERRIDE"): the page installs :meth:`apply` as the hook its ``_send`` funnel runs on every frame
    — the drip's, the stream's, the heartbeat's — and drives :meth:`tick` from its own clock thread
    at the clip's rate. Neither publishes. ``tick`` writes and ``apply`` reads the current feet and
    legs under ONE lock, so every frame carries ONE tick of the clip (feet AND legs from the same
    step), and ``apply`` never raises on a finite frame — the values it substitutes were validated
    when produced (``compose_frame`` re-checks).

    ``tick(hold=True)`` (the page's STOP is latched) FREEZES the gait where it is: the feet stay,
    the legs' velocity is zeroed (a held reference is at rest) — unless a hand-back is under way,
    which proceeds regardless: :meth:`request_release` is the operator's explicit, rate-bounded
    request, and the pursuit lands the feet on the planner's own target (see
    :class:`WalkPlayer` ``home_provider``). ``legs_enabled`` is re-read per frame (the page's
    checkbox is live); when False the caller's own ``ref_joints`` pass through (None for every
    non-walk frame).
    """

    def __init__(self, player: WalkPlayer, *, legs_enabled: Callable[[], bool] | None = None) -> None:
        self.player = player
        self._legs_enabled = legs_enabled if legs_enabled is not None else (lambda: True)
        self._lock = threading.Lock()
        self._foot_pos = player.cur_pos.copy()
        self._foot_quat = player.cur_quat.copy()
        self._legs = player.legs
        self._phase = player.phase
        self.n_ticks = 0      # player steps taken
        self.n_held = 0       # ticks frozen by a latched STOP
        self.n_frames = 0     # frames whose feet were substituted

    # --- the clock's half ------------------------------------------------------------------
    def tick(self, *, hold: bool = False) -> str:
        """Advance one clip step (or hold). Returns the player's phase after the step."""
        with self._lock:
            if self.player.done:
                self._phase = "done"
                return self._phase
            if hold and not self.player.stopping:
                self.n_held += 1
                if self._legs is not None:
                    names, q, _qd = self._legs
                    self._legs = (names, q, np.zeros_like(q))
                return self._phase
            pos, quat, phase = self.player.step()
            self._foot_pos, self._foot_quat, self._phase = pos, quat, phase
            self._legs = self.player.legs
            self.n_ticks += 1
            return phase

    def request_release(self) -> None:
        """Hand the legs back: the lead-out (a pursuit of the planner's live feet when the player has
        a ``home_provider``) starts from wherever the feet are. Idempotent."""
        with self._lock:
            self.player.request_stop()

    @property
    def stopping(self) -> bool:
        return self.player.stopping

    @property
    def done(self) -> bool:
        """The hand-back has landed: the wire's feet are the planner's. The page then waits for ONE
        more substituted frame (:attr:`n_frames`) before uninstalling — the publishers run at 60 Hz
        against this clock's 50, so without that wait the last frame to go out is the tick BEFORE the
        landing and the seam is one pursuit step (measured 0.3-1.3 mm; bounded by the ramp rate, but
        avoidable for nothing)."""
        return self.player.done

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def sending_legs(self) -> bool:
        return bool(self._legs_enabled()) and self._legs is not None

    def current(self) -> tuple[np.ndarray, np.ndarray, tuple | None]:
        """``(foot_pos (2, 3), foot_quat (2, 4), legs | None)`` of the latest tick, as copies."""
        with self._lock:
            legs = None if self._legs is None else (self._legs[0], self._legs[1].copy(), self._legs[2].copy())
            return self._foot_pos.copy(), self._foot_quat.copy(), legs

    # --- the publishers' half --------------------------------------------------------------
    def apply(self, frame: tuple, ref_joints=None) -> tuple[tuple, object]:
        """``(frame, ref_joints)`` to publish: the frame's two foot slots replaced by the current
        clip feet, the legs riding along when enabled. The three upper slots are byte-identical."""
        foot_pos, foot_quat, legs = self.current()
        out = compose_frame(frame, foot_pos, foot_quat)
        with self._lock:
            self.n_frames += 1      # a PROGRESS signal the caller waits on at the hand-back, so locked
        if self._legs_enabled() and legs is not None:
            return out, legs
        return out, ref_joints

    def progress(self) -> str:
        return self.player.progress()

    def describe(self) -> str:
        return (f"{self._phase} · {self.player.progress()} · {self.n_ticks} tick(s), {self.n_frames} frame(s) "
                f"substituted{f', {self.n_held} held by STOP' if self.n_held else ''} · speed "
                f"{self.player.speed:g}x · legs {'ON' if self.sending_legs else 'off'}")


__all__ = [
    "BUNDLED_CLIP",
    "DEFAULT_CLIP",
    "FEET_SCHEMA",
    "FOOT_BODIES",
    "FOOT_LABELS",
    "LAUNCHER_ORN_CAP_PER_TICK",
    "LOW_LEVEL_HZ",
    "LegOverride",
    "N_FEET",
    "N_JOINTS",
    "N_UPPER",
    "PELVIS_BODY",
    "SPEED_MAX",
    "SPEED_MIN",
    "WalkClip",
    "WalkPlayer",
    "WalkStats",
    "canonical_xyzw",
    "compose_frame",
    "extract_feet_local",
    "extract_joints",
    "feet_from_frame",
    "load_default_clip",
    "load_walk_clip",
    "quat_angle_rad",
    "save_feet_clip",
    "slerp_xyzw",
    "wxyz_to_xyzw",
]
