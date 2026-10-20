"""The whole-body policy's GEOMETRIC proprioception: a 10-dim view of the 29 joints.

Why this module exists
---------------------
``diffusion_umi_wholebody`` conditions on the 29 G1 joint angles the completion produced
(``layout.py``: ``state_geom == ("joints",)``). That is honest proprioception and it trains
well — and it is the one channel measured to be far OUT OF DISTRIBUTION at the robot. Over
the 159 valid episode starts of ``session_2026.09.02_merged3``, the export's joints sit

    mean |q - ready|   0.2750 rad      worst joint   0.6643 rad
    joints past the low level's 0.20 rad pre-arm gate                16 of 29

away from the pose the G1 actually holds after the operator presses **A**. Two mechanisms
absorb that on the wire (``JointClutch``) and in the observation (``--state-source
command``), but neither closes it: the model still reads a posture space the robot is not in.

The fix this module implements is to condition on what the joints MEAN rather than on the
joints: a small set of **pelvis-frame Cartesian relations between the two hands and the
body**. The same 159 episode starts, in these features:

    feature      ready pose   dataset mean   gap        dataset std   gap/std
    lr_dx         0.000 m      -0.011 m      -0.011 m    0.028 m       0.40
    lr_dy         0.297 m       0.384 m      +0.087 m    0.037 m       2.34
    lp_x          0.442 m       0.427 m      -0.015 m    0.052 m       0.28
    lp_y          0.149 m       0.241 m      +0.092 m    0.037 m       2.46
    left  r/p/y  ~0 deg        -1.3/4.2/2.2   1.3-4.2 deg  3.0-7.2 deg  0.43-0.58
    right r/p/y  ~0 deg         1.5/11.5/9.1  1.5-11.5     3.8-7.7      0.39-1.57

i.e. every x/rotation channel is INSIDE one standard deviation of the training distribution
where the raw joints are 16-of-29 past the gate. (The two ``y`` channels are the residue: the
completion places the operator's hands ~9 cm wider apart than the ready pose does, which is a
PLACEMENT property of the export and not something an observation transform can remove. It is
recorded here rather than hidden — see :mod:`flore.s1_policy.data.layout`'s mode table.)

Three properties are load-bearing
---------------------------------
* **The frame is the PELVIS, not the world.** ``KeypointFK.bodies(q)`` returns every body in
  the URDF's pelvis frame, so these features are invariant to where the take happened AND to
  which way the operator was facing — the two things that made the pre-v2 policy's absolute
  pose state OOD (``hl_dataset``'s module docstring). A world-frame hand-to-hand difference
  would rotate with the operator's heading, and this rig's operator reads -169 deg.
* **``lp_*`` is literally ``p_left``.** "Left TCP relative to the pelvis, in xy" is
  ``(p_L - p_pelvis)[:2]`` and the pelvis IS the frame origin, so the subtraction is the
  identity. Written as ``p_L[:2]`` with this note rather than as a subtraction of zeros.
* **ONE FK, shared with deploy.** :class:`~flore.s1_policy.inference.ready_pose.KeypointFK` is
  the same class ``hl_dense``'s post-completion command anchor and the ready-pose frame use,
  off the same ``g1_29dof_yam.urdf`` and the same 0.242 m fingertip lever. A second FK here
  that disagreed by a lever is exactly the failure ``ready_pose._cross_check_bake`` exists to
  catch, so there is no second FK — this module only re-keys and differences its output.

Euler angles, deliberately and with a guard
-------------------------------------------
The repo's rotation convention is rot6d (``flore/utils/rot6d.py``) precisely because Euler
angles wrap and gimbal-lock. RPY is used here anyway because it is what the 6 channels are
FOR: a per-axis wrist attitude a human can reason about and put a "10 degrees" noise number
on. That is only safe while the angles stay small, so it is MEASURED rather than assumed —
on ``session_2026.09.02_merged3`` the worst single |angle| over 7153 frames is 41.5 deg and
0.00% exceed 160 deg. :func:`geom_series` re-checks that per dataset and warns loudly if a
future task reaches past :data:`RPY_WRAP_WARN_RAD`, because a wrapped channel is a 2-pi jump
the loss would chase and no shape would reveal.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: URDF bodies the hand features are read off. Same bodies
#: :data:`flore.s1_policy.inference.ready_pose._ROLE_BODY` assigns to the ``left``/``right``
#: layout roles, and the 0.242 m fingertip lever is already applied by ``KeypointFK``.
LEFT_BODY = "left_wrist_yaw_link"
RIGHT_BODY = "right_wrist_yaw_link"

#: The CHEST body, for :data:`REL_XY3_NAMES`. holosoma's third tracked point is ``torso_link``
#: plus its own lever (``home_anchor.json::vr_3point_offsets_tcp[2]``, nominally +0.35 z), and
#: ``KeypointFK`` has already applied it — so this is the same point the 3pt/5pt wire carries
#: and the same one the completion takes as its root conditioning. G1 HAS NO NECK JOINT: this
#: is the upper torso, not a head.
CHEST_BODY = "torso_link"

#: Column names of the ``rel_xy`` block, in order. ``lr_*`` is left-minus-right; ``lp_*`` is
#: left-minus-PELVIS, which is ``p_left`` itself because FK's frame origin IS the pelvis.
REL_XY_NAMES: tuple[str, ...] = ("lr_dx", "lr_dy", "lp_x", "lp_y")

#: Column names of the ``rel_xy3`` block, in order: the THREE pairwise xy differences among
#: the three TRACKED points (left TCP, right TCP, chest), and nothing else.
#:
#: This is ``rel_xy``'s sibling, not its extension, and the difference is the whole point.
#: ``rel_xy`` carries ``lp_x``/``lp_y`` = the left hand's position relative to the pelvis, i.e.
#: an ABSOLUTE placement in the pelvis frame — measured 0.28 sigma on ``merged3``, so it is a
#: perfectly good channel, but it is a channel only the completion can produce. Every column
#: here is a DIFFERENCE BETWEEN TWO TRACKED POINTS, which makes the block
#: **translation-invariant**: it does not depend on where the pelvis is, only on how the frame
#: is ORIENTED (a difference vector still rotates with the frame, which is why the FK's pelvis
#: frame is still required — see the module docstring's first load-bearing property).
#:
#: The chest's own ``hp_x``/``hp_y`` is deliberately ABSENT. Including it would re-introduce
#: exactly the absolute placement this block exists to avoid, and it is the placement that the
#: ``home`` / ``absolute --dz-anchor hand`` export choice moves (measured on the two
#: placements' completions of one take: 61-140 mm on a hand, 48-57 mm on the chest, after the
#: one rigid offset is removed — while the per-chunk CUMULATIVE hand delta agrees to 4.8-38 mm
#: against a 72-505 mm signal). A pairwise difference keeps only the part that survives.
REL_XY3_NAMES: tuple[str, ...] = (
    "lr_dx", "lr_dy",   # left  - right
    "lh_dx", "lh_dy",   # left  - chest
    "rh_dx", "rh_dy",   # right - chest
)

#: Column names of the ``rpy`` block, in order: extrinsic-XYZ Euler of each hand's rotation
#: in the pelvis frame, radians. No pelvis RPY — in the pelvis frame the pelvis is identity.
RPY_NAMES: tuple[str, ...] = (
    "left_roll", "left_pitch", "left_yaw", "right_roll", "right_pitch", "right_yaw")

#: block name -> its ordered column names. The single source of truth for both the WIDTH
#: (``layout.WB_STATE_BLOCK_DIM``) and the ORDER :func:`geom_from_bodies` writes.
GEOM_BLOCK_NAMES: dict[str, tuple[str, ...]] = {
    "rel_xy": REL_XY_NAMES,
    "rel_xy3": REL_XY3_NAMES,
    "rpy": RPY_NAMES,
}

#: Blocks that need forward kinematics. ``joints`` (the original ``wholebody`` state) is read
#: straight off the zarr and is deliberately NOT here.
FK_BLOCKS: frozenset[str] = frozenset(GEOM_BLOCK_NAMES)

#: Warn above this |angle| (150 deg). Not an error: a legitimate task could reach there, and
#: refusing to train would be worse than saying the channel is about to wrap. Chosen below
#: 180 deg by a margin wider than any measured excursion (41.5 deg on merged3).
RPY_WRAP_WARN_RAD = float(np.deg2rad(150.0))


def load_geom_fk(**kwargs) -> tuple[object | None, str]:
    """``(fk, reason)`` — the shared :class:`KeypointFK`, or why it is unavailable.

    Imported lazily, INSIDE the function, on purpose: ``ready_pose`` reaches into
    ``inference/`` and ``serving/``, and a module-level import here would make ``data/``
    depend on both at import time — the layering inversion ``layout.WB_JOINT_NAMES``' comment
    calls out. Every caller in this package already handles a ``(None, reason)`` return by
    refusing with the reason attached.
    """
    from flore.s1_policy.inference.ready_pose import KeypointFK  # noqa: PLC0415

    return KeypointFK.load(**kwargs)


def _rpy(rotmat: np.ndarray) -> np.ndarray:
    """Extrinsic-XYZ Euler angles (roll, pitch, yaw) of a 3x3 rotation, radians."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    return np.asarray(Rotation.from_matrix(np.asarray(rotmat, dtype=np.float64)).as_euler("xyz"),
                      dtype=np.float64)


def geom_from_bodies(pos: dict, rot: dict, blocks: tuple[str, ...]) -> np.ndarray:
    """One frame's geometric features, concatenated in ``blocks`` order.

    ``pos`` / ``rot`` are ``KeypointFK.bodies()``'s output — PELVIS frame, levers applied.
    Takes the FK output rather than the joint vector so a caller that already has the bodies
    (the deploy side, which needs the same FK for its wire) pays for one FK, not two.
    """
    out: list[np.ndarray] = []
    for block in blocks:
        if block == "rel_xy":
            p_l = np.asarray(pos[LEFT_BODY], dtype=np.float64).reshape(3)
            p_r = np.asarray(pos[RIGHT_BODY], dtype=np.float64).reshape(3)
            # p_l[:2] IS (p_left - p_pelvis)[:2]: the pelvis is this frame's origin.
            out.append(np.concatenate([(p_l - p_r)[:2], p_l[:2]]))
        elif block == "rel_xy3":
            p_l = np.asarray(pos[LEFT_BODY], dtype=np.float64).reshape(3)
            p_r = np.asarray(pos[RIGHT_BODY], dtype=np.float64).reshape(3)
            p_h = np.asarray(pos[CHEST_BODY], dtype=np.float64).reshape(3)
            # Three differences, no absolute term — see REL_XY3_NAMES. The pelvis ORIGIN
            # cancels out of every column here (unlike `rel_xy`'s `lp_*`); its ORIENTATION
            # does not, which is why this still needs the pelvis-frame FK.
            out.append(np.concatenate([(p_l - p_r)[:2], (p_l - p_h)[:2], (p_r - p_h)[:2]]))
        elif block == "rpy":
            out.append(np.concatenate([_rpy(rot[LEFT_BODY]), _rpy(rot[RIGHT_BODY])]))
        else:
            raise ValueError(
                f"{block!r} is not a geometric state block; known blocks are "
                f"{sorted(GEOM_BLOCK_NAMES)} (the 'joints' block is read off the zarr, "
                f"not computed here)")
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def geom_from_joints(fk, dof, blocks: tuple[str, ...]) -> np.ndarray:
    """One frame's geometric features from 29 SDK joint angles. Convenience over FK+differences."""
    pos, rot = fk.bodies(dof)
    return geom_from_bodies(pos, rot, blocks)


def geom_series(fk, joints: np.ndarray, covered: np.ndarray, blocks: tuple[str, ...],
                *, label: str = "", log_every: int = 5000) -> np.ndarray:
    """``(T, D)`` features for a whole file's per-frame joints. Uncovered rows stay ZERO.

    ``covered`` is the exporter's ``wb_state_valid``: a frame the completion never reached
    holds whatever the exporter zero-filled, and 0 rad is a legal posture, so FK'ing it would
    produce a plausible-looking pose for a frame that has no pose. Those rows are left zero
    and the caller marks them ``state_valid=False`` — the same discipline the ``joints`` block
    already follows.

    ~0.66 ms/frame (yourdfpy FK + two ``as_euler``), i.e. ~25 s for a 37k-frame dataset. Done
    ONCE here, in ``HLDataset.__init__`` before the DataLoader forks, rather than per
    ``__getitem__``: the result is a small float array that pickles to every worker, exactly
    like ``_ts``, and a per-sample FK would put 0.66 ms of single-threaded work behind every
    image decode. Progress is logged because 25 s of silence reads as a hang.
    """
    joints = np.asarray(joints, dtype=np.float64)
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    if joints.ndim != 2 or joints.shape[0] != covered.shape[0]:
        raise ValueError(
            f"joints {joints.shape} and covered {covered.shape} must agree on the frame count")
    dim = sum(len(GEOM_BLOCK_NAMES[b]) for b in blocks)
    out = np.zeros((joints.shape[0], dim), dtype=np.float64)
    if dim == 0:
        return out
    idx = np.nonzero(covered)[0]
    for n, f in enumerate(idx):
        out[f] = geom_from_joints(fk, joints[f], blocks)
        if log_every and (n + 1) % log_every == 0:
            logger.info("wb_geom%s: FK %d/%d covered frames", label, n + 1, idx.size)
    _warn_on_wrap(out, blocks, label)
    return out


def _warn_on_wrap(feats: np.ndarray, blocks: tuple[str, ...], label: str) -> None:
    """Warn if any Euler channel approaches the +-pi wrap (see the module docstring)."""
    if "rpy" not in blocks or feats.size == 0:
        return
    start = sum(len(GEOM_BLOCK_NAMES[b]) for b in blocks[:blocks.index("rpy")])
    block = feats[:, start:start + len(RPY_NAMES)]
    worst = np.abs(block).max(axis=0)
    for name, value in zip(RPY_NAMES, worst):
        if value > RPY_WRAP_WARN_RAD:
            logger.warning(
                "wb_geom%s: %s reaches %.1f deg, past the %.0f deg wrap warning threshold. "
                "Euler channels jump by 2*pi across +-180 deg and the loss would chase the "
                "jump with nothing in the shapes to say so — check this task's wrist range "
                "before trusting the 'rpy' block, or train a mode without it.",
                label, name, np.rad2deg(value), np.rad2deg(RPY_WRAP_WARN_RAD))


def noise_sigma(layout, *, xy_m: float, rpy_rad: float) -> np.ndarray:
    """``(state_dim,)`` per-column noise sigma for a whole-body layout. Zero elsewhere.

    Built from the LAYOUT rather than from a slice literal, so a state re-composition cannot
    leave the jitter landing on a gripper column. The gripper slots are deliberately zero:
    the encoder opening is a direct measurement with no posture-space gap to absorb, and it is
    the one state channel the action also predicts.
    """
    sigma = np.zeros(int(layout.state_dim), dtype=np.float64)
    slots = layout.state_geom_slots()
    # Keyed on the block's UNITS, not on its name, so a new position-valued block gets the
    # metre sigma by being listed here once. `rel_xy3` is the `fivepoint_*` family's
    # translation-invariant sibling of `rel_xy` — same units, same physical claim ("how well
    # does the robot know where its hands are, in metres").
    per_block = {"rel_xy": float(xy_m), "rel_xy3": float(xy_m), "rpy": float(rpy_rad)}
    unknown = sorted(b for b in slots if b not in per_block and b != "joints")
    if unknown:
        # A block with no sigma entry would be silently un-jittered: the wrapper would still
        # build (other blocks are nonzero), the loss would look healthy, and the augmentation
        # would simply not cover the channel someone asked it to cover.
        raise ValueError(
            f"noise_sigma has no unit for state block(s) {unknown} in layout "
            f"{layout.mode!r}; add them to `per_block` with the right units (metres or "
            f"radians) rather than letting them go un-jittered")
    for block, value in per_block.items():
        if block in slots:
            sigma[slots[block]] = value
    return sigma
