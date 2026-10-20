"""The Bota F/T sensor frame, the ONE rotation into yam-TCP axes, and the meas->cmd SIGN.

Everything on disk and on the wire is in the **sensor** frame: the converter stamps
``attrs["hl_force_frame"] == "bota_sensor"``, ``HLDataset`` passes the force columns through
un-relativized on the strength of that attr, and ``hl_serving``'s ``F^cmd`` wire carries the
same triple. That is deliberate and is NOT changed here — this module holds the two
conversions a viewer or a publisher needs on top of it: the sensor->TCP axis relabel
(:data:`TCP_FROM_SENSOR`, so an arrow is drawn in the frame the operator reasons about instead
of implicitly asserting that the sensor and the TCP share axes), and the measured->commanded
sign flip (:data:`CMD_FROM_MEAS_SIGN`, below).

The measured relabel
--------------------
The Bota is bolted on rotated relative to the yam gripper, so the two frames do NOT share an
orientation::

    force +x  is the TCP's  +z
    force +y  is the TCP's  -y
    force +z  is the TCP's  +x

i.e. ``f_tcp = (f_sensor_z, -f_sensor_y, f_sensor_x)``, which is :data:`TCP_FROM_SENSOR`. It is
a PROPER rotation (det ``+1``, a 180 deg turn about ``(1, 0, 1)/sqrt(2)``), not a mirror — a
mirror would flip the handedness of every torque it touched.

Why this is not a guess
-----------------------
Verified against ``session_2026.08.12_merged_v3`` (115 episodes, 4150 contact frames above
5 N) by fitting, with no prior, the sensor->TCP rotation AND a fixed world direction that
jointly make ``world_R_tcp @ R @ f_sensor`` point the same way at every contact frame — i.e.
asking the data where the wiped SURFACE is. The fit recovered a normal only **7.7 deg off
world-vertical** (the table), and landed **19.9 deg from the relabel above** versus
**169.0 deg from identity**. Ranked over all 24 proper signed axis permutations by how
orientation-invariant the resulting world direction is, this relabel is **1st** and identity
is **21st**. The residual ~20 deg is the tangential (friction) part of a wiping stroke, which
no rotation can remove.

The sign matters as much as the axes: under this relabel a press reads as a force pointing
**out of** the table (the reaction the surface applies to the tool), whereas identity read as
a force pointing 165 deg away, i.e. straight INTO the table. Every viewer that drew the raw
triple against a TCP-oriented frame was therefore drawing it both permuted and backwards.

Torque: rotated, NOT re-referenced
----------------------------------
:func:`tcp_from_sensor_wrench` applies the same rotation to the torque triple, which is
correct for a change of AXES. It does **not** add the ``r x f`` term for the sensor-origin ->
TCP-origin lever, because that lever is not part of the measured relabel above. So a drawn
torque arrow is "the sensor's torque, in TCP axes" — not the torque about the fingertip. Force
is unaffected by the origin and needs no such caveat, which is why force is what the viewers
draw by default.

Measured wrench vs COMMANDED wrench: the sign
---------------------------------------------
Everything FLoRe stores or predicts is ``F_meas`` — what the Bota READ while a human did the
task. The low level does not take a measurement, it takes a **command**, and its force loop is

    e = F_cmd + F_meas   ->   0

(holosoma ``wbt_wrappers_inference/utils/force_deploy_math.py::servo_step``, and the same
relation is UniFP Eq.2's ``x_target = x_contact + (F_cmd + F_ext)/K`` with ``F_ext = F_meas``).
So at the fixed point the measured force OPPOSES the command, and the command that reproduces a
demonstrated contact is the NEGATIVE of the wrench that demonstration measured::

    F_cmd = -F_meas          (:data:`CMD_FROM_MEAS_SIGN`)

This is a sign, not a frame: the low level applies the SAME ``sensor_force_to_pelvis`` chain and
the same ``HOLOSOMA_FORCE_MEAS_SIGN`` to the wire's ``F^cmd`` as it does to its own ``F_meas``,
so there is no second convention hiding on that side to cancel this one. And because a scalar
commutes with every rotation, it does not matter whether the flip is applied before or after
:data:`TCP_FROM_SENSOR` — which is what makes it safe to apply planner-side (where the model's
output semantics are known) while the rotation stays low-level-side (where the joint readback
is).

Concretely: a press reads as ``F_meas`` pointing OUT of the table (the reaction on the tool, see
above), and the command that presses into it must point INTO the table. An un-flipped command
would ask the robot to pull away from the surface the demo pressed on, and the servo error
``F_cmd + F_meas`` would be ``2 * F_meas`` instead of ~0 — i.e. it winds up rather than settles.
Every viewer therefore draws the COMMAND by default (``as_command=True``), so what is on the
page is the direction the arm will push.
"""

from __future__ import annotations

import numpy as np

#: Sensor -> TCP axis relabel: ``f_tcp = TCP_FROM_SENSOR @ f_sensor``.
#:
#: SYMMETRIC, hence its own inverse (``R == R.T == R^-1``), so there is deliberately no
#: separate ``SENSOR_FROM_TCP`` constant to drift out of sync — see
#: :func:`sensor_from_tcp_force`. Pinned by ``tests/test_force_frame.py``.
TCP_FROM_SENSOR = np.array(
    [
        [0.0, 0.0, 1.0],   # TCP x  <-  sensor +z
        [0.0, -1.0, 0.0],  # TCP y  <-  sensor -y
        [1.0, 0.0, 0.0],   # TCP z  <-  sensor +x
    ],
    dtype=np.float64,
)

#: Human-readable form of :data:`TCP_FROM_SENSOR`, for GUI labels and log lines, so a viewer
#: states the convention it is drawing rather than leaving the operator to infer it.
TCP_FROM_SENSOR_NOTE = "sensor +x->TCP +z, +y->TCP -y, +z->TCP +x"

#: ``F_cmd = CMD_FROM_MEAS_SIGN * F_meas`` — the measured->commanded flip derived in the module
#: docstring from the low level's ``e = F_cmd + F_meas -> 0``. A scalar, so it commutes with
#: :data:`TCP_FROM_SENSOR` and is its own inverse (which is why there is no ``MEAS_FROM_CMD``
#: constant to drift). Pinned against holosoma's ``servo_step`` by
#: ``tests/test_deploy_force_contract.py``.
CMD_FROM_MEAS_SIGN = -1.0

#: Human-readable form of :data:`CMD_FROM_MEAS_SIGN`, shown next to
#: :data:`TCP_FROM_SENSOR_NOTE` wherever a viewer states what it is drawing. The two together
#: are the complete answer to "which way is this arrow pointing, and whose force is it?".
CMD_FROM_MEAS_NOTE = "F^cmd = -F_meas (the low level drives F^cmd + F_meas -> 0)"


def _rotate_triples(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a (3, 3) rotation to the LAST axis of an ``(..., 3)`` array."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"expected trailing axis of size 3, got shape {array.shape}")
    return np.einsum("ij,...j->...i", matrix, array)


def tcp_from_sensor_force(force: np.ndarray) -> np.ndarray:
    """Rotate an ``(..., 3)`` sensor-frame force into TCP axes."""
    return _rotate_triples(force, TCP_FROM_SENSOR)


def sensor_from_tcp_force(force: np.ndarray) -> np.ndarray:
    """Rotate an ``(..., 3)`` TCP-axes force back into the sensor frame.

    Shares :data:`TCP_FROM_SENSOR` because that matrix is its own inverse; spelling the
    inverse as a second literal is exactly how the two directions would drift apart.
    """
    return _rotate_triples(force, TCP_FROM_SENSOR.T)


def tcp_from_sensor_wrench(wrench: np.ndarray) -> np.ndarray:
    """Rotate an ``(..., 6)`` sensor-frame ``[Fx Fy Fz Tx Ty Tz]`` into TCP axes.

    Both triples get the same rotation. The torque is re-EXPRESSED, not re-referenced to the
    TCP origin (no ``r x f`` lever term) — see the module docstring.
    """
    array = np.asarray(wrench, dtype=np.float64)
    if array.shape[-1] != 6:
        raise ValueError(f"expected trailing axis of size 6, got shape {array.shape}")
    out = np.empty_like(array)
    out[..., 0:3] = tcp_from_sensor_force(array[..., 0:3])
    out[..., 3:6] = tcp_from_sensor_force(array[..., 3:6])
    return out


def cmd_from_meas_force(force: np.ndarray) -> np.ndarray:
    """A MEASURED ``(..., 3)`` force -> the force COMMAND that reproduces it.

    Pure sign flip (:data:`CMD_FROM_MEAS_SIGN`), frame-agnostic: it may be applied on either
    side of :data:`TCP_FROM_SENSOR`, and it is its own inverse, so this same function reads a
    command back as the measurement it targets. Returns a fresh array — the callers publish or
    draw the result, and must never alias the model's own output buffer.
    """
    array = np.asarray(force, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"expected trailing axis of size 3, got shape {array.shape}")
    return CMD_FROM_MEAS_SIGN * array


def cmd_from_meas_wrench(wrench: np.ndarray) -> np.ndarray:
    """A MEASURED ``(..., 6)`` ``[Fx Fy Fz Tx Ty Tz]`` -> the commanded wrench.

    BOTH triples flip. The wire is force-only (holosoma's F^cmd buffer and Jᵀ servo carry no
    torque), so this exists for the VIEWERS: a torque arrow or chart drawn with the opposite
    sign from the force arrow beside it would read as a wrench no rigid body can produce.
    """
    array = np.asarray(wrench, dtype=np.float64)
    if array.shape[-1] != 6:
        raise ValueError(f"expected trailing axis of size 6, got shape {array.shape}")
    return CMD_FROM_MEAS_SIGN * array


def world_force_from_sensor(force: np.ndarray, world_rotmat_tcp: np.ndarray) -> np.ndarray:
    """Sensor-frame force -> the WORLD/display frame a viewer draws arrows in.

    ``world_rotmat_tcp`` is the orientation the TCP is DRAWN at — a ``(3, 3)`` matrix, or
    ``(..., 3, 3)`` broadcasting against ``force``'s leading axes. Composing both legs here
    (``R_world_tcp @ TCP_FROM_SENSOR @ f``) is what keeps an arrow attached to the pose the
    viewer actually rendered: pass the display-rebased rotation and the arrow follows the
    rebase, pass the raw absolute one and it stays in the recorded world frame.
    """
    tcp = tcp_from_sensor_force(force)
    matrix = np.asarray(world_rotmat_tcp, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"world_rotmat_tcp must have trailing shape (3, 3), got {matrix.shape}")
    return np.einsum("...ij,...j->...i", matrix, tcp)


__all__ = [
    "CMD_FROM_MEAS_NOTE",
    "CMD_FROM_MEAS_SIGN",
    "TCP_FROM_SENSOR",
    "TCP_FROM_SENSOR_NOTE",
    "cmd_from_meas_force",
    "cmd_from_meas_wrench",
    "sensor_from_tcp_force",
    "tcp_from_sensor_force",
    "tcp_from_sensor_wrench",
    "world_force_from_sensor",
]
