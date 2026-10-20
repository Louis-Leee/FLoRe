"""Convert S1 action (32D, ABSOLUTE pelvis-frame) to holosoma's vr_3point wire.

The DP-Flow model is TRAINED with anchor-relative action chunks (GCL
``relative`` type: every row of the H-step chunk is relative to the single obs
anchor). The **server** un-relativizes each chunk to absolute pelvis-frame poses
before returning (``DPFlowChunkPolicy._chunk_to_absolute``) — this has to be
server-side because only the server knows the chunk's anchor; a drip-fed client
can't see chunk boundaries and would compound anchor-relative deltas into
runaway motion.

So by the time an action reaches this module it is ALREADY ABSOLUTE. This class
is therefore a STATELESS splitter: record the absolute pose (for the next
obs under ``obs_mode=last_target``) and reformat pos3+rot6d → wire
``vr_3point_pos[9] + vr_3point_orn[12]`` (raw xyzw quats). No integration.

Body order [left_wrist, right_wrist, head/torso] — matching holosoma's
``vr_3point_bodies``. The wire carries absolute pelvis-frame targets; S0's
ViveLiveSource drops them straight into the obs.
"""

from __future__ import annotations

import numpy as np

from flore.utils.rot6d import rotmat_to_quat_xyzw


ACTION_DIM = 32
REAL_DIMS = 29  # [L:10 | R:10 | H:9]

# Slice indices into the 32D action vector:
#   L: pos(0:3), rot6d(3:9), grip(9)
#   R: pos(10:13), rot6d(13:19), grip(19)
#   H: pos(20:23), rot6d(23:29)
#   pad: (29:32) = 0

_L_POS = slice(0, 3)
_L_ROT = slice(3, 9)
_L_GRIP = 9
_R_POS = slice(10, 13)
_R_ROT = slice(13, 19)
_R_GRIP = 19
_H_POS = slice(20, 23)
_H_ROT = slice(23, 29)


class ActionTo3pt:
    """Stateless splitter: ABSOLUTE S1 action → vr_3point wire arrays.

    The server already un-relativized the anchor-relative model output to
    absolute pelvis-frame poses (see the module docstring), so :meth:`update`
    does NOT integrate — it records the absolute pose and reformats it to the
    wire ``(pos[9], orn[12])``. The only state it holds is the last absolute
    pose, exposed as :attr:`current_state` for the next obs under
    ``obs_mode=last_target`` (what the model sees as "state" equals what it last
    commanded). Seeded from the S0 ready-pose home at construction.
    """

    def __init__(self, home_pos: np.ndarray, home_rotmat: np.ndarray):
        """Initialize with the S0 ready-pose homes.

        Parameters
        ----------
        home_pos : (3, 3) float
            Home positions for [left, right, head] in pelvis frame.
        home_rotmat : (3, 3, 3) float
            Home rotation matrices for [left, right, head].
        """
        assert home_pos.shape == (3, 3), f"expected (3,3), got {home_pos.shape}"
        assert home_rotmat.shape == (3, 3, 3), f"expected (3,3,3), got {home_rotmat.shape}"
        self._pos = home_pos.copy().astype(np.float64)
        self._rotmat = home_rotmat.copy().astype(np.float64)

    @staticmethod
    def from_flat_home(home_pos_9: np.ndarray, home_orn_12_xyzw: np.ndarray) -> "ActionTo3pt":
        """Construct from the flat wire arrays (pos[9], orn[12] xyzw) that
        holosoma publishes at START / ready-pose."""
        from flore.utils.rot6d import quat_xyzw_to_rotmat

        pos = np.asarray(home_pos_9, dtype=np.float64).reshape(3, 3)
        quats = np.asarray(home_orn_12_xyzw, dtype=np.float64).reshape(3, 4)
        rotmats = quat_xyzw_to_rotmat(quats)  # (3, 3, 3)
        return ActionTo3pt(pos, rotmats)

    def reset(self, home_pos: np.ndarray, home_rotmat: np.ndarray) -> None:
        """Re-seed to a new home (e.g. on episode reset)."""
        self._pos = np.asarray(home_pos, dtype=np.float64).reshape(3, 3).copy()
        self._rotmat = np.asarray(home_rotmat, dtype=np.float64).reshape(3, 3, 3).copy()

    def update(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split one ABSOLUTE 32D action into the wire arrays; return them.

        ⚠️ The action is ABSOLUTE pelvis-frame (pos3+rot6d per keypoint), NOT a
        delta. The server (DPFlowChunkPolicy._chunk_to_absolute) already
        un-relativized the anchor-relative model output against each chunk's
        anchor — this MUST happen server-side because only the server knows the
        chunk anchor; the drip-fed client can't see chunk boundaries and would
        COMPOUND anchor-relative deltas into runaway motion (the exact bug the
        end-to-end review caught). So the client is a STATELESS splitter: it
        just records the absolute pose (for the next obs) and reformats it to
        the wire (rot6d -> xyzw quat). No integration, no accumulation.

        Returns
        -------
        pos : (9,) float32   Absolute pelvis-frame [Lx,Ly,Lz, Rx,Ry,Rz, Hx,Hy,Hz].
        orn : (12,) float32  Absolute orientations [Lxyzw, Rxyzw, Hxyzw].
        """
        from flore.utils.rot6d import rot6d_to_rotmat

        a = np.asarray(action, dtype=np.float64)
        for i, (pos_sl, rot_sl) in enumerate(
            [
                (_L_POS, _L_ROT),
                (_R_POS, _R_ROT),
                (_H_POS, _H_ROT),
            ]
        ):
            self._pos[i] = a[pos_sl]
            self._rotmat[i] = rot6d_to_rotmat(a[rot_sl])

        # Convert to wire format.
        pos_flat = self._pos.reshape(9).astype(np.float32)
        orn_flat = np.empty(12, dtype=np.float32)
        for i in range(3):
            orn_flat[i * 4 : (i + 1) * 4] = rotmat_to_quat_xyzw(self._rotmat[i])
        return pos_flat, orn_flat

    @property
    def current_state(self) -> np.ndarray:
        """Current running target as a 32D state vector (same layout as action,
        for feeding back as obs under obs_mode=last_target). Pad dims = 0."""
        state = np.zeros(ACTION_DIM, dtype=np.float32)
        from flore.utils.rot6d import rotmat_to_rot6d

        for i, (pos_sl, rot_sl) in enumerate(
            [
                (_L_POS, _L_ROT),
                (_R_POS, _R_ROT),
                (_H_POS, _H_ROT),
            ]
        ):
            state[pos_sl] = self._pos[i].astype(np.float32)
            state[rot_sl] = rotmat_to_rot6d(self._rotmat[i]).astype(np.float32)
        return state

    @property
    def gripper_from_action(self) -> tuple[float, float]:
        """Extract the left/right gripper values from the LAST action processed.

        Returns (left_grip, right_grip) in [0,1] (normalized). Head gripper
        slot is always 0 and ignored.
        """
        return 0.0, 0.0  # TODO(gripper): extract from action once gripper channel wired
