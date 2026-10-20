"""FIVE keypoint poses -> a whole G1 body, by per-limb inverse kinematics.

WHY THIS EXISTS, AND WHY IT IS NOT MOTION COMPLETION
====================================================
``diffusion_umi_5pt`` predicts five keypoint POSES; a viser page (offline or at the rig) wants
to draw a BODY. The 3-tracker presets solve that with motion completion — a second learned
model, in a second interpreter, that invents the 26 unconditioned DoF. With five points there
is nothing left to invent: the two hands, the chest and both feet pin every limb, so the body
is a KINEMATIC consequence of the chunk and an IK is the whole of it. No ``g1_mosyn`` env, no
backend subprocess, no 30-60 s JIT, no checkpoint.

⚠️ **DISPLAY ONLY.** Nothing published to ``:8770`` comes from here. The 5pt low level is
handed the five poses themselves (``hl_dense.FivePointWire``) and solves its own whole-body
problem with a balance objective this has none of. A drawn body that disagrees with the robot
by a null-space rotation is therefore expected and harmless; a drawn body that disagrees by
CENTIMETRES at a keypoint is a real error, which is why :attr:`KeypointIKResult.residual_mm`
is returned per keypoint per step rather than summarised away.

THE DECOMPOSITION: THE CHEST IS THE ROOT
========================================
``torso_link`` carries both arms and is reached from ``pelvis`` through the 3 waist joints; the
legs hang off ``pelvis`` directly. So with the waist HELD at its seed the whole problem splits
into one placement and four independent 6-DoF targets::

    root       T_root = T_chest_target  o  inv(T_pelvis->chest_keypoint(waist_seed))
    left arm   7 joints, target = left  TCP  - 0.242 m x_hat   -> left_wrist_yaw_link  (pelvis frame)
    right arm  7 joints, target = right TCP  - 0.242 m x_hat   -> right_wrist_yaw_link
    left leg   6 joints, target = left  foot (no lever)        -> left_ankle_roll_link
    right leg  6 joints, target = right foot                   -> right_ankle_roll_link

Two consequences worth stating because both are visible on the page:

* **The chest is matched EXACTLY, by construction** — it is the root placement, not a solve, so
  its residual is 0 to machine precision. That is the user-facing meaning of "treat the chest as
  the pelvis": the body is hung ON the predicted chest and the limbs reach from there.
* **The waist never moves.** A predicted chest pose that would need a torso bend is absorbed by
  the FLOATING ROOT instead — the drawn pelvis translates/rotates where the real robot's pelvis
  would stay planted and bend its waist. For a page whose subject is the five keypoints that is
  the right trade (every keypoint still lands where it was predicted); it is the wrong trade for
  anything that reads the drawn pelvis as a physical claim, so :attr:`KeypointIKResult.root_pos`
  is exposed and the callers label it.

The arms are 7-DoF against a 6-DoF target, so one null-space dimension is free and the solve
warm-starts from the previous step to keep it continuous — the same choice, for the same reason,
as ``scripts/hl_serving.py``'s single-arm ghost. The legs are 6-against-6 and generically
determined.

THE SOLVER IS THE GHOST'S, GENERALISED
======================================
Damped least squares on the geometric Jacobian, warm-started, joint-limit clamped — lifted from
``hl_serving.py``'s ``_ghost_dls``/``_ghost_arm_jacobian`` (verified there against finite
differences to 1.7e-6, see ``tests/test_ghost_arm_ik.py``) and generalised from one chain to
four. The analytic Jacobian is the reason this is affordable: ``yourdfpy.update_cfg`` costs
0.382 ms while a cached ``get_transform`` is ~0.4 us, so a finite-difference derivative would
be ~88% of the run time. One ``update_cfg`` per limb per iteration is the floor and this sits on
it.

⚠️ MINK IS DELIBERATELY NOT USED. It solves the HuMI MJCF — a different kinematic model, missing
the wrist-yaw and waist joints — and its solution renders in the yam URDF up to ~170 mm off the
target it was given (measured; see ``hl_serving.py``'s ghost comment). A body drawn against a
trajectory has to be frame-faithful to the trajectory, so the solve happens in the SAME URDF the
render loads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from flore.s1_policy.inference.ready_pose import (
    DEFAULT_HOME_ANCHOR_PATH,
    DEFAULT_YAM_URDF_PATH,
    ROLE_BODY,
    KeypointFK,
)
from flore.utils.rot6d import rotmat_to_rotvec

#: Layout role -> (URDF body the IK drives, name substrings that select its joint chain).
#:
#: Matched by SUBSTRING against ``actuated_joint_names`` rather than listed as literals, so the
#: chains track a URDF rename instead of drifting from it; :data:`CHAIN_DOF` then asserts the
#: count, which is what makes the loose match safe. ``head`` is absent on purpose — the chest is
#: the ROOT PLACEMENT (see the module docstring), not a chain.
CHAIN_SPEC: dict[str, tuple[str, tuple[str, ...]]] = {
    "left": ("left_wrist_yaw_link", ("left_shoulder", "left_elbow", "left_wrist")),
    "right": ("right_wrist_yaw_link", ("right_shoulder", "right_elbow", "right_wrist")),
    "left_foot": ("left_ankle_roll_link", ("left_hip", "left_knee", "left_ankle")),
    "right_foot": ("right_ankle_roll_link", ("right_hip", "right_knee", "right_ankle")),
}

#: How many actuated joints each chain must have. An arm is 7 (redundant against a 6-DoF
#: target); a leg is 6 (generically determined). A mismatch means the URDF changed and the
#: substrings now select the wrong set — refused by name rather than solved with a wrong chain.
CHAIN_DOF: dict[str, int] = {"left": 7, "right": 7, "left_foot": 6, "right_foot": 6}

#: The role whose pose becomes the floating ROOT. Named rather than inlined because "the chest
#: is the root" is the one design decision on this page that a reader cannot infer from a shape.
ROOT_ROLE = "head"

#: The five roles this module solves for, in the layout's own order.
IK_ROLES: tuple[str, ...] = (ROOT_ROLE, *CHAIN_SPEC)

#: DLS damping, SCALED BY THE RESIDUAL rather than fixed, and this is a real departure from
#: ``_ghost_dls``' constant 0.05 — made because the constant measurably does not converge here.
#:
#: A fixed lambda has to be large enough to stay stable on the biggest step it will ever see, and
#: it then DOMINATES once the error is small: ``J^T (J J^T + lambda I)^-1 e`` shrinks toward
#: ``J^T e / lambda``, so the last millimetres arrive geometrically slowly. On a hand-held demo
#: that never mattered, because the single-arm ghost followed a SMOOTH recorded path. A 5pt
#: policy's chunk is not smooth — measured worst step-to-step hand jump **215 mm** on this
#: checkpoint — and at 0.05 the arms stalled at **51-56 mm** after 4 warm iterations while more
#: iterations kept improving (31 mm at 12, 13 mm at 40, 5.8 mm at 120). That is an UNCONVERGED
#: solve, not an unreachable target, and drawing it would have read as "the IK cannot reach the
#: prediction" — a claim about the model that was actually a claim about the solver.
#:
#: ``lambda_eff = max(DAMPING_MIN, DAMPING * min(1, |e_p| / DAMPING_KNEE))``: full damping for a
#: far target (stable on a 215 mm jump), near-Gauss-Newton for a close one (fast last mile).
DAMPING = 0.05
DAMPING_MIN = 5e-4
#: Position error (m) at and above which the damping is at its full :data:`DAMPING`.
DAMPING_KNEE = 0.05
#: Convergence: stop when the position error is under 0.2 mm AND the rotation error under
#: 2 mrad (0.11 deg). Both are two orders below the model's own accuracy (this checkpoint's
#: aggregate position error is 40 mm), so a converged step is exact for display purposes and the
#: iteration budget is spent only on the hard ones.
TOL_POS_M = 2e-4
TOL_ROT_RAD = 2e-3
#: Iteration CAPS for the FIRST step of a chunk (cold, seeded from ``seed_dof``) and for each
#: subsequent one (warm, seeded from its predecessor). Caps, not counts: the loop breaks on
#: :data:`TOL_POS_M`, so a smooth chunk costs a few iterations per limb and only a jerky one
#: pays the cap. Sized from the measurement above — 24 warm brings the jerkiest predicted chunk
#: inside a few mm while a target chunk still exits in 3-5.
ITERS_FIRST = 40
ITERS_WARM = 24


@dataclass(frozen=True)
class KeypointIKResult:
    """One chunk solved. Every array's leading axis is the action step.

    ``cfg`` is what ``ViserUrdf.update_cfg`` wants (the URDF's own actuated-joint order,
    including the two prismatic jaw joints); ``dof`` is the same solution in the 29-element SDK
    order, so a caller can FK it back through :class:`~ready_pose.KeypointFK` and check itself.
    The two are the same numbers under :func:`live_robot.build_sdk_to_urdf_index`'s remap.
    """

    cfg: np.ndarray            # (H, n_urdf) actuated-joint config
    dof: np.ndarray            # (H, 29) SDK order
    root_pos: np.ndarray       # (H, 3) floating root (the URDF's `pelvis`) in the target frame
    root_wxyz: np.ndarray      # (H, 4)
    #: Per-step, per-role residual of the SOLVED body against the pose it was ASKED for, in the
    #: role order of :data:`IK_ROLES`. The root role's is 0 by construction (see the module
    #: docstring) and is still reported, because a column that is structurally zero is how a
    #: reader learns which one was placed rather than solved.
    residual_mm: np.ndarray    # (H, 5)
    residual_deg: np.ndarray   # (H, 5)
    roles: tuple[str, ...] = IK_ROLES

    @property
    def n(self) -> int:
        return int(self.cfg.shape[0])

    def worst(self) -> tuple[float, float, str]:
        """``(worst mm, worst deg, the role the worst mm belongs to)`` over the whole chunk."""
        if self.residual_mm.size == 0:
            return 0.0, 0.0, ""
        flat = int(np.argmax(self.residual_mm))
        role = self.roles[flat % self.residual_mm.shape[1]]
        return (float(self.residual_mm.max()), float(self.residual_deg.max()), role)

    def describe(self) -> str:
        mm, deg, role = self.worst()
        return f"IK residual worst {mm:.1f} mm / {deg:.2f} deg ({role})"


class KeypointIK:
    """Per-limb DLS IK from five keypoint poses into one G1 configuration.

    Built off the SAME ``g1_29dof_yam.urdf`` and the SAME levers as
    :class:`~flore.s1_policy.inference.ready_pose.KeypointFK` — it holds one internally and uses
    it both for the seed's forward geometry and for the self-check, so an IK/FK lever
    disagreement cannot exist by construction.

    NOT thread-safe: a ``yourdfpy.URDF`` carries the configuration it was last set to, and every
    method here mutates it. One instance per solving thread (the viewer's worker owns one; the
    rig planner's page owns another).
    """

    def __init__(self, model, remap, fk: KeypointFK, chest_lever: np.ndarray,
                 hand_lever: np.ndarray) -> None:
        self._m = model
        self._remap = remap
        self.fk = fk
        self._chest_lever = np.asarray(chest_lever, dtype=np.float64).reshape(3)
        self._hand_lever = np.asarray(hand_lever, dtype=np.float64).reshape(3)
        names = list(model.actuated_joint_names)
        self.actuated_names: tuple[str, ...] = tuple(names)
        jmap = model.joint_map
        self._idx: dict[str, np.ndarray] = {}
        self._lo: dict[str, np.ndarray] = {}
        self._hi: dict[str, np.ndarray] = {}
        self._axes: dict[str, list[np.ndarray]] = {}
        self._childs: dict[str, list[str]] = {}
        self._body: dict[str, str] = {}
        for role, (body, keys) in CHAIN_SPEC.items():
            chain = [n for n in names if any(k in n for k in keys)]
            if len(chain) != CHAIN_DOF[role]:
                raise ValueError(
                    f"chain for {role!r} matched {len(chain)} joints {chain}, expected "
                    f"{CHAIN_DOF[role]}. The URDF's joint names changed; fix CHAIN_SPEC before "
                    f"solving — a chain of the wrong width mis-poses the limb silently.")
            nonrev = [n for n in chain if str(jmap[n].type) != "revolute"]
            if nonrev:
                raise ValueError(
                    f"{role!r} chain has non-revolute joint(s) {nonrev}; the analytic Jacobian "
                    f"below builds a revolute column only.")
            self._idx[role] = np.array([names.index(n) for n in chain], dtype=np.intp)
            self._lo[role] = np.array([float(jmap[n].limit.lower) for n in chain])
            self._hi[role] = np.array([float(jmap[n].limit.upper) for n in chain])
            self._axes[role] = [np.asarray(jmap[n].axis, dtype=np.float64) for n in chain]
            self._childs[role] = [jmap[n].child for n in chain]
            self._body[role] = body

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, *, urdf_path: str | None = None,
             home_anchor_path: str | None = None) -> tuple["KeypointIK | None", str]:
        """``(ik, reason)``. Never raises — a fresh clone has neither submodule nor yourdfpy."""
        urdf = urdf_path or DEFAULT_YAM_URDF_PATH
        if not os.path.exists(urdf):
            return None, f"the yam URDF is not checked out ({urdf})"
        try:
            import yourdfpy  # noqa: PLC0415 - optional, viz extra
        except ImportError as exc:
            return None, (f"yourdfpy is unavailable ({exc}); install the viz extra: "
                          'uv pip install --python .venv/bin/python -e ".[viz]"')
        fk, why = KeypointFK.load(urdf_path=urdf, home_anchor_path=home_anchor_path)
        if fk is None:
            return None, why
        try:
            from flore.s1_policy.serving.live_robot import (  # noqa: PLC0415
                build_sdk_to_urdf_index,
            )

            # A SECOND yourdfpy model, not the FK's: both carry the configuration they were last
            # set to, and this solver mutates its own thousands of times per chunk. Sharing one
            # would make every `fk.bodies()` call race the solve.
            model = yourdfpy.URDF.load(urdf, load_meshes=False, build_scene_graph=True)
            remap = build_sdk_to_urdf_index(model.actuated_joint_names)
            # The levers come from the FK's own table, so the IK targets the exact point the
            # dataset stores and the ready pose reports. Read through the public body map rather
            # than re-deriving: `_HAND_LEVER` is 0.242 m x_hat and the chest lever is
            # home_anchor.json's, and re-spelling either here is how the two would drift.
            chest = fk.lever(ROLE_BODY[ROOT_ROLE])
            hand = fk.lever(ROLE_BODY["left"])
            return cls(model, remap, fk, chest, hand), ""
        except Exception as exc:  # noqa: BLE001 - any input problem disables the feature
            return None, f"could not build the keypoint IK ({exc!r})"

    # ------------------------------------------------------------------ FK bits
    def _full_cfg(self, dof29: np.ndarray) -> np.ndarray:
        from flore.s1_policy.serving.live_robot import remap_dof_to_urdf  # noqa: PLC0415

        return np.asarray(remap_dof_to_urdf(np.asarray(dof29, dtype=np.float64).reshape(-1),
                                            self._remap), dtype=np.float64)

    def _dof29(self, cfg: np.ndarray) -> np.ndarray:
        """The inverse of :meth:`_full_cfg` for the 29 body joints (the jaws are not SDK DoF)."""
        out = np.zeros(29, dtype=np.float64)
        for j, s in enumerate(self._remap.src):
            if s >= 0:
                out[s] = cfg[j]
        return out

    def _pose(self, cfg: np.ndarray, body: str, frame_from: str = "pelvis"):
        self._m.update_cfg(cfg)
        T = np.asarray(self._m.get_transform(body, frame_from), dtype=np.float64)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def _jacobian(self, role: str, p_tip: np.ndarray) -> np.ndarray:
        """``(6, k)`` geometric Jacobian of ``role``'s body in the pelvis frame.

        ASSUMES the model is already at the configuration of interest (the caller has just FK'd
        it), so this is k cached ``get_transform`` lookups and no ``update_cfg``. Columns are the
        textbook revolute ``[z x (p_tip - p_i); z]``.
        """
        childs, axes = self._childs[role], self._axes[role]
        J = np.empty((6, len(childs)), dtype=np.float64)
        for i, (child, axis) in enumerate(zip(childs, axes)):
            T = np.asarray(self._m.get_transform(child, "pelvis"), dtype=np.float64)
            z = T[:3, :3] @ axis
            J[:3, i] = np.cross(z, p_tip - T[:3, 3])
            J[3:, i] = z
        return J

    def _solve_chain(self, cfg: np.ndarray, role: str, q: np.ndarray,
                     p_tgt: np.ndarray, R_tgt: np.ndarray, *, iters: int) -> np.ndarray:
        """DLS-step ``role``'s joints in ``cfg`` toward a pelvis-frame target. Never raises."""
        idx, body = self._idx[role], self._body[role]
        lo, hi = self._lo[role], self._hi[role]
        for _ in range(max(1, int(iters))):
            cfg[idx] = q
            p, R = self._pose(cfg, body)     # leaves the model AT this cfg for the Jacobian
            e_p = p_tgt - p
            e_r = R @ rotmat_to_rotvec(R.T @ R_tgt)
            if np.linalg.norm(e_p) < TOL_POS_M and np.linalg.norm(e_r) < TOL_ROT_RAD:
                break
            J = self._jacobian(role, p)
            e = np.concatenate([e_p, e_r])
            lam = max(DAMPING_MIN,
                      DAMPING * min(1.0, float(np.linalg.norm(e_p)) / DAMPING_KNEE))
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            if not np.all(np.isfinite(dq)):
                break
            q = np.clip(q + dq, lo, hi)
        cfg[idx] = q
        return q

    # ------------------------------------------------------------------ the solve
    def solve_chunk(self, poses: dict, *, seed_dof: np.ndarray | None = None,
                    jaw_m: dict | None = None, yield_gate=None) -> KeypointIKResult:
        """Solve every step of a chunk.

        Args:
          poses: ``role -> (pos (H, 3), rotmat (H, 3, 3))`` for all five of :data:`IK_ROLES`,
            in ONE frame — whatever frame the caller draws in. The result's ``root_pos`` /
            ``root_wxyz`` come back in that same frame, so the caller parents the URDF there
            and the drawn body lands on the drawn trajectory with no second transform.
          seed_dof: the 29 joint angles to warm-start step 0 from. Default: the FK's ready
            pose is NOT assumed here — ``None`` seeds from zeros, which is a straight-legged
            arms-down body and converges fine; a caller that has the deploy anchor's ``dof``
            should pass it, because a seed nearer the answer picks a nicer null space.
          jaw_m: ``role -> (H,) metres`` gripper aperture, written into the two prismatic jaw
            joints so the drawn hand opens. Absent roles keep the jaw shut.
          yield_gate: optional object with ``wait()``, called once per step. The rig page passes
            its GIL gate here so a solve never competes with an in-flight inference (measured on
            the single-arm ghost: 46 ms -> 3543 ms without it).

        Never raises for a hard target: an unreachable step is clamped at the joint limits and
        its residual is reported. It DOES raise for a missing role or a shape mismatch, because
        those are wiring errors and a body drawn from four of five points is a lie.
        """
        from flore.utils.rot6d import rotmat_to_quat_xyzw  # noqa: PLC0415

        missing = [r for r in IK_ROLES if r not in poses]
        if missing:
            raise ValueError(
                f"solve_chunk needs all {len(IK_ROLES)} roles {list(IK_ROLES)}; missing "
                f"{missing}. A five-point body cannot be drawn from fewer.")
        pos = {r: np.asarray(poses[r][0], dtype=np.float64).reshape(-1, 3) for r in IK_ROLES}
        rot = {r: np.asarray(poses[r][1], dtype=np.float64).reshape(-1, 3, 3) for r in IK_ROLES}
        H = pos[ROOT_ROLE].shape[0]
        for r in IK_ROLES:
            if pos[r].shape[0] != H or rot[r].shape[0] != H:
                raise ValueError(
                    f"role {r!r} has {pos[r].shape[0]}/{rot[r].shape[0]} steps but "
                    f"{ROOT_ROLE!r} has {H}; every role must be the same chunk")

        seed = (np.zeros(29, dtype=np.float64) if seed_dof is None
                else np.asarray(seed_dof, dtype=np.float64).reshape(-1))
        if seed.size != 29 or not np.all(np.isfinite(seed)):
            raise ValueError(f"seed_dof must be 29 finite joint angles, got shape {seed.shape}")
        cfg_seed = self._full_cfg(seed)

        # The root placement is a FIXED transform of the chest target for the whole chunk,
        # because the waist stays at the seed: T_pelvis->chest = T_pelvis->torso o translate(lever).
        self._m.update_cfg(cfg_seed)
        T_pt = np.asarray(self._m.get_transform("torso_link", "pelvis"), dtype=np.float64)
        R_pc = T_pt[:3, :3].copy()                                  # chest keypoint rotation
        p_pc = T_pt[:3, 3] + R_pc @ self._chest_lever               # chest keypoint position

        n_urdf = len(self.actuated_names)
        cfgs = np.tile(cfg_seed, (H, 1))
        root_pos = np.zeros((H, 3), dtype=np.float64)
        root_rot = np.zeros((H, 3, 3), dtype=np.float64)
        res_mm = np.zeros((H, len(IK_ROLES)), dtype=np.float64)
        res_deg = np.zeros((H, len(IK_ROLES)), dtype=np.float64)
        q = {role: cfg_seed[self._idx[role]].copy() for role in CHAIN_SPEC}

        for i in range(H):
            if yield_gate is not None:
                yield_gate.wait()
            # ROOT: place `pelvis` so the seed's chest keypoint coincides with the target's.
            R_root = rot[ROOT_ROLE][i] @ R_pc.T
            p_root = pos[ROOT_ROLE][i] - R_root @ p_pc
            root_pos[i], root_rot[i] = p_root, R_root
            cfg = cfg_seed.copy()
            iters = ITERS_FIRST if i == 0 else ITERS_WARM
            for role in CHAIN_SPEC:
                # Into the pelvis frame the chain is solved in, then off the stored point onto
                # the URDF body: a hand's dataset point is the fingertip TCP, the body is the
                # wrist. The feet get no lever (the keypoint IS the ankle origin).
                lever = self._hand_lever if role in ("left", "right") else np.zeros(3)
                R_t = R_root.T @ rot[role][i]
                p_t = R_root.T @ (pos[role][i] - p_root) - R_t @ lever
                q[role] = self._solve_chain(cfg, role, q[role], p_t, R_t, iters=iters)
            # Residuals: FK the SOLVED cfg once and compare every keypoint against its target,
            # in the caller's frame. This is the honest check — it re-applies the same levers
            # forward, so a lever mistake shows up here instead of cancelling.
            self._m.update_cfg(cfg)
            for ri, role in enumerate(IK_ROLES):
                body = ROLE_BODY[role]
                T = np.asarray(self._m.get_transform(body, "pelvis"), dtype=np.float64)
                lever = self.fk.lever(body)
                p_got = p_root + R_root @ (T[:3, 3] + T[:3, :3] @ lever)
                R_got = R_root @ T[:3, :3]
                res_mm[i, ri] = float(np.linalg.norm(p_got - pos[role][i]) * 1e3)
                res_deg[i, ri] = float(np.degrees(
                    np.linalg.norm(rotmat_to_rotvec(R_got.T @ rot[role][i]))))
            if jaw_m:
                self._write_jaws(cfg, jaw_m, i)
            cfgs[i] = cfg

        dof = np.stack([self._dof29(cfgs[i]) for i in range(H)]) if H else np.zeros((0, 29))
        wxyz = np.asarray(rotmat_to_quat_xyzw(root_rot), dtype=np.float64)
        wxyz = np.concatenate([wxyz[:, 3:4], wxyz[:, :3]], axis=1) if H else np.zeros((0, 4))
        assert cfgs.shape == (H, n_urdf)
        return KeypointIKResult(cfg=cfgs, dof=dof, root_pos=root_pos, root_wxyz=wxyz,
                                residual_mm=res_mm, residual_deg=res_deg)

    def _write_jaws(self, cfg: np.ndarray, jaw_m: dict, i: int) -> None:
        from flore.s1_policy.serving.live_robot import jaw_joint_from_width_m  # noqa: PLC0415

        for role, slot in (("left", self._remap.left_finger),
                           ("right", self._remap.right_finger)):
            widths = jaw_m.get(role)
            if slot < 0 or widths is None:
                continue
            w = np.asarray(widths, dtype=np.float64).reshape(-1)
            if w.shape[0] > i and np.isfinite(w[i]):
                cfg[slot] = jaw_joint_from_width_m(float(w[i]))

    # ------------------------------------------------------------------ self-check
    def check_round_trip(self, dof29: np.ndarray) -> tuple[float, float]:
        """``(worst mm, worst deg)`` of solving the FK of ``dof29`` back to itself.

        The one test that proves the IK's levers, chains and root placement agree with the FK
        the dataset and the wire are built on: FK a real configuration into five keypoints, hand
        those five to :meth:`solve_chunk`, and FK the answer back. A lever mistake, a swapped
        chain or a wrong root transform all show up here as millimetres. Called at startup by
        both pages, so a broken IK says so before it draws anything.
        """
        q = np.asarray(dof29, dtype=np.float64).reshape(-1)
        pos_b, rot_b = self.fk.bodies(q)
        poses = {role: (pos_b[ROLE_BODY[role]][None, :], rot_b[ROLE_BODY[role]][None, ...])
                 for role in IK_ROLES}
        out = self.solve_chunk(poses, seed_dof=q)
        return float(out.residual_mm.max()), float(out.residual_deg.max())


def load_keypoint_ik(**kwargs) -> tuple[KeypointIK | None, str]:
    """``(ik, reason)`` — module-level alias, mirroring ``wb_geom.load_geom_fk``'s shape."""
    return KeypointIK.load(**kwargs)


__all__ = [
    "CHAIN_DOF",
    "CHAIN_SPEC",
    "IK_ROLES",
    "ROOT_ROLE",
    "KeypointIK",
    "KeypointIKResult",
    "load_keypoint_ik",
    "DEFAULT_HOME_ANCHOR_PATH",
    "DEFAULT_YAM_URDF_PATH",
]
