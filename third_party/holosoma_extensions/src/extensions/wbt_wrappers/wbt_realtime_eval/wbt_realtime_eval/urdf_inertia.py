"""Re-derive MJCF body inertials from the TRAINING URDF (mass parity).

Why this exists
---------------
The browser render scenes (``scene_g1_29dof_wbt_plane*.xml`` →
``g1_29dof_yam.xml``) are hand-maintained MJCF translations of the URDF that
IsaacSim trains on. MuJoCo has no fixed joints, so the MJCF collapses each
fixed-joint subtree into its parent body and hard-codes the lumped
``mass``/``ipos``/``fullinertia``. Those literals are a SNAPSHOT: when
``g1_29dof_yam.urdf`` changed (the 2026-07-10 palm standoff 0.0295 → 0.0565 also
carried ``*_rubber_hand`` mass 0.17 → 1.107992 kg), the MJCF's *geoms* were
shifted +0.027 m to follow but the fused ``<inertial>`` was NOT recomputed.

Result: each ``*_wrist_yaw_link`` in the browser weighs 0.879804 kg where the
training URDF says 1.584577 kg — **0.705 kg too light per hand**. The reference
ghost is FK-driven so it is unaffected, but the live robot is stepped by real
MuJoCo dynamics, so the policy's arm torques lift a hand 44% too light: the arms
visibly ride HIGH, and the gravity-load / ``F_meas`` inertial (m·a) terms are
wrong. The policy is fine — the viz model was lying about the payload.

Rather than patch the MJCF in ``thirdparty/`` (gitignored, and clobbered by
``bootstrap-holosoma.sh``), we recompute the fused inertials from the URDF at
BUILD time and write them into the ``MjSpec`` before ``add_scene``. The URDF used
is the one BAKED INTO THE CHECKPOINT (ONNX ``robot_urdf`` metadata), so the
browser payload matches what that specific run trained against rather than
whatever happens to sit on disk.

The math
--------
For MJCF body ``B``, walk the URDF fixed-joint (and ``prismatic``, frozen at
q=0 — the gripper jaws, whose ``lower`` limit IS 0) closure from link ``B``,
STOPPING at any link that is itself an MJCF body (else e.g.
``left_foot_contact_point`` would be counted both in ``left_ankle_roll_link``
and in its own body). That makes the lumped sets an exact partition of the URDF
links. Then, in B's frame:

    M   = Σ mᵢ
    c   = Σ mᵢ (pᵢ + Rᵢ cᵢ) / M
    I_c = Σ [ (Rᵢ Cᵢ) Iᵢ (Rᵢ Cᵢ)ᵀ + mᵢ (‖dᵢ‖² 1 − dᵢ dᵢᵀ) ],  dᵢ = pᵢ + Rᵢ cᵢ − c

with ``(pᵢ, Rᵢ)`` link i's pose in B's frame and ``(cᵢ, Cᵢ, Iᵢ)`` its URDF
``<inertial>`` origin / rotation / inertia tensor. This is the same
parallel-axis sum the MJCF comments describe having done by hand — we just do it
from the current URDF instead of trusting a stale literal.

``I_c`` is expressed in the URDF LINK frame, so transplanting it requires the
MJCF body frame to equal that link frame. That is checked, not assumed
(:func:`_frame_mismatches`) — the G1 scenes agree to ~1.4e-6 (quat text
rounding).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from xml.etree.ElementTree import Element

import mujoco
import numpy as np
from loguru import logger

# Joint types whose child link is rigidly fused into the parent for inertia
# purposes. "fixed" is exact. "prismatic" covers the yam gripper jaws, which the
# MJCF also lumps at q=0 (their URDF limit lower bound IS 0, i.e. q=0 is the
# closed/default pose both IsaacSim and the MJCF use).
_FUSED_JOINT_TYPES = frozenset({"fixed", "prismatic"})

# Max position / rotation deviation between an MJCF body frame and its URDF
# joint-origin frame before we refuse to transplant that body's inertia. The G1
# MJCF matches to ~1.4e-6 (quat text rounding), so this is ~70x headroom over
# observed noise while still catching a genuinely different frame convention.
_FRAME_ATOL = 1e-4


@dataclass(frozen=True)
class FusedInertial:
    """Lumped inertial of one MJCF body, in that body's own frame."""

    mass: float
    com: np.ndarray
    """(3,) center of mass."""
    inertia: np.ndarray
    """(3, 3) inertia tensor ABOUT THE COM, in body axes."""
    links: tuple[str, ...]
    """URDF links lumped into this body (own link first, then the subtree)."""

    @property
    def fullinertia(self) -> list[float]:
        """MuJoCo ``fullinertia`` order: ixx iyy izz ixy ixz iyz."""
        i = self.inertia
        return [float(i[0, 0]), float(i[1, 1]), float(i[2, 2]), float(i[0, 1]), float(i[0, 2]), float(i[1, 2])]


@dataclass(frozen=True)
class InertialCorrection:
    """One applied body correction (for logging / tests)."""

    body: str
    mass_before: float
    mass_after: float
    com_shift: float
    """‖Δipos‖ in metres."""
    links: tuple[str, ...]


def _rot_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (Rz @ Ry @ Rx) → 3x3."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _read_origin(elem: Element | None) -> tuple[np.ndarray, np.ndarray]:
    """URDF ``<origin>`` → (xyz, R). Absent/partial attrs default to zero."""
    xyz_s, rpy_s = "0 0 0", "0 0 0"
    if elem is not None:
        origin = elem.find("origin")
        if origin is not None:
            xyz_s = origin.get("xyz") or xyz_s
            rpy_s = origin.get("rpy") or rpy_s
    xyz = np.array([float(v) for v in xyz_s.split()], dtype=np.float64)
    rpy = tuple(float(v) for v in rpy_s.split())
    return xyz, _rot_from_rpy(rpy)  # type: ignore[arg-type]


class _Urdf:
    """The three URDF facts this module needs, parsed once."""

    def __init__(self, urdf_text: str) -> None:
        root = ET.fromstring(urdf_text)

        # link -> (mass, com, com_rot, inertia_tensor)
        self.inertials: dict[str, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
        for link in root.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            if name is None or inertial is None:
                continue
            mass_el, tensor_el = inertial.find("mass"), inertial.find("inertia")
            if mass_el is None or tensor_el is None:
                continue
            com, com_rot = _read_origin(inertial)
            ixx = float(tensor_el.get("ixx", 0.0))
            iyy = float(tensor_el.get("iyy", 0.0))
            izz = float(tensor_el.get("izz", 0.0))
            ixy = float(tensor_el.get("ixy", 0.0))
            ixz = float(tensor_el.get("ixz", 0.0))
            iyz = float(tensor_el.get("iyz", 0.0))
            tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
            self.inertials[name] = (float(mass_el.get("value", 0.0)), com, com_rot, tensor)

        # parent -> [(child, xyz, R)] for FUSED joints only, and
        # child -> (xyz, R) for EVERY joint (the frame guard).
        self.fused_children: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
        self.child_frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for joint in root.findall("joint"):
            parent_el, child_el = joint.find("parent"), joint.find("child")
            if parent_el is None or child_el is None:
                continue
            parent, child = parent_el.get("link"), child_el.get("link")
            if parent is None or child is None:
                continue
            xyz, rot = _read_origin(joint)
            self.child_frame[child] = (xyz, rot)
            if joint.get("type") in _FUSED_JOINT_TYPES:
                self.fused_children.setdefault(parent, []).append((child, xyz, rot))


def fused_inertials_from_urdf(urdf_text: str, body_names: set[str]) -> dict[str, FusedInertial]:
    """Lump each URDF link named in ``body_names`` with its fixed-joint subtree.

    The closure STOPS at any link that is itself in ``body_names`` — that body
    carries its own inertial, so descending past it would double-count mass.
    Bodies whose link has no ``<inertial>`` (or is absent from the URDF) are
    omitted rather than zeroed.
    """
    urdf = _Urdf(urdf_text)

    out: dict[str, FusedInertial] = {}
    for body in sorted(body_names):
        if body not in urdf.inertials:
            continue
        # (link, pos_in_body_frame, rot_in_body_frame), DFS over fused joints.
        stack: list[tuple[str, np.ndarray, np.ndarray]] = [(body, np.zeros(3), np.eye(3))]
        parts: list[tuple[str, tuple[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]] = []
        while stack:
            link, pos, rot = stack.pop()
            if link in urdf.inertials:
                parts.append((link, urdf.inertials[link], pos, rot))
            for child, jxyz, jrot in urdf.fused_children.get(link, []):
                if child in body_names:
                    continue  # its own MJCF body owns it — do not double count
                stack.append((child, pos + rot @ jxyz, rot @ jrot))

        total_mass = float(sum(part[1][0] for part in parts))
        if total_mass <= 0.0:
            continue

        com = np.zeros(3)
        for _, (mass, link_com, _, _), pos, rot in parts:
            com += mass * (pos + rot @ link_com)
        com /= total_mass

        tensor = np.zeros((3, 3))
        for _, (mass, link_com, com_rot, link_tensor), pos, rot in parts:
            axes = rot @ com_rot  # link COM axes -> body axes
            offset = (pos + rot @ link_com) - com
            tensor += axes @ link_tensor @ axes.T
            tensor += mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))

        names = tuple([body] + sorted(part[0] for part in parts if part[0] != body))
        out[body] = FusedInertial(mass=total_mass, com=com, inertia=tensor, links=names)
    return out


def _frame_mismatches(spec: mujoco.MjSpec, urdf: _Urdf, bodies: set[str]) -> list[str]:
    """Bodies whose MJCF frame differs from the URDF joint-origin frame.

    ``I_c`` is expressed in URDF link axes; transplanting it into a body whose
    frame is rotated relative to that link would silently mis-orient the tensor.
    """
    bad: list[str] = []
    rot_buf = np.zeros(9)
    for body in spec.bodies:
        if body.name not in bodies:
            continue
        frame = urdf.child_frame.get(body.name)
        if frame is None:
            continue  # root link (pelvis) has no parent joint; frames coincide
        xyz, rot = frame
        if np.linalg.norm(np.asarray(body.pos, dtype=np.float64) - xyz) > _FRAME_ATOL:
            bad.append(body.name)
            continue
        mujoco.mju_quat2Mat(rot_buf, np.asarray(body.quat, dtype=np.float64))
        if np.linalg.norm(rot_buf.reshape(3, 3) - rot) > _FRAME_ATOL:
            bad.append(body.name)
    return bad


def align_spec_inertials_to_urdf(
    spec: mujoco.MjSpec, urdf_text: str, *, mass_atol: float = 1e-3
) -> list[InertialCorrection]:
    """Rewrite ``spec``'s body inertials to the URDF-derived fused values.

    Only bodies whose lumped mass differs by more than ``mass_atol`` (default
    1 g, which absorbs the MJCF's 6-significant-figure text rounding) are
    touched, so an already-correct scene is left byte-identical. A body whose
    MJCF frame disagrees with the URDF link frame is SKIPPED with a warning
    rather than corrupted.

    Mutates ``spec`` in place and returns the corrections applied.
    """
    body_names = {b.name for b in spec.bodies if b.name and b.name != "world"}
    fused = fused_inertials_from_urdf(urdf_text, body_names)
    if not fused:
        logger.warning(
            "URDF inertia alignment: no MJCF body matched a URDF link with an "
            "<inertial>; scene left unchanged (is this the right URDF?)."
        )
        return []

    skip = set(_frame_mismatches(spec, _Urdf(urdf_text), set(fused)))
    if skip:
        logger.warning(
            f"URDF inertia alignment: skipping {sorted(skip)} — MJCF body frame "
            f"differs from the URDF link frame by >{_FRAME_ATOL} m/rad, so the "
            f"URDF inertia tensor is not expressed in these bodies' axes."
        )

    corrections: list[InertialCorrection] = []
    for body in spec.bodies:
        target = fused.get(body.name)
        if target is None or body.name in skip:
            continue
        mass_before = float(body.mass)
        if abs(mass_before - target.mass) <= mass_atol:
            continue
        com_before = np.asarray(body.ipos, dtype=np.float64).copy()

        body.mass = target.mass
        body.ipos = target.com.tolist()
        # fullinertia is about the COM in BODY axes, so the principal-axis
        # fields must be neutral or MuJoCo would combine the two specs.
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        body.inertia = [0.0, 0.0, 0.0]
        body.fullinertia = target.fullinertia
        body.explicitinertial = True

        corrections.append(
            InertialCorrection(
                body=body.name,
                mass_before=mass_before,
                mass_after=target.mass,
                com_shift=float(np.linalg.norm(target.com - com_before)),
                links=target.links,
            )
        )

    if corrections:
        net = sum(c.mass_after - c.mass_before for c in corrections)
        logger.info(
            f"URDF inertia alignment: corrected {len(corrections)} body(ies), "
            f"net {net:+.4f} kg — browser payload now matches the training URDF."
        )
        for c in corrections:
            logger.info(
                f"  {c.body}: mass {c.mass_before:.6f} -> {c.mass_after:.6f} kg "
                f"({c.mass_after - c.mass_before:+.4f}), |Δcom| {c.com_shift * 100:.2f} cm, "
                f"lumped {list(c.links)}"
            )
    else:
        logger.info("URDF inertia alignment: scene already matches the training URDF (no change).")
    return corrections
