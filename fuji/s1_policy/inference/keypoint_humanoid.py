"""The offline viewer's FIVE-POINT G1 body: an IK overlay, not a second model.

A sibling of :mod:`fuji.s1_policy.inference.mc_humanoid` with the same six-method surface
(``available`` / ``build_gui`` / ``on_rollout`` / ``poll`` / ``render_at`` / ``clear`` /
``stop``), so ``viewer.py`` drives both from the same five call sites and neither knows about the
other. What differs is everything underneath::

    --motion-completion   3 poses -> a LEARNED model in the g1_mosyn interpreter -> 29 joints
    --keypoint-ik         5 poses -> per-limb DLS in the render URDF               -> 29 joints

which is the whole reason ``diffusion_umi_5pt`` exists: five points leave nothing to infer, so
the body is a kinematic consequence of the chunk (see
:mod:`fuji.s1_policy.inference.keypoint_ik`). No conda env, no subprocess, no checkpoint, ~0.3 s
per 50-step target chunk on one core.

TWO BODIES, THE SAME COLOURS AS EVERY OTHER PAGE
================================================
Violet = the PREDICTION, pale blue = the dataset TARGET, byte-identical to
``wholebody_viewer``'s two-body page and to ``hl_serving``'s ghost. That is the point of drawing
both: the five trajectory lines already answer "where does each point go", and the two bodies
answer the question a keypoint page otherwise cannot — *is the posture those points imply a
posture at all*, and does it differ from the demonstration's. The target body defaults ON for
that reason, and can be hidden when the two overlap.

⚠️ THE FRAME IS A PREREQUISITE, NOT A PREFERENCE — this overlay REFUSES the display-home frame.
The five roles' anchors are **not in one frame in the dataset**: the three tracked ones carry the
operator's recorded world pose, while the two feet are ``KeypointFK(wb_action_joints)`` in the
completion's PELVIS frame, and nothing stored relates them (``wb_root_pose`` is provenance and is
never read). Decoded against their own anchors and drawn together, the feet land metres from the
body — measured, the IK residual was **543 mm**. What makes five points one body is re-basing
every role onto ONE physically coherent set of homes, which is exactly
``--display-frame g1-ready-pose``: each role's step 0 lands on the pose the robot holds after the
operator presses **A**, and the chunk is the displacement from there. With that frame the IK
closes on the TARGET to **0.2-0.4 mm** on all five points. So the flag is required and the panel
says so instead of drawing a body that is 543 mm wrong in a way no residual on screen would
explain.

A KEYPOINT CHECKPOINT WITH FEWER THAN FIVE ROLES (``left_only`` first of all) still gets a body:
the roles it does not predict are **HELD at the ready pose** — the deploy anchor's own pose for that
slot, in the same ``g1-ready-pose`` frame the predicted roles are re-based onto — for BOTH series,
so the violet and the pale body differ only in the predicted hand(s). That is the offline twin of
what ``scripts/hl_serving_5pt.py`` publishes for such a checkpoint (``FivePointDirectWire.held``),
and the panel names the held roles so a motionless right arm is not read as a prediction of one.
``held_poses`` is the viewer's to supply; without it a missing role still disables the body by
name, exactly as before.

⚠️ THE DRAWN PELVIS IS NOT A PHYSICAL CLAIM. The chest keypoint is the ROOT (the user-facing
meaning of "treat the chest as the pelvis"), the waist joints stay at the seed, and a chest pose
that would need a torso bend is absorbed by the floating root instead. Every keypoint still lands
where it was predicted — which is what this page is about — but the drawn pelvis translates and
yaws where a real G1's would stay planted. The panel reports the root's travel for that reason.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fuji.s1_policy.inference.keypoint_ik import IK_ROLES, ROOT_ROLE, KeypointIK
from fuji.s1_policy.serving.gil_yield import YieldGate
from fuji.utils.rot6d import quat_wxyz_to_xyzw, quat_xyzw_to_rotmat

#: The PREDICTION's colour: translucent violet. Pinned equal to
#: ``wholebody_viewer.PREDICTION_RGBA`` (and therefore to ``hl_serving._GHOST_RGBA``) by a test —
#: one vocabulary across the offline pages and the rig. Same NAMES as that module too, so a reader
#: comparing the two files is not also comparing two spellings.
PREDICTION_RGBA = (0.62, 0.36, 0.95, 0.45)
#: The TARGET's: pale desaturated blue, dimmer so the prediction reads as the foreground.
TARGET_RGBA = (0.55, 0.62, 0.72, 0.30)

#: Which drawn series the body follows. Both are drawn by default; the dropdown here picks which
#: one the RESIDUAL readout reports, because a single "IK residual" line for two bodies would be
#: ambiguous in exactly the case that matters (a prediction the arm cannot reach).
SERIES = ("prediction", "target")
#: The two ways a body can be drawn. ``tinted`` = the translucent violet/pale bodies above (the
#: default: two overlapping bodies stay readable). ``real`` = the URDF's OWN materials — Unitree's
#: white/dark scheme, the yam's green jaws, the F/T sensor's dark body — drawn OPAQUE, for a page or
#: a screenshot that should look like the robot. A mesh's colour cannot be changed after creation
#: and two ``ViserUrdf`` under ONE root would collide on ``<root>/visual/<link>``, so the real body
#: is a SECOND ``ViserUrdf`` per series under its own root, created lazily on the first toggle
#: (48 meshes per body over the websocket) and posed from the very same solve.
VARIANTS = ("tinted", "real")
REAL_MESH_LABEL = "Real G1 mesh (URDF materials)"
_MESH_NOTE_TINTED = "tinted: violet = PREDICTION · pale = TARGET (translucent)"
_MESH_NOTE_REAL = "URDF materials, opaque — both bodies look alike; untick one above to compare"


def _urdf_geometry_is_invisible(geom) -> bool:
    """True for a URDF visual whose material alpha is 0 (the yam URDF's foot contact points, IMUs and
    mid360 placeholders: 12-face boxes). yourdfpy keeps them as alpha-0 vertex colours, which the GLB
    path does not honour — drawn with the real materials they come out as small BLACK boxes on the
    feet and torso — so the real-mesh body drops them. The tinted body never sees them either: the
    colour override paints every mesh alike, so this only changes what was already invisible."""
    visual = getattr(geom, "visual", None)
    main_color = getattr(visual, "main_color", None)
    if main_color is None:
        return False
    try:
        rgba = np.asarray(main_color).reshape(-1)
    except Exception:  # noqa: BLE001 - an exotic visual is simply kept
        return False
    return rgba.shape[0] == 4 and int(rgba[3]) == 0


@dataclass(frozen=True)
class _Solved:
    """One rollout solved for both series. Immutable; the render loop only indexes it."""

    generation: int
    per_series: dict            # series -> KeypointIKResult
    identity: str
    ms: float


class KeypointHumanoid:
    """Five predicted keypoints -> two G1 bodies on the offline viewer's page (translucent tints by
    default; the panel's "Real G1 mesh" checkbox swaps in the URDF's own materials, opaque).

    One instance per viewer. The solve runs on a worker thread (~0.3-1.0 s per chunk for the two
    series) so dragging the Action-step slider never waits on it, and the worker owns its OWN
    :class:`KeypointIK` — a ``yourdfpy.URDF`` carries the configuration it was last set to, so a
    shared one would race the render thread's ``update_cfg``.
    """

    def __init__(self, server, *, enabled: bool = False, urdf_path: str | None = None,
                 home_anchor_path: str | None = None, log=None, show_target: bool = True,
                 series: str = "prediction", seed_dof: np.ndarray | None = None,
                 frame_ok: bool = True, frame_note: str = "",
                 held_poses: dict | None = None) -> None:
        self.server = server
        self.enabled = bool(enabled)
        #: ``role -> (pos (3,), rotmat (3, 3))`` in the DISPLAY frame for every IK role the layout
        #: does not predict — the ready pose's own pose for that slot (see the module docstring).
        #: Broadcast over the chunk for both series in :meth:`on_rollout`. Empty = nothing held.
        self.held_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role, (pos, rot) in (held_poses or {}).items():
            p = np.asarray(pos, dtype=np.float64).reshape(-1)
            r = np.asarray(rot, dtype=np.float64)
            if p.shape != (3,) or r.shape != (3, 3):
                raise ValueError(f"held pose for {role!r} must be a (3,) position and a (3, 3) "
                                 f"rotation, got {p.shape} / {r.shape}")
            self.held_poses[str(role)] = (p.copy(), r.copy())
        self.log = log or (lambda m: print(f"[s1-inference][ik] {m}", flush=True))
        if str(series) not in SERIES:
            raise ValueError(f"series must be one of {SERIES}, got {series!r}")
        self._series = str(series)
        self._show_target = bool(show_target)
        self._seed_dof = None if seed_dof is None else np.asarray(seed_dof, dtype=np.float64)
        self._frame_ok = bool(frame_ok)
        self._frame_note = str(frame_note)
        self.disabled_reason = "" if self.enabled else "not requested (--keypoint-ik)"
        self.ik: KeypointIK | None = None
        self._urdf_path = urdf_path
        self._anchor_path = home_anchor_path
        #: ``series -> variant -> FrameHandle | None`` and ``series -> variant -> ViserUrdf | None``:
        #: the ``real`` slot stays None until the checkbox first asks for it (see VARIANTS).
        self._roots: dict[str, dict[str, Any]] = {}
        self._urdfs: dict[str, dict[str, Any]] = {}
        self._urdf_file: Path | None = None
        self._real_failed: str = ""   # non-empty = the real body could not be built; stay tinted
        self._gen = 0
        self._solved: _Solved | None = None
        self._drawn = -1
        self._q: queue.Queue = queue.Queue(maxsize=1)
        #: GIL priority for the SAMPLER. The solve is pure Python (yourdfpy FK + a 6x6 linear
        #: solve, thousands of times per chunk) while the viewer's rollout worker is a Python loop
        #: issuing CUDA launches, so the two fight for one interpreter. Measured on the rig page,
        #: which has the identical pair: a sample costs 386 ms with the solver idle and 3758 ms with
        #: it running — 9.7x, the same failure ``hl_serving``'s single-arm ghost hit (46 -> 3543 ms).
        #: Offline that is only slowness, but it is the SAME slowness, so it gets the same gate.
        self.infer_gate = YieldGate()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._gui: dict[str, Any] = {}
        if not self.enabled:
            return
        if not self._frame_ok:
            # The ONE refusal that is about a flag and not a missing prerequisite. Stated as its
            # own branch so the panel's reason names the flag to pass rather than a URDF path.
            self.disabled_reason = self._frame_note or (
                "the five-point body needs --display-frame g1-ready-pose (the five roles' "
                "anchors are not in one frame otherwise)")
            self.log(f"DISABLED: {self.disabled_reason}")
            return
        ik, why = KeypointIK.load(urdf_path=urdf_path, home_anchor_path=home_anchor_path)
        if ik is None:
            self.disabled_reason = why
            self.log(f"DISABLED: {why}")
            return
        self.ik = ik
        if self._seed_dof is not None:
            # The self-check that proves the IK's levers, chains and root placement agree with
            # the FK the dataset and the wire are built on. Run BEFORE anything is drawn, on the
            # very pose the chunk is decoded against, so a broken IK says so instead of quietly
            # drawing a body a few centimetres wrong.
            try:
                mm, deg = ik.check_round_trip(self._seed_dof)
                self.log(f"round-trip on the decode anchor: {mm:.4f} mm / {deg:.4f} deg "
                         f"({'OK' if mm < 1.0 else 'HIGH — the drawn body will sit off the plan'})")
            except Exception as exc:  # noqa: BLE001 - a self-check failure disables, never kills
                self.disabled_reason = f"the IK self-check failed ({exc!r})"
                self.log(f"DISABLED: {self.disabled_reason}")
                self.ik = None
                return
        self._build_scene(urdf_path)

    # ------------------------------------------------------------------ scene
    def _build_scene(self, urdf_path: str | None) -> None:
        from fuji.s1_policy.inference.keypoint_ik import DEFAULT_YAM_URDF_PATH  # noqa: PLC0415

        try:
            from viser.extras import ViserUrdf  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self.disabled_reason = f"viser.extras.ViserUrdf is unavailable ({exc!r})"
            self.log(f"DISABLED: {self.disabled_reason}")
            self.ik = None
            return
        path = Path(urdf_path or DEFAULT_YAM_URDF_PATH)
        self._urdf_file = path
        for series, rgba in (("target", TARGET_RGBA), ("prediction", PREDICTION_RGBA)):
            # A FRAME per body, because the root is part of the solution: the URDF renders in its
            # parent's coordinates, so the floating root goes on the frame and the 31 joint angles
            # go on the ViserUrdf. Hidden until the first solve lands (viser visibility
            # propagates to children, so one flag hides the whole body). Only the TINTED variant
            # is built here; the real-material one waits for the checkbox (see _ensure_real_body).
            name = f"/kp_ik/{series}"
            self._roots[series] = {
                "tinted": self.server.scene.add_frame(name, show_axes=False, visible=False),
                "real": None,
            }
            self._urdfs[series] = {
                "tinted": ViserUrdf(self.server, urdf_or_path=path, root_node_name=name,
                                    mesh_color_override=rgba),
                "real": None,
            }

    def _load_real_urdf(self):
        """The URDF with its own materials, loaded the way ``ViserUrdf`` loads a path (same filename
        handler, visual meshes only) and with the alpha-0 placeholder boxes removed. Returns
        ``(yourdfpy.URDF, dropped geometry names)``. One object per body: ``update_cfg`` sets the
        configuration ON the URDF object, so two bodies must not share one."""
        from functools import partial  # noqa: PLC0415

        import yourdfpy  # noqa: PLC0415

        path = Path(self._urdf_file)
        urdf = yourdfpy.URDF.load(
            path, build_scene_graph=True, load_meshes=True,
            build_collision_scene_graph=False, load_collision_meshes=False,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
        )
        dropped = [name for name, geom in urdf.scene.geometry.items() if _urdf_geometry_is_invisible(geom)]
        for name in dropped:
            urdf.scene.delete_geometry(name)
        return urdf, dropped

    def _ensure_real_body(self, series: str) -> bool:
        """Create the real-material body for ``series`` on first use (loop thread). False when it
        cannot be built — the reason goes on the panel once and the tinted body stays."""
        if self._urdfs[series].get("real") is not None:
            return True
        if self._real_failed:
            return False
        try:
            from viser.extras import ViserUrdf  # noqa: PLC0415

            urdf, dropped = self._load_real_urdf()
            name = f"/kp_ik_real/{series}"
            root = self.server.scene.add_frame(name, show_axes=False, visible=False)
            # NO colour override: ViserUrdf then adds each mesh with its trimesh visual, i.e. the
            # URDF material, instead of a flat tinted add_mesh_simple.
            body = ViserUrdf(self.server, urdf_or_path=urdf, root_node_name=name)
        except Exception as exc:  # noqa: BLE001 - a missing mesh must not kill the page
            self._real_failed = f"{exc!r}"
            self.log(f"real G1 mesh unavailable, keeping the tinted body: {self._real_failed}")
            self._set("mesh", f"real mesh unavailable: {self._real_failed}")
            return False
        self._roots[series]["real"] = root
        self._urdfs[series]["real"] = body
        self.log(f"real G1 mesh built for {series} (URDF materials; {len(dropped)} alpha-0 placeholder "
                 f"meshes dropped)")
        return True

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return self.enabled and self.ik is not None and not self.disabled_reason

    def build_gui(self, gui) -> None:
        """Add the panel. A no-op without the flag, so the page is unchanged."""
        if not self.enabled:
            return
        if not self.available:
            with gui.add_folder("Five-point IK body (G1)"):
                gui.add_text("status", initial_value="DISABLED", disabled=True)
                gui.add_text("reason", initial_value=self.disabled_reason[:110], disabled=True)
            return
        with gui.add_folder("Five-point IK body (G1)"):
            gui.add_text("what", disabled=True, initial_value=(
                "violet = PREDICTION · pale = dataset TARGET · per-limb DLS, no model"))
            self._gui["show_pred"] = gui.add_checkbox("Prediction body", initial_value=True)
            self._gui["show_tgt"] = gui.add_checkbox("Target body",
                                                     initial_value=self._show_target)
            # The URDF's own materials, opaque, instead of the tints — for both bodies that are
            # shown, so with both on they look alike; the two checkboxes above pick one.
            self._gui["real_mesh"] = gui.add_checkbox(REAL_MESH_LABEL, initial_value=False)
            self._gui["mesh"] = gui.add_text("mesh", initial_value=_MESH_NOTE_TINTED, disabled=True)
            self._gui["series"] = gui.add_dropdown("Residual for", SERIES,
                                                  initial_value=self._series)
            self._gui["res"] = gui.add_text("IK residual", initial_value="-", disabled=True)
            self._gui["root"] = gui.add_text("drawn pelvis", initial_value="-", disabled=True)
            self._gui["cost"] = gui.add_text("solve", initial_value="-", disabled=True)
            # The two facts a reader cannot get from any number on the page.
            gui.add_text("root", disabled=True, initial_value=(
                "the CHEST keypoint is the root; the waist is held, so the drawn pelvis is not "
                "a physical claim"))
            gui.add_text("frame", disabled=True,
                         initial_value=(self._frame_note or "g1-ready-pose")[:110])
            if self.held_poses:
                gui.add_text("held", disabled=True, initial_value=(
                    f"{', '.join(sorted(self.held_poses))}: not predicted by this layout — HELD at "
                    f"the ready pose for BOTH bodies")[:150])

            @self._gui["show_pred"].on_update
            def _(_e) -> None:                       # GUI thread: flag only, loop redraws
                self._drawn = -1

            @self._gui["show_tgt"].on_update
            def _(_e) -> None:
                self._drawn = -1

            @self._gui["real_mesh"].on_update
            def _(_e) -> None:                       # flag only; the loop swaps the bodies
                self._drawn = -1

            @self._gui["series"].on_update
            def _(_e) -> None:
                self._series = str(self._gui["series"].value)
                self._drawn = -1

    # ------------------------------------------------------------------ driving
    def on_rollout(self, result, display_poses: dict, file_attrs, home_anchored: bool) -> None:
        """Queue a solve for a fresh rollout. Loop thread; returns immediately.

        ``display_poses`` is the viewer's own ``role -> _DisplayPoses`` for the CURRENT display
        frame — taken rather than recomputed, for the same reason ``MCHumanoid.on_rollout`` takes
        it: a body solved in a different frame than the lines it is drawn against is the exact
        failure this page exists to make visible.
        """
        del file_attrs
        if not self.available:
            return
        missing = [role for role in IK_ROLES if role not in display_poses]
        held = [role for role in missing if role in self.held_poses]
        unfilled = [role for role in missing if role not in self.held_poses]
        if unfilled:
            self.disabled_reason = (
                f"layout {result.layout_mode!r} has no {', '.join(unfilled)} role(s) and no held "
                f"pose for them; a five-point body needs all of {list(IK_ROLES)}. Use a "
                f"diffusion_umi_5pt checkpoint (fivepoint_geom_*), or pass the ready-pose "
                f"held_poses for the missing roles (the viewer does under --display-frame "
                f"g1-ready-pose).")
            self.log(f"DISABLED: {self.disabled_reason}")
            self.ik = None
            self._hide()
            return
        if not home_anchored:
            # The raw frame leaves each role at its own absolute anchor, i.e. back in the two
            # incompatible source frames. Hidden with a reason rather than drawn 543 mm wrong.
            # RETIRED, not merely dropped: a solve from the previous sample would otherwise land
            # afterwards and draw a body under the "no body" message.
            self._retire()
            self._hide()
            self._set("res", "raw frame: no body (the roles are not in one frame)")
            return
        self._gen += 1
        payload = {"generation": self._gen, "poses": {}, "jaw": {},
                   "identity": (f"{result.algorithm} sample {result.sample.dataset_index} "
                                f"seed {result.seed}")}
        n_steps = None
        for role in IK_ROLES:
            if role in held:
                continue
            entry = display_poses[role]
            payload["poses"][role] = {
                "prediction": (np.asarray(entry.predicted_position, dtype=np.float64),
                               _rotmats_from_wxyz(entry.predicted_wxyz)),
                "target": (np.asarray(entry.target_position, dtype=np.float64),
                           _rotmats_from_wxyz(entry.target_wxyz))}
            n_steps = int(np.asarray(entry.predicted_position).shape[0])
        for role in held:
            # A held role is the ready pose on EVERY step of BOTH series: the same constant row the
            # rig's wire publishes for it, so the two bodies differ only in the predicted role(s).
            pos, rot = self.held_poses[role]
            const = (np.tile(pos.reshape(1, 3), (int(n_steps), 1)),
                     np.tile(rot.reshape(1, 3, 3), (int(n_steps), 1, 1)))
            payload["poses"][role] = {"prediction": const, "target": const}
        for rc in result.roles:
            for series, widths in (("prediction", rc.predicted_gripper_m),
                                   ("target", rc.target_gripper_m)):
                if widths is not None:
                    payload["jaw"].setdefault(series, {})[rc.role] = np.asarray(
                        widths, dtype=np.float64).reshape(-1)
        self._ensure_worker()
        # Depth-1 queue, drained first: a rollout superseded before its solve started is simply
        # dropped, and the generation check in the worker covers one already in flight.
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(payload)
        self._set("res", "solving …")

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name="kp-ik", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        """Worker: solve both series for the newest queued rollout. Owns its own KeypointIK."""
        import time  # noqa: PLC0415

        ik, why = KeypointIK.load(urdf_path=self._urdf_path,
                                  home_anchor_path=self._anchor_path)
        if ik is None:  # pragma: no cover - the constructor already loaded one successfully
            self.log(f"IK worker could not load its own solver ({why}); body disabled")
            return
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            gen = int(payload["generation"])
            t0 = time.perf_counter()
            out: dict = {}
            try:
                for series in SERIES:
                    if gen != self._gen:
                        break
                    poses = {r: payload["poses"][r][series] for r in IK_ROLES}
                    out[series] = ik.solve_chunk(
                        poses, seed_dof=self._seed_dof,
                        jaw_m=payload["jaw"].get(series), yield_gate=self.infer_gate)
            except Exception as exc:  # noqa: BLE001 - a failed solve hides the body, never kills
                self.log(f"IK solve failed: {exc!r}")
                # Only if this failure is still the CURRENT solve: unguarded, a stale worker's
                # exception would blank a body that a newer generation had already published.
                if gen == self._gen:
                    self._solved = None
                    self._drawn = -1
                continue
            if gen != self._gen or len(out) != len(SERIES):
                continue
            self._solved = _Solved(generation=gen, per_series=out,
                                   identity=str(payload["identity"]),
                                   ms=(time.perf_counter() - t0) * 1e3)
            self._drawn = -1

    def poll(self, action_step: int) -> bool:
        """True when a fresh solve has landed and the scene should be redrawn. Loop thread."""
        del action_step
        if not self.available:
            return False
        return self._solved is not None and self._drawn < 0

    def render_at(self, action_step: int) -> None:
        """Pose both bodies at this action step. Loop thread."""
        if not self.available:
            return
        solved = self._solved
        if solved is None:
            self._hide()
            return
        step = int(action_step)
        want = {"prediction": self._checked("show_pred", True),
                "target": self._checked("show_tgt", self._show_target)}
        real_requested = self._checked("real_mesh", False)
        drawn_real = False
        for series, res in solved.per_series.items():
            roots, urdfs = self._roots.get(series), self._urdfs.get(series)
            if not roots or not urdfs:
                continue
            variant = "real" if real_requested and self._ensure_real_body(series) else "tinted"
            # Exactly one variant of a body is ever visible; the other is hidden BEFORE the chosen
            # one is posed so a toggle never shows two robots for a frame.
            for other in VARIANTS:
                if other != variant and roots.get(other) is not None:
                    roots[other].visible = False
            root, urdf = roots[variant], urdfs[variant]
            if not want.get(series, True) or res.n == 0:
                root.visible = False
                continue
            i = min(max(step, 0), res.n - 1)
            root.position = tuple(float(v) for v in res.root_pos[i])
            root.wxyz = tuple(float(v) for v in res.root_wxyz[i])
            urdf.update_cfg(res.cfg[i])
            root.visible = True
            drawn_real = drawn_real or variant == "real"
        if not self._real_failed:
            self._set("mesh", _MESH_NOTE_REAL if drawn_real else _MESH_NOTE_TINTED)
        res = solved.per_series.get(self._series)
        if res is not None and res.n:
            i = min(max(step, 0), res.n - 1)
            worst = int(np.argmax(res.residual_mm[i]))
            self._set("res", (f"{self._series}: step {i} worst {res.residual_mm[i, worst]:.1f} mm "
                              f"({res.roles[worst]}) · chunk worst {res.residual_mm.max():.1f} mm"))
            # The root's travel from step 0, which is the honest measure of how much of the
            # predicted chest motion the FLOATING root absorbed instead of a torso bend.
            travel = float(np.linalg.norm(res.root_pos[i] - res.root_pos[0]) * 1e3)
            self._set("root", f"{np.round(res.root_pos[i], 3).tolist()} m · moved "
                              f"{travel:.0f} mm from step 0 (display only)")
        self._set("cost", f"{solved.ms:.0f} ms for {len(SERIES)} bodies · {solved.identity}")
        self._drawn = step

    def clear(self) -> None:
        """Drop the drawn bodies — and retire any solve still on its way to becoming one.

        ⚠️ Bumping the generation is the load-bearing half. A worker captures ``self._gen`` when it
        starts and publishes only while it is still current, so dropping ``_solved`` alone lets a
        solve already in flight put the PREVIOUS sample's bodies straight back on a page that has
        moved on — for seconds, at H = 200. Same defect and same fix as
        ``hl_5pt.PreviewGate.retire``; the bodies are display-only, so nothing downstream notices.
        """
        self._retire()
        self._hide()

    def _retire(self) -> None:
        """Make every outstanding solve unpublishable and forget the current one."""
        self._gen += 1
        self._solved = None
        while True:                      # a QUEUED solve must die with the in-flight one
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)

    # ------------------------------------------------------------------ helpers
    def _hide(self) -> None:
        for variants in self._roots.values():
            for root in variants.values():
                if root is None:
                    continue
                try:
                    root.visible = False
                except Exception:  # noqa: BLE001, S110 - the scene may already be torn down
                    pass

    def _checked(self, key: str, default: bool) -> bool:
        handle = self._gui.get(key)
        return default if handle is None else bool(handle.value)

    def _set(self, key: str, value: str) -> None:
        handle = self._gui.get(key)
        if handle is not None:
            handle.value = value[:150]


def _rotmats_from_wxyz(wxyz_seq) -> np.ndarray:
    """``[H wxyz tuples]`` -> ``(H, 3, 3)``.

    The display poses are held as quaternions (that is what viser is handed), and re-deriving
    the rotation from ``rot6d`` here would bypass the display-frame rebase this overlay exists
    to share — the same reason ``mc_humanoid._build_cond`` converts rather than re-reads.
    """
    wxyz = np.asarray(wxyz_seq, dtype=np.float64).reshape(-1, 4)
    return quat_xyzw_to_rotmat(quat_wxyz_to_xyzw(wxyz))


def add_cli_args(parser) -> None:
    """Add the ``--keypoint-ik`` flag group to an ``infer_s1_*.py`` parser."""
    group = parser.add_argument_group("five-point IK body (diffusion_umi_5pt)")
    group.add_argument(
        "--keypoint-ik", action="store_true",
        help="draw the G1 whole body implied by the FIVE predicted keypoints, by per-limb IK in "
             "the render URDF (violet = prediction, pale = dataset target). No second model, no "
             "conda env: five points leave nothing to infer. A fivepoint_geom_* checkpoint "
             "predicts all five; a left_only / both_hands* one gets the roles it does not "
             "predict HELD at the ready pose (the panel says which). IMPLIES --display-frame "
             "g1-ready-pose, because the tracked roles and the FK'd feet are not in one frame "
             "otherwise — and because the held roles are defined only in that frame.")
    group.add_argument(
        "--no-keypoint-ik-target", action="store_true",
        help="with --keypoint-ik, draw only the violet prediction body (the pale target body is "
             "on by default: it is what says whether a posture DIFFERS from the demonstration).")


def options_from_args(args) -> dict | None:
    """``run_viewer(keypoint_ik=...)`` kwargs, or ``None`` when the flag is absent."""
    if not getattr(args, "keypoint_ik", False):
        return None
    return {"enabled": True,
            "show_target": not bool(getattr(args, "no_keypoint_ik_target", False)),
            "home_anchor_path": getattr(args, "home_anchor_json", None)}


__all__ = [
    "IK_ROLES",
    "PREDICTION_RGBA",
    "TARGET_RGBA",
    "ROOT_ROLE",
    "SERIES",
    "KeypointHumanoid",
    "add_cli_args",
    "options_from_args",
]
