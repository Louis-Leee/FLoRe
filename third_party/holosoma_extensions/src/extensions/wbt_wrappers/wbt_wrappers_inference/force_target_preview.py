"""Offline viser previewer: how F_cmd / F_ext reshape a reference motion into the
UniFP Eq.2 target (design spec
``docs/superpowers/specs/2026-07-09-force-target-preview-design.md``).

Renders, for a retargeted-motion ``.npz`` played back at its native fps:

  * ``/robot_ref``    — the SOLID reference robot (yam URDF), driven by the npz.
  * ``/robot_target`` — a TRANSPARENT-ORANGE ghost at the Eq.2 target, produced by
    arm-only IK so the active hand's TCP reaches ``x_target = x_contact +
    (F_cmd + F_ext)/K`` while legs / torso / base / other-arm stay at reference.
  * an ``F_cmd`` arrow at the **TCP** (where the force is commanded) and an
    ``F_meas`` arrow at the **FT sensor** (where a real sensor reads it), with the
    0.16 m FT→TCP lever drawn between them, so the sensor-vs-TCP anchor question
    is visible on real motion.
  * TCP and FT target markers + faint target paths, and a GUI HUD with the
    per-axis shift (cm), phase, IK residual, and the sensor-vs-TCP divergence.

HONESTY BANNER (carried from the playground): ``F_ext = F_meas`` is NOT physically
simulated here — there is no contact object and no policy. Both force profiles are
SCRIPTED. Scenario B's ``F_ext → −F_cmd`` equilibrium (target relaxing back to the
reference) is HAND-DRAWN, NOT loop closure. This tool answers "where does the
target land" — not "does the controller achieve it".

Run (headless AWS box, hssim env):
    ~/.holosoma_deps/miniconda3/envs/hssim/bin/python \
      -m wbt_wrappers_inference.force_target_preview \
      /data/dataset/sonic_manipulation_holosoma/avoid_bump_push_R_003__A460_M.npz --port 8080
Then tunnel from your laptop:  ssh -L 8080:localhost:8080 <box>  →  http://localhost:8080
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wbt_wrappers_inference import force_target_math as ftm

logger = logging.getLogger(__name__)

# Default yam URDF (the mesh the ghost + reference are drawn with).
_DEFAULT_URDF = (
    Path(__file__).resolve().parents[4]
    / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof_yam.urdf"
)

# Curated contact-motion library for the in-browser selector. Updated 2026-07-09
# (user: "重新选 20 个下肢不咋动的 … 就在原地的 … 动起来完全看不到"): every clip
# here is IN-PLACE — pelvis horizontal travel < ~0.05 m, no walk/jog/crouch-walk —
# so the reference and ghost don't translate out of view. Chosen for task diversity
# (one/two-hand pick & put-down at various heights/sides, button pushes, door
# handles, bottle throws, typing, mosquito/ball catch, gesture) AND single-right-hand
# IK reachability at the 12 N default (each ≥ ~98% of sampled frames converge). All
# verified to load + have right-hand motion. The `--npz` CLI arg, if it names a file
# not in this list, is prepended so you can still preview an arbitrary clip.
_MOTION_DIR = Path("/data/dataset/sonic_manipulation_holosoma")
_MOTION_LIBRARY = (
    "medium_heavy_one_hand_pick_up_front_high_R_001__A522_M.npz",
    "big_heavy_one_hand_pick_up_front_low_R_001__A511_M.npz",
    "big_light_one_hand_pick_up_right_side_medium_R_001__A509_M.npz",
    "medium_big_heavy_one_hand_pick_up_right_side_medium_R_001__A508_M.npz",
    "big_heavy_one_hand_pick_up_behind_low_R_001__A508_M.npz",
    "big_light_one_hand_put_down_front_low_R_002__A511_M.npz",
    "medium_heavy_one_hand_put_down_front_high_R_003__A522_M.npz",
    "neutral_item_put_away_R_002__A543_M.npz",
    "item_pick_up_crouch_idle_R_003__A352_M.npz",
    "high_button_right_side_push_002__A483_M.npz",
    "high_button_left_side_push_001__A480_M.npz",
    "mid_button_under_push_001__A486_M.npz",
    "inside_door_handle_left_side_open_R_001__A514_M.npz",
    "inside_door_handle_right_side_open_R_001__A514_M.npz",
    "drinking_bottle_throw_270_R_001__A548_M.npz",
    "drinking_bottle_throw_180_R_003__A472_M.npz",
    "dont_come_any_closer_one_hand_R_001__A457_M.npz",
    "cellphone_typing_sitting_one_hand_idle_R_001__A427_M.npz",
    "mosquito_catch_one_hand_R_001__A502_M.npz",
    "catch_playing_R_003__A456_M.npz",
)

# Colors (0-255 RGB for markers/arrows; RGBA float for the URDF override).
_CMD_RGB = (60, 120, 250)  # blue  — commanded force at the TCP
_EXT_RGB = (240, 90, 60)  # red   — external/measured force at the FT sensor
_LEVER_RGB = (150, 150, 150)  # grey  — the FT↔TCP lever
_TCP_MARK_RGB = (250, 170, 40)  # orange — TCP target marker + path
_FT_MARK_RGB = (120, 120, 120)  # grey  — FT target marker + path
_GHOST_RGBA = (1.0, 0.55, 0.0, 0.4)  # transparent orange target mesh (IK reached the target)
_GHOST_FAIL_RGBA = (0.9, 0.1, 0.1, 0.3)  # transparent red ghost (IK could NOT fully reach — closest pose)

_ARROW_M_PER_N = 0.01  # 1 N → 1 cm arrow (matches eval_utils.FORCE_ARROW_M_PER_N scale intent)


# ---------------------------------------------------------------------------
# Precomputation: per-frame reference + target configs and render quantities.
# Kept OFF the viser callback threads — the render callback only indexes cached
# numpy (the replay_common thread-safety pattern).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Solution:
    """One immutable, self-consistent recompute result. Published by a single
    swap of the ``PreviewData.sol`` reference so a render on the GUI thread always
    sees a coherent set (never new frames with a stale tgt_cfg)."""

    frames: list[ftm.FrameTargets]
    tgt_cfg: np.ndarray
    residual_mm: np.ndarray
    converged: np.ndarray


class PreviewData:
    """All per-frame quantities for the current parameter set.

    ``recompute`` is the only expensive call (arm IK per frame). It is run from the
    main loop when parameters change, never from a GUI callback, so rapid slider
    drags cannot race. Its result is an immutable ``Solution`` published by a single
    atomic attribute assignment (CPython reference swap), which the render callback
    reads without a lock — so a render can never see a torn half-updated result.
    """

    def __init__(self, motion: ftm.Motion, ik: ftm.ArmIK, actuated_names: list[str]):
        self.motion = motion
        self.ik = ik
        self.actuated_names = actuated_names
        self.n = motion.n_frames

        # npz joint name → reference value, per frame, as the actuated 31-vector.
        # Gripper jaws (not in npz) stay 0.
        self._ref_cfg = np.zeros((self.n, len(actuated_names)))
        name_to_col = {n: i for i, n in enumerate(actuated_names)}
        for t in range(self.n):
            row = dict(zip(motion.joint_names, motion.dof(t)))
            for jn, val in row.items():
                if jn in name_to_col:
                    self._ref_cfg[t, name_to_col[jn]] = val
        self._name_to_col = name_to_col

        # The published solution (swapped atomically by recompute()).
        self.sol: Solution = Solution(
            frames=[], tgt_cfg=self._ref_cfg.copy(), residual_mm=np.zeros(self.n), converged=np.ones(self.n, dtype=bool)
        )

    def base_pose(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat wxyz) of the floating base at frame ``t``."""
        return self.motion.base_pos(t), self.motion.base_quat(t)

    def ref_cfg(self, t: int) -> np.ndarray:
        return self._ref_cfg[t]

    def recompute(self, side: str, direction_pelvis: np.ndarray, peak_N: float, spec: ftm.PulseSpec, scenario: str):
        """Recompute per-frame targets + IK ghost configs; publish as one Solution."""
        frames = []
        tgt_cfg = self._ref_cfg.copy()
        residual_mm = np.zeros(self.n)
        converged = np.ones(self.n, dtype=bool)
        for t in range(self.n):
            ft = ftm.frame_targets(self.motion, t, side, direction_pelvis, peak_N, spec, scenario)
            frames.append(ft)
            res = self.ik.solve(self.motion, t, side, ft.x_target_tcp)
            residual_mm[t] = res.residual_m * 1000.0
            converged[t] = res.converged
            # ALWAYS apply the IK solution — even when not "converged". A shifted
            # target near/just outside the arm's workspace (e.g. a downward force at
            # full arm extension) leaves the IK a few cm short; the best-effort
            # closest pose is still the most useful thing to show. If we instead held
            # the arm at reference on those frames the ghost would coincide EXACTLY
            # with the solid reference robot and appear to "vanish" (the user-reported
            # bug). The render() colours a non-converged ghost differently (red, more
            # transparent) and the HUD shows the residual, so "not fully reached" is
            # visible rather than looking like a disappearance.
            for jn, val in res.q_arm.items():
                if jn in self._name_to_col:
                    tgt_cfg[t, self._name_to_col[jn]] = val
        # Single atomic publish — readers see the whole coherent set or the old one.
        self.sol = Solution(frames=frames, tgt_cfg=tgt_cfg, residual_mm=residual_mm, converged=converged)


# ---------------------------------------------------------------------------
# viser scene assembly.
# ---------------------------------------------------------------------------
def _run_viewer(npz_path: str, urdf_path: str, host: str, port: int) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    ik = ftm.ArmIK(urdf_path)

    # Build the clip menu: every library clip that exists on disk, with the CLI-named
    # clip first (prepended if it is not already in the library).
    clip_paths: dict[str, str] = {}
    for fn in _MOTION_LIBRARY:
        p = _MOTION_DIR / fn
        if p.exists():
            clip_paths[fn] = str(p)
    cli_name = Path(npz_path).name
    if cli_name not in clip_paths:
        clip_paths = {cli_name: npz_path, **clip_paths}
    clip_names = list(clip_paths.keys())
    initial_clip = cli_name if cli_name in clip_paths else clip_names[0]

    # `mut` holds everything that a clip-reload swaps (single-renderer: only the main
    # loop mutates it, so no lock is needed on the objects themselves).
    mut = {"motion": ftm.load_motion(clip_paths[initial_clip]), "data": None}
    motion = mut["motion"]

    server = viser.ViserServer(host=host, port=port, label="Force-target preview")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.1)

    # Base frames (the floating base of each robot); the URDFs mount under them.
    # NOTE: each ViserUrdf gets its OWN yourdfpy.URDF instance. update_cfg mutates
    # the wrapped model and reads transforms back, so a SHARED model would let a
    # ref render and a ghost render tear each other's FK. Separate models + the
    # single-threaded render loop below (all rendering on the main loop, never on a
    # viser websocket thread) keep the two robots' poses independent and coherent.
    # ViserUrdf's mesh colour is fixed at construction, so we build TWO ghost robots:
    # an ORANGE one (IK reached the target) and a RED one (IK fell short — the closest
    # pose). Each frame shows exactly one of them via `visible`, so the ghost never
    # "vanishes" into the solid reference — it just turns red when unreachable.
    ref_base = server.scene.add_frame("/robot_ref", show_axes=False)
    tgt_base = server.scene.add_frame("/robot_target", show_axes=False)
    tgt_base_fail = server.scene.add_frame("/robot_target_fail", show_axes=False)
    ref_urdf = ViserUrdf(server, yourdfpy.URDF.load(urdf_path), root_node_name="/robot_ref/urdf", load_meshes=True)
    tgt_urdf = ViserUrdf(
        server,
        yourdfpy.URDF.load(urdf_path),
        root_node_name="/robot_target/urdf",
        load_meshes=True,
        mesh_color_override=_GHOST_RGBA,
    )
    tgt_urdf_fail = ViserUrdf(
        server,
        yourdfpy.URDF.load(urdf_path),
        root_node_name="/robot_target_fail/urdf",
        load_meshes=True,
        mesh_color_override=_GHOST_FAIL_RGBA,
    )
    actuated_names = list(ref_urdf.get_actuated_joint_names())

    mut["data"] = PreviewData(motion, ik, actuated_names)

    # --- GUI ---------------------------------------------------------------
    server.gui.add_markdown(
        "**Force-target preview** — solid = reference motion; **orange** ghost = the Eq.2 "
        "target `x_ref + (F_cmd+F_ext)/K`, realized by **arm-only IK** (one pose that "
        "reaches the shifted grasp point — NOT what a trained policy will necessarily do). "
        "When the target is out of the arm's reach the ghost turns **red** (the closest "
        "pose the IK could find) so it never vanishes into the solid robot.\n\n"
        "`F_ext = F_meas` is **scripted, not simulated** (no contact object, no policy). "
        "Scenario B's equilibrium is hand-drawn, **not** loop closure.\n\n"
        "Blue `F_cmd` arrow acts at the **TCP** (grasp point); red `F_ext` arrow is drawn at "
        "the **FT sensor** (where a real sensor reads it). The two target markers stay a "
        "**constant** 13.3 cm apart — that is only the rigid FT→TCP lever, **not** an "
        "anchor-dependent shift: for a pure 3-D force the shift is identical at either anchor."
    )
    gui_clip = server.gui.add_dropdown("clip", options=tuple(clip_names), initial_value=initial_clip)
    gui_play = server.gui.add_checkbox("play", initial_value=False)
    gui_frame = server.gui.add_slider("frame", min=0, max=motion.n_frames - 1, step=1, initial_value=0)
    gui_scenario = server.gui.add_dropdown(
        "scenario", options=("A: F_cmd only", "B: equilibrium"), initial_value="A: F_cmd only"
    )
    gui_hand = server.gui.add_dropdown("active hand", options=("right", "left"), initial_value="right")
    gui_peak = server.gui.add_slider("F_cmd peak (N)", min=0.0, max=25.0, step=1.0, initial_value=ftm.DEFAULT_PEAK_N)
    gui_dir = server.gui.add_vector3("F_cmd dir (pelvis)", initial_value=(1.0, 0.0, 0.0), step=0.1)
    gui_show_ft = server.gui.add_checkbox("show FT-sensor anchor overlay", initial_value=True)
    gui_show_path = server.gui.add_checkbox("show target paths", initial_value=True)
    gui_arrow_scale = server.gui.add_slider(
        "arrow scale (m/N)", min=0.002, max=0.05, step=0.002, initial_value=_ARROW_M_PER_N
    )
    gui_hud = server.gui.add_markdown("")

    # Threading model (single renderer): GUI callbacks NEVER render, recompute, or
    # reload — they only raise flags. The main loop is the SOLE thread that touches the
    # scene / URDF FK / motion data, so concurrent websocket callbacks can't tear a
    # pose or a mid-swap reload. `reload` (⇒ load a new clip) ⊃ `dirty` (⇒ recompute +
    # redraw paths) ⊃ `render_req` (⇒ redraw current frame only).
    state = {"reload": False, "dirty": True, "render_req": True, "playing": False}
    state_lock = threading.Lock()

    def _flag(key: str):
        def _cb(_=None):
            with state_lock:
                state[key] = True

        return _cb

    # Clip change → full reload.
    gui_clip.on_update(_flag("reload"))
    # Params that change the SOLUTION (need a recompute) → dirty.
    for h in (gui_scenario, gui_hand, gui_peak, gui_dir):
        h.on_update(_flag("dirty"))
    # Display-only controls + frame scrub → just re-render the current frame.
    for h in (gui_frame, gui_show_ft, gui_show_path, gui_arrow_scale):
        h.on_update(_flag("render_req"))

    def _toggle_play(_=None):
        with state_lock:
            state["playing"] = bool(gui_play.value)

    gui_play.on_update(_toggle_play)

    def _params():
        side = gui_hand.value
        scenario = "A" if gui_scenario.value.startswith("A") else "B"
        d = np.array(gui_dir.value, dtype=float)
        return side, scenario, d, float(gui_peak.value)

    # --- draw the two full target PATHS (once per recompute; §4.2) ----------
    def draw_paths(sol: Solution) -> None:
        show = bool(gui_show_path.value)
        tcp_pts = np.array([f.x_target_tcp for f in sol.frames], dtype=np.float32)
        ft_pts = np.array([f.x_target_ft for f in sol.frames], dtype=np.float32)
        server.scene.add_spline_catmull_rom(
            "/path_tcp", points=tcp_pts, color=_TCP_MARK_RGB, line_width=2.0, visible=show
        )
        server.scene.add_spline_catmull_rom(
            "/path_ft", points=ft_pts, color=_FT_MARK_RGB, line_width=1.0, visible=show and bool(gui_show_ft.value)
        )

    # --- render one frame (ONLY called from the main loop) ------------------
    def render(t: int, sol: Solution) -> None:
        data = mut["data"]
        motion = mut["motion"]
        if not sol.frames:
            return
        t = int(np.clip(t, 0, data.n - 1))
        ft = sol.frames[t]

        # Base poses (ghosts and reference share the SAME base; only the arm differs).
        p0, q0 = data.base_pose(t)
        ref_base.position = tuple(p0)
        ref_base.wxyz = tuple(q0)
        for base in (tgt_base, tgt_base_fail):
            base.position = tuple(p0)
            base.wxyz = tuple(q0)
        ref_urdf.update_cfg(data.ref_cfg(t))
        # Both ghosts get the same best-effort IK pose; only ONE is visible per frame:
        # orange when the IK reached the target, red (more transparent) when it fell
        # short (closest pose). This guarantees the ghost is always visible against the
        # solid reference instead of vanishing into it on unreachable frames.
        reached = bool(sol.converged[t])
        tgt_urdf.update_cfg(sol.tgt_cfg[t])
        tgt_urdf_fail.update_cfg(sol.tgt_cfg[t])
        tgt_base.visible = reached
        tgt_base_fail.visible = not reached

        arrow_scale = float(gui_arrow_scale.value)
        show_ft = bool(gui_show_ft.value)

        # F_cmd arrow at the TCP; F_meas arrow at the FT sensor.
        _draw_arrow(server, "/fcmd", ft.p_tcp, ft.f_cmd_world, arrow_scale, _CMD_RGB)
        _draw_arrow(server, "/fmeas", ft.p_ft, ft.f_ext_world, arrow_scale, _EXT_RGB, visible=show_ft)
        # Lever FT↔TCP.
        server.scene.add_line_segments(
            "/lever",
            points=np.array([[ft.p_ft, ft.p_tcp]], dtype=np.float32),
            colors=_LEVER_RGB,
            line_width=2.0,
            visible=show_ft,
        )
        # Target markers.
        server.scene.add_icosphere("/tgt_tcp", radius=0.02, color=_TCP_MARK_RGB, position=tuple(ft.x_target_tcp))
        server.scene.add_icosphere(
            "/tgt_ft", radius=0.015, color=_FT_MARK_RGB, position=tuple(ft.x_target_ft), visible=show_ft
        )

        # HUD — organized into labeled, boxed sections (monospace code fences render
        # as framed panels in viser markdown).
        s = ft.shift * 100.0
        fc, fe = ft.f_cmd_world, ft.f_ext_world
        conv = (
            "reached (orange ghost)"
            if sol.converged[t]
            else "NOT fully reached — red ghost = closest pose; marker = commanded target"
        )
        gui_hud.content = (
            f"### {gui_clip.value}\n"
            "```text\n"
            f"frame   {t:4d} / {data.n - 1:<4d}   t = {t / motion.fps:5.2f} s\n"
            f"phase   {ft.phase}\n"
            "```\n"
            "**Forces (world)**\n"
            "```text\n"
            f"F_cmd @TCP     |F| = {np.linalg.norm(fc):5.1f} N   [{fc[0]:+5.1f} {fc[1]:+5.1f} {fc[2]:+5.1f}]\n"
            f"F_ext @sensor  |F| = {np.linalg.norm(fe):5.1f} N   [{fe[0]:+5.1f} {fe[1]:+5.1f} {fe[2]:+5.1f}]\n"
            "```\n"
            "**Target shift = (F_cmd+F_ext)/K**\n"
            "```text\n"
            f"shift  [{s[0]:+5.1f} {s[1]:+5.1f} {s[2]:+5.1f}] cm    |shift| = {np.linalg.norm(s):5.1f} cm\n"
            f"same shift at BOTH anchors (pure 3-D force)\n"
            f"TCP vs FT markers: {ftm.FT_TCP_LEVER_X * 100:.1f} cm apart = rigid lever,\n"
            f"  cancels in reward residual cur-ref (matters for 6-D moments, D7)\n"
            "```\n"
            "**IK ghost**\n"
            "```text\n"
            f"pos residual = {sol.residual_mm[t]:5.2f} mm   ({conv})\n"
            "```"
        )

    # --- main loop: the SOLE renderer. reload / recompute / playback clock ---
    logger.info("viser up on http://%s:%d  (tunnel + open in a browser)", host, port)
    last = 0.0
    while True:
        with state_lock:
            do_reload = state["reload"]
            dirty = state["dirty"]
            render_req = state["render_req"]
            playing = state["playing"]
            state["reload"] = False
            state["dirty"] = False
            state["render_req"] = False
        if do_reload:
            try:
                new_motion = ftm.load_motion(clip_paths[gui_clip.value])
            except Exception as exc:  # keep the server alive on a bad clip
                logger.warning("clip load failed (%s): %s", gui_clip.value, exc)
                new_motion = None
            if new_motion is not None:
                mut["motion"] = new_motion
                mut["data"] = PreviewData(new_motion, ik, actuated_names)
                # Re-range the frame slider and reset to frame 0 for the new clip.
                gui_frame.max = new_motion.n_frames - 1
                gui_frame.value = 0
                dirty = True  # recompute + redraw paths for the new clip
        if dirty:
            side, scenario, d, peak = _params()
            mut["data"].recompute(side, d, peak, ftm.PulseSpec(), scenario)
            draw_paths(mut["data"].sol)
            render_req = True  # redraw the current frame against the new solution
        if playing:
            now = time.time()
            if now - last >= mut["motion"].dt:
                last = now
                gui_frame.value = (int(gui_frame.value) + 1) % mut["motion"].n_frames
                render_req = True
            else:
                last = last if last else now  # avoid an instant fire after pause→resume
        if render_req:
            render(int(gui_frame.value), mut["data"].sol)
        time.sleep(0.005)


def _draw_arrow(server, name: str, origin: np.ndarray, vec: np.ndarray, scale: float, rgb, visible: bool = True) -> None:
    """Draw a single force arrow from ``origin`` along ``vec`` (world), length ∝ |vec|.

    Hidden (zero-length, invisible) when the force is ~0 or ``visible=False``, so an
    idle phase (or a toggled-off overlay) shows no arrow rather than a degenerate glyph."""
    v = np.asarray(vec, dtype=float)
    mag = float(np.linalg.norm(v))
    if mag < 1e-6 or not visible:
        server.scene.add_arrows(name, points=np.zeros((1, 2, 3), dtype=np.float32), colors=rgb, visible=False)
        return
    tip = np.asarray(origin, dtype=float) + v * scale
    pts = np.array([[origin, tip]], dtype=np.float32)  # (1, 2, 3): one segment
    server.scene.add_arrows(name, points=pts, colors=rgb, shaft_radius=0.006, head_radius=0.015, head_length=0.03)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "npz_path",
        nargs="?",
        default=None,
        help="a retargeted-motion .npz (defaults to the first library clip; the "
        "in-browser 'clip' dropdown lists all curated in-place clips)",
    )
    p.add_argument("--urdf", default=str(_DEFAULT_URDF), help="yam URDF for the mesh")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)
    npz_path = args.npz_path or str(_MOTION_DIR / _MOTION_LIBRARY[0])
    if not Path(npz_path).exists():
        raise SystemExit(f"[force-preview] npz not found: {npz_path}")
    if not Path(args.urdf).exists():
        raise SystemExit(f"[force-preview] urdf not found: {args.urdf}")
    _run_viewer(npz_path, args.urdf, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
