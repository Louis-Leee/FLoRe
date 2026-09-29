"""UniFP-style force-tracking variant for G1 29-DOF whole-body tracking.

This module ports GentleHumanoid's `G1_unifp` baseline (the estimator-less
variant per `thirdparty/gentle_humanoid/train.sh:80`) to the holosoma WBT
pipeline. The PPO privileged force estimator decoder, GH v2's keypoint /
lower-keypoint shaping, and the multi-stage train→adapt→finetune pipeline
are all separate followup plans.

Layout (one module per feature, mirroring `terrain.py` / `maskedmimic_observations.py`):
- `HandForceConfig`             : dataclass holding all knobs.
- `_advance_force_pulses`       : trapezoidal-pulse helper (sole writer; called
                                  only by `HandForceInjector`).
- `force_cmd_local`             : obs term — F^cmd in root frame, 6D (pure reader).
- `force_ext_local`             : obs term — F^ext in root frame, 6D (pure reader).
- `force_track_pos_eq2`         : reward term — exp(-‖cur - (ref + F/K)‖_1 · 2/σ).
- `force_track_pos_pen`         : reward term — graded raw-drift indicator [0, 1].
- `HandForceInjector`           : CommandTerm; sole owner of `env._wbt_force_*_b`
                                  buffers, monkey-patches `_apply_force_in_physics_step`
                                  for substep wrench writes (advances pulse at substep 0).
                                  `reset(env_ids)` restarts the pulse + zeros the force
                                  buffers for reset envs (GH `sample_init` parity).
- `ForceAlphaCurriculum`        : CurriculumTerm; sole owner of `env._wbt_force_alpha`,
                                  ramps from `force_alpha_init` (default 0.0) to
                                  `force_alpha_max` (1.0) based on
                                  `env.average_episode_length`.

Force application: PhysX targets `*_wrist_yaw_link` (in holosoma's 32-body list)
but applies the wrench at the rubber_hand offset within wrist_yaw's local frame
via IsaacLab's `set_external_force_and_torque(positions=offset_b)`. This is
rigid-body-equivalent to applying force directly on `*_rubber_hand`
(8.15 cm forward of wrist_yaw for the yam gripper: wrist→ft_sensor 0.072 +
palm 0.0095), which holosoma's 32-body name list does not include. Reward
`cur_w` and `ref_w` are reconstructed via forward kinematics:
`wrist_yaw_pos_w + R(wrist_yaw_quat) @ offset_b`. Both sides go through the
same FK, so the URDF offset cancels in the residual.

Frame convention (2026-05-28 root/world alignment to GH _force_track):
    F^cmd and F^ext are SAMPLED in the root (torso) frame and stored as
    `_wbt_force_cmd_b` / `_wbt_force_ext_b` (root). `_advance_force_pulses`
    rotates them into the world frame (`_wbt_force_cmd_w` / `_wbt_force_ext_w`)
    via the live root quaternion — the single-writer point. The Eq.2 reward and
    the PhysX injection both consume the world-frame buffers; injection does a
    final world->body-local conversion (IsaacLab is_global=False, like push.py).
    The actor obs is the root-frame buffer, so the deploy planner emits F^cmd in
    the root frame.

    The rubber_hand FK offset is rotated by the LIVE wrist quaternion on both the
    cur and ref sides (_force_cur_pos_w / _force_ref_pos_w) so it cancels exactly
    in the Eq.2 residual, regardless of wrist orientation.

Single source of truth for force state: env-attached buffers (allocated by
`HandForceInjector.setup()` and `ForceAlphaCurriculum.setup()`):
    env._wbt_force_cmd_b      # (num_envs, len(body_names), 3) root frame
    env._wbt_force_ext_b      # (num_envs, len(body_names), 3) root frame
    env._wbt_force_cmd_w      # (num_envs, len(body_names), 3) world frame
    env._wbt_force_ext_w      # (num_envs, len(body_names), 3) world frame
    env._wbt_force_alpha      # () scalar in [0, 1]
    env._wbt_force_state      # dict (pulse RNG, last_step, per-pulse phase tensors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HandForceConfig:
    """Knobs for the force-tracking variant.

    Defaults match GentleHumanoid v2 (`G1_unifp_estimator_v2.yaml`) where
    applicable. Curriculum defaults are tuned for holosoma's
    AverageEpisodeLengthTracker, which produces values in the
    [0, max_episode_length] range (typically up to ~1000 for a 20-second
    episode at 50 Hz).
    """

    # Magnitudes and stiffness.
    f_cmd_max: float = 25.0  # N
    f_ext_max: float = 25.0  # N
    stiffness_K: float = 200.0  # N/m, used in Eq.2 target shift (Math Foundation §6)

    # Reward shaping (UniFP code form: L1 norm + scale 2/σ; Math Foundation §5).
    sigma: float = 0.3  # m, from GH G1_unifp (force_track_pos: [0.3]); UniFP's own tracking_ee_sigma is 1.0
    pen_threshold: float = 0.6  # m, raw drift cutoff for [0, 1] safety penalty (weight=-6.0 in cfg)

    # Trapezoidal pulse phases, in env steps (50 Hz).
    ramp_steps: tuple[int, int] = (50, 150)
    hold_steps: tuple[int, int] = (175, 450)
    idle_steps: tuple[int, int] = (50, 150)

    # Curriculum (linear ramp on average_episode_length).
    #
    # Defaults disable the ramp — full force from step 0. The
    # ForceAlphaCurriculum term remains registered (so env._wbt_force_alpha is
    # still allocated and `_advance_force_pulses` keeps multiplying by alpha),
    # but with alpha_step=0.0 reset() is a no-op, and force_alpha_init=1.0 means
    # alpha starts at the plateau. To reinstate the curriculum, set
    # force_alpha_init=0.0 and alpha_step=0.01.
    #
    # Threshold knobs are kept (defensive only) so the if/elif branches in
    # ForceAlphaCurriculum.reset() stay well-typed when alpha_step != 0.
    force_alpha_init: float = 1.0  # full force from training step 0 (no curriculum)
    force_alpha_max: float = 1.0  # plateau (matches init when curriculum disabled)
    level_down_threshold: float = 100.0  # avg_ep_length below this: keep alpha low (only used if alpha_step>0)
    level_up_threshold: float = 400.0  # avg_ep_length above this: alpha ramps up (only used if alpha_step>0)
    alpha_step: float = 0.0  # 0.0 ⇒ curriculum disabled. Set 0.01 to re-enable.

    # Force application bodies and the URDF-fixed offset to the contact point.
    #
    # `body_names` are the parent articulation links that PhysX targets via
    # `simulator.body_ids`. We pick `*_wrist_yaw_link` because:
    #   1. They ARE in holosoma's 32-body name list (so PhysX can address them).
    #   2. Each has a fixed-child `*_rubber_hand` link at offset `*_offset_b`
    #      below — the actual physical contact point.
    # Force application uses IsaacLab's `positions` kwarg on
    # `set_external_force_and_torque` to apply the wrench at the wrist_yaw-local
    # `*_offset_b` position, which is rigid-body-equivalent to applying force on
    # the rubber_hand link itself (no force-arm error). Reward functions read
    # cur/ref via forward kinematics: `wrist_yaw_pos_w + R(wrist_yaw_quat) @ offset_b`.
    body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    # URDF-fixed offset from each `body_names[i]` link's local frame to its
    # rubber_hand-link mount. For the YAM gripper the wrist→hand chain is
    # `wrist_yaw -(ft_sensor_joint 0.072)-> ft_sensor -(hand_palm_joint 0.0095,±0.003)-> rubber_hand`,
    # so the net wrist_yaw-local offset is `xyz="0.0815 ±0.003 0"` (0.072 + 0.0095).
    # See g1_29dof_yam.urdf left_ft_sensor_joint / left_hand_palm_joint (and the
    # right-hand mirror). All chain joints have rpy=0, so R(rubber_hand)==R(wrist_yaw)
    # and the residual reward still cancels the offset on both sides.
    # (The retired Pika gripper used 0.0415 — a direct wrist_yaw→rubber_hand mount.)
    # Shape: (len(body_names), 3) in wrist_yaw_link local frame.
    body_offsets_b: tuple[tuple[float, float, float], ...] = ((0.0815, +0.003, 0.0), (0.0815, -0.003, 0.0))

    # Fixed rotation of the `{side}_ft_sensor_link` (the FORCE-SENSOR frame) relative
    # to its parent `{side}_wrist_yaw_link`, as URDF roll-pitch-yaw. Both hands share
    # the same value: `left/right_ft_sensor_joint` origin rpy="0 -pi/2 pi"
    # (g1_29dof_yam.urdf:877 / :1206). This is a genuine axis-relabel (R symmetric
    # involution, det +1): sensor +x → wrist +z (up), +y → wrist -y, +z → wrist +x
    # (forward/axial), matching the physical Bota FT-sensor convention. Used ONLY by
    # the `sensor_frame=True` obs branch of `force_cmd_local` (the `--F_on_sensor_frame`
    # study) to re-express F^cmd in the sensor frame for the actor; physics, sampling,
    # and the Eq.2 reward are untouched. Default matches the committed yam URDF; a
    # different gripper would override it. The `ft_sensor_link` is a fixed child that
    # PhysX fuses into wrist_yaw (not an addressable sim body), so this constant is
    # composed with the LIVE wrist quat rather than read from a sensor-body quat.
    sensor_rpy: tuple[float, float, float] = (0.0, -1.5707963267948966, 3.141592653589793)

    seed: int = 0


# ---------------------------------------------------------------------------
# Quaternion-vector rotation (wxyz convention; matches isaaclab.utils.math).
# Defined locally so this module is testable without the omni runtime, which
# isaaclab.utils.math transitively imports. Math is the well-known formula
# for unit quaternions: v' = v + 2 * w * (q.xyz × v) + 2 * q.xyz × (q.xyz × v).
# ---------------------------------------------------------------------------


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a wxyz quaternion rotation to a vector. Same semantics as
    `isaaclab.utils.math.quat_apply`."""
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = xyz.cross(v, dim=-1) * 2
    return (v + q[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def _quat_apply_inverse_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply an inverse wxyz quaternion rotation. Same semantics as
    `isaaclab.utils.math.quat_apply_inverse`."""
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = xyz.cross(v, dim=-1) * 2
    return (v - q[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def _rot_matrix_from_rpy(rpy: tuple[float, float, float]) -> torch.Tensor:
    """Build the 3×3 rotation matrix for a URDF `rpy` (extrinsic X-Y-Z: R = Rz·Ry·Rx).

    Returns the child-frame-in-parent-frame rotation `R` (columns = child axes
    expressed in the parent frame), matching how a URDF fixed joint's `origin rpy`
    orients its child link relative to its parent. Used to derive the constant
    wrist→sensor rotation `R_sensor` from `HandForceConfig.sensor_rpy`.
    """
    roll, pitch, yaw = rpy
    cr, sr = torch.cos(torch.tensor(roll)), torch.sin(torch.tensor(roll))
    cp, sp = torch.cos(torch.tensor(pitch)), torch.sin(torch.tensor(pitch))
    cy, sy = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


# ---------------------------------------------------------------------------
# Pulse helpers + obs functions (single-writer rule: only HandForceInjector
# drives _advance_force_pulses; obs functions are pure readers).
# ---------------------------------------------------------------------------


def _allocate_pulse_state(env: Any, cfg: HandForceConfig) -> dict:
    """Allocate the per-env pulse state dict (cmd + ext samplers, last-step
    tracker, RNG). Called once from HandForceInjector.setup() — never lazily
    from obs/reward. Returns the dict so the caller can store it on env."""
    n = env.num_envs
    h = len(cfg.body_names)
    device = env.device
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    return {
        "gen": gen,
        "last_step": torch.full((n,), -1, dtype=torch.long, device=device),
        "cmd": _make_pulse_state(n, h, cfg.f_cmd_max, cfg, gen, device),
        "ext": _make_pulse_state(n, h, cfg.f_ext_max, cfg, gen, device),
    }


def _advance_force_pulses(env: Any, cfg: HandForceConfig) -> None:
    """Advance the per-env trapezoidal force schedule by one env-step.

    Pre-conditions (caller's responsibility — HandForceInjector.setup() does this):
        - env._wbt_force_alpha   : scalar tensor in [0, 1]
        - env._wbt_force_cmd_b   : (num_envs, num_hands, 3) zeros  (root frame)
        - env._wbt_force_ext_b   : (num_envs, num_hands, 3) zeros  (root frame)
        - env._wbt_force_cmd_w   : (num_envs, num_hands, 3) zeros  (world frame)
        - env._wbt_force_ext_w   : (num_envs, num_hands, 3) zeros  (world frame)
        - env._wbt_force_state   : dict from _allocate_pulse_state(env, cfg)

    Idempotent within a single env step (guarded by env.episode_length_buf).
    F^cmd and F^ext have independent per-(env, hand) timing.
    """
    state = env._wbt_force_state
    cur_step = env.episode_length_buf
    if not bool((cur_step != state["last_step"]).any()):
        return  # already advanced for this env-step

    alpha = float(env._wbt_force_alpha)
    env._wbt_force_cmd_b[:] = _step_pulse(state["cmd"], cfg, gen=state["gen"]) * alpha
    env._wbt_force_ext_b[:] = _step_pulse(state["ext"], cfg, gen=state["gen"]) * alpha

    # GH _force_track contract: F is sampled in the ROOT frame and rotated into
    # the WORLD frame here (the single-writer point), so PhysX injection and the
    # Eq.2 reward both consume world-frame forces. root_quat_w is wxyz (IsaacSim
    # native); expand to (num_envs, num_hands, 4) for the per-hand rotation.
    n, h = env.num_envs, len(cfg.body_names)
    root_quat = env.simulator._robot.data.root_quat_w  # (n, 4) wxyz
    root_quat_h = root_quat.unsqueeze(1).expand(n, h, 4)
    env._wbt_force_cmd_w[:] = _quat_apply_wxyz(root_quat_h, env._wbt_force_cmd_b)
    env._wbt_force_ext_w[:] = _quat_apply_wxyz(root_quat_h, env._wbt_force_ext_b)
    state["last_step"] = cur_step.clone()


def _make_pulse_state(num_envs, num_hands, f_max, cfg, gen, device) -> dict:
    """Allocate phase tensors for one trapezoidal-pulse sampler."""
    state = {
        "f_max": float(f_max),
        "f_target": torch.zeros(num_envs, num_hands, 3, device=device),
        "t": torch.zeros(num_envs, num_hands, dtype=torch.long, device=device),
        "ramp_end": torch.zeros(num_envs, num_hands, dtype=torch.long, device=device),
        "hold_end": torch.zeros(num_envs, num_hands, dtype=torch.long, device=device),
        "down_end": torch.zeros(num_envs, num_hands, dtype=torch.long, device=device),
        "idle_end": torch.zeros(num_envs, num_hands, dtype=torch.long, device=device),
    }
    _resample_pulse(state, cfg, gen, device, env_ids=None)
    return state


def _resample_pulse(state, cfg, gen, device, env_ids=None) -> None:
    """Resample (target, ramp/hold/idle durations) for the given env subset."""
    n = state["f_target"].shape[0] if env_ids is None else env_ids.numel()
    h = state["f_target"].shape[1]
    if n == 0:
        return
    f_max = state["f_max"]
    ft = (torch.rand((n, h, 3), generator=gen) * 2 - 1) * f_max
    rs = torch.randint(cfg.ramp_steps[0], cfg.ramp_steps[1] + 1, (n, h), generator=gen)
    hs = torch.randint(cfg.hold_steps[0], cfg.hold_steps[1] + 1, (n, h), generator=gen)
    is_ = torch.randint(cfg.idle_steps[0], cfg.idle_steps[1] + 1, (n, h), generator=gen)

    target_idx = slice(None) if env_ids is None else env_ids
    state["f_target"][target_idx] = ft.to(device)
    state["t"][target_idx] = 0
    state["ramp_end"][target_idx] = rs.to(device)
    state["hold_end"][target_idx] = (rs + hs).to(device)
    state["down_end"][target_idx] = (rs + hs + rs).to(device)  # symmetric ramps
    state["idle_end"][target_idx] = (rs + hs + rs + is_).to(device)


def _step_pulse(state, cfg, gen) -> torch.Tensor:
    """Advance one tick; return current force, auto-resample finished env subsets."""
    t = state["t"].unsqueeze(-1).float()
    re = state["ramp_end"].unsqueeze(-1).float()
    he = state["hold_end"].unsqueeze(-1).float()
    de = state["down_end"].unsqueeze(-1).float()

    in_ramp_up = (t <= re) & (re > 0)
    in_hold = (t > re) & (t <= he)
    in_ramp_down = (t > he) & (t <= de)

    scale = torch.zeros_like(t)
    scale = torch.where(in_ramp_up, t / re.clamp(min=1.0), scale)
    scale = torch.where(in_hold, torch.ones_like(t), scale)
    scale = torch.where(in_ramp_down, (de - t) / (de - he).clamp(min=1.0), scale)
    f_now = state["f_target"] * scale

    state["t"] += 1

    # Auto-resample envs whose entire idle window has elapsed.
    has_idle = (state["idle_end"] > state["down_end"]).all(dim=-1)
    past_idle = (state["t"] >= state["idle_end"]).all(dim=-1)
    done = past_idle & has_idle
    if done.any():
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        _resample_pulse(state, cfg, gen, state["t"].device, env_ids=env_ids)
    return f_now


def force_cmd_local(
    env: Any, *, cfg: HandForceConfig | None = None, hand_frame: bool = False, sensor_frame: bool = False
) -> torch.Tensor:
    """Per-hand commanded force in the ROOT (pelvis) frame by default, shape (num_envs, 6).

    Root frame = the PELVIS frame (the articulation root link `pelvis`, whose
    orientation is `_robot.data.root_quat_w` — the quat used to rotate F into the
    world at _advance_force_pulses). NOT the torso: the G1 has a 3-DOF waist
    (waist_yaw/roll/pitch) between pelvis and torso, so under a whole-body lean
    the pelvis pitches and a pelvis-frame `F^cmd=(Fx,0,0)` acquires a world-down
    component `-Fx·sin(pitch)`. This matches GH _force_track's `force_cmd_history`
    obs (motion_tracking.py:1733, also full-root_quat). The high-level planner
    supplies F^cmd; at deploy it must transform its hand-mounted F/T reading into
    the pelvis frame before sending it to the controller (see the 2026-05-28
    frame-alignment design §6).

    ── HAND-FRAME ABLATION (`hand_frame=True`, the `--F_on_hand_frame` study) ──
    The force is STILL sampled, scheduled, and applied along the PELVIS x/y/z axes
    (the physics and the commanded force are byte-identical to the pelvis-frame
    baseline — see HandForceInjector + the world-frame Eq.2 reward, both untouched
    by this flag). ONLY the actor's OBS ENCODING of F^cmd changes: we re-express
    the same world-frame command vector in each hand's (= wrist_yaw's) own frame:

        F^cmd_hand = R(q_wrist_yaw)^-1 @ F^cmd_world,   F^cmd_world = env._wbt_force_cmd_w

    Rationale: a constant pelvis-frame F^cmd is (near-)TIME-INVARIANT in the obs
    during a hold-plateau even as the arm/wrist swings; expressed in the hand
    frame it becomes TIME-VARIANT (its per-axis components track the live wrist
    rotation), giving the actor a richer pose-relative signal at the contact
    point. Because every joint in the rubber_hand → wrist_yaw chain has rpy="0 0 0"
    (g1_29dof_yam.urdf left_ft_sensor_joint + left_hand_palm_joint, right mirror),
    R(q_rubber_hand) == R(q_wrist_yaw); a force is a free vector (rotation-only,
    translation-invariant), so the URDF offset does NOT enter — the inverse wrist
    rotation is exact. F^cmd=0 ⇒ R^-1·0 = 0, so the F=0⇒2pt collapse and the idle
    phases are preserved bit-for-bit. We rotate the WORLD buffer (_wbt_force_cmd_w),
    NOT the pelvis buffer (_wbt_force_cmd_b): rotating _b would double-count the
    pelvis rotation. The live `body_quat_w` is the post-physics wrist quat (read
    after _refresh_sim_tensors, the SAME tensor the Eq.2 reward's _force_cur_pos_w
    consumes), so the hand-frame obs is time-coherent with the scored contact point.

    ── SENSOR-FRAME ABLATION (`sensor_frame=True`, the `--F_on_sensor_frame` study) ──
    Same actor-obs-only re-encoding as the hand-frame branch, but the target frame is
    the physical FORCE-SENSOR link `{side}_ft_sensor_link` instead of wrist_yaw. UNLIKE
    the rubber_hand (which is axis-aligned with wrist_yaw), the ft_sensor frame is a
    genuine axis-relabel of wrist_yaw: `{side}_ft_sensor_joint` carries rpy="0 -pi/2 pi"
    (g1_29dof_yam.urdf), so `R_sensor` maps sensor +x→wrist +z, +y→wrist -y, +z→wrist +x.
    The obs is:

        F^cmd_sensor = R_sensor^T @ ( R(q_wrist_yaw)^-1 @ F^cmd_world )
                     = R_sensor^T @ F^cmd_hand,   F^cmd_world = env._wbt_force_cmd_w

    i.e. the hand-frame result left-multiplied by the CONSTANT R_sensor^T. We do NOT
    read a live ft_sensor body_quat_w: the ft_sensor link is a fixed child that PhysX
    FUSES into wrist_yaw (it is absent from the 32-body sim list), so the sensor
    orientation in the world is `R(q_wrist_yaw) @ R_sensor` and the world→sensor map is
    its transpose. R_sensor is a constant from `cfg.sensor_rpy` (URDF single source of
    truth), so deploy reproduces it exactly by composing the same constant with its own
    Pinocchio wrist FK (no ft_sensor URDF frame needed for parity). F^cmd=0 ⇒ 0 (a pure
    rotation), so the F=0⇒2pt collapse is preserved bit-for-bit. Mutually exclusive with
    hand_frame (an assert guards misuse).

    The `hand_frame` / `sensor_frame` flags are wired ONLY on the ACTOR term (via
    ObsTermCfg.params), so the CRITIC's force_cmd_local stays pelvis-frame (privileged
    value function in a stable frame). force_ext_local is never re-framed.

    PURE READER — does NOT advance the pulse (single-writer rule, Math Foundation
    §8): even the hand/sensor-frame branches only READ _wbt_force_cmd_w + body_quat_w
    and mutate nothing. The pulse is owned by HandForceInjector which advances it once
    per env-step at substep 0 of _apply_force_in_physics_step.

    The `cfg` kwarg is unused on the pelvis path (kept for API symmetry with reward
    functions); the hand/sensor-frame branches resolve wrist body_names (and, for the
    sensor frame, sensor_rpy) from `cfg or HandForceConfig()`.
    """
    assert not (hand_frame and sensor_frame), (
        "force_cmd_local: hand_frame and sensor_frame are mutually exclusive (--F_on_hand_frame vs --F_on_sensor_frame)"
    )
    if hand_frame or sensor_frame:
        hf_cfg = cfg or HandForceConfig()
        body_ids = _ensure_force_isaac_indices(env, hf_cfg.body_names)
        wrist_quat_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]  # (n, h, 4) wxyz
        f_cmd_hand = _quat_apply_inverse_wxyz(wrist_quat_wxyz, env._wbt_force_cmd_w)  # world -> hand
        if sensor_frame:
            # world -> sensor = R_sensor^T @ (world -> hand). R_sensor is constant.
            r_sensor = _rot_matrix_from_rpy(hf_cfg.sensor_rpy).to(f_cmd_hand)  # (3, 3)
            f_cmd_sensor = torch.einsum("ji,nhj->nhi", r_sensor, f_cmd_hand)  # R_sensor^T @ v
            return f_cmd_sensor.reshape(env.num_envs, -1)
        return f_cmd_hand.reshape(env.num_envs, -1)
    return env._wbt_force_cmd_b.reshape(env.num_envs, -1)


def force_ext_local(env: Any, *, cfg: HandForceConfig | None = None) -> torch.Tensor:
    """Per-hand external disturbance force in the ROOT frame, shape (num_envs, 6).

    Root frame matches `force_cmd_local`. Critic-only (privileged). The world-frame
    `_wbt_force_ext_w` (root-rotated) is what gets injected into PhysX after a
    further world->body-local conversion in HandForceInjector.

    PURE READER — does NOT advance the pulse. See force_cmd_local docstring.
    """
    return env._wbt_force_ext_b.reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Pulse-restart helper (GH `_force_track.sample_init` parity).
#
# Resamples both cmd/ext pulse schedules for `env_ids` (back to t=0, fresh
# target + ramp/hold/idle durations), zeros the root- AND world-frame force
# buffers for those envs, and clears their idempotency marker so the next
# substep-0 advance re-fires. Without it, an env that begins a NEW trajectory
# (full episode reset OR a maskedmimic mid-episode motion-boundary resample)
# while mid-hold-plateau carries a held ~25 N onto the freshly snapped/teleported
# pose — a force shock GH does not have.
#
# This is the SOLE mutation of force state outside `_advance_force_pulses`. It is
# safe to call from any reset path (single-writer rule is about the per-STEP pulse
# advance; a reset legitimately re-phases the schedule). Callers: HandForceInjector.reset()
# (full episode reset) and MaskedMimicMultiMotionCommand.step() (mid-episode
# motion-boundary teleport, which is NOT a full env reset so the injector's reset()
# is not dispatched for those envs). No-op if force buffers are not allocated
# (the env is not running a force preset).
# ---------------------------------------------------------------------------


def restart_force_pulse_for_envs(env: Any, env_ids, *, cfg: "HandForceConfig | None" = None) -> None:
    """Restart the trapezoidal force pulse + zero the force buffers for `env_ids`.

    `cfg` defaults to `env._wbt_force_cfg` (stashed by HandForceInjector.setup()),
    so external callers (the maskedmimic command) need not hold the cfg. No-op if
    the force buffers / state have not been allocated (non-force preset)."""
    if env_ids is None:
        return
    state = getattr(env, "_wbt_force_state", None)
    if state is None or not hasattr(env, "_wbt_force_cmd_b"):
        return  # force preset not active on this env
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return
    cfg = cfg or getattr(env, "_wbt_force_cfg", None) or HandForceConfig()
    gen = state["gen"]
    _resample_pulse(state["cmd"], cfg, gen, env.device, env_ids=env_ids)
    _resample_pulse(state["ext"], cfg, gen, env.device, env_ids=env_ids)
    env._wbt_force_cmd_b[env_ids] = 0.0
    env._wbt_force_ext_b[env_ids] = 0.0
    env._wbt_force_cmd_w[env_ids] = 0.0
    env._wbt_force_ext_w[env_ids] = 0.0
    # Force the next substep-0 advance to re-fire for these envs (a stale last_step
    # would otherwise compare equal and skip the advance).
    state["last_step"][env_ids] = -1


# ---------------------------------------------------------------------------
# HandForceInjector: substep wrench writer + sole pulse-advance owner
# ---------------------------------------------------------------------------
from holosoma.managers.command.base import CommandTermBase  # noqa: E402


class HandForceInjector(CommandTermBase):
    """Sole owner of the force-pulse buffers and the PhysX wrench-write hook.

    setup() does three things:
      1. Allocate env._wbt_force_cmd_b / _wbt_force_ext_b (root-frame buffers
         read by obs) plus _wbt_force_cmd_w / _wbt_force_ext_w (world-frame
         buffers read by reward + PhysX injection).
      2. Allocate env._wbt_force_state via _allocate_pulse_state(env, cfg).
      3. Install a monkey-patch on env._apply_force_in_physics_step.

    The patched hook (called at every PhysX substep) does:
      a. _maybe_advance_pulse(): on substep 0 of each env-step, advance the
         pulse to F(k). Idempotent via env.episode_length_buf so substeps
         1..decimation-1 are no-ops.
      b. _inject_force_ext(): write the body-local F^ext wrench to PhysX.

    This is the SINGLE place that calls _advance_force_pulses. Reward and obs
    are pure readers of env._wbt_force_*_b — they MUST NOT advance the pulse
    because episode_length_buf increments before they run, which would desync
    F(k+1) reads against F(k)-integrated cur_pos_w. See Math Foundation §8.

    F^cmd is NOT written to PhysX (only F^ext). F^cmd lives in the obs/reward
    channels only — that's UniFP's whole point: implicit force tracking via
    position residual, no force sensor.

    Backend assumption (IsaacSim only): the lockstep guarantee in Math
    Foundation §8 relies on `_compute_reward` running *before* `command_manager.step()`.
    Holosoma's `BaseTask` selects this ordering for IsaacSim (`base_task.py:457-458`,
    where `_update_tasks_before_termination` is False — see `base_task.py:125`).
    For the IsaacGym manager, `_update_tasks_callback` runs at line 440 *before*
    `_compute_reward`, so motion_command would advance to k+1 at reward time and
    the (cur(k), ref(k+1), F(k)) triple would be desynced by one step. This
    preset (g1_wbt_2pt_force) targets IsaacSim exclusively, but if this preset
    is ever ported to IsaacGym the timing guarantee must be re-derived.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hf_cfg: HandForceConfig = cfg.params["hand_force_config"]
        self.body_names = list(self.hf_cfg.body_names)

        # Resolved at setup() time.
        self.body_indices: torch.Tensor | None = None
        self._isaac_body_ids: torch.Tensor | None = None
        self._body_offsets_b: torch.Tensor | None = None
        self._original_apply_force = None

    def setup(self) -> None:
        env = self.env
        cfg = self.hf_cfg

        # 1. Allocate force buffers (single source of truth read by obs/reward).
        n, h = env.num_envs, len(self.body_names)
        env._wbt_force_cmd_b = torch.zeros(n, h, 3, device=env.device)
        env._wbt_force_ext_b = torch.zeros(n, h, 3, device=env.device)
        env._wbt_force_cmd_w = torch.zeros(n, h, 3, device=env.device)
        env._wbt_force_ext_w = torch.zeros(n, h, 3, device=env.device)
        env._wbt_force_state = _allocate_pulse_state(env, cfg)
        # Surface the cfg on env so a non-owning writer (e.g. the maskedmimic
        # multi-motion command's mid-episode motion-boundary resample) can restart
        # the pulse via `restart_force_pulse_for_envs(env, env_ids)` without holding
        # a reference to this injector instance.
        env._wbt_force_cfg = cfg
        # Surface the URDF offset on env so reward/obs can reuse the same
        # tensor for the FK reconstruction (rubber_hand_pos_w = wrist_yaw_pos_w
        # + R(wrist_yaw_quat) @ offset_b).
        env._wbt_force_offsets_b = torch.tensor(cfg.body_offsets_b, dtype=torch.float, device=env.device)

        # 2. Resolve wrist_yaw body indices (force application bodies).
        sim = env.simulator
        idx_list = [sim.find_rigid_body_indice(name) for name in self.body_names]
        self.body_indices = torch.tensor(idx_list, dtype=torch.long, device=env.device)
        if hasattr(sim, "body_ids"):
            self._isaac_body_ids = torch.tensor(
                [sim.body_ids[i] for i in idx_list], dtype=torch.long, device=env.device
            )
        else:
            self._isaac_body_ids = self.body_indices.clone()
        self._body_offsets_b = env._wbt_force_offsets_b

        # 3. Monkey-patch the substep hook.
        self._original_apply_force = env._apply_force_in_physics_step

        def _patched_apply_force():
            self._original_apply_force()
            self._maybe_advance_pulse()  # substep 0 → advance F to env-step k
            self._inject_force_ext()  # every substep → write same F(k) wrench

        env._apply_force_in_physics_step = _patched_apply_force

    def reset(self, env_ids) -> None:
        # Restart the force pulse for the reset envs — GH `_force_track.sample_init`
        # parity (motion_tracking.py:1412-1434). Without this the pulse phase and
        # force buffers leak across episode boundaries, so an env that resets
        # mid-hold-plateau begins its NEW episode with a held ~25 N already on the
        # hand while the robot has just been snapped back to its reference pose —
        # a start-of-episode force shock GH does not have (it also zeros force +
        # restarts the pulse on every reset).
        #
        # We (a) resample both cmd/ext pulse schedules for env_ids (back to t=0,
        # fresh target + ramp/hold/idle durations), (b) zero the root- AND
        # world-frame force buffers for those envs, and (c) clear their
        # idempotency marker so the next substep-0 advance re-fires for them.
        restart_force_pulse_for_envs(self.env, env_ids, cfg=self.hf_cfg)

    def step(self) -> None:
        # No-op: pulse advance is locked to the substep-0 hook, not
        # command_manager.step() (which runs after reward in _post_physics_step
        # and is too late for the next env-step's first PhysX substep).
        return

    def _maybe_advance_pulse(self) -> None:
        """Advance the pulse exactly once per env-step (substep 0 of physics).
        Idempotency keyed off env._wbt_force_state["last_step"], maintained by
        _advance_force_pulses itself."""
        _advance_force_pulses(self.env, self.hf_cfg)

    def _inject_force_ext(self) -> None:
        env = self.env
        if not hasattr(env, "_wbt_force_ext_b"):
            return  # safety: buffers not yet allocated
        sim = env.simulator
        robot = sim._robot
        if not hasattr(robot, "set_external_force_and_torque"):
            return

        # GH _force_track contract: F_ext is stored in the WORLD frame
        # (root-rotated in _advance_force_pulses). IsaacLab 2.1 hardcodes
        # is_global=False, so set_external_force_and_torque consumes BODY-LOCAL
        # forces. Convert world -> body-local with the live wrist quaternion,
        # exactly like holosoma's push agent (agents/callbacks/push.py:228-231).
        body_quat_w = robot.data.body_quat_w[:, self._isaac_body_ids, :]  # (n, h, 4) wxyz
        force_body = _quat_apply_inverse_wxyz(body_quat_w, env._wbt_force_ext_w)  # world -> body-local
        torques = torch.zeros_like(force_body)

        # Apply the wrench at the rubber_hand offset within the wrist_yaw
        # local frame. Rigid-body-equivalent to applying force on the
        # rubber_hand_link itself (which holosoma's 32-body list cannot address).
        # IsaacLab broadcasts (1, num_bodies, 3) → (num_envs, num_bodies, 3).
        # _body_offsets_b is guaranteed non-None: set by setup(), and this
        # method is only invoked by the patched hook setup() installs.
        assert self._body_offsets_b is not None
        positions = self._body_offsets_b.unsqueeze(0).expand(env.num_envs, -1, -1)

        env_ids = torch.arange(env.num_envs, device=env.device)
        robot.set_external_force_and_torque(
            forces=force_body, torques=torques, positions=positions, env_ids=env_ids, body_ids=self._isaac_body_ids
        )


# ---------------------------------------------------------------------------
# Reward functions: UniFP Eq.2 position-shifted tracking + safety penalty
# (see Math Foundation §5-6 in the plan for derivation and constants)
#
# Force application body: *_wrist_yaw_link is the parent link PhysX targets
# (it IS in holosoma's 32-body list). The actual contact point is the
# fixed-child *_rubber_hand_link at offset `cfg.body_offsets_b` from wrist_yaw.
# `HandForceInjector._inject_force_ext` uses IsaacLab's `positions=` kwarg to
# apply the wrench at that offset — rigid-body-equivalent to applying force on
# the rubber_hand link itself. cur_pos_w and motion_ref are reconstructed via
# forward kinematics: `wrist_yaw_pos_w + R(wrist_yaw_quat) @ offset_b`. Both
# sides include the same offset, so the comparison stays self-consistent.
#
# Quaternion convention note: both the ref and cur FK sides rotate the offset
# by the LIVE `_robot.data.body_quat_w` (IsaacSim native WXYZ) via
# `_quat_apply_wxyz`, so the offset cancels exactly in `cur_w - target_w`
# regardless of wrist orientation. The motion quaternion is not used here.
# ---------------------------------------------------------------------------


def _cached_indices(env: Any, attr: str, body_names, resolve) -> torch.Tensor:
    """Per-`body_names` index cache on `env`, keyed by `tuple(body_names)`.

    ⚠️ The key matters (review 2026-08-05). These caches previously stored a BARE tensor and
    returned it before even looking at `body_names`, so the FIRST caller's body list won
    process-wide and any later caller with a different list silently received the first
    one's indices — i.e. it would read the wrong bodies with no error. Benign at the time
    only because every caller happened to pass the same two wrist names, but the failure is
    invisible, so it is keyed properly now.

    A bare tensor under `attr` is still honoured: several unit tests pre-seed these
    attributes directly (test_hybrid.py, test_hybrid_unified.py) to bypass mock resolution.
    That legacy form is a deliberate test escape hatch, not the production path.
    """
    cache = getattr(env, attr, None)
    if isinstance(cache, torch.Tensor):
        return cache  # legacy/test-injected form: one tensor, no key
    key = tuple(body_names)
    if isinstance(cache, dict):
        hit = cache.get(key)
        if isinstance(hit, torch.Tensor):
            return hit
    else:
        cache = {}
        setattr(env, attr, cache)
    idx = resolve()
    cache[key] = idx
    return idx


def _ensure_force_motion_indices(env: Any, body_names) -> torch.Tensor:
    """Cache the `body_names` indices into motion_command.body_pos_w.
    body_names_to_track is fixed across an experiment run, so resolved once per body list."""

    def _resolve() -> torch.Tensor:
        mc = env.command_manager.get_state("motion_command")
        names = list(mc.motion_cfg.body_names_to_track)
        return torch.tensor([names.index(n) for n in body_names], dtype=torch.long, device=env.device)

    return _cached_indices(env, "_wbt_force_motion_indices", body_names, _resolve)


def _ensure_force_isaac_indices(env: Any, body_names) -> torch.Tensor:
    """Cache the `body_names` IsaacSim body_ids on env (PhysX index space — see
    `hybrid.HybridContactInjector._sync_distal_mass` for why that distinction matters)."""

    def _resolve() -> torch.Tensor:
        sim = env.simulator
        idx_list = [sim.find_rigid_body_indice(name) for name in body_names]
        if hasattr(sim, "body_ids"):
            idx_list = [sim.body_ids[i] for i in idx_list]
        return torch.tensor(idx_list, dtype=torch.long, device=env.device)

    return _cached_indices(env, "_wbt_force_isaac_indices", body_names, _resolve)


def _ensure_force_offsets_b(env: Any, cfg: HandForceConfig) -> torch.Tensor:
    """Cache the URDF wrist_yaw → rubber_hand offsets on env.
    Allocated by `HandForceInjector.setup()` in production; this fallback covers
    the unit-test path where reward functions are exercised without the injector."""
    off = getattr(env, "_wbt_force_offsets_b", None)
    if off is not None and isinstance(off, torch.Tensor):
        return off
    off = torch.tensor(cfg.body_offsets_b, dtype=torch.float, device=env.device)
    env._wbt_force_offsets_b = off
    return off


def _force_ref_pos_w(env: Any, cfg: HandForceConfig) -> torch.Tensor:
    """World-frame rubber_hand motion reference (yaw-relocated to robot's
    current root xy + yaw, with z absolute), reconstructed via FK on wrist_yaw.

    Reads `body_pos_relative_w` / `body_quat_relative_w` (NOT the raw
    `body_pos_w`), matching:
      - GH `reward_keypoints_w` (motion_tracking.py:443-446)
      - holosoma baseline `motion_relative_body_position_error_exp`
        (reward/terms/wbt.py:87-90)

    Why yaw-relocated, not raw motion world frame: cur_w (live FK) tracks the
    robot's current yaw. If ref_w stayed on the motion file's absolute yaw,
    every degree of yaw drift would inject hidden cross-track error into
    `cur_w − ref_w`, eating the σ=0.3 m budget independent of force-tracking
    quality. See plan §F (frame-consistency review) for the full audit.

    Returns shape (num_envs, len(body_names), 3). The position anchor is
    `body_pos_relative_w` (yaw-relocated motion ref); the rubber_hand offset is
    rotated by the live `_robot.data.body_quat_w` (wxyz) so it cancels against
    `_force_cur_pos_w`. See the 2026-05-28 frame-alignment design §4.2.
    """
    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, cfg.body_names)
    offsets_b = _ensure_force_offsets_b(env, cfg)
    body_ids = _ensure_force_isaac_indices(env, cfg.body_names)
    wrist_pos_ref_w = mc.body_pos_relative_w[:, fm_idx, :]  # (n, h, 3) yaw-aligned motion ref
    # Rotate the rubber_hand offset by the LIVE wrist quaternion (wxyz) — the
    # SAME quaternion _force_cur_pos_w uses — so the offset cancels exactly in
    # cur_w - target_w. (Previously this used the motion quat, which only cancels
    # when the wrist orientation is already perfectly tracked.)
    wrist_quat_cur_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]  # (n, h, 4) wxyz
    n_bodies = len(cfg.body_names)
    offsets_b_b = offsets_b.unsqueeze(0).expand(env.num_envs, n_bodies, -1)
    return wrist_pos_ref_w + _quat_apply_wxyz(wrist_quat_cur_wxyz, offsets_b_b)


def _force_cur_pos_w(env: Any, cfg: HandForceConfig) -> torch.Tensor:
    """World-frame live rubber_hand position, reconstructed via FK on wrist_yaw.

    `_robot.data.body_quat_w` is wxyz (IsaacSim native).
    """
    body_ids = _ensure_force_isaac_indices(env, cfg.body_names)
    offsets_b = _ensure_force_offsets_b(env, cfg)
    wrist_pos_w = env.simulator._robot.data.body_pos_w[:, body_ids, :]
    wrist_quat_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]
    n_bodies = len(cfg.body_names)
    offsets_b_b = offsets_b.unsqueeze(0).expand(env.num_envs, n_bodies, -1)
    return wrist_pos_w + _quat_apply_wxyz(wrist_quat_wxyz, offsets_b_b)


def _force_eq2_target_w(env: Any, cfg: HandForceConfig) -> torch.Tensor:
    """Position-shifted Eq.2 target in world frame: ref + (F^cmd + F^ext) / K.

    Implements the simplified Eq.2 (paper §3.2): x^target = x^cmd + (F^cmd + F^ext) / K.
    See Math Foundation §3 and the 2026-05-28 frame-alignment design.

    Frame convention: F^cmd_w / F^ext_w are stored in the WORLD frame
    (root-rotated in _advance_force_pulses, matching GH _force_track), so the
    position shift adds directly to the world-frame ref. The rubber_hand FK
    offset is rotated by the LIVE wrist quaternion on both the ref and cur sides
    (_force_ref_pos_w / _force_cur_pos_w) so it cancels in the residual.
    """
    ref_w = _force_ref_pos_w(env, cfg)
    # F_cmd_w / F_ext_w are already world-frame (root-rotated in
    # _advance_force_pulses), so no per-step rotation is needed here.
    f_total_w = env._wbt_force_cmd_w + env._wbt_force_ext_w  # (n, h, 3) world frame
    return ref_w + f_total_w / cfg.stiffness_K


def force_track_pos_eq2(env: Any, *, cfg: HandForceConfig | None = None) -> torch.Tensor:
    """UniFP Eq.2 position-shifted tracking reward — exact UniFP form for two hands.

        err_l1 = Σ_{h ∈ L,R} Σ_{xyz} | cur_h^w − (ref_h^w + (F^cmd_h^w + F^ext_h^w) / K) |
        r      = exp(−err_l1 · 2 / σ)        (single scalar per env, NOT mean(exp(...)))

    PURE READER of env._wbt_force_*_w (the world-frame buffers — root-rotated in
    _advance_force_pulses; consumed here via _force_eq2_target_w). Does not
    advance the pulse (single-writer rule, see Math Foundation §8). Matches
    `_reward_tracking_ee_force_world` in UniFP `legged_robot_b2z1_pos_force.py:1891-1900`.
    """
    cfg = cfg or HandForceConfig()
    target_w = _force_eq2_target_w(env, cfg)
    cur_w = _force_cur_pos_w(env, cfg)
    err_l1 = (cur_w - target_w).abs().sum(dim=(-2, -1))  # sum over (hands, xyz) → (n,)
    return torch.exp(-err_l1 * 2.0 / cfg.sigma)


def force_track_pos_pen(env: Any, *, cfg: HandForceConfig | None = None) -> torch.Tensor:
    """Anti-reward-hacking safety penalty on raw motion-ref drift.

    Returns a POSITIVE [0, 1] term:
        r = mean over hands of {1 if ‖cur^w − ref^w‖₂ > 0.6 m else 0}

    The negative sign is applied at scoring time via the RewardTermCfg weight
    (default -6.0). Holosoma convention: penalty *func* returns positive
    magnitude, weight is negative — never both.

    PURE READER (does not advance the pulse).
    """
    cfg = cfg or HandForceConfig()
    ref_w = _force_ref_pos_w(env, cfg)
    cur_w = _force_cur_pos_w(env, cfg)
    drift = (cur_w - ref_w).norm(dim=-1)  # (n, h) raw L2 drift per hand
    overshoot = (drift > cfg.pen_threshold).float()
    return overshoot.mean(dim=-1)  # (n,) ∈ [0, 1]


# ---------------------------------------------------------------------------
# GLOBAL-FRAME Eq.2 variant (for the maskedmimic-force preset).
#
# The 2pt-force preset above anchors the Eq.2 target on the YAW-RELOCATED motion
# reference (`body_pos_relative_w`) because the 2pt baseline tracks bodies in the
# robot-anchored relative frame. The maskedmimic preset instead tracks bodies in
# the ABSOLUTE WORLD frame (`motion_global_body_position_error_exp` reads
# `mc.body_pos_w`, which already includes `env_origins`). To keep Eq.2 consistent
# with maskedmimic's own position reward — comparing global-to-global — the global
# variant anchors the shifted target on `mc.body_pos_w` (the same field maskedmimic's
# global body reward consumes) rather than `body_pos_relative_w`.
#
# Everything else is identical to the relative variant: the rubber_hand FK offset is
# rotated by the LIVE wrist quaternion on BOTH the cur and ref sides so it cancels
# exactly in the residual, F^cmd_w/F^ext_w are already world-frame (root-rotated in
# _advance_force_pulses), and the reduction is exp(−err_l1·2/σ) over (hands, xyz).
# F=0 ⇒ target = ref ⇒ identical to pure global tracking (the F=0⇒tracking invariant).
# These are PURE READERS — they never advance the pulse (single-writer rule, §8).
# ---------------------------------------------------------------------------


def _force_ref_pos_w_global(env: Any, cfg: HandForceConfig) -> torch.Tensor:
    """World-frame rubber_hand motion reference in the ABSOLUTE motion frame
    (`mc.body_pos_w`, incl. env_origins), reconstructed via FK on wrist_yaw.

    Differs from `_force_ref_pos_w` ONLY in the position anchor: this reads the
    absolute global motion ref (matching maskedmimic's `motion_global_*` rewards),
    whereas `_force_ref_pos_w` reads the yaw-relocated `body_pos_relative_w`
    (matching the 2pt relative-frame baseline). The FK offset is rotated by the
    LIVE wrist quaternion (the SAME quat `_force_cur_pos_w` uses) so it cancels.
    """
    mc = env.command_manager.get_state("motion_command")
    fm_idx = _ensure_force_motion_indices(env, cfg.body_names)
    offsets_b = _ensure_force_offsets_b(env, cfg)
    body_ids = _ensure_force_isaac_indices(env, cfg.body_names)
    wrist_pos_ref_w = mc.body_pos_w[:, fm_idx, :]  # (n, h, 3) absolute global motion ref
    wrist_quat_cur_wxyz = env.simulator._robot.data.body_quat_w[:, body_ids, :]  # (n, h, 4) wxyz
    n_bodies = len(cfg.body_names)
    offsets_b_b = offsets_b.unsqueeze(0).expand(env.num_envs, n_bodies, -1)
    return wrist_pos_ref_w + _quat_apply_wxyz(wrist_quat_cur_wxyz, offsets_b_b)


def _force_eq2_target_w_global(env: Any, cfg: HandForceConfig) -> torch.Tensor:
    """Position-shifted Eq.2 target in the ABSOLUTE world frame:
    ref_global + (F^cmd_w + F^ext_w) / K. See `_force_eq2_target_w` for the
    relative-frame analogue; only the ref anchor differs (global vs yaw-relocated)."""
    ref_w = _force_ref_pos_w_global(env, cfg)
    f_total_w = env._wbt_force_cmd_w + env._wbt_force_ext_w  # (n, h, 3) world frame
    return ref_w + f_total_w / cfg.stiffness_K


def force_track_pos_eq2_global(env: Any, *, cfg: HandForceConfig | None = None) -> torch.Tensor:
    """UniFP Eq.2 position-shifted tracking reward, ABSOLUTE-world-frame variant
    (for the maskedmimic-force preset).

        err_l1 = Σ_{h ∈ L,R} Σ_{xyz} | cur_h^w − (ref_h^global + (F^cmd_h^w + F^ext_h^w) / K) |
        r      = exp(−err_l1 · 2 / σ)        (single scalar per env)

    Identical math to `force_track_pos_eq2` except the ref anchor is the absolute
    global motion ref (`mc.body_pos_w`) so it is consistent with maskedmimic's own
    `motion_global_body_position_error_exp`. PURE READER (does not advance the pulse)."""
    cfg = cfg or HandForceConfig()
    target_w = _force_eq2_target_w_global(env, cfg)
    cur_w = _force_cur_pos_w(env, cfg)
    err_l1 = (cur_w - target_w).abs().sum(dim=(-2, -1))  # sum over (hands, xyz) → (n,)
    return torch.exp(-err_l1 * 2.0 / cfg.sigma)


# ---------------------------------------------------------------------------
# motion_relative_body_position_error_exp_no_wrist:
# A drop-in replacement for the holosoma baseline 14-link relative
# position-tracking reward
# (`holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp`)
# that excludes the wrist_yaw_link indices from the per-body mean. With the
# default `exclude_body_names`, this is a 12-body mean.
#
# Why: under UniFP Eq.2 the rubber_hand target is `ref + (F^cmd + F^ext) / K`
# (per-axis shift up to (25+25)/200 = 12.5 cm from each of F^cmd and F^ext, so a
# combined peak ≈ 25 cm at K=200 N/m since they are independent pulses). The
# baseline 14-link reward
# asks the wrist to track raw `ref` while `force_track_pos_eq2` asks the
# rubber_hand (rigidly attached to the wrist) to track the shifted target.
# With both active the policy receives two competing position targets on the
# same rigid sub-chain and must compromise at the 1.0:4.0 weighting — neither
# reward saturates. Removing wrist from this term hands the wrist's position
# entirely to Eq.2.
#
# Only POSITION is excluded for the wrist. ORIENTATION is handled separately:
# the `g1_29dof_wbt_2pt_force_reward` preset KEEPS the full 14-body
# `motion_relative_body_orientation_error_exp` (incl. wrist_yaw == hand
# orientation), so the hand contact pose IS supervised. The companion
# `motion_relative_body_orientation_error_exp_no_wrist` function below is kept
# for reuse / ablation but is NOT wired into this preset.
# ---------------------------------------------------------------------------


def _ensure_no_wrist_kept_indices(env: Any, exclude_body_names: tuple[str, ...]) -> torch.Tensor:
    """Cache the kept (non-excluded) indices into motion_command.body_pos_relative_w.

    Resolved once: `body_names_to_track` and `exclude_body_names` are both
    fixed across an experiment run.
    """
    cached = getattr(env, "_wbt_force_no_wrist_kept_indices", None)
    if isinstance(cached, torch.Tensor):
        return cached
    mc = env.command_manager.get_state("motion_command")
    names = list(mc.motion_cfg.body_names_to_track)
    kept = [i for i, n in enumerate(names) if n not in exclude_body_names]
    idx = torch.tensor(kept, dtype=torch.long, device=env.device)
    env._wbt_force_no_wrist_kept_indices = idx
    return idx


def motion_relative_body_position_error_exp_no_wrist(
    env: Any, sigma: float = 0.3, exclude_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
) -> torch.Tensor:
    """14-link relative position-tracking reward with wrist excluded.

    Math identical to upstream
    `holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp`
    except the per-body mean is taken only over indices not in
    `exclude_body_names`:

        diff[n, b] = body_pos_relative_w[n, b] - robot_body_pos_w[n, b]
        err[n, b]  = Σ_xyz diff[n, b]²
        r[n]       = exp(-mean_{b ∈ kept} err[n, b] / σ²)
    """
    mc = env.command_manager.get_state("motion_command")
    kept = _ensure_no_wrist_kept_indices(env, exclude_body_names)
    diff = mc.body_pos_relative_w[:, kept, :] - mc.robot_body_pos_w[:, kept, :]
    error = torch.sum(diff.square(), dim=-1)  # (num_envs, len(kept))
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_position_error_exp_no_wrist(
    env: Any, sigma: float = 0.3, exclude_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
) -> torch.Tensor:
    """GLOBAL body position-tracking reward with the wrist bodies excluded.

    The maskedmimic-force analogue of `motion_relative_body_position_error_exp_no_wrist`:
    same exclude-the-wrist logic so Eq.2 owns the wrist position alone (no tug-of-war
    on the rubber_hand rigid sub-chain), but reads the ABSOLUTE world frame to match
    maskedmimic's own `motion_global_body_position_error_exp`
    (`maskedmimic_observations:motion_global_body_position_error_exp`):

        diff[n, b] = body_pos_w[n, b] - robot_body_pos_w[n, b]   (both absolute world)
        err[n, b]  = Σ_xyz diff[n, b]²
        r[n]       = exp(-mean_{b ∈ kept} err[n, b] / σ²)

    PURE READER. Uses the shared `_ensure_no_wrist_kept_indices` cache (indices into
    `body_names_to_track`, identical for both the relative and global variants).
    """
    mc = env.command_manager.get_state("motion_command")
    kept = _ensure_no_wrist_kept_indices(env, exclude_body_names)
    diff = mc.body_pos_w[:, kept, :] - mc.robot_body_pos_w[:, kept, :]
    error = torch.sum(diff.square(), dim=-1)  # (num_envs, len(kept))
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp_no_wrist(
    env: Any, sigma: float = 0.4, exclude_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
) -> torch.Tensor:
    """14-link relative orientation-tracking reward with wrist excluded.

    Math identical to upstream
    `holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp`
    except the per-body mean is taken only over indices not in
    `exclude_body_names`:

        err[n, b] = quat_error_magnitude(body_quat_relative_w[n, b],
                                         robot_body_quat_w[n, b]) ** 2
        r[n]      = exp(-mean_{b in kept} err[n, b] / sigma**2)

    NOT wired into `g1_29dof_wbt_2pt_force_reward` (kept for reuse / ablation).
    The preset deliberately KEEPS the full 14-body baseline orientation term,
    including wrist_yaw, because wrist_yaw orientation == rubber_hand orientation
    (rigid offset) — the contact pose a manipulation task must track. The earlier
    rationale for excluding the wrist ("Eq.2 shifts the hand, achievable only by
    rotating the wrist, so orientation tracking fights Eq.2") was WRONG: the FK
    offset cancels in the Eq.2 residual, so Eq.2 rewards wrist-ORIGIN translation
    (reach the arm out by F/K), not wrist rotation — orientation tracking and
    Eq.2 are orthogonal. Both quaternions are xyzw (see wbt.py:812, :909).
    """
    mc = env.command_manager.get_state("motion_command")
    kept = _ensure_no_wrist_kept_indices(env, exclude_body_names)
    error = quat_error_magnitude(mc.body_quat_relative_w[:, kept, :], mc.robot_body_quat_w[:, kept, :]) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


# ---------------------------------------------------------------------------
# feet_slip: penalise horizontal foot velocity *while in contact* (port of GH
# `active_adaptation/envs/mdp/rewards/locomotion.py:125-141`).
#
# Math (GH form):
#     in_contact[e, f] = 1 if ankle f of env e touched ground this frame else 0
#     foot_v_xy[e, f]  = robot body linear velocity in world, xy components
#     slip[e]          = Σ_f in_contact[e, f] · ‖foot_v_xy[e, f]‖²
#     return -slip                                          (GH: negative func)
#
# UniFP rationale: F^ext at hand pulls CoM sideways; cheapest way to balance is
# to slide the stance foot. Without this term, policy reward-hacks via foot
# slip → looks fine in sim, deploy fails (real friction is bounded).
#
# SIGN CONVENTION (holosoma) — this function returns a POSITIVE magnitude and
# the cfg supplies `weight=-2.0`. GH does it the other way (func returns
# negative, weight=+2.0). Same loss contribution; we follow holosoma's
# `force_track_pos_pen` / `action_rate_l2` convention so the W&B reward log
# reads as a magnitude rather than a signed term.
#
# Contact detection: holosoma exposes `simulator.contact_forces_history` of
# shape (num_envs, history_length, num_bodies, 3). We threshold the L2 norm
# (>1 N) over history to decide "in contact this env-step" — same predicate
# `UndesiredContacts` (`thirdparty/holosoma/.../reward/terms/wbt.py:149-156`)
# already uses. GH uses `contact_sensor.data.current_contact_time > 0.0` which
# is a strictly tighter window (single substep) but the difference is below
# the timescale of foot kinematics.
#
# Velocity source: `simulator._rigid_body_vel` shape (num_envs, num_bodies, 3)
# — same backing tensor `MotionCommand.robot_body_lin_vel_w` reads from
# (`thirdparty/holosoma/.../command/terms/wbt.py:912-913`).
# ---------------------------------------------------------------------------


def _ensure_feet_slip_indices(env: Any, body_names: tuple[str, ...]) -> torch.Tensor:
    """Cache simulator body indices for the configured ankle links.

    Uses the `getattr(...)` + `isinstance(..., Tensor)` pattern (not `hasattr`)
    because MagicMock-backed test envs auto-create attributes on access; an
    `hasattr` cache check would trip on every call in tests.
    """
    cached = getattr(env, "_wbt_feet_slip_body_indices", None)
    if isinstance(cached, torch.Tensor):
        return cached
    sim_body_names = list(env.simulator.body_names)
    missing = [n for n in body_names if n not in sim_body_names]
    if missing:
        raise ValueError(f"feet_slip: bodies {missing} not in simulator.body_names")
    idx = torch.tensor([sim_body_names.index(n) for n in body_names], dtype=torch.long, device=env.device)
    env._wbt_feet_slip_body_indices = idx
    return idx


def feet_slip(
    env: Any,
    body_names: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Sum-of-squared horizontal foot velocity while in contact, per env.

    Returns POSITIVE magnitude `[num_envs]` (sign supplied by RewardTermCfg
    `weight=-2.0` at scoring time, see GH parity comment above).
    """
    idx = _ensure_feet_slip_indices(env, body_names)

    # contact_forces_history: (num_envs, history_length, num_bodies, 3)
    forces_hist = env.simulator.contact_forces_history[:, :, idx, :]  # (N, H, F, 3)
    force_norm_max_over_history = forces_hist.norm(dim=-1).amax(dim=1)  # (N, F)
    in_contact = (force_norm_max_over_history > contact_force_threshold).float()

    # _rigid_body_vel: (num_envs, num_bodies, 3) — world-frame linear velocity
    feet_vel_xy = env.simulator._rigid_body_vel[:, idx, :2]  # (N, F, 2)
    speed_sq = feet_vel_xy.square().sum(dim=-1)  # (N, F)

    return (in_contact * speed_sq).sum(dim=-1)  # (N,) positive magnitude


# ---------------------------------------------------------------------------
# ForceAlphaCurriculum: start at 0, ramp to 1 based on average episode length
# ---------------------------------------------------------------------------
from holosoma.managers.curriculum.base import CurriculumTermBase  # noqa: E402


class ForceAlphaCurriculum(CurriculumTermBase):
    """Linear curriculum on `env._wbt_force_alpha`.

    Sole owner of `env._wbt_force_alpha` allocation. Starts at
    `cfg.force_alpha_init` (default 0.0 — NO force at training step 0).
    Each reset() reads `env.average_episode_length` (maintained by
    `AverageEpisodeLengthTracker`, which the WBT base curriculum already
    registers); ramps `force_alpha` up by `alpha_step` when episodes are long
    enough (>= level_up_threshold) and down when too short
    (< level_down_threshold). Clamped to `[0, force_alpha_max]`.

    `_advance_force_pulses` reads `env._wbt_force_alpha` each env-step and
    scales F^cmd / F^ext accordingly — so alpha=0 means truly zero forces both
    in obs and in PhysX.

    Setup order in holosoma `BaseTask.__init__`: `command_manager.setup()`
    runs at base_task.py:170 (allocating _wbt_force_cmd_b/_ext_b/_state via
    HandForceInjector), then `curriculum_manager.setup()` runs at line 172
    (allocating _wbt_force_alpha here). The two managers cooperate via
    env-attached buffers; neither depends on a private attribute of the other.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hf_cfg: HandForceConfig = cfg.params["hand_force_config"]

    def setup(self) -> None:
        # Allocate _wbt_force_alpha. Use getattr+isinstance instead of hasattr
        # to handle MagicMock test doubles (hasattr is always True on MagicMock).
        existing = getattr(self.env, "_wbt_force_alpha", None)
        if isinstance(existing, torch.Tensor):
            existing.fill_(self.hf_cfg.force_alpha_init)
        else:
            self.env._wbt_force_alpha = torch.tensor(
                self.hf_cfg.force_alpha_init, device=self.env.device, dtype=torch.float
            )

    def reset(self, env_ids) -> None:
        # CurriculumManager calls this on every env reset; we update the global
        # alpha based on the running average_episode_length.
        #
        # IMPORTANT: `WholeBodyTrackingManager(BaseTask)` does NOT inherit from
        # `LocomotionManager`, so it has no `env.average_episode_length`
        # property — only the underlying `AverageEpisodeLengthTracker`
        # registered on `curriculum_manager`. We read the tracker directly
        # (same path as `wbt_manager.py:_update_log_dict` line 74). Falling
        # back to `getattr(env, "average_episode_length", ...)` would be
        # `None` in production WBT and silently no-op the curriculum — which
        # is exactly the bug we're fixing here. See plan-summary doc Q5.
        cm = getattr(self.env, "curriculum_manager", None)
        if cm is not None and hasattr(cm, "get_term"):
            tracker = cm.get_term("average_episode_tracker")
            if tracker is not None and hasattr(tracker, "get_average"):
                avg = float(tracker.get_average())
            else:
                return
        else:
            # Unit-test path: env.average_episode_length set directly as a
            # Tensor or float on a MagicMock — preserve test compatibility.
            avg_attr = getattr(self.env, "average_episode_length", None)
            if isinstance(avg_attr, torch.Tensor):
                avg = float(avg_attr)
            elif isinstance(avg_attr, (int, float)):
                avg = float(avg_attr)
            else:
                return
        cur = float(self.env._wbt_force_alpha)
        if avg >= self.hf_cfg.level_up_threshold:
            cur += self.hf_cfg.alpha_step
        elif avg < self.hf_cfg.level_down_threshold:
            cur -= self.hf_cfg.alpha_step
        cur = max(0.0, min(self.hf_cfg.force_alpha_max, cur))
        self.env._wbt_force_alpha.fill_(cur)
        log_dict = getattr(self.env, "log_dict", None)
        if isinstance(log_dict, dict):
            log_dict["force_alpha"] = self.env._wbt_force_alpha.detach().clone()

    def step(self) -> None:
        # Clamp guard (defensive — reset() already clamps; this catches any
        # accidental external mutation).
        cur = float(self.env._wbt_force_alpha)
        cur = max(0.0, min(self.hf_cfg.force_alpha_max, cur))
        self.env._wbt_force_alpha.fill_(cur)


# ---------------------------------------------------------------------------
# Wired-up cfg objects: command, observation, reward, curriculum
# ---------------------------------------------------------------------------
from dataclasses import replace as _replace  # noqa: E402

from holosoma.config_types.command import CommandTermCfg  # noqa: E402
from holosoma.config_types.curriculum import CurriculumTermCfg  # noqa: E402
from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg, ObservationManagerCfg  # noqa: E402
from holosoma.config_types.reward import RewardTermCfg  # noqa: E402
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum  # noqa: E402
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward as _core_reward  # noqa: E402

from holosoma.config_values.wbt.g1.observation import critic_obs_shared_terms  # noqa: E402
from holosoma.utils.rotations import quat_error_magnitude  # noqa: E402

from wbt_training.config_values.command import g1_29dof_wbt_command as _core_force_command  # noqa: E402
from wbt_training.config_values.observation import _actor_obs_2pt_terms  # noqa: E402

_DEFAULT_HAND_FORCE_CFG = HandForceConfig()


# Command preset: motion_command (inherited) + hand_force_injector (new).
#
# CRITICAL: the injector must be registered in BOTH setup_terms (one-time buffer
# allocation + monkey-patch) AND reset_terms (per-episode pulse restart + force
# zeroing). CommandManager only dispatches a class-based term's lifecycle hook
# for the stages it is registered under (`_register_class_term` builds a
# per-instance `stages` set; `CommandManager.reset()` skips any term whose set
# lacks "reset" — manager.py:153-156). Registering only in setup_terms (the
# original bug) left HandForceInjector.reset() as DEAD CODE: the trapezoidal
# pulse phase and the F^ext buffer leaked across episode boundaries, so an env
# that reset mid-hold-plateau began its NEW episode with a held ~25 N already on
# the hand (IsaacLab persists the external wrench across substeps until
# overwritten — v2.3.0 articulation.py, is_global=False, buffer-based). That is
# the exact start-of-episode force shock HandForceInjector.reset() exists to
# prevent. step_terms is intentionally omitted (HandForceInjector.step() is a
# no-op — the pulse advances in the substep-0 physics hook, not in step()).
_hand_force_injector_term = CommandTermCfg(
    func="wbt_training.config_values.force_tracking:HandForceInjector",
    params={"hand_force_config": _DEFAULT_HAND_FORCE_CFG},
)
g1_29dof_wbt_force_command = _replace(
    _core_force_command,
    setup_terms={
        "motion_command": _core_force_command.setup_terms["motion_command"],
        "hand_force_injector": _hand_force_injector_term,
    },
    reset_terms={**_core_force_command.reset_terms, "hand_force_injector": _hand_force_injector_term},
)


# Observation preset: 2pt actor obs + F^cmd in actor, F^cmd + F^ext in critic.
# Mirrors gentle_humanoid's UniFP `command()` (actor) vs `force_track_priv()` (critic)
# asymmetry: actor only sees its own compliance target (`F^cmd`); critic gets the
# external disturbance (`F^ext`) as privileged info for value-function shaping.
# See `thirdparty/gentle_humanoid/.../motion_tracking.py:1626-1707`.
_actor_obs_2pt_force_terms = {
    "obs0_command_lower_body": _actor_obs_2pt_terms["obs0_command_lower_body"],
    "obs1_vr_2point_pos": _actor_obs_2pt_terms["obs1_vr_2point_pos"],
    "obs2_vr_2point_orn": _actor_obs_2pt_terms["obs2_vr_2point_orn"],
    "obs3_force_cmd_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0
    ),
    "obs4_motion_ref_ori_b": _actor_obs_2pt_terms["obs3_motion_ref_ori_b"],
    "obs5_projected_gravity": _actor_obs_2pt_terms["obs4_projected_gravity"],
    "obs6_base_ang_vel": _actor_obs_2pt_terms["obs5_base_ang_vel"],
    "obs7_dof_pos": _actor_obs_2pt_terms["obs6_dof_pos"],
    "obs8_dof_vel": _actor_obs_2pt_terms["obs7_dof_vel"],
    "obs9_actions": _actor_obs_2pt_terms["obs8_actions"],
}  # 141 + 6 = 147D actor (vr_2point_orn is the 12D 6D-rot rep, inherited by ref)

# Critic = TML critic + privileged force info (F^cmd + F^ext). Holosoma concats
# obs terms in DICT INSERTION ORDER (manager.py:139-142), so the two force_*
# terms land at the END of the critic vector (their position in this dict
# literal). Plain (un-prefixed) names are fine because the critic side has no
# on-disk-layout constraint (unlike the actor / TML format, which uses obsN_
# numeric prefixes so insertion order reproduces the fixed TML layout).
_critic_obs_2pt_force_terms = {
    **critic_obs_shared_terms,
    "force_cmd_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0
    ),
    "force_ext_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_ext_local", scale=0.04, noise=0.0
    ),
}
critic_obs_2pt_force = ObsGroupCfg(
    concatenate=True, enable_noise=False, history_length=10, terms=_critic_obs_2pt_force_terms
)

g1_29dof_wbt_observation_2pt_force = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_2pt_force_terms
        ),
        "critic_obs": critic_obs_2pt_force,
    }
)


# --- --F_on_hand_frame ablation: actor-only hand-frame F^cmd obs encoding ---
#
# Returns a COPY of `observation_cfg` in which ONLY the ACTOR's force_cmd_local
# obs term carries `params={"hand_frame": True}` — so `force_cmd_local` re-expresses
# the (still pelvis-sampled, pelvis-applied) command vector in each hand's frame
# for the actor only. The CRITIC's force_cmd_local term is left UNTOUCHED
# (pelvis-frame), giving strict actor-only scoping (the value function keeps a
# stable privileged frame; force_ext_local is never hand-framed). Everything else
# — term names, ordering, the 147D layout, the critic group, history, scale,
# noise — is preserved byte-for-byte, so checkpoint/inference loadability and all
# other presets are unaffected. F^cmd=0 ⇒ hand-frame 0, so the F=0⇒2pt collapse is
# intact. See force_cmd_local's docstring for the math and the rpy=0 justification.
#
# This helper operates on whatever observation cfg is live at CLI-parse time, so
# any unrelated CLI obs overrides are preserved (only the one actor term is
# flipped). The named preset below is the canonical pre-built variant for tests.
def with_force_cmd_hand_frame(observation_cfg: ObservationManagerCfg) -> ObservationManagerCfg:
    """Flip the ACTOR's force_cmd_local obs term to hand frame; critic untouched."""
    groups = dict(observation_cfg.groups)
    actor = groups.get("actor_obs")
    if actor is None:
        return observation_cfg
    new_terms: dict[str, ObsTermCfg] = {}
    changed = False
    for name, term in actor.terms.items():
        if term.func.endswith(":force_cmd_local"):
            new_terms[name] = _replace(term, params={**term.params, "hand_frame": True})
            changed = True
        else:
            new_terms[name] = term
    if not changed:
        return observation_cfg
    groups["actor_obs"] = _replace(actor, terms=new_terms)
    return _replace(observation_cfg, groups=groups)


# Canonical pre-built hand-frame variant (actor F^cmd in hand frame, critic pelvis).
g1_29dof_wbt_observation_2pt_force_handframe = with_force_cmd_hand_frame(g1_29dof_wbt_observation_2pt_force)


# Sensor-frame counterpart of with_force_cmd_hand_frame: flips ONLY the ACTOR's
# force_cmd_local term to params={"sensor_frame": True}, so force_cmd_local
# re-expresses the (still pelvis-sampled, wrist_yaw-applied) command in each hand's
# {side}_ft_sensor_link frame for the actor only. Critic untouched (pelvis-frame),
# force_ext_local never re-framed. Same non-invasion guarantees as the hand-frame
# helper: term names / ordering / layout / scale / noise preserved byte-for-byte,
# other presets unaffected, F^cmd=0 ⇒ 0. See force_cmd_local's sensor-frame docstring.
def with_force_cmd_sensor_frame(observation_cfg: ObservationManagerCfg) -> ObservationManagerCfg:
    """Flip the ACTOR's force_cmd_local obs term to sensor frame; critic untouched."""
    groups = dict(observation_cfg.groups)
    actor = groups.get("actor_obs")
    if actor is None:
        return observation_cfg
    new_terms: dict[str, ObsTermCfg] = {}
    changed = False
    for name, term in actor.terms.items():
        if term.func.endswith(":force_cmd_local"):
            new_terms[name] = _replace(term, params={**term.params, "sensor_frame": True})
            changed = True
        else:
            new_terms[name] = term
    if not changed:
        return observation_cfg
    groups["actor_obs"] = _replace(actor, terms=new_terms)
    return _replace(observation_cfg, groups=groups)


# Canonical pre-built sensor-frame variant (actor F^cmd in ft_sensor frame, critic pelvis).
g1_29dof_wbt_observation_2pt_force_sensorframe = with_force_cmd_sensor_frame(g1_29dof_wbt_observation_2pt_force)


# Reward preset: inherited dense tracking + force_track_pos_eq2 + feet_slip,
# with ONLY the baseline 14-link relative-POSITION term
# (`motion_relative_body_position_error_exp`) swapped for a no-wrist (12-body)
# variant, so the wrist_yaw position is owned solely by Eq.2
# (`ref + (F^cmd+F^ext)/K`), avoiding the tug-of-war on the rubber_hand rigid
# sub-chain.
#
# WRIST/HAND ORIENTATION (kept — this is the contact pose for manipulation):
# the baseline 14-link relative-ORIENTATION term
# (`motion_relative_body_orientation_error_exp`, incl. left/right_wrist_yaw_link)
# is INHERITED UNCHANGED. rubber_hand is rigidly fixed to wrist_yaw (constant
# offset, no relative rotation), so wrist_yaw orientation == hand orientation —
# the DOF a manipulation task must get right. It does NOT conflict with Eq.2:
# the FK offset cancels in the Eq.2 residual (cur and ref both rotate it by the
# LIVE wrist quat), so Eq.2 rewards wrist_yaw-ORIGIN translation (reach the arm
# out by F/K), not wrist rotation — orientation and Eq.2 are orthogonal.
# (Earlier this preset also excluded the wrist from the orientation term, on the
# mistaken premise that Eq.2 demands a wrist rotation; it does not, so the hand
# contact orientation was left unsupervised. Fixed: orientation now covers all
# 14 bodies, exactly as the 2pt baseline does.)
#
# `force_track_pos_pen` is intentionally NOT in this preset (removed per user
# request — keep the function defined above so it can be wired back with one
# line if reward-hacking shows up in W&B).
#
# joint_pos_tracking / joint_vel_tracking removed as redundant: holosoma's
# baseline already supervises 14-link world-frame positions and 32-body
# global lin/ang velocities, which determine joint angles/velocities up to
# the small wrist degree of freedom we now intentionally leave to Eq.2.
g1_29dof_wbt_2pt_force_reward = _replace(
    _core_reward,
    terms={
        # Only the relative-POSITION term is swapped for the no_wrist variant:
        # the wrist_yaw position is owned by Eq.2 (`ref + (F^cmd+F^ext)/K`), so we
        # drop it from the 14-link position reward to avoid two competing position
        # targets on the rubber_hand rigid sub-chain.
        #
        # The 14-link relative-ORIENTATION term is KEPT UNCHANGED (inherited from
        # _core_reward, full 14 bodies incl. wrist_yaw). rubber_hand is rigidly
        # fixed to wrist_yaw, so wrist_yaw orientation == hand orientation —
        # exactly the contact-pose DOF manipulation must get right. It does NOT
        # conflict with Eq.2: because the FK offset cancels (both cur and ref
        # rotate it by the LIVE wrist quat), Eq.2 rewards wrist_yaw-ORIGIN
        # translation (reach the arm out by F/K), not wrist rotation — so the
        # orientation term and Eq.2 are orthogonal. (Earlier this preset wrongly
        # excluded the wrist from orientation too, on the mistaken premise that
        # Eq.2 demands a wrist rotation; it does not.)
        **{k: v for k, v in _core_reward.terms.items() if k != "motion_relative_body_position_error_exp"},
        "motion_relative_body_position_error_exp_no_wrist": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:motion_relative_body_position_error_exp_no_wrist",
            params={"sigma": 0.3, "exclude_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link")},
            weight=1.0,
        ),
        # weight=6.0: bumped from GH v2 baseline 4.0. We removed
        # `motion_relative_body_position_error_exp` from the wrist via the
        # no_wrist variant (wrist position is now Eq.2's exclusive territory),
        # so we shift more of the position-tracking budget into Eq.2 to match.
        # If W&B shows force_track plateauing at ≤0.7 while no_wrist saturates,
        # try 8.0 next; if force saturates at 0.95+ but no_wrist drops <0.6,
        # fall back to 4.0.
        "force_track_pos_eq2": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:force_track_pos_eq2",
            weight=6.0,
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
        # Feet-slip penalty: GH `G1_unifp.yaml:135` `feet_slip: {weight: 2.0}`.
        # GH func returns negative magnitude × positive yaml weight = negative
        # final reward. Holosoma convention: func returns POSITIVE magnitude,
        # cfg supplies the negative weight. Same loss contribution. Sign rule:
        # exactly one of {func, weight} carries the minus, never both, never
        # neither. See `feet_slip` docstring.
        "feet_slip": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:feet_slip",
            weight=-2.0,
            params={"body_names": ("left_ankle_roll_link", "right_ankle_roll_link")},
        ),
    },
)


# ===========================================================================
# exp:g1-wbt-2pt-force-ext — F^ext ADDED BACK TO THE ACTOR (first-commit recipe).
#
# This is a deliberate REVERT of `g1_wbt_2pt_force` to its ORIGINAL first-commit
# design (648c78b, the version behind wandb run 09wvlz16), packaged as its own
# preset so `g1_wbt_2pt_force` (the shipped UniFP-asymmetric variant) and
# `g1_wbt_2pt_hybrid` are BOTH left byte-for-byte untouched. Two differences vs
# the current `g1_wbt_2pt_force` above:
#
#   1. OBS — F^ext is in the ACTOR obs (not critic-only). The actor sees BOTH
#      force_cmd_local (6D) AND force_ext_local (6D), so the actor obs is 153D
#      (141D 2pt baseline + 6D F^cmd + 6D F^ext) vs the 147D of `g1_wbt_2pt_force`.
#      This drops the UniFP `command()`/`force_track_priv()` actor/critic
#      asymmetry (commit b0039b6) — the actor no longer has to INFER F^ext from
#      proprioception; it observes the disturbance directly. The critic obs is
#      UNCHANGED (still F^cmd + F^ext, `critic_obs_2pt_force`), so the value
#      function is identical to `g1_wbt_2pt_force`'s.
#
#   2. REWARD — the "最最开始" (earliest) reward weights: the FULL 14-body
#      relative POSITION tracking is KEPT (no `no_wrist` swap → the whole body,
#      not just the non-wrist bodies, is tracked at weight 1.0), the 14-body
#      relative ORIENTATION and the global lin/ang-vel terms are inherited from
#      `_core_reward` unchanged, and force is added as UniFP position-overshoot:
#      force_track_pos_eq2 @ weight=4.0 (NOT the later 6.0 bump) + the
#      force_track_pos_pen safety penalty @ weight=-6.0 (removed from the current
#      2pt-force, restored here). NO feet_slip. This is exactly the first-commit
#      `g1_29dof_wbt_2pt_force_reward` (see git show 648c78b:…/force_tracking.py).
#
# Everything else matches `g1_wbt_2pt_force`: same HandForceInjector command
# (trapezoidal F^ext pulse, F^cmd compliance target, NEVER written to PhysX),
# same K=200 N/m / σ=0.3 m Eq.2 kernel, same ForceAlphaCurriculum (disabled by
# default → full force from step 0). Pure position-overshoot UniFP — NO F_meas
# (no simulated F/T sensor), NO Jᵀ servo, NO integrator. The preset in
# experiment.py additionally bakes in the yam URDF (g1/g1_29dof_yam.urdf).
#
# `--F_on_hand_frame` / `--F_on_sensor_frame` keep working: they re-encode the
# ACTOR's force_cmd_local term only (force_ext_local is never re-framed), so the
# actor's F^ext obs channel stays pelvis-frame under both flags.
# ===========================================================================
# Actor obs: 2pt baseline + F^cmd (obs3) + F^ext (obs4), sequential obs0..obs10
# (exactly the first-commit layout). Re-keys _actor_obs_2pt_force_terms to splice
# force_ext_local in right after force_cmd_local and renumber the trailing terms.
_actor_obs_2pt_force_ext_terms = {
    "obs0_command_lower_body": _actor_obs_2pt_terms["obs0_command_lower_body"],
    "obs1_vr_2point_pos": _actor_obs_2pt_terms["obs1_vr_2point_pos"],
    "obs2_vr_2point_orn": _actor_obs_2pt_terms["obs2_vr_2point_orn"],
    "obs3_force_cmd_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_cmd_local", scale=0.04, noise=0.0
    ),
    "obs4_force_ext_local": ObsTermCfg(
        func="wbt_training.config_values.force_tracking:force_ext_local", scale=0.04, noise=0.0
    ),
    "obs5_motion_ref_ori_b": _actor_obs_2pt_terms["obs3_motion_ref_ori_b"],
    "obs6_projected_gravity": _actor_obs_2pt_terms["obs4_projected_gravity"],
    "obs7_base_ang_vel": _actor_obs_2pt_terms["obs5_base_ang_vel"],
    "obs8_dof_pos": _actor_obs_2pt_terms["obs6_dof_pos"],
    "obs9_dof_vel": _actor_obs_2pt_terms["obs7_dof_vel"],
    "obs10_actions": _actor_obs_2pt_terms["obs8_actions"],
}  # 141 + 6 (F^cmd) + 6 (F^ext) = 153D actor

# Critic obs: UNCHANGED from g1_wbt_2pt_force (F^cmd + F^ext, pelvis frame). Reusing
# the same ObsGroupCfg keeps the value function identical to the asymmetric variant.
g1_29dof_wbt_observation_2pt_force_ext = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True, enable_noise=True, history_length=4, terms=_actor_obs_2pt_force_ext_terms
        ),
        "critic_obs": critic_obs_2pt_force,
    }
)


# Reward preset: first-commit recipe — inherited dense tracking (full 14-body
# position AND orientation, unchanged from _core_reward) + the two original
# force terms (Eq.2 @ 4.0 + pen @ -6.0). NO no_wrist swap, NO feet_slip.
g1_29dof_wbt_2pt_force_ext_reward = _replace(
    _core_reward,
    terms={
        **_core_reward.terms,
        "force_track_pos_eq2": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:force_track_pos_eq2",
            weight=4.0,
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
        "force_track_pos_pen": RewardTermCfg(
            func="wbt_training.config_values.force_tracking:force_track_pos_pen",
            weight=-6.0,
            params={"cfg": _DEFAULT_HAND_FORCE_CFG},
        ),
    },
)


# Curriculum preset: inherited average-episode-tracker + force_alpha ramp.
# Registered in all three role-dicts under the same key (CurriculumManager
# deduplicates by name → one instance with full lifecycle hooks).
_force_alpha_term = CurriculumTermCfg(
    func="wbt_training.config_values.force_tracking:ForceAlphaCurriculum",
    params={"hand_force_config": _DEFAULT_HAND_FORCE_CFG},
)

g1_29dof_wbt_force_curriculum = _replace(
    g1_29dof_wbt_curriculum,
    setup_terms={**g1_29dof_wbt_curriculum.setup_terms, "force_alpha": _force_alpha_term},
    reset_terms={**g1_29dof_wbt_curriculum.reset_terms, "force_alpha": _force_alpha_term},
    step_terms={**g1_29dof_wbt_curriculum.step_terms, "force_alpha": _force_alpha_term},
)


__all__ = [
    "HandForceConfig",
    "HandForceInjector",
    "ForceAlphaCurriculum",
    "force_cmd_local",
    "force_ext_local",
    "force_track_pos_eq2",
    "force_track_pos_eq2_global",
    "force_track_pos_pen",
    "motion_relative_body_position_error_exp_no_wrist",
    "motion_global_body_position_error_exp_no_wrist",
    "motion_relative_body_orientation_error_exp_no_wrist",
    "feet_slip",
    "g1_29dof_wbt_force_command",
    "g1_29dof_wbt_observation_2pt_force",
    "g1_29dof_wbt_observation_2pt_force_handframe",
    "g1_29dof_wbt_observation_2pt_force_sensorframe",
    "g1_29dof_wbt_observation_2pt_force_ext",
    "with_force_cmd_hand_frame",
    "with_force_cmd_sensor_frame",
    "critic_obs_2pt_force",
    "g1_29dof_wbt_2pt_force_reward",
    "g1_29dof_wbt_2pt_force_ext_reward",
    "g1_29dof_wbt_force_curriculum",
]
