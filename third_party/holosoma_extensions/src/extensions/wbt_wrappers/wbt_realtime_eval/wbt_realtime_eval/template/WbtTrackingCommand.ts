import type { CommandConfigEntry, CommandTermContext } from '@/core/command/types';
import { TrackingCommand } from '@/core/command/TrackingCommand';
import { loadNpz } from '@/core/scene/npz';
import * as THREE from 'three';
import { getPolicyExtra, setPolicyExtra } from '@/core/policy/extras';

/**
 * WBT extension to mjswan's built-in TrackingCommand.
 *
 * mjswan's ``TrackingCommand`` already gives us the joint/body trajectories
 * and a motion picker. The holosoma-*-wbt presets need a handful of
 * additional per-frame tables that are too expensive to compute in the
 * browser (FK, dof-vel reordering, root-frame anchoring). The Python build
 * step pre-bakes them INTO the same motion NPZ — see
 * ``bake_motions.bake_motion`` — and this command term re-parses the NPZ on
 * motion select to expose them through getters.
 *
 * Baked tables (always present after ``bake_motion_dir``)::
 *   vr_pos_local      (T, 6)   left/right wrist xyz, root-frame  [2pt only]
 *   vr_orn_local      (T, 12)  left/right wrist 6D rot rep        [2pt only]
 *   vr3_pos_local     (T, 9)   3-body xyz, root-frame            [3pt only]
 *   vr3_pos_tcp_local (T, 9)   as above, hands at the 0.242 TCP  [3pt TCP-unified]
 *   vr3_orn_local     (T, 12)  3-body xyzw quat                  [3pt only]
 *   vr3_orn6d_local   (T, 18)  3-body 6D rot rep                 [3pt 6D-orn only]
 *   vr5_pos_tcp_local (T, 15)  5-body xyz (3pt-TCP + both feet)  [5pt only]
 *   vr5_orn_local     (T, 20)  5-body xyzw quat                  [5pt only]
 *   lower_body_jp     (T, 12)  ref dof_pos[LOWER_BODY_INDICES]
 *   lower_body_jv     (T, 12)  ref dof_vel[LOWER_BODY_INDICES]
 *   joint_pos         (T, 29)  ref dof_pos (full DOF set)
 *   joint_vel         (T, 29)  ref dof_vel (full DOF set)
 *   ref_root_pos_w    (T, 3)   anchor (pelvis) world position
 *   ref_root_vel_w    (T, 3)   anchor (pelvis) world linear velocity
 *
 * Presets that don't need a table (e.g. holosoma-wbt does not use VR FK) get
 * empty getters that return zero-filled fixed-width arrays. That's safe
 * because the matching obs class for the preset is also disabled.
 */
export class WbtTrackingCommand extends TrackingCommand {
  private readonly bakedCache: Map<
    string,
    {
      vr2PosLocal: Float32Array | null;
      vr2OrnLocal: Float32Array | null;
      vr3PosLocal: Float32Array | null;
      vr3PosTcpLocal: Float32Array | null;
      vr3OrnLocal: Float32Array | null;
      vr3Orn6dLocal: Float32Array | null;
      vr5PosTcpLocal: Float32Array | null;
      vr5OrnLocal: Float32Array | null;
      lowerBodyJp: Float32Array;
      lowerBodyJv: Float32Array;
      refDofPos: Float32Array;
      refDofVel: Float32Array;
      refRootPosW: Float32Array;
      refRootVelW: Float32Array;
      forceHandRefLocal: Float32Array | null;
      T: number;
    }
  > = new Map();
  private activeBaked: {
    vr2PosLocal: Float32Array | null;
    vr2OrnLocal: Float32Array | null;
    vr3PosLocal: Float32Array | null;
    vr3PosTcpLocal: Float32Array | null;
    vr3OrnLocal: Float32Array | null;
    vr3Orn6dLocal: Float32Array | null;
    vr5PosTcpLocal: Float32Array | null;
    vr5OrnLocal: Float32Array | null;
    lowerBodyJp: Float32Array;
    lowerBodyJv: Float32Array;
    refDofPos: Float32Array;
    refDofVel: Float32Array;
    refRootPosW: Float32Array;
    refRootVelW: Float32Array;
    forceHandRefLocal: Float32Array | null;
    T: number;
  } | null = null;
  private readonly motionPaths: Map<string, string>;
  // Mirror of the parent's context — TrackingCommand keeps its own copy
  // private, so we cache one here for the force-viz code below to read
  // mjModel / mjData / scene roots without relying on parent internals.
  private readonly wbtContext: CommandTermContext;

  // ── MOTION-COMPLETION REFERENCE GHOST (5pt / dense LIVE teleop) ───────────────────────
  // A second translucent body, LIGHT PURPLE, holding the whole 29-DoF completion the live
  // command was sliced out of. This is the GROUND TRUTH for VIVE teleop: the operator moves
  // three trackers, Motion_Completion turns that into a whole body, and the purple mesh is
  // that body. The question it answers, live and continuously, is "can the low-level policy
  // keep up with it?" — which is unanswerable against five commanded spheres, and which
  // also separates the two failures that look identical in the sim: the completion never
  // asked for the motion, versus it asked and the controller did not follow.
  //
  // Mode-independent by construction: the wire carries the same four ghost keys in 5pt and
  // dense, so there is ONE code path here and no way for a mode to be silently unsupported.
  // For a RECORDED take you do not need any of this — convert it with
  // UMI_collection/run_mc_to_holosoma.sh and bake it as an ordinary reference motion.
  //
  // WHY A SECOND GHOST INSTEAD OF DRIVING THE PARENT'S. The parent's ghost is built from
  // `refJointPos/refRootPos/refRootQuat[refIdx]`, and those same arrays ARE the policy's
  // reference — `getCommand()` and every motion_ref_* obs read them. Overwriting them to
  // move the ghost would silently change the policy's input, turning a viewer into an
  // intervention. So this one is fully additive: its own MjData, its own meshes, its own
  // qpos, and not one write into anything the parent or the obs terms read.
  //
  // Keeping BOTH visible is deliberate and is most of the diagnostic value: the PALE GREEN
  // ghost is the reference the policy is actually given (frozen — `idle_default` in these
  // presets, which is why `motion_ref_vel_b` is zeroed), and the PURPLE one is what the
  // completion wants. Both hide together with the existing reference toggle.
  //
  // ⚠️ IT HIDES WHEN THE STREAM GOES STALE, and that is not cosmetic. A frozen mesh that an
  // operator is actively trying to track is worse than no mesh: they would read their own
  // tracking error off a target that stopped updating. Same 0.5 s freshness budget the rest
  // of the stack uses.
  private static readonly MC_GHOST_RGB: readonly [number, number, number] = [0.72, 0.60, 0.96];
  private static readonly MC_GHOST_STALE_MS = 500;
  private mcGhostRoot: THREE.Group | null = null;
  private readonly mcGhostBodies: Map<number, THREE.Group> = new Map();
  private mcGhostData: import('mujoco').MjData | null = null;
  // Render-model joint name -> its qpos address, resolved ONCE. Only names the render model
  // actually has; a completion joint that is not in the model is dropped here rather than
  // written to a guessed address.
  private mcGhostJointAdr: Array<{ name: string; adr: number }> = [];
  private mcGhostFreeAdr = -1;
  private mcGhostInitialized = false;
  private mcGhostFailed = false;
  // Last (wire seq, reference frame index) actually solved. The render loop runs faster than
  // the 50 Hz wire, so without this the ghost would re-run mj_forward several times per new
  // frame for an identical pose. refIdx is in the key because the ANCHOR comes from it.
  private mcGhostLastSeq = -1;
  private mcGhostLastRefIdx = -1;
  private mcGhostStale = false;

  // Force-tracking debug visualization: per-hand RGB axis triad + TWO force
  // arrows (purple F^cmd intent, orange F^ext real disturbance) anchored at
  // `wrist_yaw_pos_w + R(wrist_yaw_quat) * offset_b`. Populated when
  // `config.force_viz_bodies` is present (force preset only); otherwise these
  // stay null and the debug-visual hook is a no-op.
  //
  // Colors are FIXED (no magnitude lerp) and length matches the benchmark
  // force-arrow videos (eval_utils): length = max(|F|*0.018, 0.08) m, uncapped.
  private readonly forceVizConfig:
    | Array<{
        bodyName: string;
        offsetB: [number, number, number];
        cmdSliderKeys: [string, string, string]; // purple F^cmd
        extSliderKeys: [string, string, string]; // orange F^ext
        measSliderKeys: [string, string, string] | null; // cyan F^meas (hybrid only)
        arrowColorCmd: [number, number, number]; // RGB 0-1, purple
        arrowColorExt: [number, number, number]; // RGB 0-1, orange
        arrowColorMeas: [number, number, number]; // RGB 0-1, cyan
        measFrame: 'pelvis' | 'world'; // F^meas native frame (world for hybrid)
      }>
    | null = null;
  private forceVizInitialized = false;
  private forceVizGroups: Array<{
    root: THREE.Group;
    triad: THREE.Group; // contains 3 line segments (R/G/B)
    cmdArrow: WbtForceArrow; // purple — F^cmd intent
    extArrow: WbtForceArrow; // orange — F^ext real disturbance
    measArrow: WbtForceArrow | null; // cyan — F^meas (hybrid only; null otherwise)
    bodyId: number;
  }> = [];

  // Hybrid spring-wall + servo-phase viz (exp:g1-wbt-2pt-hybrid). Populated from
  // config.hybrid_viz; null (and every hybrid hook a no-op) for non-hybrid presets.
  private readonly hybridVizConfig:
    | {
        servoEnabled: boolean;
        // C1 unified (exp:g1-wbt-3pt-hybrid[-global-vel]-unified[-anneal]): the Jᵀ
        // servo is a REFERENCE GENERATOR only — I/Δx still feed the actor obs, but no
        // Δq is injected and the policy is the sole executor. The UI must say so: no
        // Layer-B checkbox (the runtime ignores the toggle), and the badge reads
        // "UNIFIED POLICY" rather than "FORCE SERVO / POSITION TRACKING (servo off)".
        servoObserveOnly: boolean;
        tcpOffsetX: number;
        maxPenetrationM: number;
        planeDist: number;
        hands: Array<{ bodyName: string; cmdSliderKeys: [string, string, string] }>;
      }
    | null = null;
  private hybridWallInitialized = false;
  private hybridWallGroups: Array<{ mesh: THREE.Mesh; mat: THREE.MeshBasicMaterial; bodyId: number }> = [];
  // Per-hand wall seed (world frame), mirroring the runtime: normal = −unit(F_cmd_world),
  // point = TCP − plane_dist·normal, captured when F^cmd first goes non-zero.
  // `seedTcpW` is the seed-time TCP — the pivot the LIVE-normal mode rotates the plane
  // about (see the runtime's matching field; rotating about the live TCP would pin depth
  // to 0). Unused in the default episode-fixed mode.
  private hybridWallSeed: Array<{
    seeded: boolean;
    normalW: [number, number, number];
    pointW: [number, number, number];
    seedTcpW: [number, number, number];
  }> = [];
  private hybridBadgeEl: HTMLDivElement | null = null;
  // Live Hybrid control panel (browser-only toggles + legend, built once). The
  // two checkboxes write wbt_hybrid_wall_on / wbt_hybrid_servo_on to the extras
  // bus, which the runtime reads each substep to gate the causal spring wall
  // (Layer A) and the Jᵀ servo (Layer B). NEVER a policy obs — pure control/UI.
  private hybridControlsEl: HTMLDivElement | null = null;
  private hybridWallOn = true;
  private hybridServoOn = false; // seeded from the baked servoEnabled at build time
  // Live-normal wall (browser-only diagnostic, DEFAULT OFF). When on, both the runtime
  // physics wall and this viz plane re-derive their normal from the CURRENT F^cmd every
  // frame, so sweeping Fx→Fy in one episode visibly turns the wall. Off = the
  // training-faithful episode-fixed direction (hybrid.py holds it for the whole episode).
  private hybridWallLive = false;
  // F^meas inertial term (browser-only, DEFAULT ON = training-faithful include_inertial).
  // The runtime's simulated F/T reading is F_ext + m_distal·a_wrist with a_wrist DOUBLE
  // finite-differenced from the TCP position at the substep dt, so sub-millimetre jitter
  // reads as tens of N of flicker in free space (nothing random is injected — the demo's
  // sensor_noise_std is 0 and the runtime never applies it). OFF = F^meas is the wall
  // reaction + F^ext only: a still arrow/readout for screenshots. One F^meas: the policy's
  // force_meas_local obs, contact_phase and e follow the switch (runtime computeHybridFMeas).
  private hybridFmeasInertial = true;

  // Control-point marker: an orange sphere at the EXACT point the VR tracker
  // steers — the policy's vr_2point contact point
  // `wrist_yaw_pos_w + R(wrist_yaw_quat) * offset_b`, offset_b = the 0.18 m VR
  // lever (DIFFERENT from the force/rubber-hand offset). Populated when
  // `config.vr_control_point_bodies` is present (opt-in --show-control-point);
  // otherwise stays null and its hook is a no-op. INDEPENDENT of force_viz so it
  // works on any 2pt preset (force or not).
  private readonly controlPointConfig:
    | Array<{
        bodyName: string;
        offsetB: [number, number, number];
        color: [number, number, number];
        radius: number;
      }>
    | null = null;
  private controlPointInitialized = false;
  private controlPointGroups: Array<{ mesh: THREE.Mesh; bodyId: number; offsetB: [number, number, number] }> = [];

  // LIVE VIVE 3pt target triads: one RGB axis triad + a coloured ball at each of
  // the three COMMANDED target poses the policy is being fed (left TCP, right
  // TCP, head), read off the policy-extras bus that WbtVr3PointPosTcpLive /
  // WbtVr3PointOrnLive publish. The wire pose is PELVIS-frame, so each frame we
  // compose it with the live pelvis world pose before drawing — which is exactly
  // what makes this a correspondence check: the triad shows where the policy was
  // TOLD to put the point, next to the white TCP sphere showing where the robot
  // actually got it. Populated from `config.vr_live_target_triads`; null (and
  // every hook a no-op) for every other preset.
  private readonly vrLiveTriadConfig:
    | {
        axisLen: number;
        ballRadius: number;
        okKey: string;
        prefix: string;
        points: Array<{ tag: string; label: string; color: [number, number, number] }>;
      }
    | null = null;
  private vrLiveTriadInitialized = false;
  private vrLiveTriadGroups: Array<{ root: THREE.Group; ball: THREE.Mesh; mats: THREE.MeshBasicMaterial[] }> = [];
  // vr_3point TRACKING-point markers (exp:g1-wbt-3pt* presets): small spheres at
  // the THREE points the 3pt policy tracks — left hand / right hand / head — drawn
  // on BOTH the LIVE robot (solid) and the REFERENCE ghost (translucent, same hue),
  // so a point's commanded-target vs actual reads as light-vs-solid. Anchor math is
  // identical to the control-point marker: `body_pos_w + R(body_quat) * offset_b`,
  // offset_b = the vr_3point offsets (0.18 hands, 0.35 head). Populated when
  // `config.vr_tracking_point_markers` is present (emitted for vr_3point presets
  // only); otherwise stays null and its hook is a no-op. INDEPENDENT of force_viz
  // and the control-point marker, so it coexists with the hybrid TCP white ball.
  private readonly trackingMarkerConfig:
    | Array<{
        bodyName: string;
        offsetB: [number, number, number];
        realColor: [number, number, number];
        refColor: [number, number, number];
        radius: number;
      }>
    | null = null;
  private trackingMarkerInitialized = false;
  // Real spheres ride the live robot (mjData); reference spheres ride the ghost
  // (ghostData). Each entry keeps its body id (same body in both models — the ghost
  // shares the render mjModel) and offset for the per-frame reposition.
  private trackingMarkerReal: Array<{ mesh: THREE.Mesh; bodyId: number; offsetB: [number, number, number] }> = [];
  private trackingMarkerRef: Array<{ mesh: THREE.Mesh; bodyId: number; offsetB: [number, number, number] }> = [];

  constructor(termName: string, config: CommandConfigEntry, context: CommandTermContext) {
    super(termName, config, context);
    this.wbtContext = context;
    this.motionPaths = new Map();
    const motions = Array.isArray(config.motions) ? config.motions : [];
    for (const m of motions) {
      const entry = m as Record<string, unknown>;
      const name = entry.name;
      const path = entry.path;
      if (typeof name === 'string' && typeof path === 'string') {
        this.motionPaths.set(name, path);
      }
    }
    // Optional: force-tracking debug viz. Shape:
    //   force_viz_bodies: [
    //     {body_name:str, offset_b:[x,y,z],
    //      slider_keys:[fx,fy,fz],        // F^cmd (purple)
    //      ext_slider_keys:[fx,fy,fz],    // F^ext (orange)
    //      arrow_color_cmd?:[r,g,b], arrow_color_ext?:[r,g,b]},
    //     ...
    //   ]
    const fvbRaw = (config as Record<string, unknown>).force_viz_bodies;
    if (Array.isArray(fvbRaw) && fvbRaw.length > 0) {
      const out: NonNullable<typeof this.forceVizConfig> = [];
      for (const e of fvbRaw) {
        if (
          e &&
          typeof e === 'object' &&
          typeof (e as { body_name?: unknown }).body_name === 'string' &&
          Array.isArray((e as { offset_b?: unknown }).offset_b) &&
          Array.isArray((e as { slider_keys?: unknown }).slider_keys)
        ) {
          const ent = e as Record<string, unknown>;
          const off = ent.offset_b as unknown[];
          const cmdKeys = ent.slider_keys as unknown[];
          // ext keys fall back to cmd keys only defensively; main.py always
          // supplies distinct wbt_force_ext_local__* keys for the force preset.
          const extKeysRaw = Array.isArray(ent.ext_slider_keys)
            ? (ent.ext_slider_keys as unknown[])
            : cmdKeys;
          // Optional cyan F^meas arrow (hybrid preset only). Its native frame is
          // WORLD (the causal wall normal is world-fixed), unlike F^cmd/F^ext
          // which are pelvis-frame sliders. measFrame drives the compose in
          // updateForceArrow (pelvis → root-quat compose; world → no compose).
          const measKeysRaw = Array.isArray(ent.meas_slider_keys) ? (ent.meas_slider_keys as unknown[]) : null;
          const measFrame = (ent.meas_frame === 'world' ? 'world' : 'pelvis') as 'pelvis' | 'world';
          out.push({
            bodyName: ent.body_name as string,
            offsetB: [Number(off[0]) || 0, Number(off[1]) || 0, Number(off[2]) || 0],
            cmdSliderKeys: [String(cmdKeys[0] ?? ''), String(cmdKeys[1] ?? ''), String(cmdKeys[2] ?? '')],
            extSliderKeys: [String(extKeysRaw[0] ?? ''), String(extKeysRaw[1] ?? ''), String(extKeysRaw[2] ?? '')],
            measSliderKeys: measKeysRaw
              ? [String(measKeysRaw[0] ?? ''), String(measKeysRaw[1] ?? ''), String(measKeysRaw[2] ?? '')]
              : null,
            arrowColorCmd: this.parseColor(ent.arrow_color_cmd, [0.62, 0.2, 0.92]), // purple
            arrowColorExt: this.parseColor(ent.arrow_color_ext, [1.0, 0.45, 0.0]), // orange
            arrowColorMeas: this.parseColor(ent.arrow_color_meas, [0.0, 0.75, 1.0]), // cyan
            measFrame,
          });
        }
      }
      if (out.length > 0) this.forceVizConfig = out;
    }

    // Optional: hybrid spring-wall + servo-phase viz (exp:g1-wbt-2pt-hybrid).
    //   hybrid_viz: { servo_enabled:bool, tcp_offset_x:number, max_penetration_m:number,
    //                 hands:[{body_name:str, cmd_slider_keys:[fx,fy,fz]}, ...] }
    const hvRaw = (config as Record<string, unknown>).hybrid_viz;
    if (hvRaw && typeof hvRaw === 'object') {
      const hv = hvRaw as Record<string, unknown>;
      const handsRaw = hv.hands;
      if (Array.isArray(handsRaw) && handsRaw.length > 0) {
        const hands: NonNullable<typeof this.hybridVizConfig>['hands'] = [];
        for (const h of handsRaw) {
          if (h && typeof h === 'object') {
            const he = h as Record<string, unknown>;
            const ck = Array.isArray(he.cmd_slider_keys) ? (he.cmd_slider_keys as unknown[]) : [];
            if (typeof he.body_name === 'string') {
              hands.push({
                bodyName: he.body_name,
                cmdSliderKeys: [String(ck[0] ?? ''), String(ck[1] ?? ''), String(ck[2] ?? '')],
              });
            }
          }
        }
        if (hands.length > 0) {
          this.hybridVizConfig = {
            servoEnabled: hv.servo_enabled === true,
            servoObserveOnly: hv.servo_observe_only === true,
            tcpOffsetX: Number(hv.tcp_offset_x) || 0.242,
            maxPenetrationM: Number(hv.max_penetration_m) || 0.08,
            planeDist: Number(hv.plane_dist) || 0.05,
            hands,
          };
          this.hybridWallSeed = hands.map(() => ({
            seeded: false,
            normalW: [0, 0, 0] as [number, number, number],
            pointW: [0, 0, 0] as [number, number, number],
            seedTcpW: [0, 0, 0] as [number, number, number],
          }));
          // Live toggle state: wall ON, servo = the checkpoint's baked servo_enabled
          // (ON for 31wei7by, OFF for the Layer-A run 6y8314vq). These instance
          // fields drive the checkboxes (ensureHybridControls) + the badge.
          this.hybridWallOn = true;
          this.hybridServoOn = this.hybridVizConfig.servoEnabled;
          // NOTE on frame 0: this constructor runs during commandManager.initialize,
          // BEFORE the runtime's resetPolicyExtras() wipes the extras bus, so these
          // two setPolicyExtra calls are wiped and do NOT survive to the first
          // substep. Frame-0 correctness is instead guaranteed by the runtime's
          // getPolicyExtra FALLBACKS being deliberately matched to these same values
          // (wall fallback 1.0; servo fallback this.hybridConfig.servoEnabled, parsed
          // from the identical servo_enabled config field). The panel re-publishes
          // the authoritative values on its first update() (ensureHybridControls) and
          // on every checkbox change. Kept as a harmless belt-and-suspenders seed —
          // if the runtime fallbacks ever diverge, the panel's update() reconciles.
          setPolicyExtra('wbt_hybrid_wall_on', this.hybridWallOn ? 1.0 : 0.0);
          setPolicyExtra('wbt_hybrid_servo_on', this.hybridServoOn ? 1.0 : 0.0);
          // Same frame-0 story: the runtime's getPolicyExtra fallback for this key is
          // 0.0, matched to hybridWallLive=false (episode-fixed, training-faithful).
          setPolicyExtra('wbt_hybrid_wall_live', this.hybridWallLive ? 1.0 : 0.0);
          // Runtime fallback 1.0, matched to hybridFmeasInertial=true (include_inertial).
          setPolicyExtra('wbt_hybrid_fmeas_inertial', this.hybridFmeasInertial ? 1.0 : 0.0);
        }
      }
    }

    // Optional: control-point marker. Shape:
    //   vr_control_point_bodies: [
    //     {body_name:str, offset_b:[x,y,z], color?:[r,g,b], radius?:number}, ...
    //   ]
    const cpRaw = (config as Record<string, unknown>).vr_control_point_bodies;
    if (Array.isArray(cpRaw) && cpRaw.length > 0) {
      const out: NonNullable<typeof this.controlPointConfig> = [];
      for (const e of cpRaw) {
        if (
          e &&
          typeof e === 'object' &&
          typeof (e as { body_name?: unknown }).body_name === 'string' &&
          Array.isArray((e as { offset_b?: unknown }).offset_b)
        ) {
          const ent = e as Record<string, unknown>;
          const off = ent.offset_b as unknown[];
          const r = Number((ent as { radius?: unknown }).radius);
          out.push({
            bodyName: ent.body_name as string,
            offsetB: [Number(off[0]) || 0, Number(off[1]) || 0, Number(off[2]) || 0],
            color: this.parseColor(ent.color, [1.0, 0.55, 0.0]), // orange
            radius: Number.isFinite(r) && r > 0 ? r : 0.025,
          });
        }
      }
      if (out.length > 0) this.controlPointConfig = out;
    }

    // Optional: live 3pt target triads. Shape:
    //   vr_live_target_triads: {
    //     extras_prefix:str, ok_key:str, axis_len?:number, ball_radius?:number,
    //     points: [{tag:'l'|'r'|'h', label:str, color:[r,g,b]}, ...]
    //   }
    const ltRaw = (config as Record<string, unknown>).vr_live_target_triads;
    if (ltRaw && typeof ltRaw === 'object') {
      const lt = ltRaw as Record<string, unknown>;
      const ptsRaw = lt.points;
      if (Array.isArray(ptsRaw) && ptsRaw.length > 0) {
        const points: NonNullable<typeof this.vrLiveTriadConfig>['points'] = [];
        for (const p of ptsRaw) {
          if (p && typeof p === 'object' && typeof (p as { tag?: unknown }).tag === 'string') {
            const pe = p as Record<string, unknown>;
            points.push({
              tag: pe.tag as string,
              label: typeof pe.label === 'string' ? pe.label : (pe.tag as string),
              color: this.parseColor(pe.color, [1.0, 1.0, 1.0]),
            });
          }
        }
        const axisLen = Number(lt.axis_len);
        const ballRadius = Number(lt.ball_radius);
        if (points.length > 0) {
          this.vrLiveTriadConfig = {
            axisLen: Number.isFinite(axisLen) && axisLen > 0 ? axisLen : 0.12,
            ballRadius: Number.isFinite(ballRadius) && ballRadius > 0 ? ballRadius : 0.018,
            okKey: typeof lt.ok_key === 'string' ? lt.ok_key : 'wbt_vr3_live__ok',
            prefix: typeof lt.extras_prefix === 'string' ? lt.extras_prefix : 'wbt_vr3_tgt',
            points,
          };
        }
      }
    }

    // Optional: vr_3point tracking-point markers. Shape:
    //   vr_tracking_point_markers: [
    //     {body_name:str, offset_b:[x,y,z], real_color?:[r,g,b], ref_color?:[r,g,b],
    //      radius?:number}, ...
    //   ]
    const tmRaw = (config as Record<string, unknown>).vr_tracking_point_markers;
    if (Array.isArray(tmRaw) && tmRaw.length > 0) {
      const out: NonNullable<typeof this.trackingMarkerConfig> = [];
      for (const e of tmRaw) {
        if (
          e &&
          typeof e === 'object' &&
          typeof (e as { body_name?: unknown }).body_name === 'string' &&
          Array.isArray((e as { offset_b?: unknown }).offset_b)
        ) {
          const ent = e as Record<string, unknown>;
          const off = ent.offset_b as unknown[];
          const r = Number((ent as { radius?: unknown }).radius);
          const real = this.parseColor(ent.real_color, [0.0, 0.85, 0.9]); // cyan default
          out.push({
            bodyName: ent.body_name as string,
            offsetB: [Number(off[0]) || 0, Number(off[1]) || 0, Number(off[2]) || 0],
            realColor: real,
            refColor: this.parseColor(ent.ref_color, real), // default: same hue as real
            radius: Number.isFinite(r) && r > 0 ? r : 0.03,
          });
        }
      }
      if (out.length > 0) this.trackingMarkerConfig = out;
    }
  }

  private parseColor(raw: unknown, fallback: [number, number, number]): [number, number, number] {
    if (Array.isArray(raw) && raw.length === 3) {
      return [Number(raw[0]) || 0, Number(raw[1]) || 0, Number(raw[2]) || 0];
    }
    return fallback;
  }

  // Remove the hybrid DOM overlays (controls panel + servo badge) on dispose, so a
  // live preset switch AWAY from a hybrid preset doesn't leave a stale 'Hybrid
  // controls' panel / phase badge floating over a non-hybrid scene. The parent
  // TrackingCommand.dispose only cleans up its ghost meshes. Guarded by id (the
  // element may have been created by this or a prior instance). No-op for a
  // non-hybrid term (the overlays were never mounted). THREE meshes (wall planes,
  // arrows, control-point markers) live under the scene root the runtime tears
  // down on scene reload, so only the DOM overlays need explicit removal here.
  override dispose(): void {
    if (typeof document !== 'undefined') {
      for (const id of ['wbt-hybrid-controls', 'wbt-hybrid-servo-badge']) {
        document.getElementById(id)?.remove();
      }
    }
    this.hybridControlsEl = null;
    this.hybridBadgeEl = null;
    // Explicitly remove the vr_3point tracking-point spheres (real + reference) from
    // the scene root and free their GPU resources. The runtime tears down the scene
    // root on a full reload, but on a LIVE preset switch these THREE meshes would
    // otherwise linger, leaving stale markers over the next policy. Idempotent:
    // clears the arrays so a double-dispose is a no-op.
    for (const grp of [...this.trackingMarkerReal, ...this.trackingMarkerRef]) {
      grp.mesh.parent?.remove(grp.mesh);
      grp.mesh.geometry.dispose();
      (grp.mesh.material as THREE.Material).dispose();
    }
    this.trackingMarkerReal = [];
    this.trackingMarkerRef = [];
    this.trackingMarkerInitialized = false;
    // The MC completion ghost: same reason as the markers above — on a LIVE preset switch
    // the scene root is not torn down, so a second translucent G1 would linger over the
    // next policy. Its MjData is a WASM allocation and must be freed explicitly, exactly as
    // the parent frees its own ghost's. Idempotent (cleared, so a double-dispose is a no-op).
    if (this.mcGhostRoot) {
      this.mcGhostRoot.parent?.remove(this.mcGhostRoot);
      this.mcGhostRoot.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            for (const m of obj.material) m.dispose?.();
          } else {
            obj.material?.dispose?.();
          }
        }
      });
    }
    this.mcGhostRoot = null;
    this.mcGhostBodies.clear();
    this.mcGhostData?.delete?.();
    this.mcGhostData = null;
    this.mcGhostJointAdr = [];
    this.mcGhostInitialized = false;
    super.dispose();
  }

  override async setSelectedMotion(name: string | null): Promise<boolean> {
    const ok = await super.setSelectedMotion(name);
    // Re-seed the hybrid spring wall on every motion switch / reset: the runtime
    // re-seeds its wall in resetHybridState, so clear our display seed too or the
    // viz wall would stay pinned to the pre-reset plane while the physics wall
    // moves to the new TCP. Cheap + safe on non-hybrid presets (empty array).
    for (const s of this.hybridWallSeed) {
      s.seeded = false;
      s.seedTcpW = [0, 0, 0];
    }
    if (!ok || name === null) {
      this.activeBaked = null;
      return ok;
    }
    let entry = this.bakedCache.get(name);
    if (!entry) {
      const path = this.motionPaths.get(name);
      if (!path) {
        throw new Error(`WbtTrackingCommand: no path registered for motion '${name}'.`);
      }
      // The parent TrackingCommand already fetched this URL; the browser HTTP
      // cache makes this a re-parse from cache, not a re-fetch.
      const npz = await loadNpz(path);

      // Two acceptance paths:
      //   1. Full bake — all 6 of {lower_body_jp/jv, joint_pos, joint_vel,
      //      ref_root_pos_w/vel_w} are present. Used for real motions baked
      //      via bake_motion_dir().
      //   2. Bake-less synthetic motion (e.g. mjswan's
      //      add_synthetic_default_pose_motion). Some baked keys are absent —
      //      we zero-fill them so the obs classes return zeros (the policy
      //      effectively sees "no motion command, no force tracking" on
      //      this clip, which is exactly the intent for a static-default
      //      placeholder).
      const required = [
        'lower_body_jp',
        'lower_body_jv',
        'joint_pos',
        'joint_vel',
        'ref_root_pos_w',
        'ref_root_vel_w',
      ];
      const missing = required.filter((k) => !npz[k]);

      const jp = npz['joint_pos'];
      const jv = npz['joint_vel'];

      let T: number;
      let lowerBodyJp: Float32Array;
      let lowerBodyJv: Float32Array;
      let refDofPos: Float32Array;
      let refDofVel: Float32Array;
      let refRootPosW: Float32Array;
      let refRootVelW: Float32Array;

      if (missing.length === 0 && jp && jv) {
        // ── Full-bake path: validate widths + shape consistency. ─────────
        const lJp = npz['lower_body_jp']!;
        const lJv = npz['lower_body_jv']!;
        const rPosW = npz['ref_root_pos_w']!;
        const rVelW = npz['ref_root_vel_w']!;
        if (lJp.shape[1] !== 12) throw new Error(`lower_body_jp width ${lJp.shape[1]} != 12`);
        if (lJv.shape[1] !== 12) throw new Error(`lower_body_jv width ${lJv.shape[1]} != 12`);
        if (jp.shape[1] !== 29) throw new Error(`joint_pos width ${jp.shape[1]} != 29`);
        if (jv.shape[1] !== 29) throw new Error(`joint_vel width ${jv.shape[1]} != 29`);
        if (rPosW.shape[1] !== 3) throw new Error(`ref_root_pos_w width ${rPosW.shape[1]} != 3`);
        if (rVelW.shape[1] !== 3) throw new Error(`ref_root_vel_w width ${rVelW.shape[1]} != 3`);
        T = lJp.shape[0];
        const widths: Array<[string, number]> = [
          ['lower_body_jv', lJv.shape[0]],
          ['joint_pos', jp.shape[0]],
          ['joint_vel', jv.shape[0]],
          ['ref_root_pos_w', rPosW.shape[0]],
          ['ref_root_vel_w', rVelW.shape[0]],
        ];
        for (const [k, t] of widths) {
          if (t !== T) throw new Error(`merged NPZ ${path} key '${k}' frame count ${t} != ${T}`);
        }
        lowerBodyJp = lJp.data;
        lowerBodyJv = lJv.data;
        refDofPos = jp.data;
        refDofVel = jv.data;
        refRootPosW = rPosW.data;
        refRootVelW = rVelW.data;
      } else {
        // ── Synthetic / bake-less path: zero-fill all derived tables. ────
        // We still need a frame count `T` for the slice() helpers. Fall
        // back to joint_pos.shape[0] when present (synthetic motions ship
        // it) or to a single frame otherwise.
        if (jp && jp.shape[0] >= 1) {
          T = jp.shape[0];
        } else {
          T = 1;
        }
        console.warn(
          `[WbtTrackingCommand] motion '${name}' missing baked keys [${missing.join(', ')}] — zero-filling. ` +
            'Force/VR/lower-body obs terms will read zeros for this clip.',
        );
        lowerBodyJp = new Float32Array(T * 12);
        lowerBodyJv = new Float32Array(T * 12);
        // joint_pos/vel ARE used by WbtCommand (58D ref pose+vel) for the
        // holosoma-wbt preset. Two shapes accepted:
        //   (T, 29) — synthetic default-pose NPZ from
        //              add_synthetic_default_pose_motion (DOF-only, matches
        //              mjswan parent TrackingCommand expectation).
        //   (T, >=36) — bake-source layout [root_xyz(3), root_quat(4),
        //              dof(29)]. Strip the 7-prefix to recover the 29 DOFs.
        if (jp && jp.shape[1] === 29) {
          refDofPos = jp.data;
        } else if (jp && jp.shape[1] >= 7 + 29) {
          const flat = jp.data;
          const out = new Float32Array(T * 29);
          for (let t = 0; t < T; t++) {
            for (let d = 0; d < 29; d++) {
              out[t * 29 + d] = flat[t * jp.shape[1] + 7 + d] ?? 0.0;
            }
          }
          refDofPos = out;
        } else {
          refDofPos = new Float32Array(T * 29);
        }
        refDofVel = new Float32Array(T * 29);
        refRootPosW = new Float32Array(T * 3);
        refRootVelW = new Float32Array(T * 3);
      }

      // Optional 2pt + 3pt tables — same logic for both paths.
      const vr2Pos = npz['vr_pos_local'] ?? null;
      const vr2Orn = npz['vr_orn_local'] ?? null;
      const vr3Pos = npz['vr3_pos_local'] ?? null;
      // TCP-unified 3pt hand point (0.242 m force TCP instead of the 0.18 m VR lever) —
      // exp:g1-wbt-3pt-hybrid-unified[-anneal]. Absent on bakes older than 2026-08-03.
      const vr3PosTcp = npz['vr3_pos_tcp_local'] ?? null;
      const vr3Orn = npz['vr3_orn_local'] ?? null;
      const vr3Orn6d = npz['vr3_orn6d_local'] ?? null;
      if (vr2Pos && vr2Pos.shape[1] !== 6) throw new Error(`vr_pos_local width ${vr2Pos.shape[1]} != 6`);
      if (vr2Orn && vr2Orn.shape[1] !== 12) throw new Error(`vr_orn_local width ${vr2Orn.shape[1]} != 12`);
      if (vr3Pos && vr3Pos.shape[1] !== 9) throw new Error(`vr3_pos_local width ${vr3Pos.shape[1]} != 9`);
      if (vr3PosTcp && vr3PosTcp.shape[1] !== 9)
        throw new Error(`vr3_pos_tcp_local width ${vr3PosTcp.shape[1]} != 9`);
      if (vr3Orn && vr3Orn.shape[1] !== 12) throw new Error(`vr3_orn_local width ${vr3Orn.shape[1]} != 12`);
      if (vr3Orn6d && vr3Orn6d.shape[1] !== 18) throw new Error(`vr3_orn6d_local width ${vr3Orn6d.shape[1]} != 18`);

      // 5-point tables (the 3pt-TCP set + both feet) — exp:g1-wbt-5pt-hybrid-unified-anneal.
      // Absent on bakes older than 2026-08-27; the accessors zero-fill rather than throw so a
      // stale clip still renders (bake_motions lists both keys in its required set, so
      // bake_motion_dir re-bakes instead of reusing an old NPZ).
      const vr5PosTcp = npz['vr5_pos_tcp_local'] ?? null;
      const vr5Orn = npz['vr5_orn_local'] ?? null;
      if (vr5PosTcp && vr5PosTcp.shape[1] !== 15)
        throw new Error(`vr5_pos_tcp_local width ${vr5PosTcp.shape[1]} != 15`);
      if (vr5Orn && vr5Orn.shape[1] !== 20) throw new Error(`vr5_orn_local width ${vr5Orn.shape[1]} != 20`);

      // Optional force-tracking reference: rubber-hand pos [L_xyz, R_xyz] in the
      // reference pelvis frame (force offset 0.0815, ±0.003). Absent on
      // synthetic/bake-less clips — getForceHandErrCm then returns null.
      const forceHandRef = npz['force_hand_ref_local'] ?? null;
      if (forceHandRef && forceHandRef.shape[1] !== 6)
        throw new Error(`force_hand_ref_local width ${forceHandRef.shape[1]} != 6`);

      entry = {
        vr2PosLocal: vr2Pos ? vr2Pos.data : null,
        vr2OrnLocal: vr2Orn ? vr2Orn.data : null,
        vr3PosLocal: vr3Pos ? vr3Pos.data : null,
        vr3PosTcpLocal: vr3PosTcp ? vr3PosTcp.data : null,
        vr3OrnLocal: vr3Orn ? vr3Orn.data : null,
        vr3Orn6dLocal: vr3Orn6d ? vr3Orn6d.data : null,
        vr5PosTcpLocal: vr5PosTcp ? vr5PosTcp.data : null,
        vr5OrnLocal: vr5Orn ? vr5Orn.data : null,
        lowerBodyJp,
        lowerBodyJv,
        refDofPos,
        refDofVel,
        refRootPosW,
        refRootVelW,
        forceHandRefLocal: forceHandRef ? forceHandRef.data : null,
        T,
      };
      this.bakedCache.set(name, entry);
    }
    this.activeBaked = entry;
    return ok;
  }

  private slice(buf: Float32Array | null, width: number): Float32Array {
    if (!this.activeBaked || !buf) return new Float32Array(width);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * width;
    return buf.subarray(off, off + width);
  }

  // ── 2pt baked accessors ──────────────────────────────────────────────────
  getVr2PosLocal(): Float32Array {
    return this.slice(this.activeBaked?.vr2PosLocal ?? null, 6);
  }
  getVr2OrnLocal(): Float32Array {
    // 12D = 2 wrists x 6D rot rep (bake_motions stores rot6d, not xyzw quat).
    return this.slice(this.activeBaked?.vr2OrnLocal ?? null, 12);
  }

  // ── force-tracking reference rubber-hand pos (pelvis frame, [L_xyz, R_xyz]) ──
  // Already stored PELVIS-RELATIVE by the bake, so plain slice() — do NOT
  // live-rotate it like getMotionRefPosB (that is for world-frame data).
  getForceHandRefLocal(): Float32Array {
    return this.slice(this.activeBaked?.forceHandRefLocal ?? null, 6);
  }

  // ── 3pt baked accessors ──────────────────────────────────────────────────
  getVr3PosLocal(): Float32Array {
    return this.slice(this.activeBaked?.vr3PosLocal ?? null, 9);
  }
  // TCP-unified hand point (0.242 m force TCP) — exp:g1-wbt-3pt-hybrid-unified[-anneal].
  // Falls back to the 0.18 m table when a pre-2026-08-03 bake lacks vr3_pos_tcp_local, so
  // an old clip still renders rather than feeding the policy zeros; re-bake for fidelity.
  getVr3PosTcpLocal(): Float32Array {
    return this.slice(this.activeBaked?.vr3PosTcpLocal ?? this.activeBaked?.vr3PosLocal ?? null, 9);
  }
  getVr3OrnLocal(): Float32Array {
    return this.slice(this.activeBaked?.vr3OrnLocal ?? null, 12);
  }
  getVr3Orn6dLocal(): Float32Array {
    // 18D = 3 bodies x 6D rot rep (bake_motions stores rot6d, first two columns of R).
    return this.slice(this.activeBaked?.vr3Orn6dLocal ?? null, 18);
  }

  // ── 5pt baked accessors (exp:g1-wbt-5pt-hybrid-unified-anneal) ───────────
  // 15D = 5 bodies x 3: [L hand @ 0.242 TCP, R hand @ TCP, head, L foot, R foot].
  // NO fallback to the 9D vr3_pos_tcp_local table (unlike getVr3PosTcpLocal, whose
  // fallback is width-COMPATIBLE): a 9D buffer sliced at width 15 would read across
  // frame boundaries and hand the policy garbage. Zero-fill on a stale bake instead —
  // visibly wrong (arms/feet driven to the pelvis origin) rather than subtly wrong.
  getVr5PosTcpLocal(): Float32Array {
    return this.slice(this.activeBaked?.vr5PosTcpLocal ?? null, 15);
  }
  getVr5OrnLocal(): Float32Array {
    // 20D = 5 bodies x 4 (xyzw quat, same rep as vr3_orn_local).
    return this.slice(this.activeBaked?.vr5OrnLocal ?? null, 20);
  }

  // ── lower-body command (used by command_lower_body) ─────────────────────
  getLowerBodyJp(): Float32Array {
    if (!this.activeBaked) return new Float32Array(12);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * 12;
    return this.activeBaked.lowerBodyJp.subarray(off, off + 12);
  }
  getLowerBodyJv(): Float32Array {
    if (!this.activeBaked) return new Float32Array(12);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * 12;
    return this.activeBaked.lowerBodyJv.subarray(off, off + 12);
  }

  // ── full 29-DOF reference pose / velocity (used by `command` 58D term) ──
  getRefDofPos(): Float32Array {
    if (!this.activeBaked) return new Float32Array(29);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * 29;
    return this.activeBaked.refDofPos.subarray(off, off + 29);
  }
  getRefDofVel(): Float32Array {
    if (!this.activeBaked) return new Float32Array(29);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * 29;
    return this.activeBaked.refDofVel.subarray(off, off + 29);
  }

  // ── motion_ref_pos_b / motion_ref_vel_b (live-rotated by robot root) ────
  // The bake stores the world-frame anchor pose/vel of the reference motion;
  // the rotation into the live robot's pelvis frame happens here so we never
  // have to round-trip the live state through the bake.
  getMotionRefPosB(rootPosW: ArrayLike<number>, rootQuatWxyz: ArrayLike<number>): Float32Array {
    if (!this.activeBaked) return new Float32Array(3);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * 3;
    const ref = this.activeBaked.refRootPosW.subarray(off, off + 3);
    const dx = (ref[0] ?? 0) - (rootPosW[0] ?? 0);
    const dy = (ref[1] ?? 0) - (rootPosW[1] ?? 0);
    const dz = (ref[2] ?? 0) - (rootPosW[2] ?? 0);
    return wbtRotateInverseLocal(rootQuatWxyz, dx, dy, dz);
  }
  getMotionRefVelB(rootQuatWxyz: ArrayLike<number>): Float32Array {
    if (!this.activeBaked) return new Float32Array(3);
    const off = Math.min(this.refIdx, this.activeBaked.T - 1) * 3;
    const v = this.activeBaked.refRootVelW.subarray(off, off + 3);
    return wbtRotateInverseLocal(rootQuatWxyz, v[0] ?? 0, v[1] ?? 0, v[2] ?? 0);
  }

  // ──────────────────────────────────────────────────────────────────────
  // Force-tracking debug visualization
  // ──────────────────────────────────────────────────────────────────────

  /** Builds the per-hand THREE.js groups for the triad + force arrow. Called
   *  lazily on the first updateDebugVisuals() call so that mjModel and the
   *  scene root are guaranteed populated. Idempotent (returns false on retry
   *  if the scene isn't ready yet). */
  private initForceViz(): boolean {
    if (this.forceVizInitialized || !this.forceVizConfig) return this.forceVizInitialized;
    const mjModel = this.wbtContext.mjModel;
    const sceneRoot = this.wbtContext.mujocoRoot ?? this.wbtContext.scene;
    if (!mjModel || !sceneRoot) return false;

    const triadLen = 0.10;  // 10 cm — long enough to read at typical zoom.
    const triadRadius = 0.0035;
    for (const cfg of this.forceVizConfig) {
      // Resolve body index by name. Scene-prefixed names (e.g. "scene/foo")
      // are also handled, mirroring TrackingCommand.findBodyIdByName.
      let bodyId = -1;
      for (let b = 0; b < mjModel.nbody; b++) {
        const name = mjModel.body(b).name;
        if (name === cfg.bodyName || name.endsWith(`/${cfg.bodyName}`)) {
          bodyId = b;
          break;
        }
      }
      if (bodyId < 0) {
        console.warn(`[WbtTrackingCommand] force_viz body '${cfg.bodyName}' not found`);
        continue;
      }

      const root = new THREE.Group();
      root.name = `wbt_force_viz_${cfg.bodyName}`;

      // Triad: three thin cylinders along ±x/y/z (red / green / blue).
      const triad = new THREE.Group();
      triad.name = 'triad';
      const axisGeom = new THREE.CylinderGeometry(triadRadius, triadRadius, triadLen, 8);
      // CylinderGeometry is +Y-aligned by default — rotate per axis.
      const axes: Array<[number, [number, number, number], 'x' | 'y' | 'z']> = [
        [0xff4040, [0, 0, -Math.PI / 2], 'x'],
        [0x40ff40, [0, 0, 0], 'y'],
        [0x4080ff, [Math.PI / 2, 0, 0], 'z'],
      ];
      for (const [color, rot, ax] of axes) {
        const mat = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 });
        const cyl = new THREE.Mesh(axisGeom, mat);
        cyl.rotation.set(rot[0], rot[1], rot[2]);
        // Shift so the cylinder starts at origin and extends in +axis.
        if (ax === 'x') cyl.position.set(triadLen / 2, 0, 0);
        if (ax === 'y') cyl.position.set(0, triadLen / 2, 0);
        if (ax === 'z') cyl.position.set(0, 0, triadLen / 2);
        cyl.renderOrder = 999;
        triad.add(cyl);
      }
      root.add(triad);

      // TWO arrows per hand, both anchored at the EXACT SAME rubber-hand point
      // (matching the benchmark videos — no positional offset):
      //   orange = F^ext (real disturbance) — thick shaft, drawn first (rO 1000)
      //   purple = F^cmd (intent)           — thin shaft,  drawn on top (rO 1001)
      // Fixed colors (no magnitude lerp). Differentiating by shaft radius (not a
      // positional nudge) keeps anchor parity with the benchmark while the
      // anti-parallel cancel case (F^ext = -F^cmd) still reads as a purple core
      // inside an orange halo rather than z-fighting into a single line.
      const extArrow = makeForceArrow(cfg.arrowColorExt, 1000, 0.012);
      const cmdArrow = makeForceArrow(cfg.arrowColorCmd, 1001, 0.006);
      root.add(extArrow.group);
      root.add(cmdArrow.group);
      // Cyan F^meas arrow (hybrid preset only) — the force the wrist actually
      // FEELS. Radius 0.009 (between ext 0.012 and cmd 0.006), drawn on top.
      let measArrow: WbtForceArrow | null = null;
      if (cfg.measSliderKeys) {
        measArrow = makeForceArrow(cfg.arrowColorMeas, 1003, 0.009);
        root.add(measArrow.group);
      }

      sceneRoot.add(root);
      this.forceVizGroups.push({
        root,
        triad,
        cmdArrow,
        extArrow,
        measArrow,
        bodyId,
      });
    }
    this.forceVizInitialized = true;
    return true;
  }

  /** Build the per-hand orange control-point sphere. Lazy, idempotent; returns
   *  false until the scene/model are ready. Mirrors initForceViz's body lookup
   *  and sceneRoot attach, but is gated on its OWN config so it works with or
   *  without the force viz. */
  private initControlPoint(): boolean {
    if (this.controlPointInitialized || !this.controlPointConfig) return this.controlPointInitialized;
    const mjModel = this.wbtContext.mjModel;
    const sceneRoot = this.wbtContext.mujocoRoot ?? this.wbtContext.scene;
    if (!mjModel || !sceneRoot) return false;

    for (const cfg of this.controlPointConfig) {
      let bodyId = -1;
      for (let b = 0; b < mjModel.nbody; b++) {
        const name = mjModel.body(b).name;
        if (name === cfg.bodyName || name.endsWith(`/${cfg.bodyName}`)) {
          bodyId = b;
          break;
        }
      }
      if (bodyId < 0) {
        console.warn(`[WbtTrackingCommand] control-point body '${cfg.bodyName}' not found`);
        continue;
      }
      const geom = new THREE.SphereGeometry(cfg.radius, 16, 12);
      // depthTest off so the marker reads even when behind a mesh; high render
      // order so it draws on top, matching the triad/arrow convention.
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(cfg.color[0], cfg.color[1], cfg.color[2]),
        depthTest: false,
        transparent: true,
        opacity: 0.95,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.renderOrder = 1002;
      mesh.name = `wbt_control_point_${cfg.bodyName}`;
      sceneRoot.add(mesh);
      this.controlPointGroups.push({ mesh, bodyId, offsetB: cfg.offsetB });
    }
    this.controlPointInitialized = true;
    return true;
  }

  /** Reposition each control-point sphere to the live vr_2point contact point
   *  (wristPos + R(wrist) @ offset_b), using the SAME swizzle path as the force
   *  triad. Positions only — a sphere needs no orientation. */
  private updateControlPoint(): void {
    if (!this.controlPointConfig) return;
    if (!this.controlPointInitialized && !this.initControlPoint()) return;
    const mjData = this.wbtContext.mjData;
    if (!mjData) return;
    for (const grp of this.controlPointGroups) {
      if (grp.bodyId < 0) continue;
      const wristPosThree = mjPosToThree(mjData.xpos, grp.bodyId);
      const wristQ = mjQuatToThree(mjData.xquat, grp.bodyId);
      const ob = grp.offsetB;
      // offset is in the wrist body's MuJoCo frame: swizzle to THREE local,
      // then rotate by the wrist's THREE quaternion (identical to the force-viz
      // anchor math at updateDebugVisuals).
      const offsetThreeLocal = new THREE.Vector3(ob[0], ob[2], -ob[1]);
      const offsetWorld = offsetThreeLocal.applyQuaternion(wristQ);
      grp.mesh.position.set(
        wristPosThree.x + offsetWorld.x,
        wristPosThree.y + offsetWorld.y,
        wristPosThree.z + offsetWorld.z,
      );
    }
  }

  // ──────────────────────────────────────────────────────────────────────
  // LIVE VIVE 3pt commanded-target triads
  // ──────────────────────────────────────────────────────────────────────

  /** Build one RGB axis triad + ball per commanded 3pt target. Lazy + idempotent,
   *  mirroring initForceViz/initControlPoint. Unlike those, these groups are NOT
   *  parented to a body — the target is a free pose in space, so nothing to look
   *  up in mjModel; we only need the scene root. */
  private initVrLiveTriads(): boolean {
    if (this.vrLiveTriadInitialized || !this.vrLiveTriadConfig) return this.vrLiveTriadInitialized;
    const sceneRoot = this.wbtContext.mujocoRoot ?? this.wbtContext.scene;
    if (!sceneRoot) return false;
    const cfg = this.vrLiveTriadConfig;
    for (const pt of cfg.points) {
      const root = new THREE.Group();
      root.name = `wbt_vr3_target_${pt.tag}`;
      const mats: THREE.MeshBasicMaterial[] = [];
      // Axis triad: +x red, +y green, +z blue, in the TARGET's own frame. Built
      // here rather than reusing the force-viz triad so that bundle's geometry
      // (length 0.10 / radius 0.0035, no ball) stays byte-identical.
      const axisGeom = new THREE.CylinderGeometry(0.004, 0.004, cfg.axisLen, 8);
      const axes: Array<[number, [number, number, number], 'x' | 'y' | 'z']> = [
        [0xff4040, [0, 0, -Math.PI / 2], 'x'],
        [0x40ff40, [0, 0, 0], 'y'],
        [0x4080ff, [Math.PI / 2, 0, 0], 'z'],
      ];
      for (const [color, rot, ax] of axes) {
        const mat = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 });
        const cyl = new THREE.Mesh(axisGeom, mat);
        cyl.rotation.set(rot[0], rot[1], rot[2]);
        if (ax === 'x') cyl.position.set(cfg.axisLen / 2, 0, 0);
        if (ax === 'y') cyl.position.set(0, cfg.axisLen / 2, 0);
        if (ax === 'z') cyl.position.set(0, 0, cfg.axisLen / 2);
        cyl.renderOrder = 1004; // above the white TCP sphere (1002) and arrows
        root.add(cyl);
        mats.push(mat);
      }
      // Per-point ball so L / R / head stay tellable apart when the three triads
      // overlap on screen (and so the point is readable at a zoomed-out camera).
      const ballMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(pt.color[0], pt.color[1], pt.color[2]),
        depthTest: false,
        transparent: true,
        opacity: 0.85,
      });
      const ball = new THREE.Mesh(new THREE.SphereGeometry(cfg.ballRadius, 16, 12), ballMat);
      ball.renderOrder = 1004;
      root.add(ball);
      mats.push(ballMat);
      sceneRoot.add(root);
      this.vrLiveTriadGroups.push({ root, ball, mats });
    }
    this.vrLiveTriadInitialized = true;
    return true;
  }

  /** Pose each triad at its commanded target: read the PELVIS-frame pos + xyzw
   *  quat off the extras bus, compose with the live pelvis world pose, and swizzle
   *  to THREE — the same swizzle-then-rotate path updateControlPoint uses for a
   *  body-local offset, extended with the orientation compose.
   *
   *  Frame note: the obs term is `R(q_pelvis)^-1 (p_target_w - p_pelvis_w)` with
   *  the FULL pelvis rotation (holosoma_wbt.py `_compute_vr_fk`, anchor = the
   *  reference pelvis), not a yaw-only heading frame — so un-projecting with the
   *  full live pelvis quat is the exact inverse. */
  private updateVrLiveTriads(): void {
    const cfg = this.vrLiveTriadConfig;
    if (!cfg) return;
    if (!this.vrLiveTriadInitialized && !this.initVrLiveTriads()) return;
    const mjData = this.wbtContext.mjData;
    if (!mjData) return;
    const rootBodyId = this.findRootBodyId();
    if (rootBodyId < 0) return;
    const pelvisPos = mjPosToThree(mjData.xpos, rootBodyId);
    const pelvisQ = mjQuatToThree(mjData.xquat, rootBodyId);
    // Dim every triad while the bridge is down / stale, so a HOME-fallback pose
    // never reads as a live command.
    const live = getPolicyExtra(cfg.okKey, 0.0) >= 0.5;
    for (let i = 0; i < this.vrLiveTriadGroups.length; i++) {
      const grp = this.vrLiveTriadGroups[i];
      const pt = cfg.points[i];
      if (!grp || !pt) continue;
      const px = getPolicyExtra(`${cfg.prefix}__${pt.tag}_px`, 0.0);
      const py = getPolicyExtra(`${cfg.prefix}__${pt.tag}_py`, 0.0);
      const pz = getPolicyExtra(`${cfg.prefix}__${pt.tag}_pz`, 0.0);
      // Pelvis-frame offset -> THREE local (mx, mz, -my), then rotate by pelvis.
      const offsetWorld = new THREE.Vector3(px, pz, -py).applyQuaternion(pelvisQ);
      grp.root.position.set(
        pelvisPos.x + offsetWorld.x,
        pelvisPos.y + offsetWorld.y,
        pelvisPos.z + offsetWorld.z,
      );
      // Wire orn is xyzw in the pelvis frame. MuJoCo wxyz = (qw,qx,qy,qz), and
      // mjswan's quat swizzle is THREE xyzw = (-mqx, -mqz, mqy, -mqw) — see
      // mjQuatToThree. That map is a homomorphism up to sign (it is conjugation
      // by a fixed rotation), so composing pelvis * local in THREE space is the
      // same rotation as composing them in MuJoCo space and swizzling once.
      const qx = getPolicyExtra(`${cfg.prefix}__${pt.tag}_qx`, 0.0);
      const qy = getPolicyExtra(`${cfg.prefix}__${pt.tag}_qy`, 0.0);
      const qz = getPolicyExtra(`${cfg.prefix}__${pt.tag}_qz`, 0.0);
      const qw = getPolicyExtra(`${cfg.prefix}__${pt.tag}_qw`, 1.0);
      const qLocalThree = new THREE.Quaternion(-qx, -qz, qy, -qw).normalize();
      grp.root.quaternion.copy(pelvisQ).multiply(qLocalThree);
      for (const m of grp.mats) m.opacity = live ? (m === grp.ball.material ? 0.85 : 0.9) : 0.25;
    }
  }

  /** Build the vr_3point tracking-point spheres: one SOLID sphere per point on the
   *  live robot + one TRANSLUCENT sphere per point on the reference ghost (same hue).
   *  Lazy, idempotent; returns false until the scene/model are ready. Body ids are
   *  resolved once against mjModel (the ghost shares the same render mjModel, so the
   *  id is valid for both mjData and ghostData). */
  private initTrackingMarkers(): boolean {
    if (this.trackingMarkerInitialized || !this.trackingMarkerConfig) return this.trackingMarkerInitialized;
    const mjModel = this.wbtContext.mjModel;
    const sceneRoot = this.wbtContext.mujocoRoot ?? this.wbtContext.scene;
    if (!mjModel || !sceneRoot) return false;

    for (const cfg of this.trackingMarkerConfig) {
      let bodyId = -1;
      for (let b = 0; b < mjModel.nbody; b++) {
        const name = mjModel.body(b).name;
        if (name === cfg.bodyName || name.endsWith(`/${cfg.bodyName}`)) {
          bodyId = b;
          break;
        }
      }
      if (bodyId < 0) {
        console.warn(`[WbtTrackingCommand] tracking-marker body '${cfg.bodyName}' not found`);
        continue;
      }
      // REAL marker: solid, opaque, draws on top (depthTest off) like the other markers.
      const realMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(cfg.realColor[0], cfg.realColor[1], cfg.realColor[2]),
        depthTest: false,
        transparent: true,
        opacity: 0.95,
      });
      const realMesh = new THREE.Mesh(new THREE.SphereGeometry(cfg.radius, 16, 12), realMat);
      realMesh.renderOrder = 1002;
      realMesh.name = `wbt_track_real_${cfg.bodyName}`;
      sceneRoot.add(realMesh);
      this.trackingMarkerReal.push({ mesh: realMesh, bodyId, offsetB: cfg.offsetB });

      // REFERENCE marker: same hue, translucent (target-vs-actual reads as light-vs-solid).
      const refMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(cfg.refColor[0], cfg.refColor[1], cfg.refColor[2]),
        depthTest: false,
        transparent: true,
        opacity: 0.35,
      });
      const refMesh = new THREE.Mesh(new THREE.SphereGeometry(cfg.radius, 16, 12), refMat);
      refMesh.renderOrder = 1001; // just under the solid real marker
      refMesh.name = `wbt_track_ref_${cfg.bodyName}`;
      sceneRoot.add(refMesh);
      this.trackingMarkerRef.push({ mesh: refMesh, bodyId, offsetB: cfg.offsetB });
    }
    this.trackingMarkerInitialized = true;
    return true;
  }

  /** Reposition the vr_3point spheres each frame. REAL spheres from the live robot
   *  (mjData.xpos/xquat); REFERENCE spheres from the ghost (ghostData.xpos/xquat,
   *  refreshed by the base class's updateGhostPose each frame). Same anchor math as
   *  updateControlPoint: `bodyPos + R(bodyQuat) @ offset_b`. The reference spheres
   *  hide with the ghost (referenceVisible) and when no motion is selected. */
  private updateTrackingMarkers(): void {
    if (!this.trackingMarkerConfig) return;
    if (!this.trackingMarkerInitialized && !this.initTrackingMarkers()) return;

    const place = (
      grp: { mesh: THREE.Mesh; bodyId: number; offsetB: [number, number, number] },
      xpos: Float32Array | Float64Array,
      xquat: Float32Array | Float64Array,
    ): void => {
      if (grp.bodyId < 0) return;
      const bodyPos = mjPosToThree(xpos, grp.bodyId);
      const bodyQ = mjQuatToThree(xquat, grp.bodyId);
      const ob = grp.offsetB;
      // offset is in the body's MuJoCo frame: swizzle to THREE local (x, z, -y),
      // then rotate by the body's THREE quaternion (identical to updateControlPoint).
      const offsetWorld = new THREE.Vector3(ob[0], ob[2], -ob[1]).applyQuaternion(bodyQ);
      grp.mesh.position.set(bodyPos.x + offsetWorld.x, bodyPos.y + offsetWorld.y, bodyPos.z + offsetWorld.z);
    };

    // REAL spheres: always follow the live robot.
    const mjData = this.wbtContext.mjData;
    if (mjData) {
      for (const grp of this.trackingMarkerReal) {
        grp.mesh.visible = true;
        place(grp, mjData.xpos, mjData.xquat);
      }
    }

    // REFERENCE spheres: follow the ghost, and share its visibility (hidden when the
    // reference is toggled off or no motion is selected). refLen (public) is >0 only
    // once a motion is loaded, so it mirrors the base's `selectedMotion !== null`
    // guard in updateGhostPose without touching the base's private selectedMotion.
    const refVisible = this.referenceVisible && this.refLen > 0;
    for (const grp of this.trackingMarkerRef) {
      grp.mesh.visible = refVisible;
    }
    if (refVisible && this.ghostData) {
      for (const grp of this.trackingMarkerRef) {
        place(grp, this.ghostData.xpos, this.ghostData.xquat);
      }
    }
  }

  /** Orient + scale one force arrow from a live slider triple.
   *  frame='pelvis' (default): the triple is in the robot ROOT (pelvis) frame, so
   *  swizzle to THREE and compose with the pelvis world quat (F^cmd / F^ext).
   *  frame='world': the triple is already in the MuJoCo WORLD frame (the hybrid
   *  causal F^meas — the wall normal is world-fixed), so swizzle to THREE and use
   *  it directly (NO root-quat compose). rootQuatThree is the live pelvis world
   *  quaternion in THREE coords (ignored for frame='world'). */
  private updateForceArrow(
    arrow: WbtForceArrow,
    keys: [string, string, string],
    rootQuatThree: THREE.Quaternion,
    frame: 'pelvis' | 'world' = 'pelvis',
  ): void {
    const fx = getPolicyExtra(keys[0], 0.0);
    const fy = getPolicyExtra(keys[1], 0.0);
    const fz = getPolicyExtra(keys[2], 0.0);
    const mag = Math.sqrt(fx * fx + fy * fy + fz * fz);
    if (mag < 1e-6) {
      arrow.group.visible = false;
      return;
    }
    arrow.group.visible = true;
    // Benchmark length: max(|F|*M_PER_N, MIN_LEN), uncapped (25 N -> 0.45 m).
    const arrowLen = Math.max(mag * WBT_FORCE_ARROW_M_PER_N, WBT_FORCE_ARROW_MIN_LEN);
    // Swizzle the MuJoCo-frame force to THREE (mx,mz,-my), build the rotation
    // from +Y to that direction. For a pelvis-frame force, compose with the
    // pelvis world quat; for a world-frame force, the swizzled direction is
    // already the world direction, so use it as-is.
    const dirThree = new THREE.Vector3(fx, fz, -fy).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const rotToDir = new THREE.Quaternion().setFromUnitVectors(up, dirThree);
    if (frame === 'world') {
      arrow.group.quaternion.copy(rotToDir);
    } else {
      arrow.group.quaternion.copy(rootQuatThree).multiply(rotToDir);
    }
    arrow.shaft.scale.set(1, arrowLen, 1);
    arrow.head.position.set(0, arrowLen, 0);
  }

  /** True for a body that moves — the same test the parent's ghost uses to skip static
   *  world geometry (floor, props). Cloning those would double-draw the scene in green. */
  private mcGhostIsDynamicBody(bodyId: number): boolean {
    const mjModel = this.wbtContext.mjModel;
    if (!mjModel || bodyId <= 0 || bodyId >= mjModel.nbody) return false;
    let current = bodyId;
    while (current > 0) {
      if (mjModel.body_jntnum[current] > 0) return true;
      current = mjModel.body_parentid[current];
    }
    return false;
  }

  /** Build the light-purple completion ghost. Called lazily on the FIRST frame that actually
   *  carries completion data, so a preset without a `--publish-mc-qpos` bridge — i.e. every
   *  non-MC preset — never allocates a second body or a second MjData. Returns false and
   *  latches on failure so a missing model does not retry every frame. */
  private initMcGhost(): boolean {
    if (this.mcGhostFailed) return false;
    const mjModel = this.wbtContext.mjModel;
    const bodies = this.wbtContext.bodies ?? null;
    if (!mjModel || !bodies) {
      this.mcGhostFailed = true;
      return false;
    }
    // Joint name -> qpos address, from the RENDER model. Stored under the BARE name (mjswan
    // prefixes model joints with the scene name); the completion publishes bare URDF names.
    // mjtJoint: 0=free, 1=ball, 2=slide, 3=hinge. Ball is skipped, not indexed: it occupies
    // four qpos slots and a scalar angle written into its first one is a corrupt quaternion.
    for (let j = 0; j < mjModel.njnt; j++) {
      const type = mjModel.jnt_type[j];
      if (type === 0) {
        if (this.mcGhostFreeAdr < 0) this.mcGhostFreeAdr = mjModel.jnt_qposadr[j];
        continue;
      }
      if (type === 1) continue;
      const full = mjModel.jnt(j).name ?? '';
      const bare = full.includes('/') ? full.slice(full.lastIndexOf('/') + 1) : full;
      if (bare) this.mcGhostJointAdr.push({ name: bare, adr: mjModel.jnt_qposadr[j] });
    }
    const root = new THREE.Group();
    root.name = 'MC Completion Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
      const id = Number(bodyId);
      if (!this.mcGhostIsDynamicBody(id)) continue;
      const clone = body.clone(true);
      let hasMesh = false;
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          hasMesh = true;
          const paint = (m: THREE.Material): THREE.Material => {
            const next = m.clone();
            if ('transparent' in next) next.transparent = true;
            if ('opacity' in next) next.opacity = 0.55;
            if ('depthWrite' in next) next.depthWrite = false;
            // LIGHT PURPLE (lavender). Picked to be unmistakable against the three things it
            // shares a screen with: the parent ghost's pale GREEN (0.5,0.7,0.5), the solid
            // robot, and the SATURATED purple F^cmd force arrow (0.62,0.20,0.92) — the arrow
            // is a different quantity entirely, so the ghost is deliberately far lighter and
            // far less saturated than it.
            if ('color' in next && next.color instanceof THREE.Color) {
              next.color = new THREE.Color(WbtTrackingCommand.MC_GHOST_RGB[0],
                WbtTrackingCommand.MC_GHOST_RGB[1], WbtTrackingCommand.MC_GHOST_RGB[2]);
            }
            return next;
          };
          obj.material = Array.isArray(obj.material)
            ? obj.material.map(paint)
            : paint(obj.material);
          // Above the parent ghost (2) so the completion wins the depth-less blend when the
          // two overlap — which they do for the whole first second.
          obj.renderOrder = 3;
        }
      });
      if (!hasMesh) continue;
      this.mcGhostBodies.set(id, clone);
      root.add(clone);
    }
    if (this.mcGhostBodies.size === 0) {
      this.mcGhostFailed = true;
      return false;
    }
    this.mcGhostData = new this.wbtContext.mujoco.MjData(mjModel);
    (this.wbtContext.mujocoRoot ?? this.wbtContext.scene).add(root);
    this.mcGhostRoot = root;
    this.mcGhostInitialized = true;
    console.log(
      `[WbtMcGhost] LIGHT-PURPLE completion reference ghost built: ` +
      `${this.mcGhostBodies.size} bodies, ${this.mcGhostJointAdr.length} indexable joints, ` +
      `free-joint qposadr=${this.mcGhostFreeAdr}. It is the completion's GT body — the solid ` +
      'robot is the low-level policy trying to follow it.',
    );
    return true;
  }

  /** Drive the completion ghost from the extras bus (published by WbtVrLiveStore off the
   *  bridge's `--publish-mc-qpos` stream). No-op until a frame has arrived. */
  private updateMcGhost(): void {
    // `wbt_mcghost__seq` is written LAST by the producer, so a nonzero value means a whole
    // body — joints AND root — landed. 0 means no MC ghost stream on this page at all,
    // which is the normal state of every non-MC preset: stay silent and allocate nothing.
    const seq = getPolicyExtra('wbt_mcghost__seq', 0);
    if (seq <= 0) return;
    if (!this.mcGhostInitialized && !this.initMcGhost()) return;
    const mjModel = this.wbtContext.mjModel;
    const ghost = this.mcGhostData;
    const root = this.mcGhostRoot;
    if (!mjModel || !ghost || !root) return;

    // Hidden by the SAME toggle as the parent's reference ghost: both are "the reference",
    // and an operator hiding one and getting the other is the kind of surprise that gets
    // read as a policy behaviour.
    if (!this.referenceVisible) {
      root.visible = false;
      return;
    }

    // FRESHNESS. The bridge can die, the completion backend can wedge, the socket can drop —
    // and every one of those leaves the last body sitting there looking authoritative. During
    // teleop the operator is steering AT this mesh, so a frozen one turns their own tracking
    // error into a phantom policy failure. Hide it and say so, once per transition.
    const age = performance.now() - getPolicyExtra('wbt_mcghost__rx', 0);
    if (age > WbtTrackingCommand.MC_GHOST_STALE_MS) {
      root.visible = false;
      if (!this.mcGhostStale) {
        this.mcGhostStale = true;
        console.warn(
          `[WbtMcGhost] completion stream stale (${age.toFixed(0)} ms > ` +
          `${WbtTrackingCommand.MC_GHOST_STALE_MS} ms) — hiding the purple reference rather ` +
          'than showing a frozen target. Check the bridge.',
        );
      }
      return;
    }
    if (this.mcGhostStale) {
      this.mcGhostStale = false;
      console.log('[WbtMcGhost] completion stream is live again; purple reference restored.');
    }
    // Same pose as last solve -> the meshes are already where they belong (hiding the group
    // does not move them), so re-showing is a visibility flip and nothing else.
    if (seq === this.mcGhostLastSeq && this.refIdx === this.mcGhostLastRefIdx) {
      root.visible = true;
      return;
    }

    const qpos = ghost.qpos;
    qpos.set(mjModel.qpos0);

    // WHERE THE GHOST STANDS. The wire's root is relative to the completion's READOUT frame
    // (`_root_in_readout_frame_locked`) — the same frozen pelvis the 5pt keypoints are
    // relativised into. So it composes onto whatever this scene calls that frame, which is
    // the REFERENCE CLIP's root: in these presets that clip is the static `idle_default`,
    // i.e. the held reference pose the policy is anchored on. Using the live robot's pelvis
    // instead would cancel the very motion we are trying to see, the same way `--ref-frame
    // live` cancels 97.6% of the operator's walk.
    //
    // NOT exactly coincident at MC frame 0, and the residual is expected: the ghost sits at
    // the completion's OWN output root, which stands low and tilted relative to the frozen
    // reference (measured 12.7 mm, -12.0 of it in z, 3.2 deg of tilt). The HANDS are the
    // sub-mm match, not the pelvis. Two centimetres of standing offset at t=0 is this, not a
    // frame bug — a step is 20-50x larger.
    const anchorPos = this.refRootPos[this.refIdx];
    const anchorQuat = this.refRootQuat[this.refIdx];
    const aP: [number, number, number] = anchorPos
      ? [anchorPos[0] ?? 0, anchorPos[1] ?? 0, anchorPos[2] ?? 0]
      : this.mcGhostFreeAdr >= 0
        ? [
          mjModel.qpos0[this.mcGhostFreeAdr] ?? 0,
          mjModel.qpos0[this.mcGhostFreeAdr + 1] ?? 0,
          mjModel.qpos0[this.mcGhostFreeAdr + 2] ?? 0,
        ]
        : [0, 0, 0];
    const aQ: [number, number, number, number] = anchorQuat
      ? [anchorQuat[0] ?? 1, anchorQuat[1] ?? 0, anchorQuat[2] ?? 0, anchorQuat[3] ?? 0]
      : [1, 0, 0, 0];

    const rP: [number, number, number] = [
      getPolicyExtra('wbt_mcghost__rp0', 0),
      getPolicyExtra('wbt_mcghost__rp1', 0),
      getPolicyExtra('wbt_mcghost__rp2', 0),
    ];
    const rQ: [number, number, number, number] = [
      getPolicyExtra('wbt_mcghost__rq0', 1),
      getPolicyExtra('wbt_mcghost__rq1', 0),
      getPolicyExtra('wbt_mcghost__rq2', 0),
      getPolicyExtra('wbt_mcghost__rq3', 0),
    ];
    if (this.mcGhostFreeAdr >= 0) {
      const off = wbtQuatApplyWxyz(aQ, rP[0], rP[1], rP[2]);
      const q = wbtQuatMulWxyz(aQ, rQ);
      qpos[this.mcGhostFreeAdr + 0] = aP[0] + (off[0] ?? 0);
      qpos[this.mcGhostFreeAdr + 1] = aP[1] + (off[1] ?? 0);
      qpos[this.mcGhostFreeAdr + 2] = aP[2] + (off[2] ?? 0);
      qpos[this.mcGhostFreeAdr + 3] = q[0] ?? 1;
      qpos[this.mcGhostFreeAdr + 4] = q[1] ?? 0;
      qpos[this.mcGhostFreeAdr + 5] = q[2] ?? 0;
      qpos[this.mcGhostFreeAdr + 6] = q[3] ?? 0;
    }

    // BY NAME. An unset key comes back NaN and leaves that joint at qpos0, so a joint the
    // completion does not publish visibly stays put instead of being fed someone else's angle.
    for (const j of this.mcGhostJointAdr) {
      const v = getPolicyExtra(`wbt_mcghost__q__${j.name}`, Number.NaN);
      if (Number.isFinite(v)) qpos[j.adr] = v;
    }

    this.wbtContext.mujoco.mj_forward(mjModel, ghost);
    for (const [bodyId, body] of this.mcGhostBodies) {
      const p = mjPosToThree(ghost.xpos, bodyId);
      // A body that came back non-finite means the qpos we wrote was not a valid state. Skip
      // it rather than writing NaN into a THREE transform, which silently poisons the whole
      // subtree's matrix and can take the RENDER down with it — i.e. one bad ghost frame
      // would look like the sim itself crashing.
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) continue;
      body.position.copy(p);
      body.quaternion.copy(mjQuatToThree(ghost.xquat, bodyId));
    }
    root.visible = true;
    this.mcGhostLastSeq = seq;
    this.mcGhostLastRefIdx = this.refIdx;
  }

  override updateDebugVisuals(): void {
    super.updateDebugVisuals?.();
    // The motion-completion reference ghost. Independent of every config below (it is armed
    // by the wire, not by a preset flag) and a no-op until a completion frame arrives.
    this.updateMcGhost();
    // Control-point marker runs INDEPENDENTLY of the force viz (it may be on for
    // a non-force 2pt preset), so update it first and don't gate it on
    // forceVizConfig.
    this.updateControlPoint();
    // Same independence for the live 3pt target triads: they are driven by the
    // extras bus, not by any force config, and must run before the early return
    // below (its preset IS a force preset, but that is incidental).
    this.updateVrLiveTriads();
    // vr_3point tracking-point markers (real + reference) — also independent of the
    // force viz; a no-op unless the preset emitted vr_tracking_point_markers.
    this.updateTrackingMarkers();

    if (!this.forceVizConfig) return;
    if (!this.forceVizInitialized && !this.initForceViz()) return;

    const mjData = this.wbtContext.mjData;
    if (!mjData) return;

    // Coordinate convention reminder: mjData.xpos / mjData.xquat are in
    // MuJoCo z-up frame. mjswan renders bodies in THREE.js y-up by
    // SWIZZLING reads from xpos/xquat via getPosition/getQuaternion (see
    // third_party/mjswan/src/mjswan/template/src/core/scene/scene.ts:728).
    // Three's transforms — including this Group's local frame inside
    // mujocoRoot — are therefore in the THREE y-up frame. Any quantity we
    // pull from mjData must be swizzled the same way mjswan does or the
    // triad lands in the wrong scene location with the wrong orientation.
    //
    // Swizzle map (mjswan-compatible):
    //   pos:   (mx, my, mz)  ->  (mx,  mz, -my)
    //   quat:  (mw, mx, my, mz)  ->  THREE xyzw = (-mx, -mz, my, -mw)
    // See helpers at the bottom of this file (mjPosToThree / mjQuatToThree).

    // Robot root (pelvis) world quaternion in THREE coords — used to
    // interpret slider F^cmd values, which training stores in robot ROOT
    // (MuJoCo z-up) frame.
    const rootBodyId = this.findRootBodyId();
    const rootQuatThree = rootBodyId >= 0
      ? mjQuatToThree(mjData.xquat, rootBodyId)
      : new THREE.Quaternion();

    for (let i = 0; i < this.forceVizGroups.length; i++) {
      const grp = this.forceVizGroups[i];
      const cfg = this.forceVizConfig[i];
      if (!grp || !cfg || grp.bodyId < 0) continue;

      // Parent body (wrist_yaw_link) world pose, swizzled to THREE coords.
      const wristPosThree = mjPosToThree(mjData.xpos, grp.bodyId);
      const wristQ = mjQuatToThree(mjData.xquat, grp.bodyId);

      // The URDF-local offset (0.0815, +0.003, 0) is in the WRIST BODY's
      // own MuJoCo frame. To project it into world, first swizzle to
      // THREE local coords, then rotate by the wrist's THREE quaternion:
      //   offsetThreeLocal_xyz = (mx, mz, -my)
      const ob = cfg.offsetB;
      const offsetThreeLocal = new THREE.Vector3(ob[0], ob[2], -ob[1]);
      const offsetWorld = offsetThreeLocal.clone().applyQuaternion(wristQ);
      grp.root.position.set(
        wristPosThree.x + offsetWorld.x,
        wristPosThree.y + offsetWorld.y,
        wristPosThree.z + offsetWorld.z,
      );

      // Triad orientation = wrist body world quaternion (in THREE coords),
      // so R/G/B axes align with the wrist's body frame at the contact point.
      grp.triad.quaternion.copy(wristQ);

      // Two arrows, same anchor (grp.root.position). Each reads its own slider
      // triple (pelvis-frame raw Newtons) and is oriented via the verified
      // swizzle-then-rotate path. Orange F^ext (thick) underneath, purple F^cmd
      // (thin) on top; colors + radii are fixed at init, lengths use the
      // benchmark formula. The radius difference (set in makeForceArrow via
      // initForceViz) keeps the anti-parallel cancel case legible.
      this.updateForceArrow(grp.extArrow, cfg.extSliderKeys, rootQuatThree, 'pelvis');
      this.updateForceArrow(grp.cmdArrow, cfg.cmdSliderKeys, rootQuatThree, 'pelvis');
      // Cyan F^meas arrow (hybrid only): the wrist's felt force. World-native
      // (the causal wall normal is world-fixed), so drawn WITHOUT the root-quat
      // compose (frame='world').
      if (grp.measArrow && cfg.measSliderKeys) {
        this.updateForceArrow(grp.measArrow, cfg.measSliderKeys, rootQuatThree, cfg.measFrame);
      }
    }

    // Hybrid live controls panel + spring-wall plane + servo-phase badge
    // (all no-ops for non-hybrid presets).
    this.ensureHybridControls();
    this.updateHybridWall();
    this.updateHybridServoBadge();
  }

  // ──────────────────────────────────────────────────────────────────────
  // Hybrid virtual spring wall + servo-phase badge (exp:g1-wbt-2pt-hybrid)
  // ──────────────────────────────────────────────────────────────────────

  /** Live pelvis (root) world quat (wxyz) from qpos[3:7] — native MuJoCo z-up,
   *  the frame the F^cmd sliders live in. Used to rotate F^cmd → world for the
   *  wall normal, mirroring the runtime's hybridRootMat. */
  private hybridPelvisQuatWxyz(): [number, number, number, number] {
    const q = this.wbtContext.mjData?.qpos;
    if (!q) return [1, 0, 0, 0];
    return [q[3] ?? 1, q[4] ?? 0, q[5] ?? 0, q[6] ?? 0];
  }

  /** Build the per-hand translucent wall plane (lazy, idempotent). One thin
   *  double-sided quad per hand, re-posed + tinted each frame in updateHybridWall. */
  private initHybridWall(): boolean {
    if (this.hybridWallInitialized || !this.hybridVizConfig) return this.hybridWallInitialized;
    const mjModel = this.wbtContext.mjModel;
    const sceneRoot = this.wbtContext.mujocoRoot ?? this.wbtContext.scene;
    if (!mjModel || !sceneRoot) return false;
    for (const hand of this.hybridVizConfig.hands) {
      let bodyId = -1;
      for (let b = 0; b < mjModel.nbody; b++) {
        const name = mjModel.body(b).name;
        if (name === hand.bodyName || name.endsWith(`/${hand.bodyName}`)) {
          bodyId = b;
          break;
        }
      }
      // A 0.4 m square quad in its local XY plane (+Z = wall normal). Repositioned
      // + oriented (normal→+Z) each frame. Translucent so the hand visibly sinks in.
      const geom = new THREE.PlaneGeometry(0.4, 0.4);
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(0.55, 0.6, 0.7),
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.renderOrder = 998; // under the arrows (>=1000)
      mesh.visible = false;
      mesh.name = `wbt_hybrid_wall_${hand.bodyName}`;
      sceneRoot.add(mesh);
      this.hybridWallGroups.push({ mesh, mat, bodyId });
    }
    this.hybridWallInitialized = true;
    return true;
  }

  /** Reposition + tint each wall plane from the live TCP + F^cmd, mirroring the
   *  runtime's seeding (normal=−unit(F_cmd_world), point=TCP−plane_dist·n̂) and
   *  reddening as penetration→max. Hidden when F^cmd≈0 (no wall commanded). */
  private updateHybridWall(): void {
    if (!this.hybridVizConfig) return;
    if (!this.hybridWallInitialized && !this.initHybridWall()) return;
    const mjData = this.wbtContext.mjData;
    if (!mjData) return;
    // Wall toggled OFF (live): hide every plane; no causal contact is applied by
    // the runtime, so drawing the plane would misrepresent the physics. The seed
    // is preserved (so re-enabling snaps the plane back to the same place).
    if (!this.hybridWallOn) {
      for (const wall of this.hybridWallGroups) wall.mesh.visible = false;
      return;
    }
    const cfg = this.hybridVizConfig;
    const planeDist = cfg.planeDist; // seed offset (m), same value the runtime uses
    const qP = this.hybridPelvisQuatWxyz();
    for (let i = 0; i < this.hybridWallGroups.length; i++) {
      const wall = this.hybridWallGroups[i];
      const hand = cfg.hands[i];
      const seed = this.hybridWallSeed[i];
      if (!wall || wall.bodyId < 0) continue;
      // Live TCP (world, MuJoCo z-up): xpos + R_wrist @ (tcp_offset,0,0).
      const p3 = wall.bodyId * 3;
      const m9 = wall.bodyId * 9;
      const xm = mjData.xmat;
      const off = cfg.tcpOffsetX;
      const tcp: [number, number, number] = [
        (mjData.xpos[p3 + 0] ?? 0) + xm[m9 + 0] * off,
        (mjData.xpos[p3 + 1] ?? 0) + xm[m9 + 3] * off,
        (mjData.xpos[p3 + 2] ?? 0) + xm[m9 + 6] * off,
      ];
      // F^cmd (pelvis) → world via the pelvis quat (wxyz), matching the runtime.
      const fc = [
        getPolicyExtra(hand.cmdSliderKeys[0], 0.0),
        getPolicyExtra(hand.cmdSliderKeys[1], 0.0),
        getPolicyExtra(hand.cmdSliderKeys[2], 0.0),
      ];
      const fcW = wbtQuatApplyWxyz(qP, fc[0], fc[1], fc[2]);
      const fcMag = Math.hypot(fcW[0], fcW[1], fcW[2]);
      if (fcMag < 1e-6) {
        wall.mesh.visible = false;
        seed.seeded = false;
        seed.seedTcpW = [0, 0, 0];
        continue;
      }
      if (!seed.seeded) {
        seed.normalW = [-fcW[0] / fcMag, -fcW[1] / fcMag, -fcW[2] / fcMag];
        seed.seedTcpW = [tcp[0], tcp[1], tcp[2]];
        seed.pointW = [
          tcp[0] - planeDist * seed.normalW[0],
          tcp[1] - planeDist * seed.normalW[1],
          tcp[2] - planeDist * seed.normalW[2],
        ];
        seed.seeded = true;
      } else if (this.hybridWallLive) {
        // Mirror the runtime's live-normal branch EXACTLY (same pivot, same formula) so
        // the drawn plane keeps matching the plane the physics actually reacts on. Any
        // divergence here would make the viz lie about where contact happens.
        seed.normalW = [-fcW[0] / fcMag, -fcW[1] / fcMag, -fcW[2] / fcMag];
        seed.pointW = [
          seed.seedTcpW[0] - planeDist * seed.normalW[0],
          seed.seedTcpW[1] - planeDist * seed.normalW[1],
          seed.seedTcpW[2] - planeDist * seed.normalW[2],
        ];
      }
      // Penetration depth (world): clamp(max(0, n̂·(point−TCP)), 0, max_pen).
      const dx = seed.pointW[0] - tcp[0];
      const dy = seed.pointW[1] - tcp[1];
      const dz = seed.pointW[2] - tcp[2];
      const signed = dx * seed.normalW[0] + dy * seed.normalW[1] + dz * seed.normalW[2];
      const depth = Math.min(Math.max(signed, 0), cfg.maxPenetrationM);
      const frac = cfg.maxPenetrationM > 0 ? depth / cfg.maxPenetrationM : 0;
      // Position the quad at the plane point (swizzled z-up→y-up), orient +Z → n̂.
      wall.mesh.visible = true;
      const pointThree = new THREE.Vector3(seed.pointW[0], seed.pointW[2], -seed.pointW[1]);
      const normalThree = new THREE.Vector3(seed.normalW[0], seed.normalW[2], -seed.normalW[1]).normalize();
      wall.mesh.position.copy(pointThree);
      wall.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normalThree);
      // Cool grey → red as penetration → max; opacity rises slightly with contact.
      wall.mat.color.setRGB(0.55 + 0.45 * frac, 0.6 - 0.5 * frac, 0.7 - 0.6 * frac);
      wall.mat.opacity = 0.18 + 0.22 * frac;
    }
  }

  /** Fixed DOM badge (top-center) showing the control phase per hand:
   *  POSITION TRACKING (grey, free space / gate 0) vs FORCE SERVO (green, gate 1),
   *  or "POSITION TRACKING (Layer A — servo off)" for the servo-off preset. Reads
   *  the per-hand contact from the wall seed / live penetration computed above. */
  private updateHybridServoBadge(): void {
    if (!this.hybridVizConfig || typeof document === 'undefined') return;
    if (!this.hybridBadgeEl) {
      const el = document.createElement('div');
      el.id = 'wbt-hybrid-servo-badge';
      Object.assign(el.style, {
        position: 'fixed',
        top: '12px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: '10001',
        padding: '8px 16px',
        borderRadius: '6px',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: '13px',
        fontWeight: 'bold',
        color: '#fff',
        background: 'rgba(80, 88, 100, 0.9)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        pointerEvents: 'none',
      } as Partial<CSSStyleDeclaration>);
      document.body.appendChild(el);
      this.hybridBadgeEl = el;
    }
    const el = this.hybridBadgeEl;
    // Which hands are penetrating the (live) wall? Only meaningful when the wall
    // is ON — with the wall toggled off there is no causal contact at all.
    const inContact: string[] = [];
    if (this.hybridWallOn) {
      for (let i = 0; i < this.hybridWallSeed.length; i++) {
        const seed = this.hybridWallSeed[i];
        const wall = this.hybridWallGroups[i];
        if (!seed || !seed.seeded || !wall || !this.wbtContext.mjData) continue;
        const p3 = wall.bodyId * 3;
        const m9 = wall.bodyId * 9;
        const xm = this.wbtContext.mjData.xmat;
        const off = this.hybridVizConfig.tcpOffsetX;
        const tcp = [
          (this.wbtContext.mjData.xpos[p3 + 0] ?? 0) + xm[m9 + 0] * off,
          (this.wbtContext.mjData.xpos[p3 + 1] ?? 0) + xm[m9 + 3] * off,
          (this.wbtContext.mjData.xpos[p3 + 2] ?? 0) + xm[m9 + 6] * off,
        ];
        const signed =
          (seed.pointW[0] - tcp[0]) * seed.normalW[0] +
          (seed.pointW[1] - tcp[1]) * seed.normalW[1] +
          (seed.pointW[2] - tcp[2]) * seed.normalW[2];
        if (signed > 0) inContact.push(i === 0 ? 'L' : 'R');
      }
    }
    // Badge reflects the LIVE toggle state (not the baked servo_enabled), so the
    // user sees exactly which layers are active after flipping a toggle:
    //   servo OFF          → grey "POSITION TRACKING (servo off)"
    //   servo ON, free      → grey "POSITION TRACKING (free space)"
    //   servo ON, in contact→ green "FORCE SERVO (L+R in contact)"
    // A "wall off" suffix is appended whenever the causal wall is disabled.
    const wallSuffix = this.hybridWallOn ? '' : '  · wall OFF';
    // C1 unified (observe-only): there IS no Jᵀ servo actuating, so "FORCE SERVO"
    // would be a lie and "servo off" would wrongly suggest Layer B was disabled
    // (the integrator + I/Δx obs are fully live — only Δq injection is absent). The
    // POLICY is the executor in both phases, so both read "UNIFIED POLICY"; the
    // free-space/in-contact distinction (from the same live wall penetration) is
    // what still matters to the operator.
    if (this.hybridVizConfig.servoObserveOnly) {
      if (inContact.length > 0) {
        el.textContent = `UNIFIED POLICY  (${inContact.join('+')} in contact)${wallSuffix}`;
        el.style.background = 'rgba(38, 166, 91, 0.92)'; // green
      } else {
        el.textContent = `UNIFIED POLICY  (free space)${wallSuffix}`;
        el.style.background = 'rgba(80, 88, 100, 0.9)'; // grey
      }
      return;
    }
    if (!this.hybridServoOn) {
      el.textContent = `POSITION TRACKING  (servo off — Layer A)${wallSuffix}`;
      el.style.background = 'rgba(80, 88, 100, 0.9)'; // grey
    } else if (inContact.length > 0) {
      el.textContent = `FORCE SERVO  (${inContact.join('+')} in contact)${wallSuffix}`;
      el.style.background = 'rgba(38, 166, 91, 0.92)'; // green
    } else {
      el.textContent = `POSITION TRACKING  (free space)${wallSuffix}`;
      el.style.background = 'rgba(80, 88, 100, 0.9)'; // grey
    }
  }

  /** Build the live Hybrid control panel once: two toggle checkboxes (Wall /
   *  Servo) that write wbt_hybrid_wall_on / wbt_hybrid_servo_on to the extras
   *  bus (the runtime reads them each substep), plus an arrow + layer legend.
   *  Positioned top-left under the slider panel. No-op for non-hybrid presets. */
  private ensureHybridControls(): void {
    if (!this.hybridVizConfig || typeof document === 'undefined') return;
    if (this.hybridControlsEl) return;
    // Reuse an existing panel from a prior WbtTrackingCommand instance (live preset
    // switch: the old term is disposed without DOM cleanup, so a stale
    // #wbt-hybrid-controls may linger). getElementById-guard, like mountRecalibButton
    // — prevents a duplicate-id panel with conflicting checkbox writers. On reuse we
    // re-sync the checkboxes to THIS instance's live toggle state + re-seed the bus.
    const existing = document.getElementById('wbt-hybrid-controls') as HTMLDivElement | null;
    // The two modes render DIFFERENT panels (observe-only has no Layer-B checkbox), so a
    // panel left over from the other mode cannot be reused — it would show a servo
    // checkbox on a unified run (or hide one on a servo run) with the listener still
    // bound to the disposed instance. Drop it and rebuild below on a mode mismatch;
    // reuse only within the SAME mode (the original stale-panel guard).
    const wantMode = this.hybridVizConfig.servoObserveOnly ? 'observe-only' : 'servo';
    if (existing && existing.dataset.wbtHybridMode !== wantMode) {
      existing.remove();
    } else if (existing) {
      this.hybridControlsEl = existing;
      // Query by explicit data-role, NOT by position: in observe-only mode there is no
      // servo checkbox, so positional indexing would bind the wrong input.
      const wallCb = existing.querySelector('input[data-wbt-role="wall"]') as HTMLInputElement | null;
      const servoCb = existing.querySelector('input[data-wbt-role="servo"]') as HTMLInputElement | null;
      const wallLiveCb = existing.querySelector('input[data-wbt-role="wall-live"]') as HTMLInputElement | null;
      const inertialCb = existing.querySelector('input[data-wbt-role="fmeas-inertial"]') as HTMLInputElement | null;
      if (wallCb) wallCb.checked = this.hybridWallOn;
      if (servoCb) servoCb.checked = this.hybridServoOn;
      if (wallLiveCb) wallLiveCb.checked = this.hybridWallLive;
      if (inertialCb) inertialCb.checked = this.hybridFmeasInertial;
      setPolicyExtra('wbt_hybrid_wall_on', this.hybridWallOn ? 1.0 : 0.0);
      setPolicyExtra('wbt_hybrid_servo_on', this.hybridServoOn ? 1.0 : 0.0);
      setPolicyExtra('wbt_hybrid_wall_live', this.hybridWallLive ? 1.0 : 0.0);
      setPolicyExtra('wbt_hybrid_fmeas_inertial', this.hybridFmeasInertial ? 1.0 : 0.0);
      return;
    }
    const root = document.createElement('div');
    root.id = 'wbt-hybrid-controls';
    root.dataset.wbtHybridMode = wantMode;
    Object.assign(root.style, {
      position: 'fixed',
      left: '12px',
      bottom: '12px',
      zIndex: '10001',
      padding: '10px 14px',
      background: 'rgba(20, 24, 32, 0.88)',
      color: '#f0f0f0',
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: '12px',
      lineHeight: '1.5',
      borderRadius: '6px',
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
      minWidth: '230px',
    } as Partial<CSSStyleDeclaration>);

    const title = document.createElement('div');
    title.textContent = 'Hybrid controls';
    Object.assign(title.style, {
      fontWeight: 'bold',
      marginBottom: '6px',
      borderBottom: '1px solid rgba(255,255,255,0.2)',
      paddingBottom: '4px',
    } as Partial<CSSStyleDeclaration>);
    root.appendChild(title);

    // A labeled checkbox row wired to a setPolicyExtra key + a local mirror.
    // `role` is stamped as data-wbt-role on the <input> so the stale-panel reuse path
    // can find each checkbox BY NAME (observe-only mode omits the servo box, which
    // would make positional lookup bind the wrong input).
    const addToggle = (
      role: string,
      label: string,
      hint: string,
      initial: boolean,
      onChange: (v: boolean) => void,
    ): void => {
      const row = document.createElement('label');
      Object.assign(row.style, {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        cursor: 'pointer',
        margin: '3px 0',
      } as Partial<CSSStyleDeclaration>);
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = initial;
      cb.style.cursor = 'pointer';
      cb.dataset.wbtRole = role;
      const txt = document.createElement('span');
      txt.textContent = label;
      txt.style.fontWeight = 'bold';
      const hintEl = document.createElement('span');
      hintEl.textContent = hint;
      hintEl.style.color = '#9aa4b2';
      cb.addEventListener('change', () => onChange(cb.checked));
      row.appendChild(cb);
      row.appendChild(txt);
      row.appendChild(hintEl);
      root.appendChild(row);
    };

    addToggle('wall', 'Spring wall (Layer A)', '', this.hybridWallOn, (v) => {
      this.hybridWallOn = v;
      setPolicyExtra('wbt_hybrid_wall_on', v ? 1.0 : 0.0);
    });
    // Live wall normal (diagnostic). Default OFF = training-faithful: hybrid.py fixes
    // F_cmd's WORLD direction for the whole episode, so a slider change mid-episode
    // does NOT turn the wall. Turn this ON to sweep F^cmd across axes (Fx→Fy → a 45°
    // wall) without needing a reset — Reset zeroes the sliders first, so the
    // "set both, then Reset" route cannot express a multi-axis command.
    addToggle('wall-live', 'Live wall normal', 'follows F^cmd (not training-faithful)', this.hybridWallLive, (v) => {
      this.hybridWallLive = v;
      setPolicyExtra('wbt_hybrid_wall_live', v ? 1.0 : 0.0);
    });
    // F^meas inertial term. Default ON = the training-time sensor model (include_inertial).
    // The m·a_wrist term is what makes F^meas flicker in free space (double finite
    // difference of the TCP position at 5 ms — there is no injected noise to switch off).
    // OFF = F^meas is the wall reaction + F^ext only; the obs/contact_phase/e follow too.
    addToggle('fmeas-inertial', 'F meas inertial term', 'm·a (FD flicker); off = wall + F^ext only', this.hybridFmeasInertial, (v) => {
      this.hybridFmeasInertial = v;
      setPolicyExtra('wbt_hybrid_fmeas_inertial', v ? 1.0 : 0.0);
    });
    if (this.hybridVizConfig.servoObserveOnly) {
      // C1 unified: the runtime IGNORES wbt_hybrid_servo_on in observe-only mode (see
      // runtime.ts executeSimulationSteps) — there is no Δq injection to gate, and
      // routing to zeroHybridServoState would zero the I/Δx the policy's obs NEED.
      // Render a read-only notice instead of a checkbox that silently does nothing.
      const note = document.createElement('div');
      Object.assign(note.style, { margin: '3px 0', lineHeight: '1.5' } as Partial<CSSStyleDeclaration>);
      note.innerHTML =
        '<span style="font-weight:bold">J<sup>T</sup> servo (Layer B): observe-only</span>' +
        '<div style="color:#9aa4b2;font-size:11px">integrator + I/Δx live (reference), no Δq — the POLICY executes</div>';
      root.appendChild(note);
    } else {
      addToggle('servo', 'Force servo (Layer B)', '', this.hybridServoOn, (v) => {
        this.hybridServoOn = v;
        setPolicyExtra('wbt_hybrid_servo_on', v ? 1.0 : 0.0);
      });
    }

    // Arrow + layer legend.
    const legend = document.createElement('div');
    Object.assign(legend.style, {
      marginTop: '6px',
      paddingTop: '6px',
      borderTop: '1px solid rgba(255,255,255,0.15)',
      fontSize: '11px',
      lineHeight: '1.6',
    } as Partial<CSSStyleDeclaration>);
    const swatch = (rgb: string, text: string): string =>
      `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${rgb};margin-right:6px;vertical-align:middle"></span>${text}`;
    legend.innerHTML = [
      swatch('rgb(158,51,235)', 'F<sup>cmd</sup> — commanded force (intent / servo target)'),
      swatch('rgb(255,115,0)', 'F<sup>ext</sup> — manual disturbance (slider)'),
      swatch('rgb(0,191,255)', 'F<sup>meas</sup> — felt force (= wall reaction + F<sup>ext</sup> [+ m·a if on])'),
      '<div style="margin-top:4px;color:#9aa4b2">Layer A = causal spring wall (both runs)</div>',
      this.hybridVizConfig.servoObserveOnly
        ? '<div style="color:#9aa4b2">Layer B = J<sup>T</sup> servo as REFERENCE only (no Δq)</div>'
        : '<div style="color:#9aa4b2">Layer B = J<sup>T</sup> force servo (adds Δq)</div>',
      '<div style="margin-top:4px;color:#cfd6df">Tip: hold an F<sup>cmd</sup>, watch F meas → −F<sup>cmd</sup> and e→0</div>',
    ].join('');
    root.appendChild(legend);

    document.body.appendChild(root);
    this.hybridControlsEl = root;
  }

  /** Lazy lookup of the pelvis body id. Caches the result; -1 means
   *  "not found" (returns identity quaternion downstream — arrows still
   *  display, just in world frame instead of root frame). */
  private findRootBodyId(): number {
    if (this.cachedRootBodyId === -2) {
      const mjModel = this.wbtContext.mjModel;
      this.cachedRootBodyId = -1;
      if (mjModel) {
        for (let b = 0; b < mjModel.nbody; b++) {
          const name = mjModel.body(b).name;
          if (name === 'pelvis' || name.endsWith('/pelvis')) {
            this.cachedRootBodyId = b;
            break;
          }
        }
      }
    }
    return this.cachedRootBodyId;
  }
  private cachedRootBodyId = -2;

  // ──────────────────────────────────────────────────────────────────────
  // Live per-hand force-tracking error (raw drift + Eq.2 residual)
  // ──────────────────────────────────────────────────────────────────────

  // Rubber-hand offsets (wrist_yaw local, MuJoCo z-up), [left, right]. Mirror
  // eval_utils.FORCE_HAND_OFFSETS_B exactly — the per-hand y sign differs.
  // yam gripper: wrist_yaw → ft_sensor (0.072) → rubber_hand (0.0095) = 0.0815.
  private static readonly FORCE_OFFSETS: ReadonlyArray<readonly [number, number, number]> = [
    [0.0815, 0.003, 0.0],
    [0.0815, -0.003, 0.0],
  ];
  private static readonly FORCE_HAND_BODIES = ['left_wrist_yaw_link', 'right_wrist_yaw_link'];
  // F^cmd slider keys (raw N, pelvis frame). MUST match WbtForceCmdLocal.KEYS
  // and main.py _FORCE_SLIDER_KEYS_*. Read WITHOUT the 0.04 obs scale here.
  private static readonly FORCE_CMD_KEYS: ReadonlyArray<readonly [string, string, string]> = [
    ['wbt_force_cmd_local__l_fx', 'wbt_force_cmd_local__l_fy', 'wbt_force_cmd_local__l_fz'],
    ['wbt_force_cmd_local__r_fx', 'wbt_force_cmd_local__r_fy', 'wbt_force_cmd_local__r_fz'],
  ];
  private static readonly FORCE_K = 200.0; // FORCE_STIFFNESS_K (N/m)
  private forceHandBodyIds: [number, number] = [-2, -2];

  private resolveForceBodyId(h: number): number {
    if (this.forceHandBodyIds[h] === -2) {
      const mjModel = this.wbtContext.mjModel;
      this.forceHandBodyIds[h] = -1;
      const want = WbtTrackingCommand.FORCE_HAND_BODIES[h];
      if (mjModel) {
        for (let b = 0; b < mjModel.nbody; b++) {
          const name = mjModel.body(b).name;
          if (name === want || name.endsWith(`/${want}`)) {
            this.forceHandBodyIds[h] = b;
            break;
          }
        }
      }
    }
    return this.forceHandBodyIds[h];
  }

  /**
   * Per-hand force-tracking error for the live panel, or null if unavailable
   * (force-less / synthetic clip, or wrist body not found).
   * Returns [rawL_cm, rawR_cm, eq2L_cm, eq2R_cm].
   *
   * Mirrors eval_utils._log_force_frame exactly (all pelvis-frame, z-up):
   *   pol_hand_w   = xpos[wrist] + R(wrist) @ offset_b
   *   pol_hand_rel = R_root^T @ (pol_hand_w - root_p)          // own pelvis frame
   *   disp         = pol_hand_rel - ref_hand_rel(baked)        // ref already pelvis-frame
   *   raw  = |disp|                  (drift-immune position error)
   *   eq2  = |disp - F_cmd/K|        (force-target residual; ×K = |F_cmd - F_hat|)
   * F_cmd is raw Newtons from the SAME sliders the obs uses (NO 0.04 scale).
   */
  getForceHandErrCm(): [number, number, number, number] | null {
    const mjData = this.wbtContext.mjData;
    if (!mjData || !this.activeBaked || !this.activeBaked.forceHandRefLocal) return null;
    const ref = this.getForceHandRefLocal(); // 6D [L_xyz, R_xyz], pelvis frame
    const q = mjData.qpos;
    // Live policy pelvis world pose (native MuJoCo z-up, wxyz quat).
    const rootP: [number, number, number] = [q[0] ?? 0, q[1] ?? 0, q[2] ?? 0];
    const rootQ: [number, number, number, number] = [q[3] ?? 1, q[4] ?? 0, q[5] ?? 0, q[6] ?? 0];
    const raw: [number, number] = [NaN, NaN];
    const eq2: [number, number] = [NaN, NaN];
    const K = WbtTrackingCommand.FORCE_K;
    for (let h = 0; h < 2; h++) {
      const bId = this.resolveForceBodyId(h);
      if (bId < 0) continue;
      const off = WbtTrackingCommand.FORCE_OFFSETS[h];
      const p3 = bId * 3;
      const m9 = bId * 9;
      const xm = mjData.xmat;
      // pol_hand_w = xpos + R_wrist @ offset_b (row-major xmat).
      const hwx = (mjData.xpos[p3] ?? 0) + xm[m9] * off[0] + xm[m9 + 1] * off[1] + xm[m9 + 2] * off[2];
      const hwy = (mjData.xpos[p3 + 1] ?? 0) + xm[m9 + 3] * off[0] + xm[m9 + 4] * off[1] + xm[m9 + 5] * off[2];
      const hwz = (mjData.xpos[p3 + 2] ?? 0) + xm[m9 + 6] * off[0] + xm[m9 + 7] * off[1] + xm[m9 + 8] * off[2];
      // pol_hand_rel = R_root^T @ (hand_w - root_p) == rotate-inverse(rootQ, ·).
      const rel = wbtRotateInverseLocal(rootQ, hwx - rootP[0], hwy - rootP[1], hwz - rootP[2]);
      // disp = pol_hand_rel - ref_hand_rel (ref already in ref-pelvis frame).
      const ex = rel[0] - (ref[h * 3] ?? 0);
      const ey = rel[1] - (ref[h * 3 + 1] ?? 0);
      const ez = rel[2] - (ref[h * 3 + 2] ?? 0);
      raw[h] = Math.hypot(ex, ey, ez) * 100.0;
      // Eq.2: subtract the intended F_cmd/K displacement (raw N, pelvis frame).
      const k = WbtTrackingCommand.FORCE_CMD_KEYS[h];
      const fx = getPolicyExtra(k[0], 0.0);
      const fy = getPolicyExtra(k[1], 0.0);
      const fz = getPolicyExtra(k[2], 0.0);
      eq2[h] = Math.hypot(ex - fx / K, ey - fy / K, ez - fz / K) * 100.0;
    }
    return [raw[0], raw[1], eq2[0], eq2[1]];
  }
}

// Helpers that swizzle MuJoCo z-up <-> THREE y-up. Equivalent to
// third_party/mjswan/src/mjswan/template/src/core/scene/scene.ts:getPosition
// and :getQuaternion (the swizzle=true branches), inlined here because the
// mjswan auto-gen pipeline doesn't resolve cross-file relative imports
// from core/command/.
function mjPosToThree(buf: Float32Array | Float64Array, idx: number): THREE.Vector3 {
  const i = idx * 3;
  // (mx, my, mz)  ->  (mx, mz, -my)
  return new THREE.Vector3(buf[i] ?? 0, buf[i + 2] ?? 0, -(buf[i + 1] ?? 0));
}
function mjQuatToThree(buf: Float32Array | Float64Array, idx: number): THREE.Quaternion {
  const i = idx * 4;
  // mjswan's getQuaternion swizzle=true:
  //   THREE xyzw = ( -mqx, -mqz,  mqy, -mqw )
  return new THREE.Quaternion(
    -(buf[i + 1] ?? 0),
    -(buf[i + 3] ?? 0),
    buf[i + 2] ?? 0,
    -(buf[i] ?? 1),
  );
}

// ── Force-arrow geometry (matches the benchmark force-arrow videos) ──────────
// eval_utils: FORCE_ARROW_M_PER_N=0.018 (10 N -> 18 cm), MIN_LEN=0.08 m floor,
// uncapped (25 N -> 45 cm). WIDTH≈0.018 m shaft radius.
const WBT_FORCE_ARROW_M_PER_N = 0.018;
const WBT_FORCE_ARROW_MIN_LEN = 0.08;

type WbtForceArrow = {
  group: THREE.Group; // reoriented per frame; holds shaft + head
  shaft: THREE.Mesh; // +Y unit cylinder, scaled on Y per frame
  head: THREE.Mesh; // +Y cone, repositioned to the shaft tip per frame
  mat: THREE.MeshBasicMaterial;
};

// Build one force arrow: a +Y unit shaft (base at origin) + cone head wrapped
// in a group, with a FIXED color (no magnitude lerp). The group is oriented +
// the shaft is scaled on Y per frame; the head is repositioned to the tip.
// `shaftRadius` differentiates overlapping arrows: the orange F^ext arrow is
// drawn thicker so it reads as a halo AROUND the thinner purple F^cmd core in
// the Eval-3 cancel case (F^ext = -F^cmd), instead of z-fighting into one line.
// Both arrows share the exact same anchor (the rubber-hand point), matching the
// benchmark videos — no positional offset.
function makeForceArrow(
  colorRgb: [number, number, number],
  renderOrder: number,
  shaftRadius: number,
): WbtForceArrow {
  const mat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(colorRgb[0], colorRgb[1], colorRgb[2]),
    depthTest: false,
    transparent: true,
    opacity: 0.95,
  });
  const shaftGeom = new THREE.CylinderGeometry(shaftRadius, shaftRadius, 1.0, 12);
  shaftGeom.translate(0, 0.5, 0); // base at origin, extends +Y; scaled on Y per frame
  const shaft = new THREE.Mesh(shaftGeom, mat);
  shaft.renderOrder = renderOrder;
  const headGeom = new THREE.ConeGeometry(shaftRadius * 2.4, 0.045, 12);
  headGeom.translate(0, 0.0225, 0); // base at the head's local origin
  const head = new THREE.Mesh(headGeom, mat);
  head.renderOrder = renderOrder;
  const group = new THREE.Group();
  group.name = 'wbt_force_arrow';
  group.add(shaft);
  group.add(head);
  return { group, shaft, head, mat };
}

// Inlined rotate-inverse helper. Duplicated from WbtObservations.ts because
// the auto-generation pipeline concatenates ts_src files line-by-line and
// cross-file relative imports do not resolve in core/command/. Keeping this
// self-contained keeps the diff small.
function wbtRotateInverseLocal(qWxyz: ArrayLike<number>, vx: number, vy: number, vz: number): Float32Array {
  const w = qWxyz[0] ?? 1.0;
  const x = qWxyz[1] ?? 0.0;
  const y = qWxyz[2] ?? 0.0;
  const z = qWxyz[3] ?? 0.0;
  // Inverse-rotate v by q  ==  rotate v by q^{-1}.
  // Computed via standard q^{-1} * (0,v) * q, then dropping the scalar part.
  const qiw = w, qix = -x, qiy = -y, qiz = -z;
  // r1 = qi * (0,v)
  const r1w = -qix * vx - qiy * vy - qiz * vz;
  const r1x = qiw * vx + qiy * vz - qiz * vy;
  const r1y = qiw * vy - qix * vz + qiz * vx;
  const r1z = qiw * vz + qix * vy - qiy * vx;
  // r = r1 * q
  const rx = r1w * x + r1x * w + r1y * z - r1z * y;
  const ry = r1w * y - r1x * z + r1y * w + r1z * x;
  const rz = r1w * z + r1x * y - r1y * x + r1z * w;
  return new Float32Array([rx, ry, rz]);
}

// Forward-rotate v by q (wxyz): v_world = R(q) @ v. Used by the hybrid wall to
// rotate the pelvis-frame F^cmd slider into world (mirrors the runtime's
// hybridRootMat @ f_pelvis). Native MuJoCo z-up — NO THREE swizzle here (the
// caller swizzles the resulting world vector for rendering).
// Hamilton product, wxyz, a ⊗ b — "rotate by b, then by a". Same convention and same
// element order as `mc_teleop_source.quat_mul_wxyz`, which is what produces the operands
// this composes (the MC ghost's root: anchor ⊗ readout-relative).
function wbtQuatMulWxyz(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const aw = a[0] ?? 1; const ax = a[1] ?? 0; const ay = a[2] ?? 0; const az = a[3] ?? 0;
  const bw = b[0] ?? 1; const bx = b[1] ?? 0; const by = b[2] ?? 0; const bz = b[3] ?? 0;
  const out = new Float32Array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ]);
  const n = Math.hypot(out[0], out[1], out[2], out[3]) || 1.0;
  for (let i = 0; i < 4; i++) out[i] /= n;
  return out;
}
function wbtQuatApplyWxyz(qWxyz: ArrayLike<number>, vx: number, vy: number, vz: number): Float32Array {
  const w = qWxyz[0] ?? 1.0;
  const x = qWxyz[1] ?? 0.0;
  const y = qWxyz[2] ?? 0.0;
  const z = qWxyz[3] ?? 0.0;
  // r1 = q * (0,v)
  const r1w = -x * vx - y * vy - z * vz;
  const r1x = w * vx + y * vz - z * vy;
  const r1y = w * vy - x * vz + z * vx;
  const r1z = w * vz + x * vy - y * vx;
  // r = r1 * q^{-1}
  const qiw = w, qix = -x, qiy = -y, qiz = -z;
  const rx = r1w * qix + r1x * qiw + r1y * qiz - r1z * qiy;
  const ry = r1w * qiy - r1x * qiz + r1y * qiw + r1z * qix;
  const rz = r1w * qiz + r1x * qiy - r1y * qix + r1z * qiw;
  return new Float32Array([rx, ry, rz]);
}
