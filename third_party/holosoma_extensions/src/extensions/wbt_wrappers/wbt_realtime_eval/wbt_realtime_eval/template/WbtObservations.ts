// IMPORTANT: each `import ... from '...';` MUST be a single line.
// mjswan's _build_client.py concatenates all ts_src files into
// core/observation/custom_observations.ts using line-by-line dedup; multi-line
// import blocks get sliced into "import {" and a stray closing brace.
import { ObservationBase } from '@/core/observation/ObservationBase';
import type { ObservationConfig } from '@/core/observation/ObservationBase';
import { getCommandManager } from '@/core/command';
import type { PolicyState } from '@/core/policy/types';
import type { PolicyRunner } from '@/core/policy/PolicyRunner';
import type { CommandTerm } from '@/core/command/types';
import { getPolicyExtra } from '@/core/policy/extras';
import { setPolicyExtra } from '@/core/policy/extras';

// We cannot import WbtTrackingCommand here — mjswan inlines this file under
// core/observation/, but WbtTrackingCommand lives under core/command/, so a
// `./WbtTrackingCommand` import would fail to resolve. Use a structural
// (duck-typed) check instead.
type WbtTrackingLike = CommandTerm & {
  isReady(): boolean;
  refIdx: number;
  refLen: number;
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  getSelectedMotionName(): string | null;
  // 2pt baked tables (left/right wrist).
  getVr2PosLocal(): Float32Array;
  getVr2OrnLocal(): Float32Array;
  // 3pt baked tables (left wrist + right wrist + head).
  getVr3PosLocal(): Float32Array;
  // TCP-unified 3pt positions (hands on the 0.242 m force TCP) — the vr3_pos_tcp_local
  // table, read by WbtVr3PointPosTcp. MUST be declared here: WbtTrackingLike intersects
  // the STRICT CommandTerm interface (no index signature), so an undeclared method is a
  // hard `tsc` error (TS2551) and would break the bundle build for the TCP-unified presets.
  getVr3PosTcpLocal(): Float32Array;
  getVr3OrnLocal(): Float32Array;
  getVr3Orn6dLocal(): Float32Array;
  // 5-point baked tables (the 3pt-TCP set + both feet) — vr5_pos_tcp_local (15D) and
  // vr5_orn_local (20D), read by WbtVr5PointPosTcp / WbtVr5PointOrn. Declared here for
  // the same hard reason as getVr3PosTcpLocal above: WbtTrackingLike intersects the
  // STRICT CommandTerm interface, so an undeclared method is a `tsc` error (TS2551).
  getVr5PosTcpLocal(): Float32Array;
  getVr5OrnLocal(): Float32Array;
  getLowerBodyJp(): Float32Array;
  getLowerBodyJv(): Float32Array;
  getRefDofPos(): Float32Array;
  getRefDofVel(): Float32Array;
  getMotionRefPosB(rootPosW: ArrayLike<number>, rootQuatWxyz: ArrayLike<number>): Float32Array;
  getMotionRefVelB(rootQuatWxyz: ArrayLike<number>): Float32Array;
};

function getWbtTracking(): WbtTrackingLike | null {
  const term = getCommandManager().getTerm('motion');
  if (!term) return null;
  const candidate = term as Partial<WbtTrackingLike>;
  if (
    typeof candidate.isReady === 'function' &&
    typeof candidate.getLowerBodyJp === 'function' &&
    typeof candidate.getLowerBodyJv === 'function' &&
    typeof candidate.getRefDofPos === 'function' &&
    typeof candidate.getRefDofVel === 'function'
  ) {
    return term as WbtTrackingLike;
  }
  return null;
}

// Quaternion math helpers, inlined to avoid colliding with mjswan's own
// core/observation/math.ts (different exports). All quats are wxyz; MUST
// match holosoma_inference.utils.math.quat byte-for-byte.
function wbtQuatNormalize(q: ArrayLike<number>): Float32Array {
  const w = q[0] ?? 1.0;
  const x = q[1] ?? 0.0;
  const y = q[2] ?? 0.0;
  const z = q[3] ?? 0.0;
  const n = Math.hypot(w, x, y, z) || 1.0;
  return new Float32Array([w / n, x / n, y / n, z / n]);
}

function wbtQuatInverse(q: ArrayLike<number>): Float32Array {
  return new Float32Array([q[0] ?? 1.0, -(q[1] ?? 0.0), -(q[2] ?? 0.0), -(q[3] ?? 0.0)]);
}

function wbtQuatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const aw = a[0] ?? 1.0, ax = a[1] ?? 0.0, ay = a[2] ?? 0.0, az = a[3] ?? 0.0;
  const bw = b[0] ?? 1.0, bx = b[1] ?? 0.0, by = b[2] ?? 0.0, bz = b[3] ?? 0.0;
  return new Float32Array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ]);
}

function wbtQuatYaw(q: ArrayLike<number>): number {
  const w = q[0] ?? 1.0;
  const x = q[1] ?? 0.0;
  const y = q[2] ?? 0.0;
  const z = q[3] ?? 0.0;
  return Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

function wbtStripYaw(q: ArrayLike<number>, yaw: number): Float32Array {
  const c = Math.cos(yaw / 2);
  const s = Math.sin(yaw / 2);
  // q' = q_yaw_inv * q  where q_yaw = [c, 0, 0, s], so q_yaw_inv = [c, 0, 0, -s]
  return wbtQuatMultiply([c, 0.0, 0.0, -s], q);
}

function wbtSubtractFrameOri(qA: ArrayLike<number>, qB: ArrayLike<number>): Float32Array {
  return wbtQuatMultiply(wbtQuatInverse(wbtQuatNormalize(qA)), wbtQuatNormalize(qB));
}

// Row-major flatten of the first two columns of the rotation matrix.
// Matches numpy `matrix_from_quat(q)[..., :2].reshape(-1)` (C-order on (3,2)).
function wbtQuatToRot6d(q: ArrayLike<number>): Float32Array {
  const [w, x, y, z] = wbtQuatNormalize(q);
  const r00 = 1.0 - 2.0 * (y * y + z * z);
  const r01 = 2.0 * (x * y - w * z);
  const r10 = 2.0 * (x * y + w * z);
  const r11 = 1.0 - 2.0 * (x * x + z * z);
  const r20 = 2.0 * (x * z - w * y);
  const r21 = 2.0 * (y * z + w * x);
  return new Float32Array([r00, r01, r10, r11, r20, r21]);
}

function wbtQuatRotateInverse(q: ArrayLike<number>, v: ArrayLike<number>): Float32Array {
  const qn = wbtQuatNormalize(q);
  const qi = wbtQuatInverse(qn);
  const vq = new Float32Array([0.0, v[0] ?? 0.0, v[1] ?? 0.0, v[2] ?? 0.0]);
  const r = wbtQuatMultiply(wbtQuatMultiply(qi, vq), qn);
  return new Float32Array([r[1], r[2], r[3]]);
}

// Forward rotation: v_world = R(q) @ v. Inverse of wbtQuatRotateInverse; used by
// the hand-frame F^cmd path (rotate pelvis-frame Newtons to world, then world to
// the wrist frame). Mirrors holosoma_inference.utils.math.quat_apply.
function wbtQuatRotate(q: ArrayLike<number>, v: ArrayLike<number>): Float32Array {
  const qn = wbtQuatNormalize(q);
  const qi = wbtQuatInverse(qn);
  const vq = new Float32Array([0.0, v[0] ?? 0.0, v[1] ?? 0.0, v[2] ?? 0.0]);
  const r = wbtQuatMultiply(wbtQuatMultiply(qn, vq), qi);
  return new Float32Array([r[1], r[2], r[3]]);
}

// Per-term history length for the holosoma_2pt_wbt preset. The training-time
// observation manager stacks 4 frames per term as `[t-3, t-2, t-1, t]` (oldest
// first); see HolosomaWBTPolicy._build_observation in
// src/extensions/wbt_wrappers/wbt_wrappers_inference/policies/holosoma_wbt.py.
const WBT_HISTORY = 4;

// Common base for all WBT 2pt obs terms. Each subclass implements
// `computeCurrent(state)` returning the current single-frame value (size =
// `baseDim`); this base maintains the per-term ring buffer of length WBT_HISTORY
// and exposes the stacked output `[t-3, t-2, t-1, t]` (size = baseDim*WBT_HISTORY).
abstract class WbtHistoryBase extends ObservationBase {
  protected readonly baseDim: number;
  protected readonly history: Float32Array[];
  // Number of times update() has been called. Used to pad the history with the
  // first observed value before WBT_HISTORY frames have arrived (matches the
  // Python runtime which prepends zeros / first-value).
  private filled: number;

  constructor(runner: PolicyRunner, config: ObservationConfig, baseDim: number) {
    super(runner, config);
    this.baseDim = baseDim;
    this.history = Array.from({ length: WBT_HISTORY }, () => new Float32Array(baseDim));
    this.filled = 0;
  }

  get size(): number {
    return this.baseDim * WBT_HISTORY;
  }

  override reset(state?: PolicyState): void {
    // Seed all 4 slots with the current observation so that the first stacked
    // output isn't dominated by zeros (matches the runtime's reset behavior).
    const current = this.computeCurrent(state);
    for (const slot of this.history) slot.set(current);
    this.filled = WBT_HISTORY;
  }

  override update(state: PolicyState): void {
    // Shift older frames toward higher indices, write new at index 0 (newest).
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
    if (this.filled < WBT_HISTORY) this.filled += 1;
  }

  compute(_state: PolicyState): Float32Array {
    const out = new Float32Array(this.size);
    // Layout: [t-3, t-2, t-1, t] — oldest first, newest last. Matches
    // numpy `np.concatenate(history[-4:])` where history is an
    // append-only deque in chronological order.
    for (let k = 0; k < WBT_HISTORY; k++) {
      // history[i] holds the (i+1)-th most recent frame: history[0]=t,
      // history[1]=t-1, ..., history[3]=t-3. We want output[k=0]=t-3
      // ... output[k=3]=t. So index = WBT_HISTORY - 1 - k.
      const srcIdx = WBT_HISTORY - 1 - k;
      out.set(this.history[srcIdx], k * this.baseDim);
    }
    return out;
  }

  protected abstract computeCurrent(state?: PolicyState): Float32Array;
}

export class WbtCommandLowerBody extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 24);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    const out = new Float32Array(24);
    if (!tr) return out;
    out.set(tr.getLowerBodyJp(), 0);
    out.set(tr.getLowerBodyJv(), 12);
    return out;
  }
}

export class WbtVr2PointPos extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(6);
    return new Float32Array(tr.getVr2PosLocal());
  }
}

// 6D rotation rep per body (12D = 2 wrists x 6) — training feeds the rot6d
// (first two columns of the rotation matrix), NOT a quaternion, for the 2pt VR
// orientation term, to avoid the antipodal/double-cover OOD bug at deploy. The
// bake (bake_motions.py) already stores rot6d, so we forward it verbatim — no
// quat math needed here (avoids the xyzw/wxyz pitfall).
export class WbtVr2PointOrn extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 12);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(12);
    return new Float32Array(tr.getVr2OrnLocal());
  }
}

export class WbtVr3PointPos extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 9);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(9);
    return new Float32Array(tr.getVr3PosLocal());
  }
}

// TCP-UNIFIED 3pt position (9D) for exp:g1-wbt-3pt-hybrid-unified[-anneal] (user
// 2026-08-03). Identical layout to WbtVr3PointPos, but the bake put both HAND points on
// the 0.242 m fingertip TCP (the force anchor) instead of the 0.18 m VR-tracker lever,
// so position and force are supervised at ONE point. Head point unchanged. Separate
// class + separate bake table (vr3_pos_tcp_local) so exp:g1-wbt-3pt-hybrid and the
// global-vel twins keep the 0.18 m table byte-identically.
export class WbtVr3PointPosTcp extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 9);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(9);
    return new Float32Array(tr.getVr3PosTcpLocal());
  }
}

// 3pt orientation is rot6d (3 bodies × 6) — wbt_wrappers_inference's
// holosoma_3pt_wbt observation uses obs_dim=12, which is rot6d-per-wrist (×2)
// + an extra 0 padding? Actually it's 3 bodies × 4 (xyzw)... but obs_dim is
// 12, so 3 × 4 xyzw. The bake stores xyzw triplets contiguously to match.
export class WbtVr3PointOrn extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 12);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(12);
    return new Float32Array(tr.getVr3OrnLocal());
  }
}

// 3pt orientation as the 6D continuous rotation rep (3 bodies × 6 = 18D) — the 6D-orn
// variant used by exp:g1-wbt-3pt-hybrid-global-vel (obs term `vr_3point_orn_6d`). Same
// rationale as the 2pt 6D rep: no quaternion double-cover / antipodal OOD at deploy. The
// bake (bake_motions.py) stores rot6d in vr3_orn6d_local; forward it verbatim. Distinct
// from WbtVr3PointOrn (12D quat) so exp:g1-wbt-3pt-hybrid is untouched.
export class WbtVr3PointOrn6d extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 18);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(18);
    return new Float32Array(tr.getVr3Orn6dLocal());
  }
}

// 5-POINT position (15D = 5 bodies × 3) for exp:g1-wbt-5pt-hybrid-unified-anneal (user
// 2026-08-27). The 3pt-TCP vector (both hands on the 0.242 m fingertip TCP, head at
// torso_link +0.35 z) with BOTH FEET appended at the bare *_ankle_roll_link origin, so
// slots [0:9] are byte-identically WbtVr3PointPosTcp's output and [9:15] are left foot
// then right foot. Distinct class + distinct bake table (vr5_pos_tcp_local) so the 3pt
// presets keep their 9D table untouched; the width difference also makes a mix-up a hard
// error rather than a silent mis-slice.
export class WbtVr5PointPosTcp extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 15);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(15);
    return new Float32Array(tr.getVr5PosTcpLocal());
  }
}

// 5-POINT orientation (20D = 5 bodies × 4, xyzw QUATERNION — the pelvis-local unified
// family's rep, matching WbtVr3PointOrn; the 6D rep lives only in the -global-vel
// family). Slots [0:12] are WbtVr3PointOrn's output unchanged. Offsets do not affect
// orientation (a rigid offset does not rotate a body), so the feet contribute their link
// frames directly.
export class WbtVr5PointOrn extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 20);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    if (!tr) return new Float32Array(20);
    return new Float32Array(tr.getVr5OrnLocal());
  }
}

// 58D: ref_dof_pos(29) || ref_dof_vel(29). Used by holosoma-wbt and the
// global variants. The bake stores both as 29-vectors per frame.
export class WbtCommand extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 58);
  }
  protected computeCurrent(): Float32Array {
    const tr = getWbtTracking();
    const out = new Float32Array(58);
    if (!tr) return out;
    out.set(tr.getRefDofPos(), 0);
    out.set(tr.getRefDofVel(), 29);
    return out;
  }
}

// (motion_ref_pos_w - robot_root_pos_w) rotated by the robot root inverse
// quaternion — i.e. the motion reference body position expressed in the
// robot pelvis frame. Mirrors HolosomaWBTPolicy at
// src/extensions/wbt_wrappers/wbt_wrappers_inference/policies/holosoma_wbt.py
// (the `motion_ref_pos_b` branch of `_compute_obs_terms`). For the
// pelvis-anchored presets we use the raw root_quat (no yaw strip).
export class WbtMotionRefPosB extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 3);
  }
  protected computeCurrent(state?: PolicyState): Float32Array {
    const tr = getWbtTracking();
    if (!tr || !tr.isReady() || tr.refLen === 0) return new Float32Array(3);
    const rootPosW = state?.rootPos ?? new Float32Array(3);
    const rootQuat = wbtQuatNormalize(state?.rootQuat ?? new Float32Array([1, 0, 0, 0]));
    return tr.getMotionRefPosB(rootPosW, rootQuat);
  }
}

// motion_ref_vel_w rotated into the robot root frame.
export class WbtMotionRefVelB extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 3);
  }
  protected computeCurrent(state?: PolicyState): Float32Array {
    const tr = getWbtTracking();
    if (!tr || !tr.isReady() || tr.refLen === 0) return new Float32Array(3);
    const rootQuat = wbtQuatNormalize(state?.rootQuat ?? new Float32Array([1, 0, 0, 0]));
    return tr.getMotionRefVelB(rootQuat);
  }
}

// Linear velocity of the floating root, expressed in the body frame (rotate
// the world-frame `state.rootLinVel` by `state.rootQuat`'s inverse).
export class WbtBaseLinVel extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 3);
  }
  protected computeCurrent(state?: PolicyState): Float32Array {
    const linVelW = state?.rootLinVel ?? new Float32Array(3);
    const rootQuat = state?.rootQuat ?? new Float32Array([1, 0, 0, 0]);
    return wbtQuatRotateInverse(rootQuat, linVelW);
  }
}

export class WbtMotionRefOriB extends WbtHistoryBase {
  private headingInitialized = false;
  private robotYawOffset = 0.0;
  private motionYawOffset = 0.0;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
  }

  override reset(state?: PolicyState): void {
    this.headingInitialized = false;
    this.robotYawOffset = 0.0;
    this.motionYawOffset = 0.0;
    super.reset(state);
  }

  protected computeCurrent(state?: PolicyState): Float32Array {
    const tr = getWbtTracking();
    if (!tr || !tr.isReady() || tr.refLen === 0) return new Float32Array(6);

    const rootQuat = wbtQuatNormalize(state?.rootQuat ?? new Float32Array([1, 0, 0, 0]));
    const motionQuat = wbtQuatNormalize(tr.refRootQuat[tr.refIdx]);

    if (!this.headingInitialized) {
      this.robotYawOffset = wbtQuatYaw(rootQuat);
      this.motionYawOffset = wbtQuatYaw(tr.refRootQuat[0]);
      this.headingInitialized = true;
    }

    const rootStripped = wbtStripYaw(rootQuat, this.robotYawOffset);
    const motionStripped = wbtStripYaw(motionQuat, this.motionYawOffset);
    const rel = wbtSubtractFrameOri(rootStripped, motionStripped);
    return wbtQuatToRot6d(rel);
  }
}

export class WbtProjectedGravity extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 3);
  }
  protected computeCurrent(state?: PolicyState): Float32Array {
    const q = state?.rootQuat ?? new Float32Array([1, 0, 0, 0]);
    return wbtQuatRotateInverse(q, [0, 0, -1]);
  }
}

export class WbtBaseAngVel extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 3);
  }
  protected computeCurrent(state?: PolicyState): Float32Array {
    return new Float32Array(state?.rootAngVel ?? new Float32Array(3));
  }
}

// Subtracts the ONNX-baked default joint angles (passed via config.params
// as `default_joint_angles`) from the live joint pos. Falls back to
// runner.getDefaultJointPos() if the param is absent — but for correctness
// against training, main.py MUST pass the ONNX-baked defaults via the
// ObservationTermCfg.params.
export class WbtJointPos extends WbtHistoryBase {
  private readonly defaultAngles: Float32Array;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 29);
    const fromConfig = (config as { default_joint_angles?: unknown }).default_joint_angles;
    if (Array.isArray(fromConfig) && fromConfig.length === 29) {
      this.defaultAngles = new Float32Array(fromConfig.map((v) => Number(v) || 0.0));
    } else {
      const fromRunner = runner.getDefaultJointPos();
      this.defaultAngles = new Float32Array(29);
      for (let i = 0; i < 29; i++) this.defaultAngles[i] = fromRunner[i] ?? 0.0;
    }
  }

  protected computeCurrent(state?: PolicyState): Float32Array {
    const q = state?.jointPos ?? new Float32Array(29);
    const out = new Float32Array(29);
    for (let i = 0; i < 29; i++) out[i] = (q[i] ?? 0.0) - (this.defaultAngles[i] ?? 0.0);
    return out;
  }
}

export class WbtJointVel extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 29);
  }
  protected computeCurrent(state?: PolicyState): Float32Array {
    return new Float32Array(state?.jointVel ?? new Float32Array(29));
  }
}

export class WbtPrevActions extends WbtHistoryBase {
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 29);
  }
  protected computeCurrent(): Float32Array {
    return new Float32Array(this.runner.getLastActions());
  }
}

// 6D UniFP-style F^cmd in robot root frame: [L_Fx, L_Fy, L_Fz, R_Fx, R_Fy, R_Fz].
// Sourced live from the slider panel via getPolicyExtra(...) — the matching
// extras keys are wbt_force_cmd_local__l_fx ... __r_fz, in raw Newtons.
//
// Training pipeline applies a 0.04 obs scale (= 1/f_cmd_max=25 N) to land in
// [-1, 1]. The matching inference-side ObservationConfig
// (`holosoma_2pt_force_wbt`) declares obs_scales['force_cmd_local']=0.04, but
// the browser runtime DOES NOT apply per-term obs scaling — values flow
// straight from compute() into the ONNX input. We therefore multiply by the
// scale here so the policy sees what it was trained to see.
//
// If the future need arises to expose the raw Newtons separately (e.g. for
// the debug arrow), read the same key elsewhere with getPolicyExtra(...).
//
// HAND-FRAME ABLATION (--F_on_hand_frame checkpoints): when the baked obs
// config carries `hand_frame: true` (+ `wrist_body_names`), the SAME pelvis-frame
// slider Newtons are rotated into each wrist (wrist_yaw) frame before scaling —
// matching what a hand-frame checkpoint was trained to see. The slider semantics
// and the purple F^cmd debug arrow stay PELVIS-frame (commanded intent); only the
// actor's OBSERVATION is rotated. Default (no hand_frame field) = the original
// pelvis path, byte-identical, so existing pelvis bundles are unaffected.
//
// Parity with training/inference: training feeds R(q_wrist_world)^-1 @
// R(q_pelvis_world) @ F_pelvis (force_tracking.py rotates the WORLD F^cmd buffer
// by the inverse wrist quat); inference (holosoma_wbt.py) feeds R_rel^-1 @
// F_pelvis with R_rel = wrist-in-pelvis from FK on the actual joints — equal
// because the world root orientation cancels (pelvis is the FK freeflyer root).
// In the browser we have BOTH world quats live, so we compute the world form
// directly: F_world = R(q_pelvis_world) @ F_pelvis ; F_hand = R(q_wrist_world)^-1
// @ F_world. Quats are native MuJoCo wxyz (qpos[3:7], mjData.xquat[bid*4:]); do
// NOT apply the THREE.js y-up swizzle (that belongs only to the render layer).
export class WbtForceCmdLocal extends WbtHistoryBase {
  // Matches HandForceConfig.f_cmd_max in
  // src/extensions/wbt_wrappers/wbt_training/config_values/force_tracking.py.
  // Inverted scale used to normalize raw Newtons into the [-1, 1] band the
  // policy was trained on.
  private static readonly OBS_SCALE = 0.04;
  // Slider keys MUST stay in sync with the keys main.py emits in
  // extras.sliders. Order matches the training observation flatten:
  // [left_x, left_y, left_z, right_x, right_y, right_z].
  private static readonly KEYS = [
    'wbt_force_cmd_local__l_fx',
    'wbt_force_cmd_local__l_fy',
    'wbt_force_cmd_local__l_fz',
    'wbt_force_cmd_local__r_fx',
    'wbt_force_cmd_local__r_fy',
    'wbt_force_cmd_local__r_fz',
  ];

  // Opt-in hand-frame mode (default false => pelvis path, unchanged).
  private readonly handFrame: boolean;
  // Wrist body names (left, right) whose live world orientation we rotate into.
  private readonly wristBodyNames: [string, string];
  // Lazily-resolved MuJoCo body ids (mjModel may not be ready at construction).
  // -2 = unresolved, -1 = not found, >=0 = resolved.
  private wristBodyIds: [number, number];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
    this.handFrame = (config as Record<string, unknown>).hand_frame === true;
    const wbn = (config as Record<string, unknown>).wrist_body_names;
    this.wristBodyNames =
      Array.isArray(wbn) && wbn.length === 2
        ? [String(wbn[0]), String(wbn[1])]
        : ['left_wrist_yaw_link', 'right_wrist_yaw_link'];
    this.wristBodyIds = [-2, -2];
  }

  protected computeCurrent(): Float32Array {
    // Raw pelvis-frame Newtons from the live sliders (same source either path).
    const fp = new Float32Array(6);
    for (let i = 0; i < 6; i++) fp[i] = getPolicyExtra(WbtForceCmdLocal.KEYS[i], 0.0);

    if (!this.handFrame) {
      // ---- PELVIS path (default, byte-identical to the original) ----
      const out = new Float32Array(6);
      for (let i = 0; i < 6; i++) out[i] = fp[i] * WbtForceCmdLocal.OBS_SCALE;
      return out;
    }

    // ---- HAND-FRAME path: rotate pelvis Newtons into each wrist frame ----
    // F^cmd = 0 rotates to 0 in any frame, so an all-zero slider state yields
    // exactly the pelvis (zero) obs — the F=0 invariant holds without any sim.
    const ctx = this.runner.getContext();
    const mjData = ctx?.mjData ?? null;
    const mjModel = ctx?.mjModel ?? null;
    if (!mjData || !mjModel) {
      // Sim not ready yet: fall back to the pelvis-scaled value (safe; only
      // affects the first frame(s) before physics is live).
      const out = new Float32Array(6);
      for (let i = 0; i < 6; i++) out[i] = fp[i] * WbtForceCmdLocal.OBS_SCALE;
      return out;
    }

    // Live pelvis (root) world quat from the free-joint qpos[3:7] (wxyz).
    const q = mjData.qpos;
    const qPelvis: [number, number, number, number] = [
      q[3] ?? 1.0,
      q[4] ?? 0.0,
      q[5] ?? 0.0,
      q[6] ?? 0.0,
    ];

    const out = new Float32Array(6);
    for (let hi = 0; hi < 2; hi++) {
      const fPelvis: [number, number, number] = [fp[hi * 3], fp[hi * 3 + 1], fp[hi * 3 + 2]];
      // Resolve the wrist body id once (mjModel may not have been ready at ctor).
      if (this.wristBodyIds[hi] === -2) {
        this.wristBodyIds[hi] = -1;
        const want = this.wristBodyNames[hi];
        for (let b = 0; b < mjModel.nbody; b++) {
          const name = mjModel.body(b).name;
          if (name === want || name.endsWith(`/${want}`)) {
            this.wristBodyIds[hi] = b;
            break;
          }
        }
        if (this.wristBodyIds[hi] < 0) {
          console.warn(`[WbtForceCmdLocal] hand-frame wrist body '${want}' not found — using pelvis frame for this hand`);
        }
      }
      const bid = this.wristBodyIds[hi];
      if (bid <= 0) {
        // Body unresolved/not found: pelvis fallback for this hand only.
        out[hi * 3] = fPelvis[0] * WbtForceCmdLocal.OBS_SCALE;
        out[hi * 3 + 1] = fPelvis[1] * WbtForceCmdLocal.OBS_SCALE;
        out[hi * 3 + 2] = fPelvis[2] * WbtForceCmdLocal.OBS_SCALE;
        continue;
      }
      // Wrist world quat (wxyz) from mjData.xquat[bid*4 .. bid*4+3].
      const xq = mjData.xquat;
      const qWrist: [number, number, number, number] = [
        xq[bid * 4] ?? 1.0,
        xq[bid * 4 + 1] ?? 0.0,
        xq[bid * 4 + 2] ?? 0.0,
        xq[bid * 4 + 3] ?? 0.0,
      ];
      // F_world = R(q_pelvis) @ F_pelvis ; F_hand = R(q_wrist)^-1 @ F_world.
      const fWorld = wbtQuatRotate(qPelvis, fPelvis);
      const fHand = wbtQuatRotateInverse(qWrist, fWorld);
      out[hi * 3] = fHand[0] * WbtForceCmdLocal.OBS_SCALE;
      out[hi * 3 + 1] = fHand[1] * WbtForceCmdLocal.OBS_SCALE;
      out[hi * 3 + 2] = fHand[2] * WbtForceCmdLocal.OBS_SCALE;
    }
    return out;
  }
}

// 6D UniFP-style F^ext in robot root frame: [L_Fx, L_Fy, L_Fz, R_Fx, R_Fy, R_Fz].
// The REAL external-disturbance wrench (same slider keys main.py emits for the
// orange F^ext arrow + the runtime xfrc injection), now ALSO fed to the ACTOR.
//
// This is Preset A (`g1-29dof-holosoma-2pt-force-ext-wbt`), which reverts the
// UniFP actor/critic asymmetry: the actor observes F^ext directly instead of
// having to infer it (matches training run fmotwon3 /
// force_tracking.py:1447-1463, where force_ext_local reads env._wbt_force_ext_b
// at scale 0.04, ROOT frame). The shipped `g1-29dof-holosoma-2pt-force-wbt`
// keeps F^ext critic-only and does NOT register this obs class, so its bundle is
// unaffected.
//
// A near-verbatim clone of WbtForceCmdLocal: same 0.04 obs scale, same pelvis /
// hand-frame branch, but reads the wbt_force_ext_local__* slider keys. Both obs
// classes read the SAME extras bus, so the orange F^ext arrow and this obs term
// stay in lockstep. F^ext=0 maps to 0 in every frame, so the idle state is
// frame-invariant.
export class WbtForceExtLocal extends WbtHistoryBase {
  // Matches HandForceConfig.f_ext_max (=25 N) in
  // src/extensions/wbt_wrappers/wbt_training/config_values/force_tracking.py;
  // the inverted scale (1/25 = 0.04) lands raw Newtons in the [-1, 1] band the
  // policy was trained on. Same value as WbtForceCmdLocal.OBS_SCALE by design.
  private static readonly OBS_SCALE = 0.04;
  // Slider keys MUST match main.py _FORCE_EXT_SLIDER_KEYS_* and the ext_slider_keys
  // WbtTrackingCommand reads for the orange arrow. Order [Lxyz, Rxyz] matches the
  // training force_ext_local flatten (env._wbt_force_ext_b.reshape(n, -1)).
  private static readonly KEYS = [
    'wbt_force_ext_local__l_fx',
    'wbt_force_ext_local__l_fy',
    'wbt_force_ext_local__l_fz',
    'wbt_force_ext_local__r_fx',
    'wbt_force_ext_local__r_fy',
    'wbt_force_ext_local__r_fz',
  ];

  // Opt-in hand-frame mode (default false => pelvis path, unchanged). Parallel to
  // WbtForceCmdLocal's hand_frame ablation — kept for symmetry / future --F_on_hand_frame
  // ext checkpoints; the fmotwon3 run is pelvis-frame, so this is inert there.
  private readonly handFrame: boolean;
  private readonly wristBodyNames: [string, string];
  private wristBodyIds: [number, number];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
    this.handFrame = (config as Record<string, unknown>).hand_frame === true;
    const wbn = (config as Record<string, unknown>).wrist_body_names;
    this.wristBodyNames =
      Array.isArray(wbn) && wbn.length === 2
        ? [String(wbn[0]), String(wbn[1])]
        : ['left_wrist_yaw_link', 'right_wrist_yaw_link'];
    this.wristBodyIds = [-2, -2];
  }

  protected computeCurrent(): Float32Array {
    // Raw pelvis-frame Newtons from the live sliders (same source as the arrow).
    const fp = new Float32Array(6);
    for (let i = 0; i < 6; i++) fp[i] = getPolicyExtra(WbtForceExtLocal.KEYS[i], 0.0);

    if (!this.handFrame) {
      // ---- PELVIS path (default, matches training force_ext_local) ----
      const out = new Float32Array(6);
      for (let i = 0; i < 6; i++) out[i] = fp[i] * WbtForceExtLocal.OBS_SCALE;
      return out;
    }

    // ---- HAND-FRAME path: rotate pelvis Newtons into each wrist frame ----
    const ctx = this.runner.getContext();
    const mjData = ctx?.mjData ?? null;
    const mjModel = ctx?.mjModel ?? null;
    if (!mjData || !mjModel) {
      const out = new Float32Array(6);
      for (let i = 0; i < 6; i++) out[i] = fp[i] * WbtForceExtLocal.OBS_SCALE;
      return out;
    }
    const q = mjData.qpos;
    const qPelvis: [number, number, number, number] = [q[3] ?? 1.0, q[4] ?? 0.0, q[5] ?? 0.0, q[6] ?? 0.0];
    const out = new Float32Array(6);
    for (let hi = 0; hi < 2; hi++) {
      const fPelvis: [number, number, number] = [fp[hi * 3], fp[hi * 3 + 1], fp[hi * 3 + 2]];
      if (this.wristBodyIds[hi] === -2) {
        this.wristBodyIds[hi] = -1;
        const want = this.wristBodyNames[hi];
        for (let b = 0; b < mjModel.nbody; b++) {
          const name = mjModel.body(b).name;
          if (name === want || name.endsWith(`/${want}`)) {
            this.wristBodyIds[hi] = b;
            break;
          }
        }
        if (this.wristBodyIds[hi] < 0) {
          console.warn(`[WbtForceExtLocal] hand-frame wrist body '${want}' not found — using pelvis frame for this hand`);
        }
      }
      const bid = this.wristBodyIds[hi];
      if (bid <= 0) {
        out[hi * 3] = fPelvis[0] * WbtForceExtLocal.OBS_SCALE;
        out[hi * 3 + 1] = fPelvis[1] * WbtForceExtLocal.OBS_SCALE;
        out[hi * 3 + 2] = fPelvis[2] * WbtForceExtLocal.OBS_SCALE;
        continue;
      }
      const xq = mjData.xquat;
      const qWrist: [number, number, number, number] = [
        xq[bid * 4] ?? 1.0,
        xq[bid * 4 + 1] ?? 0.0,
        xq[bid * 4 + 2] ?? 0.0,
        xq[bid * 4 + 3] ?? 0.0,
      ];
      const fWorld = wbtQuatRotate(qPelvis, fPelvis);
      const fHand = wbtQuatRotateInverse(qWrist, fWorld);
      out[hi * 3] = fHand[0] * WbtForceExtLocal.OBS_SCALE;
      out[hi * 3 + 1] = fHand[1] * WbtForceExtLocal.OBS_SCALE;
      out[hi * 3 + 2] = fHand[2] * WbtForceExtLocal.OBS_SCALE;
    }
    return out;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Hybrid position-force controller actor obs (exp:g1-wbt-2pt-hybrid, servo + Layer-A).
//
// Three NEW actor terms (obs04/05/06 in hybrid.py's 165D layout) that the browser
// runtime (mjswan fork runtime.ts) computes each substep from the causal virtual
// spring wall + Jᵀ servo and publishes to the policy-extras bus in the WORLD frame:
//   force_meas_local  wbt_force_meas__{l,r}_f{x,y,z}   simulated F/T reading  [N]
//   integrator_state  wbt_integ__{l,r}_f{x,y,z}        servo integrator I     [N]
//   x_tgt_shift       wbt_xtgt__{l,r}_f{x,y,z}         servo setpoint shift Δx [m]
//
// Training (hybrid.py force_meas_local / integrator_state / x_tgt_shift) stores
// these in WORLD-frame env buffers and rotates them world→ROOT (pelvis) via the
// live root quat before feeding the actor. We reproduce that EXACTLY: read the
// world-frame bus values, rotate by the inverse pelvis quat (qpos[3:7], wxyz),
// then apply the training obs scale (force_meas 0.04; integrator/x_tgt 1.0).
//
// When the runtime has not published a value yet (or for the Layer-A / servo-off
// preset, where I and Δx are never written) the keys read 0 via getPolicyExtra's
// fallback, so the terms are all-zero — exactly matching training's zeroed
// _wbt_integ_state / _wbt_delta_x buffers when servo_enabled=False (obs stays 165D).
// ────────────────────────────────────────────────────────────────────────────

// Shared helper: read a per-hand world-frame [Lxyz, Rxyz] vector from the extras
// bus, rotate each hand's 3-vector world→pelvis by the live inverse pelvis quat,
// and scale. Pelvis quat is qpos[3:7] (wxyz), native MuJoCo (no THREE swizzle) —
// the same source WbtForceCmdLocal's hand-frame path reads.
function wbtHybridWorldToPelvisScaled(
  runner: PolicyRunner,
  keyPrefix: string,
  scale: number,
): Float32Array {
  const out = new Float32Array(6);
  const raw = new Float32Array(6);
  const sides = ['l', 'r'];
  for (let hi = 0; hi < 2; hi++) {
    const s = sides[hi];
    raw[hi * 3] = getPolicyExtra(`${keyPrefix}__${s}_fx`, 0.0);
    raw[hi * 3 + 1] = getPolicyExtra(`${keyPrefix}__${s}_fy`, 0.0);
    raw[hi * 3 + 2] = getPolicyExtra(`${keyPrefix}__${s}_fz`, 0.0);
  }
  const ctx = runner.getContext();
  const mjData = ctx?.mjData ?? null;
  // No sim yet (or all-zero): scale the raw values in place. world==pelvis when
  // the value is 0, and before physics is live the pelvis quat is identity, so
  // this is safe and matches the F=0 invariant.
  if (!mjData) {
    for (let i = 0; i < 6; i++) out[i] = raw[i] * scale;
    return out;
  }
  const q = mjData.qpos;
  const qPelvis: [number, number, number, number] = [q[3] ?? 1.0, q[4] ?? 0.0, q[5] ?? 0.0, q[6] ?? 0.0];
  for (let hi = 0; hi < 2; hi++) {
    const vW: [number, number, number] = [raw[hi * 3], raw[hi * 3 + 1], raw[hi * 3 + 2]];
    const vP = wbtQuatRotateInverse(qPelvis, vW); // world → pelvis (root)
    out[hi * 3] = vP[0] * scale;
    out[hi * 3 + 1] = vP[1] * scale;
    out[hi * 3 + 2] = vP[2] * scale;
  }
  return out;
}

// force_meas_local: simulated F/T reading in the ROOT frame, scale 0.04
// (= 1/f_cmd_max=25 N, matching hybrid.py force_meas_local obs cfg).
export class WbtForceMeasLocal extends WbtHistoryBase {
  private static readonly OBS_SCALE = 0.04;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
  }
  protected computeCurrent(): Float32Array {
    return wbtHybridWorldToPelvisScaled(this.runner, 'wbt_force_meas', WbtForceMeasLocal.OBS_SCALE);
  }
}

// integrator_state: servo integrator I in the ROOT frame, scale 1.0. Reads zeros
// for the Layer-A (servo-off) preset (the runtime never publishes I there).
export class WbtIntegratorState extends WbtHistoryBase {
  private static readonly OBS_SCALE = 1.0;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
  }
  protected computeCurrent(): Float32Array {
    return wbtHybridWorldToPelvisScaled(this.runner, 'wbt_integ', WbtIntegratorState.OBS_SCALE);
  }
}

// x_tgt_shift: servo task-space setpoint shift Δx in the ROOT frame, scale 1.0.
// Reads zeros for the Layer-A (servo-off) preset.
export class WbtXTgtShift extends WbtHistoryBase {
  private static readonly OBS_SCALE = 1.0;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
  }
  protected computeCurrent(): Float32Array {
    return wbtHybridWorldToPelvisScaled(this.runner, 'wbt_xtgt', WbtXTgtShift.OBS_SCALE);
  }
}

// contact_phase: soft deployable contact label ∈[0,1] per hand (C1 unified preset).
// Hysteresis(‖F_meas‖; 2/1.5 N) + LPF(50 ms), computed by the runtime's spring-wall
// sim and published as wbt_cphase__{l,r}. A per-hand scalar (no world→pelvis rotation
// needed). Zeros before the runtime publishes (free space) — matches training's
// zeroed _wbt_contact_phase buffer at reset. Scale 1.0. 2D.
export class WbtContactPhase extends WbtHistoryBase {
  private static readonly OBS_SCALE = 1.0;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 2);
  }
  protected computeCurrent(): Float32Array {
    const out = new Float32Array(2);
    out[0] = getPolicyExtra('wbt_cphase__l', 0.0) * WbtContactPhase.OBS_SCALE;
    out[1] = getPolicyExtra('wbt_cphase__r', 0.0) * WbtContactPhase.OBS_SCALE;
    return out;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// LIVE VIVE-tracker 2pt teleop (the parallel live path; replay path untouched).
//
// The Python bridge (wbt_realtime_eval.vive.vive_teleop_bridge) reads two VIVE
// trackers, calibrates them onto the robot's default-pose hand target, and
// serves the pelvis-frame 2pt VR command over a WebSocket:
//     { seq:int, t:float, pos:[6], orn:[12], orientation:bool }
// where pos = [L_xyz, R_xyz] (pelvis frame) and orn = [L_rot6d, R_rot6d]
// (rot6d = first two COLUMNS of R, C-order [r00,r01,r10,r11,r20,r21]) — byte-
// for-byte the convention WbtVr2PointPos/Orn (replay) forward from the bake.
//
// A single module-level store holds the latest frame; the two obs classes read
// it each control step. Until the first message arrives (or if the bridge is
// down) they return the HOME anchor (default-pose target) passed via config —
// NEVER zeros, because zeros command both hands to the pelvis origin and
// collapse the arms. The WS is opened lazily, once, on first obs construction
// (which happens on the main thread with a live document, once per scene load).
// ────────────────────────────────────────────────────────────────────────────

class WbtVrLiveStore {
  pos: Float32Array | null = null;   // 6 — [L_xyz, R_xyz], pelvis frame (2pt wire)
  orn: Float32Array | null = null;   // 12 — [L_rot6d, R_rot6d]      (2pt wire)
  // 3pt wire (bridge --mode 3pt): [L,R,head] xyz and 3x xyzw QUATERNION. Held in
  // their OWN fields, never in pos/orn: the 3pt orn is 12D just like the 2pt one
  // but a different REP (3 quats vs 2 rot6d), so one shared field would let a
  // mode mismatch feed 2pt rot6d bytes to a 3pt policy as quaternions at exactly
  // the right width — the one class of error no dimension check can catch.
  pos3: Float32Array | null = null;  // 9 — [L_xyz, R_xyz, H_xyz], pelvis frame
  orn3: Float32Array | null = null;  // 12 — [L_xyzw, R_xyzw, H_xyzw]
  // MOTION-COMPLETION wires (bridge --mode 5pt / dense). The same three trackers,
  // but routed through Motion_Completion first, so what arrives is a slice of a
  // COMPLETED 29-DoF body: five keypoints, or all 29 joint targets. Again their own
  // fields — 5pt's pos[15]/orn[20] and dense's joint arrays share no width with
  // anything above, but keeping them separate is what makes a mode mismatch read as
  // "parked at HOME" instead of as a plausible pose.
  pos5: Float32Array | null = null;  // 15 — [L,R,head,Lfoot,Rfoot] xyz, pelvis frame
  orn5: Float32Array | null = null;  // 20 — 5x xyzw quaternion
  jointPos: Float32Array | null = null;  // 29 — dense joint targets (MC's joint order)
  jointVel: Float32Array | null = null;  // 29 — finite-difference velocities
  jointNames: string[] | null = null;    // 29 — MC's order; the remap is BY NAME
  mcLagS = 0;                            // reported by the bridge, for the HUD
  seq = 0;
  // REPLAY-ONLY episode telemetry (vive/mc_replay_bridge.py). `nEpisodes > 0` is the ONE
  // signal that says "this bridge is a recorded take, not an operator", and it is what gates
  // the episode picker: a live rig never sends these keys, so its page never grows the panel
  // and the rehearsal flow is untouched.
  //
  // `episode` is the episode the CURRENT COMMAND came from, not the one last clicked. The two
  // differ for the ~1 s a switch takes to fill its first window (the bridge holds the old
  // command through it on purpose), and showing the bridge's own number is what makes a click
  // the bridge ignored — out of range, or a socket that never opened — visible.
  episode = -1;
  nEpisodes = 0;
  episodeFrames = 0;
  episodeFrame = 0;
  // THE REFERENCE-GHOST CHANNEL (bridge `--publish-mc-qpos`, both MC modes). This is the
  // WHOLE completed body — 29 joints + the completion's own root in the readout frame — and
  // it is drawn, never fed to the policy. It answers the one question the five commanded
  // spheres cannot: when the sim's feet do not move, did the COMPLETION ask for a step?
  //
  // Republished onto the extras bus rather than read directly by the command term, because
  // mjswan's `_build_client.py` concatenates this file under core/observation/ and
  // WbtTrackingCommand.ts under core/command/ — they end up in different modules, so this
  // singleton is simply not in scope over there. The extras bus is the seam that already
  // exists for exactly this (it is how the live 3pt target triads reach the same term).
  //
  // Keyed BY JOINT NAME (`wbt_mcghost__q__<name>`), not by index. MC's joint order and the
  // render model's are both real orders and both 29 long, so an index-aligned handoff would
  // scramble the ghost into a plausible-looking pose; a name the render model does not have
  // is simply never read, which fails loud (that joint does not move) instead of silently.
  mcQposSeq = 0;                         // seq of the newest ghost frame; 0 = none ever
  private mcExtrasWritten = false;
  private mcGhostAbsentWarned = false;

  // ── FUJI OFFLINE DRIVER telemetry (scripts/hl_offline_drive.py) ───────────────────
  // That planner replays a DATASET anchor through the policy and publishes the chunk on this
  // same socket, so the page is its whole control surface: pick a sample, Predict, Run, Reset.
  //
  // `fujiNSamples > 0` is the ONE signal that says so, and it gates the panel exactly the way
  // `nEpisodes` gates the episode picker: no live tracker bridge, no MC replay bridge and none
  // of the three deploy planners ever send it, so every other page is untouched by construction
  // rather than by a build flag someone has to remember.
  //
  // `fujiAck` is a PROTOCOL, not a readout: it counts the control verbs the planner has finished
  // acting on, so a click can be shown as pending until it has actually been handled. Without it
  // the panel reads the PREVIOUS state as the answer to the click it just sent.
  fujiNSamples = 0;
  fujiNEpisodes = 0;
  fujiSample = 0;
  fujiEpisode = -1;
  fujiEpisodeIndex = -1;
  fujiEpisodeFrame = -1;
  fujiEpisodeAnchors = 0;
  fujiSplit = '';
  fujiState = '';
  fujiMsg = '';
  fujiLayout = '';
  fujiChunkSteps = 0;
  fujiExecSteps = 0;
  fujiStep = -1;
  fujiAck = 0;

  private publishGhostExtras(
    qpos: ArrayLike<number>,
    names: readonly string[],
    rootPos: ArrayLike<number>,
    rootQuat: ArrayLike<number>,
    seq: number,
  ): void {
    for (let i = 0; i < 3; i++) setPolicyExtra(`wbt_mcghost__rp${i}`, Number(rootPos[i]) || 0.0);
    // A quaternion is the one field where "fall back to 0" is not safe: an all-zero quat is
    // not a rotation, and MuJoCo hands back NaN world poses for it, which propagates into the
    // render as an invisible or exploded body. Normalize here, identity if degenerate.
    let qw = Number(rootQuat[0]) || 0.0;
    let qx = Number(rootQuat[1]) || 0.0;
    let qy = Number(rootQuat[2]) || 0.0;
    let qz = Number(rootQuat[3]) || 0.0;
    const qn = Math.hypot(qw, qx, qy, qz);
    if (!(qn > 1e-6)) {
      qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0;
    } else {
      qw /= qn; qx /= qn; qy /= qn; qz /= qn;
    }
    setPolicyExtra('wbt_mcghost__rq0', qw);
    setPolicyExtra('wbt_mcghost__rq1', qx);
    setPolicyExtra('wbt_mcghost__rq2', qy);
    setPolicyExtra('wbt_mcghost__rq3', qz);
    for (let i = 0; i < names.length; i++) {
      setPolicyExtra(`wbt_mcghost__q__${names[i]}`, Number(qpos[i]) || 0.0);
    }
    // The consumer's FRESHNESS clock. `performance.now()` and not the bridge's `t`: the two
    // may be different hosts, and a teleop operator watching a mesh they are trying to track
    // must not be shown a stale pose because of clock skew. Monotonic, so it also survives a
    // wall-clock step.
    setPolicyExtra('wbt_mcghost__rx', performance.now());
    // Written LAST: the consumer gates on this, so it can never latch a half-written body.
    setPolicyExtra('wbt_mcghost__seq', seq);
    this.mcQposSeq = seq;
    if (!this.mcExtrasWritten) {
      this.mcExtrasWritten = true;
      console.log(
        `[WbtVrLive] MC reference ghost armed (LIGHT PURPLE): ${names.length} joints + root, ` +
        'from the bridge\'s --publish-mc-qpos stream. This is the completion\'s GT pose — ' +
        'watch whether the solid robot keeps up with it.',
      );
    }
  }
  connected = false;
  private url: string | null = null;
  private ws: WebSocket | null = null;
  private retryMs = 1000;
  private readonly retryMaxMs = 8000;
  private opened = false;
  private reconnectPending = false;

  // Idempotent: only the first call with a URL actually opens the socket.
  ensureOpen(url: string): void {
    if (this.opened) return;
    this.opened = true;
    this.url = url;
    this.connect();
  }

  // Detach handlers + close any prior socket so a late event from a dead socket
  // can't schedule extra reconnects (handler-leak guard).
  private teardown(sock: WebSocket | null): void {
    if (!sock) return;
    sock.onopen = null;
    sock.onmessage = null;
    sock.onclose = null;
    sock.onerror = null;
    try {
      sock.close();
    } catch {
      /* already closing/closed */
    }
  }

  private connect(): void {
    if (!this.url) return;
    this.reconnectPending = false;
    this.teardown(this.ws);
    this.ws = null;
    let sock: WebSocket;
    try {
      sock = new WebSocket(this.url);
    } catch (e) {
      console.warn('[WbtVrLive] WebSocket construct failed; retrying', e);
      this.scheduleReconnect();
      return;
    }
    this.ws = sock;
    sock.onopen = () => {
      this.connected = true;
      this.retryMs = 1000; // reset backoff on a good connection
      console.log('[WbtVrLive] connected to', this.url);
    };
    sock.onmessage = (ev: MessageEvent) => {
      this.ingest(ev.data);
    };
    sock.onclose = () => {
      this.connected = false;
      // Only react to the socket we still own (ignore stale dead-socket events).
      if (sock === this.ws) this.scheduleReconnect();
    };
    sock.onerror = () => {
      // onclose fires next and handles the reconnect; swallow to avoid throwing.
      this.connected = false;
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectPending) return; // single pending reconnect — no compounding
    this.reconnectPending = true;
    const delay = this.retryMs;
    this.retryMs = Math.min(this.retryMs * 2, this.retryMaxMs); // capped backoff
    setTimeout(() => this.connect(), delay);
  }

  // Stop reconnecting + close (for an explicit teardown; the store is otherwise
  // page-lifetime since obs classes are rebuilt per scene but the page persists).
  close(): void {
    this.reconnectPending = true; // suppress any further scheduleReconnect
    this.teardown(this.ws);
    this.ws = null;
    this.opened = false;
    this.connected = false;
  }

  // Send a control message back to the bridge (e.g. {cmd:'recalibrate'}). The
  // bridge's WS handler parses these and queues the action under its lock. Safe
  // no-op if the socket isn't open yet.
  sendCmd(obj: Record<string, unknown>): boolean {
    if (this.ws && this.connected) {
      try {
        this.ws.send(JSON.stringify(obj));
        return true;
      } catch {
        return false;
      }
    }
    return false;
  }

  private ingest(data: unknown): void {
    if (typeof data !== 'string') return;
    let msg: {
      pos?: unknown; orn?: unknown; seq?: unknown; mode?: unknown;
      joint_pos?: unknown; joint_vel?: unknown; joint_names?: unknown; mc_lag_s?: unknown;
      mc_qpos?: unknown; mc_joint_names?: unknown;
      mc_root_pos?: unknown; mc_root_quat?: unknown;
      episode?: unknown; n_episodes?: unknown;
      episode_frames?: unknown; episode_frame?: unknown;
      fuji_n_samples?: unknown; fuji_n_episodes?: unknown; fuji_sample?: unknown;
      fuji_episode?: unknown; fuji_episode_index?: unknown; fuji_episode_frame?: unknown;
      fuji_episode_anchors?: unknown; fuji_split?: unknown; fuji_state?: unknown;
      fuji_msg?: unknown; fuji_layout?: unknown; fuji_chunk_steps?: unknown;
      fuji_exec_steps?: unknown; fuji_step?: unknown; fuji_ack?: unknown;
    };
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }
    const p = msg.pos;
    const o = msg.orn;
    // The bridge tags every frame with its mode ("2pt" / "3pt"). Route on that
    // tag, not on length alone: pos disambiguates (6 vs 9) but orn does NOT
    // (12 either way, different rep), so a 2pt bundle pointed at a 3pt bridge
    // used to keep HOME for pos while accepting the 3pt QUATERNIONS as rot6d.
    const is3pt = msg.mode === '3pt';
    if (!is3pt && Array.isArray(p) && p.length === 6) {
      const arr = new Float32Array(6);
      for (let i = 0; i < 6; i++) arr[i] = Number(p[i]) || 0.0;
      this.pos = arr;
    }
    if (!is3pt && Array.isArray(o) && o.length === 12) {
      const arr = new Float32Array(12);
      for (let i = 0; i < 12; i++) arr[i] = Number(o[i]) || 0.0;
      this.orn = arr;
    }
    // 3pt: pos and orn are set TOGETHER or not at all, so a half-parsed frame
    // can never pair one point's live position with another frame's rotation.
    if (is3pt && Array.isArray(p) && p.length === 9 && Array.isArray(o) && o.length === 12) {
      const ap = new Float32Array(9);
      for (let i = 0; i < 9; i++) ap[i] = Number(p[i]) || 0.0;
      const ao = new Float32Array(12);
      for (let i = 0; i < 12; i++) ao[i] = Number(o[i]) || 0.0;
      this.pos3 = ap;
      this.orn3 = ao;
    }
    // 5pt (motion completion): pos[15] + orn[20], again set TOGETHER.
    if (msg.mode === '5pt' && Array.isArray(p) && p.length === 15
        && Array.isArray(o) && o.length === 20) {
      const ap = new Float32Array(15);
      for (let i = 0; i < 15; i++) ap[i] = Number(p[i]) || 0.0;
      const ao = new Float32Array(20);
      for (let i = 0; i < 20; i++) ao[i] = Number(o[i]) || 0.0;
      this.pos5 = ap;
      this.orn5 = ao;
    }
    // dense (motion completion): 29 joint targets + velocities, plus the NAMES they are
    // in. Stored and consumed by name, never by index: MC's g1_fk_params.npz joint order
    // and the policy's policy_joint_names are both real orders and they are not the same,
    // and both are 29 long, so an index-aligned read would be silently scrambled.
    if (msg.mode === 'dense' && Array.isArray(msg.joint_pos) && msg.joint_pos.length === 29
        && Array.isArray(msg.joint_vel) && msg.joint_vel.length === 29) {
      const jp = new Float32Array(29);
      const jv = new Float32Array(29);
      for (let i = 0; i < 29; i++) {
        jp[i] = Number((msg.joint_pos as unknown[])[i]) || 0.0;
        jv[i] = Number((msg.joint_vel as unknown[])[i]) || 0.0;
      }
      if (Array.isArray(msg.joint_names) && msg.joint_names.length === 29) {
        this.jointNames = (msg.joint_names as unknown[]).map((x) => String(x));
      }
      // Without the names there is no safe way to place these 29 numbers, so the frame
      // is DROPPED rather than assumed to be in the policy's order.
      if (this.jointNames) {
        this.jointPos = jp;
        this.jointVel = jv;
      }
    }
    if (typeof msg.mc_lag_s === 'number') this.mcLagS = msg.mc_lag_s;
    if (typeof msg.seq === 'number') this.seq = msg.seq;
    // Episode telemetry. Read UNCONDITIONALLY of the mode tag, because the bridge's very
    // first frames are metadata-only (no pose at all) so that the picker can be used during
    // the window fill rather than after it.
    if (typeof msg.n_episodes === 'number' && msg.n_episodes > 0) {
      this.nEpisodes = msg.n_episodes;
      if (typeof msg.episode === 'number') this.episode = msg.episode;
      if (typeof msg.episode_frames === 'number') this.episodeFrames = msg.episode_frames;
      if (typeof msg.episode_frame === 'number') this.episodeFrame = msg.episode_frame;
    }
    // FUJI offline-driver telemetry. Read UNCONDITIONALLY of the mode tag, like the episode
    // keys above and for the same reason: the panel must work on the dense wire AND the 3pt one,
    // and the gate is `fuji_n_samples` rather than any mode.
    if (typeof msg.fuji_n_samples === 'number' && msg.fuji_n_samples > 0) {
      this.fujiNSamples = msg.fuji_n_samples;
      if (typeof msg.fuji_n_episodes === 'number') this.fujiNEpisodes = msg.fuji_n_episodes;
      if (typeof msg.fuji_sample === 'number') this.fujiSample = msg.fuji_sample;
      if (typeof msg.fuji_episode === 'number') this.fujiEpisode = msg.fuji_episode;
      if (typeof msg.fuji_episode_index === 'number') this.fujiEpisodeIndex = msg.fuji_episode_index;
      if (typeof msg.fuji_episode_frame === 'number') this.fujiEpisodeFrame = msg.fuji_episode_frame;
      if (typeof msg.fuji_episode_anchors === 'number') {
        this.fujiEpisodeAnchors = msg.fuji_episode_anchors;
      }
      if (typeof msg.fuji_split === 'string') this.fujiSplit = msg.fuji_split;
      if (typeof msg.fuji_state === 'string') this.fujiState = msg.fuji_state;
      if (typeof msg.fuji_msg === 'string') this.fujiMsg = msg.fuji_msg;
      if (typeof msg.fuji_layout === 'string') this.fujiLayout = msg.fuji_layout;
      if (typeof msg.fuji_chunk_steps === 'number') this.fujiChunkSteps = msg.fuji_chunk_steps;
      if (typeof msg.fuji_exec_steps === 'number') this.fujiExecSteps = msg.fuji_exec_steps;
      if (typeof msg.fuji_step === 'number') this.fujiStep = msg.fuji_step;
      if (typeof msg.fuji_ack === 'number') this.fujiAck = msg.fuji_ack;
    }
    // Reference ghost — mode-independent ON PURPOSE. The four keys mean the same thing in
    // 5pt and dense (the completed body, in the readout frame), so the ghost has ONE code
    // path and can never be crossed with the mode-tagged COMMAND arrays above. All four are
    // required together: a body placed by a stale root, or a fresh root on stale joints, is
    // a pose that never existed and would read as a completion artifact.
    if (Array.isArray(msg.mc_qpos) && Array.isArray(msg.mc_joint_names)
        && msg.mc_qpos.length === msg.mc_joint_names.length && msg.mc_qpos.length > 0
        && Array.isArray(msg.mc_root_pos) && msg.mc_root_pos.length === 3
        && Array.isArray(msg.mc_root_quat) && msg.mc_root_quat.length === 4) {
      this.publishGhostExtras(
        msg.mc_qpos as number[],
        (msg.mc_joint_names as unknown[]).map((x) => String(x)),
        msg.mc_root_pos as number[],
        msg.mc_root_quat as number[],
        typeof msg.seq === 'number' ? msg.seq : this.mcQposSeq + 1,
      );
    } else if ((msg.mode === '5pt' || msg.mode === 'dense') && !this.mcGhostAbsentWarned) {
      // An MC frame arrived without the ghost payload. Say so ONCE, from here — this is the
      // only place that can tell "the bridge is not publishing it" apart from "this preset
      // has no MC stream at all", and the ghost's own code path stays silent so a non-MC
      // preset never sees this. Same warning as vive/mc_teleop_gate_viz.py's.
      this.mcGhostAbsentWarned = true;
      console.warn(
        '[WbtVrLive] the wire has no `mc_qpos`, so the LIGHT-PURPLE completion reference ' +
        'ghost is not drawn. Restart the bridge with --publish-mc-qpos to see what the ' +
        'completion asked for next to what the controller did.',
      );
    }
  }
}

// Module-level singleton — survives the line-by-line ts_src concat the same way
// getWbtTracking()'s state does. Shared by both live obs classes.
const _wbtVrLiveStore = new WbtVrLiveStore();

// Read the WS url + HOME fallback from the obs term's config (main_vive.py
// injects them per-component, exactly like WbtJointPos gets default_joint_angles).
// Defaults keep the class usable if a field is missing.
function wbtVrLiveUrl(config: ObservationConfig): string {
  const u = (config as { vive_ws_url?: unknown }).vive_ws_url;
  return typeof u === 'string' && u.length > 0 ? u : 'ws://localhost:8770';
}

function wbtVrLiveHome(config: ObservationConfig, size: number): Float32Array {
  const h = (config as { home?: unknown }).home;
  const out = new Float32Array(size);
  if (Array.isArray(h) && h.length === size) {
    for (let i = 0; i < size; i++) out[i] = Number(h[i]) || 0.0;
  }
  return out;
}

// ── Recalibrate button (top-right) ──────────────────────────────────────────
// One-shot DOM button (built once, like the slider panel). Click sequence:
//   1. reset the simulation to the reference/default pose (so the robot snaps
//      back to a clean stand before re-anchoring), via the CommandManager's
//      system reset button (-> runtime.resetSimulation()).
//   2. wait 5 s (countdown shown on the button) so the operator can get into
//      the calibration pose.
//   3. send {cmd:'recalibrate'} to the bridge, which re-captures the yaw +
//      per-hand anchor on the next dual frame.
const WBT_RECALIB_BTN_ID = 'wbt-recalib-btn';
const WBT_RECALIB_DELAY_S = 5;

function wbtTriggerSimReset(): void {
  // CommandManager owns the reset callback wired to runtime.resetSimulation().
  try {
    getCommandManager().triggerButton('_system:reset');
  } catch (e) {
    console.warn('[WbtVrLive] sim reset trigger failed', e);
  }
}

function mountRecalibButton(): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(WBT_RECALIB_BTN_ID)) return; // build once
  const btn = document.createElement('button');
  btn.id = WBT_RECALIB_BTN_ID;
  btn.textContent = 'Recalibrate';
  Object.assign(btn.style, {
    position: 'fixed',
    top: '12px',
    right: '12px',
    zIndex: '10001',
    padding: '10px 16px',
    background: 'rgba(98, 51, 235, 0.92)',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '13px',
    fontWeight: 'bold',
    cursor: 'pointer',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    userSelect: 'none',
  } as Partial<CSSStyleDeclaration>);

  let busy = false;
  btn.addEventListener('click', () => {
    if (busy) return;
    busy = true;
    btn.style.background = 'rgba(120, 120, 130, 0.92)';
    // 1) reset the sim to the reference pose.
    wbtTriggerSimReset();
    // 2) 5 s countdown, then 3) tell the bridge to recalibrate.
    let remaining = WBT_RECALIB_DELAY_S;
    btn.textContent = `Hold pose… ${remaining}s`;
    const tick = setInterval(() => {
      remaining -= 1;
      if (remaining > 0) {
        btn.textContent = `Hold pose… ${remaining}s`;
        return;
      }
      clearInterval(tick);
      const sent = _wbtVrLiveStore.sendCmd({ cmd: 'recalibrate' });
      btn.textContent = sent ? 'Recalibrating…' : 'Bridge offline';
      setTimeout(() => {
        btn.textContent = 'Recalibrate';
        btn.style.background = 'rgba(98, 51, 235, 0.92)';
        busy = false;
      }, 1500);
    }, 1000);
  });
  document.body.appendChild(btn);
}

// ── Episode picker (REPLAY bridge only) ─────────────────────────────────────
// A recorded session is 100+ takes and the bridge spawns a Motion_Completion backend at
// startup, so "restart the bridge with --episode N" costs a backend respawn per take you want
// to look at. vive/mc_replay_bridge.py therefore reads its socket, and this panel is the thing
// that talks to it: `{cmd:'episode', index:N}` over the SAME WebSocket the commands arrive on.
//
// It mounts ONLY once the bridge has reported `n_episodes`, which is the one thing a live
// tracker bridge never sends. That is deliberate: the rig's page must not grow a control that
// would do nothing there, and the gate is the wire itself rather than a build flag, so there is
// no mode to get wrong.
//
// The label under the picker is the BRIDGE's episode, not the one selected. They differ for the
// ~1 s a switch takes to fill its first MC window (the bridge holds the previous command
// through it, so the robot never snaps to HOME), and they differ FOREVER if the request was
// rejected — which is the only way a bad click is visible at all.
const WBT_EPISODE_PANEL_ID = 'wbt-episode-panel';
let _wbtEpisodePanelArmed = false;

function mountEpisodePanel(): void {
  if (typeof document === 'undefined') return;
  if (_wbtEpisodePanelArmed) return; // arm once; the poller below builds at most one panel
  _wbtEpisodePanelArmed = true;
  const poll = setInterval(() => {
    if (_wbtVrLiveStore.nEpisodes <= 0) return; // live bridge -> never mounts
    clearInterval(poll);
    buildEpisodePanel(_wbtVrLiveStore.nEpisodes);
  }, 400);
}

function buildEpisodePanel(nEpisodes: number): void {
  if (document.getElementById(WBT_EPISODE_PANEL_ID)) return;
  const box = document.createElement('div');
  box.id = WBT_EPISODE_PANEL_ID;
  Object.assign(box.style, {
    position: 'fixed',
    top: '58px',
    right: '12px',
    zIndex: '10001',
    padding: '8px 10px',
    background: 'rgba(24, 24, 32, 0.92)',
    color: '#fff',
    border: '1px solid rgba(160, 130, 255, 0.5)',
    borderRadius: '6px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '12px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    userSelect: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    minWidth: '190px',
  } as Partial<CSSStyleDeclaration>);

  const title = document.createElement('div');
  title.textContent = `recorded take (${nEpisodes} episodes)`;
  Object.assign(title.style, { opacity: '0.7', fontSize: '11px' } as Partial<CSSStyleDeclaration>);

  const row = document.createElement('div');
  Object.assign(row.style, {
    display: 'flex', alignItems: 'center', gap: '4px',
  } as Partial<CSSStyleDeclaration>);

  const sel = document.createElement('select');
  Object.assign(sel.style, {
    flex: '1',
    background: 'rgba(0,0,0,0.5)',
    color: '#fff',
    border: '1px solid rgba(255,255,255,0.25)',
    borderRadius: '4px',
    fontFamily: 'inherit',
    fontSize: '12px',
    padding: '3px 4px',
  } as Partial<CSSStyleDeclaration>);
  for (let i = 0; i < nEpisodes; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `ep ${String(i).padStart(3, '0')}`;
    sel.appendChild(opt);
  }

  const mkStep = (text: string, delta: number): HTMLButtonElement => {
    const b = document.createElement('button');
    b.textContent = text;
    Object.assign(b.style, {
      padding: '3px 7px',
      background: 'rgba(98, 51, 235, 0.85)',
      color: '#fff',
      border: 'none',
      borderRadius: '4px',
      fontFamily: 'inherit',
      fontSize: '12px',
      cursor: 'pointer',
    } as Partial<CSSStyleDeclaration>);
    b.addEventListener('click', () => {
      // Step from what is SELECTED, so repeated clicks walk the list even while the bridge is
      // still filling the previous switch's window (its own number lags by ~1 s).
      const next = Math.min(nEpisodes - 1, Math.max(0, Number(sel.value) + delta));
      sel.value = String(next);
      request(next);
    });
    return b;
  };

  const status = document.createElement('div');
  Object.assign(status.style, {
    opacity: '0.85', fontSize: '11px', whiteSpace: 'pre',
  } as Partial<CSSStyleDeclaration>);

  let lastSent = -1;
  let sendFailed = false;
  const request = (idx: number): void => {
    lastSent = idx;
    sendFailed = !_wbtVrLiveStore.sendCmd({ cmd: 'episode', index: idx });
    if (sendFailed) {
      console.warn('[WbtVrLive] episode request not sent — the bridge socket is not open');
    }
  };
  sel.addEventListener('change', () => request(Number(sel.value)));

  // Sync the dropdown to the bridge's STARTING episode exactly once (`--episode N`), then
  // never again: after the operator has touched it, the selection is theirs and the bridge's
  // number belongs in the status line, not in the control.
  let synced = false;
  setInterval(() => {
    const s = _wbtVrLiveStore;
    if (!synced && s.episode >= 0 && s.episode < nEpisodes) {
      synced = true;
      sel.value = String(s.episode);
    }
    const at = s.episodeFrames > 0
      ? `${s.episodeFrame}/${s.episodeFrames}`
      : '--';
    let line = `bridge: ep ${String(Math.max(s.episode, 0)).padStart(3, '0')}  ${at}`;
    if (sendFailed) {
      line += '\n⚠ bridge offline — not sent';
    } else if (lastSent >= 0 && s.episode !== lastSent) {
      line += `\n→ switching to ep ${String(lastSent).padStart(3, '0')}…`;
    }
    status.textContent = line;
  }, 250);

  row.appendChild(mkStep('◀', -1));
  row.appendChild(sel);
  row.appendChild(mkStep('▶', +1));
  box.appendChild(title);
  box.appendChild(row);
  box.appendChild(status);
  document.body.appendChild(box);
  console.log(
    `[WbtVrLive] episode picker mounted: ${nEpisodes} recorded episodes on this bridge. ` +
    'A switch costs ~1 s of MC window fill, during which the bridge holds the last command.',
  );
}

// ── FUJI offline-driver panel (scripts/hl_offline_drive.py only) ─────────────────────
// The whole control surface for a FUJI offline drive: pick a dataset anchor, Predict, watch the
// LIGHT-PURPLE ghost play the chunk, Run it into this controller, Reset. It lives HERE rather
// than in a second viser window because everything the operator has to watch — the robot, the
// ghost, the tracking-error panel — is already on this page, and a second window means reading
// two clocks.
//
// Gated on `fujiNSamples`, which only that planner sends (see the store's field comment): the
// rig's page and every other preset's page never grow it, and there is no build flag to get
// wrong. Every verb is `{cmd:'fuji', action:...}` over the SAME socket the commands arrive on,
// which is namespaced so the MC replay bridge's `{cmd:'episode'}` keeps meaning what it means
// there.
//
// PENDING IS SHOWN, NOT ASSUMED. Each click records the `fujiAck` it was sent at and the button
// stays dimmed until the planner's own counter passes it. That is what makes a verb the planner
// REFUSED (nothing staged, low level not connected, a sample out of range) visible as a message
// instead of as a click that seemed to work.
const WBT_FUJI_PANEL_ID = 'wbt-fuji-panel';
let _wbtFujiPanelArmed = false;

function mountFujiPanel(): void {
  if (typeof document === 'undefined') return;
  if (_wbtFujiPanelArmed) return; // arm once; the poller below builds at most one panel
  _wbtFujiPanelArmed = true;
  const poll = setInterval(() => {
    if (_wbtVrLiveStore.fujiNSamples <= 0) return; // any other producer -> never mounts
    clearInterval(poll);
    // CAUGHT AND NAMED. A throw inside a setInterval callback is swallowed into an "Uncaught
    // TypeError" with a minified stack and no clue which panel died, and since the interval has
    // already been cleared there is no retry either — the symptom is a panel that simply is not
    // there while the wire is demonstrably fine. That cost a whole browser build to diagnose
    // once (Object.assign onto a CSSStyleDeclaration); this makes the next one say so.
    try {
      buildFujiPanel();
    } catch (e) {
      console.error(
        '[WbtFuji] the offline-driver panel FAILED to build — the wire is fine, the DOM code is '
        + 'not. Nothing else on this page is affected.', e,
      );
    }
  }, 400);
}

function wbtFujiButton(label: string, bg: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  Object.assign(b.style, {
    flex: '1',
    padding: '5px 8px',
    background: bg,
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontFamily: 'inherit',
    fontSize: '12px',
    fontWeight: 'bold',
    cursor: 'pointer',
  } as Partial<CSSStyleDeclaration>);
  return b;
}

function buildFujiPanel(): void {
  if (document.getElementById(WBT_FUJI_PANEL_ID)) return;
  const s = _wbtVrLiveStore;
  const box = document.createElement('div');
  box.id = WBT_FUJI_PANEL_ID;
  Object.assign(box.style, {
    position: 'fixed',
    // BELOW the page's own title bar and the Recalibrate button (12-48 px), not at 58 px where
    // the episode picker sits: that one never coexists with this panel (a FUJI driver sends no
    // `n_episodes`), but the TITLE BAR does, and at 58 px this box covered it.
    top: '96px',
    right: '12px',
    zIndex: '10001',
    padding: '8px 10px',
    background: 'rgba(24, 24, 32, 0.94)',
    color: '#fff',
    border: '1px solid rgba(160, 130, 255, 0.6)',
    borderRadius: '6px',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '12px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.35)',
    userSelect: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    minWidth: '250px',
    maxWidth: '320px',
  } as Partial<CSSStyleDeclaration>);

  const title = document.createElement('div');
  title.textContent = 'FUJI offline driver';
  Object.assign(title.style, {
    fontWeight: 'bold', color: '#c3b0ff',
  } as Partial<CSSStyleDeclaration>);
  const layout = document.createElement('div');
  Object.assign(layout.style, { opacity: '0.7', fontSize: '11px' } as Partial<CSSStyleDeclaration>);

  // --- the cursor: an anchor index AND an episode, because they are not interchangeable.
  // The index is what the policy is conditioned on; the episode is what an operator thinks in,
  // and an episode's anchors are NOT a contiguous range (the glitch/validity filters punch
  // holes), so the planner owns the mapping and this only sends a request.
  let pending = -1;                                   // the ack a click is waiting for
  const request = (action: string, extra: Record<string, unknown> = {}): void => {
    pending = s.fujiAck;
    if (!s.sendCmd({ cmd: 'fuji', action, ...extra })) {
      pending = -1;
      console.warn(`[WbtFuji] ${action} not sent — the driver socket is not open`);
    }
  };

  const mkRow = (): HTMLDivElement => {
    const r = document.createElement('div');
    Object.assign(r.style, {
      display: 'flex', alignItems: 'center', gap: '4px',
    } as Partial<CSSStyleDeclaration>);
    return r;
  };

  const anchorRow = mkRow();
  const anchorLabel = document.createElement('span');
  anchorLabel.textContent = 'anchor';
  Object.assign(anchorLabel.style, { opacity: '0.7', minWidth: '52px' } as Partial<CSSStyleDeclaration>);
  // ⚠️ ONE plain style LITERAL, applied to both inputs. Do NOT reuse the first input's
  // `.style` as the source for the second: `Object.assign(b.style, a.style)` enumerates the
  // CSSStyleDeclaration's indexed properties and throws
  // `TypeError: Failed to set an indexed property [0] ... setter is not supported`, which kills
  // this whole builder from inside its setInterval — so the panel silently never mounts. Both
  // `tsc --noEmit` and `eslint` accept it (a cast satisfies the type), so nothing but running
  // the page catches it.
  const wbtFujiInputStyle = {
    flex: '1',
    minWidth: '0',
    background: 'rgba(0,0,0,0.5)',
    color: '#fff',
    border: '1px solid rgba(255,255,255,0.25)',
    borderRadius: '4px',
    fontFamily: 'inherit',
    fontSize: '12px',
    padding: '3px 4px',
  } as Partial<CSSStyleDeclaration>;
  const anchorInput = document.createElement('input');
  anchorInput.type = 'number';
  anchorInput.min = '0';
  anchorInput.max = String(s.fujiNSamples - 1);
  anchorInput.value = String(s.fujiSample);
  Object.assign(anchorInput.style, wbtFujiInputStyle);
  const clampAnchor = (v: number): number => Math.min(s.fujiNSamples - 1, Math.max(0, v));
  const sendAnchor = (v: number): void => {
    const idx = clampAnchor(v);
    anchorInput.value = String(idx);
    request('sample', { index: idx });
  };
  anchorInput.addEventListener('change', () => sendAnchor(Number(anchorInput.value)));
  const mkStep = (text: string, fn: () => void): HTMLButtonElement => {
    const b = wbtFujiButton(text, 'rgba(98, 51, 235, 0.85)');
    Object.assign(b.style, { flex: '0 0 auto', padding: '3px 7px' } as Partial<CSSStyleDeclaration>);
    b.addEventListener('click', fn);
    return b;
  };
  // Steps move from what is TYPED, not from the planner's number, so repeated clicks walk the
  // list even while a switch is still settling (its hold pose is a ~1 s ramp).
  anchorRow.appendChild(anchorLabel);
  anchorRow.appendChild(mkStep('◀', () => sendAnchor(Number(anchorInput.value) - 1)));
  anchorRow.appendChild(anchorInput);
  anchorRow.appendChild(mkStep('▶', () => sendAnchor(Number(anchorInput.value) + 1)));

  const epRow = mkRow();
  const epLabel = document.createElement('span');
  epLabel.textContent = 'episode';
  Object.assign(epLabel.style, { opacity: '0.7', minWidth: '52px' } as Partial<CSSStyleDeclaration>);
  const epInput = document.createElement('input');
  epInput.type = 'number';
  epInput.min = '0';
  epInput.max = String(Math.max(0, s.fujiNEpisodes - 1));
  epInput.value = String(Math.max(0, s.fujiEpisodeIndex));
  Object.assign(epInput.style, wbtFujiInputStyle);
  const sendEpisode = (v: number): void => {
    const idx = Math.min(Math.max(0, s.fujiNEpisodes - 1), Math.max(0, v));
    epInput.value = String(idx);
    request('episode', { index: idx });
  };
  epInput.addEventListener('change', () => sendEpisode(Number(epInput.value)));
  epRow.appendChild(epLabel);
  epRow.appendChild(mkStep('◀', () => sendEpisode(Number(epInput.value) - 1)));
  epRow.appendChild(epInput);
  epRow.appendChild(mkStep('▶', () => sendEpisode(Number(epInput.value) + 1)));

  const where = document.createElement('div');
  Object.assign(where.style, {
    opacity: '0.85', fontSize: '11px', whiteSpace: 'pre-wrap',
  } as Partial<CSSStyleDeclaration>);

  // --- the verbs. Predict and Run are SEPARATE buttons on purpose: Predict stages a chunk and
  // publishes NOTHING (the violet ghost is a different wire channel), so the operator sees what
  // is about to happen before any of it reaches the controller.
  const runRow = mkRow();
  const bPredict = wbtFujiButton('Predict', 'rgba(98, 51, 235, 0.92)');
  const bRun = wbtFujiButton('Run', 'rgba(40, 140, 90, 0.92)');
  bPredict.addEventListener('click', () => request('predict'));
  bRun.addEventListener('click', () => request('run'));
  runRow.appendChild(bPredict);
  runRow.appendChild(bRun);

  const ctlRow = mkRow();
  const bAuto = wbtFujiButton('▶▶ auto', 'rgba(40, 110, 160, 0.92)');
  const bStop = wbtFujiButton('■ Stop', 'rgba(180, 50, 50, 0.95)');
  const bReset = wbtFujiButton('Reset', 'rgba(90, 90, 105, 0.92)');
  bAuto.addEventListener('click', () => request('auto'));
  // Stop does NOT go through `request`: it must not be dimmed waiting for an ack, because it is
  // the one button an operator may need to press twice in a second.
  bStop.addEventListener('click', () => { _wbtVrLiveStore.sendCmd({ cmd: 'fuji', action: 'stop' }); });
  bReset.addEventListener('click', () => request('reset'));
  ctlRow.appendChild(bAuto);
  ctlRow.appendChild(bStop);
  ctlRow.appendChild(bReset);

  const status = document.createElement('div');
  Object.assign(status.style, {
    opacity: '0.9', fontSize: '11px', whiteSpace: 'pre-wrap', lineHeight: '1.35',
  } as Partial<CSSStyleDeclaration>);

  let synced = false;
  setInterval(() => {
    const st = _wbtVrLiveStore;
    // Sync the two inputs to the planner's own cursor exactly ONCE (its --sample / --episode),
    // then never again: after the operator has typed, the value is theirs and the planner's
    // number belongs in the readout below.
    if (!synced && st.fujiEpisodeIndex >= 0) {
      synced = true;
      anchorInput.value = String(st.fujiSample);
      epInput.value = String(st.fujiEpisodeIndex);
    }
    if (pending >= 0 && st.fujiAck > pending) pending = -1;
    const busy = pending >= 0 || st.fujiState === 'running' || st.fujiState === 'predicting';
    for (const b of [bPredict, bRun, bAuto, bReset]) {
      b.disabled = busy;
      b.style.opacity = busy ? '0.45' : '1';
      b.style.cursor = busy ? 'default' : 'pointer';
    }
    layout.textContent = st.fujiLayout;
    // The two `episode` numbers are DIFFERENT things and both are shown: the input is the
    // episode's POSITION in the dataset's list (what a picker can walk), the `id` is its global
    // episode id (what a QC sheet and a log line name). They diverge whenever a take lost all
    // its anchors to the glitch filter, which on this session is 9 of 167.
    where.textContent = `anchor ${st.fujiSample} / ${st.fujiNSamples}\n`
      + `episode ${st.fujiEpisodeIndex} / ${st.fujiNEpisodes}  (id ${Math.max(st.fujiEpisode, 0)})`
      + `  frame ${st.fujiEpisodeFrame} of ${st.fujiEpisodeAnchors} anchors  ${st.fujiSplit}`;
    const step = st.fujiStep >= 0 && st.fujiExecSteps > 0
      ? `  step ${st.fujiStep + 1}/${st.fujiExecSteps}` : '';
    const chunk = st.fujiChunkSteps > 0 ? `  chunk ${st.fujiChunkSteps}` : '';
    status.textContent = `${st.fujiState}${chunk}${step}${pending >= 0 ? '  (sending…)' : ''}\n`
      + st.fujiMsg;
  }, 200);

  box.appendChild(title);
  box.appendChild(layout);
  box.appendChild(anchorRow);
  box.appendChild(epRow);
  box.appendChild(where);
  box.appendChild(runRow);
  box.appendChild(ctlRow);
  box.appendChild(status);
  document.body.appendChild(box);
  console.log(
    `[WbtFuji] offline-driver panel mounted: ${s.fujiNSamples} dataset anchors in `
    + `${s.fujiNEpisodes} episode(s). Predict STAGES (nothing is published; the light-purple `
    + 'ghost plays the chunk), Run publishes it one step per 30 Hz tick.',
  );
}

// vr_2point_pos fed live from the trackers (6D, pelvis frame, [L_xyz, R_xyz]).
export class WbtVr2PointPosLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 6);
    this.home = wbtVrLiveHome(config, 6);
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
    mountRecalibButton();
    // Self-gating: it only appears if the producer is a FUJI offline driver.
    mountFujiPanel();
  }
  protected computeCurrent(): Float32Array {
    return new Float32Array(_wbtVrLiveStore.pos ?? this.home);
  }
}

// vr_2point_orn fed live (12D rot6d, [L_rot6d, R_rot6d]). In the bridge's
// translation-first mode this stays at the default-pose home rot6d.
export class WbtVr2PointOrnLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 12);
    this.home = wbtVrLiveHome(config, 12);
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
  }
  protected computeCurrent(): Float32Array {
    return new Float32Array(_wbtVrLiveStore.orn ?? this.home);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// LIVE VIVE-tracker 3pt teleop (the 3pt hybrid UNIFIED anneal path — the sim
// twin of the rig's `MODE=3pt-unified-anneal`). Additive: the 2pt live classes
// above and every replay class are untouched.
//
// Wire (bridge --mode 3pt --hand-point tcp):
//     { seq, t, mode:"3pt", pos:[9], orn:[12], orientation:bool }
// pos = [L_xyz, R_xyz, Head_xyz] in the PELVIS frame, where each hand point is
// the 0.242 m FINGERTIP TCP (--hand-point tcp) and the head point is
// torso_link + (0,0,0.35). orn = 3x xyzw QUATERNION — the `vr_3point_orn` rep,
// NOT rot6d. Both are byte-for-byte what `_compute_vr_fk` produces for this
// preset, which is what makes the same bridge drive the sim and the robot.
//
// Both classes also PUBLISH what they fed onto the policy-extras bus, so
// WbtTrackingCommand can draw an RGB axis triad at each of the three commanded
// target poses (see `vr_live_target_triads`). Publishing the RESOLVED value —
// live frame or HOME fallback — is deliberate: the triad then shows what the
// policy actually saw, so a dead bridge reads as "triads parked at HOME"
// instead of a plausible-looking pose. The extras bus is the only channel
// available: WbtObservations.ts is inlined under core/observation/ and
// WbtTrackingCommand.ts under core/command/, so they cannot import each other.
// ────────────────────────────────────────────────────────────────────────────

const WBT_VR3_POINT_TAGS = ['l', 'r', 'h'] as const;
// 1 while a live 3pt frame is driving the obs, 0 while the HOME fallback is.
const WBT_VR3_LIVE_OK_KEY = 'wbt_vr3_live__ok';

function wbtPublishVr3TargetPos(pos: Float32Array, live: boolean): void {
  setPolicyExtra(WBT_VR3_LIVE_OK_KEY, live ? 1.0 : 0.0);
  for (let k = 0; k < 3; k++) {
    const tag = WBT_VR3_POINT_TAGS[k];
    setPolicyExtra(`wbt_vr3_tgt__${tag}_px`, pos[k * 3] ?? 0.0);
    setPolicyExtra(`wbt_vr3_tgt__${tag}_py`, pos[k * 3 + 1] ?? 0.0);
    setPolicyExtra(`wbt_vr3_tgt__${tag}_pz`, pos[k * 3 + 2] ?? 0.0);
  }
}

function wbtPublishVr3TargetOrn(orn: Float32Array): void {
  for (let k = 0; k < 3; k++) {
    const tag = WBT_VR3_POINT_TAGS[k];
    // xyzw, exactly as the wire carries it (and as vr_3point_orn is defined).
    setPolicyExtra(`wbt_vr3_tgt__${tag}_qx`, orn[k * 4] ?? 0.0);
    setPolicyExtra(`wbt_vr3_tgt__${tag}_qy`, orn[k * 4 + 1] ?? 0.0);
    setPolicyExtra(`wbt_vr3_tgt__${tag}_qz`, orn[k * 4 + 2] ?? 0.0);
    setPolicyExtra(`wbt_vr3_tgt__${tag}_qw`, orn[k * 4 + 3] ?? 1.0);
  }
}

// vr_3point_pos_tcp fed live (9D, pelvis frame, [L_xyz, R_xyz, Head_xyz]).
export class WbtVr3PointPosTcpLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 9);
    this.home = wbtVrLiveHome(config, 9);
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
    mountRecalibButton();
    // Self-gating: it only appears if the producer is a FUJI offline driver.
    mountFujiPanel();
  }
  protected computeCurrent(): Float32Array {
    const live = _wbtVrLiveStore.pos3;
    const out = new Float32Array(live ?? this.home);
    wbtPublishVr3TargetPos(out, live !== null);
    return out;
  }
}

// vr_3point_orn fed live (12D = 3x xyzw quaternion). With the bridge in
// translation-first mode (--orientation off) this stays at the default-pose
// home quats, so the wrists/torso hold the default orientation while position
// tracks.
export class WbtVr3PointOrnLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 12);
    this.home = wbtVrLiveHome(config, 12);
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
  }
  protected computeCurrent(): Float32Array {
    const out = new Float32Array(_wbtVrLiveStore.orn3 ?? this.home);
    wbtPublishVr3TargetOrn(out);
    return out;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// MOTION-COMPLETION live terms (bridge --mode 5pt / dense)
//
// Same three trackers, but the bridge routes them through Motion_Completion, so what
// arrives is a slice of a COMPLETED 29-DoF body. The five keypoints extend the 3pt set
// with both feet; the dense wire replaces keypoints entirely with joint targets.
//
// These carry a ~667 ms COMMAND LAG that belongs to the MC checkpoint (dense conditioning
// over a 30-frame window, 20-frame stride) — the bridge reports it per frame as
// `mc_lag_s` and it is republished on the extras bus so the HUD can show it. It is NOT a
// symptom of a slow browser: measured compute is ~45 ms/window, ~7% duty.
// ────────────────────────────────────────────────────────────────────────────

const WBT_VR5_POINT_TAGS = ['l', 'r', 'h', 'lf', 'rf'] as const;
const WBT_VR5_LIVE_OK_KEY = 'wbt_vr5_live__ok';
const WBT_MC_LAG_KEY = 'wbt_mc__lag_s';

function wbtPublishVr5TargetPos(pos: Float32Array, live: boolean): void {
  setPolicyExtra(WBT_VR5_LIVE_OK_KEY, live ? 1.0 : 0.0);
  setPolicyExtra(WBT_MC_LAG_KEY, _wbtVrLiveStore.mcLagS);
  for (let k = 0; k < 5; k++) {
    const tag = WBT_VR5_POINT_TAGS[k];
    setPolicyExtra(`wbt_vr5_tgt__${tag}_px`, pos[k * 3] ?? 0.0);
    setPolicyExtra(`wbt_vr5_tgt__${tag}_py`, pos[k * 3 + 1] ?? 0.0);
    setPolicyExtra(`wbt_vr5_tgt__${tag}_pz`, pos[k * 3 + 2] ?? 0.0);
  }
}

function wbtPublishVr5TargetOrn(orn: Float32Array): void {
  for (let k = 0; k < 5; k++) {
    const tag = WBT_VR5_POINT_TAGS[k];
    setPolicyExtra(`wbt_vr5_tgt__${tag}_qx`, orn[k * 4] ?? 0.0);
    setPolicyExtra(`wbt_vr5_tgt__${tag}_qy`, orn[k * 4 + 1] ?? 0.0);
    setPolicyExtra(`wbt_vr5_tgt__${tag}_qz`, orn[k * 4 + 2] ?? 0.0);
    setPolicyExtra(`wbt_vr5_tgt__${tag}_qw`, orn[k * 4 + 3] ?? 1.0);
  }
}

// vr_5point_pos_tcp fed live (15D, pelvis frame, [L, R, head, Lfoot, Rfoot] xyz).
// Slots [0:9] mean exactly what the 3pt term's do, so the two are comparable by eye.
export class WbtVr5PointPosTcpLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 15);
    this.home = wbtVrLiveHome(config, 15);
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
    mountRecalibButton();
    // Self-gating: it only appears if the producer is a FUJI offline driver.
    mountFujiPanel();
    // Self-gating: it only appears if the bridge turns out to be a recorded replay.
    mountEpisodePanel();
  }
  protected computeCurrent(): Float32Array {
    const live = _wbtVrLiveStore.pos5;
    const out = new Float32Array(live ?? this.home);
    wbtPublishVr5TargetPos(out, live !== null);
    return out;
  }
}

// vr_5point_orn fed live (20D = 5x xyzw quaternion).
export class WbtVr5PointOrnLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 20);
    this.home = wbtVrLiveHome(config, 20);
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
  }
  protected computeCurrent(): Float32Array {
    const out = new Float32Array(_wbtVrLiveStore.orn5 ?? this.home);
    wbtPublishVr5TargetOrn(out);
    return out;
  }
}

// `command` (58D = 29 joint targets ++ 29 joint velocities) fed live from the completion,
// for exp:g1-wbt-hybrid-unified-anneal. Replaces WbtCommand, which reads the baked clip.
//
// THE REMAP IS BY NAME. The wire carries MC's joint order and the policy's order comes
// from the bundle's `policy_joint_names`; both are 29 long and they are NOT the same
// order, so an index-aligned copy would scramble the body while looking perfectly
// well-formed. A name with no counterpart holds its HOME value and is reported ONCE.
export class WbtCommandLive extends WbtHistoryBase {
  private readonly home: Float32Array;
  private readonly policyNames: string[];
  private map: Int32Array | null = null;    // policy slot -> wire slot, built once
  private warned = false;
  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config, 58);
    this.home = wbtVrLiveHome(config, 58);
    const pn = (config as { policy_joint_names?: unknown }).policy_joint_names;
    this.policyNames = Array.isArray(pn) ? pn.map((x) => String(x)) : [];
    if (this.policyNames.length !== 29) {
      console.warn(
        '[WbtCommandLive] policy_joint_names missing/short — the dense wire cannot be '
        + 'remapped by name and will hold HOME. This is a build-time wiring bug.',
      );
    }
    _wbtVrLiveStore.ensureOpen(wbtVrLiveUrl(config));
    mountRecalibButton();
    // Self-gating: it only appears if the producer is a FUJI offline driver.
    mountFujiPanel();
    // Self-gating: it only appears if the bridge turns out to be a recorded replay.
    mountEpisodePanel();
  }

  private buildMap(wireNames: string[]): Int32Array {
    const idx = new Int32Array(29).fill(-1);
    const missing: string[] = [];
    for (let i = 0; i < 29; i++) {
      const j = wireNames.indexOf(this.policyNames[i] ?? '');
      idx[i] = j;
      if (j < 0) missing.push(this.policyNames[i] ?? `#${i}`);
    }
    if (missing.length > 0 && !this.warned) {
      this.warned = true;
      console.warn('[WbtCommandLive] joints absent from the wire (held at HOME):', missing);
    }
    return idx;
  }

  protected computeCurrent(): Float32Array {
    const jp = _wbtVrLiveStore.jointPos;
    const jv = _wbtVrLiveStore.jointVel;
    const names = _wbtVrLiveStore.jointNames;
    setPolicyExtra(WBT_MC_LAG_KEY, _wbtVrLiveStore.mcLagS);
    const out = new Float32Array(this.home);
    if (!jp || !jv || !names || this.policyNames.length !== 29) {
      setPolicyExtra(WBT_VR5_LIVE_OK_KEY, 0.0);
      return out;
    }
    if (this.map === null) this.map = this.buildMap(names);
    for (let i = 0; i < 29; i++) {
      const j = this.map[i];
      if (j < 0) continue;                 // no counterpart -> keep HOME for that joint
      out[i] = jp[j];
      out[29 + i] = jv[j];
    }
    setPolicyExtra(WBT_VR5_LIVE_OK_KEY, 1.0);
    return out;
  }
}

