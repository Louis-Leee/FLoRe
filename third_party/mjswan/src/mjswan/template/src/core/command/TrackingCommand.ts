import * as THREE from 'three';

import { getPosition, getQuaternion } from '../scene/scene';
import { type NpzEntry, loadNpz } from '../scene/npz';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type TrackingMotionConfig = {
  name: string;
  path: string;
  fps: number;
  anchor_body_name: string;
  body_names: string[];
  dataset_joint_names?: string[];
  default?: boolean;
  loop?: boolean;
};

type LoadedTrackingMotion = TrackingMotionConfig & {
  jointPos: Float32Array[];
  jointVel: Float32Array[];
  bodyPosW: Float32Array[];
  bodyQuatW: Float32Array[];
  bodyLinVelW: Float32Array[];
  bodyAngVelW: Float32Array[];
  frameCount: number;
};

type ScalarRange = [number, number];
type PoseRange = Partial<Record<'x' | 'y' | 'z' | 'roll' | 'pitch' | 'yaw', ScalarRange>>;

function normalizeQuat(quat: ArrayLike<number>): Float32Array {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return new Float32Array([
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ]);
}

function splitFrames(entry: NpzEntry): Float32Array[] {
  const totalFrames = entry.shape[0] ?? 0;
  const width = entry.shape.length <= 1 ? 1 : entry.shape.slice(1).reduce((acc, v) => acc * v, 1);
  const frames: Float32Array[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const out = new Float32Array(width);
    const start = i * width;
    for (let j = 0; j < width; j++) {
      out[j] = entry.data[start + j] ?? 0.0;
    }
    frames.push(out);
  }
  return frames;
}

function normalizeScalarRange(value: unknown, fallback: ScalarRange): ScalarRange {
  if (!Array.isArray(value) || value.length < 2) {
    return fallback;
  }
  const lo = Number(value[0]);
  const hi = Number(value[1]);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    return fallback;
  }
  return [lo, hi];
}

function normalizeRangeMap(value: unknown): PoseRange {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }
  const out: PoseRange = {};
  for (const key of ['x', 'y', 'z', 'roll', 'pitch', 'yaw'] as const) {
    const range = normalizeScalarRange((value as Record<string, unknown>)[key], [0.0, 0.0]);
    if (range[0] !== 0.0 || range[1] !== 0.0) {
      out[key] = range;
    }
  }
  return out;
}

function sampleRangeValue(range: ScalarRange | undefined): number {
  if (!range) {
    return 0.0;
  }
  return range[0] + Math.random() * (range[1] - range[0]);
}

function quatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const aw = a[0] ?? 1;
  const ax = a[1] ?? 0;
  const ay = a[2] ?? 0;
  const az = a[3] ?? 0;
  const bw = b[0] ?? 1;
  const bx = b[1] ?? 0;
  const by = b[2] ?? 0;
  const bz = b[3] ?? 0;
  return new Float32Array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ]);
}

function quatFromEulerXYZ(roll: number, pitch: number, yaw: number): Float32Array {
  const cr = Math.cos(roll * 0.5);
  const sr = Math.sin(roll * 0.5);
  const cp = Math.cos(pitch * 0.5);
  const sp = Math.sin(pitch * 0.5);
  const cy = Math.cos(yaw * 0.5);
  const sy = Math.sin(yaw * 0.5);
  return new Float32Array([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ]);
}

function setGhostMaterial(material: THREE.Material): THREE.Material {
  const next = material.clone();
  if ('transparent' in next) {
    next.transparent = true;
  }
  if ('opacity' in next) {
    next.opacity = 0.5;
  }
  if ('depthWrite' in next) {
    next.depthWrite = false;
  }
  if ('color' in next && next.color instanceof THREE.Color) {
    next.color = new THREE.Color(0.5, 0.7, 0.5);
  }
  return next;
}

function hasRenderableMesh(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      found = true;
    }
  });
  return found;
}

export class TrackingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly motions: TrackingMotionConfig[];
  private readonly loadedMotions: Map<string, LoadedTrackingMotion>;
  private sampleHz: number;
  private readonly ghostRoot: THREE.Group | null;
  private readonly ghostBodies: Map<number, THREE.Group>;
  // protected (not private) so a subclass (WbtTrackingCommand) can read the
  // reference-motion body world poses (ghostData.xpos/xquat, refreshed by
  // updateGhostPose each frame) to draw reference-side vr_3point markers on the
  // ghost. Read-only for the subclass — the base class owns all writes.
  protected readonly ghostData: import('mujoco').MjData | null;
  private refBodyPosW: Float32Array[];
  private refBodyQuatW: Float32Array[];
  private refBodyLinVelW: Float32Array[];
  private refBodyAngVelW: Float32Array[];
  private selectedMotionName: string | null;
  private selectedMotion: LoadedTrackingMotion | null;
  private selectedAnchorBodyIndex: number;
  private selectedRootBodyIndex: number;
  private datasetQposAdr: number[];
  private frameAccumulator: number;
  private justReset: boolean;
  // protected so the subclass can tie the reference-side vr_3point markers'
  // visibility to the ghost's reference-visible toggle (markers hide with the ghost).
  protected referenceVisible: boolean;
  private readonly samplingMode: string;
  private readonly poseRange: PoseRange;
  private readonly velocityRange: PoseRange;
  private readonly jointPositionRange: ScalarRange;
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  nJoints: number;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    context: CommandTermContext,
  ) {
    this.context = context;
    this.motions = Array.isArray(config.motions) ? config.motions as TrackingMotionConfig[] : [];
    this.loadedMotions = new Map();
    this.sampleHz = 50.0;
    this.selectedMotionName =
      this.motions.find((motion) => motion.default)?.name ??
      this.motions[0]?.name ??
      null;
    this.selectedMotion = null;
    this.selectedAnchorBodyIndex = 0;
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = [];
    this.frameAccumulator = 0.0;
    this.justReset = true;
    this.referenceVisible = true;
    this.samplingMode = typeof config.sampling_mode === 'string' ? config.sampling_mode : 'start';
    this.poseRange = normalizeRangeMap(config.pose_range);
    this.velocityRange = normalizeRangeMap(config.velocity_range);
    this.jointPositionRange = normalizeScalarRange(config.joint_position_range, [0.0, 0.0]);
    this.refJointPos = [];
    this.refRootPos = [];
    this.refRootQuat = [];
    this.refIdx = 0;
    this.refLen = 0;
    this.nJoints = this.motions.find((motion) => motion.name === this.selectedMotionName)?.dataset_joint_names?.length ?? 0;

    this.ghostBodies = new Map();
    this.ghostData = context.mjModel ? new context.mujoco.MjData(context.mjModel) : null;
    this.ghostRoot = this.createGhostRoot();
    this.refBodyPosW = [];
    this.refBodyQuatW = [];
    this.refBodyLinVelW = [];
    this.refBodyAngVelW = [];
  }

  getCommand(): Float32Array {
    if (!this.selectedMotion || this.refLen === 0) {
      return new Float32Array(this.nJoints * 2);
    }
    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(this.nJoints);
    const jointVel = this.selectedMotion.jointVel[this.refIdx] ?? new Float32Array(this.nJoints);
    const out = new Float32Array(jointPos.length + jointVel.length);
    out.set(jointPos, 0);
    out.set(jointVel, jointPos.length);
    return out;
  }

  getUiConfig(): CommandUiConfig | null {
    return null;
  }

  async setSelectedMotion(name: string | null): Promise<boolean> {
    if (name === null) {
      this.selectedMotionName = null;
      this.selectedMotion = null;
      this.refJointPos = [];
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      this.refLen = 0;
      this.nJoints = 0;
      this.updateGhostPose();
      return false;
    }

    const config = this.motions.find((motion) => motion.name === name);
    if (!config) {
      return false;
    }

    const loaded = this.loadedMotions.get(name) ?? await this.loadMotion(config);
    this.loadedMotions.set(name, loaded);
    this.selectedMotionName = name;
    this.selectedMotion = loaded;
    this.selectedAnchorBodyIndex = Math.max(
      0,
      loaded.body_names.indexOf(loaded.anchor_body_name),
    );
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = this.resolveQposAdr(loaded.dataset_joint_names ?? []);
    this.refLen = loaded.frameCount;
    this.refJointPos = loaded.jointPos;
    this.refIdx = this.sampleInitialFrame(this.refLen);
    this.nJoints = loaded.jointPos[0]?.length ?? 0;
    this.frameAccumulator = 0.0;
    this.justReset = true;
    this.updateReferenceState();
    this.applyReferenceStateToSim();
    this.updateGhostPose();
    return true;
  }

  setReferenceVisible(visible: boolean): void {
    this.referenceVisible = visible;
    if (this.ghostRoot) {
      this.ghostRoot.visible = visible && this.selectedMotion !== null;
    }
  }

  reset(): void {
    this.refIdx = this.sampleInitialFrame(this.refLen);
    this.frameAccumulator = 0.0;
    this.justReset = true;
    this.updateReferenceState();
    this.applyReferenceStateToSim();
    this.updateGhostPose();
  }

  update(dt: number): void {
    if (!this.selectedMotion || this.refLen <= 1) {
      return;
    }
    if (this.justReset) {
      this.justReset = false;
      this.updateGhostPose();
      return;
    }
    const shouldLoop = this.selectedMotion?.loop !== false;
    this.frameAccumulator += dt * this.sampleHz;
    let motionLooped = false;
    while (this.frameAccumulator >= 1.0) {
      this.refIdx += 1;
      if (this.refIdx >= this.refLen) {
        if (!shouldLoop) {
          this.refIdx = this.refLen - 1;
          this.frameAccumulator = 0.0;
          break;
        }
        this.refIdx = 0;
        motionLooped = true;
      }
      this.frameAccumulator -= 1.0;
    }
    if (motionLooped) {
      this.context.requestReset?.();
    }
    this.updateGhostPose();
  }

  updateDebugVisuals(): void {
    if (this.ghostRoot) {
      this.ghostRoot.visible = this.referenceVisible && this.selectedMotion !== null;
    }
  }

  dispose(): void {
    if (this.ghostRoot) {
      this.ghostRoot.parent?.remove(this.ghostRoot);
      this.ghostRoot.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            for (const material of obj.material) {
              material.dispose?.();
            }
          } else {
            obj.material?.dispose?.();
          }
        }
      });
    }
    this.ghostData?.delete?.();
  }

  isReady(): boolean {
    return this.selectedMotion !== null && this.refLen > 0;
  }

  getSelectedMotion(): LoadedTrackingMotion | null {
    return this.selectedMotion;
  }

  getSelectedMotionName(): string | null {
    return this.selectedMotionName;
  }

  getAnchorBodyName(): string | null {
    return this.selectedMotion?.anchor_body_name
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.anchor_body_name
      ?? null;
  }

  getBodyNames(): string[] {
    return this.selectedMotion?.body_names
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.body_names
      ?? [];
  }

  getAnchorBodyIndex(): number {
    return this.selectedAnchorBodyIndex;
  }

  getAnchorPos(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 3;
    return frame.slice(offset, offset + 3);
  }

  getAnchorQuat(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyQuatW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 4;
    return normalizeQuat(frame.slice(offset, offset + 4));
  }

  getBodyPosW(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    return frame ? frame.slice() : null;
  }

  private createGhostRoot(): THREE.Group | null {
    const bodies = this.context.bodies ?? null;
    const mjModel = this.context.mjModel;
    if (!bodies || !mjModel) {
      return null;
    }
    const root = new THREE.Group();
    root.name = 'Tracking Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
      const numericBodyId = Number(bodyId);
      if (!this.isDynamicBody(numericBodyId)) {
        continue;
      }
      const clone = body.clone(true);
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            obj.material = obj.material.map(setGhostMaterial);
          } else {
            obj.material = setGhostMaterial(obj.material);
          }
          obj.renderOrder = 2;
        }
      });
      if (!hasRenderableMesh(clone)) {
        continue;
      }
      this.ghostBodies.set(numericBodyId, clone);
      root.add(clone);
    }
    (this.context.mujocoRoot ?? this.context.scene).add(root);
    return root;
  }

  private isDynamicBody(bodyId: number): boolean {
    const mjModel = this.context.mjModel;
    if (!mjModel || bodyId <= 0 || bodyId >= mjModel.nbody) {
      return false;
    }

    let current = bodyId;
    while (current > 0) {
      if (mjModel.body_jntnum[current] > 0) {
        return true;
      }
      current = mjModel.body_parentid[current];
    }
    return false;
  }

  private async loadMotion(config: TrackingMotionConfig): Promise<LoadedTrackingMotion> {
    this.sampleHz = config.fps;
    const npz = await loadNpz(config.path);
    const required = ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'] as const;
    for (const key of required) {
      if (!npz[key]) {
        throw new Error(`Motion asset is missing '${key}'`);
      }
    }
    const jointPos = splitFrames(npz['joint_pos']!);
    const jointVel = splitFrames(npz['joint_vel']!);
    const sourceBodyNames = npz['body_names']?.strings ?? null;
    const bodyPosW = this.selectMotionBodyFrames(splitFrames(npz['body_pos_w']!), config.body_names, 3, sourceBodyNames);
    const bodyQuatW = this.selectMotionBodyFrames(splitFrames(npz['body_quat_w']!), config.body_names, 4, sourceBodyNames);
    const bodyLinVelW = this.selectMotionBodyFrames(splitFrames(npz['body_lin_vel_w']!), config.body_names, 3, sourceBodyNames);
    const bodyAngVelW = this.selectMotionBodyFrames(splitFrames(npz['body_ang_vel_w']!), config.body_names, 3, sourceBodyNames);
    return { ...config, jointPos, jointVel, bodyPosW, bodyQuatW, bodyLinVelW, bodyAngVelW, frameCount: jointPos.length };
  }

  private selectMotionBodyFrames(
    frames: Float32Array[],
    bodyNames: string[],
    stride: number,
    sourceBodyNames: string[] | null = null,
  ): Float32Array[] {
    const mjModel = this.context.mjModel;
    const first = frames[0];
    if (!mjModel || !first || bodyNames.length === 0) {
      return frames;
    }

    const sourceBodyCount = Math.floor(first.length / stride);
    if (sourceBodyCount === bodyNames.length) {
      return frames;
    }

    let bodySourceIndices: number[];
    if (sourceBodyNames !== null) {
      // Use the npz's own body-name manifest for unambiguous index lookup.
      bodySourceIndices = bodyNames.map((name) => sourceBodyNames.indexOf(name));
    } else {
      // Fall back: assume source bodies are laid out in mjModel body-ID order
      // starting from the first body in body_names.
      const rootBodyId = this.findBodyIdByName(bodyNames[0]);
      const bodyIds = bodyNames.map((name) => this.findBodyIdByName(name));
      bodySourceIndices = bodyIds.map((id) => id - rootBodyId);
    }

    if (bodySourceIndices.some((idx) => idx < 0 || idx >= sourceBodyCount)) {
      console.warn('[TrackingCommand] Could not map all motion body names to source body indices; using raw body frames.');
      return frames;
    }

    return frames.map((frame) => {
      const selected = new Float32Array(bodyNames.length * stride);
      for (let i = 0; i < bodySourceIndices.length; i++) {
        const sourceOffset = bodySourceIndices[i] * stride;
        const targetOffset = i * stride;
        for (let j = 0; j < stride; j++) {
          selected[targetOffset + j] = frame[sourceOffset + j] ?? 0.0;
        }
      }
      return selected;
    });
  }

  private updateReferenceState(): void {
    const motion = this.selectedMotion;
    if (!motion || motion.frameCount === 0) {
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      return;
    }

    this.refBodyPosW = motion.bodyPosW.map((frame) => frame.slice());
    this.refBodyQuatW = motion.bodyQuatW.map((frame) => frame.slice());
    this.refBodyLinVelW = motion.bodyLinVelW.map((frame) => frame.slice());
    this.refBodyAngVelW = motion.bodyAngVelW.map((frame) => frame.slice());
    this.refRootPos = this.refBodyPosW.map((frame) =>
      frame.slice(this.selectedRootBodyIndex * 3, this.selectedRootBodyIndex * 3 + 3),
    );
    this.refRootQuat = this.refBodyQuatW.map((frame) =>
      normalizeQuat(frame.slice(this.selectedRootBodyIndex * 4, this.selectedRootBodyIndex * 4 + 4)),
    );
  }

  private applyReferenceStateToSim(): void {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    const motion = this.selectedMotion;
    if (!mjModel || !mjData || !motion || this.refLen === 0) {
      return;
    }

    const rootPos = this.sampleRootPos(this.refIdx);
    const rootQuat = this.sampleRootQuat(this.refIdx);
    const freeJointIndex = this.findFreeJointIndex();
    if (rootPos && rootQuat && freeJointIndex >= 0) {
      const qposAdr = mjModel.jnt_qposadr[freeJointIndex];
      const qvelAdr = mjModel.jnt_dofadr[freeJointIndex];
      mjData.qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      mjData.qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      mjData.qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      mjData.qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      mjData.qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      mjData.qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      mjData.qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;

      const linVel = this.sampleRootVelocity(this.refIdx, this.refBodyLinVelW);
      const angVel = this.sampleRootAngularVelocity(this.refIdx);
      if (linVel && angVel) {
        mjData.qvel[qvelAdr + 0] = linVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 1] = linVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 2] = linVel[2] ?? 0.0;
        mjData.qvel[qvelAdr + 3] = angVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 4] = angVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 5] = angVel[2] ?? 0.0;
      }
    }

    const jointPos = this.sampleJointPos(this.refIdx);
    const jointVel = motion.jointVel[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      mjData.qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }
    for (let i = 0; i < this.datasetQposAdr.length && i < jointVel.length; i++) {
      const dofAdr = this.resolveQvelAdrForQposAdr(this.datasetQposAdr[i]);
      if (dofAdr >= 0) {
        mjData.qvel[dofAdr] = jointVel[i] ?? 0.0;
      }
    }

    this.context.mujoco.mj_forward(mjModel, mjData);
  }

  private sampleRootPos(frameIndex: number): Float32Array | null {
    const rootPos = this.refRootPos[frameIndex];
    if (!rootPos) {
      return null;
    }
    const sampled = rootPos.slice();
    sampled[0] += sampleRangeValue(this.poseRange.x);
    sampled[1] += sampleRangeValue(this.poseRange.y);
    sampled[2] += sampleRangeValue(this.poseRange.z);
    return sampled;
  }

  private sampleRootQuat(frameIndex: number): Float32Array | null {
    const rootQuat = this.refRootQuat[frameIndex];
    if (!rootQuat) {
      return null;
    }
    const roll = sampleRangeValue(this.poseRange.roll);
    const pitch = sampleRangeValue(this.poseRange.pitch);
    const yaw = sampleRangeValue(this.poseRange.yaw);
    if (roll === 0.0 && pitch === 0.0 && yaw === 0.0) {
      return rootQuat;
    }
    return normalizeQuat(quatMultiply(quatFromEulerXYZ(roll, pitch, yaw), rootQuat));
  }

  private sampleRootVelocity(frameIndex: number, source: Float32Array[]): Float32Array | null {
    const rootVel = source[frameIndex]?.slice(
      this.selectedRootBodyIndex * 3,
      this.selectedRootBodyIndex * 3 + 3,
    );
    if (!rootVel) {
      return null;
    }
    rootVel[0] += sampleRangeValue(this.velocityRange.x);
    rootVel[1] += sampleRangeValue(this.velocityRange.y);
    rootVel[2] += sampleRangeValue(this.velocityRange.z);
    return rootVel;
  }

  private sampleRootAngularVelocity(frameIndex: number): Float32Array | null {
    const rootVel = this.refBodyAngVelW[frameIndex]?.slice(
      this.selectedRootBodyIndex * 3,
      this.selectedRootBodyIndex * 3 + 3,
    );
    if (!rootVel) {
      return null;
    }
    rootVel[0] += sampleRangeValue(this.velocityRange.roll);
    rootVel[1] += sampleRangeValue(this.velocityRange.pitch);
    rootVel[2] += sampleRangeValue(this.velocityRange.yaw);
    return rootVel;
  }

  private sampleInitialFrame(frameCount: number): number {
    if (frameCount <= 1 || this.samplingMode === 'start') {
      return 0;
    }
    if (this.samplingMode === 'uniform') {
      return Math.floor(Math.random() * frameCount);
    }
    return 0;
  }

  private sampleJointPos(frameIndex: number): Float32Array {
    const jointPos = this.refJointPos[frameIndex] ?? new Float32Array(0);
    if (this.jointPositionRange[0] === 0.0 && this.jointPositionRange[1] === 0.0) {
      return jointPos;
    }
    const sampled = jointPos.slice();
    for (let i = 0; i < sampled.length; i++) {
      sampled[i] += sampleRangeValue(this.jointPositionRange);
    }
    return sampled;
  }

  private resolveQposAdr(jointNames: string[]): number[] {
    const mjModel = this.context.mjModel;
    if (!mjModel || jointNames.length === 0) {
      return [];
    }
    const resolved: number[] = [];
    for (const jointName of jointNames) {
      let adr = -1;
      for (let j = 0; j < mjModel.njnt; j++) {
        const modelJointName = mjModel.jnt(j).name;
        if (modelJointName === jointName || modelJointName.endsWith(`/${jointName}`)) {
          adr = mjModel.jnt_qposadr[j];
          break;
        }
      }
      if (adr >= 0) {
        resolved.push(adr);
      }
    }
    return resolved;
  }

  private resolveQvelAdrForQposAdr(qposAdr: number): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let j = 0; j < mjModel.njnt; j++) {
      if (mjModel.jnt_qposadr[j] === qposAdr) {
        return mjModel.jnt_dofadr[j];
      }
    }
    return -1;
  }

  private findBodyIdByName(bodyName: string): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let b = 0; b < mjModel.nbody; b++) {
      const name = mjModel.body(b).name;
      if (name === bodyName || name.endsWith(`/${bodyName}`)) {
        return b;
      }
    }
    return -1;
  }

  private updateGhostPose(): void {
    if (!this.ghostRoot || !this.ghostData || !this.context.mjModel || !this.selectedMotion || !this.refLen) {
      if (this.ghostRoot) {
        this.ghostRoot.visible = false;
      }
      return;
    }

    const qpos = this.ghostData.qpos;
    qpos.set(this.context.mjModel.qpos0);

    const rootPos = this.refRootPos[this.refIdx];
    const rootQuat = this.refRootQuat[this.refIdx];
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex >= 0 && rootPos && rootQuat) {
      const qposAdr = this.context.mjModel.jnt_qposadr[freeJointIndex];
      qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;
    }

    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }

    this.context.mujoco.mj_forward(this.context.mjModel, this.ghostData);

    for (const [bodyId, body] of this.ghostBodies) {
      getPosition(this.ghostData.xpos, bodyId, body.position);
      getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
    }
    this.ghostRoot.visible = this.referenceVisible;
  }

  private findFreeJointIndex(): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) {
        return i;
      }
    }
    return -1;
  }
}
