import * as THREE from 'three';
import type { MjModel } from 'mujoco';
import { ObservationBase } from './ObservationBase';
import type { ObservationConfig } from './ObservationBase';
import { CustomObservations } from './custom_observations';
import {
  clampFutureIndices,
  normalizeQuat,
  quatApplyInv,
  quatInverse,
  quatMultiply,
  quatToRot6d,
} from './math';
import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';
import { getCommandManager } from '../command';
import {
  VelocityCommandObservation,
  VelocityCommandWithOscillatorsObservation,
  GeneratedCommandsObservation,
} from './CommandObservation';
import { TrackingCommand } from '../command/TrackingCommand';

type TrackingSource = {
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  nJoints: number;
  isReady(): boolean;
  getAnchorPos(frameIndex?: number): Float32Array | null;
  getAnchorQuat(frameIndex?: number): Float32Array | null;
  getAnchorBodyName(): string | null;
  getBodyNames(): string[];
};

function getTrackingContext(_runner: PolicyRunner): TrackingSource | null {
  const command = getCommandManager().getTerm('motion');
  return command instanceof TrackingCommand ? command : null;
}

function getBodyIdByName(mjModel: MjModel | null, bodyName: string): number {
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

function normalizeScale(scale: unknown, size: number, fallback = 1.0): Float32Array | null {
  if (typeof scale === 'number') {
    const values = new Float32Array(size);
    values.fill(scale);
    return values;
  }
  if (Array.isArray(scale)) {
    const values = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const value = scale[i];
      values[i] = typeof value === 'number' ? value : fallback;
    }
    return values;
  }
  return null;
}

function normalizeVector(values: unknown, size: number, fallback = 0.0): Float32Array | null {
  if (!Array.isArray(values)) {
    return null;
  }
  const output = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    output[i] = typeof values[i] === 'number' ? values[i] : fallback;
  }
  return output;
}

export class BootIndicator extends ObservationBase {
  get size(): number {
    return 1;
  }

  compute(): Float32Array {
    return new Float32Array([0.0]);
  }
}

export class RootAngVelB extends ObservationBase {
  get size(): number {
    return 3;
  }

  compute(state: PolicyState): Float32Array {
    const value = state.rootAngVel ?? new Float32Array(3);
    return new Float32Array(value);
  }
}

export class ProjectedGravityB extends ObservationBase {
  private gravity: THREE.Vector3;
  private historySteps: number;
  private history: Float32Array[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const gravity = Array.isArray(config.gravity) ? config.gravity : [0, 0, -1];
    this.gravity = new THREE.Vector3(gravity[0] ?? 0, gravity[1] ?? 0, gravity[2] ?? -1);
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));
    this.history = Array.from({ length: this.historySteps }, () => new Float32Array(3));
  }

  get size(): number {
    return 3 * this.historySteps;
  }

  reset(state?: PolicyState): void {
    const value = this.computeCurrent(state);
    for (const buffer of this.history) {
      buffer.set(value);
    }
  }

  update(state: PolicyState): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
  }

  compute(): Float32Array {
    if (this.historySteps === 1) {
      return new Float32Array(this.history[0]);
    }
    const output = new Float32Array(this.size);
    let offset = 0;
    for (const buffer of this.history) {
      output.set(buffer, offset);
      offset += buffer.length;
    }
    return output;
  }

  private computeCurrent(state?: PolicyState): Float32Array {
    const quat = state?.rootQuat ?? new Float32Array([1, 0, 0, 0]);
    const quatObj = new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]);
    const gravityLocal = this.gravity.clone().applyQuaternion(quatObj.clone().invert());
    return new Float32Array([gravityLocal.x, gravityLocal.y, gravityLocal.z]);
  }
}

export class JointPos extends ObservationBase {
  private posSteps: number[];
  private numJoints: number;
  private maxStep: number;
  private history: Float32Array[];
  private subtractDefault: boolean;
  private biased: boolean;
  private defaultJointPos: Float32Array;
  private encoderBias: Float32Array;
  private scale: Float32Array | null;
  private qposAdr: number[] | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    let posSteps: number[];
    if (Array.isArray(config.pos_steps)) {
      posSteps = config.pos_steps.map((value: number) => Math.max(0, Math.floor(value)));
    } else if (typeof config.history_steps === 'number') {
      const steps = Math.max(1, Math.floor(config.history_steps));
      posSteps = Array.from({ length: steps }, (_, idx) => idx);
    } else {
      posSteps = [0, 1, 2, 3, 4, 8];
    }
    this.posSteps = posSteps;
    const jointNames = this.getJointNames(config);
    this.qposAdr = jointNames ? this.resolveQposAdr(jointNames) : null;
    this.numJoints = jointNames?.length ?? (
      typeof config.num_joints === 'number'
        ? Math.max(1, Math.floor(config.num_joints))
        : runner.getNumActions()
    );
    this.maxStep = Math.max(...this.posSteps);
    this.history = Array.from({ length: this.maxStep + 1 }, () => new Float32Array(this.numJoints));
    this.subtractDefault = Boolean(config.subtract_default);
    this.biased = Boolean(config.biased);
    this.defaultJointPos = this.normalizeDefaultJointPos(config.default_joint_pos);
    this.encoderBias = this.normalizeEncoderBias(config.encoder_bias);
    this.scale = this.normalizeScale(config.scale);
  }

  get size(): number {
    return this.posSteps.length * this.numJoints;
  }

  reset(state?: PolicyState): void {
    this.history[0].set(this.computeCurrent(state));
    for (let i = 1; i < this.history.length; i++) {
      this.history[i].set(this.history[0]);
    }
  }

  update(state: PolicyState): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
  }

  compute(): Float32Array {
    const out = new Float32Array(this.posSteps.length * this.numJoints);
    let offset = 0;
    for (const step of this.posSteps) {
      const idx = Math.min(step, this.history.length - 1);
      const source = this.history[idx];
      for (let j = 0; j < this.numJoints; j++) {
        let value = source[j];
        if (this.subtractDefault && this.defaultJointPos.length > j) {
          value -= this.defaultJointPos[j];
        }
        if (this.scale && this.scale.length > j) {
          value *= this.scale[j];
        }
        out[offset + j] = value;
      }
      offset += this.numJoints;
    }
    return out;
  }

  private computeCurrent(state?: PolicyState): Float32Array {
    if (this.qposAdr !== null) {
      const qpos = this.runner.getContext()?.mjData?.qpos;
      const out = new Float32Array(this.numJoints);
      for (let i = 0; i < this.numJoints; i++) {
        const adr = this.qposAdr[i];
        out[i] = qpos !== undefined ? qpos[adr] : 0.0;
        if (this.biased && this.encoderBias.length > i) {
          out[i] += this.encoderBias[i];
        }
      }
      return out;
    }
    const source = state?.jointPos ?? new Float32Array(this.numJoints);
    if (!this.biased) {
      return source;
    }
    const out = new Float32Array(this.numJoints);
    for (let i = 0; i < this.numJoints; i++) {
      out[i] = (source[i] ?? 0.0) + (this.encoderBias[i] ?? 0.0);
    }
    return out;
  }

  private getJointNames(config: ObservationConfig): string[] | null {
    if (!Array.isArray(config.joint_names)) {
      return null;
    }
    const names = config.joint_names.filter((value): value is string => typeof value === 'string');
    return names.length > 0 ? names : null;
  }

  private normalizeDefaultJointPos(values: unknown): Float32Array {
    const explicit = normalizeVector(values, this.numJoints, 0.0);
    if (explicit) {
      return explicit;
    }
    const defaults = this.runner.getDefaultJointPos();
    const output = new Float32Array(this.numJoints);
    for (let i = 0; i < this.numJoints; i++) {
      output[i] = defaults[i] ?? 0.0;
    }
    return output;
  }

  private normalizeEncoderBias(values: unknown): Float32Array {
    const explicit = normalizeVector(values, this.numJoints, 0.0);
    if (explicit) {
      return explicit;
    }
    const bias = this.runner.getEncoderBias();
    const output = new Float32Array(this.numJoints);
    for (let i = 0; i < this.numJoints; i++) {
      output[i] = bias[i] ?? 0.0;
    }
    return output;
  }

  private resolveQposAdr(jointNames: string[]): number[] {
    const mjModel = this.runner.getContext()?.mjModel ?? null;
    if (mjModel === null) {
      return Array.from({ length: jointNames.length }, () => 0);
    }
    const names = this.getModelJointNames(mjModel);
    return jointNames.map((jointName) => {
      const idx = names.indexOf(jointName);
      if (idx < 0) {
        throw new Error(`JointPos: joint "${jointName}" not found in model`);
      }
      return mjModel.jnt_qposadr[idx];
    });
  }

  private getModelJointNames(mjModel: MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let j = 0; j < mjModel.njnt; j++) {
      let start = mjModel.name_jntadr[j];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }

  private normalizeScale(scale: unknown): Float32Array | null {
    if (typeof scale === 'number') {
      const values = new Float32Array(this.numJoints);
      values.fill(scale);
      return values;
    }
    if (Array.isArray(scale)) {
      const values = new Float32Array(this.numJoints);
      for (let i = 0; i < this.numJoints; i++) {
        const value = scale[i];
        values[i] = typeof value === 'number' ? value : 1.0;
      }
      return values;
    }
    return null;
  }
}

export class TrackingCommandObsRaw extends ObservationBase {
  private futureSteps: number[];
  private outputLength: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = Array.isArray(config.future_steps)
      ? config.future_steps.map((value: number) => Math.max(0, Math.floor(value)))
      : [0, 2, 4, 8, 16];
    const nFut = this.futureSteps.length;
    this.outputLength = (nFut - 1) * 3 + nFut * 6;
  }

  get size(): number {
    return this.outputLength;
  }

  compute(state: PolicyState): Float32Array {
    const tracking = getTrackingContext(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.outputLength);
    }

    const baseIdx = tracking.refIdx;
    const refLen = tracking.refLen;
    const indices = clampFutureIndices(baseIdx, this.futureSteps, refLen);

    const basePos = tracking.refRootPos[indices[0]];
    const baseQuat = normalizeQuat(tracking.refRootQuat[indices[0]]);

    const posDiff: number[] = [];
    for (let i = 1; i < indices.length; i++) {
      const pos = tracking.refRootPos[indices[i]];
      const diff = [pos[0] - basePos[0], pos[1] - basePos[1], pos[2] - basePos[2]];
      const diffB = quatApplyInv(baseQuat, diff);
      posDiff.push(diffB[0], diffB[1], diffB[2]);
    }

    const qCur = normalizeQuat(state.rootQuat ?? [1, 0, 0, 0]);
    const qCurInv = quatInverse(qCur);

    const rot6d: number[] = [];
    for (let i = 0; i < indices.length; i++) {
      const refQuat = normalizeQuat(tracking.refRootQuat[indices[i]]);
      const rel = quatMultiply(qCurInv, refQuat);
      const r6 = quatToRot6d(rel);
      rot6d.push(r6[0], r6[1], r6[2], r6[3], r6[4], r6[5]);
    }

    return Float32Array.from([...posDiff, ...rot6d]);
  }
}

export class TargetRootZObs extends ObservationBase {
  private futureSteps: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = Array.isArray(config.future_steps)
      ? config.future_steps.map((value: number) => Math.max(0, Math.floor(value)))
      : [0, 2, 4, 8, 16];
  }

  get size(): number {
    return this.futureSteps.length;
  }

  compute(): Float32Array {
    const tracking = getTrackingContext(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      out[i] = tracking.refRootPos[indices[i]][2];
    }
    return out;
  }
}

export class TargetJointPosObs extends ObservationBase {
  private futureSteps: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = Array.isArray(config.future_steps)
      ? config.future_steps.map((value: number) => Math.max(0, Math.floor(value)))
      : [0, 2, 4, 8, 16];
  }

  get size(): number {
    const tracking = getTrackingContext(this.runner);
    const nJoints = tracking?.nJoints ?? 0;
    return this.futureSteps.length * nJoints;
  }

  compute(): Float32Array {
    const tracking = getTrackingContext(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length * tracking.nJoints);
    let offset = 0;
    for (const idx of indices) {
      out.set(tracking.refJointPos[idx], offset);
      offset += tracking.nJoints;
    }
    return out;
  }
}

export class TargetProjectedGravityBObs extends ObservationBase {
  private futureSteps: number[];

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.futureSteps = Array.isArray(config.future_steps)
      ? config.future_steps.map((value: number) => Math.max(0, Math.floor(value)))
      : [0, 2, 4, 8, 16];
  }

  get size(): number {
    return this.futureSteps.length * 3;
  }

  compute(): Float32Array {
    const tracking = getTrackingContext(this.runner);
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(this.size);
    }
    const indices = clampFutureIndices(tracking.refIdx, this.futureSteps, tracking.refLen);
    const out = new Float32Array(indices.length * 3);
    const gravity = [0.0, 0.0, -1.0];
    let offset = 0;
    for (const idx of indices) {
      const quat = normalizeQuat(tracking.refRootQuat[idx]);
      const gLocal = quatApplyInv(quat, gravity);
      out[offset++] = gLocal[0];
      out[offset++] = gLocal[1];
      out[offset++] = gLocal[2];
    }
    return out;
  }
}

export class MotionAnchorPosB extends ObservationBase {
  get size(): number {
    return 3;
  }

  compute(_state: PolicyState): Float32Array {
    const tracking = getTrackingContext(this.runner);
    const context = this.runner.getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(3);
    }
    const anchorPos = tracking.getAnchorPos();
    const anchorQuat = tracking.getAnchorQuat();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorPos || !anchorQuat) {
      return new Float32Array(3);
    }
    if (!anchorName || !mjModel || !mjData) {
      return new Float32Array(3);
    }
    const currentAnchorId = getBodyIdByName(mjModel, anchorName);
    if (currentAnchorId < 0) {
      return new Float32Array(3);
    }
    const currentPos = mjData.xpos.slice(currentAnchorId * 3, currentAnchorId * 3 + 3);
    const currentQuat = normalizeQuat(mjData.xquat.slice(currentAnchorId * 4, currentAnchorId * 4 + 4));
    const diff = [
      anchorPos[0] - currentPos[0],
      anchorPos[1] - currentPos[1],
      anchorPos[2] - currentPos[2],
    ];
    const diffB = quatApplyInv(currentQuat, diff);
    return Float32Array.from(diffB);
  }
}

export class MotionAnchorOriB extends ObservationBase {
  get size(): number {
    return 6;
  }

  compute(_state: PolicyState): Float32Array {
    const tracking = getTrackingContext(this.runner);
    const context = this.runner.getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!tracking || !tracking.isReady()) {
      return new Float32Array(6);
    }
    const anchorQuat = tracking.getAnchorQuat();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorQuat) {
      return new Float32Array(6);
    }
    if (!anchorName || !mjModel || !mjData) {
      return new Float32Array(6);
    }
    const currentAnchorId = getBodyIdByName(mjModel, anchorName);
    if (currentAnchorId < 0) {
      return new Float32Array(6);
    }
    const currentQuat = normalizeQuat(mjData.xquat.slice(currentAnchorId * 4, currentAnchorId * 4 + 4));
    const rel = quatMultiply(quatInverse(currentQuat), normalizeQuat(anchorQuat));
    return Float32Array.from(quatToRot6d(rel));
  }
}

export class RobotBodyPosB extends ObservationBase {
  get size(): number {
    const tracking = getTrackingContext(this.runner);
    return tracking ? tracking.getBodyNames().length * 3 : 0;
  }

  compute(): Float32Array {
    const tracking = getTrackingContext(this.runner);
    const context = this.runner.getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!tracking || !tracking.isReady() || !mjModel || !mjData) {
      return new Float32Array(this.size);
    }
    const bodyNames = tracking.getBodyNames();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorName) {
      return new Float32Array(this.size);
    }
    const anchorId = getBodyIdByName(mjModel, anchorName);
    if (anchorId < 0) {
      return new Float32Array(this.size);
    }
    const anchorPos = mjData.xpos.slice(anchorId * 3, anchorId * 3 + 3);
    const anchorQuat = normalizeQuat(mjData.xquat.slice(anchorId * 4, anchorId * 4 + 4));
    const out = new Float32Array(bodyNames.length * 3);
    let offset = 0;
    for (const bodyName of bodyNames) {
      const bodyId = getBodyIdByName(mjModel, bodyName);
      if (bodyId < 0) {
        offset += 3;
        continue;
      }
      const pos = mjData.xpos.slice(bodyId * 3, bodyId * 3 + 3);
      const diff = [
        pos[0] - anchorPos[0],
        pos[1] - anchorPos[1],
        pos[2] - anchorPos[2],
      ];
      const local = quatApplyInv(anchorQuat, diff);
      out.set(local, offset);
      offset += 3;
    }
    return out;
  }
}

export class RobotBodyOriB extends ObservationBase {
  get size(): number {
    const tracking = getTrackingContext(this.runner);
    return tracking ? tracking.getBodyNames().length * 6 : 0;
  }

  compute(): Float32Array {
    const tracking = getTrackingContext(this.runner);
    const context = this.runner.getContext();
    const mjModel = context?.mjModel ?? null;
    const mjData = context?.mjData ?? null;
    if (!tracking || !tracking.isReady() || !mjModel || !mjData) {
      return new Float32Array(this.size);
    }
    const bodyNames = tracking.getBodyNames();
    const anchorName = tracking.getAnchorBodyName();
    if (!anchorName) {
      return new Float32Array(this.size);
    }
    const anchorId = getBodyIdByName(mjModel, anchorName);
    if (anchorId < 0) {
      return new Float32Array(this.size);
    }
    const anchorQuat = normalizeQuat(mjData.xquat.slice(anchorId * 4, anchorId * 4 + 4));
    const anchorInv = quatInverse(anchorQuat);
    const out = new Float32Array(bodyNames.length * 6);
    let offset = 0;
    for (const bodyName of bodyNames) {
      const bodyId = getBodyIdByName(mjModel, bodyName);
      if (bodyId < 0) {
        offset += 6;
        continue;
      }
      const bodyQuat = normalizeQuat(mjData.xquat.slice(bodyId * 4, bodyId * 4 + 4));
      const localQuat = quatMultiply(anchorInv, bodyQuat);
      out.set(quatToRot6d(localQuat), offset);
      offset += 6;
    }
    return out;
  }
}

export class PrevActions extends ObservationBase {
  private steps: number;
  private numActions: number;
  private actionBuffer: Float32Array[];
  private transpose: boolean;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const history = typeof config.history_steps === 'number' ? config.history_steps : 4;
    this.steps = Math.max(1, Math.floor(history));
    this.numActions = runner.getNumActions();
    this.actionBuffer = Array.from({ length: this.steps }, () => new Float32Array(this.numActions));
    this.transpose = Boolean(config.transpose);
  }

  get size(): number {
    return this.steps * this.numActions;
  }

  reset(): void {
    for (const buffer of this.actionBuffer) {
      buffer.fill(0.0);
    }
  }

  update(): void {
    for (let i = this.actionBuffer.length - 1; i > 0; i--) {
      this.actionBuffer[i].set(this.actionBuffer[i - 1]);
    }
    const source = this.runner.getLastActions();
    this.actionBuffer[0].set(source);
  }

  compute(): Float32Array {
    const flattened = new Float32Array(this.steps * this.numActions);
    if (this.transpose) {
      for (let j = 0; j < this.numActions; j++) {
        for (let i = 0; i < this.steps; i++) {
          flattened[j * this.steps + i] = this.actionBuffer[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.steps; i++) {
        for (let j = 0; j < this.numActions; j++) {
          flattened[i * this.numActions + j] = this.actionBuffer[i][j];
        }
      }
    }
    return flattened;
  }
}

export class BaseLinearVelocity extends ObservationBase {
  private historySteps: number;
  private history: Float32Array[];
  private scale: Float32Array | null;
  private worldFrame: boolean;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));
    this.history = Array.from({ length: this.historySteps }, () => new Float32Array(3));
    this.scale = normalizeScale(config.scale, 3, 1.0);
    this.worldFrame = (config.world_frame as boolean | undefined) ?? true;
  }

  get size(): number {
    return 3 * this.historySteps;
  }

  reset(state?: PolicyState): void {
    const value = this.computeCurrent(state);
    for (const buffer of this.history) {
      buffer.set(value);
    }
  }

  update(state: PolicyState): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
  }

  compute(): Float32Array {
    if (this.historySteps === 1) {
      return new Float32Array(this.history[0]);
    }
    const output = new Float32Array(this.size);
    let offset = 0;
    for (const buffer of this.history) {
      output.set(buffer, offset);
      offset += buffer.length;
    }
    return output;
  }

  private computeCurrent(state?: PolicyState): Float32Array {
    const value = state?.rootLinVel ?? new Float32Array(3);
    let out = new Float32Array(value);
    if (!this.worldFrame) {
      const quat = normalizeQuat(state?.rootQuat ?? [1, 0, 0, 0]);
      const rotated = quatApplyInv(quat, out);
      out = new Float32Array(rotated);
    }
    if (this.scale) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= this.scale[i] ?? 1.0;
      }
    }
    return out;
  }
}

export class BaseAngularVelocity extends ObservationBase {
  private historySteps: number;
  private history: Float32Array[];
  private scale: Float32Array | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));
    this.history = Array.from({ length: this.historySteps }, () => new Float32Array(3));
    this.scale = normalizeScale(config.scale, 3, 1.0);
  }

  get size(): number {
    return 3 * this.historySteps;
  }

  reset(state?: PolicyState): void {
    const value = this.computeCurrent(state);
    for (const buffer of this.history) {
      buffer.set(value);
    }
  }

  update(state: PolicyState): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
  }

  compute(): Float32Array {
    if (this.historySteps === 1) {
      return new Float32Array(this.history[0]);
    }
    const output = new Float32Array(this.size);
    let offset = 0;
    for (const buffer of this.history) {
      output.set(buffer, offset);
      offset += buffer.length;
    }
    return output;
  }

  private computeCurrent(state?: PolicyState): Float32Array {
    const value = state?.rootAngVel ?? new Float32Array(3);
    const out = new Float32Array(value);
    if (this.scale) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= this.scale[i] ?? 1.0;
      }
    }
    return out;
  }
}

export class JointVelocities extends ObservationBase {
  private numJoints: number;
  private historySteps: number;
  private history: Float32Array[];
  private scale: Float32Array | null;
  private qvelAdr: number | null;
  private qvelAdrList: number[] | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const jointNames = Array.isArray(config.joint_names)
      ? config.joint_names.filter((value): value is string => typeof value === 'string')
      : null;
    const jointName = config.joint_name as string | undefined;
    if (jointNames && jointNames.length > 0) {
      this.qvelAdr = null;
      this.qvelAdrList = this.resolveQvelAdrList(runner.getContext()?.mjModel ?? null, jointNames);
      this.numJoints = jointNames.length;
    } else if (jointName !== undefined) {
      const mjModel = runner.getContext()?.mjModel ?? null;
      this.qvelAdr = mjModel !== null ? this.resolveQvelAdr(mjModel, jointName) : 0;
      this.qvelAdrList = null;
      this.numJoints = 1;
    } else {
      this.qvelAdr = null;
      this.qvelAdrList = null;
      this.numJoints = runner.getNumActions();
    }
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));
    this.history = Array.from(
      { length: this.historySteps },
      () => new Float32Array(this.numJoints)
    );
    this.scale = normalizeScale(config.scale, this.numJoints, 1.0);
  }

  get size(): number {
    return this.numJoints * this.historySteps;
  }

  reset(state?: PolicyState): void {
    const value = this.computeCurrent(state);
    for (const buffer of this.history) {
      buffer.set(value);
    }
  }

  update(state: PolicyState): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent(state));
  }

  compute(): Float32Array {
    if (this.historySteps === 1) {
      return new Float32Array(this.history[0]);
    }
    const output = new Float32Array(this.size);
    let offset = 0;
    for (const buffer of this.history) {
      output.set(buffer, offset);
      offset += buffer.length;
    }
    return output;
  }

  private computeCurrent(state?: PolicyState): Float32Array {
    if (this.qvelAdrList !== null) {
      const qvel = this.runner.getContext()?.mjData?.qvel;
      const out = new Float32Array(this.numJoints);
      for (let i = 0; i < this.numJoints; i++) {
        out[i] = qvel !== undefined ? qvel[this.qvelAdrList[i]] : 0.0;
      }
      if (this.scale) {
        for (let i = 0; i < out.length; i++) {
          out[i] *= this.scale[i] ?? 1.0;
        }
      }
      return out;
    }
    if (this.qvelAdr !== null) {
      const qvel = this.runner.getContext()?.mjData?.qvel;
      const vel = qvel !== undefined ? qvel[this.qvelAdr] : 0.0;
      const out = new Float32Array([vel]);
      if (this.scale) out[0] *= this.scale[0] ?? 1.0;
      return out;
    }
    const value = state?.jointVel ?? new Float32Array(this.numJoints);
    const out = new Float32Array(value);
    if (this.scale) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= this.scale[i] ?? 1.0;
      }
    }
    return out;
  }

  private resolveQvelAdr(mjModel: MjModel, jointName: string): number {
    const names = this.getModelJointNames(mjModel);
    const idx = names.indexOf(jointName);
    if (idx < 0) {
      throw new Error(`JointVelocities: joint "${jointName}" not found in model`);
    }
    return mjModel.jnt_dofadr[idx];
  }

  private resolveQvelAdrList(mjModel: MjModel | null, jointNames: string[]): number[] {
    if (mjModel === null) {
      return Array.from({ length: jointNames.length }, () => 0);
    }
    const names = this.getModelJointNames(mjModel);
    return jointNames.map((jointName) => {
      const idx = names.indexOf(jointName);
      if (idx < 0) {
        throw new Error(`JointVelocities: joint "${jointName}" not found in model`);
      }
      return mjModel.jnt_dofadr[idx];
    });
  }

  private getModelJointNames(mjModel: MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let j = 0; j < mjModel.njnt; j++) {
      let start = mjModel.name_jntadr[j];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) end++;
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }
}

export class SimpleVelocityCommand extends ObservationBase {
  private historySteps: number;
  private history: Float32Array[];
  private scale: Float32Array | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));
    this.history = Array.from({ length: this.historySteps }, () => new Float32Array(3));
    this.scale = normalizeScale(config.scale, 3, 1.0);
  }

  get size(): number {
    return 3 * this.historySteps;
  }

  reset(): void {
    const value = this.computeCurrent();
    for (const buffer of this.history) {
      buffer.set(value);
    }
  }

  update(): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent());
  }

  compute(): Float32Array {
    if (this.historySteps === 1) {
      return new Float32Array(this.history[0]);
    }
    const output = new Float32Array(this.size);
    let offset = 0;
    for (const buffer of this.history) {
      output.set(buffer, offset);
      offset += buffer.length;
    }
    return output;
  }

  private computeCurrent(): Float32Array {
    // Get velocity command from CommandManager
    const commandManager = getCommandManager();
    const velocityCmd = commandManager.getVelocityCommand();
    const out = new Float32Array(velocityCmd);
    if (this.scale) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= this.scale[i] ?? 1.0;
      }
    }
    return out;
  }
}

export class VelocityCommandWithOscillators extends ObservationBase {
  private scale: Float32Array | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.scale = normalizeScale(config.scale, 3, 1.0);
  }

  get size(): number {
    return 16;
  }

  compute(): Float32Array {
    const output = new Float32Array(16);
    // Get velocity command from CommandManager
    const commandManager = getCommandManager();
    const velocityCmd = commandManager.getVelocityCommand();
    const base = new Float32Array(velocityCmd);
    if (this.scale) {
      for (let i = 0; i < 3; i++) {
        base[i] *= this.scale[i] ?? 1.0;
      }
    }
    output[0] = base[0];
    output[1] = base[1];
    output[2] = base[2];
    return output;
  }
}

export class ImpedanceCommand extends ObservationBase {
  get size(): number {
    return 27;
  }

  compute(): Float32Array {
    return new Float32Array(27);
  }
}

export class JointPosCosSin extends ObservationBase {
  private qposAdr: number;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    const mjModel = runner.getContext()?.mjModel ?? null;
    this.qposAdr = mjModel !== null ? this.resolveQposAdr(mjModel) : 0;
  }

  get size(): number {
    return 2;
  }

  compute(): Float32Array {
    const qpos = this.runner.getContext()?.mjData?.qpos;
    const angle = qpos !== undefined ? qpos[this.qposAdr] : 0.0;
    return new Float32Array([Math.cos(angle), Math.sin(angle)]);
  }

  private resolveQposAdr(mjModel: MjModel): number {
    const jointName = this.config.joint_name as string | undefined;
    if (jointName !== undefined) {
      const names = this.getJointNames(mjModel);
      const idx = names.indexOf(jointName);
      if (idx < 0) {
        throw new Error(`JointPosCosSin: joint "${jointName}" not found in model`);
      }
      return mjModel.jnt_qposadr[idx];
    }
    const jointIndex = (this.config.joint_index as number | undefined) ?? 0;
    return mjModel.jnt_qposadr[jointIndex];
  }

  private getJointNames(mjModel: MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let j = 0; j < mjModel.njnt; j++) {
      let start = mjModel.name_jntadr[j];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }
}

export class BuiltinSensor extends ObservationBase {
  private readonly sensorName: string;
  private readonly sensorAdr: number;
  private readonly sensorDim: number;
  private readonly historySteps: number;
  private readonly history: Float32Array[];
  private readonly scale: Float32Array | null;
  private readonly clipRange: [number, number] | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.sensorName = (config.sensor_name as string | undefined) ?? '';
    const mjModel = runner.getContext()?.mjModel ?? null;
    const resolved = this.resolveSensor(mjModel, this.sensorName);
    this.sensorAdr = resolved.adr;
    this.sensorDim = resolved.dim;
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));
    this.history = Array.from({ length: this.historySteps }, () => new Float32Array(this.sensorDim));
    this.scale = normalizeScale(config.scale, this.sensorDim, 1.0);
    this.clipRange = this.normalizeClipRange(config.clip);
  }

  get size(): number {
    return this.sensorDim * this.historySteps;
  }

  reset(): void {
    const value = this.computeCurrent();
    for (const buffer of this.history) {
      buffer.set(value);
    }
  }

  update(): void {
    for (let i = this.history.length - 1; i > 0; i--) {
      this.history[i].set(this.history[i - 1]);
    }
    this.history[0].set(this.computeCurrent());
  }

  compute(): Float32Array {
    if (this.historySteps === 1) {
      return new Float32Array(this.history[0]);
    }
    const output = new Float32Array(this.size);
    let offset = 0;
    for (const buffer of this.history) {
      output.set(buffer, offset);
      offset += buffer.length;
    }
    return output;
  }

  private computeCurrent(): Float32Array {
    const sensordata = this.runner.getContext()?.mjData?.sensordata;
    const out = new Float32Array(this.sensorDim);
    if (sensordata === undefined) {
      return out;
    }
    for (let i = 0; i < this.sensorDim; i++) {
      out[i] = sensordata[this.sensorAdr + i] ?? 0.0;
    }
    if (this.scale) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= this.scale[i] ?? 1.0;
      }
    }
    if (this.clipRange) {
      const [clipMin, clipMax] = this.clipRange;
      for (let i = 0; i < out.length; i++) {
        out[i] = Math.min(clipMax, Math.max(clipMin, out[i]));
      }
    }
    return out;
  }

  private resolveSensor(mjModel: MjModel | null, sensorName: string): { adr: number; dim: number } {
    if (mjModel === null) {
      return { adr: 0, dim: 1 };
    }
    const names = this.getModelSensorNames(mjModel);
    const sensorIndex = names.indexOf(sensorName);
    if (sensorIndex < 0) {
      throw new Error(`BuiltinSensor: sensor "${sensorName}" not found in model`);
    }
    return {
      adr: mjModel.sensor_adr[sensorIndex],
      dim: mjModel.sensor_dim[sensorIndex],
    };
  }

  private getModelSensorNames(mjModel: MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let i = 0; i < mjModel.nsensor; i++) {
      let start = mjModel.name_sensoradr[i];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }

  private normalizeClipRange(clip: unknown): [number, number] | null {
    if (!Array.isArray(clip) || clip.length < 2) {
      return null;
    }
    const minValue = typeof clip[0] === 'number' ? clip[0] : -Infinity;
    const maxValue = typeof clip[1] === 'number' ? clip[1] : Infinity;
    return [minValue, maxValue];
  }
}

// Legacy aliases for config compatibility.
export class ProjectedGravity extends ProjectedGravityB { }
export class JointPositions extends JointPos { }
export class PreviousActions extends PrevActions { }

export const Observations = {
  ...CustomObservations,
  PrevActions,
  PreviousActions,
  BootIndicator,
  RootAngVelB,
  ProjectedGravityB,
  ProjectedGravity,
  JointPos,
  JointPositions,
  TrackingCommandObsRaw,
  TargetRootZObs,
  TargetJointPosObs,
  TargetProjectedGravityBObs,
  MotionAnchorPosB,
  MotionAnchorOriB,
  RobotBodyPosB,
  RobotBodyOriB,
  BaseLinearVelocity,
  BaseAngularVelocity,
  JointVelocities,
  SimpleVelocityCommand,
  VelocityCommandWithOscillators,
  VelocityCommandObservation,
  VelocityCommandWithOscillatorsObservation,
  GeneratedCommands: GeneratedCommandsObservation,
  GeneratedCommandsObservation,
  ImpedanceCommand,
  JointPosCosSin,
  BuiltinSensor,
};
