import { EventBase, type EventConfig, type EventContext } from './EventBase';
import { CustomEvents } from './custom_events';

/**
 * Reset root state with uniform random pose sampling.
 *
 * Applies random offsets to the free joint root position and orientation.
 * Mirrors mjlab's reset_root_state_uniform.
 *
 * Params:
 *   pose_range: { x?, y?, z?, roll?, pitch?, yaw? } — each a [min, max] tuple
 */
export class ResetRootStateUniform extends EventBase {
  private poseRange: Record<string, [number, number]>;

  constructor(config: EventConfig) {
    super(config);
    this.poseRange = (config.params?.pose_range as Record<string, [number, number]>) ?? {};
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;

    // Find the free joint (root body)
    const freeJointIdx = this._findFreeJoint(mjModel);
    if (freeJointIdx === -1) return;

    const qposAdr = mjModel.jnt_qposadr[freeJointIdx];

    const sample = (key: string): number => {
      const range = this.poseRange[key];
      if (!range) return 0;
      return range[0] + Math.random() * (range[1] - range[0]);
    };

    // Apply x/y/z offset
    mjData.qpos[qposAdr + 0] += sample('x');
    mjData.qpos[qposAdr + 1] += sample('y');
    mjData.qpos[qposAdr + 2] += sample('z');

    // Apply yaw rotation (compose with existing quaternion)
    const yaw = sample('yaw');
    if (yaw !== 0) {
      this._applyYawRotation(mjData.qpos, qposAdr + 3, yaw);
    }
  }

  private _findFreeJoint(mjModel: import('mujoco').MjModel): number {
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) return i; // mjJNT_FREE = 0
    }
    return -1;
  }

  private _applyYawRotation(
    qpos: Float64Array,
    quatAdr: number,
    yaw: number
  ): void {
    const hw = Math.cos(yaw / 2);
    const hz = Math.sin(yaw / 2);
    // delta quaternion: (hw, 0, 0, hz)
    const w = qpos[quatAdr + 0];
    const x = qpos[quatAdr + 1];
    const y = qpos[quatAdr + 2];
    const z = qpos[quatAdr + 3];
    qpos[quatAdr + 0] = hw * w - hz * z;
    qpos[quatAdr + 1] = hw * x + hz * y;  // Note: MuJoCo quat is (w, x, y, z)
    qpos[quatAdr + 2] = hw * y - hz * x;
    qpos[quatAdr + 3] = hw * z + hz * w;
  }
}

/**
 * Reset root state by placing the robot on a random flat terrain patch.
 *
 * Selects a random [x, y, z] from terrain_data.flat_patches[patch_name]
 * and sets the root position. Falls back to ResetRootStateUniform if no
 * patch data is available.
 *
 * Params:
 *   patch_name: string (default: "spawn")
 *   pose_range: optional additional offset (same as ResetRootStateUniform)
 */
export class ResetRootStateFromFlatPatches extends EventBase {
  private patchName: string;
  private fallback: ResetRootStateUniform;

  constructor(config: EventConfig) {
    super(config);
    this.patchName = (config.params?.patch_name as string) ?? 'spawn';
    this.fallback = new ResetRootStateUniform(config);
  }

  onReset(context: EventContext): void {
    const { mjModel, mjData, terrainData } = context;
    if (!mjModel || !mjData) return;

    const patches = terrainData?.flat_patches?.[this.patchName];
    if (!patches || patches.length === 0) {
      this.fallback.onReset(context);
      return;
    }

    const freeJointIdx = this._findFreeJoint(mjModel);
    if (freeJointIdx === -1) return;

    const qposAdr = mjModel.jnt_qposadr[freeJointIdx];

    // Pick a random patch position
    const patch = patches[Math.floor(Math.random() * patches.length)];

    // Set x, y from patch; z = patch z + default root z height
    const defaultZ = mjData.qpos[qposAdr + 2];
    mjData.qpos[qposAdr + 0] = patch[0];
    mjData.qpos[qposAdr + 1] = patch[1];
    mjData.qpos[qposAdr + 2] = patch[2] + defaultZ;

    // Apply random yaw
    const poseRange = (this.config.params?.pose_range as Record<string, [number, number]>) ?? {};
    const yawRange = poseRange['yaw'] ?? [-Math.PI, Math.PI];
    const yaw = yawRange[0] + Math.random() * (yawRange[1] - yawRange[0]);
    this._applyYawRotation(mjData.qpos, qposAdr + 3, yaw);
  }

  private _findFreeJoint(mjModel: import('mujoco').MjModel): number {
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) return i;
    }
    return -1;
  }

  private _applyYawRotation(
    qpos: Float64Array,
    quatAdr: number,
    yaw: number
  ): void {
    const hw = Math.cos(yaw / 2);
    const hz = Math.sin(yaw / 2);
    const w = qpos[quatAdr + 0];
    const x = qpos[quatAdr + 1];
    const y = qpos[quatAdr + 2];
    const z = qpos[quatAdr + 3];
    qpos[quatAdr + 0] = hw * w - hz * z;
    qpos[quatAdr + 1] = hw * x + hz * y;
    qpos[quatAdr + 2] = hw * y - hz * x;
    qpos[quatAdr + 3] = hw * z + hz * w;
  }
}

export type { EventConstructor } from './EventBase';

const BuiltinEvents: Record<string, import('./EventBase').EventConstructor> = {
  ResetRootStateUniform,
  ResetRootStateFromFlatPatches,
};

export const Events: Record<string, import('./EventBase').EventConstructor> = {
  ...BuiltinEvents,
  ...CustomEvents,
};
