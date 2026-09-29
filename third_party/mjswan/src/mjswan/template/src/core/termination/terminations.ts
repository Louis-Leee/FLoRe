import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { PolicyState } from '../policy/types';
import { CustomTerminations } from './custom_terminations';

/**
 * Terminate when the episode step count exceeds max_episode_length.
 *
 * mjlab: env.episode_length_buf >= env.max_episode_length
 */
export class TimeOut extends TerminationBase {
  private stepCount = 0;
  private maxSteps: number;

  constructor(config: TerminationConfig) {
    super(config);
    const params = config.params ?? {};
    this.maxSteps = (params.max_episode_length as number) ?? Infinity;
  }

  evaluate(): boolean {
    this.stepCount++;
    return this.stepCount >= this.maxSteps;
  }

  reset(): void {
    this.stepCount = 0;
  }
}

/**
 * Terminate when the robot's orientation exceeds a limit angle.
 *
 * Uses the projected gravity vector from PolicyState to compute the
 * angle between the robot's up axis and world up.
 *
 * mjlab: torch.acos(-projected_gravity[:, 2]).abs() > limit_angle
 */
export class BadOrientation extends TerminationBase {
  private limitAngle: number;

  constructor(config: TerminationConfig) {
    super(config);
    const params = config.params ?? {};
    this.limitAngle = (params.limit_angle as number) ?? 1.0;
  }

  evaluate(state: PolicyState): boolean {
    const rootQuat = state.rootQuat;
    if (!rootQuat || rootQuat.length < 4) return false;

    // Compute tilt angle from quaternion.
    const x = rootQuat[1];
    const y = rootQuat[2];
    const gz = 1.0 - 2.0 * (x * x + y * y);
    const angle = Math.acos(Math.max(-1.0, Math.min(1.0, gz)));

    return Math.abs(angle) > this.limitAngle;
  }
}

/**
 * Terminate when the robot's root height drops below a minimum.
 *
 * mjlab: asset.data.root_link_pos_w[:, 2] < minimum_height
 */
export class RootHeightBelowMinimum extends TerminationBase {
  private minimumHeight: number;

  constructor(config: TerminationConfig) {
    super(config);
    const params = config.params ?? {};
    this.minimumHeight = (params.minimum_height as number) ?? 0.0;
  }

  evaluate(state: PolicyState): boolean {
    const rootPos = state.rootPos;
    if (!rootPos || rootPos.length < 3) return false;
    return rootPos[2] < this.minimumHeight;
  }
}

export type TerminationConstructor = new (config: TerminationConfig) => TerminationBase;

const BuiltinTerminations: Record<string, TerminationConstructor> = {
  TimeOut,
  BadOrientation,
  RootHeightBelowMinimum,
};

/**
 * Registry mapping termination class names to constructors.
 */
export const Terminations: Record<string, TerminationConstructor> = {
  ...BuiltinTerminations,
  ...CustomTerminations,
};
