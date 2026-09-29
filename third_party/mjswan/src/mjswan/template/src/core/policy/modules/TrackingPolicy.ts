import { PolicyModule } from '../PolicyModule';
import type { PolicyRunnerContext, PolicyState } from '../types';

export type TrackingConfig = {
  policy_joint_names?: string[];
  dataset_joint_names?: string[];
  transition_steps?: number;
};

export class TrackingHelper {
  policyJointNames: string[];
  datasetJointNames: string[];
  nJoints: number;
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  currentName: string;
  currentDone: boolean;

  constructor(config: TrackingConfig) {
    this.policyJointNames = (config.policy_joint_names ?? []).slice();
    this.datasetJointNames = (config.dataset_joint_names ?? []).slice();
    this.nJoints = this.datasetJointNames.length || this.policyJointNames.length || 0;
    this.refJointPos = [];
    this.refRootPos = [];
    this.refRootQuat = [];
    this.refIdx = 0;
    this.refLen = 0;
    this.currentName = 'default';
    this.currentDone = true;
  }

  reset(state?: PolicyState): void {
    const jointPos = state?.jointPos ?? new Float32Array(this.nJoints);
    const rootPos = state?.rootPos ?? new Float32Array([0.0, 0.0, 0.0]);
    const rootQuat = state?.rootQuat ?? new Float32Array([1.0, 0.0, 0.0, 0.0]);

    this.refJointPos = [Float32Array.from(jointPos)];
    this.refRootPos = [Float32Array.from(rootPos)];
    this.refRootQuat = [Float32Array.from(rootQuat)];
    this.refIdx = 0;
    this.refLen = 1;
    this.currentDone = true;
    this.currentName = 'default';
  }

  advance(): void {
    if (this.refLen <= 1) {
      return;
    }
    if (this.refIdx < this.refLen - 1) {
      this.refIdx += 1;
      if (this.refIdx === this.refLen - 1) {
        this.currentDone = true;
      }
    }
  }

  isReady(): boolean {
    return this.refLen > 0;
  }
}

export class TrackingPolicy extends PolicyModule {
  private tracking: TrackingHelper | null;

  constructor(config: Record<string, unknown>) {
    super(config);
    this.tracking = null;
  }

  async init(context: PolicyRunnerContext): Promise<void> {
    await super.init(context);
    const policyJointNames = (this.config.policy_joint_names as string[] | undefined) ?? [];
    const trackingConfig: TrackingConfig = {
      ...(this.config.tracking as TrackingConfig | undefined),
      policy_joint_names: policyJointNames,
    };
    this.tracking = new TrackingHelper(trackingConfig);
  }

  reset(state?: PolicyState): void {
    this.tracking?.reset(state);
  }

  update(): void {
    this.tracking?.advance();
  }

  getContext(): Record<string, unknown> {
    return {
      tracking: this.tracking,
    };
  }
}
