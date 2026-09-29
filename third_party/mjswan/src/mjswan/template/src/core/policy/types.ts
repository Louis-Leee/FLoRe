import type { MainModule, MjData, MjModel } from 'mujoco';
import type { Scene } from 'three';

import type { CommandsConfig } from '../command';

export type PolicyRunnerContext = {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene?: Scene | null;
};

export type PolicyState = {
  jointPos: Float32Array;
  jointVel?: Float32Array;
  rootPos?: Float32Array;
  rootQuat?: Float32Array;
  rootLinVel?: Float32Array;
  rootAngVel?: Float32Array;
  [key: string]: unknown;
};

export type ObservationConfigEntry = {
  name: string;
  [key: string]: unknown;
};

export type ObservationGroupConfig =
  | ObservationConfigEntry[]
  | {
    history_steps?: number;
    interleaved?: boolean;
    components?: ObservationConfigEntry[];
  };

export type ActionConfigEntry = {
  type: string;
  scale?: number | number[] | Record<string, number>;
  offset?: number | Record<string, number>;
  use_default_offset?: boolean;
  stiffness?: number | number[] | Record<string, number>;
  damping?: number | number[] | Record<string, number>;
  actuator_names?: string[];
  [key: string]: unknown;
};

export type TerminationConfigEntry = {
  name: string;
  params?: Record<string, unknown>;
  time_out?: boolean;
};

export type PolicyConfig = {
  policy_module?: string;
  policy_joint_names?: string[];
  default_joint_pos?: number[];
  encoder_bias?: number[];
  action_scale?: number[] | number;
  stiffness?: number[] | number;
  damping?: number[] | number;
  control_type?: string;
  onnx?: {
    path: string;
    meta?: {
      in_keys?: string[];
      out_keys?: (string | string[])[];
    };
  };
  commands?: CommandsConfig;
  motions?: Array<{
    name: string;
    path: string;
    anchor_body_name: string;
    body_names: string[];
    dataset_joint_names?: string[];
    default?: boolean;
  }>;
  observations?: Record<string, ObservationGroupConfig>;
  actions?: Record<string, ActionConfigEntry>;
  terminations?: Record<string, TerminationConfigEntry>;
  [key: string]: unknown;
};
