import type * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';

export type CommandType = 'slider' | 'button' | 'checkbox';

export interface SliderCommandConfig {
  type: 'slider';
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  enabled_when?: string;
}

export interface ButtonCommandConfig {
  type: 'button';
  name: string;
  label: string;
}

export interface CheckboxCommandConfig {
  type: 'checkbox';
  name: string;
  label: string;
  default: boolean;
}

export type CommandInputConfig = SliderCommandConfig | ButtonCommandConfig | CheckboxCommandConfig;

export interface CommandUiConfig {
  inputs?: CommandInputConfig[];
}

export interface CommandConfigEntry {
  name: string;
  ui?: CommandUiConfig;
  [key: string]: unknown;
}

export type CommandsConfig = Record<string, CommandConfigEntry>;

export interface CommandDefinition {
  id: string;
  groupName: string;
  config: CommandInputConfig;
}

export function getCommandInputId(groupName: string, inputName: string): string {
  return `${groupName}:${inputName}`;
}

export type CommandEventType = 'change' | 'reset' | 'button' | 'group_registered' | 'clear';

export interface CommandEvent {
  type: CommandEventType;
  commandId: string;
  groupName?: string;
  value?: number;
}

export type CommandEventListener = (event: CommandEvent) => void;

export interface CommandTermContext {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene: THREE.Scene;
  bodies?: Record<number, THREE.Group> | null;
  mujocoRoot?: THREE.Group | null;
  requestReset?: () => void;
}

export interface CommandTerm {
  getCommand(): Float32Array;
  getUiConfig?(): CommandUiConfig | null;
  reset?(): void;
  update?(dt: number): void;
  updateDebugVisuals?(): void;
  setValue?(inputName: string, value: number): number | void;
  triggerButton?(inputName: string): void;
  dispose?(): void;
}

export interface TrackingSource {
  isReady(): boolean;
  getAnchorBodyName(): string | null;
  getAnchorPos(frameIndex?: number): Float32Array | null;
  getAnchorQuat(frameIndex?: number): Float32Array | null;
  getBodyNames(): string[];
  getBodyPosW(frameIndex?: number): Float32Array | null;
}

export function isTrackingSource(term: unknown): term is TrackingSource {
  return (
    typeof term === 'object' &&
    term !== null &&
    'isReady' in term &&
    'getAnchorBodyName' in term &&
    'getBodyNames' in term
  );
}

export type CommandTermConstructor = new (
  termName: string,
  config: CommandConfigEntry,
  context: CommandTermContext
) => CommandTerm;
