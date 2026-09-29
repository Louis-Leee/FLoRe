/**
 * CommandObservation - Observation that reads command values from the CommandManager.
 *
 * This observation provides user-controlled inputs (like velocity commands)
 * to the ONNX policy. It reads values from the global CommandManager and
 * returns them as an observation array.
 */

import { ObservationBase } from './ObservationBase';
import type { ObservationConfig } from './ObservationBase';
import type { PolicyRunner } from '../policy/PolicyRunner';
import { getCommandManager } from '../command';

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

/**
 * VelocityCommandObservation - Provides velocity commands from the UI
 *
 * Config options:
 * - scale: number | number[] - Scale factors for velocity values
 * - history_steps: number - Number of history steps to include
 */
export class VelocityCommandObservation extends ObservationBase {
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

/**
 * VelocityCommandWithOscillatorsObservation - Velocity commands with oscillator states
 *
 * Returns a 16-dimensional observation:
 * - [0-2]: velocity command (vel_x, vel_y, yaw_rate)
 * - [3-15]: oscillator states (placeholder zeros for now)
 *
 * Config options:
 * - scale: number | number[] - Scale factors for velocity values
 */
export class VelocityCommandWithOscillatorsObservation extends ObservationBase {
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
    const commandManager = getCommandManager();
    const velocityCmd = commandManager.getVelocityCommand();

    // Apply scale if configured
    if (this.scale) {
      for (let i = 0; i < 3; i++) {
        velocityCmd[i] *= this.scale[i] ?? 1.0;
      }
    }

    // Set velocity command in first 3 elements
    output[0] = velocityCmd[0];
    output[1] = velocityCmd[1];
    output[2] = velocityCmd[2];

    // Remaining 13 elements are oscillator states (zeros for now)
    return output;
  }
}

/**
 * GeneratedCommandsObservation - Reads command values from a named command group
 *
 * This is the main observation class for reading user-controlled commands.
 * It mirrors the mjlab pattern where `generated_commands(env, command_name)`
 * retrieves command values by group name.
 *
 * Config options:
 * - command_name: string - The name of the command group to read from (e.g., "velocity")
 * - scale: number | number[] - Scale factors for command values
 * - history_steps: number - Number of history steps to include (default: 1)
 *
 * Example obs_config entry:
 * {"name": "GeneratedCommands", "command_name": "velocity"}
 */
export class GeneratedCommandsObservation extends ObservationBase {
  private commandName: string;
  private commandSize: number;
  private historySteps: number;
  private history: Float32Array[];
  private scale: Float32Array | null;

  constructor(runner: PolicyRunner, config: ObservationConfig) {
    super(runner, config);
    this.commandName = (config.command_name as string | undefined) ?? 'velocity';
    this.historySteps = Math.max(1, Math.floor((config.history_steps as number | undefined) ?? 1));

    // Get command size from CommandManager
    const commandManager = getCommandManager();
    const command = commandManager.getCommand(this.commandName);
    this.commandSize = command.length || 3; // Default to 3 for velocity

    this.history = Array.from({ length: this.historySteps }, () => new Float32Array(this.commandSize));
    this.scale = normalizeScale(config.scale, this.commandSize, 1.0);
  }

  get size(): number {
    return this.commandSize * this.historySteps;
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
    const commandManager = getCommandManager();
    const commandValues = commandManager.getCommand(this.commandName);

    // Ensure we have the right size
    const out = new Float32Array(this.commandSize);
    for (let i = 0; i < this.commandSize && i < commandValues.length; i++) {
      out[i] = commandValues[i];
    }

    // Apply scale if configured
    if (this.scale) {
      for (let i = 0; i < out.length; i++) {
        out[i] *= this.scale[i] ?? 1.0;
      }
    }
    return out;
  }
}
