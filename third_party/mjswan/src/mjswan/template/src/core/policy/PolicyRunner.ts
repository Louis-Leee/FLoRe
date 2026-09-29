import { ObservationBase } from '../observation/ObservationBase';
import { PolicyModule } from './PolicyModule';
import type {
  ObservationConfigEntry,
  PolicyConfig,
  PolicyRunnerContext,
  PolicyState,
} from './types';

export type PolicyModuleConstructor = new (config: PolicyConfig) => PolicyModule;
export type ObservationConstructor = new (
  runner: PolicyRunner,
  config: ObservationConfigEntry
) => ObservationBase;

export type PolicyRunnerOptions = {
  policyModules?: Record<string, PolicyModuleConstructor>;
  observations?: Record<string, ObservationConstructor>;
};

export class PolicyRunner {
  private config: PolicyConfig;
  private options: PolicyRunnerOptions;
  private policyModule: PolicyModule | null;
  private obsGroups: Record<string, ObservationBase[]>;
  private obsLayouts: Record<string, { name: string; size: number }[]>;
  private obsSizes: Record<string, number>;
  private historyConfig: Record<string, { steps: number; interleaved: boolean }>;
  private historyBuffers: Record<string, Float32Array>;
  private defaultObsKey: string | null;
  private context: PolicyRunnerContext | null;
  private policyJointNames: string[];
  private defaultJointPos: Float32Array;
  private encoderBias: Float32Array;
  private numActions: number;
  private lastActions: Float32Array;

  constructor(config: PolicyConfig, options: PolicyRunnerOptions = {}) {
    this.config = config;
    this.options = options;
    this.policyModule = null;
    this.obsGroups = {};
    this.obsLayouts = {};
    this.obsSizes = {};
    this.historyConfig = {};
    this.historyBuffers = {};
    this.defaultObsKey = null;
    this.context = null;

    this.policyJointNames = (config.policy_joint_names ?? []).slice();
    this.numActions = this.policyJointNames.length;
    this.lastActions = new Float32Array(this.numActions);
    this.defaultJointPos = this.normalizeArray(
      config.default_joint_pos ?? [],
      this.numActions,
      0.0
    );
    this.encoderBias = this.normalizeArray(
      config.encoder_bias ?? [],
      this.numActions,
      0.0
    );
  }

  async init(context: PolicyRunnerContext): Promise<void> {
    this.context = context;
    this.policyModule = await this.buildPolicyModule(context);
    this.buildObservationGroups();
  }

  reset(state?: PolicyState): void {
    this.lastActions.fill(0.0);
    this.policyModule?.reset();
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (obs.reset) {
          obs.reset(state);
        }
      }
    }
    if (state) {
      for (const [key, config] of Object.entries(this.historyConfig)) {
        if (config.steps > 1) {
          const frame = this.buildFrame(this.obsGroups[key] ?? [], state);
          const buffer = this.historyBuffers[key];
          for (let i = 0; i < config.steps; i++) {
            buffer.set(frame, i * frame.length);
          }
        }
      }
    }
  }

  update(state: PolicyState): void {
    this.policyModule?.update();
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (obs.update) {
          obs.update(state);
        }
      }
    }
  }

  collectObservationsByKey(state: PolicyState): Record<string, Float32Array> {
    this.update(state);
    const outputs: Record<string, Float32Array> = {};

    for (const [key, obsList] of Object.entries(this.obsGroups)) {
      const history = this.historyConfig[key];
      if (history && history.steps > 1) {
        const frame = this.buildFrame(obsList, state);
        const buffer = this.historyBuffers[key];
        for (let i = buffer.length - 1; i >= frame.length; i--) {
          buffer[i] = buffer[i - frame.length];
        }
        buffer.set(frame, 0);
        outputs[key] = new Float32Array(buffer);
      } else {
        outputs[key] = this.buildFrame(obsList, state);
      }
    }
    return outputs;
  }

  collectObservations(state: PolicyState): Float32Array {
    const outputs = this.collectObservationsByKey(state);
    if (this.defaultObsKey && outputs[this.defaultObsKey]) {
      return outputs[this.defaultObsKey];
    }
    const first = Object.keys(outputs)[0];
    return first ? outputs[first] : new Float32Array(0);
  }

  getObservationSize(): number {
    if (this.defaultObsKey && this.obsSizes[this.defaultObsKey] !== undefined) {
      return this.obsSizes[this.defaultObsKey];
    }
    const first = Object.keys(this.obsSizes)[0];
    return first ? this.obsSizes[first] : 0;
  }

  getObservationLayout(): { name: string; size: number }[] {
    if (this.defaultObsKey && this.obsLayouts[this.defaultObsKey]) {
      return this.obsLayouts[this.defaultObsKey].map((entry) => ({ ...entry }));
    }
    const first = Object.keys(this.obsLayouts)[0];
    return first ? this.obsLayouts[first].map((entry) => ({ ...entry })) : [];
  }

  getPolicyModuleContext(): Record<string, unknown> {
    return this.policyModule?.getContext() ?? {};
  }

  getPolicyModule(): PolicyModule | null {
    return this.policyModule;
  }

  getContext(): PolicyRunnerContext | null {
    return this.context;
  }

  getPolicyJointNames(): string[] {
    return this.policyJointNames.slice();
  }

  getNumActions(): number {
    return this.numActions;
  }

  getDefaultJointPos(): Float32Array {
    return new Float32Array(this.defaultJointPos);
  }

  getEncoderBias(): Float32Array {
    return new Float32Array(this.encoderBias);
  }

  getLastActions(): Float32Array {
    return new Float32Array(this.lastActions);
  }

  setLastActions(actions: Float32Array): void {
    if (actions.length !== this.lastActions.length) {
      this.lastActions = new Float32Array(actions);
      return;
    }
    this.lastActions.set(actions);
  }

  private async buildPolicyModule(
    context: PolicyRunnerContext
  ): Promise<PolicyModule | null> {
    const registry = this.options.policyModules ?? {};
    const moduleKey = this.config.policy_module;
    const Module = moduleKey ? registry[moduleKey] : registry.default;

    if (moduleKey && !Module) {
      throw new Error(`Unknown policy module: ${moduleKey}`);
    }

    if (!Module) {
      return null;
    }

    const module = new Module(this.config);
    await module.init(context);
    return module;
  }

  private buildObservationGroups(): void {
    const registry = this.options.observations ?? {};
    const obsConfig = this.config.observations ?? {};
    this.obsGroups = {};
    this.obsLayouts = {};
    this.obsSizes = {};
    this.historyConfig = {};
    this.historyBuffers = {};
    this.defaultObsKey = null;

    for (const [key, value] of Object.entries(obsConfig)) {
      if (Array.isArray(value)) {
        const obsList = value.map((entry) => {
          const ObsClass = registry[entry.name];
          if (!ObsClass) {
            throw new Error(`Unknown observation type: ${entry.name}`);
          }
          return new ObsClass(this, entry);
        });
        this.registerGroup(key, obsList, value);
        continue;
      }
      if (value && typeof value === 'object') {
        const configValue = value as {
          history_steps?: number;
          interleaved?: boolean;
          components?: ObservationConfigEntry[];
        };
        if (Array.isArray(configValue.components)) {
          const obsList = configValue.components.map((entry) => {
            const ObsClass = registry[entry.name];
            if (!ObsClass) {
              throw new Error(`Unknown observation type: ${entry.name}`);
            }
            const entryConfig = { ...entry, history_steps: 1 };
            return new ObsClass(this, entryConfig);
          });
          const steps = Math.max(1, Math.floor(configValue.history_steps ?? 1));
          const interleaved = Boolean(configValue.interleaved);
          this.registerGroup(key, obsList, configValue.components, {
            steps,
            interleaved,
          });
        }
      }
    }

    if (this.obsGroups.policy) {
      this.defaultObsKey = 'policy';
    } else if (this.obsGroups.observation) {
      this.defaultObsKey = 'observation';
    } else if (this.obsGroups.obs_history) {
      this.defaultObsKey = 'obs_history';
    } else {
      this.defaultObsKey = Object.keys(this.obsGroups)[0] ?? null;
    }
  }

  private registerGroup(
    key: string,
    obsList: ObservationBase[],
    configList: ObservationConfigEntry[],
    history?: { steps: number; interleaved: boolean }
  ): void {
    this.obsGroups[key] = obsList;
    this.obsLayouts[key] = obsList.map((obs, index) => ({
      name: configList[index]?.name ?? `obs_${index}`,
      size: obs.size,
    }));
    const baseSize = this.obsLayouts[key].reduce((sum, entry) => sum + entry.size, 0);
    if (history && history.steps > 1) {
      this.historyConfig[key] = history;
      this.historyBuffers[key] = new Float32Array(baseSize * history.steps);
      this.obsSizes[key] = baseSize * history.steps;
    } else {
      this.obsSizes[key] = baseSize;
    }
  }

  private buildFrame(obsList: ObservationBase[], state: PolicyState): Float32Array {
    const size = obsList.reduce((sum, obs) => sum + obs.size, 0);
    const output = new Float32Array(size);
    let offset = 0;
    for (const obs of obsList) {
      const value = obs.compute(state);
      const array = value instanceof Float32Array ? value : Float32Array.from(value);
      if (array.length !== obs.size) {
        throw new Error(
          `Observation size mismatch: expected ${obs.size}, got ${array.length}`
        );
      }
      output.set(array, offset);
      offset += array.length;
    }
    return output;
  }

  private normalizeArray(
    values: number[],
    length: number,
    fallback: number
  ): Float32Array {
    const output = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      output[i] = typeof values[i] === 'number' ? values[i] : fallback;
    }
    return output;
  }
}
