import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

export type ObservationConfig = {
  name: string;
  [key: string]: unknown;
};

export abstract class ObservationBase<TConfig extends ObservationConfig = ObservationConfig> {
  protected runner: PolicyRunner;
  protected config: TConfig;

  constructor(runner: PolicyRunner, config: TConfig) {
    this.runner = runner;
    this.config = config;
  }

  abstract get size(): number;

  reset?(_state?: PolicyState): void;

  update?(_state: PolicyState): void;

  abstract compute(state: PolicyState): Float32Array | number[];
}
