import type { PolicyState } from '../policy/types';

export type TerminationConfig = {
  name: string;
  params?: Record<string, unknown>;
  time_out?: boolean;
};

export abstract class TerminationBase {
  protected config: TerminationConfig;

  constructor(config: TerminationConfig) {
    this.config = config;
  }

  abstract evaluate(state: PolicyState): boolean;

  reset?(): void;
}
