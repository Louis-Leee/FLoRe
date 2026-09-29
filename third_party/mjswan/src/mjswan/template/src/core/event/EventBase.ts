export type EventConfig = {
  name: string;
  params?: Record<string, unknown>;
};

export type EventContext = {
  mjModel: import('mujoco').MjModel;
  mjData: import('mujoco').MjData;
  terrainData?: TerrainData | null;
};

export type TerrainData = {
  flat_patches?: Record<string, number[][]>;
};

export abstract class EventBase {
  protected config: EventConfig;

  constructor(config: EventConfig) {
    this.config = config;
  }

  /** Called on every episode reset. */
  abstract onReset(context: EventContext): void;
}

export type EventConstructor = new (config: EventConfig) => EventBase;
