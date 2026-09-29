import { EventBase, type EventConfig, type EventContext } from './EventBase';
import type { EventConstructor } from './EventBase';

export class EventManager {
  private terms: EventBase[] = [];

  constructor(
    configs: EventConfig[],
    registry: Record<string, EventConstructor>
  ) {
    for (const config of configs) {
      const EventClass = registry[config.name];
      if (!EventClass) {
        console.warn(`[EventManager] Unknown event type: ${config.name}`);
        continue;
      }
      this.terms.push(new EventClass(config));
    }
  }

  onReset(context: EventContext): void {
    for (const term of this.terms) {
      term.onReset(context);
    }
  }

  get size(): number {
    return this.terms.length;
  }
}
