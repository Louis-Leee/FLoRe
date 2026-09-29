/**
 * Runtime-wide singleton store for policy "extras" — small, live floats that
 * any observation class or debug visualizer can read each frame.
 *
 * Use cases:
 *   - User-controlled inputs (sliders, gamepad axes) that act as additional
 *     observation channels (e.g. UniFP-style F^cmd, target velocity, gait
 *     period).
 *   - Debug overlays that depend on the same live values (e.g. force arrows
 *     drawn at hand contact points).
 *
 * Why a singleton rather than passing through the runtime / runner:
 *   - Observation classes are constructed by name lookup against a registry
 *     and don't hold a reference to the runtime. Plumbing per-instance state
 *     through their constructors would force every new observation to update
 *     PolicyRunner internals.
 *   - Sliders are global UI; multiple obs classes (or a debug visualizer in
 *     a CommandTerm) may consume the same slider value. A central read-mostly
 *     map matches that fan-out cleanly.
 *
 * Lifetime:
 *   - Reset at the start of each `loadEnvironment` (via `resetPolicyExtras`)
 *     so a previous policy's extras don't leak into the new scene.
 *   - Slider DOM is also rebuilt per scene; orphaned keys do no harm but are
 *     cleared for hygiene.
 */

const _extras: Record<string, number> = {};

/** Read a numeric extra by key. Returns the configured fallback (default 0)
 *  when the key is unset — making it safe to read from `compute()` before
 *  the slider panel has populated the map. */
export function getPolicyExtra(key: string, fallback = 0): number {
  const v = _extras[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** Bulk read: returns the full extras snapshot. Mostly for debug overlays. */
export function getPolicyExtrasSnapshot(): Readonly<Record<string, number>> {
  return _extras;
}

/** Slider panel + scripted writers call this. */
export function setPolicyExtra(key: string, value: number): void {
  _extras[key] = value;
}

/** Called by mjswanRuntime.loadEnvironment when a new policy is loaded so
 *  the previous policy's extras don't bleed through. */
export function resetPolicyExtras(): void {
  for (const k of Object.keys(_extras)) {
    delete _extras[k];
  }
}

export type SliderConfig = {
  /** Display label (rendered next to the slider). */
  label: string;
  /** Key written into the extras map. Reused by obs classes / visualizers. */
  key: string;
  min: number;
  max: number;
  /** Optional default value at scene load (defaults to 0 if absent). */
  default?: number;
  /** Optional discrete step (defaults to (max-min)/200). */
  step?: number;
  /** Optional group label — sliders with the same group are rendered together. */
  group?: string;
};
