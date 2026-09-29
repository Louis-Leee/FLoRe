/**
 * Generic slider panel for live policy inputs.
 *
 * Driven by `policyConfig.extras.sliders: SliderConfig[]`. Each entry creates
 * an HTML <input type=range> whose live value is written into the singleton
 * policy-extras store via `setPolicyExtra(key, value)`. Observation classes
 * and debug visualizers read from the same store via `getPolicyExtra(key)`.
 *
 * Layout: a sticky panel in the top-left of the page, grouped by the
 * optional `group` field (groups render as labeled rows). Pure DOM — no
 * three.js dependency, no Vue/React, no external CSS framework. The panel
 * is removed and rebuilt every time `mountSliderPanel(...)` is called, so
 * loading a new scene with a different slider set does not leak inputs.
 *
 * Why DOM, not three.js: range inputs are a solved UI problem in the
 * browser. Re-implementing them in WebGL would balloon the diff and break
 * accessibility (keyboard, screen reader, scroll wheel, touch).
 */

import { setPolicyExtra, type SliderConfig } from './extras';

const PANEL_ID = 'mjswan-slider-panel';

// Registry of the currently-mounted slider rows, rebuilt on every
// mountSliderPanel call. resetSlidersToDefault() walks this to restore each
// input to its seeded default. Empty for non-force presets (panel removed),
// so resetSlidersToDefault is a no-op there.
type SliderEntry = { input: HTMLInputElement; key: string; default: number };
const _sliderRegistry: SliderEntry[] = [];

function ensurePanelRoot(): HTMLDivElement {
  let root = document.getElementById(PANEL_ID) as HTMLDivElement | null;
  if (root) {
    root.replaceChildren();
    return root;
  }
  root = document.createElement('div');
  root.id = PANEL_ID;
  Object.assign(root.style, {
    position: 'fixed',
    top: '12px',
    left: '12px',
    padding: '10px 14px',
    background: 'rgba(20, 24, 32, 0.85)',
    color: '#f0f0f0',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '12px',
    lineHeight: '1.4',
    minWidth: '260px',
    maxWidth: '320px',
    borderRadius: '6px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    zIndex: '10000',
    pointerEvents: 'auto',
    userSelect: 'none',
  } as Partial<CSSStyleDeclaration>);
  document.body.appendChild(root);
  return root;
}

function renderGroupHeader(label: string): HTMLElement {
  const h = document.createElement('div');
  h.textContent = label;
  Object.assign(h.style, {
    fontWeight: 'bold',
    marginTop: '6px',
    marginBottom: '4px',
    paddingBottom: '2px',
    borderBottom: '1px solid rgba(255,255,255,0.2)',
  } as Partial<CSSStyleDeclaration>);
  return h;
}

function renderSliderRow(cfg: SliderConfig): HTMLElement {
  const row = document.createElement('div');
  Object.assign(row.style, {
    display: 'grid',
    gridTemplateColumns: '60px 1fr 56px',
    gap: '6px',
    alignItems: 'center',
    margin: '3px 0',
  } as Partial<CSSStyleDeclaration>);

  const labelEl = document.createElement('span');
  labelEl.textContent = cfg.label;
  labelEl.style.color = '#a0a0a0';
  labelEl.style.whiteSpace = 'nowrap';
  labelEl.style.overflow = 'hidden';
  labelEl.style.textOverflow = 'ellipsis';
  labelEl.title = `${cfg.label}  (key: ${cfg.key})`;
  row.appendChild(labelEl);

  const initial = typeof cfg.default === 'number' ? cfg.default : 0;
  const step = cfg.step ?? Math.max((cfg.max - cfg.min) / 200, 1e-6);

  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(cfg.min);
  input.max = String(cfg.max);
  input.step = String(step);
  input.value = String(initial);
  input.style.width = '100%';

  const valEl = document.createElement('span');
  valEl.textContent = formatValue(initial);
  Object.assign(valEl.style, {
    fontWeight: 'bold',
    textAlign: 'right',
    fontVariantNumeric: 'tabular-nums',
  } as Partial<CSSStyleDeclaration>);

  // Double-click the value field to reset to default (or zero if no default).
  valEl.style.cursor = 'pointer';
  valEl.title = 'Double-click to reset';
  valEl.addEventListener('dblclick', () => {
    input.value = String(initial);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });

  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    valEl.textContent = formatValue(v);
    setPolicyExtra(cfg.key, v);
  });

  // Seed the initial value into the extras store so observation classes don't
  // see `undefined` on the first frame before the user has interacted.
  setPolicyExtra(cfg.key, initial);

  row.appendChild(input);
  row.appendChild(valEl);
  // Register so resetSlidersToDefault() can restore this row on reset.
  _sliderRegistry.push({ input, key: cfg.key, default: initial });
  return row;
}

function formatValue(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

/** Mount a slider panel from a list of configs. Idempotent — rebuilds the
 *  panel each call. Pass `[]` (or omit) to remove the panel. */
export function mountSliderPanel(sliders: SliderConfig[] | undefined): void {
  // Rebuild the registry from scratch every mount so a previous scene's rows
  // (or a removed panel) never linger.
  _sliderRegistry.length = 0;
  if (!sliders || sliders.length === 0) {
    const existing = document.getElementById(PANEL_ID);
    if (existing) existing.remove();
    return;
  }
  const root = ensurePanelRoot();

  const title = document.createElement('div');
  title.textContent = 'Live policy inputs';
  Object.assign(title.style, {
    fontWeight: 'bold',
    marginBottom: '6px',
    paddingBottom: '4px',
    borderBottom: '1px solid rgba(255,255,255,0.3)',
  } as Partial<CSSStyleDeclaration>);
  root.appendChild(title);

  // Group sliders by their `group` field (preserving insertion order). Sliders
  // without a group land in an implicit '' group rendered without a header.
  const groups = new Map<string, SliderConfig[]>();
  for (const s of sliders) {
    const g = s.group ?? '';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g)!.push(s);
  }

  for (const [groupName, groupSliders] of groups) {
    if (groupName) root.appendChild(renderGroupHeader(groupName));
    for (const s of groupSliders) root.appendChild(renderSliderRow(s));
  }
}

/** Restore every currently-mounted slider to its seeded default and fire the
 *  input listener so the value label AND the policy-extras store stay in sync
 *  via the single existing code path (renderSliderRow's 'input' handler). No-op
 *  when no panel is mounted (the registry is empty for non-force presets).
 *  Called on every simulation reset so a lingering F^cmd / F^ext can't carry
 *  across a Reset / motion-switch / fall. */
export function resetSlidersToDefault(): void {
  for (const e of _sliderRegistry) {
    e.input.value = String(e.default);
    // Re-runs the 'input' listener: valEl label update + setPolicyExtra(key, default).
    e.input.dispatchEvent(new Event('input', { bubbles: true }));
  }
}
