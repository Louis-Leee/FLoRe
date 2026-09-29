// IMPORTANT: each `import ... from '...';` MUST be a single line.
// mjswan inlines this file into core/observation/custom_observations.ts and
// dedupes imports line-by-line; multi-line import blocks get truncated.
import { ObservationBase } from '@/core/observation/ObservationBase';
import { getCommandManager } from '@/core/command';
import type { PolicyState } from '@/core/policy/types';
import type { CommandTerm } from '@/core/command/types';
import { getPolicyExtra } from '@/core/policy/extras';

// Same structural type as in WbtObservations.ts. Duplicated rather than
// shared so each ts_src is self-contained — mjswan auto-generation does not
// resolve cross-file relative imports beyond top-level dedup.
type WbtTrackingLikeMetrics = CommandTerm & {
  isReady(): boolean;
  refIdx: number;
  refLen: number;
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  getSelectedMotionName(): string | null;
  // Force preset only: [rawL_cm, rawR_cm, eq2L_cm, eq2R_cm] or null when the
  // clip has no baked force reference (synthetic/idle) or wrists unresolved.
  getForceHandErrCm?(): [number, number, number, number] | null;
};

function getWbtTrackingForMetrics(): WbtTrackingLikeMetrics | null {
  const term = getCommandManager().getTerm('motion');
  if (!term) return null;
  const candidate = term as Partial<WbtTrackingLikeMetrics>;
  if (
    typeof candidate.isReady === 'function' &&
    typeof candidate.getSelectedMotionName === 'function' &&
    Array.isArray(candidate.refJointPos) &&
    Array.isArray(candidate.refRootPos)
  ) {
    return term as WbtTrackingLikeMetrics;
  }
  return null;
}

const WBT_RAD2DEG = 180.0 / Math.PI;
const WBT_BOOT_WINDOW = 50;

/** Format an L/R cm pair as "L 1.23 / R 4.56 cm", with '—' for null/NaN. */
function _fmtLR(l: number | undefined, r: number | undefined): string {
  const f = (v: number | undefined) =>
    v === undefined || Number.isNaN(v) ? '—' : `${v.toFixed(2)}`;
  if ((l === undefined || Number.isNaN(l)) && (r === undefined || Number.isNaN(r))) return '—';
  return `L ${f(l)} / R ${f(r)} cm`;
}

type WbtPanelEls = {
  root: HTMLDivElement;
  motion: HTMLSpanElement;
  frame: HTMLSpanElement;
  jointErr: HTMLSpanElement;
  bodyErr: HTMLSpanElement;
  rootErr: HTMLSpanElement;
  bootErr: HTMLSpanElement;
  forceRaw: HTMLSpanElement;
  forceEq2: HTMLSpanElement;
  // Hybrid-only rows (exp:g1-wbt-2pt-hybrid). Their parent <div>s are hidden for
  // non-hybrid presets so the panel stays unchanged there.
  forceMeas: HTMLSpanElement;
  integ: HTMLSpanElement;
  deltaX: HTMLSpanElement;
  forceErr: HTMLSpanElement;
  // C1 unified only: the soft contact_phase label ∈[0,1] per hand that the runtime
  // publishes as wbt_cphase__{l,r} and the 170D/185D actor obs consumes. Its own
  // visibility gate (the cphase keys), so the pre-unified hybrid panels are unchanged.
  contactPhase: HTMLSpanElement;
  forceMeasLine: HTMLDivElement;
  integLine: HTMLDivElement;
  deltaXLine: HTMLDivElement;
  forceErrLine: HTMLDivElement;
  contactPhaseLine: HTMLDivElement;
};

// Hybrid extras-bus keys (world frame) the runtime publishes each step. The panel
// reads them directly to show F^meas magnitude, integrator, and setpoint shift.
const _WBT_HYBRID_KEYS = {
  meas: { l: ['wbt_force_meas__l_fx', 'wbt_force_meas__l_fy', 'wbt_force_meas__l_fz'], r: ['wbt_force_meas__r_fx', 'wbt_force_meas__r_fy', 'wbt_force_meas__r_fz'] },
  integ: { l: ['wbt_integ__l_fx', 'wbt_integ__l_fy', 'wbt_integ__l_fz'], r: ['wbt_integ__r_fx', 'wbt_integ__r_fy', 'wbt_integ__r_fz'] },
  xtgt: { l: ['wbt_xtgt__l_fx', 'wbt_xtgt__l_fy', 'wbt_xtgt__l_fz'], r: ['wbt_xtgt__r_fx', 'wbt_xtgt__r_fy', 'wbt_xtgt__r_fz'] },
  // Servo force error e = F_cmd + F_meas (world frame), published by the runtime's
  // computeHybridFMeas every substep. ‖e‖ → 0 when the wall reaction cancels F_cmd
  // (F_meas → −F_cmd) — the numeric proof of the position-force cancellation.
  ferr: { l: ['wbt_ferr__l_fx', 'wbt_ferr__l_fy', 'wbt_ferr__l_fz'], r: ['wbt_ferr__r_fx', 'wbt_ferr__r_fy', 'wbt_ferr__r_fz'] },
};

// Soft contact-phase label keys (C1 unified presets). Per-hand SCALARS ∈[0,1] —
// hysteresis(‖F_meas‖; 2/1.5 N) + 50 ms LPF, computed by the runtime's spring-wall sim.
// This is the actual actor obs channel (WbtContactPhase), so surfacing it lets the
// operator see what the policy's contact input is doing vs the raw force.
const _WBT_CPHASE_KEYS = { l: 'wbt_cphase__l', r: 'wbt_cphase__r' };

/** True only on a C1 unified build. The runtime publishes `wbt_hybrid_observe_only`
 *  from extras.hybrid.servo_observe_only (1.0 / 0.0) every substep of the hybrid path;
 *  an UNSET key reads the NaN fallback, which is neither ≥0.5 nor finite, so every
 *  pre-unified hybrid preset (and every non-hybrid preset) reads false and keeps its
 *  panel byte-identical. Distinct from just checking whether wbt_cphase__* is published:
 *  the runtime computes that label for ALL hybrid presets, but only the unified actors
 *  actually OBSERVE it. */
function _wbtCphaseObserved(): boolean {
  return getPolicyExtra('wbt_hybrid_observe_only', Number.NaN) >= 0.5;
}

function _wbtVecMag(keys: string[]): number {
  const x = getPolicyExtra(keys[0], 0.0);
  const y = getPolicyExtra(keys[1], 0.0);
  const z = getPolicyExtra(keys[2], 0.0);
  return Math.sqrt(x * x + y * y + z * z);
}

let _wbtPanel: WbtPanelEls | null = null;

function ensureWbtPanel(): WbtPanelEls {
  if (_wbtPanel) return _wbtPanel;
  const root = document.createElement('div');
  root.id = 'wbt-realtime-metrics';
  Object.assign(root.style, {
    position: 'fixed',
    right: '12px',
    bottom: '12px',
    padding: '12px 16px',
    background: 'rgba(20, 24, 32, 0.85)',
    color: '#f0f0f0',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    fontSize: '13px',
    lineHeight: '1.5',
    minWidth: '240px',
    borderRadius: '6px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    zIndex: '10000',
    pointerEvents: 'none',
  } as Partial<CSSStyleDeclaration>);

  const title = document.createElement('div');
  title.textContent = 'Tracking errors';
  Object.assign(title.style, {
    fontWeight: 'bold',
    marginBottom: '6px',
    borderBottom: '1px solid rgba(255,255,255,0.2)',
    paddingBottom: '4px',
  } as Partial<CSSStyleDeclaration>);
  root.appendChild(title);

  // Returns [valueSpan, lineDiv] so hybrid-only rows can be hidden per preset.
  function row(label: string): [HTMLSpanElement, HTMLDivElement] {
    const line = document.createElement('div');
    const lab = document.createElement('span');
    lab.textContent = label;
    Object.assign(lab.style, {
      color: '#a0a0a0',
      display: 'inline-block',
      minWidth: '80px',
    } as Partial<CSSStyleDeclaration>);
    const val = document.createElement('span');
    val.textContent = '—';
    val.style.fontWeight = 'bold';
    line.appendChild(lab);
    line.appendChild(val);
    root.appendChild(line);
    return [val, line];
  }
  const [motionEl] = row('motion: ');
  const [frameEl] = row('frame:  ');
  const [jointEl] = row('joint:  ');
  const [bodyEl] = row('body:   ');
  const [rootEl] = row('root:   ');
  const [bootEl] = row('boot:   ');
  // Per-hand force-tracking error (force preset only; '—' otherwise). raw =
  // |hand − ref|; eq2 = |hand − (ref + F^cmd/K)| (force-target residual, the
  // honest force-tracking number when F^cmd ≠ 0). Each shows L / R.
  const [forceRawEl] = row('F raw:  ');
  const [forceEq2El] = row('F eq2:  ');
  // Hybrid-only rows (hidden unless the runtime publishes the hybrid extras):
  // F^meas magnitude (what the wrist feels), the servo integrator I (N), and the
  // task-space setpoint shift Δx (cm). Each shows L / R.
  const [forceMeasEl, forceMeasLine] = row('F meas: ');
  const [forceErrEl, forceErrLine] = row('e=Fc+Fm:');
  // C1 unified only: the soft contact label the actor actually observes.
  const [contactPhaseEl, contactPhaseLine] = row('φ contact:');
  const [integEl, integLine] = row('I (N):  ');
  const [deltaXEl, deltaXLine] = row('Δx(cm): ');
  forceMeasLine.style.display = 'none';
  forceErrLine.style.display = 'none';
  contactPhaseLine.style.display = 'none';
  integLine.style.display = 'none';
  deltaXLine.style.display = 'none';

  document.body.appendChild(root);
  _wbtPanel = {
    root,
    motion: motionEl,
    frame: frameEl,
    jointErr: jointEl,
    bodyErr: bodyEl,
    rootErr: rootEl,
    bootErr: bootEl,
    forceRaw: forceRawEl,
    forceEq2: forceEq2El,
    forceMeas: forceMeasEl,
    integ: integEl,
    deltaX: deltaXEl,
    forceErr: forceErrEl,
    contactPhase: contactPhaseEl,
    forceMeasLine,
    integLine,
    deltaXLine,
    forceErrLine,
    contactPhaseLine,
  };
  return _wbtPanel;
}

export class WbtMetricsTap extends ObservationBase {
  private bootBuf: number[] = [];
  private currentMotion: string = '';

  get size(): number {
    return 0;
  }

  compute(state: PolicyState): Float32Array {
    const tr = getWbtTrackingForMetrics();
    if (!tr || !tr.isReady() || tr.refLen === 0) {
      return new Float32Array(0);
    }
    const motionName = tr.getSelectedMotionName() ?? '';
    if (motionName !== this.currentMotion) {
      this.currentMotion = motionName;
      this.bootBuf = [];
    }

    const refIdx = Math.min(tr.refIdx, tr.refLen - 1);
    const refDof = tr.refJointPos[refIdx];
    if (!refDof) return new Float32Array(0);
    const q = state.jointPos ?? new Float32Array(29);

    let je = 0.0;
    const n = Math.min(q.length, refDof.length);
    for (let i = 0; i < n; i++) je += Math.abs((q[i] ?? 0) - (refDof[i] ?? 0));
    je = (je / Math.max(1, n)) * WBT_RAD2DEG;

    const refRoot = tr.refRootPos[refIdx];
    let re = 0.0;
    if (refRoot && state.rootPos) {
      const dx = state.rootPos[0] - refRoot[0];
      const dy = state.rootPos[1] - refRoot[1];
      const dz = state.rootPos[2] - refRoot[2];
      re = Math.hypot(dx, dy, dz) * 100;
    }

    // body_err is approximated by root_err in v1 — a real per-body err would
    // require iterating mjData.xpos against the parent's refBodyPosW (not
    // exposed publicly). The panel labels this clearly as "(root proxy)".
    const be = re;

    this.bootBuf.push(je);
    if (this.bootBuf.length > WBT_BOOT_WINDOW) this.bootBuf.shift();
    let bootSq = 0.0;
    for (const v of this.bootBuf) bootSq += v * v;
    const bootRms = Math.sqrt(bootSq / Math.max(1, this.bootBuf.length));

    const p = ensureWbtPanel();
    p.motion.textContent = motionName || '—';
    p.frame.textContent = `${refIdx}/${tr.refLen}`;
    p.jointErr.textContent = `${je.toFixed(2)}°`;
    p.bodyErr.textContent = `${be.toFixed(2)} cm  (root proxy)`;
    p.rootErr.textContent = `${re.toFixed(2)} cm`;
    p.bootErr.textContent = `${bootRms.toFixed(2)}°  (RMS over ${this.bootBuf.length}/${WBT_BOOT_WINDOW} frames)`;

    // Per-hand force-tracking error (force preset only). Render '—' when the
    // clip has no force reference (synthetic/idle) or a value is NaN.
    const fe = tr.getForceHandErrCm?.() ?? null;
    p.forceRaw.textContent = _fmtLR(fe?.[0], fe?.[1]);
    p.forceEq2.textContent = _fmtLR(fe?.[2], fe?.[3]);

    // Hybrid-only rows: F^meas magnitude, servo integrator I, setpoint shift Δx.
    // Detected by whether the runtime has published any F^meas key (it publishes
    // wbt_force_meas__* only for the hybrid preset — via getPolicyExtra's NaN
    // fallback, an UNSET key reads NaN, a published key reads a finite number).
    // The three lines stay hidden for every other preset, so the panel is
    // unchanged there.
    const measPublished =
      Number.isFinite(getPolicyExtra('wbt_force_meas__l_fx', Number.NaN)) ||
      Number.isFinite(getPolicyExtra('wbt_force_meas__r_fx', Number.NaN)) ||
      Number.isFinite(getPolicyExtra('wbt_force_meas__l_fy', Number.NaN)) ||
      Number.isFinite(getPolicyExtra('wbt_force_meas__r_fy', Number.NaN)) ||
      Number.isFinite(getPolicyExtra('wbt_force_meas__l_fz', Number.NaN)) ||
      Number.isFinite(getPolicyExtra('wbt_force_meas__r_fz', Number.NaN));
    if (measPublished) {
      p.forceMeasLine.style.display = '';
      p.forceErrLine.style.display = '';
      p.integLine.style.display = '';
      p.deltaXLine.style.display = '';
      const measL = _wbtVecMag(_WBT_HYBRID_KEYS.meas.l);
      const measR = _wbtVecMag(_WBT_HYBRID_KEYS.meas.r);
      const errL = _wbtVecMag(_WBT_HYBRID_KEYS.ferr.l);
      const errR = _wbtVecMag(_WBT_HYBRID_KEYS.ferr.r);
      const integL = _wbtVecMag(_WBT_HYBRID_KEYS.integ.l);
      const integR = _wbtVecMag(_WBT_HYBRID_KEYS.integ.r);
      const dxL = _wbtVecMag(_WBT_HYBRID_KEYS.xtgt.l) * 100.0; // m → cm
      const dxR = _wbtVecMag(_WBT_HYBRID_KEYS.xtgt.r) * 100.0;
      p.forceMeas.textContent = `L ${measL.toFixed(1)} / R ${measR.toFixed(1)} N`;
      // ‖e‖ = ‖F_cmd + F_meas‖ → 0 when the wall reaction cancels the command.
      p.forceErr.textContent = `L ${errL.toFixed(1)} / R ${errR.toFixed(1)} N`;
      p.integ.textContent = `L ${integL.toFixed(1)} / R ${integR.toFixed(1)}`;
      p.deltaX.textContent = `L ${dxL.toFixed(2)} / R ${dxR.toFixed(2)}`;
      // Soft contact phase φ ∈[0,1] per hand — the C1 unified actor obs channel.
      // Gated SEPARATELY from the F^meas block: the runtime publishes wbt_cphase__*
      // for EVERY hybrid preset (it is computed alongside F^meas), but only the
      // unified presets OBSERVE it, so showing the row on a servo-ON 2pt/3pt-hybrid
      // build would imply an obs channel that policy does not have. `_wbtCphaseObserved`
      // reads the flag the runtime sets from extras.hybrid.servo_observe_only.
      if (_wbtCphaseObserved()) {
        p.contactPhaseLine.style.display = '';
        const phL = getPolicyExtra(_WBT_CPHASE_KEYS.l, 0.0);
        const phR = getPolicyExtra(_WBT_CPHASE_KEYS.r, 0.0);
        p.contactPhase.textContent = `L ${phL.toFixed(2)} / R ${phR.toFixed(2)}`;
      } else {
        p.contactPhaseLine.style.display = 'none';
      }
    } else {
      // Re-hide the hybrid rows if F^meas is no longer published (e.g. a live
      // switch to a non-hybrid preset without a page reload). The panel is a
      // module singleton, so we must actively hide, not just skip the update.
      p.forceMeasLine.style.display = 'none';
      p.forceErrLine.style.display = 'none';
      p.contactPhaseLine.style.display = 'none';
      p.integLine.style.display = 'none';
      p.deltaXLine.style.display = 'none';
    }

    return new Float32Array(0);
  }

  override reset(): void {
    this.bootBuf = [];
  }
}
