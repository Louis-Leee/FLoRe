import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import type { MainModule, MjData, MjModel } from 'mujoco';
import {
  downloadExampleScenesFolder,
  getPosition,
  getQuaternion,
  loadSceneFromURL,
} from '../scene/scene';
import { type SplatConfig, type SplatMesh, loadSplat, disposeSplat, applySplatTransform } from '../scene/splat';
import { loadCollider, disposeCollider } from '../scene/collider';
import { DragStateManager } from '../utils/dragStateManager';
import { createTendonState, updateTendonGeometry, updateTendonRendering } from '../scene/tendons';
import { updateHeadlightFromCamera, updateLightsFromData } from '../scene/lights';
import { threeToMjcCoordinate } from '../scene/coordinate';
import {
  type ViewerConfig,
  type ViewerState,
  applyViewerConfig,
  updateCameraFromData,
} from './viewer_config';
import { SceneCacheManager } from '../cache/sceneCacheManager';
import { SceneResourceTracker } from '../cache/resourceTracker';
import { MemoryMonitor } from '../cache/memoryMonitor';
import { Observations } from '../observation/observations';
import { TerminationManager } from '../termination/TerminationManager';
import { Terminations } from '../termination/terminations';
import * as ort from 'onnxruntime-web';
import { PolicyRunner } from '../policy/PolicyRunner';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { PolicyConfig } from '../policy/types';
import { resetPolicyExtras, getPolicyExtra, setPolicyExtra, type SliderConfig } from '../policy/extras';
import { mountSliderPanel, resetSlidersToDefault } from '../policy/SliderPanel';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { getCommandManager, type CommandTermContext, type CommandsConfig } from '../command';
import { EventManager } from '../event/EventManager';
import { Events } from '../event/events';
import type { TerrainData } from '../event/EventBase';

type RuntimeOptions = {
  baseUrl?: string;
};

type MotionCommandTerm = {
  setSelectedMotion(name: string | null): Promise<boolean> | boolean;
  setReferenceVisible?(visible: boolean): void;
  getSelectedMotionName?(): string | null;
};

function isMotionCommandTerm(term: unknown): term is MotionCommandTerm {
  return (
    typeof term === 'object' &&
    term !== null &&
    typeof (term as MotionCommandTerm).setSelectedMotion === 'function'
  );
}

/** Thrown when a scene exceeds the browser's 2 GB WebAssembly memory limit. */
export class WasmMemoryLimitError extends Error {
  constructor() {
    super(
      "This scene cannot be loaded because it exceeds the browser's " +
        'WebAssembly 2 GB memory limit. ' +
        'Try closing other browser tabs or reloading the page to free memory.'
    );
    this.name = 'WasmMemoryLimitError';
  }
}

// mj_loadXML surfaces OOM via four distinct paths depending on which internal
// allocator hits the 2 GB ceiling first (null return, MuJoCo error string,
// lodepng error string, or raw bad_alloc).
function isWasmOom(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    msg.includes('MjModel loading returned null') ||
    msg.includes('Could not allocate memory') ||
    msg.includes('memory allocation failed') ||
    msg.includes('bad_alloc')
  );
}

type BodyState = {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

// One external-force (F^ext) injection target. bodyId: -2 = unresolved,
// -1 = not found, >=0 = resolved (lazy by-name resolution in
// applyExternalForces). offset = wrist-local contact-point offset (z-up).
type ForceExtBody = {
  bodyName: string;
  bodyId: number;
  offset: [number, number, number];
  sliderKeys: [string, string, string];
};

// ── Hybrid position-force controller (exp:g1-wbt-2pt-hybrid) browser-side sim ──
// Reconstructs, in the browser, the training-time causal virtual spring wall +
// simulated F/T sensor (F_meas) + Jᵀ force servo so the servo-ON checkpoint (which
// was trained assuming a live servo adds Δq and that F_meas/I/Δx are live obs)
// behaves correctly. All quantities are native MuJoCo z-up; the obs classes rotate
// world→pelvis themselves. Everything is gated on `hybridConfig` being present, so
// non-hybrid policies are entirely unaffected. See
// docs/superpowers/specs/2026-07-11-hybrid-force-viz-design.md and
// src/extensions/wbt_wrappers/wbt_training/config_values/hybrid.py.
type HybridHandCfg = {
  bodyName: string; // wrist_yaw link
  // Per-hand F^cmd slider keys (pelvis-frame Newtons) the servo + wall read.
  cmdSliderKeys: [string, string, string];
  // 7 arm joint names (shoulder p/r/y, elbow, wrist r/p/y) the Jᵀ servo drives.
  armJointNames: string[];
};
type HybridConfig = {
  servoEnabled: boolean;
  // C1 observe-only integrator (exp:g1-wbt-3pt-hybrid-global-vel-unified[-anneal]):
  // the integrator keeps computing I/Δx (published to the obs bus) but Δq is NEVER
  // injected into ctrl — the policy is the executor (hybrid.py servo_observe_only).
  servoObserveOnly: boolean;
  // Soft contact-phase label (hysteresis on ‖F_meas‖ + first-order LPF), ∈[0,1] per
  // hand, published as wbt_cphase__{l,r} for the 185D unified obs (WbtContactPhase).
  contactPhaseThetaOn: number; // N — hysteresis ON threshold (training 2.0)
  contactPhaseThetaOff: number; // N — hysteresis OFF threshold (training 1.5)
  contactPhaseTauS: number; // s — LPF time constant (training 0.05)
  // Post-reset suppression window, in ENV steps (training contact_phase_warmup_steps=5).
  // Converted to substeps internally (× decimation) since we run the LPF per substep.
  contactPhaseWarmupSteps: number;
  stiffnessK: number; // 200 N/m — maps task wrench → Δx setpoint shift
  fCmdMax: number; // 25 N (unused for clamping here; kept for parity/labels)
  kEnv: number; // demo contact stiffness N/m (geo-mean of [k_env_min,k_env_max])
  planeDist: number; // m — wall placed this far ahead of the rest TCP along +F_cmd
  maxPenetrationM: number; // 0.08 — depth clamp
  maxForceN: number; // 35 — |F_ext| clamp post-spring
  distalMassKg: number; // 1.5 — inertial term in F_meas
  sensorNoiseStd: number; // demo default 0 (clean arrows); training 0.5
  servoGainK: number; // 0.05 rad/(N·m-ish) — Δq = k·Jᵀ·task_w
  integratorGainKi: number; // 0.5 /s
  integratorMax: number; // 40 N — anti-windup clip on I (force units)
  servoDqMax: number; // 0.15 rad — per-joint |Δq| clamp
  tcpOffsetX: number; // 0.242 — the SINGLE contact anchor (no lateral y)
  hands: HybridHandCfg[];
};
// Resolved-once runtime handles + evolving per-hand state for the hybrid sim.
type HybridHandRt = {
  bodyId: number; // wrist_yaw body id (-2 unresolved, -1 not found)
  armQposAdr: number[]; // qpos address of each of the 7 arm joints
  armQvelAdr: number[]; // qvel/dof address of each arm joint (for Jacobian cols)
  armCtrl: Array<{ ctrlIndex: number; positionActuator: boolean; kp: number }>;
  // Wall (world frame), seeded when F^cmd first becomes non-zero (and on reset).
  wallSeeded: boolean;
  normalW: [number, number, number]; // outward wall normal = −unit(F_cmd_world)
  pointW: [number, number, number]; // a point on the wall plane
  fcmdDirW: [number, number, number]; // episode-fixed unit F_cmd WORLD direction (= −normalW)
  // TCP at seed time (world) — the PIVOT the plane is anchored to. Training anchors
  // plane_point = tcp0 − plane_dist·n̂ at reset (hybrid.py reset()); we keep tcp0 so
  // the LIVE-normal mode (wbt_hybrid_wall_live) can re-derive the plane about the SAME
  // pivot as the direction turns. Re-deriving the pivot from the LIVE TCP instead would
  // keep the plane permanently plane_dist ahead of the hand → depth pinned at 0, i.e.
  // a wall that can never be pushed. Unused in the default episode-fixed mode.
  seedTcpW: [number, number, number];
  // Evolving servo/contact state (world frame).
  integ: [number, number, number]; // I_force [N]
  deltaX: [number, number, number]; // Δx [m]
  fExtW: [number, number, number]; // causal contact force [N] (cache for arrow)
  fMeasW: [number, number, number]; // simulated F/T reading [N] (cache for arrow)
  depth: number; // penetration [m]
  gate: number; // contact flag (1 if depth>0)
  // a_wrist finite-difference state.
  prevTcpW: [number, number, number] | null;
  prevVelW: [number, number, number] | null;
  // Soft contact-phase label state (hysteresis + LPF on ‖F_meas‖), C1 unified preset.
  cphase: number; // LPF output ∈[0,1] (published as wbt_cphase__{l,r})
  chyst: number; // binary hysteresis state (0/1)
  // Substeps elapsed since the last reset/motion-switch. Mirrors training's
  // contact_phase_warmup_steps gate (hybrid.py): the reset teleport puts a huge
  // spurious m·a spike into F_meas (a_wrist is double-finite-differenced from TCP
  // position, so a pose JUMP reads as enormous acceleration), which would slam the
  // soft label to 1 in free space. Training forces hysteresis OFF for the first few
  // env steps of each episode; we do the same, counted in substeps.
  cwarm: number;
};

export class mjswanRuntime {
  private mujoco: MainModule;
  private container: HTMLElement;
  private baseUrl: string;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private mjModel: MjModel | null;
  private mjData: MjData | null;
  private bodies: Record<number, THREE.Group> | null;
  private lights: THREE.Light[];
  private mujocoRoot: THREE.Group | null;
  private lastSimState: {
    bodies: Map<number, BodyState>;
    tendons: ReturnType<typeof createTendonState>;
  };
  private dynamicBodyIds: Set<number> | null;
  private loopPromise: Promise<void> | null;
  private running: boolean;
  private timestep: number;
  private decimation: number;
  private loadingScene: Promise<void> | null;
  private resizeObserver: ResizeObserver | null;
  private dragStateManager: DragStateManager | null;
  private dragForceScale: number;
  // Optional external-force (F^ext) injection config, populated from
  // policyConfig.extras.force_ext_bodies (holosoma-2pt-force preset only; null
  // otherwise so this is a no-op for every other policy). Each entry pins a
  // body (by resolved id), a wrist-local offset to the contact point, and the
  // 3 slider keys carrying the pelvis-frame Newtons. See applyExternalForces.
  private forceExtBodies: ForceExtBody[] | null;
  // Hybrid position-force controller browser-side sim (exp:g1-wbt-2pt-hybrid).
  // null for every non-hybrid policy → all hybrid code paths are no-ops. Parsed
  // from policyConfig.extras.hybrid in loadPolicyConfig; per-hand runtime handles
  // are resolved lazily in the substep loop once mjModel is ready.
  private hybridConfig: HybridConfig | null;
  private hybridHands: HybridHandRt[] | null;
  private sceneCacheManager: SceneCacheManager;
  private resourceTracker: SceneResourceTracker;
  private memoryMonitor: MemoryMonitor;
  private policyRunner: PolicyRunner | null;
  private policyStateBuilder: PolicyStateBuilder | null;
  private policyConfigPath: string | null;
  private policyDebugCounter: number;
  private policyControl: Array<{
    controlType: string;
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    // Indices into the flat policy action vector for this term's joints.
    actionIndices: number[];
    actionScale: Float32Array;
    actionOffset: Float32Array;
    defaultJointPos: Float32Array;
    encoderBias: Float32Array;
    // Per-actuator flag: true = position actuator (ctrl=target_pos, PD internal),
    // false = motor actuator (ctrl=torque, PD computed in browser from kp/kd).
    positionActuator: boolean[];
    kp: Float32Array;
    kd: Float32Array;
  }> | null;
  private onnxModule: OnnxModule | null;
  private onnxInputDict: Record<string, ort.Tensor> | null;
  private onnxInferencing: boolean;
  private onnxTimeStep: number;
  // Optional pre-clip of observation values before they hit a policy with a
  // baked-in normalizer (cf. holosoma's wbt_wrappers_inference.utils.normalizer
  // — `obs' = clip((obs - mean)/std, -clip, clip) * std + mean`). Populated
  // from policyConfig.extras when present; null disables pre-clip.
  private onnxNormMean: Float32Array | null;
  private onnxNormStd: Float32Array | null;
  private onnxNormClip: number;
  private terminationManager: TerminationManager | null;
  private eventManager: EventManager | null;
  private terrainData: TerrainData | null;
  private vrButton: HTMLElement | null;
  private splatMesh: SplatMesh | null;
  private colliderMesh: THREE.Group | null;
  private cameraState: ViewerState;

  constructor(mujoco: MainModule, container: HTMLElement, options: RuntimeOptions = {}) {
    this.mujoco = mujoco;
    this.container = container;
    this.baseUrl = options.baseUrl || '/';

    const workingPath = '/working';
    try {
      this.mujoco.FS.mkdir(workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST') {
        console.warn('Failed to create /working directory:', error);
      }
    }
    try {
      this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST' && error.code !== 'EBUSY') {
        console.warn('Failed to mount MEMFS at /working:', error);
      }
    }

    const { width, height } = this.getSize();

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 1000);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.xr.enabled = true;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    this.vrButton = null;
    navigator.xr?.isSessionSupported('immersive-vr').then((supported) => {
      if (supported) {
        this.vrButton = VRButton.createButton(this.renderer);
        document.body.appendChild(this.vrButton);
      }
    });

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.2, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this.renderer.setAnimationLoop(this.render);
    window.addEventListener('resize', this.onWindowResize);

    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(() => this.onWindowResize());
      this.resizeObserver.observe(this.container);
    } else {
      this.resizeObserver = null;
    }

    this.lastSimState = {
      bodies: new Map(),
      tendons: createTendonState(),
    };
    this.dynamicBodyIds = null;

    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.loopPromise = null;
    this.running = false;
    this.timestep = 0.001;
    this.decimation = 1;
    this.loadingScene = null;
    this.dragStateManager = null;
    this.dragForceScale = 100.0;
    this.forceExtBodies = null;
    this.hybridConfig = null;
    this.hybridHands = null;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyConfigPath = null;
    this.policyDebugCounter = 0;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.onnxNormMean = null;
    this.onnxNormStd = null;
    this.onnxNormClip = 3.0;
    this.terminationManager = null;
    this.eventManager = null;
    this.terrainData = null;
    this.splatMesh = null;
    this.colliderMesh = null;
    this.cameraState = { trackBodyId: null, prevBodyPos: null };

    // Initialize cache system (singleton shared across runtime instances)
    this.sceneCacheManager = SceneCacheManager.getInstance(this.mujoco);
    this.resourceTracker = new SceneResourceTracker();
    this.memoryMonitor = new MemoryMonitor();
  }

  async loadEnvironment(
    scenePath: string,
    policyConfigPath: string | null = null,
    splatConfig: SplatConfig | null = null,
    cameraConfig: ViewerConfig | null = null,
    eventsConfig: import('../event/EventBase').EventConfig[] | null = null,
    terrainData: TerrainData | null = null
  ): Promise<void> {
    this.terrainData = terrainData;
    if (eventsConfig && eventsConfig.length > 0) {
      this.eventManager = new EventManager(eventsConfig, Events);
      console.log(`[EventManager] ${this.eventManager.size} reset event(s) loaded`);
    } else {
      this.eventManager = null;
    }
    await this.stop();

    // Dispose previous splat/collider before switching scenes
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    const startTime = performance.now();

    // Initialize CommandManager with default velocity commands
    this.initializeCommands();

    // Check cache first
    if (this.sceneCacheManager.has(scenePath)) {
      await this.restoreFromCache(scenePath);
      const elapsed = performance.now() - startTime;
      this.memoryMonitor.logCacheOperation('hit', scenePath, { elapsedMs: elapsed });
    } else {
      this.memoryMonitor.logCacheOperation('miss', scenePath);

      // Prepare cache for new scene (may trigger eviction)
      await this.sceneCacheManager.prepareForNewScene();

      // Clear current references before loading new scene
      // This prevents loadSceneFromURL from deleting cached objects
      this.mjModel = null;
      this.mjData = null;
      this.bodies = null;
      this.lights = [];
      this.mujocoRoot = null;
      this.dynamicBodyIds = null;

      // Start tracking resources
      this.resourceTracker.startTracking(this.mujoco);

      // Normal load
      await downloadExampleScenesFolder(this.mujoco, scenePath, this.baseUrl);
      await this.loadSceneWithOomRetry(scenePath);

      // Capture and cache resources
      await this.captureAndCacheResources(scenePath);
    }

    // Load splat and optional collider
    if (splatConfig) {
      this.splatMesh = loadSplat(splatConfig, this.scene);
      if (splatConfig.colliderUrl) {
        this.colliderMesh = await loadCollider(
          this.resolveAssetUrl(splatConfig.colliderUrl),
          this.scene
        );
      }
    }

    await this.loadPolicyConfig(policyConfigPath);

    this.applyViewerConfig(cameraConfig);

    this.running = true;
    void this.startLoop();
  }

  /**
   * Initialize the CommandManager (clear and set up reset callback)
   * Commands are registered from policy config in loadPolicyConfig()
   */
  private initializeCommands(): void {
    const commandManager = getCommandManager();
    commandManager.clear();
    commandManager.setResetCallback(() => this.resetSimulation());
  }

  /**
   * Initialize commands from policy config
   */
  private initializeCommandsFromConfig(
    commands: CommandsConfig,
    context: CommandTermContext
  ): void {
    const commandManager = getCommandManager();
    commandManager.initialize(commands, context);
    console.log('[mjswanRuntime] Commands loaded from policy config:', Object.keys(commands));
  }

  /**
   * Public method to reset the simulation state
   * Can be called from UI components via the CommandManager
   */
  resetSimulation(): void {
    this.resetSimulationState();
    if (this.policyRunner && this.policyStateBuilder) {
      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
    }
    console.log('[mjswanRuntime] Simulation reset');
  }

  async setSelectedMotion(motionName: string | null): Promise<boolean> {
    const term = getCommandManager().getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return false;
    }
    const accepted = await term.setSelectedMotion(motionName);
    if (accepted) {
      this.resetSimulation();
    }
    return accepted;
  }

  getSelectedMotionName(): string | null {
    const term = getCommandManager().getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return null;
    }
    return term.getSelectedMotionName?.() ?? null;
  }

  setReferenceVisible(visible: boolean): void {
    const term = getCommandManager().getTerm('motion');
    if (!isMotionCommandTerm(term) || typeof term.setReferenceVisible !== 'function') {
      return;
    }
    term.setReferenceVisible(visible);
  }

  async loadScene(scenePath: string): Promise<void> {
    if (this.loadingScene) {
      await this.loadingScene;
    }

    this.loadingScene = (async () => {
      const existingRoot = this.scene.getObjectByName('MuJoCo Root');
      if (existingRoot) {
        this.scene.remove(existingRoot);
      }

      const parent = {
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
      };

      [this.mjModel, this.mjData, this.bodies, this.lights] = await loadSceneFromURL(
        this.mujoco,
        scenePath,
        parent
      );

      if (!this.mjModel || !this.mjData) {
        throw new Error('Failed to load MuJoCo model.');
      }

      this.mujocoRoot = this.scene.getObjectByName('MuJoCo Root') as THREE.Group | null;

      this.mujoco.mj_forward(this.mjModel, this.mjData);
      updateLightsFromData(this.mujoco, this.mjData, this.lights);
      updateHeadlightFromCamera(this.camera, this.lights);
      this.dynamicBodyIds = this.computeDynamicBodyIds(this.mjModel);
      this.syncStaticBodiesFromData();

      this.timestep = this.mjModel.opt.timestep || 0.001;
      this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

      this.lastSimState.bodies.clear();
      this.updateCachedState();

      // Initialize DragStateManager
      if (!this.dragStateManager) {
        this.dragStateManager = new DragStateManager({
          scene: this.scene,
          renderer: this.renderer,
          camera: this.camera,
          container: this.container,
          controls: this.controls,
          draggableBodyIds: this.dynamicBodyIds,
        });
      } else {
        this.dragStateManager.setDraggableBodyIds(this.dynamicBodyIds);
      }

      this.loadingScene = null;
    })();

    await this.loadingScene;
  }

  // On OOM, evict all cached scenes to reclaim the WASM malloc free-list and
  // retry once.  If the retry also fails, throw WasmMemoryLimitError.
  private async loadSceneWithOomRetry(scenePath: string): Promise<void> {
    try {
      await this.loadScene(scenePath);
    } catch (error) {
      if (!isWasmOom(error)) {
        throw error;
      }
      console.warn('[mjswanRuntime] OOM — clearing cache and retrying...');
      this.loadingScene = null;
      await this.sceneCacheManager.clear();
      try {
        await this.loadScene(scenePath);
      } catch (retryError) {
        this.loadingScene = null;
        if (isWasmOom(retryError)) {
          throw new WasmMemoryLimitError();
        }
        throw retryError;
      }
    }
  }

  async startLoop(): Promise<void> {
    if (this.loopPromise) {
      return this.loopPromise;
    }
    this.running = true;
    this.loopPromise = this.mainLoop();
    return this.loopPromise;
  }

  async setSplat(config: SplatConfig | null): Promise<void> {
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }
    if (config) {
      this.splatMesh = loadSplat(config, this.scene);
      if (config.colliderUrl) {
        this.colliderMesh = await loadCollider(
          this.resolveAssetUrl(config.colliderUrl),
          this.scene
        );
      }
    }
  }

  /** Update transform of the existing splat without disposing/reloading. */
  calibrateSplat(config: SplatConfig): void {
    if (this.splatMesh) {
      applySplatTransform(this.splatMesh, config);
    }
  }

  setSplatVisible(visible: boolean): void {
    if (this.splatMesh) {
      this.splatMesh.visible = visible;
    }
  }

  async stop(): Promise<void> {
    this.running = false;
    const pending = this.loopPromise;
    if (pending) {
      await pending;
    }
    this.loopPromise = null;
  }

  private async mainLoop(): Promise<void> {
    while (this.running) {
      const loopStart = performance.now();
      const target = this.timestep * this.decimation;

      if (this.mjModel && this.mjData) {
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        if (this.policyRunner && this.policyStateBuilder) {
          const state = this.policyStateBuilder.build();
          const obs = this.policyRunner.collectObservationsByKey(state);
          await this.runOnnxInference(obs);
          if (this.policyDebugCounter % 60 === 0) {
            const debugKey =
              'policy' in obs
                ? 'policy'
                : 'observation' in obs
                  ? 'observation'
                  : Object.keys(obs)[0];
            const debugObs = debugKey ? obs[debugKey] : null;
            const preview = debugObs ? Array.from(debugObs.slice(0, 8)) : [];
            console.log('[PolicyRunner] obs', {
              key: debugKey,
              size: debugObs ? debugObs.length : 0,
              sample: preview,
            });
          }
          this.policyDebugCounter += 1;
        }
        this.executeSimulationSteps();
        this.updateCachedState();

        // Evaluate termination conditions after simulation step
        if (this.terminationManager && this.policyStateBuilder) {
          const postState = this.policyStateBuilder.build();
          const result = this.terminationManager.evaluate(postState);
          if (result.done) {
            this.resetSimulationState();
            this.terminationManager.reset();
            if (this.policyRunner) {
              const resetState = this.policyStateBuilder.build();
              this.policyRunner.reset(resetState);
            }
          }
        }

        // Commands are updated after the physics step to match mjlab training-time semantics:
        // the policy sees command values from the previous step, consistent with how mjlab
        // computes observations before stepping the environment.
        getCommandManager().update(target);
        getCommandManager().updateDebugVisuals();
      }

      const elapsed = (performance.now() - loopStart) / 1000;
      const sleepTime = Math.max(0, target - elapsed);
      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
      }
    }
    this.loopPromise = null;
  }

  private async loadPolicyConfig(policyConfigPath: string | null): Promise<void> {
    const previousPolicyConfigPath = this.policyConfigPath;
    this.policyConfigPath = policyConfigPath;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyDebugCounter = 0;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.onnxNormMean = null;
    this.onnxNormStd = null;
    this.onnxNormClip = 3.0;
    this.forceExtBodies = null;
    this.hybridConfig = null;
    this.hybridHands = null;
    this.terminationManager = null;
    // Note: eventManager and terrainData are scene-level state set in loadEnvironment; do not clear here.

    // Clear existing commands when switching policies
    const commandManager = getCommandManager();
    commandManager.clear();
    commandManager.setResetCallback(() => this.resetSimulation());

    if (!policyConfigPath) {
      return;
    }

    if (!this.mjModel || !this.mjData) {
      console.warn('Policy config loaded before MuJoCo model is ready.');
      return;
    }

    if (policyConfigPath !== previousPolicyConfigPath) {
      this.resetSimulationState();
    }

    try {
      const { config } = await this.fetchPolicyConfig(policyConfigPath);
      if (Array.isArray(config.motions)) {
        config.motions = config.motions.map((motion) => ({
          ...motion,
          path: this.resolveAssetUrl(
            this.resolvePolicyAssetPath(policyConfigPath, motion.path)
          ),
        }));
      }
      if (config.commands?.motion && Array.isArray(config.motions)) {
        config.commands.motion = {
          ...config.commands.motion,
          motions: config.motions,
        };
      }
      this.resetSimulationState();
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();

      // Initialize commands from policy config if present
      if (config.commands && typeof config.commands === 'object') {
        this.initializeCommandsFromConfig(config.commands as CommandsConfig, {
          mujoco: this.mujoco,
          mjModel: this.mjModel,
          mjData: this.mjData,
          scene: this.scene,
          bodies: this.bodies,
          mujocoRoot: this.mujocoRoot,
          requestReset: () => this.resetSimulation(),
        });
        getCommandManager().resetTerms();
        const motionTerm = getCommandManager().getTerm('motion');
        if (isMotionCommandTerm(motionTerm)) {
          await motionTerm.setSelectedMotion(
            config.motions?.find((motion) => motion.default)?.name
              ?? config.motions?.[0]?.name
              ?? null
          );
          motionTerm.setReferenceVisible?.(true);
        }
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();
      }

      if (!config.policy_joint_names || config.policy_joint_names.length === 0) {
        throw new Error('Policy config missing policy_joint_names.');
      }

      const runner = new PolicyRunner(config, {
        policyModules: {
          tracking: TrackingPolicy,
          locomotion: LocomotionPolicy,
        },
        observations: Observations,
      });

      await runner.init({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
      });

      this.policyRunner = runner;
      this.policyStateBuilder = new PolicyStateBuilder(
        this.mujoco,
        this.mjModel,
        this.mjData,
        runner.getPolicyJointNames()
      );

      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
      this.policyControl = this.buildPolicyControl(config, runner, this.policyStateBuilder);

      // Initialize termination manager if termination config is present
      if (config.terminations && Object.keys(config.terminations).length > 0) {
        this.terminationManager = new TerminationManager(config.terminations, Terminations);
        console.log(`[TerminationManager] ${this.terminationManager.size} termination term(s) loaded`);
      }

      if (config.onnx?.path) {
        const onnxPath = this.resolvePolicyAssetPath(policyConfigPath, config.onnx.path);
        const onnxUrl = this.resolveAssetUrl(onnxPath);
        const onnxConfig = { ...config.onnx, path: onnxUrl };
        const module = new OnnxModule(onnxConfig);
        await module.init();
        this.onnxModule = module;
        this.onnxInputDict = module.initInput();
      }

      // Optional baked-normalizer pre-clip parameters supplied by the policy
      // exporter (see holosoma_inference's load_normalizer_from_path). Extras
      // shape: { normalizer_mean: number[], normalizer_std: number[],
      // normalizer_clip?: number, normalizer_eps?: number }. Empty/missing
      // disables pre-clip (no-op for policies without baked normalization).
      this.onnxNormMean = null;
      this.onnxNormStd = null;
      this.onnxNormClip = 3.0;

      // Reset live policy extras (slider values, etc.) so a previous policy's
      // values don't leak into the new scene. Sliders are then re-mounted from
      // the new config below if it declares any.
      resetPolicyExtras();

      const extras = (config as { extras?: Record<string, unknown> }).extras;
      if (extras && typeof extras === 'object') {
        const meanRaw = extras['normalizer_mean'];
        const stdRaw = extras['normalizer_std'];
        if (Array.isArray(meanRaw) && Array.isArray(stdRaw) && meanRaw.length === stdRaw.length && meanRaw.length > 0) {
          const eps = typeof extras['normalizer_eps'] === 'number' ? (extras['normalizer_eps'] as number) : 1e-2;
          const mean = new Float32Array(meanRaw.length);
          const stdEps = new Float32Array(stdRaw.length);
          for (let i = 0; i < meanRaw.length; i++) {
            mean[i] = Number(meanRaw[i]) || 0.0;
            stdEps[i] = (Number(stdRaw[i]) || 0.0) + eps;
          }
          this.onnxNormMean = mean;
          this.onnxNormStd = stdEps;
          if (typeof extras['normalizer_clip'] === 'number') {
            this.onnxNormClip = extras['normalizer_clip'] as number;
          }
          console.log('[PolicyRunner] normalizer pre-clip enabled', {
            dim: mean.length,
            clip: this.onnxNormClip,
          });
        }

        // Generic live-policy slider panel. Each slider writes its current
        // value into the policy-extras singleton (`getPolicyExtra(key)`) so
        // observation classes and debug visualizers can read the same source.
        const slidersRaw = extras['sliders'];
        if (Array.isArray(slidersRaw) && slidersRaw.length > 0) {
          const sliders: SliderConfig[] = [];
          for (const s of slidersRaw) {
            if (
              s &&
              typeof s === 'object' &&
              typeof (s as SliderConfig).label === 'string' &&
              typeof (s as SliderConfig).key === 'string' &&
              typeof (s as SliderConfig).min === 'number' &&
              typeof (s as SliderConfig).max === 'number'
            ) {
              sliders.push(s as SliderConfig);
            }
          }
          mountSliderPanel(sliders);
        } else {
          mountSliderPanel(undefined);
        }

        // Optional external-force (F^ext) injection config. Shape:
        //   force_ext_bodies: [{ body_name: string, offset_b: [x,y,z],
        //                        slider_keys: [fx_key, fy_key, fz_key] }, ...]
        // Each lists a body whose rubber-hand contact point receives a real
        // disturbance wrench injected into xfrc_applied (see
        // applyExternalForces). Body ids are resolved lazily (mjModel may not
        // be ready here). Absent => no-op (this.forceExtBodies stays null), so
        // every non-force policy is unaffected.
        this.forceExtBodies = null;
        const fextRaw = extras['force_ext_bodies'];
        if (Array.isArray(fextRaw) && fextRaw.length > 0) {
          const parsed: ForceExtBody[] = [];
          for (const e of fextRaw) {
            if (
              e &&
              typeof e === 'object' &&
              typeof (e as { body_name?: unknown }).body_name === 'string' &&
              Array.isArray((e as { offset_b?: unknown }).offset_b) &&
              Array.isArray((e as { slider_keys?: unknown }).slider_keys)
            ) {
              const ent = e as Record<string, unknown>;
              const off = ent.offset_b as unknown[];
              const keys = ent.slider_keys as unknown[];
              parsed.push({
                bodyName: ent.body_name as string,
                bodyId: -2,
                offset: [Number(off[0]) || 0, Number(off[1]) || 0, Number(off[2]) || 0],
                sliderKeys: [String(keys[0] ?? ''), String(keys[1] ?? ''), String(keys[2] ?? '')],
              });
            }
          }
          if (parsed.length > 0) {
            this.forceExtBodies = parsed;
            console.log('[PolicyRunner] F^ext injection enabled', {
              bodies: parsed.map((p) => p.bodyName),
            });
          }
        }

        // Optional hybrid position-force controller config (exp:g1-wbt-2pt-hybrid).
        // Drives the browser-side causal spring wall + F_meas + Jᵀ servo. Absent =>
        // no-op (hybridConfig stays null), so every non-hybrid policy is unaffected.
        this.hybridConfig = this.parseHybridConfig(extras['hybrid']);
        this.hybridHands = null; // resolved lazily once mjModel is ready
        if (this.hybridConfig) {
          console.log('[PolicyRunner] hybrid controller enabled', {
            servo: this.hybridConfig.servoEnabled,
            kEnv: this.hybridConfig.kEnv,
            hands: this.hybridConfig.hands.map((h) => h.bodyName),
          });
        }
      } else {
        mountSliderPanel(undefined);
      }

      console.log('[PolicyRunner] config loaded', {
        obsSize: runner.getObservationSize(),
        obsLayout: runner.getObservationLayout(),
        pdEnabled: this.policyControl !== null,
      });
    } catch (error) {
      console.warn('Failed to load policy config:', error);
    }
  }

  private async fetchPolicyConfig(
    policyConfigPath: string
  ): Promise<{ config: PolicyConfig; resolvedUrl: string }> {
    const resolved = this.resolveAssetUrl(policyConfigPath);
    const response = await fetch(resolved, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to fetch policy config: ${response.status}`);
    }
    const payload = await response.json();
    return { config: payload as PolicyConfig, resolvedUrl: resolved };
  }

  private resolveAssetUrl(assetPath: string): string {
    if (/^[a-z]+:\/\//i.test(assetPath)) {
      return assetPath;
    }
    const base = (this.baseUrl || '/').replace(/\/+$/, '/');
    const baseUrl = new URL(base, window.location.origin + '/').toString();
    return new URL(assetPath.replace(/^\/+/, ''), baseUrl).toString();
  }

  private resolvePolicyAssetPath(configPath: string, assetPath: string): string {
    const normalizedConfig = configPath.replace(/\\/g, '/');
    const lastSlash = normalizedConfig.lastIndexOf('/');
    if (lastSlash >= 0) {
      const dir = normalizedConfig.slice(0, lastSlash + 1);
      return `${dir}${assetPath}`.replace(/\/+/g, '/');
    }
    return assetPath;
  }

  private buildPolicyControl(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder
  ): Array<{
    controlType: string;
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    actionIndices: number[];
    actionScale: Float32Array;
    actionOffset: Float32Array;
    defaultJointPos: Float32Array;
    encoderBias: Float32Array;
    positionActuator: boolean[];
    kp: Float32Array;
    kd: Float32Array;
  }> | null {
    const jointNames = runner.getPolicyJointNames();
    const affineBiasValue = this.mujoco.mjtBias?.mjBIAS_AFFINE?.value ?? 1;

    const buildEntry = (
      termKey: string,
      controlType: string,
      mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] },
      configScale: number[] | number | Record<string, number> | undefined,
      configOffset: number[] | number | Record<string, number> | undefined,
      configStiffness: number[] | number | Record<string, number> | undefined,
      configDamping: number[] | number | Record<string, number> | undefined,
      useDefaultOffset: boolean
    ) => {
      const n = mapping.qposAdr.length;
      const subsetJointNames = mapping.actionIndices.map((i) => jointNames[i]);

      const actionScale = this.normalizeControlArray(configScale, n, 1.0, subsetJointNames);
      const actionOffset = this.normalizeControlArray(configOffset, n, 0.0, subsetJointNames);

      const allDefaultJointPos = useDefaultOffset
        ? runner.getDefaultJointPos()
        : new Float32Array(jointNames.length);
      const defaultJointPos = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        defaultJointPos[i] = allDefaultJointPos[mapping.actionIndices[i]];
      }

      const allEncoderBias = Array.isArray(config.encoder_bias)
        ? Float32Array.from(config.encoder_bias)
        : new Float32Array(jointNames.length);
      const encoderBias = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        encoderBias[i] = allEncoderBias[mapping.actionIndices[i]] ?? 0.0;
      }

      const kp = this.normalizeControlArray(configStiffness, n, 0.0, subsetJointNames);
      const kd = this.normalizeControlArray(configDamping, n, 0.0, subsetJointNames);

      // Detect per-actuator whether the scene uses position actuators (biastype=affine,
      // ctrl=target_pos, PD handled internally by MuJoCo) or motor actuators
      // (biastype=none, ctrl=torque, PD must be computed externally from kp/kd).
      const positionActuator: boolean[] = mapping.ctrlAdr.map((adr) => {
        if (adr < 0 || !this.mjModel) return false;
        return this.mjModel.actuator_biastype[adr] === affineBiasValue;
      });

      const isPosition = positionActuator.some(Boolean);
      const isMotor = positionActuator.some((v) => !v);
      if (isPosition && isMotor) {
        console.warn(`[PolicyRunner] Action term "${termKey}": mixed actuator types detected.`);
      }
      console.log(
        `[PolicyRunner] Action term "${termKey}" (${controlType}): ${n} joint(s), ` +
        `mode: ${isPosition ? 'position (ctrl=target_pos)' : 'motor (ctrl=torque, external PD)'}`
      );

      return {
        controlType,
        ctrlAdr: mapping.ctrlAdr,
        qposAdr: mapping.qposAdr,
        qvelAdr: mapping.qvelAdr,
        actionIndices: mapping.actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
      };
    };

    // ── Legacy path: no `actions` block, use flat top-level fields ──────────
    const actionsConfig = config.actions;
    if (!actionsConfig || Object.keys(actionsConfig).length === 0) {
      const controlType = config.control_type ?? 'joint_position';
      if (controlType !== 'joint_position' && controlType !== 'torque') {
        console.warn(`[PolicyRunner] Unsupported control_type: ${controlType}`);
        return null;
      }
      const baseMapping = stateBuilder.getControlMapping();
      if (!baseMapping) {
        console.warn('[PolicyRunner] Failed to build control mapping.');
        return null;
      }
      const mapping = {
        ...baseMapping,
        actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
      };
      return [buildEntry(
        'legacy',
        controlType,
        mapping,
        config.action_scale,
        undefined,
        config.stiffness,
        config.damping,
        true
      )];
    }

    // ── Multi-term path: iterate every entry in the `actions` block ─────────
    const results: Array<ReturnType<typeof buildEntry>> = [];

    for (const [termKey, actionTerm] of Object.entries(actionsConfig)) {
      const controlType = actionTerm.type ?? 'joint_position';
      if (controlType !== 'joint_position' && controlType !== 'torque') {
        console.warn(`[PolicyRunner] Action term "${termKey}": unsupported type "${controlType}", skipping.`);
        continue;
      }

      // If actuator_names is absent or [".*"], match all joints (backward-compatible).
      const patterns = actionTerm.actuator_names ?? ['.*'];
      const isMatchAll = patterns.length === 1 && patterns[0] === '.*';

      let mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] } | null;

      if (isMatchAll) {
        const baseMapping = stateBuilder.getControlMapping();
        if (!baseMapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": failed to build control mapping, skipping.`);
          continue;
        }
        mapping = {
          ...baseMapping,
          actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
        };
      } else {
        mapping = stateBuilder.getControlMappingFor(patterns, jointNames);
        if (!mapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": no joints matched patterns [${patterns.join(', ')}], skipping.`);
          continue;
        }
      }

      const useDefaultOffset = actionTerm.use_default_offset !== undefined
        ? actionTerm.use_default_offset
        : controlType === 'joint_position';

      results.push(buildEntry(
        termKey,
        controlType,
        mapping,
        actionTerm.scale as number[] | number | Record<string, number> | undefined,
        actionTerm.offset as number[] | number | Record<string, number> | undefined,
        actionTerm.stiffness as number[] | number | Record<string, number> | undefined,
        actionTerm.damping as number[] | number | Record<string, number> | undefined,
        useDefaultOffset
      ));
    }

    if (results.length === 0) {
      console.warn('[PolicyRunner] No valid action terms found in config.actions.');
      return null;
    }
    return results;
  }

  private normalizeControlArray(
    values: number[] | number | Record<string, number> | undefined,
    length: number,
    fallback: number,
    jointNames?: string[]
  ): Float32Array {
    const output = new Float32Array(length);
    output.fill(fallback);
    if (typeof values === 'number') {
      output.fill(values);
      return output;
    }
    if (Array.isArray(values)) {
      for (let i = 0; i < length; i++) {
        output[i] = typeof values[i] === 'number' ? values[i] : fallback;
      }
      return output;
    }
    if (values !== null && typeof values === 'object' && jointNames) {
      for (const [name, val] of Object.entries(values)) {
        const idx = jointNames.indexOf(name);
        if (idx >= 0 && idx < length) {
          output[idx] = val;
        } else {
          console.warn(`[PolicyRunner] Joint name "${name}" not found in policy_joint_names; skipping.`);
        }
      }
      return output;
    }
    return output;
  }

  private resetSimulationState(): void {
    // Zero the live force sliders (F^cmd + F^ext) on EVERY reset. This is the
    // universal choke point: the Reset button and motion-select reach it via
    // resetSimulation(), and fall/termination calls it DIRECTLY (so adding this
    // to resetSimulation() alone would miss terminations). Restores each slider
    // to its seeded default (0 N) and re-syncs the DOM label + policy-extras
    // store through the slider's own input listener. Pure DOM — runs ABOVE the
    // mjModel guard so it never depends on physics readiness; no-op when no
    // slider panel is mounted (non-force presets).
    resetSlidersToDefault();
    // Reset the hybrid servo/wall state (I, Δx, cached F_meas, a_wrist FD state,
    // and the wall seed) so a fall / motion-switch re-seats the wall and the
    // integrator does not carry across episodes. No-op for non-hybrid policies.
    this.resetHybridState();
    if (!this.mjModel || !this.mjData) {
      return;
    }
    if (this.mjModel.nkey > 0) {
      this.mujoco.mj_resetDataKeyframe(this.mjModel, this.mjData, 0);
    } else {
      this.mujoco.mj_resetData(this.mjModel, this.mjData);
    }
    if (this.eventManager) {
      this.eventManager.onReset({
        mjModel: this.mjModel,
        mjData: this.mjData,
        terrainData: this.terrainData,
      });
    }
    getCommandManager().resetTerms();
    if (this.onnxModule) {
      this.onnxInputDict = this.onnxModule.initInput();
    }
    this.onnxTimeStep = 0;
    this.mujoco.mj_forward(this.mjModel, this.mjData);
    this.lastSimState.bodies.clear();
    this.updateCachedState();
  }

  private executeSimulationSteps(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    // Apply drag forces (clears ALL xfrc_applied, then writes the dragged body).
    this.applyDragForces();

    // ── NON-hybrid path (unchanged): F^ext once per control step, then substeps.
    // The hybrid path re-does F^ext + causal wall + F_meas + servo EVERY substep
    // (training recomputes contact/sensor per substep), so it owns the whole loop.
    if (!this.hybridConfig) {
      // Add F^ext on top — must run AFTER applyDragForces' clear, and ADDs (+=)
      // so a concurrent mouse-drag on the same body still sums correctly.
      this.applyExternalForces();
      for (let substep = 0; substep < this.decimation; substep++) {
        this.applyPolicyControl();
        this.mujoco.mj_step(this.mjModel, this.mjData);
      }
      return;
    }

    // ── HYBRID path: mirror hybrid.py's per-substep _patched() ordering.
    // Per substep: re-clear the wrist xfrc rows (drag cleared once, before the
    // loop), re-inject the manual F^ext slider + the causal spring-wall F_ext
    // from the LIVE TCP, recompute F_meas, run the policy control, then (servo
    // on) the Jᵀ servo AFTER applyPolicyControl (so ctrl.fill(0) can't erase Δq),
    // then step. `dt = this.timestep` (per-substep physics dt), matching
    // hybrid.py's use of physics_dt for the servo integrator.
    this.ensureHybridHands();
    // Live control toggles (extras bus, browser-only — NEVER a policy obs). The
    // Hybrid control panel (WbtTrackingCommand) writes these; before it does, the
    // fallback preserves the baked behavior: wall ON, servo = the checkpoint's
    // servo_enabled (ON for run 31wei7by, OFF for the Layer-A run 6y8314vq). This
    // lets the operator (a) isolate the causal spring wall (Layer A) from the manual
    // F^ext disturbance, and (b) A/B the Jᵀ servo (Layer B) live on either run — the
    // "control authority" needed to verify F_meas → −F_cmd cancellation.
    const wallOn = getPolicyExtra('wbt_hybrid_wall_on', 1.0) >= 0.5;
    // Observe-only (C1 unified) presets IGNORE the live servo toggle: toggling it
    // off would route to zeroHybridServoState and zero the I/Δx the policy's obs
    // NEED live (training always winds the integrator — servo_enabled is True in
    // both unified twins). There is no Δq injection to toggle anyway.
    const servoOn = this.hybridConfig.servoObserveOnly
      ? this.hybridConfig.servoEnabled
      : getPolicyExtra('wbt_hybrid_servo_on', this.hybridConfig.servoEnabled ? 1.0 : 0.0) >= 0.5;
    // Publish the observe-only mode as an extras-bus flag so UI consumers (the metrics
    // panel's φ contact row) can distinguish "the runtime COMPUTES contact_phase" (true
    // for every hybrid preset) from "the ACTOR observes it" (unified presets only).
    // Never a policy obs — no obs class reads this key.
    setPolicyExtra('wbt_hybrid_observe_only', this.hybridConfig.servoObserveOnly ? 1.0 : 0.0);
    for (let substep = 0; substep < this.decimation; substep++) {
      this.clearHybridXfrc();
      this.applyExternalForces(); // manual F^ext slider disturbance (additive)
      this.applySpringWallForces(wallOn); // causal F_ext = clamp(K_env·depth,35)·n̂ (skipped if wall off)
      this.computeHybridFMeas(); // F_meas = F_ext + m·a_wrist(+bias) → extras bus
      this.applyPolicyControl();
      if (servoOn) {
        this.hybridServoStep(); // Jᵀ servo → I, Δx to bus; ctrl_arm += Δq
      } else {
        this.zeroHybridServoState(); // servo off → publish I=Δx=0 (else stale values leak into obs/panel)
      }
      this.mujoco.mj_step(this.mjModel, this.mjData);
    }
  }

  /**
   * Inject the UniFP external disturbance force (F^ext) into xfrc_applied,
   * replicating the headless benchmark's mj_applyFT-at-the-rubber-hand-point
   * (utils/headless_sim_factory.py:_apply_force_ext / utils/eval_utils.py).
   *
   * Frame discipline — this is the PHYSICS path, so EVERYTHING here is native
   * MuJoCo z-up; do NOT apply the THREE.js y-up swizzle (that swizzle belongs
   * only to the arrow viz in WbtTrackingCommand). The slider tuple is raw
   * Newtons in the robot ROOT (pelvis) frame, identical to what the F^cmd obs
   * stores, so we rotate it to world by the LIVE pelvis quat.
   *
   * Equivalence to mj_applyFT: xfrc_applied is a world-frame wrench applied
   * about the body's CENTER OF MASS (xipos), not its origin (xpos). So a pure
   * force F_world applied at the world contact point hand_pt becomes the COM
   * wrench { force = F_world, torque = (hand_pt - xipos) x F_world }. The pivot
   * is xipos — NOT xpos. (mjswan's mouse-drag code uses xpos as its pivot,
   * which is an acceptable approximation for hand-tuned dragging but is the
   * wrong pivot for matching the headless ground truth; we deliberately do not
   * copy it.) Omitting the torque term would silently drop the ~4.15 cm
   * moment arm the benchmark and the training-time injector both include.
   */
  private applyExternalForces(): void {
    if (!this.forceExtBodies || !this.mjModel || !this.mjData) {
      return;
    }
    const mjData = this.mjData;
    const mjModel = this.mjModel;
    // Live pelvis (root) world rotation from the free-joint quat qpos[3:7]
    // (wxyz). Same R_root matrix as headless_sim_factory _apply_force_ext.
    const q = mjData.qpos;
    const w = q[3] ?? 1.0;
    const x = q[4] ?? 0.0;
    const y = q[5] ?? 0.0;
    const z = q[6] ?? 0.0;
    // Row-major 3x3.
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
    const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
    const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);

    for (const fb of this.forceExtBodies) {
      // Lazy body-id resolve (mjModel may not have been ready at config load).
      if (fb.bodyId === -2) {
        fb.bodyId = -1;
        for (let b = 0; b < mjModel.nbody; b++) {
          const name = mjModel.body(b).name;
          if (name === fb.bodyName || name.endsWith(`/${fb.bodyName}`)) {
            fb.bodyId = b;
            break;
          }
        }
        if (fb.bodyId < 0) {
          console.warn(`[PolicyRunner] F^ext body '${fb.bodyName}' not found — skipping`);
        }
      }
      const bid = fb.bodyId;
      if (bid <= 0) continue;

      // Pelvis-frame Newtons from the live sliders (raw N, NOT obs-scaled).
      const fpx = getPolicyExtra(fb.sliderKeys[0], 0.0);
      const fpy = getPolicyExtra(fb.sliderKeys[1], 0.0);
      const fpz = getPolicyExtra(fb.sliderKeys[2], 0.0);
      if (fpx === 0 && fpy === 0 && fpz === 0) continue;

      // f_world = R_root @ f_pelvis.
      const fwx = r00 * fpx + r01 * fpy + r02 * fpz;
      const fwy = r10 * fpx + r11 * fpy + r12 * fpz;
      const fwz = r20 * fpx + r21 * fpy + r22 * fpz;

      // Contact point: hand_pt = xpos[bid] + R_wrist(xmat[bid]) @ offset_b.
      const p3 = bid * 3;
      const m9 = bid * 9;
      const xm = mjData.xmat;
      const o = fb.offset;
      const hx = mjData.xpos[p3 + 0] + xm[m9 + 0] * o[0] + xm[m9 + 1] * o[1] + xm[m9 + 2] * o[2];
      const hy = mjData.xpos[p3 + 1] + xm[m9 + 3] * o[0] + xm[m9 + 4] * o[1] + xm[m9 + 5] * o[2];
      const hz = mjData.xpos[p3 + 2] + xm[m9 + 6] * o[0] + xm[m9 + 7] * o[1] + xm[m9 + 8] * o[2];

      // Moment arm about the body COM (xipos), NOT the origin (xpos).
      const rx = hx - (mjData.xipos[p3 + 0] ?? mjData.xpos[p3 + 0]);
      const ry = hy - (mjData.xipos[p3 + 1] ?? mjData.xpos[p3 + 1]);
      const rz = hz - (mjData.xipos[p3 + 2] ?? mjData.xpos[p3 + 2]);
      // torque = r x F_world.
      const tx = ry * fwz - rz * fwy;
      const ty = rz * fwx - rx * fwz;
      const tz = rx * fwy - ry * fwx;

      // ADD (not assign) so F^ext coexists with a concurrent mouse drag.
      const off = bid * 6;
      mjData.xfrc_applied[off + 0] += fwx;
      mjData.xfrc_applied[off + 1] += fwy;
      mjData.xfrc_applied[off + 2] += fwz;
      mjData.xfrc_applied[off + 3] += tx;
      mjData.xfrc_applied[off + 4] += ty;
      mjData.xfrc_applied[off + 5] += tz;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // Hybrid position-force controller — browser-side causal spring wall + F_meas
  // + Jᵀ force servo (exp:g1-wbt-2pt-hybrid). All native MuJoCo z-up. Gated on
  // this.hybridConfig; every method is a no-op when it is null. Faithful to
  // src/extensions/wbt_wrappers/wbt_training/config_values/hybrid.py.
  // ══════════════════════════════════════════════════════════════════════════

  // Parse policyConfig.extras.hybrid into a typed HybridConfig (or null). Tolerant
  // of missing fields (fills training defaults) so a partial extras block still runs.
  private parseHybridConfig(raw: unknown): HybridConfig | null {
    if (!raw || typeof raw !== 'object') return null;
    const o = raw as Record<string, unknown>;
    const num = (k: string, d: number): number => (typeof o[k] === 'number' ? (o[k] as number) : d);
    const handsRaw = o['hands'];
    if (!Array.isArray(handsRaw) || handsRaw.length === 0) return null;
    const hands: HybridHandCfg[] = [];
    for (const h of handsRaw) {
      if (!h || typeof h !== 'object') continue;
      const he = h as Record<string, unknown>;
      const bodyName = he['body_name'];
      const cmdKeys = he['cmd_slider_keys'];
      const armJoints = he['arm_joints'];
      if (typeof bodyName !== 'string' || !Array.isArray(cmdKeys) || !Array.isArray(armJoints)) continue;
      hands.push({
        bodyName,
        cmdSliderKeys: [String(cmdKeys[0] ?? ''), String(cmdKeys[1] ?? ''), String(cmdKeys[2] ?? '')],
        armJointNames: armJoints.map((n) => String(n)),
      });
    }
    if (hands.length === 0) return null;
    return {
      servoEnabled: o['servo_enabled'] === true,
      servoObserveOnly: o['servo_observe_only'] === true,
      contactPhaseThetaOn: num('contact_phase_theta_on', 2.0),
      contactPhaseThetaOff: num('contact_phase_theta_off', 1.5),
      contactPhaseTauS: num('contact_phase_tau_s', 0.05),
      contactPhaseWarmupSteps: num('contact_phase_warmup_steps', 5),
      stiffnessK: num('stiffness_K', 200.0),
      fCmdMax: num('f_cmd_max', 25.0),
      kEnv: num('k_env', Math.sqrt(100.0 * 800.0)),
      planeDist: num('plane_dist', 0.05),
      maxPenetrationM: num('max_penetration_m', 0.08),
      maxForceN: num('max_force_n', 35.0),
      distalMassKg: num('distal_mass_kg', 1.5),
      sensorNoiseStd: num('sensor_noise_std', 0.0),
      servoGainK: num('servo_gain_k', 0.05),
      integratorGainKi: num('integrator_gain_ki', 0.5),
      integratorMax: num('integrator_max', 40.0),
      servoDqMax: num('servo_dq_max', 0.15),
      tcpOffsetX: num('tcp_offset_x', 0.242),
      hands,
    };
  }

  // Lazily resolve per-hand runtime handles (wrist body id, arm joint qpos/qvel
  // addresses, and the ctrl-index / kp / actuator-type of each arm joint from the
  // policyControl term). Idempotent — once every hand is resolved it early-returns.
  private ensureHybridHands(): void {
    if (!this.hybridConfig || !this.mjModel || !this.mjData) return;
    if (this.hybridHands && this.hybridHands.length === this.hybridConfig.hands.length) return;
    const mjModel = this.mjModel;
    const findBody = (name: string): number => {
      for (let b = 0; b < mjModel.nbody; b++) {
        const bn = mjModel.body(b).name;
        if (bn === name || bn.endsWith(`/${name}`)) return b;
      }
      return -1;
    };
    // Map a joint name → {qposAdr, qvelAdr} via mjModel.jnt_qposadr / jnt_dofadr.
    const jointAdr = (name: string): { qpos: number; qvel: number } | null => {
      const jid = this.mujoco.mj_name2id(mjModel, this.mujoco.mjtObj.mjOBJ_JOINT.value, name);
      if (jid < 0) return null;
      return { qpos: mjModel.jnt_qposadr[jid], qvel: mjModel.jnt_dofadr[jid] };
    };
    // Find the ctrl-index / kp / actuator-type for an arm joint from policyControl.
    // The joint's qpos address uniquely identifies it across the joint_position term.
    const armCtrlFor = (qposAdr: number): { ctrlIndex: number; positionActuator: boolean; kp: number } => {
      const fallback = { ctrlIndex: -1, positionActuator: true, kp: 0.0 };
      if (!this.policyControl) return fallback;
      for (const term of this.policyControl) {
        if (term.controlType !== 'joint_position') continue;
        for (let i = 0; i < term.ctrlAdr.length; i++) {
          if (term.qposAdr[i] === qposAdr) {
            return { ctrlIndex: term.ctrlAdr[i], positionActuator: term.positionActuator[i], kp: term.kp[i] };
          }
        }
      }
      return fallback;
    };
    const hands: HybridHandRt[] = [];
    for (const hc of this.hybridConfig.hands) {
      const armQposAdr: number[] = [];
      const armQvelAdr: number[] = [];
      const armCtrl: HybridHandRt['armCtrl'] = [];
      for (const jn of hc.armJointNames) {
        const adr = jointAdr(jn);
        if (!adr) {
          console.warn(`[PolicyRunner] hybrid arm joint '${jn}' not found — servo will skip it`);
          continue;
        }
        armQposAdr.push(adr.qpos);
        armQvelAdr.push(adr.qvel);
        armCtrl.push(armCtrlFor(adr.qpos));
      }
      hands.push({
        bodyId: findBody(hc.bodyName),
        armQposAdr,
        armQvelAdr,
        armCtrl,
        wallSeeded: false,
        normalW: [0, 0, 0],
        pointW: [0, 0, 0],
        fcmdDirW: [0, 0, 0],
        seedTcpW: [0, 0, 0],
        integ: [0, 0, 0],
        deltaX: [0, 0, 0],
        fExtW: [0, 0, 0],
        fMeasW: [0, 0, 0],
        depth: 0,
        gate: 0,
        prevTcpW: null,
        prevVelW: null,
        cphase: 0,
        chyst: 0,
        cwarm: 0,
      });
    }
    this.hybridHands = hands;
  }

  // World-frame TCP point of hand `hi`: xpos[wrist] + R_wrist @ (tcp_offset_x,0,0).
  // Native z-up. Returns null if the wrist body is unresolved.
  private hybridTcpWorld(hi: number): [number, number, number] | null {
    if (!this.hybridConfig || !this.hybridHands || !this.mjData) return null;
    const h = this.hybridHands[hi];
    if (!h || h.bodyId <= 0) return null;
    const off = this.hybridConfig.tcpOffsetX;
    const p3 = h.bodyId * 3;
    const m9 = h.bodyId * 9;
    const xm = this.mjData.xmat;
    const xp = this.mjData.xpos;
    // r local = (off, 0, 0) → world column 0 of R_wrist * off.
    return [
      (xp[p3 + 0] ?? 0) + xm[m9 + 0] * off,
      (xp[p3 + 1] ?? 0) + xm[m9 + 3] * off,
      (xp[p3 + 2] ?? 0) + xm[m9 + 6] * off,
    ];
  }

  // Pelvis (root) world rotation matrix (row-major 3x3) from qpos[3:7] wxyz.
  private hybridRootMat(): number[] {
    const q = this.mjData!.qpos;
    const w = q[3] ?? 1.0, x = q[4] ?? 0.0, y = q[5] ?? 0.0, z = q[6] ?? 0.0;
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ];
  }

  // Zero the xfrc_applied rows of the hybrid wrist bodies at the top of each
  // substep. applyDragForces clears the WHOLE buffer once before the substep
  // loop; but the causal F_ext (and manual F^ext) are re-injected with += every
  // substep from the LIVE penetration, so we must re-zero these rows each substep
  // or they would accumulate over the decimation substeps. Only the hybrid wrist
  // rows are touched (a concurrent mouse-drag on another body is preserved).
  private clearHybridXfrc(): void {
    if (!this.hybridHands || !this.mjData) return;
    const xfrc = this.mjData.xfrc_applied;
    for (const h of this.hybridHands) {
      if (h.bodyId <= 0) continue;
      const off = h.bodyId * 6;
      for (let k = 0; k < 6; k++) xfrc[off + k] = 0.0;
    }
  }

  // Causal virtual spring wall: recompute penetration from the LIVE TCP and inject
  // F_ext = clamp(K_env·depth, max_force_n)·n̂ into xfrc_applied at the TCP with the
  // r×F moment arm about the body COM (xipos), exactly like applyExternalForces.
  // Seeds the per-hand wall plane the first time F^cmd is non-zero (normal =
  // −unit(F_cmd_world), plane_point = TCP − plane_dist·n̂), matching hybrid.py.
  private applySpringWallForces(wallOn = true): void {
    if (!this.hybridConfig || !this.hybridHands || !this.mjModel || !this.mjData) return;
    const cfg = this.hybridConfig;
    // Live-normal opt-in (default OFF = training-faithful episode-fixed direction).
    // Read per call, not per episode, so the checkbox takes effect immediately.
    const wallLive = getPolicyExtra('wbt_hybrid_wall_live', 0.0) >= 0.5;
    // Wall toggled OFF (live): no causal contact force. Zero the cached contact
    // state so F_meas (computeHybridFMeas reads h.fExtW) and the depth/gate the
    // servo + badge read all reflect free space. F^cmd stays a pure intent/servo
    // target; the ONLY external force is then the manual F^ext slider — which is
    // exactly the isolation the operator needs to reason about cause and effect.
    if (!wallOn) {
      for (let hi = 0; hi < this.hybridHands.length; hi++) {
        const h = this.hybridHands[hi];
        const hc = cfg.hands[hi];
        h.depth = 0;
        h.gate = 0;
        h.fExtW = [0, 0, 0];
        // Keep wallSeeded so the servo's episode-fixed F^cmd direction (fcmdDirW,
        // captured at seed) persists across a toggle; only the reaction FORCE is
        // suppressed. Re-seed if F^cmd→0 (mirrors the wall-on deadband branch).
        const fmag = Math.hypot(
          getPolicyExtra(hc.cmdSliderKeys[0], 0.0),
          getPolicyExtra(hc.cmdSliderKeys[1], 0.0),
          getPolicyExtra(hc.cmdSliderKeys[2], 0.0),
        );
        if (fmag < 1e-6) {
          h.wallSeeded = false;
          h.seedTcpW = [0, 0, 0];
        }
      }
      return;
    }
    const R = this.hybridRootMat(); // pelvis→world (for F^cmd direction)
    const xfrc = this.mjData.xfrc_applied;
    for (let hi = 0; hi < this.hybridHands.length; hi++) {
      const h = this.hybridHands[hi];
      const hc = cfg.hands[hi];
      if (!h || h.bodyId <= 0) continue;
      const tcp = this.hybridTcpWorld(hi);
      if (!tcp) continue;

      // F^cmd in pelvis frame (raw N) from the sliders → world.
      const fcx = getPolicyExtra(hc.cmdSliderKeys[0], 0.0);
      const fcy = getPolicyExtra(hc.cmdSliderKeys[1], 0.0);
      const fcz = getPolicyExtra(hc.cmdSliderKeys[2], 0.0);
      const fcmdW: [number, number, number] = [
        R[0] * fcx + R[1] * fcy + R[2] * fcz,
        R[3] * fcx + R[4] * fcy + R[5] * fcz,
        R[6] * fcx + R[7] * fcy + R[8] * fcz,
      ];
      const fcmdMag = Math.hypot(fcmdW[0], fcmdW[1], fcmdW[2]);

      // Wall exists only while F^cmd is commanded (matches training: the plane
      // normal = −unit(F_cmd)). Below a small deadband, no wall / no F_ext, and
      // the wall re-seeds next time F^cmd is pressed.
      if (fcmdMag < 1e-6) {
        h.wallSeeded = false;
        h.seedTcpW = [0, 0, 0];
        h.depth = 0;
        h.gate = 0;
        h.fExtW = [0, 0, 0];
        continue;
      }
      if (!h.wallSeeded) {
        // Outward normal points back at the hand along −F_cmd; the plane sits
        // plane_dist ahead of the current TCP along +F_cmd (so a small overshoot
        // penetrates it). n̂ = −unit(F_cmd_world). Capture the EPISODE-FIXED unit
        // F_cmd world direction (= −n̂) at seed time, mirroring hybrid.py's
        // _wbt_fcmd_dir_w (= −plane_normal): training holds F_cmd's WORLD direction
        // fixed for the episode, so the servo/reward do NOT drift if the pelvis
        // reorients under a held force. The servo uses this fixed direction (scaled
        // by the live |F_cmd| slider magnitude) instead of re-rotating by the live
        // pelvis each substep.
        h.normalW = [-fcmdW[0] / fcmdMag, -fcmdW[1] / fcmdMag, -fcmdW[2] / fcmdMag];
        h.fcmdDirW = [fcmdW[0] / fcmdMag, fcmdW[1] / fcmdMag, fcmdW[2] / fcmdMag];
        h.seedTcpW = [tcp[0], tcp[1], tcp[2]];
        h.pointW = [
          tcp[0] - cfg.planeDist * h.normalW[0],
          tcp[1] - cfg.planeDist * h.normalW[1],
          tcp[2] - cfg.planeDist * h.normalW[2],
        ];
        h.wallSeeded = true;
      } else if (wallLive) {
        // LIVE-NORMAL mode (operator opt-in, `wbt_hybrid_wall_live`): re-derive the
        // plane from the CURRENT F^cmd every substep instead of holding the seeded
        // direction. This exists purely so the operator can sweep F^cmd across axes in
        // one episode (e.g. Fx=10 then Fy=10 → a 45° wall) and watch the wall follow;
        // in the DEFAULT mode the seed block above runs once and a later slider change
        // leaves the plane on its original axis, which reads as "the wall is stuck".
        //
        // NOT training-faithful, and deliberately so: hybrid.py holds the direction
        // fixed for the whole episode, and F_cmd's direction is sampled uniformly on
        // the unit sphere ACROSS episodes (sample_plane), so a live-turning wall is a
        // regime the policy never saw. Keep it OFF for A/B or accept-criteria runs.
        //
        // The plane rotates about the SEED TCP (h.seedTcpW), not the live one: anchoring
        // to the live TCP would hold the plane plane_dist ahead of the hand forever, so
        // depth ≡ 0 and no force could ever build. Rotating about the fixed pivot keeps
        // any depth already accumulated along the new axis, which is what makes the
        // reddening + F^ext arrow respond as you turn the command.
        h.normalW = [-fcmdW[0] / fcmdMag, -fcmdW[1] / fcmdMag, -fcmdW[2] / fcmdMag];
        h.fcmdDirW = [fcmdW[0] / fcmdMag, fcmdW[1] / fcmdMag, fcmdW[2] / fcmdMag];
        h.pointW = [
          h.seedTcpW[0] - cfg.planeDist * h.normalW[0],
          h.seedTcpW[1] - cfg.planeDist * h.normalW[1],
          h.seedTcpW[2] - cfg.planeDist * h.normalW[2],
        ];
      }

      // depth = clamp(max(0, n̂·(plane_point − TCP)), 0, max_pen). Penetrating
      // means the TCP has crossed the plane in the −normal (into-wall) direction.
      const dx = h.pointW[0] - tcp[0];
      const dy = h.pointW[1] - tcp[1];
      const dz = h.pointW[2] - tcp[2];
      const signed = dx * h.normalW[0] + dy * h.normalW[1] + dz * h.normalW[2];
      const depth = Math.min(Math.max(signed, 0), cfg.maxPenetrationM);
      h.depth = depth;
      h.gate = depth > 0 ? 1 : 0;

      // F_ext = clamp(K_env·depth, max_force_n) · n̂  (compression-only, +normal).
      const mag = Math.min(cfg.kEnv * depth, cfg.maxForceN);
      const fExtW: [number, number, number] = [mag * h.normalW[0], mag * h.normalW[1], mag * h.normalW[2]];
      h.fExtW = fExtW;
      if (mag <= 0) continue;

      // Inject at the TCP with r×F about COM (xipos), mirroring applyExternalForces.
      const p3 = h.bodyId * 3;
      const rx = tcp[0] - (this.mjData.xipos[p3 + 0] ?? this.mjData.xpos[p3 + 0]);
      const ry = tcp[1] - (this.mjData.xipos[p3 + 1] ?? this.mjData.xpos[p3 + 1]);
      const rz = tcp[2] - (this.mjData.xipos[p3 + 2] ?? this.mjData.xpos[p3 + 2]);
      const off6 = h.bodyId * 6;
      xfrc[off6 + 0] += fExtW[0];
      xfrc[off6 + 1] += fExtW[1];
      xfrc[off6 + 2] += fExtW[2];
      xfrc[off6 + 3] += ry * fExtW[2] - rz * fExtW[1];
      xfrc[off6 + 4] += rz * fExtW[0] - rx * fExtW[2];
      xfrc[off6 + 5] += rx * fExtW[1] - ry * fExtW[0];
    }
  }

  // Simulated F/T sensor: F_meas_w = F_ext_w + distal_mass·a_wrist (+ bias/noise).
  // a_wrist is finite-differenced from the TCP world position across substeps
  // (mjData exposes no body_lin_acc). Writes the world-frame value to the extras
  // bus under wbt_force_meas__{l,r}_f{x,y,z} for the obs class (which rotates
  // world→pelvis) + the cyan arrow. Matches hybrid.py simulated_measured_force
  // (include_inertial=True, no gravity term — HW does static gravity comp).
  // No noise is injected anywhere here: sensorNoiseStd is parsed but never applied
  // (the demo config sets it to 0). What flickers on the readout is the INERTIAL
  // term — a_wrist is DOUBLE finite-differenced at the substep dt, so sub-millimetre
  // TCP jitter is amplified by ~1/dt² (0.1 mm at 5 ms ≈ 4 m/s² ≈ 6 N on 1.5 kg).
  // `wbt_hybrid_fmeas_inertial` (operator checkbox, fallback 1.0 = ON = training-
  // faithful) gates that term: OFF ⇒ F_meas = F_ext only (wall reaction + the F^ext
  // slider), a still arrow/readout. It is the ONE F_meas, so the policy's
  // force_meas_local obs, contact_phase and e = F_cmd + F_meas all follow the switch.
  // The FD state keeps updating while off, so switching back on cannot spike.
  private computeHybridFMeas(): void {
    if (!this.hybridConfig || !this.hybridHands || !this.mjData) return;
    const cfg = this.hybridConfig;
    const dt = this.timestep > 0 ? this.timestep : 0.005;
    const inertialOn = getPolicyExtra('wbt_hybrid_fmeas_inertial', 1.0) >= 0.5;
    const mEff = inertialOn ? cfg.distalMassKg : 0.0;
    const sides = ['l', 'r'];
    const R = this.hybridRootMat(); // pelvis→world, for F^cmd when the wall isn't seeded
    for (let hi = 0; hi < this.hybridHands.length; hi++) {
      const h = this.hybridHands[hi];
      const hc = cfg.hands[hi];
      const tcp = this.hybridTcpWorld(hi);
      let aW: [number, number, number] = [0, 0, 0];
      // SELF-ARMING contact-phase warmup: a null prevTcpW is the canonical "I have no
      // velocity history" signal, set by BOTH resetHybridState and the hand-state
      // initializer. Re-arming the warmup window here — rather than only in
      // resetHybridState — removes a cross-object ordering dependency: the pose teleport
      // lives in TrackingCommand.applyReferenceStateToSim (a different object on a
      // different call path), and on the initial-load path loadPolicyConfig calls the
      // command term's setSelectedMotion directly, without any resetSimulation(). Keying
      // off the state that the FD itself depends on makes the gate impossible to bypass.
      if (h.prevTcpW === null) h.cwarm = 0;
      if (tcp && h.prevTcpW) {
        const velW: [number, number, number] = [
          (tcp[0] - h.prevTcpW[0]) / dt,
          (tcp[1] - h.prevTcpW[1]) / dt,
          (tcp[2] - h.prevTcpW[2]) / dt,
        ];
        if (h.prevVelW) {
          aW = [(velW[0] - h.prevVelW[0]) / dt, (velW[1] - h.prevVelW[1]) / dt, (velW[2] - h.prevVelW[2]) / dt];
        }
        h.prevVelW = velW;
      }
      if (tcp) h.prevTcpW = tcp;
      const fMeasW: [number, number, number] = [
        h.fExtW[0] + mEff * aW[0],
        h.fExtW[1] + mEff * aW[1],
        h.fExtW[2] + mEff * aW[2],
      ];
      h.fMeasW = fMeasW;
      const s = sides[hi] ?? String(hi);
      setPolicyExtra(`wbt_force_meas__${s}_fx`, fMeasW[0]);
      setPolicyExtra(`wbt_force_meas__${s}_fy`, fMeasW[1]);
      setPolicyExtra(`wbt_force_meas__${s}_fz`, fMeasW[2]);

      // Soft contact phase (C1 unified): hysteresis on ‖F_meas‖ + first-order LPF,
      // per substep dt — the SAME math as hybrid.py _compute_contact_and_sensor.
      // Published unconditionally (zeros-by-fallback for presets whose obs don't
      // read wbt_cphase__*, so non-unified presets are unaffected).
      const fmag = Math.hypot(fMeasW[0], fMeasW[1], fMeasW[2]);
      if (fmag > cfg.contactPhaseThetaOn) h.chyst = 1.0;
      else if (fmag < cfg.contactPhaseThetaOff) h.chyst = 0.0;
      // Post-reset warmup, mirroring training's contact_phase_warmup_steps gate: for the
      // first N env steps after a reset/motion-switch the hysteresis is FORCED off, so the
      // teleport's spurious m·a spike in F_meas can't slam the label to 1 in free space.
      // Training counts env steps; we count substeps, hence × decimation.
      h.cwarm += 1;
      if (h.cwarm < cfg.contactPhaseWarmupSteps * this.decimation) h.chyst = 0.0;
      const beta = 1.0 - Math.exp(-dt / Math.max(cfg.contactPhaseTauS, 1e-6));
      h.cphase += beta * (h.chyst - h.cphase);
      setPolicyExtra(`wbt_cphase__${s}`, h.cphase);

      // Servo error e = F_cmd_w + F_meas_w (world). At servo equilibrium the wall
      // reaction cancels the command, so F_meas → −F_cmd and ‖e‖ → 0 — the direct
      // numeric proof of the cancellation. Published every substep (servo on OR
      // off) so the panel shows it even for the Layer-A run; the panel takes ‖e‖.
      // F_cmd world vector: episode-fixed seeded direction × live |F_cmd| when the
      // wall is seeded (matches the servo), else live slider rotated by the root.
      const fcx = getPolicyExtra(hc.cmdSliderKeys[0], 0.0);
      const fcy = getPolicyExtra(hc.cmdSliderKeys[1], 0.0);
      const fcz = getPolicyExtra(hc.cmdSliderKeys[2], 0.0);
      let fcmdW: [number, number, number];
      if (h.wallSeeded) {
        const mag = Math.hypot(fcx, fcy, fcz);
        fcmdW = [h.fcmdDirW[0] * mag, h.fcmdDirW[1] * mag, h.fcmdDirW[2] * mag];
      } else {
        fcmdW = [
          R[0] * fcx + R[1] * fcy + R[2] * fcz,
          R[3] * fcx + R[4] * fcy + R[5] * fcz,
          R[6] * fcx + R[7] * fcy + R[8] * fcz,
        ];
      }
      setPolicyExtra(`wbt_ferr__${s}_fx`, fcmdW[0] + fMeasW[0]);
      setPolicyExtra(`wbt_ferr__${s}_fy`, fcmdW[1] + fMeasW[1]);
      setPolicyExtra(`wbt_ferr__${s}_fz`, fcmdW[2] + fMeasW[2]);
    }
  }

  // Finite-difference world-frame TCP linear Jacobian for hand `hi`: column j =
  // ∂TCP_world / ∂q_arm[j]. Perturbs each arm joint's qpos by ε, runs mj_kinematics,
  // reads the shifted TCP, and restores qpos. This yields the TCP-anchored linear
  // Jacobian directly (no −[r]×J_ω shift needed) and uses only mj_kinematics +
  // xpos/xmat (well-exercised in this binding), sidestepping mj_jacBody's untyped
  // emscripten pointer args. Returns a (3 x n_arm) matrix (column-major list of
  // [dx,dy,dz]); restores mjData to the pre-perturbation kinematics on exit.
  private hybridTcpJacobianFD(hi: number): Array<[number, number, number]> {
    const h = this.hybridHands![hi];
    const cols: Array<[number, number, number]> = [];
    if (!this.mjModel || !this.mjData || h.bodyId <= 0) {
      return h.armQposAdr.map(() => [0, 0, 0] as [number, number, number]);
    }
    const eps = 1e-4;
    const qpos = this.mjData.qpos;
    const base = this.hybridTcpWorld(hi)!;
    // try/finally so a perturbed qpos is ALWAYS restored + kinematics rebuilt,
    // even if hybridTcpWorld returns null mid-loop — a corrupted qpos would feed
    // mj_step and diverge the sim.
    let jSaved = -1;
    let savedVal = 0;
    try {
      for (let j = 0; j < h.armQposAdr.length; j++) {
        const adr = h.armQposAdr[j];
        jSaved = adr;
        savedVal = qpos[adr];
        qpos[adr] = savedVal + eps;
        this.mujoco.mj_kinematics(this.mjModel, this.mjData);
        const pert = this.hybridTcpWorld(hi)!;
        qpos[adr] = savedVal;
        jSaved = -1;
        cols.push([(pert[0] - base[0]) / eps, (pert[1] - base[1]) / eps, (pert[2] - base[2]) / eps]);
      }
    } finally {
      if (jSaved >= 0) qpos[jSaved] = savedVal; // restore an in-flight perturbation
      // Restore the unperturbed kinematics (so downstream reads see the real state).
      this.mujoco.mj_kinematics(this.mjModel, this.mjData);
    }
    return cols;
  }

  // Contact-gated Jᵀ force servo (hybrid.py servo_residual_step), run per substep
  // AFTER applyPolicyControl. World frame, force channels only (S=(1,1,1)):
  //   e        = F_cmd_w + F_meas_w
  //   I        = gate·clip(I + Ki·e·dt, ±I_max)          [N] (winds only in contact)
  //   task_w   = F_cmd_w + I                             [N]
  //   Δq_arm   = clip(k·Jᵀ·task_w, ±dq_max)              [rad]
  //   Δx       = task_w / K                              [m] (obs/reward setpoint shift)
  // Δq is injected into mjData.ctrl AFTER applyPolicyControl (which does ctrl.fill(0)
  // then rewrites every actuator from the policy action): position actuator →
  // ctrl += Δq (radians, direct); motor actuator → ctrl += kp·Δq (torque). Both
  // yield the intended extra PD torque Kp·Δq, matching training's
  // _actions_after_delay += servo_dq/action_scales. getLastActions() is NOT touched,
  // so the `actions` obs term stays the raw policy action (as in training).
  private hybridServoStep(): void {
    if (!this.hybridConfig || !this.hybridHands || !this.mjModel || !this.mjData) return;
    const cfg = this.hybridConfig;
    const dt = this.timestep > 0 ? this.timestep : 0.005;
    const R = this.hybridRootMat();
    const ctrl = this.mjData.ctrl;
    for (let hi = 0; hi < this.hybridHands.length; hi++) {
      const h = this.hybridHands[hi];
      const hc = cfg.hands[hi];
      if (!h || h.bodyId <= 0) continue;

      // F^cmd world vector. Training holds the F_cmd WORLD direction FIXED for the
      // episode (_wbt_fcmd_dir_w = −plane_normal), so we use the seeded episode-fixed
      // unit direction (h.fcmdDirW) scaled by the live |F_cmd| slider magnitude —
      // NOT the live-pelvis-rotated slider — so the servo does not drift if the
      // pelvis reorients under a held force. Falls back to the live-rotated value
      // before the wall is seeded (F=0 → the servo is inert anyway, gate=0).
      const fcx = getPolicyExtra(hc.cmdSliderKeys[0], 0.0);
      const fcy = getPolicyExtra(hc.cmdSliderKeys[1], 0.0);
      const fcz = getPolicyExtra(hc.cmdSliderKeys[2], 0.0);
      const fcmdMag = Math.hypot(fcx, fcy, fcz); // slider magnitude is frame-invariant
      let fcmdW: [number, number, number];
      if (h.wallSeeded) {
        fcmdW = [h.fcmdDirW[0] * fcmdMag, h.fcmdDirW[1] * fcmdMag, h.fcmdDirW[2] * fcmdMag];
      } else {
        fcmdW = [
          R[0] * fcx + R[1] * fcy + R[2] * fcz,
          R[3] * fcx + R[4] * fcy + R[5] * fcz,
          R[6] * fcx + R[7] * fcy + R[8] * fcz,
        ];
      }
      const e: [number, number, number] = [
        fcmdW[0] + h.fMeasW[0],
        fcmdW[1] + h.fMeasW[1],
        fcmdW[2] + h.fMeasW[2],
      ];
      // Gated, anti-windup FORCE-unit integral (frozen in free space).
      const clip = (v: number, lim: number): number => Math.min(Math.max(v, -lim), lim);
      for (let k = 0; k < 3; k++) {
        h.integ[k] = h.gate * clip(h.integ[k] + cfg.integratorGainKi * e[k] * dt, cfg.integratorMax);
      }
      // Desired task wrench = feedforward F_cmd + integral (both N), +F_cmd sign.
      const taskW: [number, number, number] = [
        fcmdW[0] + h.integ[0],
        fcmdW[1] + h.integ[1],
        fcmdW[2] + h.integ[2],
      ];
      // Δx = task_w / K (obs/reward setpoint shift only).
      h.deltaX = [taskW[0] / cfg.stiffnessK, taskW[1] / cfg.stiffnessK, taskW[2] / cfg.stiffnessK];

      // Δq_arm = clip(k · Jᵀ · task_w, ±dq_max). Jᵀ·task_w = Σ_axis J[axis][j]·task_w[axis].
      // C1 observe-only mode (unified preset): SKIP the ctrl injection entirely — the
      // integrator math above + the I/Δx publishing below stay live (reference
      // generator), but the policy is the sole executor (hybrid.py _observe_step).
      // The FD Jacobian is also skipped (only needed to actuate).
      if (!cfg.servoObserveOnly) {
        const jac = this.hybridTcpJacobianFD(hi); // n_arm columns of [dx,dy,dz]
        for (let j = 0; j < h.armCtrl.length; j++) {
          const col = jac[j] ?? [0, 0, 0];
          const jt = col[0] * taskW[0] + col[1] * taskW[1] + col[2] * taskW[2];
          const dq = clip(cfg.servoGainK * jt, cfg.servoDqMax);
          const ac = h.armCtrl[j];
          if (ac.ctrlIndex < 0) continue;
          if (ac.positionActuator) {
            ctrl[ac.ctrlIndex] += dq; // ctrl is a joint target in radians
          } else {
            ctrl[ac.ctrlIndex] += ac.kp * dq; // ctrl is a torque
          }
        }
      }

      // Publish I / Δx (world frame) to the extras bus for the obs classes
      // (WbtIntegratorState / WbtXTgtShift rotate world→pelvis) + the panel.
      const s = ['l', 'r'][hi] ?? String(hi);
      setPolicyExtra(`wbt_integ__${s}_fx`, h.integ[0]);
      setPolicyExtra(`wbt_integ__${s}_fy`, h.integ[1]);
      setPolicyExtra(`wbt_integ__${s}_fz`, h.integ[2]);
      setPolicyExtra(`wbt_xtgt__${s}_fx`, h.deltaX[0]);
      setPolicyExtra(`wbt_xtgt__${s}_fy`, h.deltaX[1]);
      setPolicyExtra(`wbt_xtgt__${s}_fz`, h.deltaX[2]);
    }
  }

  // Servo OFF path (Layer A, or the live servo toggle set off): the Jᵀ servo does
  // NOT run, so no Δq is injected. But I / Δx are actor OBS terms + panel rows, so
  // we must PUBLISH zeros every substep — otherwise the last servo-on I/Δx values
  // stay on the extras bus and leak into the obs (WbtIntegratorState/WbtXTgtShift)
  // and the metrics panel after a live toggle. Also zeros the internal integ/deltaX
  // so a later servo-ON start begins from a clean integrator (no windup carryover).
  // Matches training's zeroed _wbt_integ_state / _wbt_delta_x when servo_enabled=False.
  private zeroHybridServoState(): void {
    if (!this.hybridConfig || !this.hybridHands) return;
    for (const h of this.hybridHands) {
      h.integ = [0, 0, 0];
      h.deltaX = [0, 0, 0];
    }
    for (const s of ['l', 'r']) {
      for (const c of ['fx', 'fy', 'fz']) {
        setPolicyExtra(`wbt_integ__${s}_${c}`, 0.0);
        setPolicyExtra(`wbt_xtgt__${s}_${c}`, 0.0);
      }
    }
    // NOTE: contact_phase is NOT zeroed here — it tracks ‖F_meas‖ (the wall is
    // still live with the servo off), exactly like training where the label is
    // computed in _compute_contact_and_sensor regardless of the servo mode.
  }

  // Reset all evolving hybrid state on episode reset (fall / motion-switch / Reset
  // button). Zeros the integrator, Δx, cached F_ext/F_meas, the a_wrist FD state,
  // re-seeds the wall, and clears the extras-bus keys (which are NOT sliders, so
  // resetSlidersToDefault does not touch them). No-op for non-hybrid policies.
  private resetHybridState(): void {
    if (!this.hybridConfig) return;
    if (this.hybridHands) {
      for (const h of this.hybridHands) {
        h.integ = [0, 0, 0];
        h.deltaX = [0, 0, 0];
        h.fExtW = [0, 0, 0];
        h.fMeasW = [0, 0, 0];
        h.depth = 0;
        h.gate = 0;
        h.wallSeeded = false;
        h.normalW = [0, 0, 0];
        h.fcmdDirW = [0, 0, 0];
        h.seedTcpW = [0, 0, 0];
        h.prevTcpW = null;
        h.prevVelW = null;
        h.cphase = 0;
        h.chyst = 0;
        // Re-arm the post-reset warmup: the pose jump this reset just performed makes
        // the next few a_wrist finite differences enormous (see cwarm's declaration).
        h.cwarm = 0;
      }
    }
    for (const s of ['l', 'r']) {
      setPolicyExtra(`wbt_cphase__${s}`, 0.0);
      for (const c of ['fx', 'fy', 'fz']) {
        setPolicyExtra(`wbt_force_meas__${s}_${c}`, 0.0);
        setPolicyExtra(`wbt_integ__${s}_${c}`, 0.0);
        setPolicyExtra(`wbt_xtgt__${s}_${c}`, 0.0);
        setPolicyExtra(`wbt_ferr__${s}_${c}`, 0.0);
      }
    }
  }

  private applyPolicyControl(): void {
    if (!this.policyControl || !this.mjData) {
      return;
    }

    const ctrl = this.mjData.ctrl;
    ctrl.fill(0.0);

    // Fetch the full action vector once; each term reads its own slice via actionIndices.
    const allActions = this.policyRunner?.getLastActions() ?? new Float32Array(0);

    for (const term of this.policyControl) {
      const {
        controlType,
        ctrlAdr,
        qposAdr,
        qvelAdr,
        actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
      } =
        term;
      const numJoints = ctrlAdr.length;

      if (controlType === 'joint_position') {
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex < 0) continue;
          const actionValue = allActions[actionIndices[i]] ?? 0;
          const target =
            defaultJointPos[i] +
            actionOffset[i] +
            actionScale[i] * actionValue -
            encoderBias[i];

          if (positionActuator[i]) {
            // Position actuator (biastype=affine): ctrl = target joint position.
            // MuJoCo computes force = kp*(ctrl - qpos) - kd*qvel internally.
            ctrl[ctrlIndex] = target;
          } else {
            // Motor actuator (biastype=none): ctrl = torque.
            // PD must be computed externally using kp/kd from the policy config.
            const qpos = this.mjData.qpos[qposAdr[i]];
            const qvel = this.mjData.qvel[qvelAdr[i]];
            ctrl[ctrlIndex] = kp[i] * (target - qpos) + kd[i] * (0 - qvel);
          }
        }
      } else if (controlType === 'torque') {
        for (let i = 0; i < numJoints; i++) {
          const ctrlIndex = ctrlAdr[i];
          if (ctrlIndex >= 0) {
            ctrl[ctrlIndex] = actionScale[i] * (allActions[actionIndices[i]] ?? 0);
          }
        }
      }
    }
  }

  // Pre-clip the obs vector so the ONNX-baked normalizer produces values in
  // [-clip, +clip]. Only applied to the primary obs input ('policy' / 'obs');
  // sidecar inputs (time_step, adapt_hx) pass through untouched. No-op when
  // normalizer params absent or when the obs length doesn't match mean's.
  private applyOnnxNormalizerPreClip(key: string, raw: Float32Array): Float32Array {
    const mean = this.onnxNormMean;
    const std = this.onnxNormStd;
    if (!mean || !std) return raw;
    if (key === 'time_step' || key === 'is_init' || key === 'adapt_hx') return raw;
    if (raw.length !== mean.length) {
      // Mismatch is a configuration bug — log once per inference and skip.
      console.warn('[PolicyRunner] normalizer dim mismatch', {
        key,
        obsLen: raw.length,
        meanLen: mean.length,
      });
      return raw;
    }
    const clip = this.onnxNormClip;
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const m = mean[i];
      const s = std[i];
      let n = (raw[i] - m) / s;
      if (n > clip) n = clip;
      else if (n < -clip) n = -clip;
      out[i] = n * s + m;
    }
    return out;
  }

  private async runOnnxInference(obs: Record<string, Float32Array>): Promise<void> {
    if (!this.onnxModule || !this.policyRunner || this.onnxInferencing) {
      return;
    }

    this.onnxInferencing = true;
    try {
      if (!this.onnxInputDict) {
        this.onnxInputDict = this.onnxModule.initInput();
      }
      const input: Record<string, ort.Tensor> = { ...this.onnxInputDict };
      if (this.onnxModule.inKeys.includes('time_step')) {
        input.time_step = new ort.Tensor('float32', new Float32Array([this.onnxTimeStep]), [1, 1]);
      }
      for (const [key, value] of Object.entries(obs)) {
        const tensorData = this.applyOnnxNormalizerPreClip(key, value);
        input[key] = new ort.Tensor('float32', tensorData, [1, tensorData.length]);
      }
      for (const key of this.onnxModule.inKeys) {
        if (!input[key]) {
          console.warn('[PolicyRunner] Missing ONNX input:', {
            key,
            available: Object.keys(input),
          });
          return;
        }
      }

      const [result, carry] = await this.onnxModule.runInference(input);
      if (Object.keys(carry).length > 0) {
        this.onnxInputDict = { ...this.onnxInputDict, ...carry };
      }
      if (this.onnxModule.inKeys.includes('time_step')) {
        this.onnxTimeStep += 1;
      }

      const outKey = this.onnxModule.outKeys[0];
      const actionTensor = result.action ?? (outKey ? result[outKey] : null) ?? result.policy ?? null;
      if (!actionTensor) {
        return;
      }

      const raw = actionTensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      const expectedActionCount = this.policyRunner?.getNumActions() ?? 0;
      if (this.policyControl && action.length !== expectedActionCount) {
        console.warn('[PolicyRunner] Action size mismatch:', {
          expected: expectedActionCount,
          got: action.length,
        });
        return;
      }
      this.policyRunner.setLastActions(action);
    } catch (error) {
      console.warn('[PolicyRunner] ONNX inference failed:', error);
    } finally {
      this.onnxInferencing = false;
    }
  }

  private applyDragForces(): void {
    if (!this.dragStateManager || !this.mjModel || !this.mjData || !this.bodies) {
      return;
    }

    // Clear xfrc_applied (reset to zero at each step)
    for (let i = 0; i < this.mjData.xfrc_applied.length; i++) {
      this.mjData.xfrc_applied[i] = 0.0;
    }

    const dragged = this.dragStateManager.physicsObject;
    if (!dragged || !('bodyID' in dragged) || typeof dragged.bodyID !== 'number' || dragged.bodyID <= 0) {
      return;
    }

    const bodyId = dragged.bodyID as number;
    if (this.dynamicBodyIds && !this.dynamicBodyIds.has(bodyId)) {
      return;
    }

    // Update body positions (for drag calculation)
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.mjData.xpos, b, this.bodies[b].position);
        getQuaternion(this.mjData.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix(true, false);
      }
    }

    // Update offset
    this.dragStateManager.update();

    // Calculate force (Three.js coordinate system → MuJoCo coordinate system)
    const forceThree = this.dragStateManager.offset
      .clone()
      .multiplyScalar(this.dragForceScale);
    const force = threeToMjcCoordinate(forceThree);

    // Point where force is applied (world coordinates)
    const pointThree = this.dragStateManager.worldHit.clone();
    const point = threeToMjcCoordinate(pointThree);
    // Body position
    const bodyPos = new THREE.Vector3(
      this.mjData.xpos[bodyId * 3 + 0],
      this.mjData.xpos[bodyId * 3 + 1],
      this.mjData.xpos[bodyId * 3 + 2]
    );

    // Calculate torque: τ = r × F
    const r = new THREE.Vector3(
      point.x - bodyPos.x,
      point.y - bodyPos.y,
      point.z - bodyPos.z
    );
    const f = new THREE.Vector3(force.x, force.y, force.z);
    const torque = new THREE.Vector3().crossVectors(r, f);

    // Set xfrc_applied
    // xfrc_applied: (nbody, 6) = [fx, fy, fz, tx, ty, tz] for each body
    const offset = bodyId * 6;
    this.mjData.xfrc_applied[offset + 0] = force.x;
    this.mjData.xfrc_applied[offset + 1] = force.y;
    this.mjData.xfrc_applied[offset + 2] = force.z;
    this.mjData.xfrc_applied[offset + 3] = torque.x;
    this.mjData.xfrc_applied[offset + 4] = torque.y;
    this.mjData.xfrc_applied[offset + 5] = torque.z;
  }

  private updateCachedState(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (dynamicBodyIds && !dynamicBodyIds.has(b)) {
        continue;
      }
      if (this.bodies[b]) {
        if (!this.lastSimState.bodies.has(b)) {
          this.lastSimState.bodies.set(b, {
            position: new THREE.Vector3(),
            quaternion: new THREE.Quaternion(),
          });
        }
        const state = this.lastSimState.bodies.get(b) as BodyState;
        getPosition(this.mjData.xpos, b, state.position);
        getQuaternion(this.mjData.xquat, b, state.quaternion);
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      updateTendonGeometry(
        this.mjModel,
        this.mjData,
        {
          cylinders: this.mujocoRoot.cylinders,
          spheres: this.mujocoRoot.spheres!,
        },
        this.lastSimState.tendons
      );
    }
  }

  private applyViewerConfig(config: ViewerConfig | null): void {
    this.cameraState = applyViewerConfig(config, this.camera, this.controls, this.mjModel, this.mjData);
  }

  private computeDynamicBodyIds(mjModel: MjModel): Set<number> {
    const dynamic = new Set<number>();
    for (let bodyId = 1; bodyId < mjModel.nbody; bodyId++) {
      let current = bodyId;
      while (current > 0) {
        if (mjModel.body_jntnum[current] > 0) {
          dynamic.add(bodyId);
          break;
        }
        current = mjModel.body_parentid[current];
      }
    }
    return dynamic;
  }

  private syncStaticBodiesFromData(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let bodyId = 0; bodyId < this.mjModel.nbody; bodyId++) {
      if (dynamicBodyIds?.has(bodyId)) {
        continue;
      }
      const body = this.bodies[bodyId];
      if (!body) {
        continue;
      }
      getPosition(this.mjData.xpos, bodyId, body.position);
      getQuaternion(this.mjData.xquat, bodyId, body.quaternion);
    }
  }

  private render = (): void => {
    getCommandManager().updateDebugVisuals();

    if (this.mjData) {
      updateCameraFromData(this.mjData, this.camera, this.controls, this.cameraState);
    }
    this.controls.update();

    if (this.mjModel && this.mjData && this.bodies) {
      updateHeadlightFromCamera(this.camera, this.lights);

      for (const [b, state] of this.lastSimState.bodies) {
        const body = this.bodies[b];
        if (body) {
          body.position.copy(state.position);
          body.quaternion.copy(state.quaternion);
        }
      }

      updateLightsFromData(this.mujoco, this.mjData, this.lights);

      if (this.mujocoRoot && this.mujocoRoot.cylinders) {
        updateTendonRendering(
          {
            cylinders: this.mujocoRoot.cylinders,
            spheres: this.mujocoRoot.spheres!,
          },
          this.lastSimState.tendons
        );
      }
    }

    this.renderer.render(this.scene, this.camera);
  };

  private onWindowResize = (): void => {
    const { width, height } = this.getSize();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  dispose(): void {
    this.stop();
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyConfigPath = null;

    if (this.dragStateManager) {
      this.dragStateManager.dispose();
      this.dragStateManager = null;
    }

    // NOTE: Do NOT delete mjData/mjModel here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    this.mjData = null;
    this.mjModel = null;

    // NOTE: Do NOT dispose Three.js resources here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    // this.disposeThreeJSResources();

    window.removeEventListener('resize', this.onWindowResize);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    this.controls.dispose();
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();

    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }

    if (this.vrButton?.parentElement) {
      this.vrButton.parentElement.removeChild(this.vrButton);
      this.vrButton = null;
    }

    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;
    this.lastSimState.bodies.clear();
  }

  private disposeThreeJSResources(): void {
    if (!this.scene) {
      return;
    }

    this.scene.traverse((object) => {
      if ('geometry' in object && object.geometry) {
        (object.geometry as THREE.BufferGeometry).dispose();
      }
      if ('material' in object && object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => this.disposeMaterial(material));
        } else {
          this.disposeMaterial(object.material as THREE.Material);
        }
      }
    });

    while (this.scene.children.length > 0) {
      this.scene.remove(this.scene.children[0]);
    }
  }

  private disposeMaterial(material: THREE.Material): void {
    const anyMaterial = material as THREE.MeshStandardMaterial & {
      map?: THREE.Texture;
      aoMap?: THREE.Texture;
      emissiveMap?: THREE.Texture;
      metalnessMap?: THREE.Texture;
      normalMap?: THREE.Texture;
      roughnessMap?: THREE.Texture;
    };

    if (anyMaterial.map) {
      anyMaterial.map.dispose();
    }
    if (anyMaterial.aoMap) {
      anyMaterial.aoMap.dispose();
    }
    if (anyMaterial.emissiveMap) {
      anyMaterial.emissiveMap.dispose();
    }
    if (anyMaterial.metalnessMap) {
      anyMaterial.metalnessMap.dispose();
    }
    if (anyMaterial.normalMap) {
      anyMaterial.normalMap.dispose();
    }
    if (anyMaterial.roughnessMap) {
      anyMaterial.roughnessMap.dispose();
    }
    material.dispose();
  }

  private getSize(): { width: number; height: number } {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    return {
      width: Math.max(1, width),
      height: Math.max(1, height),
    };
  }

  /**
   * Restore scene from cache
   */
  private async restoreFromCache(scenePath: string): Promise<void> {
    const resources = this.sceneCacheManager.get(scenePath);
    if (!resources) {
      throw new Error(`Scene ${scenePath} not found in cache`);
    }

    // Remove existing root if present
    const existingRoot = this.scene.getObjectByName('MuJoCo Root');
    if (existingRoot) {
      this.scene.remove(existingRoot);
    }

    // Restore MuJoCo objects
    this.mjModel = resources.mjModel;
    this.mjData = resources.mjData;

    // Restore Three.js resources
    this.bodies = resources.bodies;
    this.lights = resources.lights;
    this.mujocoRoot = resources.mujocoRoot;

    // Re-add root to scene
    this.scene.add(this.mujocoRoot);

    // Restore skybox background
    this.scene.background = resources.skybox;

    // Run forward dynamics
    this.mujoco.mj_forward(this.mjModel, this.mjData);
    this.dynamicBodyIds = this.computeDynamicBodyIds(this.mjModel);
    this.syncStaticBodiesFromData();

    // Update runtime parameters
    this.timestep = this.mjModel.opt.timestep || 0.001;
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

    // Clear and update cached state
    this.lastSimState.bodies.clear();
    this.updateCachedState();

    // Initialize DragStateManager if needed
    if (!this.dragStateManager) {
      this.dragStateManager = new DragStateManager({
        scene: this.scene,
        renderer: this.renderer,
        camera: this.camera,
        container: this.container,
        controls: this.controls,
        draggableBodyIds: this.dynamicBodyIds,
      });
    } else {
      this.dragStateManager.setDraggableBodyIds(this.dynamicBodyIds);
    }
  }

  /**
   * Capture resources and add to cache
   */
  private async captureAndCacheResources(scenePath: string): Promise<void> {
    // Stop tracking and get FS files
    const fsFiles = this.resourceTracker.stopTracking(this.mujoco);

    if (!this.mjModel || !this.mjData || !this.bodies || !this.mujocoRoot) {
      console.warn('[SceneCache] Cannot cache scene: missing resources');
      return;
    }

    // Estimate memory usage
    const estimatedMemoryBytes = this.resourceTracker.estimateSceneMemory({
      mjModel: this.mjModel,
      mjData: this.mjData,
      bodies: this.bodies,
      meshes: {}, // Meshes are part of the scene
      mujocoRoot: this.mujocoRoot,
    });

    // Create cache entry
    await this.sceneCacheManager.set(scenePath, {
      scenePath,
      lastAccessed: Date.now(),
      loadedAt: Date.now(),
      mjModel: this.mjModel,
      mjData: this.mjData,
      bodies: this.bodies,
      lights: this.lights,
      meshes: {},
      mujocoRoot: this.mujocoRoot,
      skybox: this.scene.background instanceof THREE.CubeTexture ? this.scene.background : null,
      fsFiles,
      estimatedMemoryBytes,
    });

    // Log the cache operation
    const metrics = this.sceneCacheManager.getMetrics();
    this.memoryMonitor.logCacheOperation('load', scenePath, {
      memoryMB: estimatedMemoryBytes / 1048576,
      totalScenes: metrics.totalScenes,
      totalMemoryMB: metrics.totalMemoryBytes / 1048576,
    });
  }
}
