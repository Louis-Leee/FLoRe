import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MantineProvider } from '@mantine/core';
import MjswanViewer from './components/MjswanViewer';
import ControlPanel from './ControlPanel';
import type { mjswanRuntime } from './core/engine/runtime';
import type { SplatConfig } from './core/scene/splat';
import { theme } from './AppTheme';
import { LoadingProvider, useLoading } from './contexts/LoadingContext';
import { Loader } from './components/Loader';
import './App.css';

interface PolicyConfig {
  name: string;
  metadata: Record<string, unknown>;
  config?: string;
  default?: boolean;
  motions?: Array<{
    name: string;
    default?: boolean;
  }>;
}

interface ViewerConfig {
  lookat?: [number, number, number];
  distance?: number;
  fovy?: number;
  elevation?: number;
  azimuth?: number;
  originType?: 'AUTO' | 'WORLD' | 'ASSET_ROOT' | 'ASSET_BODY';
  entityName?: string;
  bodyName?: string;
  enableReflections?: boolean;
  enableShadows?: boolean;
  height?: number;
  width?: number;
}

interface SceneConfig {
  name: string;
  metadata: Record<string, unknown>;
  policies: PolicyConfig[];
  path?: string;
  splats?: SplatConfig[];
  splatSection?: boolean;
  camera?: ViewerConfig;
  events?: import('./core/event/EventBase').EventConfig[];
  terrainData?: import('./core/event/EventBase').TerrainData;
}

interface ProjectConfig {
  name: string;
  id: string | null;
  metadata: Record<string, unknown>;
  scenes: SceneConfig[];
}

interface AppConfig {
  version: string;
  projects: ProjectConfig[];
}

const PANEL_QUERY_PARAM = 'panel';
const REF_QUERY_PARAM = 'ref';

function getProjectIdFromLocation(): string | null {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '/');
  const pathname = window.location.pathname;

  let pathClean = pathname.replace(/^\/+|\/+$/g, '');
  const baseClean = base.replace(/^\/+|\/+$/g, '');

  if (baseClean) {
    if (pathClean === baseClean) {
      pathClean = '';
    } else if (pathClean.startsWith(`${baseClean}/`)) {
      pathClean = pathClean.slice(baseClean.length + 1);
    }
  }

  if (!pathClean) {
    return null;
  }

  const projectId = pathClean.split('/')[0];
  if (projectId === 'main') {
    return null;
  }
  if (projectId.includes('.') || projectId === 'assets') {
    return null;
  }
  return projectId;
}

function sanitizeName(name: string): string {
  return name.toLowerCase().replace(/ /g, '_').replace(/-/g, '_');
}

function buildConfigCandidates(baseUrl: string, projectId: string | null): string[] {
  const normalizedBase = (baseUrl || '/').replace(/\/+$/, '/');
  const candidates = new Set<string>();
  const add = (path: string, base?: string) => {
    if (!path) {
      return;
    }
    try {
      const resolved = new URL(path, base || window.location.href).toString();
      candidates.add(resolved);
    } catch {
      candidates.add(path.replace(/\/+/g, '/'));
    }
  };

  const originBase = `${window.location.origin}/`;
  const appBase = new URL(normalizedBase, originBase).toString();
  add('assets/config.json', appBase);

  const pathname = window.location.pathname;
  const parts = pathname.split('/').filter(Boolean);
  if (parts.length > 0) {
    const last = parts[parts.length - 1];
    if (last === 'index.html') {
      parts.pop();
    }
  }
  if (parts.length > 0) {
    const last = parts[parts.length - 1];
    if (last === (projectId ?? 'main')) {
      parts.pop();
    }
  }
  const rootPath = `/${parts.join('/')}${parts.length ? '/' : ''}`;
  const rootBase = `${window.location.origin}${rootPath}`;
  add('assets/config.json', rootBase);

  add('assets/config.json');
  add('../assets/config.json');
  add('../../assets/config.json');

  return Array.from(candidates);
}

async function loadConfig(baseUrl: string, projectId: string | null): Promise<AppConfig> {
  const params = new URLSearchParams(window.location.search);
  const override = params.get('config');
  const candidates = buildConfigCandidates(baseUrl, projectId);
  if (override) {
    try {
      candidates.unshift(new URL(override, window.location.href).toString());
    } catch {
      candidates.unshift(override);
    }
  }
  let lastError: Error | null = null;

  for (const url of candidates) {
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status}`);
      }
      const text = await response.text();
      const trimmed = text.trim();
      const contentType = response.headers.get('content-type') || '';
      if (
        contentType.includes('text/html') ||
        trimmed.startsWith('<!doctype') ||
        trimmed.startsWith('<html')
      ) {
        throw new Error(`Received HTML from ${url}`);
      }
      try {
        return JSON.parse(text) as AppConfig;
      } catch (error) {
        throw new Error(
          `Invalid JSON from ${url}: ${error instanceof Error ? error.message : String(error)}`
        );
      }
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
    }
  }

  throw lastError ?? new Error('Failed to load config.json.');
}

function pickScene(project: ProjectConfig, sceneQuery: string | null): SceneConfig | null {
  if (!project.scenes.length) {
    return null;
  }
  if (!sceneQuery) {
    return project.scenes[0];
  }
  const normalized = sceneQuery.trim().toLowerCase();
  return (
    project.scenes.find((scene) => scene.name.toLowerCase() === normalized) ||
    project.scenes.find((scene) => sanitizeName(scene.name) === normalized) ||
    project.scenes[0]
  );
}

function pickPolicy(scene: SceneConfig, policyQuery: string | null): string | null {
  if (!scene.policies.length) {
    return null;
  }
  const fallback = scene.policies.find((policy) => policy.default) ?? scene.policies[0];
  if (!policyQuery) {
    return fallback.name;
  }
  const normalized = policyQuery.trim().toLowerCase();
  const found =
    scene.policies.find((policy) => policy.name.toLowerCase() === normalized) ||
    scene.policies.find((policy) => sanitizeName(policy.name) === normalized);
  return found?.name ?? fallback.name;
}

function pickMotion(policy: PolicyConfig | null, motionQuery: string | null): string | null {
  if (!policy?.motions?.length) {
    return null;
  }
  const fallback = policy.motions.find((motion) => motion.default) ?? policy.motions[0];
  if (!motionQuery) {
    return fallback.name;
  }
  const normalized = motionQuery.trim().toLowerCase();
  const found =
    policy.motions.find((motion) => motion.name.toLowerCase() === normalized) ||
    policy.motions.find((motion) => sanitizeName(motion.name) === normalized);
  return found?.name ?? fallback.name;
}

function isPanelVisibleFromSearch(search: string): boolean {
  const params = new URLSearchParams(search);
  return params.get(PANEL_QUERY_PARAM) !== '0';
}

function isRefVisibleFromSearch(search: string): boolean {
  const params = new URLSearchParams(search);
  return params.get(REF_QUERY_PARAM) !== '0';
}

function buildProjectPath(projectId: string | null): string {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '/');
  const normalizedBase = base.replace(/^\//g, '').replace(/\/+$/g, '');

  let pathname = normalizedBase ? `/${normalizedBase}/` : '/';
  if (projectId && projectId !== 'main') {
    pathname += `${projectId}/`;
  }

  return pathname;
}

function updateUrlParams({
  projectId,
  sceneName,
  policyName,
  panelVisible,
  showReference,
}: {
  projectId: string | null;
  sceneName: string | null;
  policyName: string | null;
  panelVisible: boolean;
  showReference: boolean;
}) {
  const pathname = buildProjectPath(projectId);
  const params = new URLSearchParams(window.location.search);

  if (sceneName) {
    params.set('scene', sceneName);
  } else {
    params.delete('scene');
  }
  if (policyName) {
    params.set('policy', policyName);
  } else {
    params.delete('policy');
  }
  if (panelVisible) {
    params.delete(PANEL_QUERY_PARAM);
  } else {
    params.set(PANEL_QUERY_PARAM, '0');
  }
  if (showReference) {
    params.delete(REF_QUERY_PARAM);
  } else {
    params.set(REF_QUERY_PARAM, '0');
  }

  const search = params.toString();
  const newUrl = pathname + (search ? `?${search}` : '') + window.location.hash;
  window.history.replaceState({}, '', newUrl);
}

function AppContent() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [currentProject, setCurrentProject] = useState<ProjectConfig | null>(null);
  const [currentScene, setCurrentScene] = useState<SceneConfig | null>(null);
  const [selectedPolicy, setSelectedPolicy] = useState<string | null>(null);
  const [selectedMotion, setSelectedMotion] = useState<string | null>(null);
  const [showReferenceMotion, setShowReferenceMotion] = useState(() =>
    isRefVisibleFromSearch(window.location.search)
  );
  const [error, setError] = useState<string | null>(null);
  const [selectedSplat, setSelectedSplat] = useState<string | null>(null);
  const [customSplatUrl, setCustomSplatUrl] = useState<string | null>(null);
  const [panelVisible, setPanelVisible] = useState(() =>
    isPanelVisibleFromSearch(window.location.search)
  );
  const runtimeRef = useRef<mjswanRuntime | null>(null);
  const { showLoading, hideLoading, setLoadingMessage } = useLoading();

  const projectId = useMemo(() => getProjectIdFromLocation(), []);
  const sceneQuery = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('scene');
  }, []);
  const policyQuery = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('policy');
  }, []);

  useEffect(() => {
    showLoading('Loading…');
    loadConfig(import.meta.env.BASE_URL || '/', projectId)
      .then((data: AppConfig) => {
        setConfig(data);
        const project = data.projects.find((p) => {
          if (projectId === null) {
            return p.id === null;
          }
          return p.id === projectId;
        });
        if (!project) {
          throw new Error(`Project "${projectId ?? '(main)'}" not found in config.json.`);
        }
        setCurrentProject(project);
        const selectedScene = pickScene(project, sceneQuery);
        setCurrentScene(selectedScene);
        const initialPolicy = selectedScene ? pickPolicy(selectedScene, policyQuery) : null;
        setSelectedPolicy(initialPolicy);
        const initialPolicyConfig = selectedScene?.policies.find((policy) => policy.name === initialPolicy) ?? null;
        setSelectedMotion(pickMotion(initialPolicyConfig, null));
      })
      .catch((err) => {
        console.error('Failed to load config:', err);
        setError(err.message || 'Failed to load config.');
        hideLoading();
      });
  }, [projectId, sceneQuery, policyQuery, showLoading, hideLoading]);

  const scenePath = useMemo(() => {
    if (!currentProject || !currentScene) {
      return null;
    }
    const projectDir = currentProject.id ? currentProject.id : 'main';
    const sceneRelPath = currentScene.path
      ? currentScene.path
      : `scene/${sanitizeName(currentScene.name)}/scene.xml`;
    return `${projectDir}/assets/${sceneRelPath}`.replace(/\/+/g, '/');
  }, [currentProject, currentScene]);
  const selectedPolicyConfig = useMemo(() => {
    if (!currentScene || !selectedPolicy) {
      return null;
    }
    return currentScene.policies.find((policy) => policy.name === selectedPolicy) ?? null;
  }, [currentScene, selectedPolicy]);
  const policyConfigPath = useMemo(() => {
    if (!currentProject || !selectedPolicyConfig?.config) {
      return null;
    }
    const projectDir = currentProject.id ? currentProject.id : 'main';
    return `${projectDir}/assets/${selectedPolicyConfig.config}`.replace(/\/+/g, '/');
  }, [currentProject, selectedPolicyConfig]);
  const motionOptions = useMemo(() => {
    if (!selectedPolicyConfig?.motions?.length) {
      return [] as { value: string; label: string }[];
    }
    return selectedPolicyConfig.motions.map((motion) => ({
      value: motion.name,
      label: motion.name,
    }));
  }, [selectedPolicyConfig]);

  // Resolve bundled splat paths to URLs the viewer can fetch.
  // When config.json uses "path" (bundled), convert it to a relative URL.
  // When config.json uses "url" (external), pass it through unchanged.
  const resolvedSplats = useMemo(() => {
    if (!currentProject || !currentScene?.splats?.length) return [] as SplatConfig[];
    const projectDir = currentProject.id ? currentProject.id : 'main';
    return currentScene.splats.map((splat) => {
      if (splat.path) {
        const resolvedUrl = `${projectDir}/assets/${splat.path}`.replace(/\/+/g, '/');
        return { ...splat, url: resolvedUrl };
      }
      return splat;
    });
  }, [currentProject, currentScene?.splats]);

  const resolvedSplatConfig = useMemo(() => {
    if (!selectedSplat) return customSplatUrl ? { name: 'Custom', url: customSplatUrl } : null;
    return resolvedSplats.find((s) => s.name === selectedSplat) ?? null;
  }, [resolvedSplats, selectedSplat, customSplatUrl]);

  const projectOptions = useMemo(() => {
    if (!config) {
      return [] as { value: string; label: string }[];
    }
    return config.projects.map((project) => ({
      value: project.id ?? 'main',
      label: project.name || (project.id ?? 'Main'),
    }));
  }, [config]);

  const sceneOptions = useMemo(() => {
    if (!currentProject) {
      return [] as { value: string; label: string }[];
    }
    return currentProject.scenes.map((scene) => ({ value: scene.name, label: scene.name }));
  }, [currentProject]);

  const policyOptions = useMemo(() => {
    if (!currentScene || !currentScene.policies) {
      return [] as { value: string; label: string }[];
    }
    return currentScene.policies.map((policy) => ({ value: policy.name, label: policy.name }));
  }, [currentScene]);

  const projectValue = currentProject ? (currentProject.id ?? 'main') : null;
  const sceneValue = currentScene?.name ?? null;
  const handleViewerError = useCallback((err: Error) => {
    setError(err.message);
    hideLoading();
  }, [hideLoading]);

  const handleViewerReady = useCallback(() => {
    hideLoading();
  }, [hideLoading]);

  const handleViewerStatus = useCallback((status: string) => {
    // "Running simulation" is the done sentinel — onReady drives hideLoading.
    // "Failed to load scene" is handled by the error HUD.
    if (status === 'Running simulation' || status === 'Failed to load scene') {
      return;
    }
    setLoadingMessage(status);
  }, [setLoadingMessage]);

  // Reset splat selection when switching scenes
  useEffect(() => {
    const firstSplat = currentScene?.splats?.[0];
    setSelectedSplat(firstSplat ? firstSplat.name : null);
    setCustomSplatUrl(null);
  }, [currentScene]);

  const handleRuntimeReady = useCallback((runtime: mjswanRuntime) => {
    runtimeRef.current = runtime;
  }, []);

  useEffect(() => {
    setSelectedMotion(pickMotion(selectedPolicyConfig, null));
    setShowReferenceMotion(Boolean(selectedPolicyConfig?.motions?.length));
  }, [selectedPolicyConfig]);

  const splatOptions = useMemo(() => {
    if (!currentScene?.splats?.length) return [] as { value: string; label: string }[];
    return currentScene.splats.map((s) => ({ value: s.name, label: s.name }));
  }, [currentScene?.splats]);

  const handleSplatChange = useCallback(async (value: string | null) => {
    setSelectedSplat(value);
    setCustomSplatUrl(null);
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    const splat = value === null ? null : (resolvedSplats.find((s) => s.name === value) ?? null);
    if (value !== null && !splat) {
      return;
    }
    showLoading(value === null ? 'Removing splat…' : `Loading splat "${value}"…`);
    try {
      await runtime.setSplat(splat);
    } catch (e) {
      console.error('Failed to load splat:', e);
    } finally {
      hideLoading();
    }
  }, [resolvedSplats, showLoading, hideLoading]);

  const handleSplatUrlLoad = useCallback(async (url: string): Promise<boolean> => {
    try {
      const res = await fetch(url, { method: 'HEAD' });
      if (!res.ok) return false;
    } catch {
      return false;
    }
    const runtime = runtimeRef.current;
    if (!runtime) {
      return false;
    }
    showLoading('Loading splat "Custom"…');
    try {
      await runtime.setSplat({ name: 'Custom', url });
      setCustomSplatUrl(url);
      return true;
    } catch (e) {
      console.error('Failed to load custom splat:', e);
      return false;
    } finally {
      hideLoading();
    }
  }, [showLoading, hideLoading]);

  const handleCalibrateSplat = useCallback((scale: number, xOffset: number, yOffset: number, zOffset: number, roll: number, pitch: number, yaw: number) => {
    const splat = resolvedSplatConfig ?? (customSplatUrl ? { name: 'Custom', url: customSplatUrl } : null);
    if (splat) {
      runtimeRef.current?.calibrateSplat({ ...splat, scale, xOffset, yOffset, zOffset, roll, pitch, yaw });
    }
  }, [resolvedSplatConfig, customSplatUrl]);

  const syncUrlState = useCallback((next: {
    projectId?: string | null;
    sceneName?: string | null;
    policyName?: string | null;
    panelVisible?: boolean;
    showReference?: boolean;
  }) => {
    updateUrlParams({
      projectId: next.projectId ?? currentProject?.id ?? null,
      sceneName: next.sceneName ?? currentScene?.name ?? null,
      policyName: next.policyName ?? selectedPolicy,
      panelVisible: next.panelVisible ?? panelVisible,
      showReference: next.showReference ?? showReferenceMotion,
    });
  }, [currentProject?.id, currentScene?.name, selectedPolicy, panelVisible, showReferenceMotion]);

  const handleProjectChange = useCallback(
    (value: string | null) => {
      if (!config || !value) {
        return;
      }
      const normalized = value === 'main' ? null : value;
      const project = config.projects.find((p) => (p.id ?? 'main') === (normalized ?? 'main'));
      if (!project) {
        return;
      }
      const nextScene = pickScene(project, null);
      showLoading(nextScene ? `Loading scene "${nextScene.name}"…` : 'Loading…');
      setCurrentProject(project);
      setCurrentScene(nextScene);
      const nextPolicy = nextScene ? pickPolicy(nextScene, null) : null;
      setSelectedPolicy(nextPolicy);
      const nextPolicyConfig = nextScene?.policies.find((policy) => policy.name === nextPolicy) ?? null;
      setSelectedMotion(pickMotion(nextPolicyConfig, null));
      syncUrlState({
        projectId: project.id,
        sceneName: nextScene?.name ?? null,
        policyName: nextPolicy,
      });
    },
    [config, showLoading, syncUrlState]
  );

  const handleSceneChange = useCallback(
    (value: string | null) => {
      if (!currentProject || !value) {
        return;
      }
      const scene = currentProject.scenes.find((s) => s.name === value);
      if (!scene) {
        return;
      }
      showLoading(`Loading scene "${scene.name}"…`);
      setCurrentScene(scene);
      const nextPolicy = pickPolicy(scene, null);
      setSelectedPolicy(nextPolicy);
      const nextPolicyConfig = scene.policies.find((policy) => policy.name === nextPolicy) ?? null;
      setSelectedMotion(pickMotion(nextPolicyConfig, null));
      syncUrlState({
        projectId: currentProject.id,
        sceneName: value,
        policyName: nextPolicy,
      });
    },
    [currentProject, showLoading, syncUrlState]
  );

  const handlePolicyChange = useCallback(
    (value: string | null) => {
      if (value !== selectedPolicy) {
        showLoading(value ? `Loading policy "${value}"…` : 'Loading policy…');
      }
      setSelectedPolicy(value);
      const nextPolicyConfig = currentScene?.policies.find((policy) => policy.name === value) ?? null;
      setSelectedMotion(pickMotion(nextPolicyConfig, null));
      syncUrlState({ policyName: value });
    },
    [selectedPolicy, showLoading, syncUrlState]
  );

  const handlePanelVisibleChange = useCallback(
    (visible: boolean) => {
      setPanelVisible(visible);
      syncUrlState({ panelVisible: visible });
    },
    [syncUrlState]
  );

  const handleMotionChange = useCallback(async (value: string | null) => {
    const previousMotion = selectedMotion;
    setSelectedMotion(value);
    const runtime = runtimeRef.current;
    if (!runtime) {
      return;
    }
    showLoading(value === null ? 'Clearing motion…' : `Loading motion "${value}"…`);
    try {
      const accepted = await runtime.setSelectedMotion(value);
      if (accepted === false && value !== null) {
        setSelectedMotion(runtime.getSelectedMotionName() ?? previousMotion);
      }
    } catch (e) {
      console.error('Failed to load motion:', e);
    } finally {
      hideLoading();
    }
  }, [selectedMotion, showLoading, hideLoading]);

  const handleShowReferenceChange = useCallback((value: boolean) => {
    setShowReferenceMotion(value);
    runtimeRef.current?.setReferenceVisible(value);
    syncUrlState({ showReference: value });
  }, [syncUrlState]);

  if (error) {
    return (
      <MantineProvider theme={theme} defaultColorScheme="auto">
        <div className="app">
          <div className="hud hud-error">
            <h1 className="hud-title">mjswan</h1>
            <p className="hud-message">{error}</p>
          </div>
        </div>
      </MantineProvider>
    );
  }

  if (!currentProject || !currentScene || !scenePath) {
    return null;
  }

  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <div className="app">
        <Loader />
        <ControlPanel
          visible={panelVisible}
          onVisibleChange={handlePanelVisibleChange}
          projects={projectOptions}
          projectValue={projectValue}
          projectLabel={currentProject?.name ?? 'mjswan'}
          onProjectChange={handleProjectChange}
          scenes={sceneOptions}
          sceneValue={sceneValue}
          onSceneChange={handleSceneChange}
          splats={splatOptions}
          splatSection={currentScene?.splatSection ?? false}
          splatValue={selectedSplat}
          onSplatChange={handleSplatChange}
          splatConfig={resolvedSplatConfig}
          onCalibrateSplat={handleCalibrateSplat}
          onSplatUrlLoad={handleSplatUrlLoad}
          policies={policyOptions}
          policyValue={selectedPolicy}
          onPolicyChange={handlePolicyChange}
          motions={motionOptions}
          motionValue={selectedMotion}
          onMotionChange={handleMotionChange}
          showReferenceMotion={showReferenceMotion}
          onShowReferenceMotionChange={handleShowReferenceChange}
          commandsEnabled={!!policyConfigPath}
        />
        <MjswanViewer
          scenePath={scenePath}
          baseUrl={import.meta.env.BASE_URL || '/'}
          policyConfigPath={policyConfigPath}
          splatConfig={resolvedSplatConfig}
          cameraConfig={currentScene?.camera}
          eventsConfig={currentScene?.events}
          terrainData={currentScene?.terrainData}
          selectedMotion={selectedMotion}
          showReferenceMotion={showReferenceMotion}
          onError={handleViewerError}
          onReady={handleViewerReady}
          onStatusChange={handleViewerStatus}
          onRuntimeReady={handleRuntimeReady}
        />
      </div>
    </MantineProvider>
  );
}

function App() {
  return (
    <LoadingProvider>
      <AppContent />
    </LoadingProvider>
  );
}

export default App;
