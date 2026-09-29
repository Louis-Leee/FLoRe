import { useEffect, useRef } from 'react';
import { mjswanRuntime, WasmMemoryLimitError } from '../core/engine/runtime';
import type { ViewerConfig } from '../core/engine/viewer_config';
import type { SplatConfig } from '../core/scene/splat';
import type { MainModule } from 'mujoco';
import type { EventConfig, TerrainData } from '../core/event/EventBase';

type MjswanViewerProps = {
  scenePath: string;
  baseUrl: string;
  policyConfigPath?: string | null;
  splatConfig?: SplatConfig | null;
  cameraConfig?: ViewerConfig | null;
  eventsConfig?: EventConfig[] | null;
  terrainData?: TerrainData | null;
  selectedMotion?: string | null;
  showReferenceMotion?: boolean;
  onStatusChange?: (status: string) => void;
  onError?: (error: Error) => void;
  onReady?: () => void;
  onRuntimeReady?: (runtime: mjswanRuntime) => void;
};

// Prevents infinite reload loops: stores the scene path that triggered a
// reload so we attempt the reload at most once per scene.
const OOM_RELOAD_KEY = 'mjswan_oom_reload_scene';

const MjswanViewer = ({
  scenePath,
  baseUrl,
  policyConfigPath,
  splatConfig,
  cameraConfig,
  eventsConfig,
  terrainData,
  selectedMotion,
  showReferenceMotion = true,
  onStatusChange,
  onError,
  onReady,
  onRuntimeReady,
}: MjswanViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<mjswanRuntime | null>(null);
  const mujocoRef = useRef<MainModule | null>(null);
  // Ref so the async init reads the latest splatConfig after awaiting WASM/scene load,
  // by which time App.tsx's selectedSplat effect has already fired.
  const splatConfigRef = useRef(splatConfig);
  splatConfigRef.current = splatConfig;
  const selectedMotionRef = useRef(selectedMotion);
  selectedMotionRef.current = selectedMotion;
  const showReferenceRef = useRef(showReferenceMotion);
  showReferenceRef.current = showReferenceMotion;

  useEffect(() => {
    let cancelled = false;

    const notify = (status: string) => {
      onStatusChange?.(status);
    };

    const init = async () => {
      if (!mujocoRef.current) {
        notify('Loading MuJoCo…');
        const mujocoModule = __MUJOCO_MT__
          ? await import('mujoco/mt')
          : await import('mujoco');
        mujocoRef.current = await mujocoModule.default();
      }
      if (cancelled) {
        return;
      }

      const container = containerRef.current;
      if (!container) {
        throw new Error('Failed to find viewer container.');
      }

      const mujoco = mujocoRef.current;
      if (!mujoco) {
        throw new Error('MuJoCo not loaded.');
      }

      if (!runtimeRef.current) {
        runtimeRef.current = new mjswanRuntime(mujoco, container, { baseUrl });
        onRuntimeReady?.(runtimeRef.current);
      }

      notify('Loading scene…');
      await runtimeRef.current.loadEnvironment(scenePath, policyConfigPath ?? null, splatConfigRef.current ?? null, cameraConfig ?? null, eventsConfig ?? null, terrainData ?? null);
      await runtimeRef.current.setSelectedMotion(selectedMotionRef.current ?? null);
      runtimeRef.current.setReferenceVisible(showReferenceRef.current);
      if (cancelled) {
        return;
      }
      sessionStorage.removeItem(OOM_RELOAD_KEY);
      notify('Running simulation');
      onReady?.();
    };

    init().catch((error) => {
      if (!cancelled) {
        if (error instanceof WasmMemoryLimitError) {
          // WASM linear memory cannot shrink within a session; a reload gives a
          // fresh heap.  Guard against looping if the scene is truly too large.
          if (sessionStorage.getItem(OOM_RELOAD_KEY) !== scenePath) {
            console.warn('[MjswanViewer] OOM — reloading page to free memory...');
            sessionStorage.setItem(OOM_RELOAD_KEY, scenePath);
            window.location.reload();
            return;
          }
          sessionStorage.removeItem(OOM_RELOAD_KEY);
        }
        console.error('Failed to initialize MuJoCo viewer:', error);
        onError?.(error instanceof Error ? error : new Error(String(error)));
        notify('Failed to load scene');
      }
    });

    return () => {
      cancelled = true;
      runtimeRef.current?.dispose();
      runtimeRef.current = null;
    };
  }, [scenePath, baseUrl, policyConfigPath, cameraConfig, eventsConfig, terrainData, onStatusChange, onError, onReady]);

  useEffect(() => {
    runtimeRef.current?.setReferenceVisible(showReferenceMotion);
  }, [showReferenceMotion]);

  return <div ref={containerRef} className="viewer" />;
};

export default MjswanViewer;
