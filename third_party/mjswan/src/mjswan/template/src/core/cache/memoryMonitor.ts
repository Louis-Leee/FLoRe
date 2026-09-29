import * as THREE from 'three';
import type { CacheMetrics } from './sceneCacheManager';

/**
 * Memory statistics including JS heap and cache info
 */
export interface MemoryStats {
  // JavaScript heap
  jsHeapUsedMB: number;
  jsHeapTotalMB: number;
  jsHeapLimitMB: number;

  // Cache metrics
  cachedScenesCount: number;
  estimatedCacheMemoryMB: number;

  // GPU memory estimate (heuristic)
  estimatedGPUMemoryMB: number;

  // Per-scene breakdown
  sceneBreakdown: Array<{
    scenePath: string;
    memoryMB: number;
    lastAccessed: Date;
  }>;
}

/**
 * Performance memory interface for browsers that support it
 */
interface PerformanceMemory {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
}

interface PerformanceWithMemory extends Performance {
  memory?: PerformanceMemory;
}

/**
 * Monitors and logs memory usage
 */
export class MemoryMonitor {
  /**
   * Get current memory statistics
   */
  public getStats(cacheMetrics: CacheMetrics, sceneData?: Map<string, { memoryBytes: number; lastAccessed: number }>): MemoryStats {
    const jsHeap = this.getJSHeapStats();

    const stats: MemoryStats = {
      jsHeapUsedMB: jsHeap.used,
      jsHeapTotalMB: jsHeap.total,
      jsHeapLimitMB: jsHeap.limit,
      cachedScenesCount: cacheMetrics.totalScenes,
      estimatedCacheMemoryMB: cacheMetrics.totalMemoryBytes / 1048576,
      estimatedGPUMemoryMB: 0, // Difficult to estimate accurately
      sceneBreakdown: [],
    };

    if (sceneData) {
      for (const [scenePath, data] of sceneData.entries()) {
        stats.sceneBreakdown.push({
          scenePath,
          memoryMB: data.memoryBytes / 1048576,
          lastAccessed: new Date(data.lastAccessed),
        });
      }
    }

    return stats;
  }

  /**
   * Log cache operation to console
   */
  public logCacheOperation(operation: 'load' | 'hit' | 'miss' | 'evict', scenePath: string, details?: {
    memoryMB?: number;
    totalScenes?: number;
    totalMemoryMB?: number;
    elapsedMs?: number;
  }): void {
    const prefix = '[SceneCache]';

    switch (operation) {
      case 'load': {
        const memStr = details?.memoryMB ? ` (${details.memoryMB.toFixed(1)} MB)` : '';
        const cacheStr = details?.totalScenes !== undefined && details?.totalMemoryMB !== undefined
          ? ` - Cache: ${details.totalScenes}/5 scenes, ${details.totalMemoryMB.toFixed(1)} MB total`
          : '';
        console.log(`${prefix} Loaded: ${scenePath}${memStr}${cacheStr}`);
        break;
      }

      case 'hit': {
        const timeStr = details?.elapsedMs !== undefined ? ` (~${details.elapsedMs.toFixed(0)}ms)` : '';
        console.log(`${prefix} Cache HIT: ${scenePath}${timeStr}`);
        break;
      }

      case 'miss': {
        console.log(`${prefix} Cache MISS: ${scenePath}`);
        break;
      }

      case 'evict': {
        const memStr = details?.memoryMB ? ` (${details.memoryMB.toFixed(1)} MB)` : '';
        console.log(`${prefix} Evicted LRU: ${scenePath}${memStr}`);
        break;
      }
    }
  }

  /**
   * Log detailed statistics to console
   */
  public logStats(stats: MemoryStats): void {
    console.log('[SceneCache] Memory Statistics:');
    console.log(`  JS Heap: ${stats.jsHeapUsedMB.toFixed(1)} MB / ${stats.jsHeapTotalMB.toFixed(1)} MB (Limit: ${stats.jsHeapLimitMB.toFixed(1)} MB)`);
    console.log(`  Cached Scenes: ${stats.cachedScenesCount}/5`);
    console.log(`  Estimated Cache Memory: ${stats.estimatedCacheMemoryMB.toFixed(1)} MB`);

    if (stats.sceneBreakdown.length > 0) {
      console.log('  Scene Breakdown:');
      for (const scene of stats.sceneBreakdown) {
        console.log(`    - ${scene.scenePath}: ${scene.memoryMB.toFixed(1)} MB (accessed: ${scene.lastAccessed.toLocaleTimeString()})`);
      }
    }
  }

  /**
   * Read JS heap via performance.memory
   */
  private getJSHeapStats(): { used: number; total: number; limit: number } {
    const perf = performance as PerformanceWithMemory;

    if (perf.memory) {
      return {
        used: perf.memory.usedJSHeapSize / 1048576,
        total: perf.memory.totalJSHeapSize / 1048576,
        limit: perf.memory.jsHeapSizeLimit / 1048576,
      };
    }

    // performance.memory not available (not Chrome or disabled)
    return {
      used: 0,
      total: 0,
      limit: 0,
    };
  }

  /**
   * Estimate GPU memory from WebGL context (very approximate)
   */
  private estimateGPUMemory(renderer: THREE.WebGLRenderer): number {
    // This is extremely difficult to measure accurately
    // WebGL doesn't provide direct access to GPU memory usage
    const info = renderer.info;

    // Rough estimate based on geometry and texture counts
    const geometries = info.memory.geometries;
    const textures = info.memory.textures;

    // Very rough heuristic: 1MB per geometry, 2MB per texture
    const estimatedMB = geometries * 1 + textures * 2;

    return estimatedMB;
  }
}
