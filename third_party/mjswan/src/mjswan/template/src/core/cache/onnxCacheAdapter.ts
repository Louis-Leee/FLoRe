/**
 * ONNX Policy cache structure
 *
 * This is a stub for future ONNX model caching support.
 * When ONNX policy loading is implemented, this adapter will handle
 * loading, caching, and disposing of ONNX inference sessions.
 */
export interface ONNXPolicyCache {
  // Model binary data
  modelBuffer: ArrayBuffer;

  // Input/output information
  inputNames: string[];
  outputNames: string[];

  // Metadata
  metadata: {
    policyName: string;
    scenePath: string;
    version?: string;
  };

  // Memory estimation
  estimatedMemoryBytes: number;
}

/**
 * Adapter for caching ONNX models alongside scenes
 *
 * Future implementation will:
 * - Load ONNX models from scene directories
 * - Cache inference sessions with scene resources
 * - Dispose sessions when scenes are evicted
 * - Integrate with onnxruntime-web
 */
export class ONNXCacheAdapter {
  /**
   * Load ONNX model for scene
   *
   * @param scenePath - Path to the scene
   * @param policyName - Name of the policy to load
   * @returns Policy cache object or null if not found
   *
   * @future This will:
   * 1. Look for .onnx files in scene directory
   * 2. Parse policy metadata from config.json
   * 3. Create ONNX runtime inference session
   * 4. Return cached policy structure
   */
  async loadONNXPolicy(scenePath: string, policyName: string): Promise<ONNXPolicyCache | null> {
    // TODO: Implement ONNX model loading
    // 1. Fetch .onnx file from scene directory
    // 2. Create InferenceSession from onnxruntime-web
    // 3. Extract input/output names
    // 4. Estimate memory usage
    // 5. Return ONNXPolicyCache object

    console.log(`[ONNXCacheAdapter] ONNX loading not yet implemented for ${scenePath}/${policyName}`);
    return null;
  }

  /**
   * Attach ONNX policy to scene resources
   *
   * @param scenePath - Path to the scene
   * @param policy - Policy cache object
   *
   * @future This will associate the ONNX model with cached scene resources
   * so they are evicted together
   */
  attachToScene(scenePath: string, policy: ONNXPolicyCache): void {
    // TODO: Store policy reference with scene in cache manager
    console.log(`[ONNXCacheAdapter] Attaching policy ${policy.metadata.policyName} to scene ${scenePath}`);
  }

  /**
   * Dispose ONNX policy resources
   *
   * @param policy - Policy cache object to dispose
   *
   * @future This will:
   * 1. Close ONNX inference session
   * 2. Release model buffer
   * 3. Clear references
   */
  disposePolicy(policy: ONNXPolicyCache): void {
    // TODO: Implement ONNX session disposal
    // 1. session.release() or equivalent
    // 2. Clear ArrayBuffer reference
    console.log(`[ONNXCacheAdapter] Disposing policy ${policy.metadata.policyName}`);
  }

  /**
   * Estimate memory usage for ONNX model
   *
   * @param modelBuffer - Model binary data
   * @returns Estimated memory in bytes
   *
   * @future Implement more accurate estimation based on:
   * - Model buffer size
   * - Session initialization overhead
   * - Tensor allocations
   */
  estimateMemory(modelBuffer: ArrayBuffer): number {
    // Rough estimate: buffer size + 50% overhead for session
    return modelBuffer.byteLength * 1.5;
  }
}
