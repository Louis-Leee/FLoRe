/**
 * Normalize a scene path for consistent caching
 *
 * This function ensures consistent path normalization across the codebase:
 * - Trims whitespace
 * - Removes leading "./" patterns
 * - Collapses multiple consecutive slashes
 *
 * @param scenePath The path to normalize
 * @returns Normalized path
 */
export function normalizeScenePath(scenePath: string): string {
  return scenePath
    .trim()
    .replace(/^(\.\/)+/, '')
    .replace(/\/+/g, '/');
}
