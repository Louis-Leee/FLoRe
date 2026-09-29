export class MuJoCoAssetCollector {
  private REFERENCE_ATTRS: Set<string>;
  private TAG_DIRECTORY_HINTS: Record<string, string[]>;
  private BINARY_EXTENSIONS: string[];
  private cache: Map<string, string[]>;
  private debug: boolean;

  constructor(options: { debug?: boolean } = {}) {
    this.REFERENCE_ATTRS = new Set([
      'file',
      'href',
      'src',
      'fileup',
      'fileback',
      'filedown',
      'filefront',
      'fileleft',
      'fileright',
    ]);

    this.TAG_DIRECTORY_HINTS = {
      include: ['includedir'],
      mesh: ['meshdir'],
      texture: ['texturedir'],
      heightfield: ['heightfielddir'],
      skin: ['skindir'],
    };

    this.BINARY_EXTENSIONS = ['.png', '.stl', '.skn', '.mjb', '.mjz'];
    this.cache = new Map();
    this.debug = options.debug || false;
  }

  async analyzeScene(xmlPath: string, baseUrl: string = './'): Promise<Array<string>> {
    if (!xmlPath || typeof xmlPath !== 'string') {
      throw new Error(`Invalid xmlPath: ${xmlPath}`);
    }

    if (!baseUrl || typeof baseUrl !== 'string') {
      throw new Error(`Invalid baseUrl: ${baseUrl}`);
    }

    const normalizedXmlPath = this._normalizePath(xmlPath);

    // For binary .mjb/.mjz files, just return the file itself - no XML parsing needed
    if (this._isBinary(normalizedXmlPath)) {
      return [normalizedXmlPath];
    }

    const cacheKey = `${baseUrl}/${normalizedXmlPath}`;
    if (this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      return Array.isArray(cached) ? cached : [];
    }

    try {
      const result = await this._collectAssets(normalizedXmlPath, baseUrl);
      const validResult = Array.isArray(result) ? result : [];
      this.cache.set(cacheKey, validResult);
      return validResult;
    } catch (error) {
      console.error(`[MuJoCoAssetCollector] Error analyzing scene ${xmlPath}:`, error);
      return [];
    }
  }

  clearCache(): void {
    this.cache.clear();
  }

  private async _collectAssets(rootPath: string, baseUrl: string): Promise<Array<string>> {
    const rootDir = this._getDirectoryPath(rootPath);
    const visited = new Set<string>();
    const collected = new Set<string>();

    const walk = async (filePath: string, parentHints: Record<string, string> = {}) => {
      const normalizedPath = this._normalizePath(filePath);
      const fullFilePath = `${baseUrl}/${normalizedPath}`.replace(/\/+/g, '/');

      if (visited.has(normalizedPath)) {
        return;
      }
      visited.add(normalizedPath);
      collected.add(normalizedPath);

      let xmlContent: string;
      try {
        const response = await fetch(fullFilePath);
        if (!response.ok) {
          console.warn(
            `[MuJoCoAssetCollector] Failed to fetch ${fullFilePath}: ${response.status}`
          );
          return;
        }
        xmlContent = await response.text();
      } catch (error) {
        console.error(`[MuJoCoAssetCollector] Error fetching ${filePath}:`, error);
        return;
      }

      const baseDir = this._getDirectoryPath(normalizedPath);
      const localHints = this._parseCompilerDirectories(xmlContent, baseDir);
      const directoryHints = this._mergeDirectoryHints(parentHints, localHints);

      const parser = new DOMParser();
      let xmlDoc: Document;
      try {
        xmlDoc = parser.parseFromString(xmlContent, 'text/xml');
        const parseError = xmlDoc.querySelector('parsererror');
        if (parseError) {
          throw new Error(parseError.textContent || 'Parser error');
        }
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        console.warn(`[MuJoCoAssetCollector] Failed to parse XML ${filePath}:`, message);
        return;
      }

      const allElements = xmlDoc.getElementsByTagName('*');

      for (const element of allElements) {
        const tagName = this._stripNamespace(element.tagName.toLowerCase());

        for (const attrName of this.REFERENCE_ATTRS) {
          const attrValue = element.getAttribute(attrName);
          if (!attrValue) {
            continue;
          }

          const reference = await this._resolveReference(
            attrValue,
            tagName,
            attrName,
            baseDir,
            directoryHints,
            baseUrl,
            rootDir
          );

          if (reference) {
            if ('path' in reference) {
              collected.add(reference.path);

              if (tagName === 'include' && attrName === 'file') {
                await walk(reference.path, directoryHints);
              }
            } else if ('text' in reference) {
              collected.add(reference.text);
            }
          }
        }
      }
    };

    await walk(rootPath);

    const result = Array.from(collected).sort();
    if (!Array.isArray(result)) {
      console.error('[MuJoCoAssetCollector] Internal error: result is not an array');
      return [];
    }

    if (this.debug) {
      console.log(
        `[MuJoCoAssetCollector] Successfully analyzed ${rootPath}: found ${result.length} assets`
      );
    }
    return result;
  }

  private _parseCompilerDirectories(xmlContent: string, baseDir: string): Record<string, string> {
    const directories: Record<string, string> = {};
    const parser = new DOMParser();
    let xmlDoc: Document;
    try {
      xmlDoc = parser.parseFromString(xmlContent, 'text/xml');
    } catch {
      return directories;
    }

    const compilerElements = xmlDoc.getElementsByTagName('compiler');
    for (const compiler of compilerElements) {
      for (const [tag, hints] of Object.entries(this.TAG_DIRECTORY_HINTS)) {
        for (const hint of hints) {
          const attrValue = compiler.getAttribute(hint);
          if (attrValue) {
            directories[tag] = this._normalizePath(this._joinPath(baseDir, attrValue));
          }
        }
      }
    }

    return directories;
  }

  private _mergeDirectoryHints(
    parentHints: Record<string, string>,
    localHints: Record<string, string>
  ): Record<string, string> {
    const merged = { ...parentHints };
    for (const [key, value] of Object.entries(localHints)) {
      merged[key] = value;
    }
    return merged;
  }

  private async _resolveReference(
    attrValue: string,
    tagName: string,
    attrName: string,
    baseDir: string,
    directoryHints: Record<string, string>,
    baseUrl: string,
    rootDir: string
  ): Promise<{ path: string } | { text: string } | null> {
    if (!attrValue) {
      return null;
    }

    const cleanedAttr = attrValue.trim();
    if (!cleanedAttr) {
      return null;
    }

    if (cleanedAttr.startsWith('http://') || cleanedAttr.startsWith('https://')) {
      return { path: cleanedAttr };
    }

    if (cleanedAttr.startsWith('<') && cleanedAttr.endsWith('>')) {
      return { text: cleanedAttr };
    }

    const hintDir = directoryHints[tagName];
    const resolved = this._normalizePath(this._joinPath(hintDir ?? baseDir, cleanedAttr));

    if (!resolved) {
      return null;
    }

    if (tagName === 'include' && attrName === 'file') {
      return { path: resolved };
    }

    if (this._isBinary(resolved)) {
      return { path: resolved };
    }

    const fullPath = `${baseUrl}/${resolved}`.replace(/\/+/g, '/');
    try {
      const response = await fetch(fullPath, { method: 'HEAD' });
      if (!response.ok) {
        return { path: resolved };
      }
    } catch {
      return { path: resolved };
    }

    if (resolved.startsWith(rootDir)) {
      return { path: resolved };
    }

    return { path: resolved };
  }

  private _normalizePath(path: string): string {
    if (!path) {
      return '';
    }

    const isAbsolute = path.startsWith('/');
    let normalized = path.replace(/\\/g, '/').replace(/\/+/g, '/').replace(/\/$/, '');
    const parts = normalized.split('/');
    const resolved: string[] = [];

    for (const part of parts) {
      if (!part || part === '.') {
        continue;
      }
      if (part === '..') {
        if (resolved.length > 0) {
          resolved.pop();
        }
        continue;
      }
      resolved.push(part);
    }

    let result = resolved.join('/');
    if (isAbsolute && result) {
      result = `/${result}`;
    }
    return result || '';
  }

  private _stripNamespace(tagName: string): string {
    const parts = tagName.split(':');
    return parts.length > 1 ? parts[1] : tagName;
  }

  private _isBinary(path: string): boolean {
    const lower = path.toLowerCase();
    return this.BINARY_EXTENSIONS.some((ext) => lower.endsWith(ext));
  }

  private _getDirectoryPath(path: string): string {
    if (!path) {
      return '';
    }
    const parts = this._normalizePath(path).split('/');
    return parts.slice(0, -1).join('/');
  }

  private _joinPath(...parts: (string | undefined)[]): string {
    const filtered = parts.filter((part) => part !== null && part !== undefined && part !== '.');
    if (filtered.length === 0) {
      return '';
    }

    const joined = filtered.join('/').replace(/\/+/g, '/');

    if (parts[0] && parts[0].startsWith('/')) {
      return this._normalizePath(joined);
    }

    return this._normalizePath(joined.replace(/^\//, ''));
  }
}

export const mujocoAssetCollector = new MuJoCoAssetCollector();
