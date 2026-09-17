/**
 * Cache directory resolution. Extracted binaries are stored per release tag so
 * several versions can coexist; `STARLING_SERVE_CACHE` overrides the location.
 */
import { homedir } from "node:os";
import { join } from "node:path";

export const CACHE_ENV = "STARLING_SERVE_CACHE";

/**
 * Default cache root following OS conventions:
 * - Windows: `%LOCALAPPDATA%\starling-serve\cache`
 * - macOS: `~/Library/Caches/starling-serve`
 * - Linux: `$XDG_CACHE_HOME/starling-serve` or `~/.cache/starling-serve`
 */
export function defaultCacheDir(os: string = process.platform): string {
  if (os === "win32") {
    const localAppData =
      process.env["LOCALAPPDATA"] && process.env["LOCALAPPDATA"].trim() !== ""
        ? process.env["LOCALAPPDATA"]
        : join(homedir(), "AppData", "Local");

    return join(localAppData, "starling-serve", "cache");
  }

  if (os === "darwin") {
    return join(homedir(), "Library", "Caches", "starling-serve");
  }

  const xdg =
    process.env["XDG_CACHE_HOME"] && process.env["XDG_CACHE_HOME"].trim() !== ""
      ? process.env["XDG_CACHE_HOME"]
      : join(homedir(), ".cache");

  return join(xdg, "starling-serve");
}

/** Honor `STARLING_SERVE_CACHE` from `env` (defaults to `process.env`). */
export function cacheDir(env: NodeJS.ProcessEnv = process.env): string {
  const override = env["STARLING_SERVE_CACHE"];

  return override && override.trim() !== "" ? override : defaultCacheDir();
}

/** Layout inside the cache root: `<root>/releases/<tag>/<binary>`. */
export function releaseCachePath(root: string, tag: string, binary: string): string {
  return join(root, "releases", tag, binary);
}

/** Marker written next to a verified binary: `<binary>.verified`. */
export function verifiedMarkerPath(binaryPath: string): string {
  return `${binaryPath}.verified`;
}
