/**
 * Cache directory resolution. Extracted binaries are stored per repository and
 * release tag so versions — and alternate `STARLING_SERVE_REPO` sources — can
 * coexist without one source's binary ever shadowing another's;
 * `STARLING_SERVE_CACHE` overrides the location.
 */
import { homedir } from "node:os";
import { join } from "node:path";

export const CACHE_ENV = "STARLING_SERVE_CACHE";

/**
 * Characters allowed in a single cache path component. Anything else —
 * notably `/` and `\` — could escape the cache root when the value comes
 * from the environment (`STARLING_SERVE_REPO`, `STARLING_SERVE_RELEASE`).
 * `+` stays allowed so semver build-metadata tags (`v0.2.0+build.1`) work.
 */
const SAFE_COMPONENT = /^[A-Za-z0-9._+-]+$/;

export class InvalidCacheComponentError extends Error {
  constructor(
    readonly component: string,
    readonly value: string,
  ) {
    super(
      `Invalid ${component} ${JSON.stringify(value)} for the serve cache: ` +
        `path components may only contain letters, digits, ".", "_", "-", and "+" ` +
        `so values from the environment cannot escape the cache directory.`,
    );
    this.name = "InvalidCacheComponentError";
  }
}

/**
 * Throw unless `value` is safe to embed as one path level below the cache
 * root. Exported for unit tests; everyone else should go through
 * {@link releaseCachePath}.
 */
export function assertCacheComponent(component: string, value: string): void {
  // `.` and `..` pass the character class but would collapse the layout, and
  // `\` is a separator on Windows, so they are rejected explicitly; the empty
  // string fails the class itself. With separators and dot-segments gone, a
  // component can never escape the cache root through `join`.
  if (value === "." || value === ".." || value.includes("\\") || !SAFE_COMPONENT.test(value)) {
    throw new InvalidCacheComponentError(component, value);
  }
}

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

/**
 * Layout inside the cache root, namespaced by repository provenance:
 * `<root>/releases/<owner>/<repo>/<tag>/<binary>`. The repository namespace
 * keeps an alternate `STARLING_SERVE_REPO` source from ever shadowing the
 * default release's binary (or vice versa) when tags and asset names match.
 * `repo` is the normalized `owner/name` coordinate (see
 * {@link normalizeRepo} in `release.ts`).
 *
 * Every component is validated so values from the environment or config
 * cannot escape the cache root; see {@link assertCacheComponent}.
 */
export function releaseCachePath(root: string, repo: string, tag: string, binary: string): string {
  const [owner, name, ...extra] = repo.split("/");

  if (owner === undefined || name === undefined || extra.length > 0) {
    throw new InvalidCacheComponentError("repository", repo);
  }

  assertCacheComponent("repository owner", owner);
  assertCacheComponent("repository name", name);
  assertCacheComponent("release tag", tag);
  assertCacheComponent("binary name", binary);

  return join(root, "releases", owner, name, tag, binary);
}

/** Marker written next to a verified binary: `<binary>.verified`. */
export function verifiedMarkerPath(binaryPath: string): string {
  return `${binaryPath}.verified`;
}
