/**
 * Platform, backend, and release-asset mapping for the starling-serve launcher.
 *
 * This module is pure: every function is a table lookup or a validation so the
 * os/arch/backend → asset mapping can be unit tested without touching the
 * network or the filesystem. Anything that inspects the machine lives in
 * {@link ./detect.ts} and is injected as an option.
 */

export type Os = "linux" | "win32" | "darwin";

export type Arch = "x64" | "arm64";

export const BACKEND_NAMES = ["cpu", "vulkan", "cuda", "rocm", "metal"] as const;

export type Backend = (typeof BACKEND_NAMES)[number];

const BACKEND_SET: ReadonlySet<string> = new Set(BACKEND_NAMES);

/** Narrow an unvalidated string (CLI flag, environment variable) to a Backend. */
export function isBackend(value: string): value is Backend {
  return BACKEND_SET.has(value);
}

/** Everything the launcher needs to download and run one release artifact. */
export interface ArtifactSpec {
  /** Release asset file name, e.g. `starling-serve-linux-cpu.tar.gz`. */
  readonly archive: string;
  /** Executable member inside the archive, e.g. `starling-serve-linux-cpu.exe`. */
  readonly binary: string;
  /** Archive container format; Windows releases are zip, everything else tar.gz. */
  readonly archiveExt: ".tar.gz" | ".zip";
}

export class UnsupportedPlatformError extends Error {
  constructor(
    readonly os: string,
    readonly arch: string,
  ) {
    super(
      `starling-serve has no release artifacts for ${os}/${arch}. ` +
        "Published builds: linux/x64, win32/x64, and darwin/arm64. " +
        "See https://github.com/sims1253/starling/blob/master/docs/native-serving.md#build " +
        "to build a local executable from source.",
    );
    this.name = "UnsupportedPlatformError";
  }
}

export class UnsupportedBackendError extends Error {
  constructor(
    readonly os: string,
    readonly arch: string,
    readonly backend: string,
    readonly supported: readonly Backend[],
  ) {
    super(
      `starling-serve has no ${backend} artifact for ${os}/${arch}. ` +
        `Available backends: ${supported.join(", ")}.`,
    );
    this.name = "UnsupportedBackendError";
  }
}

/** Backends with a published artifact, keyed by os/arch. */
export const SUPPORTED = {
  "linux/x64": ["cpu", "vulkan", "cuda", "rocm"],
  "win32/x64": ["cpu", "vulkan", "cuda"],
  "darwin/arm64": ["cpu", "metal"],
} as const satisfies Record<string, readonly Backend[]>;

export function supportedBackends(os: string, arch: string): readonly Backend[] {
  if (os === "linux" && arch === "x64") return SUPPORTED["linux/x64"];

  if (os === "win32" && arch === "x64") return SUPPORTED["win32/x64"];

  if (os === "darwin" && arch === "arm64") return SUPPORTED["darwin/arm64"];

  return [];
}

export function resolveArtifact(os: string, arch: string, backend: Backend): ArtifactSpec {
  const supported = supportedBackends(os, arch);

  if (supported.length === 0) {
    throw new UnsupportedPlatformError(os, arch);
  }

  if (!supported.includes(backend)) {
    throw new UnsupportedBackendError(os, arch, backend, supported);
  }

  const suffix = `${os === "win32" ? "windows" : os}-${backend}`;
  const archiveExt = os === "win32" ? (".zip" as const) : (".tar.gz" as const);
  const binary = os === "win32" ? `${baseName(suffix)}.exe` : baseName(suffix);

  return { archive: `${baseName(suffix)}${archiveExt}`, binary, archiveExt };
}

function baseName(suffix: string): string {
  return `starling-serve-${suffix}`;
}

/**
 * Default backend for a platform.
 *
 * The rules come from docs/release-runtime.md:
 * - darwin/arm64 → metal: Metal ships with macOS 14+, so the metal artifact
 *   has no extra runtime prerequisites and is the fast path on Apple Silicon.
 * - linux/x64 → vulkan when the Vulkan loader is discoverable (cross-vendor
 *   GPU acceleration), cpu otherwise: the vulkan artifact additionally needs
 *   `libvulkan1` plus a vendor driver, which servers and containers often lack.
 * - win32/x64 → cpu: stock Windows has no Vulkan loader or CUDA runtime, so
 *   cpu is the only artifact guaranteed to start; opt in to vulkan/cuda with
 *   STARLING_SERVE_BACKEND or --starling-backend.
 */
export function defaultBackend(
  os: string,
  arch: string,
  hasVulkanLoader: () => boolean = () => false,
): Backend {
  if (os === "darwin") return "metal";

  if (os === "linux") return hasVulkanLoader() ? "vulkan" : "cpu";

  return "cpu";
}
