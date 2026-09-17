/**
 * Public API of the starling-serve launcher package.
 *
 * Programmatic consumers get the same behavior as the bin: `ensureBinary`
 * resolves, downloads, and verifies the executable for the running platform
 * without executing it.
 */
export {
  defaultBackend,
  resolveArtifact,
  supportedBackends,
  SUPPORTED,
  UnsupportedBackendError,
  UnsupportedPlatformError,
  type Arch,
  type ArtifactSpec,
  type Backend,
  type Os,
} from "./platforms.js";

export { DEFAULT_REPO, normalizeTag, releaseAssetUrl, releaseTag } from "./release.js";

export {
  assertChecksum,
  ChecksumFormatError,
  ChecksumMismatchError,
  parseChecksum,
  sha256File,
} from "./checksum.js";

export { cacheDir, defaultCacheDir, releaseCachePath, verifiedMarkerPath } from "./cache.js";

export {
  DownloadError,
  ensureBinary,
  ReleaseAssetError,
  type EnsureOptions,
  type EnsureResult,
} from "./install.js";

export { defaultExecFile, extractBinary, tarArgs, type ExecFileFn } from "./archive.js";

export {
  packageVersion,
  parseWrapperArgs,
  runCli,
  WrapperUsageError,
  type CliDeps,
  type SpawnFn,
  type WrapperArgs,
} from "./cli.js";

export { detectVulkanLoader, detectVulkanLoaderViaLdconfig } from "./detect.js";
