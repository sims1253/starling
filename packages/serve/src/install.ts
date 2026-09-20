/**
 * First-run installer for starling-serve release binaries.
 *
 * Deliberately NOT a postinstall script: the download happens the first time
 * the `starling-serve` bin runs. Install-time downloads break offline installs,
 * poison CI logs with surprise network I/O, and get silently swallowed by
 * package managers that skip lifecycle scripts for security; first-run keeps
 * `npm install` hermetic and puts the one-time cost where the user asked for
 * the server anyway.
 *
 * Every downloaded artifact is verified twice before execution: the archive
 * hash against the release's `SHA256SUMS.txt`, and the extracted executable
 * against the `.sha256` file shipped inside the archive. The verified hash is
 * then written as a marker next to the cached binary and re-checked on every
 * launch, so on-disk corruption is caught before exec.
 */
import { createWriteStream } from "node:fs";
import { chmod, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import type { ReadableStream as NodeWebReadableStream } from "node:stream/web";
import { pipeline } from "node:stream/promises";
import { cacheDir as resolveCacheDir, releaseCachePath, verifiedMarkerPath } from "./cache.js";
import { defaultExecFile, extractBinary, type ExecFileFn } from "./archive.js";
import { assertChecksum, parseChecksum, sha256File } from "./checksum.js";
import { detectVulkanLoader } from "./detect.js";
import { defaultBackend, resolveArtifact, type Backend } from "./platforms.js";
import { DEFAULT_REPO, normalizeRepo, releaseAssetUrl, releaseTag } from "./release.js";

export class ReleaseAssetError extends Error {
  constructor(
    readonly asset: string,
    readonly tag: string,
    readonly repo: string,
  ) {
    super(
      `Release ${tag} of ${repo} has no ${asset} asset. ` +
        "The release may predate this artifact; pick a backend that exists in it, " +
        "or set STARLING_SERVE_RELEASE to a tag that ships the artifact.",
    );
    this.name = "ReleaseAssetError";
  }
}

export class DownloadError extends Error {
  constructor(
    readonly url: string,
    readonly reason: string,
    readonly cacheDirPath: string,
  ) {
    super(
      `Could not download ${url}: ${reason}. ` +
        "If you are offline, run once on a machine with network access to populate " +
        `the cache, or place a verified binary there manually. Cache directory: ${cacheDirPath} ` +
        "(override with STARLING_SERVE_CACHE).",
    );
    this.name = "DownloadError";
  }
}

export interface EnsureOptions {
  /** Package version, e.g. "0.2.0" → release tag "v0.2.0". */
  version: string;
  /** Defaults to `process.platform`. */
  readonly os?: string;
  /** Defaults to `process.arch`. */
  readonly arch?: string;
  /** Backend override; defaults to the platform default (see platforms.ts). */
  readonly backend?: Backend;
  /** GitHub `owner/name`; defaults to sims1253/starling. */
  readonly repo?: string;
  /** Release tag override; defaults to `v${version}`. */
  readonly releaseTag?: string;
  /** Cache root override; defaults to the OS cache dir / STARLING_SERVE_CACHE. */
  readonly cacheDir?: string;
  /** Fetch implementation override for tests. */
  readonly fetchImpl?: typeof fetch;
  /** Vulkan-loader probe override for tests. */
  readonly hasVulkanLoader?: () => boolean;
  /** tar executor override for tests. */
  readonly execFile?: ExecFileFn;
  /** Progress sink (stderr in the CLI); defaults to no-op. */
  readonly log?: (message: string) => void;
}

export interface EnsureResult {
  /** Absolute path of the verified executable. */
  readonly binaryPath: string;
  readonly backend: Backend;
  readonly tag: string;
  /** Normalized `owner/name` of the repository the binary was verified from. */
  readonly repo: string;
  /** True when this call downloaded and verified the binary. */
  readonly downloaded: boolean;
}

/**
 * Resolve the platform's binary, downloading and verifying it on first use.
 * Returns the executable path; never executes it.
 */
export async function ensureBinary(options: EnsureOptions): Promise<EnsureResult> {
  const os = options.os ?? process.platform;
  const arch = options.arch ?? process.arch;

  const backend =
    options.backend ?? defaultBackend(os, arch, options.hasVulkanLoader ?? detectVulkanLoader);

  const spec = resolveArtifact(os, arch, backend);
  // Normalize early: the normalized coordinate is the cache identity, so
  // case-differing spellings of the same repository share one entry and an
  // invalid coordinate throws before any download is attempted.
  const repo = normalizeRepo(options.repo ?? DEFAULT_REPO);
  const tag = releaseTag(options.version, options.releaseTag);
  const root = options.cacheDir ?? resolveCacheDir();
  const binaryPath = releaseCachePath(root, repo, tag, spec.binary);
  const log = options.log ?? (() => {});

  if (await cachedBinaryMatchesMarker(binaryPath, repo, tag, spec.binary)) {
    log(`starling-serve ${tag} (${os}-${backend}) found in cache: ${binaryPath}`);

    return { binaryPath, backend, tag, repo, downloaded: false };
  }

  log(`Downloading starling-serve ${tag} (${os}-${backend}) from ${repo}...`);
  await mkdir(dirname(binaryPath), { recursive: true });
  // Stage inside the release directory, not os.tmpdir(): the verified binary
  // is rename(2)d to binaryPath, and a cross-device rename fails with EXDEV
  // after the whole download already succeeded — /tmp is a tmpfs on several
  // distros, and STARLING_SERVE_CACHE may live on another volume entirely.
  // The dotted mkdtemp name can never collide with the cached binary itself,
  // so an interrupted download just re-runs; the finally below cleans up.
  const workDir = await mkdtemp(join(dirname(binaryPath), ".starling-serve-"));

  try {
    const fetchImpl = options.fetchImpl ?? fetch;
    const archivePath = join(workDir, spec.archive);
    await downloadToFile(fetchImpl, repo, tag, spec.archive, archivePath, root);
    const sums = await downloadText(fetchImpl, repo, tag, "SHA256SUMS.txt", root);
    assertChecksum(spec.archive, parseChecksum(sums, spec.archive), await sha256File(archivePath));

    const extracted = await extractBinary(
      archivePath,
      spec,
      workDir,
      options.execFile ?? defaultExecFile,
    );

    const innerChecksum = parseChecksum(await readFile(extracted.checksumPath, "utf8"));
    assertChecksum(spec.binary, innerChecksum, await sha256File(extracted.binaryPath));

    if (os !== "win32") {
      await chmod(extracted.binaryPath, 0o755);
    }

    await rename(extracted.binaryPath, binaryPath);
    await writeFile(
      verifiedMarkerPath(binaryPath),
      `repo ${repo}\ntag ${tag}\nbinary ${spec.binary}\nsha256 ${innerChecksum}\n`,
      "utf8",
    );

    return { binaryPath, backend, tag, repo, downloaded: true };
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
}

/**
 * A cached binary is usable when its marker carries the requested artifact
 * identity (`repo`, `tag`, `binary`) and its checksum matches a fresh
 * re-hash. A legacy marker (a bare checksum line from before provenance was
 * recorded) has unknowable origin, so it is treated as a miss: the binary is
 * re-downloaded and re-verified, replacing the marker with one that carries
 * the full identity.
 */
async function cachedBinaryMatchesMarker(
  binaryPath: string,
  repo: string,
  tag: string,
  binary: string,
): Promise<boolean> {
  let marker: string;

  try {
    marker = await readFile(verifiedMarkerPath(binaryPath), "utf8");
  } catch {
    return false;
  }

  const fields = parseMarker(marker);

  if (
    fields === undefined ||
    fields.repo !== repo ||
    fields.tag !== tag ||
    fields.binary !== binary
  ) {
    return false;
  }

  try {
    return (await sha256File(binaryPath)) === fields.sha256;
  } catch {
    return false;
  }
}

/** Parsed verification-marker fields, or `undefined` for a legacy marker. */
function parseMarker(
  marker: string,
): { repo: string; tag: string; binary: string; sha256: string } | undefined {
  const fields = new Map<string, string>();

  for (const rawLine of marker.split("\n")) {
    const line = rawLine.trim();

    if (line === "") continue;
    const space = line.indexOf(" ");

    if (space <= 0) return undefined;
    fields.set(line.slice(0, space), line.slice(space + 1));
  }

  const repo = fields.get("repo");
  const tag = fields.get("tag");
  const binary = fields.get("binary");
  const sha256 = fields.get("sha256");

  if (repo === undefined || tag === undefined || binary === undefined || sha256 === undefined) {
    return undefined;
  }

  return { repo, tag, binary, sha256 };
}

async function downloadToFile(
  fetchImpl: typeof fetch,
  repo: string,
  tag: string,
  asset: string,
  dest: string,
  root: string,
): Promise<void> {
  const url = releaseAssetUrl(repo, tag, asset);
  let response: Response;

  try {
    response = await fetchImpl(url);
  } catch (error) {
    const reason = error instanceof Error ? describeError(error) : String(error);
    throw new DownloadError(url, reason, root);
  }

  if (response.status === 404) {
    throw new ReleaseAssetError(asset, tag, repo);
  }

  if (!response.ok || response.body === null) {
    throw new DownloadError(url, `HTTP ${response.status}`, root);
  }

  try {
    // SAFETY: Node's fetch returns the same WHATWG stream that node:stream/web
    // types describe; the DOM lib types structurally match, so this widens
    // only the type-level name, not the runtime value.
    const body = Readable.fromWeb(response.body as NodeWebReadableStream<Uint8Array>);
    await pipeline(body, createWriteStream(dest));
  } catch (error) {
    const reason = error instanceof Error ? describeError(error) : String(error);

    throw new DownloadError(url, reason, root);
  }
}

async function downloadText(
  fetchImpl: typeof fetch,
  repo: string,
  tag: string,
  asset: string,
  root: string,
): Promise<string> {
  const url = releaseAssetUrl(repo, tag, asset);
  let response: Response;

  try {
    response = await fetchImpl(url);
  } catch (error) {
    const reason = error instanceof Error ? describeError(error) : String(error);
    throw new DownloadError(url, reason, root);
  }

  if (response.status === 404) {
    throw new ReleaseAssetError(asset, tag, repo);
  }

  if (!response.ok) {
    throw new DownloadError(url, `HTTP ${response.status}`, root);
  }

  return response.text();
}

function describeError(error: Error): string {
  const causeMessage = error.cause instanceof Error ? ` (${error.cause.message})` : "";

  return `${error.message}${causeMessage}`;
}
