import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";
import { afterEach, describe, it } from "vite-plus/test";
import { resolveTarExecutable, tarArgs, type ExecFileFn } from "../src/archive.js";
import {
  ChecksumMismatchError,
  DownloadError,
  ensureBinary,
  ReleaseAssetError,
  verifiedMarkerPath,
} from "../src/index.js";
import { buildReleaseFixture, fakeFetch, type ReleaseFixture } from "./fixtures.js";

const REPO = "sims1253/starling";

const TAG = "v0.2.0";

const ARCHIVE = "starling-serve-linux-cpu.tar.gz";

const BINARY = "starling-serve-linux-cpu";

const cleanups: string[] = [];

afterEach(async () => {
  while (cleanups.length > 0) {
    const dir = cleanups.pop();

    if (dir !== undefined) {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

interface Harness {
  cacheDir: string;
  fixture: ReleaseFixture;
  fetches: string[];
  fetchImpl: typeof fetch;
}

async function harness(
  options: Partial<Parameters<typeof buildReleaseFixture>[0]> = {},
): Promise<Harness> {
  const cacheDir = await mkdtemp(join(tmpdir(), "starling-install-"));
  cleanups.push(cacheDir);

  const fixture = buildReleaseFixture({
    repo: REPO,
    tag: TAG,
    archiveName: ARCHIVE,
    binaryName: BINARY,
    ...options,
  });

  const fetches: string[] = [];
  const serving = fakeFetch(fixture.assets);

  const fetchImpl: typeof fetch = async (input, init) => {
    fetches.push(String(input));

    return serving(input, init);
  };

  return { cacheDir, fixture, fetches, fetchImpl };
}

function ensure(h: Harness, overrides: Partial<Parameters<typeof ensureBinary>[0]> = {}) {
  return ensureBinary({
    version: "0.2.0",
    os: "linux",
    arch: "x64",
    backend: "cpu",
    cacheDir: h.cacheDir,
    fetchImpl: h.fetchImpl,
    ...overrides,
  });
}

function offlineFetch(): typeof fetch {
  const offline: typeof fetch = async () => {
    throw new TypeError("fetch failed");
  };

  return offline;
}

describe("ensureBinary first-run install", () => {
  it("downloads, double-verifies, extracts, and caches the binary", async () => {
    const h = await harness();
    const result = await ensure(h);
    const cached = join(h.cacheDir, "releases", TAG, BINARY);
    const marker = (await readFile(verifiedMarkerPath(cached), "utf8")).trim();

    assert.equal(result.downloaded, true);
    assert.equal(result.tag, "v0.2.0");
    assert.equal(result.backend, "cpu");
    assert.equal(result.binaryPath, cached);
    assert.deepEqual(await readFile(cached), h.fixture.binaryContent);
    assert.match(marker, /^[0-9a-f]{64}$/);
    // Archive + SHA256SUMS.txt were both fetched from the right release.
    assert.deepEqual(h.fetches.sort(), [
      `https://github.com/${REPO}/releases/download/${TAG}/SHA256SUMS.txt`,
      `https://github.com/${REPO}/releases/download/${TAG}/${ARCHIVE}`,
    ]);
  });

  it("extracts only the executable and its checksum by member name", async () => {
    const invocations: { file: string; args: string[] }[] = [];

    const execFile: ExecFileFn = async (file, args) => {
      invocations.push({ file, args: [...args] });

      return { stdout: "", stderr: "" };
    };

    const h = await harness();

    // The stub does not extract anything, so reading the extracted checksum
    // fails afterwards — but the invocation is what this test asserts.
    await assert.rejects(
      ensure(h, { execFile }),
      (cause) => cause instanceof Error && /ENOENT/.test(cause.message),
    );

    const invocation = invocations[0];
    assert.equal(invocations.length, 1);
    assert.ok(invocation);
    assert.equal(invocation.file, resolveTarExecutable());
    assert.equal(invocation.args[0], "-xzf");
    assert.ok(invocation.args.indexOf("-C") > 0);
    // Staging happens inside the cache's release directory, so the final move
    // is a same-filesystem rename even when os.tmpdir() is another device.
    const destDir = invocation.args[invocation.args.indexOf("-C") + 1];
    assert.ok(
      destDir !== undefined && destDir.startsWith(join(h.cacheDir, "releases", TAG) + sep),
      `staging dir ${destDir} is not inside ${join(h.cacheDir, "releases", TAG)}`,
    );
    assert.deepEqual(invocation.args.slice(invocation.args.indexOf("-C") + 2), [
      BINARY,
      `${BINARY}.sha256`,
    ]);
  });

  it("stages next to the cached binary and removes the staging dir on failure", async () => {
    const h = await harness({
      tamperArchive: (archive) => Buffer.concat([archive, Buffer.from("x")]),
    });

    await assert.rejects(ensure(h), (cause) => cause instanceof ChecksumMismatchError);

    // The finally cleanup removed the staging dir; nothing that a later run
    // could mistake for a verified cache entry is left behind.
    assert.deepEqual(await readdir(join(h.cacheDir, "releases", TAG)), []);
  });

  it("leaves only the binary and its marker after a successful install", async () => {
    const h = await harness();
    await ensure(h);

    assert.deepEqual((await readdir(join(h.cacheDir, "releases", TAG))).sort(), [
      BINARY,
      `${BINARY}.verified`,
    ]);
  });

  it("reuses the cached binary without any network access", async () => {
    const h = await harness();
    await ensure(h);
    const firstFetches = h.fetches.length;
    const second = await ensure(h);

    assert.equal(second.downloaded, false);
    assert.equal(second.binaryPath, join(h.cacheDir, "releases", TAG, BINARY));
    assert.equal(h.fetches.length, firstFetches);
  });

  it("re-downloads when the cached binary no longer matches its marker", async () => {
    const h = await harness();
    const first = await ensure(h);
    await writeFile(first.binaryPath, Buffer.from("corrupted payload"));
    const second = await ensure(h);

    assert.equal(first.downloaded, true);
    assert.equal(second.downloaded, true);
    assert.deepEqual(await readFile(second.binaryPath), h.fixture.binaryContent);
  });

  it("rejects an archive whose hash disagrees with SHA256SUMS.txt", async () => {
    const h = await harness({
      tamperArchive: (archive) => Buffer.concat([archive, Buffer.from("x")]),
    });

    await assert.rejects(
      ensure(h),
      (cause) => cause instanceof ChecksumMismatchError && cause.file === ARCHIVE,
    );
  });

  it("rejects a binary whose hash disagrees with the in-archive checksum", async () => {
    const h = await harness({
      tamperInnerChecksum: () => `${"0".repeat(64)}  ${BINARY}\n`,
    });

    await assert.rejects(
      ensure(h),
      (cause) =>
        cause instanceof ChecksumMismatchError &&
        cause.file === BINARY &&
        /refusing to run/.test(cause.message),
    );
  });

  it("reports a missing release asset with the tag and asset name", async () => {
    const h = await harness();
    h.fixture.assets.delete(`https://github.com/${REPO}/releases/download/${TAG}/${ARCHIVE}`);

    await assert.rejects(
      ensure(h),
      (cause) =>
        cause instanceof ReleaseAssetError &&
        cause.tag === TAG &&
        cause.asset === ARCHIVE &&
        /STARLING_SERVE_RELEASE/.test(cause.message),
    );
  });

  it("reports an actionable error when offline without a cached binary", async () => {
    const h = await harness();

    await assert.rejects(
      ensure(h, { fetchImpl: offlineFetch() }),
      (cause) =>
        cause instanceof DownloadError &&
        /Could not download/.test(cause.message) &&
        /STARLING_SERVE_CACHE/.test(cause.message) &&
        cause.cacheDirPath === h.cacheDir,
    );
  });

  it("serves a valid cache with no network at all", async () => {
    const h = await harness();
    await ensure(h);
    const result = await ensure(h, { fetchImpl: offlineFetch() });

    assert.equal(result.downloaded, false);
  });

  it("honors a release-tag override for custom and test builds", async () => {
    const customTag = "v0.1.0-test";
    const cacheDir = await mkdtemp(join(tmpdir(), "starling-install-"));
    cleanups.push(cacheDir);

    const fixture = buildReleaseFixture({
      repo: REPO,
      tag: customTag,
      archiveName: ARCHIVE,
      binaryName: BINARY,
    });

    const result = await ensureBinary({
      version: "0.2.0",
      os: "linux",
      arch: "x64",
      backend: "cpu",
      releaseTag: customTag,
      cacheDir,
      fetchImpl: fakeFetch(fixture.assets),
    });

    assert.equal(result.tag, customTag);
    assert.equal(result.binaryPath, join(cacheDir, "releases", customTag, BINARY));
  });

  it("caches separate release tags side by side", async () => {
    const h = await harness();
    await ensure(h);
    const otherTag = "v0.3.0";

    const otherFixture = buildReleaseFixture({
      repo: REPO,
      tag: otherTag,
      archiveName: ARCHIVE,
      binaryName: BINARY,
    });

    const result = await ensureBinary({
      version: "0.2.0",
      os: "linux",
      arch: "x64",
      backend: "cpu",
      releaseTag: otherTag,
      cacheDir: h.cacheDir,
      fetchImpl: fakeFetch(otherFixture.assets),
    });

    assert.equal(result.binaryPath, join(h.cacheDir, "releases", otherTag, BINARY));
  });

  it("marks the cached executable as executable on POSIX", async () => {
    const h = await harness();
    const result = await ensure(h);

    if (process.platform !== "win32") {
      assert.equal((await stat(result.binaryPath)).mode & 0o111, 0o111);
    }
  });
});

describe("tar argument construction", () => {
  it("uses gzip flags for tar.gz and plain extract for zip", () => {
    assert.deepEqual(tarArgs("/tmp/a.tar.gz", ["m1", "m2"], "/dest", ".tar.gz"), [
      "-xzf",
      "/tmp/a.tar.gz",
      "-C",
      "/dest",
      "m1",
      "m2",
    ]);
    assert.deepEqual(tarArgs("/tmp/a.zip", ["m.exe"], "/dest", ".zip"), [
      "-xf",
      "/tmp/a.zip",
      "-C",
      "/dest",
      "m.exe",
    ]);
  });
});

describe("tar executable resolution", () => {
  it("uses the PATH lookup on non-Windows platforms", () => {
    assert.equal(resolveTarExecutable("linux"), "tar");
    assert.equal(resolveTarExecutable("darwin"), "tar");
  });

  it("prefers System32 bsdtar on Windows so Git Bash's GNU tar cannot shadow it", () => {
    assert.equal(
      resolveTarExecutable("win32", "C:\\Windows", () => true),
      "C:\\Windows\\System32\\tar.exe",
    );
  });

  it("falls back to the PATH lookup when the System32 bsdtar is absent", () => {
    assert.equal(
      resolveTarExecutable("win32", "C:\\Windows", () => false),
      "tar",
    );
    assert.equal(
      resolveTarExecutable("win32", undefined, () => true),
      "tar",
    );
  });
});
