import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, it } from "vite-plus/test";
import {
  assertChecksum,
  ChecksumFormatError,
  ChecksumMismatchError,
  parseChecksum,
  sha256File,
} from "../src/checksum.js";
import { cacheDir, defaultCacheDir, releaseCachePath, verifiedMarkerPath } from "../src/cache.js";
import { sha256 } from "./fixtures.js";

const cleanups: string[] = [];

afterEach(async () => {
  while (cleanups.length > 0) {
    const dir = cleanups.pop();

    if (dir !== undefined) {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

describe("checksum parsing", () => {
  it("parses a single sha256sum line", () => {
    const digest = "a".repeat(64);
    assert.equal(parseChecksum(`${digest}  starling-serve-linux-cpu\n`), digest);
  });

  it("selects the entry for a named asset from SHA256SUMS.txt", () => {
    const first = "1".repeat(64);
    const second = "2".repeat(64);
    const sums = `${first}  starling-serve-linux-cpu.tar.gz\n${second}  starling-serve-macos-metal.tar.gz\n`;
    assert.equal(parseChecksum(sums, "starling-serve-macos-metal.tar.gz"), second);
    assert.equal(parseChecksum(sums, "starling-serve-linux-cpu.tar.gz"), first);
  });

  it("tolerates CRLF and the GNU binary-mode asterisk", () => {
    const digest = "b".repeat(64);
    assert.equal(parseChecksum(`${digest} *file.bin\r\n`, "file.bin"), digest);
  });

  it("ignores blank and comment lines", () => {
    const digest = "c".repeat(64);
    assert.equal(parseChecksum(`# header\n\n${digest}  a.zip\n`), digest);
  });

  it("rejects malformed lines and missing entries", () => {
    assert.throws(() => parseChecksum("not-a-checksum  file"), ChecksumFormatError);
    assert.throws(() => parseChecksum("zzzz  file"), ChecksumFormatError);
    assert.throws(
      () => parseChecksum(`${"d".repeat(64)}  a.zip\n`, "missing.zip"),
      /no checksum entry for missing.zip/,
    );
    assert.throws(() => parseChecksum(""), /exactly one checksum entry/);
    assert.throws(
      () => parseChecksum(`${"e".repeat(64)}  a.zip\n${"f".repeat(64)}  b.zip\n`),
      /exactly one checksum entry/,
    );
  });
});

describe("file hashing", () => {
  it("hashes file contents with streaming SHA-256", async () => {
    const dir = await mkdtemp(join(tmpdir(), "starling-checksum-"));
    cleanups.push(dir);
    const file = join(dir, "payload.bin");
    await writeFile(file, Buffer.from("starling-serve test payload"));
    assert.equal(await sha256File(file), sha256(Buffer.from("starling-serve test payload")));
    assert.notEqual(await sha256File(file), sha256(Buffer.from("tampered")));
  });

  it("assertChecksum passes on equality and reports mismatches", () => {
    const digest = sha256("x");
    assert.doesNotThrow(() => assertChecksum("file", digest, digest));
    assert.throws(
      () => assertChecksum("file", digest, "0".repeat(64)),
      (cause) =>
        cause instanceof ChecksumMismatchError &&
        cause.file === "file" &&
        cause.expected === digest &&
        cause.actual === "0".repeat(64) &&
        /refusing to run/.test(cause.message),
    );
  });
});

describe("cache directory resolution", () => {
  it("honors STARLING_SERVE_CACHE over the OS default", () => {
    assert.equal(cacheDir({ STARLING_SERVE_CACHE: "/custom/cache" }), "/custom/cache");
    assert.equal(cacheDir({ STARLING_SERVE_CACHE: "  " }), cacheDir({}));
  });

  it("uses OS-conventional roots", () => {
    const darwin = defaultCacheDir("darwin");
    assert.match(darwin, /Library[\\/]Caches[\\/]starling-serve$/);
    const win32 = defaultCacheDir("win32");
    assert.match(win32, /[\\/]starling-serve[\\/]cache$/);
    const linuxXdg = defaultCacheDirWithEnv("linux", { XDG_CACHE_HOME: "/xdg" });
    assert.equal(linuxXdg, "/xdg/starling-serve");
    const linuxDefault = defaultCacheDirWithEnv("linux", {});
    assert.match(linuxDefault, /\.cache[\\/]starling-serve$/);
  });

  it("lays out binaries per release tag and marks verification", () => {
    const path = releaseCachePath("/cache", "v0.2.0", "starling-serve-linux-cpu");
    assert.equal(path, join("/cache", "releases", "v0.2.0", "starling-serve-linux-cpu"));
    assert.equal(verifiedMarkerPath(path), `${path}.verified`);
  });
});

function defaultCacheDirWithEnv(os: string, env: NodeJS.ProcessEnv): string {
  const original = process.env["XDG_CACHE_HOME"];

  if (env["XDG_CACHE_HOME"] === undefined) {
    delete process.env["XDG_CACHE_HOME"];
  } else {
    process.env["XDG_CACHE_HOME"] = env["XDG_CACHE_HOME"];
  }

  try {
    return defaultCacheDir(os);
  } finally {
    if (original === undefined) {
      delete process.env["XDG_CACHE_HOME"];
    } else {
      process.env["XDG_CACHE_HOME"] = original;
    }
  }
}
