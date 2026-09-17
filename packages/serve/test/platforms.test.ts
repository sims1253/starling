import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";
import {
  defaultBackend,
  resolveArtifact,
  supportedBackends,
  UnsupportedBackendError,
  UnsupportedPlatformError,
} from "../src/platforms.js";
import { normalizeTag, releaseAssetUrl, releaseTag } from "../src/release.js";

describe("platform + backend → artifact mapping", () => {
  it("maps every published linux/x64 artifact", () => {
    assert.deepEqual(resolveArtifact("linux", "x64", "cpu"), {
      archive: "starling-serve-linux-cpu.tar.gz",
      binary: "starling-serve-linux-cpu",
      checksum: "starling-serve-linux-cpu.sha256",
      archiveExt: ".tar.gz",
    });
    assert.deepEqual(resolveArtifact("linux", "x64", "vulkan"), {
      archive: "starling-serve-linux-vulkan.tar.gz",
      binary: "starling-serve-linux-vulkan",
      checksum: "starling-serve-linux-vulkan.sha256",
      archiveExt: ".tar.gz",
    });
    assert.deepEqual(resolveArtifact("linux", "x64", "cuda"), {
      archive: "starling-serve-linux-cuda.tar.gz",
      binary: "starling-serve-linux-cuda",
      checksum: "starling-serve-linux-cuda.sha256",
      archiveExt: ".tar.gz",
    });
    assert.deepEqual(resolveArtifact("linux", "x64", "rocm"), {
      archive: "starling-serve-linux-rocm.tar.gz",
      binary: "starling-serve-linux-rocm",
      checksum: "starling-serve-linux-rocm.sha256",
      archiveExt: ".tar.gz",
    });
  });

  it("maps win32 artifacts to zip archives with .exe members", () => {
    for (const backend of ["cpu", "vulkan", "cuda"] as const) {
      const spec = resolveArtifact("win32", "x64", backend);
      assert.equal(spec.archive, `starling-serve-windows-${backend}.zip`);
      assert.equal(spec.binary, `starling-serve-windows-${backend}.exe`);
      assert.equal(spec.archiveExt, ".zip");
    }
  });

  it("names the in-archive checksum member without the Windows .exe suffix", () => {
    // The workflow ships starling-serve-windows-cpu.sha256 next to
    // starling-serve-windows-cpu.exe; POSIX stems are unchanged.
    assert.equal(
      resolveArtifact("win32", "x64", "cpu").checksum,
      "starling-serve-windows-cpu.sha256",
    );
    assert.equal(
      resolveArtifact("linux", "x64", "cpu").checksum,
      "starling-serve-linux-cpu.sha256",
    );
  });

  it("maps darwin/arm64 to the macos-named cpu and metal tarballs", () => {
    assert.deepEqual(resolveArtifact("darwin", "arm64", "cpu"), {
      archive: "starling-serve-macos-cpu.tar.gz",
      binary: "starling-serve-macos-cpu",
      checksum: "starling-serve-macos-cpu.sha256",
      archiveExt: ".tar.gz",
    });
    assert.deepEqual(resolveArtifact("darwin", "arm64", "metal"), {
      archive: "starling-serve-macos-metal.tar.gz",
      binary: "starling-serve-macos-metal",
      checksum: "starling-serve-macos-metal.sha256",
      archiveExt: ".tar.gz",
    });
  });

  it("rejects platforms without artifacts with an actionable message", () => {
    assert.throws(
      () => resolveArtifact("darwin", "x64", "cpu"),
      (cause) =>
        cause instanceof UnsupportedPlatformError &&
        /darwin\/x64/.test(cause.message) &&
        /darwin\/arm64/.test(cause.message) &&
        /build a local executable from source/.test(cause.message),
    );
    assert.throws(() => resolveArtifact("linux", "arm64", "cpu"), UnsupportedPlatformError);
    assert.throws(() => resolveArtifact("freebsd", "x64", "cpu"), UnsupportedPlatformError);
  });

  it("rejects backends without an artifact for the platform", () => {
    assert.throws(
      () => resolveArtifact("win32", "x64", "metal"),
      (cause) =>
        cause instanceof UnsupportedBackendError &&
        /no metal artifact for win32\/x64/.test(cause.message) &&
        /cpu, vulkan, cuda/.test(cause.message),
    );
    assert.throws(() => resolveArtifact("darwin", "arm64", "rocm"), UnsupportedBackendError);
  });

  it("lists supported backends per platform", () => {
    assert.deepEqual(supportedBackends("linux", "x64"), ["cpu", "vulkan", "cuda", "rocm"]);
    assert.deepEqual(supportedBackends("win32", "x64"), ["cpu", "vulkan", "cuda"]);
    assert.deepEqual(supportedBackends("darwin", "arm64"), ["cpu", "metal"]);
    assert.deepEqual(supportedBackends("sunos", "x64"), []);
  });
});

describe("default backend selection", () => {
  it("prefers metal on Apple Silicon (Metal ships with macOS)", () => {
    assert.equal(defaultBackend("darwin", "arm64"), "metal");
  });

  it("uses vulkan on Linux only when the loader is present", () => {
    assert.equal(
      defaultBackend("linux", "x64", () => true),
      "vulkan",
    );
    assert.equal(
      defaultBackend("linux", "x64", () => false),
      "cpu",
    );
    assert.equal(defaultBackend("linux", "x64"), "cpu");
  });

  it("defaults to cpu on Windows (no Vulkan loader or CUDA runtime is guaranteed)", () => {
    assert.equal(
      defaultBackend("win32", "x64", () => true),
      "cpu",
    );
  });
});

describe("release coordinates", () => {
  it("maps package versions to v-prefixed release tags", () => {
    assert.equal(normalizeTag("0.2.0"), "v0.2.0");
    assert.equal(normalizeTag("v0.2.0"), "v0.2.0");
    assert.equal(releaseTag("0.2.0"), "v0.2.0");
    assert.equal(releaseTag("0.2.0", "v0.1.0-test"), "v0.1.0-test");
    assert.equal(releaseTag("0.2.0", "1.2.3"), "v1.2.3");
  });

  it("builds GitHub release download URLs", () => {
    assert.equal(
      releaseAssetUrl("sims1253/starling", "v0.2.0", "starling-serve-linux-cpu.tar.gz"),
      "https://github.com/sims1253/starling/releases/download/v0.2.0/starling-serve-linux-cpu.tar.gz",
    );
  });
});
