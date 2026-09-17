import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, it } from "vite-plus/test";
import {
  parseWrapperArgs,
  runCli,
  WrapperUsageError,
  type SpawnFn,
  type SpawnedProcess,
} from "../src/cli.js";
import { buildReleaseFixture, fakeFetch } from "./fixtures.js";

const cleanups: string[] = [];

afterEach(async () => {
  while (cleanups.length > 0) {
    const dir = cleanups.pop();

    if (dir !== undefined) {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

/** The cpu artifact for the host platform (all CI matrices have one). */
function hostCpuArtifact() {
  const os = process.platform === "win32" ? "windows" : process.platform;

  const archiveName =
    process.platform === "win32"
      ? `starling-serve-${os}-cpu.zip`
      : `starling-serve-${os}-cpu.tar.gz`;

  const stem = process.platform === "win32" ? archiveName.slice(0, -4) : archiveName.slice(0, -7);
  const binaryName = process.platform === "win32" ? `${stem}.exe` : stem;

  return { archiveName, binaryName };
}

class FakeChild extends EventEmitter implements SpawnedProcess {
  kill(): boolean {
    return true;
  }
}

describe("wrapper argument parsing", () => {
  it("forwards everything untouched when no wrapper flags are present", () => {
    const parsed = parseWrapperArgs([
      "--model",
      "parakeet",
      "--gguf",
      "model.gguf",
      "--port",
      "8181",
      "--warmup",
    ]);

    assert.equal(parsed.backend, undefined);
    assert.deepEqual(parsed.forward, [
      "--model",
      "parakeet",
      "--gguf",
      "model.gguf",
      "--port",
      "8181",
      "--warmup",
    ]);
  });

  it("consumes --starling-backend in space and equals forms", () => {
    assert.deepEqual(parseWrapperArgs(["--starling-backend", "cuda", "--version"]), {
      backend: "cuda",
      forward: ["--version"],
    });
    assert.deepEqual(parseWrapperArgs(["--version", "--starling-backend=vulkan"]), {
      backend: "vulkan",
      forward: ["--version"],
    });
  });

  it("keeps forwarded argument order across wrapper flags", () => {
    const parsed = parseWrapperArgs(["a", "--starling-backend", "cpu", "b"]);

    assert.deepEqual(parsed.forward, ["a", "b"]);
  });

  it("rejects unknown --starling-* flags instead of forwarding them", () => {
    assert.throws(
      () => parseWrapperArgs(["--starling-bakcend", "cpu"]),
      (cause) =>
        cause instanceof WrapperUsageError &&
        /Unknown wrapper flag --starling-bakcend/.test(cause.message),
    );
  });

  it("rejects --starling-backend without a value or with an unknown backend", () => {
    assert.throws(() => parseWrapperArgs(["--starling-backend"]), WrapperUsageError);
    assert.throws(() => parseWrapperArgs(["--starling-backend="]), WrapperUsageError);
    assert.throws(
      () => parseWrapperArgs(["--starling-backend", "tensor"]),
      /Unknown backend tensor/,
    );
  });

  it("does not treat unrelated dashed flags as wrapper flags", () => {
    const parsed = parseWrapperArgs(["--backendish", "--starling"]);

    assert.equal(parsed.backend, undefined);
    assert.deepEqual(parsed.forward, ["--backendish", "--starling"]);
  });
});

function cliHarness() {
  const { archiveName, binaryName } = hostCpuArtifact();

  const fixture = buildReleaseFixture({
    repo: "sims1253/starling",
    tag: "v0.2.0",
    archiveName,
    binaryName,
  });

  return { fixture, binaryName };
}

describe("runCli", () => {
  it("installs, then execs the resolved binary with forwarded args", async () => {
    const cacheDir = await mkdtemp(join(tmpdir(), "starling-cli-"));
    cleanups.push(cacheDir);
    const { fixture, binaryName } = cliHarness();

    const spawned: { file: string; args: readonly string[] }[] = [];
    const exitCodes: number[] = [];
    const logs: string[] = [];
    const children: FakeChild[] = [];

    const spawnFn: SpawnFn = (file, args) => {
      spawned.push({ file, args });
      const child = new FakeChild();
      children.push(child);

      return child;
    };

    await runCli({
      argv: ["--starling-backend", "cpu", "--model", "parakeet", "--port", "8181"],
      env: { STARLING_SERVE_CACHE: cacheDir, STARLING_SERVE_BACKEND: "vulkan" },
      version: "0.2.0",
      spawnFn,
      exit: (code) => exitCodes.push(code),
      log: (message) => logs.push(message),
      fetchImpl: fakeFetch(fixture.assets),
    });

    const child = children[0];

    assert.ok(child);
    assert.equal(spawned.length, 1);
    assert.equal(spawned[0]?.file, join(cacheDir, "releases", "v0.2.0", binaryName));
    assert.deepEqual(spawned[0]?.args, ["--model", "parakeet", "--port", "8181"]);
    // The CLI flag wins over STARLING_SERVE_BACKEND.
    assert.match(logs.join("\n"), /cpu/);

    child.emit("close", 7, null);
    assert.deepEqual(exitCodes, [7]);
  });

  it("propagates death-by-signal as the conventional 128+n code", async () => {
    const cacheDir = await mkdtemp(join(tmpdir(), "starling-cli-"));
    cleanups.push(cacheDir);
    const { fixture } = cliHarness();
    const exitCodes: number[] = [];
    const child = new FakeChild();

    await runCli({
      argv: [],
      env: { STARLING_SERVE_CACHE: cacheDir, STARLING_SERVE_BACKEND: "cpu" },
      version: "0.2.0",
      spawnFn: () => child,
      exit: (code) => exitCodes.push(code),
      log: () => {},
      fetchImpl: fakeFetch(fixture.assets),
    });

    child.emit("close", null, "SIGTERM");
    assert.deepEqual(exitCodes, [143]);
  });

  it("exits 1 with the installer's message when the download fails", async () => {
    const cacheDir = await mkdtemp(join(tmpdir(), "starling-cli-"));
    cleanups.push(cacheDir);

    const offline: typeof fetch = async () => {
      throw new TypeError("fetch failed");
    };

    const exitCodes: number[] = [];
    const logs: string[] = [];

    await runCli({
      argv: ["--version"],
      env: { STARLING_SERVE_CACHE: cacheDir, STARLING_SERVE_BACKEND: "cpu" },
      version: "0.2.0",
      spawnFn: () => new FakeChild(),
      exit: (code) => exitCodes.push(code),
      log: (message) => logs.push(message),
      fetchImpl: offline,
    });

    assert.deepEqual(exitCodes, [1]);
    assert.match(logs.join("\n"), /starling-serve: Could not download/);
  });

  it("exits 2 on a wrapper usage error without touching the network", async () => {
    const exitCodes: number[] = [];
    const logs: string[] = [];
    let spawnCalls = 0;

    const spawnFn: SpawnFn = () => {
      spawnCalls += 1;

      return new FakeChild();
    };

    await runCli({
      argv: ["--starling-bakcend", "cpu"],
      env: {},
      version: "0.2.0",
      spawnFn,
      exit: (code) => exitCodes.push(code),
      log: (message) => logs.push(message),
    });

    assert.deepEqual(exitCodes, [2]);
    assert.equal(spawnCalls, 0);
    assert.match(logs.join("\n"), /Unknown wrapper flag/);
  });

  it("exits 1 when the exec itself fails", async () => {
    const cacheDir = await mkdtemp(join(tmpdir(), "starling-cli-"));
    cleanups.push(cacheDir);
    const { fixture } = cliHarness();
    const exitCodes: number[] = [];
    const child = new FakeChild();

    await runCli({
      argv: [],
      env: { STARLING_SERVE_CACHE: cacheDir, STARLING_SERVE_BACKEND: "cpu" },
      version: "0.2.0",
      spawnFn: () => child,
      exit: (code) => exitCodes.push(code),
      log: () => {},
      fetchImpl: fakeFetch(fixture.assets),
    });

    child.emit("error", new Error("EACCES: permission denied"));
    assert.deepEqual(exitCodes, [1]);
  });
});
