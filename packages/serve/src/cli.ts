#!/usr/bin/env node
/**
 * `starling-serve` bin: resolve, fetch-if-needed, verify, and exec the native
 * server, forwarding every argument to it.
 *
 * Wrapper-specific flags are namespaced with `--starling-` so they can never
 * collide with the server's own CLI (the server rejects unknown flags):
 *   --starling-backend <cpu|vulkan|cuda|rocm|metal>
 * Environment overrides:
 *   STARLING_SERVE_BACKEND   same as the flag (flag wins)
 *   STARLING_SERVE_RELEASE   release tag instead of the package's v<version>
 *   STARLING_SERVE_REPO      GitHub owner/name instead of sims1253/starling
 *   STARLING_SERVE_CACHE     cache directory override
 */
import { spawn } from "node:child_process";
import { readFileSync, realpathSync } from "node:fs";
import process from "node:process";
import { fileURLToPath } from "node:url";
import type { EventEmitter } from "node:events";
import { ensureBinary } from "./install.js";
import { isBackend, type Backend } from "./platforms.js";

const WRAPPER_FLAG_PREFIX = "--starling-";

/** The process surface the CLI needs from spawn; injectable for tests. */
export interface SpawnedProcess extends EventEmitter {
  kill(signal?: NodeJS.Signals | number): boolean;
}

/** Minimal spawn surface the CLI needs; injectable for tests. */
export type SpawnFn = (
  file: string,
  args: readonly string[],
  options: { stdio: "inherit" },
) => SpawnedProcess;

export interface WrapperArgs {
  readonly backend?: Backend;
  /** Arguments to hand to the server executable, verbatim and in order. */
  readonly forward: readonly string[];
}

export class WrapperUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WrapperUsageError";
  }
}

/**
 * Split argv into wrapper flags and forwarded arguments. Any unrecognized
 * `--starling-*` flag is a hard error: a typo would otherwise be silently
 * passed to the server (which rejects unknown flags) and hide the mistake.
 */
export function parseWrapperArgs(argv: readonly string[]): WrapperArgs {
  let backend: Backend | undefined;
  const forward: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg === undefined) {
      continue;
    }

    if (!arg.startsWith(WRAPPER_FLAG_PREFIX)) {
      forward.push(arg);

      continue;
    }

    const [name, inlineValue] = splitFlag(arg);

    if (name !== "--starling-backend") {
      throw new WrapperUsageError(
        `Unknown wrapper flag ${name}. The only wrapper flag is ` +
          `--starling-backend <backend>; everything else is forwarded to the ` +
          `server, which would reject ${name} as an unknown option.`,
      );
    }

    const value = inlineValue ?? argv[++i];

    if (value === undefined || value === "") {
      throw new WrapperUsageError(`${name} requires a backend name`);
    }

    if (!isBackend(value)) {
      throw new WrapperUsageError(
        `Unknown backend ${value}. Known backends: cpu, vulkan, cuda, rocm, metal.`,
      );
    }

    backend = value;
  }

  const wrapper: WrapperDraft = { forward };

  if (backend !== undefined) {
    wrapper.backend = backend;
  }

  return wrapper;
}

/** Mutable draft of {@link WrapperArgs}; the backend key is added only when set. */
interface WrapperDraft {
  backend?: Backend;
  forward: string[];
}

function splitFlag(arg: string): [string, string | undefined] {
  const eq = arg.indexOf("=");

  return eq === -1 ? [arg, undefined] : [arg.slice(0, eq), arg.slice(eq + 1)];
}

/** Read the package version from the package.json above this file. */
export function packageVersion(): string {
  const manifest = readFileSync(fileURLToPath(new URL("../package.json", import.meta.url)), "utf8");

  const match = /"version"\s*:\s*"([^"]+)"/.exec(manifest);

  if (match === null || match[1] === undefined) {
    throw new Error("packages/serve/package.json has no parsable version field");
  }

  return match[1];
}

export interface CliDeps {
  readonly argv?: readonly string[];
  readonly env?: NodeJS.ProcessEnv;
  readonly version?: string;
  readonly spawnFn?: SpawnFn;
  readonly exit?: (code: number) => void;
  readonly log?: (message: string) => void;
  /** Fetch override for tests. */
  readonly fetchImpl?: typeof fetch;
}

/** Mutable draft of {@link EnsureOptions}; optional keys are added only when set. */
interface OptionsDraft {
  version: string;
  backend?: Backend;
  repo?: string;
  releaseTag?: string;
  cacheDir?: string;
  fetchImpl?: typeof fetch;
  log: (message: string) => void;
}

export async function runCli(deps: CliDeps = {}): Promise<void> {
  const env = deps.env ?? process.env;
  const exit = deps.exit ?? ((code: number) => process.exit(code));
  const log = deps.log ?? ((message: string) => process.stderr.write(`${message}\n`));

  let wrapper: WrapperArgs;

  try {
    wrapper = parseWrapperArgs(deps.argv ?? process.argv.slice(2));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`starling-serve: ${message}`);
    exit(2);

    return;
  }

  const backend = wrapper.backend ?? nonEmpty(env["STARLING_SERVE_BACKEND"]);
  const repo = nonEmpty(env["STARLING_SERVE_REPO"]);
  const releaseTagOverride = nonEmpty(env["STARLING_SERVE_RELEASE"]);
  const cacheDirOverride = nonEmpty(env["STARLING_SERVE_CACHE"]);
  const options: OptionsDraft = { version: deps.version ?? packageVersion(), log };

  if (backend !== undefined) {
    if (!isBackend(backend)) {
      log(
        `starling-serve: STARLING_SERVE_BACKEND=${backend} is not a backend ` +
          "(cpu, vulkan, cuda, rocm, metal).",
      );
      exit(2);

      return;
    }

    options.backend = backend;
  }

  if (repo !== undefined) {
    options.repo = repo;
  }

  if (releaseTagOverride !== undefined) {
    options.releaseTag = releaseTagOverride;
  }

  if (cacheDirOverride !== undefined) {
    options.cacheDir = cacheDirOverride;
  }

  if (deps.fetchImpl !== undefined) {
    options.fetchImpl = deps.fetchImpl;
  }

  let binaryPath: string;

  try {
    binaryPath = (await ensureBinary(options)).binaryPath;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`starling-serve: ${message}`);
    exit(1);

    return;
  }

  const spawnFn = deps.spawnFn ?? defaultSpawn;
  const child = spawnFn(binaryPath, wrapper.forward, { stdio: "inherit" });
  forwardSignals(child);

  child.on("error", (error) => {
    log(`starling-serve: failed to start ${binaryPath}: ${error.message}`);
    exit(1);
  });

  child.on("close", (code, signal) => {
    if (signal !== null) {
      // Our own signal handlers suppress default termination, so translate
      // the child's death-by-signal into the conventional 128+n exit code.
      exit(signalExitCode(signal) ?? 1);

      return;
    }

    exit(code ?? 0);
  });
}

const defaultSpawn: SpawnFn = (file, args, options) => spawn(file, [...args], options);

function signalExitCode(signal: string): number | undefined {
  if (signal === "SIGHUP") return 129;

  if (signal === "SIGINT") return 130;

  if (signal === "SIGTERM") return 143;

  return undefined;
}

function forwardSignals(child: SpawnedProcess): void {
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
    process.on(signal, () => {
      child.kill(signal);
    });
  }
}

function nonEmpty(value: string | undefined): string | undefined {
  return value && value.trim() !== "" ? value : undefined;
}

/**
 * True when this module is the Node entry point. npm/npx/pnpm run the bin
 * through a `node_modules/.bin` symlink, so the invoked `argv[1]` is the
 * symlink while this module's own path is already resolved to the real file —
 * the raw strings never match. Both sides are therefore resolved to their
 * realpaths before comparing, which is invariant to how the bin was linked.
 * `invokedPath` is taken by the caller (a parameter default would collapse an
 * explicitly-absent `argv[1]` back to the real one).
 */
export function isEntryPoint(
  invokedPath: string | undefined,
  modulePath: string = fileURLToPath(import.meta.url),
): boolean {
  if (invokedPath === undefined) {
    return false;
  }

  try {
    return samePath(realpathSync(invokedPath), realpathSync(modulePath));
  } catch {
    // The invoked path does not resolve to this file (another script, a stale
    // path): this module was imported, not executed.
    return false;
  }
}

/** Path equality that tolerates Windows case-insensitive filesystems. */
function samePath(left: string, right: string): boolean {
  return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;
}

// Only self-execute when invoked as the bin; imports (tests, programmatic use)
// stay side-effect free.
if (isEntryPoint(process.argv[1])) {
  await runCli();
}
