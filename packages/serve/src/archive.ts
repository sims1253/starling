/**
 * Archive extraction via the system `tar`.
 *
 * The release archives are flat (`starling-serve-<os>-<backend>[.exe]`,
 * its `.sha256`, `RUNTIME.md`), and only the two files the launcher needs are
 * extracted — by member name, never a wildcard — into a fresh staging
 * directory. Linux and macOS releases are tar.gz, which GNU tar and bsdtar
 * both read; Windows releases are zip, which only Windows' bundled bsdtar
 * reads, so on Windows the System32 bsdtar is resolved explicitly (see
 * {@link resolveTarExecutable}). The executor is injected so tests can assert
 * the argument vector without extracting anything.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { execFile as execFileCb } from "node:child_process";
import { promisify } from "node:util";
import type { ArtifactSpec } from "./platforms.js";

export type ExecFileFn = (
  file: string,
  args: readonly string[],
) => Promise<{ stdout: string; stderr: string }>;

export const defaultExecFile: ExecFileFn = promisify(execFileCb);

/**
 * Resolve the tar executable to run on `platform`.
 *
 * A bare `"tar"` goes through `PATH` — and in a Git Bash session the MSYS
 * directories precede `System32`, so it resolves to Git for Windows' GNU tar,
 * which cannot read the zip Windows releases are packaged as. Windows 10+
 * ships bsdtar (reads both zip and tar.gz) as
 * `<SystemRoot>\System32\tar.exe`; preferring it explicitly sidesteps PATH
 * ordering entirely. Other platforms (and Windows images without the bundled
 * bsdtar) use the `PATH` lookup.
 */
export function resolveTarExecutable(
  platform: string = process.platform,
  systemRoot: string | undefined = process.env["SystemRoot"] ?? process.env["windir"],
  fileExists: (path: string) => boolean = existsSync,
): string {
  if (platform !== "win32" || systemRoot === undefined) {
    return "tar";
  }

  const systemTar = `${systemRoot}\\System32\\tar.exe`;

  return fileExists(systemTar) ? systemTar : "tar";
}

/**
 * Build the argv for extracting `members` from `archive` into `destDir`.
 * Exported for tests.
 */
export function tarArgs(
  archive: string,
  members: readonly string[],
  destDir: string,
  archiveExt: ".tar.gz" | ".zip",
): string[] {
  // GNU tar needs -z for gzip; bsdtar (the only extractor used for zip, via
  // resolveTarExecutable) accepts it too and auto-detects zip with -xf.
  const flags = archiveExt === ".tar.gz" ? ["-xzf"] : ["-xf"];

  return [...flags, archive, "-C", destDir, ...members];
}

/** Extract the executable and its checksum from a release archive. */
export async function extractBinary(
  archivePath: string,
  spec: ArtifactSpec,
  destDir: string,
  execFile: ExecFileFn = defaultExecFile,
): Promise<{ binaryPath: string; checksumPath: string }> {
  const members: readonly string[] = [spec.binary, spec.checksum];
  await execFile(resolveTarExecutable(), tarArgs(archivePath, members, destDir, spec.archiveExt));

  return {
    binaryPath: join(destDir, spec.binary),
    checksumPath: join(destDir, spec.checksum),
  };
}
