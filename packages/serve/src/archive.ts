/**
 * Archive extraction via the system `tar`.
 *
 * The release archives are flat (`starling-serve-<os>-<backend>[.exe]`,
 * its `.sha256`, `RUNTIME.md`), and only the two files the launcher needs are
 * extracted — by member name, never a wildcard — into a fresh temporary
 * directory. Windows 10+ ships bsdtar, which reads both zip and tar.gz; Linux
 * and macOS releases are tar.gz, which GNU tar and bsdtar both read. The
 * executor is injected so tests can assert the argument vector without
 * extracting anything.
 */
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
 * Build the argv for extracting `members` from `archive` into `destDir`.
 * Exported for tests.
 */
export function tarArgs(
  archive: string,
  members: readonly string[],
  destDir: string,
  archiveExt: ".tar.gz" | ".zip",
): string[] {
  // bsdtar auto-detects zip; GNU tar needs -z for gzip (and cannot read zip,
  // but Windows artifacts are zip and Windows always has bsdtar).
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
  await execFile("tar", tarArgs(archivePath, members, destDir, spec.archiveExt));

  return {
    binaryPath: join(destDir, spec.binary),
    checksumPath: join(destDir, spec.checksum),
  };
}
