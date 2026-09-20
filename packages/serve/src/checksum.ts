/**
 * SHA-256 helpers. Checksums are checked twice before a downloaded binary is
 * executed: the archive hash against the release's consolidated
 * `SHA256SUMS.txt`, and the extracted executable against the `.sha256` file
 * shipped inside each archive. Both use the `sha256sum` text format
 * (`<hex>  <name>`), which the release workflow produces on every platform.
 */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

export class ChecksumFormatError extends Error {
  constructor(line: string) {
    super(`Malformed sha256 checksum line: ${JSON.stringify(line)}`);
    this.name = "ChecksumFormatError";
  }
}

export class ChecksumMismatchError extends Error {
  constructor(
    readonly file: string,
    readonly expected: string,
    readonly actual: string,
  ) {
    super(
      `Checksum mismatch for ${file}: expected ${expected}, got ${actual}. ` +
        "The download was corrupted or tampered with; refusing to run it. " +
        "Clear the cache directory and retry.",
    );
    this.name = "ChecksumMismatchError";
  }
}

/** Streaming SHA-256 of a file, so 100 MB binaries never load into memory. */
export async function sha256File(path: string): Promise<string> {
  const stream = createReadStream(path);
  const hash = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve());
    stream.on("error", reject);
  });

  return hash.digest("hex");
}

/**
 * Parse `sha256sum`-format checksum text and return the hex digest. `contents`
 * may be a full SHA256SUMS file; `name`, when given, selects the matching
 * line. Without `name` the text must contain exactly one entry.
 */
export function parseChecksum(contents: string, name?: string): string {
  const entries: string[] = [];

  for (const rawLine of contents.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (line === "" || line.startsWith("#")) continue;
    const match = /^([0-9a-fA-F]{64})\s+\*?(.+)$/.exec(line);

    if (!match || match[1] === undefined || match[2] === undefined) {
      throw new ChecksumFormatError(rawLine);
    }

    const entry = match[1].toLowerCase();
    const file = match[2];

    if (name === undefined) {
      entries.push(entry);
    } else if (file.trim() === name) {
      return entry;
    }
  }

  if (name === undefined) {
    if (entries.length !== 1) {
      throw new ChecksumFormatError(`expected exactly one checksum entry, got ${entries.length}`);
    }

    const only = entries[0];

    if (only !== undefined) return only;
  }

  throw new ChecksumFormatError(`no checksum entry for ${name}`);
}

/** Throw {@link ChecksumMismatchError} unless `actual` equals `expected`. */
export function assertChecksum(file: string, expected: string, actual: string): void {
  if (expected !== actual) {
    throw new ChecksumMismatchError(file, expected, actual);
  }
}
