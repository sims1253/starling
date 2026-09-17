/**
 * Test fixtures: build real release archives (tar.gz via ustar + gzip) in pure
 * Node, so install tests exercise the actual system `tar` extraction without
 * network access or OS-specific tooling.
 */
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";

export interface ReleaseFixture {
  /** Fetched asset map keyed by full download URL. */
  readonly assets: Map<string, string | Buffer>;
  /** The executable payload the archive contains. */
  readonly binaryContent: Buffer;
  readonly archiveName: string;
  readonly binaryName: string;
}

export interface FixtureOptions {
  readonly repo: string;
  readonly tag: string;
  readonly archiveName: string;
  readonly binaryName: string;
  /** Tamper hooks: mutate members before checksums are computed. */
  readonly tamperBinary?: (content: Buffer) => Buffer;
  readonly tamperInnerChecksum?: (content: string) => string;
  readonly tamperArchive?: (archive: Buffer) => Buffer;
}

export function sha256(content: Buffer | string): string {
  return createHash("sha256").update(content).digest("hex");
}

export function buildReleaseFixture(options: FixtureOptions): ReleaseFixture {
  const binaryContent =
    options.tamperBinary?.(Buffer.from(`#!/bin/sh\necho starling-serve test\n`, "utf8")) ??
    Buffer.from(`#!/bin/sh\necho starling-serve test\n`, "utf8");

  const binarySha = sha256(binaryContent);

  const innerChecksum =
    options.tamperInnerChecksum?.(`${binarySha}  ${options.binaryName}\n`) ??
    `${binarySha}  ${options.binaryName}\n`;

  const runtime = `# RUNTIME.md test fixture\nNo accelerator libraries.\n`;

  let archive = tarGz([
    { name: options.binaryName, mode: 0o755, content: binaryContent },
    { name: `${options.binaryName}.sha256`, content: innerChecksum },
    { name: "RUNTIME.md", content: runtime },
  ]);

  // Checksum the pristine archive, then optionally tamper with the bytes that
  // get served, so SHA256SUMS.txt disagrees with the download.
  const archiveSha = sha256(archive);
  archive = options.tamperArchive?.(archive) ?? archive;

  const base = `https://github.com/${options.repo}/releases/download/${options.tag}`;

  const assets = new Map<string, string | Buffer>([
    [`${base}/${options.archiveName}`, archive],
    [`${base}/SHA256SUMS.txt`, `${archiveSha}  ${options.archiveName}\n`],
  ]);

  return {
    assets,
    binaryContent,
    archiveName: options.archiveName,
    binaryName: options.binaryName,
  };
}

/** A fetch stub serving `assets` from memory; unmatched URLs 404. */
export function fakeFetch(assets: Map<string, string | Buffer>): typeof fetch {
  const fetcher: typeof fetch = async (input) => {
    const url = String(input);
    const body = assets.get(url);

    if (body === undefined) {
      return new Response("not found", { status: 404 });
    }

    if (Buffer.isBuffer(body)) {
      // Copying into a plain Uint8Array keeps the DOM-lib BodyInit type
      // happy without an assertion; bytes are unchanged.
      return new Response(new Uint8Array(body), { status: 200 });
    }

    return new Response(body, { status: 200 });
  };

  return fetcher;
}

interface TarMember {
  name: string;
  mode?: number;
  content: string | Buffer;
}

/** Minimal ustar writer: flat members, no symlinks, no pax extensions. */
export function tarGz(members: readonly TarMember[]): Buffer {
  const chunks: Buffer[] = [];

  for (const member of members) {
    chunks.push(ustarHeader(member.name, member.mode ?? 0o644, Buffer.byteLength(member.content)));
    const content = Buffer.from(member.content);
    chunks.push(content, zeroBlock(-content.length & 511));
  }

  chunks.push(zeroBlock(1024));

  return gzipSync(Buffer.concat(chunks));
}

function ustarHeader(name: string, mode: number, size: number): Buffer {
  const header = Buffer.alloc(512, 0);
  header.write(name.slice(0, 99), 0, "utf8");
  header.write(octal(mode, 7), 100);
  header.write(octal(0, 7), 108);
  header.write(octal(0, 7), 116);
  header.write(octal(size, 11), 124);
  header.write(octal(0, 11), 136);
  header.write("        ", 148); // checksum placeholder: eight spaces
  header.write("0", 156); // typeflag: regular file
  header.write("ustar", 257, "utf8");
  header.write("00", 263);
  const checksum = header.reduce((sum, byte) => sum + byte, 0);
  header.write(`${octal(checksum, 6)}\u0000 `, 148);

  return header;
}

function octal(value: number, width: number): string {
  return value.toString(8).padStart(width, "0");
}

function zeroBlock(length: number): Buffer {
  return Buffer.alloc(length, 0);
}
