#!/usr/bin/env python3
"""Assemble the bundled `engines/` directory inside a desktop archive.

Takes the standalone `starling-serve-<platform>-<backend>` release artifacts
(tar.gz on Linux, zip on Windows) produced earlier in the same workflow run,
verifies each binary against the `.sha256` file inside its archive, renames
the binaries to the fixed layout the desktop app's engine manager discovers,
requires every engine to report the same version and ABI, and writes the
`engines.json` manifest and `SHA256SUMS.txt`:

    <out>/
      starling-serve-vulkan[.exe]
      starling-serve-cpu[.exe]
      engines.json          {"version": ..., "abi": <int>, "engines": [...]}
      SHA256SUMS.txt        "<sha256 hex>  <file name>" per binary, sorted
      RUNTIME.md            copied from a server archive (docs/release-runtime.md)

Stdlib only. Exit code 1 with a message on any verification failure.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

# Preference order: the app tries these backends first to last.
DEFAULT_BACKENDS = "vulkan cpu"


class BundleError(RuntimeError):
    """A bundled-engine verification or assembly failure."""


def engine_file_name(platform: str, backend: str) -> str:
    """The fixed binary name inside `engines/` (the app discovers exactly this)."""
    name = f"starling-serve-{backend}"
    return name + ".exe" if platform == "windows" else name


def archive_name(platform: str, backend: str) -> str:
    """The standalone release archive name for a platform/backend."""
    suffix = ".zip" if platform == "windows" else ".tar.gz"
    return f"starling-serve-{platform}-{backend}{suffix}"


def parse_backends(value: str) -> list[str]:
    backends = value.replace(",", " ").split()
    if not backends:
        raise BundleError("no backends given")
    seen = set()
    for backend in backends:
        if not backend.isalnum():
            raise BundleError(f"invalid backend name: {backend!r}")
        if backend in seen:
            raise BundleError(f"duplicate backend: {backend!r}")
        seen.add(backend)
    return backends


def find_archive(artifacts: Path, platform: str, backend: str) -> Path:
    """Locate the release archive under `artifacts` (download-artifact layout
    puts it in `<artifacts>/<artifact-name>/<archive>`; a flat directory also
    works)."""
    expected = archive_name(platform, backend)
    matches = sorted(path for path in artifacts.rglob(expected) if path.is_file())
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "none"
        raise BundleError(
            f"expected exactly one {expected} under {artifacts} "
            f"(artifact starling-serve-{platform}-{backend}); found: {found}"
        )
    return matches[0]


def _ensure_inside(archive: Path, dest: Path, member: str, link: str | None = None) -> None:
    """Reject an archive member that would extract outside `dest`.

    Covers absolute paths and `..` segments (via resolve) plus, for tar
    links, a link target pointing outside the destination.
    """
    root = dest.resolve()
    target = (dest / member).resolve()
    if not target.is_relative_to(root):
        raise BundleError(
            f"{archive.name}: archive member escapes the destination: {member}")
    if link is not None:
        linked = (target.parent / link).resolve()
        if not linked.is_relative_to(root):
            raise BundleError(
                f"{archive.name}: archive member {member} links outside "
                f"the destination: {link}")


def extract_archive(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            try:
                # PEP 706's data filter rejects absolute paths, "..", and
                # links escaping the destination.
                tar.extractall(dest, filter="data")
            except TypeError:
                # Python without the extraction filters (pre-backport
                # 3.10/3.11): validate members and link targets by hand.
                for member in tar.getmembers():
                    _ensure_inside(
                        archive, dest, member.name,
                        member.linkname if member.issym() or member.islnk() else None)
                tar.extractall(dest)
            except tarfile.TarError as error:
                # The data filter's rejections (absolute or escaping links,
                # ".." members, ...) become the script's error type.
                raise BundleError(f"{archive.name}: {error}") from error
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                _ensure_inside(archive, dest, name)
            bundle.extractall(dest)
    else:
        raise BundleError(f"unsupported archive type: {archive.name}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(binary: Path, sha_file: Path, label: str) -> None:
    """Verify `binary` against the `<hex>  <name>` line inside `sha_file`."""
    if not sha_file.is_file():
        raise BundleError(f"{label}: archive is missing {sha_file.name}")
    tokens = sha_file.read_text().split()
    if not tokens:
        raise BundleError(f"{label}: {sha_file.name} is empty")
    expected = tokens[0].lower()
    actual = sha256_of(binary)
    if actual != expected:
        raise BundleError(
            f"{label}: checksum mismatch for {binary.name}: "
            f"archive says {expected}, binary is {actual}"
        )


def query_metadata(binary: Path) -> tuple[str, int]:
    """Run `--version` and `--abi-version`; return (version, abi)."""
    def run(flag: str) -> str:
        result = subprocess.run(
            [str(binary), flag], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise BundleError(
                f"{binary.name} {flag} failed with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    version_output = run("--version")
    first_line = version_output.splitlines()[0] if version_output.splitlines() else ""
    parts = first_line.split()
    if len(parts) < 2 or parts[0] != "starling-serve":
        raise BundleError(f"{binary.name} --version has unexpected output: {first_line!r}")
    try:
        abi = int(run("--abi-version").strip())
    except ValueError as error:
        raise BundleError(f"{binary.name} --abi-version did not print an integer") from error
    return parts[1], abi


def engines_json(version: str, abi: int, platform: str, backends: list[str]) -> str:
    payload = {
        "version": version,
        "abi": abi,
        "engines": [
            {"backend": backend, "file": engine_file_name(platform, backend)}
            for backend in backends
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def sha256sums_content(out: Path, platform: str, backends: list[str]) -> str:
    """One `<sha256 hex>  <file name>` line per engine binary, sorted, LF."""
    names = sorted(engine_file_name(platform, backend) for backend in backends)
    return "".join(f"{sha256_of(out / name)}  {name}\n" for name in names)


def bundle(platform: str, artifacts: Path, out: Path, backends: list[str],
           runtime_md: Path | None = None) -> None:
    if platform not in ("linux", "windows"):
        raise BundleError(f"unsupported platform: {platform!r} (linux or windows)")
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if platform == "windows" else ""
    version: str | None = None
    abi: int | None = None
    with tempfile.TemporaryDirectory(prefix="starling-engines-") as work:
        work_dir = Path(work)
        for backend in backends:
            archive = find_archive(artifacts, platform, backend)
            extracted = work_dir / f"starling-serve-{platform}-{backend}"
            extract_archive(archive, extracted)
            binary = extracted / f"starling-serve-{platform}-{backend}{suffix}"
            if not binary.is_file():
                raise BundleError(f"{backend}: archive {archive.name} does not contain {binary.name}")
            # The exec bit is meaningless for Windows CreateProcess, and
            # Python's zipfile does not preserve it, so set it on POSIX before
            # the metadata probe below runs the binary.
            if os.name == "posix":
                binary.chmod(binary.stat().st_mode | 0o111)
            sha_file = extracted / f"starling-serve-{platform}-{backend}.sha256"
            verify_checksum(binary, sha_file, backend)
            engine_version, engine_abi = query_metadata(binary)
            if engine_version != version and version is not None:
                raise BundleError(
                    f"engines disagree on version: {backends[0]} reports {version}, "
                    f"{backend} reports {engine_version}"
                )
            if engine_abi != abi and abi is not None:
                raise BundleError(
                    f"engines disagree on abi: {backends[0]} reports {abi}, "
                    f"{backend} reports {engine_abi}"
                )
            if version is None:
                version, abi = engine_version, engine_abi
            target = out / engine_file_name(platform, backend)
            shutil.copyfile(binary, target)
            if os.name == "posix":
                target.chmod(target.stat().st_mode | 0o111)
            print(f"bundled {backend}: {target.name} (version {engine_version}, abi {engine_abi})")
        runtime_source = (
            runtime_md if runtime_md is not None
            else work_dir / f"starling-serve-{platform}-{backends[0]}" / "RUNTIME.md"
        )
        if not runtime_source.is_file():
            raise BundleError(f"RUNTIME.md not found: {runtime_source}")
        shutil.copyfile(runtime_source, out / "RUNTIME.md")
    assert version is not None and abi is not None  # at least one backend required
    (out / "engines.json").write_text(engines_json(version, abi, platform, backends))
    (out / "SHA256SUMS.txt").write_text(sha256sums_content(out, platform, backends))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", required=True, choices=["linux", "windows"])
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="directory containing the downloaded server artifacts")
    parser.add_argument("--out", type=Path, required=True,
                        help="engines output directory inside the desktop dist")
    parser.add_argument("--backends", default=DEFAULT_BACKENDS,
                        help=f"backends to bundle in preference order "
                             f"(default: {DEFAULT_BACKENDS!r})")
    parser.add_argument("--runtime-md", type=Path, default=None,
                        help="RUNTIME.md to copy (default: the one inside the "
                             "first backend's server archive)")
    args = parser.parse_args()
    try:
        bundle(args.platform, args.artifacts, args.out,
               parse_backends(args.backends), args.runtime_md)
    except (BundleError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
