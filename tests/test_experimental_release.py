"""Exercise release completeness and publication failure/retry behavior without GitHub."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experimental_prepare", ROOT / "scripts/experimental-release/prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
BUNDLE_SPEC = importlib.util.spec_from_file_location("bundle_engines", ROOT / "scripts/experimental-release/bundle-engines.py")
bundle_engines = importlib.util.module_from_spec(BUNDLE_SPEC)
BUNDLE_SPEC.loader.exec_module(bundle_engines)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.args = argparse.Namespace(
            artifacts=self.dist, notes=self.root / "notes.md",
            version="0.0.0-experimental.12.gabcdef123456", tag="experimental-12-abcdef123456",
            sha="abcdef123456" + "0" * 28, repository="owner/project", run_id="123", run_number="12",
        )
        # Representative output from all four producers: six servers, two apps, one APK.
        self.packages = [
            "starling-serve-linux-cpu.tar.gz", "starling-serve-linux-vulkan.tar.gz",
            "starling-serve-linux-cuda.tar.gz", "starling-serve-windows-cpu.zip",
            "starling-serve-windows-vulkan.zip", "starling-serve-windows-cuda.zip",
            "starling-gpui-linux-x64.tar.gz", "starling-gpui-windows-x64.zip",
            f"starling-mobile-{self.args.version}-i8mm.apk",
        ]
        for name in self.packages:
            (self.dist / name).write_bytes(name.encode())
        apk = self.packages[-1]
        digest = hashlib.sha256(apk.encode()).hexdigest()
        (self.dist / "SHA256SUMS-android.txt").write_text(f"{digest}  {apk}\n")

    def test_complete_release_has_verifiable_checksums_and_commit(self):
        prepare.prepare(self.args)
        sums = (self.dist / "SHA256SUMS.txt").read_text().splitlines()
        self.assertEqual(len(sums), 11)
        for line in sums:
            digest, name = line.split()
            self.assertEqual(digest, hashlib.sha256((self.dist / name).read_bytes()).hexdigest())
        info = json.loads((self.dist / "build-info.json").read_text())
        self.assertEqual(info["commit"], self.args.sha)
        # Preference order, like BUNDLED_ENGINES in package-desktop.yml.
        self.assertEqual(info["desktop_bundled_engines"], ["vulkan", "cpu"])
        self.assertIn(self.args.sha, self.args.notes.read_text())

    def test_notes_describe_one_self_contained_desktop_download(self):
        prepare.prepare(self.args)
        notes = self.args.notes.read_text()
        self.assertIn("pick a model in the app", notes)
        # The desktop row is the app archive alone; servers are the advanced path.
        self.assertIn("| Linux desktop (CPU or Vulkan GPU) | `starling-gpui-linux-x64.tar.gz` |", notes)
        self.assertIn("Manual server mode", notes)
        self.assertNotIn("+ `starling-serve-linux-vulkan.tar.gz`", notes)

    def test_missing_package_blocks_publication(self):
        (self.dist / self.packages[0]).unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*starling-serve-linux-cpu"):
            prepare.prepare(self.args)
        self.assertFalse(self.args.notes.exists())

    def test_stale_apk_version_blocks_publication(self):
        apk = self.dist / self.packages[-1]
        apk.rename(self.dist / "starling-mobile-old-i8mm.apk")
        with self.assertRaisesRegex(ValueError, "Asset mismatch"):
            prepare.prepare(self.args)

    def test_empty_package_blocks_publication(self):
        (self.dist / self.packages[1]).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty or invalid"):
            prepare.prepare(self.args)

    def test_corrupt_apk_blocks_publication(self):
        (self.dist / self.packages[-1]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Android checksum"):
            prepare.prepare(self.args)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "dist").mkdir()
        (self.root / "dist/package.zip").write_bytes(b"package")
        (self.root / "experimental-notes.md").write_text("notes")
        # Stateful gh substitute: failures happen before changing remote state.
        gh = self.root / "gh"
        gh.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
state = Path("state")
with Path("calls").open("a") as log:
    log.write(json.dumps(args) + "\\n")
command = args[1]
if os.environ.get("FAIL_COMMAND") == command:
    print("authentication failed", file=sys.stderr)
    sys.exit(1)
if command == "view":
    if not state.exists():
        print("release not found", file=sys.stderr)
        sys.exit(1)
    print("true" if state.read_text() == "draft" else "false")
elif command == "create":
    assert "--draft" in args and "--prerelease" in args and "--latest=false" in args
    assert args[args.index("--target") + 1] == os.environ["GITHUB_SHA"]
    state.write_text("draft")
elif command == "upload":
    assert state.read_text() == "draft"
    Path("uploaded").touch()
elif command == "edit":
    assert Path("uploaded").exists()
    assert "--draft=false" in args and "--prerelease" in args and "--latest=false" in args
    state.write_text("published")
else:
    sys.exit(2)
''')
        gh.chmod(0o755)
        self.env = {
            **os.environ, "PATH": f"{self.root}:{os.environ['PATH']}",
            "RELEASE_TAG": "experimental-12-abcdef123456", "GITHUB_SHA": "abcdef123456" + "0" * 28,
            "GITHUB_RUN_NUMBER": "12", "RUNNER_TEMP": str(self.root),
            "GH_REPO": "owner/project", "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
        }

    def publish(self, fail=""):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/experimental-release/publish.sh")],
            cwd=self.root, env={**self.env, "FAIL_COMMAND": fail}, capture_output=True, text=True,
        )

    def test_first_publish_and_idempotent_rerun(self):
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "state").read_text(), "published")
        (self.root / "calls").unlink()
        self.assertEqual(self.publish().returncode, 0)
        calls = [json.loads(line) for line in (self.root / "calls").read_text().splitlines()]
        self.assertEqual([call[1] for call in calls], ["view"])

    def test_failed_upload_remains_draft_and_retry_completes(self):
        self.assertNotEqual(self.publish(fail="upload").returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "draft")
        self.assertFalse((self.root / "uploaded").exists())
        self.assertEqual(self.publish().returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "published")

    def test_failed_create_never_uploads(self):
        self.assertNotEqual(self.publish(fail="create").returncode, 0)
        self.assertFalse((self.root / "uploaded").exists())

    def test_failed_lookup_never_creates_or_uploads(self):
        result = self.publish(fail="view")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("authentication failed", result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls").read_text().splitlines()]
        self.assertEqual([call[1] for call in calls], ["view"])

    def test_failed_publish_can_resume(self):
        self.assertNotEqual(self.publish(fail="edit").returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "draft")
        self.assertEqual(self.publish().returncode, 0)


def fake_executable(path: Path, version: str, abi: int) -> None:
    """A stand-in starling-serve binary answering --version/--abi-version."""
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        'if "--abi-version" in sys.argv[1:]:\n'
        f"    print({abi})\n"
        "else:\n"
        f'    print("starling-serve {version}")\n'
        f'    print("abi-version: {abi}")\n'
        '    print("backend: contract-fixture")\n'
    )
    path.chmod(0o755)


def fake_server_archive(artifacts: Path, platform: str, backend: str,
                        version="0.0.0-test", abi=8, checksum=None) -> Path:
    """Create artifacts/<artifact>/<archive> like actions/download-artifact.

    The archive contains the platform-named binary, its .sha256 sidecar, and
    RUNTIME.md, exactly like the release-starling-serve packaging steps.
    """
    name = f"starling-serve-{platform}-{backend}"
    exe = ".exe" if platform == "windows" else ""
    root = artifacts / name
    root.mkdir(parents=True)
    staging = root / "staging"
    staging.mkdir()
    binary = staging / f"{name}{exe}"
    fake_executable(binary, version, abi)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    (staging / f"{name}.sha256").write_text(f"{checksum or digest}  {name}{exe}\n")
    (staging / "RUNTIME.md").write_text(f"# Runtime prerequisites ({backend})\n")
    archive = root / (name + (".zip" if platform == "windows" else ".tar.gz"))
    if platform == "windows":
        with zipfile.ZipFile(archive, "w") as bundle:
            for item in sorted(staging.iterdir()):
                bundle.write(item, item.name)
    else:
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(staging.iterdir()):
                tar.add(item, arcname=item.name)
    shutil.rmtree(staging)
    return archive


class BundleEnginesTests(unittest.TestCase):
    """The engines/ assembly inside the desktop archive (issue #362)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "engines-artifacts"
        self.artifacts.mkdir()
        self.out = self.root / "dist" / "engines"

    def bundle(self, platform="linux", backends=("vulkan", "cpu"), runtime_md=None):
        bundle_engines.bundle(platform, self.artifacts, self.out, list(backends), runtime_md)

    def fake_sidecar_archive(self, artifacts: Path, platform: str, backend: str,
                             sidecar: str) -> None:
        """A server archive whose .sha256 sidecar is exactly `sidecar`."""
        fake_server_archive(artifacts, platform, backend)
        name = f"starling-serve-{platform}-{backend}"
        root = artifacts / name
        staging = root / "staging"
        staging.mkdir()
        binary = staging / f"{name}{'.exe' if platform == 'windows' else ''}"
        fake_executable(binary, "0.0.0-test", 8)
        (staging / f"{name}.sha256").write_text(sidecar)
        (staging / "RUNTIME.md").write_text("# Runtime prerequisites\n")
        archive = root / (name + (".zip" if platform == "windows" else ".tar.gz"))
        if platform == "windows":
            with zipfile.ZipFile(archive, "w") as bundle_zip:
                for item in sorted(staging.iterdir()):
                    bundle_zip.write(item, item.name)
        else:
            with tarfile.open(archive, "w:gz") as tar:
                for item in sorted(staging.iterdir()):
                    tar.add(item, arcname=item.name)
        shutil.rmtree(staging)

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_bundle_produces_the_discovered_layout(self):
        fake_server_archive(self.artifacts, "linux", "vulkan")
        fake_server_archive(self.artifacts, "linux", "cpu")
        self.bundle()
        self.assertEqual(
            sorted(path.name for path in self.out.iterdir()),
            ["RUNTIME.md", "SHA256SUMS.txt", "engines.json",
             "starling-serve-cpu", "starling-serve-vulkan"])
        manifest = json.loads((self.out / "engines.json").read_text())
        self.assertEqual(manifest["version"], "0.0.0-test")
        self.assertEqual(manifest["abi"], 8)
        self.assertEqual(manifest["engines"], [
            {"backend": "vulkan", "file": "starling-serve-vulkan"},
            {"backend": "cpu", "file": "starling-serve-cpu"},
        ])
        # One "<sha256 hex>  <file name>" line per engine binary, sorted, LF.
        sums = (self.out / "SHA256SUMS.txt").read_text()
        self.assertTrue(sums.endswith("\n"))
        self.assertNotIn("\r", sums)
        names = []
        for line in sums.splitlines():
            digest, _, name = line.partition("  ")
            names.append(name)
            self.assertEqual(digest, hashlib.sha256((self.out / name).read_bytes()).hexdigest())
        self.assertEqual(names, ["starling-serve-cpu", "starling-serve-vulkan"])
        # RUNTIME.md is copied from the first (preference-order) server archive.
        self.assertEqual((self.out / "RUNTIME.md").read_text(),
                         "# Runtime prerequisites (vulkan)\n")
        # A --runtime-md override replaces the archive's copy.
        override = self.root / "RUNTIME-override.md"
        override.write_text("override\n")
        shutil.rmtree(self.out)
        self.bundle(runtime_md=override)
        self.assertEqual((self.out / "RUNTIME.md").read_text(), "override\n")

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_windows_platform_uses_exe_names_and_zip_archives(self):
        fake_server_archive(self.artifacts, "windows", "vulkan")
        fake_server_archive(self.artifacts, "windows", "cpu")
        self.bundle(platform="windows")
        manifest = json.loads((self.out / "engines.json").read_text())
        self.assertEqual([engine["file"] for engine in manifest["engines"]],
                         ["starling-serve-vulkan.exe", "starling-serve-cpu.exe"])
        self.assertEqual(
            sorted(path.name for path in self.out.iterdir()),
            ["RUNTIME.md", "SHA256SUMS.txt", "engines.json",
             "starling-serve-cpu.exe", "starling-serve-vulkan.exe"])

    def test_checksum_verification_failure_blocks_bundling(self):
        fake_server_archive(self.artifacts, "linux", "vulkan", checksum="0" * 64)
        fake_server_archive(self.artifacts, "linux", "cpu")
        with self.assertRaisesRegex(bundle_engines.BundleError, "checksum mismatch"):
            self.bundle()
        self.assertFalse(self.out.joinpath("engines.json").exists())

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_sidecar_must_name_exactly_the_binary_on_one_line(self):
        # The .sha256 sidecar pairs one digest with ONE file name; a sidecar
        # naming something else, carrying extra lines, or empty must be
        # rejected instead of silently verifying the first token. (These
        # rejections fire before the digest comparison, so a dummy digest
        # is enough to pin the shape.)
        dummy = "0" * 64
        for sidecar, pattern in [
            (f"{dummy}  starling-serve-linux-cpu\n{dummy}  other-file\n",
             "exactly one"),
            ("", "exactly one"),
            (f"{dummy}  starling-serve-linux-vulkan\n", "names"),
            (f"{dummy} starling-serve-linux-cpu\n", "not a '<sha256>  <name>'"),
        ]:
            with self.subTest(sidecar=sidecar):
                shutil.rmtree(self.artifacts)
                self.artifacts.mkdir()
                self.fake_sidecar_archive(self.artifacts, "linux", "cpu", sidecar)
                with self.assertRaisesRegex(bundle_engines.BundleError, pattern):
                    self.bundle(backends=("cpu",))

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_sidecar_accepts_the_sha256sum_binary_mode_marker(self):
        # `sha256sum -b` prefixes the name with `*`; that is still the same
        # single pairing and must verify. The fake binary is deterministic,
        # so a probe copy yields the archive binary's digest.
        probe = self.root / "probe-binary"
        fake_executable(probe, "0.0.0-test", 8)
        digest = hashlib.sha256(probe.read_bytes()).hexdigest()
        probe.unlink()
        self.fake_sidecar_archive(
            self.artifacts, "linux", "cpu", f"{digest}  *starling-serve-linux-cpu\n")
        self.bundle(backends=("cpu",))
        self.assertTrue((self.out / "engines.json").exists())

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_rerun_removes_stale_engines_this_run_does_not_stage(self):
        # A rerun into an existing --out must not leave an engine without a
        # checksum entry: the stale starling-serve-cuda from the previous
        # run is gone, the ones staged now are exactly the manifest's.
        fake_server_archive(self.artifacts, "linux", "vulkan")
        fake_server_archive(self.artifacts, "linux", "cpu")
        self.bundle()
        stale = self.out / "starling-serve-cuda"
        stale.write_text("left over from an earlier run")
        self.bundle(backends=("cpu",))
        self.assertFalse(stale.exists())
        self.assertTrue((self.out / "starling-serve-cpu").exists())
        self.assertFalse((self.out / "starling-serve-vulkan").exists())
        manifest = json.loads((self.out / "engines.json").read_text())
        self.assertEqual([engine["file"] for engine in manifest["engines"]],
                         ["starling-serve-cpu"])
        sums = (self.out / "SHA256SUMS.txt").read_text().splitlines()
        self.assertEqual([line.split("  ", 1)[1] for line in sums],
                         ["starling-serve-cpu"])

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_version_disagreement_blocks_bundling(self):
        fake_server_archive(self.artifacts, "linux", "vulkan", version="0.0.0-a")
        fake_server_archive(self.artifacts, "linux", "cpu", version="0.0.0-b")
        with self.assertRaisesRegex(bundle_engines.BundleError, "disagree on version"):
            self.bundle()
        self.assertFalse(self.out.joinpath("engines.json").exists())

    @unittest.skipIf(os.name != "posix", "fake executables need the POSIX exec bit")
    def test_abi_disagreement_blocks_bundling(self):
        fake_server_archive(self.artifacts, "linux", "vulkan", abi=8)
        fake_server_archive(self.artifacts, "linux", "cpu", abi=9)
        with self.assertRaisesRegex(bundle_engines.BundleError, "disagree on abi"):
            self.bundle()

    def test_missing_server_artifact_blocks_bundling(self):
        fake_server_archive(self.artifacts, "linux", "cpu")
        with self.assertRaisesRegex(bundle_engines.BundleError, "expected exactly one"):
            self.bundle()


if __name__ == "__main__":
    unittest.main()
