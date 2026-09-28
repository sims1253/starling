"""Exercise release completeness and publication failure/retry behavior without GitHub."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experimental_prepare", ROOT / "scripts/experimental-release/prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


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
        self.assertIn(self.args.sha, self.args.notes.read_text())

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
    sys.exit(1)
if command == "view":
    if not state.exists(): sys.exit(1)
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

    def test_failed_publish_can_resume(self):
        self.assertNotEqual(self.publish(fail="edit").returncode, 0)
        self.assertEqual((self.root / "state").read_text(), "draft")
        self.assertEqual(self.publish().returncode, 0)


if __name__ == "__main__":
    unittest.main()
