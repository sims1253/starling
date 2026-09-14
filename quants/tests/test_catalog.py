import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from quants.starling_quants.cli import RECIPE_DIR, build, catalog, main, plan


class QuantArtifacts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.gguf"
        self.source.write_bytes(b"GGUF" + b"\0" * 24)
        self.binary = self.root / "quantizer"
        self.binary.write_bytes(b"test quantizer")
        self.output = self.root / "output.gguf"

    def spec(self):
        return plan("parakeet-q8", self.source, self.output, self.binary, None)

    def test_every_recipe_has_a_catalog_entry(self):
        recipes = {profile["recipe"] for profile in catalog() if "recipe" in profile}
        self.assertEqual(recipes, {path.name for path in RECIPE_DIR.glob("*.recipe")})

    def test_calibrated_profiles_require_imatrix(self):
        for profile in catalog():
            if profile["requires_imatrix"]:
                with self.subTest(profile=profile["id"]), self.assertRaisesRegex(ValueError, "importance matrix"):
                    plan(profile["id"], self.source, self.output, self.binary, None)

    def test_recipe_required_flags_survive_planning(self):
        compact = plan("parakeet-iq2-compact", self.source, self.output, self.binary, self.root / "cal.imx")
        self.assertIn("--shrink-f16", compact["argv"])
        moss = plan("moss-q4e4", self.source, self.output, self.binary, self.root / "cal.imx")
        self.assertIn("--f32-1d", moss["argv"])

    def test_refuses_in_place_quantization(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            plan("parakeet-q8", self.source, self.source, self.binary, None)

    def test_failure_does_not_publish_partial_artifact(self):
        def fail(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"partial")
            raise subprocess.CalledProcessError(1, command)
        with patch("quants.starling_quants.cli.subprocess.run", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                build(self.spec())
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".starling-quant-*")))

    def test_success_records_inputs_and_detects_tampering(self):
        def write(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"GGUF" + b"\0" * 32)
        with patch("quants.starling_quants.cli.subprocess.run", side_effect=write):
            record = build(self.spec())
        self.assertEqual(record["evaluation"], "not_evaluated")
        self.assertIn("source_sha256", record)
        self.assertIn("quantizer_sha256", record)
        self.assertEqual(json.loads(self.output.with_suffix(".gguf.json").read_text())["sha256"], record["sha256"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["verify", str(self.output)]), 0)
            self.output.write_bytes(b"tampered")
            self.assertEqual(main(["verify", str(self.output)]), 1)

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"precious")
        with self.assertRaisesRegex(ValueError, "already exists"):
            build(self.spec())
        self.assertEqual(self.output.read_bytes(), b"precious")

    def test_invalid_quantizer_output_is_not_published(self):
        def write(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(b"bad output")
        with patch("quants.starling_quants.cli.subprocess.run", side_effect=write):
            with self.assertRaisesRegex(ValueError, "GGUF"):
                build(self.spec())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
