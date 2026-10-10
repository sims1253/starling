"""Offline stitch replay cache keys (issue #357).

Hermetic: stand-in binary and model files, no server.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stream_stitch_sim as sim  # noqa: E402


class FakeTranscriber(sim.Transcriber):
    def transcribe(self, wav: bytes) -> str:
        return f"text {len(self.cache)}"


class CacheKeyTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for name, data in (("serve-a", b"build a"), ("serve-b", b"build b"),
                           ("model-a.gguf", b"weights a"), ("model-b.gguf", b"weights b")):
            (self.dir / name).write_bytes(data)
        self.pcm = bytes(range(256)) * 8

    def transcriber(self, binary="serve-a", model="model-a.gguf", slug="parakeet"):
        args = argparse.Namespace(binary=self.dir / binary, model=self.dir / model,
                                  model_slug=slug, cache=self.dir / "cache.json", port=0)
        return FakeTranscriber(args)

    def test_cached_window_is_reused_across_runs(self):
        tr = self.transcriber()
        first = tr.text(self.pcm, 10, 100)
        tr.close()
        again = self.transcriber()
        self.assertEqual(again.text(self.pcm, 10, 100), first)
        self.assertEqual(again.misses, 0)

    def test_engine_or_audio_change_misses(self):
        tr = self.transcriber()
        tr.text(self.pcm, 10, 100)
        tr.close()
        for kwargs in ({"binary": "serve-b"}, {"model": "model-b.gguf"}, {"slug": "other"}):
            other = self.transcriber(**kwargs)
            other.text(self.pcm, 10, 100)
            self.assertEqual(other.misses, 1, kwargs)
        same = self.transcriber()
        same.text(self.pcm, 10, 100)
        self.assertEqual(same.misses, 0)
        same.text(self.pcm, 11, 100)  # another span
        same.text(self.pcm, 10, 99)
        changed = bytearray(self.pcm)
        changed[30] ^= 1  # another recording, same take name and span
        same.text(bytes(changed), 10, 100)
        self.assertEqual(same.misses, 3)

    def test_identity_is_content_not_path(self):
        tr = self.transcriber()
        tr.text(self.pcm, 0, 50)
        tr.close()
        (self.dir / "serve-a").write_bytes(b"rebuilt")
        rebuilt = self.transcriber()
        rebuilt.text(self.pcm, 0, 50)
        self.assertEqual(rebuilt.misses, 1)


class SrcTest(unittest.TestCase):
    """--src picks the replayed stitcher in either argument form."""

    def fake_src(self) -> Path:
        src = Path(tempfile.mkdtemp())
        (src / "starling").mkdir()
        (src / "starling" / "__init__.py").write_text("")
        (src / "starling" / "stream_chunk.py").write_text("FAKE = True\n")
        return src

    def test_space_and_equals_forms_select_the_checkout(self):
        src = self.fake_src()
        for form in (["--src", str(src)], [f"--src={src}"]):
            code = ("import sys, stream_stitch_sim as sim\n"
                    "ap = sim.argparse.ArgumentParser()\n"
                    "ap.add_argument('--src', type=sim.Path)\n"
                    f"sim.use_stitcher(ap.parse_args({form!r}).src)\n"
                    "import starling.stream_chunk as sc\n"
                    "print(sc.FAKE)\n")
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 cwd=Path(__file__).resolve().parent, check=True)
            self.assertEqual(out.stdout.strip(), "True", form)

    def test_another_copy_already_imported_is_refused(self):
        import starling.stream_chunk  # noqa: F401

        src = self.fake_src().resolve()
        try:
            with self.assertRaises(SystemExit):
                sim.use_stitcher(src)
        finally:
            sys.path.remove(str(src))


if __name__ == "__main__":
    unittest.main()
