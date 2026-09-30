"""Supervision contract tests for starling-serve (issue #362).

Covers the process-level behaviors a supervisor (the desktop app) relies on:

- `--port 0` binds any free port and announces it on stdout as exactly one
  `STARLING_SERVE_LISTENING <host>:<port>` line once the socket is bound;
- a fixed port is announced with that port;
- out-of-range ports are rejected at startup;
- `--no-eager-load` + `POST /warmup` loads a deferred model then warms it,
  observable through the additive `/health` fields (`backend`, `warm`,
  `load_error`);
- `--parent-pid` exits the process cleanly when the parent disappears.

Unittest style, run by `python -m unittest discover -s backends/native/tests -v`
with starling-serve-contract-fixture built.
"""
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

# The sibling import below needs the tests directory on sys.path when this
# file is run directly (python test_supervision.py) instead of through
# unittest discovery, which puts the directory there itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_openai_api import (
    ANNOUNCE_RE, fixture_binary, read_announce, start_fixture, stop_fixture, wait_healthy,
)


class PortAnnouncement(unittest.TestCase):
    def test_port_zero_announces_and_serves(self):
        # start_fixture itself proves the --port 0 contract: it parses the
        # announce line (ANNOUNCE_RE) before /health is polled, so reaching a
        # healthy server means a valid line was printed for a bound socket.
        base, process = start_fixture("parakeet")
        self.addCleanup(stop_fixture, process)
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            health = json.load(response)
        self.assertEqual(health["status"], "ok")
        self.assertGreater(int(base.rsplit(":", 1)[1]), 0)

    def test_announce_line_matches_the_contract_exactly(self):
        process = subprocess.Popen(
            [str(fixture_binary()), "--model", "parakeet", "--gguf", __file__, "--port", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.addCleanup(stop_fixture, process)
        announce = read_announce(process)
        self.assertIsNotNone(announce)
        match = ANNOUNCE_RE.match(announce)
        self.assertIsNotNone(match, announce)
        self.assertEqual(match.group(1), "127.0.0.1")
        base = f"http://{match.group(1)}:{match.group(2)}"
        # The socket is accepting when the line is printed: /health must
        # answer immediately afterwards.
        self.assertTrue(wait_healthy(base, process))
        self.assertEqual(process.poll(), None)

    def test_fixed_port_announces_that_port(self):
        # The test must pass a FIXED port (that is the contract pinned
        # here), but a port learned by bind-then-close hands the socket to
        # the server with a race: another process can grab it between the
        # close and the server's listen. On a startup failure (no announce
        # line) retry with a fresh port instead — what is pinned is that
        # the announce reports exactly the port that was passed.
        announce = None
        process = None
        for _ in range(5):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            process = subprocess.Popen(
                [str(fixture_binary()), "--model", "parakeet", "--gguf", __file__, "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            announce = read_announce(process)
            if announce is not None:
                break
            stop_fixture(process)
        else:
            self.fail("fixture never started on any fixed port")
        self.addCleanup(stop_fixture, process)
        self.assertEqual(announce, f"STARLING_SERVE_LISTENING 127.0.0.1:{port}")
        self.assertTrue(wait_healthy(f"http://127.0.0.1:{port}", process))

    def test_out_of_range_ports_exit_non_zero(self):
        for port in ("70000", "-1"):
            with self.subTest(port=port):
                process = subprocess.run(
                    [str(fixture_binary()), "--model", "parakeet", "--gguf", __file__, "--port", port],
                    capture_output=True, text=True, timeout=10)
                self.assertNotEqual(process.returncode, 0)


class DeferredLoad(unittest.TestCase):
    """--no-eager-load + POST /warmup + the additive /health fields."""

    @classmethod
    def setUpClass(cls):
        # start_fixture stops its own process on failure, but guard anyway:
        # if it ever raises without cleaning up, nothing may leak.
        cls.process = None
        try:
            cls.base, cls.process = start_fixture("parakeet", ["--no-eager-load"])
        except Exception:
            if cls.process is not None:
                stop_fixture(cls.process)
            raise
        cls.addClassCleanup(stop_fixture, cls.process)

    def health(self):
        with urllib.request.urlopen(self.base + "/health", timeout=5) as response:
            return json.load(response)

    def test_starts_unloaded_and_introspectable(self):
        health = self.health()
        self.assertFalse(health["loaded"])
        self.assertFalse(health["warm"])
        self.assertIsNone(health["load_error"])
        # The compile-time backend family before a load; the fixture always
        # reports contract-fixture.
        self.assertEqual(health["backend"], "contract-fixture")

    def test_warmup_loads_deferred_model_then_warms(self):
        request = urllib.request.Request(self.base + "/warmup", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 202)
            accepted = json.load(response)
        self.assertEqual(accepted["status"], "warmup started")
        self.assertIn(accepted["phase"], ("loading", "ready"))
        deadline = time.monotonic() + 10
        health = self.health()
        while not (health["loaded"] and health["warm"]):
            if time.monotonic() > deadline:
                self.fail(f"model did not load and warm up: {health}")
            time.sleep(0.05)
            health = self.health()
        self.assertEqual(health["backend"], "contract-fixture")
        self.assertIsNone(health["load_error"])


class ParentWatchdog(unittest.TestCase):
    """--parent-pid: the server exits cleanly when the parent disappears."""

    def test_server_exits_when_parent_dies(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(self.kill_sleeper, sleeper)
        base, process = start_fixture("parakeet", ["--parent-pid", str(sleeper.pid)])
        # Close the stdout pipe however the assertions below turn out, not
        # only on success (a failing assert would otherwise leak the pipe
        # until garbage collection).
        self.addCleanup(process.stdout.close)
        # The server is healthy while the parent lives.
        self.assertTrue(wait_healthy(base, process))
        self.assertEqual(process.poll(), None)
        sleeper.kill()
        sleeper.wait(timeout=5)
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(process.poll(), 0, "server must exit with code 0 within 5 s of the parent dying")

    @staticmethod
    def kill_sleeper(sleeper):
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
