"""Deterministic stub server for the experiment-runner TESTS only.

A minimal starling-serve HTTP double: GET /health reports loaded, POST
/inference sleeps STUB_DELAY_MS and returns a fixed transcript. It exists
to exercise the runner's protocol (fresh processes, cold/warm split,
timeouts, records) in CI without models or GPUs. It is NOT benchmark
evidence: the demo command refuses to use it, and any record produced
against it says so via its binary provenance.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import sys
import time

DELAY_MS = float(os.environ.get("STUB_DELAY_MS", "0"))


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence — the runner keeps its own logs
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok", "loaded": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/inference":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        while length > 0:
            chunk = self.rfile.read(min(length, 65536))
            if not chunk:
                break
            length -= len(chunk)
        if DELAY_MS > 0:
            time.sleep(DELAY_MS / 1000.0)
        body = json.dumps({"text": "stub", "duration_s": 0.0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", port if port else 0))
        actual = s.getsockname()[1]
    # health must answer even while a request sleeps: serve per-request threads.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", actual), Handler)
    print(f"stub listening on {actual}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
