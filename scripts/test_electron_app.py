#!/usr/bin/env python3
"""Launch the built Electron renderer, exercise its IPC, and measure the shell.

Linux without a display: xvfb-run -a npm run test:electron
Requires npm ci, npm run build, the native contract fixture, and Playwright.
Set STARLING_ELECTRON_EXECUTABLE to smoke-test an unpacked application binary.
"""
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import urllib.request

from playwright.sync_api import expect, sync_playwright
from test_desktop_app import ROOT, RAW, audio_file, free_port, stop, wait_ready


def check_transport_failures(page, stack):
    """Check HTTP status preservation and interruption after response headers."""
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/v1/models":
                payload = b"Test proxy is unavailable"
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"status":')
            self.wfile.flush()
            self.connection.settimeout(3)
            try:
                if self.connection.recv(1) == b"":
                    disconnected.set()
            except ConnectionError:
                disconnected.set()
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    stack.callback(server.server_close)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stack.callback(server.shutdown)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    error = page.evaluate("""async (endpoint) => {
        try {
            await window.starlingDesktop.health({endpoint, protocol: 'openai'});
            return 'unexpected success';
        } catch (cause) { return String(cause); }
    }""", endpoint)
    assert "503" in error and "Test proxy is unavailable" in error, error
    started = time.perf_counter()
    error = page.evaluate("""async (endpoint) => {
        try {
            await window.starlingDesktop.health({endpoint, protocol: 'starling', timeoutMs: 80});
            return 'unexpected success';
        } catch (cause) { return String(cause); }
    }""", endpoint)
    assert "timed out" in error.lower(), error
    assert time.perf_counter() - started < 2, "Timeout did not cover the response body"
    assert disconnected.wait(timeout=2), "Timed-out response body kept its connection open"


def main():
    executable = ROOT / "node_modules/electron/dist/electron"
    if os.name == "nt":
        executable = executable.with_name("electron.exe")
    elif os.uname().sysname == "Darwin":
        executable = ROOT / "node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
    packaged = os.environ.get("STARLING_ELECTRON_EXECUTABLE")
    if packaged:
        executable = Path(packaged).resolve()
    fixture = Path(os.environ.get("STARLING_CONTRACT_BIN", ROOT / "build/native-cpu/starling-serve-contract-fixture"))
    if not executable.is_file() or not fixture.is_file():
        raise RuntimeError("Install Electron and build starling-serve-contract-fixture first")
    with ExitStack() as stack:
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="starling-electron-test-")))
        server_log = stack.enter_context((directory / "server.log").open("w+"))
        electron_log = stack.enter_context((directory / "electron.log").open("w+"))
        api_port, debug_port = free_port(), free_port()
        server = subprocess.Popen([str(fixture), "--model", "parakeet", "--gguf", __file__, "--port", str(api_port)], cwd=ROOT, start_new_session=os.name != "nt", stdout=server_log, stderr=subprocess.STDOUT)
        stack.callback(stop, server)
        endpoint = f"http://127.0.0.1:{api_port}"
        wait_ready(endpoint + "/health", server)
        environment = {**os.environ, "XDG_CONFIG_HOME": str(directory / "config")}
        environment.pop("STARLING_RENDERER_URL", None)
        args = [str(executable), f"--remote-debugging-port={debug_port}", f"--user-data-dir={directory / 'userdata'}"]
        if not packaged:
            args.append(str(ROOT / "apps/desktop/dist-electron/main.mjs"))
        if os.environ.get("STARLING_TEST_NO_SANDBOX") == "1":
            args.insert(1, "--no-sandbox")
        started = time.perf_counter()
        electron = subprocess.Popen(args, cwd=ROOT, env=environment, start_new_session=os.name != "nt", stdout=electron_log, stderr=subprocess.STDOUT)
        stack.callback(stop, electron)
        try:
            wait_ready(f"http://127.0.0.1:{debug_port}/json/version", electron)
            with sync_playwright() as playwright:
                browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{debug_port}")
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else context.wait_for_event("page")
                page.wait_for_load_state("networkidle")
                expect(page.get_by_role("button", name="Start recording")).to_be_visible()
                launch_to_ui_ms = round((time.perf_counter() - started) * 1000)
                assert page.evaluate("typeof window.starlingDesktop?.transcribe") == "function"
                assert page.evaluate("typeof require") == "undefined", "Renderer must not expose Node"
                page.get_by_role("button", name="Open server settings").click()
                page.get_by_label("Server endpoint", exact=True).fill(endpoint)
                page.get_by_label(re.compile(r"^API format")).select_option("openai")
                page.get_by_label("Model", exact=True).fill("parakeet")
                page.get_by_role("button", name="Save settings", exact=True).click()
                expect(page.locator(".connection")).to_contain_text("ready")
                page.locator('input[type="file"]').set_input_files(audio_file())
                expect(page.locator(".transcript-body")).to_contain_text(RAW)
                page.get_by_role("button", name="Import an audio file", exact=True).click(trial=True)
                check_transport_failures(page, stack)
                diagnostic = None if packaged else page.evaluate("window.starlingDesktop.diagnostics()")
                report = {"launch_to_ui_ms": launch_to_ui_ms, "shell": diagnostic,
                          "scope": ("Packaged app" if packaged else "Development Electron shell") + ", no inference model; Xvfb if supplied; launch_to_ui includes CDP polling and network-idle wait"}
                print(json.dumps(report, indent=2))
                output = os.environ.get("STARLING_ELECTRON_REPORT")
                if output:
                    Path(output).write_text(json.dumps(report, indent=2) + "\n")
                # Close the CDP connection; process cleanup belongs to ExitStack.
                browser.close()
            print("Electron checks passed: bundled renderer, sandboxed preload, OpenAI multipart via IPC, exact transcript, HTTP errors, stalled-body timeout and connection cleanup")
        except BaseException:
            electron_log.seek(0)
            print(electron_log.read()[-6000:])
            raise


if __name__ == "__main__":
    main()
