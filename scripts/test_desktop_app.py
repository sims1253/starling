#!/usr/bin/env python3
"""Exercise the real desktop web UI and HTTP adapter with a test engine.

Run from the checkout after pnpm install and building starling-serve-contract-fixture:
  uv run --no-project --with playwright python scripts/test_desktop_app.py
Install Chromium with the same uv environment's `python -m playwright install chromium`.
"""
from contextlib import ExitStack
import io
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
import wave

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RAW = "5. Keep auth.\n6. I'd prefer to never merge this.\n7. I like orange, err, yellow.\n8. A. Agreed. café 🎙"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(url, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before {url} became ready")
        try:
            urllib.request.urlopen(url, timeout=1).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {url}")


def stop(process):
    if process.poll() is None:
        if os.name != "nt" and os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def audio_file():
    """A noncanonical stereo/48k recording exercises the client resampler."""
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        samples = [int(math.sin(index * math.tau * 440 / 48000) * 2000) for index in range(12000)]
        stream.writeframes(b"".join(struct.pack("<hh", value, value) for value in samples))
    return {"name": "recording.wav", "mimeType": "audio/wav", "buffer": output.getvalue()}


def main():
    binary = Path(os.environ.get("STARLING_CONTRACT_BIN", ROOT / "build/native-cpu/starling-serve-contract-fixture"))
    if not binary.is_file():
        raise RuntimeError("Build starling-serve-contract-fixture before running the app tests")
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise RuntimeError("pnpm is required")
    backend_port, frontend_port = free_port(), free_port()
    with ExitStack() as stack:
        log_directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="starling-app-test-"))
        server_log = stack.enter_context(open(Path(log_directory) / "server.log", "w+"))
        frontend_log = stack.enter_context(open(Path(log_directory) / "frontend.log", "w+"))
        server = subprocess.Popen([str(binary), "--model", "parakeet", "--gguf", __file__, "--port", str(backend_port),
                                   "--min-chunk-seconds", "0.05", "--stream-chunk-seconds", "0.2",
                                   "--stream-overlap-seconds", "0.05", "--partial-interval-seconds", "0.05"], cwd=ROOT, start_new_session=os.name != "nt", stdout=server_log, stderr=subprocess.STDOUT)
        stack.callback(stop, server)
        wait_ready(f"http://127.0.0.1:{backend_port}/health", server)
        environment = {**os.environ, "STARLING_API_TARGET": f"http://127.0.0.1:{backend_port}"}
        frontend = subprocess.Popen([pnpm, "--filter", "@starling/desktop", "dev", "--host", "127.0.0.1", "--port", str(frontend_port)], cwd=ROOT, start_new_session=os.name != "nt", env=environment, stdout=frontend_log, stderr=subprocess.STDOUT)
        stack.callback(stop, frontend)
        try:
            wait_ready(f"http://127.0.0.1:{frontend_port}", frontend)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, executable_path=os.environ.get("STARLING_BROWSER_EXECUTABLE"), args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])
                context = browser.new_context(permissions=["microphone", "clipboard-read", "clipboard-write"], viewport={"width": 1180, "height": 760})
                page = context.new_page()
                failures = []
                page.on("pageerror", lambda error: failures.append(str(error)))
                page.goto(f"http://127.0.0.1:{frontend_port}")
                page.wait_for_load_state("networkidle")
                expect(page.get_by_role("button", name="Start recording")).to_be_visible()
                try:
                    expect(page.locator(".connection")).to_contain_text("ready")
                except AssertionError:
                    print(page.locator("body").inner_text())
                    raise

                page.locator('input[type="file"]').set_input_files(audio_file())
                expect(page.locator(".transcript-body")).to_contain_text(RAW)
                expect(page.locator(".history-row")).to_have_count(1)
                page.get_by_role("button", name="Import an audio file", exact=True).click(trial=True)
                assert "[object Object]" not in page.locator("body").inner_text()
                page.get_by_role("button", name="Copy", exact=True).click()
                assert page.evaluate("navigator.clipboard.readText()") == RAW
                page.reload()
                page.wait_for_load_state("networkidle")
                expect(page.locator(".transcript-body")).to_contain_text(RAW)

                # An upload failure must leave the recording in durable history.
                upload = re.compile(r"/api/(inference|transcribe|v1/audio/transcriptions)$")
                page.route(upload, lambda route: route.abort())
                page.locator('input[type="file"]').set_input_files(audio_file())
                expect(page.locator(".history-row")).to_have_count(2)
                expect(page.get_by_role("button", name="Retry", exact=True)).to_be_enabled()
                page.reload()
                page.wait_for_load_state("networkidle")
                expect(page.locator(".history-row")).to_have_count(2)
                expect(page.get_by_role("button", name="Retry", exact=True)).to_be_enabled()
                page.unroute(upload)
                page.get_by_role("button", name="Retry", exact=True).click()
                expect(page.locator(".transcript-body")).to_contain_text(RAW)

                # Two uploads may overlap. Changing server settings must not
                # run crash recovery on either live request, and finishing one
                # must not enable deletion of the other recording.
                pending = []
                page.route(upload, lambda route: pending.append(route))
                page.locator('input[type="file"]').set_input_files(audio_file())
                expect(page.locator(".history-row")).to_have_count(3)
                expect(page.locator(".take-state.transcribing")).to_have_count(1)
                page.locator('input[type="file"]').set_input_files(audio_file())
                expect(page.locator(".history-row")).to_have_count(4)
                expect(page.locator(".take-state.transcribing")).to_have_count(2)
                page.get_by_role("button", name="Open server settings").click()
                page.get_by_label(re.compile(r"^API format")).select_option("openai")
                page.get_by_role("button", name="Save settings", exact=True).click()
                expect(page.locator(".connection")).to_contain_text("ready")
                expect(page.locator(".take-state.transcribing")).to_have_count(2)
                for row_index in [1, 0]:
                    page.locator(".history-row").nth(row_index).click()
                    expect(page.get_by_role("button", name="Delete saved recording")).to_be_disabled()
                assert len(pending) == 2
                pending[0].continue_()
                expect(page.locator(".take-state.transcribing")).to_have_count(1)
                expect(page.get_by_role("button", name="Delete saved recording")).to_be_disabled()
                pending[1].continue_()
                expect(page.locator(".take-state.transcribing")).to_have_count(0)
                expect(page.locator(".transcript-body")).to_contain_text(RAW)
                page.unroute(upload)

                # The overlap block saved settings with the OpenAI API format;
                # live streaming is Starling-API-only, so restore it first.
                page.get_by_role("button", name="Open server settings").click()
                page.get_by_label(re.compile(r"^API format")).select_option("starling")
                page.get_by_role("button", name="Save settings", exact=True).click()

                # Fake microphone supplies real PCM through the browser capture
                # path. Live streaming is on by default for the Starling API:
                # partials appear while recording and Stop commits the stream.
                page.get_by_role("button", name="Start recording").click()
                expect(page.get_by_role("button", name="Stop recording")).to_be_visible()
                expect(page.locator(".live-stream.live")).to_be_visible()
                expect(page.locator(".live-text")).to_contain_text(RAW, timeout=10_000)
                page.wait_for_timeout(700)  # Collect several audio processing buffers.
                page.get_by_role("button", name="Stop recording").click()
                expect(page.locator(".history-row")).to_have_count(5)
                expect(page.locator(".transcript-body")).to_contain_text(RAW)

                # A streaming socket that cannot connect must degrade to the
                # batch upload of the saved WAV: same transcript, nothing lost.
                # (A page-level WebSocket stub keeps this deterministic across
                # browsers; network-level WS routing is not universally wired.)
                page.evaluate("""() => {
                    window.realWebSocket = window.WebSocket;
                    window.WebSocket = function (url, protocols) {
                        if (String(url).includes("/stream")) {
                            // Fail like an unreachable server: error, then close, never open.
                            const socket = new EventTarget();
                            setTimeout(() => {
                                socket.dispatchEvent(new Event("error"));
                                socket.dispatchEvent(new CloseEvent("close", { code: 1006 }));
                            }, 0);
                            return Object.assign(socket, { readyState: 0, send() {}, close() {} });
                        }
                        return protocols === undefined
                            ? new window.realWebSocket(url)
                            : new window.realWebSocket(url, protocols);
                    };
                }""")
                page.get_by_role("button", name="Start recording").click()
                expect(page.get_by_role("button", name="Stop recording")).to_be_visible()
                expect(page.locator(".live-stream.unavailable")).to_be_visible()
                page.wait_for_timeout(500)
                page.get_by_role("button", name="Stop recording").click()
                expect(page.locator(".history-row")).to_have_count(6)
                expect(page.locator(".transcript-body")).to_contain_text(RAW)
                page.evaluate("() => { window.WebSocket = window.realWebSocket; }")

                # Storage failure must leave every take recoverable, even if
                # a later recording saves successfully or a download is cancelled.
                page.evaluate("""() => {
                    window.savedIdbAdd = IDBObjectStore.prototype.add;
                    IDBObjectStore.prototype.add = function () {
                        throw new DOMException('Test storage is full', 'QuotaExceededError');
                    };
                }""")
                for number in [1, 2]:
                    page.locator('input[type="file"]').set_input_files(audio_file())
                    expect(page.get_by_role("button", name=f"Download WAV {number}", exact=True)).to_be_visible()
                expect(page.locator(".history-row")).to_have_count(6)
                page.evaluate("() => { IDBObjectStore.prototype.add = window.savedIdbAdd; }")
                page.locator('input[type="file"]').set_input_files(audio_file())
                expect(page.locator(".history-row")).to_have_count(7)
                expect(page.locator(".transcript-body")).to_contain_text(RAW)
                with page.expect_download() as download_event:
                    page.get_by_role("button", name="Download WAV 1", exact=True).click()
                downloaded = Path(download_event.value.path()).read_bytes()
                assert downloaded[:4] == b"RIFF" and len(downloaded) > 44
                expect(page.get_by_role("button", name="Download WAV 1", exact=True)).to_be_visible()
                expect(page.get_by_role("button", name="Download WAV 2", exact=True)).to_be_visible()
                page.once("dialog", lambda dialog: dialog.accept())
                page.get_by_role("button", name="Discard unsaved", exact=True).click()
                expect(page.get_by_role("button", name="Download WAV 1", exact=True)).to_have_count(0)
                screenshot = os.environ.get("STARLING_APP_SCREENSHOT")
                if screenshot:
                    page.screenshot(path=screenshot, full_page=True, animations="disabled")
                assert not failures, "Unhandled browser exceptions: " + "; ".join(failures)
                browser.close()
            print("Desktop browser checks passed: import/resample, raw text, copy, durable reload, failed upload/retry, overlapping uploads, settings changes, microphone capture with live streaming and blocked-socket fallback, storage failure recovery")
        except BaseException:
            frontend_log.seek(0)
            print(frontend_log.read()[-4000:])
            raise


if __name__ == "__main__":
    main()
