"""Real HTTP contract tests with a deterministic engine, no GPU/model downloads.

Build starling-serve-contract-fixture, then run unittest discover in this folder.
"""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
import uuid
import wave

ROOT = Path(__file__).resolve().parents[3]
RAW = "5. Keep auth.\n6. I'd prefer to never merge this.\n7. I like orange, err, yellow.\n8. A. Agreed. café 🎙"


def wav(rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\0\0" * 160)
    return output.getvalue()


def start_fixture(model):
    """Spawn the contract fixture with `model` and wait for /health.

    Returns (base_url, process); the caller terminates the process.
    """
    binary = Path(os.environ.get("STARLING_CONTRACT_BIN", ROOT / "build/native-cpu/starling-serve-contract-fixture"))
    if not binary.is_file():
        raise RuntimeError(f"Build the contract fixture first: {binary}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen([str(binary), "--model", model, "--gguf", __file__, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base + "/health", timeout=1).close()
            return base, process
        except OSError:
            time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError("Fixture server did not start")


class OpenAIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.process = start_fixture("parakeet")
        cls.addClassCleanup(cls.stop)

    @classmethod
    def stop(cls):
        cls.process.terminate()
        cls.process.wait(timeout=5)

    def request(self, fields=None, audio=None, path="/v1/audio/transcriptions", file_name="file"):
        boundary = uuid.uuid4().hex
        body = bytearray()
        for key, value in fields if fields is not None else [("model", "parakeet")]:
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{file_name}"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        body += audio if audio is not None else wav()
        body += f'\r\n--{boundary}--\r\n'.encode()
        req = urllib.request.Request(self.base + path, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Request-Id": uuid.uuid4().hex})
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read().decode()

    def test_json_preserves_raw_text(self):
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"text": RAW})
        self.assertTrue(headers["X-Request-Id"])

    def test_plain_text(self):
        status, headers, body = self.request([("model", "parakeet"), ("response_format", "text")])
        self.assertEqual((status, body), (200, RAW))
        self.assertIn("text/plain", headers["Content-Type"])

    def test_legacy_route_remains_compatible(self):
        status, _, body = self.request([], path="/inference")
        result = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(result["text"], RAW)
        self.assertIn("duration_s", result)
        self.assertIn("segments", result)

    def test_models(self):
        with urllib.request.urlopen(self.base + "/v1/models") as response:
            result = json.load(response)
        self.assertEqual(result["object"], "list")
        self.assertEqual(result["data"][0]["id"], "parakeet")

    def test_missing_or_wrong_model(self):
        for fields, expected in [([], 400), ([("model", "other")], 404)]:
            with self.subTest(fields=fields):
                status, _, body = self.request(fields)
                self.assertEqual(status, expected)
                self.assertEqual(json.loads(body)["error"]["param"], "model")

    def test_unsupported_options_are_explicit(self):
        for option, value in [("prompt", "auth"), ("language", "de"), ("stream", "true"), ("temperature", "0.7"), ("response_format", "verbose_json"), ("timestamp_granularities[]", "word")]:
            with self.subTest(option=option):
                status, _, body = self.request([("model", "parakeet"), (option, value)])
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"]["param"], option)

    def test_supported_defaults(self):
        self.assertEqual(self.request([("model", "parakeet"), ("stream", "false"), ("temperature", "0")])[0], 200)

    def test_rejects_duplicate_fields(self):
        self.assertEqual(self.request([("model", "parakeet"), ("model", "parakeet")])[0], 400)

    def test_requires_file_part(self):
        self.assertEqual(self.request(file_name="audio")[0], 400)

    def test_compressed_audio_is_not_reinterpreted_as_pcm(self):
        status, _, body = self.request(audio=b"ID3 unsupported mp3 data")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "file")

    def test_wrong_sample_rate_has_standard_error_envelope(self):
        status, _, body = self.request(audio=wav(48000))
        self.assertEqual(status, 400)
        self.assertIn("sample rate", json.loads(body)["error"]["message"])

    def test_capabilities_are_honest(self):
        with urllib.request.urlopen(self.base + "/v1/starling/capabilities") as response:
            result = json.load(response)
        self.assertFalse(result["prompt"])
        self.assertFalse(result["word_timestamps"])
        self.assertEqual(result["sample_rate_hz"], 16000)


class NormalizeContract(unittest.TestCase):
    """POST /normalize transport on the fixture s1 text model.

    The engine double echoes the decoded transcript back, so the JSON string
    reader's \\u decoding is pinned byte-for-byte without a real model: the
    raw-UTF-8 and ASCII-escaped forms of the same string must decode to the
    same transcript bytes (issue #123 — surrogate halves used to be decoded
    independently, emitting invalid UTF-8).
    """

    @classmethod
    def setUpClass(cls):
        cls.base, cls.process = start_fixture("s1")
        cls.addClassCleanup(cls.stop)

    @classmethod
    def stop(cls):
        cls.process.terminate()
        cls.process.wait(timeout=5)

    def request(self, body):
        req = urllib.request.Request(self.base + "/normalize", data=body.encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_raw_and_escaped_supplementary_chars_decode_identically(self):
        # Emoji U+1F600 and supplementary-plane CJK U+20BB7, raw and as
        # \uD8XX\uDXXX surrogate-pair escapes: identical decoded transcripts.
        # ensure_ascii=False keeps the first body raw UTF-8 (json.dumps would
        # otherwise \u-escape it too, exercising only the escape path).
        expected = "hi 😀 𠮷 bye"
        status_raw, body_raw = self.request(json.dumps({"transcript": expected}, ensure_ascii=False))
        status_escaped, body_escaped = self.request('{"transcript": "hi \\ud83d\\ude00 \\ud842\\udfb7 bye"}')
        self.assertEqual((status_raw, status_escaped), (200, 200))
        self.assertEqual(body_raw["text"], expected)
        self.assertEqual(body_escaped["text"], body_raw["text"])

    def test_bmp_escapes_and_simple_escapes(self):
        status, body = self.request('{"transcript": "caf\\u00e9 line\\nbreak \\"quoted\\" \\\\ok"}')
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], 'café line\nbreak "quoted" \\ok')

    def test_unpaired_surrogates_decode_to_replacement_character(self):
        for escaped, expected in [
            ('"a\\ud83db"', '"a\\ufffdb"'),   # lone high surrogate
            ('"x\\ude00y"', '"x\\ufffdy"'),   # lone low surrogate
        ]:
            with self.subTest(escaped=escaped):
                status, body = self.request('{"transcript": ' + escaped + "}")
                self.assertEqual(status, 200)
                self.assertEqual(body["text"], json.loads(expected))

    def test_malformed_escape_is_rejected(self):
        status, body = self.request('{"transcript": "bad \\uZZZZ"}')
        self.assertEqual(status, 400)
        self.assertIn("transcript", body["error"])


if __name__ == "__main__":
    unittest.main()
