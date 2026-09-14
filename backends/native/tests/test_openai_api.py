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


class OpenAIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = Path(os.environ.get("STARLING_CONTRACT_BIN", ROOT / "build/native-cpu/starling-serve-contract-fixture"))
        if not binary.is_file():
            raise RuntimeError(f"Build the contract fixture first: {binary}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen([str(binary), "--model", "parakeet", "--gguf", __file__, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.addClassCleanup(cls.stop)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(cls.base + "/health", timeout=1).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("Fixture server did not start")

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


if __name__ == "__main__":
    unittest.main()
