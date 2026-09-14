"""SONAR (``psdn-sonar``) adapter for a running Starling ASR server.

SONAR resolves models through ``psdn_sonar.models.registry``, which ships
adapters only for its own HuggingFace/API models. This module adds a Starling
adapter and a registration helper so ``create_model("starling_<variant>")``
returns one bound to a running server URL.

The server process is owned by :mod:`run_sonar` (one isolated server per
variant); this adapter is a pure HTTP client over the batch endpoint shared by
both servers::

    POST /inference   multipart/form-data "file" -> {"text": ..., ...}

Both ``starling-serve`` (native ggml: cpu / vulkan / cuda / hip / metal) and
``python -m starling.server`` (CUDA-graph pipelines) implement that route, so
one adapter covers every Starling implementation and quantization.

Why HTTP rather than importing Starling in-process: SONAR pins Python 3.10-3.12
plus its own torch/transformers, and the whole point of the native ggml engines
is that they are *not* the Python runtime. A process boundary keeps the two
dependency trees from interfering and matches how Starling is actually served.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests

from psdn_sonar.models.base import ASRModel, LatencyMetrics


class StarlingServerModel(ASRModel):
    """Transcribe by POSTing audio to a running Starling server.

    ``model_snapshot`` is recorded in ``scores_<model>.json`` under the
    submission block; pass a human-readable descriptor (e.g.
    ``parakeet q4_0 / vulkan``) so the artifact identifies the exact engine and
    quant behind the numbers.

    ``transcribe`` returns ``(text, LatencyMetrics)`` so SONAR records the
    adapter's own wall-clock (which includes the HTTP round trip) rather than
    only its outer measurement. Set ``supports_latency_metrics`` accordingly.
    """

    provider = "starling"
    supports_latency_metrics = True

    def __init__(
        self,
        base_url: str,
        model_snapshot: Optional[str] = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider_model_id = model_snapshot
        self.timeout_s = float(timeout_s)
        self._session = requests.Session()

    def transcribe(self, audio_path: str):
        """POST one WAV to ``/inference``; return text + latency, or None on failure.

        Returning ``None`` (rather than raising) matches SONAR's adapter
        contract: one bad clip must not abort a multi-hundred-sample run. The
        cause is left on ``last_transcribe_error`` for the artifact.
        """
        try:
            t0 = time.perf_counter()
            with open(audio_path, "rb") as fh:
                files = {"file": (Path(audio_path).name, fh, "audio/wav")}
                response = self._session.post(
                    f"{self.base_url}/inference",
                    files=files,
                    timeout=self.timeout_s,
                )
            complete_s = time.perf_counter() - t0
            if response.status_code != 200:
                self.last_transcribe_error = (
                    f"HTTP {response.status_code} from {self.base_url}/inference: "
                    f"{response.text[:300]}"
                )
                return None
            text = (response.json().get("text") or "").strip()
            return text, LatencyMetrics(complete_s=complete_s, ttft_s=None)
        except Exception as exc:  # noqa: BLE001 - adapter contract: never raise
            self._record_transcribe_failure(audio_path, exc)
            return None


def register_starling_model(
    name: str,
    base_url: str,
    *,
    model_snapshot: Optional[str] = None,
    timeout_s: float = 600.0,
) -> str:
    """Register ``name`` -> a :class:`StarlingServerModel` bound to ``base_url``.

    Mutates ``registry._MODEL_CONFIGS`` (the only extension point the installed
    package exposes; there is no public ``register_model`` yet). The class path
    is a module name so the registry's lazy ``importlib.import_module`` resolves
    it — this module must therefore be importable as top-level
    ``starling_sonar_adapter``, which is true when the orchestrator is run as a
    script from this directory.
    """
    from psdn_sonar.models import registry

    registry._MODEL_CONFIGS[name] = (
        "starling_sonar_adapter.StarlingServerModel",
        {
            "base_url": base_url,
            "model_snapshot": model_snapshot,
            "timeout_s": timeout_s,
        },
    )
    return name
