"""Starling variant matrix for the SONAR harness.

A "variant" is one point in the (model x implementation/backend x quantization)
grid. :mod:`run_sonar` turns each variant into its own ``starling-serve``
process, so variants are process-isolated and their numbers are attributable to
exactly one engine + one GGUF.

Only the native ggml ``starling-serve`` path is wired here, because that is what
this box can run (no NVIDIA GPU) and what carries the quantization ladder. The
adapter itself is backend-agnostic, so a CUDA build or ``python -m
starling.server`` drops in by adding binary/backend entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GGUF_DIR = REPO_ROOT / "models"

# Native serve binaries built in this checkout, keyed by the ggml backend they
# are compiled against (see ``starling-serve --version``). Add CUDA/HIP/Metal
# builds here on hardware that has them.
DEFAULT_BINARIES: dict[str, Path] = {
    "vulkan": REPO_ROOT / "build-ar-vk" / "starling-serve",
    "cpu": REPO_ROOT / "build-ar-cpu" / "starling-serve",
}

# model slug (starling-serve --model) -> {quant tag: gguf filename}
QUANT_LADDERS: dict[str, dict[str, str]] = {
    "parakeet": {
        "bf16": "parakeet-tdt-0.6b-v3-bf16-exact.gguf",
        "q8_0": "parakeet-tdt-0.6b-v3-q8_0.gguf",
        "q6k_imx": "parakeet-tdt-0.6b-v3-q6k-imx.gguf",
        "q5_0": "parakeet-tdt-0.6b-v3-q5_0.gguf",
        "q4_0": "parakeet-tdt-0.6b-v3-q4_0.gguf",
        "q4_imx": "parakeet-tdt-0.6b-v3-q4-fullimx.gguf",
        "q2_imx": "parakeet-tdt-0.6b-v3-q2-fullimx.gguf",
    },
    "moss": {
        "bf16": "moss-transcribe-preview-2b-bf16-exact.gguf",
        "q8_0": "moss-transcribe-preview-2b-q8_0.gguf",
        "q4e8_imx": "moss-transcribe-preview-2b-q4e8-fullimx.gguf",
        "q4e4_imx": "moss-transcribe-preview-2b-q4e4-fullimx.gguf",
        "q2e4_imx": "moss-transcribe-preview-2b-q2e4-fullimx.gguf",
    },
}

# Named starting points. "quants" answers "does quantization cost accuracy?";
# "backends" answers "does the implementation change the transcript?".
PRESETS: dict[str, dict[str, list[str]]] = {
    "smoke": {"models": ["parakeet"], "quants": ["bf16", "q4_0"], "backends": ["vulkan"]},
    "quants": {
        "models": ["parakeet"],
        "quants": ["bf16", "q8_0", "q6k_imx", "q5_0", "q4_0", "q4_imx", "q2_imx"],
        "backends": ["vulkan"],
    },
    "backends": {"models": ["parakeet"], "quants": ["bf16", "q4_0"], "backends": ["vulkan", "cpu"]},
    "moss_quants": {
        "models": ["moss"],
        "quants": ["bf16", "q8_0", "q4e8_imx", "q4e4_imx", "q2e4_imx"],
        "backends": ["vulkan"],
    },
}
DEFAULT_PRESET = "quants"


@dataclass(frozen=True)
class StarlingVariant:
    """One (model, backend, quant) cell, ready to launch as a server."""

    label: str
    model_slug: str
    gguf: Path
    backend: str
    binary: Path

    @property
    def sonar_name(self) -> str:
        """Registry/model name SONAR uses (and names its artifacts after)."""
        return f"starling_{self.label}"

    @property
    def describe(self) -> str:
        """Human-readable snapshot recorded in the scores artifact."""
        return f"{self.model_slug} | {self.gguf.name} | {self.backend}"


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def build_variants(
    preset: str = DEFAULT_PRESET,
    *,
    models: list[str] | None = None,
    quants: list[str] | None = None,
    backends: list[str] | None = None,
    gguf_dir: Path = DEFAULT_GGUF_DIR,
    binaries: dict[str, Path] | None = None,
) -> list[StarlingVariant]:
    """Resolve a preset plus overrides into concrete variants.

    ``models``/``quants``/``backends`` replace the preset's lists when given.
    Raises ``ValueError`` naming the bad key rather than producing a silently
    empty sweep.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    spec = PRESETS[preset]
    models = models if models else spec["models"]
    backends = backends if backends else spec["backends"]
    binaries = binaries or DEFAULT_BINARIES

    variants: list[StarlingVariant] = []
    for slug in models:
        ladder = QUANT_LADDERS.get(slug)
        if ladder is None:
            raise ValueError(f"no quant ladder for model {slug!r}; have {sorted(QUANT_LADDERS)}")
        selected_quants = quants if quants else spec["quants"]
        for quant in selected_quants:
            if quant not in ladder:
                raise ValueError(f"model {slug!r} has no quant {quant!r}; have {sorted(ladder)}")
            gguf = (gguf_dir / ladder[quant]).expanduser()
            if not gguf.is_file():
                raise FileNotFoundError(f"GGUF not found for {slug}/{quant}: {gguf}")
            for backend in backends:
                binary = Path(binaries.get(backend, "")).expanduser()
                if not binary.is_file():
                    raise FileNotFoundError(
                        f"no starling-serve binary for backend {backend!r} at {binary}; "
                        f"build it or override --binary {backend}=/path/to/starling-serve"
                    )
                label = _sanitize(f"{slug}_{backend}_{quant}")
                variants.append(
                    StarlingVariant(
                        label=label,
                        model_slug=slug,
                        gguf=gguf,
                        backend=backend,
                        binary=binary,
                    )
                )
    return variants
