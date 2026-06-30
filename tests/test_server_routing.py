"""CPU-safe tests for the unified server's model routing.

These exercise the pure, GPU-free parts of :mod:`starling.server`: the slug ->
backend class resolution, config plumbing, the CLI arg parser, and error
handling for unknown models. No model is loaded and no CUDA is required, so the
test runs in any environment (CI / CPU-only).

The end-to-end transcribe paths are covered by the per-model pipeline tests
(``test_smoke``, ``test_qwen3_pipeline``, ...) and the live ``test_server.py``
script; here we only assert routing wires up the right backend for each slug.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from starling.server import (  # noqa: E402
    MODEL_SLUGS,
    ArkBackend,
    CohereBackend,
    GraniteBackend,
    HiggsBackend,
    MossBackend,
    ParakeetBackend,
    Qwen3Backend,
    ServerConfig,
    StarlingServer,
    get_backend,
)


def test_model_slugs_are_the_supported_set() -> None:
    assert set(MODEL_SLUGS) == {
        "granite", "parakeet", "moss", "qwen3", "ark", "cohere", "higgs",
    }


def test_get_backend_resolves_each_slug_to_the_right_class() -> None:
    cfg = ServerConfig()
    assert isinstance(get_backend("granite", cfg), GraniteBackend)
    assert isinstance(get_backend("parakeet", cfg), ParakeetBackend)
    assert isinstance(get_backend("moss", cfg), MossBackend)
    assert isinstance(get_backend("qwen3", cfg), Qwen3Backend)
    assert isinstance(get_backend("ark", cfg), ArkBackend)
    assert isinstance(get_backend("cohere", cfg), CohereBackend)
    assert isinstance(get_backend("higgs", cfg), HiggsBackend)


def test_get_backend_unknown_slug_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_backend("whisper", ServerConfig())


def test_backend_slug_round_trips() -> None:
    """Each backend class advertises the slug that resolves to it."""
    cfg = ServerConfig()
    for slug in MODEL_SLUGS:
        assert get_backend(slug, cfg).slug == slug


def test_server_config_carries_model_slug() -> None:
    for slug in MODEL_SLUGS:
        srv = StarlingServer(config=ServerConfig(model=slug))
        assert srv.model_slug == slug
        # not loaded yet -> backend is None and load() has not run
        assert srv.backend is None
        assert srv.loaded is False


def test_cli_arg_parser_accepts_each_model() -> None:
    from starling.server import _build_arg_parser

    parser = _build_arg_parser()
    for slug in MODEL_SLUGS:
        args = parser.parse_args(["--model", slug])
        assert args.model == slug
    # default is granite
    assert parser.parse_args([]).model == "granite"


def test_cli_arg_parser_rejects_unknown_model() -> None:
    from starling.server import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "nope"])
