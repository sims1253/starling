"""Local speech recognition with CUDA-graph pipelines and native ggml engines."""

__all__ = ["FusedEncoder"]


def __getattr__(name: str):
    # GPU lock tools must work without importing the inference stack.
    if name == "FusedEncoder":
        from .granite.encoder_mega import FusedEncoder

        globals()[name] = FusedEncoder
        return FusedEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
