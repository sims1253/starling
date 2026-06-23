"""Environment verification for the starling bootstrap.

Checks: python version, torch + cuda + arch + device, FP8 capabilities, and that
transformers (from git source) can load the nvidia/parakeet-tdt-0.6b-v3 config
(i.e. the `parakeet_tdt` model type is registered). Prints every dim that drives
kernel sizing in later phases.

This script MUST NOT die silently: every stage is wrapped so any exception is
printed clearly.
"""

from __future__ import annotations

import sys
import traceback


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    _section("PYTHON")
    print(f"python version: {sys.version}")
    print(f"executable    : {sys.executable}")


def check_torch() -> None:
    _section("TORCH / CUDA / DEVICE")
    import torch  # noqa: WPS433

    print(f"torch version   : {torch.__version__}")
    print(f"torch cuda ver  : {torch.version.cuda}")
    print(f"cuda arch list  : {torch.cuda.get_arch_list()}")
    print(f"is_available    : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False -- no CUDA device visible")
    dev_name = torch.cuda.get_device_name(0)
    print(f"device name     : {dev_name}")
    if "5090" not in dev_name:
        print(f"WARNING: expected RTX 5090, got {dev_name!r}")
    else:
        print("OK: RTX 5090 detected")

    _section("FP8 CAPABILITIES")
    has_scaled_mm = hasattr(torch, "_scaled_mm")
    has_fp8_dtype = hasattr(torch, "float8_e4m3fn")
    print(f"has torch._scaled_mm       : {has_scaled_mm}")
    print(f"has torch.float8_e4m3fn    : {has_fp8_dtype}")
    if not (has_scaled_mm and has_fp8_dtype):
        raise RuntimeError("FP8 capabilities missing -- later-phase FP8 work will break")


def check_transformers() -> None:
    _section("TRANSFORMERS + PARAKEET CONFIG")
    import transformers  # noqa: WPS433

    print(f"transformers version: {transformers.__version__}")

    from transformers import AutoConfig  # noqa: WPS433

    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    print(f"loading config for: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"config class      : {type(config).__name__}")
    print(f"model_type        : {getattr(config, 'model_type', '<missing>')}")

    # Dims that drive kernel sizing. Use getattr with a safe fallback so missing
    # keys never crash the script.
    keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "conv_dim",
        "kernel_size",
        # TDT-specific fields
        "tdt_num_durations",
        "tdt_decoder_layers",
        "tdt_decoder_hidden",
        "decoder_layers",
        "decoder_hidden",
        # common extras that matter for kernel tiling
        "d_model",
        "encoder_layers",
        "decoder_ffn_dim",
        "feedforward_intermediate",
        "n_head",
        "n_layer",
    ]

    print("\n--- extracted dims (top-level config) ---")
    for key in keys:
        val = getattr(config, key, "<missing>")
        print(f"  {key:30s} = {val!r}")

    # Parakeet-TDT nests the Conformer encoder config under `encoder_config`, so
    # pull the kernel-sizing dims from there too. These drive tile/block sizing
    # in the megakernel phase.
    raw = config.to_dict() if hasattr(config, "to_dict") else {}
    encoder_cfg = raw.get("encoder_config", {}) if isinstance(raw, dict) else {}
    if encoder_cfg:
        enc_keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "conv_kernel_size",
            "subsampling_factor",
            "subsampling_conv_channels",
            "num_mel_bins",
            "subsampling_conv_kernel_size",
            "subsampling_conv_stride",
            "max_position_embeddings",
            "model_type",
        ]
        print("\n--- extracted dims (nested encoder_config / Conformer) ---")
        for key in enc_keys:
            print(f"  {key:30s} = {encoder_cfg.get(key, '<missing>')!r}")

    # Also surface any TDT/duration-ish fields so we don't miss them.
    tdt_like = {k: v for k, v in raw.items() if "tdt" in str(k).lower() or "duration" in str(k).lower()}
    if tdt_like:
        print("\n--- raw TDT/duration-ish fields from config.to_dict() ---")
        for k, v in tdt_like.items():
            print(f"  {k:30s} = {v!r}")


def main() -> int:
    failures: list[str] = []
    for stage in (check_python, check_torch, check_transformers):
        try:
            stage()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{stage.__name__}: {exc!r}")
            print(f"\n[FAIL in {stage.__name__}] {exc!r}")
            traceback.print_exc()

    _section("SUMMARY")
    if failures:
        print("VERIFICATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFICATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
