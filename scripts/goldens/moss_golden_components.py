"""Capture staged, self-describing MOSS-Transcribe reference goldens.

This deliberately uses ``moss_reference.py`` for the audio-encoder and
embedding-merge stages. ``last_prefill_logits`` is the prefill portion of
``greedy_generate`` verbatim, so its logits are the distribution whose argmax
seeds that reference decoder.

The capture device defaults to CUDA when available and falls back to CPU
(``STARLING_GOLDEN_DEVICE=cpu|cuda`` overrides); the chosen device and library
versions are recorded in each ``moss_<fixture>_meta.json`` so a component run
can prove WHICH reference it was measured against (issue #167). The captured
ids are asserted equal to ``moss_<fixture>_ids.pt`` (the ``moss_golden.py``
reference) before anything is written — a component capture that disagrees
with the text-golden reference aborts rather than staging a second,
contradicting reference.

The 21 staged files are published ATOMICALLY: everything is written to
``golden/.staging_components/`` first and only moved into ``golden/`` after
every fixture has been captured and verified, so a failed or interrupted run
can never leave a mixed half-old/half-new reference set for
``scripts/goldens/golden_to_raw.py`` to consume. Existing staged files are only
overwritten with ``--force`` (a reference update is a separately reviewed
change, mirroring ``moss_golden.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from moss_reference import (
    REFERENCE_PATH,
    REPO,
    audio_features,
    build_inputs_embeds,
    capture_device,
    greedy_generate,
    last_prefill_logits,
    load_model_and_processor,
    load_wav,
    repo_revision,
)

NAMES = ("short", "medium", "long")
STAGES = ("mel", "encoder_hidden", "audio_embeds", "prompt_ids", "inputs_embeds", "prefill_logits")


def cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def shape_dtype(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def verify_saved(gdir: Path) -> None:
    expected = [gdir / f"moss_{name}_{stage}.pt" for name in NAMES for stage in STAGES]
    expected += [gdir / f"moss_{name}_meta.json" for name in NAMES]
    assert len(expected) == 21, f"expected 21 component files, got {len(expected)}"
    missing = [str(path) for path in expected if not path.is_file()]
    assert not missing, f"missing goldens: {missing}"
    for path in expected:
        if path.suffix == ".pt":
            value = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(value, torch.Tensor), f"{path} did not reload as a tensor"
        else:
            json.loads(path.read_text())


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing staged component goldens (default: refuse)")
    args = ap.parse_args()

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)

    existing = [gdir / f"moss_{name}_{stage}.pt"
                for name in NAMES for stage in STAGES]
    existing += [gdir / f"moss_{name}_meta.json" for name in NAMES]
    existing = [p for p in existing if p.exists()]
    if existing and not args.force:
        print("[moss-components] REFUSING to overwrite existing staged goldens "
              "(no --force):")
        for p in existing:
            print(f"  {p}")
        return 3

    rows: list[tuple[str, str, str, str]] = []

    import shutil

    device = capture_device()
    capture_provenance = {
        "device": device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "repo_revision": repo_revision(),
        "reference_path": REFERENCE_PATH,
    }
    print(f"[moss-components] loading model ... (device={device})")

    # Staging directory: the capture is only published into golden/ after ALL
    # fixtures verify, so a mid-run failure can never leave a mixed
    # half-old/half-new reference set (greptile P2).
    sdir = gdir / ".staging_components"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)

    def _capture() -> None:
        model, proc = load_model_and_processor(device=device)
        for name in NAMES:
            wav, sr = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
            seconds = wav.shape[0] / sr
            raw = proc(wav.numpy())
            # Preserve the processor's tensor (bf16) for the reference forward.
            inp = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in raw.items()}

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                encoder_hidden = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
                inputs_embeds = build_inputs_embeds(model, inp["input_ids"], encoder_hidden, inp["audio_input_mask"])
                # This is exactly the adapter output consumed by masked_scatter_ above.
                audio_embeds = model.model.audio_adapter(encoder_hidden)
                prefill_logits = last_prefill_logits(model, inputs_embeds)
                ids = greedy_generate(model, inputs_embeds, max_new_tokens=200, max_cache_len=2048)
            if device == "cuda":
                torch.cuda.synchronize()

            expected_ids = torch.load(gdir / f"moss_{name}_ids.pt", map_location="cpu", weights_only=True)
            assert torch.equal(ids.cpu(), expected_ids), (
                f"{name}: component capture diverged from moss_{name}_ids.pt "
                f"(got {ids.tolist()}, expected {expected_ids.tolist()})"
            )
            first_token = int(prefill_logits.argmax(dim=-1).item())
            assert first_token == int(ids[0, 0]), f"{name}: prefill argmax does not seed greedy decode"

            # The processor feeds bf16 to the bf16 model.  Persist mel as fp32 as
            # the C++ frontend gate, retaining those exact bf16-quantized values.
            saved = {
                "mel": cpu_float(inp["audio_data"]),
                "encoder_hidden": cpu_float(encoder_hidden),
                "audio_embeds": cpu_float(audio_embeds),
                "prompt_ids": inp["input_ids"].detach().to(device="cpu", dtype=torch.int64).contiguous(),
                "inputs_embeds": cpu_float(inputs_embeds),
                "prefill_logits": cpu_float(prefill_logits),
            }
            for stage, tensor in saved.items():
                torch.save(tensor, sdir / f"moss_{name}_{stage}.pt")
                rows.append((name, stage, str(tuple(tensor.shape)), str(tensor.dtype).removeprefix("torch.")))

            meta = {
                "model": "MOSS-Transcribe-preview-2B",
                "fixture": f"tests/fixtures/{name}.wav",
                "seconds": seconds,
                "mel_frontend": {"sample_rate": 16000, "mel_dim": 128, "n_fft": 640, "hop_length": 160},
                "audio_data_seqlens": inp["audio_data_seqlens"].detach().cpu().tolist(),
                "audio_input_mask_sum": int(inp["audio_input_mask"].sum().item()),
                "n_audio_tokens": int(inp["audio_input_mask"].sum().item()),
                "first_generated_token_id": first_token,
                "reference_ids_file": f"moss_{name}_ids.pt",
                "capture": capture_provenance,
                "tensors": {stage: shape_dtype(tensor) for stage, tensor in saved.items()},
            }
            (sdir / f"moss_{name}_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            elapsed = time.perf_counter() - t0
            print(f"[moss-components] {name}: ids verified, {elapsed:.1f}s")

    try:
        _capture()
        # Publish only a fully verified, complete capture (atomic w.r.t.
        # failures: nothing in golden/ changes unless all 21 files verify).
        verify_saved(sdir)
        for name in NAMES:
            for stage in STAGES:
                shutil.move(str(sdir / f"moss_{name}_{stage}.pt"),
                            str(gdir / f"moss_{name}_{stage}.pt"))
            shutil.move(str(sdir / f"moss_{name}_meta.json"),
                        str(gdir / f"moss_{name}_meta.json"))
        print("[moss-components] published 21 verified files into golden/")
    finally:
        if sdir.exists():
            shutil.rmtree(sdir)

    verify_saved(gdir)
    print("\nfixture  stage            shape                 dtype")
    print("-------  ---------------  --------------------  -------")
    for name, stage, shape, dtype in rows:
        print(f"{name:<7}  {stage:<15}  {shape:<20}  {dtype}")
    print("\n[moss-components] verified 21 component files reload correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
