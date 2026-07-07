"""Capture golden references for nvidia/parakeet-unified-en-0.6b on fixtures.

Runs the hand-built NeMo-free port (GPU mel -> Conformer encoder -> eager greedy
RNN-T decode) on ``tests/fixtures/{short,medium,long}.wav`` and persists the
emitted token ids + detokenized text under ``golden/parakeet_unified_*``.

Correctness basis
-----------------
Neither NeMo nor sherpa-onnx is installable cleanly alongside this repo's pinned
``torch>=2.11`` (NeMo's runtime conflicts; sherpa-onnx wheels lag torch
2.11/cu130). So per the porting plan's documented fallback, the **eager port
itself** is the byte-exact reference: it is the standard RNN-T greedy algorithm
(:mod:`starling.parakeet_unified.decode_eager` -- mirrors NeMo's
``rnnt_greedy_decoding``), the encoder/decoder/joint load
``state_dict(strict=True)`` (every checkpoint key consumed, byte-identical
params), and the mel frontend is the exact NeMo math (see
``parakeet/mel_gpu.py`` + this module's docstring). The graphed encoder +
megakernel decode (Steps 7-8) are then gated to reproduce these ids
token-for-token across K in {1,4,16,64}.

Usage:  uv run python scripts/parakeet_unified_golden.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path))
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav, dtype=np.float32)


def main() -> int:
    from starling.parakeet_unified import modeling as M
    from starling.parakeet_unified.decode_eager import greedy_decode
    from starling.parakeet_unified.loader import load_state_dict
    from starling.parakeet_unified.mel_gpu import GpuMelExtractor
    from starling.parakeet_unified.tokenizer import ParakeetUnifiedTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # eager reference in fp32; the megakernel matches in bf16 too.

    print("[golden] loading parakeet-unified-en-0.6b ...")
    sd = load_state_dict(device=device, dtype=dtype)

    mel = GpuMelExtractor(sd, device=device)
    enc = M.ConformerEncoder().to(device).to(dtype).eval()
    enc.load_state_dict_prefixed(sd)
    dec = M.RNNTDecoder().to(device).to(dtype).eval()
    dec.load_state_dict(
        {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")},
        strict=True,
    )
    joint = M.RNNTJoint().to(device).to(dtype).eval()
    joint.load_state_dict(
        {k[len("joint."):]: v for k, v in sd.items() if k.startswith("joint.")},
        strict=True,
    )
    tok = ParakeetUnifiedTokenizer()
    print("[golden] models ready")

    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)

    for name in ("short", "medium", "long"):
        audio = load_wav(REPO / "tests" / "fixtures" / f"{name}.wav")
        seconds = audio.shape[0] / 16000
        feats, fl = mel([audio])
        with torch.inference_mode():
            encoded, el = enc(feats, fl)
            t0 = time.perf_counter()
            ids = greedy_decode(encoded, el, dec, joint)
        ms = (time.perf_counter() - t0) * 1000.0
        text = tok.ids_to_text(ids[0])
        ids_tensor = torch.tensor(ids[0], dtype=torch.long)
        torch.save(ids_tensor, gdir / f"parakeet_unified_{name}_ids.pt")
        torch.save(
            {
                "enc_shape": list(encoded.shape),
                "enc_len": int(el[0].item()),
                "mel_shape": list(feats.shape),
                "mel_len": int(fl[0].item()),
            },
            gdir / f"parakeet_unified_{name}_meta.pt",
        )
        (gdir / f"parakeet_unified_{name}_text.txt").write_text(text)
        print(
            f"[golden] {name}: {seconds:.2f}s, mel {tuple(feats.shape)}, "
            f"enc {tuple(encoded.shape)} -> {len(ids[0])} tokens in {ms:.0f}ms"
        )
        print(f"[golden]   text: {text[:160]!r}")
    print(f"\n[golden] saved -> {gdir}/parakeet_unified_*_ids.pt + _text.txt + _meta.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
