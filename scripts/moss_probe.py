"""Probe MOSS-Transcribe loading + stock transcription on a test fixture.

Captures the stock-transformers golden transcript + token ids for the short
fixture. Run once to populate golden/moss_*.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
# vendor the model code so we don't depend on a transformers bump
sys.path.insert(0, str(REPO / ".moss_ref"))

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-preview-2B"


def main() -> int:
    from transformers import AutoTokenizer
    from modeling_Moss import MossForCausalLM, MossConfig
    from processing_Moss import MossProcessor

    cfg = MossConfig.from_pretrained(MODEL_ID)
    print("[probe] loading model (bf16, cuda) ...")
    t0 = time.perf_counter()
    model = MossForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    print(f"[probe] loaded in {time.perf_counter()-t0:.1f}s")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    from processing_Moss import MelConfig
    # model conv_out expects 128 mel bins (7680 = 480*16 from 128->64->32->16 conv downsampling)
    proc = MossProcessor(tok, config=MelConfig(mel_dim=128),
                         template_path=str(REPO / ".moss_ref" / "chat_template_default.py"))

    # load short fixture
    wav, sr = sf.read(str(REPO / "tests" / "fixtures" / "short.wav"))
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = torch.from_numpy(wav.astype("float32"))
    audio_seconds = wav.shape[0] / sr
    print(f"[probe] fixture: {audio_seconds:.2f}s, {wav.shape[0]} samples")

    inp = proc(wav.numpy())
    # move to cuda
    inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
    print("[probe] input_ids", inp["input_ids"].shape, "audio_data", inp["audio_data"].shape)
    print("[probe] num audio tokens (mask sum):", int(inp["audio_input_mask"].sum()))

    print("[probe] running stock generate ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(
            **inp,
            max_new_tokens=200,
            do_sample=False,
        )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    n_new = gen.shape[1] - inp["input_ids"].shape[1]
    print(f"[probe] generated {n_new} tokens in {ms:.0f} ms  ({n_new/(ms/1000):.0f} tok/s, RTFx {audio_seconds/(ms/1000):.1f}x)")
    new_ids = gen[0, inp["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    print(f"[probe] transcript:\n{text}")

    # save golden
    gdir = REPO / "golden"
    gdir.mkdir(exist_ok=True)
    torch.save(new_ids.cpu(), gdir / "moss_short_ids.pt")
    (gdir / "moss_short_text.txt").write_text(text)
    print(f"[probe] saved golden -> {gdir}/moss_short_*.pt/txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
