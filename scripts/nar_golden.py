"""Capture golden reference for granite-speech-4.1-2b-nar on the fixture tiers.

The golden is the stock ``model.transcribe`` output (token ids + text) for
short/medium/long fixtures, captured eager/SDPA. The starling pipeline must
reproduce these token ids exactly.

Run:  uv run python scripts/nar_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"
FIXTURES = [
    ("short", "tests/fixtures/short.wav"),
    ("medium", "tests/fixtures/medium.wav"),
    ("long", "tests/fixtures/long.wav"),
]
GOLDEN_DIR = Path("golden/nar")


def main() -> int:
    from transformers import AutoModel, AutoProcessor
    import soundfile as sf

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    ).to("cuda")
    model.eval()

    summary = {}
    for tier, path in FIXTURES:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).float().to("cuda")
        inputs = processor(audios=wav_t, device="cuda")
        feats = inputs["input_features"].to(torch.bfloat16)
        attn = inputs["attention_mask"]
        dur = len(wav) / sr

        # Run twice to confirm determinism (cuBLAS bf16 can be nondeterministic).
        with torch.inference_mode():
            out1 = model.transcribe(feats, attention_mask=attn)
            out2 = model.transcribe(feats, attention_mask=attn)
        p1, p2 = out1.preds[0].cpu(), out2.preds[0].cpu()
        det = bool((p1 == p2).all().item())
        text = processor.batch_decode([p1.tolist()])[0]
        print(f"[{tier}] {dur:.1f}s -> {len(p1)} tok, deterministic={det}")
        print(f"   {text!r}")

        torch.save(p1, GOLDEN_DIR / f"{tier}_preds.pt")
        # Also save the encoder + projector + LLM intermediates for byte-exact
        # stage checks in tests.
        with torch.inference_mode():
            enc_out = model.encoder(feats, attention_mask=attn, output_hidden_states=True)
            torch.save(enc_out.last_hidden_state.cpu(), GOLDEN_DIR / f"{tier}_enc_last.pt")
            torch.save(enc_out.logits.cpu(), GOLDEN_DIR / f"{tier}_enc_bpe_logits.pt")
            multilayer = torch.cat(
                [enc_out.all_hidden_states[idx] for idx in model.config.encoder_layer_indices],
                dim=-1,
            )
            audio_embeds = model.projector(multilayer)
            em = getattr(model.config.text_config, "embedding_multiplier", 1.0)
            audio_embeds = (audio_embeds / em).to(model.language_model.model.embed_tokens.weight.dtype)
            torch.save(audio_embeds.cpu(), GOLDEN_DIR / f"{tier}_audio_embeds.pt")

        summary[tier] = {
            "duration_s": dur,
            "n_tokens": int(len(p1)),
            "text": text,
            "deterministic": det,
            "enc_last_shape": list(enc_out.last_hidden_state.shape),
            "audio_embeds_shape": list(audio_embeds.shape),
        }

    out_path = GOLDEN_DIR / "golden.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[golden] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
