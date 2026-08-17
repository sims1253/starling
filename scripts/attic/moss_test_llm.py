"""Quick byte-exactness test for the MOSS LLM megakernel vs the golden reference."""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from starling.moss.loader import load_model_and_processor
    from starling.moss.reference import audio_features, build_inputs_embeds
    from starling.moss.llm_mega import MossLLMMega
    from starling.moss.multistep import MossMultiStepMega

    print("[test] loading model ...")
    model, proc = load_model_and_processor()
    wav, sr = sf.read(str(REPO / "tests" / "fixtures" / "short.wav"))
    inp = proc(wav.astype("float32"))
    inp = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}

    with torch.inference_mode():
        feats = audio_features(model, inp["audio_data"], inp["audio_data_seqlens"])
        emb = build_inputs_embeds(model, inp["input_ids"], feats, inp["audio_input_mask"])

    # golden
    gold = torch.load(REPO / "golden" / "moss_short_ids.pt")
    print(f"[test] golden: {gold.shape[1]} tokens")

    comps_inner = model.model
    lm, lm_head = comps_inner.language_model, model.lm_head

    # single-step
    print("[test] single-step decoder ...")
    dec1 = MossLLMMega(lm, lm_head, max_cache_len=1024)
    with torch.inference_mode():
        r1 = dec1.generate(emb, max_new_tokens=gold.shape[1])
    text1 = proc.tokenizer.decode(r1.ids[0], skip_special_tokens=True)
    match1 = bool((r1.ids[0] == gold[0]).all().item())
    print(f"[test] single-step: {r1.n_tokens} tokens, match={match1}")
    print(f"[test]   {text1[:120]}")
    if not match1:
        # show first divergence
        for i in range(min(r1.n_tokens, gold.shape[1])):
            if r1.ids[0, i].item() != gold[0, i].item():
                print(f"[test]   first divergence at token {i}: mine={r1.ids[0,i].item()} gold={gold[0,i].item()}")
                break

    # multi-step K=16
    print("[test] multi-step K=16 ...")
    dec16 = MossMultiStepMega(lm, lm_head, max_cache_len=1024, steps_per_replay=16)
    with torch.inference_mode():
        r16 = dec16.generate(emb, max_new_tokens=gold.shape[1])
    match16 = bool((r16.ids[0] == gold[0]).all().item())
    print(f"[test] multi-step K=16: {r16.n_tokens} tokens, match={match16}")
    return 0 if (match1 and match16) else 1


if __name__ == "__main__":
    raise SystemExit(main())
