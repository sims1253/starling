"""A/B the attention backends on the real ARK decode path:
manual (current) vs SDPA-math (byte-exact, no repeat_kv) vs FP8 (fastest).

Measures real decode tok/s + checks transcript drift. This isolates the
attention-backend optimization from everything else.
"""
from __future__ import annotations

import sys
import time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from starling.ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel
from starling.ark.config import EOS_TOKEN_ID
from starling.ark.pipeline import MegaPipeline
from starling.attention import gqa_attention
from starling.flags import OptFlags


def make_decode(backend: str):
    """Return a _decode_step_eager replacement using the chosen attention backend."""
    def decode(self):
        k = self._k
        hd = self._head_dim; n_q = self._n_q_heads; n_kv = self._n_kv_heads
        qkv_split = [n_q*hd, n_kv*hd, n_kv*hd]; inter = self._intermediate
        flags = {"manual": OptFlags(), "sdpa": OptFlags(sdpa_attention=True),
                 "fp8": OptFlags(fp8_attention=True, tolerance_mode=True)}[backend]
        hidden = self._embed(self.static_input_ids) * 1.0
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        for idx, layer in enumerate(self._layers):
            f = self._fused[idx]
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)
            x2 = normed.view(1, -1)
            qkv = F.linear(x2, f["qkv_w"], f["qkv_b"]).view(-1)
            q, kv, v = qkv.split(qkv_split, dim=0)
            q = q.view(1, n_q, 1, hd); kv = kv.view(1, n_kv, 1, hd); v = v.view(1, n_kv, 1, hd)
            q, kv = k.fused_rope(q, kv, cos, sin)
            kv, v = self.cache.update(kv, v, idx)
            attn_out = gqa_attention(q, kv, v, self.static_attn_mask, self._attn_scale, self.dtype, flags)
            attn_out = attn_out.transpose(1,2).reshape(1,1,n_q*hd)
            attn_out = f["o_proj"](attn_out)
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)
            x3 = normed.view(1, -1)
            gu = F.linear(x3, f["gu_w"], None).view(-1)
            gate, up = gu.split([inter, inter], dim=0)
            gate = gate.view(1,1,inter); up = up.view(1,1,inter)
            act = k.fused_silu_mul(gate, up)
            mlp_out = f["down_proj"](act)
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = self.lm_head(hidden) / 1.0
        self.static_logits.copy_(logits)
    return decode


def measure(pipe, wav, backend):
    orig = type(pipe.llm)._decode_step_eager
    type(pipe.llm)._decode_step_eager = make_decode(backend)
    pipe.llm._ms_captured = False
    pipe.llm._captured = False
    hop = int(pipe.processor.feature_extractor.hop_length)
    n_mel = len(wav) // hop
    mel = extract_mel(pipe.processor, [wav]).to(dtype=torch.bfloat16, device="cuda")
    ids = build_prompt_ids(pipe.processor.tokenizer, "Transcribe the audio to text.", n_mel_frames=n_mel).to("cuda")
    af = pipe.fused_encoder(mel)
    ie = build_inputs_embeds(pipe.model, ids, af)
    T = int(ie.shape[1]); llm = pipe._get_llm(T)
    for _ in range(3):
        llm.generate(ie, max_new_tokens=60, eos_token_id=EOS_TOKEN_ID)
    torch.cuda.synchronize()
    times, texts = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        res = llm.generate(ie, max_new_tokens=60, eos_token_id=EOS_TOKEN_ID)
        torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)/max(res.n_tokens,1)*1000)
        texts.append(pipe.processor.tokenizer.decode(res.ids[0], skip_special_tokens=True))
    type(pipe.llm)._decode_step_eager = orig
    return float(np.median(times)), texts[0]


def main():
    print("loading ...", flush=True)
    pipe = MegaPipeline.from_pretrained(max_cache_len=4096)
    pipe.prewarm()
    wav, sr = sf.read("tests/fixtures/medium.wav")
    wav = np.ascontiguousarray(wav, dtype=np.float32)

    results = {}
    for backend in ["manual", "sdpa"]:
        ms, text = measure(pipe, wav, backend)
        results[backend] = (ms, text)
        print(f"  {backend:8s}: {ms:.2f} ms/tok = {1000/ms:.0f} tok/s")

    print("\ntranscript (first 100 chars):")
    for b, (_, t) in results.items():
        print(f"  {b:8s}: {t[:100]}")
    print(f"\nsdpa byte-identical to manual: {results['manual'][1].strip()==results['sdpa'][1].strip()}")
    base = results["manual"][0]
    print(f"  sdpa speedup: {base/results['sdpa'][0]:.2f}x")


if __name__ == "__main__":
    main()
