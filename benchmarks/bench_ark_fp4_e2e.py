"""End-to-end NVFP4 proof: temporarily monkey-patch ARK's GEMVs to fp4 and
measure real decode tok/s + transcript drift. This is the ground truth that
the isolated microbench couldn't give us (cache noise).
"""
from __future__ import annotations

import os
import sys
os.environ["MEGAPAR_LLM_AUTOTUNE"] = "0"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf
import torch

from starling.ark.audio import build_inputs_embeds, build_prompt_ids, extract_mel
from starling.ark.config import EOS_TOKEN_ID
from starling.ark.pipeline import MegaPipeline
from starling.granite.fp4 import quantize_fp4_packed, _fp4_linear_fused


def measure(pipe, wav, label):
    hop = int(pipe.processor.feature_extractor.hop_length)
    n_mel = len(wav) // hop
    mel = extract_mel(pipe.processor, [wav]).to(dtype=torch.bfloat16, device="cuda")
    ids = build_prompt_ids(pipe.processor.tokenizer, "Transcribe the audio to text.", n_mel_frames=n_mel).to("cuda")
    af = pipe.fused_encoder(mel)
    ie = build_inputs_embeds(pipe.model, ids, af)
    T = int(ie.shape[1])
    llm = pipe._get_llm(T)
    # warm
    for _ in range(3):
        llm.generate(ie, max_new_tokens=60, eos_token_id=EOS_TOKEN_ID)
    torch.cuda.synchronize()
    # timed
    import time
    texts, toks, times = [], [], []
    for _ in range(3):
        t0 = time.perf_counter()
        res = llm.generate(ie, max_new_tokens=60, eos_token_id=EOS_TOKEN_ID)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt / max(res.n_tokens, 1) * 1000)
        toks.append(res.n_tokens)
        texts.append(pipe.processor.tokenizer.decode(res.ids[0], skip_special_tokens=True))
    print(f"  {label}: {np.median(times):.2f} ms/tok = {1000/np.median(times):.0f} tok/s  ({toks[0]} tok)")
    return texts[0]


def main():
    print("loading ...", flush=True)
    pipe = MegaPipeline.from_pretrained(max_cache_len=4096)
    pipe.prewarm()

    wav, sr = sf.read("tests/fixtures/medium.wav")
    wav = np.ascontiguousarray(wav, dtype=np.float32)

    # baseline
    print("\nbaseline (bf16):")
    ref_text = measure(pipe, wav, "bf16 ")

    # ---- patch: quantize every GEMM weight and swap the calls ----
    llm = pipe.llm
    fp4_layers = []
    for idx, f in enumerate(llm._fused):
        fp4_layers.append({
            "qkv_w": quantize_fp4_packed(f["qkv_w"].detach()),
            "gu_w": quantize_fp4_packed(f["gu_w"].detach()),
            "o_proj": quantize_fp4_packed(f["o_proj"].weight.detach()),
            "down_proj": quantize_fp4_packed(f["down_proj"].weight.detach()),
        })
    fp4_lm_head = quantize_fp4_packed(llm._lm_head.weight.detach() if hasattr(llm,"_lm_head") and llm._lm_head is not None else llm.lm_head.weight.detach())

    # monkeypatch _decode_step_eager's GEMV calls by swapping the fused dict to
    # sentinel-bearing proxies is hard; instead patch F.linear temporarily.
    # Simpler: replace the _fused weights with fp4 and swap the decode_step body.
    # We'll patch the method to use fp4.
    orig_decode = type(llm)._decode_step_eager
    def fp4_decode(self):
        k = self._k
        hd = self._head_dim; n_q = self._n_q_heads; n_kv = self._n_kv_heads
        qkv_split = [n_q*hd, n_kv*hd, n_kv*hd]; inter = self._intermediate
        hidden = self._embed(self.static_input_ids) * 1.0
        cos, sin = self._rotary(hidden, position_ids=self.static_position_ids)
        for idx, layer in enumerate(self._layers):
            f4 = fp4_layers[idx]
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.input_layernorm.weight, self._rms_eps)
            x2 = normed.view(1, -1)
            qkv = _fp4_linear_fused(x2, f4["qkv_w"]).view(-1)
            q, kv, v = qkv.split(qkv_split, dim=0)
            q = q.view(1, n_q, 1, hd); kv = kv.view(1, n_kv, 1, hd); v = v.view(1, n_kv, 1, hd)
            q, kv = k.fused_rope(q, kv, cos, sin)
            kv, v = self.cache.update(kv, v, idx)
            from starling.ark.llm_mega import _repeat_kv
            kv_r = _repeat_kv(kv, self._n_kv_groups); v_r = _repeat_kv(v, self._n_kv_groups)
            scores = torch.matmul(q, kv_r.transpose(2,3)) * self._attn_scale
            scores = scores + self.static_attn_mask
            attn = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
            attn_out = torch.matmul(attn, v_r)
            attn_out = attn_out.transpose(1,2).reshape(1,1,n_q*hd)
            attn_out = _fp4_linear_fused(attn_out, f4["o_proj"])
            hidden = k.fused_residual_scale(residual, attn_out, self._res_mult)
            residual = hidden
            normed = k.fused_rmsnorm(hidden, layer.post_attention_layernorm.weight, self._rms_eps)
            x3 = normed.view(1, -1)
            gu = _fp4_linear_fused(x3, f4["gu_w"]).view(-1)
            gate, up = gu.view(-1).split([inter, inter], dim=0)
            gate = gate.view(1,1,inter); up = up.view(1,1,inter)
            act = k.fused_silu_mul(gate, up)
            mlp_out = _fp4_linear_fused(act, f4["down_proj"])
            hidden = k.fused_residual_scale(residual, mlp_out, self._res_mult)
        hidden = k.fused_rmsnorm(hidden, self._final_norm.weight, self._rms_eps)
        logits = _fp4_linear_fused(hidden, fp4_lm_head)
        self.static_logits.copy_(logits)

    type(llm)._decode_step_eager = fp4_decode
    # force recapture of the K-step graph with the fp4 body
    llm._ms_captured = False
    llm._captured = False

    print("\nNVFP4 (all GEMMs quantized):")
    fp4_text = measure(pipe, wav, "fp4 ")
    type(llm)._decode_step_eager = orig_decode

    print("\ntranscript drift:")
    print(f"  bf16: {ref_text[:120]}")
    print(f"  fp4 : {fp4_text[:120]}")
    same = ref_text.strip() == fp4_text.strip()
    print(f"  byte-identical: {same}")


if __name__ == "__main__":
    main()
