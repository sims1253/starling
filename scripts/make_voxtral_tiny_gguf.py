#!/usr/bin/env python3
"""Tiny synthetic Voxtral GGUF + torch-CPU reference outputs for the encoder test.

Emits models/tiny/voxtral-tiny.gguf (SAME metadata schema and tensor names as
the real converter, scaled dims) plus models/tiny/voxtral-tiny-ref.json (torch
reference outputs computed with tiny modules replicating the exact op order).

Tiny dims (every loader relation holds: attention width == heads*head_dim,
proj input == d_model*downsample, per-tensor shapes match the metadata):
  enc: 2 layers, d_model 128, 2 heads x head_dim 64 (AW 128), ffn 256,
       sliding window 16, conv pads 2/1, stride 2, theta 1e6, eps 1e-5.
  proj: input 512 (128*4), output 128, downsample 4.
  llm block (schema only; the encoder test never runs it): 2 layers, hidden
       128, 8q/2kv x head_dim 16 (QW 128, KVW 32), inter 256, vocab 512,
       ada bottleneck 4, time_embedding_dim 128.
  tokenizer: 512-entry gpt2 table (ids >= 256 decode as latin-1 bytes, so the
       loader-guard decode spot-checks still exercise CONTROL skipping).
  prompt_prefix: [1] + [32]*38 (same shape as the real prefix).

Reference pipeline (0.5 s synthetic waveform, torch CPU, bf16 wherever the
C++ oracle rounds):
  pcm (8000 samples) -> offline pad (ceil to 1280 + 49*1280 zeros) ->
  stock-formula mel (torch.stft center=True hann-400, mag^2 with the
  stft[..., :-1] trailing-TIME-frame drop canceling center's +1, log10 clamp
  1e-10, floor (1.5-8), (x+4)/4) -> embedder out -> each encoder layer out ->
  final norm -> projector rows.
The mel uses torch.stft (not the C++ pocketfft path), so the JSON mel checks
the C++ frontend end-to-end (window/bank synthesis + STFT + fixed max).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import gguf

# ---- tiny dims ---------------------------------------------------------------
N_MEL = 128
N_LAYERS, D_MODEL, N_HEADS, HEAD_DIM, FFN = 2, 128, 2, 64, 256
AW = N_HEADS * HEAD_DIM  # 128
WINDOW = 16
PROJ_OUT, DS = 128, 4
PROJ_IN = D_MODEL * DS  # 512
LLM_LAYERS, LLM_H, LLM_Q, LLM_KV, LLM_HD, LLM_I = 2, 128, 8, 2, 16, 256
VOCAB, ADA, TCOND = 512, 4, 128
SR, N_FFT, HOP, MEL_MAX = 16000, 400, 160, 1.5


def bf16(a: torch.Tensor) -> torch.Tensor:
    return a.to(torch.bfloat16)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # F.rms_norm semantics: normalize + affine in f32, ONE bf16 round.
    y = F.rms_norm(x.float(), (x.shape[-1],)).to(torch.float32)
    return bf16(y * w.float()).to(x.dtype == torch.bfloat16 and torch.bfloat16 or x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("models/tiny/voxtral-tiny.gguf"))
    ap.add_argument("--ref", type=Path, default=Path("models/tiny/voxtral-tiny-ref.json"))
    args = ap.parse_args()
    torch.manual_seed(0xC0FFEE)
    g = torch.Generator().manual_seed(1234)

    def rand(*shape: int) -> torch.Tensor:
        return bf16(torch.randn(*shape, generator=g) * 0.4)

    # ---- weights ---------------------------------------------------------------
    W: dict[str, torch.Tensor] = {}
    W["enc.conv1.weight"] = rand(D_MODEL, N_MEL, 3)
    W["enc.conv1.bias"] = rand(D_MODEL)
    W["enc.conv2.weight"] = rand(D_MODEL, D_MODEL, 3)
    W["enc.conv2.bias"] = rand(D_MODEL)
    for i in range(N_LAYERS):
        p = f"enc.blk.{i}."
        W[p + "attn_norm.weight"] = bf16(torch.ones(D_MODEL))
        W[p + "attn.q.weight"] = rand(AW, D_MODEL)
        W[p + "attn.q.bias"] = rand(AW)
        W[p + "attn.k.weight"] = rand(AW, D_MODEL)
        W[p + "attn.v.weight"] = rand(AW, D_MODEL)
        W[p + "attn.v.bias"] = rand(AW)
        W[p + "attn.o.weight"] = rand(D_MODEL, AW)
        W[p + "attn.o.bias"] = rand(D_MODEL)
        W[p + "ffn_norm.weight"] = bf16(torch.ones(D_MODEL))
        W[p + "ffn.gate.weight"] = rand(FFN, D_MODEL)
        W[p + "ffn.up.weight"] = rand(FFN, D_MODEL)
        W[p + "ffn.down.weight"] = rand(D_MODEL, FFN)
        W[p + "ffn.down.bias"] = rand(D_MODEL)
    W["enc.final_norm.weight"] = bf16(torch.ones(D_MODEL))
    W["proj.fc0.weight"] = rand(PROJ_OUT, PROJ_IN)
    W["proj.fc2.weight"] = rand(PROJ_OUT, PROJ_OUT)
    W["llm.embed.weight"] = rand(VOCAB, LLM_H)
    W["llm.final_norm.weight"] = bf16(torch.ones(LLM_H))
    W["llm.t_cond"] = torch.randn(TCOND, dtype=torch.float32)
    QW, KVW = LLM_Q * LLM_HD, LLM_KV * LLM_HD
    for i in range(LLM_LAYERS):
        p = f"llm.blk.{i}."
        W[p + "attn_norm.weight"] = bf16(torch.ones(LLM_H))
        W[p + "attn.q.weight"] = rand(QW, LLM_H)
        W[p + "attn.k.weight"] = rand(KVW, LLM_H)
        W[p + "attn.v.weight"] = rand(KVW, LLM_H)
        W[p + "attn.o.weight"] = rand(LLM_H, QW)
        W[p + "ffn_norm.weight"] = bf16(torch.ones(LLM_H))
        W[p + "ffn.gate.weight"] = rand(LLM_I, LLM_H)
        W[p + "ffn.up.weight"] = rand(LLM_I, LLM_H)
        W[p + "ffn.down.weight"] = rand(LLM_H, LLM_I)
        W[p + "ada.fc0.weight"] = rand(ADA, LLM_H)
        W[p + "ada.fc2.weight"] = rand(LLM_H, ADA)

    # ---- reference pipeline ------------------------------------------------------
    t = torch.arange(8000, dtype=torch.float32) / SR
    pcm = (0.5 * torch.sin(2 * math.pi * 440.0 * t) +
           0.25 * torch.sin(2 * math.pi * 880.0 * t + 1.0)).tolist()
    # Offline pad: ceil(S/1280)*1280 + 49*1280 zeros.
    S = len(pcm)
    body = ((S + 1279) // 1280) * 1280
    padded = torch.tensor(pcm + [0.0] * (body - S + 49 * 1280), dtype=torch.float32)
    # Stock-formula mel (torch.stft, center=True, hann-400, mag^2 -- the
    # stft[..., :-1] drops the trailing TIME frame, canceling center's +1, so
    # mel_T == padded//hop exactly -- log10 clamp 1e-10, fixed floor (1.5-8),
    # (x+4)/4).
    from transformers.audio_utils import mel_filter_bank
    bank = torch.from_numpy(mel_filter_bank(201, 128, 0.0, 8000.0, 16000,
                                            norm="slaney", mel_scale="slaney")).float()
    window = torch.hann_window(N_FFT)
    stft = torch.stft(padded, N_FFT, HOP, window=window, return_complex=True, center=True)
    mag = stft[..., :-1].abs() ** 2
    mel = (bank.T @ mag).clamp(min=1e-10).log10()
    mel = torch.maximum(mel, torch.tensor(MEL_MAX - 8.0))
    mel = (mel + 4.0) / 4.0
    mel_T = mel.shape[-1]
    assert mel_T % 8 == 0, mel_T
    T_enc = mel_T // 2
    ref = {"mel_T": mel_T, "T_enc": T_enc,
           "n_tokens": T_enc // DS, "width": PROJ_OUT,
           "mel": bf16(mel).flatten().tolist()}

    # Embedder: causal conv1d k3 s1 left-pad 2 -> GELU -> k3 s2 left-pad 1 ->
    # GELU -> transpose to (T_enc, D_MODEL). bf16 oracle: GEMM in bf16, bias +
    # GELU boundaries matching linear_bf16/gelu_erf_bf16.
    def lin(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        y = bf16(x) @ bf16(w).t().to(torch.bfloat16)
        if b is not None:
            y = y.float() + b.float().to(torch.float32)
            y = bf16(y)
        return y

    x = bf16(mel).unsqueeze(0)  # [1, 128, mel_T]
    x = F.pad(x, (2, 0))
    x = F.conv1d(x.float(), W["enc.conv1.weight"].float(),
                 W["enc.conv1.bias"].float(), stride=1)
    # Exact GELU (erf, approximate="none"): 0.5*x*(1+erf(x/sqrt2)).
    x = bf16(0.5 * x.float() * (1.0 + torch.erf(x.float() / math.sqrt(2.0))))
    x = F.pad(x, (1, 0))
    x = F.conv1d(x.float(), W["enc.conv2.weight"].float(),
                 W["enc.conv2.bias"].float(), stride=2)
    x = bf16(0.5 * x.float() * (1.0 + torch.erf(x.float() / math.sqrt(2.0))))
    h = bf16(x.permute(0, 2, 1)).squeeze(0)  # [T_enc, D_MODEL]
    ref["embedder"] = h.flatten().tolist()

    # RoPE tables + band mask.
    inv = torch.pow(1000000.0, -2.0 * torch.arange(HEAD_DIM // 2).float() / HEAD_DIM)
    pos = torch.arange(T_enc).float().unsqueeze(1) * inv.unsqueeze(0)
    cos, sin = pos.cos(), pos.sin()
    # Stock emb = cat((freqs, freqs)): full-dim tables [T, head_dim].
    cos_full = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, D]
    sin_full = torch.cat((sin, sin), dim=-1).unsqueeze(0)
    band = torch.full((T_enc, T_enc), -3.3895313892515355e38)
    for i in range(T_enc):
        for j in range(max(0, i - WINDOW + 1), i + 1):
            band[i, j] = 0.0

    layer_outs = []
    # Layer-0 per-stage bisect dumps (kept in the reference JSON so future
    # debugging can compare each stage without re-deriving layouts; all
    # [T, W] row-major flattened, att0 is softmax head 0 [T, T] rows).
    stages: dict[str, list[float]] = {}
    for i in range(N_LAYERS):
        p = f"enc.blk.{i}."
        r = h
        n = rms_norm(h, W[p + "attn_norm.weight"], 1e-5)
        q = lin(n, W[p + "attn.q.weight"], W[p + "attn.q.bias"])
        k = lin(n, W[p + "attn.k.weight"], None)
        v = lin(n, W[p + "attn.v.weight"], W[p + "attn.v.bias"])
        # MHA: fold heads, rotate-half RoPE on full head_dim, band mask.
        def heads(z: torch.Tensor) -> torch.Tensor:
            return z.view(T_enc, N_HEADS, HEAD_DIM).transpose(0, 1)  # [H, T, D]
        def unheads(z: torch.Tensor) -> torch.Tensor:
            return z.transpose(0, 1).reshape(T_enc, AW)  # [H, T, D] -> [T, AW]
        qh, kh, vh = heads(q), heads(k), heads(v)
        # C++ mirror: cos/sin cast to the activation dtype, per-op bf16
        # rounds through rope (first = (x1*cos)-(x2*sin), second = +).
        cos_b, sin_b = bf16(cos_full), bf16(sin_full)
        cos_h, sin_h = cos_b[..., :HEAD_DIM // 2], sin_b[..., :HEAD_DIM // 2]
        def rope(z: torch.Tensor) -> torch.Tensor:
            z1, z2 = z[..., :HEAD_DIM // 2], z[..., HEAD_DIM // 2:]
            r1 = bf16(bf16(z1 * cos_h) - bf16(z2 * sin_h))
            r2 = bf16(bf16(z2 * cos_h) + bf16(z1 * sin_h))
            return torch.cat((r1, r2), dim=-1)
        if i == 0:
            stages["n0"] = n.flatten().tolist()
            stages["q0"] = q.flatten().tolist()
            stages["k0"] = k.flatten().tolist()
            stages["v0"] = v.flatten().tolist()
        qh, kh = rope(qh), rope(kh)
        if i == 0:
            stages["qr0"] = unheads(qh).flatten().tolist()
            stages["kr0"] = unheads(kh).flatten().tolist()
        # HF op order: bf16-rounded matmul output, bf16 scale, the f32 band
        # mask added in f32, softmax in f32, bf16 context matmul.
        att = (qh @ kh.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD_DIM))
        att = att.float() + band.unsqueeze(0)
        att = torch.softmax(att, dim=-1).to(torch.bfloat16)
        if i == 0:
            stages["att0"] = att[0].flatten().tolist()
        ctx = (att @ vh).transpose(0, 1).reshape(T_enc, AW)
        if i == 0:
            stages["ctx0"] = ctx.flatten().tolist()
        a = lin(bf16(ctx), W[p + "attn.o.weight"], W[p + "attn.o.bias"])
        if i == 0:
            stages["a0"] = a.flatten().tolist()
        h = bf16(r.float() + a.float())
        r = h
        n = rms_norm(h, W[p + "ffn_norm.weight"], 1e-5)
        gg = lin(n, W[p + "ffn.gate.weight"], None)
        u = lin(n, W[p + "ffn.up.weight"], None)
        # SwiGLU product in f32 with ONE bf16 round (the oracle order: silu
        # and the product stay f32 until the store).
        ffn = lin(bf16(F.silu(gg.float()) * u.float()),
                  W[p + "ffn.down.weight"], W[p + "ffn.down.bias"])
        if i == 0:
            stages["ffn0"] = ffn.flatten().tolist()
        h = bf16(r.float() + ffn.float())
        layer_outs.append(h.flatten().tolist())
    ref["layers"] = layer_outs
    ref.update(stages)
    h = rms_norm(h, W["enc.final_norm.weight"], 1e-5)
    ref["final_norm"] = h.flatten().tolist()
    # Projector: group-by-4 reshape -> fc0 -> GELU -> fc2 (no biases).
    pr = h.view(T_enc // DS, PROJ_IN)
    pr = lin(pr, W["proj.fc0.weight"], None)
    pr = bf16(0.5 * pr.float() * (1.0 + torch.erf(pr.float() / math.sqrt(2.0))))
    pr = lin(pr, W["proj.fc2.weight"], None)
    ref["projected"] = pr.flatten().tolist()

    # ---- GGUF --------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.output, "voxtral", use_temp_file=True)
    V = gguf.GGUFValueType

    def ints(**xs):
        for k, v in xs.items():
            w.add_key_value("voxtral." + k, v, V.UINT32)

    def floats(**xs):
        for k, v in xs.items():
            w.add_key_value("voxtral." + k, float(v), V.FLOAT32)

    def strings(**xs):
        for k, v in xs.items():
            w.add_string("voxtral." + k, v)

    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    w.add_string("general.architecture", "voxtral")
    ints(**{"frontend.sample_rate": SR, "frontend.n_fft": N_FFT,
            "frontend.win_length": N_FFT, "frontend.hop_length": HOP,
            "frontend.n_mels": N_MEL, "frontend.center": 1,
            "frontend.unit_samples": 1280, "frontend.left_pad_tokens": 32,
            "frontend.right_pad_tokens": 17})
    floats(**{"frontend.mel_floor": 1e-10, "frontend.log_mel_max": MEL_MAX,
              "frontend.normalization_offset": 4.0,
              "frontend.normalization_divisor": 4.0, "frontend.dynamic_range": 8.0})
    strings(**{"frontend.mel_scale": "slaney", "frontend.log": "log10",
               "frontend.output_dtype": "bf16"})
    ints(**{"enc.num_mel_bins": N_MEL, "enc.encoder_layers": N_LAYERS,
            "enc.d_model": D_MODEL, "enc.encoder_attention_heads": N_HEADS,
            "enc.head_dim": HEAD_DIM, "enc.encoder_ffn_dim": FFN,
            "enc.sliding_window": WINDOW, "enc.conv_kernel": 3,
            "enc.conv_left_pad1": 2, "enc.conv_left_pad2": 1, "enc.conv_stride2": 2})
    floats(**{"enc.rope_theta": 1000000.0, "enc.rms_norm_eps": 1e-5})
    ints(**{"proj.input_size": PROJ_IN, "proj.output_size": PROJ_OUT,
            "proj.downsample": DS, "proj.mel_per_token": 8})
    strings(**{"proj.act": "gelu"})
    ints(**{"llm.hidden_size": LLM_H, "llm.num_layers": LLM_LAYERS,
            "llm.num_heads": LLM_Q, "llm.num_kv_heads": LLM_KV,
            "llm.head_dim": LLM_HD, "llm.intermediate_size": LLM_I,
            "llm.vocab_size": VOCAB, "llm.sliding_window": 8192, "llm.tied": 1,
            "llm.num_delay_tokens": 6, "llm.time_embedding_dim": TCOND,
            "llm.ada_bottleneck": ADA, "llm.max_cache_len": 4096})
    floats(**{"llm.rope_theta": 1000000.0, "llm.rms_norm_eps": 1e-5})
    w.add_key_value("voxtral.llm.time_embedding_theta", 10000.0, V.FLOAT32)
    ints(**{"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 11,
            "streaming_pad_id": 32, "left_pad_tokens": 32,
            "right_pad_tokens": 17, "max_new_tokens": 200})
    w.add_key_value("voxtral.prompt_prefix", [1] + [32] * 38, V.ARRAY, V.INT32)
    # Tokenizer: 512-entry gpt2 table (ids 0..255 CONTROL specials like the
    # real tekken head, ids 256..511 latin-1 single bytes).
    w.add_tokenizer_model("gpt2")
    toks = [f"<s{i}>" for i in range(256)] + \
        [bytes([i - 256]).decode("latin-1") for i in range(256, VOCAB)]
    w.add_token_list(toks)
    w.add_token_scores([0.0] * VOCAB)
    w.add_token_types([gguf.TokenType.CONTROL] * 256 + [gguf.TokenType.NORMAL] * 256)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_pad_token_id(11)
    for name, t in W.items():
        if name == "llm.t_cond":
            w.add_tensor(name, np.ascontiguousarray(t.numpy(), dtype=np.float32))
            continue
        a = np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint16).numpy())
        w.add_tensor(name, a, raw_shape=a.shape,
                     raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # Reference JSON: f32 lists (bf16 values cast up exactly) + the pcm.
    ref["pcm"] = pcm
    args.ref.write_text(json.dumps(ref))
    import os
    print(f"wrote {args.output}: {os.path.getsize(args.output)} bytes, "
          f"mel_T={mel_T} T_enc={T_enc} tokens={T_enc // DS}")
    print(f"wrote {args.ref}: {os.path.getsize(args.ref)} bytes")


if __name__ == "__main__":
    main()
