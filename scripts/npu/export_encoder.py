"""Export the Qwen3-ASR audio encoder (+ projector) of an HF checkpoint as the
TPU artifact's source graph: one attention window of n_window_infer mel
frames, inputs mel [1, n_mels, frames] f32 and key mask [1, 1, 1, tokens] f32
(0 valid / -1e4 padding), output [1, tokens, output_dim] f32 — the contract in
cpp/npu/encoder_contract.hpp. Weights are stored as fp16 (the TPU computes in
fp16 either way), and the converter's `keep_stablehlo_constant` metadata key,
which the phone's on-device compiler rejects, is renamed in place.

python -I export_encoder.py MODEL OUT.tflite      (MODEL: HF id or snapshot dir)
Prints one JSON line: {"key": ..., "in": [...], "out": [...], "model_hash": "0x...", "tflite": ...}.

python -I export_encoder.py --fingerprint MODEL
Prints only the checkpoint's model fingerprint (0x...), without exporting.
"""
import glob
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

fingerprint_only = sys.argv[1] == "--fingerprint"
model = sys.argv[2] if fingerprint_only else sys.argv[1]
out_path = None if fingerprint_only else sys.argv[2]
if os.path.isdir(model):
    snap = model
else:
    cands = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))
    if not cands:
        raise SystemExit(f"{model}: not in the HF cache (hf download {model})")
    snap = cands[0]

from safetensors import safe_open  # noqa: E402

# The model fingerprint of cpp/npu/encoder_contract.hpp: FNV-1a 64 over the
# f32 bytes of these tensors, in this order (kQwen3EncFingerprintTensors).
# Starling GGUFs keep them F32, widened exactly from the checkpoint's values,
# so a checkpoint and the GGUFs converted from it share the fingerprint; a
# fine-tune with the same dims does not.
FINGERPRINT_TENSORS = [  # (GGUF name, checkpoint name)
    ("enc.conv1.bias", "model.audio_tower.conv2d1.bias"),
    ("enc.ln_post.weight", "model.audio_tower.ln_post.weight"),
    ("enc.ln_post.bias", "model.audio_tower.ln_post.bias"),
    ("proj.linear_2.bias", "model.multi_modal_projector.linear_2.bias"),
    ("enc.blk.0.attn_norm.weight", "model.audio_tower.layers.0.self_attn_layer_norm.weight"),
]


def fingerprint(snap_dir):
    found = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for _, k in FINGERPRINT_TENSORS:
                if k in h.keys():
                    found[k] = h.get_tensor(k)
    h = 0xCBF29CE484222325
    for g, k in FINGERPRINT_TENSORS:
        if k not in found:
            raise SystemExit(f"{snap_dir}: no {k} (for {g}); not a Qwen3-ASR checkpoint")
        for b in found[k].float().contiguous().numpy().astype("<f4").tobytes():
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"0x{h:016x}"


model_hash = fingerprint(snap)
if fingerprint_only:
    print(model_hash)
    sys.exit(0)

from transformers import AutoConfig, AutoModel  # noqa: E402

torch.set_grad_enabled(False)
cfg = AutoConfig.from_pretrained(snap)
ac = cfg.audio_config
enc = AutoModel.from_config(ac).float().eval()


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(ac.d_model, ac.d_model)
        self.linear_2 = nn.Linear(ac.d_model, ac.output_dim)


proj = Projector().eval()
enc_sd, proj_sd = {}, {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("model.audio_tower."):
                enc_sd[k[len("model.audio_tower."):]] = h.get_tensor(k).float()
            elif k.startswith("model.multi_modal_projector."):
                proj_sd[k[len("model.multi_modal_projector."):]] = h.get_tensor(k).float()
missing = enc.load_state_dict(enc_sd, strict=False)
assert not missing.missing_keys, missing.missing_keys
proj.load_state_dict(proj_sd)
del enc_sd, proj_sd   # the converter needs the memory (1.7B: ~1.3 GB of f32 weights)

FR = ac.n_window_infer                       # mel frames per attention window
CH = ac.n_window * 2                         # frames per conv chunk (100)
NCH = FR // CH                               # chunks per window
half = lambda x: (x - 1) // 2 + 1  # noqa: E731
TPC = half(half(half(CH)))                   # tokens per chunk (13)
TOK = TPC * NCH                              # tokens per window
D, H, NM = ac.d_model, ac.encoder_attention_heads, ac.num_mel_bins
DS = ac.downsample_hidden_size
FQ = half(half(half(NM)))                    # frequency bins after the convs (16)


class Window(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.proj = enc, proj
        self.register_buffer("pos", enc.positional_embedding.positional_embedding[:TPC].clone())

    def forward(self, mel, kbias):
        e = self.enc
        x = mel.reshape(NM, NCH, CH).permute(1, 0, 2).reshape(NCH, 1, NM, CH)
        x = F.gelu(e.conv2d1(x))
        x = F.gelu(e.conv2d2(x))
        x = F.gelu(e.conv2d3(x))
        x = x.permute(0, 3, 1, 2).reshape(NCH, TPC, DS * FQ)
        h = (e.conv_out(x) + self.pos).reshape(1, TOK, D)
        for L in e.layers:
            a = L.self_attn
            r = h
            y = L.self_attn_layer_norm(h)
            q = a.q_proj(y).reshape(1, TOK, H, D // H).transpose(1, 2)
            k = a.k_proj(y).reshape(1, TOK, H, D // H).transpose(1, 2)
            v = a.v_proj(y).reshape(1, TOK, H, D // H).transpose(1, 2)
            s = torch.matmul(q * a.scaling, k.transpose(2, 3)) + kbias
            o = torch.matmul(torch.softmax(s, dim=-1), v).transpose(1, 2).reshape(1, TOK, D)
            h = r + a.out_proj(o)
            h = h + L.fc2(F.gelu(L.fc1(L.final_layer_norm(h))))
        h = e.ln_post(h)
        return self.proj.linear_2(F.gelu(self.proj.linear_1(h)))


mel = torch.randn(1, NM, FR) * 0.5
kb = torch.zeros(1, 1, 1, TOK)
import litert_torch  # noqa: E402
from ai_edge_quantizer import quantizer, recipe_manager, qtyping  # noqa: E402
from ai_edge_quantizer.algorithm_manager import AlgorithmName  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    f32 = os.path.join(td, "f32.tflite")
    # Big encoders (1.7B: ~0.3 B parameters) only convert within 8 GB with
    # lightweight_conversion; the fp16 weight casting below cannot read that
    # output, so those are compiled from f32 weights (the TPU computes in
    # fp16 either way and stores fp16 bytecode; only the .tflite is bigger).
    light = sum(p.numel() for p in enc.parameters()) > 200_000_000
    litert_torch.convert(Window().eval(), (mel, kb), lightweight_conversion=light).export(f32)
    src = f32
    if not light:
        rm = recipe_manager.RecipeManager()
        rm.add_weight_only_config(".*", qtyping.TFLOperationName.ALL_SUPPORTED, 16,
                                  algorithm_key=AlgorithmName.FLOAT_CASTING)
        src = os.path.join(td, "fp16.tflite")
        quantizer.Quantizer(f32, rm.get_quantization_recipe()).quantize(None, serialize_to_path=src)
    b = bytearray(open(src, "rb").read())
    key = b"keep_stablehlo_constant"
    assert b.count(key) <= 1, b.count(key)
    b = b.replace(key, b"x" * len(key))
    open(out_path, "wb").write(b)

ckey = (f"qwen3-asr-enc/1 mels={NM} frames={FR} tokens={TOK} d={D} L={ac.encoder_layers} H={H} "
        f"ff={ac.encoder_ffn_dim} ds={DS} out={ac.output_dim}")
print(json.dumps({"key": ckey, "in": [NM * FR * 4, TOK * 4], "out": [TOK * ac.output_dim * 4],
                  "model_hash": model_hash, "tflite": out_path, "bytes": os.path.getsize(out_path)}))
