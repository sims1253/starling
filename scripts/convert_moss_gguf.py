#!/usr/bin/env python3
"""Convert the pinned MOSS Transcribe safetensors checkpoint to Starling GGUF."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import gguf

REVISION = "c98175cb20e48bd9be4e95f6c85f2af18899f780"
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Transcribe-preview-2B/snapshots" / REVISION

# This is deliberately a complete, explicit mapping rather than a best-effort
# rename: a checkpoint addition should fail conversion.
def gguf_name(name: str) -> str:
    if name == "model.audio_model.conv_out.weight": return "enc.conv_out.weight"
    if name.startswith("model.audio_model.conv2d"):
        tail = name.removeprefix("model.audio_model.")
        return tail.replace("conv2d", "enc.conv", 1)
    if name.startswith("model.audio_model.layers."):
        p = name.removeprefix("model.audio_model.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "self_attn_layer_norm": "attn_norm", "final_layer_norm": "ffn_norm",
            "self_attn.q_proj": "attn.q", "self_attn.k_proj": "attn.k",
            "self_attn.v_proj": "attn.v", "self_attn.out_proj": "attn.o",
            "fc1": "ffn.fc1", "fc2": "ffn.fc2",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."): return f"enc.blk.{i}.{new}." + rest[len(old)+1:]
    for old, new in (("model.audio_model.ln_post.", "enc.ln_post."),
                     ("model.audio_model.proj1.", "enc.proj1."),
                     ("model.audio_model.proj2.", "enc.proj2.")):
        if name.startswith(old): return new + name[len(old):]
    if name.startswith("model.audio_adapter."):
        return "adapter." + name.removeprefix("model.audio_adapter.").replace("_proj.", ".")
    if name == "model.language_model.embed_tokens.weight": return "llm.embed.weight"
    if name.startswith("model.language_model.layers."):
        p = name.removeprefix("model.language_model.layers.").split(".")
        i, rest = p[0], ".".join(p[1:])
        repl = {
            "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
            "self_attn.q_proj": "attn.q", "self_attn.k_proj": "attn.k",
            "self_attn.v_proj": "attn.v", "self_attn.o_proj": "attn.o",
            "self_attn.q_norm": "attn.q_norm", "self_attn.k_norm": "attn.k_norm",
            "mlp.gate_proj": "ffn.gate", "mlp.up_proj": "ffn.up", "mlp.down_proj": "ffn.down",
        }
        for old, new in repl.items():
            if rest.startswith(old + "."): return f"llm.blk.{i}.{new}." + rest[len(old)+1:]
    if name == "model.language_model.norm.weight": return "llm.final_norm.weight"
    raise KeyError(f"no section 9.2 mapping for {name}")

def add_metadata(w: gguf.GGUFWriter) -> None:
    V = gguf.GGUFValueType
    # GGUF custom integer metadata needs an explicit type.
    w.add_key_value("starling.format_version", 1, V.UINT32)
    w.add_string("starling.numeric_profile", "bf16_exact")
    def ints(**xs):
        for k, v in xs.items(): w.add_key_value("moss_transcribe." + k, v, V.UINT32)
    def floats(**xs):
        for k, v in xs.items(): w.add_key_value("moss_transcribe." + k, float(v), V.FLOAT32)
    def strings(**xs):
        for k, v in xs.items(): w.add_string("moss_transcribe." + k, v)
    ints(**{"enc.num_mel_bins":128,"enc.encoder_layers":32,"enc.d_model":1280,"enc.encoder_attention_heads":20,"enc.head_dim":64,"enc.encoder_ffn_dim":5120,"enc.downsample_hidden_size":480,"enc.max_source_positions":1500,"enc.n_window":50,"enc.n_window_infer":800,"enc.conv_chunksize":500,"enc.output_dim":2048,"adapter.input_size":2048,"adapter.hidden_size":8192,"adapter.output_size":2048,"llm.hidden_size":2048,"llm.num_layers":28,"llm.num_heads":16,"llm.num_kv_heads":8,"llm.head_dim":128,"llm.intermediate_size":6144,"llm.vocab_size":151936,"llm.max_position_embeddings":40960,"frontend.sample_rate":16000,"frontend.n_fft":640,"frontend.win_length":640,"frontend.hop_length":160,"frontend.n_mels":128,"frontend.power":2,"pad_token_id":151643,"eos_token_id":151645,"start_token_id":151644,"audio_start_id":151669,"audio_end_id":151670,"audio_placeholder_id":0,"max_new_tokens":200,"max_cache_len":2048})
    floats(**{"enc.layer_norm_eps":1e-5,"llm.rope_theta":1000000,"llm.rms_norm_eps":1e-6,"frontend.mel_floor":1e-10,"frontend.dynamic_range":8,"frontend.normalization_offset":4,"frontend.normalization_divisor":4})
    strings(**{"llm.rope_scaling":"none","frontend.pad_mode":"reflect","frontend.mel_scale":"slaney","frontend.mel_norm":"slaney","frontend.log":"log10","frontend.output_dtype":"bf16"})
    w.add_key_value("moss_transcribe.llm.tied_embeddings", True, V.BOOL)
    w.add_key_value("moss_transcribe.frontend.center", True, V.BOOL)
    w.add_key_value("moss_transcribe.prompt_prefix", [151644,872,198,151669], V.ARRAY, V.INT32)
    w.add_key_value("moss_transcribe.prompt_suffix", [151670,151645,198,151644,77091,198], V.ARRAY, V.INT32)

def tokenizer(w: gguf.GGUFWriter, snapshot: Path) -> None:
    data = json.loads((snapshot / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    tokens = [None] * 151936
    for token, i in vocab.items(): tokens[i] = token
    special = set()
    for item in data.get("added_tokens", []):
        tokens[item["id"]] = item["content"]
        if item.get("special", False): special.add(item["id"])
    # The checkpoint declares 151936 entries while its tokenizer assets define
    # only 151669. Preserve defined entries verbatim and make the undeclared
    # model-vocabulary tail explicit control placeholders; this retains a
    # complete GGUF table without pretending those strings came from HF.
    undefined = set()
    for i, token in enumerate(tokens):
        if token is None:
            # tokenizer.json ends at 151668 although config.vocab_size is
            # 151936.  The checkpoint does not define strings/classification
            # for this reserved tail. Use explicit conventional PAD spellings
            # and mark them UNUSED rather than claiming they are HF specials.
            tokens[i] = "[PAD" + str(i) + "]"
            undefined.add(i)
    # The processor contract names these two otherwise undeclared IDs.
    tokens[151669] = "<|audio_start|>"; tokens[151670] = "<|audio_end|>"
    special.update((151669, 151670))
    merges = data["model"]["merges"]
    merges = [" ".join(x) if isinstance(x, list) else x for x in merges]
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * len(tokens))
    w.add_token_types([gguf.TokenType.UNUSED if i in undefined and i not in special else gguf.TokenType.CONTROL if i in special else gguf.TokenType.NORMAL for i in range(len(tokens))])
    w.add_token_merges(merges)
    w.add_bos_token_id(151643); w.add_eos_token_id(151645); w.add_pad_token_id(151643)

def frontend(snapshot: Path) -> tuple[np.ndarray, np.ndarray]:
    # Use precisely the same Transformers Whisper implementation called by the
    # vendored MossProcessor, with its Starling-required n_fft=640.
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    from transformers.audio_utils import window_function
    fx = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, n_fft=640)
    return np.asarray(fx.mel_filters.T, dtype=np.float32), np.asarray(window_function(640, "hann"), dtype=np.float32)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT); ap.add_argument("--output", type=Path, default=Path("models/moss-transcribe-preview-2b-bf16-exact.gguf")); args = ap.parse_args()
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    shard = args.snapshot / next(iter(set(index["weight_map"].values())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.output, "moss_transcribe", use_temp_file=True)
    add_metadata(w); tokenizer(w, args.snapshot)
    learned = 0; dtypes = set()
    with safe_open(shard, framework="pt", device="cpu") as f:
        for source in sorted(index["weight_map"]):
            t = f.get_tensor(source); dtypes.add(str(t.dtype))
            if t.dtype is not torch.bfloat16: raise TypeError(f"{source}: expected BF16, found {t.dtype}")
            # GGUF stores dimensions innermost-first; GGUFReader consequently
            # displays this native HF [out,in] metadata as ggml [in,out].
            # Keep the BF16 byte stream in checkpoint row-major order.
            a = np.ascontiguousarray(t.view(torch.uint16).numpy())
            w.add_tensor(gguf_name(source), a, raw_shape=a.shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
            learned += 1
    mel, window = frontend(args.snapshot)
    w.add_tensor("audio.mel_filters", mel.T) # GGML display shape is [128,321]
    w.add_tensor("audio.mel_window", window)
    # Torch's recorded F32 formulation, stored to avoid host math drift.
    k = np.arange(640, dtype=np.float32); inv = np.exp(-np.float32(np.log(10000.0) / 639.0) * k)
    p = np.arange(1500, dtype=np.float32)[:, None]; angles = p * inv[None, :]
    pos = np.concatenate((np.sin(angles), np.cos(angles)), axis=1).astype(np.float32)
    w.add_tensor("enc.positional_embedding", pos)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.output}: {learned + 3} tensors ({learned} learned), source dtypes={sorted(dtypes)}")
if __name__ == "__main__": main()
