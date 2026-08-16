import sys; sys.path.insert(0,"src")
import numpy as np, torch
from dataclasses import asdict
from starling.higgs.vendor.modeling import HiggsAudio3Config, HiggsAudio3Model
from starling.higgs.vendor import HiggsAudioSampleCollator, ChatMLDatasetSample
from transformers import AutoTokenizer, WhisperProcessor
import soundfile as sf
from transformers import DynamicCache

MODEL_ID="bosonai/higgs-audio-v3-stt"
cfg = HiggsAudio3Config.from_pretrained(MODEL_ID)
model = HiggsAudio3Model.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
coll = HiggsAudioSampleCollator(whisper_processor=wp, audio_in_token_id=cfg.audio_in_token_idx, audio_out_token_id=cfg.audio_out_token_idx, audio_stream_bos_id=cfg.audio_stream_bos_id, audio_stream_eos_id=cfg.audio_stream_eos_id, encode_whisper_embed=cfg.encode_whisper_embed, pad_token_id=cfg.pad_token_id, return_audio_in_tokens=cfg.encode_audio_in_tokens, use_delay_pattern=cfg.use_delay_pattern, round_to=1, audio_num_codebooks=cfg.audio_num_codebooks, chunk_size_seconds=30.0, pad_left=False)

def enc(s): return tok.encode(s, add_special_tokens=False)
input_ids = enc("<|im_start|>user\n") + enc("Transcribe the speech. Output only the spoken words in lowercase with no punctuation.") + enc("<|audio_bos|><|AUDIO|><|audio_eos|>") + enc("<|im_end|>\n") + enc("<|im_start|>assistant\n")
audio_np, sr = sf.read("/home/m0hawk/Documents/starling/tests/fixtures/short.wav"); audio_np=np.asarray(audio_np,dtype=np.float32)
sample = ChatMLDatasetSample(input_ids=torch.LongTensor(input_ids), label_ids=torch.LongTensor([-100]*len(input_ids)), audio_ids_concat=torch.zeros((1,0),dtype=torch.long), audio_ids_start=torch.tensor([0],dtype=torch.long), audio_waveforms_concat=torch.tensor(audio_np,dtype=torch.float32), audio_waveforms_start=torch.tensor([0]), audio_sample_rate=torch.tensor([16000]), audio_speaker_indices=torch.tensor([0]))
batch = asdict(coll([sample])); batch={k:(v.to("cuda").contiguous() if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
with torch.inference_mode():
    out = model.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], audio_features=batch["audio_features"], audio_feature_attention_mask=batch["audio_feature_attention_mask"], past_key_values=DynamicCache(), use_cache=True)
print("LOGITS SHAPE:", tuple(out.logits.shape), "dtype", out.logits.dtype)
print("first-token argmax:", int(out.logits[0,-1].argmax()))
