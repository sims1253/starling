# coding: utf-8
from typing import Optional, Tuple, Union
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import torchaudio
    import librosa
except ImportError:
    torchaudio = None
    librosa = None
try:
    from omegaconf import OmegaConf
except Exception as e:
    OmegaConf = None  # only needed for ckpt/yaml loader
# xcodec is only needed when audio_encoder_config.model_type == "higgs_audio_encoder_xcodec"
# Make imports lazy so the model can be loaded for ASR/understanding without xcodec installed.
try:
    import xcodec.descriptaudiocodec.dac.model.dac as dac2
    from xcodec.modules.semantic_module import Encoder as SemanticEncoder
    from xcodec.models.semantic.w2v_bert2 import W2V2Wrapper
    from xcodec.models.semantic.whisper_kimi import WhisperKimiWrapper
    from xcodec.models.semantic.mert import MERTBatchWrapper
    _XCODEC_AVAILABLE = True
except ImportError:
    dac2 = None
    SemanticEncoder = None
    W2V2Wrapper = None
    WhisperKimiWrapper = None
    MERTBatchWrapper = None
    _XCODEC_AVAILABLE = False
from transformers import AutoModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

try:
    from .common import HiggsAudioPreTrainedModel
except ImportError:
    from common import HiggsAudioPreTrainedModel


def _encoder_semantic_out_dim(st: str) -> int:
    if st in ("hubert_base", "hubert_base_general", "wavlm_base_plus"):
        return 768
    if st == "w2v_bert2":
        return 1024
    if st in ("whisper_kimi", "whisper-kimi"):
        return 512
    if st == "mert_music":
        return 768
    if st == "mert_music-330m":
        return 1024
    raise ValueError(f"Unsupported semantic_techer: {st}")

class HiggsAudioEncoderXcodecConfig(PretrainedConfig):
    """XCodec-based audio tower that returns the pre-quantizer embedding `e` (fc_prior output)."""
    model_type = "higgs_audio_encoder_xcodec"

    def __init__(
        self,
        # Will be overridden by YAML if you use the loader below
        sample_rate: int = 24000,
        ratios=(8, 5, 4, 2, 3),     # 24k / (8*5*4*2*3=960) -> 25 Hz
        D: int = 256,               # acoustic encoder channels
        semantic_techer: str = "hubert_base_general",
        downsample_mode: str = "step_down",  # 'step_down' | 'avg'
        vq_scale: int = 1,          # affects quantizer_dim = (D + sem_dim) // vq_scale
        max_source_seconds: int = 30,
        pad_token_id: int = 128001,
        # Whether to freeze the teacher (recommended)
        freeze_semantic_teacher: bool = True,
        # Optionally override teacher sampling rate (usually auto from wrapper)
        semantic_sample_rate: Optional[int] = None,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.ratios = tuple(ratios)
        self.D = D
        self.semantic_techer = semantic_techer
        self.downsample_mode = downsample_mode
        self.vq_scale = vq_scale
        self.max_source_seconds = max_source_seconds
        self.pad_token_id = pad_token_id
        self.freeze_semantic_teacher = freeze_semantic_teacher
        self.semantic_sample_rate = semantic_sample_rate
        self.init_std = float(init_std)
        self.d_model = (D + _encoder_semantic_out_dim(semantic_techer)) // max(1, vq_scale)
        hop = 1
        for r in self.ratios:
            hop *= r
        self.hop_length = hop
        self.frame_rate = max(1, self.sample_rate / self.hop_length)
        self.max_source_positions = self.frame_rate * self.max_source_seconds


class _SemanticTeacher(nn.Module):
    """Produces frame-level semantic features at teacher SR, then we downsample/step to match acoustic frames."""
    def __init__(self, which: str, explicit_sr: Optional[int] = None, last_layer_semantic: bool = True):
        super().__init__()
        self.which = which
        self.last_layer_semantic = last_layer_semantic
        if which in ("hubert_base", "hubert_base_general"):
            self.model = AutoModel.from_pretrained(
                "facebook/hubert-base-ls960" if which == "hubert_base"
                else "ZhenYe234/hubert_base_general_audio"
            )
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
        elif which == "wavlm_base_plus":
            self.model = AutoModel.from_pretrained("microsoft/wavlm-base-plus")
            self.semantic_sample_rate = 16000
            self.semantic_dim = 768
        elif which == "w2v_bert2":
            self.model = W2V2Wrapper(model_name="facebook/w2v-bert-2.0", sampling_rate=16000)
            self.semantic_sample_rate = 16000
            self.semantic_dim = 1024
        elif which in ("whisper_kimi", "whisper-kimi"):
            self.model = WhisperKimiWrapper(model_name="openai/whisper-base",
                                            sampling_rate=16000,
                                            requires_hidden_states=not last_layer_semantic)
            self.semantic_sample_rate = 16000
            self.semantic_dim = 512
        elif which == "mert_music":
            self.model = MERTBatchWrapper(model_name="/ceph/models/MERT/v1-95M/", input_sampling_rate=24000)
            self.semantic_sample_rate = 24000
            self.semantic_dim = 768
        elif which == "mert_music-330m":
            self.model = MERTBatchWrapper(model_name="/ceph/models/MERT/v1-330M/", input_sampling_rate=24000)
            self.semantic_sample_rate = 24000
            self.semantic_dim = 1024
        else:
            raise ValueError(f"Unsupported semantic_techer: {which}")
        if explicit_sr is not None:
            self.semantic_sample_rate = explicit_sr

    @torch.no_grad()
    def forward(self, x_24k_mono: torch.Tensor, avg_layers_if_available: bool = False) -> torch.Tensor:
        """
        x_24k_mono: (B, 1, T) @ (any SR) — caller must resample to self.semantic_sample_rate before calling...
        Returns: (B, T_sem, C_sem)
        """
        if hasattr(self.model, "forward"):
            if isinstance(self.model, (W2V2Wrapper, WhisperKimiWrapper, MERTBatchWrapper)):
                out = self.model(x_24k_mono, avg_layers=avg_layers_if_available if hasattr(self.model, "forward") else False)
                return out  # already (B, T_sem, C)
            else:
                # from Xcodec's original code
                xt = x_24k_mono[:, 0, :]
                xt = F.pad(xt, (160, 160))
                o = self.model(xt, output_hidden_states=True).hidden_states
                o = torch.stack(o, dim=1).mean(1)
                return o  # (B, T_sem, C)
        raise RuntimeError("Unexpected teacher model type")


class XCodecFrontEnd(nn.Module):
    """
    Front-end to compute pre-quantizer embedding without quantizer/decoder.
    e = fc_prior(cat( e_acoustic, e_semantic ).transpose(1, 2))  # -> (B, T', Q)
    """
    def __init__(self, cfg: HiggsAudioEncoderXcodecConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = dac2.Encoder(64, list(cfg.ratios), cfg.D)
        self.teacher = _SemanticTeacher(cfg.semantic_techer, explicit_sr=cfg.semantic_sample_rate)
        self.semantic_dim = _encoder_semantic_out_dim(cfg.semantic_techer)
        self.encoder_semantic = self._make_semantic_encoder(self.teacher.semantic_dim, self.semantic_dim)
        self.quantizer_dim = (cfg.D + self.semantic_dim) // max(1, cfg.vq_scale)
        self.fc_prior = nn.Linear(cfg.D + self.semantic_dim, self.quantizer_dim)
        if cfg.downsample_mode == "avg":
            # semantic_downsample_factor matches SoundStream logic:
            # hop_length / (audio_sr / teacher_sr) / 320
            sem_factor = int(self._semantic_downsample_factor())
            self.semantic_pool = nn.AvgPool1d(kernel_size=sem_factor, stride=sem_factor)
        else:
            self.semantic_pool = None
        if cfg.freeze_semantic_teacher:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def _semantic_downsample_factor(self) -> int:
        # Mirrors: hop_length / (sample_rate / semantic_sample_rate) / 320
        sr = self.cfg.sample_rate
        tsr = self.teacher.semantic_sample_rate
        return int(self.cfg.hop_length / (sr / tsr) / 320)

    def _make_semantic_encoder(self, in_channels: int, out_channels: int) -> nn.Module:
        # Mirrors XCodec: encoder_semantic = Encoder(input_channels=semantic_dim, encode_channels=encoder_semantic_dim); kept default hyperparams so the state_dict from model.pth maps cleanly.
        return SemanticEncoder(
            input_channels=in_channels,
            encode_channels=out_channels,
            channel_ratios=(1, 1),
            strides=(1, 1),
            kernel_size=3,
            bias=True,
            block_dilations=(1, 1),
            unit_kernel_size=3,
        )

    @torch.no_grad()
    def _teacher_feats(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) @ cfg.sample_rate; Returns semantic feats aligned (downsampled) to approx acoustic frames.
        if self.teacher.semantic_sample_rate != self.cfg.sample_rate:
            xt = torchaudio.functional.resample(x, self.cfg.sample_rate, self.teacher.semantic_sample_rate)
        else:
            xt = x
        sem = self.teacher(xt, avg_layers_if_available=not self.teacher.last_layer_semantic)  # (B, T_sem, C_sem)
        if self.cfg.downsample_mode == "step_down":
            factor = max(1, self._semantic_downsample_factor())
            if factor > 1:
                sem = sem[:, ::factor, :]
        else:
            sem = self.semantic_pool(sem.transpose(1, 2)).transpose(1, 2)
        return sem

    def forward(self, x_wave: torch.Tensor) -> torch.Tensor:
        """
        x_wave: (B, 1, T) @ cfg.sample_rate
        Returns e: (B, T_frames, quantizer_dim)
        """
        # acoustic path
        e_acoustic = self.encoder(x_wave)  # (B, D, T')
        # semantic path
        sem = self._teacher_feats(x_wave)  # (B, T_sem, C_sem)
        e_semantic = self.encoder_semantic(sem.transpose(1, 2))  # (B, C_out, T_sem')
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            Ta, Ts = e_acoustic.shape[2], e_semantic.shape[2]
            if Ta > Ts:
                e_acoustic = e_acoustic[:, :, :Ts]
            else:
                e_semantic = e_semantic[:, :, :Ta]
        z = torch.cat([e_acoustic, e_semantic], dim=1)          # (B, D+C_out, T)
        e = self.fc_prior(z.transpose(1, 2))                    # (B, T, Q)
        return e

    @staticmethod
    def filter_state_dict_for_frontend(sd: dict) -> dict:
        """Keep only encoder / encoder_semantic / fc_prior weights. Strip 'module.' if present."""
        out = {}
        for k, v in sd.items():
            kk = k[7:] if k.startswith("module.") else k
            if kk.startswith("encoder.") or kk.startswith("encoder_semantic.") or kk.startswith("fc_prior."):
                out[kk] = v
        return out


# HiggsAudioEncoderXcodec - High-level tower module, FIXME later can be merged with HiggsAudioEncoder
class HiggsAudioEncoderXcodec(HiggsAudioPreTrainedModel):
    config_class = HiggsAudioEncoderXcodecConfig
    main_input_name = "input_values"  # raw waveform

    def __init__(self, config: HiggsAudioEncoderXcodecConfig):
        super().__init__(config)
        self.config = config
        self.frontend = XCodecFrontEnd(config)
        self.post_init()

    def _ensure_wave(self, input_values: torch.Tensor) -> torch.Tensor:
        # Accept (B, T) or (B, 1, T)
        x = input_values
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(1) == 1, "input_values must be (B, T) or (B, 1, T)"
        return x

    def _resample_if_needed(self, x: torch.Tensor, sr_in: Optional[int]) -> torch.Tensor:
        if sr_in is not None and sr_in != self.config.sample_rate:
            return torchaudio.functional.resample(x, sr_in, self.config.sample_rate)
        return x

    def forward(
        self,
        input_values: torch.Tensor, # (B, T) or (B, 1, T)
        sampling_rate: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # ignored for xcodec branch
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        check_seq_length: bool = False,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        x = input_values
        # Produce pre-quantizer embedding e
        e = self.frontend(x)    # (B, T', Q)
        if check_seq_length and e.size(1) > self.config.max_source_positions:
            raise ValueError(
                f"Too many frames ({e.size(1)}) for max_source_positions={self.config.max_source_positions}."
            )
        return (e, None, None) if not return_dict else BaseModelOutput(last_hidden_state=e, hidden_states=None, attentions=None)


# Utilities for loading from YAML + model.pth
def build_xcodec_config_from_yaml(
    yaml_path: str,
    base_cfg_overrides: Optional[dict] = None
) -> HiggsAudioEncoderXcodecConfig:
    if OmegaConf is None:
        raise ImportError("omegaconf is required to read XCodec YAML")
    oc = OmegaConf.load(yaml_path)
    gen = oc.generator
    assert gen.name == "SoundStream", f"Expected SoundStream in YAML, got {gen.name}"
    gcfg = gen.config
    ratios = tuple(int(r) for r in gcfg.get("ratios"))
    cfg = HiggsAudioEncoderXcodecConfig(
        sample_rate=int(gcfg.get("sample_rate", 24000)),
        ratios=ratios,
        D=int(gcfg.get("D", 256)),
        semantic_techer=str(gcfg.get("semantic_techer", "hubert_base_general")),
        downsample_mode="step_down",
        vq_scale=int(gcfg.get("vq_scale", 1)),
        max_source_seconds=300,  # this shouldn't matter at inference time, but training can get oom if too long
        pad_token_id=128001,
        freeze_semantic_teacher=True,  # teacher frozen by default
        semantic_sample_rate=None,
    )
    if base_cfg_overrides:
        for k, v in base_cfg_overrides.items():
            setattr(cfg, k, v)
    return cfg


def load_xcodec_frontend_weights_from_model_pth(
    tower: HiggsAudioEncoderXcodec,
    model_pth: str,
) -> None:
    # Load only encoder / encoder_semantic / fc_prior weights from XCodec model.pth - these are the Front-End weights
    sd = torch.load(model_pth, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = XCodecFrontEnd.filter_state_dict_for_frontend(sd)
    fcw = sd.get("fc_prior.weight")
    assert fcw is not None, "fc_prior.weight not found in XCodec checkpoint"
    out_features, in_features = fcw.shape
    assert out_features == tower.config.d_model == tower.frontend.quantizer_dim, "fc_prior.weight shape mismatch"
    missing, unexpected = tower.frontend.load_state_dict(sd, strict=False)
    if missing:
        print(f"[XCodec] Missing keys in frontend load (ok if unrelated): {len(missing)}")
    if unexpected:
        print(f"[XCodec] Unexpected keys (ignored): {len(unexpected)}")


class XCodecProcessor:
    """
    Batch processor for XCodec raw waveform inputs, analogous to WhisperProcessor.feature_extractor
    Returns:
      - input_features: np.ndarray, shape (N, 1, T_pad), float32, raw 24k waveforms
      - attention_mask: usually None to save memory
      - audio_wv_lengths: np.ndarray, shape (N,), int64, true (unpadded) lengths in samples at 24k
    """

    class FeatureExtractor:
        def __init__(
            self,
            sampling_rate: int = 24000,
            hop_length: int = 960,   # product of ratios (8*5*4*2*3) -> 25 fps at 24k; for 12.5 it should be 960*2
            chunk_size_seconds: float = 0.0,   # use 0.0 to disable chunking
            pad_value: float = 0.0,
            temporal_downsample: int = 1,
        ):
            self.sampling_rate = int(sampling_rate)
            self.hop_length = int(hop_length)
            self.chunk_size_seconds = float(chunk_size_seconds) if chunk_size_seconds else 0.0
            self.max_samples = int(round(self.chunk_size_seconds * self.sampling_rate)) if self.chunk_size_seconds > 0 else 0
            self.pad_value = float(pad_value)
            self.temporal_downsample = int(temporal_downsample)

        def _resample(self, wav: np.ndarray, orig_sr: int) -> np.ndarray:
            if orig_sr == self.sampling_rate:
                return wav
            return librosa.resample(wav, orig_sr=orig_sr, target_sr=self.sampling_rate)

        def _ceil_to_nearest(self, n: int, m: int) -> int:
            return ((n + m - 1) // m) * m

        def __call__(
            self,
            audios: list[np.ndarray] | list[float] | list[list[float]],
            sampling_rate: int,
            return_attention_mask: bool = False,  # should not use this in processor to save memory
            padding: str = "longest",  # kept for API similarity; only "longest" is meaningful here
        ) -> dict:
            assert sampling_rate == self.sampling_rate, f"Sampling rate mismatch: {sampling_rate} != {self.sampling_rate}"
            assert padding == "longest", f"Padding mode {padding} not supported"
            normed = []
            for a in audios:
                arr = np.asarray(a, dtype=np.float32).reshape(-1)
                normed.append(arr)
            # resample + chunk
            chunks = []
            for wav_24k in normed:
                # wav_24k = self._resample(wav, orig_sr=sampling_rate)  # should enforce this in collator previous part
                if self.max_samples > 0:
                    assert wav_24k.shape[0] <= self.max_samples, f"Audio chunk longer than chunk_size_seconds {self.chunk_size_seconds} seconds"
                chunks.append(wav_24k)
            if len(chunks) == 0:
                # consistent empty outputs
                return {
                    "input_features": np.zeros((0, 1, 0), dtype=np.float32),
                    "attention_mask": np.zeros((0, 0), dtype=np.int32) if return_attention_mask else None,
                    "audio_wv_lengths": np.zeros((0,), dtype=np.int64),
                }
            lengths = np.asarray([c.shape[0] for c in chunks], dtype=np.int64)
            T_max = int(lengths.max())
            if self.hop_length > 0:
                T_max = self._ceil_to_nearest(T_max, self.hop_length * self.temporal_downsample)
            feats = np.full((len(chunks), 1, T_max), fill_value=self.pad_value, dtype=np.float32)
            attn = np.zeros((len(chunks), T_max), dtype=np.int32) if return_attention_mask else None
            for i, c in enumerate(chunks):
                t = c.shape[0]
                feats[i, 0, :t] = c
                if attn is not None:
                    attn[i, :t] = 1
            return {
                "input_features": feats,   # (N, 1, T_pad) float32
                "attention_mask": attn,    # (N, T_pad) int32 or None
                "audio_wv_lengths": lengths,        # (N,) int64
            }

    def __init__(
        self,
        sampling_rate: int = 24000,
        hop_length: int = 960,
        chunk_size_seconds: float = 30.0,
        pad_value: float = 0.0,
        temporal_downsample: int = 1,
    ):
        # expose a "feature_extractor" attr to match WhisperProcessor usage
        self.feature_extractor = self.FeatureExtractor(
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_size_seconds=chunk_size_seconds,
            pad_value=pad_value,
            temporal_downsample=temporal_downsample,
        )

