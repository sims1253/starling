from dataclasses import dataclass
from typing import Optional, Union, List
import torch
import numpy as np
from transformers import BatchEncoding
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import importlib.util
import sys


@dataclass
class MelConfig:
    mel_sr: int = 16000
    mel_dim: int = 80
    mel_n_fft: int = 640
    mel_hop_length: int = 160
    mel_dtype: torch.dtype = torch.bfloat16


def load_chat_template(template_path: str, package_path: Optional[str] = None) -> List:
    """Dynamically import a chat template module by file path and return its `chat_template`."""
    import os

    if package_path and package_path not in sys.path:
        sys.path.insert(0, package_path)

    spec = importlib.util.spec_from_file_location("chat_template_module", template_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["chat_template_module"] = module
    spec.loader.exec_module(module)
    return module.chat_template


class MossProcessor:
    """Audio processor for MOSS-Transcribe: mel-spectrogram + chat-template-driven token layout."""

    def __init__(
        self,
        tokenizer,
        config: Optional[MelConfig] = None,
        template_path: Optional[str] = None,
        enable_time_marker: bool = False,
    ):
        self.tokenizer = tokenizer
        self.config = config or MelConfig()

        # Whisper log-mel frontend — matches the front-end the model was trained with.
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=int(self.config.mel_dim),
            sampling_rate=int(self.config.mel_sr),
            hop_length=int(self.config.mel_hop_length),
            n_fft=int(self.config.mel_n_fft),
        )

        # Special token ids (Qwen3 tokenizer).
        self.start_token_id = 151644
        self.end_token_id = 151645
        self.audio_start_token_id = 151669
        self.audio_end_token_id = 151670
        self.audio_placeholder_id = 0

        self.chat_template = None if template_path is None else load_chat_template(template_path)
        self.enable_time_marker = enable_time_marker

        # Digit tokens 0-9 in the Qwen3 tokenizer, used for time markers.
        self._digit_token_ids = {str(d): 15 + d for d in range(10)}
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )

    def load_template(self, template_path: str):
        self.chat_template = load_chat_template(template_path)
        print(f"Loaded chat template from {template_path}")
        return self

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Map raw mel-frame count to number of audio tokens after the encoder downsample."""
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        )
        return output_lengths

    def _get_time_marker_token_ids(self, second: int) -> List[int]:
        return [self._digit_token_ids[c] for c in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> List[int]:
        """Interleave time markers every `time_marker_every_seconds` seconds of audio tokens."""
        num_full_seconds = int(audio_seq_len / self.audio_tokens_per_second)

        tokens_list: List[int] = []
        audio_tokens_consumed = 0

        for second in range(
            self.time_marker_every_seconds, num_full_seconds + 1, self.time_marker_every_seconds
        ):
            marker_pos = (
                (second // self.time_marker_every_seconds) * self.time_marker_every_audio_tokens
            )
            segment_len = marker_pos - audio_tokens_consumed
            if segment_len > 0:
                tokens_list.extend([self.audio_placeholder_id] * segment_len)
                audio_tokens_consumed += segment_len
            tokens_list.extend(self._get_time_marker_token_ids(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            tokens_list.extend([self.audio_placeholder_id] * remaining)
        return tokens_list

    def _build_input_from_template(self, num_audio_tokens: int) -> tuple:
        """Walk the loaded chat_template and emit (input_ids, audio_input_mask) for inference."""
        if self.chat_template is None:
            raise ValueError("Chat template not loaded. Call load_template() first.")

        input_ids: List[int] = []
        audio_mask: List[bool] = []

        for segment in self.chat_template:
            seg_type = segment.type

            if seg_type == "constant_text_token":
                text_ids = segment.text_ids.tolist()
                input_ids.extend(text_ids)
                audio_mask.extend([False] * len(text_ids))

            elif seg_type in ("audio_contiguous", "audio_token"):
                if self.enable_time_marker:
                    audio_ids = self._build_audio_tokens_with_time_markers(num_audio_tokens)
                    input_ids.extend(audio_ids)
                    audio_mask.extend(
                        [tok == self.audio_placeholder_id for tok in audio_ids]
                    )
                else:
                    input_ids.extend([self.audio_placeholder_id] * num_audio_tokens)
                    audio_mask.extend([True] * num_audio_tokens)

            elif seg_type == "text_token":
                # Generation starts here at inference time.
                break

        return input_ids, audio_mask

    def _build_input_legacy(self, num_audio_tokens: int) -> tuple:
        """Hardcoded [start, audio_start, audio*, audio_end] layout, used when no template is loaded."""
        if self.enable_time_marker:
            audio_ids = self._build_audio_tokens_with_time_markers(num_audio_tokens)
            ids = (
                [self.start_token_id, self.audio_start_token_id]
                + audio_ids
                + [self.audio_end_token_id]
            )
            audio_mask = [tok == self.audio_placeholder_id for tok in audio_ids]
            mask = [False, False] + audio_mask + [False]
        else:
            ids = (
                [self.start_token_id, self.audio_start_token_id]
                + [self.audio_placeholder_id] * num_audio_tokens
                + [self.audio_end_token_id]
            )
            mask = [False, False] + [True] * num_audio_tokens + [False]
        return ids, mask

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        return_tensors: str = "pt",
        **kwargs,
    ):
        if audio is None:
            raise ValueError("Audio input is required.")

        if isinstance(audio, torch.Tensor):
            waveform = audio.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform[0]

        try:
            mel = self.feature_extractor._np_extract_fbank_features(
                waveform[None, ...], device="cpu"
            )[0]
        except TypeError:
            mel = self.feature_extractor._np_extract_fbank_features(waveform[None, ...])[0]
        input_features = torch.from_numpy(mel).to(self.config.mel_dtype)
        if input_features.dim() == 3:
            input_features = input_features.squeeze(0)

        raw_mel_len = input_features.shape[-1]
        num_audio_tokens = self._get_feat_extract_output_lengths(raw_mel_len)

        if self.chat_template is not None:
            ids, mask = self._build_input_from_template(num_audio_tokens)
        else:
            ids, mask = self._build_input_legacy(num_audio_tokens)

        input_ids_tensor = torch.tensor([ids], dtype=torch.long)
        audio_mask_tensor = torch.tensor([mask], dtype=torch.bool)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)
        seq_lens_tensor = torch.tensor([raw_mel_len], dtype=torch.long)

        data = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "audio_data": input_features,
            "audio_data_seqlens": seq_lens_tensor,
            "audio_input_mask": audio_mask_tensor,
        }
        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
