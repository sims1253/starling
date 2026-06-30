"""Vendored ``ChatMLDatasetSample`` dataclass.

Byte-for-byte copy of the upstream ``boson_multimodal.dataset.chatml_dataset``
class (the part the collator needs). Vendored so this package does not depend on
``boson_multimodal``, which pins ``transformers==4.46.3`` and is incompatible
with this repo's ``transformers 5.13.0.dev0``. The only edit vs upstream is
replacing the ``loguru`` ``logger`` with stdlib ``logging`` (used only inside
``ChatMLDatasetSample.merge``, which the collator never calls). Everything else
-- fields, methods, numerics -- is identical so prefill embeddings/masks match
the reference ``transcribe()`` path bit-for-bit.
"""

import logging
import numpy as np
import torch
from abc import ABC, abstractmethod  # noqa: F401  (kept for parity with upstream imports)
from dataclasses import dataclass, fields
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC = 25


@dataclass
class ChatMLDatasetSample:
    input_ids: torch.LongTensor  # Shape (seq_len,): The input text tokens.
    label_ids: torch.LongTensor  # Shape (seq_len,): The label ids.
    audio_ids_concat: torch.LongTensor  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    audio_ids_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio token in the concatenated audio tokens.
    audio_waveforms_concat: (
        torch.Tensor
    )  # Shape (total_wv_length,): The concatenated audio waveforms for audio-in features.
    audio_waveforms_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio waveform in the concatenated audio waveforms.
    audio_sample_rate: torch.Tensor  # Shape (num_audios,): The sampling rate of the audio waveforms.
    audio_speaker_indices: (
        torch.LongTensor
    )  # Shape (num_audios,) -1 means unknown speaker: The speaker indices for each audio.
    audio_label_ids_concat: Optional[torch.LongTensor] = (
        None  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    )
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    reward: Optional[float] = None

    def num_audios(self):
        return max(len(self.audio_waveforms_start), len(self.audio_ids_start))

    def get_audio_codes(self, idx):
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_ids_concat[:, code_start:code_end]

    def get_audio_codes_labels(self, idx):
        if self.audio_label_ids_concat is None:
            return None
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_label_ids_concat[:, code_start:code_end]

    def get_wv(self, idx):
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[wv_start:wv_end], sr

    def cal_num_tokens(
        self,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = False,
        encode_audio_out_tokens: bool = True,
        audio_in_token_id: int = 128015,
        audio_out_token_id: int = 128016,
    ) -> int:
        # we firstly exclude <|AUDIO|> and <|AUDIO_OUT|> because we do late merging and replace those position with actual audio features and audio token ids
        # It's assumed that we always have audio_ids when audio_waveforms are there (but not vice-versa)
        num_tokens = len(self.input_ids) - len(self.audio_ids_start)

        if encode_whisper_embed and len(self.audio_waveforms_concat) > 0:
            audio_lengths = torch.diff(self.audio_waveforms_start)
            if len(audio_lengths):
                # Sum before calling .item()
                num_tokens += (
                    (
                        np.ceil(WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC * audio_lengths / self.audio_sample_rate[:-1])
                    ).sum()
                ).item()
            # add the last audio's token estimation
            num_tokens += (
                np.ceil(
                    WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC
                    * (self.audio_waveforms_concat.shape[0] - self.audio_waveforms_start[-1])
                    / self.audio_sample_rate[-1]
                )
            ).item()

        if self.audio_ids_concat.size(1) > 0:
            audio_io_ids = self.input_ids[
                (self.input_ids == audio_in_token_id) | (self.input_ids == audio_out_token_id)
            ]
            audio_io_id_lengths = torch.concat(
                [
                    torch.diff(self.audio_ids_start),
                    torch.tensor([self.audio_ids_concat.shape[-1] - self.audio_ids_start[-1]]),
                ]
            )
            if encode_audio_in_tokens:
                num_tokens += torch.sum(audio_io_id_lengths[audio_io_ids == audio_in_token_id]).item()

            if encode_audio_out_tokens:
                num_tokens += torch.sum(audio_io_id_lengths[audio_io_ids == audio_out_token_id]).item()

        return int(num_tokens)

    @classmethod
    def merge(
        cls,
        samples: List["ChatMLDatasetSample"],
        eos_token_id: int,
        ignore_index: int,
        padding_size: Optional[int] = None,
    ) -> "ChatMLDatasetSample":
        """Merges a list of ChatMLDatasetSample instances, inserting eos_token_id and ignore_index between them, and adjusting offsets for audio_ids_start and audio_waveforms_start.

        Args:
            samples (List[ChatMLDatasetSample]): List of samples to merge.
            eos_token_id (int): Tokens to be inserted into input_ids between samples.
            ignore_index (int): Default label for padding.
            padding_size (Optional[int]): If provided, pad the sequence to with this length.

        Returns:
            ChatMLDatasetSample: Merged and potentially padded sample.
        """
        if not samples:
            logger.fatal("The samples list is empty and cannot be merged.")
            raise ValueError("The samples list is empty and cannot be merged.")

        # Initialize empty lists for concatenation
        input_ids_list = []
        label_ids_list = []
        audio_ids_concat_list = []
        audio_ids_start_list = []
        audio_waveforms_concat_list = []
        audio_waveforms_start_list = []
        audio_sample_rate_list = []
        audio_speaker_indices_list = []

        # Track offsets
        audio_ids_offset = 0
        audio_waveforms_offset = 0

        for sample in samples:
            # Add input_ids and label_ids with padding
            if input_ids_list:
                input_ids_list.append(torch.tensor([eos_token_id], dtype=torch.long))
                label_ids_list.append(torch.tensor([ignore_index], dtype=torch.long))
            input_ids_list.append(sample.input_ids)
            label_ids_list.append(sample.label_ids)

            # Add audio_ids_concat and handle empty audio ids
            if sample.audio_ids_concat.size(1) > 0:
                audio_ids_concat_list.append(sample.audio_ids_concat)

                # Offset and add audio_ids_start
                audio_ids_start_list.append(sample.audio_ids_start + audio_ids_offset)
                audio_ids_offset += sample.audio_ids_concat.size(
                    1
                )  # (num_codebooks, seq_len): Update offset by audio_seq_len

            # Add audio_waveforms_concat
            if sample.audio_waveforms_concat.size(0) > 0:
                # Check dimensions of the audio waveform to ensure consistency
                if (
                    audio_waveforms_concat_list
                    and sample.audio_waveforms_concat.dim() != audio_waveforms_concat_list[0].dim()
                ):
                    logger.warning(
                        f"Skipping audio waveform with inconsistent dimensions: expected {audio_waveforms_concat_list[0].dim()}D, got {sample.audio_waveforms_concat.dim()}D"
                    )
                    continue

                audio_waveforms_concat_list.append(sample.audio_waveforms_concat)
                audio_waveforms_start_list.append(sample.audio_waveforms_start + audio_waveforms_offset)
                audio_waveforms_offset += sample.audio_waveforms_concat.size(0)

                # Add audio_sample_rate and audio_speaker_indices
                audio_sample_rate_list.append(sample.audio_sample_rate)

            audio_speaker_indices_list.append(sample.audio_speaker_indices)

        # Concatenate all tensors
        input_ids = torch.cat(input_ids_list, dim=0)
        label_ids = torch.cat(label_ids_list, dim=0)

        # Apply padding if padding_size is specified
        if padding_size is not None and padding_size > 0:
            input_ids = torch.cat([input_ids, torch.full((padding_size,), eos_token_id, dtype=torch.long)], dim=0)
            label_ids = torch.cat([label_ids, torch.full((padding_size,), ignore_index, dtype=torch.long)], dim=0)

        # Safely concatenate audio tensors with proper error handling
        try:
            audio_ids_concat = torch.cat(audio_ids_concat_list, dim=1) if audio_ids_concat_list else torch.tensor([[]])
            audio_ids_start = torch.cat(audio_ids_start_list, dim=0) if audio_ids_start_list else torch.tensor([])

            # Check for dimensional consistency in audio waveforms
            if audio_waveforms_concat_list:
                dims = [t.dim() for t in audio_waveforms_concat_list]
                if not all(d == dims[0] for d in dims):
                    # If dimensions don't match, log warning and filter out the problematic tensors
                    logger.warning(
                        f"Inconsistent dimensions in audio waveforms: {dims}. Filtering to keep only consistent ones."
                    )
                    expected_dim = max(set(dims), key=dims.count)  # Most common dimension
                    audio_waveforms_concat_list = [t for t in audio_waveforms_concat_list if t.dim() == expected_dim]

                    # Recalculate audio_waveforms_start with the filtered list
                    if audio_waveforms_concat_list:
                        audio_waveforms_offset = 0
                        audio_waveforms_start_list = []
                        for waveform in audio_waveforms_concat_list:
                            audio_waveforms_start_list.append(torch.tensor([audio_waveforms_offset]))
                            audio_waveforms_offset += waveform.size(0)

            audio_waveforms_concat = (
                torch.cat(audio_waveforms_concat_list, dim=0) if audio_waveforms_concat_list else torch.tensor([])
            )
            audio_waveforms_start = (
                torch.cat(audio_waveforms_start_list, dim=0) if audio_waveforms_start_list else torch.tensor([])
            )
            audio_sample_rate = (
                torch.cat(audio_sample_rate_list, dim=0) if audio_sample_rate_list else torch.tensor([])
            )
            audio_speaker_indices = (
                torch.cat(audio_speaker_indices_list, dim=0) if audio_speaker_indices_list else torch.tensor([])
            )

        except RuntimeError as e:
            logger.error(f"Error during tensor concatenation: {str(e)}")
            logger.warning("Falling back to empty audio tensors")
            # Fall back to empty tensors
            audio_ids_concat = torch.tensor([[]])
            audio_ids_start = torch.tensor([])
            audio_waveforms_concat = torch.tensor([])
            audio_waveforms_start = torch.tensor([])
            audio_sample_rate = torch.tensor([])
            audio_speaker_indices = torch.tensor([])

        # Create the merged sample
        merged_sample = cls(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices,
        )

        return merged_sample

