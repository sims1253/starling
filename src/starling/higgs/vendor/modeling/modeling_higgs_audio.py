"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""
### need transformers==4.51.0
import torch
import torch.nn as nn
import math
import glob
import functools
import copy
import os
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from safetensors.torch import load_file, save_file
from typing import Optional, Tuple, Union, List, Dict, Any

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding, Qwen3Config
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache, StaticCache, EncoderDecoderCache
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerateDecoderOnlyOutput
from transformers.utils import logging, ModelOutput
from transformers.integrations import is_deepspeed_available
from transformers.modeling_flash_attention_utils import _flash_attention_forward

DistributedAttention = None
vocab_sequence_parallel_cross_entropy = None
if is_deepspeed_available():
    import importlib
    _ds_layer = importlib.import_module("deepspeed.sequence.layer")
    _ds_ce = importlib.import_module("deepspeed.sequence.cross_entropy")
    DistributedAttention = _ds_layer.DistributedAttention
    vocab_sequence_parallel_cross_entropy = _ds_ce.vocab_sequence_parallel_cross_entropy

from .attention import HiggsDistributedWhisperFlashAttention2
from transformers.modeling_utils import PreTrainedModel
from .utils import (
    merge_input_ids_with_audio_features,
    sequence_chunking_per_rank,
    count_parameters,
    deepspeed_ulysses_attention,
    support_deepspeed_ulysses,
    is_deepspeed_ulysses_enabled,
    all_gather_tensors,
    deepspeed_ulysses_rope,
    drop_tokens,
    gather_tokens,
)
from .common import HiggsAudioPreTrainedModel
from .configuration_higgs_audio import HiggsAudio3Config, HiggsAudioEncoderConfig
from .custom_modules import PartiallyFrozenLinear, PartiallyFrozenEmbedding
from .cuda_graph_runner import CUDAGraphRunner
try:
    from .modeling_higgs_audio_xcodec import HiggsAudioEncoderXcodec
except ImportError:
    HiggsAudioEncoderXcodec = None

logger = logging.get_logger(__name__)


class GenerationMode(Enum):
    """Enum for different generation modes in HiggsAudio model."""
    TEXT = 0                    # Text generation mode
    AUDIO_INIT = 1             # Audio generation mode initialization
    AUDIO_IN_PROGRESS = 2      # Audio generation mode in progress


class HiggsAudioFeatureProjector(nn.Module):
    """
    Projector that maps audio features extracted by Whisper to hidden state of the text model. Two selectable implementations:
      - 'linear' (backward-compatible, old behavior)
      - 'mlp'    (new: optional temporal downsample + 2-layer MLP + activation)
    """
    def __init__(self, config: HiggsAudio3Config):
        super().__init__()
        audio_dim = config.audio_encoder_config.d_model
        llm_hidden_dim = config.text_config.hidden_size
        self.stride = int(getattr(config, "projector_temporal_downsample", 1))
        self.projector_type = getattr(config, "projector_type", "linear")
        if self.projector_type == "linear":
            assert self.stride == 1, "Temporal downsample is not supported for linear projector."
            self.linear = nn.Linear(audio_dim, llm_hidden_dim, bias=True)
        else: 
            if self.stride == 1: 
                self.temporal = nn.Identity()
            elif self.stride == 2:
                self.temporal = nn.Conv1d(audio_dim, audio_dim, 3, 2, padding=1, groups=audio_dim, bias=True)
            elif self.stride == 4:
                self.temporal = nn.Sequential(
                    nn.Conv1d(audio_dim, audio_dim, 3, 2, padding=1, groups=audio_dim, bias=True),
                    nn.Conv1d(audio_dim, audio_dim, 3, 2, padding=1, groups=audio_dim, bias=True),
                )
            else:
                raise ValueError(f"Unsupported stride: {self.stride}")
            # fix to 2 layer mlp with 2048 hidden and ReLU as https://huggingface.co/stepfun-ai/Step-Audio-2-mini/blob/main/modeling_step_audio_2.py#L266
            hidden = 2048
            self.linear1 = nn.Linear(audio_dim, hidden, bias=True)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden, llm_hidden_dim, bias=True)

    def forward(self, audio_features):
        # Input: (B, T, audio_dim)
        # Output: (B, T', llm_hidden_dim), where T' depends on the downsample stride get from conv
        x = audio_features  # (B, T, C_in)
        if self.projector_type == "linear":
            return self.linear(x)
        else: 
            if self.stride > 1:
                x = x.permute(0, 2, 1)
                x = self.temporal(x)  # apply on the time dimension
                x = x.permute(0, 2, 1)
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)

    def downsample_lengths(self, lengths):
        # lengths: (B,)
        if self.projector_type == "linear" or self.stride == 1:
            return lengths
        if self.stride in (2, 4):
            # because the temporal is built from k=3,p=1,stride=2 blocks, effective behavior is ceil_div by stride
            return (lengths - 1) // self.stride + 1
        raise ValueError(f"Unsupported stride: {self.stride}")

@dataclass
class HiggsAudioDecoderLayerOutput:
    logits: torch.FloatTensor
    audio_logits: torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@support_deepspeed_ulysses
class HiggsAudioDecoderProjector(HiggsAudioPreTrainedModel):
    """Projection layers that map hidden states from the LLM component to audio / text logits.
    Removed RQ Transformer for Qwen3 for now. 
    Check this https://github.com/boson-ai/boson-multimodal/blob/5ae26565a5f750716a47ccdf35ea2cc7270b2a45/boson_multimodal/model/higgs_audio/modeling_higgs_audio.py for the implementation before RQ Transformer.
    """

    def __init__(self, config: HiggsAudio3Config, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.text_lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(
            config.text_config.hidden_size, config.audio_num_codebooks * (config.audio_codebook_size + 2), bias=False
        )
        self.gradient_checkpointing = False
        # FIXME: audio_decoder_proj_num_layers is never used in M2 model; we always have dec=0... 
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        hidden_states,
        audio_out_mask,
        label_audio_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_audio_hidden_states=False,
        cache_position=None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            label_audio_ids (`torch.Tensor` of shape `(num_codebooks, num_audio_out_tokens)`):
                Label tokens for the audio-out part. This is used for calculating the logits if RQ-Transformer is used.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask to avoid performing attention on padding token indices
            position_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Position ids for the input tokens

        Returns:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure `num_text_tokens + num_audio_tokens == batch_size * seq_len`
        """
        logits = self.text_lm_head(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        next_cache = next_decoder_cache if use_cache else None
        if is_deepspeed_ulysses_enabled():
            audio_out_mask = sequence_chunking_per_rank(
                getattr(self, "sp_size", 1),
                getattr(self, "sp_rank", 0),
                audio_out_mask,
                dim=1,
            )
        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])
        if output_audio_hidden_states:
            audio_hidden_states = hidden_states[audio_out_mask]
        else:
            audio_hidden_states = None
        return logits, audio_logits, all_self_attns, all_hidden_states, audio_hidden_states, next_cache


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """Placeholder, not used in HiggsAudio3."""

    def __init__(
        self, config: HiggsAudio3Config, layer_idx: int, fast_forward: bool = False, use_audio_attention: bool = False
    ):
        super().__init__()
        pass


# Revised on top of transformers.models.qwen2_audio.modeling_qwen2_audio with Qwen2AudioEncoder --> HiggsAudioEncoder
# The code was originally borrowed from WhisperEncoder
@support_deepspeed_ulysses
class HiggsAudioEncoder(HiggsAudioPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        # Flash Attention 2 does not support zero shape tensor, so we have to use sdpa implementation for the Whisper component.
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.gradient_checkpointing = False
        self.disable_sp_for_audio = False   # FIXME (DongmingShenDS) there's a dim-related bug in the audio tower when using SP we fix later
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        check_seq_length=True,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                HiggsAudio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # FIXME(DongmingShenDS) - disable for now; we are exploring streaming whisper and this is no longer used
        # expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        # if check_seq_length and (input_features.shape[-1] != expected_seq_length):
        #     raise ValueError(
        #         f"HiggsAudio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
        #     )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight
        # when using streaming whisper, the we need to chunk the embed_pos to match the inputs_embeds because the inputs_embeds is of variable length
        T = inputs_embeds.size(1)
        assert T <= self.config.max_source_positions, f"HiggsAudio expects the mel input features to be of length {self.config.max_source_positions}, but found {T}. Make sure to pad the input mel features to {self.config.max_source_positions}."
        embed_pos = self.embed_positions.weight[:T, :].unsqueeze(0)  # (1, T, D)
        if is_deepspeed_ulysses_enabled() and not self.disable_sp_for_audio:
            # The grad_output at this point needs to be scaled down by sp_size because the gradients will be duplicated among the SP ranks.
            inputs_embeds = drop_tokens(
                inputs_embeds,
                dim=1,
                group=getattr(self, "sp_group", None),
                grad_scale=getattr(self, "sp_size", 1),
            )
            embed_pos = sequence_chunking_per_rank(
                getattr(self, "sp_size", 1),
                getattr(self, "sp_rank", 0),
                embed_pos,
                dim=1,
            )

        hidden_states = inputs_embeds + embed_pos  # (B, T, D)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                # transformers >=5.0: ``WhisperEncoderLayer.forward`` returns a bare
                # ``hidden_states`` tensor instead of the historical ``(hidden_states,
                # attn_weights)`` tuple, and it always discards the attention weights
                # (``self_attn`` is unpacked as ``hidden_states, _ = ...``). The code
                # below indexes ``layer_outputs[0]`` / ``layer_outputs[1]``, so on tf5
                # ``[0]`` would wrongly slice the batch axis off the tensor. Normalize
                # to a tuple here so the historical unpacking keeps working on both
                # tf4.51 (already a tuple) and tf5.x (bare tensor). Attention weights
                # are no longer surfaced by the stock layer under tf5.x, so pad the
                # attentions slot with ``None`` to preserve the legacy tuple shape.
                if torch.is_tensor(layer_outputs):
                    layer_outputs = (layer_outputs, None)

                hidden_states = layer_outputs[0]

            if output_attentions:
                attn = layer_outputs[1] if len(layer_outputs) > 1 else None
                all_attentions = all_attentions + ((attn,) if attn is not None else ())

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        # If the sequence length after average pooling is not divisible by the sequence parallel size, we would duplicate it across the sequence parallel ranks.
        # In this case, gradients need to be scaled up because the subsequent scaling up in the function _apply_audio_tower is skipped.
        if is_deepspeed_ulysses_enabled() and not self.disable_sp_for_audio:
            hidden_states = gather_tokens(
                hidden_states,
                dim=2,
                group=getattr(self, "sp_group", None),
                grad_scale=1 if ((hidden_states.size(2) - 2) // 2 + 1) % getattr(self, "sp_size", 1) == 0 else getattr(self, "sp_size", 1),
            )
        hidden_states = self.avg_pooler(hidden_states)
        if is_deepspeed_ulysses_enabled() and not self.disable_sp_for_audio:
            if hidden_states.size(2) % getattr(self, "sp_size", 1) == 0:
                hidden_states = drop_tokens(
                    hidden_states,
                    dim=2,
                    group=getattr(self, "sp_group", None),
                    grad_scale=1,
                )
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # TODO(sxjscience) Double confirm the formula
        input_lengths = (input_lengths - 1) // 2 + 1  # post-conv before whisper layers - used for attention mask
        output_lengths = (input_lengths - 2) // 2 + 1  # post-avg-pool
        return input_lengths, output_lengths


@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    codebook_losses: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    audio_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HiggsAudioGenerationOutput(ModelOutput):
    """
    Outputs of HiggsAudio generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        audio_sequences (`tuple(torch.LongTensor)` *optional*):
            The generated discrete audio codes. These codes can be used to fill-in related locations of <|AUDIO_OUT|> at input sequences.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor = None
    audio_sequences: Optional[List[torch.LongTensor]] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


_AUDIO_TOWER_REGISTRY = {
    "higgs_audio_encoder": HiggsAudioEncoder,              # Whisper-backed
}
if HiggsAudioEncoderXcodec is not None:
    _AUDIO_TOWER_REGISTRY["higgs_audio_encoder_xcodec"] = HiggsAudioEncoderXcodec

def _build_audio_tower(audio_encoder_config):
    model_type = getattr(audio_encoder_config, "model_type", None)
    if model_type not in _AUDIO_TOWER_REGISTRY:
        raise ValueError(f"Unsupported audio encoder model_type: {model_type}")
    return _AUDIO_TOWER_REGISTRY[model_type](audio_encoder_config)


@support_deepspeed_ulysses
class HiggsAudio3Model(HiggsAudioPreTrainedModel, GenerationMixin):
    """Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio.

    Consider the following example for mixed text/audio understanding / generation:

    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_bos|>[AUDIO]<|audio_eos|><text_token4>
    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_out_bos|>[AUDIO_OUT]<|audio_eos|><text_token4>

    We will fill [AUDIO] with the audio features extracted by Whisper and fill [AUDIO_OUT] with the audio tokens.

    Consider the following example for mixed text/audio generation:

    text: <|audio_out_bos|>    MASK           MASK           MASK          MASK               MASK         <|audio_eos|> [text_token1]
    audio:     MASK    <|audio_stream_bos|> [audio_token1] [audio_token2] [audio_token3] <|audio_stream_eos|>   MASK           MASK
    token_type: 0               1              1              1             1                  1                 0              0
    """

    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: HiggsAudio3Config):
        # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py
        super().__init__(config)
        # handle attention implementation
        attn_impl = getattr(config, "_attn_implementation", None) or getattr(config.text_config, "_attn_implementation", None)
        if hasattr(config, "audio_encoder_config") and config.audio_encoder_config is not None:
            setattr(config.audio_encoder_config, "_attn_implementation", attn_impl)
        if hasattr(config, "text_config") and config.text_config is not None:
            setattr(config.text_config, "_attn_implementation", attn_impl)
        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = getattr(config, "audio_out_bos_token_id", None)
        self.audio_eos_token_id = getattr(config, "audio_eos_token_id", None)
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention
        self.projector_temporal_downsample = config.projector_temporal_downsample
        self.embed_tokens = nn.Embedding(self.vocab_size, config.text_config.hidden_size, self.padding_idx)
        assert config.audio_adapter_type == "stack", f"Audio adapter type {config.audio_adapter_type} not implemented."
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config.text_config, layer_idx)
            for layer_idx in range(config.text_config.num_hidden_layers)
        ])
        layer_idx = config.text_config.num_hidden_layers
        self.num_activation_checkpointing_layers = len(self.layers)
        self.decode_graph_runners = defaultdict(dict[bool, CUDAGraphRunner])
        self.norm = Qwen3RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config.text_config)
        self.has_sliding_layers = "sliding_attention" in getattr(config.text_config, "layer_types", [])  # never used but keep for signature
        if not config.skip_audio_tower:
            self.audio_tower = _build_audio_tower(config.audio_encoder_config)
            self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
            self.encoder_backend = "xcodec" if isinstance(self.audio_tower, HiggsAudioEncoderXcodec) else "whisper"  
            self.sample_rate = int(getattr(self.audio_tower.config, "sample_rate", 24000)) if self.encoder_backend == "xcodec" else 16000
            self.hop_length = int(getattr(self.audio_tower.config, "hop_length", 960 * 2)) if self.encoder_backend == "xcodec" else -1
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config, layer_idx=layer_idx)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token
        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=False
            )
        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )
        self.gradient_checkpointing = False
        self.audio_codebook_weights = torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks # default to equal weights
        print(f"Model TPS: {config.tps}", flush=True)
        self.post_init()

    def set_num_activation_checkpointing_layers(self, num_layers):
        self.num_activation_checkpointing_layers = num_layers

    def set_audio_special_tokens(self, tokenizer: AutoTokenizer):
        self.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids into hidden states using the audio codebook embeddings

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_ids.device) * self.audio_codebook_size
        )
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift.unsqueeze(-1))
        if self.config.audio_embed_avg:
            audio_embed = torch.mean(audio_embed, dim=0)
        else:
            audio_embed = torch.sum(audio_embed, dim=0)
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed
    
    def _apply_audio_tower_whisper(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features"""
        # FIXME (DongmingShenDS) need to check if this leads to any issues
        if audio_features is None or audio_features.shape[0] == 0:
            return None, None

        audio_feat_lengths, audio_feat_out_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = audio_features.shape
        # Post-conv (pre-avgpool) sequence length: the two Whisper conv1d layers
        # halve 3000 mel frames -> 1500. The vendored HiggsAudioEncoder runs its
        # 32 WhisperEncoderLayers at THIS resolution, then avg-pools 2x at the end.
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1  # 1500
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range < lengths_expand  # (B, 1500) True for valid keys
        # transformers 5.x: tf4.51 accepted a bool mask and converted it inside the
        # whisper attention; tf5's ``eager_attention_forward`` does a raw
        # ``attn_weights + attention_mask`` add, so we must pass an ADDITIVE float
        # mask (0.0 for valid keys, -inf for padded keys) shaped (B,1,q,k) for
        # broadcasting. This keeps padded mel frames from corrupting the valid
        # span via self-attention -- byte-exact with the reference encoder.
        min_dtype = torch.finfo(self.audio_tower.conv1.weight.dtype).min
        if self.config._attn_implementation != "flash_attention_2":
            add_mask = torch.where(
                padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len),
                0.0,
                min_dtype,
            ).to(self.audio_tower.conv1.weight.dtype)
        else:
            add_mask = padding_mask  # flash path takes bool
        audio_outputs = self.audio_tower(audio_features, attention_mask=add_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)

        if is_deepspeed_ulysses_enabled():
            # The grad_output at this point needs to be scaled up by sp_size to cancel out the scaling down in the subsequent drop_tokens.
            if audio_features_embed.size(1) * getattr(self, "sp_size", 1) == (max_seq_len - 2) // 2 + 1:
                audio_features_embed = gather_tokens(
                    audio_features_embed,
                    dim=1,
                    group=getattr(self, "sp_group", None),
                    grad_scale=getattr(self, "sp_size", 1),
                )
        
        # adjust final lengths by projector temporal stride in audio feature projector (downsample might happen in the projector layers)
        audio_feat_out_lengths = self.audio_encoder_proj.downsample_lengths(audio_feat_out_lengths)
        audio_feat_out_lengths = audio_feat_out_lengths.clamp_max(audio_features_embed.size(1))  # safety, might be redundant
        return audio_features_embed, audio_feat_out_lengths

    def _apply_audio_tower_xcodec(self, audio_features, audio_wv_lengths):
        """Apply the audio tower to the audio features with the xcodec encoder backend"""
        # Apply the XCodec audio tower:
        #   - input: raw padded waveforms (24 kHz bfloat16) and true wv lengths (pre-padding) in samples in `audio_wv_lengths`
        #   - output: projected embeddings (N, T_frames, hidden) and per-item frame lengths (N,)
        # Lengths are computed as ceil_div(true_samples, hop_length), then clamped to T_frames.
        # 0) Empty-batch: touch parameters to keep ZeRO/grad graphs happy, then return empty views.
        if audio_features.shape[0] == 0:
            return None, None
        # 1) Run XCodec encoder on raw waveforms at 24 kHz and project to LLM dim
        if self.gradient_checkpointing and self.training:
            def _tower_call(x):
                return self.audio_tower(input_values=x, sampling_rate=self.sample_rate, return_dict=True).last_hidden_state
            e = self._gradient_checkpointing_func(_tower_call, audio_features)
        else:
            out = self.audio_tower(input_values=audio_features, sampling_rate=self.sample_rate, return_dict=True)
            e = out.last_hidden_state  # (N, T_frames, Q); where T_frames = ceil_div(true_samples, hop_length) and Q = d_model in xcodec
        audio_features_embed = self.audio_encoder_proj(e)  # (N, T_frames_after_pool, hidden)
        # 2) Compute true lengths in frames (aka embedding frames) from raw waveform lengths
        audio_feat_out_lengths = ((audio_wv_lengths + (self.hop_length - 1)) // self.hop_length)  # this assumes we pad to nearest hop_length
        if self.projector_temporal_downsample > 1: 
            audio_feat_out_lengths = (audio_feat_out_lengths - 1) // self.projector_temporal_downsample + 1
        audio_feat_out_lengths = audio_feat_out_lengths.clamp_max(audio_features_embed.size(1))  # safety, might be redundant
        return audio_features_embed, audio_feat_out_lengths

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=False,  # no sliding window for now
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config.text_config,     # note: use text_config
            past_key_values=past_key_values,
        )
        if (
            self.config.text_config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class - NOT USED???
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask

    def _prepare_all_static_kv_cache_masks(self, hidden_states, attention_mask, audio_out_mask, past_key_values):
        target_length = hidden_states.shape[1]
        cur_pos = audio_out_mask.shape[1]
        min_dtype = torch.finfo(hidden_states.dtype).min
        assert len(attention_mask.shape) == 4, "Only support SDPA for now"
        kv_cache_len = past_key_values.get_max_cache_shape()
        audio_out_mask_padded = torch.nn.functional.pad(audio_out_mask, (0, kv_cache_len - cur_pos), value=True)
        fast_forward_attention_mask = attention_mask.masked_fill(
            audio_out_mask_padded[:, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]].reshape(
                audio_out_mask_padded.shape[0], 1, target_length, 1
            )
            | audio_out_mask_padded.reshape(audio_out_mask_padded.shape[0], 1, 1, audio_out_mask_padded.shape[1]),
            min_dtype,
        )
        no_audio_out_mask = ~audio_out_mask
        no_audio_out_mask = torch.nn.functional.pad(
            no_audio_out_mask, (0, kv_cache_len - audio_out_mask.shape[1]), value=False
        )
        no_audio_out_mask = no_audio_out_mask[
            :, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]
        ].reshape(audio_out_mask.shape[0], 1, target_length, 1) | no_audio_out_mask.reshape(
            audio_out_mask.shape[0], 1, 1, kv_cache_len
        )
        audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
        return fast_forward_attention_mask, audio_attention_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]],
        use_cache: bool,
        audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        is_decoding_audio_token: Optional[bool] = None,
        is_using_cuda_graph: Optional[bool] = False,
    ):
        # https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py#L498
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if self.gradient_checkpointing and self.training and layer_idx < self.num_activation_checkpointing_layers:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            # transformers >=5.0: ``Qwen3DecoderLayer.forward`` returns a bare
            # ``hidden_states`` tensor instead of the historical
            # ``(hidden_states, attn_weights)`` tuple, and no longer surfaces
            # attention weights. ``layer_outputs[0]`` below would otherwise slice
            # the batch axis off the tensor (collapsing it to 2D, which then
            # mis-reshapes the next layer's attention). Normalize to a tuple so
            # the legacy unpacking works on both tf4.51 (tuple) and tf5.x (bare
            # tensor); pad the attentions slot with ``None``. (Same fix pattern
            # as the Whisper encoder layers above.)
            if torch.is_tensor(layer_outputs):
                layer_outputs = (layer_outputs, None)

            hidden_states = layer_outputs[0]

            if output_attentions:
                attn = layer_outputs[1] if len(layer_outputs) > 1 else None
                all_self_attns += ((attn,) if attn is not None else ())

        return hidden_states, all_hidden_states, all_self_attns

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids_start_group_loc: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_audio_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        reward: Optional[torch.FloatTensor] = None,
        group_id: Optional[torch.LongTensor] = None,
        audio_wv_lengths: Optional[torch.LongTensor] = None  # (num_audio_in,) - for xcodec
    ):
        """Forward pass for the Higgs-Audio model.

        Args:
            input_ids (:obj:`torch.LongTensor`):
                The input ids of the prompt. It will have shape (bsz, seq_len).
                When use_cache is enabled, the input_ids will have
                shape (bsz, 1) for incremental decode or None
            inputs_embeds:
                Input embeddings. This flag won't be used.
            attention_mask (:obj:`torch.LongTensor`):
                The attention mask of the prompt. It will have shape (bsz, seq_len).
            audio_features (:obj:`torch.FloatTensor`):
                The audio features extracted by Whisper. It will have shape (num_audio_in, feature_dim, max_mel_seq_len).
            audio_feature_attention_mask (:obj:`torch.LongTensor`):
                The attention mask of the audio features. It will have shape (num_audio_in, max_mel_seq_len).
            audio_in_ids (:obj:`torch.LongTensor`):
                The discretized audio tokens. It will have shape (num_codebooks, audio_in_total_length).
            audio_in_ids_start (:obj:`torch.LongTensor`):
                The start indices for each audio in audio_in_ids. It will have shape (num_audio_in,)
            audio_out_ids (:obj:`torch.LongTensor`):
                The discretized audio tokens. It will have shape (num_codebooks, audio_out_total_length).
            audio_out_ids_start (:obj:`torch.LongTensor`):
                The start indices for each audio in audio_out_ids. It will have shape (num_audio_out,)
            audio_out_ids_start_group_loc (:obj:`torch.LongTensor`):
                The sample indices in a batch that map to each element in the audio_out_ids_start. It will have shape (num_audio_out,)
            label_text_ids (:obj:`torch.LongTensor`):
                The labels of the prompt. It will have shape (bsz, seq_len).
            label_audio_ids (:obj:`torch.LongTensor`):
                The labels of the audio tokens. It will have the same shape as audio_out_ids, i.e., (num_codebooks, audio_out_total_length)
            past_key_values (:obj:`Tuple`):
                Tuple of past key values.
            use_cache (:obj:`bool`):
                Whether to use cache.
            output_attentions (:obj:`bool`):
                Whether to output attentions.
            output_hidden_states (:obj:`bool`):
                Whether to output hidden states.
            output_audio_hidden_states (:obj:`bool`):
                Whether to output audio hidden states. This will be used when running RQ-Transformer decoding in the inference mode.
            return_dict (:obj:`bool`):
                Whether to return a dictionary.
            cache_position (:obj:`torch.LongTensor`):
                The position of the cache.
            cache_audio_discrete_codes_mask (:obj:`torch.LongTensor`):
                The cached audio discrete codes mask. It will only be used when use_cache is turned on.
            past_key_values_buckets (:obj:`OrderedDict`):
                The buckets of past key values.
            reward (:obj:`torch.FloatTensor`):
                The reward for DPO training.
            audio_wv_lengths (:obj:`torch.LongTensor`):
                The lengths of the audio waveforms in the batch.
        """
        target_device = input_ids.device

        # not used
        del inputs_embeds

        if audio_features is not None:
            audio_features = audio_features.to(target_device, dtype=torch.bfloat16)  # otherwise see errors
        if audio_feature_attention_mask is not None:
            audio_feature_attention_mask = audio_feature_attention_mask.to(target_device)
        if audio_wv_lengths is not None:
            audio_wv_lengths = audio_wv_lengths.to(target_device)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # 1. Extract the input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # 2. Extract audio embeddings
        if self.config.skip_audio_tower:
            audio_features_embed = audio_features_length = None
        else:
            if self.encoder_backend == "whisper":
                audio_features_embed, audio_features_length = self._apply_audio_tower_whisper(
                    audio_features, audio_feature_attention_mask
                )
            elif self.encoder_backend == "xcodec":
                # raw 24k wav in `audio_features`, real wv lengths (pre-padding 24khz) in samples in `audio_wv_lengths`
                audio_features_embed, audio_features_length = self._apply_audio_tower_xcodec(
                    audio_features, audio_wv_lengths
                )
            else:
                raise ValueError(f"Invalid encoder backend: {self.encoder_backend}")

        if self.config.encode_audio_in_tokens:
            if audio_in_ids is not None and audio_in_ids.shape[-1] > 0:
                audio_in_ids = audio_in_ids.to(target_device)
            else:
                audio_in_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
            audio_in_embed = self._embed_audio_ids(audio_in_ids)
        else:
            audio_in_embed = None

        if audio_out_ids is not None and audio_out_ids.shape[-1] > 0:
            audio_out_ids = audio_out_ids.to(target_device)
        else:
            audio_out_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
        audio_out_embed = self._embed_audio_ids(audio_out_ids)

        # 3. Merge text, audio-in embeddings, and audio-out embeddings
        # use_cache is turned on during inference time, we should set round_to to 1 to avoid extra padding in the end.
        round_to = 1 if use_cache else 8
        left_padding = True if use_cache or input_ids.shape[0] == 1 else False
        (
            inputs_embeds,
            attention_mask,
            labels,
            position_ids,
            input_ids,
            audio_in_mask,
            audio_in_discrete_codes_mask,
            audio_out_mask,
        ) = merge_input_ids_with_audio_features(
            audio_features_embed,
            audio_features_length,
            audio_in_embed,
            audio_in_ids_start,
            audio_out_embed,
            audio_out_ids_start,
            self.audio_in_token_idx,
            self.audio_out_token_idx,
            inputs_embeds,
            input_ids,
            attention_mask,
            label_ids,
            pad_token_id=self.padding_idx,
            round_to=round_to,
            left_padding=left_padding,
        )

        if is_deepspeed_ulysses_enabled():
            # The grad_output at this point needs to be scaled down by sp_size because the gradients will be duplicated among the SP ranks.
            inputs_embeds = drop_tokens(
                inputs_embeds,
                dim=1,
                group=getattr(self, "sp_group", None),
                grad_scale=getattr(self, "sp_size", 1),
            )

        # re-check if we use the correct kv cache bucket after
        # the input_embeds has been merged with audio features
        if past_key_values_buckets is not None and \
            inputs_embeds.shape[1] > past_key_values.get_max_cache_shape():
            past_key_values, self.current_past_key_values_bucket = \
                self._prepare_kv_cache(inputs_embeds.shape[1], None, past_key_values_buckets)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            if isinstance(past_key_values, StaticCache) and past_seen_tokens >= past_key_values.get_max_cache_shape():
                raise ValueError(f"The current sequence length ({past_seen_tokens}) exceeds "
                                 f"the maximum cache shape. "
                                 f"Please consider increasing the cache size.")
            # FIXME!!!!! in original qwen code there's a line "if position_ids is None:", if added here, will trigger bug in generate when using cache - need to inspect
            position_ids = cache_position.unsqueeze(0)

        # Use torch compile
        use_static_cache = isinstance(past_key_values, StaticCache)

        # Apply the LLM component
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        audio_discrete_codes_mask = audio_in_discrete_codes_mask | audio_out_mask
        if cache_audio_discrete_codes_mask is not None and use_cache:
            audio_discrete_codes_mask = torch.concat(
                [cache_audio_discrete_codes_mask, audio_discrete_codes_mask], dim=1
            )

        # Generate the audio attention mask outside the layer to avoid recompilation
        if use_static_cache:
            fast_forward_attention_mask, audio_attention_mask = self._prepare_all_static_kv_cache_masks(
                hidden_states, causal_mask, audio_discrete_codes_mask, past_key_values
            )
            # Set the audio out mask to the last token
            if hidden_states.shape[1] == 1:
                audio_discrete_codes_mask = audio_discrete_codes_mask[:, -1:]
                audio_discrete_codes_mask = audio_discrete_codes_mask.reshape((-1, 1)).contiguous()
                is_decoding_audio_token = audio_discrete_codes_mask.item()
            else:
                is_decoding_audio_token = False

        # Use the captured cuda graph runner for decoding
        # if it exists, otherwise use the normal forward pass
        if (
            past_key_values is not None
            and past_key_values.get_max_cache_shape() in self.decode_graph_runners
            and (input_ids.shape[-1] == 1)
        ):
            _forward_core = self.decode_graph_runners[past_key_values.get_max_cache_shape()][is_decoding_audio_token]
            is_using_cuda_graph = True
        else:
            _forward_core = self._forward_core
            is_using_cuda_graph = False

        hidden_states, all_hidden_states, all_self_attns = _forward_core(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            audio_discrete_codes_mask=audio_discrete_codes_mask,
            is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            audio_attention_mask=audio_attention_mask if use_static_cache else None,
            fast_forward_attention_mask=fast_forward_attention_mask if use_static_cache else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            is_using_cuda_graph=is_using_cuda_graph,
        )
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Apply the audio decoder projector
        logits, audio_logits, decoder_all_self_attns, decoder_all_hidden_states, audio_hidden_states, _ = self.audio_decoder_proj(
            hidden_states,
            audio_out_mask,
            label_audio_ids=label_audio_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_audio_hidden_states=output_audio_hidden_states,
            cache_position=cache_position,
        )

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.audio_num_codebooks, self.audio_codebook_size
            ).float()

        if output_hidden_states:
            if decoder_all_hidden_states is not None and len(decoder_all_hidden_states) > 1:
                all_hidden_states += decoder_all_hidden_states[1:]

        if output_attentions:
            all_self_attns += decoder_all_self_attns
        # Loss calculation when label_ids is not None
        loss = None
        llm_loss = None
        audio_loss = None
        codebook_losses = None

        # Calculate the loss function
        # There will be two loss functions, one for the text-stream and one for the audio stream.
        if label_ids is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            if is_deepspeed_ulysses_enabled():
                sp_seqlen = logits.size(1)
                shift_logits = logits.contiguous()

                if self.sp_rank == self.sp_size - 1:
                    # add an ignore_index to the end of the labels
                    shift_labels = torch.cat(
                        (labels[..., -(sp_seqlen - 1) :], torch.full_like(labels[:, :1], -100)), dim=-1
                    ).contiguous()
                else:
                    shift_labels = labels[
                        ..., (sp_seqlen * self.sp_rank) + 1 : (sp_seqlen * (self.sp_rank + 1)) + 1
                    ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            if (shift_labels == -100).all() and not is_deepspeed_ulysses_enabled():
                loss = shift_logits.sum() * 0.0  # Connect the gradient
            else:
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                if is_deepspeed_ulysses_enabled():
                    loss = vocab_sequence_parallel_cross_entropy(
                        shift_logits.unsqueeze(1), shift_labels.unsqueeze(1), getattr(self, "sp_group", None)
                    ).squeeze(1)
                    loss = loss[torch.nonzero(loss)].mean()
                else:
                    loss = nn.functional.cross_entropy(shift_logits, shift_labels)

            # # FIXME! This is a hack to make ZeRO work for heterogenous computation.
            if audio_features_embed is not None and audio_features_embed.shape[0] == 0:
                loss += torch.sum(audio_features_embed, dtype=torch.float32) * 0

            if audio_in_embed is not None and audio_in_embed.shape[0] == 0:
                loss += torch.sum(audio_in_embed, dtype=torch.float32) * 0
        else:
            loss = torch.tensor(0, dtype=torch.float32, device=target_device)

        if label_audio_ids is not None and label_audio_ids.shape[-1] > 0:
            # audio_logits have shape (num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)
            if is_deepspeed_ulysses_enabled():
                audio_out_size = None
                if audio_out_mask.size(0) > 1:
                    audio_out_size = []
                    for this_rank in range(getattr(self, "sp_size", 1)):
                        this_audio_out_mask = sequence_chunking_per_rank(
                            getattr(self, "sp_size", 1),
                            this_rank,
                            audio_out_mask,
                            dim=1,
                        )
                        this_audio_out_size = this_audio_out_mask.sum(-1)
                        audio_out_size.append(this_audio_out_size)
                audio_logits = all_gather_tensors(
                    audio_logits, size=audio_out_size, dim=0, group=getattr(self, "sp_group", None)
                )
            audio_shift_logits = audio_logits[:-1, :, :].contiguous()

            # Ignore the first label token for each audio for proper auto-regressive training.
            # input:                    a1, a2, a3,   b1, b2, b3, b4, c1, d1
            # label (masked):           a1, a2, a3, -100, b2, b3, b4, c1, -100
            # label (shifted):          a2, a3, -100, b2, b3, b4, c1, -100
            # label_audio_ids have shape (num_codebooks, num_audio_out_tokens)
            label_audio_ids[:, audio_out_ids_start] = -100

            audio_shift_labels = label_audio_ids[:, 1:]

            audio_loss_fct = CrossEntropyLoss()
            codebook_losses = torch.zeros([self.audio_num_codebooks], device=target_device)
            for codebook in range(self.audio_num_codebooks):
                codebook_logits = audio_shift_logits[:, codebook, :].contiguous().view(-1, self.audio_codebook_size)
                codebook_labels = audio_shift_labels[codebook, :].contiguous().to(codebook_logits.device)
                if (codebook_labels == -100).all():
                    codebook_loss = audio_shift_logits.sum() * 0.0  # connect the gradient
                else:
                    codebook_loss = audio_loss_fct(codebook_logits, codebook_labels)
                codebook_losses[codebook] = codebook_loss
            
            audio_loss = torch.sum(codebook_losses * self.audio_codebook_weights.to(target_device))
            loss += audio_loss
        elif label_audio_ids is not None and label_audio_ids.shape[-1] == 0:
            # FIXME! This is a hack to make ZeRO work for heterogenous computation.
            # It is possible that one worker received a batch that contains no audio-out tokens while the other workers received batches with audio-out tokens.
            # In this scenario, we fake the compute associated with `self._embed_audio_ids()` and `audio_logits`
            # to ensure that ZeRO will still work.
            # This essentially fakes the forward + backward call for the layers.
            codebook_losses = torch.zeros([self.audio_num_codebooks], device=target_device)
            audio_loss = (
                torch.sum(audio_logits, dtype=torch.float32) + torch.sum(audio_out_embed, dtype=torch.float32)
            ) * 0
            loss += audio_loss
        else:
            codebook_losses = torch.zeros([self.audio_num_codebooks], device=target_device)
            audio_loss = torch.tensor(0, dtype=torch.float32, device=target_device)

        if loss is not None and audio_loss is None:
            llm_loss = loss
        elif loss is not None and audio_loss is not None:
            llm_loss = loss - audio_loss

        next_cache = past_key_values if use_cache else None

        ret = HiggsAudioModelOutputWithPast(
            loss=loss,
            llm_loss=llm_loss,
            audio_loss=audio_loss,
            codebook_losses=codebook_losses,
            logits=logits,
            audio_logits=audio_logits,
            expanded_input_ids=input_ids,
            expanded_labels=labels,
            audio_in_mask=audio_in_mask,
            audio_in_discrete_codes_mask=audio_in_discrete_codes_mask,
            audio_out_mask=audio_out_mask,
            attention_mask=attention_mask,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            audio_hidden_states=audio_hidden_states,
            attentions=all_self_attns,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            outputs = ret.to_tuple()
            return outputs

        return ret

    # Overwrite GenerationMixin._update_model_kwargs_for_generation
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        extend_attention_mask: bool = True
    ) -> Dict[str, Any]:
        """Update the model kwargs for each step."""
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if extend_attention_mask:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        if "cache_audio_discrete_codes_mask" in model_kwargs:
            if model_kwargs["cache_audio_discrete_codes_mask"] is None:
                model_kwargs["cache_audio_discrete_codes_mask"] = (
                    outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask
                )
            else:
                model_kwargs["cache_audio_discrete_codes_mask"] = torch.concat(
                    [
                        model_kwargs["cache_audio_discrete_codes_mask"],
                        outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask,
                    ],
                    1,
                )

        return model_kwargs

    def _copy_kv_cache(self, from_cache: Cache, to_cache: Cache):
        num_layers = self.config.text_config.num_hidden_layers
        """ Copy the key-value pairs from one cache to another. """
        for layer_idx in range(num_layers):
            from_cache_size = from_cache.get_max_cache_shape()
            assert to_cache.get_max_cache_shape() >= from_cache_size, \
                f"The target cache size {to_cache.get_max_cache_shape()} is smaller than the source cache size {from_cache_size}."
            to_cache.key_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.key_cache[layer_idx]
            to_cache.value_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.value_cache[layer_idx]

    def _prepare_kv_cache(self, current_sequence_length: int,
                          current_past_key_values_bucket: Optional[int],
                          past_key_values_buckets: OrderedDict[int, Cache],
        ) -> Tuple[Optional[Cache], Optional[int]]:
        """ Prepare the KV cache for the current sequence length. """
        for cache_length in past_key_values_buckets.keys():
            if cache_length >= current_sequence_length:
                # Promote to the next KV cache bucket, copy the current KV cache bucket
                # to the new one.
                if current_past_key_values_bucket is not None and \
                    cache_length != current_past_key_values_bucket:
                    self._copy_kv_cache(
                        past_key_values_buckets[current_past_key_values_bucket],
                        past_key_values_buckets[cache_length]
                    )

                return past_key_values_buckets[cache_length], cache_length
        
        raise ValueError(f"The current sequence length {current_sequence_length} is larger than "
                         f"all past key values buckets {past_key_values_buckets.keys()}.")

    def _sample_audio_tokens(
        self,
        hidden_states: torch.Tensor,
        audio_logits: torch.Tensor,
        audio_out_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        torch_generator: Optional[torch.Generator],
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[int]]:
        """Sample audio tokens and its corresponding text tokens from the logits"""

        # parameters related to repetition aware sampling
        ras_win_len = generation_config.generation_kwargs.get("ras_win_len", None)
        ras_win_max_num_repeat = generation_config.generation_kwargs.get("ras_win_max_num_repeat", 2)
        audio_eos_token_id = generation_config.generation_kwargs.get("audio_eos_token_id", None)

        # In the audio generation mode, we sample from audio_logits and keep updating audio_out_ids.
        next_audio_token_logits = audio_logits.clone()[-1, :, :].float().to(device)
        # TopP, TopK logits processor supports empty input_ids
        next_audio_token_scores = logits_processor(None, next_audio_token_logits)

        # token selection
        if do_sample:
            # next_audio_token_scores has been applied top_p, top_k, and temperature.
            probs = nn.functional.softmax(next_audio_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_audio_tokens = \
                torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
        else:
            next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)

        # next_tokens: (num_codebooks, )
        if ras_win_len is not None:
            # check if there are repetitions over a window of tokens.
            rep_num = (audio_out_ids[:, -ras_win_len:] == next_audio_tokens.unsqueeze(1)).sum(dim=1)

            # if we saw repeated tokens in the most recent window of tokens, resample without temperature.
            row_indices = torch.nonzero(rep_num >= ras_win_max_num_repeat).squeeze(1)
            resampled_next_tokens = next_audio_token_logits[row_indices] \
                                    .softmax(dim=-1) \
                                    .multinomial(1, replacement=True, generator=torch_generator) \
                                    .squeeze(1)
            next_audio_tokens[row_indices] = resampled_next_tokens

        # Force the next text tokens to be <|AUDIO_OUT|> in audio generation mode
        next_tokens = torch.full(
            (audio_logits.shape[0],),
            self.config.audio_out_token_idx,
            dtype=torch.long,
            device=device,
        )
        return (next_tokens, next_audio_tokens, next_audio_token_logits, next_audio_token_scores,
                num_delay, num_remaining_delays)

    def _sample_text_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        generation_mode: GenerationMode,
        torch_generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample text tokens from the logits"""
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if generation_mode == GenerationMode.AUDIO_INIT:
            # See the audio bos token, we should start generating audio tokens
            next_tokens = torch.full(
                (input_ids.shape[0],),
                self.audio_out_token_idx,
                dtype=torch.long,
                device=device,
            )
            next_audio_tokens = torch.full(
                (self.config.audio_num_codebooks,),
                self.config.audio_stream_bos_id,
                dtype=torch.long,
                device=device,
            )
        else:
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
        
            next_audio_tokens = None
        
        return next_tokens, next_audio_tokens, next_token_logits, next_token_scores

    # Built on top of GenerationMixin._sample.
    # We revise the implementation to support generating both audio / text.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        past_key_values_buckets: Optional[OrderedDict[int, Cache]],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for joint text/audio models using **multinomial sampling**.

        This function may also be revised to support generating samples from HiggsAudio-like end-to-end text/audio models built on top of LLMs.
        If the input_ids ends with <|audio_out_bos|>, we will switch to the audio-generation mode.

        ```
        ...<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>
        ```

        Otherwise, we will keep generating the text tokens.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # Batched generation is safe only when no row will ever enter audio generation
        # (true for ASR / understanding use cases — the model output is text-only).
        # For mixed text/audio generation, stay on bs=1 to preserve the original
        # multimodal state-machine semantics.
        _bs_gt_1_ok = input_ids.shape[0] > 1 and not (
            (input_ids[:, -1] == self.config.audio_out_token_idx).any()
            or (generation_config.generation_kwargs.get("audio_out_bos_token_id") is not None
                and (input_ids[:, -1] == generation_config.generation_kwargs["audio_out_bos_token_id"]).any())
        )
        assert input_ids.shape[0] == 1 or _bs_gt_1_ok, (
            "_sample() only supports batch_size>1 for text-only generation (ASR). "
            "For mixed text/audio generation, use batch_size=1."
        )
        audio_out_bos_token_id = generation_config.generation_kwargs.get("audio_out_bos_token_id", None)

        # torch generator for sampling
        seed = generation_config.generation_kwargs.get("seed", None)
        if seed is not None:
            torch_generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        else:
            torch_generator = None

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        # Used to track which past_key_va
        self.current_past_key_values_bucket = None

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        if generation_config.use_cache:
            model_kwargs["cache_audio_discrete_codes_mask"] = None

        init_model_input = True
        num_delay = 0
        num_remaining_delays = None
        audio_sequences = []
        # A tensor to keep track of all the audio placeholder tokens.
        input_ids_full = input_ids.clone()

        # Initialize the audio variables based on the input prompt.
        if input_ids[0][-1] == self.config.audio_out_token_idx:
            audio_sequences = [
                model_kwargs["audio_out_ids"][:, model_kwargs["audio_out_ids_start"][-1]:]
            ]

        # NOTE: https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/generation/utils.py#L2025
        # does not take in cur_len=cur_len, max_length=max_length anymore
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # Check which multimodal stage we are in.
            # For batched ASR (bs>1), all rows are in TEXT mode by construction
            # (caller guarantees no audio generation); derive mode from row 0.
            # Single-sample multimodal generation still uses input_ids[0][-1].
            if input_ids[0][-1] == audio_out_bos_token_id:
                generation_mode = GenerationMode.AUDIO_INIT
            elif input_ids[0][-1] == self.audio_out_token_idx:
                generation_mode = GenerationMode.AUDIO_IN_PROGRESS
            else:
                generation_mode = GenerationMode.TEXT

            is_audio_generation_mode = generation_mode == GenerationMode.AUDIO_IN_PROGRESS

            if init_model_input or not generation_config.use_cache:
                model_inputs = {"input_ids": input_ids, **model_kwargs}
            else:
                model_inputs = {"input_ids": input_ids[:, -1:], **model_kwargs}

                if is_audio_generation_mode and generation_config.use_cache:
                    model_inputs["audio_out_ids"] = model_kwargs["audio_out_ids"][:, -1:]
                    model_inputs["audio_out_ids_start"] = torch.tensor([0], dtype=torch.long, device=input_ids.device)
                elif not is_audio_generation_mode:
                    del model_inputs["audio_out_ids"]
                    del model_inputs["audio_out_ids_start"]

                if generation_config.use_cache:
                    if "audio_features" in model_inputs and model_inputs["audio_features"] is not None:
                        model_inputs["audio_features"] = model_inputs["audio_features"][:0, ...]
                    if "audio_feature_attention_mask" in model_inputs and model_inputs["audio_feature_attention_mask"] is not None: # for xcodec this does not exist
                        model_inputs["audio_feature_attention_mask"] = model_inputs["audio_feature_attention_mask"][:0, ...]
                    if "audio_in_ids" in model_inputs and model_inputs["audio_in_ids"] is not None:
                        model_inputs["audio_in_ids"] = None
                        model_inputs["audio_in_ids_start"] = None

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if past_key_values_buckets is not None:
                past_key_values, self.current_past_key_values_bucket = \
                    self._prepare_kv_cache(cur_len, self.current_past_key_values_bucket,
                                           past_key_values_buckets)
                if past_key_values is not None:
                    model_inputs.update({"past_key_values": past_key_values})
                model_inputs["past_key_values_buckets"] = past_key_values_buckets

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # Update the actual sequence length after the first forward pass
            if init_model_input and past_key_values_buckets is not None:
                cur_len = \
                    past_key_values_buckets[self.current_past_key_values_bucket].get_seq_length().item()

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                extend_attention_mask=True,
            )

            # After the first forward pass, we can set init_model_input to False.
            init_model_input = False

            if synced_gpus and this_peer_finished:
                continue

            if is_audio_generation_mode:
                # In audio generation mode, we sample the audio tokens from audio logits. 
                # It might also generate the audio eos token to end the audio generation.
                next_tokens, next_audio_tokens, next_audio_token_logits, next_audio_token_scores, num_delay, num_remaining_delays = \
                    self._sample_audio_tokens(
                        hidden_states=outputs.audio_hidden_states,
                        audio_logits=outputs.audio_logits,
                        audio_out_ids=model_kwargs["audio_out_ids"],
                        do_sample=do_sample,
                        logits_processor=logits_processor,
                        device=input_ids.device,
                        torch_generator=torch_generator,
                        generation_config=generation_config,
                        num_delay=num_delay,
                        num_remaining_delays=num_remaining_delays,
                    )

                # update generated ids, model inputs, and length for next step
                model_kwargs["audio_out_ids"] = torch.cat(
                    [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=-1
                )
                audio_sequences[-1] = torch.cat([audio_sequences[-1], next_audio_tokens[:, None]], dim=-1)

                if streamer is not None:
                    streamer.put(next_audio_tokens.cpu())
            else:
                # In text generation mode, we sample the text tokens from text logits.
                # It might also generate the audio placeholder token to start the audio generation.
                next_tokens, next_audio_tokens, next_token_logits, next_token_scores = \
                    self._sample_text_tokens(
                        input_ids=input_ids,
                        logits=outputs.logits,
                        do_sample=do_sample,
                        logits_processor=logits_processor,
                        device=input_ids.device,
                        generation_mode=generation_mode,
                        torch_generator=torch_generator,
                    )

                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                if next_audio_tokens is not None:
                    # If the token is audio bos token, we will generate the audio placeholder token
                    # and the corrensponding audio stream bos token to start the audio generation.
                    audio_sequences.append(next_audio_tokens[:, None])
                    if streamer is not None:
                        streamer.put(next_audio_tokens.cpu())
                    if model_kwargs["audio_out_ids"] is None or model_kwargs["audio_out_ids"].shape[0] == 0:
                        # Initialize audio_out_ids
                        model_kwargs["audio_out_ids"] = next_audio_tokens[:, None]
                        model_kwargs["audio_out_ids_start"] = torch.tensor([0], dtype=torch.long, device=input_ids.device)
                    else:
                        model_kwargs["audio_out_ids_start"] = torch.concat(
                            [
                                model_kwargs["audio_out_ids_start"],
                                torch.tensor(
                                    [model_kwargs["audio_out_ids"].shape[1]], dtype=torch.long, device=input_ids.device
                                ),
                            ],
                            dim=0,
                        )
                        model_kwargs["audio_out_ids"] = torch.concat(
                            [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=1
                        )

            if return_dict_in_generate:
                if output_scores:
                    if is_audio_generation_mode:
                        scores += (next_audio_token_scores,)
                    else:
                        scores += (next_token_scores,)
                if output_logits:
                    if is_audio_generation_mode:
                        raw_logits += (next_audio_token_logits,)
                    else:
                        raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if "tokenizer_length" in generation_config.generation_kwargs:
                tokenizer_length = generation_config.generation_kwargs["tokenizer_length"]
                if torch.max(next_tokens) >= tokenizer_length:
                    raise ValueError(f"Next generated token has max value {torch.max(next_tokens)} which is greater than the tokenizer's vocabulary size {tokenizer_length}, this is undesired behavior.")

            # update generated ids, model inputs, and length for next step
            if not is_audio_generation_mode or next_tokens[0] != self.audio_out_token_idx:
                # We only add one <|AUDIO_OUT|> token to the input_ids for simplicity.
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids_full = torch.cat([input_ids_full, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return HiggsAudioGenerationOutput(
                sequences=input_ids,
                audio_sequences=audio_sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, audio_sequences

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_wv_lengths: Optional[torch.LongTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        audio_out_bos_token_id: int = None,
        audio_eos_token_id: int = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        The generate function in huggingface generally follows these steps:

        for sample_step in 1, 2, 3, 4, 5, ...
            ...

        """
        # Batched generation is allowed for text-only (ASR / understanding) use
        # cases, where no row will ever enter audio generation. Mixed text/audio
        # generation still requires batch_size=1.
        if input_ids.shape[0] > 1:
            _audio_bos = audio_out_bos_token_id if audio_out_bos_token_id is not None \
                else getattr(self, "audio_out_bos_token_id", None)
            _audio_out = getattr(self.config, "audio_out_token_idx", None)
            _has_audio_token = False
            if _audio_bos is not None:
                _has_audio_token = _has_audio_token or (input_ids == _audio_bos).any().item()
            if _audio_out is not None:
                _has_audio_token = _has_audio_token or (input_ids == _audio_out).any().item()
            assert not _has_audio_token, (
                "HiggsAudioModel.generate() with batch_size>1 is only supported for "
                "text-only generation (ASR / understanding). The input contains an "
                "audio-generation token; use batch_size=1 for mixed text/audio generation."
            )
        generation_config, kwargs = self._prepare_generation_config(kwargs.pop("generation_config", None), **kwargs)
        if audio_out_bos_token_id is not None:
            generation_config.generation_kwargs["audio_out_bos_token_id"] = audio_out_bos_token_id
        else:
            try:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = self.audio_out_bos_token_id
            except:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = None

        if audio_eos_token_id is not None:
            generation_config.generation_kwargs["audio_eos_token_id"] = audio_eos_token_id
        else:
            try:
                generation_config.generation_kwargs["audio_eos_token_id"] = self.audio_eos_token_id
            except:
                generation_config.generation_kwargs["audio_eos_token_id"] = None
        
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

        generation_config.generation_kwargs["ras_win_len"] = kwargs.pop("ras_win_len", None)
        generation_config.generation_kwargs["ras_win_max_num_repeat"] = kwargs.pop("ras_win_max_num_repeat", 2)
        # Set generation seed if determinstic generation is required
        if seed is not None:
            generation_config.generation_kwargs["seed"] = seed

        # Store tokenizer in generation config if it is in kwargs without popping it
        if "tokenizer" in kwargs:
            generation_config.generation_kwargs["tokenizer_length"] = len(kwargs["tokenizer"])

        # input_ids: [bsz, seq_len]
        # The merging of audio features happens inside the forward path. The input_ids does not need to change.
        # TODO: prepare the final input embeddings to improve generation performance
        input_ids_length = input_ids.shape[-1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=None,
            inputs_tensor=None,
            input_ids_length=input_ids_length,
        )
        assert generation_config.num_beams == 1, "Currently, we only support beam search with num_beams=1"
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_scores = generation_config.output_scores

        # When attn_implement is spda or flash-attention, it will create causal mask automatically.
        attention_mask = kwargs.pop("attention_mask", None)
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_wv_lengths=audio_wv_lengths,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            past_key_values=past_key_values,
            generation_config=generation_config,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            past_key_values_buckets=past_key_values_buckets,
            **kwargs,
        )

    def parameter_count_per_component(self):
        """Count the number of parameters per component in the model.

        HiggsAudio has the following main components:
            audio_tower: For mapping audio features to hidden states),
            llm_embed: The size of embedding layer of the LLM
            llm_non_embed: The size of non-embedding layer of the LLM
            audio_adapter: The overall size of additional layers for audio generation

        """
        trainable_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }
        total_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }

        total_stats["overall"] = count_parameters(self, trainable_only=False)
        trainable_stats["overall"] = count_parameters(self, trainable_only=True)

        for mod in [self.audio_tower]:
            if mod is not None:
                total_stats["audio_tower"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_tower"] += count_parameters(mod, trainable_only=True)

        total_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=False)
        trainable_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=True)

        total_stats["audio_embed"] = count_parameters(self.audio_codebook_embeddings, trainable_only=False)
        trainable_stats["audio_embed"] = count_parameters(self.audio_codebook_embeddings, trainable_only=True)

        # Calculate number of parameters for LLM
        for layer in self.layers:
            total_stats["llm_non_embed"] += count_parameters(layer, trainable_only=False)
            trainable_stats["llm_non_embed"] += count_parameters(layer, trainable_only=True)
        total_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=False)
        trainable_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=True)

        total_stats["audio_adapter"] += count_parameters(self.audio_decoder_proj.audio_lm_head, trainable_only=False)
        trainable_stats["audio_adapter"] += count_parameters(self.audio_decoder_proj.audio_lm_head, trainable_only=True)
        total_stats["llm_embed"] += count_parameters(self.audio_decoder_proj.text_lm_head, trainable_only=False)
        trainable_stats["llm_embed"] += count_parameters(self.audio_decoder_proj.text_lm_head, trainable_only=True)

        other_audio_modules = [self.audio_encoder_proj]
        if self.use_audio_out_embed_projector:
            other_audio_modules.append(self.audio_out_embed_projector)

        for mod in other_audio_modules:
            if mod is not None:
                total_stats["audio_adapter"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_adapter"] += count_parameters(mod, trainable_only=True)
        return {"trainable": trainable_stats, "total": total_stats}

    def set_skip_audio_tower(self):
        self.config.skip_audio_tower = True
        self.config.encode_whisper_embed = False

    def set_encode_audio_in_tokens(self):
        self.config.encode_audio_in_tokens = True

    def freeze_audio_tower(self):
        if self.audio_tower is not None:
            for param in self.audio_tower.parameters():
                param.requires_grad = False

    def freeze_audio_encoder_proj(self):
        if self.audio_encoder_proj is not None:
            for param in self.audio_encoder_proj.parameters():
                param.requires_grad = False

    def freeze_llm(
        self, 
        freeze_embed=True, 
        freeze_embed_until_idx: Optional[int] = None,
        unfreeze_first_n_layers: Optional[int] = None,
    ):
        total_layers = len(self.layers)
        unfreeze_until = total_layers if unfreeze_first_n_layers is None else min(unfreeze_first_n_layers, total_layers)
        logger.info(f"Unfreeze first {unfreeze_until} layers and freezing the remaining {total_layers - unfreeze_until} layers.")
        for idx, layer in enumerate(self.layers):
            if idx < unfreeze_until: continue
            for param in layer.parameters():
                param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

        if freeze_embed:
            if freeze_embed_until_idx is None:
                for param in self.embed_tokens.parameters():
                    param.requires_grad = False
            else:
                assert isinstance(self.embed_tokens, nn.Embedding)
                self.embed_tokens = PartiallyFrozenEmbedding(
                    original_embedding=self.embed_tokens, freeze_until_idx=freeze_embed_until_idx
                )

    def freeze_text_head(self, freeze_text_head_until_idx: Optional[int] = None):
        """Freeze the final text head"""
        if freeze_text_head_until_idx is None:
            for param in self.audio_decoder_proj.text_lm_head.parameters():
                param.requires_grad = False
        else:
            assert isinstance(self.audio_decoder_proj.text_lm_head, nn.Linear)
            self.audio_decoder_proj.text_lm_head = PartiallyFrozenLinear(original_linear=self.audio_decoder_proj.text_lm_head, freeze_until_idx=freeze_text_head_until_idx)

    @classmethod
    def merge_weights_from_checkpoint(cls, checkpoint_dir: str, merged_output_dir: str, *model_args, **kwargs):
        # For users' convenience, we merge back embedding and text_lm_head if they are splitted
        splitted_model = super().from_pretrained(
            checkpoint_dir,
            *model_args,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            **{**kwargs, "state_dict": None}  # Prevent auto-loading state_dict
        )

        # Load all safetensor shards
        state_dict = {}
        shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

        for shard_path in shard_paths:
            shard_dict = load_file(shard_path)  # Load each shard
            state_dict.update(shard_dict)  # Merge into a single dict

        # Merge weights
        if "audio_decoder_proj.text_lm_head.linear_frozen.weight" in state_dict and "audio_decoder_proj.text_lm_head.linear_trainable.weight" in state_dict:
            state_dict["audio_decoder_proj.text_lm_head.weight"] = torch.cat([
                state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"],
                state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"]
            ], dim=0)

            del state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"]
            del state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"]

        if "embed_tokens.embedding_frozen.weight" in state_dict and "embed_tokens.embedding_trainable.weight" in state_dict:
            state_dict["embed_tokens.weight"] = torch.cat([
                state_dict["embed_tokens.embedding_frozen.weight"],
                state_dict["embed_tokens.embedding_trainable.weight"]
            ], dim=0)

            del state_dict["embed_tokens.embedding_frozen.weight"]
            del state_dict["embed_tokens.embedding_trainable.weight"]

        # Load the final state_dict
        splitted_model.load_state_dict(state_dict, strict=True)

        if merged_output_dir:
            splitted_model.save_pretrained(merged_output_dir, is_main_process=True, state_dict=state_dict)

    @torch.inference_mode()
    def capture_model(self, past_key_values: list[Union[Cache, List[torch.FloatTensor]]]) -> None:
        """Capture CUDA graphs for the model's forward pass with different KV cache lengths.

        Args:
            past_key_values: List of KV caches to capture graphs for
        """
        for past_key_value in past_key_values:
            kv_cache_length = past_key_value.get_max_cache_shape()
            # We capture two graphs, one for decoding audio tokens and one for decoding text tokens
            for is_decoding_audio_token in [True, False]:
                runner = CUDAGraphRunner(self._forward_core)

                # Create dummy inputs for graph capture
                batch_size = 1
                hidden_dim = self.text_config.hidden_size

                hidden_states = torch.zeros((batch_size, 1, hidden_dim), dtype=self.config.torch_dtype, device="cuda")
                causal_mask = torch.ones(
                    (batch_size, 1, 1, kv_cache_length), dtype=self.config.torch_dtype, device="cuda"
                )
                position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device="cuda")
                audio_discrete_codes_mask = torch.tensor([[is_decoding_audio_token]], dtype=torch.bool, device="cuda")
                cache_position = torch.tensor([kv_cache_length - 1], dtype=torch.long, device="cuda")
                audio_attention_mask = torch.ones_like(causal_mask)
                fast_forward_attention_mask = torch.ones_like(causal_mask)

                runner.capture(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    audio_discrete_codes_mask=audio_discrete_codes_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_value,
                    use_cache=True,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    is_decoding_audio_token=is_decoding_audio_token,
                    is_using_cuda_graph=True,
                )

                self.decode_graph_runners[kv_cache_length][is_decoding_audio_token] = runner
