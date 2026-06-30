import warnings
import torch

from typing import Optional, Tuple
from transformers.cache_utils import Cache, StaticCache, EncoderDecoderCache
try:
    from transformers.models.whisper.modeling_whisper import WhisperFlashAttention2, WHISPER_ATTENTION_CLASSES
except ImportError:
    WhisperFlashAttention2 = object
    WHISPER_ATTENTION_CLASSES = {}
    warnings.warn("WhisperFlashAttention2 could not be imported from transformers.models.whisper.modeling_whisper. "
                  "Make sure you have transformers<4.53.0 installed when using HiggsAudio models. This might cause errors "
                  "if you use HiggsAudio in this environment.", ImportWarning)
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.integrations import is_deepspeed_available


from .utils import support_deepspeed_ulysses, is_deepspeed_ulysses_enabled, deepspeed_ulysses_attention, deepspeed_ulysses_rope, sequence_chunking_per_rank

DistributedAttention = None
vocab_sequence_parallel_cross_entropy = None
if is_deepspeed_available():
    import importlib
    DistributedAttention = importlib.import_module("deepspeed.sequence.layer").DistributedAttention
    vocab_sequence_parallel_cross_entropy = importlib.import_module("deepspeed.sequence.cross_entropy").vocab_sequence_parallel_cross_entropy


logger = logging.get_logger(__name__)



@deepspeed_ulysses_attention(seq_dim=1, head_dim=2)
def _distributed_flash_attention_forward(*args, **kwargs):
    return _flash_attention_forward(*args, **kwargs)


def _patched_flash_attention_forward(
    query, key, value, *args, **kwargs,
) -> torch.Tensor:
    # IMPORTANT! Implementation here is wrong and is only for the purpose of obtaining the correct attn_weight shape
    if query.size(-2) != key.size(-2):
        key = key.repeat_interleave(query.size(-2) // key.size(-2), -2)
        value = value.repeat_interleave(query.size(-2) // value.size(-2), -2)

    return (query + key + value) / 3.0


def _distributed_higgs_flash_attention_forward(*args, **kwargs):
    if args[0].size(0) == 0 or args[1].size(1) == 0:
        return _patched_flash_attention_forward(*args, **kwargs)
    else:
        return _distributed_flash_attention_forward(*args, **kwargs)

@support_deepspeed_ulysses
class HiggsDistributedWhisperFlashAttention2(WhisperFlashAttention2):
    """
    Higgs Whisper flash attention module. This module inherits from `WhisperFlashAttention2` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Remove the redundant transpose and apply the monkey-patch for zero-shape input."""
        if seq_len == -1:
            return tensor.view(bsz, tensor.shape[1], self.num_heads, self.head_dim)
        else:
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "The `static` cache implementation is not compatible with `attn_implementation='flash_attention_2'`. "
                "Use `attn_implementation='sdpa'` in the meantime, and open an issue at https://github.com/huggingface/transformers"
            )
        # WhisperFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("HiggsWhisperFlashAttention2 attention does not support output_attentions")

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = torch.reshape(self.q_proj(hidden_states), (bsz, tgt_len, self.num_heads, self.head_dim))

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, {"cache_position": cache_position}
                )

        # Cache layout [batch_size, num_heads, sequence_length, head_dim]
        if past_key_value is not None:
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            if is_deepspeed_ulysses_enabled():
                causal_mask = attention_mask[:, : key_states.shape[-3] * getattr(self, "sp_size", 1)]
            else:
                causal_mask = attention_mask[:, : key_states.shape[-3]]

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _distributed_higgs_flash_attention_forward(
            query_states,
            key_states,
            value_states,
            causal_mask,
            tgt_len * getattr(self, "sp_size", 1) if is_deepspeed_ulysses_enabled() else tgt_len,
            dropout=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, hidden_states.size(2))
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

WHISPER_ATTENTION_CLASSES["flash_attention_2"] = HiggsDistributedWhisperFlashAttention2
