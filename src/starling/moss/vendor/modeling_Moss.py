from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils.auto_docstring import auto_docstring
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

class MossConfig(Qwen3Config):
    model_type = "moss"
    is_composition = True
    # Make the architecture discoverable by Megatron-Bridge's AutoBridge
    # when loading configs from disk.
    architectures = ["MossForCausalLM"]

    def __init__(
        self,
        audio_config=None,
        language_config=None,
        adapter_hidden_size=8192,
        ignore_index=-100,
        **kwargs
    ):
        num_hidden_layers = None
        if language_config is not None:
            if isinstance(language_config, dict):
                num_hidden_layers = language_config.get("num_hidden_layers", None)
            elif isinstance(language_config, Qwen3Config):
                num_hidden_layers = language_config.num_hidden_layers
        
        if num_hidden_layers is not None:
            kwargs.update({"num_hidden_layers": num_hidden_layers})
            
        # Initialize parent Qwen3Config with kwargs to handle standard config params
        super().__init__(**kwargs)
        
        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        if isinstance(audio_config, Qwen3OmniMoeAudioEncoderConfig):
            audio_config = audio_config
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
            
        if isinstance(language_config, dict):
            language_config = Qwen3Config(**language_config)
        elif isinstance(language_config, Qwen3Config):
            language_config = language_config
        elif language_config is None:
            language_config = Qwen3Config()

        self.audio_config = audio_config
        self.language_config = language_config
        self.adapter_hidden_size = adapter_hidden_size
        self.ignore_index = ignore_index
        self.dtype = language_config.dtype

    def to_dict(self):
        output = super().to_dict()
        if self.audio_config is not None:
            if hasattr(self.audio_config, "to_dict"):
                 output["audio_config"] = self.audio_config.to_dict()
            else:
                 output["audio_config"] = self.audio_config
        if self.language_config is not None:
            if hasattr(self.language_config, "to_dict"):
                output["language_config"] = self.language_config.to_dict()
            else:
                output["language_config"] = self.language_config
        return output

class MossGatedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

@auto_docstring
class MossPreTrainedModel(PreTrainedModel):
    config: MossConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
    }

class MossModel(MossPreTrainedModel):
    config_class = MossConfig
    
    def __init__(self, config: MossConfig):
        super().__init__(config)
        
        self.audio_model = Qwen3OmniMoeAudioEncoder(config.audio_config)
        self.language_model = Qwen3Model(config.language_config)
        
        self.audio_adapter = MossGatedMLP(
            input_size=config.audio_config.output_dim,
            hidden_size=config.adapter_hidden_size,
            output_size=config.language_config.hidden_size
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_audio_features(self, input_features, feature_lens):
        audio_outputs = self.audio_model(
            input_features=input_features,
            feature_lens=feature_lens,
        )
        return audio_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Process audio and merge embeddings if audio is present
        if audio_data is not None:
            # [B, Audio_Len, D]
            audio_embeds = self.get_audio_features(audio_data, audio_data_seqlens)
            audio_embeds = self.audio_adapter(audio_embeds)

            # audio_input_mask: [B, L] -> [B, L, 1] -> [B, L, D]
            # D elements will be replaced by audio embeddings
            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

        # 3. Forward pass through language model
        return self.language_model(
            input_ids=None, # We pass inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

class MossForCausalLM(MossPreTrainedModel, GenerationMixin):
    config_class = MossConfig
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_save = ["lm_head.weight"]
    
    def __init__(self, config: MossConfig):
        super().__init__(config)
        self.model = MossModel(config)
        self.vocab_size = config.language_config.vocab_size
        self.lm_head = nn.Linear(config.language_config.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        # accept/forward kwargs (e.g. recompute_mapping) for newer transformers
        super().tie_weights(*args, **kwargs)

        # tie lm_head to input embeddings
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            shift_logits = shift_logits.view(-1, self.config.language_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs
    ):
        # decoding step (KV cache present) keeps only the last token and drops audio inputs;
        # prefill step pulls audio inputs from kwargs.
        position_ids = kwargs.get("position_ids", None)
        if cache_position is not None and cache_position[0] > 0:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            audio_data = None
            audio_input_mask = None
            audio_data_seqlens = None
        else:
            audio_data = kwargs.get("audio_data", None)
            audio_input_mask = kwargs.get("audio_input_mask", None)
            audio_data_seqlens = kwargs.get("audio_data_seqlens", None)

        # prefer inputs_embeds at the first step when present
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "audio_data": audio_data,
            "audio_input_mask": audio_input_mask,
            "audio_data_seqlens": audio_data_seqlens,
        })
        
        return model_inputs

__all__ = [
    "MossConfig",
    "MossModel",
    "MossForCausalLM",
]
