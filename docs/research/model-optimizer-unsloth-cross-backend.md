# Model Optimizer and Unsloth: portable ideas for Starling

Reviewed [NVIDIA Model Optimizer at `23355eda9`](https://github.com/NVIDIA/Model-Optimizer/tree/23355eda90a25c290f9b1fdfb928ad54caae7d10)
against Starling's [quantization path](../quantization.md) and native ggml
runtime. The useful pieces are mostly **offline methods**: produce better
weights or choose which supported format each layer uses, then deploy those
weights through the existing CPU, Vulkan, and CUDA paths. This is a source
assessment, not a quality or speed measurement on Starling models.

| Method in Model Optimizer | What transfers to Starling | First check |
| --- | --- | --- |
| [GGML IQ1_S/IQ2_XS/IQ2_XXS codecs](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/quantization/ggml/registry.py) | Offline candidate packing and fake quantization. [IQ2_XXS](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/quantization/ggml/iq2_xxs.py#L16-L35) packs 256 weights into 66 bytes and has a [PyTorch fallback when its optional CUDA packer is absent](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/quantization/ggml/iq2_xxs.py#L187-L223). The encoded bytes can be tested with any ggml backend that supports the format. | Cross-decode packed blocks in Starling's pinned ggml, compare with ModelOpt's decoder, then test held-out tensors and WER. ModelOpt names a llama.cpp codebook revision `9b05354`; Starling pins ggml `e91ded11`, so layout compatibility must be measured. |
| [AutoQuantize](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/docs/source/announcements/autoquantize.rst) | Its gradient-squared-weighted output-error score and grouped integer search can rank **formats Starling already runs**. The score is a proxy; the assignment can be exported as a recipe. | Replace the paper's eligible-weight effective-bits budget with measured resident bytes and per-device latency where those matter, especially co-residency and the Pixel IQ2 fallback. Check each candidate against [#50](https://github.com/sims1253/starling/issues/50) and [#316](https://github.com/sims1253/starling/issues/316) held-out WER gates. |
| [Local-Hessian scale choice](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/docs/source/announcements/local-hessian.rst) | Uses an input second-moment matrix `XXᵀ` to choose per-block scales that reduce *layer output* error. Scale search is offline, so a retained packed format needs no new inference kernel. | Adapt the objective to Starling's actual block/codebook layout and compare against its importance-matrix quantizer. ModelOpt's implementation targets NVFP4 16-weight blocks with FP8 scales; it is not a drop-in IQ2 encoder. |
| [GPTQ](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/quantization/utils/calib_utils.py#L116-L281) | Offline, sequential weight updates are relevant to [#49](https://github.com/sims1253/starling/issues/49). | Try one layer with Starling's exact pack/decode callable and measure runtime and held-out error. The inspected helper uses full-matrix fake quantization during column updates; cost and validity for IQ blocks are still unknown. |
| [Straight-through GGML fake quantization](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/quantization/ggml/common.py#L75-L119) | `inputs + (reconstructed - inputs).detach()` gives an offline QAT/QAD training primitive with the packed IQ codec in the forward pass. | Consider only after an offline candidate fails the quality gate and a training set is available; run Starling's WER and output checks again. |
| [EAGLE/DFlash draft training](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/examples/speculative_decoding/README.md) | Learned draft heads and their training procedure do not require a CUDA *deployment* algorithm in principle. | [#314](https://github.com/sims1253/starling/issues/314) still needs native hidden-state/KV hooks, a draft format, and the [#310](https://github.com/sims1253/starling/issues/310) dictation workload to justify added cost. |

ModelOpt's [HF](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/export/unified_export_hf.py#L638)
and [Megatron](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/modelopt/torch/export/unified_export_megatron.py#L1188)
paths pack IQ payloads into their exported checkpoints. **Inference from the
inspected tree:** there is no generic GGUF container writer in those paths; a
Starling bridge still has to map tensor names, shapes, types, and metadata
before a model can load. Codec byte parity alone does not establish model
quality. ModelOpt's IQ2_XXS encoder uses an empirical super-block scale and a
grid search, whereas Starling's [importance-matrix path](../quantization.md)
weights quantization error from recorded activations. Matching payload format
does not imply matching encoder choices or WER.

The [AutoQuantize source and announcement](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/docs/source/announcements/autoquantize.rst)
exclude embeddings, norms, and other ineligible parameters from effective-bit
accounting and sum per-group sensitivity scores. Both are useful search
simplifications, but Starling's deployment objective includes entire-model
resident memory, quantized layout conversion, and backend latency. For the
Pixel IQ2 route that [repacks to W8](../fast-engine.md), file bits are a poor
proxy for resident bytes. A format assignment must be judged by those measured
costs and by WER, not the paper's LLM benchmark gains.

[Dense channel/layer pruning followed by distillation](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/docs/source/guides/3_pruning.rst)
could eventually produce smaller ordinary GEMMs on several backends. Its
[2:4 sparsity speed claim](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/docs/source/guides/6_sparsity.rst#L129-L150)
depends on sparse Tensor Core support and is a lower priority for Starling's
Pixel path. The NVFP4/Qwen results in the Local-Hessian announcement are
evidence for that model and format, not evidence of ASR or Pixel gains.

**Recommended first experiment:** pack a small, fixed set of Starling linear
weights with ModelOpt IQ2_XXS on its non-CUDA path. Cross-decode each block
with pinned ggml, check decoded values and metadata, and compare layer output
error on held-out inputs against Starling's current imatrix recipe. Only after
that parity check should a whole-model mixed-format search or QAD run be considered.
No ModelOpt dependency, model rewrite, or runtime change is proposed here.

## Unsloth: deployment and training paths

Reviewed the [Unsloth documentation](https://unsloth.ai/docs) on 2026-09-28.
Its [Dynamic 3.0 GGUF description](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)
reports a more varied importance-matrix calibration corpus, improved layer
selection, and post-training quantization (PTQ). It explicitly says Dynamic
3.0 uses neither quantization-aware training (QAT) nor quantization-aware
distillation (QAD). Its published Qwen3.8 results are vendor measurements on
an LLM; they do not establish a gain for Starling's ASR models or Pixel
runtime. The public imatrix makes a candidate calibration input available,
but the cited page does not provide a complete recipe or implementation for
reproducing its layer-selection decisions. The ordinary
[`save_pretrained_gguf` export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
must not be assumed to reproduce Dynamic 3.0.

| Unsloth technique | Relevance and boundary for Starling |
| --- | --- |
| Dynamic layer formats and imatrix calibration | Another candidate recipe for [#316](https://github.com/sims1253/starling/issues/316). Compare per-layer formats and calibration on the *same* model and held-out set, then measure resident bytes, Pixel latency, and WER. Dynamic 3.0's Qwen-oriented coding/chat corpus cannot substitute for representative ASR input. |
| [Held-out 32-token divergence and KL divergence](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs#divergence-300-32) | Useful secondary diagnostics for Starling's text decoder: a one-token agreement rate can conceal trajectory drift. Keep [#50](https://github.com/sims1253/starling/issues/50) WER/CER, critical spans, and protected text as the deployment gates. Separate calibration from evaluation, and retain the same prompt template and decoding rules for baseline and candidate. |
| [GGUF export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) | Could help [#306](https://github.com/sims1253/starling/issues/306) merge a text-side LoRA into source precision and export. Each actual artifact still needs a load/decode check against Starling's pinned ggml, tensor types and architecture support, tokenizer, chat template, BOS/EOS, and fast-engine repacking where applicable. A `.gguf` extension alone is insufficient. |
| [TorchAO QAT](https://unsloth.ai/docs/blog/quantization-aware-training-qat) and [phone deployment](https://unsloth.ai/docs/basics/inference-and-deployment/deploy-llms-phone) | QAT is a possible offline quality-recovery experiment, but the documented `int8-int4` fake-quant recipe and `.pte` ExecuTorch/XNNPACK output target different formats and runtime from Starling's ggml IQ/Q_K path. `save_pretrained_torchao` is not a Starling GGUF export. Its Qwen3-0.6B Pixel 8 result is a separate model/runtime measurement. |
| [LoRA fine-tuning](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide) and [speech guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) | Tooling could be evaluated for [#306](https://github.com/sims1253/starling/issues/306) once reviewed local pairs exist. The inspected docs do not demonstrate the Parakeet FastConformer adapter path needed by [#307](https://github.com/sims1253/starling/issues/307); its documented speech examples include Whisper and TTS models. Preserve #307's NeMo-first viability check and source-precision merge/export plan. No personal dataset was inspected or used for this assessment. |
| [Two-model speculative decoding](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf/speculative-decoding) | Documents a smaller draft model with the same tokenizer in llama.cpp. It gives [#314](https://github.com/sims1253/starling/issues/314) an ordinary draft baseline, not a trained EAGLE/DFlash head or evidence that a second resident model beats Starling's free drafts on a phone. |

The first joint experiment remains **codec and export compatibility before
quality claims**: cross-decode a ModelOpt IQ block, then load one ordinary
Unsloth-exported GGUF for an architecture Starling already supports. Pin the
source model, export tool and ggml revisions, compare token IDs and metadata,
and only then evaluate Starling's held-out workload. Dynamic 3.0's published
numbers and the Unsloth QAT phone demo do not replace these checks.
