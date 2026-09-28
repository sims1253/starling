# NVIDIA Model Optimizer: ideas Starling can use beyond CUDA

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
