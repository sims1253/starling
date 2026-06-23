"""CUDA-graph-captured Conformer encoder for nvidia/parakeet-tdt-0.6b-v3.

Same I/O and byte-exact output as the stock
``model.get_audio_features(input_features, attention_mask)`` call, but captured
into a single ``torch.cuda.CUDAGraph`` and served by ``graph.replay()``.

Why graph the encoder
---------------------
The stock 24-layer Conformer encoder is ~43 ms at batch=8 medium but only ~10%
GPU-busy: it is launch-overhead bound (hundreds of tiny per-layer kernels with
sequential dependencies, each paying ~us of host launch latency). A CUDA graph
collapses those launches into one replay, measured at ~32 ms (1.36x, byte-exact,
max_diff 0.0). See ``outputs/parakeet/pipeline_bench_graphed.json`` for the
integrated RTF impact.

Static-buffer strategy
----------------------
The graph reads two static buffers (allocated once per shape, tagged with
``torch._dynamo.mark_static_address``) and writes its outputs at fixed addresses:

* ``static_inp``  (B, T_mel, 128) -- the bf16 mel features
* ``static_mask`` (B, T_mel)       -- the bool attention mask

On each call, the new input data is copied into the static buffers and the graph
is replayed. The captured encoder output (a ``ParakeetEncoderModelOutput``) is
stored by reference; ``__call__`` returns a *fresh* dataclass whose
``pooler_output`` and ``attention_mask`` are clones of the static output tensors,
so callers cannot mutate the captured state (a subsequent replay would otherwise
overwrite it). The decode step consumes ``pooler_output.contiguous()`` and
``attention_mask.sum(-1)`` -- neither of which mutates -- so the clone is
defensive but never required by the integrated pipeline.

Shape caching
-------------
One graph per ``(B, T_mel)`` shape is cached in a dict, so the one-off capture
cost is amortised across all calls of the same shape (the production-realistic
shape: capture once, encode many). A first call for a new shape pays capture;
every later same-shape call is a dict lookup + buffer copy + replay.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from torch._dynamo import mark_static_address as _mark_static
except Exception:  # pragma: no cover - older torch
    def _mark_static(t):  # type: ignore[misc]
        return t


def _mark_many(tensors) -> None:
    for t in tensors:
        try:
            _mark_static(t)
        except Exception:
            pass


# ---------------------------------------------------------------------- #
# BatchNorm1d -> depthwise_conv folding (exact affine fold for inference)
# ---------------------------------------------------------------------- #
def fold_conformer_batchnorm(model) -> Tuple[List[dict], dict]:
    """Fold every Conformer conv-module BatchNorm1d into its depthwise_conv.

    For inference (``model.eval()``) BatchNorm1d is a deterministic per-channel
    affine transform applied to the depthwise conv output::

        BN: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias

    which folds into the preceding conv as::

        scale   = bn.weight / sqrt(running_var + eps)            # (C,)
        W'      = W * scale[:, None, None]                        # (C, 1, K)
        b'      = (b - running_mean) * scale + bn.bias            # (C,)

    In exact arithmetic this is identical (no approximation). In bf16 there can
    be sub-ULP rounding differences because the scale is baked into the weight
    rather than applied to the conv output. The *purpose* of folding is to
    remove the BatchNorm1d op entirely so ``torch.compile`` cannot amplify bf16
    rounding differences ~316x via the tiny ``running_var`` (granite sibling:
    ``running_var ~ 4e-10`` -> ``1 / sqrt(running_var + eps) ~ 316``).

    Mutates the conv modules in-place on ``model``:
      * ``depthwise_conv.weight`` / ``.bias`` <- folded params (new tensors).
      * ``conv.norm`` <- ``nn.Identity()`` (the BatchNorm1d is removed).

    Args:
        model: a loaded ``ParakeetForTDT`` (its ``model.encoder.layers[*].conv``
            modules are mutated). Pass a model instance dedicated to compiled
            mode; the pipeline loads a fresh model per ``encoder_mode``, so the
            graphed/eager stock path is preserved for A/B.

    Returns:
        ``(originals, bn_stats)`` where ``originals`` is a list of per-layer
        param dicts (so the fold can be reverted via
        :func:`restore_conformer_batchnorm`) and ``bn_stats`` reports the
        ``running_var`` distribution (to answer "is parakeet's BN as unstable as
        granite's?").
    """
    layers = model.encoder.layers
    originals: List[dict] = []
    all_rvar: List[torch.Tensor] = []
    for layer in layers:
        conv = layer.conv                 # ParakeetEncoderConvolutionModule
        bn = conv.norm                    # nn.BatchNorm1d
        dw = conv.depthwise_conv          # nn.Conv1d (groups == channels)

        running_mean = bn.running_mean.to(torch.float32)
        running_var = bn.running_var.to(torch.float32)
        bn_weight = bn.weight.to(torch.float32)
        bn_bias = bn.bias.to(torch.float32)
        eps = float(bn.eps)

        scale = bn_weight / torch.sqrt(running_var + eps)   # (C,)

        w = dw.weight.to(torch.float32)                      # (C, 1, K)
        new_w = w * scale.view(-1, 1, 1)
        if dw.bias is not None:
            b = dw.bias.to(torch.float32)
            new_b = (b - running_mean) * scale + bn_bias
        else:
            new_b = bn_bias - running_mean * scale

        originals.append({
            "layer": int(layer.self_attn.layer_idx) if hasattr(layer.self_attn, "layer_idx") else len(originals),
            "depthwise_weight": dw.weight.detach().clone(),
            "depthwise_bias": dw.bias.detach().clone() if dw.bias is not None else None,
            "bn_weight": bn.weight.detach().clone(),
            "bn_bias": bn.bias.detach().clone(),
            "bn_running_mean": bn.running_mean.detach().clone(),
            "bn_running_var": bn.running_var.detach().clone(),
            "bn_eps": eps,
        })
        all_rvar.append(running_var)

        dw_dtype = dw.weight.dtype
        dw.weight = nn.Parameter(new_w.to(dw_dtype).contiguous())
        dw.bias = nn.Parameter(new_b.to(dw_dtype).contiguous())
        conv.norm = nn.Identity()

    cat_rvar = torch.cat(all_rvar)
    bn_stats = {
        "n_layers": len(originals),
        "n_channels_total": int(cat_rvar.numel()),
        "running_var_min": float(cat_rvar.min().item()),
        "running_var_mean": float(cat_rvar.mean().item()),
        "running_var_max": float(cat_rvar.max().item()),
        "eps": float(originals[0]["bn_eps"]) if originals else None,
        "max_inv_std": float((1.0 / torch.sqrt(cat_rvar + originals[0]["bn_eps"])).max().item()) if originals else None,
    }
    return originals, bn_stats


def restore_conformer_batchnorm(model, originals: List[dict]) -> None:
    """Revert a prior :func:`fold_conformer_batchnorm` call (restore BN)."""
    layers = model.encoder.layers
    for layer, orig in zip(layers, originals):
        conv = layer.conv
        dw = conv.depthwise_conv
        dw.weight = nn.Parameter(orig["depthwise_weight"].clone())
        if orig["depthwise_bias"] is not None:
            dw.bias = nn.Parameter(orig["depthwise_bias"].clone())
        bn = nn.BatchNorm1d(
            orig["bn_running_mean"].shape[0],
            eps=orig["bn_eps"],
            affine=True,
            track_running_stats=True,
        ).to(dtype=orig["bn_weight"].dtype, device=orig["bn_weight"].device)
        bn.weight.data.copy_(orig["bn_weight"])
        bn.bias.data.copy_(orig["bn_bias"])
        bn.running_mean.data.copy_(orig["bn_running_mean"])
        bn.running_var.data.copy_(orig["bn_running_var"])
        conv.norm = bn


class GraphedEncoder:
    """Capture ``model.get_audio_features`` into one CUDA graph; encode many inputs.

    The graph is shape-specific (``B``, ``T_mel`` fixed at capture time); one
    :class:`GraphedEncoder` caches one captured graph per ``(B, T_mel)`` shape so
    the capture cost is amortised across same-shape calls.

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16).
        warmup_iters: side-stream warmup iterations before graph capture
            (stabilises cudnn/cublas autotune for the conv subsampling + the
            24 conformer layers).
    """

    def __init__(self, model, *, warmup_iters: int = 3) -> None:
        self.model = model
        self.warmup_iters = int(warmup_iters)
        # (B, T_mel) -> bundle(dict): static_inp, static_mask, static_out, graph
        self._graphs: Dict[Tuple[int, int], dict] = {}

    # ------------------------------------------------------------------ #
    # shape-keyed capture (amortise capture across same-shape calls)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _capture(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """Allocate static buffers for this ``(B, T_mel)`` shape and capture."""
        B, T_mel = int(input_features.shape[0]), int(input_features.shape[1])

        # static input buffers (fixed GPU addresses for the graph's lifetime)
        static_inp = torch.empty_like(input_features)
        static_mask = torch.empty_like(attention_mask)
        _mark_many([static_inp, static_mask])
        static_inp.copy_(input_features)
        static_mask.copy_(attention_mask)

        # warmup on a side stream (stabilises cudnn/cublas autotune before
        # capture; mirrors the GraphedDecoder capture pattern in decode_mega).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self.model.get_audio_features(
                    input_features=static_inp, attention_mask=static_mask
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # capture the encoder forward into a CUDAGraph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.model.get_audio_features(
                input_features=static_inp, attention_mask=static_mask
            )

        bundle = {
            "B": B,
            "T_mel": T_mel,
            "static_inp": static_inp,
            "static_mask": static_mask,
            "static_out": static_out,
            "graph": graph,
        }
        self._graphs[(B, T_mel)] = bundle
        return bundle

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Run the captured encoder for this batch; return a fresh output.

        Args:
            input_features: ``(B, T_mel, 128)`` bf16 mel features on cuda.
            attention_mask: ``(B, T_mel)`` bool attention mask on cuda.

        Returns:
            A new ``ParakeetEncoderModelOutput``-typed object whose
            ``pooler_output`` (B, T_enc, 640) and ``attention_mask`` (B, T_enc)
            are clones of the static captured output, so the caller cannot
            mutate the captured state. ``last_hidden_state`` etc. are left at
            their dataclass defaults (the integrated pipeline only reads
            ``pooler_output`` and ``attention_mask``).

        Byte-exact with the eager ``model.get_audio_features`` path (max_diff
        0.0); the graph only removes host launch overhead.
        """
        B, T_mel = int(input_features.shape[0]), int(input_features.shape[1])
        key = (B, T_mel)
        bundle = self._graphs.get(key)
        if bundle is None:
            bundle = self._capture(input_features, attention_mask)

        # copy the new input data into the static buffers and replay
        bundle["static_inp"].copy_(input_features)
        bundle["static_mask"].copy_(attention_mask)
        bundle["graph"].replay()

        out = bundle["static_out"]
        out_cls = type(out)
        # fresh dataclass; clone pooler_output + attention_mask so callers can't
        # mutate the captured state that the next replay would overwrite.
        return out_cls(
            pooler_output=out.pooler_output.clone(),
            attention_mask=out.attention_mask.clone(),
        )


class CompiledEncoder:
    """torch.compile + (optional) BN-fold encoder for the parakeet Conformer.

    A fast, *non-guaranteed-byte-exact* alternative to :class:`GraphedEncoder`.
    The 24-block Conformer spends ~36ms of its ~32ms-at-B8 budget on
    elementwise + memop glue (per profiler: matmul 58% / elementwise 15% / conv
    12% / memops 5% / ...); ``torch.compile(mode="reduce-overhead")`` fuses that
    glue and captures the whole encoder into cudagraph trees.

    The catch is the conv module's BatchNorm1d: ``running_var`` is tiny, so
    ``1 / sqrt(running_var + eps)`` is a large per-channel gain that amplifies
    any bf16 rounding difference (which torch.compile introduces by reordering
    reductions / upcasting attention). To remove that amplification we fold the
    BatchNorm1d into the preceding depthwise conv (see
    :func:`fold_conformer_batchnorm`) -- a deterministic affine fold that
    eliminates the BN op entirely.

    Args:
        model: a loaded ``ParakeetForTDT`` on cuda (eval mode, bf16). This
            encoder MUTATES the model's conv modules when ``fold_bn=True`` --
            pass a model instance dedicated to compiled mode (the pipeline
            loads a fresh model per ``encoder_mode``, so the graphed/eager stock
            path is preserved for A/B).
        fold_bn: if True (default), fold BatchNorm1d into depthwise_conv before
            compiling. Set False to test raw ``torch.compile`` (Approach 1).
        compile_mode: ``torch.compile`` mode (default ``"reduce-overhead"``).
        warmup_iters: per-shape warmup iterations; the first call(s) for a new
            shape pay torch.compile tracing + autotune + cudagraph capture.

    Attributes:
        bn_stats: the ``running_var`` distribution (populated when ``fold_bn``);
            ``None`` otherwise. Reported so the caller can compare parakeet's BN
            stability against granite's (``running_var ~ 4e-10``).
        bn_originals: per-layer original params; the fold is revertable via
            :func:`restore_conformer_batchnorm`.
    """

    def __init__(
        self,
        model,
        *,
        fold_bn: bool = True,
        compile_mode: str = "reduce-overhead",
        warmup_iters: int = 3,
    ) -> None:
        self.model = model
        self.fold_bn = bool(fold_bn)
        self.compile_mode = str(compile_mode)
        self.warmup_iters = int(warmup_iters)
        self.bn_stats: dict | None = None
        self.bn_originals: List[dict] | None = None

        if self.fold_bn:
            self.bn_originals, self.bn_stats = fold_conformer_batchnorm(model)

        # compile a closure with explicit tensor args (cleaner than compiling the
        # bound method `model.get_audio_features`, which recompiles per instance).
        def _encode(input_features, attention_mask):
            return model.get_audio_features(
                input_features=input_features, attention_mask=attention_mask
            )

        self._compiled = torch.compile(_encode, mode=self.compile_mode)
        # warmup state per (B, T_mel) shape: True once warmup is done for it
        self._warmed: Dict[Tuple[int, int], bool] = {}

    # ------------------------------------------------------------------ #
    # per-shape compile warmup (tracing + autotune + cudagraph capture)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _warmup(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> None:
        B, T_mel = int(input_features.shape[0]), int(input_features.shape[1])
        key = (B, T_mel)
        if self._warmed.get(key):
            return
        for _ in range(self.warmup_iters):
            self._compiled(input_features, attention_mask)
        torch.cuda.synchronize()
        self._warmed[key] = True

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Run the compiled encoder for this batch; return a fresh output.

        Same return contract as :class:`GraphedEncoder`: a
        ``ParakeetEncoderModelOutput``-typed object whose ``pooler_output``
        ``(B, T_enc, 640)`` and ``attention_mask`` ``(B, T_enc)`` are clones of
        the compiled output (the compiled fn may reuse output buffers under
        cudagraph trees, so callers must not retain the raw output across
        calls).

        Accuracy: NOT guaranteed byte-exact with the eager / graphed path.
        BN folding removes the dominant amplification, but torch.compile can
        still reorder reductions and upcast attention, so expect sub-ULP bf16
        differences. The integrated pipeline's correctness gate is text-level
        transcript match vs the oracle (see ``test_parakeet_pipeline.py``).
        """
        self._warmup(input_features, attention_mask)
        out = self._compiled(input_features, attention_mask)
        out_cls = type(out)
        # clone pooler_output + attention_mask: reduce-overhead cudagraph trees
        # may reuse the output's storage on the next call.
        return out_cls(
            pooler_output=out.pooler_output.clone(),
            attention_mask=out.attention_mask.clone(),
        )

