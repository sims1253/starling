"""Calibration study: is ASR encoder KV low-rank enough to justify compression?

Motivation
----------
The wiki concept `/mnt/z/concepts/kv-cache-spectral-compression.md` measured
that for LLMs only ~3-4% of head dimensions carry meaningful signal in the KEY
vectors (effective dim ~4 out of 128), while VALUES are higher rank (~30-40%).
If ASR *encoder* KV has similar low-rank structure, spectral compression is
worth building; if not, we save the engineering effort.

This is a MEASUREMENT script only. No production code is modified.

Method
------
For each attention layer we register a ``forward_hook`` that captures the raw K
and V tensors (per head, per position) on a calibration set of audio clips. We
then run PCA per (layer, head) via ``torch.linalg.svd`` on the centered
(N_positions, head_dim) matrix and report ``d_eff`` -- the number of principal
components needed to capture 95 / 99 / 99.9% of the variance -- and the ratio
``d_eff / head_dim``, averaged across heads per layer.

  * granite encoder: hook ``layer.attn.to_kv`` (fused K||V Linear, output split
    into K=first inner_dim, V=last inner_dim). 16 layers x 8 heads x hd=128,
    block-local attention over context_size=200.
  * qwen3 encoder:  hook ``layer.self_attn.k_proj`` / ``.v_proj`` separately.
    24 layers x 16 heads x hd=64, windowed attention.

Outputs ``outputs/kv_spectral.json`` and prints a per-layer table + verdict.

Usage
-----
    uv run python benchmarks/bench_kv_spectral.py [--clips N] [--models granite,qwen3]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_PATH = REPO_ROOT / "outputs" / "kv_spectral.json"

# Wiki LLM reference numbers (K ~3%, V ~30-40%) for the verdict.
WIKI_K_RATIO = 0.03
WIKI_V_RATIO = 0.35


# =========================================================================== #
# Calibration audio set
# =========================================================================== #
def gather_calibration_clips(max_clips: int) -> list[tuple[np.ndarray, int, str]]:
    """Return a diverse set of (audio_float32_mono, 16000, name) calibration clips.

    Prefers the leaderboard corpus (7 datasets, real ASR distribution) and
    supplements with the synthetic short/medium/long fixtures for variety.
    """
    clips: list[tuple[np.ndarray, int, str]] = []
    corpus_dir = REPO_ROOT / "tests" / "fixtures" / "leaderboard_corpus"
    if corpus_dir.exists():
        # One clip per (dataset, n8 bucket) spread across datasets.
        seen_datasets: set[str] = set()
        per_dataset_cap = max(1, max_clips // 7)
        for bucket in sorted(corpus_dir.iterdir()):
            if not bucket.is_dir():
                continue
            ds = bucket.name.split("__")[0]
            if ds in seen_datasets:
                continue
            wavs = sorted(bucket.glob("clip_*.wav"))
            # take a small spread of clip indices (not all from the start)
            pick = wavs[:per_dataset_cap]
            for w in pick:
                if len(clips) >= max_clips:
                    break
                a, sr = _read_wav_mono(str(w))
                clips.append((a, sr, f"{bucket.name}/{w.name}"))
            if pick:
                seen_datasets.add(ds)
            if len(clips) >= max_clips:
                break

    # Always include the synthetic fixtures (varied durations) if room remains.
    for name in ("short.wav", "medium.wav", "long.wav"):
        if len(clips) >= max_clips:
            break
        p = REPO_ROOT / "tests" / "fixtures" / name
        if p.exists():
            a, sr = _read_wav_mono(str(p))
            clips.append((a, sr, f"fixtures/{name}"))

    return clips[:max_clips]


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    a, sr = sf.read(path)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return np.ascontiguousarray(a, dtype=np.float32), int(sr)


# =========================================================================== #
# Hooked K/V capture
# =========================================================================== #
class GraniteKVCapture:
    """Capture K and V per layer for the granite CTC conformer encoder.

    Hooks every ``layer.attn.to_kv`` Linear. Its output is the concatenated
    K||V projection of shape (B, T_pad, 2*inner_dim); the encoder splits it
    ``k, v = kv.chunk(2, dim=-1)``. We split the same way and reshape to
    (T, num_heads, head_dim), trimming the context_size padding using the
    true input seq length recorded per forward.
    """

    def __init__(self, encoder) -> None:
        self.encoder = encoder
        cfg = encoder.config
        self.num_heads = int(cfg.num_heads)
        self.head_dim = int(cfg.dim_head)
        self.inner_dim = self.num_heads * self.head_dim
        self.num_layers = int(encoder.num_layers)
        # kv_per_layer[layer] -> list of (K (T, nh, hd), V (T, nh, hd)) fp32 cpu
        self.kv_per_layer: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_layers)
        ]
        self._cur_layer: int | None = None
        self._cur_T: int | None = None
        self.handles: list[Any] = []

    def _install(self) -> None:
        for idx, layer in enumerate(self.encoder.layers):
            to_kv = layer.attn.to_kv

            def make_hook(layer_idx):
                def hook(_mod, _inp, out):
                    # out: (B, T_pad, 2*inner_dim). Split K||V.
                    T_true = self._cur_T
                    o = out.detach().to(torch.float32)
                    if T_true is not None and o.shape[1] > T_true:
                        o = o[:, :T_true, :]
                    k = o[..., : self.inner_dim]
                    v = o[..., self.inner_dim :]
                    B = o.shape[0]
                    # (B, T, inner_dim) -> (B, T, nh, hd) -> (B*T, nh, hd)
                    k = k.reshape(B, -1, self.num_heads, self.head_dim)
                    v = v.reshape(B, -1, self.num_heads, self.head_dim)
                    self.kv_per_layer[layer_idx].append(
                        (k.reshape(-1, self.num_heads, self.head_dim).cpu(),
                         v.reshape(-1, self.num_heads, self.head_dim).cpu())
                    )
                return hook

            self.handles.append(to_kv.register_forward_hook(make_hook(idx)))

    def attach(self) -> None:
        self._install()

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def record_seq_len(self, T: int) -> None:
        """Tell the capture the true (unpadded) input seq length for this fwd."""
        self._cur_T = int(T)

    def stacked(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per layer: (K (N, nh, hd), V (N, nh, hd)) concatenated over samples."""
        out = []
        for layer_kvs in self.kv_per_layer:
            ks = torch.cat([k for k, _ in layer_kvs], dim=0)
            vs = torch.cat([v for _, v in layer_kvs], dim=0)
            out.append((ks, vs))
        return out


class Qwen3KVCapture:
    """Capture K and V per layer for the Qwen3-ASR windowed encoder.

    Hooks every ``layer.self_attn.k_proj`` / ``.v_proj`` Linear. Their output is
    the per-head projection of shape (P_packed, inner_dim) where P is the
    valid-only packed sequence (padding already removed by the encoder's
    index_select). Reshape to (P, num_heads, head_dim).
    """

    def __init__(self, encoder) -> None:
        self.encoder = encoder
        # all layers share num_heads / head_dim via the attention module.
        a0 = encoder.layers[0].self_attn
        self.num_heads = int(a0.num_heads)
        self.head_dim = int(a0.head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.num_layers = len(encoder.layers)
        self.kv_per_layer: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_layers)
        ]
        self.handles: list[Any] = []

    def _install(self) -> None:
        for idx, layer in enumerate(self.encoder.layers):
            attn = layer.self_attn

            def make_k_hook(kl):
                def hook(_mod, _inp, out):
                    o = out.detach().to(torch.float32)
                    P = o.shape[0]
                    kl.append(o.reshape(P, self.num_heads, self.head_dim).cpu())
                return hook

            def make_v_hook(vl):
                def hook(_mod, _inp, out):
                    o = out.detach().to(torch.float32)
                    P = o.shape[0]
                    vl.append(o.reshape(P, self.num_heads, self.head_dim).cpu())
                return hook

            k_list: list[torch.Tensor] = []
            v_list: list[torch.Tensor] = []
            self._k_lists.append(k_list)
            self._v_lists.append(v_list)
            self.handles.append(attn.k_proj.register_forward_hook(make_k_hook(k_list)))
            self.handles.append(attn.v_proj.register_forward_hook(make_v_hook(v_list)))

    def attach(self) -> None:
        self._k_lists: list[list[torch.Tensor]] = []
        self._v_lists: list[list[torch.Tensor]] = []
        self._install()

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for k_list, v_list in zip(self._k_lists, self._v_lists):
            ks = torch.cat(k_list, dim=0)
            vs = torch.cat(v_list, dim=0)
            out.append((ks, vs))
        return out


def o_P(out: torch.Tensor) -> int:
    return out.shape[0]


# =========================================================================== #
# PCA via SVD per (layer, head)
# =========================================================================== #
def effective_dim(singular_values: torch.Tensor, thresholds: tuple[float, ...]) -> dict[float, int]:
    """Number of components to reach each cumulative-variance threshold.

    singular_values: 1-D tensor (>=0), one per principal component.
    Returns {threshold: d_eff}.
    """
    var = (singular_values ** 2).to(torch.float64)
    total = var.sum()
    if total <= 0:
        return {t: 1 for t in thresholds}
    cum = torch.cumsum(var, dim=0) / total
    out = {}
    for t in thresholds:
        # first index where cumulative >= t (1-based count)
        idx = int(torch.searchsorted(cum, torch.tensor(float(t))).item())
        idx = min(max(idx + 1, 1), int(cum.numel()))
        out[t] = idx
    return out


def pca_layer(mat: torch.Tensor, head_dim: int, thresholds: tuple[float, ...]) -> dict[str, Any]:
    """Per-layer PCA across heads.

    mat: (N, num_heads, head_dim) fp32. Returns per-threshold mean d_eff,
    mean d_eff/head_dim, and the head-level d_eff arrays.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_heads = mat.shape[1]
    per_head_deff = {t: np.zeros(num_heads, dtype=np.int32) for t in thresholds}
    sv_list: list[np.ndarray] = []
    for h in range(num_heads):
        x = mat[:, h, :].to(device=device, dtype=torch.float32)
        # Center.
        mean = x.mean(dim=0, keepdim=True)
        xc = x - mean
        # SVD on centered matrix: singular values are sqrt of eigenvalues of cov.
        try:
            sv = torch.linalg.svdvals(xc)
        except RuntimeError:
            # fallback to full svd if svdvals hits a driver issue
            sv = torch.linalg.svd(xc, full_matrices=False)[1]
        sv = sv.cpu()
        sv_list.append(sv.double().numpy())
        deff = effective_dim(sv, thresholds)
        for t in thresholds:
            per_head_deff[t][h] = deff[t]
    result: dict[str, Any] = {}
    for t in thresholds:
        result[f"d_eff@{t}"] = per_head_deff[t].tolist()
        result[f"d_eff_mean@{t}"] = float(per_head_deff[t].mean())
        result[f"d_eff_ratio_mean@{t}"] = float(per_head_deff[t].mean() / head_dim)
    return result


# =========================================================================== #
# Per-model measurement
# =========================================================================== #
@torch.inference_mode()
def measure_granite(clips: list[tuple[np.ndarray, int, str]]) -> dict[str, Any]:
    from starling.granite.audio import build_inputs
    from starling.granite.loader import get_components, load_model_and_processor

    print("\n=== GRANITE encoder ===", flush=True)
    print("loading granite model ...", flush=True)
    model, processor = load_model_and_processor()
    comps = get_components(model)
    encoder = comps["encoder"]
    dtype = model.dtype

    cap = GraniteKVCapture(encoder)
    cap.attach()
    try:
        for i, (wav, sr, name) in enumerate(clips):
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            inputs = build_inputs(processor, wav_t)
            feats = inputs["input_features"].to(dtype).cuda()
            T = int(feats.shape[1])
            cap.record_seq_len(T)
            _ = encoder(feats, return_dict=True)
            print(f"  [{i+1}/{len(clips)}] {name}: T={T}", flush=True)
    finally:
        cap.detach()

    stacked = cap.stacked()
    head_dim = cap.head_dim
    num_heads = cap.num_heads
    thresholds = (0.95, 0.99, 0.999)
    layers_out = []
    for li, (k_mat, v_mat) in enumerate(stacked):
        k_res = pca_layer(k_mat, head_dim, thresholds)
        v_res = pca_layer(v_mat, head_dim, thresholds)
        print(f"  layer {li:2d}: K d_eff@0.99={k_res['d_eff_mean@0.99']:.2f} "
              f"({k_res['d_eff_ratio_mean@0.99']*100:.1f}% of {head_dim})  "
              f"V d_eff@0.99={v_res['d_eff_mean@0.99']:.2f} "
              f"({v_res['d_eff_ratio_mean@0.99']*100:.1f}%)", flush=True)
        layers_out.append({
            "layer": li,
            "n_positions": int(k_mat.shape[0]),
            "K": k_res,
            "V": v_res,
        })

    # Free model before loading the next one.
    del model, processor, encoder, comps
    gc.collect()
    torch.cuda.empty_cache()

    summary = _summarise(layers_out, head_dim, thresholds)
    summary["model"] = "granite-speech-4.1-2b"
    summary["num_layers"] = len(layers_out)
    summary["num_heads"] = num_heads
    summary["head_dim"] = head_dim
    summary["layers"] = layers_out
    summary["block_attention_context"] = 200
    return summary


@torch.inference_mode()
def measure_qwen3(clips: list[tuple[np.ndarray, int, str]]) -> dict[str, Any]:
    from starling.qwen3.audio import build_inputs
    from starling.qwen3.loader import get_components, load_model_and_processor

    print("\n=== QWEN3 encoder ===", flush=True)
    print("loading qwen3 model ...", flush=True)
    model, processor = load_model_and_processor()
    comps = get_components(model)
    encoder = comps["encoder"]
    dtype = model.dtype

    cap = Qwen3KVCapture(encoder)
    cap.attach()
    try:
        for i, (wav, sr, name) in enumerate(clips):
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            inputs = build_inputs(processor, wav_t, sr=sr)
            feats = inputs["input_features"].to(dtype).cuda()
            mask = inputs.get("input_features_mask")
            if mask is not None:
                mask = mask.cuda()
            _ = encoder(
                input_features=feats,
                input_features_mask=mask,
                return_dict=True,
            )
            print(f"  [{i+1}/{len(clips)}] {name}: "
                  f"feat={tuple(feats.shape)}", flush=True)
    finally:
        cap.detach()

    stacked = cap.stacked()
    head_dim = cap.head_dim
    num_heads = cap.num_heads
    thresholds = (0.95, 0.99, 0.999)
    layers_out = []
    for li, (k_mat, v_mat) in enumerate(stacked):
        k_res = pca_layer(k_mat, head_dim, thresholds)
        v_res = pca_layer(v_mat, head_dim, thresholds)
        print(f"  layer {li:2d}: K d_eff@0.99={k_res['d_eff_mean@0.99']:.2f} "
              f"({k_res['d_eff_ratio_mean@0.99']*100:.1f}% of {head_dim})  "
              f"V d_eff@0.99={v_res['d_eff_mean@0.99']:.2f} "
              f"({v_res['d_eff_ratio_mean@0.99']*100:.1f}%)", flush=True)
        layers_out.append({
            "layer": li,
            "n_positions": int(k_mat.shape[0]),
            "K": k_res,
            "V": v_res,
        })

    del model, processor, encoder, comps
    gc.collect()
    torch.cuda.empty_cache()

    summary = _summarise(layers_out, head_dim, thresholds)
    summary["model"] = "qwen3-asr-1.7b"
    summary["num_layers"] = len(layers_out)
    summary["num_heads"] = num_heads
    summary["head_dim"] = head_dim
    summary["layers"] = layers_out
    summary["windowed_attention_n_window"] = 50
    return summary


def _summarise(layers_out: list[dict], head_dim: int, thresholds: tuple[float, ...]) -> dict[str, Any]:
    s: dict[str, Any] = {}
    for t in thresholds:
        k_means = [l["K"][f"d_eff_ratio_mean@{t}"] for l in layers_out]
        v_means = [l["V"][f"d_eff_ratio_mean@{t}"] for l in layers_out]
        s[f"K_ratio_overall_mean@{t}"] = float(np.mean(k_means))
        s[f"V_ratio_overall_mean@{t}"] = float(np.mean(v_means))
        s[f"K_ratio_overall_min@{t}"] = float(np.min(k_means))
        s[f"K_ratio_overall_max@{t}"] = float(np.max(k_means))
        s[f"V_ratio_overall_min@{t}"] = float(np.min(v_means))
        s[f"V_ratio_overall_max@{t}"] = float(np.max(v_means))
    return s


# =========================================================================== #
# Verdict
# =========================================================================== #
def verdict_line(ratio: float, ref: float, kind: str) -> str:
    """Compare a measured K/V ratio to the wiki LLM reference."""
    if ratio <= ref * 1.5:
        verdict = "LOW-RANK (compression viable)"
    elif ratio <= ref * 4.0:
        verdict = "BORDERLINE (marginal compression gain)"
    else:
        verdict = "HIGH-RANK (compression NOT viable)"
    return f"{kind}: {ratio*100:.1f}% vs wiki {ref*100:.1f}% -> {verdict}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=24,
                    help="number of calibration clips (default 24)")
    ap.add_argument("--models", type=str, default="granite,qwen3",
                    help="comma list: granite,qwen3 (default both)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    clips = gather_calibration_clips(args.clips)
    print(f"calibration set: {len(clips)} clips", flush=True)
    for _, _, name in clips:
        print(f"  - {name}", flush=True)
    if not clips:
        print("ERROR: no calibration clips found", file=sys.stderr)
        return 1

    thresholds = (0.95, 0.99, 0.999)
    results: dict[str, Any] = {
        "wiki_llm_reference": {"K_ratio": WIKI_K_RATIO, "V_ratio": WIKI_V_RATIO},
        "thresholds": list(thresholds),
        "n_clips": len(clips),
        "clip_names": [name for _, _, name in clips],
    }

    if "granite" in models:
        results["granite"] = measure_granite(clips)
    if "qwen3" in models:
        results["qwen3"] = measure_qwen3(clips)

    # ---- verdict ----
    print("\n" + "=" * 72, flush=True)
    print("VERDICT (d_eff/head_dim @ 99% variance, averaged across heads & layers)", flush=True)
    print("-" * 72, flush=True)
    print(f"wiki LLM reference: K={WIKI_K_RATIO*100:.1f}%  V={WIKI_V_RATIO*100:.1f}%", flush=True)
    verdicts: dict[str, Any] = {}
    for m in ("granite", "qwen3"):
        if m not in results:
            continue
        kr = results[m]["K_ratio_overall_mean@0.99"]
        vr = results[m]["V_ratio_overall_mean@0.99"]
        print(f"\n[{m}] {results[m]['num_layers']} layers x "
              f"{results[m]['num_heads']} heads x hd={results[m]['head_dim']}", flush=True)
        print("  " + verdict_line(kr, WIKI_K_RATIO, "K"), flush=True)
        print("  " + verdict_line(vr, WIKI_V_RATIO, "V"), flush=True)
        # Overall build recommendation: K must be low-rank to be worth it.
        if kr <= WIKI_K_RATIO * 1.5:
            build = "BUILD (K is as low-rank as the LLM)"
        elif kr <= WIKI_K_RATIO * 4.0:
            build = "BORDERLINE -- only K compression, marginal; re-measure with more clips"
        else:
            build = "NO-BUILD (K is NOT low-rank; compression would lose signal)"
        print(f"  recommendation: {build}", flush=True)
        verdicts[m] = {
            "K_ratio@0.99": kr,
            "V_ratio@0.99": vr,
            "build_recommendation": build,
        }
    results["verdict"] = verdicts
    print("=" * 72, flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nresults written to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
