"""Calibration study: is ASR encoder KV low-rank enough to justify compression?

Motivation
----------
Issue #59 asks whether the ASR encoder's K/V projections are compressible.
That must be tested on held-out audio before changing an encoder.

This is a MEASUREMENT script only. No production code is modified.

Method
------
For each attention layer we capture raw K and V tensors. We fit a PCA basis on
even-indexed clips and report reconstruction error on odd-indexed clips, which
were not used to fit the basis. In-sample explained variance alone cannot tell
us whether a compressed cache preserves other utterances.

  * granite encoder: hook ``layer.attn.to_kv`` (fused K||V Linear, output split
    into K=first inner_dim, V=last inner_dim). 16 layers x 8 heads x hd=128,
    block-local attention over context_size=200.
  * qwen3 encoder:  hook ``layer.self_attn.k_proj`` / ``.v_proj`` separately.
    24 layers x 16 heads x hd=64, windowed attention.

Outputs ``outputs/kv_spectral.json`` and prints a per-layer table. The result
is a screening measurement, not a latency, memory, or WER claim.

Usage
-----
    uv run python benchmarks/bench_kv_spectral.py [--clips N] [--models granite,qwen3]
    uv run python benchmarks/bench_kv_spectral.py --audio clip1.wav clip2.wav ...
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_PATH = REPO_ROOT / "outputs" / "kv_spectral.json"

# =========================================================================== #
# Calibration audio set
# =========================================================================== #
def gather_calibration_clips(max_clips: int,
                             audio_paths: list[Path] | None = None) -> list[tuple[np.ndarray, int, str]]:
    """Return a diverse set of (audio_float32_mono, 16000, name) calibration clips.

    Prefers the leaderboard corpus (7 datasets, real ASR distribution) and
    supplements with the synthetic short/medium/long fixtures for variety.
    """
    clips: list[tuple[np.ndarray, int, str]] = []
    if audio_paths:
        for path in audio_paths[:max_clips]:
            a, sr = _read_wav_mono(str(path))
            clips.append((a, sr, path.name))
        return clips
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
    if sr != 16000:
        raise ValueError(f"calibration WAV {path} has sample rate {sr}; expected 16000 Hz")
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

    def stacked(self, clip_indices: range) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per layer: K and V concatenated over selected clips."""
        out = []
        for layer_kvs in self.kv_per_layer:
            ks = torch.cat([layer_kvs[i][0] for i in clip_indices], dim=0)
            vs = torch.cat([layer_kvs[i][1] for i in clip_indices], dim=0)
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

    def stacked(self, clip_indices: range) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out = []
        for k_list, v_list in zip(self._k_lists, self._v_lists):
            ks = torch.cat([k_list[i] for i in clip_indices], dim=0)
            vs = torch.cat([v_list[i] for i in clip_indices], dim=0)
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
        idx = int(torch.searchsorted(cum, cum.new_tensor(float(t))).item())
        idx = min(max(idx + 1, 1), int(cum.numel()))
        out[t] = idx
    return out


def pca_layer(train: torch.Tensor, held_out: torch.Tensor,
              thresholds: tuple[float, ...]) -> dict[str, Any]:
    """Fit each head on training clips; measure relative squared error elsewhere.

    Error is normalized by held-out energy around the training mean. The
    orthogonal PCA projection makes this ratio lie in [0, 1], apart from
    floating-point roundoff; a distribution shift can move it toward one.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_heads, head_dim = train.shape[1:]
    if held_out.shape[1:] != train.shape[1:] or not len(train) or not len(held_out):
        raise ValueError("train and held-out K/V must have matching nonempty heads")
    per_head_deff = {t: np.zeros(num_heads, dtype=np.int32) for t in thresholds}
    ranks = sorted({max(1, head_dim // 4), max(1, head_dim // 2), head_dim})
    errors = {r: [] for r in ranks}
    threshold_errors = {t: [] for t in thresholds}
    for h in range(num_heads):
        x = train[:, h, :].to(device=device, dtype=torch.float32)
        y = held_out[:, h, :].to(device=device, dtype=torch.float32)
        mean = x.mean(dim=0, keepdim=True)
        xc = x - mean
        # Keep the small [head_dim, head_dim] basis. A full SVD on long audio
        # would otherwise materialize an enormous [positions, positions] U.
        _, sv, vh = torch.linalg.svd(xc, full_matrices=x.shape[0] < head_dim)
        deff = effective_dim(sv, thresholds)
        for t in thresholds:
            per_head_deff[t][h] = deff[t]
        yc = y - mean
        energy = float(yc.square().sum())
        def relative_error(rank: int) -> float:
            basis = vh[:rank]
            residual = yc - (yc @ basis.T) @ basis
            return float(residual.square().sum()) / energy if energy else 0.0
        for r in ranks:
            errors[r].append(relative_error(r))
        for t in thresholds:
            threshold_errors[t].append(relative_error(deff[t]))
    result: dict[str, Any] = {}
    for t in thresholds:
        result[f"d_eff@{t}"] = per_head_deff[t].tolist()
        result[f"d_eff_mean@{t}"] = float(per_head_deff[t].mean())
        result[f"d_eff_ratio_mean@{t}"] = float(per_head_deff[t].mean() / head_dim)
        result[f"held_out_relative_mse@d_eff_{t}"] = threshold_errors[t]
        result[f"held_out_relative_mse_mean@d_eff_{t}"] = float(np.mean(threshold_errors[t]))
    result["held_out_relative_mse_by_rank"] = {str(r): v for r, v in errors.items()}
    result["held_out_relative_mse_mean_by_rank"] = {
        str(r): float(np.mean(v)) for r, v in errors.items()
    }
    return result


# =========================================================================== #
# Per-model measurement
# =========================================================================== #
@torch.inference_mode()
def measure_granite(clips: list[tuple[np.ndarray, int, str]],
                    snapshot: Path | None = None) -> dict[str, Any]:
    for _, sr, name in clips:
        if sr != 16000:
            raise ValueError(f"Granite clip {name} has sample rate {sr}; expected 16000 Hz")
    from starling.granite.audio import build_inputs
    from starling.granite.loader import get_components, load_model_and_processor

    print("\n=== GRANITE encoder ===", flush=True)
    print("loading granite model ...", flush=True)
    if snapshot is None:
        model, processor = load_model_and_processor()
    else:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            snapshot, device_map="cuda", dtype=torch.bfloat16,
            attn_implementation="eager").eval()
        processor = AutoProcessor.from_pretrained(snapshot)
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

    train = cap.stacked(range(0, len(clips), 2))
    held_out = cap.stacked(range(1, len(clips), 2))
    head_dim = cap.head_dim
    num_heads = cap.num_heads
    thresholds = (0.95, 0.99, 0.999)
    layers_out = []
    for li, ((k_mat, v_mat), (k_test, v_test)) in enumerate(zip(train, held_out)):
        k_res = pca_layer(k_mat, k_test, thresholds)
        v_res = pca_layer(v_mat, v_test, thresholds)
        print(f"  layer {li:2d}: K d_eff@0.99={k_res['d_eff_mean@0.99']:.2f} "
              f"({k_res['d_eff_ratio_mean@0.99']*100:.1f}% of {head_dim})  "
              f"held-out error={k_res['held_out_relative_mse_mean@d_eff_0.99']:.3f}; "
              f"V d_eff@0.99={v_res['d_eff_mean@0.99']:.2f} "
              f"held-out error={v_res['held_out_relative_mse_mean@d_eff_0.99']:.3f}", flush=True)
        layers_out.append({
            "layer": li,
            "n_train_positions": int(k_mat.shape[0]),
            "n_held_out_positions": int(k_test.shape[0]),
            "K": k_res,
            "V": v_res,
        })

    # Free model before loading the next one.
    del model, processor, encoder, comps
    gc.collect()
    torch.cuda.empty_cache()

    summary = _summarise(layers_out, head_dim, thresholds)
    summary["model"] = "granite-speech-4.1-2b"
    summary["source_snapshot"] = snapshot.name if snapshot is not None else "default HF revision"
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

    train = cap.stacked(range(0, len(clips), 2))
    held_out = cap.stacked(range(1, len(clips), 2))
    head_dim = cap.head_dim
    num_heads = cap.num_heads
    thresholds = (0.95, 0.99, 0.999)
    layers_out = []
    for li, ((k_mat, v_mat), (k_test, v_test)) in enumerate(zip(train, held_out)):
        k_res = pca_layer(k_mat, k_test, thresholds)
        v_res = pca_layer(v_mat, v_test, thresholds)
        print(f"  layer {li:2d}: K d_eff@0.99={k_res['d_eff_mean@0.99']:.2f} "
              f"({k_res['d_eff_ratio_mean@0.99']*100:.1f}% of {head_dim})  "
              f"held-out error={k_res['held_out_relative_mse_mean@d_eff_0.99']:.3f}; "
              f"V d_eff@0.99={v_res['d_eff_mean@0.99']:.2f} "
              f"held-out error={v_res['held_out_relative_mse_mean@d_eff_0.99']:.3f}", flush=True)
        layers_out.append({
            "layer": li,
            "n_train_positions": int(k_mat.shape[0]),
            "n_held_out_positions": int(k_test.shape[0]),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=24,
                    help="number of calibration clips (default 24)")
    ap.add_argument("--models", type=str, default="granite,qwen3",
                    help="comma list: granite,qwen3 (default both)")
    ap.add_argument("--audio", type=Path, nargs="+",
                    help="explicit local WAV files; at least two clips are required")
    ap.add_argument("--granite-snapshot", type=Path,
                    help="pinned local Granite HF snapshot directory for reproducible loading")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models or set(models) - {"granite", "qwen3"}:
        ap.error("--models must contain granite and/or qwen3")
    try:
        clips = gather_calibration_clips(args.clips, args.audio)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print(f"calibration set: {len(clips)} clips", flush=True)
    for _, _, name in clips:
        print(f"  - {name}", flush=True)
    if len(clips) < 2:
        print("ERROR: at least two clips are needed for held-out evaluation", file=sys.stderr)
        return 1
    if args.granite_snapshot is not None and not args.granite_snapshot.is_dir():
        ap.error("--granite-snapshot must be an existing directory")
    if args.audio and len({p.name for p in args.audio[:len(clips)]}) != len(clips):
        ap.error("--audio filenames must be distinct for portable result hashes")

    thresholds = (0.95, 0.99, 0.999)
    results: dict[str, Any] = {
        "method": "per-head PCA fitted on even-indexed clips; reconstruction evaluated on odd-indexed clips",
        "thresholds": list(thresholds),
        "n_clips": len(clips),
        "train_clip_names": [clips[i][2] for i in range(0, len(clips), 2)],
        "held_out_clip_names": [clips[i][2] for i in range(1, len(clips), 2)],
    }
    if args.audio:
        results["audio_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.audio[:len(clips)]
        }

    if "granite" in models:
        results["granite"] = measure_granite(clips, args.granite_snapshot)
    if "qwen3" in models:
        results["qwen3"] = measure_qwen3(clips)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nresults written to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
