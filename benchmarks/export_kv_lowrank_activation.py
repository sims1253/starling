"""Export ACTIVATION-space low-rank K factors for the granite encoder (#59).

Second basis provenance: the rank-r basis is fit on REAL K activations
captured from the Granite encoder reference (HF bf16) on calibration speech,
reusing bench_kv_spectral.py's capture classes unmodified, then folded into
the runtime weight:

    B_h = top-r right singular vectors of K_h activations  [128, r]
    f1  = (W_h^T B_h)^T,  f2 = B_h   (z = n @ f1^T,  k_h ~= z @ B_h^T)

Same v1 factor-file format as the weight-space exporter.

Measured outcome (2026-10-01, this campaign): FAILED the 0.2-point WER gate
at r=32 (+0.42) despite held-out K projection error of only ~0.04 mean. The
bf16-fitted basis transfers no better than a basis fitted on the runtime's
own K (see export_kv_lowrank_runtime.py) — basis provenance is not the
failure cause; PCA-style bases zero out-of-basis directions that OOD audio
(FLEURS test clips vs 8 train clips) excites, while weight-space SVD keeps
uniform spectral error. Kept for reproducibility of the negative result.

Usage:
    PYTHONPATH=src:. uv run python benchmarks/export_kv_lowrank_activation.py \
        --gguf models/granite-2b-dynq4.imx.gguf --audio a1.wav ... \
        --rank-k 32 --out benchmarks/results/kvfactors_granite_act_k32.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import bench_kv_spectral as kvs  # noqa: E402  (capture classes, unmodified)
from export_kv_lowrank import attn_kv_weight, check_rank, dequant_2d, fit_pca_fold  # noqa: E402

import gguf  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--rank-k", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.audio) < 2:
        raise ValueError("need at least 2 --audio clips (even = fit, odd = held-out)")
    clips = kvs.gather_calibration_clips(len(args.audio),
                                         [Path(a) for a in args.audio])
    if len(clips) != len(args.audio):
        raise RuntimeError(f"clip gathering mismatch: {len(clips)} of {len(args.audio)}")

    # --- capture K per layer over the calibration clips (torch, bf16 oracle)
    from starling.config import DEFAULT_TASK_PROMPT
    from starling.granite.loader import get_components, load_model_and_processor
    print("loading granite (HF snapshot, cpu) ...", flush=True)
    model, processor = load_model_and_processor(device="cpu")
    comps = get_components(model)
    encoder = comps["encoder"]
    cap = kvs.GraniteKVCapture(encoder)
    cap.attach()
    try:
        for i, (wav, sr, name) in enumerate(clips):
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            # CPU stand-in for starling.granite.audio.build_inputs (which is
            # CUDA-hardcoded): same chat template + processor call.
            chat = [{"role": "user", "content": f"<|audio|>{DEFAULT_TASK_PROMPT}"}]
            prompt = processor.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, wav_t, device="cpu", return_tensors="pt")
            feats = inputs["input_features"].to(dtype=model.dtype)
            cap.record_seq_len(int(feats.shape[1]))
            _ = encoder(feats, return_dict=True)
            print(f"  captured {i + 1}/{len(clips)}: {name}", flush=True)
    finally:
        cap.detach()

    n_layers = cap.num_layers
    n_heads, head_dim = cap.num_heads, cap.head_dim
    check_rank(args.rank_k, head_dim, "--rank-k")
    hidden = None

    # Fit: even clips. Held-out check: odd clips.
    fit = cap.stacked(range(0, len(clips), 2))
    held = cap.stacked(range(1, len(clips), 2))

    reader = gguf.GGUFReader(args.gguf)
    import struct
    f1k_l, f2k_l = [], []
    print("fitting bases + folding ...", flush=True)
    for li in range(n_layers):
        K_fit = fit[li][0]          # [T, n_heads, head_dim] (K only)
        K_held = held[li][0]
        w = dequant_2d(attn_kv_weight(reader, li))  # [2*hidden, hidden] runtime weight
        if hidden is None:
            hidden = w.shape[1]
        wk = w[: w.shape[0] // 2]   # K half [hidden, hidden] (out, in)
        f1, f2, e = fit_pca_fold(K_fit.numpy(), K_held.numpy(), wk, n_heads, args.rank_k)
        f1k_l.append(f1)
        f2k_l.append(f2)
        print(f"  layer {li:2d}: uncentered K proj rel-MSE "
              f"fit={e['err_fit'] / e['en_fit']:.4f} "
              f"held-out={e['err_held'] / e['en_held']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"STLGKVF1")
        f.write(struct.pack("<6I", 1, n_layers, hidden, n_heads, args.rank_k, 0))
        for i in range(n_layers):
            f1k_l[i].tofile(f)
            f2k_l[i].tofile(f)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) rank_k={args.rank_k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
