"""Measure low-rank Granite CTC BPE draft heads on pinned, real speech.

This is an offline study. It does not change the runtime or imply speculative
speedup: a draft needs the decoder verifier to accept enough tokens first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from starling.granite.audio import build_inputs
from starling.granite.encoder_mega import FusedEncoder
from starling.granite.loader import get_components
from starling.granite.speculative import CTCBPEDraft


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collapse(labels: torch.Tensor) -> list[int]:
    labels = torch.unique_consecutive(labels)
    return (labels[labels > 0] - 1).cpu().tolist()


def edit_distance(a: list[int], b: list[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        row = [i]
        for j, right in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (left != right)))
        prev = row
    return prev[-1]


def timed_ms(fn, reps: int = 12) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return float(np.median(values))


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--audio", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256, 512, 768, 1024])
    args = ap.parse_args()
    if args.snapshot.name != "de575db64086f84fdc79da4932d1076e965bc546":
        ap.error("snapshot must be pinned Granite revision de575db6")
    if not args.audio or any(r < 1 or r > 1024 for r in args.ranks):
        ap.error("provide audio and ranks in 1..1024")

    from safetensors.torch import load_file
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    t0 = time.monotonic()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.snapshot, device_map="cuda", dtype=torch.bfloat16,
        attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(args.snapshot)
    encoder = FusedEncoder(get_components(model)["encoder"], mode="eager")
    head_path = args.snapshot / "out_llm.safetensors"
    state = load_file(str(head_path), device="cpu")
    weight = state["weight"].to(device="cuda", dtype=torch.float32)
    weight_bf16 = weight.to(torch.bfloat16)
    head_vocab, head_hidden = weight.shape
    bias = state["bias"].to(device="cuda", dtype=torch.bfloat16)
    # Best Frobenius low-rank approximation of the weight. The 1024x1024 Gram
    # avoids materializing a 100353x100353 covariance matrix.
    gram = weight.T @ weight
    _, basis = torch.linalg.eigh(gram)
    basis = basis.flip(-1).contiguous()
    factor = weight @ basis
    del weight, gram
    basis = basis.to(torch.bfloat16)
    factor = factor.to(torch.bfloat16)
    torch.cuda.synchronize()
    print(f"loaded/factorized in {time.monotonic() - t0:.1f}s", flush=True)

    records = []
    pooled_all = []
    for path in args.audio:
        audio, sr = sf.read(path, dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            ap.error(f"expected mono 16 kHz WAV: {path}")
        inputs = build_inputs(processor, torch.from_numpy(audio).unsqueeze(0))
        mid, hidden = CTCBPEDraft(encoder, None).encode_with_mid(inputs["input_features"])
        mid_logits = encoder.out(mid)
        importance = 1 - torch.softmax(mid_logits.float(), dim=-1)[..., 0]
        pooled = CTCBPEDraft._posterior_weighted_pool(hidden, importance, 4)
        pooled_all.append(pooled.squeeze(0).to(torch.bfloat16))
        records.append({"audio": path.name, "sha256": sha256(path),
                        "duration_s": len(audio) / sr, "pooled_frames": pooled.shape[1]})
        print(f"encoded {path.name}: {pooled.shape[1]} pooled frames", flush=True)

    pooled = torch.cat(pooled_all, dim=0)
    baseline = torch.nn.functional.linear(pooled, weight_bf16, bias)
    baseline_labels = baseline.argmax(dim=-1)
    baseline_ms = timed_ms(lambda: torch.nn.functional.linear(pooled, weight_bf16, bias))
    del baseline

    offset = 0
    for record, chunk in zip(records, pooled_all):
        length = len(chunk)
        labels = baseline_labels[offset:offset + length]
        record["baseline_ids"] = collapse(labels)
        offset += length

    ranks = {}
    for rank in sorted(set(args.ranks)):
        left = basis[:, :rank].contiguous()
        right = factor[:, :rank].contiguous()
        def project():
            z = torch.nn.functional.linear(pooled, left.T)
            return torch.nn.functional.linear(z, right, bias)
        predicted = project().argmax(dim=-1)
        measured_ms = timed_ms(project)
        offset = 0
        per_audio = []
        for record, chunk in zip(records, pooled_all):
            length = len(chunk)
            candidate = predicted[offset:offset + length]
            reference = baseline_labels[offset:offset + length]
            ids = collapse(candidate)
            original_ids = record["baseline_ids"]
            per_audio.append({
                "audio": record["audio"],
                "frame_match": float((candidate == reference).float().mean().item()),
                "draft_ids": ids,
                "draft_exact": ids == original_ids,
                "draft_edit_distance": edit_distance(ids, original_ids),
            })
            offset += length
        ranks[str(rank)] = {
            "projection_ms_median": measured_ms,
            # Above hidden * vocab / (hidden + vocab) the factors exceed the full head.
            "head_weight_fraction": (rank * (head_hidden + head_vocab)) / (head_hidden * head_vocab),
            "frame_match": float((predicted == baseline_labels).float().mean().item()),
            "audio": per_audio,
        }
        print(f"rank={rank} frame-match={ranks[str(rank)]['frame_match']:.3%} "
              f"projection={measured_ms:.2f}ms", flush=True)

    result = {
        "model_revision": args.snapshot.name,
        "source_out_llm_sha256": sha256(head_path),
        "method": "top eigenspace of W.T@W, BF16 two-stage projection",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "baseline_projection_ms_median": baseline_ms,
        "audio": records,
        "ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
