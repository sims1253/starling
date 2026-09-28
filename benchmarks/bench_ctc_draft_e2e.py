"""Warmed full-wall Granite low-rank CTC draft comparison on public WAVs.

The factorization is prepared once, as an exported artifact would be. Every
timed trial starts from decoded audio and includes processor, encoder, draft head,
projector/prompt, decoder prefill, and verified generation. This uses Granite's
Python verifier; it cannot certify native/phone latency or energy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from bench_ctc_draft_rank import collapse, sha256
from starling.config import LLM_EOS_TOKEN_ID
from starling.granite.audio import build_inputs
from starling.granite.pipeline import MegaPipeline
from starling.granite.speculative import CTCBPEDraft, SpeculativeDecoder


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--rank-result", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[64, 256, 512, 1024])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()
    study = json.loads(args.rank_result.read_text())
    if args.snapshot.name != study["model_revision"]:
        ap.error("snapshot differs from rank study")
    if sha256(args.snapshot / "out_llm.safetensors") != study["source_out_llm_sha256"]:
        ap.error("CTC head hash differs from rank study")
    if (args.repeats < 1 or args.max_new_tokens < 1 or
            len(set(args.ranks)) != len(args.ranks) or
            any(str(rank) not in study["ranks"] for rank in args.ranks)):
        ap.error("invalid repeats/ranks/max-new-tokens")

    from safetensors.torch import load_file
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.snapshot, device_map="cuda", dtype=torch.bfloat16,
        attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(args.snapshot)
    pipeline = MegaPipeline(model, processor, encoder_mode="eager", use_fused_llm=True)
    verifier = SpeculativeDecoder(pipeline.llm, pipeline.embed_tokens)
    ctc = CTCBPEDraft(pipeline.fused_encoder, None)

    head = load_file(str(args.snapshot / "out_llm.safetensors"), device="cpu")
    weight = head["weight"].to(device="cuda", dtype=torch.float32)
    full_weight = weight.to(torch.bfloat16)
    bias = head["bias"].to(device="cuda", dtype=torch.bfloat16)
    _, basis = torch.linalg.eigh(weight.T @ weight)
    basis = basis.flip(-1).contiguous()
    factor = weight @ basis
    basis = basis.to(torch.bfloat16)
    factor = factor.to(torch.bfloat16)
    del weight
    torch.cuda.synchronize()

    methods = ["target", "full", *[str(r) for r in args.ranks]]
    rank_slices = {
        rank: (basis[:, :rank].contiguous(), factor[:, :rank].contiguous())
        for rank in args.ranks
    }
    rank_audio = {
        str(rank): {row["audio"]: row for row in study["ranks"][str(rank)]["audio"]}
        for rank in args.ranks
    }
    for rank in args.ranks:
        rows = study["ranks"][str(rank)]["audio"]
        if len(rank_audio[str(rank)]) != len(rows):
            ap.error(f"rank {rank} study contains duplicate audio names")
    def stamp() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def trial(audio: torch.Tensor, method: str) -> dict:
        t0 = stamp()
        inputs = build_inputs(processor, audio)
        t1 = stamp()
        if method == "target":
            hidden = pipeline.fused_encoder(inputs["input_features"])
            draft_ids: list[int] | None = None
        else:
            mid, hidden = ctc.encode_with_mid(inputs["input_features"])
        t2 = stamp()
        if method != "target":
            mid_logits = pipeline.fused_encoder.out(mid)
            importance = 1 - torch.softmax(mid_logits.float(), dim=-1)[..., 0]
            pooled = CTCBPEDraft._posterior_weighted_pool(hidden, importance, 4)
            pooled = pooled.to(torch.bfloat16)
            if method == "full":
                logits = torch.nn.functional.linear(pooled, full_weight, bias)
            else:
                rank = int(method)
                left, right = rank_slices[rank]
                z = torch.nn.functional.linear(pooled, left.T)
                logits = torch.nn.functional.linear(z, right, bias)
            draft_ids = collapse(logits.argmax(dim=-1)[0])
        t3 = stamp()
        embeds = pipeline.projector(hidden)
        prompt = pipeline.build_inputs_embeds(
            inputs["input_ids"], embeds, inputs.get("input_features_mask"))
        t4 = stamp()
        budget = min(args.max_new_tokens, pipeline.llm.max_cache_len - prompt.shape[1] + 1)
        if budget < 1:
            ap.error(f"prompt {prompt.shape[1]} leaves no generation budget "
                     f"(max_cache_len={pipeline.llm.max_cache_len})")
        if method == "target":
            result = pipeline.llm.generate(prompt, max_new_tokens=budget,
                                           eos_token_id=LLM_EOS_TOKEN_ID)
        else:
            result = verifier.generate(prompt, draft_ids, max_new_tokens=budget,
                                       eos_token_id=LLM_EOS_TOKEN_ID)
        t5 = stamp()
        row = {
            "ids": result.ids[0].tolist(),
            "prompt_tokens": int(prompt.shape[1]),
            "draft_tokens": len(draft_ids) if draft_ids is not None else 0,
            "draft_ids": draft_ids,
            "prep_ms": (t1-t0)*1000,
            "encoder_ms": (t2-t1)*1000,
            "head_ms": (t3-t2)*1000,
            "projector_prompt_ms": (t4-t3)*1000,
            "generation_wall_ms": (t5-t4)*1000,
            "generation_decode_ms": result.total_ms,
            "full_wall_ms": (t5-t0)*1000,
        }
        if method != "target":
            row.update(accepted=result.accepted,
                       acceptance_rate=result.acceptance_rate,
                       verify_forwards=result.verify_forwards,
                       decode_steps=result.decode_steps)
        return row

    output = []
    for ix, record in enumerate(study["audio"]):
        path = args.audio_dir / record["audio"]
        if sha256(path) != record["sha256"]:
            ap.error(f"audio hash differs from rank study: {path}")
        samples, sr = sf.read(path, dtype="float32")
        if sr != 16000 or samples.ndim != 1:
            ap.error(f"expected mono 16 kHz WAV: {path}")
        audio = torch.from_numpy(samples).unsqueeze(0)
        # Exclude each input shape's first-use costs from timed trials.
        for method in methods:
            trial(audio, method)
        rows: dict[str, list[dict]] = {method: [] for method in methods}
        for rep in range(args.repeats):
            order = methods if (ix + rep) % 2 == 0 else list(reversed(methods))
            for method in order:
                rows[method].append(trial(audio, method))
        gold = rows["target"][0]["ids"]
        for method, values in rows.items():
            for row in values:
                row["greedy_identical"] = row["ids"] == gold
                if method == "full":
                    row["draft_matches_rank_study"] = row["draft_ids"] == record["baseline_ids"]
                elif method != "target":
                    candidate = rank_audio[method].get(record["audio"])
                    if candidate is None:
                        ap.error(f"rank {method} study lacks audio {record['audio']}")
                    row["draft_matches_rank_study"] = row["draft_ids"] == candidate["draft_ids"]
                row.pop("ids")
                row.pop("draft_ids")
        output.append({"audio": record["audio"], "sha256": record["sha256"],
                       "duration_s": record["duration_s"], "trials": rows})
        print(f"[{ix+1}/{len(study['audio'])}] {record['audio']} "
              + " ".join(f"{m}={sum(v['full_wall_ms'] for v in rows[m])/len(rows[m]):.0f}ms"
                         for m in methods), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "rank_study": args.rank_result.name,
        "model_revision": study["model_revision"],
        "source_out_llm_sha256": study["source_out_llm_sha256"],
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "method": "warmed full-wall decoded waveform to verified tokens; factor preparation excluded",
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "audio": output,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
