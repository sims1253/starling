"""Measure copy-draft token acceptance against recorded S1-mini greedy IDs.

Run from the repo root with a locally captured golden directory:
  python benchmarks/speculative/eval_copy_drafts.py --golden-dir golden/s1

Only acceptance and verify-pass counts are observable here. Native latency,
energy, and numerical parity require issue #311's verifier and a device run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

if __package__:
    from .copy_draft import simulate
else:
    from copy_draft import simulate

ROOT = Path(__file__).resolve().parents[2]


def evaluate(golden_dir: Path, max_k: int) -> dict:
    import torch
    from transformers import AutoTokenizer

    config = runpy.run_path(str(ROOT / "src/starling/s1/config.py"))
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_ID"], local_files_only=True)
    fixtures = runpy.run_path(str(ROOT / "tests/fixtures/s1_transcripts.py"))["LENGTH_TIERS"]
    cases = []
    for tier, transcript in fixtures.items():
        golden = golden_dir / f"greedy_ids_{tier}.pt"
        if not golden.is_file():
            raise FileNotFoundError(f"missing golden capture for tier {tier!r}: {golden}")
        ids = torch.load(golden, map_location="cpu", weights_only=True).reshape(-1).tolist()
        source = tokenizer.encode(transcript, add_special_tokens=False)
        result = simulate(source, ids, max_k=max_k)
        cases.append({
            "tier": tier,
            "source_tokens": len(source),
            "golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
            **result.to_dict(),
        })
    return {
        "source": "S1-mini raw transcript tokenized with its own tokenizer",
        "target": "eager stock greedy IDs captured by starling.s1.golden",
        "max_k": max_k,
        "limitations": "Offline oracle replay; no model verify, latency, energy, or device parity",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    config = runpy.run_path(str(ROOT / "src/starling/s1/config.py"))
    parser.add_argument("--golden-dir", type=Path, default=config["GOLDEN_DIR"])
    parser.add_argument("--max-k", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.golden_dir, args.max_k), indent=2))


if __name__ == "__main__":
    main()
