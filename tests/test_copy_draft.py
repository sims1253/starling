"""Acceptance behavior of the offline copy proposer."""

import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/speculative/copy_draft.py"
    spec = importlib.util.spec_from_file_location("copy_draft", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module  # dataclasses resolves postponed annotations here
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[spec.name]
        else:
            sys.modules[spec.name] = previous
    return module


def test_deletions_realign_and_accept_multiple_tokens():
    m = _module()
    # Source's fillers are skipped when a later target token matches.
    result = m.simulate([10, 99, 11, 12, 13], [10, 11, 12, 13], max_k=2)
    assert result.full_accept_passes >= 1
    assert result.verify_passes < result.output_tokens
    assert result.accepted_tokens >= 2


def test_rejected_edits_still_reproduce_greedy_stream():
    m = _module()
    result = m.simulate([10, 11, 12], [20, 21, 22], max_k=2)
    assert result.accepted_tokens == 0
    assert result.verify_passes == result.output_tokens


def test_lookup_only_uses_emitted_prefix():
    m = _module()
    drafter = m.CopyDrafter([])
    assert drafter.propose([4, 5, 4], 2) == [5, 4]
    assert drafter.propose([4, 5, 4, 5], 2) == [4, 5]


def test_stalled_source_without_lookup_uses_target_only_pass():
    m = _module()
    drafter = m.CopyDrafter([1, 2, 3])
    drafter.misses = 3
    assert drafter.propose([9], 2) == []
    result = m.simulate([1, 2, 3], [9, 8, 7, 6], max_k=2)
    assert result.passes_without_draft >= 1


def test_truncated_final_draft_counts_as_full_accept():
    # The last pass verifies only the remaining target token; a match is full.
    assert _module().simulate([1, 2], [1], max_k=2).full_accept_passes == 1
