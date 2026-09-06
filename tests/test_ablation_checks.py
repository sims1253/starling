"""A matching prefix must not pass the ablation token-equality check."""
from types import SimpleNamespace

import pytest
import torch

from benchmarks.bench_ablate import _byte_exact


@pytest.mark.parametrize("generated, expected", [
    ([1, 2, 3], True),
    ([1, 2], False),
    ([1, 2, 3, 4], False),
    ([1, 9, 3], False),
    ([], False),
])
def test_ablation_checks_complete_token_sequence(generated, expected):
    decoder = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(
        ids=torch.tensor([generated], dtype=torch.int64), n_tokens=len(generated),
    ))
    assert _byte_exact(decoder, None, torch.tensor([[9, 1, 2, 3]]), 1, None) is expected


def test_ablation_respects_requested_token_budget():
    decoder = SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(
        ids=torch.tensor([[1, 2]]), n_tokens=2,
    ))
    assert _byte_exact(decoder, None, torch.tensor([[9, 1, 2, 3]]), 1, None,
                       max_new_tokens=2) is True
