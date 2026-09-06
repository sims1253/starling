"""Exercise the real decoder state updates without model weights or a GPU."""

from contextlib import contextmanager
from importlib import import_module

import pytest
import torch


MODELS = ("ark", "audex", "granite", "higgs", "moss", "qwen3")


@pytest.fixture(params=MODELS)
def decoder(request, monkeypatch):
    model = request.param
    module = import_module(f"starling.{model}.multistep")
    cls = getattr(module, "MossMultiStepMega" if model == "moss" else "MultiStepLLMMega")
    dec = object.__new__(cls)
    dec.device = "cpu"
    dec.max_cache_len = 8
    dec.warmup_iters = 3
    dec.prefill_use_graph = False
    dec.eos_token_id = 15
    dec._neg_val = float("-inf")
    dec.static_attn_mask = torch.full((1, 1, 1, 8), dec._neg_val)
    dec.static_input_ids = torch.zeros((1, 1), dtype=torch.int64)
    dec.static_position_ids = torch.zeros((1, 1), dtype=torch.int64)
    dec.static_cache_pos = torch.zeros(1, dtype=torch.int64)
    dec.static_cache_position = torch.zeros(1, dtype=torch.int64)
    dec.pos_buf = torch.zeros((), dtype=torch.int64)
    dec.cpos_buf = torch.zeros((), dtype=torch.int64)
    dec.static_logits = torch.zeros((1, 1, 16))
    dec._init_multistep(4, "cpu")
    dec.positions = []
    dec.cache_cursor = 0

    def reset_cache(pos):
        dec.cache_cursor = pos

    def decode():
        position = int(dec.static_position_ids.item())
        assert 0 <= position < dec.max_cache_len
        mask = dec.static_attn_mask.view(-1)
        assert torch.all(mask[: position + 1] == 0)
        assert torch.all(torch.isneginf(mask[position + 1 :]))
        if model in ("ark", "audex", "granite", "qwen3"):
            assert dec.cache_cursor < dec.max_cache_len
            dec.cache_cursor += 1
        elif model == "moss":
            assert int(dec.static_cache_pos.item()) == position
        else:
            assert int(dec.static_cache_position.item()) == position
        dec.positions.append(position)
        dec.static_logits.zero_()
        dec.static_logits[0, 0, (int(dec.static_input_ids.item()) + 1) % 16] = 1

    dec._reset_cache_pos = reset_cache
    dec._decode_step_eager = decode
    dec._finalize = lambda ids, elapsed, *args: ids

    def prefill(inputs, **kwargs):
        prompt = inputs["inputs_embeds"] if isinstance(inputs, dict) else inputs
        dec.cache_cursor = prompt.shape[1]
        first = torch.tensor([[1]])
        return (first, prompt.shape[1]) if model == "higgs" else first

    dec.prefill = prefill

    class Graph:
        def replay(self):
            self.run()

    @contextmanager
    def graph_context(graph):
        # Parent capture runs a single step. Shared capture runs K steps.
        graph.run = dec._run_k_steps if getattr(dec, "_captured", False) else decode
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return dec


def generate(decoder, prompt_len, count, **kwargs):
    embeds = torch.zeros((1, prompt_len, 1))
    if decoder.__class__.__module__.split(".")[-2] == "higgs":
        return decoder.generate({"inputs_embeds": embeds}, max_new_tokens=count, **kwargs)
    return decoder.generate(embeds, max_new_tokens=count, **kwargs)


def test_capture_warmups_reset_at_cache_boundary(decoder):
    # Only one K-step chunk fits. Three warmups must reuse those same slots.
    decoder.cache_cursor = 4
    decoder.capture(torch.tensor([[1]]), 4)
    assert max(decoder.positions) == 7
    assert decoder.valid_len_buf.item() == 5
    assert decoder.static_input_ids.item() == 1
    assert decoder._ms_captured


@pytest.mark.parametrize("capture", [False, True])
def test_exact_cache_budget_and_second_request(decoder, capture):
    # Seven generated tokens require six cache slots: a full chunk and two steps.
    assert generate(decoder, 2, 7, capture=capture) == list(range(1, 8))
    assert generate(decoder, 5, 4, capture=capture) == list(range(1, 5))
    assert max(decoder.positions) <= 7


def test_prefill_eos_needs_no_graph_or_decode(decoder):
    assert decoder._generate_multistep(torch.tensor([[15]]), 4, 5, (15,), True)[0] == [15]
    assert decoder.positions == []
    assert decoder._ms_graph is None


@pytest.mark.parametrize("eos", [2, 4, 5])
def test_eos_trims_replay(decoder, eos):
    assert decoder._generate_multistep(torch.tensor([[1]]), 2, 7, (eos,), True)[0] == list(
        range(1, eos + 1)
    )


def test_uncaptured_short_request_does_not_capture(decoder):
    assert generate(decoder, 7, 2) == [1, 2]
    assert decoder._ms_graph is None
    assert decoder.positions == [7]


def test_invalid_budget_and_zero_tokens(decoder):
    with pytest.raises(ValueError, match="overflows cache"):
        generate(decoder, 7, 3)
    assert generate(decoder, 7, 0) == []
    assert decoder.positions == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph validation requires a GPU")
@torch.inference_mode()
def test_real_cuda_graph_replay_and_partial_tail():
    from starling.multistep import MultiStepDecoder

    class SingleStep:
        def capture(self, first_token, prefill_len):
            self._reset_to_chunk_start(prefill_len, first_token)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._decode_step_eager()

    class Decoder(MultiStepDecoder, SingleStep):
        def _reset_cache_pos(self, pos):
            pass

        def _decode_step_eager(self):
            self.static_logits.zero_()
            self.static_logits.scatter_(-1, ((self.static_input_ids + 1) % 16).unsqueeze(-1), 1)

    dec = Decoder()
    dec.max_cache_len = 8
    dec.warmup_iters = 3
    dec._neg_val = float("-inf")
    dec.static_attn_mask = torch.full((1, 1, 1, 8), dec._neg_val, device="cuda")
    dec.static_input_ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    dec.static_position_ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    dec.static_logits = torch.zeros((1, 1, 16), device="cuda")
    dec._init_multistep(4, "cuda")
    first = torch.tensor([[1]], device="cuda")
    # Warm the tiny decode operations before the parent's graph capture.
    dec._decode_step_eager()
    torch.cuda.synchronize()
    assert dec._generate_multistep(first, 2, 7, (15,), True)[0] == list(range(1, 8))
    assert dec._generate_multistep(first, 4, 5, (3,), True)[0] == [1, 2, 3]


def test_graph_reuse_changes_prompt_without_replacing_buffers(decoder):
    assert generate(decoder, 2, 5) == list(range(1, 6))
    graph = decoder._ms_graph
    names = (
        "static_input_ids",
        "static_position_ids",
        "static_attn_mask",
        "valid_len_buf",
        "output_ids",
    )
    addresses = [getattr(decoder, name).data_ptr() for name in names]
    for prompt_len, capture in ((3, True), (4, False)):
        decoder.positions.clear()
        assert generate(decoder, prompt_len, 5, capture=capture) == list(range(1, 6))
        assert decoder.positions == list(range(prompt_len, prompt_len + 4))
        assert decoder._ms_graph is graph
        assert [getattr(decoder, name).data_ptr() for name in names] == addresses
