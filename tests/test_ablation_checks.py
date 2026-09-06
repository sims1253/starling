"""A matching prefix must not pass the ablation token-equality check."""
import json
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


def test_multistep_ablation_measures_selected_pipeline_decoder(monkeypatch, tmp_path):
    """Run harness dispatch and real pipeline selection without GPU allocation."""
    from benchmarks import bench_ablate as ablate
    from starling.granite import audio, long_audio, multistep, pipeline

    class SingleStep:
        def __init__(self, *args, **kwargs):
            pass

    class MultiStep(SingleStep):
        pass

    language_model = SimpleNamespace(get_input_embeddings=lambda: object())
    model = SimpleNamespace(lm_head=object())
    monkeypatch.setattr(ablate, "load_model_and_processor", lambda **kw: (model, object()))
    monkeypatch.setattr(pipeline, "get_components", lambda model: {
        "encoder": object(), "projector": object(), "language_model": language_model,
    })
    monkeypatch.setattr(pipeline, "FusedEncoder", lambda *args, **kw: object())
    monkeypatch.setattr(pipeline, "FusedLLMMega", SingleStep)
    monkeypatch.setattr(multistep, "MultiStepLLMMega", MultiStep)
    monkeypatch.setattr(audio, "load_sample_audio", lambda: (torch.zeros(300), 1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(ablate, "OUTPUTS", tmp_path)

    measured = []

    def transcribe(pipe, processor, wav, sr, *, chunk_seconds, speculative):
        # Speculative decoding bypasses the selected greedy decoder; it must
        # not be used to report this flag's effect.
        assert speculative is False
        measured.append(type(pipe.llm))
        return SimpleNamespace(text="same transcript")

    monkeypatch.setattr(long_audio, "transcribe_long", transcribe)

    def wrong_measurement(*args, **kwargs):
        pytest.fail("multistep must not use the single-step graph timing harness")

    monkeypatch.setattr(ablate, "bench_decode_step", wrong_measurement)
    args = SimpleNamespace(mode="long_audio", flag="multistep_graph", trials=1)
    saved = ablate.get_default_flags()
    selected = ablate._select_flags(args.mode, args.flag)
    assert ablate._main_locked(args, selected) == 0
    assert measured == [SingleStep, MultiStep]
    assert ablate.get_default_flags() is saved
    report = json.loads((tmp_path / "ablate_long_audio.json").read_text())
    assert report["decoding"] == "greedy"
    assert report["baseline_flags"]["multistep_graph"] is False
    assert report["rows"][0]["text_exact"] is True


def test_wrong_mode_flag_rejected_before_gpu_or_model_setup(monkeypatch, capsys):
    from benchmarks import bench_ablate as ablate
    from starling.parakeet import gpu_lock

    def unexpected_setup(*args, **kwargs):
        pytest.fail("invalid mode/flag combination must fail before GPU/model setup")

    monkeypatch.setattr(gpu_lock, "with_gpu_lock", unexpected_setup)
    monkeypatch.setattr(ablate, "load_model_and_processor", unexpected_setup)
    monkeypatch.setattr("sys.argv", ["bench_ablate.py", "--flag", "multistep_graph"])
    with pytest.raises(SystemExit) as exc:
        ablate.main()
    assert exc.value.code == 2
    assert "not available in decode_step" in capsys.readouterr().err
