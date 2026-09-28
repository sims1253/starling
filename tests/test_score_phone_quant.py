"""The phone WER scorer must reject gaps, reordering, and false device labels."""

from pathlib import Path

import pytest

from benchmarks.fast_engine.score_phone_quant import parse_logs, score


DEVICE = "PowerVR D-Series DXT-48-1536 MC1"
ENGINE = (f"[fast] device '{DEVICE}' vendor=1010\n"
          f"[fast] parakeet engine: fast/vulkan '{DEVICE}' weights=719 MiB\n")


def _logs(tmp_path: Path, groups: list[list[str]]) -> tuple[list[Path], list[Path]]:
    logs, engine_logs = [], []
    for index, names in enumerate(groups):
        log = tmp_path / f"run-{index}.stdout"
        engine = tmp_path / f"run-{index}.stderr"
        lines = ["load 1.0 ms  backend=cpu",
                 "short.wav run=0 audio=1.00s time=1.0ms rtf=0.0010", "  warmup"]
        for name in names:
            lines += [f"{name} run=0 audio=1.00s time=2.0ms rtf=0.0020", "  text"]
        log.write_text("\n".join(lines) + "\n")
        engine.write_text(ENGINE)
        logs.append(log)
        engine_logs.append(engine)
    return logs, engine_logs


def test_accepts_contiguous_fresh_process_segments(tmp_path):
    logs, engine_logs = _logs(tmp_path, [["a.wav"], ["b.wav"]])
    result = parse_logs(logs, engine_logs, ["a.wav", "b.wav"], DEVICE, "short.wav")
    assert list(result) == ["a.wav", "b.wav"]


@pytest.mark.parametrize("groups", [
    [["b.wav", "a.wav"]],
    [["a.wav"], ["a.wav", "b.wav"]],
    [["a.wav"]],
])
def test_rejects_reorder_duplicate_or_gap(tmp_path, groups):
    logs, engine_logs = _logs(tmp_path, groups)
    with pytest.raises(ValueError):
        parse_logs(logs, engine_logs, ["a.wav", "b.wav"], DEVICE, "short.wav")


def test_rejects_generic_header_without_real_fast_identity(tmp_path):
    logs, engine_logs = _logs(tmp_path, [["a.wav"]])
    engine_logs[0].write_text("[fast] parakeet falls back to ggml: unavailable\n")
    with pytest.raises(ValueError, match="actual fast Vulkan device"):
        parse_logs(logs, engine_logs, ["a.wav"], DEVICE, "short.wav")


def test_rejects_malformed_protocol_before_scoring(tmp_path):
    from argparse import Namespace

    protocol = tmp_path / "protocol.json"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    protocol.write_text('{"schema":"quant-wer-noninferiority-v1"}')
    with pytest.raises(ValueError, match="protocol cohorts"):
        score(Namespace(protocol=protocol, corpus=corpus))
