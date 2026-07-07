"""Byte-exact correctness tests for the parakeet-unified-en-0.6b megakernel port.

These verify byte-exactness vs the golden reference (captured by
``scripts/parakeet_unified_golden.py``) for each stage:

* the eager greedy RNN-T decode reproduces the golden token sequence,
* the graphed encoder (Step 7) is byte-exact with eager,
* the RNNT megakernel decode (Step 8) reproduces the golden token sequence
  across K in {1,4,16,64},
* the integrated pipeline ``transcribe`` reproduces the golden transcript.

Golden references live in ``golden/parakeet_unified_{short,medium,long}_ids.pt``
/ ``_text.txt`` / ``_meta.pt`` (gitignored). Tests skip if the goldens or
fixtures are absent.

Reference basis: the eager port is itself the byte-exact reference. Neither
NeMo nor sherpa-onnx is installable alongside the pinned torch (see
``scripts/parakeet_unified_golden.py``); the eager greedy loop mirrors NeMo's
``rnnt_greedy_decoding`` and the encoder/decoder/joint load
``state_dict(strict=True)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO / "golden"
FIXTURES = REPO / "tests" / "fixtures"
sys_path_added = False
try:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    sys_path_added = True
except Exception:  # noqa: BLE001
    pass


def _have_torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


_SKIP_REASON = (
    None
    if _have_torch_cuda()
    else "CUDA / torch unavailable -- skipping parakeet_unified tests"
)
pytestmark = pytest.mark.skipif(
    _SKIP_REASON is not None, reason=_SKIP_REASON or ""
)

_FIXTURE_OK = (FIXTURES / "short.wav").exists()
_GOLDEN_OK = (GOLDEN_DIR / "parakeet_unified_short_ids.pt").exists()

FIXTURE_NAMES = ["short", "medium", "long"]


# module-level cache for the loaded eager reference (encoder/decoder/joint/mel/tok)
_REF = None


def _load_reference():
    """Load the eager reference (mel + encoder + decoder + joint + tokenizer)."""
    global _REF
    if _REF is None:
        import numpy as np  # noqa: F401
        import torch

        from starling.parakeet_unified import modeling as M
        from starling.parakeet_unified.loader import load_state_dict
        from starling.parakeet_unified.mel_gpu import GpuMelExtractor
        from starling.parakeet_unified.tokenizer import ParakeetUnifiedTokenizer

        device = "cuda"
        dtype = torch.float32
        sd = load_state_dict(device=device, dtype=dtype)
        mel = GpuMelExtractor(sd, device=device)
        enc = M.ConformerEncoder().to(device).to(dtype).eval()
        enc.load_state_dict_prefixed(sd)
        dec = M.RNNTDecoder().to(device).to(dtype).eval()
        dec.load_state_dict(
            {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")},
            strict=True,
        )
        joint = M.RNNTJoint().to(device).to(dtype).eval()
        joint.load_state_dict(
            {k[len("joint."):]: v for k, v in sd.items() if k.startswith("joint.")},
            strict=True,
        )
        tok = ParakeetUnifiedTokenizer()
        _REF = {"mel": mel, "enc": enc, "dec": dec, "joint": joint, "tok": tok}
    return _REF


def _load_wav(name: str):
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(str(FIXTURES / f"{name}.wav"))
    if wav.ndim > 1:
        wav = wav.mean(1)
    return np.ascontiguousarray(wav.astype(np.float32)), sr


def _golden_ids(name: str):
    import torch

    return torch.load(GOLDEN_DIR / f"parakeet_unified_{name}_ids.pt")


def _golden_text(name: str) -> str:
    return (GOLDEN_DIR / f"parakeet_unified_{name}_text.txt").read_text()


# --------------------------------------------------------------------------- #
# 1. eager greedy decode reproduces the golden token sequence (self-consistency:
# the golden was captured by this exact path; this guards against regressions
# in modeling/decode_eager/mel_gpu/loader).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_eager_decode_matches_golden(name):
    import torch

    from starling.parakeet_unified.decode_eager import greedy_decode

    ref = _load_reference()
    audio, _ = _load_wav(name)
    feats, fl = ref["mel"]([audio])
    with torch.inference_mode():
        encoded, el = ref["enc"](feats, fl)
        ids = greedy_decode(encoded, el, ref["dec"], ref["joint"])
    got = torch.tensor(ids[0], dtype=torch.long)
    want = _golden_ids(name)
    assert got.equal(want), (
        f"eager decode drifted from golden: got {got.tolist()[:20]}..., "
        f"want {want.tolist()[:20]}... (len {len(ids[0])} vs {len(want)})"
    )
    # text round-trip is also byte-exact
    assert ref["tok"].ids_to_text(ids[0]) == _golden_text(name)


# --------------------------------------------------------------------------- #
# 2. encoder forward is deterministic (re-running on the same input reproduces
# the same encoded tensor -> the graphed-encoder gate has a stable fp32 oracle).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
def test_encoder_deterministic():
    import torch

    ref = _load_reference()
    audio, _ = _load_wav("short")
    feats, fl = ref["mel"]([audio])
    with torch.inference_mode():
        a, _ = ref["enc"](feats, fl)
        b, _ = ref["enc"](feats, fl)
    assert torch.equal(a, b), "encoder forward not deterministic"


# --------------------------------------------------------------------------- #
# 3. graphed encoder is byte-exact with eager (max_diff 0.0) across shapes.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_graphed_encoder_byte_exact(name):
    import torch

    from starling.parakeet_unified.encoder_graph import GraphedEncoder

    ref = _load_reference()
    audio, _ = _load_wav(name)
    feats, fl = ref["mel"]([audio])
    with torch.inference_mode():
        eager_enc, eager_lens = ref["enc"](feats, fl)
        ge = GraphedEncoder(ref["enc"])
        g_enc, g_lens = ge(feats, fl)
        # second call exercises the shape cache (amortised replay path)
        g_enc2, g_lens2 = ge(feats, fl)
    assert torch.equal(eager_enc, g_enc), (
        f"graphed encoder drifted from eager: max_diff {(eager_enc - g_enc).abs().max().item()}"
    )
    assert eager_lens.equal(g_lens), "graphed encoder lengths != eager"
    assert torch.equal(g_enc, g_enc2), "graphed encoder not stable across calls"


# --------------------------------------------------------------------------- #
# 4. RNNT megakernel reproduces the eager greedy token sequence across K in
# {1,4,16,64} (fp32). The golden was captured by the eager path, so this also
# asserts mega == golden.
# --------------------------------------------------------------------------- #
_K_VALUES = [1, 4, 16, 64]


@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.parametrize("K", _K_VALUES)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_mega_decode_matches_eager(name, K):
    import torch

    from starling.parakeet_unified import config as C
    from starling.parakeet_unified.decode_mega import GraphedDecoder

    ref = _load_reference()
    audio, _ = _load_wav(name)
    feats, fl = ref["mel"]([audio])
    with torch.inference_mode():
        encoded, el = ref["enc"](feats, fl)
        gd = GraphedDecoder(
            ref["dec"], ref["joint"],
            blank_id=C.BLANK_ID, vocab_size=C.VOCAB_SIZE,
            max_symbols=C.MAX_SYMBOLS_PER_STEP,
            pred_hidden=C.PRED_HIDDEN, n_layers=C.PRED_RNN_LAYERS,
            steps_per_replay=K,
        )
        gd.capture(encoded, el)
        mega_ids = gd.decode(encoded, el)
    want = _golden_ids(name).tolist()
    assert mega_ids[0] == want, (
        f"mega K={K} drifted from golden on {name}: got {mega_ids[0][:20]}..., "
        f"want {want[:20]}... (len {len(mega_ids[0])} vs {len(want)})"
    )


# --------------------------------------------------------------------------- #
# 5. integrated pipeline transcribe matches the golden transcript byte-for-byte
# (graphed encoder + mega decode, fp32).
# --------------------------------------------------------------------------- #
_PIPE = None


def _get_pipeline():
    global _PIPE
    if _PIPE is None:
        import torch

        from starling.parakeet_unified.pipeline import MegaParakeetUnifiedPipeline

        _PIPE = MegaParakeetUnifiedPipeline(
            dtype=torch.float32, encoder_mode="graphed"
        )
    return _PIPE


@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_pipeline_transcribe_matches_golden(name):
    pipe = _get_pipeline()
    audio, _ = _load_wav(name)
    text = pipe.transcribe([audio])[0]
    assert text == _golden_text(name), (
        f"pipeline transcript drifted from golden on {name}"
    )


# --------------------------------------------------------------------------- #
# 6. single-chunk chunker is byte-exact with the one-shot pipeline path
# (a single chunk forms a mini-batch of B=1 -> identical mel/encoder/decode).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_FIXTURE_OK and _GOLDEN_OK), reason="fixtures/golden absent")
@pytest.mark.slow
def test_chunker_single_chunk_matches_pipeline():
    from starling.parakeet_unified.chunking import ChunkedTranscriber

    pipe = _get_pipeline()
    ct = ChunkedTranscriber(pipe, chunk_seconds=30.0, overlap_seconds=2.0)
    audio, _ = _load_wav("short")  # 7.4s -> single chunk
    one_shot = pipe.transcribe([audio])[0]
    chunked = ct.transcribe(audio)
    assert one_shot == chunked, (
        "single-chunk chunker != one-shot pipeline (B=1 must be byte-exact)"
    )
