"""Native ggml parity gates — manifest-driven correctness contract (issue #167).

The executable contract lives in ``tests/native_parity_manifest.json``
(loader/validator: ``tests/parity_contract.py``). Every gate in this file is
claimed by a manifest target; ``tests/test_native_parity_contract.py`` fails
if the manifest, this file's test selection, and the rendered table in
``docs/ggml-engine.md`` ever disagree.

Contract classes (manifest ``contract.class``):

* ``exact_token_equality`` — the full emitted token stream must match the
  reference element-for-element (a matching prefix is a FAILURE).
* ``exact_text_equality`` — the entire transcript must match after the
  recorded normalization (default: trailing-whitespace rstrip only).
* ``component_numerical_tolerance`` — staged component tensors with explicit
  per-component bounds (metric, aggregation, tolerance, justification).
* ``corpus_quality_acceptance`` — an explicitly documented approximate path:
  metric + aggregation + language normalization + acceptance margin are
  recorded in the manifest BEFORE evaluation, never invented at failure time.

Skipping rules (CI compatibility):

* A target whose GGUF / lib / goldens are ABSENT in this environment skips and
  is reported as UNAVAILABLE coverage by the accounting test (distinct, not
  silent; fails only under ``STARLING_PARITY_STRICT=1``).
* A REQUIRED target whose assets are all present but which executes ZERO
  tests FAILS (a skip-vacuum is a broken gate, not a green one).
* A present asset whose sha256 does not match the manifest pin FAILS outright
  (never silently measure — or regenerate — the reference).

Run with:  uv run pytest tests/test_ggml_parity.py -q
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests"))
sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

import make_fixtures as mkfx  # noqa: E402  (tests/fixtures added to sys.path above)
import parity_contract as pc  # noqa: E402  (the executable contract)

pc.mark_parity_module_loaded()

GOLDEN = _REPO_ROOT / "golden"
FIXTURES = mkfx.load_fixtures()  # {short, medium, long} -> 1-D float32 @16kHz

#: manifest target ids used below (single source of truth for the mapping).
T_MOSS_TEXT = "moss.intree.text"
T_MOSS_MEL = "moss.intree.mel"
T_MOSS_ENCODER = "moss.intree.encoder"
T_QWEN_DECODE = "qwen_decode.shared_fixture"
T_QWEN_TOKENS = "qwen_decode.shared_fixture.token_stream"
T_PARAKEET_TEXT = "parakeet.intree.text"
T_PARAKEET_IDS = "parakeet.intree.ids"
T_MOSS_CRISPASR = "moss.crispasr.external"
T_PARAKEET_EXTERNAL = "parakeet.external"
T_MOSS_KSTEP = "moss.intree.kstep_regression"
T_ARK_TEXT = "ark.intree.text"
T_HIGGS_TEXT = "higgs.intree.text"
T_HOJO_TEXT = "hojo.intree.text"
T_GRANITE_TEXT = "granite.intree.text"
T_QWEN3_TEXT = "qwen3.intree.text"
T_AUDEX_TEXT = "audex.intree.text"
T_S1_TEXT = "s1.intree.text"
T_S1_CONTROL = "s1.intree.control_matrix_smoke"


def _manifest_target(target_id: str) -> dict:
    m = pc.load_manifest()
    for t in m["targets"]:
        if t["id"] == target_id:
            return t
    raise pc.ManifestError(f"target {target_id!r} missing from the manifest")


def _manifest_gate(target_id: str) -> None:
    """Manifest-driven skip/fail gate for one target.

    Wrong-asset problems (sha mismatch) FAIL; missing assets record the target
    as UNAVAILABLE coverage and skip distinctly.
    """
    probe = pc.probe_target_assets(_manifest_target(target_id))
    if probe.problems:
        pytest.fail("; ".join(probe.problems), pytrace=False)
    if probe.missing:
        pc.record_unavailable(target_id, ", ".join(probe.missing))
        pytest.skip(f"{target_id}: assets unavailable here "
                    f"({', '.join(probe.missing)})")


def _ggml_available() -> bool:
    """True iff the parakeet-server binary + model exist (so the test can run)."""
    try:
        from engines import GgmlParakeet
    except Exception:
        return False
    try:
        return GgmlParakeet().available
    except Exception:
        return False


@pytest.fixture(scope="module")
def ggml_engine():
    """One persistent parakeet engine for the whole module (load paid once).

    The default path is the in-process ctypes binding (fastest). ggml's global
    Backend static destructor aborts at process exit on some builds; that crash
    is after all tests pass and does not affect their outcome (pytest reports
    before the atexit crash on stdout-flushed runs). For environments where the
    atexit crash must be fully isolated, set GGML_PARAKEET_NATIVE=0 to use the
    HTTP-server path (ggml runs in a child process).
    """
    if not _ggml_available():
        pc.record_unavailable(T_PARAKEET_EXTERNAL, "parakeet-server binary/model unavailable")
        pytest.skip("parakeet-server binary or model unavailable")
    from engines import GgmlParakeet

    eng = GgmlParakeet()
    eng.load()
    yield eng
    eng.close()


@pytest.mark.skipif(not _ggml_available(), reason="parakeet-server binary or model unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_ggml_parakeet_external_text_gate(ggml_engine, name: str) -> None:
    """External parakeet.cpp server: SMOKE-QUALITY gate, not certification.

    Renamed from ``test_ggml_parakeet_byte_exact``: the long fixture only
    holds a similarity ratio (>= 0.90), so the old name overclaimed. The
    in-tree certification lives in ``test_starling_ggml_parakeet_text_parity``
    (manifest target ``parakeet.intree.text``). Manifest:
    ``parakeet.external`` — short/medium byte-exact against the golden
    captured by ``scripts/parakeet_tdt_golden.py``; long is
    corpus-quality (transformers 5.14 SDPA kernel-path drift, WER-verified
    benign; see parakeet.cpp 147ba98 for the guarded-out multistep on long).
    """
    _manifest_gate(T_PARAKEET_EXTERNAL)
    pc.record_executed(T_PARAKEET_EXTERNAL)
    golden_text = (GOLDEN / f"parakeet_tdt_{name}_text.txt").read_text()
    out = ggml_engine._run_one(FIXTURES[name])
    if name == "long":
        ratio = difflib.SequenceMatcher(None, out, golden_text).ratio()
        assert ratio >= 0.90, (
            f"parakeet-tdt ggml transcript drift too high on {name} (ratio={ratio:.3f}):\n"
            f"  golden: {golden_text[:160]!r}\n  ggml:   {out[:160]!r}"
        )
    else:
        pc.assert_exact_text(out, golden_text, target=T_PARAKEET_EXTERNAL, fixture=name)


# --------------------------------------------------------------------------- #
# Moss-Transcribe-preview-2B (CrispASR moss-transcribe backend) — DEPRECATED
# --------------------------------------------------------------------------- #
def _ggml_moss_available() -> bool:
    try:
        from engines import GgmlMoss
    except Exception:
        return False
    try:
        return GgmlMoss().available
    except Exception:
        return False


moss = pytest.mark.skipif(
    not (_ggml_available() and _ggml_moss_available()),
    reason="parakeet-server or CrispASR MOSS binary/model unavailable (preserves "
           "the historical external-engine module gate)",
)


@pytest.fixture(scope="module")
def ggml_moss_engine():
    """One CrispASR moss-transcribe engine for the whole module (skipped if
    unavailable)."""
    if not (_ggml_available() and _ggml_moss_available()):
        pc.record_unavailable(T_MOSS_CRISPASR, "parakeet/CrispASR MOSS unavailable")
        pytest.skip("historical external-engine gate: parakeet/CrispASR MOSS unavailable")
    from engines import GgmlMoss

    eng = GgmlMoss()
    eng.load()
    yield eng
    eng.close()


@moss
@pytest.mark.parametrize("name", ["short"])
def test_ggml_moss_crispasr_short_exact(ggml_moss_engine, name: str) -> None:
    """Deprecated external CrispASR engine: SHORT is byte-exact.

    Renamed from ``test_ggml_moss_byte_exact`` to name the engine explicitly.
    Both invocation paths (persistent server and one-shot CLI fallback)
    reproduce the golden exactly on short: the audio fits one 30 s chunk and
    the decode has not yet accumulated enough KV-cache context to diverge from
    the golden's HF eager greedy path.
    """
    _manifest_gate(T_MOSS_CRISPASR)
    pc.record_executed(T_MOSS_CRISPASR)
    golden_text = (GOLDEN / f"moss_{name}_text.txt").read_text()
    out = ggml_moss_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_MOSS_CRISPASR, fixture=name)


@moss
@pytest.mark.parametrize("name", ["medium", "long"])
def test_ggml_moss_crispasr_approx_smoke(ggml_moss_engine, name: str) -> None:
    """Deprecated external CrispASR engine: APPROXIMATE SMOKE gate.

    Renamed from ``test_ggml_moss_near_exact`` so the name cannot be mistaken
    for accuracy certification. The residual divergence is NOT a flag or
    CrispASR-post-processing issue — it is an inherent numeric-path difference
    between CrispASR's ggml f16 KV-cache decode and the golden's HF bf16 eager
    greedy decode. At the low-confidence ``eyes`` repetition boundary the two
    argmax results flip: CrispASR emits ``eyes. It`` (period + capital) where
    the golden emits ``eyes it``. Confirmed on the raw token stream
    (``moss_transcribe: N tokens`` verbose log): the period appears in the LLM
    output itself, before any post-processing. Flags tried that do NOT fix it:
    ``--no-punctuation`` (strips ALL punctuation, including the golden's own
    commas — too aggressive), ``-nfa`` (no flash attention),
    ``--frequency-penalty 0``, ``-bs greedy``.

    Characterized residual (one-shot CLI / server single-chunk, byte-identical):
      * medium: normalized CER = 0.0000 (2 inserted periods).
      * long:   normalized CER < 0.02 (6 inserted periods + 6 capitalizations
        + a different EOS/truncation point).

    The CER bound catches any regression beyond this known, documented gap.
    The in-tree engine's certification is a SEPARATE contract
    (``moss.intree.text``); this deprecated engine must never define
    Starling's native correctness.
    """
    _manifest_gate(T_MOSS_CRISPASR)
    pc.record_executed(T_MOSS_CRISPASR)
    golden_text = (GOLDEN / f"moss_{name}_text.txt").read_text()
    out = ggml_moss_engine._run_one(FIXTURES[name])
    pc.assert_cer_below(out, golden_text, target=T_MOSS_CRISPASR, fixture=name,
                        margin=0.10)


# --------------------------------------------------------------------------- #
# In-tree MOSS C API
# --------------------------------------------------------------------------- #
def _starling_ggml_moss_available() -> bool:
    try:
        from engines import StarlingGgmlMoss
        return StarlingGgmlMoss().available
    except Exception:
        return False


@pytest.fixture(scope="module")
def starling_ggml_moss_engine():
    if not _starling_ggml_moss_available():
        pc.record_unavailable(T_MOSS_TEXT, "in-tree libstarling_ggml or MOSS GGUF unavailable")
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_MOSS_MODEL unavailable")
    from engines import StarlingGgmlMoss

    eng = StarlingGgmlMoss()
    eng.load()
    yield eng
    eng.close()


@pytest.mark.skipif(not _starling_ggml_moss_available(),
                    reason="in-tree libstarling_ggml or MOSS GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_moss_text_parity(starling_ggml_moss_engine, name: str) -> None:
    """The in-tree C API returns the golden MOSS transcript EXACTLY on all
    three fixtures (contract class exact_text_equality, no tolerance).

    This deliberately has no CrispASR-style tolerance: it gates Starling's own
    loader → mel → encoder → adapter → prompt → LLM → detokenizer pipeline
    against the pinned HF eager greedy reference captured by
    ``scripts/moss_golden.py`` (device + library versions + output hashes in
    ``golden/moss_reference_provenance.json``).

    Resolution of the issue #167 drift: the previous normalized-CER < 0.10
    escape hatch on the long fixture was inherited from the parakeet
    "transformers 5.14 SDPA kernel-path drift" story, but the MOSS golden path
    is eager everywhere (the loader propagates ``attn_implementation="eager"``
    to every nested config), so that drift mechanism does not apply. Measured
    on the reference capture recorded in this PR, the in-tree engine
    reproduces the long fixture byte-exactly, and the exact gate is restored.
    See the manifest changelog for the measurement record.
    """
    _manifest_gate(T_MOSS_TEXT)
    pc.record_executed(T_MOSS_TEXT)
    golden_text = (GOLDEN / f"moss_{name}_text.txt").read_text()
    out = starling_ggml_moss_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_MOSS_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# In-tree MOSS component gates (staged golden/raw tensors)
#
# The shared-Qwen decoder contract (issue #167's third fixture) is
# ``moss_llm_test`` below: it drives cpp/lib/qwen_decode through
# moss::llm_prefill + moss::greedy_generate against the staged HF eager
# reference tensors, gating bitwise embeds, exact argmax, a bounded logits
# max-abs (every backend), and EXACT token ids + text (CUDA: all fixtures;
# CPU: short — cpp/moss/llm.cpp documents the CPU bf16 GEMM fallback; see
# _run_moss_llm_gate and the manifest enforcement block).
# --------------------------------------------------------------------------- #
_CPP_COMPONENT_TIMEOUT = 1200


def _run_cpp_component_test(binary: str, target_id: str) -> None:
    _manifest_gate(target_id)
    pc.record_executed(target_id)
    bin_path = _REPO_ROOT / "build" / binary
    if not bin_path.exists():
        pc.record_unavailable(target_id, f"build/{binary} not built")
        pytest.skip(f"build/{binary} not built (run cmake --build build -j)")
    proc = subprocess.run(
        [str(bin_path), str(_REPO_ROOT)],
        capture_output=True, text=True, timeout=_CPP_COMPONENT_TIMEOUT,
    )
    assert proc.returncode == 0, (
        f"{binary} exited {proc.returncode} (component contract {target_id} "
        f"violated):\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


_LLM_ROW = re.compile(
    r"^(short|medium|long)\s+([\d.]+)%\s+(\S+)\s+([\d.]+)%\s+(\S+)\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)( FAIL)?\s*$")
_LLM_FAIL = re.compile(
    r"\[fail-components\] prompt=(\d) embeds=(\d)\(eSz=(\d),eFin=(\d)\) "
    r"logitSz=(\d) logitFin=(\d) maxabs=([\d.e+-]+[a-z]?) argmax=(\d) "
    r"ids=(\d) text=(\d)")


def _run_moss_llm_gate() -> None:
    """Shared-Qwen decoder fixture (manifest targets ``qwen_decode.shared_fixture``
    [components] + ``qwen_decode.shared_fixture.token_stream`` [exact tokens]).

    One ``build/moss_llm_test`` run produces one final row per fixture
    (columns: embeds% embeds-maxabs prefill% prefill-maxabs top5 argmax ids
    text) plus a ``[fail-components]`` stderr line per failing fixture.

    Enforcement is manifest-driven per the ACTIVE backend:

    * every backend: prompt ids exact, merged embeds 100% bitwise, prefill
      logits exact-width/finite with max-abs <= 8.0, prefill argmax exact
      (the component contract);
    * fixtures listed in the token-stream target's ``enforcement.<backend>
      .fixtures_exact``: additionally the FULL emitted id stream + text
      exact. On CUDA that is all three fixtures; on CPU only ``short``
      (cpp/moss/llm.cpp: 'CPU bf16 GEMMs are not bit-identical to cuBLAS
      and are a fallback only' — measured: medium/long flip near-tie
      argmax vs the eager reference, and NOT a K-step artifact). Medium/long
      id/text mismatches on CPU are REPORTED distinctly, never silent, and
      end-to-end CPU text exactness is carried by ``moss.intree.text``.
    """
    for target_id in (T_QWEN_DECODE, T_QWEN_TOKENS):
        _manifest_gate(target_id)
        pc.record_executed(target_id)
    bin_path = _REPO_ROOT / "build" / "moss_llm_test"
    if not bin_path.exists():
        for target_id in (T_QWEN_DECODE, T_QWEN_TOKENS):
            pc.record_unavailable(target_id, "build/moss_llm_test not built")
        pytest.skip("build/moss_llm_test not built (run cmake --build build -j)")

    from starling._ggml import backend_name
    device = backend_name()
    scope = "cpu" if device.upper() == "CPU" else "cuda"
    enforcement = _manifest_target(T_QWEN_TOKENS).get("enforcement", {})
    exact_fixtures = set(
        enforcement.get(scope, {}).get("fixtures_exact", ["short", "medium", "long"]))

    proc = subprocess.run(
        [str(bin_path), str(_REPO_ROOT)],
        capture_output=True, text=True, timeout=_CPP_COMPONENT_TIMEOUT,
    )
    rows = {m.group(1): m for m in
            (_LLM_ROW.match(line) for line in proc.stdout.splitlines()) if m}
    fails = {m.group(1): m for m in
             (_LLM_FAIL.search(line) for line in proc.stderr.splitlines()) if m}
    assert set(rows) == {"short", "medium", "long"}, (
        f"moss_llm_test output not parseable (backend {device}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    reported: list[str] = []
    for fixture in ("short", "medium", "long"):
        fail = fails.get(fixture)
        if fail is None:
            continue  # fixture fully passed, nothing to check
        prompt, embeds, _e_sz, _e_fin, logit_sz, logit_fin, maxabs, argmax, ids, text = \
            fail.groups()
        component_ok = (prompt == "1" and embeds == "1" and logit_sz == "1"
                        and logit_fin == "1" and argmax == "1"
                        and float(maxabs) <= 8.0)
        assert component_ok, (
            f"[{T_QWEN_DECODE}] {fixture}: component contract violated "
            f"(prompt={prompt} embeds={embeds} logitSz={logit_sz} "
            f"logitFin={logit_fin} maxabs={maxabs} argmax={argmax}):\n"
            f"{proc.stdout}\n{proc.stderr}")
        if fixture in exact_fixtures:
            pytest.fail(
                f"[{T_QWEN_TOKENS}] {fixture}: exact-token contract violated on "
                f"backend {device} (ids={ids} text={text}):\n{proc.stdout}\n"
                f"{proc.stderr}",
                pytrace=False)
        reported.append(
            f"[{T_QWEN_TOKENS}] {fixture}: id/text stream mismatch REPORTED "
            f"(not gated) on backend {device} per the manifest enforcement "
            f"note — components passed; see the CPU bf16 GEMM fallback note "
            f"in tests/native_parity_manifest.json")
    if reported:
        print("\n".join(reported))


@pytest.mark.skipif(not _starling_ggml_moss_available(),
                    reason="in-tree libstarling_ggml or MOSS GGUF unavailable")
def test_starling_ggml_moss_mel_components() -> None:
    """Component gate: log-mel vs the staged processor reference
    (``scripts/moss_golden_components.py`` → ``scripts/golden_to_raw.py``).

    Tolerances are the C++ binary's own recorded contract: at most 64
    mismatching bins out of ~951 680, each within one bf16 ULP measured at
    that bin's magnitude (per-bin, not a global absolute constant), no
    non-finite output. Manifest: ``moss.intree.mel`` /
    component_numerical_tolerance.
    """
    _run_cpp_component_test("moss_mel_test", T_MOSS_MEL)


@pytest.mark.skipif(not _starling_ggml_moss_available(),
                    reason="in-tree libstarling_ggml or MOSS GGUF unavailable")
def test_starling_ggml_moss_encoder_components() -> None:
    """Component gate: encoder hidden state + audio adapter embeds vs the
    staged eager reference. Tolerances: max-abs <= 0.02 (encoder_hidden,
    ~1.5x the observed eager-path spread after 32 bf16 attention layers) and
    <= 0.001 (audio_embeds); exact shapes; no non-finite values. Manifest:
    ``moss.intree.encoder`` / component_numerical_tolerance.
    """
    _run_cpp_component_test("moss_encoder_test", T_MOSS_ENCODER)


@pytest.mark.skipif(not _starling_ggml_moss_available(),
                    reason="in-tree libstarling_ggml or MOSS GGUF unavailable")
def test_starling_ggml_moss_llm_components() -> None:
    """Shared-Qwen decoder fixture: cpp/lib/qwen_decode vs the staged HF
    eager reference (prompt ids, merged embeds, prefill logits, emitted ids).

    Two manifest targets: ``qwen_decode.shared_fixture`` (component contract:
    bitwise-equal merged inputs_embeds, exact argmax, prefill logits max-abs
    <= 8.0 — enforced on EVERY backend) and
    ``qwen_decode.shared_fixture.token_stream`` (the FULL emitted token id
    stream + text EXACT, starting from the independent golden merged
    embedding so decoder failures are not hidden by the encoder/adapter
    gates — enforced on all fixtures on CUDA, on short on CPU where
    cpp/moss/llm.cpp documents the bf16 GEMM fallback; see _run_moss_llm_gate).
    This is the component-level contract shared by every Qwen-decoder engine
    (moss/ark/granite/qwen3/audex/s1).
    """
    _run_moss_llm_gate()


# --------------------------------------------------------------------------- #
# In-tree ARK-ASR-3B C API
# --------------------------------------------------------------------------- #
def _starling_ggml_ark_available() -> bool:
    try:
        from engines import StarlingGgmlArk
        return StarlingGgmlArk().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_ark_engine():
    if not _starling_ggml_ark_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_ARK_MODEL unavailable")
    from engines import StarlingGgmlArk
    engine = StarlingGgmlArk()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_ark_available(),
                    reason="in-tree libstarling_ggml or ARK GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_ark_text_parity(starling_ggml_ark_engine, name: str) -> None:
    """The in-tree C API returns the golden ARK transcript exactly.

    This deliberately has no tolerance: it gates Starling's own
    loader -> mel -> encoder -> adapter -> prompt -> LLM -> detokenizer pipeline
    for the ARK-ASR-3B model against the byte-exact reference.
    """
    _manifest_gate(T_ARK_TEXT)
    pc.record_executed(T_ARK_TEXT)
    golden_text = json.loads((GOLDEN / "ark_reference.json").read_text())[name]["text"]
    out = starling_ggml_ark_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_ARK_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# In-tree higgs-audio-v3-stt C API
# --------------------------------------------------------------------------- #
def _starling_ggml_higgs_available() -> bool:
    try:
        from engines import StarlingGgmlHiggs
        return StarlingGgmlHiggs().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_higgs_engine():
    if not _starling_ggml_higgs_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_HIGGS_MODEL unavailable")
    from engines import StarlingGgmlHiggs
    engine = StarlingGgmlHiggs()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_higgs_available(),
                    reason="in-tree libstarling_ggml or HIGGS GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_higgs_text_parity(starling_ggml_higgs_engine, name: str) -> None:
    """The in-tree C API returns the golden higgs transcript.

    Gates Starling's own mel -> Whisper encoder (+ avg pool) -> MLP projector
    -> ChatML prompt -> Qwen3 decoder (with qk_norm) -> BPE detokenizer pipeline
    for bosonai/higgs-audio-v3-stt against the eager reference captured by
    scripts/capture_golden_ref.py (golden/higgs_golden.json). Asserts exact text
    parity with no tolerance.
    """
    _manifest_gate(T_HIGGS_TEXT)
    pc.record_executed(T_HIGGS_TEXT)
    golden = json.loads((GOLDEN / "higgs_golden.json").read_text())
    golden_text = golden["fixtures"][name]["text"]
    out = starling_ggml_higgs_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_HIGGS_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# In-tree HojoAI/Hojo-ASR-V1 C API
# --------------------------------------------------------------------------- #
def _starling_ggml_hojo_available() -> bool:
    try:
        from engines import StarlingGgmlHojo
        return StarlingGgmlHojo().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_hojo_engine():
    if not _starling_ggml_hojo_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_HOJO_MODEL unavailable")
    from engines import StarlingGgmlHojo
    engine = StarlingGgmlHojo()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_hojo_available(),
                    reason="in-tree libstarling_ggml or HOJO GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_hojo_text_parity(starling_ggml_hojo_engine, name: str) -> None:
    """The in-tree C API returns the golden Hojo transcript.

    Gates Starling's own mel -> Qwen3-Omni audio tower -> WeNet Conformer
    bottleneck -> ln_speech -> Qwen3-4B decoder (beam-4, qk_norm) -> BPE
    detokenizer pipeline for HojoAI/Hojo-ASR-V1 against the eager reference
    captured by scripts/hojo_golden_components.py (golden/hojo_reference.json).
    Asserts exact text parity with no tolerance.
    """
    _manifest_gate(T_HOJO_TEXT)
    pc.record_executed(T_HOJO_TEXT)
    golden = json.loads((GOLDEN / "hojo_reference.json").read_text())
    golden_text = golden["fixtures"][name]["text"]
    out = starling_ggml_hojo_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_HOJO_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# In-tree granite-speech-4.1-2b C API
# --------------------------------------------------------------------------- #
def _starling_ggml_granite_available() -> bool:
    try:
        from engines import StarlingGgmlGranite
        return StarlingGgmlGranite().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_granite_engine():
    if not _starling_ggml_granite_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_GRANITE_MODEL unavailable")
    from engines import StarlingGgmlGranite
    engine = StarlingGgmlGranite()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_granite_available(),
                    reason="in-tree libstarling_ggml or granite GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_granite_text_parity(starling_ggml_granite_engine, name: str) -> None:
    """The in-tree C API returns the golden granite transcript.

    Gates Starling's own torchaudio mel (odd-drop + pair stack) -> CTC
    conformer encoder (block-local Shaw attention, mid-stack self-conditioned
    CTC) -> BLIP2 Q-Former projector -> Granite-4.0-1b decoder (bias-free, no
    qk-norm, untied lm_head, granite multipliers) pipeline including the serve
    chunk policy for long audio, against the stock-numerics reference captured
    by scripts/make_granite_golden.py (golden/granite_reference.json). Asserts
    exact text parity with no tolerance.
    """
    _manifest_gate(T_GRANITE_TEXT)
    pc.record_executed(T_GRANITE_TEXT)
    golden = json.loads((GOLDEN / "granite_reference.json").read_text())
    golden_text = golden["fixtures"][name]["text"]
    out = starling_ggml_granite_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_GRANITE_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# In-tree Qwen3-ASR-1.7B C API
# --------------------------------------------------------------------------- #
def _starling_ggml_qwen3_available() -> bool:
    try:
        from engines import StarlingGgmlQwen3
        return StarlingGgmlQwen3().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_qwen3_engine():
    if not _starling_ggml_qwen3_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_QWEN3_MODEL unavailable")
    from engines import StarlingGgmlQwen3
    engine = StarlingGgmlQwen3()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_qwen3_available(),
                    reason="in-tree libstarling_ggml or qwen3 GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_qwen3_text_parity(starling_ggml_qwen3_engine, name: str) -> None:
    """The in-tree C API returns the golden qwen3 transcript.

    Gates Starling's own whisper-style mel (128 bins, drop-last-frame rule,
    zero mel-pad to 100-frame chunks) -> chunked conv2d stack + windowed
    attention encoder (104-row windows) -> MLP projector -> Qwen3 decoder
    (bias-free, qk-norm, tied lm_head) pipeline including the serve chunk
    policy for long audio and the transcription_only text extraction, against
    the stock-numerics reference captured by scripts/make_qwen3_golden.py
    (golden/qwen3_reference.json). Asserts exact text parity with no
    tolerance.
    """
    _manifest_gate(T_QWEN3_TEXT)
    pc.record_executed(T_QWEN3_TEXT)
    golden = json.loads((GOLDEN / "qwen3_reference.json").read_text())
    golden_text = golden["fixtures"][name]["text"]
    out = starling_ggml_qwen3_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_QWEN3_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# In-tree Nemotron-Labs-Audex-2B C API
# --------------------------------------------------------------------------- #
def _starling_ggml_audex_available() -> bool:
    try:
        from engines import StarlingGgmlAudex
        return StarlingGgmlAudex().available
    except Exception:
        return False

@pytest.fixture(scope="module")
def starling_ggml_audex_engine():
    if not _starling_ggml_audex_available():
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_AUDEX_MODEL unavailable")
    from engines import StarlingGgmlAudex
    engine = StarlingGgmlAudex()
    engine.load()
    yield engine
    engine.close()

@pytest.mark.skipif(not _starling_ggml_audex_available(),
                    reason="in-tree libstarling_ggml or audex GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_audex_text_parity(starling_ggml_audex_engine, name: str) -> None:
    """The in-tree C API returns the golden audex transcript.

    Gates Starling's own whisper-style mel (128 bins, drop-last-frame rule,
    fixed 3000-frame clips) -> conv1d frontend + learned positional table +
    full-attention encoder (32 layers, 20 heads) -> avg-pooler (750 tokens)
    -> RMSNorm/relu2 projector -> Nemotron-Dense decoder (bias-free GQA,
    untied lm_head, relu2 MLP, single-round RMSNorm) pipeline including the
    serve chunk policy for long audio and the quote-extraction text cleanup,
    against the stock-numerics reference captured by
    scripts/make_audex_golden.py (golden/audex_reference.json). Asserts exact
    text parity with no tolerance.
    """
    _manifest_gate(T_AUDEX_TEXT)
    pc.record_executed(T_AUDEX_TEXT)
    golden = json.loads((GOLDEN / "audex_reference.json").read_text())
    golden_text = golden["fixtures"][name]["text"]
    out = starling_ggml_audex_engine._run_one(FIXTURES[name])
    pc.assert_exact_text(out, golden_text, target=T_AUDEX_TEXT, fixture=name)


# --------------------------------------------------------------------------- #
# Wave G regression: MOSS K-step decode must not access the KV cache / RoPE
# tables past max_cache when a block's remaining token budget < K.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _starling_ggml_moss_available(),
                    reason="in-tree libstarling_ggml or MOSS GGUF unavailable")
def test_starling_ggml_moss_kstep_cache_boundary() -> None:
    """The K-step decode must stay in-bounds at the max_cache boundary.

    Drives the C++ regression binary ``build/moss_kstep_oob_test`` (auto-built
    from ``cpp/tests/moss_kstep_oob_test.cpp``) which constructs a synthetic
    ``inputs_embeds`` with ``n_tokens + max_new_tokens == max_cache`` and runs
    ``greedy_generate`` for K=4 and K=8. On unpatched code the final K-step
    block's wasted tail steps write KV slots / read RoPE rows at indices
    ``>= max_cache`` and the resulting sticky CUDA illegal-memory-access makes
    ``greedy_generate`` return false. The fix caps each block's step count to
    the remaining budget so every device index stays ``< max_cache``.

    Gated on the same model/lib availability as the in-tree MOSS parity tests;
    a CPU backend is a vacuous pass (the K-step path is GPU-only, where the bug
    lives).
    """
    pc.record_executed(T_MOSS_KSTEP)
    bin_path = _REPO_ROOT / "build" / "moss_kstep_oob_test"
    if not bin_path.exists():
        pc.record_unavailable(T_MOSS_KSTEP, "build/moss_kstep_oob_test not built")
        pytest.skip("build/moss_kstep_oob_test not built (run cmake --build build -j)")
    proc = subprocess.run(
        [str(bin_path), str(_REPO_ROOT)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"moss_kstep_oob_test exited {proc.returncode} (K-step OOB regression):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# --------------------------------------------------------------------------- #
# In-tree Parakeet-tdt C API — the missing parity gate (Task 2, Phase 0).
# --------------------------------------------------------------------------- #
# Unlike the external ``GgmlParakeet`` engine (which drives mudler's
# parakeet.cpp server), ``StarlingGgmlParakeet`` drives Starling's OWN in-tree
# ``libstarling_ggml`` (built from ``cpp/``). Parakeet-tdt greedy decode is
# deterministic with no LLM/chunk path, so the short/medium gates below are
# EXACT (text AND non-blank id-stream equality against the golden
# ``parakeet_tdt_*_ids.pt`` captured by ``scripts/parakeet_tdt_golden.py``
# from HF ``model.generate``); the LONG fixture is the documented approximate
# path (transformers 5.14 SDPA kernel-path drift; the golden was regenerated
# from ``model.generate`` whose SDPA reduction order differs from the in-tree
# eager loop on 74 s decodes).
#
# This is the same artifact run before/after any decode change: if it is green
# on the serial-CPU engine and stays green (identical id-stream) on the K-step
# GPU engine, byte-exactness is established by construction — independent of
# any doc or commit message. Set ``STARLING_GGML_TDT_SERIAL=1`` to force the
# in-binary serial fallback for the within-binary A/B control.
#
# All three fixtures (short/medium/long) are exercised: short/medium take the
# K-step multistep fast path (T<=512); long (T=930) takes the byte-exact serial
# greedy loop. The id-stream gate covers both paths.
def _starling_ggml_parakeet_available() -> bool:
    try:
        from engines import StarlingGgmlParakeet
        return StarlingGgmlParakeet().available
    except Exception:
        return False


@pytest.fixture(scope="module")
def starling_ggml_parakeet_engine():
    """One in-tree StarlingGgmlParakeet engine for the whole module."""
    if not _starling_ggml_parakeet_available():
        pc.record_unavailable(T_PARAKEET_TEXT, "in-tree libstarling_ggml or parakeet GGUF unavailable")
        pc.record_unavailable(T_PARAKEET_IDS, "in-tree libstarling_ggml or parakeet GGUF unavailable")
        pytest.skip("in-tree libstarling_ggml or STARLING_GGML_PARAKEET_MODEL unavailable")
    from engines import StarlingGgmlParakeet
    engine = StarlingGgmlParakeet()
    engine.load()
    yield engine
    engine.close()


@pytest.mark.skipif(not _starling_ggml_parakeet_available(),
                    reason="in-tree libstarling_ggml or parakeet GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_parakeet_text_parity(
        starling_ggml_parakeet_engine, name: str) -> None:
    """The in-tree parakeet engine vs the golden transcript.

    short/medium: exact text (deterministic greedy decode, no LLM, no
    chunk-stitch — a diff is a real regression in the in-tree loader -> mel ->
    encoder -> TDT decode -> detokenize pipeline).
    long: the documented approximate path — transformers 5.14 SDPA
    kernel-path drift (the golden was regenerated from ``model.generate``;
    WER-verified benign at 3.18% == starling-vs-stock). Corpus-quality gate:
    difflib ratio >= 0.90 on the RAW transcript (no normalization); see the
    manifest tolerance block for metric/aggregation/margin.
    """
    _manifest_gate(T_PARAKEET_TEXT)
    pc.record_executed(T_PARAKEET_TEXT)
    golden_text = (GOLDEN / f"parakeet_tdt_{name}_text.txt").read_text()
    out = starling_ggml_parakeet_engine._run_one(FIXTURES[name])
    if name == "long":
        ratio = difflib.SequenceMatcher(None, out, golden_text).ratio()
        assert ratio >= 0.90, (
            f"in-tree parakeet transcript drift too high on {name} (ratio={ratio:.3f}):\n"
            f"  golden: {golden_text[:160]!r}\n  ggml:   {out[:160]!r}"
        )
    else:
        pc.assert_exact_text(out, golden_text, target=T_PARAKEET_TEXT, fixture=name)


@pytest.mark.skipif(not _starling_ggml_parakeet_available(),
                    reason="in-tree libstarling_ggml or parakeet GGUF unavailable")
@pytest.mark.parametrize("name", ["short", "medium", "long"])
def test_starling_ggml_parakeet_idstream_parity(
        starling_ggml_parakeet_engine, name: str) -> None:
    """The in-tree parakeet engine emits the golden CONTENT token stream exactly.

    Strictest gate short of blank-counting: the sequence of emitted CONTENT
    (non-blank) tokens must equal ``golden/parakeet_tdt_*_ids.pt`` element-for-
    element. This is stricter than the text gate (token-level, before
    detokenization) and catches any numeric drift / near-tie argmax flip that
    text-normalization would hide.

    Blanks (the TDT ``no-symbol-this-step`` marker, id ``blank_id`` = 8192 for
    parakeet-tdt-0.6b-v3) are EXCLUDED (the manifest's recorded suppression
    policy): the in-tree greedy loop and HF ``model.generate`` emit blanks at
    slightly different cadences but the CONTENT tokens are byte-identical.
    Blanks carry no linguistic content and are dropped by detokenization.

    long: the same documented SDPA-drift approximate path as the text gate —
    positional content-token match rate >= 0.65 divided by the LONGER stream
    (a decode that just truncates cannot score 1.0); the text gate is the
    primary quality contract for that fixture.
    """
    _manifest_gate(T_PARAKEET_IDS)
    pc.record_executed(T_PARAKEET_IDS)
    ipath = GOLDEN / f"parakeet_tdt_{name}_ids.pt"
    try:
        out_ids = starling_ggml_parakeet_engine._run_one_ids(FIXTURES[name])
    except RuntimeError as e:
        if "decode_ids" in str(e):
            pytest.skip("libstarling_ggml built without the decode_ids symbol")
        raise
    import torch
    golden_ids = [int(x) for x in torch.load(ipath).tolist()]
    # blank_id is a model-config constant (8192); prefer the golden meta if the
    # operator saved it, so the gate stays correct if the id ever moves.
    blank_id = 8192
    mpath = GOLDEN / f"parakeet_tdt_{name}_meta.pt"
    if mpath.exists():
        try:
            blank_id = int(torch.load(mpath).get("blank_id", blank_id))
        except Exception:
            pass
    out_nb = [t for t in out_ids if t != blank_id]
    golden_nb = [t for t in golden_ids if t != blank_id]
    if name == "long" and out_nb != golden_nb:
        # transformers 5.14 SDPA kernel-path drift on long decodes: near-tie
        # argmax flips from kernel reduction-order differences (~30% of
        # content tokens) that normalization absorbs (text CER = 0.0).
        rate = pc.token_match_rate(out_nb, golden_nb)
        assert rate >= 0.65, (
            f"in-tree parakeet content-token drift too high on {name} "
            f"(blank_id={blank_id}): match rate {rate:.3f} "
            f"(non-blank len ggml={len(out_nb)} golden={len(golden_nb)}):\n"
            f"  ggml[:16]={out_nb[:16]} golden[:16]={golden_nb[:16]}"
        )
        return
    if out_nb != golden_nb:
        n = min(len(out_nb), len(golden_nb))
        first = next((i for i in range(n) if out_nb[i] != golden_nb[i]), n)
        assert out_nb == golden_nb, (
            f"in-tree parakeet content-token mismatch on {name} "
            f"(blank_id={blank_id}): non-blank len ggml={len(out_nb)} "
            f"golden={len(golden_nb)}; first content diverge @ {first}: "
            f"ggml={out_nb[first:first + 8]} golden={golden_nb[first:first + 8]} "
            f"(raw id len ggml={len(out_ids)} golden={len(golden_ids)})"
        )


# --------------------------------------------------------------------------- #
# S1-mini (superwhisper/s1-mini) — text normalizer parity (ABI 6, slug "s1").
# --------------------------------------------------------------------------- #
STARLING_GGML_S1_MODEL = Path(os.environ.get(
    "STARLING_GGML_S1_MODEL",
    str(_REPO_ROOT / "models" / "s1-mini-bf16-exact.gguf"),
)).expanduser()


def _starling_ggml_s1_available() -> bool:
    try:
        from starling._ggml import available
        return available() and STARLING_GGML_S1_MODEL.exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def starling_ggml_s1_engine():
    if not _starling_ggml_s1_available():
        pytest.skip("in-tree libstarling_ggml (ABI 6) or s1 GGUF unavailable")
    from starling._ggml import S1, GgmlModel

    engine = GgmlModel(S1, str(STARLING_GGML_S1_MODEL))
    yield engine
    engine.close()


@pytest.mark.skipif(not _starling_ggml_s1_available(),
                    reason="in-tree libstarling_ggml (ABI 6) or s1 GGUF unavailable")
@pytest.mark.parametrize("tier", ["short", "medium", "long"])
def test_starling_ggml_s1_text_parity(starling_ggml_s1_engine, tier: str) -> None:
    """The in-tree C engine returns the stock greedy normalization text.

    Gates the whole text path: C++ BPE encode (Qwen pre-tokenizer regex +
    byte-level merges) + baked chat template + plain embedding lookup +
    Qwen3 trunk greedy decode stopping on <|im_end|> OR <|endoftext|>, against
    the eager stock golden captured by starling.s1.golden (which runs the
    model-card quickstart verbatim). Asserts exact text parity.
    """
    pc.record_executed(T_S1_TEXT)
    from starling.s1.golden import GREEDY_TEXT, load_golden_text

    sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
    import s1_transcripts as fx  # noqa: E402

    golden_text = load_golden_text(GREEDY_TEXT.format(tier=tier))
    out = starling_ggml_s1_engine.normalize_text(fx.LENGTH_TIERS[tier])
    pc.assert_exact_text(out, golden_text, target=T_S1_TEXT, fixture=tier)


@pytest.mark.skipif(not _starling_ggml_s1_available(),
                    reason="in-tree libstarling_ggml (ABI 6) or s1 GGUF unavailable")
def test_starling_ggml_s1_control_matrix_smoke(starling_ggml_s1_engine) -> None:
    """SMOKE check (renamed; not an accuracy certification): every trained
    control combination (4 styling x 2 structure x 2 context) produces
    non-degenerate output on the trained path (the transcript has real
    content, so an empty return IS the hallucination-shaped degenerate case),
    and unknown control values are rejected with a clear error.

    The accuracy contract for s1 is the exact-text parity gate above; this
    test only certifies non-degeneracy on the 16 control cells.
    """
    pc.record_executed(T_S1_CONTROL)
    import s1_transcripts as fx  # noqa: E402  (tests/fixtures on sys.path)

    n = 0
    for transcript, styling, structure, context in fx.CONTROL_MATRIX:
        out = starling_ggml_s1_engine.normalize_text(
            transcript, styling, structure, context)
        assert isinstance(out, str) and out.strip(), (
            f"degenerate (empty) output for {styling}/{structure}/{context}"
        )
        n += 1
    assert n == 16

    for bad in ({"styling": "pirate"}, {"structure": "table"}, {"context": "space"}):
        with pytest.raises(RuntimeError, match="unknown"):
            starling_ggml_s1_engine.normalize_text(
                "hello", bad.get("styling"), bad.get("structure"), bad.get("context"))


# --------------------------------------------------------------------------- #
# Coverage accounting (must run AFTER the gates above)
# --------------------------------------------------------------------------- #
def test_parity_coverage_accounting(request) -> None:
    """Required targets must not silently pass (issue #167 acceptance).

    Summarizes executed vs unavailable coverage for every manifest target and
    fails when a REQUIRED target with all assets present executed ZERO tests
    or an asset hash disagrees with the manifest pin. Genuinely unavailable
    coverage is reported distinctly and only fails under
    ``STARLING_PARITY_STRICT=1``. An explicitly deselected run (-k/-m)
    suspends the zero-executed enforcement (scoping is deliberate, not
    silent).
    """
    report, failures = pc.coverage_lines(
        pc.load_manifest(), deselection=pc.pytest_deselection(request.config))
    if report:
        print("\n[native-parity coverage report — distinct, not silent]")
        for line in report:
            print(f"  {line}")
    if pc.strict_mode():
        assert not failures, f"strict-mode coverage failures:\n{failures}"
    else:
        hard = [f for f in failures if "UNAVAILABLE" not in f]
        assert not hard, (
            f"required-target coverage failures:\n{hard}")
