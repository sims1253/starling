#!/usr/bin/env python3
"""Run SONAR-OSS evaluation over a grid of Starling models / backends / quants.

Each variant (one ``starling-serve`` model x ggml backend x GGUF) is launched as
its own process on a free port, registered into SONAR as a model adapter, scored
over a SONAR-prepared dataset, then torn down. Results land in
``results/<tag>/<variant>/scores_<model>.json`` and are rendered with SONAR's own
``leaderboard`` aggregation, so the table carries only measured numbers.

Typical use (from the repo root)::

    # one-time: create the isolated SONAR venv
    uv sync --project benchmarks/sonar

    # one-time: prepare the native SONAR dataset (FLEURS en, capped)
    uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
        --prepare-only --language en --max-samples 100

    # run the parakeet quant ladder on the Vulkan backend
    uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
        --preset quants --language en --max-samples 100

    # compare CPU vs Vulkan at two quants
    uv run --project benchmarks/sonar python benchmarks/sonar/run_sonar.py \
        --preset backends --language en --max-samples 100

The adapter works against any server exposing ``POST /inference``, so a CUDA
``starling-serve`` build or ``python -m starling.server`` can be added by
pointing a backend at that binary in :mod:`variants`.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REPO_ROOT = HERE.parents[1]

import requests  # noqa: E402

from starling_sonar_adapter import register_starling_model  # noqa: E402
from variants import (  # noqa: E402
    DEFAULT_BINARIES,
    DEFAULT_GGUF_DIR,
    DEFAULT_PRESET,
    PRESETS,
    StarlingVariant,
    build_variants,
)

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# server lifecycle
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    """Reserve an ephemeral port, then close it so the server can bind it.

    Small race window, acceptable for a local harness: the OS will not hand the
    same port to another process in the microseconds before ``Popen``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _health(base_url: str, timeout_s: float = 2.0) -> dict | None:
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout_s)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def _wait_healthy(base_url: str, proc: subprocess.Popen, timeout_s: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"starling-serve exited rc={proc.returncode} before becoming healthy; see {log_path}"
            )
        health = _health(base_url)
        if health and health.get("loaded") and health.get("phase") in ("ready", "busy"):
            return
        time.sleep(0.5)
    raise TimeoutError(f"starling-serve not healthy after {timeout_s:.0f}s; see {log_path}")


def _silent_wav(seconds: float = 3.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))
    return buffer.getvalue()


def _warmup(base_url: str, seconds: float = 3.0) -> None:
    """One untimed inference so shader/allocator/model setup is not billed to the first clip.

    SONAR times every clip; the ggml backends lazily build pipelines (Vulkan) or
    kernels (CPU) on first use, which would otherwise land on clip #1. Failures
    are logged by the caller, not fatal — the run still produces numbers.
    """
    files = {"file": ("warmup.wav", _silent_wav(seconds), "audio/wav")}
    response = requests.post(f"{base_url}/inference", files=files, timeout=600.0)
    response.raise_for_status()


def _disable_audio_quality() -> None:
    """Replace SONAR's per-clip audio-quality pass with the empty record.

    SONAR recomputes SNR/UTMOS/SQUIM/DNSMOS for every model in a run, on CPU
    here; across a multi-variant sweep that is hours of work with no bearing on
    WER/RTFx. The per-utterance CSV schema is unchanged (the quality columns
    stay empty). This is the one place the harness reaches into SONAR internals
    and it is opt-in via --skip-audio-quality.
    """
    from psdn_sonar.evaluators import single_speaker

    def _empty(item: dict) -> tuple:
        return item["audio_path"], dict(single_speaker._EMPTY_AUDIO_QUALITY)

    single_speaker.SingleSpeakerEvaluator._compute_audio_quality = staticmethod(_empty)


def _start_server(
    variant: StarlingVariant, *, log_path: Path, timeout_s: float
) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        str(variant.binary),
        "--model",
        variant.model_slug,
        "--gguf",
        str(variant.gguf),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The child inherits its own descriptor; the parent handle is closed by the
    # `with` block right after spawn so a long sweep does not leak one per run.
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
    try:
        _wait_healthy(base_url, proc, timeout_s, log_path)
    except Exception:
        _stop_server(proc)
        raise
    return proc, base_url


def _stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


# --------------------------------------------------------------------------- #
# dataset preparation (delegates to `psdn-sonar discover`)
# --------------------------------------------------------------------------- #
def _sonar_cli(args: list[str]) -> int:
    """Invoke the installed ``psdn-sonar`` CLI in a child process."""
    exe = shutil.which("psdn-sonar")
    if exe:
        cmd = [exe, *args]
    else:
        cmd = [sys.executable, "-c", "from psdn_sonar.cli import main; main()", *args]
    return subprocess.run(cmd).returncode


def _ensure_dataset(
    language: str, dataset: str, max_samples: int, data_root: Path, *, force: bool
) -> Path:
    """Prepare (or reuse) the SONAR TSV for ``language``/``dataset``.

    SONAR's preparer writes ``<data_root>/<language>/<dataset>/test.tsv`` plus
    extracted WAVs. ``max_samples`` bounds each split; the underlying HF split
    is still downloaded in full on first run (SONAR's documented behavior).
    """
    tsv = data_root / language / dataset / "test.tsv"
    if tsv.is_file() and not force:
        print(f"[sonar] reusing dataset TSV {tsv}")
        return tsv
    print(
        f"[sonar] preparing dataset {dataset}/{language} (max_samples={max_samples}) -> {data_root}"
    )
    rc = _sonar_cli(
        [
            "discover",
            "--language",
            language,
            "--datasets",
            dataset,
            "--output",
            str(data_root / language),
            "--max-samples",
            str(max_samples),
        ]
    )
    if rc != 0 or not tsv.is_file():
        raise RuntimeError(f"`psdn-sonar discover` failed (rc={rc}); expected TSV at {tsv}")
    return tsv


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _evaluate_variant(
    variant: StarlingVariant,
    *,
    base_url: str,
    tsv: Path,
    language: str,
    max_samples: int,
    out_dir: Path,
) -> dict:
    """Register the running server as a SONAR model and score the dataset."""
    from psdn_sonar.benchmark.submission import SubmissionConfig
    from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

    name = register_starling_model(
        variant.sonar_name,
        base_url,
        model_snapshot=variant.describe,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant.json").write_text(
        json.dumps(
            {
                "label": variant.label,
                "sonar_model": name,
                "model_slug": variant.model_slug,
                "backend": variant.backend,
                "gguf": str(variant.gguf),
                "binary": str(variant.binary),
                "base_url": base_url,
                "dataset_tsv": str(tsv),
                "language": language,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # SONAR derives `device` from local torch ("cpu" here, since this box has no
    # CUDA torch), which hides the ggml backend behind a Vulkan run. Build the
    # submission explicitly so the backend is what the artifact records; the
    # gguf/quant live in model_snapshot and in variant.json.
    submission = SubmissionConfig.from_env(
        provider="starling",
        model_snapshot=variant.describe,
        protocol="batch",
        inference_params={"language_code": language},
    ).model_copy(update={"device": variant.backend})
    return SingleSpeakerEvaluator.run_evaluation(
        tsv_path=str(tsv),
        output_dir=str(out_dir),
        models=[name],
        max_samples=max_samples,
        compute_sem=True,
        language=language,
        submission=submission,
    )


def _render_leaderboard(run_dir: Path, language: str) -> list[dict]:
    """Aggregate every ``scores_*.json`` under ``run_dir`` with SONAR's own logic."""
    from psdn_sonar.benchmark.leaderboard import (
        build_leaderboard,
        collect_scores,
        render_leaderboard,
        rows_as_json,
    )

    loaded, skipped = collect_scores([run_dir])
    for message in skipped:
        print(f"[sonar] WARNING: {message}")
    if not loaded:
        return []
    rows = build_leaderboard(loaded, language=language, sort="wer")
    if not rows:
        print(f"[sonar] no rows for language {language!r}")
        return []
    print("\n" + render_leaderboard(rows, sort="wer"))
    (run_dir / "leaderboard.json").write_text(rows_as_json(rows), encoding="utf-8")
    (run_dir / "leaderboard.md").write_text(render_leaderboard(rows, sort="wer"), encoding="utf-8")
    return json.loads(rows_as_json(rows))


def _write_manifest(run_dir: Path, rows: list[dict]) -> None:
    """Merge leaderboard rows with each variant's engine/quant provenance."""
    meta: dict[str, dict] = {}
    for path in run_dir.rglob("variant.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta[entry["sonar_model"]] = entry
    merged = [{**row, "variant": meta.get(row["model_name"], {})} for row in rows]
    (run_dir / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets: " + ", ".join(sorted(PRESETS)),
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    parser.add_argument(
        "--models", nargs="+", help="override preset model slugs (e.g. parakeet moss)"
    )
    parser.add_argument("--quants", nargs="+", help="override preset quant tags (e.g. bf16 q4_0)")
    parser.add_argument("--backends", nargs="+", help="override preset backends (e.g. vulkan cpu)")
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_GGUF_DIR)
    parser.add_argument(
        "--binary",
        action="append",
        default=[],
        metavar="BACKEND=PATH",
        help="override a starling-serve binary (repeatable), e.g. --binary vulkan=/path/to/starling-serve",
    )
    parser.add_argument(
        "--language", default="en", help="ISO 639-1 language for the SONAR normalizer"
    )
    parser.add_argument(
        "--dataset", default="fleurs", help="SONAR discover dataset name (default fleurs)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="cap on samples per split (discover) and per run (eval); 0 = all",
    )
    parser.add_argument(
        "--tsv", type=Path, help="use an existing SONAR TSV instead of preparing one"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=HERE / "data",
        help="directory `discover` writes prepared datasets under",
    )
    parser.add_argument("--tag", default=None, help="run name (default: timestamp)")
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument(
        "--server-timeout", type=float, default=300.0, help="seconds to wait for model load"
    )
    parser.add_argument(
        "--no-warmup", action="store_true", help="skip the untimed warmup inference"
    )
    parser.add_argument(
        "--skip-audio-quality",
        action="store_true",
        help="skip SONAR's per-clip SNR/MOS pass (much faster; WER/CER/POSEIDON unaffected)",
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="prepare the dataset, then exit"
    )
    parser.add_argument("--dry-run", action="store_true", help="list the variant grid, then exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    binaries = dict(DEFAULT_BINARIES)
    for override in args.binary:
        key, _, value = override.partition("=")
        if not key or not value:
            raise SystemExit(f"--binary expects BACKEND=PATH, got {override!r}")
        binaries[key] = Path(value).expanduser()

    variants = build_variants(
        args.preset,
        models=args.models,
        quants=args.quants,
        backends=args.backends,
        gguf_dir=args.gguf_dir,
        binaries=binaries,
    )

    print(f"[sonar] preset={args.preset} language={args.language} variants={len(variants)}")
    for variant in variants:
        print(f"  - {variant.sonar_name:40s} {variant.gguf.name}  [{variant.backend}]")
    if args.dry_run:
        return 0

    tsv = args.tsv
    if tsv is None:
        tsv = _ensure_dataset(
            args.language,
            args.dataset,
            args.max_samples,
            args.data_root,
            force=False,
        )
    if not tsv.is_file():
        raise SystemExit(f"TSV not found: {tsv}")
    if args.prepare_only:
        print(f"[sonar] dataset ready: {tsv}")
        return 0

    tag = args.tag or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = args.results_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sonar] results -> {run_dir}")

    if args.skip_audio_quality:
        _disable_audio_quality()
        print("[sonar] audio-quality pass disabled")

    failures: list[tuple[str, str]] = []
    for index, variant in enumerate(variants, 1):
        print(f"\n[sonar] ({index}/{len(variants)}) {variant.sonar_name} — {variant.describe}")
        log_path = run_dir / "logs" / f"{variant.label}.log"
        out_dir = run_dir / variant.label
        proc: subprocess.Popen | None = None
        try:
            proc, base_url = _start_server(
                variant, log_path=log_path, timeout_s=args.server_timeout
            )
            if not args.no_warmup:
                try:
                    _warmup(base_url)
                except Exception as exc:  # noqa: BLE001 - warmup is best-effort
                    print(f"[sonar]   warmup failed (continuing): {exc}")
            metrics = _evaluate_variant(
                variant,
                base_url=base_url,
                tsv=tsv,
                language=args.language,
                max_samples=args.max_samples,
                out_dir=out_dir,
            )
            summary = metrics.get(variant.sonar_name, {}).get("summary", {})
            print(f"[sonar]   WER={summary.get('avg_wer')}")
        except Exception as exc:  # noqa: BLE001 - one variant must not kill the sweep
            print(f"[sonar]   FAILED: {exc}")
            failures.append((variant.sonar_name, str(exc)))
        finally:
            _stop_server(proc)

    rows = _render_leaderboard(run_dir, args.language)
    _write_manifest(run_dir, rows)
    print(f"\n[sonar] wrote {run_dir / 'manifest.json'}")
    if failures:
        print(f"[sonar] {len(failures)} variant(s) failed:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
    return 1 if failures and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
