#!/usr/bin/env python3
"""One-off generator for benchmarks/campaign/profiles/*.json (issue #176 §2).
Run from the repo root; overwrites the generated profiles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import PROFILE_SCHEMA  # noqa: E402

OUT = Path(__file__).resolve().parent / "profiles"

WORKLOADS = [
    {"id": "clips-short-medium-long", "description":
        "tests/fixtures short/medium/long clips through the bench gate",
     "status": "available"},
    {"id": "warm-dictation", "description": "warm dictation session",
     "status": "unavailable", "owner_issue": "#226"},
    {"id": "session-5-10min", "description": "5-10 minute paced sessions",
     "status": "unavailable", "owner_issue": "#226"},
    {"id": "preload-before-speech", "description": "model preload before speech starts",
     "status": "unavailable", "owner_issue": "#229"},
    {"id": "speech-during-load", "description": "speech arriving while the model loads",
     "status": "unavailable", "owner_issue": "#229"},
    {"id": "activation-to-capture", "description": "activation-to-capture latency",
     "status": "unavailable", "owner_issue": "#310"},
    {"id": "first-partial", "description": "first partial transcript delay",
     "status": "unavailable", "owner_issue": "#310"},
    {"id": "catch-up-backlog", "description": "catch-up of accumulated audio backlog",
     "status": "unavailable", "owner_issue": "#310"},
    {"id": "stop-to-final", "description": "stop-to-final latency; sustained backlog must not drop audio",
     "status": "unavailable", "owner_issue": "#310"},
]

HELDOUT = {
    "env": "STARLING_HELDOUT_DIR",
    "sha256": None,
    "description": "FLEURS held-out slice (default corpus + languages beyond the "
                   "100-clip English smoke set); used ONLY by finalize, never per attempt",
}

QUALITY_KERNEL = {
    "policy": "exact_numerical_contract",
    "notes": "same-model kernel changes: transcripts must match the baseline "
             "exactly. Quant-change campaigns switch to quant_wer_gate: "
             "+0.2 absolute WER points per language unless stricter, and no "
             "promotion from the 100-clip English smoke set alone.",
}
QUALITY_QUANT = {
    "policy": "quant_wer_gate",
    "notes": "quant changes: +0.2 absolute WER points per language unless "
             "stricter; no promotion from the 100-clip English smoke set alone.",
}

BUDGETS_NOTEBOOK = {
    "max_attempts": 6, "campaign_wall_clock_s": 28800.0, "attempt_wall_clock_s": 3600.0,
    "gate_timeout_s": 1800.0, "agent_timeout_s": 1800.0, "token_budget": None,
    "cooldown_max_s": 900.0,
}
BUDGETS_PIXEL = {
    "max_attempts": 6, "campaign_wall_clock_s": 28800.0, "attempt_wall_clock_s": 3600.0,
    "gate_timeout_s": 1800.0, "agent_timeout_s": 1800.0, "token_budget": None,
    "cooldown_max_s": 1200.0,
}
THRESHOLDS_NOTEBOOK = {"max_temp_c": 85.0, "min_mem_available_mb": 2048.0}
THRESHOLDS_PIXEL = {"max_temp_c": 43.0, "min_battery_pct": 40.0, "min_mem_available_mb": 1500.0}

NB_BUILD_FAST = (
    "cmake -B build-campaign -DSTARLING_SERVE=ON -DSTARLING_FAST=ON "
    "-DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF "
    "&& cmake --build build-campaign --parallel ${STARLING_CAMPAIGN_BUILD_JOBS:-4} "
    "--target starling-serve"
)
NB_BUILD_GGML = (
    "cmake -B build-campaign -DSTARLING_SERVE=ON "
    "-DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF "
    "&& cmake --build build-campaign --parallel ${STARLING_CAMPAIGN_BUILD_JOBS:-4} "
    "--target starling-serve"
)
NB_BUILD_FIXTURE = (
    "cmake -B build-campaign -DSTARLING_SERVE=ON -DSTARLING_GGML_TESTS=ON "
    "-DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF "
    "&& cmake --build build-campaign --parallel ${STARLING_CAMPAIGN_BUILD_JOBS:-4} "
    "--target starling-serve-contract-fixture"
)
PX_BUILD = (
    'NDK=$(ls -d "${ANDROID_NDK_HOME:-${ANDROID_HOME:-/opt/android-sdk}/ndk}"/* 2>/dev/null '
    "| sort -V | tail -1); [ -n \"$NDK\" ] || { echo 'Android NDK not found "
    "(set ANDROID_NDK_HOME or ANDROID_HOME)' >&2; exit 1; }; "
    "cmake -S . -B build-campaign -G Ninja "
    '-DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" '
    "-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_BUILD_TYPE=Release "
    "-DSTARLING_FAST=ON -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF "
    "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm "
    "-DSTARLING_GGML_TESTS=OFF -DSTARLING_QUANTIZE=OFF "
    # Static: each arm is one self-contained binary, never the libggml*.so
    # some earlier push left on the phone.
    "-DBUILD_SHARED_LIBS=OFF "
    "&& cmake --build build-campaign --parallel ${STARLING_CAMPAIGN_BUILD_JOBS:-4} "
    "--target starling-bench"
)
PX_BUILD_GGML = PX_BUILD.replace("-DSTARLING_FAST=ON ", "")

# Fixture audio is generated (tests/fixtures/make_fixtures.py; *.wav is
# gitignored), so it is NOT in the git-archived trusted tree. Profiles take it
# from $STARLING_FIXTURES_DIR as sealed artifacts: preflight requires the
# files, start hashes them, and every attempt re-verifies the hashes.
FIXTURE_WAVS = ("short.wav", "medium.wav", "long.wav")
FIXTURE_ARTIFACTS = [
    {"name": f"fixture-{w.split('.')[0]}", "path": f"${{STARLING_FIXTURES_DIR}}/{w}",
     "sha256": None}
    for w in FIXTURE_WAVS
]

KILL_BENCHES = {
    "argv": ["adb", "shell",
             "for p in $(pidof starling-bench starling-bench-base starling-bench-cand); "
             "do kill -9 $p; done"],
    "timeout_s": 30,
}


def notebook_gates(slug: str, gguf: str) -> list[dict]:
    return [
        {
            "name": "serve-contract", "stage": "correctness",
            "argv": [
                "{python}", "{trusted}/benchmarks/campaign/gates/serve_contract_smoke.py",
                "--binary", "{candidate}/starling-serve",
                "--baseline-binary", "{baseline}/starling-serve",
                "--model", slug, "--gguf", f"${{STARLING_MODELS_DIR}}/{gguf}",
                *[a for w in FIXTURE_WAVS for a in ("--wav", f"${{STARLING_FIXTURES_DIR}}/{w}")],
            ],
            "rules": [{"metric": "contract_ok", "op": "==", "value": 1},
                      {"metric": "transcripts_match", "op": "==", "value": 1}],
            "required": True, "timeout_s": 900.0,
        },
        {
            "name": "perf", "stage": "perf",
            "argv": [
                "{python}", "{trusted}/benchmarks/campaign/gates/experiments_ab.py",
                "--base-bin", "{best}/starling-serve",
                "--cand-bin", "{candidate}/starling-serve",
                "--run-dir", "{attempt_dir}/experiment",
                "--model", slug, "--gguf", f"${{STARLING_MODELS_DIR}}/{gguf}",
            ],
            "rules": [{"metric": "verdict", "op": "==", "value": "pass"}],
            "required": True, "objective": True, "timeout_s": 1800.0,
        },
    ]


def pixel_gates(slug: str, gguf: str, engine: str, wav: str = "medium.wav") -> list[dict]:
    return [
        {
            "name": "phone-ab", "stage": "perf",
            "argv": [
                "bash", "{trusted}/benchmarks/campaign/gates/phone_bench_ab.sh",
                "--base-bin", "{best}/starling-bench",
                "--cand-bin", "{candidate}/starling-bench",
                "--model", slug, "--gguf", f"${{STARLING_MODELS_DIR}}/{gguf}",
                "--wav", "${STARLING_FIXTURES_DIR}/" + wav,
                "--engine", engine,
            ],
            "rules": [
                {"metric": "transcripts_match", "op": "==", "value": 1},
                {"metric": "total_ms_delta_pct", "op": "<=", "value": -5.0},
            ],
            "required": True, "objective": True, "timeout_s": 1800.0,
            "remote_cleanup": KILL_BENCHES,
        },
    ]


ENERGY_GATE = {
    # Finalize only: phone_energy.sh adds two model loads and a long idle
    # window, and every load costs the PowerVR driver's per-boot health
    # budget (#325) — too expensive to repeat for every attempt.
    "name": "energy-per-transcription", "stage": "resource", "required": False,
    "phase": "finalize",
    "shell": (
        "adb push {candidate}/starling-bench /data/local/tmp/starling/starling-bench-cand "
        "&& OUT={attempt_dir} bash {trusted}/benchmarks/fast_engine/phone_energy.sh "
        "> {attempt_dir}/energy.stdout 2>&1; rc=$?; cat {attempt_dir}/energy.stdout; "
        "sed -n 's/^fast: .* = \\([0-9.]*\\) mWh\\/transcription.*/"
        "METRIC energy_mwh_per_transcription=\\1/p' {attempt_dir}/energy.stdout; "
        "if [ $rc -eq 0 ]; then exit 0; "
        'elif grep -qE "unplug the phone|battery status" {attempt_dir}/energy.stdout; then '
        'echo "DIVERGENCE: energy inconclusive (charging)" >&2; exit 3; else exit $rc; fi'
    ),
    "rules": [], "timeout_s": 1800.0,
    "remote_cleanup": KILL_BENCHES,
}


def profile(pid, model_name, hf_id, slug, status, tracking, notes, device, backend,
            engine, artifacts, build, gates, objectives, quality=None,
            blocked_on=None, blocked_reason=None, thresholds=None, defaults=None,
            device_expect=None):
    p = {
        "schema": PROFILE_SCHEMA,
        "id": pid,
        "model": {"name": model_name, "hf_id": hf_id},
        "tracking_issue": tracking,
        "status": status,
        "readiness_notes": notes,
        "slug": slug,
        "device": device,
        "backend": backend,
        "engine": engine,
        "artifacts": artifacts,
        "build": build,
        "gates": gates,
        "workloads": WORKLOADS,
        "objectives": objectives,
        "quality_policy": quality or QUALITY_KERNEL,
        "heldout": HELDOUT,
        "defaults": defaults,
        "thresholds": thresholds,
    }
    if blocked_on:
        p["blocked_on"] = blocked_on
    if blocked_reason:
        p["blocked_reason"] = blocked_reason
    if device_expect:
        p["device_expect"] = device_expect
    return p


MODELS = {
    "parakeet": {
        "name": "Parakeet-TDT-0.6B-v3", "hf": "nvidia/parakeet-tdt-0.6b-v3",
        "slug": "parakeet", "status": "ready", "tracking": "#349",
        "notes_nb": "fast Vulkan engine validated on RADV; ggml fallback available",
        "notes_px": "fast engine on PowerVR (DXT-48-1536); known wedge state #325 stops the run",
        "gguf_nb": "parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf",
        "gguf_px": "parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf",
        "allowed": ["cpp/parakeet/**", "cpp/fast/**"],
    },
    "moss": {
        "name": "MOSS-Transcribe-preview-2B", "hf": "OpenMOSS-Team/MOSS-Transcribe-preview-2B",
        "slug": "moss", "status": "ready", "tracking": "#350",
        "notes_nb": "fast Vulkan engine validated on RADV",
        "notes_px": "fast engine on PowerVR; decode ms/token and energy tracked",
        "gguf_nb": "moss-transcribe-preview-2b-q4e8-fullimx.gguf",
        "gguf_px": "moss-transcribe-preview-2b-q4e8-fullimx.gguf",
        "allowed": ["cpp/moss/**", "cpp/fast/**"],
    },
    "moss-diarize": {
        "name": "MOSS-Transcribe-Diarize", "hf": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "slug": None, "status": "blocked", "tracking": "#351",
        "blocked_on": "native-support",
        "blocked_reason": "no native (ggml/fast) support: diarization architecture not ported",
        "notes_nb": "blocked until native support lands",
        "notes_px": "blocked until native support lands",
        "gguf_nb": "moss-transcribe-diarize-q8_0.gguf",
        "gguf_px": "moss-transcribe-diarize-q4-fullimx.gguf",
        "allowed": ["cpp/moss/**"],
    },
    "qwen3": {
        "name": "Qwen3-ASR-1.7B", "hf": "Qwen/Qwen3-ASR-1.7B",
        "slug": "qwen3", "status": "ready", "tracking": "#352",
        "notes_nb": "ggml engine (no fast engine); CPU/Vulkan ggml backend",
        "notes_px": "ggml engine on the phone CPU (6 threads)",
        "gguf_nb": "qwen3-asr-1.7b-q8_0.gguf",
        "gguf_px": "qwen3-asr-1.7b-q4-fullimx.gguf",
        "allowed": ["cpp/qwen3/**", "cpp/lib/**"],
    },
    "qwen3-06b": {
        "name": "Qwen3-ASR-0.6B", "hf": "Qwen/Qwen3-ASR-0.6B",
        "slug": None, "status": "blocked", "tracking": "#353",
        "blocked_on": "port-validation",
        "blocked_reason": "smaller sibling not ported/validated natively",
        "notes_nb": "blocked until the 0.6B port lands",
        "notes_px": "blocked until the 0.6B port lands",
        "gguf_nb": "qwen3-asr-0.6b-q8_0.gguf",
        "gguf_px": "qwen3-asr-0.6b-q4-fullimx.gguf",
        "allowed": ["cpp/qwen3/**"],
    },
    "ark06": {
        "name": "ARK-ASR-0.6B", "hf": "Audio8/ARK-ASR-0.6B",
        "slug": "ark06", "status": "ready", "tracking": "#354",
        "notes_nb": "ggml engine; parity pending GPU verification",
        "notes_px": "ggml engine; parity pending GPU verification",
        "gguf_nb": "ark-asr-0.6b-q8_0.gguf",
        "gguf_px": "ark-asr-0.6b-q4-fullimx.gguf",
        "allowed": ["cpp/ark/**", "cpp/lib/**"],
    },
    "ark": {
        "name": "ARK-ASR-3B", "hf": "AutoArk-AI/ARK-ASR-3B",
        "slug": "ark", "status": "ready", "tracking": "#355",
        "notes_nb": "ggml engine",
        "notes_px": "ggml engine on the phone CPU",
        "gguf_nb": "ark-asr-3b-q8_0.gguf",
        "gguf_px": "ark-asr-3b-q4-fullimx.gguf",
        "allowed": ["cpp/ark/**", "cpp/lib/**"],
    },
    "parakeet-unified": {
        "name": "Parakeet Unified EN 0.6B", "hf": "nvidia/parakeet-unified-en-0.6b",
        "slug": None, "status": "blocked", "tracking": "#358",
        "blocked_on": "native-port",
        "blocked_reason": "Python-only port (CUDA graphs); no native ggml port",
        "notes_nb": "blocked until the native port (#358) lands",
        "notes_px": "blocked until the native port (#358) lands",
        "gguf_nb": "parakeet-unified-en-0.6b-q8_0.gguf",
        "gguf_px": "parakeet-unified-en-0.6b-q4-fullimx.gguf",
        "allowed": ["cpp/parakeet/**"],
    },
}


def model_profiles() -> list[dict]:
    out = []
    for key, m in MODELS.items():
        # --- notebook ---
        fast = key in ("parakeet", "moss") and m["status"] == "ready"
        engine_nb = "fast" if fast else "ggml"
        gates_nb = notebook_gates(m["slug"], m["gguf_nb"]) if m["status"] == "ready" else []
        out.append(profile(
            f"{key}--notebook", m["name"], m["hf"], m["slug"], m["status"], m["tracking"],
            m["notes_nb"], "notebook", "vulkan" if fast else "cpu", engine_nb,
            [{"name": "gguf", "path": f"${{STARLING_MODELS_DIR}}/{m['gguf_nb']}",
              "sha256": None}, *FIXTURE_ARTIFACTS] if m["status"] == "ready" else [],
            {"shell": NB_BUILD_FAST if fast else NB_BUILD_GGML,
             "artifacts_out": ["build-campaign/starling-serve"]},
            gates_nb,
            ([{"name": "quality", "gate": "serve-contract", "metric": "contract_ok",
               "direction": "higher", "min_improvement": 0.0},
              {"name": "end-to-end latency", "gate": "perf", "metric": "verdict",
               "direction": "higher", "min_improvement": 0.0}]
             if gates_nb else []),
            blocked_on=m.get("blocked_on"), blocked_reason=m.get("blocked_reason"),
            thresholds=THRESHOLDS_NOTEBOOK,
            defaults={"allowed_paths": m["allowed"], "budgets": BUDGETS_NOTEBOOK},
            device_expect="Ryzen 5 PRO 5650U",
        ))
        # --- pixel ---
        engine_px = "fast" if fast else "ggml"
        gates_px = pixel_gates(m["slug"], m["gguf_px"], engine_px,
                               wav="short.wav" if key == "moss" else "medium.wav") \
            if m["status"] == "ready" else []
        if key == "moss":
            gates_px = gates_px + [dict(ENERGY_GATE)]
        out.append(profile(
            f"{key}--pixel", m["name"], m["hf"], m["slug"], m["status"], m["tracking"],
            m["notes_px"], "pixel", "vulkan" if fast else "cpu", engine_px,
            [{"name": "gguf", "path": f"${{STARLING_MODELS_DIR}}/{m['gguf_px']}",
              "sha256": None}, *FIXTURE_ARTIFACTS] if m["status"] == "ready" else [],
            {"shell": PX_BUILD if fast else PX_BUILD_GGML,
             "artifacts_out": ["build-campaign/starling-bench"]},
            gates_px,
            ([{"name": "latency", "gate": "phone-ab", "metric": "total_ms_delta_pct",
               "direction": "lower", "min_improvement": 5.0},
              {"name": "quality", "gate": "phone-ab", "metric": "transcripts_match",
               "direction": "higher", "min_improvement": 0.0},
              {"name": "peak memory", "gate": "phone-ab", "metric": "peak_rss_kb",
               "direction": "lower", "min_improvement": 0.0}]
             if gates_px else []),
            blocked_on=m.get("blocked_on"), blocked_reason=m.get("blocked_reason"),
            thresholds=THRESHOLDS_PIXEL,
            defaults={"allowed_paths": m["allowed"], "budgets": BUDGETS_PIXEL},
            device_expect="Pixel 10 Pro",
        ))
    return out


def fixture_profile() -> dict:
    return profile(
        "fixture--notebook-cpu", "starling-serve contract fixture", None, None, "ready",
        "#176",
        "the real-repo CPU pilot: builds the contract fixture (no models, no GPU) "
        "and runs the sealed experiment comparator as the perf gate",
        "notebook", "cpu", "auto",
        [],
        {"shell": NB_BUILD_FIXTURE,
         "artifacts_out": ["build-campaign/starling-serve-contract-fixture"]},
        [
            {
                "name": "serve-contract", "stage": "correctness",
                "argv": [
                    "{python}",
                    "{trusted}/benchmarks/campaign/gates/serve_contract_smoke.py",
                    "--binary", "{candidate}/starling-serve-contract-fixture",
                    "--baseline-binary", "{baseline}/starling-serve-contract-fixture",
                    "--model", "parakeet",
                ],
                "rules": [{"metric": "contract_ok", "op": "==", "value": 1},
                          {"metric": "transcripts_match", "op": "==", "value": 1}],
                "required": True, "timeout_s": 600.0,
            },
            {
                "name": "perf", "stage": "perf",
                "argv": [
                    "{python}",
                    "{trusted}/benchmarks/campaign/gates/experiments_ab.py",
                    "--base-bin", "{best}/starling-serve-contract-fixture",
                    "--cand-bin", "{candidate}/starling-serve-contract-fixture",
                    "--run-dir", "{attempt_dir}/experiment",
                    "--model", "parakeet",
                ],
                "rules": [{"metric": "verdict", "op": "==", "value": "pass"}],
                "required": True, "objective": True, "timeout_s": 1800.0,
            },
        ],
        [
            {"name": "end-to-end latency", "gate": "perf", "metric": "verdict",
             "direction": "higher", "min_improvement": 0.0},
            {"name": "serve contract", "gate": "serve-contract", "metric": "contract_ok",
             "direction": "higher", "min_improvement": 0.0},
        ],
        quality=dict(QUALITY_KERNEL, notes="fixture engine: exact contract by construction"),
        thresholds=THRESHOLDS_NOTEBOOK,
        defaults={"allowed_paths": ["cpp/serve/**"],
                  "budgets": {"max_attempts": 4, "campaign_wall_clock_s": 14400.0,
                              "attempt_wall_clock_s": 3600.0, "gate_timeout_s": 1800.0,
                              "agent_timeout_s": 1800.0, "token_budget": None,
                              "cooldown_max_s": 600.0}},
        device_expect=None,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = model_profiles() + [fixture_profile()]
    from spec import validate_profile
    for p in profiles:
        problems = validate_profile(p)
        if problems:
            raise SystemExit(f"{p['id']}: {problems}")
        (OUT / f"{p['id']}.json").write_text(
            json.dumps(p, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(profiles)} profiles to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
