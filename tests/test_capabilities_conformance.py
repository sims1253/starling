"""Contract tests for the server capability descriptor and wire fixtures (E10).

Three layers, all CPU-only file/JSON checks (no server, no model):

* Schema conformance — ``packages/contracts/capabilities/capability.json``
  validates against ``capability.schema.json`` (and a tampered descriptor is
  rejected, so the schema has teeth); every fixture file under
  ``capabilities/fixtures/`` validates against ``fixtures.schema.json``; the
  OpenAI-subset error bodies and the live ``/v1/starling/capabilities`` body
  validate against the schemas inside ``packages/contracts/openapi.json``.
  Validation uses ``jsonschema`` when installed and falls back to the
  self-contained ``minischema`` otherwise; no new dependencies.
* Cross-field consistency — every endpoint in the descriptor has at least one
  fixture; every fixture route exists in openapi.json or the documented native
  route set; audio fixtures agree with the descriptor's sample-rate and size
  limits; fixture status codes are covered by the descriptor's error contract;
  the canonical descriptor maps exactly onto the partial capability shape the
  live route serves; descriptor defaults still equal the C++ server's
  compile-time/CLI defaults (anti-rot check against cpp/serve sources).
* Adoption map — the README's who-consumes-what table is verified against the
  actual client code: today no client fetches /v1/starling/capabilities, the
  Rust desktop client hardcodes the 256 MiB cap that the descriptor states, and
  the documented probe is /v1/models. Wiring clients to consume
  the descriptor is explicit follow-up work listed in the README.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import minischema

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised only without the package
    jsonschema = None

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "packages" / "contracts" / "capabilities"
SCHEMA = json.loads((CONTRACT / "capability.schema.json").read_text())
DESCRIPTOR = json.loads((CONTRACT / "capability.json").read_text())
FIXTURE_SCHEMA = json.loads((CONTRACT / "fixtures.schema.json").read_text())
FIXTURES = sorted((CONTRACT / "fixtures").glob("*.json"))
OPENAPI = json.loads((REPO / "packages" / "contracts" / "openapi.json").read_text())
README = (CONTRACT / "README.md").read_text()

WS_UPGRADE_STATUS = 101  # the HTTP status of a successful WebSocket handshake


def assert_valid(instance, schema, root=None):
    if jsonschema is not None:
        jsonschema.Draft202012Validator(schema).validate(instance)
    else:
        problems = minischema.errors(instance, schema, root)
        assert not problems, problems


def assert_invalid(instance, schema):
    if jsonschema is not None:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(instance)
    else:
        assert minischema.errors(instance, schema), "expected schema rejection"


def fixture_cases() -> list[tuple[Path, dict, dict]]:
    """Yield (file, fixture-doc, case) for every case in every fixture file."""
    cases = []
    for path in FIXTURES:
        doc = json.loads(path.read_text())
        for case in doc["cases"]:
            cases.append((path, doc, case))
    return cases


def case_route(case: dict, doc: dict) -> tuple[str, str]:
    """(method, path) a case exercises; WS upgrades ride on GET."""
    request = case["request"]
    method = request["method"]
    if method == "WS":
        method = "GET"
    return method, request.get("path", doc["route"])


def route_matches(pattern: str, path: str) -> bool:
    """Match a route pattern with {param} or <param> segments against a
    literal path (openapi.json and the native docs use different styles)."""
    pattern_parts = pattern.strip("/").split("/")
    path_parts = path.strip("/").split("/")
    if len(pattern_parts) != len(path_parts):
        return False

    def is_param(part: str) -> bool:
        return (part.startswith("{") and part.endswith("}")) or (
            part.startswith("<") and part.endswith(">")
        )

    return all(
        part == literal or is_param(part)
        for part, literal in zip(pattern_parts, path_parts)
    )


def audio_specs(case: dict) -> list[dict]:
    """Every audio payload description in a case's request."""
    request = case["request"]
    specs = []
    if "body_audio" in request:
        specs.append(request["body_audio"])
    if "multipart_file" in request:
        specs.append(request["multipart_file"]["audio"])
    for frame in request.get("ws_frames", []):
        if "audio" in frame:
            specs.append(frame["audio"])
    return specs


# --------------------------------------------------------------------------- #
# 1. Schema conformance
# --------------------------------------------------------------------------- #

def test_descriptor_validates_against_schema():
    assert_valid(DESCRIPTOR, SCHEMA)


def test_schema_rejects_tampered_descriptor():
    """The schema must have teeth: fabricated capability fields are invalid."""
    bad = copy.deepcopy(DESCRIPTOR)
    bad["features"]["prompt"] = True  # not implemented; schema pins false
    assert_invalid(bad, SCHEMA)

    bad = copy.deepcopy(DESCRIPTOR)
    bad["aspirational_field"] = "soon"  # unknown property
    assert_invalid(bad, SCHEMA)

    bad = copy.deepcopy(DESCRIPTOR)
    bad["schema_version"] = 2  # unversioned bump without a schema
    assert_invalid(bad, SCHEMA)

    bad = copy.deepcopy(DESCRIPTOR)
    # an endpoint the server does not route (method enum)
    bad["endpoints"]["control"].append(
        {"method": "PATCH", "path": "/v1/audio/transcriptions", "protocol": "openai-subset"}
    )
    assert_invalid(bad, SCHEMA)


def test_one_model_server_is_not_oversubscribed():
    """Cross-field honesty: served list length must match the resident limit
    (the schema pins each entry's shape; the count is a consistency rule)."""
    models = DESCRIPTOR["models"]
    assert models["resident_limit"] == 1
    assert len(models["served"]) == models["resident_limit"]


def test_fixture_files_validate_against_envelope():
    """fixtures.schema.json must be exercised, not merely present: every file
    under fixtures/ validates against it under BOTH validators — jsonschema
    when installed, and the minischema fallback always, so the envelope has
    teeth even in environments without the third-party package (where the
    fallback path would otherwise never run). No fixture intentionally
    bypasses the envelope schema."""
    assert FIXTURES, "no conformance fixtures found"
    for path in FIXTURES:
        doc = json.loads(path.read_text())
        assert_valid(doc, FIXTURE_SCHEMA)
        assert not minischema.errors(doc, FIXTURE_SCHEMA), path.name
    # The envelope has teeth: fabricated fixture shapes must be rejected, so
    # the loop above cannot be a vacuous green.
    bad = json.loads(FIXTURES[0].read_text())
    bad["cases"][0]["wire_format"] = "invented"  # not in the envelope
    assert_invalid(bad, FIXTURE_SCHEMA)
    assert minischema.errors(bad, FIXTURE_SCHEMA)
    bad = json.loads(FIXTURES[0].read_text())
    del bad["cases"][0]["request"]  # required by the envelope
    assert_invalid(bad, FIXTURE_SCHEMA)
    assert minischema.errors(bad, FIXTURE_SCHEMA)


def test_openai_error_bodies_match_openapi_error_schema():
    error_schema = OPENAPI["components"]["schemas"]["Error"]
    checked = 0
    for _path, doc, case in fixture_cases():
        response = case["response"]
        if doc["protocol"] != "openai-subset" or response["status"] < 400:
            continue
        assert_valid(response["body_json"], error_schema)
        checked += 1
    assert checked >= 10, "expected the OpenAI-subset error ladder to be pinned"


def test_live_capabilities_body_matches_openapi_route_schema():
    schema = (
        OPENAPI["paths"]["/v1/starling/capabilities"]["get"]["responses"]["200"]
        ["content"]["application/json"]["schema"]
    )
    body = json.loads(
        (CONTRACT / "fixtures" / "starling_capabilities_route.json").read_text()
    )["cases"][0]["response"]["body_json"]
    assert_valid(body, schema)


# --------------------------------------------------------------------------- #
# 2. Cross-field consistency
# --------------------------------------------------------------------------- #

def descriptor_endpoints() -> list[tuple[str, str]]:
    endpoints = []
    for group in DESCRIPTOR["endpoints"].values():
        for endpoint in group:
            endpoints.append((endpoint["method"], endpoint["path"]))
    return endpoints


# The native routes documented in docs/native-serving.md#api-contract and
# wired in cpp/serve/main.cpp; the OpenAI subset comes from openapi.json.
DOCUMENTED_NATIVE_ROUTES = {
    ("GET", "/health"),
    ("POST", "/warmup"),
    ("POST", "/normalize"),
    ("DELETE", "/v1/audio/transcriptions/<id>"),
    ("GET", "/stream"),
}


def test_every_descriptor_endpoint_has_a_fixture():
    covered = {case_route(case, doc) for _p, doc, case in fixture_cases()}
    missing = []
    for method, path in descriptor_endpoints():
        if not any(m == method and route_matches(path, p) for m, p in covered):
            missing.append(f"{method} {path}")
    assert not missing, f"descriptor endpoints without fixtures: {missing}"


def test_every_fixture_route_is_a_real_route():
    openapi_paths = set(OPENAPI["paths"])
    unknown = []
    for _path, doc, case in fixture_cases():
        method, path = case_route(case, doc)
        known_openapi = path in openapi_paths
        known_native = any(
            m == method and route_matches(pattern, path)
            for m, pattern in DOCUMENTED_NATIVE_ROUTES
        )
        if not (known_openapi or known_native):
            unknown.append(f"{method} {path} ({doc['name']})")
    assert not unknown, f"fixture routes that are not documented server routes: {unknown}"


def test_fixture_audio_agrees_with_descriptor_limits():
    rates = set(DESCRIPTOR["audio"]["sample_rates_hz"])
    max_bytes = DESCRIPTOR["audio"]["max_upload_mb"] * 1024 * 1024
    for path, doc, case in fixture_cases():
        response = case["response"]
        for spec in audio_specs(case):
            rate = spec.get("sample_rate_hz")
            size = spec.get("bytes", 0)
            if rate is not None and rate not in rates:
                ws_refused = response["status"] == WS_UPGRADE_STATUS and any(
                    msg.get("type") == "error" for msg in response.get("ws_messages", [])
                )
                assert response["status"] == 400 or ws_refused, (
                    f"{path.name}:{case['name']} sends {rate} Hz audio but expects "
                    f"status {response['status']}; unsupported rates must be refused"
                )
            if size > max_bytes:
                assert response["status"] == 413, (
                    f"{path.name}:{case['name']} sends {size} bytes but expects "
                    f"status {response['status']}; oversize payloads must 413"
                )
            if response["status"] in (200, 202) and rate is not None:
                assert rate in rates, (
                    f"{path.name}:{case['name']} succeeds with {rate} Hz audio"
                )


def test_fixture_status_codes_are_in_the_error_contract():
    documented = {
        entry["code"] for entry in DESCRIPTOR["error_contract"]["status_codes"]
    }
    allowed = documented | {200, 202, WS_UPGRADE_STATUS}
    for path, _doc, case in fixture_cases():
        status = case["response"]["status"]
        assert status in allowed, (
            f"{path.name}:{case['name']} uses status {status}, which the "
            f"descriptor's error contract does not document"
        )


def test_descriptor_maps_onto_the_live_capabilities_route():
    """The canonical descriptor must project onto the partial shape the
    live GET /v1/starling/capabilities serves (README mapping table)."""
    features = DESCRIPTOR["features"]
    projected = {
        "schema_version": DESCRIPTOR["schema_version"],
        "model": DESCRIPTOR["models"]["served"][0]["id"],
        "audio_transcription": features["audio_transcription"],
        "audio_formats": DESCRIPTOR["audio"]["batch"]["openai_subset"]["containers"],
        "sample_rate_hz": DESCRIPTOR["audio"]["sample_rates_hz"][0],
        "response_formats": DESCRIPTOR["responses"]["openai_subset_formats"],
        "prompt": features["prompt"],
        "language_selection": features["language_selection"],
        "word_timestamps": features["word_timestamps"],
        "streaming_transcriptions": features["server_sent_transcription_events"],
        "websocket_path": DESCRIPTOR["streaming"]["path"],
    }
    live = json.loads(
        (CONTRACT / "fixtures" / "starling_capabilities_route.json").read_text()
    )["cases"][0]["response"]["body_json"]
    assert projected == live


def test_descriptor_defaults_match_the_cpp_server():
    """Anti-rot: the descriptor's stated defaults must equal what the server
    binary actually compiles in (cpp/serve/main.cpp Args and server.hpp)."""
    main_src = (REPO / "cpp" / "serve" / "main.cpp").read_text()
    hpp_src = (REPO / "cpp" / "serve" / "server.hpp").read_text()

    args = dict(
        (name, float(value))
        for name, value in re.findall(
            r"double (\w+)\s*=\s*(-?\d+(?:\.\d+)?)\s*;", main_src
        )
    )
    constants = dict(
        (name, int(value))
        for name, value in re.findall(
            r"(kSampleRate|kMaxWaiters|kMaxUploadMB)\s*=\s*(\d+)\s*;", hpp_src
        )
    )

    assert DESCRIPTOR["streaming"]["chunking"]["window_seconds"]["value"] == args["stream_chunk"]
    assert DESCRIPTOR["streaming"]["chunking"]["overlap_seconds"]["value"] == args["stream_overlap"]
    assert (
        DESCRIPTOR["streaming"]["chunking"]["min_seconds_before_first_partial"]["value"]
        == args["min_chunk"]
    )
    assert (
        DESCRIPTOR["streaming"]["chunking"]["min_seconds_between_partials"]["value"]
        == args["partial_interval"]
    )
    assert (
        DESCRIPTOR["streaming"]["live_buffer_cap"]["default_seconds"]["value"]
        == args["max_stream_seconds"]
    )
    assert (
        DESCRIPTOR["limits"]["request_timeout_seconds"]["value"]
        == args["request_timeout"]
    )
    assert DESCRIPTOR["audio"]["sample_rates_hz"] == [constants["kSampleRate"]]
    assert DESCRIPTOR["limits"]["max_queue_waiters"] == constants["kMaxWaiters"]
    assert DESCRIPTOR["audio"]["max_upload_mb"] == constants["kMaxUploadMB"]


# --------------------------------------------------------------------------- #
# 3. Adoption map (verified against the actual client code)
# --------------------------------------------------------------------------- #

CLIENT_TS_PATHS = [
    REPO / "packages" / "dictation" / "src",
]

CLIENT_RS_PATHS = [
    REPO / "apps" / "desktop-gpui" / "crates",
]


def _client_sources() -> list[Path]:
    sources: list[Path] = []
    for root in CLIENT_TS_PATHS:
        if root.is_dir():
            sources.extend(root.rglob("*.ts"))
            sources.extend(root.rglob("*.tsx"))
    for root in CLIENT_RS_PATHS:
        if root.is_dir():
            sources.extend(root.rglob("*.rs"))
    return sources


def test_no_client_consumes_the_capability_route_yet():
    """Today-state check behind the README adoption map: nothing in the client
    sources (TypeScript dictation package, Rust gpui crates) fetches
    /v1/starling/capabilities. When a client is wired to the descriptor
    (README follow-ups), update the adoption map and this assertion together."""
    offenders = [
        str(path.relative_to(REPO))
        for path in _client_sources()
        if "starling/capabilities" in path.read_text()
    ]
    assert not offenders, f"clients now consume the capability route: {offenders}"


def test_clients_probe_models_the_hardcoded_way():
    """Health probing uses /v1/models, not the capability descriptor."""
    dictation_client = (REPO / "packages" / "dictation" / "src" / "client.ts").read_text()
    assert "/v1/models" in dictation_client
    assert '"/health"' not in dictation_client
    assert '"/v1/audio/transcriptions"' in dictation_client
    rust_client = (
        REPO / "apps" / "desktop-gpui" / "crates" / "dictation" / "src" / "client.rs"
    ).read_text()
    # The Rust client builds routes with format!("{}/v1/...", base_url).
    assert "/v1/models" in rust_client
    assert "/health" not in rust_client
    assert "/v1/audio/transcriptions" in rust_client


def test_rust_client_hardcoded_upload_cap_equals_descriptor_limit():
    """The Rust desktop client hardcodes the 256 MiB audio cap instead of
    reading it from the descriptor; the two must at least agree."""
    client_rs = (
        REPO / "apps" / "desktop-gpui" / "crates" / "dictation" / "src" / "client.rs"
    ).read_text()
    mb = DESCRIPTOR["audio"]["max_upload_mb"]
    assert f"MAX_AUDIO_BYTES: usize = {mb} * 1024 * 1024" in client_rs


def test_readme_documents_the_adoption_map():
    for anchor in (
        "packages/dictation/src/client.ts",
        "apps/desktop-gpui/crates/dictation/src/client.rs",
        "backends/python",
        "TypeScript client",
        "Desktop (gpui, Rust)",
        "Android",
        "iOS",
        "Python backend",
    ):
        assert anchor in README, f"adoption map is missing {anchor!r}"
