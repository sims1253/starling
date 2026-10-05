#!/usr/bin/env python3
"""Check CUDA and ROCm install pins and concrete runtime guidance against the release version."""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ".github/workflows/release-starling-serve.yml"
DOCS = ("docs/release-runtime.md", "docs/native-serving.md")
DOCKERFILE = "scripts/release-runtime/Dockerfile.cuda"
DESKTOP_WORKFLOW = ".github/workflows/package-desktop.yml"
PREPARE = "scripts/experimental-release/prepare.py"
ENGINE_SECTION = "## Desktop bundled engines"
# The Windows Vulkan SDK installer pins: the version and the installer's
# SHA-256, each set once as an env var in both workflows that install it.
VULKAN_SDK_PINS = {
    "VULKAN_SDK_VERSION": re.compile(
        r"^\s+VULKAN_SDK_VERSION:\s*['\"]?(\d+(?:\.\d+)+)['\"]?\s*$", re.M),
    "VULKAN_SDK_SHA256": re.compile(
        r"^\s+VULKAN_SDK_SHA256:\s*['\"]?([0-9A-Fa-f]{64})['\"]?\s*$", re.M),
}


def bundled_engines_errors(desktop_workflow: str, prepare: str,
                           release_runtime: str) -> list[str]:
    """The bundled-engine backend list must agree everywhere it is stated.

    Sources: the desktop packager's BUNDLED_ENGINES env (the single source of
    truth the packaging steps use), prepare.py's build-info
    desktop_bundled_engines, and the "Desktop bundled engines" section of the
    runtime guide (which also must say CUDA is not bundled).
    """
    errors = []
    found = re.findall(r"^\s+BUNDLED_ENGINES: [\"']([^\"']+)[\"']\s*$",
                       desktop_workflow, re.M)
    workflow_engines = None
    if len(found) != 1:
        errors.append(
            f"{DESKTOP_WORKFLOW}: expected exactly one job-level "
            "BUNDLED_ENGINES: \"<engines>\" env (the single source of truth "
            "for the bundled desktop engines)")
    else:
        workflow_engines = found[0].split()

    found = re.findall(r"[\"']desktop_bundled_engines[\"']\s*:\s*\[([^\]]*)\]",
                       prepare)
    prepare_engines = None
    if len(found) != 1:
        errors.append(
            f"{PREPARE}: build-info must record desktop_bundled_engines: [...] "
            "with the bundled backend list")
    else:
        prepare_engines = re.findall(r"[\"']([a-z0-9]+)[\"']", found[0])

    section = re.search(re.escape(ENGINE_SECTION) + r"\n(.*?)(?=\n## |\Z)",
                        release_runtime, re.S)
    docs_engines = None
    if not section:
        errors.append(
            f"{DOCS[0]}: expected a {ENGINE_SECTION!r} section stating the "
            "bundled engines")
    else:
        body = section.group(1)
        # Each subsequent engine needs a ", " or " and " separator, so a
        # list with a missing separator fails instead of parsing loosely.
        sentence = re.search(
            r"desktop archives bundle exactly the "
            r"(`[a-z0-9]+`(?:(?:, | and )`[a-z0-9]+`)*)\s*engines", body)
        if not sentence:
            errors.append(
                f"{DOCS[0]}: the Desktop bundled engines section must state "
                "which engines ship inside the desktop archives "
                "(\"bundle exactly the `x` and `y` engines\")")
        else:
            docs_engines = re.findall(r"`([a-z0-9]+)`", sentence.group(1))
        if "CUDA is not bundled" not in body:
            errors.append(
                f"{DOCS[0]}: the Desktop bundled engines section must state "
                "that CUDA is not bundled")

    sources = [
        (DESKTOP_WORKFLOW + " BUNDLED_ENGINES", workflow_engines),
        (PREPARE + " desktop_bundled_engines", prepare_engines),
        (DOCS[0] + " Desktop bundled engines section", docs_engines),
    ]
    if any(engines is None for _, engines in sources):
        return errors
    # A duplicated entry inside one list would survive the ordered
    # comparison below while still being wrong — a bundle list must
    # name each backend exactly once.
    for name, engines in sources:
        duplicates = sorted({engine for engine in engines if engines.count(engine) > 1})
        if duplicates:
            errors.append(f"{name}: duplicate engine entries: {duplicates}")
    # Ordered, not set-wise: the list order is the app's preference order
    # (it tries the first entry first), so the sources must agree
    # element-for-element in the same order.
    ordered = [engines for _, engines in sources]
    if not ordered[0] or any(engines != ordered[0] for engines in ordered[1:]):
        errors.append(
            "Bundled desktop engine lists disagree (order matters, it is "
            "the preference order): "
            + "; ".join(f"{name}={engines}" for name, engines in sources))
    return errors


def vulkan_sdk_errors(desktop_workflow: str, release_workflow: str) -> list[str]:
    """Both workflows that install the Windows Vulkan SDK must pin one and
    the same installer (VULKAN_SDK_VERSION and VULKAN_SDK_SHA256).

    Sources: the desktop packager's Vulkan loader install (its assembly step
    runs the Vulkan engine's --version, which needs vulkan-1.dll) and the
    release workflow's windows-vulkan build. A silent divergence would let
    the desktop bundle be validated against an SDK the release never used.
    """
    errors = []
    for name, text in ((DESKTOP_WORKFLOW, desktop_workflow), (WORKFLOW, release_workflow)):
        if "humbletim/install-vulkan-sdk" in re.sub(r"#[^\n]*", "", text):
            errors.append(
                f"{name}: humbletim/install-vulkan-sdk cannot unpack SDK "
                "installers >= 1.4.313.0; run the pinned official installer")
    for var, pattern in VULKAN_SDK_PINS.items():
        pins = {}
        for name, text in ((DESKTOP_WORKFLOW, desktop_workflow), (WORKFLOW, release_workflow)):
            found = pattern.findall(text)
            if len(found) != 1:
                errors.append(f"{name}: expected exactly one {var} env pin")
            else:
                pins[name] = found[0].upper()
        if len(pins) == 2 and len(set(pins.values())) != 1:
            errors.append(
                f"{var} pins disagree (keep the two workflows in sync): "
                + "; ".join(f"{name} pins {pin}" for name, pin in pins.items()))
    return errors


HARDWARE_SECTION = "## Hardware verification"


def hardware_coverage_errors(release_runtime: str, release_workflow: str) -> list[str]:
    """Hardware coverage must stay explicit (#57).

    The runtime guide's Hardware verification ledger needs one row per
    artifact in its Prerequisites table, each starting "Verified" or
    "Not verified", and the release body must name exactly the
    "Not verified" artifacts in its "Not verified on hardware:" line, so a
    new artifact cannot ship without a coverage statement and the release
    notes cannot drop an unverified one.
    """
    errors = []
    table = re.search(r"^## Prerequisites\n(.*?)(?=\n## |\Z)", release_runtime, re.S | re.M)
    prerequisites = re.findall(r"^\| `([a-z0-9-]+)` \|", table.group(1) if table else "", re.M)
    if not prerequisites:
        errors.append(f"{DOCS[0]}: expected artifact rows in the Prerequisites table")
    section = re.search(re.escape(HARDWARE_SECTION) + r"\n(.*?)(?=\n## |\Z)", release_runtime, re.S)
    if not section:
        return [f"{DOCS[0]}: expected a {HARDWARE_SECTION!r} section"]
    rows = re.findall(r"^\| `([a-z0-9-]+)` \| ([^|]*) \|", section.group(1), re.M)
    ledger = [name for name, _ in rows]
    if sorted(ledger) != sorted(prerequisites) or len(set(ledger)) != len(ledger):
        errors.append(
            f"{DOCS[0]}: the Hardware verification ledger must have exactly one row per "
            f"artifact in the Prerequisites table; prerequisites={prerequisites} ledger={ledger}")
    unverified = []
    for name, status in rows:
        if status.startswith("Not verified"):
            unverified.append(name)
        elif not status.startswith("Verified"):
            errors.append(
                f"{DOCS[0]}: Hardware verification status for `{name}` must start with "
                f"\"Verified\" or \"Not verified\"; found {status!r}")
    found = re.findall(r"Not verified on hardware: ((?:`[a-z0-9-]+`(?:, )?)+)\.", release_workflow)
    if len(found) != 1:
        errors.append(f"{WORKFLOW}: the release body must state \"Not verified on hardware: "
                      "`artifact`, ...\" exactly once")
    elif sorted(re.findall(r"`([a-z0-9-]+)`", found[0])) != sorted(unverified):
        errors.append(
            f"{WORKFLOW}: the release body's \"Not verified on hardware\" list must match the "
            f"ledger's Not verified rows; release body={re.findall(r'`([a-z0-9-]+)`', found[0])} "
            f"ledger={unverified}")
    return errors


def check(workflow: str, docs: dict[str, str], executing_cuda_version: str | None = None,
          dockerfile: str = "") -> list[str]:
    errors = []
    versions = re.findall(r"^  CUDA_VERSION: ['\"]?(\d+\.\d+\.\d+)['\"]?\s*$", workflow, re.M)
    if len(versions) != 1:
        return [f"{WORKFLOW}: expected one CUDA_VERSION major.minor.patch in workflow env"]
    if executing_cuda_version is not None and executing_cuda_version != versions[0]:
        errors.append(
            f"Executing workflow CUDA_VERSION={executing_cuda_version!r} differs from "
            f"the checked-out release CUDA_VERSION={versions[0]!r}. "
            "Dispatch from a workflow ref with the same CUDA_VERSION as the release tag."
        )
    series = versions[0].rsplit(".", 1)[0]

    # Linux installs a series metapackage; Windows pins the toolkit patch.
    # Require both install sites to use the shared version, not a second pin.
    for required in (
        'cuda_series="${CUDA_VERSION%.*}"',
        'sudo apt-get install -y "cuda-toolkit-${cuda_series//./-}"',
        'cuda: ${{ env.CUDA_VERSION }}',
    ):
        if workflow.count(required) != 1:
            errors.append(f"{WORKFLOW}: expected one shared-version install expression: {required}")

    references = [
        (WORKFLOW, workflow, r"CUDA requires the (\d+\.\d+)(?:\s|$)", "release body"),
        (DOCS[0], docs[DOCS[0]], r"^\| `linux-cuda` \|.*?CUDA (\d+\.\d+) runtime", "Linux prerequisites"),
        (DOCS[0], docs[DOCS[0]], r"^\| `windows-cuda` \|.*?CUDA (\d+\.\d+) runtime", "Windows prerequisites"),
        (DOCS[1], docs[DOCS[1]], r"The workflow builds with CUDA (\d+\.\d+)\.", "serving guide"),
    ]
    for path, text, pattern, label in references:
        found = re.findall(pattern, text, re.M)
        if found != [series]:
            errors.append(f"{path}: {label} must state CUDA {series} once; found {found}")

    # The runtime-check image installs the CUDA series' runtime packages by
    # name; a series bump that skips it would keep validating the old
    # runtime (within a major the sonames still resolve), so the pin is
    # gated here like the ROCm repository (pullfrog review of #185).
    if dockerfile != "":
        series_dashed = series.replace(".", "-")
        for package in (f"cuda-cudart-{series_dashed}", f"libcublas-{series_dashed}"):
            if dockerfile.count(package) != 1:
                errors.append(
                    f"{DOCKERFILE}: expected exactly one {package} install "
                    f"matching CUDA {series}"
                )

        # The linux-cuda runner (workflow env) and this image both install the
        # userspace driver library by name so the binary's libcuda.so.1
        # resolves on GPU-less machines. Driver branches do not follow CUDA
        # versions, so the two pins cannot be derived — they must simply agree.
        workflow_driver = re.findall(r"^  CUDA_DRIVER_PACKAGE: (libnvidia-compute-\d+)", workflow, re.M)
        dockerfile_driver = re.findall(r"(libnvidia-compute-\d+)", dockerfile)
        if len(workflow_driver) != 1 or dockerfile_driver != workflow_driver:
            errors.append(
                f"{WORKFLOW}: CUDA_DRIVER_PACKAGE must be pinned exactly once and "
                f"match {DOCKERFILE}'s libnvidia-compute install; "
                f"workflow={workflow_driver} dockerfile={dockerfile_driver}"
            )

    # The ROCm release is an exact versioned apt repository, not a series:
    # the install URL is the single source of truth and the docs restate it.
    rocm_pins = re.findall(r"rocm/apt/(\d+\.\d+(?:\.\d+)?)", workflow)
    if len(rocm_pins) != 1:
        errors.append(f"{WORKFLOW}: expected one pinned rocm/apt/<version> repository URL")
        return errors
    rocm = rocm_pins[0]

    rocm_references = [
        (WORKFLOW, workflow, r"ROCm requires the (\d+\.\d+(?:\.\d+)?) HIP/BLAS runtime", "release body"),
        (DOCS[0], docs[DOCS[0]], r"^\| `linux-rocm` \|.*?ROCm (\d+\.\d+(?:\.\d+)?)", "Linux prerequisites"),
        (DOCS[0], docs[DOCS[0]], r"rocm/apt/(\d+\.\d+(?:\.\d+)?)", "pinned repository"),
        (DOCS[1], docs[DOCS[1]], r"The workflow builds with ROCm (\d+\.\d+(?:\.\d+)?)\.", "serving guide"),
    ]
    for path, text, pattern, label in rocm_references:
        found = re.findall(pattern, text, re.M)
        if found != [rocm]:
            errors.append(f"{path}: {label} must state ROCm {rocm} once; found {found}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executing-cuda-version",
                        help="CUDA_VERSION from the executing release workflow; "
                             "must match the checked-out release tag")
    args = parser.parse_args()
    workflow_text = (ROOT / WORKFLOW).read_text(encoding="utf-8")
    desktop_workflow_text = (ROOT / DESKTOP_WORKFLOW).read_text(encoding="utf-8")
    errors = check(workflow_text,
                   {name: (ROOT / name).read_text(encoding="utf-8") for name in DOCS},
                   args.executing_cuda_version,
                   (ROOT / DOCKERFILE).read_text(encoding="utf-8"))
    errors += bundled_engines_errors(
        desktop_workflow_text,
        (ROOT / PREPARE).read_text(encoding="utf-8"),
        (ROOT / DOCS[0]).read_text(encoding="utf-8"))
    errors += vulkan_sdk_errors(desktop_workflow_text, workflow_text)
    errors += hardware_coverage_errors(
        (ROOT / DOCS[0]).read_text(encoding="utf-8"), workflow_text)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("CUDA and ROCm installers and runtime guidance agree; "
          "bundled desktop engines agree; Vulkan SDK pins agree; "
          "hardware coverage is explicit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
