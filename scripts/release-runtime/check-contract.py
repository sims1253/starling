#!/usr/bin/env python3
"""Check CUDA install pins and concrete runtime guidance against the release version."""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ".github/workflows/release-starling-serve.yml"
DOCS = ("docs/release-runtime.md", "docs/native-serving.md")


def check(workflow: str, docs: dict[str, str], executing_cuda_version: str | None = None) -> list[str]:
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
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executing-cuda-version",
                        help="CUDA_VERSION from the executing release workflow; "
                             "must match the checked-out release tag")
    args = parser.parse_args()
    errors = check((ROOT / WORKFLOW).read_text(encoding="utf-8"),
                   {name: (ROOT / name).read_text(encoding="utf-8") for name in DOCS},
                   args.executing_cuda_version)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("CUDA installers and runtime guidance agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
