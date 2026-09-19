"""Release documentation must agree with the CUDA and ROCm installer configurations."""
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_contract", ROOT / "scripts/release-runtime/check-contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def inputs():
    return ((ROOT / contract.WORKFLOW).read_text(encoding="utf-8"),
            {name: (ROOT / name).read_text(encoding="utf-8") for name in contract.DOCS},
            (ROOT / contract.DOCKERFILE).read_text(encoding="utf-8"))


def test_current_release_contract():
    workflow, docs, dockerfile = inputs()
    assert contract.check(workflow, docs, dockerfile=dockerfile) == []


@pytest.mark.parametrize("path, old, new", [
    (contract.WORKFLOW, "CUDA_VERSION: '@VERSION@'", "CUDA_VERSION: '@BAD_VERSION@'"),
    (contract.WORKFLOW, 'cuda_series="${CUDA_VERSION%.*}"', 'cuda_series="@BAD_SERIES@"'),
    (contract.WORKFLOW, '"cuda-toolkit-${cuda_series//./-}"', 'cuda-toolkit-@BAD_PACKAGE@'),
    (contract.WORKFLOW, 'cuda: ${{ env.CUDA_VERSION }}', "cuda: '@BAD_VERSION@'"),
    (contract.WORKFLOW, 'CUDA requires the @SERIES@', 'CUDA requires the @BAD_SERIES@'),
    (contract.DOCS[0], 'CUDA @SERIES@ runtime and cuBLAS libraries', 'CUDA @BAD_SERIES@ runtime and cuBLAS libraries'),
    (contract.DOCS[0], 'CUDA @SERIES@ runtime and cuBLAS DLLs', 'CUDA @BAD_SERIES@ runtime and cuBLAS DLLs'),
    (contract.DOCS[1], 'The workflow builds with CUDA @SERIES@.', 'The workflow builds with CUDA @BAD_SERIES@.'),
    (contract.DOCKERFILE, 'cuda-cudart-@SERIES_DASHED@ libcublas-@SERIES_DASHED@', 'cuda-cudart-@BAD_DASHED@ libcublas-@BAD_DASHED@'),
])
def test_rejects_independent_version_drift(path, old, new):
    workflow, docs, dockerfile = inputs()
    version = re.search(r"CUDA_VERSION: '([^']+)'", workflow).group(1)
    old = old.replace("@VERSION@", version).replace("@SERIES@", version.rsplit(".", 1)[0])
    bad_series = f"{int(version.split('.')[0]) + 1}.0"
    dashed = version.rsplit(".", 1)[0].replace(".", "-")
    old = old.replace("@SERIES_DASHED@", dashed)
    new = (new.replace("@BAD_VERSION@", bad_series + ".0")
              .replace("@BAD_SERIES@", bad_series)
              .replace("@BAD_PACKAGE@", bad_series.replace(".", "-"))
              .replace("@BAD_DASHED@", bad_series.replace(".", "-")))
    text = {contract.WORKFLOW: workflow, **docs, contract.DOCKERFILE: dockerfile}[path]
    assert old in text
    changed = text.replace(old, new)
    if path == contract.WORKFLOW:
        workflow = changed
    elif path == contract.DOCKERFILE:
        dockerfile = changed
    else:
        docs[path] = changed
    assert contract.check(workflow, docs, dockerfile=dockerfile)


@pytest.mark.parametrize("path, old, new", [
    (contract.WORKFLOW, 'ROCm requires the @ROCM@ HIP/BLAS runtime', 'ROCm requires the @BAD_ROCM@ HIP/BLAS runtime'),
    (contract.WORKFLOW, 'rocm/apt/@ROCM@', 'rocm/apt/@BAD_ROCM@'),
    (contract.DOCS[0], 'ROCm @ROCM@ HIP runtime', 'ROCm @BAD_ROCM@ HIP runtime'),
    (contract.DOCS[0], 'rocm/apt/@ROCM@', 'rocm/apt/@BAD_ROCM@'),
    (contract.DOCS[1], 'The workflow builds with ROCm @ROCM@.', 'The workflow builds with ROCm @BAD_ROCM@.'),
])
def test_rejects_independent_rocm_version_drift(path, old, new):
    workflow, docs, dockerfile = inputs()
    rocm = re.search(r"rocm/apt/(\d+\.\d+(?:\.\d+)?)", workflow).group(1)
    old = old.replace("@ROCM@", rocm)
    parts = [int(p) + (i == 0) for i, p in enumerate(rocm.split("."))]
    bad_rocm = ".".join(str(p) for p in parts)
    new = new.replace("@BAD_ROCM@", bad_rocm)
    text = workflow if path == contract.WORKFLOW else docs[path]
    assert old in text
    changed = text.replace(old, new)
    if path == contract.WORKFLOW:
        workflow = changed
    else:
        docs[path] = changed
    assert contract.check(workflow, docs, dockerfile=dockerfile)


def test_coordinated_version_update_passes():
    workflow, docs, dockerfile = inputs()
    version = re.search(r"CUDA_VERSION: '([^']+)'", workflow).group(1)
    series = version.rsplit(".", 1)[0]
    bad_series = '99.1'
    workflow = workflow.replace(version, '99.1.2').replace(series, bad_series)
    docs = {name: text.replace(series, bad_series) for name, text in docs.items()}
    dockerfile = dockerfile.replace(series.replace(".", "-"),
                                    bad_series.replace(".", "-"))
    assert contract.check(workflow, docs, dockerfile=dockerfile) == []


@pytest.mark.parametrize("patch_offset", [0, 1])
def test_release_preflight_checks_executing_workflow_version(monkeypatch, capsys, patch_offset):
    workflow, _, _ = inputs()
    version = re.search(r"CUDA_VERSION: '([^']+)'", workflow).group(1)
    major, minor, patch = version.split(".")
    executing = f"{major}.{minor}.{int(patch) + patch_offset}"
    monkeypatch.setattr("sys.argv", ["check-contract.py", "--executing-cuda-version", executing])
    result = contract.main()
    captured = capsys.readouterr()
    if patch_offset:
        assert result == 1
        assert f"Executing workflow CUDA_VERSION={executing!r}" in captured.err
        assert f"checked-out release CUDA_VERSION={version!r}" in captured.err
        assert "Dispatch from a workflow ref" in captured.err
    else:
        assert result == 0
        assert captured.err == ""
