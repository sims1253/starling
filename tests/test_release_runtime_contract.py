"""Release documentation must agree with both CUDA installer configurations."""
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_contract", ROOT / "scripts/release-runtime/check-contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def inputs():
    return (ROOT / contract.WORKFLOW).read_text(), {name: (ROOT / name).read_text() for name in contract.DOCS}


def test_current_release_contract():
    assert contract.check(*inputs()) == []


@pytest.mark.parametrize("path, old, new", [
    (contract.WORKFLOW, "CUDA_VERSION: '@VERSION@'", "CUDA_VERSION: '@BAD_VERSION@'"),
    (contract.WORKFLOW, 'cuda_series="${CUDA_VERSION%.*}"', 'cuda_series="@BAD_SERIES@"'),
    (contract.WORKFLOW, '"cuda-toolkit-${cuda_series//./-}"', 'cuda-toolkit-@BAD_PACKAGE@'),
    (contract.WORKFLOW, 'cuda: ${{ env.CUDA_VERSION }}', "cuda: '@BAD_VERSION@'"),
    (contract.WORKFLOW, 'CUDA requires the @SERIES@', 'CUDA requires the @BAD_SERIES@'),
    (contract.DOCS[0], 'CUDA @SERIES@ runtime and cuBLAS libraries', 'CUDA @BAD_SERIES@ runtime and cuBLAS libraries'),
    (contract.DOCS[0], 'CUDA @SERIES@ runtime and cuBLAS DLLs', 'CUDA @BAD_SERIES@ runtime and cuBLAS DLLs'),
    (contract.DOCS[1], 'The workflow builds with CUDA @SERIES@.', 'The workflow builds with CUDA @BAD_SERIES@.'),
])
def test_rejects_independent_version_drift(path, old, new):
    workflow, docs = inputs()
    version = re.search(r"CUDA_VERSION: '([^']+)'", workflow).group(1)
    old = old.replace("@VERSION@", version).replace("@SERIES@", version.rsplit(".", 1)[0])
    bad_series = f"{int(version.split('.')[0]) + 1}.0"
    new = (new.replace("@BAD_VERSION@", bad_series + ".0")
              .replace("@BAD_SERIES@", bad_series)
              .replace("@BAD_PACKAGE@", bad_series.replace(".", "-")))
    text = workflow if path == contract.WORKFLOW else docs[path]
    assert old in text
    changed = text.replace(old, new)
    if path == contract.WORKFLOW:
        workflow = changed
    else:
        docs[path] = changed
    assert contract.check(workflow, docs)


def test_coordinated_version_update_passes():
    workflow, docs = inputs()
    version = re.search(r"CUDA_VERSION: '([^']+)'", workflow).group(1)
    series = version.rsplit(".", 1)[0]
    workflow = workflow.replace(version, '99.1.2').replace(series, '99.1')
    docs = {name: text.replace(series, '99.1') for name, text in docs.items()}
    assert contract.check(workflow, docs) == []
