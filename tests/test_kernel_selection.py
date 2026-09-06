"""Backend initialization and fallback without a CUDA device or compiler."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import Mock

import pytest
import torch

from starling import _kernels as kernels
from starling._kernels import base, cuda_backend, torch_backend


@pytest.fixture(autouse=True)
def isolate_selection(monkeypatch):
    monkeypatch.setattr(kernels, "_ACTIVE", None)
    monkeypatch.setattr(kernels, "_ACTIVE_NAME", None)
    monkeypatch.delenv("STARLING_KERNEL_BACKEND", raising=False)
    monkeypatch.setattr(kernels, "have_triton", lambda: False)
    monkeypatch.setattr(kernels, "have_cuda_compile", lambda: True)


def test_auto_compiler_failure_exports_working_torch_kernel(monkeypatch, caplog):
    compile_extension = Mock(side_effect=RuntimeError("compiler missing"))
    monkeypatch.setattr(cuda_backend, "_ext", compile_extension)

    from starling._kernels import residual_add

    x = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(residual_add(x, x), 2 * x)
    assert kernels.get_backend_name() == "torch"
    assert kernels.get_backend() is torch_backend
    compile_extension.assert_called_once_with()
    assert "compiler missing" in caplog.text


@pytest.mark.parametrize("source", ["env", "override"])
def test_explicit_cuda_propagates_compiler_failure(monkeypatch, source):
    if source == "env":
        monkeypatch.setenv("STARLING_KERNEL_BACKEND", "cuda")
    else:
        kernels.set_backend("cuda")
    monkeypatch.setattr(cuda_backend, "_ext", Mock(side_effect=RuntimeError("compiler missing")))
    with pytest.raises(RuntimeError, match="compiler missing"):
        kernels.get_backend()
    assert kernels._ACTIVE is None


def test_cuda_initialized_before_export_and_cached(monkeypatch):
    compile_extension = Mock()
    monkeypatch.setattr(cuda_backend, "_ext", compile_extension)
    assert kernels.residual_add is cuda_backend.residual_add
    assert kernels.get_backend_name() == "cuda"
    assert kernels.get_backend() is cuda_backend
    compile_extension.assert_called_once_with()


def test_selected_kernel_runtime_error_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(cuda_backend, "_ext", Mock())
    monkeypatch.setattr(cuda_backend, "residual_add", Mock(side_effect=RuntimeError("bad shape")))
    with pytest.raises(RuntimeError, match="bad shape"):
        kernels.residual_add(None, None)
    assert kernels.get_backend_name() == "cuda"


def test_reset_retries_initialization_after_fallback(monkeypatch):
    compile_extension = Mock(side_effect=[RuntimeError("compiler missing"), object()])
    monkeypatch.setattr(cuda_backend, "_ext", compile_extension)
    assert kernels.get_backend_name() == "torch"
    kernels.set_backend(None)
    assert kernels.get_backend_name() == "cuda"
    assert compile_extension.call_count == 2


def test_override_precedes_environment(monkeypatch):
    monkeypatch.setenv("STARLING_KERNEL_BACKEND", "cuda")
    kernels.set_backend(" TORCH ")
    assert kernels.get_backend_name() == "torch"


def test_auto_without_toolkit_uses_torch(monkeypatch):
    monkeypatch.setattr(kernels, "have_cuda_compile", lambda: False)
    compile_extension = Mock()
    monkeypatch.setattr(cuda_backend, "_ext", compile_extension)
    assert kernels.get_backend_name() == "torch"
    compile_extension.assert_not_called()


@pytest.mark.parametrize("value", ["nonsense", ""])
def test_unknown_environment_backend_raises(monkeypatch, value):
    monkeypatch.setenv("STARLING_KERNEL_BACKEND", value)
    with pytest.raises(ValueError, match="unknown kernel backend"):
        kernels.get_backend()


@pytest.mark.parametrize("windows", [True, False])
def test_cuda_probe_requires_device_and_nvcc(monkeypatch, tmp_path, windows):
    from torch.utils import cpp_extension

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(cpp_extension, "IS_WINDOWS", windows)
    assert not base.have_cuda_compile()
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / ("nvcc.exe" if windows else "nvcc")).touch()
    assert base.have_cuda_compile()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not base.have_cuda_compile()


def test_concurrent_first_access_initializes_only_once(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def initialize():
        if compile_extension.call_count == 1:
            started.set()
            assert release.wait(5)
            raise RuntimeError("compiler missing")
        return object()

    compile_extension = Mock(side_effect=initialize)
    monkeypatch.setattr(cuda_backend, "_ext", compile_extension)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(kernels.get_backend)
        try:
            assert started.wait(5)
            second = pool.submit(kernels.get_backend)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=5) is torch_backend
        assert second.result(timeout=5) is torch_backend
    compile_extension.assert_called_once_with()
    assert kernels.get_backend_name() == "torch"
