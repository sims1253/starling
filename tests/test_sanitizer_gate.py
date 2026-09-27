"""CPU regressions for the sanitizer runner's fail-closed accounting."""

from __future__ import annotations

from pathlib import Path

from scripts.sanitizer_gate import run_tool


def _fake_sanitizer(tmp_path: Path) -> Path:
    executable = tmp_path / "compute-sanitizer"
    executable.write_text(
        "#!/bin/sh\n"
        "if [ \"$FAKE_SANITIZER_MODE\" = timeout ]; then sleep 2; fi\n"
        "if [ \"$FAKE_SANITIZER_MODE\" = no_child ]; then\n"
        "  echo '========= ERROR SUMMARY: 0 errors'\n"
        "  exit 0\n"
        "fi\n"
        "shift 6\n"
        '"$@"\n'
        "result=$?\n"
        "if [ \"$FAKE_SANITIZER_MODE\" = finding ]; then\n"
        "  echo '========= ERROR SUMMARY: 1 errors'\n"
        "  exit 86\n"
        "fi\n"
        "echo '========= ERROR SUMMARY: 0 errors'\n"
        'exit "$result"\n'
    )
    executable.chmod(0o755)
    return executable


def _run(tmp_path: Path, monkeypatch, source: str, mode: str = "", timeout: int = 30) -> dict[str, object]:
    test = tmp_path / "test_probe.py"
    test.write_text(source)
    monkeypatch.setenv("FAKE_SANITIZER_MODE", mode)
    return run_tool(
        str(_fake_sanitizer(tmp_path)), "memcheck", tmp_path, timeout,
        test=test, expected_tests=1, cwd=tmp_path,
    )


def test_inherited_collect_only_does_not_bypass_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only -p no:cacheprovider")
    result = _run(tmp_path, monkeypatch, "def test_runs(): assert True\n")
    assert result["status"] == "pass"
    assert result["executed_tests"] == 1


def test_no_tests_and_stale_report_fail(tmp_path: Path, monkeypatch) -> None:
    stale = tmp_path / "memcheck.junit.xml"
    stale.write_text('<testsuite><testcase name="stale"/></testsuite>')
    result = _run(tmp_path, monkeypatch, "# no tests\n", mode="no_child")
    assert not stale.exists()
    assert result["status"] == "fail"
    assert result["executed_tests"] == 0
    assert "did not write" in str(result["reason"])


def test_skipped_test_fails(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, "import pytest\ndef test_skip(): pytest.skip('no GPU')\n")
    assert result["status"] == "fail"
    assert "skipped" in str(result["reason"])


def test_timeout_fails(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, "def test_runs(): assert True\n", mode="timeout", timeout=1)
    assert result["status"] == "timeout"
    assert result["exit_code"] is None
    assert "timed out" in str(result["reason"])


def test_sanitizer_finding_fails(tmp_path: Path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, "def test_runs(): assert True\n", mode="finding")
    assert result["status"] == "fail"
    assert "exit code 86" in str(result["reason"])
