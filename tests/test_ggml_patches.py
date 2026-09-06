"""Exercise the patch applicator in disposable repositories; no real ggml edits."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apply_ggml_patches.sh"
BASH = shutil.which("bash")
if os.name == "nt" and (git_executable := shutil.which("git")):
    # System32 may expose a WSL launcher even when no distribution is installed.
    BASH = next((str(candidate) for parent in Path(git_executable).resolve().parents
                 for candidate in (parent / "bin/bash.exe", parent / "usr/bin/bash.exe")
                 if candidate.is_file()), None)
pytestmark = pytest.mark.skipif(not BASH or not shutil.which("git"), reason="requires bash and git")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "project with spaces"
    repo = root / "third_party" / "ggml"
    patches = root / "third_party" / "ggml-patches"
    repo.mkdir(parents=True)
    patches.mkdir()
    (root / "scripts").mkdir()
    shutil.copyfile(SCRIPT, root / "scripts" / SCRIPT.name)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "core.autocrlf", "false")
    for name in ("a", "b", "local"):
        (repo / name).write_text(f"{name} original\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    return root


def series(project, changes):
    repo = project / "third_party" / "ggml"
    patches = project / "third_party" / "ggml-patches"
    base = git(repo, "rev-parse", "HEAD").strip().decode()
    for number, edits in enumerate(changes, 1):
        for name, text in edits.items():
            (repo / name).write_text(text)
        git(repo, "add", ".")
        (patches / f"{number:04d}.patch").write_bytes(git(repo, "diff", "--cached", "--binary"))
        git(repo, "commit", "-qm", f"patch {number}")
    git(repo, "reset", "--hard", base)
    return repo, sorted(patches.glob("*.patch"))


def run(project, **kwargs):
    return subprocess.run([BASH, (project / "scripts" / SCRIPT.name).as_posix()],
                          capture_output=True, text=True, timeout=10, **kwargs)


@pytest.mark.parametrize("present", [(), (0,), (1,), (0, 1)])
def test_independent_series_fresh_partial_tail_only_and_repeated(project, present):
    repo, patches = series(project, [{"a": "a patched\n"}, {"b": "b patched\n"}])
    for i in present:
        git(repo, "apply", str(patches[i]))
    before_index = (repo / ".git" / "index").read_bytes()
    result = run(project)
    assert result.returncode == 0, result.stderr
    assert (repo / "a").read_text() == "a patched\n"
    assert (repo / "b").read_text() == "b patched\n"
    assert (repo / ".git" / "index").read_bytes() == before_index
    repeated = run(project)
    assert repeated.returncode == 0, repeated.stderr
    assert "already applied" in repeated.stdout


@pytest.mark.parametrize("present_count", [0, 1, 2])
def test_overlapping_series(project, present_count):
    repo, patches = series(project, [{"a": "first change\n"}, {"a": "second change\n"}])
    for patch in patches[:present_count]:
        git(repo, "apply", str(patch))
    result = run(project)
    assert result.returncode == 0, result.stderr
    assert (repo / "a").read_text() == "second change\n"
    assert run(project).returncode == 0


def test_dirty_files_and_staged_changes_survive(project):
    repo, _ = series(project, [{"a": "a patched\n"}, {"b": "b patched\n"}])
    (repo / "local").write_text("staged change\n")
    git(repo, "add", "local")
    (repo / "local").write_text("unstaged change\n")
    (repo / "untracked").write_bytes(b"untracked data\x00")
    before_index = (repo / ".git" / "index").read_bytes()
    result = run(project)
    assert result.returncode == 0, result.stderr
    assert (repo / "local").read_text() == "unstaged change\n"
    assert (repo / "untracked").read_bytes() == b"untracked data\x00"
    assert (repo / ".git" / "index").read_bytes() == before_index


@pytest.mark.parametrize("partial_patch", [False, True])
def test_conflict_leaves_entire_worktree_and_index_unchanged(project, partial_patch):
    changes = [{"a": "a patched\n"}, {"b": "b patched\n", "local": "local patched\n"}]
    repo, _ = series(project, changes)
    # One file of a multi-file patch already present is an inconsistent patch.
    (repo / "b").write_text("b patched\n" if partial_patch else "user conflict\n")
    git(repo, "add", "b")
    before_files = {p.name: p.read_bytes() for p in repo.iterdir() if p.is_file()}
    before_index = (repo / ".git" / "index").read_bytes()
    result = run(project)
    assert result.returncode != 0
    assert "cannot validate complete" in result.stderr
    assert {p.name: p.read_bytes() for p in repo.iterdir() if p.is_file()} == before_files
    assert (repo / ".git" / "index").read_bytes() == before_index
    assert not (repo / ".git" / "starling-patches.lock").exists()


def test_untracked_file_collision_is_not_overwritten(project):
    repo, _ = series(project, [{"new": "new patched file\n"}])
    (repo / "new").write_text("user's untracked file\n")
    result = run(project)
    assert result.returncode != 0
    assert (repo / "new").read_text() == "user's untracked file\n"
    assert git(repo, "status", "--porcelain").decode() == "?? new\n"


def test_concurrent_runs_wait_for_lock_and_finish_idempotently(project):
    repo, _ = series(project, [{"a": "a patched\n"}, {"b": "b patched\n"}])
    lock = repo / ".git" / "starling-patches.lock"
    lock.mkdir()
    command = [BASH, (project / "scripts" / SCRIPT.name).as_posix()]
    children = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(2)]
    try:
        time.sleep(0.2)
        assert all(child.poll() is None for child in children)
        assert (repo / "a").read_text() == "a original\n"
        lock.rmdir()
        outputs = [child.communicate(timeout=10) for child in children]
        assert all(child.returncode == 0 for child in children), outputs
        assert sum("already applied" in out for out, _ in outputs) == 1
        assert (repo / "a").read_text() == "a patched\n"
        assert (repo / "b").read_text() == "b patched\n"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_stale_lock_fails_after_bounded_wait(project, tmp_path):
    repo, _ = series(project, [{"a": "a patched\n"}])
    lock = repo / ".git" / "starling-patches.lock"
    lock.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sleep = fake_bin / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    sleep.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
    result = run(project, env=env)
    assert result.returncode != 0
    assert "if no patch process is running" in result.stderr
    assert lock.exists()
    assert (repo / "a").read_text() == "a original\n"


def test_partial_hunk_fails_without_applying_an_earlier_patch(project):
    repo = project / "third_party" / "ggml"
    original = "".join(f"line {i}\n" for i in range(30))
    (repo / "a").write_text(original)
    git(repo, "add", "a")
    git(repo, "commit", "-qm", "multiple hunks")
    partial = original.replace("line 0\n", "first hunk\n")
    complete = partial.replace("line 29\n", "second hunk\n")
    repo, _ = series(project, [{"b": "b patched\n"}, {"a": complete}])
    (repo / "a").write_text(partial)
    result = run(project)
    assert result.returncode != 0
    assert (repo / "a").read_text() == partial
    assert (repo / "b").read_text() == "b original\n"


def test_unrelated_edit_in_patched_file_survives_and_diff_config_is_ignored(project):
    repo = project / "third_party" / "ggml"
    original = "".join(f"line {i}\n" for i in range(30))
    (repo / "a").write_text(original)
    git(repo, "add", "a")
    git(repo, "commit", "-qm", "long file")
    repo, _ = series(project, [{"a": original.replace("line 0\n", "patched\n")}])
    git(repo, "config", "diff.noprefix", "true")
    local = original.replace("line 29\n", "user edit\n")
    (repo / "a").write_text(local)
    result = run(project)
    assert result.returncode == 0, result.stderr
    assert (repo / "a").read_text() == local.replace("line 0\n", "patched\n")
    assert run(project).returncode == 0
