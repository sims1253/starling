"""``starling-gpu-run`` — run a command under the GPU lock for its whole life.

Wrapper that acquires a :class:`~starling.gpu.session.GpuSession` and then
``execvp``\\s the command, so the flock fd is inherited by the exec'd process
and the lock is held for the **child's entire lifetime** (released only when
the child exits and its inherited fd closes).

This gives the 26 benchmark scripts that have *no* per-file lock a uniform
lock with zero source edits::

    starling-gpu-run --session decode-bench --eta 5 -- python bench_decode.py
    starling-gpu-run --session moss        --       ./build/moss_server

The runner ``os.setsid()``\\s first so the command has an isolated process
group. Lock takeover is intentionally not automatic: stale metadata cannot
distinguish a hung process from a valid inherited native child, so the kernel
flock remains authoritative.

This module is stdlib-only (loads session.py by relative import when used as a
package, and also supports ``python -m`` / direct-file invocation).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Allow loading this file directly (importlib path) without the package.
try:
    from .session import GpuSession, GpuLockBusy, GpuLockTimeout  # type: ignore
except ImportError:  # pragma: no cover - direct-file execution path
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    from session import GpuSession, GpuLockBusy, GpuLockTimeout  # type: ignore


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="starling-gpu-run",
        description="Run a command under the Starling GPU lock.",
        # everything after the first '--' is the command to exec
    )
    p.add_argument("--session", required=True, help="logical session name")
    p.add_argument("--eta", type=int, default=5, dest="eta_min",
                   help="expected minutes the command will hold the GPU")
    p.add_argument("--note", default="", help="free-form note for the token")
    p.add_argument("--model", default="", help="model name for the token")
    p.add_argument("--lock-dir", default=None,
                   help="directory for the .flock files "
                        "(default: $STARLING_GPU_LOCK_DIR or /tmp)")
    p.add_argument("--uuid", default=None,
                   help="explicit GPU key (default: auto-resolve from nvidia-smi)")
    p.add_argument("--no-signal-handlers", action="store_true",
                   help="do not install SIGTERM/SIGINT teardown handlers")
    p.add_argument("--no-heartbeat", action="store_true",
                   help="disable the heartbeat refresh thread")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="command to run (use '--' to separate it)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cmd = list(args.command)
    # Strip a single leading "--" separator produced by `-- cmd`.
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no command given (use: starling-gpu-run [opts] -- CMD ...)")
        return 2

    # Keep the wrapped command in its own process group for operator cleanup.
    try:
        os.setsid()
    except OSError:
        pass  # already a session leader, or unsupported -> not fatal

    # execvp replaces this Python process, so its heartbeat thread cannot run in
    # the wrapped command. The token remains useful acquisition metadata; the
    # kernel flock, not heartbeat freshness, is the ownership authority.
    session = GpuSession(
        session=args.session,
        model=args.model,
        eta_min=args.eta_min,
        note=args.note,
        lock_dir=args.lock_dir,
        uuid=args.uuid,
        wait=True,
        install_signal_handlers=not args.no_signal_handlers,
        heartbeat=False,
    )
    try:
        session.acquire()
    except (GpuLockBusy, GpuLockTimeout) as e:
        print(f"starling-gpu-run: could not acquire GPU lock: {e}",
              file=sys.stderr)
        return 75  # EX_TEMPFAIL

    # Publish the inherited lock so benchmark scripts that retain their own
    # with_gpu_lock call borrow it rather than deadlocking on a second flock.
    if session._fd is not None:
        os.environ["STARLING_GPU_LOCK_FD"] = str(session._fd)
        os.environ["STARLING_GPU_LOCK_KEY"] = session._key or ""
        os.environ["STARLING_GPU_LOCK_OWNER"] = session.owner_id or ""
    # The fd is inheritable (set in GpuSession.acquire); execvp carries it into
    # the command, so the kernel lock persists for the command's whole lifetime.
    os.execvp(cmd[0], cmd)
    return 127  # unreachable if execvp succeeds


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
