"""Run a command while holding the GPU lock through an inherited descriptor.

    starling-gpu-run --session decode-bench --eta 5 -- python bench_decode.py

Use a CUDA_VISIBLE_DEVICES GPU UUID on multi-GPU hosts. When nvidia-smi is
unavailable, --uuid supplies a key that all users of the same GPU must share.
Nested GpuSession, acquire_gpu_lock, and with_gpu_lock calls must pass that
same key as uuid=KEY. The key declares the device; it does not select it.
The command replaces this process; its exit closes the lock descriptor.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow loading this file directly (importlib path) without the package.
try:
    from .session import GpuLockBusy, GpuLockTimeout, GpuSession
except ImportError:  # pragma: no cover - direct-file execution path
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    from session import (  # ty: ignore[unresolved-import]
        GpuLockBusy,
        GpuLockTimeout,
        GpuSession,
    )


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
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="command to run (use '--' to separate it)")
    return p


def main(argv: list[str] | None = None) -> int:
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
    except (AttributeError, OSError):
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
        # Nested sessions borrow a non-inheritable duplicate. Publish an FD
        # that will remain open after exec, just as a newly acquired FD does.
        os.set_inheritable(session._fd, True)
        os.environ["STARLING_GPU_LOCK_FD"] = str(session._fd)
        os.environ["STARLING_GPU_LOCK_KEY"] = session._key or ""
        os.environ["STARLING_GPU_LOCK_OWNER"] = session.owner_id or ""
    # The fd is inheritable (set in GpuSession.acquire); execvp carries it into
    # the command, so the kernel lock persists for the command's whole lifetime.
    try:
        os.execvp(cmd[0], cmd)
    finally:
        session.release()  # exec failure; successful exec replaces this process
    return 127  # unreachable if execvp succeeds


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
