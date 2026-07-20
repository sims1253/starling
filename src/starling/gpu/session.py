"""flock-based cross-process GPU lock with native-child awareness (Task 1).

The previous lock (``starling.parakeet.gpu_lock``) used an ``O_CREAT|O_EXCL``
file keyed by the ``CUDA_VISIBLE_DEVICES`` string and recorded only the *Python
parent* PID. Two hazards followed:

1. **Orphaned native children.** Starling's benchmark engines spawn native GPU
   processes (``CrispASR``/``ParakeetCpp``/``GgmlParakeet`` — see
   ``benchmarks/engines.py``). When the Python parent died, the old lock was
   released while the native child still held VRAM, so a second benchmark would
   start on a contended GPU and the two corrupted each other's numbers.
2. **Keying by ``CUDA_VISIBLE_DEVICES``.** ``CVD=0`` and unset on a single-GPU
   box refer to the same device but produced *different* lock files.

This module fixes both:

* Mutual exclusion is a POSIX ``fcntl.flock(LOCK_EX)`` on
  ``<dir>/starling-gpu-<key>.flock``. ``flock`` is associated with the *open
  file description*, so it is held as long as ANY process holds an fd on it —
  and it is released **automatically on process death / fd close** (no stale
  lock files to steal).
* The flock fd is opened *inheritable* (``os.set_inheritable(True)``) and passed
  to native children via :meth:`GpuSession.spawn` (``pass_fds``). A native child
  therefore keeps the lock held for its whole lifetime even if the Python parent
  is SIGKILLed — closing hazard #1.
* The lock key is the **GPU UUID** actually visible (``nvidia-smi``), so every
  spelling of "this GPU" (``CVD=0`` / unset / ``CVD=0,1`` on a one-GPU host)
  maps to one file — closing hazard #2. Multi-GPU visibility fails closed until
  per-device ordered lock acquisition is implemented, preventing partial-set
  overlap from silently corrupting measurements.

A v2 JSON token is written into the flock file for observability. The kernel
flock is authoritative; stale metadata is never used to kill a holder because
an inherited native child may remain valid after its Python parent exits.

This module is **stdlib-only** (no ``starling`` package import, no torch) so it
can be loaded directly by ``importlib.util.spec_from_file_location`` in
subprocess holders, keeping the test suite hermetic and GPU-free.

Environment knobs:
* ``STARLING_GPU_LOCK_DISABLE=1`` — make every session a no-op (hermetic opt-out).
* ``STARLING_GPU_LOCK_DIR``       — directory for the ``.flock`` files.
* ``STARLING_GPU_LOCK_FORCE=1``   — retained compatibility knob; takeover is
  intentionally refused unless a future design can prove the holder is dead.
"""

from __future__ import annotations

import errno
try:
    import fcntl
except ImportError:  # native Windows: import remains safe; acquire fails closed
    fcntl = None  # type: ignore[assignment]
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid as _uuid
from pathlib import Path
from typing import IO, Optional

# v2 token schema. Bumped only on a breaking change to the token *contract*.
TOKEN_VERSION = 2
TOKEN_SCHEMA = "starling.gpu.session"

HEARTBEAT_SEC = 5.0  # how often an in-process holder refreshes observability

_QUERY_CACHE: Optional[list[str]] = None


class GpuLockBusy(RuntimeError):
    """Raised when a fresh lock is held and ``wait=False`` (or the wait expired)."""


class GpuLockTimeout(TimeoutError):
    """Raised when ``wait=True`` exceeded ``max_wait_sec``."""


# --------------------------------------------------------------------------- #
# GPU discovery + key resolution
# --------------------------------------------------------------------------- #
def _query_gpu_uuids() -> list[str]:
    """Return the UUIDs of all GPUs, or ``[]`` if nvidia-smi is unavailable.

    Cached process-globally after the first successful/failed probe so repeated
    sessions don't re-shell out.
    """
    global _QUERY_CACHE
    if _QUERY_CACHE is not None:
        return list(_QUERY_CACHE)
    uuids: list[str] = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                u = line.strip()
                if u:
                    uuids.append(u)
    except (OSError, subprocess.SubprocessError):
        pass
    _QUERY_CACHE = list(uuids)
    return list(uuids)


def _parse_cvd(cvd: Optional[str]) -> Optional[list[int]]:
    """Parse ``CUDA_VISIBLE_DEVICES`` into a sorted list of indices.

    Returns ``None`` for unset/empty ("all GPUs"). Supports comma lists and
    ``M-N`` ranges (the CUDA format). Invalid entries are dropped.
    """
    if cvd is None:
        return None
    cvd = cvd.strip()
    if cvd == "":
        return None
    indices: list[int] = []
    for part in cvd.split(","):
        part = part.strip()
        if part == "":
            continue
        if "-" in part:  # range "M-N"
            lo, _, hi = part.partition("-")
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError:
                continue
            indices.extend(range(lo_i, hi_i + 1))
        else:
            try:
                indices.append(int(part))
            except ValueError:
                # CUDA also allows UUID-as-CVD; we only support integer indices
                # here (UUIDs come from nvidia-smi directly).
                continue
    return sorted(set(indices)) if indices else None


def _resolve_lock_key(
    cvd: Optional[str] = None,
    uuids: Optional[list[str]] = None,
) -> str:
    """Collapse the visible-device set into one stable lock key.

    On a one-GPU box, ``CVD=0`` / unset / ``CVD=0,1`` all select that single
    GPU and so collapse to the same key. With no nvidia-smi, we fall back to a
    sanitized ``CVD`` string (preserving the old behaviour) so the lock still
    works on a GPU-less/CPU machine.
    """
    if cvd is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if uuids is None:
        uuids = _query_gpu_uuids()
    indices = _parse_cvd(cvd)
    if not uuids:
        # No GPU discovery: key on the (sanitized) CVD string so CPU-only boxes
        # still get a deterministic, contention-aware lock.
        raw = (cvd if cvd is not None and cvd != "" else "default")
        return raw.replace("/", "_").replace(",", "-").replace(":", "_")
    if indices is None:
        chosen = list(uuids)
    else:
        chosen = [uuids[i] for i in indices if 0 <= i < len(uuids)]
        if not chosen:  # every index was out of range -> treat as "all"
            chosen = list(uuids)
    # de-dup + sort -> stable, order-independent key
    return ",".join(sorted(set(chosen)))


def _sanitize(key: str) -> str:
    return key.replace("/", "_").replace(os.sep, "_")


def _default_lock_dir() -> Path:
    return Path(os.environ.get("STARLING_GPU_LOCK_DIR",
                               tempfile.gettempdir())).expanduser()


def _lock_file_path(key: str, lock_dir: Optional[str] = None) -> Path:
    base = Path(lock_dir) if lock_dir else _default_lock_dir()
    return base / f"starling-gpu-{_sanitize(key)}.flock"


# --------------------------------------------------------------------------- #
# token (v2) helpers
# --------------------------------------------------------------------------- #
def _parse_token(path: Path) -> Optional[dict]:
    """Read + validate a token file. Returns ``None`` if absent/empty/stale-v1."""
    try:
        raw = path.read_text()
    except (FileNotFoundError, OSError):
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        tok = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(tok, dict) or tok.get("v") != TOKEN_VERSION:
        # Reject anything that isn't a v2 token (incl. legacy v:1 entries).
        return None
    return tok


class GpuSession:
    """A flock-held, native-child-aware GPU lock session.

    Use as a context manager::

        with GpuSession(session="bench", eta_min=5) as s:
            ...  # exclusive GPU access
            child = s.spawn(["parakeet-server", ...])  # inherits the lock

    The lock is held until the context exits AND every ``s.spawn``-ed child has
    exited (the flock fd is inherited). SIGKILLing this process releases the
    lock only once the native children release their inherited fd.
    """

    def __init__(
        self,
        *,
        session: str,
        model: str = "",
        eta_min: int = 5,
        note: str = "",
        lock_dir: Optional[str] = None,
        uuid: Optional[str] = None,
        wait: bool = True,
        poll_sec: float = 0.2,
        max_wait_sec: float = 600.0,
        install_signal_handlers: bool = False,
        heartbeat: bool = True,
        force: Optional[bool] = None,
    ) -> None:
        self.session = session
        self.model = model
        self.eta_min = eta_min
        self.note = note
        self._lock_dir = lock_dir
        self._uuid_arg = uuid
        self.wait = wait
        self.poll_sec = poll_sec
        self.max_wait_sec = max_wait_sec
        self._want_signal_handlers = install_signal_handlers
        self._want_heartbeat = heartbeat
        # Retained as an accepted compatibility argument. Automatic takeover is
        # deliberately unsupported because stale metadata cannot prove that an
        # inherited native child has stopped valid GPU work.
        self.force = bool(force or os.environ.get("STARLING_GPU_LOCK_FORCE", "") == "1")

        self._fd: Optional[int] = None
        self._path: Optional[Path] = None
        self._key: Optional[str] = None
        self._owner_id: Optional[str] = None
        self._disabled = (
            os.environ.get("STARLING_GPU_LOCK_DISABLE", "") == "1"
        )
        self._acquired = False
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        self._token_lock = threading.Lock()
        self._old_handlers: dict[int, object] = {}
        self._atexit_registered = False
        self._borrowed = False

    # -- introspection -----------------------------------------------------
    def read_token(self) -> Optional[dict]:
        """The current v2 token for this session (or ``None`` if not held)."""
        if self._path is None:
            return None
        return _parse_token(self._path)

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id

    @property
    def lock_path(self) -> Optional[Path]:
        return self._path

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "GpuSession":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def acquire(self) -> "GpuSession":
        if self._acquired:
            return self
        if self._disabled:
            self._acquired = True
            self._owner_id = _uuid.uuid4().hex
            return self

        if fcntl is None:
            raise RuntimeError(
                "Starling GPU isolation requires POSIX fcntl.flock on this "
                "platform. Set STARLING_GPU_LOCK_DISABLE=1 only when external "
                "serialization is guaranteed."
            )

        key = self._uuid_arg or _resolve_lock_key()
        if "," in key:
            raise RuntimeError(
                "GpuSession currently requires exactly one visible GPU; set "
                "CUDA_VISIBLE_DEVICES to one device. Multi-GPU set locking "
                "must acquire one lock per UUID to prevent partial overlap."
            )
        self._key = key
        path = _lock_file_path(key, self._lock_dir)

        # A command launched by starling-gpu-run (or GpuSession.spawn) already
        # owns this exact lock through an inherited fd. Borrow a dup instead of
        # trying to flock the same file through a new open-file description,
        # which would deadlock existing benchmark scripts that also call the
        # legacy with_gpu_lock API internally.
        inherited_fd = os.environ.get("STARLING_GPU_LOCK_FD")
        inherited_key = os.environ.get("STARLING_GPU_LOCK_KEY")
        if inherited_fd is not None and inherited_key == key:
            try:
                self._fd = os.dup(int(inherited_fd))
            except (OSError, ValueError):
                self._fd = None
            if self._fd is not None:
                self._path = path
                # This duplicate is intentionally only a nested reference. The
                # original env-published fd remains open for the wrapped
                # process's whole lifetime, so releasing an inner legacy
                # with_gpu_lock cannot release the outer runner's lock. os.dup
                # returns a non-inheritable fd (PEP 446); spawn() explicitly
                # marks it inheritable when another native child is launched.
                self._borrowed = True
                self._acquired = True
                self._owner_id = os.environ.get(
                    "STARLING_GPU_LOCK_OWNER", self._owner_id)
                return self
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._owner_id = _uuid.uuid4().hex

        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        # THE key line: the fd must survive into native children (spawn + exec).
        # Python (PEP 446) creates non-inheritable fds by default; flip it.
        os.set_inheritable(fd, True)
        self._fd = fd

        self._flock_with_contention()
        self._write_token()
        if self._want_heartbeat:
            self._start_heartbeat()
        if not self._atexit_registered:
            import atexit
            atexit.register(self.release)
            self._atexit_registered = True
        if self._want_signal_handlers:
            self._install_signal_handlers()
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        if self._disabled:
            self._fd = None
            return
        self._stop_heartbeat()
        self._restore_signal_handlers()
        # Do NOT issue LOCK_UN: spawned children share this open-file
        # description, and an explicit unlock would release their lock too.
        # Closing our fd drops only our reference; flock releases naturally
        # after the last inherited/duplicated fd closes. The token is metadata,
        # not the lock, so leaving the last owner record is intentional.
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self._borrowed = False

    # -- native children ---------------------------------------------------
    def spawn(self, args, **popen_kwargs) -> subprocess.Popen:
        """``subprocess.Popen`` that inherits the flock fd (``pass_fds``).

        The child keeps the lock held for its whole lifetime even if this
        process is killed. ``pass_fds`` is merged with any caller-supplied fds.
        """
        if not self._acquired or self._fd is None:
            raise RuntimeError("GpuSession.spawn before acquire()")
        env = dict(popen_kwargs.pop("env", os.environ))
        env["STARLING_GPU_LOCK_FD"] = str(self._fd)
        env["STARLING_GPU_LOCK_KEY"] = self._key or ""
        env["STARLING_GPU_LOCK_OWNER"] = self._owner_id or ""
        popen_kwargs["env"] = env
        if fcntl is not None:
            extra = popen_kwargs.pop("pass_fds", ())
            fds = tuple(dict.fromkeys((self._fd, *extra)))  # dedup, keep order
            os.set_inheritable(self._fd, True)
            popen_kwargs["pass_fds"] = fds
        return subprocess.Popen(args, **popen_kwargs)

    # -- internals ---------------------------------------------------------
    def _flock_with_contention(self) -> None:
        assert self._fd is not None and fcntl is not None
        deadline = time.time() + (self.max_wait_sec if self.wait else 0.0)
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if not self.wait:
                    raise GpuLockBusy(self._who())
                if time.time() >= deadline:
                    raise GpuLockTimeout(
                        f"timed out after {self.max_wait_sec}s waiting for GPU "
                        f"lock held by {self._who()}")
                time.sleep(self.poll_sec)

    def _who(self) -> str:
        tok = self.read_token()
        if tok:
            return (f"session={tok.get('session')!r} pid={tok.get('pid')} "
                    f"uuid={tok.get('uuid')}")
        return f"{self._path}"

    def _write_token(self, heartbeat: bool = False) -> None:
        assert self._path is not None and self._fd is not None
        now = time.time()
        try:
            pgid = os.getpgid(0)
        except OSError:
            pgid = os.getpid()
        payload = {
            "v": TOKEN_VERSION,
            "schema": TOKEN_SCHEMA,
            "session": self.session,
            "model": self.model,
            "pid": os.getpid(),
            "pgid": pgid,
            "hostname": socket.gethostname(),
            "uuid": self._key,
            "started_at": self._started_at if heartbeat else now,
            "heartbeat_at": now,
            "eta_min": self.eta_min,
            "note": self.note,
            "owner_id": self._owner_id,
        }
        if heartbeat:
            # preserve the original acquisition time across heartbeats
            payload["started_at"] = getattr(self, "_started_at", now)
        else:
            self._started_at = now
        with self._token_lock:
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                data = json.dumps(payload).encode("utf-8")
                os.write(self._fd, data)
                os.ftruncate(self._fd, len(data))
            except OSError:
                pass

    def _start_heartbeat(self) -> None:
        def beat():
            while not self._hb_stop.wait(HEARTBEAT_SEC):
                if not self._acquired:
                    return
                self._write_token(heartbeat=True)
        self._hb_thread = threading.Thread(
            target=beat, name="starling-gpu-heartbeat", daemon=True)
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        self._hb_stop.set()
        t = self._hb_thread
        self._hb_thread = None
        if t is not None and t.is_alive():
            t.join(timeout=2 * HEARTBEAT_SEC)

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            self._old_handlers[sig] = prev
            try:
                signal.signal(sig, self._make_handler(sig))
            except (ValueError, OSError):
                # not in main thread / unsupported -> skip gracefully
                pass

    def _make_handler(self, sig):
        def handler(signum, frame):
            try:
                self.release()
            finally:
                prev = self._old_handlers.get(signum)
                if callable(prev):
                    prev(signum, frame)
                else:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)
        return handler

    def _restore_signal_handlers(self) -> None:
        for sig, prev in list(self._old_handlers.items()):
            try:
                signal.signal(sig, prev)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass
        self._old_handlers.clear()
