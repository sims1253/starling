"""Cross-process GPU isolation using an inherited POSIX flock descriptor.

The lock survives a parent's exit while a native child still holds its inherited
FD. Closing the last FD releases it; token timestamps are observability only.
Use CUDA GPU UUIDs for unambiguous device selection on multi-GPU hosts.

STARLING_GPU_LOCK_DISABLE=1 explicitly disables isolation.
STARLING_GPU_LOCK_DIR overrides the directory containing lock files.
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # native Windows: import remains safe; acquire fails closed
    fcntl: types.ModuleType | None = None  # type: ignore[assignment]
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import types
import uuid as _uuid
from pathlib import Path

from typing_extensions import Self

# v2 token schema. Bumped only on a breaking change to the token *contract*.
TOKEN_VERSION = 2
TOKEN_SCHEMA = "starling.gpu.session"

HEARTBEAT_SEC = 5.0  # how often an in-process holder refreshes observability

_QUERY_CACHE: list[str] | None = None


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
            capture_output=True, text=True, timeout=5, check=False,
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


def _resolve_lock_key(
    cvd: str | None = None,
    uuids: list[str] | None = None,
) -> str:
    """Resolve CUDA visibility without guessing CUDA's device ordering.

    CUDA ordinals can differ from nvidia-smi ordering. On multi-GPU hosts,
    require UUID selection until runtime-based ordinal discovery is available.
    Invalid indices terminate visibility, as specified by CUDA.
    """
    if cvd is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if uuids is None:
        uuids = _query_gpu_uuids()
    if not uuids:
        raise RuntimeError(
            "Cannot discover GPU UUIDs with nvidia-smi; supply an explicit "
            "GpuSession uuid/--uuid shared by all users of this device, or "
            "set STARLING_GPU_LOCK_DISABLE=1 with external serialization."
        )
    chosen = []
    if cvd is None:
        chosen = uuids
    else:
        for part in cvd.split(","):
            if part.startswith("GPU-"):
                matches = [u for u in uuids if u.startswith(part)]
                if len(matches) != 1:
                    raise RuntimeError(f"Unknown or ambiguous CUDA GPU UUID: {part!r}")
                device = matches[0]
            elif part.isascii() and part.isdigit():
                index = int(part)
                if index >= len(uuids):
                    break
                if len(uuids) > 1:
                    raise RuntimeError(
                        "CUDA ordinal order is ambiguous on multi-GPU hosts; "
                        "set CUDA_VISIBLE_DEVICES to a GPU UUID from nvidia-smi."
                    )
                device = uuids[index]
            elif part == "" or part == "-1":
                break
            else:
                raise RuntimeError(f"Unsupported CUDA_VISIBLE_DEVICES entry: {part!r}")
            if device in chosen:
                raise RuntimeError("CUDA_VISIBLE_DEVICES selects the same GPU twice")
            chosen.append(device)
    if not chosen:
        raise RuntimeError("CUDA_VISIBLE_DEVICES selects no GPU")
    return ",".join(sorted(chosen))


def _sanitize(key: str) -> str:
    return key.replace("/", "_").replace(os.sep, "_")


def _default_lock_dir() -> Path:
    return Path(os.environ.get("STARLING_GPU_LOCK_DIR",
                               tempfile.gettempdir())).expanduser()


def _lock_file_path(key: str, lock_dir: str | None = None) -> Path:
    base = Path(lock_dir) if lock_dir else _default_lock_dir()
    return base / f"starling-gpu-{_sanitize(key)}.flock"


# --------------------------------------------------------------------------- #
# token (v2) helpers
# --------------------------------------------------------------------------- #
def _parse_token(path: Path) -> dict | None:
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
        lock_dir: str | None = None,
        uuid: str | None = None,
        wait: bool = True,
        poll_sec: float = 0.2,
        max_wait_sec: float = 600.0,
        heartbeat: bool = True,
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
        self._want_heartbeat = heartbeat
        self._fd: int | None = None
        self._path: Path | None = None
        self._key: str | None = None
        self._owner_id: str | None = None
        self._disabled = (
            os.environ.get("STARLING_GPU_LOCK_DISABLE", "") == "1"
        )
        self._acquired = False
        self._hb_thread: threading.Thread | None = None
        self._hb_stop = threading.Event()
        self._token_lock = threading.Lock()

    # -- introspection -----------------------------------------------------
    def read_token(self) -> dict | None:
        """The current v2 token for this session (or ``None`` if not held)."""
        if self._path is None:
            return None
        return _parse_token(self._path)

    @property
    def owner_id(self) -> str | None:
        return self._owner_id

    @property
    def lock_path(self) -> Path | None:
        return self._path

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def acquire(self) -> Self:
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
                if not os.path.samestat(os.fstat(self._fd), path.stat()):
                    raise ValueError("inherited GPU descriptor names a different file")
                # Environment metadata may outlive its descriptor. Verify both
                # file identity and ownership before treating it as our lock.
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, ValueError, OverflowError):
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
            if self._fd is not None:
                self._path = path
                # This duplicate is intentionally only a nested reference. The
                # original env-published fd remains open for the wrapped
                # process's whole lifetime, so releasing an inner legacy
                # with_gpu_lock cannot release the outer runner's lock. os.dup
                # returns a non-inheritable fd (PEP 446); spawn() explicitly
                # marks it inheritable when another native child is launched.
                self._acquired = True
                self._owner_id = os.environ.get(
                    "STARLING_GPU_LOCK_OWNER", self._owner_id)
                return self
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._owner_id = _uuid.uuid4().hex

        self._fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.set_inheritable(self._fd, True)
            self._flock_with_contention()
            self._acquired = True
            self._write_token()
            if self._want_heartbeat:
                self._start_heartbeat()
        except BaseException:
            self.release()
            raise
        return self

    def release(self) -> None:
        if not self._acquired and self._fd is None:
            return
        self._acquired = False
        if self._disabled:
            self._fd = None
            return
        self._stop_heartbeat()
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

    # -- native children ---------------------------------------------------
    def spawn(self, args, **popen_kwargs) -> subprocess.Popen:
        """``subprocess.Popen`` that inherits the flock fd (``pass_fds``).

        The child keeps the lock held for its whole lifetime even if this
        process is killed. ``pass_fds`` is merged with any caller-supplied fds.
        """
        if not self._acquired:
            raise RuntimeError("GpuSession.spawn before acquire()")
        if self._disabled:
            return subprocess.Popen(args, **popen_kwargs)
        assert self._fd is not None
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
        self._hb_stop.clear()

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
