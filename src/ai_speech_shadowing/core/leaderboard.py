"""Per-user evaluation-count leaderboard with an in-memory cache and a
periodic, cross-worker-safe flush.

Design (see ``docs/db.md`` for the full rationale):

* Each worker keeps an in-memory ``_cache`` (the last disk snapshot **plus** this
  worker's uncommitted deltas) — what reads are served from, with no disk I/O.
* Each evaluation bumps ``_cache`` and a per-worker ``_deltas`` map (a cheap
  ``threading.Lock`` guards the non-atomic ``count += 1``; sync handlers run in
  the anyio threadpool, so real threads mutate this concurrently).
* A background task flushes every ~15s (default): under ``fcntl.flock`` on a stable
  sidecar lock file it re-reads the shared ``db.json``, **adds** this worker's
  deltas on top, and writes back atomically (temp + ``os.replace``). Merging
  deltas (never the full cache) is what keeps counts correct across workers —
  each process only ever contributes its own increments.
* A graceful-shutdown flush commits pending deltas so counts survive restarts.

The lock file is a stable inode anchor (never ``os.replace``'d, never deleted);
``flock`` is released by the OS the instant the process dies, so leftover files
from a crash are harmless and reused.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: Path = Path("data/storage/db.json")
DEFAULT_DEDUP_DIR: Path = Path("data/storage/hashes")
SCHEMA: int = 1
_MASK_LEN: int = 8
_HASH_LEN: int = 16


# --------------------------------------------------------------------------- #
# free helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _empty_state() -> dict[str, Any]:
    return {"schema": SCHEMA, "total_evaluations": 0, "users": {}, "updated_at": None}


def _new_user() -> dict[str, Any]:
    return {"count": 0, "last_evaluated": None}


def mask_id(uid: str) -> str:
    """First 8 hex chars of a user id — enough to recognise yourself, not reversible."""
    return uid[:_MASK_LEN]


def audio_hash(data: bytes) -> str:
    """Short stable hash of an attempt's audio bytes — keys the per-user replay dedup."""
    import hashlib

    return hashlib.sha256(data).hexdigest()[:_HASH_LEN]


def _dedup_claim(path: Path) -> bool:
    """Atomically claim a dedup slot via an exclusive file create.

    ``O_CREAT | O_EXCL`` is atomic across processes on the same filesystem, so
    two workers racing to count the same (user, audio) can't both win — no lock
    needed. Returns True if we created the file (first time), False if it already
    existed (replay → deduped). The empty files live under ``dedup_dir`` (default
    ``data/storage/hashes``, inside the persisted mount) so dedup survives
    restarts; they cost ~0 bytes each.
    """
    import errno

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError as e:
        if e.errno == errno.EEXIST:
            return False
        raise
    os.close(fd)
    return True


def _max_ts(a: str | None, b: str | None) -> str | None:
    """Later of two ISO timestamps (None = lowest). Same-format strings compare lexically."""
    return b if (a is None or (b is not None and b > a)) else a


def _load(db_path: Path) -> dict[str, Any]:
    """Read + normalise db.json. Missing/corrupt → fresh empty state (never raise)."""
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    users = data.get("users")
    if not isinstance(users, dict):
        users = {}
        data["users"] = users
    for uid, u in list(users.items()):
        if not isinstance(u, dict):
            users[uid] = _new_user()
            continue
        u.setdefault("count", 0)
        u.setdefault("last_evaluated", None)
    data["schema"] = SCHEMA
    # recompute total drift-proof on every read
    data["total_evaluations"] = sum(int(u["count"]) for u in users.values())
    data.setdefault("updated_at", None)
    return data


def _atomic_write(db_path: Path, state: dict[str, Any]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, db_path)


@contextmanager
def _flock(lock_path: Path):
    """Exclusive advisory lock on a stable sidecar file. Released on close/exit.

    ``flock`` binds to the open file description (inode); because ``db.json`` is
    replaced atomically (new inode) we must lock a separate file that is never
    replaced, so all workers contend on the same lock.
    """
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # releasing happens implicitly on close too, but be explicit + robust.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
class LeaderboardStore:
    """Process-wide in-memory leaderboard state + cross-worker flush.

    One instance per worker process. Tests instantiate their own with a tmp
    ``db_path``; the API holds a singleton via ``EngineState``.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        dedup_dir: str | Path = DEFAULT_DEDUP_DIR,
    ) -> None:
        self._db_path = Path(db_path)
        self._lock_path = self._db_path.with_name(self._db_path.name + ".lock")
        # Empty-file-per-(user,audio) replay dedup. Lives under the persisted
        # volume so it survives restarts; O_EXCL makes it lock-free across workers.
        self._dedup_dir = Path(dedup_dir)
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = _empty_state()
        self._deltas: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._dirty = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def dedup_dir(self) -> Path:
        return self._dedup_dir

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._cache = _load(self._db_path)
            self._loaded = True

    # ---- hot path (no db.json write) ------------------------------------ #
    def increment(self, uid: str, audio_hash: str | None = None) -> bool:
        """Count one evaluation for ``uid``, unless the same audio was counted
        already (per-user replay dedup via an empty file under ``dedup_dir``).

        Returns True when the count increased, False when deduped. The dedup
        claim (``O_CREAT | O_EXCL``) is atomic across workers on the shared
        filesystem, so no lock is needed for the dedup itself; only the in-memory
        counter bump takes the thread lock.
        """
        if not uid:
            return False
        if audio_hash:
            # shard by the first hex char of the uid to avoid one giant directory
            claim = self._dedup_dir / uid[0] / uid / audio_hash
            if not _dedup_claim(claim):
                return False  # replay of the same recording — don't double-count
        now = _now_iso()
        with self._lock:
            self._ensure_loaded()
            u = self._cache["users"].setdefault(uid, _new_user())
            u["count"] += 1
            u["last_evaluated"] = now
            self._cache["total_evaluations"] += 1
            d = self._deltas.setdefault(uid, {"count": 0, "last_evaluated": None})
            d["count"] += 1
            d["last_evaluated"] = now
            self._dirty = True
            return True

    # ---- reads (no disk) ------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the current in-memory state (disk snapshot + deltas)."""
        with self._lock:
            self._ensure_loaded()
            return {
                "schema": self._cache.get("schema", SCHEMA),
                "total_evaluations": self._cache.get("total_evaluations", 0),
                "users": {
                    uid: {
                        "count": v.get("count", 0),
                        "last_evaluated": v.get("last_evaluated"),
                    }
                    for uid, v in self._cache.get("users", {}).items()
                },
                "updated_at": self._cache.get("updated_at"),
            }

    def leaderboard(self, limit: int = 10, *, me_uid: str | None = None) -> dict[str, Any]:
        """Ranking served from memory: total, the caller's own row, and top-N."""
        snap = self.snapshot()
        users: dict[str, dict[str, Any]] = snap["users"]
        ranked = sorted(
            users.items(),
            key=lambda kv: (kv[1].get("count", 0), kv[1].get("last_evaluated") or ""),
            reverse=True,
        )
        top = [
            {
                "rank": i + 1,
                "id": mask_id(uid),
                "count": v.get("count", 0),
                "last_evaluated": v.get("last_evaluated"),
            }
            for i, (uid, v) in enumerate(ranked[:limit])
        ]
        me: dict[str, Any] | None = None
        if me_uid:
            # Always return the caller's own row so the UI can show "my id"
            # even before they've evaluated — count 0 / rank null in that case.
            me_v = users.get(me_uid, {"count": 0, "last_evaluated": None})
            my_count = me_v.get("count", 0)
            if my_count > 0:
                my_ts = me_v.get("last_evaluated") or ""
                # rank = 1 + number of users strictly ahead (count desc, ts desc)
                rank: int | None = 1 + sum(
                    1
                    for uid, v in users.items()
                    if uid != me_uid
                    and (
                        v.get("count", 0) > my_count
                        or (
                            v.get("count", 0) == my_count
                            and (v.get("last_evaluated") or "") > my_ts
                        )
                    )
                )
            else:
                rank = None  # not ranked yet (no evaluations)
            me = {
                "id": mask_id(me_uid),
                "count": my_count,
                "rank": rank,
                "last_evaluated": me_v.get("last_evaluated"),
            }
        return {"total_evaluations": snap["total_evaluations"], "me": me, "top": top}

    # ---- flush (disk; fcntl-serialised across workers) ------------------ #
    def flush(self) -> int:
        """Merge this worker's deltas into db.json. Returns the count merged.

        No-op when there is nothing to flush. On failure the deltas are restored
        so they retry on the next cycle (no loss).
        """
        with self._lock:
            if not self._deltas:
                return 0
            snapshot = self._deltas
            self._deltas = {}
        try:
            merged = self._merge_to_disk(snapshot)
        except Exception:
            # put the deltas back so they ship next cycle
            with self._lock:
                for uid, d in snapshot.items():
                    cur = self._deltas.setdefault(uid, {"count": 0, "last_evaluated": None})
                    cur["count"] += d["count"]
                    cur["last_evaluated"] = _max_ts(cur["last_evaluated"], d["last_evaluated"])
            raise
        with self._lock:
            self._cache = merged
            self._dirty = False
        return sum(int(d["count"]) for d in snapshot.values())

    def flush_if_dirty(self) -> int:
        """Flush only when increments happened since the last successful flush."""
        with self._lock:
            dirty = self._dirty
        return self.flush() if dirty else 0

    def sync(self) -> int:
        """Periodic tick: ship pending deltas AND refresh ``_cache`` from disk.

        Without the refresh, a worker that only serves reads (never increments)
        would never re-read ``db.json`` and its view would freeze at startup —
        blind to counts other workers flush. So every tick we rebase
        ``_cache = latest disk state + this worker's uncommitted deltas`` (writes
        only when there are deltas). The disk read needs no ``flock``: ``os.replace``
        makes ``db.json`` reads never torn.
        """
        written = self.flush()  # writes if dirty; no-op (no rebase) when clean
        self._rebase_cache()
        return written

    def _rebase_cache(self) -> None:
        """Recompute ``_cache`` from the latest on-disk state plus our deltas."""
        with self._lock:
            disk = _load(self._db_path)
            users = disk["users"]
            for uid, d in self._deltas.items():
                u = users.setdefault(uid, _new_user())
                u["count"] += d["count"]
                u["last_evaluated"] = _max_ts(u.get("last_evaluated"), d.get("last_evaluated"))
            disk["total_evaluations"] = sum(int(u["count"]) for u in users.values())
            self._cache = disk

    def _merge_to_disk(self, deltas: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Re-read shared db.json, add our deltas, write atomically. Returns merged."""
        with _flock(self._lock_path):
            disk = _load(self._db_path)
            users = disk["users"]
            for uid, d in deltas.items():
                du = users.setdefault(uid, {"count": 0, "last_evaluated": None})
                du["count"] += d["count"]
                du["last_evaluated"] = _max_ts(du["last_evaluated"], d["last_evaluated"])
            disk["total_evaluations"] = sum(int(u["count"]) for u in users.values())
            disk["updated_at"] = _now_iso()
            _atomic_write(self._db_path, disk)
            return disk


def default_db_path() -> Path:
    """Resolve the DB path from ``LEADERBOARD_DB`` (default data/storage/db.json)."""
    env = os.environ.get("LEADERBOARD_DB")
    return Path(env) if env else DEFAULT_DB_PATH


def default_dedup_dir() -> Path:
    """Resolve the dedup dir from ``LEADERBOARD_DEDUP_DIR`` (default data/storage/hashes)."""
    env = os.environ.get("LEADERBOARD_DEDUP_DIR")
    return Path(env) if env else DEFAULT_DEDUP_DIR
