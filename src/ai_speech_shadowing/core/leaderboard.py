"""Per-user evaluation-count leaderboard with an in-memory cache and a
periodic, cross-worker-safe flush.

Design (see ``docs/db.md`` for the full rationale):

* Each worker keeps an in-memory ``_cache`` (the last disk snapshot **plus** this
  worker's uncommitted deltas) — what reads are served from, with no disk I/O.
* Each evaluation bumps ``_cache`` and a per-worker ``_deltas`` map (a cheap
  ``threading.Lock`` guards the non-atomic ``count += 1``; sync handlers run in
  the anyio threadpool, so real threads mutate this concurrently).
* A background task flushes every ~minute: under ``fcntl.flock`` on a stable
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
SCHEMA: int = 1
_MASK_LEN: int = 8


# --------------------------------------------------------------------------- #
# free helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _empty_state() -> dict[str, Any]:
    return {"schema": SCHEMA, "total_evaluations": 0, "users": {}, "updated_at": None}


def mask_id(uid: str) -> str:
    """First 8 hex chars of a user id — enough to recognise yourself, not reversible."""
    return uid[:_MASK_LEN]


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
            users[uid] = {"count": 0, "last_evaluated": None}
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

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._lock_path = self._db_path.with_name(self._db_path.name + ".lock")
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = _empty_state()
        self._deltas: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._dirty = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._cache = _load(self._db_path)
            self._loaded = True

    # ---- hot path (no disk) --------------------------------------------- #
    def increment(self, uid: str) -> None:
        """Bump the caller's count in memory + record a delta for the next flush."""
        if not uid:
            return
        now = _now_iso()
        with self._lock:
            self._ensure_loaded()
            d = self._deltas.setdefault(uid, {"count": 0, "last_evaluated": None})
            d["count"] += 1
            d["last_evaluated"] = now
            u = self._cache["users"].setdefault(uid, {"count": 0, "last_evaluated": None})
            u["count"] += 1
            u["last_evaluated"] = now
            self._cache["total_evaluations"] += 1
            self._dirty = True

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
