"""Tests for the in-memory leaderboard engine (core/leaderboard.py).

Covers the two correctness-critical properties that make the multi-worker
design work:
  * exact counting under concurrent threadpool-style increments (threading.Lock)
  * delta-merge flush — each worker only adds its own increments, never
    overwrites, so two stores on the same db.json sum correctly
plus the read path (masking, me/rank, ordering) and robustness (corrupt/missing
db, failed-flush delta restore).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ai_speech_shadowing.core.leaderboard import (
    SCHEMA,
    LeaderboardStore,
    default_db_path,
    mask_id,
)


@pytest.fixture
def store(tmp_path: Path) -> LeaderboardStore:
    return LeaderboardStore(tmp_path / "db.json")


def _db_json(db_path: Path) -> dict:
    return json.loads(db_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# increment + read (in-memory)
# --------------------------------------------------------------------------- #
class TestIncrementRead:
    def test_increment_bumps_count_total_and_last_evaluated(self, store: LeaderboardStore) -> None:
        uid = "a" * 64
        store.increment(uid)
        store.increment(uid)
        snap = store.snapshot()
        assert snap["total_evaluations"] == 2
        assert snap["users"][uid]["count"] == 2
        assert snap["users"][uid]["last_evaluated"] is not None

    def test_increment_empty_uid_is_noop(self, store: LeaderboardStore) -> None:
        store.increment("")
        assert store.snapshot()["total_evaluations"] == 0

    def test_read_reflects_increment_immediately_no_disk(self, store: LeaderboardStore) -> None:
        # no db.json exists yet, but reads still work (in-memory cache)
        assert not store.db_path.exists()
        store.increment("b" * 64)
        lb = store.leaderboard()
        assert lb["total_evaluations"] == 1
        assert lb["top"][0]["count"] == 1


# --------------------------------------------------------------------------- #
# masking / me / ordering
# --------------------------------------------------------------------------- #
class TestLeaderboardView:
    def test_mask_id_is_first_8_chars(self) -> None:
        uid = "abcdef0123456789" * 4  # 64 chars
        assert mask_id(uid) == "abcdef01"
        assert len(mask_id(uid)) == 8

    def test_me_for_unknown_user_shows_id_count_zero_unranked(
        self, store: LeaderboardStore
    ) -> None:
        # a user who hasn't evaluated still gets a row (id visible, count 0, no rank)
        store.increment("a" * 64)
        me = store.leaderboard(me_uid="z" * 64)["me"]
        assert me is not None
        assert me["id"] == mask_id("z" * 64)
        assert me["count"] == 0
        assert me["rank"] is None

    def test_me_rank_and_masked_id(self, store: LeaderboardStore) -> None:
        # three users: c(3) > b(2) > a(1)
        for _ in range(3):
            store.increment("c" * 64)
        for _ in range(2):
            store.increment("b" * 64)
        store.increment("a" * 64)
        lb = store.leaderboard(me_uid="b" * 64)
        assert lb["me"] is not None
        assert lb["me"]["count"] == 2
        assert lb["me"]["rank"] == 2  # one user (c) strictly ahead
        assert lb["me"]["id"] == mask_id("b" * 64)

    def test_top_ordering_count_desc(self, store: LeaderboardStore) -> None:
        for _ in range(3):
            store.increment("c" * 64)
        store.increment("b" * 64)
        store.increment("b" * 64)
        store.increment("a" * 64)
        top = store.leaderboard(limit=2)["top"]
        assert [t["count"] for t in top] == [3, 2]
        assert top[0]["rank"] == 1 and top[1]["rank"] == 2

    def test_limit_caps_top(self, store: LeaderboardStore) -> None:
        for i in range(5):
            store.increment(f"{i:064x}")
        assert len(store.leaderboard(limit=3)["top"]) == 3


# --------------------------------------------------------------------------- #
# flush — single store
# --------------------------------------------------------------------------- #
class TestFlush:
    def test_flush_writes_db_and_clears_dirty(self, store: LeaderboardStore) -> None:
        store.increment("a" * 64)
        store.increment("a" * 64)
        merged = store.flush()
        assert merged == 2
        assert store.db_path.exists()
        data = _db_json(store.db_path)
        assert data["schema"] == SCHEMA
        assert data["total_evaluations"] == 2
        assert data["users"]["a" * 64]["count"] == 2
        # second flush with no new increments is a no-op
        assert store.flush() == 0

    def test_flush_if_dirty_skips_when_clean(self, store: LeaderboardStore) -> None:
        assert store.flush_if_dirty() == 0
        assert not store.db_path.exists()  # nothing written

    def test_reload_picks_up_disk_state(self, store: LeaderboardStore) -> None:
        store.increment("a" * 64)
        store.flush()
        # a fresh store loads the persisted counts
        reloaded = LeaderboardStore(store.db_path)
        snap = reloaded.snapshot()
        assert snap["total_evaluations"] == 1
        assert snap["users"]["a" * 64]["count"] == 1

    def test_failed_flush_restores_deltas_no_loss(self, store: LeaderboardStore) -> None:
        store.increment("a" * 64)
        # force the disk merge to fail
        store._merge_to_disk = lambda deltas: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            store.flush()
        # deltas restored -> a subsequent (working) flush still ships them
        store._merge_to_disk = LeaderboardStore._merge_to_disk.__get__(store, LeaderboardStore)
        assert store.flush() == 1
        assert _db_json(store.db_path)["total_evaluations"] == 1


# --------------------------------------------------------------------------- #
# cross-worker delta-merge (the core correctness guarantee)
# --------------------------------------------------------------------------- #
class TestCrossWorkerMerge:
    def test_two_stores_on_same_db_sum_correctly(self, tmp_path: Path) -> None:
        db = tmp_path / "db.json"
        a = LeaderboardStore(db)
        b = LeaderboardStore(db)
        # worker A increments X twice; worker B increments Y once
        a.increment("x" * 64)
        a.increment("x" * 64)
        b.increment("y" * 64)
        a.flush()
        b.flush()
        # a third, fresh store reads the merged disk state
        c = LeaderboardStore(db)
        snap = c.snapshot()
        assert snap["users"]["x" * 64]["count"] == 2
        assert snap["users"]["y" * 64]["count"] == 1
        assert snap["total_evaluations"] == 3

    def test_interleaved_flushes_never_lose(self, tmp_path: Path) -> None:
        db = tmp_path / "db.json"
        a = LeaderboardStore(db)
        b = LeaderboardStore(db)
        a.increment("u" * 64)
        a.flush()  # disk: u=1
        b.increment("u" * 64)
        b.flush()  # disk: u=2 (B re-read u=1, +1)
        a.increment("u" * 64)
        a.flush()  # disk: u=3
        assert LeaderboardStore(db).snapshot()["users"]["u" * 64]["count"] == 3


# --------------------------------------------------------------------------- #
# thread-safety of the in-memory increment (sync handlers run in a threadpool)
# --------------------------------------------------------------------------- #
class TestConcurrency:
    def test_no_lost_increments_under_threads(self, tmp_path: Path) -> None:
        store = LeaderboardStore(tmp_path / "db.json")
        uid = "c" * 64
        n_threads, per_thread = 16, 500

        def worker() -> None:
            for _ in range(per_thread):
                store.increment(uid)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # with the lock, count is exact; without it, this would undercount
        assert store.snapshot()["users"][uid]["count"] == n_threads * per_thread
        assert store.snapshot()["total_evaluations"] == n_threads * per_thread
        # and the flush ships the exact total
        assert store.flush() == n_threads * per_thread


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_missing_db_starts_empty(self, tmp_path: Path) -> None:
        store = LeaderboardStore(tmp_path / "does-not-exist.json")
        assert store.snapshot()["total_evaluations"] == 0

    def test_corrupt_db_starts_empty_no_crash(self, tmp_path: Path) -> None:
        db = tmp_path / "db.json"
        db.write_text("{not valid json", encoding="utf-8")
        store = LeaderboardStore(db)
        snap = store.snapshot()
        assert snap["total_evaluations"] == 0
        assert snap["users"] == {}
        # writing still works (overwrites the corrupt file atomically)
        store.increment("a" * 64)
        store.flush()
        assert _db_json(db)["total_evaluations"] == 1

    def test_atomic_write_no_half_file(self, store: LeaderboardStore) -> None:
        # after flush, no leftover .tmp file remains alongside db.json
        store.increment("a" * 64)
        store.flush()
        assert store.db_path.exists()
        assert not store.db_path.with_name(store.db_path.name + ".tmp").exists()

    def test_lock_file_is_created_and_reused(self, tmp_path: Path) -> None:
        store = LeaderboardStore(tmp_path / "db.json")
        store.increment("a" * 64)
        store.flush()
        lock = tmp_path / "db.json.lock"
        assert lock.exists()
        # a second store can lock the same file immediately (advisory, released)
        store2 = LeaderboardStore(tmp_path / "db.json")
        store2.increment("b" * 64)
        store2.flush()  # no deadlock


def test_default_db_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEADERBOARD_DB", raising=False)
    from ai_speech_shadowing.core import leaderboard as mod

    assert str(mod.default_db_path()).replace("\\", "/").endswith("data/storage/db.json")
    monkeypatch.setenv("LEADERBOARD_DB", "/tmp/custom/lb.json")
    assert default_db_path() == Path("/tmp/custom/lb.json")
