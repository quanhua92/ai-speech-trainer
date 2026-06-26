# Leaderboard DB (`data/storage/db.json`)

> How the per-user evaluation count leaderboard is stored, kept correct under
> multiple uvicorn workers, and persisted across restarts — without a database.

## Overview

Every successful evaluation bumps a counter keyed by the caller's user id (the
SHA-256 of their `user_id` cookie, see [identity](storage.md#per-user-identity)). A
leaderboard ranks users by evaluation count and exposes the caller's own
count/rank.

The store is a single JSON file on disk:

```
data/storage/db.json
data/storage/db.json.lock   # stable sidecar for the cross-process lock (see below)
```

```json
{
  "schema": 1,
  "total_evaluations": 1234,
  "users": {
    "<64-hex-hash>": { "count": 42, "last_evaluated": "2026-06-26T12:00:00+00:00" }
  },
  "updated_at": "2026-06-26T12:00:00+00:00"
}
```

The on-disk file stores the **full** user hash (needed so counts merge across
workers). The public API masks it (first 8 hex chars) — see `GET /api/v1/leaderboard`.

## Design constraints

1. **No per-request disk I/O.** Evaluation latency must not pay for a leaderboard write.
2. **Multiple workers.** `WORKERS=2` (default) means two separate processes that do
   **not** share memory. They must not overwrite each other's counts.
3. **Sync handlers.** Every route is a plain `def` (no `async def`), so Starlette
   dispatches them to an external threadpool — real concurrent threads within a
   worker process.
4. **Survive restarts.** Counts must persist across deploys/restarts.

These four facts drive every choice below.

## Two concurrency layers

| Layer | Scope | Mechanism |
| --- | --- | --- |
| **Intra-process** (threads in the anyio threadpool, one worker) | concurrent `increment_user` calls | `threading.Lock` |
| **Inter-process** (the N worker processes) | each has its own `_deltas`; disk is shared | `fcntl.flock` + delta-merge at flush |

### Why a threading lock is needed at all

Python's GIL guarantees only **one thread runs bytecode at a time**, but it can
**release between bytecodes**. A counter increment is a *read-modify-write*
spanning several bytecodes:

```
_deltas[uid] += 1   →   get (BINARY_SUBSCR) · add 1 · store (STORE_SUBSCR)
                           └── GIL can switch threads here ──┘
```

So two threadpool threads can both `get` 0 and both `set` 1 → a **lost
increment**, even though never two threads at once. A single `d[k] = v` is
atomic; `count += 1` is the exception. Hence a tiny `threading.Lock` around the
in-memory RMW (and around the flush's snapshot/clear). It is uncontended almost
always, so cost ≈ nil.

If the increment path were `async def` (single event loop, one thread) this
layer would not exist — but the handlers are sync, so it does.

## In-memory model (per worker)

Module-level globals in `core/leaderboard.py`, **one copy per worker process**:

```
_cache:   dict    # last-known full state (disk snapshot + this worker's deltas) — serves reads
_deltas:  dict    # increments THIS worker has produced since its last successful flush
_dirty:   bool    # set on any increment; cleared on a successful flush
```

Reads (`GET /leaderboard`) are served from `_cache` — no disk I/O per read.

## Write path — per evaluation

An evaluation counts only if it is a **real attempt**: the composite score must
reach `LEADERBOARD_MIN_SCORE` (default 30), and a replay of the **same audio**
by the same user is deduped. The gate lives in the evaluate route; the store only
does the count + dedup.

### Score threshold (evaluate route)

```python
if user_id and report.composite_score >= _leaderboard_min_score():   # default 30
    store.increment(user_id, audio_hash(attempt_bytes))
```

This filters out noise/silence/garbage submissions so they don't pad the count.
The threshold is the outer gate — a below-threshold attempt never calls
`increment` (and never claims a dedup slot), so a later good attempt with the
same audio still counts.

### Replay dedup — empty files under `data/storage/hashes/`

`increment(uid, audio_hash)` claims one empty file per (user, audio) before
bumping the count:

```
data/storage/hashes/<uid[0]>/<uid>/<audio_hash>   # 0-byte marker
```

The claim is `os.open(path, O_CREAT | O_EXCL)` — **atomic across workers** on the
shared container filesystem, so two workers racing on the same recording can't
both count it. No lock is needed for the dedup itself (only the in-memory counter
bump takes the thread lock). If the file already exists, `increment` returns
`False` and the count is unchanged.

- The files live under the **persisted mount** (`data/storage/hashes`), so dedup
  survives restarts; each costs ~0 bytes.
- Dedup is **per user**: the same audio submitted by two different users counts
  once for each.
- `audio_hash` is the first 16 hex of `sha256(attempt_bytes)` — the uploaded
  bytes are byte-identical on a replay (the browser re-sends the same blob), so
  the hash is stable.

### In-memory count bump (no db.json write)

```python
# after the dedup claim succeeds (or when audio_hash is None → no dedup)
now = iso_now()
with _thread_lock:
    _cache["users"].setdefault(uid, {})["count"] += 1
    _cache["users"][uid]["last_evaluated"] = now
    _cache["total_evaluations"] += 1
    _deltas[uid] += 1
    _dirty = True
```

No db.json write, no `flock` on the hot path. The dedup file is the only disk
touch, and it's a single atomic syscall.

## Read path — `GET /api/v1/leaderboard` (hot, no disk, no flock)

A leaderboard view is served **entirely from memory** — no `db.json` read, no
`flock`. The store's `leaderboard(limit, me_uid=...)` method:

1. takes a `snapshot()` of `_cache` under the thread lock (a shallow copy of the
   users dict so ranking can't be torn by a concurrent increment);
2. ranks users by **count desc, then `last_evaluated` desc** (ties broken toward
   more-recent activity);
3. builds the response.

`_cache` already contains the current worker's uncommitted deltas (each
increment updates `_cache` live, alongside `_deltas`), so your own count is
visible immediately; what's missing is *other* workers' not-yet-flushed deltas
(see [Eventual consistency](#eventual-consistency)).

### Response

```json
{
  "total_evaluations": 1234,
  "me": { "id": "a1b2c3d4", "count": 42, "rank": 3, "last_evaluated": "2026-06-26T12:00:00+00:00" },
  "top": [
    { "rank": 1, "id": "e5f6a7b8", "count": 99, "last_evaluated": "2026-06-26T12:01:00+00:00" }
  ]
}
```

- **`id` is masked** — first 8 hex of the 64-char hash. Enough to recognise
  yourself, not reversible into a cookie. The on-disk file stores the full hash
  (needed so deltas merge across workers); masking happens only on read.
- **`me`** is computed for the caller (`request.state.user_id`): `rank` = 1 + the
  number of users ranked strictly ahead. `me` is `null` when the caller has no
  entry yet (hasn't evaluated), so a new visitor isn't shown "rank #9999".
- **`top`** is truncated to `limit` (default 10, max 100).

### Cost

`O(n log n)` in the number of ranked users (the sort), per request. Fine at
expected scale; cap `limit` if it ever matters. No disk, no lock contention with
the writer (the snapshot copy is taken in microseconds).

### Optional disk-fresh read

If a view ever needs to be exactly current (every worker's latest flushes), the
endpoint can re-read `db.json` on each view instead of using `_cache`. That is
still **no per-evaluation** disk cost — one read per page-load. Not enabled by
default; the ~15s memory lag (see [Eventual consistency](#eventual-consistency))
is acceptable for a leaderboard.

## Flush — merge deltas, never overwrite (the cross-worker rule)

Each worker tracks only the increments **it** produced since its last flush. On
flush it re-reads the shared disk state and **adds** its deltas on top, then
writes back atomically. It never writes its full `_cache` (which lacks other
workers' counts) — that would erase them.

```python
def flush():
    with _thread_lock:                 # O(1): a reference swap, NOT a copy
        snapshot = _deltas             # local name → the OLD dict object
        _deltas = {}                   # live increments now go to a fresh dict
    if not snapshot:
        return
    try:
        with flock(LOCK_EX, "db.json.lock"):
            disk = read("db.json")          # may include other workers' flushes
            for uid, n in snapshot.items():
                disk["users"].setdefault(uid, {"count": 0, "last_evaluated": None})
                disk["users"][uid]["count"] += n
                disk["users"][uid]["last_evaluated"] = max(disk[...]["last_evaluated"], ...)
            disk["total_evaluations"] = sum(u["count"] for u in disk["users"].values())
            disk["updated_at"] = iso_now()
            write_tmp(); os.replace(tmp, "db.json")    # atomic
        with _thread_lock:
            _cache = disk                    # serve fresh merged state
            _dirty = False
    except Exception:
        # flush failed (disk error, etc.) — put deltas back so they retry next cycle
        with _thread_lock:
            for uid, n in snapshot.items():
                _deltas[uid] = _deltas.get(uid, 0) + n
        raise
```

### Why each safeguard exists

- **Flush deltas, not full state.** Writing the whole `_cache` would clobber
  other workers' counts (this worker's cache doesn't contain them).
- **Re-read disk under `flock`.** The on-disk file is the shared truth; it
  changed since startup because other workers flushed. Merging into a stale
  snapshot would lose their work.
- **`flock`.** Without it, two workers could read the same snapshot and the
  second `os.replace` would discard the first's merge.
- **snapshot + clear under the thread lock.** An increment landing mid-flush
  goes into the *new* `_deltas` and ships next cycle — never lost. The swap is
  O(1) (rebind the name), so the lock is held for nanoseconds.
- **`last_evaluated = max(...)`.** Never regress a timestamp.
- **Recompute `total_evaluations` as a sum.** Drift-proof.

### Worked example (2 workers)

Disk starts `X:10, Y:5, total:15`. Both workers load it into `_cache`, `_deltas={}`.

1. eval → **A** → `inc(X)`: `_deltas_A={X:1}`
2. eval → **B** → `inc(Y)`: `_deltas_B={Y:1}`
3. eval → **A** → `inc(X)`: `_deltas_A={X:2}`

Flush, A first:

- `flock` → re-read disk `{X:10,Y:5,total:15}` → merge `{X:+2}` →
  `{X:12,Y:5,total:17}` → atomic write → clear `_deltas_A`

B flushes:

- `flock` → re-read disk `{X:12,Y:5,total:17}` (now reflects A) → merge
  `{Y:+1}` → `{X:12,Y:6,total:18}` → write → clear `_deltas_B`

Final disk: `X:12, Y:6` — correct (10+2, 5+1).

## The lock file is a stable sidecar — never `db.json`

`fcntl.flock` binds the lock to the **open file description / inode**, not the
path. Our atomic write does `os.replace(tmp, db.json)`, which **swaps the
inode**. If we flocked `db.json` directly:

- Worker A flocks the *old* `db.json` inode, does its replace → `db.json` is a
  *new* inode.
- Worker B opens `db.json` → gets the *new* inode → flocks it **unblocked**
  (A's lock is on the dead old inode) → both merge concurrently → lost updates.

A dedicated `db.json.lock` that is **never replaced** keeps a stable inode, so
every worker flocking that same file actually contends on the same lock.

- The lock file is created lazily (`open(..., "a")`) and **never deleted** — a
  stable, ~0-byte anchor that lives forever.
- It's an **advisory** lock — protects us only because all our workers flock the
  same path.
- `flock` is reliable only on a **local filesystem** (ext4/xfs/apfs). On NFS it
  is unreliable. The Docker bind mount (`./data/storage`) is on the host's local
  disk → fine. Don't point `LEADERBOARD_DB` at an NFS share.

## Crash / leftover-file behaviour

**The OS releases a `flock` the instant the process dies** — clean exit, OOM
`SIGKILL`, or segfault all close the process's file descriptors → the lock is
freed by the kernel. Nothing has to "clean up."

- Previous run crashed → kernel released its flock at crash time → the empty
  `db.json.lock` may remain on disk, but **no lock is held**.
- New run opens that same `db.json.lock` → `flock(LOCK_EX)` → **succeeds
  immediately**. The leftover file is reused.

This is the opposite of the naive `touch lock; work; rm lock` anti-pattern,
where a crash skips the `rm` and leaves a sentinel that deadlocks the next run.
`flock` sidesteps it because **lock state ≠ file existence**.

`db.json` itself is never half-written: the flush writes a temp file then
`os.replace`, which is atomic at the filesystem level. A mid-flush crash lands
in one of exactly two states — old file intact, or new file fully intact.

## Flush scheduling & jitter

A background asyncio task per worker (`_periodic_leaderboard_flush`, started in
the app lifespan) loops: sleep, then `sync()` — write pending deltas if any AND
rebase `_cache` from disk (so idle workers still see other workers' counts).
Defaults:

| Env var | Default | Effect |
| --- | --- | --- |
| `LEADERBOARD_FLUSH_SECONDS` | `15` | Flush interval |
| `LEADERBOARD_DB` | `data/storage/db.json` | DB file path |
| `LEADERBOARD_MIN_SCORE` | `30` | Composite score (0-100) an eval must reach to count |
| `LEADERBOARD_DEDUP_DIR` | `data/storage/hashes` | Dir for the per-(user,audio) dedup marker files |

`flock` already makes simultaneous flushes *safe* (they serialize). Jitter is
added only to **reduce contention**, not to prevent breakage:

- **Per-worker initial phase offset** (PID-based, deterministic, immune to the
  fork-inherited-RNG pitfall): `await asyncio.sleep(os.getpid() % 30)` so the
  two workers' cycles phase-shift apart at boot.
- **Interval jitter**: `await asyncio.sleep(interval + random.uniform(-5, 5))` so they
  don't slowly re-sync.

Net: the lock is almost never contended; if jitter ever failed to separate them,
the flock keeps it correct — just briefly serialized.

## Graceful shutdown

The lifespan `finally` block runs **one final flush** before the process exits,
so normal restarts/deploys commit all pending deltas and lose nothing. The only
data-loss window is a hard crash (`kill -9` / OOM) between flushes — up to
~`LEADERBOARD_FLUSH_SECONDS` of one worker's deltas. This is the accepted
tradeoff for "no per-request disk."

## Docker mount

```yaml
# docker-compose.yml
volumes:
  - app-data:/app/data   # named volume — persists history, references, AND the leaderboard
```

`db.json` and `hashes/` live at `/app/data/storage/` **inside the named volume**
— no separate bind mount. The named volume persists across rebuilds and
`down`/`up` (only `down -v` wipes it), which is what makes the leaderboard
survive restarts.

> **Why not a bind mount?** A bind mount of `./data/storage` *nested inside* the
> named volume would be created root-owned on the host; the container runs as a
> non-root user, so the app couldn't create `data/storage/hashes/…` →
> `PermissionError` on every counted evaluation. The named volume's ownership
> matches the image's `/app/data`, so writes just work.

Local dev (`scripts/serve.sh`) uses the same `data/storage/db.json` path relative
to the project root — no Docker involved.

`data/storage/` is git-ignored — the db.json is runtime state, not source.

## Eventual consistency

Each worker serves reads from its own `_cache`. To keep that cache from going
stale on an **idle worker** (one that only serves reads and never increments,
so it would otherwise never re-read disk), the periodic task runs `sync()` every
tick — not `flush_if_dirty()`. `sync()` does two things:

1. `flush()` — write this worker's pending deltas to `db.json` (if any);
2. `_rebase_cache()` — **always** recompute `_cache = latest disk state + this
   worker's uncommitted deltas`.

The rebase is a plain `_load(db.json)` (no `flock` needed — `os.replace` makes
reads never torn) followed by re-applying the in-memory deltas, under the thread
lock. So even a worker that has produced zero increments picks up the counts
other workers flushed, every interval.

Net: a leaderboard view can be **~`LEADERBOARD_FLUSH_SECONDS` behind** the true
global state (default 15s), regardless of whether the serving worker is active
or idle. This is acceptable for a leaderboard. If exact reads were ever required,
the endpoint could re-read disk on each view (disk only — still not per-eval).

## Test coverage

`tests/test_leaderboard.py`:

- increment bumps count, `total_evaluations`, and `last_evaluated`
- ordering by count (`top_n`)
- masked id (first 8 hex)
- missing / corrupt `db.json` → fresh start, never crash
- delta-merge correctness across two simulated worker shards
- idle worker refreshes its cache from disk via `sync()` (sees other workers' counts)
- snapshot+clear: an increment during flush ships next cycle (no loss)
- failed flush restores deltas (no loss)
- atomic write (no half-written file observable)
- replay dedup: same audio counts once (per user), distinct audios each count,
  dedup survives a new store on the same dedup dir
