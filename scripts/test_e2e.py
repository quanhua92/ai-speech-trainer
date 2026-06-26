"""End-to-end REST API test: spin up uvicorn, hit every endpoint over real HTTP.

Usage:
    uv run python scripts/test_e2e.py
    uv run python scripts/test_e2e.py --host 127.0.0.1 --port 8765
    uv run python scripts/test_e2e.py --reuse        # connect to an already-running server
    uv run python scripts/test_e2e.py --url https://localhost:8000 --insecure

Flow: health → create two near-identical references (A vs B, one word changed)
→ list → download B's audio as the "user" attempt → /evaluate/quick and
/evaluate (A's reference vs B's attempt: high but <100, so the diff actually
fires) → history list/detail → stats → leaderboard (extensive: shape, masking,
per-user increments, ranking, truncation) → delete. Exits non-zero on any failure.

The leaderboard checks run against the live server (single- or multi-worker).
Counts are in-memory per worker and flushed every LEADERBOARD_FLUSH_SECONDS, so a
read may land on a worker that hasn't seen an eval yet — we poll with a timeout
that comfortably covers a 60s-flush server.

The first run downloads the Kokoro (~330 MB) and Wav2Vec2 (~1.2 GB) models.
All audio is real Kokoro speech synthesized by the server itself — no mic, no
client-side synthesis.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import httpx

BASE_PATH = "/api/v1"
TIMEOUT = 120.0


def _wait_for_health(client: httpx.Client, tries: int = 60) -> dict:
    last = None
    for _ in range(tries):
        try:
            r = client.get(f"{BASE_PATH}/health", timeout=5.0)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError as e:
            last = e
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reuse", action="store_true", help="connect to an already-running server")
    parser.add_argument(
        "--url",
        help="full base URL of an external server (e.g. https://host:8000); implies --reuse",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS certificate verification (self-signed certs, e.g. scripts/serve.sh)",
    )
    args = parser.parse_args()

    if args.url:
        base_url = args.url.rstrip("/")
        reuse = True  # external URL → always connect, never start a local server
    else:
        base_url = f"http://{args.host}:{args.port}"
        reuse = args.reuse
    server_thread = None

    if not reuse:
        import uvicorn

        from ai_speech_shadowing.api.app import create_app

        config = uvicorn.Config(create_app(), host=args.host, port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        print(f"[e2e] server starting on {base_url}")

    failures: list[str] = []

    with httpx.Client(base_url=base_url, timeout=TIMEOUT, verify=not args.insecure) as client:
        try:
            # 1. health
            health = (
                _wait_for_health(client) if not reuse else client.get(f"{BASE_PATH}/health").json()
            )
            print(f"[e2e] health: status={health['status']} version={health['version']}")
            assert health["status"] == "healthy"

            # 2. create TWO references that differ by exactly one small word.
            #    A is the reference; B's audio becomes the "user" attempt, so
            #    the comparison is real speech that's close-but-not-identical.
            text_a = "The quick brown fox jumps over the lazy dog."
            text_b = "A quick brown fox jumps over the lazy dog."  # "The" → "A"

            ref_a = client.post(f"{BASE_PATH}/references", json={"text": text_a, "language": "en"})
            ref_a.raise_for_status()
            ref_a = ref_a.json()
            ref_b = client.post(f"{BASE_PATH}/references", json={"text": text_b, "language": "en"})
            ref_b.raise_for_status()
            ref_b = ref_b.json()
            print(
                f"[e2e] created references: A={ref_a['id']}  B={ref_b['id']} (differ by one word)"
            )

            # 3. list contains both
            refs = client.get(f"{BASE_PATH}/references").json()
            ids = {r["id"] for r in refs}
            assert {ref_a["id"], ref_b["id"]} <= ids, "created references not in list"
            print(f"[e2e] references listed: {len(refs)}")

            # 4. harvest B's real Kokoro audio as the "user" attempt
            user_audio = client.get(f"{BASE_PATH}/references/{ref_b['id']}/audio")
            user_audio.raise_for_status()
            user_bytes = user_audio.content
            assert user_audio.headers["content-type"].startswith("audio")
            print(f"[e2e] user attempt (B's audio): {len(user_bytes)} bytes")

            # 5. /evaluate/quick: server regenerates ref A on the fly, user is B
            r = client.post(
                f"{BASE_PATH}/evaluate/quick",
                files={"audio": ("user.wav", user_bytes, "audio/wav")},
                data={"text": text_a, "language": "en"},
            )
            r.raise_for_status()
            quick = r.json()
            quick_score = quick["scores"]["composite"]["score"]
            print(
                f"[e2e] quick evaluate (A ref vs B attempt): id={quick['id']} "
                f"composite={quick_score}/100 {quick['scores']['composite']['grade']}"
            )
            assert "scores" in quick
            assert quick["scores"]["pronunciation"]["phoneme_error_rate"] > 0, (
                "expected a non-zero PER — the two texts differ"
            )

            # 6. /evaluate against the pre-generated reference A
            r = client.post(
                f"{BASE_PATH}/evaluate",
                files={"audio": ("user.wav", user_bytes, "audio/wav")},
                data={"reference_id": ref_a["id"]},
            )
            r.raise_for_status()
            evaluation = r.json()
            eval_score = evaluation["scores"]["composite"]["score"]
            print(
                "[e2e] evaluate (A ref vs B attempt): "
                f"id={evaluation['id']} composite={eval_score}/100"
            )
            assert evaluation["reference_id"] == ref_a["id"]
            # close but not identical → high but not perfect
            assert eval_score < 100, "expected <100 — the attempt differs from the reference"
            assert eval_score >= 60, f"expected >=60 for near-identical speech, got {eval_score}"

            # 7. history list + detail
            history = client.get(f"{BASE_PATH}/history").json()
            print(f"[e2e] history: {history['total']} evaluation(s)")
            assert history["total"] >= 2
            detail = client.get(f"{BASE_PATH}/history/{evaluation['id']}").json()
            assert detail["id"] == evaluation["id"]
            print(f"[e2e] history detail OK: {detail['id']}")

            # 8. stats
            stats = client.get(f"{BASE_PATH}/history/stats").json()
            print(
                f"[e2e] stats: total={stats['total_evaluations']} "
                f"avg_composite={stats['average_scores']['composite']}"
            )
            assert stats["total_evaluations"] >= 2

            # 9. leaderboard — extensive checks against the live server.
            #    Counts live in-memory per worker and flush periodically, so a
            #    read may land on a worker that hasn't seen an eval yet; we poll.
            LB_POLL = 90.0
            _hex = set("0123456789abcdef")

            def _lb(c: httpx.Client, limit: int = 50) -> dict:
                r = c.get(f"{BASE_PATH}/leaderboard", params={"limit": limit})
                r.raise_for_status()
                return r.json()

            def _poll_lb(c: httpx.Client, pred, label: str) -> dict:
                deadline = time.time() + LB_POLL
                last: dict | None = None
                while time.time() < deadline:
                    last = _lb(c)
                    if pred(last):
                        return last
                    time.sleep(1.0)
                raise AssertionError(
                    f"leaderboard: {label} not satisfied within {LB_POLL:.0f}s; last={last}"
                )

            def _fresh_user() -> httpx.Client:
                c = httpx.Client(base_url=base_url, timeout=TIMEOUT, verify=not args.insecure)
                c.get(f"{BASE_PATH}/health")  # acquire the user_id cookie
                return c

            def _eval_against(c: httpx.Client, ref_id: str, audio_bytes: bytes) -> dict:
                r = c.post(
                    f"{BASE_PATH}/evaluate",
                    files={"audio": ("user.wav", audio_bytes, "audio/wav")},
                    data={"reference_id": ref_id},
                )
                r.raise_for_status()
                return r.json()

            def _me_count(lb: dict) -> int:
                return lb["me"]["count"] if lb["me"] else 0

            def _has(lb: dict, uid: str) -> bool:
                return any(e["id"] == uid for e in lb["top"])

            # 9a. shape + masking (cheap GETs)
            lb0 = _lb(client)
            assert isinstance(lb0["total_evaluations"], int) and lb0["total_evaluations"] >= 0
            assert isinstance(lb0["top"], list)
            for e in lb0["top"]:
                assert set(e) >= {"rank", "id", "count", "last_evaluated"}
                assert len(e["id"]) == 8 and set(e["id"]) <= _hex, f"masked id not 8 hex: {e['id']}"
            # ranks dense + ordered from 1
            if lb0["top"]:
                ranks = [e["rank"] for e in lb0["top"]]
                assert ranks == sorted(ranks) and ranks[0] == 1, f"ranks not dense: {ranks}"

            # 9b. a brand-new user (no evals) sees its id with count 0, unranked
            u_none = _fresh_user()
            me_none = _lb(u_none)["me"]
            assert me_none is not None, "new user should still get a me row (to show its id)"
            assert me_none["count"] == 0 and me_none["rank"] is None
            assert len(me_none["id"]) == 8  # masked id shown so they can find themselves
            print(
                "[e2e] leaderboard shape OK "
                f"(total={lb0['total_evaluations']}, top={len(lb0['top'])}); "
                f"new-user me id={me_none['id']} count=0 unranked"
            )

            # 9c. u1 evaluates with two DIFFERENT audios (count 2); u2 with one
            #     (count 1). A replay of an already-counted audio must NOT raise
            #     the count (per-user audio dedup).
            audio_a = client.get(f"{BASE_PATH}/references/{ref_a['id']}/audio").content
            u1, u2 = _fresh_user(), _fresh_user()
            base_total = lb0["total_evaluations"]
            _eval_against(u1, ref_a["id"], audio_a)  # +1
            _eval_against(u1, ref_a["id"], user_bytes)  # +1 (different audio)
            _eval_against(u1, ref_a["id"], audio_a)  # replay → deduped, no count
            _eval_against(u2, ref_a["id"], user_bytes)  # +1 for u2 (per-user dedup)

            # 9d. each user sees its own count (poll: covers cross-worker flush lag)
            lb1 = _poll_lb(u1, lambda lb: _me_count(lb) == 2, "u1 count==2 (dedup held)")
            u1_id, u1_count, u1_rank = lb1["me"]["id"], lb1["me"]["count"], lb1["me"]["rank"]
            assert set(u1_id) <= _hex and len(u1_id) == 8
            assert lb1["me"]["last_evaluated"], "last_evaluated should be set after an eval"
            print(f"[e2e] u1: id={u1_id} count={u1_count} rank=#{u1_rank} (3 evals, 1 deduped)")

            lb2 = _poll_lb(u2, lambda lb: _me_count(lb) >= 1, "u2 count>=1")
            u2_id, u2_count, u2_rank = lb2["me"]["id"], lb2["me"]["count"], lb2["me"]["rank"]
            print(f"[e2e] u2: id={u2_id} count={u2_count} rank=#{u2_rank}")

            # 9e. global ranking: u1 (2) strictly ahead of u2 (1)
            both = _poll_lb(
                client,
                lambda lb: _has(lb, u1_id) and _has(lb, u2_id),
                "both users visible in top",
            )
            u1e = next(e for e in both["top"] if e["id"] == u1_id)
            u2e = next(e for e in both["top"] if e["id"] == u2_id)
            assert u1e["count"] > u2e["count"], f"u1 should outrank u2: {u1e} vs {u2e}"
            assert u1e["rank"] < u2e["rank"], f"u1 rank should be better: {u1e} vs {u2e}"
            print(
                "[e2e] ranking OK: "
                f"u1#{u1e['rank']}({u1e['count']}) > u2#{u2e['rank']}({u2e['count']})"
            )

            # 9f. total climbed by the 3 evals we just ran (eventually consistent)
            final = _poll_lb(
                client, lambda lb: lb["total_evaluations"] >= base_total + 3, "total += 3"
            )
            print(f"[e2e] total evaluations {base_total} -> {final['total_evaluations']}")

            # 9g. limit truncation
            limited = _lb(client, limit=1)
            assert len(limited["top"]) <= 1, (
                f"limit=1 should truncate top, got {len(limited['top'])}"
            )

            for c in (u_none, u1, u2):
                c.close()
            print("[e2e] leaderboard checks passed")

            # 10. delete both references
            for rid in (ref_a["id"], ref_b["id"]):
                d = client.delete(f"{BASE_PATH}/references/{rid}")
                d.raise_for_status()
                assert d.status_code == 204
                assert client.get(f"{BASE_PATH}/references/{rid}").status_code == 404
            print("[e2e] deleted both references")

            print("\n[e2e] ALL CHECKS PASSED")

        except Exception as e:
            failures.append(f"{type(e).__name__}: {e}")
            print(f"[e2e] FAILURE: {e}", file=sys.stderr)

    _ = server_thread  # daemon thread dies with the process

    if failures:
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
