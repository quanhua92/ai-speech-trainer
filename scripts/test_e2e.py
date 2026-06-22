"""End-to-end REST API test: spin up uvicorn, hit every endpoint over real HTTP.

Usage:
    uv run python scripts/test_e2e.py
    uv run python scripts/test_e2e.py --host 127.0.0.1 --port 8765
    uv run python scripts/test_e2e.py --reuse        # connect to an already-running server

Flow: health → create two near-identical references (A vs B, one word changed)
→ list → download B's audio as the "user" attempt → /evaluate/quick and
/evaluate (A's reference vs B's attempt: high but <100, so the diff actually
fires) → history list/detail → stats → delete. Exits non-zero on any failure.

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
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    server_thread = None

    if not args.reuse:
        import uvicorn

        from ai_speech_shadowing.api.app import create_app

        config = uvicorn.Config(create_app(), host=args.host, port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        print(f"[e2e] server starting on {base_url}")

    failures: list[str] = []

    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as client:
        try:
            # 1. health
            health = (
                _wait_for_health(client)
                if not args.reuse
                else client.get(f"{BASE_PATH}/health").json()
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

            # 9. delete both references
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
