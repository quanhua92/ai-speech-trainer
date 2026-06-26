"""Latency benchmark for the REST API: time every endpoint over real HTTP.

Measures per-endpoint wall time so you can see cold-load vs warm costs
(Wav2Vec2 model load, Kokoro TTS synthesis, numba JIT) — the things that
dominate a local/CPU deployment. With multiple uvicorn workers the first
few /evaluate calls each pay a cold model load as they land on different
workers, then settle to the warm latency.

Usage:
    uv run python scripts/bench_api.py
    uv run python scripts/bench_api.py --host 127.0.0.1 --port 8765
    uv run python scripts/bench_api.py --url https://shadowing.huahongquan.com
    uv run python scripts/bench_api.py --url https://localhost:8000 --insecure
    uv run python scripts/bench_api.py --rounds 5      # more /evaluate samples

Creates one reference, reuses its Kokoro audio as the "user" attempt, runs
the timed sequence, then deletes the reference. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import httpx

BASE_PATH = "/api/v1"
TIMEOUT = 180.0
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def _time(label: str, fn, timings: list[tuple[str, float]]) -> httpx.Response:
    t0 = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t0
    timings.append((label, dt))
    print(f"{label:36s} {dt:6.2f}s  (http {r.status_code})")
    r.raise_for_status()
    return r


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
    parser.add_argument("--text", default=DEFAULT_TEXT, help="reference sentence to synthesize")
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="number of /evaluate calls (shows cold -> warm per worker)",
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
        import threading

        import uvicorn

        from ai_speech_shadowing.api.app import create_app

        config = uvicorn.Config(create_app(), host=args.host, port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        print(f"[bench] server starting on {base_url}")

    timings: list[tuple[str, float]] = []
    rid: str | None = None

    with httpx.Client(base_url=base_url, timeout=TIMEOUT, verify=not args.insecure) as client:
        try:
            # wait for health (only when we started the server ourselves)
            for _ in range(60):
                if client.get(f"{BASE_PATH}/health").status_code == 200:
                    break
                time.sleep(0.5)

            _time("GET /health", lambda: client.get(f"{BASE_PATH}/health"), timings)
            r = _time(
                "POST /references (TTS synth)",
                lambda: client.post(
                    f"{BASE_PATH}/references", json={"text": args.text, "language": "en"}
                ),
                timings,
            )
            rid = r.json()["id"]
            _time(
                "GET /references/{id}/audio",
                lambda: client.get(f"{BASE_PATH}/references/{rid}/audio"),
                timings,
            )
            audio = client.get(f"{BASE_PATH}/references/{rid}/audio").content
            files = {"audio": ("user.wav", audio, "audio/wav")}

            for i in range(args.rounds):
                tag = " (cold?)" if i == 0 else ""
                _time(
                    f"POST /evaluate #{i + 1}{tag}",
                    lambda: client.post(
                        f"{BASE_PATH}/evaluate", files=files, data={"reference_id": rid}
                    ),
                    timings,
                )

            _time(
                "POST /evaluate/quick (TTS+eval)",
                lambda: client.post(
                    f"{BASE_PATH}/evaluate/quick",
                    files=files,
                    data={"text": args.text, "language": "en"},
                ),
                timings,
            )

        except Exception as e:
            print(f"[bench] FAILURE: {e}", file=sys.stderr)
            return 1
        finally:
            if rid is not None:
                client.delete(f"{BASE_PATH}/references/{rid}")

    _ = server_thread

    evals = [dt for label, dt in timings if label.startswith("POST /evaluate #")]
    if len(evals) >= 2:
        print(
            f"\n[bench] /evaluate: first={evals[0]:.2f}s  warm median="
            f"{statistics.median(evals[1:]):.2f}s  min={min(evals[1:]):.2f}s"
        )
    print("[bench] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
