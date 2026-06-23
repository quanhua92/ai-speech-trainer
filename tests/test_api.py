"""FastAPI endpoint tests.

Fast tests use FastAPI's TestClient against the in-process app with the engine
state pointed at tmp dirs (no model needed). The full /evaluate flow is opt-in
slow (loads the Wav2Vec2 + Kokoro models).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_speech_shadowing.api import deps
from ai_speech_shadowing.api.app import create_app
from ai_speech_shadowing.tts.generator import ReferenceConfig, ReferenceManager


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    state = deps.EngineState(
        reference_manager=ReferenceManager(ReferenceConfig(base_dir=tmp_path / "refs")),
        history_dir=tmp_path / "history",
    )
    deps.reset_state(state)
    with TestClient(create_app()) as c:
        yield c
    deps.reset_state()


# --------------------------------------------------------------------------- #
# Demo page
# --------------------------------------------------------------------------- #
class TestDemo:
    def test_demo_serves_html(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "ai-speech-shadowing" in r.text
        assert "/api/v1" in r.text  # it talks to the API


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class TestHealth:
    def test_healthy(self, client: TestClient) -> None:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "wav2vec2" in body["models"]
        assert body["models"]["wav2vec2"]["loaded"] is False  # lazy


# --------------------------------------------------------------------------- #
# References (no model needed for list/get-missing)
# --------------------------------------------------------------------------- #
class TestReferences:
    def test_list_empty(self, client: TestClient) -> None:
        r = client.get("/api/v1/references")
        assert r.status_code == 200
        assert r.json() == []

    def test_get_missing_404(self, client: TestClient) -> None:
        r = client.get("/api/v1/references/nope")
        assert r.status_code == 404

    def test_delete_missing_404(self, client: TestClient) -> None:
        r = client.delete("/api/v1/references/nope")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# History (no model needed when empty)
# --------------------------------------------------------------------------- #
class TestHistory:
    def test_list_empty(self, client: TestClient) -> None:
        r = client.get("/api/v1/history")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["items"] == []
        assert body["limit"] == 100
        assert body["offset"] == 0

    def test_list_pagination(self, client: TestClient) -> None:
        r = client.get("/api/v1/history", params={"limit": 5, "offset": 0, "sort": "asc"})
        assert r.status_code == 200
        assert r.json()["limit"] == 5

    def test_stats_empty(self, client: TestClient) -> None:
        r = client.get("/api/v1/history/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_evaluations"] == 0
        assert body["trend"] == "insufficient"

    def test_get_missing_404(self, client: TestClient) -> None:
        r = client.get("/api/v1/history/eval_nope")
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_evaluate_requires_form_fields(self, client: TestClient) -> None:
        # no form fields at all -> 422
        r = client.post("/api/v1/evaluate")
        assert r.status_code == 422

    def test_create_reference_rejects_empty_text(self, client: TestClient) -> None:
        r = client.post("/api/v1/references", json={"text": ""})
        assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Path-traversal regression (HTTP layer)
# --------------------------------------------------------------------------- #
class TestPathTraversal:
    """Regression tests for the path-traversal fix.

    httpx/TestClient normalizes bare ".." out of the URL before it reaches the
    app, so the encoded "%2e%2e" form is used to actually exercise the handler.
    The critical invariant for every case: real data must survive.
    """

    @staticmethod
    def _seed_reference(slug: str = "hello-world") -> Path:
        """Plant a real reference dir under the configured base_dir."""
        base = deps.get_state().reference_manager.config.base_dir
        d = base / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text('{"text":"hi","default_speaker":"af_heart"}')
        return d

    def test_delete_encoded_dotdot_blocked_and_preserves_data(self, client: TestClient) -> None:
        ref = self._seed_reference()
        r = client.delete("/api/v1/references/%2e%2e")
        assert r.status_code == 400
        assert ref.exists()  # the real reference was NOT deleted

    def test_get_encoded_dotdot_returns_400(self, client: TestClient) -> None:
        r = client.get("/api/v1/references/%2e%2e")
        assert r.status_code == 400

    def test_get_audio_encoded_dotdot_returns_400(self, client: TestClient) -> None:
        r = client.get("/api/v1/references/%2e%2e/audio")
        assert r.status_code == 400

    def test_voice_query_traversal_returns_400(self, client: TestClient) -> None:
        # voice is validated before any Kokoro call, so no model is needed
        r = client.get("/api/v1/references/anything/audio?voice=a/../../../../tmp/x")
        assert r.status_code == 400

    def test_history_traversal_blocked(self, client: TestClient) -> None:
        # history containment returns 404 (not found), not 400
        for path in [
            "/api/v1/history/%2e%2e",
            "/api/v1/history/%2e%2e/audio",
        ]:
            assert client.get(path).status_code == 404
        assert client.delete("/api/v1/history/%2e%2e").status_code == 404

    def test_legit_reference_still_works(self, client: TestClient) -> None:
        """Happy path must not be broken by the guards."""
        self._seed_reference("hello-world")
        r = client.get("/api/v1/references/hello-world")
        assert r.status_code == 200
        assert r.json()["id"] == "hello-world"


# --------------------------------------------------------------------------- #
# Upload safety — pathological audio rejected with 400, not 500
# --------------------------------------------------------------------------- #
def _pcm_wav(sample_rate: int, n_samples: int) -> bytes:
    """A minimal 16-bit PCM mono WAV with an arbitrary (even hostile) sample rate."""
    import struct

    data = b"\x00\x00" * n_samples
    return (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(data),
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            len(data),
        )
        + data
    )


def _seed_reference_with_audio(slug: str = "ref-clip") -> str:
    """Plant a reference dir with a valid 16 kHz reference WAV + metadata."""
    import numpy as np
    import soundfile as sf

    mgr = deps.get_state().reference_manager
    audio_dir = mgr.config.base_dir / slug / "audio" / "kokoro-en-us-af_heart"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sr = 16000
    tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    sf.write(str(audio_dir / "ref.wav"), tone, sr)
    (mgr.config.base_dir / slug / "metadata.json").write_text(
        '{"text":"hi","default_speaker":"af_heart"}'
    )
    return slug


class TestUploadSafety:
    """Regression for the sample_rate=1 memory-amplification DoS: the upload is
    rejected at decode time with HTTP 400 (not a 4 GB allocation / 500)."""

    def test_evaluate_rejects_sample_rate_1_upload(self, client: TestClient) -> None:
        slug = _seed_reference_with_audio()
        # the original attack: 122 KB WAV declaring sample_rate=1, 64000 frames
        attack = _pcm_wav(1, 64_000)
        r = client.post(
            "/api/v1/evaluate",
            files={"audio": ("a.wav", attack, "audio/wav")},
            data={"reference_id": slug},
        )
        assert r.status_code == 400
        # 8 kHz (within range) is accepted at decode — covered by
        # test_audio.py::TestAudioPlausibility; not re-tested here because the
        # full /evaluate pipeline would load the Wav2Vec2 model.


# --------------------------------------------------------------------------- #
# Stored-XSS mitigation — server data is HTML-escaped before innerHTML
# --------------------------------------------------------------------------- #
class TestXssMitigation:
    """The demo SPA must HTML-escape server-derived strings before inserting
    them into innerHTML. Guards the stored-XSS fix (reference text -> word-diff)."""

    def test_demo_defines_esc_helper(self, client: TestClient) -> None:
        html = client.get("/").text
        assert "function esc(" in html

    def test_word_diff_sink_is_escaped(self, client: TestClient) -> None:
        html = client.get("/").text
        # the confirmed stored-XSS sink now escapes w.word
        assert "${esc(w.word)}" in html
        # feedback + phoneme sinks are escaped too
        assert "${esc(f)}" in html
        assert "${esc(op.phoneme)}" in html

    def test_no_unescaped_server_sinks_remain(self, client: TestClient) -> None:
        html = client.get("/").text
        # the raw (pre-fix) sinks must be gone
        assert "${w.word}" not in html
        assert ">• ${f}<" not in html


# --------------------------------------------------------------------------- #
# Run-2: CSRF guard, upload cap, text cap, voice-language validation
# --------------------------------------------------------------------------- #
class TestCsrfGuard:
    """State-changing requests with a disallowed Origin are rejected (403);
    no Origin (non-browser) and same-origin are allowed."""

    def test_cross_origin_post_blocked(self, client: TestClient) -> None:
        r = client.post("/api/v1/evaluate", headers={"Origin": "https://attacker.com"})
        assert r.status_code == 403

    def test_cross_origin_delete_blocked(self, client: TestClient) -> None:
        r = client.delete("/api/v1/references/x", headers={"Origin": "https://attacker.com"})
        assert r.status_code == 403

    def test_no_origin_allowed_reaches_handler(self, client: TestClient) -> None:
        # no Origin header -> not a browser CSRF -> reaches FastAPI validation (422)
        r = client.post("/api/v1/evaluate")
        assert r.status_code == 422

    def test_configured_origin_allowed(self, client: TestClient) -> None:
        # a configured dev origin reaches the handler (422, not 403)
        r = client.post("/api/v1/evaluate", headers={"Origin": "http://localhost:5173"})
        assert r.status_code == 422


class TestUploadCap:
    def test_oversized_upload_rejected_before_read(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.api import app as appmod

        monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 16)
        r = client.post(
            "/api/v1/evaluate/quick",
            files={"audio": ("a.wav", b"x" * 256, "audio/wav")},
            data={"text": "hi"},
        )
        assert r.status_code == 413


class TestTextCap:
    def test_create_reference_rejects_oversized_text(self, client: TestClient) -> None:
        r = client.post("/api/v1/references", json={"text": "x" * 501})
        assert r.status_code == 422

    def test_evaluate_quick_rejects_oversized_text(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/evaluate/quick",
            files={"audio": ("a.wav", b"x", "audio/wav")},
            data={"text": "x" * 501},
        )
        assert r.status_code == 422


class TestVoiceLanguageValidation:
    def test_unknown_voice_lang_returns_400(self, client: TestClient) -> None:
        # 'x' is not a Kokoro language code; must be a clean 400, not a 500
        r = client.get("/api/v1/references/anything/audio?voice=xs_foo")
        assert r.status_code == 400


class TestColdStartLock:
    def test_phoneme_extractor_loads_once_under_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading
        import time
        from unittest.mock import MagicMock

        from ai_speech_shadowing.api import deps

        state = deps.EngineState()
        calls = 0
        lock = threading.Lock()
        mock_extractor = MagicMock(name="extractor")

        def counting_get(*args, **kwargs):
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.05)  # simulate a slow model load
            return mock_extractor

        monkeypatch.setattr(deps, "get_extractor", counting_get)
        threads = [threading.Thread(target=state.phoneme_extractor) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert calls == 1  # double-checked locking -> exactly one load
        assert state._extractor is mock_extractor


# --------------------------------------------------------------------------- #
# Full flow (opt-in slow: Kokoro + Wav2Vec2)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestEvaluateFlow:
    def test_quick_evaluate_full_flow(self, client: TestClient, kokoro_ref_wav: Path) -> None:
        # use the Kokoro clip bytes as the "user" audio upload
        user_bytes = kokoro_ref_wav.read_bytes()

        r = client.post(
            "/api/v1/evaluate/quick",
            files={"audio": ("user.wav", user_bytes, "audio/wav")},
            data={"text": "Hello world, this is a Kokoro TTS test.", "language": "en"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["reference_id"] == "hello-world-this-is-a-kokoro-tts-test"
        assert "scores" in body and "composite" in body["scores"]
        assert isinstance(body["phoneme_diff"], list)
        assert isinstance(body["feedback"], list)

        # quick-eval knows the reference text -> word-level diff is attached
        assert isinstance(body.get("words"), list) and body["words"]

        # the reference was generated on the fly and is now listed
        refs = client.get("/api/v1/references").json()
        assert any(ref["id"] == body["reference_id"] for ref in refs)

        # history recorded the evaluation
        history = client.get("/api/v1/history").json()
        assert history["total"] >= 1
        assert any(item["id"] == body["id"] for item in history["items"])

        # stats reflect it
        stats = client.get("/api/v1/history/stats").json()
        assert stats["total_evaluations"] >= 1

        # health now shows models loaded
        health = client.get("/api/v1/health").json()
        assert health["models"]["wav2vec2"]["loaded"] is True
        assert health["models"]["tts"]["loaded"] is True
