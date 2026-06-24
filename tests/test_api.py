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
from ai_speech_shadowing.core.feedback import FeedbackReport
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


class TestScoringWeights:
    """All-zero scoring weights are valid per the schema but meaningless; they
    must be rejected with 400 before any model work (was an uncaught 500)."""

    def test_all_zero_weights_returns_400(self, client: TestClient) -> None:
        slug = _seed_reference_with_audio()
        audio = _pcm_wav(16_000, 16_000)  # 1s of valid 16 kHz audio
        r = client.post(
            "/api/v1/evaluate",
            files={"audio": ("a.wav", audio, "audio/wav")},
            data={
                "reference_id": slug,
                "weight_pronunciation": 0,
                "weight_intonation": 0,
                "weight_fluency": 0,
            },
        )
        assert r.status_code == 400
        # non-zero weights still pass the guard; the full pipeline is exercised
        # under --runslow by TestEvaluateFlow (not re-tested here — would load
        # the model in the fast suite).


# --------------------------------------------------------------------------- #
# Reference phoneme sourcing — G2P path vs acoustic fallback
# --------------------------------------------------------------------------- #
class TestReferencePhonemeSource:
    """Verify the API wiring around the cached G2P phoneme block.

    The full /evaluate flow needs the Wav2Vec2 model; here we exercise the
    pure helpers that decide whether the G2P path applies for a given slug.
    """

    def test_read_phonemes_returns_none_when_block_absent(self, client: TestClient) -> None:
        from ai_speech_shadowing.api.routes.evaluate import _read_reference_phonemes

        _seed_reference_with_audio("plain-ref")  # metadata has no phonemes block
        mgr = deps.get_state().reference_manager
        assert _read_reference_phonemes(mgr, "plain-ref") is None

    def test_read_phonemes_returns_none_when_slug_missing(self, client: TestClient) -> None:
        from ai_speech_shadowing.api.routes.evaluate import _read_reference_phonemes

        mgr = deps.get_state().reference_manager
        assert _read_reference_phonemes(mgr, "never-existed") is None

    def test_read_phonemes_returns_tokens_when_present(self, client: TestClient) -> None:
        from ai_speech_shadowing.api.routes.evaluate import _read_reference_phonemes

        mgr = deps.get_state().reference_manager
        meta_path = mgr.config.base_dir / "g2p-ref" / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            '{"text":"hi","phonemes":{"tokens":["h","ə","l","oʊ"],'
            '"source":"kokoro-g2p","notation":"espeak-wav2vec2"}}'
        )
        assert _read_reference_phonemes(mgr, "g2p-ref") == ["h", "ə", "l", "oʊ"]

    def test_read_phonemes_returns_none_for_empty_tokens(self, client: TestClient) -> None:
        from ai_speech_shadowing.api.routes.evaluate import _read_reference_phonemes

        mgr = deps.get_state().reference_manager
        meta_path = mgr.config.base_dir / "empty-ref" / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text('{"text":"hi","phonemes":{"tokens":[]}}')
        # Empty token list is treated as "no G2P available" → acoustic fallback.
        assert _read_reference_phonemes(mgr, "empty-ref") is None

    def test_read_phonemes_tolerates_non_dict_block(self, client: TestClient) -> None:
        from ai_speech_shadowing.api.routes.evaluate import _read_reference_phonemes

        mgr = deps.get_state().reference_manager
        meta_path = mgr.config.base_dir / "weird-ref" / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        # A corrupt phonemes block (a string instead of dict) must not crash.
        meta_path.write_text('{"text":"hi","phonemes":"not-a-dict"}')
        assert _read_reference_phonemes(mgr, "weird-ref") is None

    def test_evaluation_response_carries_source_field(self) -> None:
        # Build a synthetic report with the G2P source and verify the response
        # schema surfaces it. Pure schema-level; no FastAPI client needed.
        from ai_speech_shadowing.api.schemas import build_evaluation_response
        from ai_speech_shadowing.core.feedback import (
            DEFAULT_WEIGHTS,
            FeedbackReport,
            grade_for,
        )
        from ai_speech_shadowing.core.phoneme import diff_phonemes
        from ai_speech_shadowing.core.prosody import PitchStats

        def _stats() -> PitchStats:
            return PitchStats(
                f0_contour=__import__("numpy").zeros(1),
                times=__import__("numpy").zeros(1),
                mean_hz=200.0,
                median_hz=200.0,
                min_hz=100.0,
                max_hz=300.0,
                range_hz=200.0,
                std_hz=20.0,
                voiced_ratio=1.0,
                pitch_floor=75.0,
                pitch_ceiling=500.0,
            )

        report = FeedbackReport(
            composite_score=90,
            composite_grade=grade_for(90),
            pronunciation_score=90,
            intonation_score=90,
            fluency_score=90,
            weights=DEFAULT_WEIGHTS,
            phoneme_error_rate=0.1,
            pitch_range_ratio=1.0,
            monotone=False,
            dtw_normalized_distance=0.05,
            syllable_rate_reference=2.0,
            syllable_rate_hypothesis=2.0,
            pause_count_reference=0,
            pause_count_hypothesis=0,
            phoneme_diff=diff_phonemes(["a"], ["a"]),
            feedback=("ok",),
            reference_phoneme_source="kokoro-g2p",
        )
        resp = build_evaluation_response(
            report, reference_id="x", eval_id="eval_1", created_at="2026-01-01"
        )
        assert resp.reference_phoneme_source == "kokoro-g2p"

        # The acoustic-source report also round-trips.
        report2 = _seed_acoustic_report()
        resp2 = build_evaluation_response(
            report2, reference_id="y", eval_id="eval_2", created_at="2026-01-01"
        )
        assert resp2.reference_phoneme_source == "wav2vec2-acoustic"


def _seed_acoustic_report() -> FeedbackReport:
    """Reuse the construction above for the acoustic-source case."""
    from ai_speech_shadowing.core.feedback import FeedbackReport, grade_for
    from ai_speech_shadowing.core.phoneme import diff_phonemes

    return FeedbackReport(
        composite_score=50,
        composite_grade=grade_for(50),
        pronunciation_score=50,
        intonation_score=50,
        fluency_score=50,
        weights=(0.4, 0.3, 0.3),
        phoneme_error_rate=0.5,
        pitch_range_ratio=0.5,
        monotone=False,
        dtw_normalized_distance=0.2,
        syllable_rate_reference=2.0,
        syllable_rate_hypothesis=2.0,
        pause_count_reference=0,
        pause_count_hypothesis=0,
        phoneme_diff=diff_phonemes(["a"], ["b"]),
        feedback=("ok",),
        reference_phoneme_source="wav2vec2-acoustic",
    )


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
