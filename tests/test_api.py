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
