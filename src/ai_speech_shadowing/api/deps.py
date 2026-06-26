"""Shared engine state for the API (lazy singletons + load timing).

Centralises construction of the phoneme extractor and reference manager so that
``/health`` can report real load times and the model is loaded at most once per
process.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai_speech_shadowing.core.leaderboard import (
    LeaderboardStore,
    default_db_path,
    default_dedup_dir,
)
from ai_speech_shadowing.core.phoneme import PhonemeExtractor, get_extractor
from ai_speech_shadowing.tts.generator import ReferenceConfig, ReferenceManager

logger = logging.getLogger(__name__)

_EXTRACTOR_RETRY_COOLDOWN: float = 30.0
"""Seconds to wait before retrying a failed extractor load."""


def _default_leaderboard() -> LeaderboardStore:
    return LeaderboardStore(default_db_path(), default_dedup_dir())


@dataclass
class EngineState:
    """Mutable process-wide state. Tests may reassign the manager/history_dir."""

    reference_manager: ReferenceManager = field(
        default_factory=lambda: ReferenceManager(ReferenceConfig())
    )
    history_dir: Path = field(default_factory=lambda: Path("data/history"))
    leaderboard: LeaderboardStore = field(default_factory=_default_leaderboard)
    _extractor: PhonemeExtractor | None = None
    extractor_load_time_ms: int | None = None
    tts_available: bool = False
    tts_load_time_ms: int | None = None
    _extractor_lock: threading.Lock = field(default_factory=threading.Lock)
    _extractor_error_ts: float | None = None
    _extractor_error: Exception | None = None

    def phoneme_extractor(self) -> PhonemeExtractor:
        """Lazily load the Wav2Vec2 phoneme model (once), recording load time.

        Double-checked locking: concurrent first-requests serialise on the lock
        so the ~350 MB model is loaded exactly once per process.

        On failure the exception is cached and re-raised for a cooldown period
        (``_EXTRACTOR_RETRY_COOLDOWN``) to avoid retry storms.
        """
        if self._extractor is not None:
            return self._extractor
        # Fast-path: still in cooldown after a prior failure.
        err_ts = self._extractor_error_ts
        if err_ts is not None and time.monotonic() - err_ts < _EXTRACTOR_RETRY_COOLDOWN:
            raise self._extractor_error or RuntimeError("extractor load failed (cooldown)")
        with self._extractor_lock:
            if self._extractor is not None:
                pass
            elif (
                self._extractor_error_ts is not None
                and time.monotonic() - self._extractor_error_ts < _EXTRACTOR_RETRY_COOLDOWN
            ):
                raise self._extractor_error or RuntimeError("extractor load failed (cooldown)")
            else:
                t0 = time.perf_counter()
                try:
                    logger.info("[load] deps: requesting phoneme_extractor (first load)...")
                    self._extractor = get_extractor()
                except Exception as exc:
                    self._extractor_error_ts = time.monotonic()
                    self._extractor_error = exc
                    raise
                self.extractor_load_time_ms = int((time.perf_counter() - t0) * 1000)
                logger.info(
                    "[load] deps: phoneme_extractor ready in %sms",
                    self.extractor_load_time_ms,
                )
        return self._extractor

    def mark_tts_loaded(self, *, load_time_ms: int) -> None:
        if not self.tts_available:
            self.tts_load_time_ms = load_time_ms
        self.tts_available = True


_state: EngineState | None = None


def get_state() -> EngineState:
    global _state
    if _state is None:
        _state = EngineState()
    return _state


def reset_state(state: EngineState | None = None) -> EngineState:
    """Replace the singleton (used by tests to point at temp dirs)."""
    global _state
    _state = state or EngineState()
    return _state
