"""Shared engine state for the API (lazy singletons + load timing).

Centralises construction of the phoneme extractor and reference manager so that
``/health`` can report real load times and the model is loaded at most once per
process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ai_speech_shadowing.core.phoneme import PhonemeExtractor, get_extractor
from ai_speech_shadowing.tts.generator import ReferenceConfig, ReferenceManager

if TYPE_CHECKING:
    pass


@dataclass
class EngineState:
    """Mutable process-wide state. Tests may reassign the manager/history_dir."""

    reference_manager: ReferenceManager = field(
        default_factory=lambda: ReferenceManager(ReferenceConfig())
    )
    history_dir: Path = field(default_factory=lambda: Path("data/history"))
    _extractor: PhonemeExtractor | None = None
    extractor_load_time_ms: int | None = None
    tts_available: bool = False
    tts_load_time_ms: int | None = None
    _extractor_lock: threading.Lock = field(default_factory=threading.Lock)

    def phoneme_extractor(self) -> PhonemeExtractor:
        """Lazily load the Wav2Vec2 phoneme model (once), recording load time.

        Double-checked locking: concurrent first-requests serialise on the lock
        so the ~350 MB model is loaded exactly once per process.
        """
        if self._extractor is None:
            with self._extractor_lock:
                if self._extractor is None:
                    t0 = time.perf_counter()
                    self._extractor = get_extractor()
                    self.extractor_load_time_ms = int((time.perf_counter() - t0) * 1000)
        return self._extractor

    def mark_tts_loaded(self, *, load_time_ms: int) -> None:
        self.tts_available = True
        self.tts_load_time_ms = load_time_ms


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
