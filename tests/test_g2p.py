"""Tests for the G2P / normalization helpers in :mod:`ai_speech_shadowing.core.g2p`.

The pure-string helpers (``norm_misaki``, ``misaki_to_espeak_tokens`` with a
mocked vocab) are fast. End-to-end ``g2p_words`` is exercised by
``test_wordalign.py::TestWordLevelDiff`` (slow).
"""

# IPA characters in the test fixtures are intentional.
# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import pytest

from ai_speech_shadowing.core import g2p as g2p_mod


class TestNormMisaki:
    """Pure string normalization (no HF / no misaki)."""

    def test_strips_stress_marks(self) -> None:
        assert g2p_mod.norm_misaki("ˈʌmps") == "ʌmps"
        assert g2p_mod.norm_misaki("ˌWn") == "aʊn"  # stress gone, W -> aʊ

    def test_strips_length_mark(self) -> None:
        assert g2p_mod.norm_misaki("ɑː") == "ɑ"

    def test_unfolds_affricates(self) -> None:
        assert g2p_mod.norm_misaki("ʤˈʌmpt") == "dʒʌmpt"
        assert g2p_mod.norm_misaki("ʧ") == "tʃ"

    def test_uppercase_diphthongs(self) -> None:
        assert g2p_mod.norm_misaki("lˈAzi") == "leɪzi"  # A -> eɪ
        assert g2p_mod.norm_misaki("ˈOvəɹ") == "oʊvəɹ"  # O -> oʊ
        assert g2p_mod.norm_misaki("nˈIt") == "naɪt"  # I -> aɪ

    def test_passthrough_plain(self) -> None:
        assert g2p_mod.norm_misaki("ðə") == "ðə"


class TestMisakiToEspeakTokens:
    """``misaki_to_espeak_tokens`` with a mocked espeak vocab (no HF download)."""

    @pytest.fixture(autouse=True)
    def _mock_vocab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Longest-first ordering matters for greedy tokenization (e.g. "oʊ"
        # must be matched before "o"). Mirrors the real _get_espeak_tokens sort.
        vocab = [
            "aʊ",
            "aɪ",
            "eɪ",
            "oʊ",
            "tʃ",
            "dʒ",
            "ɜ˞",
            "h",
            "l",
            "d",
            "w",
            "ə",
            "o",
            "a",
            "ɛ",
            "ɹ",
        ]
        # Also reset the module-level cache so a stale real load doesn't leak in.
        monkeypatch.setattr(g2p_mod, "_espeak_tokens", vocab)
        monkeypatch.setattr(g2p_mod, "_get_espeak_tokens", lambda model_id="x": vocab)

    def test_drops_stress_and_tokenizes(self) -> None:
        # "həˈloʊ" → norm → "həloʊ" → tokens (h ə l oʊ)
        assert g2p_mod.misaki_to_espeak_tokens("həˈloʊ") == ("h", "ə", "l", "oʊ")

    def test_handles_multi_char_tokens(self) -> None:
        # "oʊ" must be matched as one token, not (o, ʊ)
        assert g2p_mod.misaki_to_espeak_tokens("oʊ") == ("oʊ",)

    def test_drops_unrecognized_chars(self) -> None:
        # Spaces, punctuation, and unknown codepoints are silently dropped.
        assert g2p_mod.misaki_to_espeak_tokens("h ə l . oʊ") == ("h", "ə", "l", "oʊ")

    def test_unfolds_affricates_before_tokenizing(self) -> None:
        # "ʤ" normalizes to "dʒ" which then tokenizes as one token.
        assert g2p_mod.misaki_to_espeak_tokens("ʤ") == ("dʒ",)

    def test_empty_input(self) -> None:
        assert g2p_mod.misaki_to_espeak_tokens("") == ()

    def test_returns_tuple(self) -> None:
        result = g2p_mod.misaki_to_espeak_tokens("həloʊ")
        assert isinstance(result, tuple)
        assert all(isinstance(t, str) for t in result)

    def test_concats_multi_chunk_input(self) -> None:
        # Kokoro emits one _ps string per chunk; the generator joins them with
        # spaces before calling this function. Verify that's handled cleanly.
        joined = " ".join(["həˈloʊ", "wɜ˞ld"])
        assert g2p_mod.misaki_to_espeak_tokens(joined) == (
            "h",
            "ə",
            "l",
            "oʊ",
            "w",
            "ɜ˞",
            "l",
            "d",
        )
