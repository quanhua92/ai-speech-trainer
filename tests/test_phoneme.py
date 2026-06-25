"""Tests for phoneme diff, PER, the no-drop normalizers, and (opt-in) extraction.

The diff/PER logic and the pure-string normalizers (``_strip_tones_marks``,
``_arpabet_base``, ``ARPABET_TO_IPA``) are exercised without any model. The
extraction test is marked ``slow`` — it downloads a model and only runs with
``uv run pytest --runslow``.
"""

from __future__ import annotations

# IPA characters in this module are intentional.
# ruff: noqa: RUF001, RUF003
from pathlib import Path

import pytest

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.phoneme import (
    ARPABET_TO_IPA,
    DEFAULT_MODEL_KEY,
    MODELS,
    PhonemeDiff,
    _arpabet_base,
    _segments_to_espeak_units,
    _strip_tones_marks,
    diff_phonemes,
    get_phoneme_model,
    phoneme_error_rate,
)
from ai_speech_shadowing.core.preprocess import preprocess


# --------------------------------------------------------------------------- #
# Pure diff + PER tests (no model)
# --------------------------------------------------------------------------- #
class TestDiffPhonemes:
    def test_identical_sequences_are_perfect(self) -> None:
        seq = ["h", "ə", "l", "oʊ"]
        d = diff_phonemes(seq, seq)
        assert d.matches == 4
        assert (d.substitutions, d.deletions, d.insertions) == (0, 0, 0)
        assert d.phoneme_error_rate == 0.0
        assert d.is_perfect
        assert d.accuracy == 1.0
        assert [op.tag for op in d.operations] == ["match"] * 4

    def test_single_substitution(self) -> None:
        # known-good "l" vs known-bad "ɹ" (a classic accent substitution)
        d = diff_phonemes(["h", "ə", "l", "oʊ"], ["h", "ə", "ɹ", "oʊ"])
        assert d.matches == 3
        assert d.substitutions == 1
        assert d.deletions == 0
        assert d.insertions == 0
        assert d.phoneme_error_rate == pytest.approx(0.25)
        sub_ops = [op for op in d.operations if op.tag == "sub"]
        assert len(sub_ops) == 1
        assert sub_ops[0].ref == "l" and sub_ops[0].hyp == "ɹ"

    def test_deletion_omitted_phoneme(self) -> None:
        # user dropped the final phoneme
        d = diff_phonemes(["h", "ə", "l", "oʊ"], ["h", "ə", "l"])
        assert d.deletions == 1
        assert d.substitutions == 0
        assert d.insertions == 0
        assert d.matches == 3
        assert d.phoneme_error_rate == pytest.approx(0.25)

    def test_insertion_extra_phoneme(self) -> None:
        # user inserted an extra vowel
        d = diff_phonemes(["h", "ə", "l", "oʊ"], ["h", "ə", "ə", "l", "oʊ"])
        assert d.insertions == 1
        assert d.deletions == 0
        assert d.substitutions == 0
        assert d.phoneme_error_rate == pytest.approx(0.25)

    def test_mixed_errors(self) -> None:
        # ref: a b c d e ; hyp: a x c e   (b->x sub, d deleted)
        d = diff_phonemes(["a", "b", "c", "d", "e"], ["a", "x", "c", "e"])
        assert d.matches == 3
        assert d.substitutions == 1
        assert d.deletions == 1
        assert d.phoneme_error_rate == pytest.approx(0.4)

    def test_empty_reference_with_insertions(self) -> None:
        d = diff_phonemes([], ["a", "b"])
        assert d.insertions == 2
        assert d.phoneme_error_rate == 1.0  # empty ref + errors -> 1.0

    def test_empty_hypothesis_all_deletions(self) -> None:
        d = diff_phonemes(["a", "b", "c"], [])
        assert d.deletions == 3
        assert d.phoneme_error_rate == 1.0

    def test_both_empty(self) -> None:
        d = diff_phonemes([], [])
        assert d.phoneme_error_rate == 0.0
        assert d.operations == ()

    def test_returns_typed_diff(self) -> None:
        d = diff_phonemes(["a"], ["b"])
        assert isinstance(d, PhonemeDiff)
        assert isinstance(d.reference, tuple)
        assert isinstance(d.hypothesis, tuple)

    def test_per_can_exceed_one(self) -> None:
        # 1 ref phoneme, 3 inserted extras -> errors=3, per=3.0
        d = diff_phonemes(["a"], ["a", "b", "c", "d"])
        assert d.matches == 1
        assert d.insertions == 3
        assert d.phoneme_error_rate == 3.0


class TestPhonemeErrorRateHelper:
    def test_helper_matches_diff_property(self) -> None:
        ref = ["a", "b", "c"]
        hyp = ["a", "x", "c"]
        assert phoneme_error_rate(ref, hyp) == diff_phonemes(ref, hyp).phoneme_error_rate


# --------------------------------------------------------------------------- #
# Edge cases (no model)
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_empty_audio_decodes_to_no_phonemes(self) -> None:
        # silence should collapse to zero phonemes once it hits the model; here
        # we just assert the diff path handles an empty hypothesis cleanly.
        d = diff_phonemes(["a", "b"], [])
        assert d.phoneme_error_rate == 1.0


# --------------------------------------------------------------------------- #
# espeak normalization (no model) — no-drop behavior
# --------------------------------------------------------------------------- #
class TestStripTonesMarks:
    def test_drops_tone_markers_but_keeps_everything_else(self) -> None:
        # Recognizer garbage from eval_cde21c1f: Mandarin tones + stray chars.
        # Tone tokens (digit-bearing) are dropped; non-English consonants and
        # alternate-notation vowels are KEPT (the old inventory filter deleted
        # them, silently censoring real pronunciation info).
        raw = ("ð", "ə", "ɕ", "ɑ5", "ai5", "ei5", "u5", "iɜ", "k", "ɔː", "uː")
        assert _strip_tones_marks(raw, "en-us") == ("ð", "ə", "ɕ", "iɜ", "k", "ɔ", "u")

    def test_strips_length_and_stress_marks(self) -> None:
        assert _strip_tones_marks(("ɜː", "ˈæ", "ˌt"), "en") == ("ɜ", "æ", "t")

    def test_keeps_alternate_notation_vowels_that_the_old_filter_dropped(self) -> None:
        # Regression guard: bare "e", barred "ᵻ", and r-coloured "ɔːɹ" used to
        # be silently deleted by the ENGLISH_PHONEMES inventory check. They must
        # now survive (length/stress stripped, but the token kept).
        assert _strip_tones_marks(("e", "eː", "ᵻ", "ɔːɹ"), "en") == ("e", "e", "ᵻ", "ɔɹ")

    def test_non_english_language_passes_through(self) -> None:
        raw = ("ɕ", "iɛ5", "a")
        # Spanish/French/etc. are not filtered — the model covers them natively.
        assert _strip_tones_marks(raw, "es") == raw
        assert _strip_tones_marks(raw, None) == raw

    def test_keeps_english_diphthongs(self) -> None:
        assert _strip_tones_marks(("eɪ", "oʊ", "aɪ", "θ"), "en-gb") == (
            "eɪ",
            "oʊ",
            "aɪ",
            "θ",
        )


# --------------------------------------------------------------------------- #
# ARPAbet → IPA mapping (no model)
# --------------------------------------------------------------------------- #
class TestArpabetMapping:
    def test_arpabet_base_strips_err_suffix(self) -> None:
        assert _arpabet_base("g_err") == "g"
        assert _arpabet_base("aa_err") == "aa"
        assert _arpabet_base("ih_err") == "ih"

    def test_arpabet_base_strips_star_suffix(self) -> None:
        assert _arpabet_base("b*") == "b"
        assert _arpabet_base("d*") == "d"
        assert _arpabet_base("g*") == "g"

    def test_arpabet_base_passes_plain_tokens_through(self) -> None:
        assert _arpabet_base("ih") == "ih"
        assert _arpabet_base("tʃ") == "tʃ"

    def test_table_covers_every_base_phoneme(self) -> None:
        # The slplab model's 89 phone tokens reduce (via _arpabet_base) to this
        # exact set of bases. Every one must be in ARPABET_TO_IPA — no drops.
        bases = {
            "aa",
            "ae",
            "ah",
            "ao",
            "aw",
            "ax",
            "ay",
            "b",
            "ch",
            "d",
            "dh",
            "eh",
            "er",
            "eu",
            "ey",
            "f",
            "g",
            "hh",
            "ih",
            "iy",
            "jh",
            "k",
            "l",
            "m",
            "n",
            "ng",
            "o",
            "ow",
            "oy",
            "p",
            "r",
            "s",
            "sh",
            "t",
            "th",
            "ts",
            "uh",
            "uw",
            "v",
            "w",
            "y",
            "z",
            "zh",
        }
        assert set(ARPABET_TO_IPA) >= bases
        # spot-check a few critical ones (esp. the Unicode ɡ for ARPAbet "g")
        assert ARPABET_TO_IPA["g"] == "ɡ"
        assert ARPABET_TO_IPA["ih"] == "ɪ"
        assert ARPABET_TO_IPA["er"] == "ɝ"
        assert ARPABET_TO_IPA["ng"] == "ŋ"

    def test_full_sequence_maps_with_no_drops(self) -> None:
        # "big bear" as decoded by slplab on clipped learner audio, incl. a
        # _err flag on the g. Every token maps to a valid espeak-IPA segment.
        raw = ("b", "ih", "g_err", "b", "eh", "r")
        mapped = tuple(ARPABET_TO_IPA[_arpabet_base(t)] for t in raw)
        assert mapped == ("b", "ɪ", "ɡ", "b", "ɛ", "ɹ")
        assert len(mapped) == len(raw)  # no token dropped


# --------------------------------------------------------------------------- #
# Segment -> espeak-unit collapse (no model; mocked espeak vocab)
# --------------------------------------------------------------------------- #
class TestSegmentCollapse:
    def test_rejoins_r_coloured_vowels_to_espeak_units(self, monkeypatch) -> None:
        # ɛ + ɹ must collapse to the single espeak unit ɛɹ that kokoro G2P
        # emits — this is the notation-mismatch fix that takes PER 0.4 -> 0.0.
        import ai_speech_shadowing.core.g2p as g2p_mod

        mock_vocab = sorted(
            ["b", "ɪ", "ɡ", "d", "ɛɹ", "ɛ", "ɹ", "eɪ", "oʊ", "tʃ", "t", "ʃ"],
            key=len,
            reverse=True,
        )
        monkeypatch.setattr(g2p_mod, "_get_espeak_tokens", lambda model_id="x": mock_vocab)

        assert _segments_to_espeak_units(("b", "ɪ", "ɡ", "b", "ɛ", "ɹ")) == (
            "b",
            "ɪ",
            "ɡ",
            "b",
            "ɛɹ",
        )

    def test_rejoins_affricates(self, monkeypatch) -> None:
        import ai_speech_shadowing.core.g2p as g2p_mod

        mock_vocab = sorted(["t", "ʃ", "tʃ", "d", "ʒ", "dʒ"], key=len, reverse=True)
        monkeypatch.setattr(g2p_mod, "_get_espeak_tokens", lambda model_id="x": mock_vocab)

        assert _segments_to_espeak_units(("t", "ʃ")) == ("tʃ",)
        assert _segments_to_espeak_units(("d", "ʒ")) == ("dʒ",)

    def test_already_maximal_tokens_pass_through(self, monkeypatch) -> None:
        import ai_speech_shadowing.core.g2p as g2p_mod

        mock_vocab = sorted(["eɪ", "oʊ", "aɪ", "h", "ə", "l"], key=len, reverse=True)
        monkeypatch.setattr(g2p_mod, "_get_espeak_tokens", lambda model_id="x": mock_vocab)

        assert _segments_to_espeak_units(("h", "ə", "l", "oʊ")) == ("h", "ə", "l", "oʊ")

    def test_empty_input(self, monkeypatch) -> None:
        import ai_speech_shadowing.core.g2p as g2p_mod

        monkeypatch.setattr(g2p_mod, "_get_espeak_tokens", lambda model_id="x": ["b"])
        assert _segments_to_espeak_units(()) == ()


# --------------------------------------------------------------------------- #
# Registry (no model load)
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_default_is_slplab_l2(self) -> None:
        assert DEFAULT_MODEL_KEY == "slplab-l2"
        assert "slplab-l2" in MODELS
        assert "espeak" in MODELS

    def test_get_phoneme_model_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown phoneme model"):
            get_phoneme_model(key="does-not-exist", reload=True)


# --------------------------------------------------------------------------- #
# Opt-in model test (downloads; run with --runslow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestExtractionWithModel:
    def test_extract_real_speech(self, kokoro_ref_wav: Path) -> None:
        """End-to-end: Kokoro reference → preprocess → phonemes.

        Uses the default backend (slplab-l2). Asserts the pipeline runs and
        yields a sensible IPA sequence for "Hello world, this is a Kokoro TTS
        test."
        """
        extractor = get_phoneme_model(device="cpu", reload=True)
        sample = preprocess(AudioSample.from_wav(kokoro_ref_wav))
        result = extractor.extract(sample)
        assert len(result.phonemes) > 5
        assert "h" in result.phonemes  # 'Hello' opens with /h/
        assert isinstance(result.raw_text, str)
        assert len(result.raw_text.split()) == len(result.phonemes)

    def test_extract_requires_16k(self, mono_44100_wav: Path) -> None:
        extractor = get_phoneme_model(device="cpu", reload=True)
        sample = AudioSample.from_wav(mono_44100_wav)  # 44.1kHz, not canonical
        with pytest.raises(ValueError, match="16000"):
            extractor.extract(sample)
