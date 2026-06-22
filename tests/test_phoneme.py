"""Tests for phoneme diff, PER, and (opt-in) Wav2Vec2 extraction.

The diff/PER logic is pure and tested exhaustively without any model. The
extraction test is marked ``slow`` — it downloads ~1.2GB and only runs with
``uv run pytest --runslow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.phoneme import (
    PhonemeDiff,
    PhonemeExtractor,
    diff_phonemes,
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
# Opt-in model test (downloads ~1.2GB; run with --runslow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestExtractionWithModel:
    def test_extract_real_speech(self, kokoro_ref_wav: Path) -> None:
        """End-to-end: Kokoro reference → preprocess → phonemes.

        Asserts the pipeline runs and yields a sensible IPA sequence for
        "Hello world, this is a Kokoro TTS test."
        """
        extractor = PhonemeExtractor(device="cpu")
        sample = preprocess(AudioSample.from_wav(kokoro_ref_wav))
        result = extractor.extract(sample)
        assert len(result.phonemes) > 5
        assert "h" in result.phonemes  # 'Hello' opens with /h/
        assert isinstance(result.raw_text, str)
        assert len(result.raw_text.split()) == len(result.phonemes)

    def test_extract_requires_16k(self, mono_44100_wav: Path) -> None:
        extractor = PhonemeExtractor(device="cpu")
        sample = AudioSample.from_wav(mono_44100_wav)  # 44.1kHz, not canonical
        with pytest.raises(ValueError, match="16000"):
            extractor.extract(sample)
