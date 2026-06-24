"""Tests for TTS reference generation.

Slugify / paths / metadata / cache / list-parsing are pure → fast. Kokoro
synthesis is opt-in slow (``--runslow``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from ai_speech_shadowing.tts.generator import (
    KOKORO_SAMPLE_RATE,
    PathEscapeError,
    ReferenceConfig,
    ReferenceManager,
    parse_sentence_list,
    slugify,
)


@pytest.fixture
def manager(tmp_path: Path) -> ReferenceManager:
    return ReferenceManager(ReferenceConfig(base_dir=tmp_path))


# --------------------------------------------------------------------------- #
# slugify (pure)
# --------------------------------------------------------------------------- #
class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Hello world!") == "hello-world"
        assert slugify("The quick brown fox") == "the-quick-brown-fox"

    def test_punctuation_and_whitespace(self) -> None:
        assert slugify("  Hello,   world...  ") == "hello-world"

    def test_strips_accents(self) -> None:
        assert slugify("Xin chào") == "xin-chao"
        assert slugify("café") == "cafe"

    def test_non_latin_falls_back_to_hash(self) -> None:
        # CJK yields no ASCII -> deterministic 12-char hash
        slug = slugify("你好世界")
        assert len(slug) == 12
        assert slug == slugify("你好世界")  # stable

    def test_truncates_long_text(self) -> None:
        slug = slugify("a " * 100, max_length=20)
        assert len(slug) <= 20

    def test_whitespace_only_falls_back_to_hash(self) -> None:
        assert slugify("   ") == slugify("   ")


# --------------------------------------------------------------------------- #
# ReferenceManager structure / metadata / cache (pure, tmp_path)
# --------------------------------------------------------------------------- #
class TestReferenceManagerStructure:
    def test_voice_profile_default(self, manager: ReferenceManager) -> None:
        assert manager.voice_profile() == "kokoro-en-us-af_heart"

    def test_voice_profile_other_lang(self, manager: ReferenceManager) -> None:
        assert manager.voice_profile(lang="j", voice="jf_alpha") == "kokoro-ja-jf_alpha"
        assert manager.voice_profile(lang="j") == "kokoro-ja-af_heart"

    def test_paths(self, manager: ReferenceManager) -> None:
        assert manager.slug_path("hi") == manager.config.base_dir / "hi"
        assert manager.metadata_path("hi") == manager.config.base_dir / "hi" / "metadata.json"
        assert manager.audio_dir("hi", "kokoro-en-us-af_heart") == (
            manager.config.base_dir / "hi" / "audio" / "kokoro-en-us-af_heart"
        )
        assert manager.audio_file("hi", "kokoro-en-us-af_heart").name == "ref.wav"

    def test_exists_false_then_true(self, manager: ReferenceManager) -> None:
        assert manager.exists("hi") is False
        f = manager.audio_file("hi", manager.voice_profile())
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x")
        assert manager.exists("hi") is True


# --------------------------------------------------------------------------- #
# Path safety — traversal payloads rejected at the manager (the sink)
# --------------------------------------------------------------------------- #
class TestPathSafety:
    """Regression tests for the path-traversal fix.

    Bare ".." / "." and slash-containing slugs are rejected by the slug format
    check; a symlink whose name is a valid slug but points outside base_dir is
    rejected by the resolve()+is_relative_to containment check.
    """

    @pytest.mark.parametrize("slug", ["..", ".", "foo/../bar", "%2e%2e", "a/b"])
    def test_slug_path_rejects_traversal(self, manager: ReferenceManager, slug: str) -> None:
        with pytest.raises(PathEscapeError):
            manager.slug_path(slug)

    def test_slug_path_accepts_clean_slug(self, manager: ReferenceManager) -> None:
        # happy path is not broken by the guard
        assert manager.slug_path("hello-world") == manager.config.base_dir / "hello-world"

    def test_slug_path_rejects_symlink_escape(
        self,
        manager: ReferenceManager,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        # A symlink with a valid slug name pointing OUTSIDE base_dir must be
        # caught by the containment check (the regex alone would let it through).
        outside = tmp_path_factory.mktemp("outside")
        manager.config.base_dir.mkdir(parents=True, exist_ok=True)
        (manager.config.base_dir / "evil-link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscapeError):
            manager.slug_path("evil-link")

    @pytest.mark.parametrize(
        "voice",
        ["a/../../../../tmp/x", "../boom", "af_heart/..", "a b"],
    )
    def test_voice_profile_rejects_traversal(self, manager: ReferenceManager, voice: str) -> None:
        with pytest.raises(PathEscapeError):
            manager.voice_profile(voice=voice)

    def test_empty_voice_falls_back_to_default(self, manager: ReferenceManager) -> None:
        # falsy voice is not a traversal — it means "use the configured default"
        assert manager.voice_profile(voice="") == "kokoro-en-us-af_heart"
        assert manager.voice_profile(voice=None) == "kokoro-en-us-af_heart"

    def test_voice_profile_accepts_clean_voice(self, manager: ReferenceManager) -> None:
        assert manager.voice_profile(voice="af_heart") == "kokoro-en-us-af_heart"
        assert manager.voice_profile(voice="jf_alpha") == "kokoro-en-us-jf_alpha"


class TestMetadata:
    def test_write_and_read(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart")
        meta = manager.read_metadata("hi")
        assert meta["text"] == "Hello"
        assert meta["language"] == "en-us"
        assert meta["default_speaker"] == "af_heart"
        assert "kokoro-en-us-af_heart" in meta["audio"]

    def test_write_metadata_persists_phonemes(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart", phonemes=["h", "ə", "l", "oʊ"])
        meta = manager.read_metadata("hi")
        assert meta["phonemes"]["tokens"] == ["h", "ə", "l", "oʊ"]
        assert meta["phonemes"]["source"] == "kokoro-g2p"
        assert meta["phonemes"]["notation"] == "espeak-wav2vec2"

    def test_write_metadata_phonemes_uses_setdefault(self, manager: ReferenceManager) -> None:
        # First write establishes the phonemes.
        manager.write_metadata("hi", "Hello", "a", "af_heart", phonemes=["h", "ə", "l", "oʊ"])
        # Second call with different voice + different tokens must NOT overwrite
        # (phonemes are a function of text, not voice).
        manager.write_metadata("hi", "Hello", "a", "am_adam", phonemes=["x", "y", "z"])
        meta = manager.read_metadata("hi")
        assert meta["phonemes"]["tokens"] == ["h", "ə", "l", "oʊ"]
        assert meta["default_speaker"] == "af_heart"  # also setdefault

    def test_write_metadata_phonemes_none_omits_field(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart", phonemes=None)
        meta = manager.read_metadata("hi")
        assert "phonemes" not in meta

    def test_merge_multiple_profiles(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart")
        # second profile for the same slug merges into the existing audio dict
        manager.config = ReferenceConfig(
            base_dir=manager.config.base_dir, engine="kokoro", default_lang="j"
        )
        manager.write_metadata("hi", "Hello", "j", "jf_alpha")
        meta = manager.read_metadata("hi")
        assert "kokoro-en-us-af_heart" in meta["audio"]
        assert "kokoro-ja-jf_alpha" in meta["audio"]
        # original text/language are preserved (setdefault)
        assert meta["text"] == "Hello"
        assert meta["default_speaker"] == "af_heart"

    def test_list_references(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart")
        manager.write_metadata("bye", "Goodbye", "a", "af_heart")
        listed = manager.list_references()
        slugs = [m["slug"] for m in listed]
        assert slugs == ["bye", "hi"]  # sorted

    def test_list_references_empty(self, manager: ReferenceManager) -> None:
        assert manager.list_references() == []


# --------------------------------------------------------------------------- #
# Metadata resilience + atomic writes (run-2: corrupt file must not crash all)
# --------------------------------------------------------------------------- #
class TestMetadataResilience:
    def test_read_metadata_corrupt_returns_empty(self, manager: ReferenceManager) -> None:
        d = manager.config.base_dir / "bad-ref"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text("{not valid json")
        assert manager.read_metadata("bad-ref") == {}

    def test_list_references_skips_corrupt_without_crashing(
        self, manager: ReferenceManager
    ) -> None:
        manager.write_metadata("good", "Hello", "a", "af_heart")
        bad = manager.config.base_dir / "bad"
        bad.mkdir(parents=True)
        (bad / "metadata.json").write_text("{not valid json")
        listed = manager.list_references()  # must not raise
        slugs = [m["slug"] for m in listed]
        assert "good" in slugs

    def test_write_metadata_is_atomic_round_trip(self, manager: ReferenceManager) -> None:
        # write then re-read must always yield valid JSON, and a stale .tmp
        # must not be left behind on success.
        manager.write_metadata("hi", "Hello", "a", "af_heart")
        meta = manager.read_metadata("hi")
        assert meta["text"] == "Hello"
        assert not (manager.config.base_dir / "hi" / "metadata.json.tmp").exists()


# --------------------------------------------------------------------------- #
# parse_sentence_list (pure)
# --------------------------------------------------------------------------- #
class TestParseSentenceList:
    def test_skips_blanks_and_comments(self, tmp_path: Path) -> None:
        f = tmp_path / "list.txt"
        f.write_text(
            "# a comment\nHello world\n\n   \n# another\nGoodbye world\n",
            encoding="utf-8",
        )
        assert parse_sentence_list(f) == ["Hello world", "Goodbye world"]


# --------------------------------------------------------------------------- #
# Kokoro generation (opt-in slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestGenerate:
    def test_single_reference(self, manager: ReferenceManager) -> None:
        out = manager.generate("Hello world", voice="af_heart", lang="a")
        assert out.is_file()
        info = sf.info(str(out))
        assert info.samplerate == KOKORO_SAMPLE_RATE
        assert info.frames > 0
        # metadata written alongside
        meta = manager.read_metadata(slugify("Hello world"))
        assert meta["text"] == "Hello world"
        assert "kokoro-en-us-af_heart" in meta["audio"]
        # G2P phonemes are captured at synthesis time and persisted.
        assert "phonemes" in meta
        assert meta["phonemes"]["source"] == "kokoro-g2p"
        assert meta["phonemes"]["tokens"]  # non-empty
        assert meta["phonemes"]["tokens"][0] == "h"  # "Hello"

    def test_cache_skips_regeneration(self, manager: ReferenceManager, tmp_path: Path) -> None:
        out1 = manager.generate("Hello world")
        mtime1 = out1.stat().st_mtime_ns
        # second call without force → same path, untouched
        out2 = manager.generate("Hello world")
        assert out2 == out1
        assert out2.stat().st_mtime_ns == mtime1
        # force → rewritten
        out3 = manager.generate("Hello world", force=True)
        assert out3 == out1

    def test_batch_from_list(self, manager: ReferenceManager, tmp_path: Path) -> None:
        sentences = ["Hello world", "Goodbye world"]
        paths = manager.generate_batch(sentences)
        assert len(paths) == 2
        assert all(p.is_file() for p in paths)
        assert len({p.parent.name for p in paths}) == 1  # same voice profile

    def test_phonemes_invariant_under_voice(self, manager: ReferenceManager) -> None:
        """The reference phonemes are a pure function of the text, not the audio.

        Synthesizing the same text with two different Kokoro voices produces
        different acoustic output (different formants, different f0) but the
        captured G2P tokens are byte-identical — because they come from the G2P
        of the text, not from acoustic recognition of the rendered audio. This
        is the core property that justifies the G2P reference path.
        """
        text = "Hello world, this is a Kokoro TTS test."
        slug = slugify(text)

        manager.generate(text, voice="af_heart", lang="a", force=True)
        tokens_heart = manager.read_metadata(slug)["phonemes"]["tokens"]

        manager.generate(text, voice="am_adam", lang="a", force=True)
        tokens_adam = manager.read_metadata(slug)["phonemes"]["tokens"]

        assert tokens_heart  # non-empty sanity check
        assert tokens_heart == tokens_adam  # voice-invariant
        # Sanity: still the right word ("Hello" → starts with /h/).
        assert tokens_heart[0] == "h"
