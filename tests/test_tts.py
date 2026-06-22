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
        assert manager.voice_profile() == "kokoro-en-us"

    def test_voice_profile_other_lang(self, manager: ReferenceManager) -> None:
        assert manager.voice_profile(lang="j") == "kokoro-ja"

    def test_paths(self, manager: ReferenceManager) -> None:
        assert manager.slug_path("hi") == manager.config.base_dir / "hi"
        assert manager.metadata_path("hi") == manager.config.base_dir / "hi" / "metadata.json"
        assert manager.audio_dir("hi", "kokoro-en-us") == (
            manager.config.base_dir / "hi" / "audio" / "kokoro-en-us"
        )
        assert manager.audio_file("hi", "kokoro-en-us").name == "ref.wav"

    def test_exists_false_then_true(self, manager: ReferenceManager) -> None:
        assert manager.exists("hi") is False
        f = manager.audio_file("hi", manager.voice_profile())
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x")
        assert manager.exists("hi") is True


class TestMetadata:
    def test_write_and_read(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart")
        meta = manager.read_metadata("hi")
        assert meta["text"] == "Hello"
        assert meta["language"] == "en-us"
        assert meta["default_speaker"] == "af_heart"
        assert "kokoro-en-us" in meta["audio"]

    def test_merge_multiple_profiles(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart")
        # second profile for the same slug merges into the existing audio dict
        manager.config = ReferenceConfig(
            base_dir=manager.config.base_dir, engine="kokoro", default_lang="j"
        )
        manager.write_metadata("hi", "Hello", "j", "jf_alpha")
        meta = manager.read_metadata("hi")
        assert "kokoro-en-us" in meta["audio"]
        assert "kokoro-ja" in meta["audio"]
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
        assert "kokoro-en-us" in meta["audio"]

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
