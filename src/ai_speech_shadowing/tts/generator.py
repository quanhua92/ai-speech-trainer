"""Generate and manage gold-standard native reference audio with Kokoro.

Layout (under the configured base directory, default ``data/references``)::

    <slug>/
        metadata.json              # text, language, default_speaker, per-profile audio
        audio/
            kokoro-en-us/ref.wav   # one folder per voice profile (engine + language)

References are cached: re-generating the same text/voice is a no-op unless
``force=True``. Kokoro synthesis itself is opt-in slow (loads the model); the
structure / metadata / cache logic is pure and unit-tested.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

KOKORO_SAMPLE_RATE: int = 24000
"""Kokoro's output sample rate (its only supported synthesis rate)."""

KOKORO_LANGUAGES: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "zh",
}
"""Map of Kokoro single-letter language codes → ISO-ish codes (for profile names)."""

DEFAULT_BASE_DIR: Path = Path("data/references")
DEFAULT_REF_FILENAME: str = "ref.wav"
SLUG_MAX_LENGTH: int = 50

# Kokoro voice names look like "af_heart", "am_michael", "bf_emma", "zf_xiaobei":
# 1-2 letters, an underscore, then a word. This blocks "/", "\", "..", spaces,
# and null bytes — the characters that make a voice name a path-traversal vector.
_VOICE_RE: re.Pattern[str] = re.compile(r"^[A-Za-z]{1,2}_\w+$")

# Slugs produced by slugify() are [a-z0-9-]. Enforcing the same shape on read/delete
# rejects ".", "..", and any encoded traversal before path construction.
_SLUG_RE: re.Pattern[str] = re.compile(r"^[a-z0-9-]+$")


class PathEscapeError(ValueError):
    """Raised when a user-supplied path segment would leave its base directory."""


def ensure_within(base: Path, target: Path) -> Path:
    """Resolve ``target`` and confirm it stays within ``base``.

    Flattens ``..`` segments and follows symlinks to the real location, then
    rejects anything that escapes ``base``. Returns the resolved path so callers
    operate on the true on-disk location.
    """
    base_r = base.resolve()
    target_r = target.resolve()
    try:
        target_r.relative_to(base_r)
    except ValueError:
        raise PathEscapeError(f"path {target!r} escapes base {base!r}") from None
    return target_r


def slugify(text: str, *, max_length: int = SLUG_MAX_LENGTH) -> str:
    """Derive a short filesystem-safe slug from arbitrary text.

    Accents are stripped (NFKD + ASCII); non-Latin text that yields no ASCII
    falls back to a 12-char hash so every input maps to a stable slug.
    """
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    if not slug:
        slug = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]
    return slug


def parse_sentence_list(path: str | Path) -> list[str]:
    """Read a sentence list file: one sentence per line, ``#`` comments + blanks skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Where and how references are generated."""

    base_dir: Path = field(default_factory=lambda: Path(DEFAULT_BASE_DIR))
    engine: str = "kokoro"
    default_voice: str = "af_heart"
    default_lang: str = "a"  # Kokoro single-letter language code


class ReferenceManager:
    """Owns the reference directory layout, metadata, and caching."""

    def __init__(self, config: ReferenceConfig | None = None) -> None:
        self.config = config or ReferenceConfig()
        # Serializes the read->modify->write in write_metadata so concurrent
        # same-slug requests don't lose updates or corrupt metadata.json.
        self._meta_lock = threading.Lock()

    # ---- naming & paths -------------------------------------------------
    def voice_profile(
        self,
        *,
        lang: str | None = None,
        engine: str | None = None,
        voice: str | None = None,
    ) -> str:
        """Profile folder name: ``{engine}-{iso}-{voice}`` (e.g. kokoro-en-us-af_heart).

        Including the voice lets the same text exist with different voices
        without overwriting. Falls back to ``default_voice`` / ``default_lang``.
        """
        lang = lang or self.config.default_lang
        engine = engine or self.config.engine
        voice = voice or self.config.default_voice
        # The voice becomes a filesystem folder name, so reject anything that is
        # not a clean Kokoro voice id (blocks path traversal via ?voice= / speaker).
        if not _VOICE_RE.match(voice):
            raise PathEscapeError(f"invalid voice name: {voice!r}")
        iso = KOKORO_LANGUAGES.get(lang, lang)
        return f"{engine}-{iso}-{voice}"

    def _profile_for_slug(self, slug: str) -> str:
        """Return the voice profile for an existing reference (from its metadata).

        Falls back to the default profile if the reference has no metadata or
        the old-style ``kokoro-en-us`` folder exists (backward compat).
        """
        meta = self.read_metadata(slug)
        voice = (
            str(meta.get("default_speaker", self.config.default_voice))
            if meta
            else self.config.default_voice
        )
        profile = self.voice_profile(voice=voice)
        # backward compat: if the new-profile folder doesn't exist but the old one does
        new_path = self.audio_dir(slug, profile)
        if not new_path.is_dir():
            old_profile = self.voice_profile(lang=self.config.default_lang, voice=None)
            old_profile = old_profile.rsplit("-", 1)[0]  # strip voice suffix
            if self.audio_dir(slug, old_profile).is_dir():
                return old_profile
        return profile

    def slug_path(self, slug: str) -> Path:
        # Reject non-slug values (".", "..", encoded traversal) up front, then
        # confirm the resolved path stays inside base_dir (defeats symlinks too).
        if not _SLUG_RE.match(slug):
            raise PathEscapeError(f"invalid slug: {slug!r}")
        path = self.config.base_dir / slug
        ensure_within(self.config.base_dir, path)
        return path

    def metadata_path(self, slug: str) -> Path:
        return self.slug_path(slug) / "metadata.json"

    def audio_dir(self, slug: str, profile: str) -> Path:
        return self.slug_path(slug) / "audio" / profile

    def audio_file(self, slug: str, profile: str) -> Path:
        return self.audio_dir(slug, profile) / DEFAULT_REF_FILENAME

    def exists(self, slug: str, *, profile: str | None = None) -> bool:
        """Cache check: is the reference WAV already on disk?"""
        profile = profile or self.voice_profile()
        return self.audio_file(slug, profile).is_file()

    # ---- metadata -------------------------------------------------------
    def write_metadata(self, slug: str, text: str, lang: str, voice: str) -> Path:
        """Merge this voice profile into ``<slug>/metadata.json`` (create if absent)."""
        path = self.metadata_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lock the read-modify-write so concurrent same-slug writes don't lose
        # updates, and write via a temp + atomic os.replace so a crash or
        # concurrent reader never sees a half-written (corrupt) file.
        with self._meta_lock:
            data: dict[str, object] = {}
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = {}  # treat a corrupt file as empty rather than crashing
            data.setdefault("text", text)
            data.setdefault("language", KOKORO_LANGUAGES.get(lang, lang))
            data.setdefault("default_speaker", voice)
            data["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

            profile = self.voice_profile(lang=lang, voice=voice)
            audio_entry: dict[str, object] = {
                "file": f"audio/{profile}/{DEFAULT_REF_FILENAME}",
                "sample_rate": KOKORO_SAMPLE_RATE,
                "engine": self.config.engine,
            }
            profiles = data.setdefault("audio", {})
            if isinstance(profiles, dict):
                profiles[profile] = audio_entry

            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        return path

    def read_metadata(self, slug: str) -> dict[str, object]:
        path = self.metadata_path(slug)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt metadata.json must not crash listing/get for all refs.
            return {}

    def list_references(self) -> list[dict[str, object]]:
        """List every slug folder that has a metadata.json."""
        base = self.config.base_dir
        if not base.is_dir():
            return []
        results: list[dict[str, object]] = []
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "metadata.json").is_file():
                meta = self.read_metadata(entry.name)
                if not meta:  # corrupt/unreadable metadata -> skip, don't crash the list
                    continue
                meta["slug"] = entry.name
                results.append(meta)
        return results

    # ---- generation (Kokoro; opt-in slow) -------------------------------
    def generate(
        self,
        text: str,
        *,
        voice: str | None = None,
        lang: str | None = None,
        force: bool = False,
    ) -> Path:
        """Synthesize ``text`` with Kokoro and write it under the slug folder.

        Cached: returns the existing path if the WAV already exists and
        ``force`` is False.
        """
        import soundfile as sf
        from kokoro import KPipeline

        voice = voice or self.config.default_voice
        lang = lang or self.config.default_lang
        slug = slugify(text)
        profile = self.voice_profile(lang=lang, voice=voice)
        out = self.audio_file(slug, profile)
        if out.is_file() and not force:
            return out

        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            pipeline = KPipeline(lang_code=lang)
            chunks = [audio for _gs, _ps, audio in pipeline(text, voice=voice)]
            if not chunks:
                raise RuntimeError(f"kokoro produced no audio for: {text!r}")
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            sf.write(str(out), audio, KOKORO_SAMPLE_RATE)
        except Exception:
            # Don't leave an empty profile dir behind when synthesis fails
            # (e.g. an invalid voice name passes the format check but Kokoro rejects it).
            with suppress(OSError):
                out.parent.rmdir()  # only succeeds if empty
            raise
        self.write_metadata(slug, text, lang, voice)
        return out

    def generate_batch(
        self,
        sentences: Iterable[str] | Sequence[str],
        *,
        voice: str | None = None,
        lang: str | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Generate a reference for each sentence. Returns the written paths."""
        return [
            self.generate(s, voice=voice, lang=lang, force=force) for s in sentences if s.strip()
        ]
