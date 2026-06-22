"""ai-speech-shadowing: local-first speech evaluation engine."""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    """Console-script entry point — delegates to the Typer app."""
    from ai_speech_shadowing.cli import app

    app()
