"""Tests for the Typer CLI."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ai_speech_shadowing.cli import app

runner = CliRunner()


def test_preprocess_command_writes_16k_mono(tmp_path: Path, stereo_48000_wav: Path) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["preprocess", str(stereo_48000_wav), "-o", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert "16000 Hz" in result.stdout
    assert "1ch" in result.stdout


def test_preprocess_default_output_suffix(tmp_path: Path, mono_44100_wav: Path) -> None:
    result = runner.invoke(app, ["preprocess", str(mono_44100_wav)])
    assert result.exit_code == 0, result.stdout
    expected = mono_44100_wav.with_suffix(".preprocessed.wav")
    assert expected.is_file()


def test_preprocess_disable_normalize(tmp_path: Path, quiet_wav: Path) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["preprocess", str(quiet_wav), "-o", str(out), "--normalize", "none"],
    )
    assert result.exit_code == 0, result.stdout


def test_preprocess_missing_file_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["preprocess", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0
