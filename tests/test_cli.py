"""Tests for the Typer CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ai_speech_shadowing.cli import app
from ai_speech_shadowing.core.history import save_report

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


# --------------------------------------------------------------------------- #
# Global flags
# --------------------------------------------------------------------------- #
def test_global_verbose_flag_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "version"])
    assert result.exit_code == 0


def test_global_quiet_flag_accepted() -> None:
    result = runner.invoke(app, ["--quiet", "version"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# record (mocked sounddevice — no real microphone)
# --------------------------------------------------------------------------- #
def test_record_writes_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sd = pytest.importorskip("sounddevice")

    def fake_rec(n: int, **kwargs: object) -> np.ndarray:
        return np.zeros((n, 1), dtype="float32")

    monkeypatch.setattr(sd, "rec", fake_rec)
    monkeypatch.setattr(sd, "wait", lambda: None)

    out = tmp_path / "rec.wav"
    result = runner.invoke(app, ["record", str(out), "--duration", "0.1", "--sample-rate", "8000"])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert "wrote" in result.stdout


# --------------------------------------------------------------------------- #
# report (list + view)
# --------------------------------------------------------------------------- #
def _seed_report(history_dir: Path) -> str:
    from ai_speech_shadowing.core.feedback import build_report
    from ai_speech_shadowing.core.fluency import DtwResult, FluencyDiff, PauseInfo
    from ai_speech_shadowing.core.phoneme import diff_phonemes
    from ai_speech_shadowing.core.prosody import PitchStats, ProsodyDiff

    pitch = PitchStats(
        f0_contour=np.zeros(1, dtype=np.float64),
        times=np.zeros(1, dtype=np.float64),
        mean_hz=200.0,
        median_hz=200.0,
        min_hz=100.0,
        max_hz=300.0,
        range_hz=200.0,
        std_hz=20.0,
        voiced_ratio=1.0,
        pitch_floor=75.0,
        pitch_ceiling=500.0,
    )
    prosody = ProsodyDiff(
        reference=pitch,
        hypothesis=pitch,
        pitch_range_ratio=1.0,
        monotone=False,
        monotone_threshold=0.5,
        score=1.0,
    )
    fluency = FluencyDiff(
        dtw=DtwResult(0.0, 10, 0.0),
        score=1.0,
        reference_pauses=PauseInfo(0, 0.0, ()),
        hypothesis_pauses=PauseInfo(0, 0.0, ()),
        syllable_rate_reference=2.0,
        syllable_rate_hypothesis=2.0,
        syllable_rate_ratio=1.0,
    )
    path = save_report(
        build_report(diff_phonemes(["a", "b"], ["a", "b"]), prosody, fluency),
        history_dir=history_dir,
    )
    return path.stem


def test_report_lists_saved_reports(tmp_path: Path) -> None:
    history = tmp_path / "history"
    rid = _seed_report(history)
    result = runner.invoke(app, ["report", "--history-dir", str(history)])
    assert result.exit_code == 0, result.stdout
    assert rid in result.stdout
    assert "100/100" in result.stdout


def test_report_view_by_id(tmp_path: Path) -> None:
    history = tmp_path / "history"
    rid = _seed_report(history)
    result = runner.invoke(app, ["report", rid, "--history-dir", str(history)])
    assert result.exit_code == 0, result.stdout
    assert "Composite: 100/100" in result.stdout


def test_report_view_json(tmp_path: Path) -> None:
    history = tmp_path / "history"
    rid = _seed_report(history)
    import json

    result = runner.invoke(app, ["report", rid, "--history-dir", str(history), "--format", "json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["id"] == rid


def test_report_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--history-dir", str(tmp_path / "history")])
    assert result.exit_code == 0
    assert "no saved reports" in result.stdout


def test_report_missing_id_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "eval_nope", "--history-dir", str(tmp_path / "history")])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# batch (opt-in slow: loads the phoneme model)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_batch_evaluates_directory(
    tmp_path: Path, kokoro_ref_wav: Path, mono_44100_wav: Path
) -> None:
    # two "recordings" (reuse fixtures) in a dir, evaluate against the Kokoro ref
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "r1.wav").write_bytes(mono_44100_wav.read_bytes())
    (recordings / "r2.wav").write_bytes(mono_44100_wav.read_bytes())
    history = tmp_path / "history"

    result = runner.invoke(
        app,
        [
            "batch",
            str(kokoro_ref_wav),
            str(recordings),
            "--history-dir",
            str(history),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Evaluated 2 recording(s)" in result.stdout
    assert (history / "eval_").parent == history
    # two reports saved
    from ai_speech_shadowing.core.history import list_reports

    assert len(list_reports(history)) == 2
