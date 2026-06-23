"""Tests for evaluation history persistence (pure filesystem I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_speech_shadowing.core.feedback import FeedbackReport, build_report
from ai_speech_shadowing.core.fluency import DtwResult, FluencyDiff, PauseInfo
from ai_speech_shadowing.core.history import (
    HistoryEntry,
    delete_report,
    format_summary,
    list_reports,
    load_report,
    report_path,
    save_report,
)
from ai_speech_shadowing.core.phoneme import diff_phonemes
from ai_speech_shadowing.core.prosody import PitchStats, ProsodyDiff


def _report() -> FeedbackReport:
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
    return build_report(diff_phonemes(["a", "b"], ["a", "b"]), prosody, fluency)


@pytest.fixture
def history_dir(tmp_path: Path) -> Path:
    return tmp_path / "history"


class TestSaveLoad:
    def test_save_writes_json_with_id(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        assert path.is_file()
        assert path.name.startswith("eval_")
        assert path.suffix == ".json"
        data = json.loads(path.read_text())
        assert data["id"].startswith("eval_")
        assert "created_at" in data
        assert data["composite"]["score"] == 100

    def test_load_round_trip(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        rid = path.stem
        data = load_report(rid, history_dir)
        assert data is not None
        assert data["id"] == rid

    def test_load_missing_returns_none(self, history_dir: Path) -> None:
        assert load_report("eval_nope", history_dir) is None


class TestList:
    def test_empty_when_no_dir(self, history_dir: Path) -> None:
        assert list_reports(history_dir) == []

    def test_lists_entries(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir)
        save_report(_report(), history_dir=history_dir)
        entries = list_reports(history_dir)
        assert len(entries) == 2
        assert all(isinstance(e, HistoryEntry) for e in entries)
        assert all(e.id.startswith("eval_") for e in entries)
        assert all(e.composite_score == 100 for e in entries)

    def test_skips_malformed(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir)
        (history_dir / "eval_bad.json").write_text("{not json")
        entries = list_reports(history_dir)
        assert len(entries) == 1  # malformed skipped


class TestDelete:
    def test_delete_existing(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        assert delete_report(path.stem, history_dir) is True
        assert not path.is_file()

    def test_delete_missing(self, history_dir: Path) -> None:
        assert delete_report("eval_nope", history_dir) is False


# --------------------------------------------------------------------------- #
# Path safety — report_id traversal must be contained under history_dir
# --------------------------------------------------------------------------- #
class TestReportPathSafety:
    """Regression tests: a report_id containing traversal must not read or
    delete files outside the history (recordings) folder."""

    @pytest.mark.parametrize("report_id", ["..", ".", "../secret", "..%2f..%2fetc", "a/../b"])
    def test_report_path_rejects_traversal(self, history_dir: Path, report_id: str) -> None:
        assert report_path(report_id, history_dir, suffix=".json") is None
        assert report_path(report_id, history_dir, suffix=".wav") is None

    def test_report_path_accepts_clean_id(self, history_dir: Path) -> None:
        p = report_path("eval_abc12345", history_dir, suffix=".json")
        assert p == history_dir / "eval_abc12345.json"

    def test_load_traversal_returns_none(self, history_dir: Path) -> None:
        assert load_report("..", history_dir) is None
        assert load_report("../secret", history_dir) is None

    def test_delete_traversal_returns_false_and_preserves_data(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        # attempt to delete via traversal must fail AND leave real data intact
        assert delete_report("..", history_dir) is False
        assert path.is_file()


class TestFormatSummary:
    def test_contains_scores_and_feedback(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir)
        data = load_report(list_reports(history_dir)[0].id, history_dir)
        summary = format_summary(data)
        assert "Composite: 100/100" in summary
        assert "Pronunciation" in summary
        assert "Feedback:" in summary
