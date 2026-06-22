"""Persistence for evaluation reports (the Phase 8 ``/history`` store, used by
the Phase 7 ``report`` CLI command).

Reports are stored as JSON files under a history directory (default
``data/history/<eval_id>.json``). All operations are pure filesystem I/O, so
they unit-test cleanly against ``tmp_path``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_speech_shadowing.core.feedback import FeedbackReport

DEFAULT_HISTORY_DIR: Path = Path("data/history")

_SEVERITY: dict[str, str] = {"good": "🟢", "fair": "🟡", "needs_work": "🔴"}


def _new_id() -> str:
    return "eval_" + uuid.uuid4().hex[:8]


def save_report(
    report: FeedbackReport,
    *,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
) -> Path:
    """Serialize a FeedbackReport to ``<history_dir>/<id>.json`` and return the path."""
    from ai_speech_shadowing.core.feedback import report_to_dict

    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    rid = _new_id()
    data: dict[str, object] = {
        "id": rid,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    data.update(report_to_dict(report))
    path = history_dir / f"{rid}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One row in ``report`` (the list view)."""

    id: str
    created_at: str
    path: Path
    composite_score: int
    composite_grade: str


def list_reports(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
) -> list[HistoryEntry]:
    """List every saved report, newest path-first by id (sorted by filename)."""
    history_dir = Path(history_dir)
    if not history_dir.is_dir():
        return []
    entries: list[HistoryEntry] = []
    for path in sorted(history_dir.glob("eval_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        composite = data.get("composite", {}) if isinstance(data.get("composite"), dict) else {}
        entries.append(
            HistoryEntry(
                id=str(data.get("id", path.stem)),
                created_at=str(data.get("created_at", "")),
                path=path,
                composite_score=int(composite.get("score", 0)),
                composite_grade=str(composite.get("grade", "")),
            )
        )
    return entries


def load_report(
    report_id: str,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
) -> dict[str, object] | None:
    """Load one report dict by id, or None if it doesn't exist."""
    path = Path(history_dir) / f"{report_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_report(report_id: str, history_dir: str | Path = DEFAULT_HISTORY_DIR) -> bool:
    """Delete a report; return True if something was removed."""
    path = Path(history_dir) / f"{report_id}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


def format_summary(data: dict[str, object]) -> str:
    """Render a saved report dict as a compact terminal summary."""
    rid = str(data.get("id", "?"))
    created = str(data.get("created_at", ""))
    composite = data.get("composite", {})
    c_score = int(composite.get("score", 0)) if isinstance(composite, dict) else 0
    c_grade = str(composite.get("grade", "")) if isinstance(composite, dict) else ""
    scores = data.get("scores", {})
    pron = int(scores.get("pronunciation", {}).get("score", 0)) if isinstance(scores, dict) else 0
    into = int(scores.get("intonation", {}).get("score", 0)) if isinstance(scores, dict) else 0
    flu = int(scores.get("fluency", {}).get("score", 0)) if isinstance(scores, dict) else 0

    lines = [
        f"Report {rid}  ({created})",
        f"Composite: {c_score}/100 {_SEVERITY.get(c_grade, '')} {c_grade}",
        f"Pronunciation {pron} | Intonation {into} | Fluency {flu}",
    ]
    feedback = data.get("feedback", [])
    if isinstance(feedback, list) and feedback:
        lines.append("Feedback:")
        for msg in feedback:
            lines.append(f"  • {msg}")
    return "\n".join(lines)
