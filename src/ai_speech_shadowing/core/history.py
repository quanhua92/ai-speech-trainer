"""Persistence for evaluation reports (the Phase 8 ``/history`` store, used by
the Phase 7 ``report`` CLI command).

Reports are stored as JSON files under a history directory (default
``data/history/<eval_id>.json``). All operations are pure filesystem I/O, so
they unit-test cleanly against ``tmp_path``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_speech_shadowing.core.feedback import FeedbackReport

DEFAULT_HISTORY_DIR: Path = Path("data/history")

_SEVERITY: dict[str, str] = {"good": "🟢", "fair": "🟡", "needs_work": "🔴"}

# Report ids are "eval_" + uuid hex; allow [A-Za-z0-9_-] so "..", ".", and any
# "/"-containing traversal are rejected before path construction.
_REPORT_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]+$")


def _new_id() -> str:
    return "eval_" + uuid.uuid4().hex[:8]


def report_path(
    report_id: str,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    *,
    suffix: str,
) -> Path | None:
    """Return ``history_dir / f"{report_id}{suffix}"`` only if it is a well-formed
    id AND stays inside ``history_dir`` after resolving ``..`` and symlinks;
    otherwise ``None``.

    The fixed ``.json``/``.wav`` suffix already blocks most traversal, but the
    format check + resolve() containment are defense-in-depth for the
    recordings/history folder.
    """
    if not _REPORT_ID_RE.match(report_id):
        return None
    base = Path(history_dir).resolve()
    path = Path(history_dir) / f"{report_id}{suffix}"
    try:
        path.resolve().relative_to(base)
    except ValueError:
        return None
    return path


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
    path = report_path(report_id, history_dir, suffix=".json")
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_report(report_id: str, history_dir: str | Path = DEFAULT_HISTORY_DIR) -> bool:
    """Delete a report; return True if something was removed."""
    path = report_path(report_id, history_dir, suffix=".json")
    if path is None or not path.is_file():
        return False
    path.unlink()
    return True


def compute_stats(
    history_dir: str | Path = DEFAULT_HISTORY_DIR, *, period_days: int = 30
) -> dict[str, object]:
    """Aggregate saved reports into a stats dict matching ``StatsResponse``.

    Reads every report, filters to the last ``period_days`` by ``created_at``,
    and computes average pillar/composite scores, a coarse trend, the most
    frequently mispronounced phonemes, and a per-day breakdown.
    """
    history_dir = Path(history_dir)
    now = dt.datetime.now(UTC)
    cutoff = now - dt.timedelta(days=period_days)

    valid: list[tuple[dt.datetime, dict[str, object]]] = []
    if history_dir.is_dir():
        for path in sorted(history_dir.glob("eval_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ts = _parse_created_at(str(data.get("created_at", "")))
            if ts is not None and ts >= cutoff:
                valid.append((ts, data))

    total = len(valid)
    if total:
        avg_pron = _mean(_pillar_scores(valid, "pronunciation"))
        avg_into = _mean(_pillar_scores(valid, "intonation"))
        avg_flu = _mean(_pillar_scores(valid, "fluency"))
        avg_comp = _mean(_composite_scores(valid))
    else:
        avg_pron = avg_into = avg_flu = avg_comp = 0.0

    by_day: dict[str, list[int]] = defaultdict(list)
    for ts, data in valid:
        by_day[ts.date().isoformat()].append(_composite_of(data))
    daily = [
        {"date": day, "count": len(scores), "avg_composite": round(_mean(scores))}
        for day, scores in sorted(by_day.items())
    ]

    return {
        "period_days": period_days,
        "total_evaluations": total,
        "average_scores": {
            "pronunciation": round(avg_pron, 1),
            "intonation": round(avg_into, 1),
            "fluency": round(avg_flu, 1),
            "composite": round(avg_comp, 1),
        },
        "trend": _trend(valid),
        "weakest_phonemes": _weakest_phonemes(valid),
        "daily_breakdown": daily,
    }


def _parse_created_at(raw: str) -> dt.datetime | None:
    if not raw:
        return None
    try:
        ts = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.UTC)
    return ts


def _composite_of(data: dict[str, object]) -> int:
    composite = data.get("composite", {})
    return int(composite.get("score", 0)) if isinstance(composite, dict) else 0


def _pillar_scores(valid: list[tuple[object, dict[str, object]]], pillar: str) -> list[int]:
    out: list[int] = []
    for _ts, data in valid:
        scores = data.get("scores", {})
        if isinstance(scores, dict):
            block = scores.get(pillar, {})
            if isinstance(block, dict):
                out.append(int(block.get("score", 0)))
    return out


def _composite_scores(valid: list[tuple[object, dict[str, object]]]) -> list[int]:
    return [_composite_of(data) for _ts, data in valid]


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trend(valid: list[tuple[dt.datetime, dict[str, object]]]) -> str:
    """Compare the mean composite of the earlier vs later half of reports."""
    if len(valid) < 4:
        return "insufficient"
    ordered = sorted(valid, key=lambda x: x[0])
    mid = len(ordered) // 2
    first = _mean([_composite_of(d) for _ts, d in ordered[:mid]])
    second = _mean([_composite_of(d) for _ts, d in ordered[mid:]])
    if second > first + 2:
        return "improving"
    if second < first - 2:
        return "declining"
    return "steady"


def _weakest_phonemes(valid: list[tuple[object, dict[str, object]]], *, top: int = 5) -> list[str]:
    """Most-frequently mispronounced phonemes (the 'expected' of each sub op)."""
    counter: Counter[str] = Counter()
    for _ts, data in valid:
        ops = data.get("phoneme_diff", [])
        if not isinstance(ops, list):
            continue
        for op in ops:
            if isinstance(op, dict) and op.get("type") == "sub":
                expected = op.get("expected")
                if expected:
                    counter[str(expected)] += 1
    return [phoneme for phoneme, _count in counter.most_common(top)]


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
