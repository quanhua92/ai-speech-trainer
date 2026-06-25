"""Persistence for evaluation reports (the Phase 8 ``/history`` store, used by
the Phase 7 ``report`` CLI command).

Reports are stored as JSON files under a history directory (default
``data/history/<eval_id>.json``). All operations are pure filesystem I/O, so
they unit-test cleanly against ``tmp_path``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_speech_shadowing.core.feedback import FeedbackReport

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_DIR: Path = Path("data/history")

_SEVERITY: dict[str, str] = {"good": "🟢", "fair": "🟡", "needs_work": "🔴"}

# Report ids are "eval_" + uuid hex; allow [A-Za-z0-9_-] so "..", ".", and any
# "/"-containing traversal are rejected before path construction.
_REPORT_ID_RE: re.compile[str] = re.compile(r"^[A-Za-z0-9_-]+$")

# On-disk user ids are sha256 digests (64 lowercase hex chars) or the fixed
# ``_cli`` bucket. ``[A-Za-z0-9_-]+`` is safe for a path segment (no ``/``,
# ``.``, or ``..``) while covering both forms.
_USER_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]+$")
CLI_USER_ID: str = "_cli"


def _new_id() -> str:
    return "eval_" + uuid.uuid4().hex[:8]


def _user_segment(user_id: str | None) -> str:
    """Resolve ``user_id`` to the on-disk subdirectory name.

    ``None`` (CLI ``report`` all-users view) is signalled separately by the
    caller; when a concrete user is requested it is validated against
    ``_USER_ID_RE`` before becoming a path segment.
    """
    if user_id is None:
        return CLI_USER_ID
    if not _USER_ID_RE.match(user_id):
        raise ValueError(f"invalid user_id {user_id!r}")
    return user_id


def report_path(
    report_id: str,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    user_id: str | None = None,
    *,
    suffix: str,
) -> Path | None:
    """Return the on-disk path for ``report_id`` only if it is well-formed AND
    stays inside ``history_dir`` after resolving ``..`` and symlinks; otherwise
    ``None``.

    When ``user_id`` is given the path is ``history_dir / user_id / f"{id}{suffix}"``.
    When ``user_id`` is ``None`` the top-level (legacy) location is used — this
    supports the CLI all-users ``report`` view and the cleanup sweep.
    """
    if not _REPORT_ID_RE.match(report_id):
        return None
    base = Path(history_dir).resolve()
    if user_id is not None:
        if not _USER_ID_RE.match(user_id):
            return None
        path = Path(history_dir) / user_id / f"{report_id}{suffix}"
    else:
        path = Path(history_dir) / f"{report_id}{suffix}"
    try:
        path.resolve().relative_to(base)
    except ValueError:
        return None
    return path


def _find_report_path(
    report_id: str,
    history_dir: str | Path,
    user_id: str | None,
    *,
    suffix: str,
) -> Path | None:
    """Locate a report's file: scoped to ``user_id`` when given, else search
    every user subdirectory and the top level (the CLI all-users path)."""
    if user_id is not None:
        return report_path(report_id, history_dir, user_id, suffix=suffix)
    # all-users search: top-level legacy + every subdirectory
    top = report_path(report_id, history_dir, None, suffix=suffix)
    if top is not None and top.is_file():
        return top
    base = Path(history_dir)
    if not base.is_dir():
        return None
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / f"{report_id}{suffix}"
        try:
            candidate.resolve().relative_to(base.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def save_report(
    report: FeedbackReport,
    *,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    user_id: str | None = None,
) -> Path:
    """Serialize a FeedbackReport to ``<history_dir>/<user>/<id>.json`` and
    return the path. ``user_id=None`` writes to the ``_cli`` bucket."""
    from ai_speech_shadowing.core.feedback import report_to_dict

    seg = _user_segment(user_id)
    out_dir = Path(history_dir) / seg
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = _new_id()
    data: dict[str, object] = {
        "id": rid,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    data.update(report_to_dict(report))
    path = out_dir / f"{rid}.json"
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
    reference_id: str | None = None


def list_reports(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    user_id: str | None = None,
) -> list[HistoryEntry]:
    """List saved reports, newest path-first by id.

    When ``user_id`` is given, only that user's subdirectory is scanned. When
    ``None``, every subdirectory (all users) plus the top-level legacy files
    are scanned — the CLI all-users view.
    """
    history_dir = Path(history_dir)
    if not history_dir.is_dir():
        return []
    paths: list[Path] = []
    if user_id is not None:
        if not _USER_ID_RE.match(user_id):
            return []
        paths = sorted((history_dir / user_id).glob("eval_*.json"))
    else:
        paths = sorted(history_dir.rglob("eval_*.json"))
    entries: list[HistoryEntry] = []
    for path in paths:
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
                reference_id=data.get("reference_id"),
            )
        )
    return entries


def load_report(
    report_id: str,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    user_id: str | None = None,
) -> dict[str, object] | None:
    """Load one report dict by id, or None if it doesn't exist.

    When ``user_id`` is given, only that user's directory is checked (user A
    cannot read user B's eval). When ``None``, all directories are searched.
    """
    path = _find_report_path(report_id, history_dir, user_id, suffix=".json")
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_report(
    report_id: str,
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    user_id: str | None = None,
) -> bool:
    """Delete a report; return True if something was removed."""
    path = _find_report_path(report_id, history_dir, user_id, suffix=".json")
    if path is None or not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def compute_stats(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    user_id: str | None = None,
    *,
    period_days: int = 30,
) -> dict[str, object]:
    """Aggregate saved reports into a stats dict matching ``StatsResponse``.

    Reads every report (scoped to ``user_id`` when given, else all users),
    filters to the last ``period_days`` by ``created_at``, and computes average
    pillar/composite scores, a coarse trend, the most frequently mispronounced
    phonemes, and a per-day breakdown.
    """
    history_dir = Path(history_dir)
    now = dt.datetime.now(UTC)
    cutoff = now - dt.timedelta(days=period_days)

    if user_id is not None and _USER_ID_RE.match(user_id):
        user_dir = history_dir / user_id
        paths = sorted(user_dir.glob("eval_*.json")) if user_dir.is_dir() else []
    elif history_dir.is_dir():
        paths = sorted(history_dir.rglob("eval_*.json"))
    else:
        paths = []

    valid: list[tuple[dt.datetime, dict[str, object]]] = []
    for path in paths:
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


# --------------------------------------------------------------------------- #
# Retention — delete reports older than N days
# --------------------------------------------------------------------------- #
def cleanup_old_reports(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    retention_days: int = 7,
) -> int:
    """Delete eval reports (JSON + WAV) whose ``created_at`` is older than
    ``retention_days``. Returns the number of reports removed.

    Scans every user subdirectory AND the top-level legacy files (``rglob``).
    Empty user directories left behind are removed. ``retention_days <= 0`` is
    a no-op (keep forever).
    """
    if retention_days <= 0:
        return 0
    history_dir = Path(history_dir)
    if not history_dir.is_dir():
        return 0
    cutoff = datetime.now(UTC) - dt.timedelta(days=retention_days)
    deleted = 0
    for path in sorted(history_dir.rglob("eval_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ts = _parse_created_at(str(data.get("created_at", "")))
        if ts is None or ts >= cutoff:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not delete %s", path)
            continue
        wav = path.with_suffix(".wav")
        if wav.is_file():
            with suppress(OSError):
                wav.unlink(missing_ok=True)
        deleted += 1
    # sweep up empty per-user directories
    for sub in history_dir.iterdir():
        if sub.is_dir() and not any(sub.iterdir()):
            with suppress(OSError):
                sub.rmdir()
    if deleted:
        logger.info(
            "history cleanup: deleted %d report(s) older than %d day(s)",
            deleted,
            retention_days,
        )
    return deleted
