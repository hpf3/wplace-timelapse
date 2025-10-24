"""Session discovery helpers for the timelapse backup system."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def extract_slug_from_session(session_dir: Path, backup_root: Path) -> Optional[str]:
    """Derive the timelapse slug from a session directory relative to the backup root."""
    try:
        relative = session_dir.resolve().relative_to(backup_root.resolve())
    except ValueError:
        return None

    parts = relative.parts
    return parts[0] if parts else None


def parse_session_datetime(session_dir: Path) -> Optional[datetime]:
    """Parse the datetime encoded in a session directory path."""
    try:
        return datetime.strptime(
            f"{session_dir.parent.name} {session_dir.name}",
            "%Y-%m-%d %H-%M-%S",
        )
    except ValueError:
        return None


def _collect_sessions(candidate_dirs: Iterable[Path]) -> List[Tuple[datetime, Path]]:
    sessions: List[Tuple[datetime, Path]] = []
    for session_dir in candidate_dirs:
        if not session_dir.is_dir():
            continue
        session_dt = parse_session_datetime(session_dir)
        if session_dt is None:
            continue
        sessions.append((session_dt, session_dir))
    return sessions


def get_prior_sessions(backup_root: Path, slug: str, current_session: Path) -> List[Path]:
    """Return previous sessions for a slug ordered from newest to oldest."""
    slug_dir = backup_root / slug
    if not slug_dir.exists():
        return []

    current_dt = parse_session_datetime(current_session)
    candidates: List[Tuple[datetime, Path]] = []

    for date_dir in slug_dir.iterdir():
        if not date_dir.is_dir():
            continue
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            if time_dir == current_session:
                continue
            session_dt = parse_session_datetime(time_dir)
            if session_dt is None:
                continue
            if current_dt is None or session_dt < current_dt:
                candidates.append((session_dt, time_dir))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates]


def get_session_dirs_for_date(backup_root: Path, slug: str, date_str: str) -> List[Path]:
    """Return session directories for a given slug/date ordered chronologically."""
    date_dir = backup_root / slug / date_str
    if not date_dir.exists():
        return []

    sessions = _collect_sessions(date_dir.iterdir())
    sessions.sort(key=lambda item: item[0])
    return [path for _, path in sessions]


def get_all_sessions(backup_root: Path, slug: str) -> List[Path]:
    """Collect all sessions for a slug ordered chronologically."""
    slug_dir = backup_root / slug
    if not slug_dir.exists():
        return []

    sessions: List[Tuple[datetime, Path]] = []
    for date_dir in slug_dir.iterdir():
        if not date_dir.is_dir():
            continue
        sessions.extend(_collect_sessions(date_dir.iterdir()))

    sessions.sort(key=lambda item: item[0])
    return [path for _, path in sessions]


__all__ = [
    "extract_slug_from_session",
    "parse_session_datetime",
    "get_prior_sessions",
    "get_session_dirs_for_date",
    "get_all_sessions",
]
