"""Migration utilities executed during container startup."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import re

from timelapse_backup.app import TimelapseBackup
from timelapse_backup.full_timelapse import (
    FullTimelapseSegment,
    FullTimelapseState,
)
from timelapse_backup.sessions import (
    get_all_sessions,
    parse_session_datetime,
)

LOGGER = logging.getLogger("timelapse.migrations")


@dataclass
class StatsCoverage:
    first_session: Optional[datetime]
    last_session: Optional[datetime]
    frame_count: Optional[int]


def _safe_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _segment_exists(state: FullTimelapseState) -> bool:
    """Return True if the manifest already records any segment."""
    state.load()
    return bool(state.segments)


def _log_skip(
    slug: str,
    mode_name: str,
    suffix: str,
    reason: str,
) -> None:
    LOGGER.info(
        "Skipping full timelapse migration for %s (%s) suffix '%s': %s",
        slug,
        mode_name,
        suffix,
        reason,
    )


def _estimate_frame_count(session_dirs: Iterable[Path]) -> int:
    return sum(1 for _ in session_dirs)


def _parse_datetime(value: str) -> Optional[datetime]:
    stripped = value.strip()
    if not stripped or stripped.lower() == "unknown":
        return None
    if stripped.endswith("Z"):
        stripped = stripped[:-1]
    try:
        return datetime.fromisoformat(stripped)
    except ValueError:
        LOGGER.warning("Failed to parse datetime from stats.txt value: %s", value)
        return None


def _parse_int(value: str) -> Optional[int]:
    digits = value.replace(",", "").strip()
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        LOGGER.warning("Failed to parse integer from stats.txt value: %s", value)
        return None


def _parse_stats_report(stats_path: Path) -> StatsCoverage:
    first: Optional[datetime] = None
    last: Optional[datetime] = None
    frames: Optional[int] = None

    first_pattern = re.compile(r"^First frame:\s*(.+)$")
    last_pattern = re.compile(r"^Last frame:\s*(.+)$")
    frames_pattern = re.compile(r"^Frames rendered:\s*([\d,]+)$")

    try:
        for raw_line in stats_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = first_pattern.match(line)
            if match and first is None:
                candidate = _parse_datetime(match.group(1))
                if candidate is not None:
                    first = candidate
                continue
            match = last_pattern.match(line)
            if match and last is None:
                candidate = _parse_datetime(match.group(1))
                if candidate is not None:
                    last = candidate
                continue
            match = frames_pattern.match(line)
            if match and frames is None:
                candidate = _parse_int(match.group(1))
                if candidate is not None:
                    frames = candidate
    except OSError as exc:
        LOGGER.warning("Failed to read stats report %s: %s", stats_path, exc)

    return StatsCoverage(first_session=first, last_session=last, frame_count=frames)


def _prepare_segment_file(
    slug_dir: Path,
    output_filename: str,
    segment_path: Path,
) -> bool:
    """Copy the legacy full video as the first incremental segment."""
    legacy_path = slug_dir / output_filename
    if not legacy_path.exists():
        return False

    segment_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(legacy_path, segment_path)
    except OSError as exc:
        LOGGER.error(
            "Failed to copy legacy timelapse into segment storage: %s -> %s (%s)",
            legacy_path,
            segment_path,
            exc,
        )
        return False

    return True


def migrate_full_timelapse_segments(backup: TimelapseBackup) -> None:
    """Backfill segment manifests for legacy full timelapses."""
    for timelapse in backup.get_enabled_timelapses():
        slug = timelapse["slug"]
        name = timelapse["name"]
        slug_dir = backup.output_dir / slug

        for mode in backup.get_enabled_timelapse_modes(timelapse):
            if not mode.get("create_full"):
                continue

            suffix = mode["suffix"]
            mode_name = mode["mode"]
            output_filename = f"full{suffix}.mp4"
            legacy_path = slug_dir / output_filename
            if not legacy_path.exists():
                _log_skip(slug, mode_name, suffix, "legacy full video not found")
                continue

            state = FullTimelapseState(
                slug_dir,
                output_filename,
                logger=LOGGER,
            )
            if _segment_exists(state):
                _log_skip(slug, mode_name, suffix, "segments manifest already present")
                continue

            session_dirs = get_all_sessions(backup.backup_dir, slug)
            if not session_dirs:
                _log_skip(slug, mode_name, suffix, "no captured sessions available")
                continue

            stats_path = legacy_path.with_suffix(legacy_path.suffix + ".stats.txt")
            stats_info = _parse_stats_report(stats_path) if stats_path.exists() else None

            first_dt = parse_session_datetime(session_dirs[0])
            last_dt = parse_session_datetime(session_dirs[-1])

            if stats_info is not None:
                if stats_info.first_session is not None:
                    first_dt = stats_info.first_session
                if stats_info.last_session is not None:
                    last_dt = stats_info.last_session

            if first_dt is None or last_dt is None:
                _log_skip(slug, mode_name, suffix, "unable to derive session timestamps")
                continue

            state.ensure_segments_dir()
            segment_path = state.make_segment_filename(first_dt, last_dt)
            if segment_path.exists():
                segment_path.unlink()

            if not _prepare_segment_file(slug_dir, output_filename, segment_path):
                _log_skip(slug, mode_name, suffix, "failed to create initial segment copy")
                continue

            relative_segment_path = segment_path.relative_to(slug_dir)
            if stats_info is not None and stats_info.frame_count is not None:
                frame_count = stats_info.frame_count
            else:
                frame_count = _estimate_frame_count(session_dirs)
            segment = FullTimelapseSegment(
                path=relative_segment_path.as_posix(),
                first_session=_safe_iso(first_dt),
                last_session=_safe_iso(last_dt),
                frame_count=frame_count,
            )

            state.segments = [segment]
            state.save()
            state.write_concat_file(temporary=False)

            LOGGER.info(
                "Migrated legacy full timelapse for '%s' (%s) suffix '%s' using %s frames",
                name,
                slug,
                suffix,
                frame_count,
            )


def run_all() -> None:
    """Entry point for startup migrations."""
    LOGGER.info("Starting startup migrations")
    backup = TimelapseBackup()
    migrate_full_timelapse_segments(backup)
    LOGGER.info("Startup migrations completed")


__all__ = ["migrate_full_timelapse_segments", "run_all"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
