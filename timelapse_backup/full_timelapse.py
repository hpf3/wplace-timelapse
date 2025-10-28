"""Helpers for incremental full-timelapse assembly."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from timelapse_backup.sessions import parse_session_datetime


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _escape_for_concat(path: Path) -> str:
    """Escape single quotes for FFmpeg concat demuxer entries."""
    return str(path).replace("'", "'\\''")


@dataclass
class FullTimelapseSegment:
    """Metadata describing a rendered full-timelapse segment."""

    path: str
    first_session: str
    last_session: str
    frame_count: int

    def first_datetime(self) -> datetime:
        return _from_iso(self.first_session)

    def last_datetime(self) -> datetime:
        return _from_iso(self.last_session)


class FullTimelapseState:
    """Persisted state tracking rendered full-timelapse segments."""

    MANIFEST_VERSION = 1

    def __init__(
        self,
        slug_dir: Path,
        output_filename: str,
        *,
        logger: logging.Logger,
    ) -> None:
        self.slug_dir = slug_dir
        self.output_filename = output_filename
        self.logger = logger

        output_path = Path(output_filename)
        self.output_basename = output_path.stem
        self.output_suffix = "".join(output_path.suffixes) or ".mp4"

        self.segments_dir = self.slug_dir / "segments" / self.output_basename
        self.manifest_path = self.slug_dir / f"{self.output_basename}.segments.json"
        self.concat_path = self.slug_dir / f"{self.output_basename}.concat.txt"

        self.segments: List[FullTimelapseSegment] = []

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.manifest_path.exists():
            self.segments = []
            return

        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning(
                "Failed to read full-timelapse manifest %s: %s",
                self.manifest_path,
                exc,
            )
            self.segments = []
            return

        if not isinstance(data, dict):
            self.logger.warning(
                "Unexpected manifest format for %s; ignoring",
                self.manifest_path,
            )
            self.segments = []
            return

        raw_segments = data.get("segments", [])
        segments: List[FullTimelapseSegment] = []
        for entry in raw_segments:
            if not isinstance(entry, dict):
                continue
            try:
                segment = FullTimelapseSegment(
                    path=str(entry["path"]),
                    first_session=str(entry["first_session"]),
                    last_session=str(entry["last_session"]),
                    frame_count=int(entry.get("frame_count", 0)),
                )
            except KeyError:
                continue
            segments.append(segment)

        segments.sort(key=lambda seg: seg.first_datetime())
        self.segments = segments

    def save(self) -> None:
        payload = {
            "version": self.MANIFEST_VERSION,
            "output_filename": self.output_filename,
            "segments": [asdict(segment) for segment in self.segments],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_name(
            f".tmp_{uuid.uuid4().hex}_{self.manifest_path.name}"
        )
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.manifest_path)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def last_session_datetime(self) -> Optional[datetime]:
        if not self.segments:
            return None
        return self.segments[-1].last_datetime()

    def pending_sessions(self, session_dirs: Sequence[Path]) -> List[Path]:
        """Return sessions that have not yet been included in the manifest."""
        sortable: List[Tuple[datetime, Path]] = []
        for session_dir in session_dirs:
            session_dt = parse_session_datetime(session_dir)
            if session_dt is None:
                continue
            sortable.append((session_dt, session_dir))

        sortable.sort(key=lambda item: item[0])
        last_included = self.last_session_datetime()
        if last_included is None:
            return [path for _, path in sortable]

        return [path for session_dt, path in sortable if session_dt > last_included]

    # ------------------------------------------------------------------
    # Segment management
    # ------------------------------------------------------------------

    def ensure_segments_dir(self) -> None:
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    def make_segment_filename(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Path:
        start_str = start_dt.strftime("%Y%m%dT%H%M%S")
        end_str = end_dt.strftime("%Y%m%dT%H%M%S")
        filename = f"{self.output_basename}_{start_str}_{end_str}{self.output_suffix}"
        return self.segments_dir / filename

    def add_segment(self, segment: FullTimelapseSegment) -> None:
        self.segments.append(segment)
        self.segments.sort(key=lambda seg: seg.first_datetime())

    def _resolve_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        if path.is_absolute():
            return path
        return self.slug_dir / path

    def write_concat_file(
        self,
        segments: Optional[Sequence[FullTimelapseSegment]] = None,
        *,
        temporary: bool = False,
    ) -> Path:
        entries = segments if segments is not None else self.segments
        target = (
            self.concat_path.with_name(f".tmp_{uuid.uuid4().hex}_{self.concat_path.name}")
            if temporary
            else self.concat_path.with_name(
                f".tmp_{uuid.uuid4().hex}_{self.concat_path.name}"
            )
        )
        self.concat_path.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as handle:
            for segment in entries:
                path = self._resolve_path(segment.path)
                handle.write(f"file '{_escape_for_concat(path)}'\n")

        if temporary:
            return target

        target.replace(self.concat_path)
        return self.concat_path


__all__ = [
    "FullTimelapseSegment",
    "FullTimelapseState",
]
