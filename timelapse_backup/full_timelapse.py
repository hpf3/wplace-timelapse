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
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    content_width: Optional[int] = None
    content_height: Optional[int] = None
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    pad_left: Optional[int] = None
    pad_top: Optional[int] = None

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
                video_width = entry.get("video_width")
                if video_width is None and "width" in entry:
                    video_width = entry["width"]
                video_height = entry.get("video_height")
                if video_height is None and "height" in entry:
                    video_height = entry["height"]
                content_width = entry.get("content_width")
                content_height = entry.get("content_height")
                segment = FullTimelapseSegment(
                    path=str(entry["path"]),
                    first_session=str(entry["first_session"]),
                    last_session=str(entry["last_session"]),
                    frame_count=int(entry.get("frame_count", 0)),
                    video_width=int(video_width) if video_width is not None else None,
                    video_height=int(video_height) if video_height is not None else None,
                    content_width=int(content_width) if content_width is not None else None,
                    content_height=int(content_height) if content_height is not None else None,
                    crop_x=(
                        int(entry["crop_x"])
                        if "crop_x" in entry and entry["crop_x"] is not None
                        else None
                    ),
                    crop_y=(
                        int(entry["crop_y"])
                        if "crop_y" in entry and entry["crop_y"] is not None
                        else None
                    ),
                    pad_left=(
                        int(entry["pad_left"])
                        if "pad_left" in entry and entry["pad_left"] is not None
                        else None
                    ),
                    pad_top=(
                        int(entry["pad_top"])
                        if "pad_top" in entry and entry["pad_top"] is not None
                        else None
                    ),
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

    def max_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        widths = [segment.video_width for segment in self.segments if segment.video_width]
        heights = [segment.video_height for segment in self.segments if segment.video_height]
        max_width = max(widths) if widths else None
        max_height = max(heights) if heights else None
        return max_width, max_height

    def content_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        min_x: Optional[int] = None
        min_y: Optional[int] = None
        max_x: Optional[int] = None
        max_y: Optional[int] = None

        for segment in self.segments:
            crop_x = segment.crop_x
            crop_y = segment.crop_y
            width = segment.content_width or segment.video_width
            height = segment.content_height or segment.video_height
            if crop_x is None or crop_y is None or width is None or height is None:
                continue
            if min_x is None or crop_x < min_x:
                min_x = crop_x
            if min_y is None or crop_y < min_y:
                min_y = crop_y
            candidate_max_x = crop_x + width
            candidate_max_y = crop_y + height
            if max_x is None or candidate_max_x > max_x:
                max_x = candidate_max_x
            if max_y is None or candidate_max_y > max_y:
                max_y = candidate_max_y

        if min_x is None or min_y is None or max_x is None or max_y is None:
            return None
        return (min_x, min_y, max_x, max_y)

    def segment_path(self, segment: FullTimelapseSegment) -> Path:
        return self._resolve_path(segment.path)


__all__ = [
    "FullTimelapseSegment",
    "FullTimelapseState",
]
