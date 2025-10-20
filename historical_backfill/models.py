"""
Core dataclasses and helpers for historical timelapse backfill operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Sequence


def parse_iso_datetime(value: str) -> datetime:
    """
    Parse ISO-like datetime strings, accepting a trailing 'Z' for UTC and
    MapLibre-style timestamps that omit colons in the time portion.
    """
    text = value.strip()
    if not text:
        raise ValueError("Datetime value is empty")

    # Normalise trailing Z into explicit offset
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO datetime: {value}") from exc


@dataclass(frozen=True)
class BoundingBox:
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    def validate(self) -> None:
        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax")
        if self.ymin > self.ymax:
            raise ValueError("ymin must be <= ymax")

    def coordinates(self) -> Iterator[tuple[int, int]]:
        """Yield every tile coordinate within the inclusive bounds."""
        for x in range(self.xmin, self.xmax + 1):
            for y in range(self.ymin, self.ymax + 1):
                yield x, y

    def count(self) -> int:
        return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)


@dataclass(frozen=True)
class RegionSpec:
    slug: str
    name: str
    bounding_box: BoundingBox

    def coordinates(self) -> Iterator[tuple[int, int]]:
        return self.bounding_box.coordinates()


@dataclass(frozen=True)
class BackfillRequest:
    region: RegionSpec
    start: datetime
    end: datetime
    interval: timedelta
    dest_root: Path
    source_root: Path
    tiles_subdir: Path | None
    dry_run: bool
    overwrite: bool
    deduplicate: bool

    def validate(self) -> None:
        if self.interval <= timedelta(0):
            raise ValueError("Interval must be greater than zero")
        if self.start > self.end:
            raise ValueError("Start datetime must be <= end datetime")
        self.region.bounding_box.validate()

    def frame_datetimes(self) -> List[datetime]:
        frames: List[datetime] = []
        current = self.start
        while current <= self.end:
            frames.append(current)
            current += self.interval
        return frames

    def session_path(self, frame_time: datetime) -> Path:
        date_part = frame_time.strftime("%Y-%m-%d")
        time_part = frame_time.strftime("%H-%M-%S")
        return (
            self.dest_root
            / self.region.slug
            / date_part
            / time_part
        )


PLACEHOLDER_SUFFIX = ".placeholder"


def placeholder_path(session_dir: Path, x: int, y: int) -> Path:
    return session_dir / f"{x}_{y}{PLACEHOLDER_SUFFIX}"


def tile_filename(x: int, y: int) -> str:
    return f"{x}_{y}.png"

