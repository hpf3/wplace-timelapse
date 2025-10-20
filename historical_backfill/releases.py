"""
Utilities for discovering and selecting historical map releases.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)


TIMESTAMP_REGEX = re.compile(
    r"""
    ^
    (?P<prefix>[a-zA-Z0-9_-]+?)-
    (?P<date>\d{4}-\d{2}-\d{2})
    T
    (?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})
    (?P<rest>\.\d+)?       # optional fractional seconds
    (?P<suffix>Z|[+-]\d{2}:\d{2})?
    $
    """,
    re.VERBOSE,
)


def parse_release_timestamp(name: str) -> Optional[datetime]:
    match = TIMESTAMP_REGEX.match(name)
    if not match:
        return None

    parts = match.groupdict()
    fractional = parts.get("rest") or ""
    suffix = parts.get("suffix") or "+00:00"
    iso = (
        f"{parts['date']}T"
        f"{parts['hour']}:{parts['minute']}:{parts['second']}"
        f"{fractional}"
        f"{suffix.replace('Z', '+00:00')}"
    )

    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        LOGGER.debug("Failed to parse release timestamp from %s", name, exc_info=True)
        return None


@dataclass(frozen=True)
class ReleaseInfo:
    name: str
    timestamp: datetime
    base_path: Path
    tiles_root: Path

    def tile_path(self, x: int, y: int) -> Path:
        return self.tiles_root / str(x) / f"{y}.png"


def _candidate_tiles_root(base_path: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        candidate = (base_path / override).resolve()
        return candidate if candidate.exists() else None

    if base_path.exists():
        return base_path
    return None


def scan_local_releases(
    root: Path,
    tiles_subdir: Optional[Path] = None,
) -> List[ReleaseInfo]:
    if not root.exists():
        LOGGER.warning("Release directory %s does not exist", root)
        return []

    releases: List[ReleaseInfo] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        timestamp = parse_release_timestamp(entry.name)
        if timestamp is None:
            LOGGER.debug("Skipping %s - cannot parse timestamp", entry)
            continue

        tiles_root = _candidate_tiles_root(entry, tiles_subdir)
        if tiles_root is None or not tiles_root.exists():
            LOGGER.warning("Tiles root missing for release %s (expected under %s)", entry.name, tiles_root or entry)
            continue

        releases.append(
            ReleaseInfo(
                name=entry.name,
                timestamp=timestamp,
                base_path=entry,
                tiles_root=tiles_root,
            )
        )

    releases.sort(key=lambda release: release.timestamp)
    return releases


def select_release_for_timestamp(
    releases: Sequence[ReleaseInfo],
    frame_time: datetime,
) -> Optional[ReleaseInfo]:
    if not releases:
        return None

    timestamps = [release.timestamp for release in releases]
    index = bisect_right(timestamps, frame_time) - 1
    if index < 0:
        return None
    return releases[index]

