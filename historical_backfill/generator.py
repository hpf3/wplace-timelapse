"""
Generate historical backup sessions by copying tiles from archived releases.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .models import (
    BackfillRequest,
    placeholder_path,
    tile_filename,
)
from .releases import ReleaseInfo, select_release_for_timestamp

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationSummary:
    frames_created: int = 0
    frames_skipped: int = 0
    tiles_copied: int = 0
    placeholders_written: int = 0
    missing_tiles: int = 0
    releases_used: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, object]:
        return {
            "frames_created": self.frames_created,
            "frames_skipped": self.frames_skipped,
            "tiles_copied": self.tiles_copied,
            "placeholders_written": self.placeholders_written,
            "missing_tiles": self.missing_tiles,
            "releases_used": sorted(self.releases_used),
        }


class HistoricalGenerator:
    def __init__(
        self,
        request: BackfillRequest,
        releases: Sequence[ReleaseInfo],
        logger: logging.Logger | None = None,
    ):
        self.request = request
        self.releases = sorted(releases, key=lambda release: release.timestamp)
        self.logger = logger or LOGGER
        self.summary = GenerationSummary()

    def run(self) -> GenerationSummary:
        self.request.validate()

        frames = self.request.frame_datetimes()
        if not frames:
            self.logger.info("No frames required for requested range")
            return self.summary

        coordinates = list(self.request.region.coordinates())
        if not coordinates:
            self.logger.info("Region %s has no coordinates to copy", self.request.region.slug)
            return self.summary

        previous_session: Path | None = None
        previous_release: ReleaseInfo | None = None

        for frame_time in frames:
            release = select_release_for_timestamp(self.releases, frame_time)

            if release is None:
                self._handle_missing_release(frame_time, coordinates, previous_session)
                continue

            is_duplicate = (
                previous_release is not None
                and release.timestamp == previous_release.timestamp
                and self.request.deduplicate
            )

            if is_duplicate and previous_session is not None:
                created = self._write_placeholder_session(frame_time, coordinates)
                if created:
                    self.summary.frames_created += 1
                    previous_session = self.request.session_path(frame_time)
                continue

            created = self._materialise_session(frame_time, release, coordinates, previous_session)
            if created:
                previous_release = release
                previous_session = self.request.session_path(frame_time)
                self.summary.releases_used.add(release.name)

        return self.summary

    def _handle_missing_release(
        self,
        frame_time: datetime,
        coordinates: Iterable[tuple[int, int]],
        previous_session: Path | None,
    ) -> None:
        session_dir = self.request.session_path(frame_time)
        if previous_session is None:
            self.logger.warning(
                "Skipping %s - no release available and no prior session to reference",
                frame_time.isoformat(),
            )
            self.summary.frames_skipped += 1
            return

        created = self._write_placeholder_session(frame_time, coordinates)
        if created:
            self.summary.frames_created += 1

    def _prepare_session_dir(self, session_dir: Path) -> bool:
        if session_dir.exists():
            if not self.request.overwrite:
                self.logger.info("Skipping existing session %s (use --overwrite to replace)", session_dir)
                self.summary.frames_skipped += 1
                return False
            if self.request.dry_run:
                self.logger.debug("Would clear existing session %s", session_dir)
            else:
                for item in session_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        self.logger.warning("Leaving unexpected entry in session dir: %s", item)
        else:
            if self.request.dry_run:
                self.logger.debug("Would create session directory %s", session_dir)
            else:
                session_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _write_placeholder_session(
        self,
        frame_time: datetime,
        coordinates: Iterable[tuple[int, int]],
    ) -> bool:
        session_dir = self.request.session_path(frame_time)
        if not self._prepare_session_dir(session_dir):
            return False

        if self.request.dry_run:
            self.logger.info("Dry run: would create placeholder session for %s", frame_time.isoformat())
            return True

        for x, y in coordinates:
            placeholder = placeholder_path(session_dir, x, y)
            if placeholder.exists():
                continue
            data = {
                "type": "placeholder",
                "version": 2,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            }
            with open(placeholder, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            self.summary.placeholders_written += 1

        self.logger.info("Created placeholder session %s", session_dir)
        return True

    def _materialise_session(
        self,
        frame_time: datetime,
        release: ReleaseInfo,
        coordinates: Iterable[tuple[int, int]],
        previous_session: Path | None,
    ) -> bool:
        session_dir = self.request.session_path(frame_time)
        if not self._prepare_session_dir(session_dir):
            return False

        if self.request.dry_run:
            self.logger.info(
                "Dry run: would create session %s from release %s",
                session_dir,
                release.name,
            )
            return True

        session_dir.mkdir(parents=True, exist_ok=True)
        copied_any = False

        for x, y in coordinates:
            src = release.tile_path(x, y)
            dest = session_dir / tile_filename(x, y)

            if src.exists():
                shutil.copy2(src, dest)
                copied_any = True
                self.summary.tiles_copied += 1
                continue

            if previous_session is not None:
                placeholder = placeholder_path(session_dir, x, y)
                data = {
                    "type": "placeholder",
                    "version": 2,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                }
                with open(placeholder, "w", encoding="utf-8") as handle:
                    json.dump(data, handle)
                self.summary.placeholders_written += 1
                continue

            self.logger.warning(
                "Missing tile %s in release %s (frame %s)",
                src,
                release.name,
                frame_time.isoformat(),
            )
            self.summary.missing_tiles += 1

        if copied_any:
            self.summary.frames_created += 1
            self.logger.info("Created session %s using release %s", session_dir, release.name)
            return True

        self.logger.warning(
            "No tiles copied for session %s (release %s). The frame only contains placeholders.",
            session_dir,
            release.name,
        )
        self.summary.frames_created += 1
        return True
