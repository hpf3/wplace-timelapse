"""Utilities for cleaning backup directories and backfilling from archives."""

from __future__ import annotations

import io
import json
import logging
import shutil
import tarfile
import uuid
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Optional

import cv2
import numpy as np
from zoneinfo import ZoneInfo

from historical_backfill.github_client import (
    RemoteRelease,
    download_assets,
    fetch_recent_releases,
)
from historical_backfill.releases import parse_release_timestamp


LOGGER = logging.getLogger("timelapse_cleanup")


@dataclass(frozen=True)
class BoundingBox:
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    def coordinates(self) -> Iterator[tuple[int, int]]:
        for x in range(self.xmin, self.xmax + 1):
            for y in range(self.ymin, self.ymax + 1):
                yield x, y


@dataclass(frozen=True)
class SlugConfig:
    slug: str
    name: str
    bounding_box: BoundingBox


@dataclass(frozen=True)
class SessionRecord:
    slug: str
    path: Path
    timestamp: datetime


@dataclass(frozen=True)
class GapInfo:
    slug: str
    start: datetime
    end: datetime
    previous_session: SessionRecord
    next_session: SessionRecord

    @property
    def start_utc(self) -> datetime:
        return self.start.astimezone(timezone.utc)

    @property
    def end_utc(self) -> datetime:
        return self.end.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReleaseCandidate:
    release: RemoteRelease
    capture_time: datetime


@dataclass(frozen=True)
class CachePaths:
    cache_root: Path
    archives_dir: Path


@dataclass
class _ReleaseAssignment:
    slug_config: SlugConfig
    gap: GapInfo
    session_dir: Path


@dataclass
class _ReleaseWork:
    candidate: ReleaseCandidate
    assignments: list[_ReleaseAssignment]


class ConcatenatedReader(io.RawIOBase):
    """Expose multiple file parts as a single readable stream."""

    def __init__(self, parts: Sequence[Path]):
        if not parts:
            raise ValueError("At least one archive part is required")
        self._parts = list(parts)
        self._index = 0
        self._current: io.BufferedReader | None = None
        self._closed = False

    def _ensure_stream(self) -> bool:
        while self._current is None and self._index < len(self._parts):
            path = self._parts[self._index]
            try:
                self._current = path.open("rb")
            except OSError:
                LOGGER.warning("Failed to open archive part %s", path)
                self._index += 1
                continue
        return self._current is not None

    def readable(self) -> bool:  # pragma: no cover - trivial
        return True

    def read(self, size: int | None = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if size is None:
            size = -1

        if not self._ensure_stream():
            return b""

        chunks: list[bytes] = []
        remaining = size

        while True:
            if self._current is None:
                if not self._ensure_stream():
                    break

            if size is None or size < 0:
                data = self._current.read()
            else:
                data = self._current.read(remaining)

            if data:
                chunks.append(data)
                if size is not None and size >= 0:
                    remaining -= len(data)
                    if remaining <= 0:
                        break
                continue

            if self._current is not None:
                self._current.close()
            self._current = None
            self._index += 1
            if size is not None and size >= 0 and remaining <= 0:
                break
            if self._index >= len(self._parts):
                break

        return b"".join(chunks)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        return n

    def close(self) -> None:  # pragma: no cover - trivial
        if self._closed:
            return
        if self._current is not None:
            self._current.close()
        self._current = None
        self._closed = True
        super().close()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_slug_configs(config: dict) -> dict[str, SlugConfig]:
    result: dict[str, SlugConfig] = {}
    for entry in config.get("timelapses", []):
        slug = entry.get("slug")
        coords = entry.get("coordinates") or {}
        if not slug or not coords:
            continue
        bbox = BoundingBox(
            xmin=int(coords.get("xmin")),
            xmax=int(coords.get("xmax")),
            ymin=int(coords.get("ymin")),
            ymax=int(coords.get("ymax")),
        )
        result[slug] = SlugConfig(
            slug=slug,
            name=entry.get("name") or slug,
            bounding_box=bbox,
        )
    return result


def prune_empty_directories(root: Path, dry_run: bool = False) -> list[Path]:
    removed: list[Path] = []
    if not root.exists():
        return removed

    entries = sorted(root.rglob("*"), key=lambda p: (-len(p.relative_to(root).parts), p.name))
    for path in entries:
        if not path.is_dir():
            continue
        try:
            has_entries = any(path.iterdir())
        except OSError:
            continue
        if has_entries:
            continue
        removed.append(path)
        if dry_run:
            LOGGER.info("Would remove empty directory %s", path)
        else:
            try:
                path.rmdir()
            except OSError:
                LOGGER.debug("Failed to remove %s", path, exc_info=True)
    return removed


def parse_session_path(slug: str, session_dir: Path, tz: ZoneInfo) -> SessionRecord | None:
    date_part = session_dir.parent.name
    time_part = session_dir.name
    try:
        naive = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S")
    except ValueError:
        return None
    aware = naive.replace(tzinfo=tz)
    return SessionRecord(slug=slug, path=session_dir, timestamp=aware)


def collect_sessions(slug_dir: Path, slug: str, tz: ZoneInfo) -> list[SessionRecord]:
    sessions: list[SessionRecord] = []
    if not slug_dir.exists():
        return sessions

    date_dirs = sorted((d for d in slug_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
    for date_dir in date_dirs:
        time_dirs = sorted((d for d in date_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
        for session_dir in time_dirs:
            record = parse_session_path(slug, session_dir, tz)
            if record is None:
                continue
            sessions.append(record)

    sessions.sort(key=lambda record: record.timestamp)
    return sessions


def find_gaps(
    sessions: Sequence[SessionRecord],
    interval: timedelta,
    min_gap: timedelta,
) -> list[GapInfo]:
    gaps: list[GapInfo] = []
    if len(sessions) < 2:
        return gaps

    for idx in range(len(sessions) - 1):
        current = sessions[idx]
        nxt = sessions[idx + 1]
        delta = nxt.timestamp - current.timestamp
        if delta < min_gap:
            continue
        gap_start = current.timestamp + interval
        gap_end = nxt.timestamp
        if gap_start >= gap_end:
            continue
        gaps.append(
            GapInfo(
                slug=current.slug,
                start=gap_start,
                end=gap_end,
                previous_session=current,
                next_session=nxt,
            )
        )
    return gaps


def determine_timezone(tz_name: str | None) -> ZoneInfo:
    if tz_name:
        return ZoneInfo(tz_name)
    local = datetime.now().astimezone().tzinfo
    if isinstance(local, ZoneInfo):
        return local
    return ZoneInfo("UTC")


def ensure_cache_paths(preferred_root: Path) -> CachePaths:
    cache_root = preferred_root
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path("historical_backfill/data")
        LOGGER.warning("Falling back to local cache directory %s", fallback)
        cache_root = fallback
        cache_root.mkdir(parents=True, exist_ok=True)

    archives_dir = cache_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    return CachePaths(cache_root=cache_root, archives_dir=archives_dir)


def fetch_releases_for_window(
    start: datetime,
    end: datetime,
    limit: int,
) -> list[ReleaseCandidate]:
    window_start = start.astimezone(timezone.utc)
    window_end = end.astimezone(timezone.utc)

    raw = fetch_recent_releases(limit)
    candidates: list[ReleaseCandidate] = []
    for release in raw:
        capture = parse_release_timestamp(release.tag)
        if capture is None:
            continue
        capture_utc = capture.astimezone(timezone.utc)
        if window_start <= capture_utc <= window_end:
            candidates.append(ReleaseCandidate(release=release, capture_time=capture))

    candidates.sort(key=lambda candidate: candidate.capture_time)
    return candidates


def _candidate_tar_parts(archive_dir: Path) -> Sequence[Path]:
    parts = sorted(archive_dir.glob("*.tar.gz.*"))
    if parts:
        return parts
    singles = sorted(archive_dir.glob("*.tar.gz"))
    if singles:
        return singles
    return []


def write_tiles_from_archive(
    archive_dir: Path,
    session_dir: Path,
    coords: Iterable[tuple[int, int]],
    release_tag: str,
) -> tuple[int, int]:
    parts = _candidate_tar_parts(archive_dir)
    if not parts:
        raise FileNotFoundError(f"No archive parts found in {archive_dir}")

    base_name = parts[0].name.split(".tar.gz", 1)[0]
    prefixes = [base_name, f"{base_name}/tiles"]

    targets: dict[str, tuple[int, int]] = {}
    coord_set = set(coords)
    for prefix in prefixes:
        for x, y in coord_set:
            targets[f"{prefix}/{x}/{y}.png"] = (x, y)

    extracted = 0

    session_dir.mkdir(parents=True, exist_ok=True)

    reader = ConcatenatedReader(parts)
    with tarfile.open(fileobj=reader, mode="r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            target = targets.get(member.name)
            if target is None:
                continue
            file_obj = handle.extractfile(member)
            if file_obj is None:
                continue
            x, y = target
            dest_path = session_dir / f"{x}_{y}.png"
            with dest_path.open("wb") as dest:
                shutil.copyfileobj(file_obj, dest)
            extracted += 1
            coord_set.discard((x, y))
            if not coord_set:
                break

    missing = len(coord_set)
    if missing:
        LOGGER.warning(
            "Archive %s missing %d tile(s) for session %s",
            release_tag,
            missing,
            session_dir,
        )
    return extracted, missing


class CleanupManager:
    def __init__(
        self,
        backup_root: Path,
        slug_configs: dict[str, SlugConfig],
        timezone_info: ZoneInfo,
        interval: timedelta,
        min_gap: timedelta,
        cache_paths: CachePaths,
        dry_run: bool = False,
        keep_archives: bool = False,
        release_limit: int = 1000,
        *,
        output_root: Path | None = None,
        background_color: tuple[int, int, int] | None = None,
        timelapse_quality: int = 23,
    ) -> None:
        self.backup_root = backup_root
        self.slug_configs = slug_configs
        self.timezone = timezone_info
        self.interval = interval
        self.min_gap = min_gap
        self.cache_paths = cache_paths
        self.dry_run = dry_run
        self.keep_archives = keep_archives
        self.release_limit = release_limit
        self.logger = LOGGER
        self.output_root = output_root
        self.background_color = background_color
        self._background_color_array: Optional[np.ndarray]
        if background_color is not None:
            self._background_color_array = np.array(background_color, dtype=np.uint8)
        else:
            self._background_color_array = None
        self._background_tolerance = 6
        self.timelapse_quality = timelapse_quality

    def run(self, slugs: Sequence[str] | None = None) -> None:
        target_slugs = list(slugs) if slugs else list(self.slug_configs)
        all_gaps: list[GapInfo] = []

        self._repair_output_videos(target_slugs)

        for slug in target_slugs:
            slug_config = self.slug_configs.get(slug)
            if slug_config is None:
                self.logger.warning("Skipping unknown slug %s", slug)
                continue

            slug_dir = self.backup_root / slug
            removed = prune_empty_directories(slug_dir, dry_run=self.dry_run)
            if removed:
                self.logger.info("Pruned %d empty directories under %s", len(removed), slug_dir)

            sessions = collect_sessions(slug_dir, slug, self.timezone)
            if not sessions:
                self.logger.info("No sessions discovered for %s", slug)
                continue

            gaps = find_gaps(sessions, self.interval, self.min_gap)
            if not gaps:
                self.logger.info("No gaps >= %s detected for %s", self.min_gap, slug)
                continue

            self.logger.info("Detected %d gap(s) for %s", len(gaps), slug)
            all_gaps.extend(gaps)

        if not all_gaps:
            self.logger.info("No qualifying gaps detected; cleanup complete")
            return

        earliest = min(gap.start for gap in all_gaps)
        latest = max(gap.end for gap in all_gaps)
        releases = fetch_releases_for_window(earliest, latest, self.release_limit)
        if not releases:
            self.logger.info("No historical releases found between %s and %s", earliest, latest)
            return

        assignments_by_release: dict[str, _ReleaseWork] = {}
        for gap in all_gaps:
            matches = [
                candidate
                for candidate in releases
                if gap.start_utc <= candidate.capture_time.astimezone(timezone.utc) < gap.end_utc
            ]
            if not matches:
                self.logger.info(
                    "No releases cover gap %s -> %s for %s",
                    gap.start.isoformat(),
                    gap.end.isoformat(),
                    gap.slug,
                )
                continue
            slug_config = self.slug_configs[gap.slug]
            for candidate in matches:
                session_dir = self._prepare_session_dir(slug_config.slug, candidate.capture_time)
                work = assignments_by_release.setdefault(
                    candidate.release.tag,
                    _ReleaseWork(candidate=candidate, assignments=[]),
                )
                if any(item.session_dir == session_dir for item in work.assignments):
                    continue
                work.assignments.append(
                    _ReleaseAssignment(
                        slug_config=slug_config,
                        gap=gap,
                        session_dir=session_dir,
                    )
                )

        if not assignments_by_release:
            self.logger.info("No matching releases for detected gaps; cleanup complete")
            return

        for work in sorted(
            assignments_by_release.values(),
            key=lambda item: item.candidate.capture_time,
        ):
            self._process_release_group(work)

    def _prepare_session_dir(self, slug: str, capture_time: datetime) -> Path:
        local_time = capture_time.astimezone(self.timezone)
        date_part = local_time.strftime("%Y-%m-%d")
        time_part = local_time.strftime("%H-%M-%S")
        return self.backup_root / slug / date_part / time_part

    def _clear_other_archives(self, keep: str | None = None) -> None:
        archives_dir = self.cache_paths.archives_dir
        if not archives_dir.exists():
            return
        for entry in archives_dir.iterdir():
            if not entry.is_dir():
                continue
            if keep and entry.name == keep:
                continue
            if self.dry_run:
                self.logger.info("Would remove archive directory %s", entry)
            else:
                shutil.rmtree(entry, ignore_errors=True)

    def _process_release_group(self, work: _ReleaseWork) -> None:
        candidate = work.candidate
        release_tag = candidate.release.tag
        assignments = work.assignments
        if not assignments:
            return

        self.logger.info(
            "Processing release %s for %d gap(s)",
            release_tag,
            len(assignments),
        )

        if self.dry_run:
            for assignment in assignments:
                self.logger.info(
                    "Dry run: would download %s and populate %s for gap %s -> %s",
                    release_tag,
                    assignment.session_dir,
                    assignment.gap.start.isoformat(),
                    assignment.gap.end.isoformat(),
                )
            return

        self._clear_other_archives(keep=release_tag)

        archive_dir = self.cache_paths.archives_dir / release_tag

        downloaded = download_assets(
            candidate.release,
            destination=self.cache_paths.archives_dir,
            skip_existing=True,
        )
        try:
            has_assets = archive_dir.exists() and any(archive_dir.iterdir())
        except OSError:
            has_assets = False

        if not downloaded and not has_assets:
            self.logger.warning("No assets downloaded for release %s", release_tag)
            return

        for assignment in assignments:
            self._populate_session_from_archive(
                assignment=assignment,
                candidate=candidate,
                archive_dir=archive_dir,
            )

        if not self.keep_archives:
            shutil.rmtree(archive_dir, ignore_errors=True)

    def _populate_session_from_archive(
        self,
        assignment: _ReleaseAssignment,
        candidate: ReleaseCandidate,
        archive_dir: Path,
    ) -> None:
        slug_config = assignment.slug_config
        gap = assignment.gap
        session_dir = assignment.session_dir

        self.logger.info(
            "Processing release %s for %s gap %s -> %s",
            candidate.release.tag,
            slug_config.slug,
            gap.start.isoformat(),
            gap.end.isoformat(),
        )

        if session_dir.exists():
            try:
                has_files = any(session_dir.iterdir())
            except OSError:
                has_files = True
            if has_files:
                self.logger.info("Session %s already populated; skipping", session_dir)
                return

        coords = list(slug_config.bounding_box.coordinates())
        extracted, missing = write_tiles_from_archive(
            archive_dir=archive_dir,
            session_dir=session_dir,
            coords=coords,
            release_tag=candidate.release.tag,
        )
        if extracted:
            self.logger.info(
                "Populated %s with %d tile(s) from %s",
                session_dir,
                extracted,
                candidate.release.tag,
            )
        if missing and not extracted:
            try:
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
            except OSError:
                pass

    def _repair_output_videos(self, slugs: Sequence[str]) -> None:
        if self.output_root is None or self._background_color_array is None:
            return
        if shutil.which("ffmpeg") is None:
            self.logger.debug("ffmpeg not available; skipping output repair")
            return

        for slug in slugs:
            output_dir = self.output_root / slug
            if not output_dir.exists():
                continue
            for video_path in sorted(output_dir.rglob("*.mp4")):
                if not video_path.is_file():
                    continue
                try:
                    needs_trim = self._first_frame_is_background(video_path)
                except Exception:  # pragma: no cover - defensive
                    self.logger.debug("Failed to inspect %s", video_path, exc_info=True)
                    continue
                if not needs_trim:
                    continue
                if self.dry_run:
                    self.logger.info("Dry run: would drop first frame from %s", video_path)
                    continue
                self.logger.info("Dropping solid background frame from %s", video_path)
                if not self._trim_first_frame(video_path):
                    self.logger.warning("Failed to drop first frame from %s", video_path)

    def _first_frame_is_background(self, video_path: Path) -> bool:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return False
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return False
        if self._background_color_array is None:
            return False
        background = self._background_color_array.astype(np.int16)
        diff = np.abs(frame.astype(np.int16) - background.reshape(1, 1, 3))
        max_diff = int(diff.max())
        return max_diff <= self._background_tolerance

    def _trim_first_frame(self, video_path: Path) -> bool:
        temp_path = video_path.with_name(f".tmp_trim_{uuid.uuid4().hex}{video_path.suffix}")
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        codec_args = [
            "-c:v",
            "libx264",
            "-crf",
            str(self.timelapse_quality),
            "-preset",
            "slow",
            "-tune",
            "animation",
            "-x264-params",
            "keyint=300:min-keyint=300:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "trim=start_frame=1,setpts=PTS-STARTPTS",
            "-an",
            *codec_args,
            str(temp_path),
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not temp_path.exists():
            self.logger.error(
                "ffmpeg failed to trim %s: %s",
                video_path,
                result.stderr.decode("utf-8", errors="ignore") if result.stderr else result.returncode,
            )
            temp_path.unlink(missing_ok=True)
            return False

        try:
            temp_path.replace(video_path)
        except OSError as exc:
            self.logger.error("Failed to replace %s with trimmed video: %s", video_path, exc)
            temp_path.unlink(missing_ok=True)
            return False

        return True


__all__ = [
    "BoundingBox",
    "CachePaths",
    "CleanupManager",
    "ConcatenatedReader",
    "GapInfo",
    "ReleaseCandidate",
    "SessionRecord",
    "ensure_cache_paths",
    "collect_sessions",
    "collect_slug_configs",
    "determine_timezone",
    "find_gaps",
    "load_config",
    "prune_empty_directories",
    "write_tiles_from_archive",
]
