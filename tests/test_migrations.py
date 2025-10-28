import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timelapse_backup.full_timelapse import FullTimelapseState
from timelapse_backup.migrations import migrate_full_timelapse_segments


class DummyBackup:
    def __init__(self, root: Path) -> None:
        self.output_dir = root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = root / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("timelapse-tests")
        self._timelapses = [
            {
                "slug": "canvas",
                "name": "Canvas",
                "coordinates": {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0},
            }
        ]

    def get_enabled_timelapses(self):
        return self._timelapses

    def get_enabled_timelapse_modes(self, _timelapse):
        return [{"mode": "normal", "suffix": "", "create_full": True}]


def _write_dummy_video(path: Path, *, width: int = 4, height: int = 4, frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (width, height))
    assert writer.isOpened(), "Failed to create dummy video"
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _create_session(root: Path, slug: str, dt: datetime) -> Path:
    session = root / slug / dt.strftime("%Y-%m-%d") / dt.strftime("%H-%M-%S")
    session.mkdir(parents=True, exist_ok=True)
    return session


def test_migration_creates_manifest_and_segment(tmp_path):
    backup = DummyBackup(tmp_path)
    slug = "canvas"
    slug_dir = backup.output_dir / slug
    slug_dir.mkdir()

    start = datetime(2025, 1, 1, 0, 0, 0)
    mid = start + timedelta(minutes=5)
    newer = mid + timedelta(minutes=5)
    session_dirs = []
    for ts in (start, mid, newer):
        session_dirs.append(_create_session(backup.backup_dir, slug, ts))

    legacy_full = slug_dir / "full.mp4"
    _write_dummy_video(legacy_full, width=8, height=6, frames=42)

    stats_path = legacy_full.with_suffix(legacy_full.suffix + ".stats.txt")
    stats_path.write_text(
        "\n".join(
            [
                "Timelapse Report",
                "Frame Overview",
                "Frames rendered: 42",
                "Timeline",
                f"First frame: {start.isoformat()}",
                f"Last frame: {mid.isoformat()}",
            ]
        ),
        encoding="utf-8",
    )

    migrate_full_timelapse_segments(backup)

    manifest_path = slug_dir / "full.segments.json"
    concat_path = slug_dir / "full.concat.txt"
    segment_dir = slug_dir / "segments" / "full"

    assert manifest_path.exists()
    assert concat_path.exists()
    assert list(segment_dir.glob("*.mp4"))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["segments"][0]["path"].startswith("segments/full/")
    assert manifest["segments"][0]["frame_count"] == 42
    assert manifest["segments"][0]["video_width"] == 8
    assert manifest["segments"][0]["video_height"] == 6
    assert manifest["segments"][0]["content_width"] == 8
    assert manifest["segments"][0]["content_height"] == 6
    assert manifest["segments"][0]["crop_x"] == 0
    assert manifest["segments"][0]["crop_y"] == 0
    assert manifest["segments"][0]["pad_left"] == 0
    assert manifest["segments"][0]["pad_top"] == 0
    assert manifest["segments"][0]["first_session"] == start.isoformat()
    assert manifest["segments"][0]["last_session"] == mid.isoformat()

    state = FullTimelapseState(slug_dir, "full.mp4", logger=logging.getLogger("timelapse-tests"))
    state.load()
    pending = state.pending_sessions(session_dirs)
    assert pending == [session_dirs[-1]]


def test_migration_is_idempotent(tmp_path):
    backup = DummyBackup(tmp_path)
    slug = "canvas"
    slug_dir = backup.output_dir / slug
    slug_dir.mkdir()

    start = datetime(2025, 1, 1, 0, 0, 0)
    _create_session(backup.backup_dir, slug, start)
    legacy_full = slug_dir / "full.mp4"
    _write_dummy_video(legacy_full, width=4, height=4, frames=10)

    stats_path = legacy_full.with_suffix(legacy_full.suffix + ".stats.txt")
    stats_path.write_text(
        "\n".join(
            [
                "Frame Overview",
                "Frames rendered: 10",
                "Timeline",
                f"First frame: {start.isoformat()}",
                f"Last frame: {start.isoformat()}",
            ]
        ),
        encoding="utf-8",
    )

    migrate_full_timelapse_segments(backup)
    first_manifest = (slug_dir / "full.segments.json").read_text(encoding="utf-8")

    migrate_full_timelapse_segments(backup)
    second_manifest = (slug_dir / "full.segments.json").read_text(encoding="utf-8")

    assert first_manifest == second_manifest
