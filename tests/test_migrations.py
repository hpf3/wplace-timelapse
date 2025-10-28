import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

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


def _write_dummy_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy mp4 data")


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
    for ts in (start, mid):
        _create_session(backup.backup_dir, slug, ts)

    legacy_full = slug_dir / "full.mp4"
    _write_dummy_video(legacy_full)

    migrate_full_timelapse_segments(backup)

    manifest_path = slug_dir / "full.segments.json"
    concat_path = slug_dir / "full.concat.txt"
    segment_dir = slug_dir / "segments" / "full"

    assert manifest_path.exists()
    assert concat_path.exists()
    assert list(segment_dir.glob("*.mp4"))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["segments"][0]["path"].startswith("segments/full/")
    assert manifest["segments"][0]["frame_count"] == 2


def test_migration_is_idempotent(tmp_path):
    backup = DummyBackup(tmp_path)
    slug = "canvas"
    slug_dir = backup.output_dir / slug
    slug_dir.mkdir()

    start = datetime(2025, 1, 1, 0, 0, 0)
    _create_session(backup.backup_dir, slug, start)
    legacy_full = slug_dir / "full.mp4"
    _write_dummy_video(legacy_full)

    migrate_full_timelapse_segments(backup)
    first_manifest = (slug_dir / "full.segments.json").read_text(encoding="utf-8")

    migrate_full_timelapse_segments(backup)
    second_manifest = (slug_dir / "full.segments.json").read_text(encoding="utf-8")

    assert first_manifest == second_manifest
