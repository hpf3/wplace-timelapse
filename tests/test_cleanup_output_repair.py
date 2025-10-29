import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import timelapse_backup.cleanup as cleanup_module  # noqa: E402
from timelapse_backup.cleanup import (  # noqa: E402
    BoundingBox,
    CachePaths,
    CleanupManager,
    SlugConfig,
)


class _FakeCapture:
    def __init__(self, frame: np.ndarray, opened: bool = True) -> None:
        self._frame = frame
        self._opened = opened
        self._released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        return True, self._frame.copy()

    def release(self) -> None:
        self._released = True


def _make_manager(tmp_path: Path, *, dry_run: bool = False) -> CleanupManager:
    cache_root = tmp_path / "cache"
    archives_dir = cache_root / "archives"
    cache_paths = CachePaths(cache_root=cache_root, archives_dir=archives_dir)
    slug_configs = {"slug": SlugConfig("slug", "Slug", BoundingBox(0, 0, 0, 0))}
    manager = CleanupManager(
        backup_root=tmp_path / "backups",
        slug_configs=slug_configs,
        timezone_info=ZoneInfo("UTC"),
        interval=timedelta(minutes=5),
        min_gap=timedelta(hours=2),
        cache_paths=cache_paths,
        dry_run=dry_run,
        keep_archives=False,
        release_limit=5,
        output_root=tmp_path / "output",
        background_color=(190, 150, 37),
        timelapse_quality=23,
    )
    return manager


def test_first_frame_detection_background(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    frame = np.full((4, 4, 3), (190, 150, 37), dtype=np.uint8)
    monkeypatch.setattr(
        cleanup_module.cv2,
        "VideoCapture",
        lambda *_: _FakeCapture(frame),
    )
    result = manager._first_frame_is_background(tmp_path / "dummy.mp4")
    assert result is True


def test_repair_output_trims_first_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    video_dir = manager.output_root / "slug"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "sample.mp4"
    video_path.write_bytes(b"original")

    monkeypatch.setattr(
        manager,
        "_first_frame_is_background",
        lambda _: True,
    )
    monkeypatch.setattr(
        cleanup_module.shutil,
        "which",
        lambda _: "/usr/bin/ffmpeg",
    )

    spawned: list[list[str]] = []

    def fake_run(cmd, stdout=None, stderr=None):
        spawned.append(cmd)
        temp_path = Path(cmd[-1])
        temp_path.write_bytes(b"trimmed")

        class Result:
            returncode = 0
            stderr = b""

        return Result()

    monkeypatch.setattr(cleanup_module.subprocess, "run", fake_run)

    manager._repair_output_videos(["slug"])

    assert spawned, "ffmpeg should be invoked"
    assert video_path.read_bytes() == b"trimmed"


def test_repair_output_skips_non_background(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    video_dir = manager.output_root / "slug"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "sample.mp4"
    video_path.write_bytes(b"original")

    monkeypatch.setattr(
        manager,
        "_first_frame_is_background",
        lambda _: False,
    )
    monkeypatch.setattr(
        cleanup_module.shutil,
        "which",
        lambda _: "/usr/bin/ffmpeg",
    )

    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cleanup_module.subprocess, "run", fake_run)

    manager._repair_output_videos(["slug"])

    assert not called
    assert video_path.read_bytes() == b"original"
