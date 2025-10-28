import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timelapse_backup import TimelapseBackup
from timelapse_backup.full_timelapse import FullTimelapseSegment, FullTimelapseState


def make_session_dir(root: Path, slug: str, timestamp: datetime) -> Path:
    session_dir = root / slug / timestamp.strftime("%Y-%m-%d") / timestamp.strftime("%H-%M-%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def write_tile(path: Path, color: tuple[int, int, int, int]) -> None:
    tile = np.zeros((4, 4, 4), dtype=np.uint8)
    tile[..., 0] = color[0]
    tile[..., 1] = color[1]
    tile[..., 2] = color[2]
    tile[..., 3] = color[3]
    cv2.imwrite(str(path), tile)


def count_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = 0
        while True:
            success, _ = capture.read()
            if not success:
                break
            frames += 1
    finally:
        capture.release()
    return frames


def build_backup(tmp_path: Path) -> TimelapseBackup:
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.base_url = ""
    backup.background_color = (0, 0, 0)
    backup.auto_crop_transparent_frames = False
    backup.fps = 5
    backup.quality = 20
    backup.output_dir = tmp_path / "output"
    backup.output_dir.mkdir()
    backup.backup_dir = tmp_path / "backups"
    backup.backup_dir.mkdir()
    backup.logger = logging.getLogger("timelapse-tests")
    backup.frame_prep_workers = 1
    backup.request_delay = 0.0
    backup.diff_threshold = 10
    backup.diff_visualization = "colored"
    backup.diff_fade_frames = 3
    backup.diff_enhancement_factor = 2.0
    backup.reporting_enabled = False
    backup.seconds_per_pixel = 30
    backup.coverage_gap_multiplier = None
    return backup


def test_pending_sessions_excludes_previously_encoded(tmp_path):
    slug_dir = tmp_path / "output" / "slug"
    slug_dir.mkdir(parents=True)

    state = FullTimelapseState(
        slug_dir,
        "full.mp4",
        logger=logging.getLogger("timelapse-tests"),
    )

    start = datetime(2025, 1, 1, 0, 0, 0)
    middle = start + timedelta(minutes=5)
    later = middle + timedelta(minutes=5)

    state.segments = [
        FullTimelapseSegment(
            path="segments/full/full_20250101T000000_20250101T000500.mp4",
            first_session=start.isoformat(),
            last_session=middle.isoformat(),
            frame_count=10,
        )
    ]

    backup_root = tmp_path / "backups"
    session_a = make_session_dir(backup_root, "slug", start)
    session_b = make_session_dir(backup_root, "slug", middle)
    session_c = make_session_dir(backup_root, "slug", later)

    pending = state.pending_sessions([session_c, session_a, session_b])
    assert pending == [session_c]


def test_incremental_full_timelapse_appends_segments(tmp_path):
    backup = build_backup(tmp_path)
    slug = "canvas"
    timelapse_config = {
        "coordinates": {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0},
    }

    start = datetime(2025, 1, 1, 0, 0, 0)
    session_dirs = []
    colors = [
        (0, 0, 255, 255),
        (0, 255, 0, 255),
        (255, 0, 0, 255),
    ]

    for index in range(2):
        session_dir = make_session_dir(backup.backup_dir, slug, start + timedelta(minutes=5 * index))
        write_tile(session_dir / "0_0.png", colors[index])
        session_dirs.append(session_dir)

    backup.render_incremental_full_timelapse(
        slug=slug,
        name="Canvas",
        timelapse_config=timelapse_config,
        session_dirs=session_dirs,
        mode_name="normal",
        suffix="",
        output_filename="full.mp4",
        label="across all backups",
    )

    slug_dir = backup.output_dir / slug
    full_path = slug_dir / "full.mp4"
    manifest_path = slug_dir / "full.segments.json"

    assert full_path.exists()
    assert manifest_path.exists()
    assert count_frames(full_path) == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["segments"]) == 1
    assert manifest["segments"][0]["frame_count"] == 2

    third_session = make_session_dir(backup.backup_dir, slug, start + timedelta(minutes=10))
    write_tile(third_session / "0_0.png", colors[2])
    session_dirs.append(third_session)

    backup.render_incremental_full_timelapse(
        slug=slug,
        name="Canvas",
        timelapse_config=timelapse_config,
        session_dirs=session_dirs,
        mode_name="normal",
        suffix="",
        output_filename="full.mp4",
        label="across all backups",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["segments"]) == 2
    assert manifest["segments"][-1]["frame_count"] == 1
    assert manifest["segments"][-1]["last_session"] == (start + timedelta(minutes=10)).isoformat()

    assert count_frames(full_path) == 3

