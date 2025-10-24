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


def _create_session(session_root: Path, timestamp: datetime, image: np.ndarray) -> Path:
    day_dir = session_root / timestamp.strftime("%Y-%m-%d")
    session_dir = day_dir / timestamp.strftime("%H-%M-%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    tile_path = session_dir / "0_0.png"
    cv2.imwrite(str(tile_path), image)
    return session_dir


def test_stats_report_counts_expected_pixels(tmp_path):
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.logger = logging.getLogger("stats-report-test")
    backup.logger.setLevel(logging.CRITICAL)
    backup.backup_dir = tmp_path / "backups"
    backup.backup_dir.mkdir()
    backup.output_dir = tmp_path / "output"
    backup.output_dir.mkdir()
    backup.background_color = (0, 0, 0)
    backup.auto_crop_transparent_frames = False
    backup.diff_visualization = "colored"
    backup.diff_threshold = 10
    backup.diff_enhancement_factor = 1.0
    backup.fps = 5
    backup.quality = 23
    backup.request_delay = 0.0
    backup.frame_prep_workers = 1
    backup.backup_interval = 5
    backup.reporting_enabled = True
    backup.seconds_per_pixel = 30
    backup.coverage_gap_multiplier = None
    backup.historical_cutoff = datetime(2025, 10, 13)

    slug = "stats_test"
    base_time = datetime(2025, 10, 13, 0, 0, 0)
    session_root = backup.backup_dir / slug

    height, width = 24, 24
    base_image = np.zeros((height, width, 3), dtype=np.uint8)

    rng = np.random.default_rng(42)
    positions = set()
    while len(positions) < 10:
        y = int(rng.integers(0, height))
        x = int(rng.integers(0, width))
        positions.add((y, x))
    positions = sorted(positions)

    sessions = []
    # Baseline frame before any changes.
    sessions.append(_create_session(session_root, base_time, base_image))

    for idx in range(10):
        change_time = base_time + timedelta(minutes=1 + idx * 2)
        color = (
            int((idx * 37) % 256),
            int((idx * 83 + 40) % 256),
            int((idx * 29 + 90) % 256),
        )
        changed_image = base_image.copy()
        for y, x in positions:
            changed_image[y, x] = color

        sessions.append(_create_session(session_root, change_time, changed_image))

        if idx < 9:
            placeholder_time = change_time + timedelta(minutes=1)
            sessions.append(_create_session(session_root, placeholder_time, changed_image))

    timelapse_config = {
        "slug": slug,
        "coordinates": {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0},
    }

    output_filename = "stats_test_diff.mp4"
    backup.render_timelapse_from_sessions(
        slug=slug,
        name="Stats Test",
        timelapse_config=timelapse_config,
        session_dirs=sessions,
        mode_name="diff",
        suffix="_diff",
        output_filename=output_filename,
        label="unit-test",
    )

    stats_path = backup.output_dir / slug / f"{output_filename}.stats.txt"
    assert stats_path.exists(), "Stats report was not generated"

    report = stats_path.read_text(encoding="utf-8")
    assert "Total changed pixels: 100" in report
    assert "Frames with change: 10" in report
    assert "Frames excluded from change stats" not in report
