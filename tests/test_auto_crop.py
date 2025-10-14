import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import TimelapseBackup


def build_backup(tmp_path: Path, auto_crop: bool = True) -> TimelapseBackup:
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.background_color = (0, 0, 0)
    backup.auto_crop_transparent_frames = auto_crop
    backup.fps = 5
    backup.output_dir = tmp_path / "output"
    backup.output_dir.mkdir()
    backup.logger = logging.getLogger("timelapse-tests")
    return backup


def write_tile(path: Path, color: tuple[int, int, int, int]) -> None:
    tile = np.zeros((4, 4, 4), dtype=np.uint8)
    tile[..., 0] = color[0]
    tile[..., 1] = color[1]
    tile[..., 2] = color[2]
    tile[..., 3] = color[3]
    cv2.imwrite(str(path), tile)


def prepare_sessions(tmp_path: Path, active_tiles: set[tuple[int, int]]) -> list[Path]:
    coordinates = range(3)
    sessions = []
    for idx in range(2):
        session_dir = tmp_path / f"session_{idx}"
        session_dir.mkdir()
        for x in coordinates:
            for y in coordinates:
                alpha = 255 if (x, y) in active_tiles else 0
                color = (0, 255, 0, alpha)
                write_tile(session_dir / f"{x}_{y}.png", color)
        sessions.append(session_dir)
    return sessions


def read_first_video_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        success, frame = capture.read()
    finally:
        capture.release()
    assert success, "Expected to read at least one frame from rendered video"
    return frame


def test_render_timelapse_crops_to_union_alpha(tmp_path):
    backup = build_backup(tmp_path)
    sessions = prepare_sessions(tmp_path, {(1, 1)})
    timelapse_config = {"coordinates": {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2}}

    backup.render_timelapse_from_sessions(
        slug="test",
        name="Crop Test",
        timelapse_config=timelapse_config,
        session_dirs=sessions,
        mode_name="normal",
        suffix="",
        output_filename="crop.mp4",
        label="unit test",
    )

    video_path = backup.output_dir / "test" / "crop.mp4"
    assert video_path.exists()

    frame = read_first_video_frame(video_path)
    assert frame.shape[0] == 4
    assert frame.shape[1] == 4


def test_render_timelapse_retains_size_when_transparent(tmp_path):
    backup = build_backup(tmp_path)
    sessions = prepare_sessions(tmp_path, set())
    timelapse_config = {"coordinates": {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 2}}

    backup.render_timelapse_from_sessions(
        slug="test2",
        name="No Crop",
        timelapse_config=timelapse_config,
        session_dirs=sessions,
        mode_name="normal",
        suffix="",
        output_filename="no_crop.mp4",
        label="unit test",
    )

    video_path = backup.output_dir / "test2" / "no_crop.mp4"
    assert video_path.exists()

    frame = read_first_video_frame(video_path)
    assert frame.shape[0] == 12
    assert frame.shape[1] == 12
