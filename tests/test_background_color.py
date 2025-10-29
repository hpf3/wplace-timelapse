import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timelapse_backup import TimelapseBackup
from timelapse_backup.models import PreparedFrame
from timelapse_backup.rendering import Renderer


def make_backup(**kwargs) -> TimelapseBackup:
    """Helper to build a minimal TimelapseBackup with configurable attributes."""
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.background_color = kwargs.get("background_color", (0, 0, 0))
    backup.diff_visualization = kwargs.get("diff_visualization", "colored")
    backup.diff_threshold = kwargs.get("diff_threshold", 10)
    backup.diff_enhancement_factor = kwargs.get("diff_enhancement_factor", 1.0)
    backup.auto_crop_transparent_frames = kwargs.get("auto_crop_transparent_frames", True)
    return backup


def test_parse_background_color_rgb_array_to_bgr():
    backup = TimelapseBackup.__new__(TimelapseBackup)
    result = TimelapseBackup._parse_background_color(backup, [37, 150, 190])
    assert result == (190, 150, 37)


def test_parse_background_color_hex_string():
    backup = TimelapseBackup.__new__(TimelapseBackup)
    result = TimelapseBackup._parse_background_color(backup, "#2596BE")
    assert result == (190, 150, 37)


def test_parse_background_color_explicit_bgr_to_bgr():
    backup = TimelapseBackup.__new__(TimelapseBackup)
    result = TimelapseBackup._parse_background_color(
        backup,
        {"value": [190, 150, 37], "order": "bgr"},
    )
    assert result == (190, 150, 37)


def test_differential_first_frame_uses_background_color():
    backing_color = (190, 150, 37)
    backup = make_backup(background_color=backing_color)
    current = np.zeros((2, 2, 3), dtype=np.uint8)

    first_frame = backup.create_differential_frame(None, current)

    unique_pixels = {tuple(map(int, pixel)) for pixel in first_frame.reshape(-1, 3)}
    assert unique_pixels == {backing_color}


def test_differential_colored_highlights_on_background():
    backing_color = (190, 150, 37)
    backup = make_backup(background_color=backing_color, diff_visualization="colored")
    previous = np.zeros((2, 2, 3), dtype=np.uint8)
    current = np.zeros((2, 2, 3), dtype=np.uint8)
    current[0, 0] = [255, 255, 255]

    diff_frame = backup.create_differential_frame(previous, current)

    expected_pixels = {backing_color, (0, 255, 0)}
    unique_pixels = {tuple(map(int, pixel)) for pixel in diff_frame.reshape(-1, 3)}
    assert unique_pixels == expected_pixels


def test_composite_preserves_background_for_transparent_tiles(tmp_path):
    backing_color = (190, 150, 37)
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.background_color = backing_color

    session_dir = tmp_path / "session"
    session_dir.mkdir()

    coordinates = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0}
    timelapse_config = {"coordinates": coordinates}

    tile_path = session_dir / "0_0.png"
    tile_rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    tile_rgba[..., :3] = [255, 0, 0]
    tile_rgba[..., 3] = 0  # Fully transparent
    cv2.imwrite(str(tile_path), tile_rgba)

    composite = backup.create_composite_image(session_dir, timelapse_config)

    assert composite is not None
    unique_pixels = {tuple(map(int, pixel)) for pixel in composite.color.reshape(-1, 3)}
    assert unique_pixels == {backing_color}
    assert np.count_nonzero(composite.alpha) == 0


def test_diff_generator_skips_solid_background_frame(tmp_path):
    backing_color = (190, 150, 37)
    first_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    second_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    second_frame[0, 0] = [255, 255, 255]

    first_path = tmp_path / "frame_000000.png"
    second_path = tmp_path / "frame_000001.png"
    cv2.imwrite(str(first_path), first_frame)
    cv2.imwrite(str(second_path), second_frame)

    manifest_builder = types.SimpleNamespace(background_color=backing_color)
    diff_settings = types.SimpleNamespace(
        threshold=10,
        visualization="overlay",
        enhancement_factor=1.0,
    )
    renderer = Renderer(
        manifest_builder,
        logger=logging.getLogger("timelapse-test"),
        frame_prep_workers=1,
        auto_crop_transparent_frames=True,
        diff_settings=diff_settings,
    )

    prepared_frames = [
        PreparedFrame(index=0, session_dir=tmp_path, temp_path=first_path, frame_shape=(4, 4), alpha_bounds=None),
        PreparedFrame(index=1, session_dir=tmp_path, temp_path=second_path, frame_shape=(4, 4), alpha_bounds=None),
    ]
    frame_iter, stats = renderer.frame_byte_generator(
        prepared_frames,
        "diff",
        "slug",
        "name",
        "label",
        [None, None],
    )

    encoded = list(frame_iter)

    assert len(encoded) == 1
    assert len(stats.records) == 1
