import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timelapse_backup import TimelapseBackup


def test_tile_cache_reuses_last_known_tile(tmp_path):
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.background_color = (0, 0, 0)
    backup.auto_crop_transparent_frames = False
    backup.logger = logging.getLogger("tile-cache-test")
    backup.backup_dir = tmp_path / "backups"
    backup.backup_dir.mkdir()

    slug = "cache_slug"
    day_dir = backup.backup_dir / slug / "2025-01-01"
    session_a = day_dir / "00-00-00"
    session_b = day_dir / "00-05-00"
    session_a.mkdir(parents=True)
    session_b.mkdir(parents=True)

    coords = {"xmin": 10, "xmax": 10, "ymin": 20, "ymax": 20}
    timelapse_config = {"coordinates": coords, "slug": slug}

    filename = "10_20.png"
    tile_path = session_a / filename
    tile = np.full((4, 4, 3), (12, 34, 56), dtype=np.uint8)
    cv2.imwrite(str(tile_path), tile)

    # First composite builds cache entry.
    tile_cache = {}
    composite_a = backup.create_composite_image(session_a, timelapse_config, tile_cache)
    assert composite_a is not None

    # Second session intentionally missing the tile; composite should reuse cached path.
    composite_b = backup.create_composite_image(session_b, timelapse_config, tile_cache)
    assert composite_b is not None
    unique_pixels = {tuple(map(int, pixel)) for pixel in composite_b.color.reshape(-1, 3)}
    assert unique_pixels == {(12, 34, 56)}
