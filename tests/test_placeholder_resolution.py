import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import TimelapseBackup


def test_placeholder_chain_falls_back_to_older_session(tmp_path):
    backup = TimelapseBackup.__new__(TimelapseBackup)
    backup.backup_dir = tmp_path / "backups"
    backup.backup_dir.mkdir()
    backup.logger = logging.getLogger("placeholder-tests")
    backup.background_color = (0, 0, 0)

    slug = "placeholder_slug"
    session_old = backup.backup_dir / slug / "2025-01-01" / "00-00-00"
    session_mid = backup.backup_dir / slug / "2025-01-02" / "00-00-00"
    session_new = backup.backup_dir / slug / "2025-01-03" / "00-00-00"

    for session in (session_old, session_mid, session_new):
        session.mkdir(parents=True, exist_ok=True)

    filename = "5_7.png"
    tile_path = session_old / filename
    tile = np.full((4, 4, 3), (17, 34, 51), dtype=np.uint8)
    cv2.imwrite(str(tile_path), tile)

    backup._write_placeholder(backup._placeholder_path(session_mid, filename), tile_path)
    backup._write_placeholder(backup._placeholder_path(session_new, filename), tile_path)

    coordinates = [(5, 7)]
    prior_sessions = backup.get_prior_sessions(slug, session_new)
    tile_map = backup.build_previous_tile_map(prior_sessions, coordinates)
    assert tile_map[filename] == tile_path

    resolved = backup.resolve_tile_image_path(session_new, 5, 7, prior_sessions)
    assert resolved == tile_path

    timelapse_config = {
        "slug": slug,
        "coordinates": {
            "xmin": 5,
            "xmax": 5,
            "ymin": 7,
            "ymax": 7,
        },
    }
    composite = backup.create_composite_image(session_new, timelapse_config)
    assert composite is not None
    unique_pixels = {tuple(map(int, pixel)) for pixel in composite.color.reshape(-1, 3)}
    assert unique_pixels == {(17, 34, 51)}
