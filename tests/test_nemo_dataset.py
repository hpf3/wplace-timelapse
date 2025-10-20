import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import TimelapseBackup


def test_nemo_placeholder_session_builds_full_frame():
    backup = TimelapseBackup(config_file="config.json")

    slug = "nemo"
    session_dir = Path("backups") / slug / "2025-10-17" / "11-06-29"
    assert session_dir.exists(), "Expected copied nemo dataset to include 2025-10-17/11-06-29"

    timelapse_config = next(t for t in backup.config["timelapses"] if t["slug"] == slug)
    prior_sessions = backup.get_prior_sessions(slug, session_dir)

    coordinates = backup.get_tile_coordinates(timelapse_config)
    resolved_paths = []
    for x, y in coordinates:
        tile_path = backup.resolve_tile_image_path(session_dir, x, y, prior_sessions)
        assert tile_path is not None, f"Failed to resolve tile {x}_{y}"
        assert tile_path.exists(), f"Resolved tile for {x}_{y} does not exist: {tile_path}"
        resolved_paths.append(tile_path)

    composite = backup.create_composite_image(session_dir, timelapse_config)
    assert composite is not None

    height, width = composite.color.shape[:2]
    # 3x3 tiles of 1000px, so expect 3000x3000 output
    assert height == 3000
    assert width == 3000

    background = np.array(backup.background_color, dtype=np.uint8)
    diff_mask = np.any(composite.color != background, axis=2)
    assert np.any(diff_mask), "Composite fell back entirely to background color"
