"""Data models used across the timelapse backup system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CompositeFrame:
    """Container for a rendered composite frame and its transparency mask."""

    color: np.ndarray
    alpha: np.ndarray


@dataclass
class FrameManifest:
    """Resolved tile paths for a single frame."""

    session_dir: Path
    tile_paths: Dict[Tuple[int, int], Path]


@dataclass
class PreparedFrame:
    """Prepared frame data persisted to temporary storage."""

    index: int
    session_dir: Path
    temp_path: Optional[Path]
    frame_shape: Optional[Tuple[int, int]]
    alpha_bounds: Optional[Tuple[int, int, int, int]]


@dataclass
class RenderedTimelapseResult:
    """Summary of a rendered timelapse clip."""

    output_path: Path
    frame_count: int
    session_dirs: Tuple[Path, ...]
    video_width: Optional[int]
    video_height: Optional[int]
    content_width: Optional[int]
    content_height: Optional[int]
    crop_x: Optional[int]
    crop_y: Optional[int]


__all__ = [
    "CompositeFrame",
    "FrameManifest",
    "PreparedFrame",
    "RenderedTimelapseResult",
]
