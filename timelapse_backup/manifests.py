"""Manifest building utilities for the timelapse backup system."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from timelapse_backup.models import CompositeFrame, FrameManifest
from timelapse_backup.progress import eta_string
from timelapse_backup.tiles import TileDownloader


class ManifestBuilder:
    """Create frame manifests and composite images from tile sessions."""

    def __init__(
        self,
        tile_downloader: TileDownloader,
        *,
        background_color: Tuple[int, int, int],
        logger: logging.Logger,
    ) -> None:
        self.tile_downloader = tile_downloader
        self.background_color = background_color
        self.logger = logger

    # ------------------------------------------------------------------
    # Manifest construction
    # ------------------------------------------------------------------

    def build_manifest_for_session(
        self,
        session_dir: Path,
        coordinates: Sequence[Tuple[int, int]],
        *,
        backup_root: Optional[Path] = None,
        slug: Optional[str] = None,
        prior_sessions: Optional[Sequence[Path]] = None,
        tile_cache: Optional[Dict[Tuple[int, int], Path]] = None,
    ) -> Optional[FrameManifest]:
        manifest_tile_cache = tile_cache if tile_cache is not None else {}
        tile_paths: Dict[Tuple[int, int], Path] = {}

        for x, y in coordinates:
            tile_path = self.tile_downloader.resolve_tile_image_path(
                session_dir,
                x,
                y,
                prior_sessions=prior_sessions,
                backup_root=backup_root,
                slug=slug,
            )
            if tile_path is None and tile_cache is not None:
                cached = manifest_tile_cache.get((x, y))
                if cached and cached.exists():
                    tile_path = cached

            if tile_path is None or not tile_path.exists():
                continue

            if tile_cache is not None:
                manifest_tile_cache[(x, y)] = tile_path
            tile_paths[(x, y)] = tile_path

        if not tile_paths:
            return None

        return FrameManifest(session_dir=session_dir, tile_paths=tile_paths)

    def build_frame_manifests(
        self,
        session_dirs: Sequence[Path],
        coordinates: Sequence[Tuple[int, int]],
        *,
        backup_root: Optional[Path],
        slug: Optional[str],
        mode_name: str,
        label: str,
        timelapse_name: str,
    ) -> List[FrameManifest]:
        manifests: List[FrameManifest] = []
        manifest_tile_cache: Dict[Tuple[int, int], Path] = {}
        total_sessions = len(session_dirs)
        if total_sessions == 0:
            return manifests

        progress_interval = max(1, total_sessions // 20)
        usable_sessions = 0
        progress_start = perf_counter()

        for index, session_dir in enumerate(session_dirs, start=1):
            manifest = self.build_manifest_for_session(
                session_dir,
                coordinates,
                backup_root=backup_root,
                slug=slug,
                tile_cache=manifest_tile_cache,
            )
            if manifest is not None:
                manifests.append(manifest)
                usable_sessions += 1

            if index % progress_interval == 0 or index == total_sessions:
                percent = (index / total_sessions) * 100.0
                elapsed = perf_counter() - progress_start
                eta_text = eta_string(elapsed, index, total_sessions)
                self.logger.info(
                    "Frame preparation progress for '%s' (%s) %s %s: %s/%s sessions scanned, %s usable (%0.1f%%, %s)",
                    timelapse_name,
                    slug,
                    mode_name,
                    label,
                    index,
                    total_sessions,
                    usable_sessions,
                    percent,
                    eta_text,
                )

        return manifests

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def compose_frame(
        self,
        manifest: FrameManifest,
        coordinates: Sequence[Tuple[int, int]],
    ) -> Optional[CompositeFrame]:
        x_coords = sorted({x for x, _ in coordinates})
        y_coords = sorted({y for _, y in coordinates})
        x_index_map = {value: idx for idx, value in enumerate(x_coords)}
        y_index_map = {value: idx for idx, value in enumerate(y_coords)}

        first_tile: Optional[np.ndarray] = None
        for x, y in coordinates:
            tile_path = manifest.tile_paths.get((x, y))
            if not tile_path or not tile_path.exists():
                continue
            tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
            if tile is None:
                continue
            tile = self._ensure_bgra(tile)
            if tile is not None:
                first_tile = tile
                break

        if first_tile is None:
            return None

        tile_height, tile_width = first_tile.shape[:2]
        composite_height = len(y_coords) * tile_height
        composite_width = len(x_coords) * tile_width
        composite = np.full(
            (composite_height, composite_width, 3),
            self.background_color,
            dtype=np.uint8,
        )
        alpha_mask = np.zeros((composite_height, composite_width), dtype=np.uint8)

        for x, y in coordinates:
            tile_path = manifest.tile_paths.get((x, y))
            if not tile_path or not tile_path.exists():
                continue
            tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
            tile = self._ensure_bgra(tile)
            if tile is None:
                continue

            x_idx = x_index_map.get(x)
            y_idx = y_index_map.get(y)
            if x_idx is None or y_idx is None:
                continue

            start_y = y_idx * tile_height
            end_y = start_y + tile_height
            start_x = x_idx * tile_width
            end_x = start_x + tile_width

            region = composite[start_y:end_y, start_x:end_x].astype(np.float32)
            region_alpha = alpha_mask[start_y:end_y, start_x:end_x].astype(np.float32) / 255.0

            tile_rgb = tile[:, :, :3].astype(np.float32)
            tile_alpha = tile[:, :, 3].astype(np.float32) / 255.0

            inverse_tile_alpha = 1.0 - tile_alpha
            out_alpha = tile_alpha + region_alpha * inverse_tile_alpha

            combined_color = (
                tile_rgb * tile_alpha[..., None]
                + region * (region_alpha * inverse_tile_alpha)[..., None]
            )

            divisor = np.maximum(out_alpha[..., None], 1e-6)
            out_color = combined_color / divisor

            zero_alpha_mask = out_alpha <= 0
            if np.any(zero_alpha_mask):
                out_color[zero_alpha_mask] = self.background_color

            composite[start_y:end_y, start_x:end_x] = np.clip(out_color, 0, 255).astype(np.uint8)
            alpha_mask[start_y:end_y, start_x:end_x] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8)

        return CompositeFrame(color=composite, alpha=alpha_mask)

    def create_composite_image(
        self,
        session_dir: Path,
        coordinates: Sequence[Tuple[int, int]],
        *,
        backup_root: Optional[Path] = None,
        slug: Optional[str] = None,
        prior_sessions: Optional[Sequence[Path]] = None,
        tile_cache: Optional[Dict[Tuple[int, int], Path]] = None,
    ) -> Optional[CompositeFrame]:
        manifest = self.build_manifest_for_session(
            session_dir,
            coordinates,
            backup_root=backup_root,
            slug=slug,
            prior_sessions=prior_sessions,
            tile_cache=tile_cache,
        )
        if manifest is None:
            return None
        return self.compose_frame(manifest, coordinates)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_bgra(tile: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if tile is None:
            return None
        if len(tile.shape) == 2:
            return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
        if tile.shape[2] == 1:
            return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
        if tile.shape[2] == 3:
            opaque_alpha = np.full((tile.shape[0], tile.shape[1], 1), 255, dtype=tile.dtype)
            return np.concatenate((tile, opaque_alpha), axis=2)
        if tile.shape[2] != 4:
            return None
        return tile


__all__ = ["ManifestBuilder"]
