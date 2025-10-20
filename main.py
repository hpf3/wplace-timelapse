#!/usr/bin/env python3
"""
Timelapse Backup System for WPlace Tiles
Automatically downloads tile images every 5 minutes and creates daily timelapses.
"""

import os
import requests
import logging
import cv2
import numpy as np
import time
import json
import subprocess
import shutil
import uuid
import threading
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Load environment variables
load_dotenv()


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

class TimelapseBackup:
    PLACEHOLDER_SUFFIX = '.placeholder'

    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_configuration()
        
        # Setup logging
        self.setup_logging()
        
        # Create directories
        self.backup_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from JSON file or fall back to .env"""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Set properties from config
            settings = config['global_settings']
            self.base_url = settings.get('base_url', 'https://backend.wplace.live/files/s0/tiles')
            self.backup_interval = settings.get('backup_interval_minutes', 5)
            self.backup_dir = Path(settings.get('backup_dir', 'backups'))
            self.output_dir = Path(settings.get('output_dir', 'output'))
            self.request_delay = settings.get('request_delay', 0.5)
            self.fps = settings.get('timelapse_fps', 10)
            self.quality = settings.get('timelapse_quality', 23)
            self.background_color = self._parse_background_color(settings.get('background_color', [0, 0, 0]))
            self.auto_crop_transparent_frames = self._parse_bool(
                settings.get('auto_crop_transparent_frames', True),
                True
            )
            default_workers = max(1, min(4, os.cpu_count() or 1))
            self.frame_prep_workers = self._parse_positive_int(
                settings.get('frame_prep_workers', default_workers),
                default_workers
            )
            
            # Differential settings
            diff_settings = settings.get('diff_settings', {})
            self.diff_threshold = diff_settings.get('threshold', 10)
            self.diff_visualization = diff_settings.get('visualization', 'colored')
            self.diff_fade_frames = diff_settings.get('fade_frames', 3)
            self.diff_enhancement_factor = diff_settings.get('enhancement_factor', 2.0)
            
            return config
        else:
            # Fall back to .env configuration (backward compatibility)
            self.base_url = os.getenv('BASE_URL', 'https://backend.wplace.live/files/s0/tiles')
            self.backup_interval = int(os.getenv('BACKUP_INTERVAL_MINUTES', 5))
            self.backup_dir = Path(os.getenv('BACKUP_DIR', 'backups'))
            self.output_dir = Path(os.getenv('TIMELAPSE_DIR', 'timelapses'))
            self.request_delay = float(os.getenv('REQUEST_DELAY', 0.5))
            self.fps = int(os.getenv('TIMELAPSE_FPS', 10))
            self.quality = int(os.getenv('TIMELAPSE_QUALITY', 23))
            self.background_color = (0, 0, 0)
            self.auto_crop_transparent_frames = self._parse_bool(
                os.getenv('AUTO_CROP_TRANSPARENT_FRAMES', 'true'),
                True
            )
            default_workers = max(1, min(4, os.cpu_count() or 1))
            self.frame_prep_workers = self._parse_positive_int(
                os.getenv('FRAME_PREP_WORKERS'),
                default_workers
            )
            
            # Differential settings (defaults)
            self.diff_threshold = 10
            self.diff_visualization = 'colored'
            self.diff_fade_frames = 3
            self.diff_enhancement_factor = 2.0
            
            # Create legacy single timelapse config
            return {
                'timelapses': [{
                    'slug': 'default',
                    'name': 'Default Timelapse',
                    'coordinates': {
                        'xmin': int(os.getenv('XMIN', 1031)),
                        'xmax': int(os.getenv('XMAX', 1032)),
                        'ymin': int(os.getenv('YMIN', 747)),
                        'ymax': int(os.getenv('YMAX', 748))
                    },
                    'enabled': True
                }],
                'global_settings': {
                    'base_url': self.base_url,
                    'backup_interval_minutes': self.backup_interval,
                    'backup_dir': str(self.backup_dir),
                    'output_dir': str(self.output_dir),
                    'request_delay': self.request_delay,
                    'timelapse_fps': self.fps,
                    'timelapse_quality': self.quality,
                    'background_color': list(self.background_color),
                    'auto_crop_transparent_frames': self.auto_crop_transparent_frames
                }
            }

    def _parse_background_color(self, value: Any) -> Tuple[int, int, int]:
        """Parse and clamp background color definition"""
        default = (0, 0, 0)

        def _clamp_triplet(triplet: Any) -> Optional[Tuple[int, int, int]]:
            if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                return None
            try:
                return tuple(
                    max(0, min(255, int(channel)))
                    for channel in triplet
                )
            except (TypeError, ValueError):
                return None

        def _to_bgr(channels: Tuple[int, int, int], order: Optional[str]) -> Optional[Tuple[int, int, int]]:
            color_order = (order or 'rgb').lower()
            if color_order == 'bgr':
                return channels
            if color_order == 'rgb':
                return (channels[2], channels[1], channels[0])
            return None

        if isinstance(value, dict):
            if 'hex' in value and isinstance(value['hex'], str):
                return self._parse_background_color(value['hex'])
            if 'value' in value:
                channels = _clamp_triplet(value['value'])
                if channels is None:
                    return default
                bgr = _to_bgr(channels, value.get('order') or value.get('color_space'))
                if bgr is not None:
                    return bgr
                return default

        if isinstance(value, (list, tuple)):
            channels = _clamp_triplet(value)
            if channels is None:
                return default
            bgr = _to_bgr(channels, 'rgb')
            if bgr is not None:
                return bgr
            return default

        if isinstance(value, str):
            hex_value = value.lstrip('#')
            if len(hex_value) == 6:
                try:
                    r = int(hex_value[0:2], 16)
                    g = int(hex_value[2:4], 16)
                    b = int(hex_value[4:6], 16)
                    # Convert from RGB string to BGR tuple used by OpenCV
                    return (b, g, r)
                except ValueError:
                    return default

        return default

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        """Parse truthy/falsy values from multiple input types."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}
        if isinstance(value, (int, float)):
            return value != 0
        return default

    @staticmethod
    def _parse_positive_int(value: Any, default: int) -> int:
        """Parse a positive integer with fallback to default."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
    
    def get_enabled_timelapses(self) -> List[Dict[str, Any]]:
        """Get list of enabled timelapse configurations"""
        return [tl for tl in self.config['timelapses'] if tl.get('enabled', True)]
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('timelapse_backup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_tile_coordinates(self, timelapse_config: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Generate list of tile coordinates to download for a specific timelapse"""
        coords = timelapse_config['coordinates']
        coordinates = []
        for x in range(coords['xmin'], coords['xmax'] + 1):
            for y in range(coords['ymin'], coords['ymax'] + 1):
                coordinates.append((x, y))
        return coordinates
    
    def _placeholder_filename(self, filename: str) -> str:
        """Derive placeholder filename for a tile"""
        return f"{Path(filename).stem}{self.PLACEHOLDER_SUFFIX}"

    def _placeholder_path(self, session_dir: Path, filename: str) -> Path:
        """Return path to placeholder file for given tile filename"""
        return session_dir / self._placeholder_filename(filename)

    def _write_placeholder(self, placeholder_path: Path, target_path: Path) -> bool:
        """Create placeholder metadata marking tile reuse without hardcoding a file path."""
        if not target_path.exists():
            self.logger.warning(f"Creating placeholder for missing target tile: {target_path}")
        data = {
            "type": "placeholder",
            "version": 2,
            "created_at": datetime.utcnow().isoformat()
        }
        try:
            with open(placeholder_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return True
        except OSError as exc:
            self.logger.error(f"Failed to write placeholder {placeholder_path}: {exc}")
            return False

    def _find_tile_in_sessions(
        self,
        sessions: Iterable[Path],
        filename: str
    ) -> Optional[Path]:
        """Search older sessions for the first concrete PNG for a tile."""
        for session in sessions:
            candidate = session / filename
            if candidate.exists():
                return candidate
        return None

    def _extract_slug_from_session(self, session_dir: Path) -> Optional[str]:
        """Derive slug name from a session directory path relative to the backup root."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None:
            return None
        backup_root = Path(backup_root)

        try:
            relative = session_dir.relative_to(backup_root)
        except ValueError:
            return None

        parts = relative.parts
        if not parts:
            return None
        return parts[0]

    def _parse_session_datetime(self, session_dir: Path) -> Optional[datetime]:
        """Parse datetime from session directory path"""
        try:
            return datetime.strptime(
                f"{session_dir.parent.name} {session_dir.name}",
                "%Y-%m-%d %H-%M-%S"
            )
        except ValueError:
            return None

    def get_prior_sessions(self, slug: str, current_session: Path) -> List[Path]:
        """Get list of prior session directories ordered from newest to oldest"""
        slug_dir = self.backup_dir / slug
        if not slug_dir.exists():
            return []

        current_dt = self._parse_session_datetime(current_session)
        sessions: List[Tuple[datetime, Path]] = []

        for date_dir in slug_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for time_dir in date_dir.iterdir():
                if not time_dir.is_dir() or time_dir == current_session:
                    continue
                session_dt = self._parse_session_datetime(time_dir)
                if session_dt is None:
                    continue
                if current_dt is not None and session_dt >= current_dt:
                    continue
                sessions.append((session_dt, time_dir))

        sessions.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in sessions]

    def build_previous_tile_map(
        self,
        prior_sessions: List[Path],
        coordinates: List[Tuple[int, int]]
    ) -> Dict[str, Path]:
        """
        Build map of tile filename to the most recent available PNG path from prior sessions.
        """
        filenames = {f"{x}_{y}.png" for x, y in coordinates}
        result: Dict[str, Path] = {}

        for index, session_dir in enumerate(prior_sessions):
            remaining = filenames.difference(result.keys())
            if not remaining:
                break

            for filename in list(remaining):
                direct_path = session_dir / filename
                if direct_path.exists():
                    result[filename] = direct_path
                    continue

                placeholder_path = self._placeholder_path(session_dir, filename)
                if placeholder_path.exists():
                    target_path = self._find_tile_in_sessions(
                        prior_sessions[index + 1 :],
                        filename
                    )
                    if target_path is not None:
                        result[filename] = target_path

        return result

    def resolve_tile_image_path(
        self,
        session_dir: Path,
        x: int,
        y: int,
        prior_sessions: Optional[List[Path]] = None
    ) -> Optional[Path]:
        """Resolve actual image path for a tile, following placeholders if needed"""
        filename = f"{x}_{y}.png"
        candidate = session_dir / filename
        if candidate.exists():
            return candidate

        placeholder_path = self._placeholder_path(session_dir, filename)
        if placeholder_path.exists():
            if prior_sessions is not None:
                sessions_to_search = prior_sessions
            else:
                slug = self._extract_slug_from_session(session_dir)
                sessions_to_search = self.get_prior_sessions(slug, session_dir) if slug else []

            target_path = self._find_tile_in_sessions(sessions_to_search, filename)
            if target_path is not None:
                return target_path

        return None
        
    def download_tile(
        self,
        slug: str,
        x: int,
        y: int,
        session_dir: Path,
        previous_tile_map: Dict[str, Path]
    ) -> Tuple[bool, bool]:
        """Download a single tile image and skip duplicates via placeholders.

        Returns:
            Tuple[bool, bool]: (success flag, placeholder created flag)
        """
        try:
            url = f"{self.base_url}/{x}/{y}.png"
            filename = f"{x}_{y}.png"
            filepath = session_dir / filename

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            content = response.content
            previous_path = previous_tile_map.get(filename)

            if previous_path and previous_path.exists():
                try:
                    if previous_path.read_bytes() == content:
                        placeholder_path = self._placeholder_path(session_dir, filename)
                        if self._write_placeholder(placeholder_path, previous_path):
                            self.logger.debug(
                                f"Skipped duplicate tile {slug} {x},{y}; placeholder -> {previous_path}"
                            )
                            return True, True
                        else:
                            self.logger.debug(
                                f"Placeholder creation failed for {slug} {x},{y}; saving tile to disk"
                            )
                except OSError as exc:
                    self.logger.warning(f"Failed to read previous tile {previous_path}: {exc}")

            with open(filepath, 'wb') as f:
                f.write(content)

            # Ensure no stale placeholder remains
            placeholder_path = self._placeholder_path(session_dir, filename)
            if placeholder_path.exists():
                placeholder_path.unlink(missing_ok=True)

            self.logger.debug(f"Downloaded tile {slug} {x},{y} to {filepath}")
            return True, False

        except Exception as e:
            self.logger.error(f"Failed to download tile {x},{y}: {e}")
            return False, False
            
    def backup_tiles(self):
        """Backup all tiles for current timestamp for all enabled timelapses"""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        
        enabled_timelapses = self.get_enabled_timelapses()
        self.logger.info(f"Starting backup session at {time_str} for {len(enabled_timelapses)} timelapses")
        
        total_successful = 0
        total_tiles = 0
        total_duplicates = 0
        
        for timelapse in enabled_timelapses:
            slug = timelapse['slug']
            name = timelapse['name']
            
            # Create session directory for this timelapse
            session_dir = self.backup_dir / slug / date_str / time_str
            session_dir.mkdir(parents=True, exist_ok=True)
            
            coordinates = self.get_tile_coordinates(timelapse)
            prior_sessions = self.get_prior_sessions(slug, session_dir)
            previous_tile_map = self.build_previous_tile_map(prior_sessions, coordinates)
            successful_downloads = 0
            duplicate_tiles = 0

            self.logger.info(f"Backing up '{name}' ({slug}): {len(coordinates)} tiles")
            
            for i, (x, y) in enumerate(coordinates):
                success, used_placeholder = self.download_tile(
                    slug,
                    x,
                    y,
                    session_dir,
                    previous_tile_map
                )
                if success:
                    successful_downloads += 1
                    if used_placeholder:
                        duplicate_tiles += 1
                
                # Add delay between requests (except for the last one)
                if i < len(coordinates) - 1:
                    time.sleep(self.request_delay)
                    
            if duplicate_tiles:
                self.logger.info(
                    f"'{name}' completed: {successful_downloads}/{len(coordinates)} tiles "
                    f"({duplicate_tiles} duplicates skipped)"
                )
            else:
                self.logger.info(f"'{name}' completed: {successful_downloads}/{len(coordinates)} tiles")
            
            total_successful += successful_downloads
            total_tiles += len(coordinates)
            total_duplicates += duplicate_tiles
            
            # Remove empty session directory if no downloads succeeded
            if successful_downloads == 0:
                try:
                    session_dir.rmdir()
                    # Try to remove parent directories if empty
                    if not any(session_dir.parent.iterdir()):
                        session_dir.parent.rmdir()
                        if not any(session_dir.parent.parent.iterdir()):
                            session_dir.parent.parent.rmdir()
                except:
                    pass
                    
        if total_duplicates:
            self.logger.info(
                f"Backup session completed: {total_successful}/{total_tiles} total tiles processed "
                f"({total_duplicates} duplicates skipped)"
            )
        else:
            self.logger.info(
                f"Backup session completed: {total_successful}/{total_tiles} total tiles downloaded"
            )
                
    def _build_manifest_for_session(
        self,
        session_dir: Path,
        timelapse_config: Dict[str, Any],
        coordinates: List[Tuple[int, int]],
        tile_cache: Optional[Dict[Tuple[int, int], Path]],
        prior_sessions: Optional[List[Path]] = None,
    ) -> Optional[FrameManifest]:
        """Resolve tile paths for a session using existing fallback logic."""
        slug = timelapse_config.get('slug')
        if not slug:
            slug = self._extract_slug_from_session(session_dir)

        if prior_sessions is None:
            prior_sessions = self.get_prior_sessions(slug, session_dir) if slug else []

        manifest_tile_cache = tile_cache if tile_cache is not None else {}
        tile_paths: Dict[Tuple[int, int], Path] = {}

        for x, y in coordinates:
            tile_path = self.resolve_tile_image_path(session_dir, x, y, prior_sessions)
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

    def _compose_frame_from_manifest(
        self,
        manifest: FrameManifest,
        coordinates: List[Tuple[int, int]],
        x_coords: List[int],
        y_coords: List[int],
        x_index_map: Dict[int, int],
        y_index_map: Dict[int, int],
    ) -> Optional[CompositeFrame]:
        """Compose a frame using tile paths resolved in a manifest."""
        first_tile: Optional[np.ndarray] = None
        for x, y in coordinates:
            tile_path = manifest.tile_paths.get((x, y))
            if not tile_path or not tile_path.exists():
                continue
            tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
            if tile is None:
                continue
            if len(tile.shape) == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            elif tile.shape[2] == 1:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            elif tile.shape[2] == 3:
                opaque_alpha = np.full((tile.shape[0], tile.shape[1], 1), 255, dtype=tile.dtype)
                tile = np.concatenate((tile, opaque_alpha), axis=2)
            elif tile.shape[2] != 4:
                continue
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
            if tile is None:
                continue

            if len(tile.shape) == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            elif tile.shape[2] == 1:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            elif tile.shape[2] == 3:
                opaque_alpha = np.full((tile.shape[0], tile.shape[1], 1), 255, dtype=tile.dtype)
                tile = np.concatenate((tile, opaque_alpha), axis=2)
            elif tile.shape[2] != 4:
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
        timelapse_config: Dict[str, Any],
        tile_cache: Optional[Dict[Tuple[int, int], Path]] = None
    ) -> Optional[CompositeFrame]:
        """Create a composite image from individual tiles and retain transparency."""
        coordinates = self.get_tile_coordinates(timelapse_config)

        manifest = self._build_manifest_for_session(
            session_dir,
            timelapse_config,
            coordinates,
            tile_cache,
        )
        if manifest is None:
            return None

        x_coords = sorted(set(x for x, y in coordinates))
        y_coords = sorted(set(y for x, y in coordinates))
        x_index_map = {value: idx for idx, value in enumerate(x_coords)}
        y_index_map = {value: idx for idx, value in enumerate(y_coords)}

        return self._compose_frame_from_manifest(
            manifest,
            coordinates,
            x_coords,
            y_coords,
            x_index_map,
            y_index_map,
        )
        
    def _update_content_bounds(
        self,
        alpha_mask: np.ndarray,
        bounds: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Expand a running bounding box using the provided alpha mask."""
        if alpha_mask is None or not np.any(alpha_mask):
            return bounds

        # Ensure mask is uint8 for OpenCV operations
        if alpha_mask.dtype != np.uint8:
            mask = alpha_mask.astype(np.uint8)
        else:
            mask = alpha_mask

        coords = cv2.findNonZero(mask)
        if coords is None:
            return bounds

        x, y, w, h = cv2.boundingRect(coords)
        updated = (x, y, x + w, y + h)

        if bounds is None:
            return updated

        return (
            min(bounds[0], updated[0]),
            min(bounds[1], updated[1]),
            max(bounds[2], updated[2]),
            max(bounds[3], updated[3])
        )

    @staticmethod
    def _merge_bounds(
        current: Optional[Tuple[int, int, int, int]],
        new: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Merge two bounding boxes if both exist."""
        if new is None:
            return current
        if current is None:
            return new
        return (
            min(current[0], new[0]),
            min(current[1], new[1]),
            max(current[2], new[2]),
            max(current[3], new[3]),
        )

    def _encode_with_ffmpeg(
        self,
        frame_iter: Iterable[bytes],
        output_path: Path,
        fps: int,
        crop_bounds: Tuple[int, int, int, int]
    ) -> None:
        """Encode frames streamed as PNG data into FFmpeg using a modern codec."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg with libx264/libx265/libsvtav1."
            )

        x0, y0, x1, y1 = crop_bounds
        crop_w, crop_h = (x1 - x0), (y1 - y0)
        crop_filter = f"crop={crop_w}:{crop_h}:{x0}:{y0}"

        codec_args = [
            "-c:v",
            "libx264",
            "-crf",
            str(getattr(self, "quality", 20)),
            "-preset",
            "slow",
            "-tune",
            "animation",
            "-x264-params",
            "keyint=300:min-keyint=300:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

        temp_output = output_path.with_name(f".tmp_{uuid.uuid4().hex}_{output_path.name}")
        if temp_output.exists():
            temp_output.unlink()

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-vf",
            crop_filter,
            "-an",
            *codec_args,
            str(temp_output),
        ]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        stderr_data: bytes = b""
        return_code: Optional[int] = None
        stderr_chunks: List[bytes] = []
        stderr_thread: Optional[threading.Thread] = None

        if process.stderr is not None:
            def _drain_stderr(pipe: Any, collector: List[bytes]) -> None:
                """Continuously read FFmpeg stderr so it cannot block."""
                try:
                    for chunk in iter(lambda: pipe.read(8192), b""):
                        if chunk:
                            collector.append(chunk)
                finally:
                    try:
                        pipe.close()
                    except OSError:
                        pass

            stderr_thread = threading.Thread(
                target=_drain_stderr,
                args=(process.stderr, stderr_chunks),
                daemon=True,
            )
            stderr_thread.start()

        try:
            for frame_data in frame_iter:
                if process.stdin is None:
                    raise RuntimeError("FFmpeg stdin closed unexpectedly.")
                process.stdin.write(frame_data)
            if process.stdin:
                process.stdin.close()
            return_code = process.wait()
        except Exception:
            process.kill()
            raise
        finally:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if stderr_thread is not None:
                stderr_thread.join()
            elif process.stderr is not None:
                try:
                    process.stderr.close()
                except OSError:
                    pass

        if stderr_thread is not None:
            stderr_data = b"".join(stderr_chunks)
        elif process.stderr is not None:
            try:
                stderr_data = process.stderr.read()
            except OSError:
                stderr_data = b""

        if return_code is None:
            return_code = process.poll()

        if return_code != 0:
            stderr_text = (
                stderr_data.decode("utf-8", errors="replace")
                if stderr_data
                else ''
            )
            if temp_output.exists():
                try:
                    temp_output.unlink()
                except OSError:
                    pass
            raise subprocess.CalledProcessError(return_code or -1, cmd, stderr=stderr_text)

        temp_output.replace(output_path)
        
    def create_differential_frame(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Create differential frame showing changes between two frames"""
        if prev_frame is None:
            # First frame - return configured background color
            return np.full_like(curr_frame, self.background_color)
            
        # Calculate absolute difference
        diff = cv2.absdiff(prev_frame, curr_frame)
        background = np.full_like(curr_frame, self.background_color)
        
        if self.diff_visualization == 'binary':
            # Binary difference: white changes on black background
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, binary_diff = cv2.threshold(gray_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            result = background.copy()
            result[binary_diff > 0] = (255, 255, 255)
            return result
            
        elif self.diff_visualization == 'heatmap':
            # Heatmap: color-coded intensity of changes
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # Apply threshold
            _, thresholded = cv2.threshold(gray_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            # Apply enhancement
            enhanced = cv2.multiply(thresholded, self.diff_enhancement_factor)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            # Apply colormap (COLORMAP_JET for blue->red heatmap)
            heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
            result = background.copy()
            mask = thresholded > 0
            result[mask] = heatmap[mask]
            return result
            
        elif self.diff_visualization == 'overlay':
            # Overlay: changes highlighted on semi-transparent background
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Create overlay
            overlay = curr_frame.copy()
            overlay[mask > 0] = [0, 255, 255]  # Yellow highlight for changes
            
            # Blend with original frame
            alpha = 0.7
            result = cv2.addWeighted(curr_frame, alpha, overlay, 1-alpha, 0)
            return result
            
        elif self.diff_visualization == 'colored':
            # Colored difference: green for additions, red for removals
            # This is a simplified version - in practice, detecting additions vs removals 
            # requires more sophisticated analysis
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Create colored diff image
            colored_diff = np.zeros_like(curr_frame)
            
            # Show changes in bright green
            colored_diff[mask > 0] = [0, 255, 0]  # Green for changes
            
            # Enhance visibility
            enhanced_diff = cv2.multiply(colored_diff, self.diff_enhancement_factor)
            enhanced_diff = np.clip(enhanced_diff, 0, 255).astype(np.uint8)
            
            result = background.copy()
            result[mask > 0] = enhanced_diff[mask > 0]
            return result
            
        else:
            # Default to absolute difference
            result = background.copy()
            diff_mask = np.any(diff > 0, axis=2)
            result[diff_mask] = diff[diff_mask]
            return result
    
    def get_enabled_timelapse_modes(self, timelapse_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of enabled timelapse modes for a timelapse"""
        modes = timelapse_config.get('timelapse_modes', {
            'normal': {'enabled': True, 'suffix': ''}
        })
        
        enabled_modes: List[Dict[str, Any]] = []
        for mode_name, mode_config in modes.items():
            if mode_config.get('enabled', True):
                suffix = mode_config.get('suffix', f'_{mode_name}' if mode_name != 'normal' else '')
                enabled_modes.append({
                    'mode': mode_name,
                    'suffix': suffix,
                    'create_full': mode_config.get('create_full_timelapse', False)
                })
                
        return enabled_modes

    def get_session_dirs_for_date(self, slug: str, date_str: str) -> List[Path]:
        """Return session directories for a specific date ordered chronologically"""
        date_dir = self.backup_dir / slug / date_str
        if not date_dir.exists():
            return []

        sessions: List[Tuple[datetime, Path]] = []
        for session_dir in date_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session_dt = self._parse_session_datetime(session_dir)
            if session_dt is None:
                continue
            sessions.append((session_dt, session_dir))

        sessions.sort(key=lambda item: item[0])
        return [path for _, path in sessions]

    def get_all_sessions(self, slug: str) -> List[Path]:
        """Collect all available session directories for a timelapse ordered chronologically"""
        slug_dir = self.backup_dir / slug
        if not slug_dir.exists():
            return []

        sessions: List[Tuple[datetime, Path]] = []
        for date_dir in slug_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for session_dir in date_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                session_dt = self._parse_session_datetime(session_dir)
                if session_dt is None:
                    continue
                sessions.append((session_dt, session_dir))

        sessions.sort(key=lambda item: item[0])
        return [path for _, path in sessions]

    def _build_frame_manifests(
        self,
        session_dirs: List[Path],
        timelapse_config: Dict[str, Any],
        coordinates: List[Tuple[int, int]],
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> List[FrameManifest]:
        """Generate frame manifests while respecting placeholder fallbacks."""
        manifests: List[FrameManifest] = []
        manifest_tile_cache: Dict[Tuple[int, int], Path] = {}
        prep_total = len(session_dirs)
        if prep_total == 0:
            return manifests

        prep_interval = max(1, prep_total // 20)
        valid_frames = 0

        for index, session_dir in enumerate(session_dirs, start=1):
            manifest = self._build_manifest_for_session(
                session_dir,
                timelapse_config,
                coordinates,
                manifest_tile_cache,
            )
            if manifest is not None:
                manifests.append(manifest)
                valid_frames += 1

            if index % prep_interval == 0 or index == prep_total:
                percent = (index / prep_total) * 100.0
                self.logger.info(
                    f"Frame preparation progress for '{name}' ({slug}) {mode_name} {label}: "
                    f"{index}/{prep_total} sessions scanned, {valid_frames} usable ({percent:.1f}%)"
                )

        return manifests

    def _render_frame_from_manifest(
        self,
        index: int,
        manifest: FrameManifest,
        coordinates: List[Tuple[int, int]],
        x_coords: List[int],
        y_coords: List[int],
        x_index_map: Dict[int, int],
        y_index_map: Dict[int, int],
        temp_dir: Path,
    ) -> PreparedFrame:
        """Worker routine to render a composite and persist it to disk."""
        composite = self._compose_frame_from_manifest(
            manifest,
            coordinates,
            x_coords,
            y_coords,
            x_index_map,
            y_index_map,
        )
        if composite is None:
            return PreparedFrame(
                index=index,
                session_dir=manifest.session_dir,
                temp_path=None,
                frame_shape=None,
                alpha_bounds=None,
            )

        frame_bounds = None
        if self.auto_crop_transparent_frames:
            frame_bounds = self._update_content_bounds(composite.alpha, None)

        success, buffer = cv2.imencode(".png", composite.color)
        if not success:
            raise RuntimeError(
                f"Failed to encode frame for manifest index {index} ({manifest.session_dir})"
            )

        temp_path = temp_dir / f"frame_{index:06d}.png"
        temp_path.write_bytes(buffer.tobytes())

        return PreparedFrame(
            index=index,
            session_dir=manifest.session_dir,
            temp_path=temp_path,
            frame_shape=composite.color.shape[:2],
            alpha_bounds=frame_bounds,
        )

    def _prepare_frames_from_manifests(
        self,
        manifests: List[FrameManifest],
        coordinates: List[Tuple[int, int]],
        x_coords: List[int],
        y_coords: List[int],
        x_index_map: Dict[int, int],
        y_index_map: Dict[int, int],
        temp_dir: Path,
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> List[PreparedFrame]:
        """Render frames in parallel using previously built manifests."""
        if not manifests:
            return []

        max_workers = max(1, getattr(self, "frame_prep_workers", 1))
        total = len(manifests)
        progress_interval = max(1, total // 20)
        prepared: List[Optional[PreparedFrame]] = [None] * total

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._render_frame_from_manifest,
                    index,
                    manifest,
                    coordinates,
                    x_coords,
                    y_coords,
                    x_index_map,
                    y_index_map,
                    temp_dir,
                )
                for index, manifest in enumerate(manifests)
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                prepared[result.index] = result
                completed += 1
                if completed % progress_interval == 0 or completed == total:
                    percent = (completed / total) * 100.0
                    self.logger.info(
                        f"Frame rendering progress for '{name}' ({slug}) {mode_name} {label}: "
                        f"{completed}/{total} frames ({percent:.1f}%)"
                    )

        return [frame for frame in prepared if frame and frame.temp_path is not None]

    def _frame_byte_generator(
        self,
        prepared_frames: List[PreparedFrame],
        mode_name: str,
        slug: str,
        name: str,
        label: str,
    ) -> Iterable[bytes]:
        """Yield encoded PNG data for prepared frames in order."""
        total_frames = len(prepared_frames)
        if total_frames == 0:
            return

        progress_interval = max(1, total_frames // 20)
        prev_composite_color: Optional[np.ndarray] = None

        for index, prepared in enumerate(prepared_frames, start=1):
            if prepared.temp_path is None:
                continue

            if mode_name == "diff":
                composite_color = cv2.imread(str(prepared.temp_path), cv2.IMREAD_COLOR)
                if composite_color is None:
                    raise RuntimeError(
                        f"Failed to read prepared composite for diff mode: {prepared.temp_path}"
                    )
                frame = self.create_differential_frame(prev_composite_color, composite_color)
                prev_composite_color = composite_color
                success, buffer = cv2.imencode(".png", frame)
                if not success:
                    raise RuntimeError(
                        f"Failed to encode differential frame for '{name}' ({slug}) {mode_name} {label}"
                    )
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = prepared.temp_path.read_bytes()

            yield frame_bytes

            if index % progress_interval == 0 or index == total_frames:
                percent = (index / total_frames) * 100.0
                self.logger.info(
                    f"Encoding progress for '{name}' ({slug}) {mode_name} {label}: "
                    f"{index}/{total_frames} frames ({percent:.1f}%)"
                )

    def render_timelapse_from_sessions(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        session_dirs: List[Path],
        mode_name: str,
        suffix: str,
        output_filename: str,
        label: str
    ):
        """Create timelapse from a pre-collected list of session directories"""
        if not session_dirs:
            self.logger.warning(
                f"No session directories found for '{name}' ({slug}) {label}"
            )
            return

        self.logger.info(
            f"Creating {mode_name} timelapse for '{name}' ({slug}) {label} with {len(session_dirs)} frames"
        )

        coordinates = self.get_tile_coordinates(timelapse_config)
        if not coordinates:
            self.logger.error(
                f"No valid frames created for '{name}' ({slug}) {mode_name} {label}"
            )
            return

        x_coords = sorted(set(x for x, y in coordinates))
        y_coords = sorted(set(y for x, y in coordinates))
        x_index_map = {value: idx for idx, value in enumerate(x_coords)}
        y_index_map = {value: idx for idx, value in enumerate(y_coords)}

        manifests = self._build_frame_manifests(
            session_dirs,
            timelapse_config,
            coordinates,
            slug,
            name,
            mode_name,
            label,
        )
        if not manifests:
            self.logger.error(
                f"No valid frames created for '{name}' ({slug}) {mode_name} {label}"
            )
            return

        output_dir = self.output_dir / slug
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / output_filename

        total_frames = 0

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            prepared_frames = self._prepare_frames_from_manifests(
                manifests,
                coordinates,
                x_coords,
                y_coords,
                x_index_map,
                y_index_map,
                temp_dir,
                slug,
                name,
                mode_name,
                label,
            )

            if not prepared_frames:
                self.logger.error(
                    f"No valid frames created for '{name}' ({slug}) {mode_name} {label}"
                )
                return

            first_frame_shape: Optional[Tuple[int, int]] = None
            content_bounds: Optional[Tuple[int, int, int, int]] = None

            for frame in prepared_frames:
                if frame.frame_shape is None:
                    continue
                if first_frame_shape is None:
                    first_frame_shape = frame.frame_shape
                if self.auto_crop_transparent_frames and frame.alpha_bounds is not None:
                    content_bounds = self._merge_bounds(content_bounds, frame.alpha_bounds)

            if first_frame_shape is None:
                self.logger.error(
                    f"No valid frames created for '{name}' ({slug}) {mode_name} {label}"
                )
                return

            frame_height, frame_width = first_frame_shape

            if self.auto_crop_transparent_frames and content_bounds is not None:
                min_x, min_y, max_x, max_y = content_bounds
                min_x = max(0, min(min_x, frame_width - 1))
                min_y = max(0, min(min_y, frame_height - 1))
                max_x = max(min_x + 1, min(max_x, frame_width))
                max_y = max(min_y + 1, min(max_y, frame_height))
                crop_bounds = (min_x, min_y, max_x, max_y)
            else:
                crop_bounds = (0, 0, frame_width, frame_height)

            crop_width = crop_bounds[2] - crop_bounds[0]
            crop_height = crop_bounds[3] - crop_bounds[1]
            if crop_width <= 0 or crop_height <= 0:
                crop_bounds = (0, 0, frame_width, frame_height)
                crop_width = frame_width
                crop_height = frame_height

            x0, y0, x1, y1 = crop_bounds

            if crop_width % 2 != 0:
                if x1 < frame_width:
                    x1 = min(frame_width, x1 + 1)
                elif x0 > 0:
                    x0 = max(0, x0 - 1)
                else:
                    self.logger.warning(
                        f"Unable to expand width to even dimension for '{name}' ({slug}) {mode_name} {label}"
                    )

            if crop_height % 2 != 0:
                if y1 < frame_height:
                    y1 = min(frame_height, y1 + 1)
                elif y0 > 0:
                    y0 = max(0, y0 - 1)
                else:
                    self.logger.warning(
                        f"Unable to expand height to even dimension for '{name}' ({slug}) {mode_name} {label}"
                    )

            crop_bounds = (x0, y0, x1, y1)
            crop_width = x1 - x0
            crop_height = y1 - y0

            if (
                self.auto_crop_transparent_frames
                and (crop_width != frame_width or crop_height != frame_height)
            ):
                self.logger.info(
                    f"Cropping '{name}' ({slug}) {mode_name} {label} to {crop_width}x{crop_height} "
                    f"from original {frame_width}x{frame_height}"
                )

            total_frames = len(prepared_frames)
            self.logger.info(
                f"Streaming {total_frames} frames to FFmpeg for '{name}' ({slug}) {mode_name} {label}"
            )

            try:
                frame_iter = self._frame_byte_generator(
                    prepared_frames,
                    mode_name,
                    slug,
                    name,
                    label,
                )
                self._encode_with_ffmpeg(frame_iter, output_path, self.fps, crop_bounds)
            except subprocess.CalledProcessError as exc:
                self.logger.error(
                    f"FFmpeg encoding failed for '{name}' ({slug}) {mode_name} {label}: {exc.stderr or exc}"
                )
                raise

        self.logger.info(
            f"{mode_name.title()} timelapse created for '{name}' ({slug}): {output_path} ({total_frames} frames)"
        )
        
    def create_daily_timelapse(self, date: datetime = None):
        """Create timelapse videos from previous day's images for all timelapses"""
        if date is None:
            date = datetime.now() - timedelta(days=1)
            
        date_str = date.strftime('%Y-%m-%d')
        enabled_timelapses = self.get_enabled_timelapses()
        
        self.logger.info(f"Creating timelapses for {date_str} for {len(enabled_timelapses)} timelapses")
        
        for timelapse in enabled_timelapses:
            slug = timelapse['slug']
            name = timelapse['name']
            
            enabled_modes = self.get_enabled_timelapse_modes(timelapse)
            session_dirs = self.get_session_dirs_for_date(slug, date_str)

            for mode in enabled_modes:
                mode_name = mode['mode']
                suffix = mode['suffix']
                label = f"on {date_str}"
                output_filename = f"{date_str}{suffix}.mp4"

                self.render_timelapse_from_sessions(
                    slug,
                    name,
                    timelapse,
                    session_dirs,
                    mode_name,
                    suffix,
                    output_filename,
                    label
                )

                if mode.get('create_full'):
                    full_session_dirs = self.get_all_sessions(slug)
                    self.render_timelapse_from_sessions(
                        slug,
                        name,
                        timelapse,
                        full_session_dirs,
                        mode_name,
                        suffix,
                        f"full{suffix}.mp4",
                        "across all backups"
                    )
            
    def create_full_timelapses(self):
        """Create full-history timelapses for modes configured to support them"""
        enabled_timelapses = self.get_enabled_timelapses()
        self.logger.info(f"Creating full-history timelapses for {len(enabled_timelapses)} timelapses")

        for timelapse in enabled_timelapses:
            slug = timelapse['slug']
            name = timelapse['name']

            enabled_modes = [
                mode for mode in self.get_enabled_timelapse_modes(timelapse)
                if mode.get('create_full')
            ]
            if not enabled_modes:
                continue

            session_dirs = self.get_all_sessions(slug)
            for mode in enabled_modes:
                self.render_timelapse_from_sessions(
                    slug,
                    name,
                    timelapse,
                    session_dirs,
                    mode['mode'],
                    mode['suffix'],
                    f"full{mode['suffix']}.mp4",
                    "across all backups"
                )
            
    def create_timelapse_for_slug(self, slug: str, name: str, timelapse_config: Dict[str, Any], date_str: str, mode_name: str = 'normal', suffix: str = ''):
        """Create timelapse video for a specific timelapse slug and mode"""
        session_dirs = self.get_session_dirs_for_date(slug, date_str)
        self.render_timelapse_from_sessions(
            slug,
            name,
            timelapse_config,
            session_dirs,
            mode_name,
            suffix,
            f"{date_str}{suffix}.mp4",
            f"on {date_str}"
        )
            
    def run(self):
        """Run the backup system with scheduled tasks"""
        scheduler = BlockingScheduler()
        
        # Schedule tile backups every N minutes
        scheduler.add_job(
            self.backup_tiles,
            trigger=IntervalTrigger(minutes=self.backup_interval),
            id='backup_tiles',
            name='Backup Tiles',
            max_instances=1
        )
        
        # Schedule daily timelapse creation at midnight
        scheduler.add_job(
            self.create_daily_timelapse,
            trigger=CronTrigger(hour=0, minute=1),
            id='create_timelapse',
            name='Create Daily Timelapse',
            max_instances=1
        )
        
        enabled_timelapses = self.get_enabled_timelapses()
        self.logger.info("Multi-timelapse backup system started")
        self.logger.info(f"Backup interval: {self.backup_interval} minutes")
        self.logger.info(f"Monitoring {len(enabled_timelapses)} timelapses:")
        for tl in enabled_timelapses:
            coords = tl['coordinates']
            self.logger.info(f"  - '{tl['name']}' ({tl['slug']}): X({coords['xmin']}-{coords['xmax']}), Y({coords['ymin']}-{coords['ymax']})")
        
        try:
            # Run initial backup
            self.backup_tiles()
            
            # Start scheduler
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Timelapse backup system stopped")
            scheduler.shutdown()

def main():
    """Main entry point"""
    backup_system = TimelapseBackup()
    backup_system.run()

if __name__ == "__main__":
    main()
