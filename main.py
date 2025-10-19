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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
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
        """Create placeholder metadata referencing target tile path"""
        data = {
            "type": "placeholder",
            "target": str(target_path.resolve()),
            "created_at": datetime.utcnow().isoformat()
        }
        try:
            with open(placeholder_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return True
        except OSError as exc:
            self.logger.error(f"Failed to write placeholder {placeholder_path}: {exc}")
            return False

    def _read_placeholder_target(self, placeholder_path: Path) -> Optional[Path]:
        """Read target path from placeholder metadata"""
        try:
            with open(placeholder_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            target = data.get("target")
            if target:
                return Path(target)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            self.logger.warning(f"Invalid placeholder file {placeholder_path}: {exc}")
        return None

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

        for session_dir in prior_sessions:
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
                    target_path = self._read_placeholder_target(placeholder_path)
                    if target_path and target_path.exists():
                        result[filename] = target_path

        return result

    def resolve_tile_image_path(self, session_dir: Path, x: int, y: int) -> Optional[Path]:
        """Resolve actual image path for a tile, following placeholders if needed"""
        filename = f"{x}_{y}.png"
        candidate = session_dir / filename
        if candidate.exists():
            return candidate

        placeholder_path = self._placeholder_path(session_dir, filename)
        if placeholder_path.exists():
            target = self._read_placeholder_target(placeholder_path)
            if target and target.exists():
                return target
            if target and not target.exists():
                self.logger.warning(f"Placeholder target missing for {placeholder_path}: {target}")

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
                
    def create_composite_image(self, session_dir: Path, timelapse_config: Dict[str, Any]) -> Optional[CompositeFrame]:
        """Create a composite image from individual tiles and retain transparency."""
        coordinates = self.get_tile_coordinates(timelapse_config)
        
        # Determine grid dimensions
        x_coords = sorted(set(x for x, y in coordinates))
        y_coords = sorted(set(y for x, y in coordinates))
        
        # Load first image to get tile dimensions
        first_tile = None
        for x, y in coordinates:
            tile_path = self.resolve_tile_image_path(session_dir, x, y)
            if tile_path:
                first_tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
                if first_tile is not None and len(first_tile.shape) == 2:
                    first_tile = cv2.cvtColor(first_tile, cv2.COLOR_GRAY2BGR)
                elif first_tile is not None and first_tile.shape[2] == 1:
                    first_tile = cv2.cvtColor(first_tile, cv2.COLOR_GRAY2BGR)
                if first_tile is not None:
                    break

        if first_tile is None:
            return None
            
        tile_height, tile_width = first_tile.shape[:2]
        
        # Create composite image
        composite_height = len(y_coords) * tile_height
        composite_width = len(x_coords) * tile_width
        composite = np.full(
            (composite_height, composite_width, 3),
            self.background_color,
            dtype=np.uint8
        )
        alpha_mask = np.zeros((composite_height, composite_width), dtype=np.uint8)
        
        for x, y in coordinates:
            tile_path = self.resolve_tile_image_path(session_dir, x, y)
            if not tile_path:
                continue

            tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
            if tile is None:
                continue

            if len(tile.shape) == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            elif tile.shape[2] == 1:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
            elif tile.shape[2] == 3:
                opaque_alpha = np.full(
                    (tile.shape[0], tile.shape[1], 1),
                    255,
                    dtype=tile.dtype
                )
                tile = np.concatenate((tile, opaque_alpha), axis=2)
            elif tile.shape[2] != 4:
                continue
            
            # Calculate position in composite
            x_idx = x_coords.index(x)
            y_idx = y_coords.index(y)
            
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

    def _encode_with_ffmpeg(
        self,
        temp_dir: Path,
        output_path: Path,
        fps: int,
        crop_bounds: Tuple[int, int, int, int]
    ) -> None:
        """Encode PNG frames with FFmpeg using a modern codec."""
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

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%06d.png"),
            "-vf",
            crop_filter,
            "-an",
            *codec_args,
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        
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

        safe_label = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in label)
        temp_dir = Path(f"temp_timelapse_{slug}_{safe_label}_{mode_name}")
        temp_dir.mkdir(exist_ok=True)

        try:
            frame_index = 0
            valid_frames = 0
            prev_composite_color: Optional[np.ndarray] = None
            first_frame_shape: Optional[Tuple[int, int]] = None
            content_bounds: Optional[Tuple[int, int, int, int]] = None

            for i, session_dir in enumerate(session_dirs):
                composite = self.create_composite_image(session_dir, timelapse_config)
                if composite is None:
                    continue

                if first_frame_shape is None:
                    first_frame_shape = composite.color.shape[:2]

                if self.auto_crop_transparent_frames:
                    content_bounds = self._update_content_bounds(composite.alpha, content_bounds)

                frame_path = temp_dir / f"frame_{frame_index:06d}.png"
                if mode_name == 'diff':
                    diff_frame = self.create_differential_frame(prev_composite_color, composite.color)
                    cv2.imwrite(str(frame_path), diff_frame)
                    prev_composite_color = composite.color
                else:
                    cv2.imwrite(str(frame_path), composite.color)

                valid_frames += 1
                frame_index += 1

            if valid_frames == 0:
                self.logger.error(
                    f"No valid frames created for '{name}' ({slug}) {mode_name} {label}"
                )
                return

            output_dir = self.output_dir / slug
            output_dir.mkdir(exist_ok=True)

            output_path = output_dir / output_filename

            first_frame_path = temp_dir / "frame_000000.png"
            if not first_frame_path.exists():
                self.logger.error(
                    f"No frame files found for '{name}' ({slug}) {mode_name} {label}"
                )
                return

            first_frame = cv2.imread(str(first_frame_path))
            if first_frame is None:
                self.logger.error(
                    f"Failed to read first frame for '{name}' ({slug}) {mode_name} {label}"
                )
                return

            frame_height, frame_width = first_frame.shape[:2]

            if first_frame_shape is None:
                first_frame_shape = (frame_height, frame_width)

            crop_bounds: Tuple[int, int, int, int]
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

            try:
                self._encode_with_ffmpeg(temp_dir, output_path, self.fps, crop_bounds)
            except subprocess.CalledProcessError as exc:
                self.logger.error(
                    f"FFmpeg encoding failed for '{name}' ({slug}) {mode_name} {label}: {exc}"
                )
                raise

            self.logger.info(
                f"{mode_name.title()} timelapse created for '{name}' ({slug}): {output_path} ({valid_frames} frames)"
            )

        finally:
            for file in temp_dir.glob("*.png"):
                file.unlink()
            temp_dir.rmdir()
        
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
