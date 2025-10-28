"""
Timelapse Backup System for WPlace Tiles
Automatically downloads tile images every 5 minutes and creates daily timelapses.
"""

import os
import logging
import cv2
import numpy as np
import shutil
import time
import subprocess
import uuid
import threading
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable, Iterator, Union
from dotenv import load_dotenv
from timelapse_backup.config import (
    DiffSettings,
    GlobalSettings,
    TimelapseConfig,
    _parse_background_color as config_parse_background_color,
    _parse_bool as config_parse_bool,
    _parse_positive_int as config_parse_positive_int,
    load_config,
)
from timelapse_backup.full_timelapse import (
    FullTimelapseSegment,
    FullTimelapseState,
)
from timelapse_backup.logging_setup import configure_logging
from timelapse_backup.models import (
    CompositeFrame,
    FrameManifest,
    PreparedFrame,
    RenderedTimelapseResult,
)
from timelapse_backup.stats import TimelapseStatsCollector, build_stats_report
from timelapse_backup.sessions import (
    get_all_sessions,
    get_prior_sessions,
    get_session_dirs_for_date,
    parse_session_datetime,
)
from timelapse_backup.manifests import ManifestBuilder
from timelapse_backup.rendering import Renderer
from timelapse_backup.tiles import TileDownloader
from timelapse_backup import scheduler as scheduler_module

# Load environment variables
load_dotenv()

class TimelapseBackup:
    PLACEHOLDER_SUFFIX = '.placeholder'

    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config_path = Path(config_file)
        # Historical data imported prior to this cutoff is treated as baseline-only.
        self.historical_cutoff: Optional[datetime] = datetime(2025, 10, 13)

        config_data = load_config(self.config_path)
        self.config_data = config_data
        self._apply_global_settings(config_data.global_settings)
        self.config = {
            'timelapses': [self._timelapse_to_dict(timelapse) for timelapse in config_data.timelapses],
            'global_settings': self._global_settings_to_dict(config_data.global_settings),
        }

        # Setup logging
        self.setup_logging()

        # Initialize tile downloader helper
        self._get_tile_downloader()

        # Create directories
        self.backup_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def _apply_global_settings(self, settings: GlobalSettings) -> None:
        """Populate instance attributes from parsed global settings."""
        self.base_url = settings.base_url
        self.backup_interval = settings.backup_interval_minutes
        self.backup_dir = settings.backup_dir
        self.output_dir = settings.output_dir
        self.request_delay = settings.request_delay
        self.fps = settings.timelapse_fps
        self.quality = settings.timelapse_quality
        self.background_color = settings.background_color
        self.auto_crop_transparent_frames = settings.auto_crop_transparent_frames
        self.frame_prep_workers = settings.frame_prep_workers

        diff = settings.diff_settings
        self.diff_threshold = diff.threshold
        self.diff_visualization = diff.visualization
        self.diff_fade_frames = diff.fade_frames
        self.diff_enhancement_factor = diff.enhancement_factor

        reporting = settings.reporting
        self.reporting_enabled = reporting.enable_stats_file
        self.seconds_per_pixel = reporting.seconds_per_pixel
        self.coverage_gap_multiplier = reporting.coverage_gap_multiplier

    def _timelapse_to_dict(self, config: TimelapseConfig) -> Dict[str, Any]:
        """Convert a dataclass timelapse config into the legacy dictionary shape."""
        coordinates = {
            'xmin': config.coordinates.xmin,
            'xmax': config.coordinates.xmax,
            'ymin': config.coordinates.ymin,
            'ymax': config.coordinates.ymax,
        }
        modes: Dict[str, Dict[str, Any]] = {}
        for name, mode in config.timelapse_modes.items():
            modes[name] = {
                'enabled': mode.enabled,
                'suffix': mode.suffix,
                'create_full_timelapse': mode.create_full_timelapse,
            }
        return {
            'slug': config.slug,
            'name': config.name,
            'description': config.description,
            'coordinates': coordinates,
            'enabled': config.enabled,
            'timelapse_modes': modes,
        }

    def _global_settings_to_dict(self, settings: GlobalSettings) -> Dict[str, Any]:
        """Convert parsed global settings into a legacy-compatible dictionary."""
        return {
            'base_url': settings.base_url,
            'backup_interval_minutes': settings.backup_interval_minutes,
            'backup_dir': str(settings.backup_dir),
            'output_dir': str(settings.output_dir),
            'request_delay': settings.request_delay,
            'timelapse_fps': settings.timelapse_fps,
            'timelapse_quality': settings.timelapse_quality,
            'background_color': list(settings.background_color),
            'auto_crop_transparent_frames': settings.auto_crop_transparent_frames,
            'frame_prep_workers': settings.frame_prep_workers,
            'diff_settings': {
                'threshold': settings.diff_settings.threshold,
                'visualization': settings.diff_settings.visualization,
                'fade_frames': settings.diff_settings.fade_frames,
                'enhancement_factor': settings.diff_settings.enhancement_factor,
            },
            'reporting': {
                'enable_stats_file': settings.reporting.enable_stats_file,
                'seconds_per_pixel': settings.reporting.seconds_per_pixel,
                'coverage_gap_multiplier': settings.reporting.coverage_gap_multiplier,
            },
        }

    def _parse_background_color(self, value: Any) -> Tuple[int, int, int]:
        """Backward-compatible wrapper for the background color parser."""
        return config_parse_background_color(value)

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        """Backward-compatible wrapper for boolean parsing."""
        return config_parse_bool(value, default)

    @staticmethod
    def _parse_positive_int(value: Any, default: int) -> int:
        """Backward-compatible wrapper for positive integer parsing."""
        return config_parse_positive_int(value, default)

    def get_prior_sessions(self, slug: str, current_session: Path) -> List[Path]:
        """Delegate to session helper for compatibility."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None or not slug:
            return []
        return get_prior_sessions(backup_root, slug, current_session)

    def get_session_dirs_for_date(self, slug: str, date_str: str) -> List[Path]:
        """Delegate to session helper for compatibility."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None:
            return []
        return get_session_dirs_for_date(backup_root, slug, date_str)

    def get_all_sessions(self, slug: str) -> List[Path]:
        """Delegate to session helper for compatibility."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None:
            return []
        return get_all_sessions(backup_root, slug)

    def _get_tile_downloader(self) -> TileDownloader:
        if not hasattr(self, "_tile_downloader"):
            logger = getattr(self, "logger", logging.getLogger(__name__))
            base_url = getattr(self, "base_url", "")
            self._tile_downloader = TileDownloader(
                base_url=base_url,
                logger=logger,
                placeholder_suffix=self.PLACEHOLDER_SUFFIX,
            )
        return self._tile_downloader

    def _get_manifest_builder(self) -> ManifestBuilder:
        tile_downloader = self._get_tile_downloader()
        current_logger = getattr(self, "logger", logging.getLogger(__name__))
        builder = getattr(self, "_manifest_builder", None)
        if (
            builder is None
            or builder.tile_downloader is not tile_downloader
            or builder.background_color != self.background_color
            or builder.logger is not current_logger
        ):
            builder = ManifestBuilder(
                tile_downloader=tile_downloader,
                background_color=self.background_color,
                logger=current_logger,
            )
            self._manifest_builder = builder
        return builder

    def _placeholder_filename(self, filename: str) -> str:
        return self._get_tile_downloader().placeholder_filename(filename)

    def _placeholder_path(self, session_dir: Path, filename: str) -> Path:
        return self._get_tile_downloader().placeholder_path(session_dir, filename)

    def _write_placeholder(self, placeholder_path: Path, target_path: Path) -> bool:
        return self._get_tile_downloader().write_placeholder(placeholder_path, target_path)

    def _get_renderer(self) -> Renderer:
        builder = self._get_manifest_builder()
        current_logger = getattr(self, "logger", logging.getLogger(__name__))

        if hasattr(self, "config_data"):
            diff_settings = self.config_data.global_settings.diff_settings
        else:
            diff_settings = DiffSettings(
                threshold=getattr(self, "diff_threshold", 10),
                visualization=getattr(self, "diff_visualization", "colored"),
                fade_frames=getattr(self, "diff_fade_frames", 3),
                enhancement_factor=getattr(self, "diff_enhancement_factor", 2.0),
            )

        renderer = getattr(self, "_renderer", None)
        if (
            renderer is None
            or renderer.manifest_builder is not builder
            or renderer.logger is not current_logger
            or renderer.frame_prep_workers != max(1, getattr(self, "frame_prep_workers", 1))
            or renderer.auto_crop_transparent_frames != getattr(self, "auto_crop_transparent_frames", True)
            or renderer.diff_settings != diff_settings
            or renderer.historical_cutoff != getattr(self, "historical_cutoff", None)
        ):
            renderer = Renderer(
                manifest_builder=builder,
                logger=current_logger,
                frame_prep_workers=getattr(self, "frame_prep_workers", 1),
                auto_crop_transparent_frames=getattr(self, "auto_crop_transparent_frames", True),
                diff_settings=diff_settings,
                historical_cutoff=getattr(self, "historical_cutoff", None),
            )
            self._renderer = renderer
        return renderer


    def get_enabled_timelapses(self) -> List[Dict[str, Any]]:
        """Get list of enabled timelapse configurations"""
        return [tl for tl in self.config['timelapses'] if tl.get('enabled', True)]

    def get_last_capture_time(self) -> Optional[datetime]:
        """Return the most recent session timestamp across enabled timelapses."""
        latest: Optional[datetime] = None
        for timelapse in self.get_enabled_timelapses():
            slug = timelapse.get("slug")
            if not slug:
                continue
            sessions = get_all_sessions(self.backup_dir, slug)
            if not sessions:
                continue
            session_time = parse_session_datetime(sessions[-1])
            if session_time is None:
                continue
            if latest is None or session_time > latest:
                latest = session_time
        return latest

    def get_enabled_timelapse_modes(
        self,
        timelapse_config: Union[Dict[str, Any], TimelapseConfig],
    ) -> List[Dict[str, Any]]:
        """Return enabled rendering modes for a timelapse in legacy dict format."""
        enabled_modes: List[Dict[str, Any]] = []

        if isinstance(timelapse_config, TimelapseConfig):
            for mode in timelapse_config.enabled_modes():
                enabled_modes.append(
                    {
                        "mode": mode.name,
                        "suffix": mode.suffix,
                        "create_full": mode.create_full_timelapse,
                    }
                )
        else:
            raw_modes = {}
            if isinstance(timelapse_config, dict):
                raw_modes = timelapse_config.get("timelapse_modes", {})

            if isinstance(raw_modes, dict):
                for mode_name, raw_mode in raw_modes.items():
                    if not isinstance(raw_mode, dict):
                        continue
                    if not self._parse_bool(raw_mode.get("enabled"), True):
                        continue
                    create_full_value = raw_mode.get("create_full")
                    if create_full_value is None:
                        create_full_value = raw_mode.get("create_full_timelapse")
                    enabled_modes.append(
                        {
                            "mode": mode_name,
                            "suffix": str(raw_mode.get("suffix", "")),
                            "create_full": self._parse_bool(create_full_value, False),
                        }
                    )
            elif isinstance(raw_modes, list):
                for entry in raw_modes:
                    if isinstance(entry, str):
                        enabled_modes.append(
                            {"mode": entry, "suffix": "", "create_full": False}
                        )
                    elif isinstance(entry, dict):
                        mode_name = str(entry.get("mode") or entry.get("name") or "normal")
                        if not self._parse_bool(entry.get("enabled"), True):
                            continue
                        create_full_value = entry.get("create_full")
                        if create_full_value is None:
                            create_full_value = entry.get("create_full_timelapse")
                        enabled_modes.append(
                            {
                                "mode": mode_name,
                                "suffix": str(entry.get("suffix", "")),
                                "create_full": self._parse_bool(create_full_value, False),
                            }
                        )

        if not enabled_modes:
            enabled_modes.append(
                {
                    "mode": "normal",
                    "suffix": "",
                    "create_full": False,
                }
            )
        return enabled_modes
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = configure_logging(logger_name=__name__)
        
    def get_tile_coordinates(self, timelapse_config: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Generate list of tile coordinates to download for a specific timelapse"""
        coords = timelapse_config['coordinates']
        coordinates = []
        for x in range(coords['xmin'], coords['xmax'] + 1):
            for y in range(coords['ymin'], coords['ymax'] + 1):
                coordinates.append((x, y))
        return coordinates
    




    def build_previous_tile_map(
        self,
        prior_sessions: Iterable[Path],
        coordinates: Iterable[Tuple[int, int]],
    ) -> Dict[str, Path]:
        """Delegate to tile downloader for compatibility."""
        return self._get_tile_downloader().build_previous_tile_map(
            prior_sessions,
            coordinates,
        )

    def resolve_tile_image_path(
        self,
        session_dir: Path,
        x: int,
        y: int,
        prior_sessions: Optional[List[Path]] = None
    ) -> Optional[Path]:
        """Resolve actual image path for a tile, following placeholders if needed"""
        backup_root = getattr(self, "backup_dir", None)
        return self._get_tile_downloader().resolve_tile_image_path(
            session_dir,
            x,
            y,
            prior_sessions=prior_sessions,
            backup_root=backup_root,
        )
        
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
        return self._get_tile_downloader().download_tile(
            slug,
            x,
            y,
            session_dir,
            previous_tile_map,
        )
            
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
            prior_sessions = get_prior_sessions(self.backup_dir, slug, session_dir)
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
        backup_root = getattr(self, "backup_dir", None)
        builder = self._get_manifest_builder()
        return builder.build_manifest_for_session(
            session_dir,
            coordinates,
            backup_root=backup_root,
            slug=timelapse_config.get('slug'),
            prior_sessions=prior_sessions,
            tile_cache=tile_cache,
        )

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
        return self._get_manifest_builder().compose_frame(manifest, coordinates)

    def create_composite_image(
        self,
        session_dir: Path,
        timelapse_config: Dict[str, Any],
        tile_cache: Optional[Dict[Tuple[int, int], Path]] = None
    ) -> Optional[CompositeFrame]:
        """Create a composite image from individual tiles and retain transparency."""
        coordinates = self.get_tile_coordinates(timelapse_config)
        builder = self._get_manifest_builder()
        backup_root = getattr(self, "backup_dir", None)
        return builder.create_composite_image(
            session_dir,
            coordinates,
            backup_root=backup_root,
            slug=timelapse_config.get('slug'),
            tile_cache=tile_cache,
        )
        


    def _encode_with_ffmpeg(
        self,
        frame_iter: Iterable[bytes],
        output_path: Path,
        fps: int,
        crop_bounds: Tuple[int, int, int, int]
    ) -> None:
        renderer = self._get_renderer()
        renderer.encode_with_ffmpeg(
            iter(frame_iter),
            output_path,
            fps,
            crop_bounds,
            quality=getattr(self, "quality", 20),
        )

    def _remux_segments_with_ffmpeg(
        self,
        concat_path: Path,
        output_path: Path,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg with libx264/libx265/libsvtav1."
            )

        temp_output = output_path.with_name(f".tmp_{uuid.uuid4().hex}_{output_path.name}")
        if temp_output.exists():
            temp_output.unlink()

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
        )

        temp_output.replace(output_path)

    @staticmethod
    def _probe_video_dimensions(video_path: Path) -> Optional[Tuple[int, int]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if width > 0 and height > 0:
            return width, height
        return None

    def _reframe_video_to_bounds(
        self,
        video_path: Path,
        content_width: int,
        content_height: int,
        crop_left: int,
        crop_top: int,
        target_width: int,
        target_height: int,
        left_offset: int,
        top_offset: int,
    ) -> None:
        if (
            target_width <= 0
            or target_height <= 0
            or content_width <= 0
            or content_height <= 0
        ):
            return

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg with libx264/libx265/libsvtav1."
            )

        current_width: Optional[int] = None
        current_height: Optional[int] = None
        current_dims = self._probe_video_dimensions(video_path)
        if current_dims is not None:
            current_width, current_height = current_dims
            if (
                current_width == target_width
                and current_height == target_height
                and crop_left == left_offset
                and crop_top == top_offset
            ):
                return
            if content_width > current_width or content_height > current_height:
                self.logger.warning(
                    "Skipping reframe for %s; content bounds exceed current frame",
                    video_path,
                )
                return

        color_bgr = getattr(self, "background_color", (0, 0, 0))
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        pad_color = f"0x{r:02X}{g:02X}{b:02X}"

        temp_output = video_path.with_name(f".tmp_{uuid.uuid4().hex}_{video_path.name}")
        if temp_output.exists():
            temp_output.unlink()

        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        left_offset = max(0, left_offset)
        top_offset = max(0, top_offset)

        if left_offset + content_width > target_width:
            target_width = left_offset + content_width
        if top_offset + content_height > target_height:
            target_height = top_offset + content_height

        if current_width is not None and crop_left + content_width > current_width:
            self.logger.warning(
                "Skipping reframe for %s; crop exceeds current width",
                video_path,
            )
            return
        if current_height is not None and crop_top + content_height > current_height:
            self.logger.warning(
                "Skipping reframe for %s; crop exceeds current height",
                video_path,
            )
            return

        crop_filter = f"crop={content_width}:{content_height}:{crop_left}:{crop_top}"
        pad_filter = (
            f"pad={target_width}:{target_height}:{left_offset}:{top_offset}:color={pad_color}"
        )
        filter_chain = f"{crop_filter},{pad_filter}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(getattr(self, "quality", 20)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        temp_output.replace(video_path)

    def _ensure_segment_dimensions(self, state: FullTimelapseState) -> None:
        for segment in state.segments:
            if segment.video_width is None or segment.video_height is None:
                dims = self._probe_video_dimensions(state.segment_path(segment))
                if dims is not None:
                    segment.video_width, segment.video_height = dims
            if segment.content_width is None and segment.video_width is not None:
                segment.content_width = segment.video_width
            if segment.content_height is None and segment.video_height is not None:
                segment.content_height = segment.video_height
            if segment.crop_x is None:
                segment.crop_x = 0
            if segment.crop_y is None:
                segment.crop_y = 0
            if segment.pad_left is None:
                segment.pad_left = 0
            if segment.pad_top is None:
                segment.pad_top = 0



    def create_differential_frame(
        self,
        prev_frame: Optional[np.ndarray],
        curr_frame: np.ndarray,
        return_stats: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """Delegate differential frame generation to the renderer."""
        return self._get_renderer().create_differential_frame(
            prev_frame,
            curr_frame,
            return_stats=return_stats,
        )

    def _prepare_frames_from_manifests(
        self,
        manifests: List[FrameManifest],
        coordinates: List[Tuple[int, int]],
        temp_dir: Path,
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> List[PreparedFrame]:
        return self._get_renderer().prepare_frames_from_manifests(
            manifests,
            coordinates,
            temp_dir,
            slug,
            name,
            mode_name,
            label,
        )

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
        builder = self._get_manifest_builder()
        backup_root = getattr(self, "backup_dir", None)
        return builder.build_frame_manifests(
            session_dirs,
            coordinates,
            backup_root=backup_root,
            slug=slug,
            mode_name=mode_name,
            label=label,
            timelapse_name=name,
        )

    def _frame_byte_generator(
        self,
        prepared_frames: List[PreparedFrame],
        mode_name: str,
        slug: str,
        name: str,
        label: str,
        frame_datetimes: List[Optional[datetime]],
    ) -> Tuple[Iterator[bytes], TimelapseStatsCollector]:
        renderer = self._get_renderer()
        return renderer.frame_byte_generator(
            prepared_frames,
            mode_name,
            slug,
            name,
            label,
            frame_datetimes,
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
    ) -> Optional[RenderedTimelapseResult]:
        """Create timelapse from a pre-collected list of session directories.

        Returns:
            RenderedTimelapseResult if frames were rendered, otherwise None.
        """
        if not session_dirs:
            self.logger.warning(
                "No session directories found for '%s' (%s) %s",
                name,
                slug,
                label,
            )
            return None

        self.logger.info(
            "Creating %s timelapse for '%s' (%s) %s with %s frames",
            mode_name,
            name,
            slug,
            label,
            len(session_dirs),
        )

        coordinates = self.get_tile_coordinates(timelapse_config)
        if not coordinates:
            self.logger.error(
                "No valid frames created for '%s' (%s) %s %s",
                name,
                slug,
                mode_name,
                label,
            )
            return None

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
                "No valid frames created for '%s' (%s) %s %s",
                name,
                slug,
                mode_name,
                label,
            )
            return None

        output_dir = self.output_dir / slug
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        renderer = self._get_renderer()
        total_frames = 0
        rendered_session_dirs: List[Path] = []
        rendered_video_width: Optional[int] = None
        rendered_video_height: Optional[int] = None
        rendered_content_width: Optional[int] = None
        rendered_content_height: Optional[int] = None
        rendered_crop_x: Optional[int] = None
        rendered_crop_y: Optional[int] = None

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            prepared_frames = self._prepare_frames_from_manifests(
                manifests,
                coordinates,
                temp_dir,
                slug,
                name,
                mode_name,
                label,
            )

            if not prepared_frames:
                self.logger.error(
                    "No valid frames created for '%s' (%s) %s %s",
                    name,
                    slug,
                    mode_name,
                    label,
                )
                return None

            try:
                crop_bounds, original_shape = renderer.compute_crop_bounds(
                    prepared_frames,
                    slug=slug,
                    name=name,
                    mode_name=mode_name,
                    label=label,
                )
                x0, y0, x1, y1 = crop_bounds
                rendered_crop_x = x0
                rendered_crop_y = y0
                content_width = max(0, x1 - x0)
                content_height = max(0, y1 - y0)
                if content_width == 0 or content_height == 0:
                    content_width = original_shape[1]
                    content_height = original_shape[0]
                rendered_content_width = content_width
                rendered_content_height = content_height
                rendered_video_width = content_width
                rendered_video_height = content_height
            except RuntimeError as exc:
                self.logger.error(str(exc))
                return None

            frame_datetimes: List[Optional[datetime]] = [
                parse_session_datetime(frame.session_dir)
                for frame in prepared_frames
            ]
            rendered_session_dirs = [frame.session_dir for frame in prepared_frames]

            gap_threshold: Optional[timedelta] = None
            gap_multiplier = getattr(self, "coverage_gap_multiplier", None)
            if (
                gap_multiplier is not None
                and gap_multiplier > 0
                and getattr(self, "backup_interval", 0) > 0
            ):
                gap_threshold = timedelta(
                    minutes=self.backup_interval * gap_multiplier
                )

            total_frames = len(prepared_frames)
            self.logger.info(
                "Streaming %s frames to FFmpeg for '%s' (%s) %s %s",
                total_frames,
                name,
                slug,
                mode_name,
                label,
            )

            try:
                frame_iter, stats_collector = self._frame_byte_generator(
                    prepared_frames,
                    mode_name,
                    slug,
                    name,
                    label,
                    frame_datetimes,
                )
                self._encode_with_ffmpeg(frame_iter, output_path, self.fps, crop_bounds)
            except subprocess.CalledProcessError as exc:
                self.logger.error(
                    "FFmpeg encoding failed for '%s' (%s) %s %s: %s",
                    name,
                    slug,
                    mode_name,
                    label,
                    exc.stderr or exc,
                )
                raise
            else:
                if getattr(self, "reporting_enabled", False):
                    stats_summary = stats_collector.summarize(
                        gap_threshold,
                        seconds_per_pixel=getattr(self, "seconds_per_pixel", 30),
                    )
                    generated_at = datetime.utcnow()
                    report_text = build_stats_report(
                        slug=slug,
                        name=name,
                        mode=mode_name,
                        label=label,
                        output_path=output_path,
                        generated_at=generated_at,
                        stats=stats_summary,
                    )
                    stats_path = output_path.with_suffix(output_path.suffix + ".stats.txt")
                    try:
                        stats_path.write_text(
                            report_text,
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        self.logger.error(
                            "Failed to write stats file for '%s' (%s) %s %s: %s",
                            name,
                            slug,
                            mode_name,
                            label,
                            exc,
                        )

        self.logger.info(
            "%s timelapse created for '%s' (%s): %s (%s frames)",
            mode_name.title(),
            name,
            slug,
            output_path,
            total_frames,
        )
        return RenderedTimelapseResult(
            output_path=output_path,
            frame_count=total_frames,
            session_dirs=tuple(rendered_session_dirs),
            video_width=rendered_video_width,
            video_height=rendered_video_height,
            content_width=rendered_content_width,
            content_height=rendered_content_height,
            crop_x=rendered_crop_x,
            crop_y=rendered_crop_y,
        )

    def render_incremental_full_timelapse(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        session_dirs: List[Path],
        mode_name: str,
        suffix: str,
        output_filename: str,
        label: str,
    ) -> None:
        """Render and append only new sessions to the full-history timelapse."""
        slug_dir = self.output_dir / slug
        slug_dir.mkdir(exist_ok=True)

        state = FullTimelapseState(
            slug_dir,
            output_filename,
            logger=self.logger,
        )
        state.load()

        pending_sessions = state.pending_sessions(session_dirs)
        if not pending_sessions:
            self.logger.info(
                "Full timelapse up to date for '%s' (%s) %s %s",
                name,
                slug,
                mode_name,
                label,
            )
            return

        state.ensure_segments_dir()
        temp_segment_rel = Path("segments") / state.output_basename / f"tmp_{uuid.uuid4().hex}.mp4"
        temp_segment_rel_str = temp_segment_rel.as_posix()

        try:
            render_result = self.render_timelapse_from_sessions(
                slug=slug,
                name=name,
                timelapse_config=timelapse_config,
                session_dirs=pending_sessions,
                mode_name=mode_name,
                suffix=suffix,
                output_filename=temp_segment_rel_str,
                label=f"{label} (segment)",
            )
        except subprocess.CalledProcessError:
            # render_timelapse_from_sessions already logged details
            return

        temp_segment_path = slug_dir / temp_segment_rel
        if render_result is None or render_result.frame_count == 0:
            temp_segment_path.unlink(missing_ok=True)
            self.logger.info(
                "Skipping full timelapse update for '%s' (%s) %s %s; no frames rendered",
                name,
                slug,
                mode_name,
                label,
            )
            return

        encoded_sessions = list(render_result.session_dirs)
        if not encoded_sessions:
            temp_segment_path.unlink(missing_ok=True)
            self.logger.warning(
                "Skipping full timelapse update for '%s' (%s) %s %s; no encoded sessions returned",
                name,
                slug,
                mode_name,
                label,
            )
            return

        first_dt = parse_session_datetime(encoded_sessions[0])
        last_dt = parse_session_datetime(encoded_sessions[-1])
        if first_dt is None or last_dt is None:
            temp_segment_path.unlink(missing_ok=True)
            self.logger.warning(
                "Unable to derive session timestamps for full timelapse '%s' (%s) %s %s; skipping update",
                name,
                slug,
                mode_name,
                label,
            )
            return

        segment_path = state.make_segment_filename(first_dt, last_dt)
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure unique segment names; replace duplicates to keep manifest clean.
        if segment_path.exists():
            segment_path.unlink()

        temp_segment_path.replace(segment_path)
        relative_segment_path = segment_path.relative_to(slug_dir)

        # Harmonize segment framing so the concat muxer can copy streams.
        self._ensure_segment_dimensions(state)

        probe_dims = self._probe_video_dimensions(segment_path)
        new_video_width = render_result.video_width or (probe_dims[0] if probe_dims else None)
        new_video_height = render_result.video_height or (probe_dims[1] if probe_dims else None)
        new_content_width = (
            render_result.content_width
            or render_result.video_width
            or (probe_dims[0] if probe_dims else None)
        )
        new_content_height = (
            render_result.content_height
            or render_result.video_height
            or (probe_dims[1] if probe_dims else None)
        )
        new_crop_x = render_result.crop_x if render_result.crop_x is not None else 0
        new_crop_y = render_result.crop_y if render_result.crop_y is not None else 0

        if new_content_width is None or new_content_height is None:
            dims = self._probe_video_dimensions(segment_path)
            if dims is None:
                self.logger.error(
                    "Unable to determine dimensions for new segment %s",
                    segment_path,
                )
                return
            new_content_width = dims[0]
            new_content_height = dims[1]
            if new_video_width is None:
                new_video_width = dims[0]
            if new_video_height is None:
                new_video_height = dims[1]

        existing_bounds = state.content_bounds()
        min_x = new_crop_x
        min_y = new_crop_y
        max_x = new_crop_x + new_content_width
        max_y = new_crop_y + new_content_height
        if existing_bounds is not None:
            min_x = min(min_x, existing_bounds[0])
            min_y = min(min_y, existing_bounds[1])
            max_x = max(max_x, existing_bounds[2])
            max_y = max(max_y, existing_bounds[3])

        target_width = max(0, max_x - min_x)
        target_height = max(0, max_y - min_y)

        if target_width == 0 or target_height == 0:
            dims = self._probe_video_dimensions(segment_path)
            if dims is not None:
                target_width = dims[0]
                target_height = dims[1]
                min_x = min(min_x, 0)
                min_y = min(min_y, 0)

        if target_width % 2 != 0:
            target_width += 1
        if target_height % 2 != 0:
            target_height += 1

        # Reframe existing segments.
        for segment in state.segments:
            content_width = segment.content_width or segment.video_width
            content_height = segment.content_height or segment.video_height
            if content_width is None or content_height is None:
                continue
            left_offset = max(0, segment.crop_x - min_x)
            top_offset = max(0, segment.crop_y - min_y)
            needs_reframe = (
                segment.video_width != target_width
                or segment.video_height != target_height
                or segment.pad_left != left_offset
                or segment.pad_top != top_offset
            )
            if not needs_reframe:
                continue

            seg_path = state.segment_path(segment)
            self.logger.info(
                "Reframing legacy segment %s to %sx%s (offset %s,%s)",
                seg_path,
                target_width,
                target_height,
                left_offset,
                top_offset,
            )
            self._reframe_video_to_bounds(
                seg_path,
                content_width,
                content_height,
                segment.pad_left or 0,
                segment.pad_top or 0,
                target_width,
                target_height,
                left_offset,
                top_offset,
            )
            segment.video_width = target_width
            segment.video_height = target_height
            segment.pad_left = left_offset
            segment.pad_top = top_offset

        # Reframe new segment.
        new_left_offset = max(0, new_crop_x - min_x)
        new_top_offset = max(0, new_crop_y - min_y)
        self._reframe_video_to_bounds(
            segment_path,
            new_content_width,
            new_content_height,
            0,
            0,
            target_width,
            target_height,
            new_left_offset,
            new_top_offset,
        )
        new_video_width = target_width
        new_video_height = target_height

        new_segment = FullTimelapseSegment(
            path=relative_segment_path.as_posix(),
            first_session=first_dt.replace(microsecond=0).isoformat(),
            last_session=last_dt.replace(microsecond=0).isoformat(),
            frame_count=render_result.frame_count,
            video_width=new_video_width,
            video_height=new_video_height,
            content_width=new_content_width,
            content_height=new_content_height,
            crop_x=new_crop_x,
            crop_y=new_crop_y,
            pad_left=new_left_offset,
            pad_top=new_top_offset,
        )

        updated_segments = [*state.segments, new_segment]
        concat_temp_path = state.write_concat_file(
            updated_segments,
            temporary=True,
        )

        full_output_path = slug_dir / output_filename
        try:
            self._remux_segments_with_ffmpeg(concat_temp_path, full_output_path)
        except subprocess.CalledProcessError as exc:
            segment_path.unlink(missing_ok=True)
            self.logger.error(
                "Failed to append to full timelapse for '%s' (%s) %s %s: %s",
                name,
                slug,
                mode_name,
                label,
                exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else exc,
            )
            return
        finally:
            concat_temp_path.unlink(missing_ok=True)

        state.add_segment(new_segment)
        state.write_concat_file(temporary=False)
        state.save()

        self.logger.info(
            "Full timelapse updated for '%s' (%s) %s %s with %s new frames",
            name,
            slug,
            mode_name,
            label,
            render_result.frame_count,
        )

    def create_daily_timelapse(self, date: datetime = None):
        """Create timelapse videos from previous day's images for all timelapses."""
        scheduler_module.create_daily_timelapse(self, date)
            
    def create_full_timelapses(self):
        """Create full-history timelapses for modes configured to support them."""
        scheduler_module.create_full_timelapses(self)
            
    def create_timelapse_for_slug(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        date_str: str,
        mode_name: str = 'normal',
        suffix: str = '',
    ):
        """Create timelapse video for a specific timelapse slug and mode."""
        scheduler_module.create_timelapse_for_slug(
            self,
            slug,
            name,
            timelapse_config,
            date_str,
            mode_name=mode_name,
            suffix=suffix,
        )
            
    def run(self):
        """Run the backup system with scheduled tasks."""
        scheduler_module.run(self)
