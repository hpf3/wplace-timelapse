"""
Timelapse Backup System for WPlace Tiles
Automatically downloads tile images every 5 minutes and creates daily timelapses.
"""

import os
import logging
import cv2
import numpy as np
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
from timelapse_backup.logging_setup import configure_logging
from timelapse_backup.models import (
    CompositeFrame,
    FrameManifest,
    PreparedFrame,
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
    ):
        """Create timelapse from a pre-collected list of session directories"""
        if not session_dirs:
            self.logger.warning(
                "No session directories found for '%s' (%s) %s",
                name,
                slug,
                label,
            )
            return

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
            return

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
            return

        output_dir = self.output_dir / slug
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename

        renderer = self._get_renderer()
        total_frames = 0

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
                return

            try:
                crop_bounds, _ = renderer.compute_crop_bounds(
                    prepared_frames,
                    slug=slug,
                    name=name,
                    mode_name=mode_name,
                    label=label,
                )
            except RuntimeError as exc:
                self.logger.error(str(exc))
                return

            frame_datetimes: List[Optional[datetime]] = [
                parse_session_datetime(frame.session_dir)
                for frame in prepared_frames
            ]

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
