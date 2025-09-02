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
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Load environment variables
load_dotenv()

class TimelapseBackup:
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
                    'timelapse_quality': self.quality
                }
            }
    
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
        
    def download_tile(self, x: int, y: int, session_dir: Path) -> bool:
        """Download a single tile image"""
        try:
            url = f"{self.base_url}/{x}/{y}.png"
            filename = f"{x}_{y}.png"
            filepath = session_dir / filename
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            self.logger.debug(f"Downloaded tile {x},{y} to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download tile {x},{y}: {e}")
            return False
            
    def backup_tiles(self):
        """Backup all tiles for current timestamp for all enabled timelapses"""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        
        enabled_timelapses = self.get_enabled_timelapses()
        self.logger.info(f"Starting backup session at {time_str} for {len(enabled_timelapses)} timelapses")
        
        total_successful = 0
        total_tiles = 0
        
        for timelapse in enabled_timelapses:
            slug = timelapse['slug']
            name = timelapse['name']
            
            # Create session directory for this timelapse
            session_dir = self.backup_dir / slug / date_str / time_str
            session_dir.mkdir(parents=True, exist_ok=True)
            
            coordinates = self.get_tile_coordinates(timelapse)
            successful_downloads = 0
            
            self.logger.info(f"Backing up '{name}' ({slug}): {len(coordinates)} tiles")
            
            for i, (x, y) in enumerate(coordinates):
                if self.download_tile(x, y, session_dir):
                    successful_downloads += 1
                
                # Add delay between requests (except for the last one)
                if i < len(coordinates) - 1:
                    time.sleep(self.request_delay)
                    
            self.logger.info(f"'{name}' completed: {successful_downloads}/{len(coordinates)} tiles")
            
            total_successful += successful_downloads
            total_tiles += len(coordinates)
            
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
                    
        self.logger.info(f"Backup session completed: {total_successful}/{total_tiles} total tiles downloaded")
                
    def create_composite_image(self, session_dir: Path, timelapse_config: Dict[str, Any]) -> np.ndarray:
        """Create a composite image from individual tiles"""
        coordinates = self.get_tile_coordinates(timelapse_config)
        
        # Determine grid dimensions
        x_coords = sorted(set(x for x, y in coordinates))
        y_coords = sorted(set(y for x, y in coordinates))
        
        # Load first image to get tile dimensions
        first_tile = None
        for x, y in coordinates:
            filepath = session_dir / f"{x}_{y}.png"
            if filepath.exists():
                first_tile = cv2.imread(str(filepath))
                break
                
        if first_tile is None:
            return None
            
        tile_height, tile_width = first_tile.shape[:2]
        
        # Create composite image
        composite_height = len(y_coords) * tile_height
        composite_width = len(x_coords) * tile_width
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        for x, y in coordinates:
            filepath = session_dir / f"{x}_{y}.png"
            if not filepath.exists():
                continue
                
            tile = cv2.imread(str(filepath))
            if tile is None:
                continue
                
            # Calculate position in composite
            x_idx = x_coords.index(x)
            y_idx = y_coords.index(y)
            
            start_y = y_idx * tile_height
            end_y = start_y + tile_height
            start_x = x_idx * tile_width
            end_x = start_x + tile_width
            
            composite[start_y:end_y, start_x:end_x] = tile
            
        return composite
        
    def create_differential_frame(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Create differential frame showing changes between two frames"""
        if prev_frame is None:
            # First frame - return black image
            return np.zeros_like(curr_frame)
            
        # Calculate absolute difference
        diff = cv2.absdiff(prev_frame, curr_frame)
        
        if self.diff_visualization == 'binary':
            # Binary difference: white changes on black background
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, binary_diff = cv2.threshold(gray_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(binary_diff, cv2.COLOR_GRAY2BGR)
            
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
            return heatmap
            
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
            
            return enhanced_diff
            
        else:
            # Default to absolute difference
            return diff
    
    def get_enabled_timelapse_modes(self, timelapse_config: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Get list of enabled timelapse modes for a timelapse"""
        modes = timelapse_config.get('timelapse_modes', {
            'normal': {'enabled': True, 'suffix': ''}
        })
        
        enabled_modes = []
        for mode_name, mode_config in modes.items():
            if mode_config.get('enabled', True):
                suffix = mode_config.get('suffix', f'_{mode_name}' if mode_name != 'normal' else '')
                enabled_modes.append((mode_name, suffix))
                
        return enabled_modes
        
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
            
            # Get enabled modes for this timelapse
            enabled_modes = self.get_enabled_timelapse_modes(timelapse)
            
            for mode_name, suffix in enabled_modes:
                self.create_timelapse_for_slug(slug, name, timelapse, date_str, mode_name, suffix)
            
    def create_timelapse_for_slug(self, slug: str, name: str, timelapse_config: Dict[str, Any], date_str: str, mode_name: str = 'normal', suffix: str = ''):
        """Create timelapse video for a specific timelapse slug and mode"""
        date_dir = self.backup_dir / slug / date_str
        
        if not date_dir.exists():
            self.logger.warning(f"No backup data found for '{name}' ({slug}) on {date_str}")
            return
            
        # Get all session directories for the date
        session_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])
        
        if not session_dirs:
            self.logger.warning(f"No session directories found for '{name}' ({slug}) on {date_str}")
            return
            
        self.logger.info(f"Creating {mode_name} timelapse for '{name}' ({slug}) on {date_str} with {len(session_dirs)} frames")
        
        # Create temporary directory for composite images
        temp_dir = Path(f"temp_timelapse_{slug}_{date_str}_{mode_name}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Create composite images
            valid_frames = 0
            prev_composite = None
            
            for i, session_dir in enumerate(session_dirs):
                composite = self.create_composite_image(session_dir, timelapse_config)
                if composite is not None:
                    if mode_name == 'diff':
                        # Create differential frame
                        diff_frame = self.create_differential_frame(prev_composite, composite)
                        frame_path = temp_dir / f"frame_{i:06d}.png"
                        cv2.imwrite(str(frame_path), diff_frame)
                        prev_composite = composite  # Store for next comparison
                    else:
                        # Normal mode - save composite directly
                        frame_path = temp_dir / f"frame_{i:06d}.png"
                        cv2.imwrite(str(frame_path), composite)
                    
                    valid_frames += 1
                    
            if valid_frames == 0:
                self.logger.error(f"No valid frames created for '{name}' ({slug}) {mode_name} on {date_str}")
                return
                
            # Create output directory for this timelapse
            output_dir = self.output_dir / slug
            output_dir.mkdir(exist_ok=True)
            
            # Create video using opencv
            output_filename = f"{date_str}{suffix}.mp4"
            output_path = output_dir / output_filename
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(temp_dir / "frame_000000.png"))
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                self.fps, 
                (width, height)
            )
            
            # Write frames to video
            for i in range(valid_frames):
                frame_path = temp_dir / f"frame_{i:06d}.png"
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
                    video_writer.write(frame)
                    
            video_writer.release()
            
            self.logger.info(f"{mode_name.title()} timelapse created for '{name}': {output_path} ({valid_frames} frames)")
            
        finally:
            # Cleanup temporary files
            for file in temp_dir.glob("*.png"):
                file.unlink()
            temp_dir.rmdir()
            
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
