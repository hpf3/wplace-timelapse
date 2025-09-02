# WPlace Timelapse System

A Python-based automated system that creates timelapse videos from [WPlace](https://wplace.live) collaborative pixel art canvas. The system downloads tile images at regular intervals and generates daily timelapses showing the evolution of artwork in different geographical regions.

## Features

- **Multi-Location Monitoring**: Track multiple regions simultaneously (Gaza, Toulouse, Buenos Aires, etc.)
- **Automated Collection**: Downloads tiles every 5 minutes with configurable intervals
- **Dual Timelapse Modes**:
  - **Normal Mode**: Standard timelapse showing full canvas evolution
  - **Differential Mode**: Highlights only the changes between frames
- **Flexible Configuration**: JSON-based configuration for easy setup of new regions
- **Robust Error Handling**: Continues operation despite network issues or missing tiles
- **Smart Scheduling**: Uses APScheduler for reliable automated execution
- **Video Output**: High-quality MP4 videos with configurable FPS and quality

## Sample Output

### Normal Timelapse
![Gaza Sample](images/gaza_sample.png)
*Gaza region timelapse showing collaborative pixel art evolution*

### Normal vs Differential Comparison
| Normal Mode | Differential Mode |
|-------------|-------------------|
| ![Toulouse Normal](images/toulouse_normal.png) | ![Toulouse Diff](images/toulouse_diff.png) |
| *Full canvas view* | *Changes highlighted in green* |

## How It Works

1. **Tile Collection**: Downloads PNG tiles from WPlace's tile server at specified coordinates
2. **Image Composition**: Combines individual tiles into complete canvas images
3. **Video Generation**: Creates MP4 timelapses using OpenCV with proper frame sequencing
4. **Differential Analysis**: Detects and visualizes changes between consecutive frames
5. **Automated Scheduling**: Runs backups every 5 minutes and creates daily videos at midnight

## Quick Start

### Prerequisites

- Python 3.13 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/wplace-timelapse.git
   cd wplace-timelapse
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv install

   # Or using pip with virtual environment
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure your timelapses**:
   ```bash
   cp config.example.json config.json
   # Edit config.json with your desired regions
   ```

4. **Run the system**:
   ```bash
   python main.py
   ```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Configuration

The system uses `config.json` for configuration. Here's the structure:

```json
{
  "timelapses": [
    {
      "slug": "gaza",
      "name": "Gaza",
      "description": "Gaza region monitoring",
      "coordinates": {
        "xmin": 1214,
        "xmax": 1220,
        "ymin": 832,
        "ymax": 837
      },
      "enabled": true,
      "timelapse_modes": {
        "normal": {"enabled": true, "suffix": ""},
        "diff": {"enabled": true, "suffix": "_diff"}
      }
    }
  ],
  "global_settings": {
    "base_url": "https://backend.wplace.live/files/s0/tiles",
    "backup_interval_minutes": 5,
    "backup_dir": "backups",
    "output_dir": "output",
    "request_delay": 0.5,
    "timelapse_fps": 10,
    "timelapse_quality": 23,
    "diff_settings": {
      "threshold": 10,
      "visualization": "colored",
      "fade_frames": 3,
      "enhancement_factor": 2.0
    }
  }
}
```

### Key Configuration Options

- **coordinates**: Define the tile region to monitor (x/y tile coordinates)
- **backup_interval_minutes**: How often to download tiles (default: 5 minutes)
- **timelapse_fps**: Frames per second for output videos (default: 10)
- **diff_settings**: Controls differential timelapse appearance and sensitivity

## Output Structure

```
output/
├── gaza/
│   ├── 2025-09-01.mp4      # Normal timelapse
│   ├── 2025-09-01_diff.mp4 # Differential timelapse
│   └── 2025-09-02.mp4
├── toulouse/
│   └── ...
└── ...

backups/
├── gaza/
│   └── 2025-09-01/
│       ├── 12-00-00/       # Hour-minute-second
│       │   ├── 1214_832.png
│       │   └── ...
│       └── ...
└── ...
```

## Finding Coordinates

To monitor a specific region on WPlace:

1. Visit [wplace.live](https://wplace.live)
2. Navigate to your area of interest
3. Check the browser's developer tools Network tab for tile requests
4. Look for URLs like `https://backend.wplace.live/files/s0/tiles/X/Y.png`
5. Use those X,Y coordinates to define your region bounds

## Timelapse Modes

### Normal Mode
Standard timelapse showing the complete evolution of the canvas region.

### Differential Mode
Advanced mode that highlights only the pixels that changed between frames:
- **Colored**: Shows changes in bright green
- **Binary**: Black and white change detection
- **Heatmap**: Color-coded intensity of changes
- **Overlay**: Changes highlighted on semi-transparent background

## Technical Details

- **Tile System**: WPlace uses a tile-based system similar to web maps
- **Image Format**: Downloads PNG tiles and combines them into composite images
- **Video Codec**: Outputs MP4 using OpenCV's mp4v codec
- **Scheduling**: APScheduler handles automated execution with cron-like triggers
- **Error Recovery**: Continues operation even if individual tiles fail to download

## Example Regions

The default configuration includes three interesting regions:

- **Gaza**: High activity region with frequent changes
- **Toulouse**: European city with moderate activity  
- **Buenos Aires**: South American region with varied artwork

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [WPlace](https://wplace.live) for providing the collaborative pixel art platform
- OpenCV team for the excellent computer vision library
- APScheduler developers for robust job scheduling

## Support

If you encounter issues or have questions:

1. Check the logs in `timelapse_backup.log`
2. Verify your configuration matches the expected format
3. Ensure you have a stable internet connection
4. Open an issue on GitHub with detailed error information

---

*This system is designed for educational and archival purposes. Please be respectful of WPlace's servers and don't set intervals too aggressively.*