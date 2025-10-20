# WPlace Timelapse System

A Python-based automated system that creates timelapse videos from [WPlace](https://wplace.live) collaborative pixel art canvas. The system downloads tile images at regular intervals and generates daily timelapses showing the evolution of artwork in different geographical regions.

## Features

- **Multi-Location Monitoring**: Track multiple regions simultaneously (Toulouse, Buenos Aires, etc.)
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
![Toulouse](images/toulouse_normal.png)
*Toulouse region timelapse showing collaborative pixel art evolution*

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
      "slug": "toulouse",
      "name": "Toulouse",
      "description": "Toulouse region monitoring",
      "coordinates": {
        "xmin": 1031,
        "xmax": 1032,
        "ymin": 747,
        "ymax": 748
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
    "background_color": [0, 0, 0],
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
- **background_color**: Base color (RGB array, hex string, or `{"value": [...], "order": "bgr"}`) used to fill empty areas or missing tiles (default: `[0, 0, 0]`)
- **auto_crop_transparent_frames**: When enabled (default: `true`), unions the non-transparent pixels from all merged frames and crops timelapse videos to that bounding box so unused tile borders are removed.
- **diff_settings**: Controls differential timelapse appearance and sensitivity

Numeric arrays are interpreted as `[R, G, B]` for readability. Use a hex string (for example `"#2596BE"`) or wrap values as `{"value": [B, G, R], "order": "bgr"}` if you need to specify explicit BGR ordering.

## Output Structure

```
output/
├── toulouse/
│   ├── 2025-09-01.mp4      # Normal timelapse
│   ├── 2025-09-01_diff.mp4 # Differential timelapse
│   └── 2025-09-02.mp4
├── other/
│   └── ...
└── ...

backups/
├── toulouse/
│   └── 2025-09-01/
│       ├── 12-00-00/       # Hour-minute-second
│       │   ├── 1214_832.png
│       │   └── ...
│       └── ...
└── ...
```

## Historical Backfill

The `historical_backfill/` sub-project can pre-populate the `backups/` tree with
frames that predate live capture by reusing snapshots from
[`murolem/wplace-archives`](https://github.com/murolem/wplace-archives). The CLI
mirrors the timelapse layout so the main pipeline can consume the backfilled
sessions without further changes.

```bash
# Prepare the workspace
python3 -m historical_backfill init

# Inspect the latest archive releases and fetch one
python3 -m historical_backfill download --list --count 5
python3 -m historical_backfill download --tag world-2025-10-20T03-30-04.354Z

# Optional checklist for manual download/extraction steps
python3 -m historical_backfill plan --slug toulouse

# Generate synthetic captures every 5 minutes between two timestamps
python3 -m historical_backfill generate \
  --slug toulouse \
  --start 2025-10-19T18:30:00Z \
  --end 2025-10-20T03:30:00Z \
  --interval-minutes 5 \
  --config config.json \
  --config-slug toulouse \
  --cleanup-archives

# Dedicated Nemo helper (downloads, extracts, generates in one pass)
bash historical_backfill/fetch_nemo_history.sh
# Defaults target `/data/backups` for frames and `/data/cache` for temporary
# archive storage; set the `HF_*` env vars described in the script to adjust.
```

Releases are mapped to frame timestamps by their capture time (encoded in the
folder name). Provide a `GITHUB_TOKEN` environment variable if you hit GitHub
rate limits while listing or downloading archives. When multiple frames reuse
the same archive snapshot,
placeholders are created to match the behaviour of the live collector.
Add `--cleanup-archives` to the generate command to remove the multi-gigabyte
tar parts after they are consumed (pair with `--archives-dir` if you store the
downloads elsewhere).

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
