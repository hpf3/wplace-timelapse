# WPlace Timelapse System

WPlace Timelapse System downloads tiles from the [WPlace](https://wplace.live) collaborative canvas, archives them efficiently, and renders daily MP4 timelapses. Everything you need to run the collector, renderer, and helper tooling lives in this repository.

## What It Does

- Captures tiles on a fixed schedule and stores them in timestamped session folders.
- Skips duplicate tiles by creating lightweight placeholders, keeping long-running archives manageable.
- Renders normal composites, optional differential videos, and coverage statistics from the collected sessions.
- Auto-crops transparent borders, applies tunable quality settings, and exposes programmatic hooks for custom automation.

## Default Paths

The collector is designed to run inside a container that exposes `/data` volumes:

- Backups: `/data/backups`
- Output videos: `/data/output`
- Logs: `/data/logs/timelapse_backup.log`
- Cache / temp data: `/data/cache` (used when retrieving archive/historical data)

If those directories are missing the logger falls back to the working directory, but creating `/data` up front keeps everything consistent.

## Quick Start

### Local Python environment

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/yourusername/wplace-timelapse.git
   cd wplace-timelapse
   ```
2. Install dependencies:
   ```bash
   uv sync  # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
3. Create a configuration:
   ```bash
   cp config.example.json config.json
   # Adjust coordinates or turn on extra regions as needed.
   ```
4. Run the scheduler:
   ```bash
   python main.py
   ```
5. Watch the outputs:
   - Raw frames and placeholders land in `backups/<slug>/<date>/<time>/`.
   - Videos appear under `output/<slug>/YYYY-MM-DD[_suffix].mp4`.
   - Logs flow to `/data/logs/timelapse_backup.log` when available (otherwise `./timelapse_backup.log`).

Programmatic control stays straightforward:

```python
from timelapse_backup import TimelapseBackup

backup = TimelapseBackup()
backup.create_full_timelapses()
```

### Docker

You can use the prebuilt image published from this repository:

```bash
docker pull ghcr.io/hpf3/wplace-timelapse:latest
```

1. Prepare host directories so data persists:
   ```bash
   mkdir -p ./data/backups ./data/output ./data/logs ./data/cache
   cp config.example.json ./data/config.json  # edit as needed
   ```
2. Run the container:
   ```bash
   docker run -it -v "$(pwd)/data:/data" ghcr.io/hpf3/wplace-timelapse:latest
   ```
   The entrypoint copies `/data/config.json` into place, runs migrations, starts the background cleanup helper, and launches the scheduler.
3. Review outputs under `./data` on the host; logs stream to `./data/logs/timelapse_backup.log`.

Prefer to build locally? Swap step 0 for:

```bash
docker build -t wplace-timelapse -f images/WplaceRecorder/Dockerfile .
```
and use that tag in step 2.

## Configuration Basics

`config.json` maps directly to the dataclasses in `timelapse_backup.config`. The example file in the repository looks like this:

```json
{
  "timelapses": [
    {
      "slug": "example_region",
      "name": "Example Region",
      "description": "Example timelapse configuration for a small region",
      "coordinates": {
        "xmin": 1000,
        "xmax": 1002,
        "ymin": 800,
        "ymax": 802
      },
      "enabled": true,
      "timelapse_modes": {
        "normal": {
          "enabled": true,
          "suffix": "",
          "create_full_timelapse": true
        },
        "diff": {
          "enabled": true,
          "suffix": "_diff",
          "create_full_timelapse": false
        }
      }
    },
    {
      "slug": "larger_area",
      "name": "Larger Area Example",
      "description": "Example configuration for monitoring a larger region",
      "coordinates": {
        "xmin": 1200,
        "xmax": 1210,
        "ymin": 850,
        "ymax": 860
      },
      "enabled": false,
      "timelapse_modes": {
        "normal": {
          "enabled": true,
          "suffix": "",
          "create_full_timelapse": false
        },
        "diff": {
          "enabled": false,
          "suffix": "_diff",
          "create_full_timelapse": false
        }
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
    "auto_crop_transparent_frames": true,
    "diff_settings": {
      "threshold": 10,
      "visualization": "colored",
      "fade_frames": 3,
      "enhancement_factor": 2.0
    }
  }
}
```

Key notes:

- Coordinates are inclusive tile bounds. `larger_area` is disabled by default but demonstrates how to monitor a broader region.
- `auto_crop_transparent_frames` trims unused borders after compositing so videos stay focused on the artwork. (Primarily useful when most tiles are empty.)
- Differential settings control whether change-only renders are produced. Stats reports only include meaningful change metrics when at least one diff-capable render runs; the `overlay` mode is a good compromise if you want a conventional video plus change counts.
- Environment variables in `.env` can override many settings thanks to `python-dotenv`.

## Repository Layout

- `timelapse_backup/` — scheduler, rendering, manifests, stats, and logging helpers.
- `tests/` — property-based and integration coverage for download, cleanup, diffing, and reporting routines.
- `historical_backfill/` — CLI for importing archived snapshots before the live collector started.
- `images/` — container build context that mirrors the `/data` layout used in production.

## Historical Backfill

Use the CLI to populate `backups/` with archived frames:

```bash
python -m historical_backfill init
python -m historical_backfill download --list --count 5
python -m historical_backfill download --tag world-2025-10-20T03-30-04.354Z
python -m historical_backfill generate \
  --slug example_region_historical \
  --start 2025-10-19T18:30:00Z \
  --end 2025-10-20T03:30:00Z \
  --interval-minutes 5 \
  --config config.json \
  --config-slug example_region \
  --cleanup-archives
```

`historical_backfill/fetch_nemo_history.sh` automates the download/extract/generate loop with sensible `/data` defaults. The archives come from [murolem/wplace-archives](https://github.com/murolem/wplace-archives); huge thanks to that project for keeping historical canvases available.

⚠️ Expect heavy bandwidth usage: each global archive is roughly 6–8 GB and must be fully downloaded regardless of the actually needed area, so generating multiple frames quickly adds up.

## Finding Coordinates

1. Open [wplace.live](https://wplace.live) and move to the region you want to capture.
2. Click the pixel that represents the top-left corner of your area. The info strip at the bottom shows the tile coordinates as the first two numbers (X, Y); record them as `xmin` and `ymin`.
3. Click the bottom-right pixel of the area and record the first two numbers as `xmax` and `ymax`.
4. Paste those inclusive bounds into `config.json`. For large canvases, split the region into multiple configs if you want separate renders or different mode settings.

## License

MIT — see [LICENSE](LICENSE).

## Support

If you encounter issues or have questions:

1. Check the logs in `timelapse_backup.log`
2. Verify your configuration matches the expected format
3. Ensure you have a stable internet connection
4. Open an issue on GitHub with detailed error information

---

*This system is designed for educational and archival purposes. Please be respectful of WPlace's servers and don't set intervals too aggressively.*
