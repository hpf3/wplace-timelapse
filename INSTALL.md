# Installation Guide

## Prerequisites

- **Python 3.13 or higher**
- **pip** or **uv** package manager
- **FFmpeg** (for video processing, optional - only needed for custom video operations)

## Method 1: Using uv (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/wplace-timelapse.git
cd wplace-timelapse

# Install dependencies
uv install
```

## Method 2: Using pip with virtual environment

```bash
# Clone the repository
git clone https://github.com/yourusername/wplace-timelapse.git
cd wplace-timelapse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration Setup

1. **Copy the example configuration**:
   ```bash
   cp config.example.json config.json
   ```

2. **Edit config.json** to set your desired regions and settings:
   ```json
   {
     "timelapses": [
       {
         "slug": "my_region",
         "name": "My Watched Region",
         "coordinates": {
           "xmin": 1000,
           "xmax": 1010,
           "ymin": 800,
           "ymax": 810
         },
         "enabled": true
       }
     ]
   }
   ```

## Finding Tile Coordinates

1. Visit [wplace.live](https://wplace.live)
2. Navigate to your area of interest
3. Open browser Developer Tools (F12)
4. Go to the Network tab
5. Look for tile requests like: `https://backend.wplace.live/files/s0/tiles/X/Y.png`
6. Use those X,Y coordinates to define your region

## Directory Structure

The system will create these directories automatically:

```
wplace-timelapse/
├── backups/           # Raw tile images organized by date/time
├── output/            # Generated timelapse videos  
├── images/            # Sample screenshots for README
├── config.json        # Your configuration
└── timelapse_backup.log  # Application logs
```

## Verification

Test your installation:

```bash
# Using uv
uv run python main.py --help

# Using pip/venv
python main.py --help
```

## Running the System

### Manual Execution
```bash
# Start the automated system (runs continuously)
python main.py

# The system will:
# - Download tiles every 5 minutes
# - Create daily timelapses at midnight
# - Log all activity to timelapse_backup.log
```

### As a Service (Linux)

Create a systemd service file at `/etc/systemd/system/wplace-timelapse.service`:

```ini
[Unit]
Description=WPlace Timelapse System
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/wplace-timelapse
ExecStart=/path/to/wplace-timelapse/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl enable wplace-timelapse
sudo systemctl start wplace-timelapse
sudo systemctl status wplace-timelapse
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t wplace-timelapse .
docker run -v $(pwd)/config.json:/app/config.json -v $(pwd)/output:/app/output wplace-timelapse
```

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   ```bash
   # Ensure directories are writable
   chmod 755 backups output
   ```

2. **Network Errors**:
   - Check internet connection
   - Verify WPlace server is accessible
   - Adjust request_delay in config if getting rate limited

3. **FFmpeg Not Found** (if using custom video operations):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Memory Issues with Large Regions**:
   - Reduce the coordinate range
   - Increase system swap space
   - Monitor with `htop` during operation

### Logs

Check `timelapse_backup.log` for detailed information:
```bash
tail -f timelapse_backup.log
```

### Configuration Validation

The system will validate your configuration on startup and log any issues. Common problems:

- Coordinates outside valid range
- Missing required fields
- Invalid JSON syntax
- Network connectivity to WPlace servers

## Performance Considerations

- **Disk Space**: Videos can be large (50MB+ per day per region)
- **Network**: Downloads ~1-10MB every 5 minutes depending on region size
- **CPU**: Video encoding requires processing power
- **Memory**: Large regions (10x10+ tiles) may need 1GB+ RAM

## Security Notes

- No authentication required for WPlace tile access
- System only downloads public tile data
- Consider firewall rules if running on a server
- Log files may contain timestamps and coordinates