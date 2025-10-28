FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/hpf3/wplace-timelapse"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for OpenCV/FFmpeg + git for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 ca-certificates tzdata curl jq git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app/src

WORKDIR /app/src

# Runtime data lives here (bind-mount from host)
VOLUME ["/data"]

# Default timezone (can be overridden at runtime)
ENV TZ=America/Los_Angeles

COPY images/WplaceRecorder/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
