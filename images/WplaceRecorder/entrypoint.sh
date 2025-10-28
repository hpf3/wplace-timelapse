#!/bin/sh
set -e

# Ensure persistent dirs
mkdir -p /data/backups /data/output /data/logs /data/cache

# If user supplied config.json in the volume, use it
if [ -f "/data/config.json" ]; then
  cp /data/config.json /app/src/config.json
fi

# Helpful default if user wants logs persisted
export PYTHONUNBUFFERED=1
cd /app/src

if [ "${SKIP_PREBACKUP_CLEANUP:-0}" != "1" ]; then
  CLEANUP_CMD="python cleanup_backups.py --cache-root /data/cache ${CLEANUP_EXTRA_ARGS:-}"
  echo "Starting cleanup_backups.py in background: ${CLEANUP_CMD}"
  sh -c "${CLEANUP_CMD}" >>/data/logs/cleanup.log 2>&1 &
  CLEANUP_PID=$!
  trap "kill ${CLEANUP_PID} 2>/dev/null || true" INT TERM
fi

exec python main.py
