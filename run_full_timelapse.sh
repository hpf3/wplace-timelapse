#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 - <<'PY'
from timelapse_backup import TimelapseBackup

backup = TimelapseBackup()
backup.create_full_timelapses()
PY
