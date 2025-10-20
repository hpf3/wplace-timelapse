#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

# Historical backfill automation for the Nemo region.
# Downloads relevant archive releases, extracts the 3x3 tile grid, and
# generates placeholder-backed sessions at a 6-minute cadence.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

START_ISO="${HF_START_ISO:-2025-08-25T00:00:00Z}"
END_ISO="${HF_END_ISO:-2025-10-13T11:00:00Z}"
SLUG="${HF_SLUG:-nemo-old}"
REGION_NAME="${HF_REGION_NAME:-Point Nemo Historical}"
INTERVAL_MINUTES="${HF_INTERVAL_MINUTES:-6}"
CONFIG_PATH="${HF_CONFIG_PATH:-${PROJECT_ROOT}/config.json}"
CONFIG_SLUG="${HF_CONFIG_SLUG:-nemo}"

CACHE_ROOT="${HF_CACHE_ROOT:-/data/cache}"
ARCHIVES_ROOT="${HF_ARCHIVES_ROOT:-${CACHE_ROOT}/archives}"
EXTRACTED_ROOT="${HF_EXTRACTED_ROOT:-${CACHE_ROOT}/extracted}"
DEST_BACKUPS="${HF_DEST_BACKUPS:-/data/backups}"

KEEP_ARCHIVES="${HF_KEEP_ARCHIVES:-0}"
DRY_RUN="${HF_DRY_RUN:-0}"
OVERWRITE="${HF_OVERWRITE:-0}"

X_VALUES=(293 294 295)
Y_VALUES=(1253 1254 1255)

log() {
  local level="$1"; shift
  printf '%s [%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "$*" >&2
}

ensure_directories() {
  local fallback_cache="${PROJECT_ROOT}/historical_backfill/data"
  local fallback_backups="${PROJECT_ROOT}/backups"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log INFO "DRY_RUN: would ensure directories ${ARCHIVES_ROOT}, ${EXTRACTED_ROOT}, ${DEST_BACKUPS}"
    return
  fi

  if ! mkdir -p "${ARCHIVES_ROOT}" "${EXTRACTED_ROOT}"; then
    log WARN "Falling back to project-local cache under ${fallback_cache}"
    CACHE_ROOT="${fallback_cache}"
    ARCHIVES_ROOT="${CACHE_ROOT}/archives"
    EXTRACTED_ROOT="${CACHE_ROOT}/extracted"
    mkdir -p "${ARCHIVES_ROOT}" "${EXTRACTED_ROOT}"
  fi

  if ! mkdir -p "${DEST_BACKUPS}"; then
    log WARN "Falling back to project-local backups under ${fallback_backups}"
    DEST_BACKUPS="${fallback_backups}"
    mkdir -p "${DEST_BACKUPS}"
  fi
}

prune_archives_root() {
  if [[ "${KEEP_ARCHIVES}" -eq 1 ]]; then
    return
  fi
  if [[ ! -d "${ARCHIVES_ROOT}" ]]; then
    return
  fi

  local -a dirs=("${ARCHIVES_ROOT}"/*)
  if [[ ${#dirs[@]} -eq 0 ]]; then
    return
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log INFO "DRY_RUN: would remove ${#dirs[@]} existing archive directories before next download"
    return
  fi

  log INFO "Clearing ${#dirs[@]} archive directories to keep disk usage low"
  for dir in "${dirs[@]}"; do
    rm -rf "${dir}"
  done
}

ensure_workspace() {
  log INFO "Ensuring historical_backfill workspace exists"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log INFO "DRY_RUN enabled: skipping workspace initialization"
  else
    python3 -m historical_backfill init >/dev/null
  fi
}

map_releases() {
  log INFO "Resolving release list between ${START_ISO} and ${END_ISO}"
  HF_START_ISO="${START_ISO}" HF_END_ISO="${END_ISO}" python3 - <<'PY'
import os
import sys
from datetime import datetime

from historical_backfill.github_client import fetch_recent_releases
from historical_backfill.releases import parse_release_timestamp

start_raw = os.environ["HF_START_ISO"].replace("Z", "+00:00")
end_raw = os.environ["HF_END_ISO"].replace("Z", "+00:00")

start = datetime.fromisoformat(start_raw)
end = datetime.fromisoformat(end_raw)

limit = 4000
releases = fetch_recent_releases(limit)

records = []
for release in releases:
    capture = parse_release_timestamp(release.tag)
    if capture is None:
        continue
    if start <= capture <= end:
        records.append((capture, release.tag))

if not records:
    raise SystemExit("No releases found in the requested window.")

records.sort()
if records[0][0] > start:
    print(
        f"Warning: earliest release ({records[0][1]}) is later than start window {start.isoformat()}",
        file=sys.stderr,
    )

for capture, tag in records:
    print(f"{capture.isoformat()}Z {tag}")
PY
}

download_release() {
  local tag="$1"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log INFO "DRY_RUN: would download ${tag}"
    return
  fi
  log INFO "Downloading release ${tag}"
  python3 -m historical_backfill download \
    --tag "${tag}" \
    --skip-existing \
    --output-dir "${ARCHIVES_ROOT}" \
    >/dev/null
}

extract_tiles() {
  local tag="$1"
  local archive_dir="${ARCHIVES_ROOT}/${tag}"
  local extract_dir="${EXTRACTED_ROOT}/${tag}"
  local sample="${extract_dir}/${X_VALUES[0]}/${Y_VALUES[0]}.png"

  if [[ -f "${sample}" ]]; then
    log INFO "Tiles already extracted for ${tag}, skipping extraction"
    if [[ "${KEEP_ARCHIVES}" -ne 1 && -d "${archive_dir}" ]]; then
      log INFO "Removing leftover archive directory for ${tag}"
      rm -rf "${archive_dir}"
    fi
    return
  fi

  if [[ ! -d "${archive_dir}" ]]; then
    log WARN "Archive directory ${archive_dir} not found; skipping extraction"
    return
  fi

  local -a parts=("${archive_dir}"/*.tar.gz.*)
  if [[ ${#parts[@]} -eq 0 ]]; then
    log ERROR "No archive parts found in ${archive_dir}"
    return 1
  fi

  local part_name
  part_name="$(basename "${parts[0]}")"
  local base="${part_name%.tar.gz.*}"

  mkdir -p "${extract_dir}"

  local -a targets=()
  for x in "${X_VALUES[@]}"; do
    mkdir -p "${extract_dir}/${x}"
    for y in "${Y_VALUES[@]}"; do
      targets+=("${base}/${x}/${y}.png")
    done
  done

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log INFO "DRY_RUN: would extract tiles for ${tag}"
  else
    log INFO "Extracting tiles for ${tag}"
    if ! cat "${parts[@]}" | tar xzf - \
      --strip-components=1 \
      --ignore-failed-read \
      -C "${extract_dir}" \
      "${targets[@]}"; then
        log WARN "Extraction reported warnings for ${tag}"
    fi
    local count
    count="$(find "${extract_dir}" -maxdepth 2 -name '*.png' | wc -l | tr -d ' ')"
    log INFO "Available tiles for ${tag}: ${count}"
  fi

  if [[ "${KEEP_ARCHIVES}" -ne 1 ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      log INFO "DRY_RUN: would remove archive directory ${archive_dir}"
    else
      log INFO "Deleting archive parts for ${tag}"
      rm -rf "${archive_dir}"
    fi
  fi
}

generate_frames() {
  local -a args=(
    --slug "${SLUG}"
    --name "${REGION_NAME}"
    --start "${START_ISO}"
    --end "${END_ISO}"
    --interval-minutes "${INTERVAL_MINUTES}"
    --config "${CONFIG_PATH}"
    --config-slug "${CONFIG_SLUG}"
    --source-dir "${EXTRACTED_ROOT}"
    --dest-dir "${DEST_BACKUPS}"
    --cleanup-archives
    --archives-dir "${ARCHIVES_ROOT}"
  )

  if [[ "${OVERWRITE}" -eq 1 ]]; then
    args+=(--overwrite)
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log INFO "DRY_RUN: would invoke generator with: ${args[*]}"
  else
    log INFO "Generating frames for ${SLUG} from ${START_ISO} to ${END_ISO}"
    python3 -m historical_backfill generate "${args[@]}"
  fi
}

main() {
  ensure_workspace
  ensure_directories

  mapfile -t RELEASE_LINES < <(map_releases)
  log INFO "Found ${#RELEASE_LINES[@]} releases covering requested window"

  for line in "${RELEASE_LINES[@]}"; do
    [[ -z "${line}" ]] && continue
    local capture tag
    capture="${line%% *}"
    tag="${line##* }"
    log INFO "Processing release ${tag} (capture ${capture})"
    prune_archives_root
    download_release "${tag}"
    extract_tiles "${tag}"
  done

  generate_frames
  log INFO "Nemo historical backfill complete"
}

main "$@"
