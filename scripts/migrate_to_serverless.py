#!/usr/bin/env python3
"""Convert legacy timelapse backups to the serverless S3 layout."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("wplace.migration")

Coordinate = Tuple[int, int]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from timelapse_backup.config import Config, TimelapseConfig, load_config


# ---------------------------------------------------------------------------
# Optional import of serverless package with fallback implementation
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard
    from wplace_timelapse_serverless.manifest import (  # type: ignore[import]
        DeltaManifest,
        ManifestFailure,
        ManifestPointer,
        ManifestTile,
    )
    from wplace_timelapse_serverless.storage.s3 import (  # type: ignore[import]
        S3Paths,
        S3StorageBackend,
    )
except ImportError:  # pragma: no cover - fallback mirrors serverless
    import hashlib
    import json

    try:  # pragma: no cover - import guard
        import boto3
        from botocore.client import BaseClient  # type: ignore[import]
        from botocore.exceptions import ClientError  # type: ignore[import]
        _BOTO_IMPORT_ERROR: Optional[ImportError] = None
    except ImportError as exc:  # pragma: no cover - graceful failure
        boto3 = None  # type: ignore[assignment]
        BaseClient = Any  # type: ignore[assignment]
        ClientError = Exception  # type: ignore[assignment]
        _BOTO_IMPORT_ERROR = exc

    @dataclass(frozen=True, slots=True)
    class ManifestTile:
        coordinate: Coordinate
        object_key: str
        checksum: str
        size: int

    @dataclass(frozen=True, slots=True)
    class ManifestFailure:
        coordinate: Coordinate
        reason: str

    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def format_timestamp(value: datetime) -> str:
        return _as_utc(value).isoformat().replace("+00:00", "Z")

    def parse_timestamp(value: str) -> datetime:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @dataclass(slots=True)
    class DeltaManifest:
        slug: str
        capture_time: datetime
        previous_manifest: Optional[str]
        tiles: List[ManifestTile]
        failed_tiles: List[ManifestFailure]
        metadata: Dict[str, str]

        def to_dict(self) -> Dict[str, object]:
            return {
                "slug": self.slug,
                "capture_time": format_timestamp(self.capture_time),
                "previous_manifest": self.previous_manifest,
                "tiles": [
                    {
                        "coordinate": list(tile.coordinate),
                        "object_key": tile.object_key,
                        "checksum": tile.checksum,
                        "size": tile.size,
                    }
                    for tile in self.tiles
                ],
                "failed_tiles": [
                    {"coordinate": list(failure.coordinate), "reason": failure.reason} for failure in self.failed_tiles
                ],
                "metadata": self.metadata,
            }

        def to_json(self) -> str:
            return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @dataclass(frozen=True, slots=True)
    class ManifestPointer:
        object_key: str
        capture_time: datetime

    @dataclass(frozen=True, slots=True)
    class S3Paths:
        bucket: str
        tile_prefix: str
        manifest_prefix: str
        latest_suffix: str = "latest.json"

        def tile_key(self, slug: str, checksum: str) -> str:
            return f"{self.tile_prefix.rstrip('/')}/{slug}/{checksum[:2]}/{checksum}.png"

        def manifest_key(self, slug: str, capture_time: datetime) -> str:
            stamp = _as_utc(capture_time).strftime("%Y/%m/%d/%H%M%S")
            return f"{self.manifest_prefix.rstrip('/')}/{slug}/{stamp}.json"

        def latest_key(self, slug: str) -> str:
            return f"{self.manifest_prefix.rstrip('/')}/{slug}/{self.latest_suffix}"

    @dataclass(frozen=True, slots=True)
    class StoredTile:
        tile: ManifestTile
        existed: bool

    class S3StorageBackend:
        def __init__(
            self,
            *,
            bucket: str,
            region: Optional[str] = None,
            tile_prefix: str = "tiles",
            manifest_prefix: str = "manifests",
            latest_suffix: str = "latest.json",
            endpoint_url: Optional[str] = None,
            client: Optional[BaseClient] = None,
        ) -> None:
            if boto3 is None:
                raise ImportError(
                    "boto3 is required to use the S3 migration backend. Install the 'boto3' package first."
                ) from _BOTO_IMPORT_ERROR
            self.paths = S3Paths(
                bucket=bucket,
                tile_prefix=tile_prefix,
                manifest_prefix=manifest_prefix,
                latest_suffix=latest_suffix,
            )
            self.client = client or boto3.client("s3", region_name=region, endpoint_url=endpoint_url)  # type: ignore[union-attr]

        def get_latest_manifest(self, slug: str) -> Optional[ManifestPointer]:
            key = self.paths.latest_key(slug)
            try:
                response = self.client.get_object(Bucket=self.paths.bucket, Key=key)
            except ClientError as exc:  # type: ignore[attr-defined]
                if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                    return None
                raise

            payload = response["Body"].read()
            data = json.loads(payload)
            return ManifestPointer(object_key=data["manifest_key"], capture_time=parse_timestamp(data["capture_time"]))

        def load_manifest(self, pointer: ManifestPointer) -> DeltaManifest:
            try:
                response = self.client.get_object(Bucket=self.paths.bucket, Key=pointer.object_key)
            except ClientError as exc:  # type: ignore[attr-defined]
                raise FileNotFoundError(f"Manifest object {pointer.object_key} not found") from exc
            payload = response["Body"].read().decode("utf-8")
            data = json.loads(payload)
            tiles = [
                ManifestTile(coordinate=tuple(entry["coordinate"]), object_key=entry["object_key"], checksum=entry["checksum"], size=int(entry["size"]))  # type: ignore[arg-type]
                for entry in data.get("tiles", [])
            ]
            failures = [
                ManifestFailure(coordinate=tuple(entry["coordinate"]), reason=entry["reason"])  # type: ignore[arg-type]
                for entry in data.get("failed_tiles", [])
            ]
            metadata = {str(k): str(v) for k, v in data.get("metadata", {}).items()}
            return DeltaManifest(
                slug=str(data["slug"]),
                capture_time=parse_timestamp(str(data["capture_time"])),
                previous_manifest=data.get("previous_manifest"),
                tiles=tiles,
                failed_tiles=failures,
                metadata=metadata,
            )

        def store_tile(
            self,
            slug: str,
            capture_time: datetime,
            coord: Coordinate,
            payload: bytes,
            checksum: str,
        ) -> StoredTile:
            digest = checksum or hashlib.sha256(payload).hexdigest()
            object_key = self.paths.tile_key(slug, digest)

            existed = self._object_exists(object_key)
            if not existed:
                self.client.put_object(
                    Bucket=self.paths.bucket,
                    Key=object_key,
                    Body=payload,
                    ContentType="image/png",
                    Metadata={
                        "slug": slug,
                        "capture_time": _as_utc(capture_time).isoformat(),
                        "coordinate": f"{coord[0]},{coord[1]}",
                        "checksum": digest,
                    },
                )

            tile = ManifestTile(coordinate=coord, object_key=object_key, checksum=digest, size=len(payload))
            return StoredTile(tile=tile, existed=existed)

        def write_manifest(self, manifest: DeltaManifest) -> ManifestPointer:
            key = self.paths.manifest_key(manifest.slug, manifest.capture_time)
            self.client.put_object(
                Bucket=self.paths.bucket,
                Key=key,
                Body=manifest.to_json().encode("utf-8"),
                ContentType="application/json",
            )
            return ManifestPointer(object_key=key, capture_time=manifest.capture_time)

        def update_latest_manifest(self, slug: str, pointer: ManifestPointer) -> None:
            payload = json.dumps(
                {
                    "manifest_key": pointer.object_key,
                    "capture_time": format_timestamp(pointer.capture_time),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            key = self.paths.latest_key(slug)
            self.client.put_object(
                Bucket=self.paths.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
            )

        def _object_exists(self, key: str) -> bool:
            try:
                self.client.head_object(Bucket=self.paths.bucket, Key=key)
                return True
            except ClientError as exc:  # type: ignore[attr-defined]
                code = exc.response.get("Error", {}).get("Code")
                if code in {"404", "NoSuchKey", "NotFound", "403", "AccessDenied"}:
                    return False
                raise


@dataclass
class SessionStats:
    """Lightweight stats emitted for each migrated session."""

    slug: str
    session_dir: Path
    capture_time: datetime
    total_tiles: int
    png_tiles: int = 0
    placeholder_tiles: int = 0
    missing_tiles: int = 0
    changed_tiles: int = 0
    deduplicated_tiles: int = 0
    uploaded_tiles: int = 0
    reused_tiles: int = 0
    recovered_tiles: int = 0
    failures: int = 0
    duration_seconds: float = 0.0


def _env_truthy(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [slug.strip() for slug in value.split(",")]
    return [item for item in items if item]


def parse_args(env: Mapping[str, str] = os.environ) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to legacy config.json")
    parser.add_argument("--backup-root", type=Path, default=None, help="Override backup directory")
    parser.add_argument("--slug", action="append", default=None, help="Restrict migration to the given slug(s)")
    parser.add_argument("--bucket", default=None, help="Target S3 bucket name")
    parser.add_argument("--region", default=None, help="S3 region (optional for some providers)")
    parser.add_argument("--endpoint-url", default=None, help="Custom S3 endpoint URL (for R2/MinIO/etc.)")
    parser.add_argument("--tile-prefix", default=None, help="S3 prefix for tile objects")
    parser.add_argument("--manifest-prefix", default=None, help="S3 prefix for manifest objects")
    parser.add_argument("--latest-suffix", default=None, help="Suffix used for latest manifest pointers")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from the latest manifest in S3")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Scan and report without uploading")
    parser.add_argument("--log-level", default=None, help="Logging level (DEBUG, INFO, ...)")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(env.get("WPLACE_CONFIG_PATH", "config.json"))

    if args.backup_root is None:
        backup_env = env.get("WPLACE_BACKUP_ROOT")
        if backup_env:
            args.backup_root = Path(backup_env)

    if args.slug is None:
        args.slug = _env_list(env.get("WPLACE_SLUGS"))

    if args.bucket is None:
        args.bucket = env.get("WPLACE_BUCKET_NAME")

    if args.region is None:
        args.region = env.get("WPLACE_REGION")

    if args.endpoint_url is None:
        args.endpoint_url = env.get("WPLACE_ENDPOINT_URL")

    if args.tile_prefix is None:
        args.tile_prefix = env.get("WPLACE_TILE_PREFIX", "tiles")

    if args.manifest_prefix is None:
        args.manifest_prefix = env.get("WPLACE_MANIFEST_PREFIX", "manifests")

    if args.latest_suffix is None:
        args.latest_suffix = env.get("WPLACE_LATEST_SUFFIX", "latest.json")

    if args.resume is None:
        resume_env = _env_truthy(env.get("WPLACE_MIGRATE_RESUME"))
        args.resume = bool(resume_env) if resume_env is not None else False

    if args.dry_run is None:
        dry_env = _env_truthy(env.get("WPLACE_MIGRATE_DRY_RUN"))
        args.dry_run = bool(dry_env) if dry_env is not None else False

    if args.log_level is None:
        args.log_level = env.get("WPLACE_LOG_LEVEL", "INFO")

    return args


def parse_session_timestamp(date_name: str, time_name: str) -> Optional[datetime]:
    try:
        value = datetime.strptime(f"{date_name}T{time_name}", "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None
    return value.replace(tzinfo=timezone.utc)


def discover_sessions(slug_dir: Path) -> List[Tuple[datetime, Path]]:
    """Return timestamp-sorted session directories for a slug."""
    sessions: List[Tuple[datetime, Path]] = []
    if not slug_dir.exists():
        return sessions

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    time_pattern = re.compile(r"^\d{2}-\d{2}-\d{2}$")

    for date_dir in sorted((p for p in slug_dir.iterdir() if p.is_dir()), key=lambda item: item.name):
        if not date_pattern.match(date_dir.name):
            continue
        for time_dir in sorted((p for p in date_dir.iterdir() if p.is_dir()), key=lambda item: item.name):
            if not time_pattern.match(time_dir.name):
                continue
            timestamp = parse_session_timestamp(date_dir.name, time_dir.name)
            if timestamp is None:
                LOGGER.warning("Skipping session with unparsable timestamp: %s", time_dir)
                continue
            sessions.append((timestamp, time_dir))

    sessions.sort(key=lambda entry: entry[0])
    return sessions


def resolve_tile_from_history(coord: Coordinate, prior_sessions: Sequence[Path]) -> Optional[Path]:
    """Locate a PNG tile for a coordinate across prior sessions."""
    filename = f"{coord[0]}_{coord[1]}.png"
    for session_dir in reversed(prior_sessions):
        candidate = session_dir / filename
        if candidate.exists():
            return candidate
    return None


def rebuild_tile_state(
    storage: "S3StorageBackend",
    pointer: "ManifestPointer",
    coordinates: Iterable[Coordinate],
) -> Dict[Coordinate, "ManifestTile"]:
    """Traverse manifests to recover the latest tile per coordinate."""
    tile_state: Dict[Coordinate, ManifestTile] = {}
    expected = set(coordinates)
    visited: set[str] = set()

    current = pointer
    while current and current.object_key not in visited:
        manifest = storage.load_manifest(current)
        visited.add(current.object_key)

        for tile in manifest.tiles:
            coordinate = (int(tile.coordinate[0]), int(tile.coordinate[1]))
            if coordinate not in tile_state:
                tile_state[coordinate] = tile

        if tile_state.keys() >= expected:
            break

        previous_key = manifest.previous_manifest
        if not previous_key:
            break

        current = ManifestPointer(object_key=previous_key, capture_time=manifest.capture_time)

    return tile_state


def load_manifest_chain(
    storage: "S3StorageBackend",
    pointer: Optional["ManifestPointer"],
) -> List[Tuple["ManifestPointer", "DeltaManifest"]]:
    """Walk the manifest chain from latest to oldest."""
    chain: List[Tuple[ManifestPointer, DeltaManifest]] = []
    visited: set[str] = set()
    current = pointer

    while current and current.object_key not in visited:
        manifest = storage.load_manifest(current)
        chain.append((current, manifest))
        visited.add(current.object_key)

        previous_key = manifest.previous_manifest
        if not previous_key:
            break
        current = ManifestPointer(object_key=previous_key, capture_time=manifest.capture_time)

    return chain


def process_session(
    *,
    slug: str,
    session_dir: Path,
    capture_time: datetime,
    coordinates: Sequence[Coordinate],
    storage: "S3StorageBackend",
    latest_tiles: Dict[Coordinate, "ManifestTile"],
    checksum_cache: Dict[str, "ManifestTile"],
    prior_sessions: Sequence[Path],
    dry_run: bool,
) -> Tuple[List["ManifestTile"], List["ManifestFailure"], SessionStats]:
    """Upload session tiles and build the manifest delta."""
    start = perf_counter()
    changed_tiles: List[ManifestTile] = []
    failures: List[ManifestFailure] = []
    stats = SessionStats(
        slug=slug,
        session_dir=session_dir,
        capture_time=capture_time,
        total_tiles=len(coordinates),
    )

    for coord in coordinates:
        filename = f"{coord[0]}_{coord[1]}"
        png_path = session_dir / f"{filename}.png"
        placeholder_path = session_dir / f"{filename}.placeholder"

        if png_path.exists():
            stats.png_tiles += 1
            try:
                payload = png_path.read_bytes()
            except OSError as exc:
                failures.append(ManifestFailure(coordinate=coord, reason=f"read_error:{exc}"))
                continue

            digest = sha256(payload).hexdigest()
            previous_tile = latest_tiles.get(coord)
            if previous_tile and previous_tile.checksum == digest:
                stats.deduplicated_tiles += 1
                continue

            cached_tile = checksum_cache.get(digest)
            if cached_tile:
                stats.reused_tiles += 1
                tile_ref = ManifestTile(
                    coordinate=coord,
                    object_key=cached_tile.object_key,
                    checksum=cached_tile.checksum,
                    size=cached_tile.size,
                )
            else:
                if dry_run:
                    object_key = storage.paths.tile_key(slug, digest)  # type: ignore[attr-defined]
                    tile_ref = ManifestTile(
                        coordinate=coord,
                        object_key=object_key,
                        checksum=digest,
                        size=len(payload),
                    )
                else:
                    stored = storage.store_tile(  # type: ignore[attr-defined]
                        slug=slug,
                        capture_time=capture_time,
                        coord=coord,
                        payload=payload,
                        checksum=digest,
                    )
                    tile_ref = stored.tile
                    if not stored.existed:
                        stats.uploaded_tiles += 1

                checksum_cache[digest] = tile_ref

            changed_tiles.append(tile_ref)
            latest_tiles[coord] = tile_ref
            continue

        if placeholder_path.exists():
            stats.placeholder_tiles += 1
            if coord in latest_tiles:
                stats.deduplicated_tiles += 1
                continue

            fallback = resolve_tile_from_history(coord, prior_sessions)
            if fallback is not None and fallback.exists():
                try:
                    payload = fallback.read_bytes()
                except OSError as exc:
                    failures.append(ManifestFailure(coordinate=coord, reason=f"placeholder_read_error:{exc}"))
                    continue

                digest = sha256(payload).hexdigest()
                cached_tile = checksum_cache.get(digest)
                if cached_tile:
                    latest_tiles[coord] = cached_tile
                    stats.deduplicated_tiles += 1
                    continue

                if dry_run:
                    object_key = storage.paths.tile_key(slug, digest)  # type: ignore[attr-defined]
                    tile_ref = ManifestTile(
                        coordinate=coord,
                        object_key=object_key,
                        checksum=digest,
                        size=len(payload),
                    )
                else:
                    stored = storage.store_tile(  # type: ignore[attr-defined]
                        slug=slug,
                        capture_time=capture_time,
                        coord=coord,
                        payload=payload,
                        checksum=digest,
                    )
                    tile_ref = stored.tile
                    if not stored.existed:
                        stats.uploaded_tiles += 1

                checksum_cache[digest] = tile_ref
                latest_tiles[coord] = tile_ref
                stats.recovered_tiles += 1
                stats.deduplicated_tiles += 1
                continue

            failures.append(ManifestFailure(coordinate=coord, reason="missing_placeholder_source"))
            continue

        stats.missing_tiles += 1
        failures.append(ManifestFailure(coordinate=coord, reason="missing_tile"))

    stats.changed_tiles = len(changed_tiles)
    stats.failures = len(failures)
    stats.duration_seconds = perf_counter() - start
    return changed_tiles, failures, stats


def migrate_slug(
    *,
    slug: str,
    slug_config: TimelapseConfig,
    storage: "S3StorageBackend",
    backup_root: Path,
    resume: bool,
    dry_run: bool,
) -> List[SessionStats]:
    coordinates = list(slug_config.coordinates.iter_tiles())
    slug_dir = backup_root / slug
    sessions = discover_sessions(slug_dir)
    if not sessions:
        LOGGER.warning("No sessions found for slug %s under %s", slug, backup_root)
        return []

    existing_pointer = storage.get_latest_manifest(slug)
    existing_chain: List[Tuple[ManifestPointer, DeltaManifest]] = []
    existing_latest_capture: Optional[datetime] = existing_pointer.capture_time if existing_pointer else None
    existing_oldest_capture: Optional[datetime] = None

    if existing_pointer:
        existing_chain = load_manifest_chain(storage, existing_pointer)
        if existing_chain:
            existing_oldest_capture = existing_chain[-1][1].capture_time

    session_min = sessions[0][0]
    session_max = sessions[-1][0]

    mode = "fresh"
    if existing_pointer:
        if existing_oldest_capture and session_max < existing_oldest_capture:
            mode = "prepend"
            LOGGER.info(
                "Prepending %d sessions for %s before existing earliest capture (%s)",
                len(sessions),
                slug,
                existing_oldest_capture.isoformat(),
            )
        elif existing_latest_capture and session_min > existing_latest_capture:
            mode = "append"
            if not resume and not dry_run:
                LOGGER.warning(
                    "Slug %s already has manifests in S3. Re-run with --resume to append newer sessions.",
                    slug,
                )
                return []
            LOGGER.info(
                "Appending %d sessions for %s starting after %s",
                len(sessions),
                slug,
                existing_latest_capture.isoformat(),
            )
        else:
            if not resume and not dry_run:
                LOGGER.warning(
                    "Slug %s has existing manifests overlapping requested sessions. Use --resume to merge.",
                    slug,
                )
                return []
            mode = "append"
            LOGGER.info(
                "Merging sessions for %s with existing data (overlapping capture times).",
                slug,
            )
    else:
        if resume:
            LOGGER.info("No existing manifest for %s, starting fresh despite --resume.", slug)

    latest_tiles: Dict[Coordinate, ManifestTile] = {}
    checksum_cache: Dict[str, ManifestTile] = {}
    current_pointer: Optional[ManifestPointer] = None
    last_capture_time: Optional[datetime] = None
    latest_known_capture: Optional[datetime] = existing_latest_capture
    newest_prepend_pointer: Optional[ManifestPointer] = None

    if mode == "append":
        current_pointer = existing_pointer
        last_capture_time = existing_latest_capture
        if existing_pointer:
            latest_tiles = rebuild_tile_state(storage, existing_pointer, coordinates)
            checksum_cache = {tile.checksum: tile for tile in latest_tiles.values()}
    # mode fresh and prepend start with empty state

    migrated_stats: List[SessionStats] = []
    prior_sessions: List[Path] = []

    for capture_time, session_dir in sessions:
        if last_capture_time and capture_time <= last_capture_time:
            LOGGER.debug("Skipping %s (already processed)", session_dir)
            continue

        changed_tiles, failures, stats = process_session(
            slug=slug,
            session_dir=session_dir,
            capture_time=capture_time,
            coordinates=coordinates,
            storage=storage,
            latest_tiles=latest_tiles,
            checksum_cache=checksum_cache,
            prior_sessions=prior_sessions,
            dry_run=dry_run,
        )

        metadata = {
            "source_session_dir": str(session_dir.relative_to(backup_root)),
            "source_path": str(session_dir),
            "total_tiles": str(stats.total_tiles),
            "png_tiles": str(stats.png_tiles),
            "placeholder_tiles": str(stats.placeholder_tiles),
            "missing_tiles": str(stats.missing_tiles),
            "changed_tiles": str(stats.changed_tiles),
            "deduplicated_tiles": str(stats.deduplicated_tiles),
            "uploaded_tiles": str(stats.uploaded_tiles),
            "reused_tiles": str(stats.reused_tiles),
            "recovered_tiles": str(stats.recovered_tiles),
            "failed_tiles": str(stats.failures),
            "duration_seconds": f"{stats.duration_seconds:.3f}",
        }

        if dry_run:
            LOGGER.info(
                "[DRY-RUN] %s @ %s -> %d changes, %d dedup, %d failures",
                slug,
                capture_time.isoformat(),
                stats.changed_tiles,
                stats.deduplicated_tiles,
                stats.failures,
            )
        else:
            manifest = DeltaManifest(
                slug=slug,
                capture_time=capture_time,
                previous_manifest=current_pointer.object_key if current_pointer else None,
                tiles=changed_tiles,
                failed_tiles=failures,
                metadata=metadata,
            )
            pointer = storage.write_manifest(manifest)
            if latest_known_capture is None or capture_time >= latest_known_capture:
                storage.update_latest_manifest(slug, pointer)
                latest_known_capture = capture_time
            current_pointer = pointer
            if mode == "prepend":
                newest_prepend_pointer = pointer
            last_capture_time = capture_time
            LOGGER.info(
                "Uploaded manifest %s (%d tiles, %d failures)",
                pointer.object_key,
                stats.changed_tiles,
                stats.failures,
            )

        migrated_stats.append(stats)
        prior_sessions.append(session_dir)

    if mode == "prepend" and newest_prepend_pointer and existing_chain:
        tail_pointer, tail_manifest = existing_chain[-1]
        if dry_run:
            LOGGER.info(
                "[DRY-RUN] Would update manifest %s to reference new predecessor %s",
                tail_pointer.object_key,
                newest_prepend_pointer.object_key,
            )
        else:
            previous_before = tail_manifest.previous_manifest
            tail_manifest.previous_manifest = newest_prepend_pointer.object_key
            storage.write_manifest(tail_manifest)
            LOGGER.info(
                "Updated earliest manifest %s previous pointer from %s to %s",
                tail_pointer.object_key,
                previous_before,
                newest_prepend_pointer.object_key,
            )

    return migrated_stats


def load_timelapse_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_config(config_path)


def main() -> None:
    args = parse_args()

    if not args.bucket:
        raise ValueError("Bucket name must be provided via --bucket or WPLACE_BUCKET_NAME")

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    config = load_timelapse_config(args.config)
    backup_root = Path(args.backup_root) if args.backup_root else Path(config.global_settings.backup_dir)

    slug_map: Dict[str, TimelapseConfig] = {cfg.slug: cfg for cfg in config.timelapses if cfg.enabled}
    requested_slugs = args.slug or list(slug_map.keys())
    missing_slugs = [slug for slug in requested_slugs if slug not in slug_map]
    if missing_slugs:
        raise ValueError(f"Unknown slug(s) requested: {', '.join(sorted(missing_slugs))}")

    storage = S3StorageBackend(  # type: ignore[call-arg]
        bucket=args.bucket,
        region=args.region,
        tile_prefix=args.tile_prefix,
        manifest_prefix=args.manifest_prefix,
        latest_suffix=args.latest_suffix,
        endpoint_url=args.endpoint_url,
    )

    overall_stats: List[SessionStats] = []
    for slug in requested_slugs:
        LOGGER.info("Migrating slug %s", slug)
        stats = migrate_slug(
            slug=slug,
            slug_config=slug_map[slug],
            storage=storage,
            backup_root=backup_root,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        overall_stats.extend(stats)

    migrated_sessions = len(overall_stats)
    changed_tiles = sum(stat.changed_tiles for stat in overall_stats)
    uploaded_tiles = sum(stat.uploaded_tiles for stat in overall_stats)

    LOGGER.info(
        "Migration complete: %d sessions, %d changed tiles, %d uploads%s",
        migrated_sessions,
        changed_tiles,
        uploaded_tiles,
        " (dry-run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
