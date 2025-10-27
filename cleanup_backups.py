#!/usr/bin/env python3
"""Clean backup directories and backfill gaps using historical archives."""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path

from timelapse_backup.cleanup import (
    CleanupManager,
    collect_slug_configs,
    determine_timezone,
    ensure_cache_paths,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prune empty backup directories, detect long capture gaps, and download "
            "historical archives to backfill missing sessions."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the project configuration file (default: config.json)",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        help="Override the backup root directory (default derived from config).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/cache"),
        help="Directory used as temporary cache for archives (default: /data/cache).",
    )
    parser.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Limit processing to specific slugs (repeatable).",
    )
    parser.add_argument(
        "--min-gap-hours",
        type=float,
        default=3.0,
        help="Minimum gap size (in hours) to consider for backfill (default: 3).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        help="Override expected capture interval in minutes (default: config value).",
    )
    parser.add_argument(
        "--timezone",
        help="Timezone name for interpreting backup timestamps (default: system local).",
    )
    parser.add_argument(
        "--release-limit",
        type=int,
        default=1200,
        help="Maximum number of GitHub releases to inspect (default: 1200).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without modifying files.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Retain downloaded archives after processing (default: delete).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def resolve_backup_root(config: dict, override: Path | None, config_path: Path) -> Path:
    if override is not None:
        root = override
    else:
        settings = config.get("global_settings", {})
        configured = settings.get("backup_dir", "backups")
        root = Path(configured)
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    return root


def resolve_interval(config: dict, override: float | None) -> timedelta:
    if override is not None:
        minutes = override
    else:
        settings = config.get("global_settings", {})
        minutes = settings.get("backup_interval_minutes", 5)
    return timedelta(minutes=float(minutes))


def setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(levelname)s %(message)s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    config_path = args.config.resolve()
    if not config_path.exists():
        parser.error(f"Configuration file {config_path} does not exist")

    config = load_config(config_path)
    slug_configs = collect_slug_configs(config)
    if not slug_configs:
        parser.error("No slugs found in configuration; nothing to process")

    backup_root = resolve_backup_root(config, args.backup_root, config_path)
    interval = resolve_interval(config, args.interval_minutes)
    min_gap = timedelta(hours=args.min_gap_hours)
    timezone_info = determine_timezone(args.timezone)
    cache_paths = ensure_cache_paths(args.cache_root)

    manager = CleanupManager(
        backup_root=backup_root,
        slug_configs=slug_configs,
        timezone_info=timezone_info,
        interval=interval,
        min_gap=min_gap,
        cache_paths=cache_paths,
        dry_run=args.dry_run,
        keep_archives=args.keep_archives,
        release_limit=args.release_limit,
    )

    manager.run(args.slugs)


if __name__ == "__main__":
    main()
