"""Scheduling orchestration for the timelapse backup system."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from timelapse_backup.sessions import get_all_sessions, get_session_dirs_for_date


def _get_enabled_modes(backup: Any, timelapse: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Return enabled timelapse modes using the backup helper."""
    return backup.get_enabled_timelapse_modes(timelapse)


def create_daily_timelapse(backup: Any, date: Optional[datetime] = None) -> None:
    """Create timelapse videos from previous day's images for all timelapses."""
    if date is None:
        date = datetime.now() - timedelta(days=1)

    date_str = date.strftime("%Y-%m-%d")
    enabled_timelapses = backup.get_enabled_timelapses()

    backup.logger.info(
        "Creating timelapses for %s for %s timelapses",
        date_str,
        len(enabled_timelapses),
    )

    for timelapse in enabled_timelapses:
        slug = timelapse["slug"]
        name = timelapse["name"]

        session_dirs = get_session_dirs_for_date(backup.backup_dir, slug, date_str)
        for mode in _get_enabled_modes(backup, timelapse):
            mode_name = mode["mode"]
            suffix = mode["suffix"]
            label = f"on {date_str}"
            output_filename = f"{date_str}{suffix}.mp4"

            backup.render_timelapse_from_sessions(
                slug,
                name,
                timelapse,
                session_dirs,
                mode_name,
                suffix,
                output_filename,
                label,
            )

            if mode.get("create_full"):
                full_session_dirs = get_all_sessions(backup.backup_dir, slug)
                backup.render_timelapse_from_sessions(
                    slug,
                    name,
                    timelapse,
                    full_session_dirs,
                    mode_name,
                    suffix,
                    f"full{suffix}.mp4",
                    "across all backups",
                )


def create_full_timelapses(backup: Any) -> None:
    """Create full-history timelapses for modes configured to support them."""
    enabled_timelapses = backup.get_enabled_timelapses()
    backup.logger.info(
        "Creating full-history timelapses for %s timelapses",
        len(enabled_timelapses),
    )

    for timelapse in enabled_timelapses:
        slug = timelapse["slug"]
        name = timelapse["name"]

        enabled_modes = [
            mode for mode in _get_enabled_modes(backup, timelapse) if mode.get("create_full")
        ]
        if not enabled_modes:
            continue

        session_dirs = get_all_sessions(backup.backup_dir, slug)
        for mode in enabled_modes:
            backup.render_timelapse_from_sessions(
                slug,
                name,
                timelapse,
                session_dirs,
                mode["mode"],
                mode["suffix"],
                f"full{mode['suffix']}.mp4",
                "across all backups",
            )


def create_timelapse_for_slug(
    backup: Any,
    slug: str,
    name: str,
    timelapse_config: Dict[str, Any],
    date_str: str,
    *,
    mode_name: str = "normal",
    suffix: str = "",
) -> None:
    """Create timelapse video for a specific timelapse slug and mode."""
    session_dirs = get_session_dirs_for_date(backup.backup_dir, slug, date_str)
    backup.render_timelapse_from_sessions(
        slug,
        name,
        timelapse_config,
        session_dirs,
        mode_name,
        suffix,
        f"{date_str}{suffix}.mp4",
        f"on {date_str}",
    )


def run(backup: Any) -> None:
    """Run the backup system with scheduled tasks."""
    scheduler = BlockingScheduler()

    scheduler.add_job(
        backup.backup_tiles,
        trigger=IntervalTrigger(minutes=backup.backup_interval),
        id="backup_tiles",
        name="Backup Tiles",
        max_instances=1,
    )

    scheduler.add_job(
        backup.create_daily_timelapse,
        trigger=CronTrigger(hour=0, minute=1),
        id="create_timelapse",
        name="Create Daily Timelapse",
        max_instances=1,
    )

    enabled_timelapses = backup.get_enabled_timelapses()
    backup.logger.info("Multi-timelapse backup system started")
    backup.logger.info("Backup interval: %s minutes", backup.backup_interval)
    backup.logger.info("Monitoring %s timelapses:", len(enabled_timelapses))
    for tl in enabled_timelapses:
        coords = tl["coordinates"]
        backup.logger.info(
            "  - '%s' (%s): X(%s-%s), Y(%s-%s)",
            tl["name"],
            tl["slug"],
            coords["xmin"],
            coords["xmax"],
            coords["ymin"],
            coords["ymax"],
        )

    try:
        backup.backup_tiles()
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        backup.logger.info("Timelapse backup system stopped")
        scheduler.shutdown()


__all__ = [
    "create_daily_timelapse",
    "create_full_timelapses",
    "create_timelapse_for_slug",
    "run",
]
