#!/usr/bin/env python3
"""
Identify long runs of placeholder-only backup sessions and move the surplus
directories into a quarantine location for later review.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


LOGGER = logging.getLogger("prune_placeholders")

DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H-%M-%S"


@dataclass(frozen=True)
class SessionDir:
    slug: str
    path: Path
    timestamp: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan backup sessions for placeholder-only runs. Once a run exceeds "
            "the configured threshold, move any additional sessions into a backup "
            "destination instead of deleting them."
        )
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("backups"),
        help="Root directory containing per-slug backups (default: backups/).",
    )
    parser.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        help="Restrict processing to one or more specific slugs (repeatable).",
    )
    parser.add_argument(
        "--placeholder-threshold",
        type=int,
        default=10,
        help="Maximum allowed length of consecutive placeholder-only sessions (default: 10).",
    )
    parser.add_argument(
        "--backup-dest",
        type=Path,
        help=(
            "Directory where trimmed sessions are moved. Defaults to "
            "<backup-root>/_pruned_placeholders."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on how many sessions may be moved in this run (0 means no limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Perform moves. Without this flag, the script only reports what it would do.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def load_slugs(backup_root: Path, selected: Sequence[str] | None) -> List[Path]:
    if not backup_root.exists():
        raise FileNotFoundError(f"Backup root {backup_root} does not exist")

    slug_dirs: List[Path] = []
    if selected:
        for slug in selected:
            slug_dir = backup_root / slug
            if slug_dir.is_dir():
                slug_dirs.append(slug_dir)
            else:
                LOGGER.warning("Skipping unknown slug %s", slug)
    else:
        slug_dirs = [
            path for path in backup_root.iterdir() if path.is_dir()
        ]

    slug_dirs.sort(key=lambda path: path.name)
    return slug_dirs


def iter_session_dirs(slug_dir: Path) -> Iterator[SessionDir]:
    for date_dir in sorted((d for d in slug_dir.iterdir() if d.is_dir()), key=lambda d: d.name):
        try:
            datetime.strptime(date_dir.name, DATE_FMT)
        except ValueError:
            LOGGER.debug("Skipping non-date directory %s", date_dir)
            continue

        for time_dir in sorted((d for d in date_dir.iterdir() if d.is_dir()), key=lambda d: d.name):
            try:
                timestamp = datetime.strptime(f"{date_dir.name} {time_dir.name}", f"{DATE_FMT} {TIME_FMT}")
            except ValueError:
                LOGGER.debug("Skipping non-timestamp directory %s", time_dir)
                continue

            yield SessionDir(slug=slug_dir.name, path=time_dir, timestamp=timestamp)


def is_placeholder_only(session_dir: Path) -> bool:
    has_placeholder = False
    for entry in session_dir.iterdir():
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix == ".png":
            return False
        if suffix == ".placeholder":
            has_placeholder = True
    return has_placeholder


def collect_surplus_sessions(
    sessions: Iterable[SessionDir],
    threshold: int,
) -> List[SessionDir]:
    surplus: List[SessionDir] = []
    run: List[SessionDir] = []

    for session in sessions:
        if is_placeholder_only(session.path):
            run.append(session)
            if len(run) > threshold:
                surplus.append(session)
        else:
            if run:
                LOGGER.debug(
                    "Encountered run of %d placeholder sessions ending at %s",
                    len(run),
                    run[-1].path,
                )
            run.clear()

    if run:
        LOGGER.debug(
            "Final placeholder run of %d session(s) ends at %s",
            len(run),
            run[-1].path,
        )
    return surplus


def ensure_backup_dest(
    base_dest: Path,
    session: SessionDir,
    backup_root: Path,
    create: bool,
) -> Path:
    relative = session.path.relative_to(backup_root)
    target = base_dest / relative
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Target {target} already exists; refusing to overwrite")
    return target


def move_sessions(
    sessions: Sequence[SessionDir],
    backup_root: Path,
    dest_root: Path,
    dry_run: bool,
    limit: int,
) -> int:
    moved = 0
    for session in sessions:
        if limit and moved >= limit:
            LOGGER.info("Move limit reached (%d); stopping", limit)
            break

        try:
            target = ensure_backup_dest(
                dest_root,
                session,
                backup_root,
                create=not dry_run,
            )
        except FileExistsError as exc:
            LOGGER.error("Skipping %s: %s", session.path, exc)
            continue

        LOGGER.info(
            "%s placeholder session %s -> %s",
            "Would move" if dry_run else "Moving",
            session.path,
            target,
        )

        if not dry_run:
            # Dest directories are guaranteed to exist; ensure again in case limit truncated earlier run.
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(session.path), str(target))

        moved += 1

    return moved


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.placeholder_threshold < 1:
        raise ValueError("--placeholder-threshold must be at least 1")

    dry_run = not args.force
    backup_root = args.backup_root.resolve()
    dest_root = (args.backup_dest or (backup_root / "_pruned_placeholders")).resolve()

    LOGGER.info("Scanning backup root %s", backup_root)
    LOGGER.info(
        "Placeholder run threshold: %d (dry run: %s)",
        args.placeholder_threshold,
        "yes" if dry_run else "no",
    )

    slugs = load_slugs(backup_root, args.slugs)
    if not slugs:
        LOGGER.warning("No slugs found to scan")
        return

    total_surplus = 0
    total_moved = 0

    for slug_dir in slugs:
        LOGGER.info("Processing slug %s", slug_dir.name)
        sessions = list(iter_session_dirs(slug_dir))
        if not sessions:
            LOGGER.info("No sessions found under %s", slug_dir)
            continue

        surplus_sessions = collect_surplus_sessions(sessions, args.placeholder_threshold)
        if not surplus_sessions:
            LOGGER.info("No surplus placeholder sessions detected for %s", slug_dir.name)
            continue

        LOGGER.info(
            "Found %d surplus placeholder sessions for %s",
            len(surplus_sessions),
            slug_dir.name,
        )

        moved = move_sessions(
            surplus_sessions,
            backup_root,
            dest_root,
            dry_run=dry_run,
            limit=args.limit,
        )

        total_surplus += len(surplus_sessions)
        total_moved += moved

    LOGGER.info(
        "Run complete. Surplus sessions detected: %d; %s: %d",
        total_surplus,
        "would move" if dry_run else "moved",
        total_moved,
    )


if __name__ == "__main__":
    main()
