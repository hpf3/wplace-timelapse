"""
Command line utilities for the historical backfill workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from textwrap import dedent

from .generator import HistoricalGenerator
from .github_client import (
    GitHubError,
    download_assets,
    fetch_recent_releases,
    fetch_release_by_tag,
)
from .models import BackfillRequest, BoundingBox, RegionSpec, parse_iso_datetime
from .releases import scan_local_releases

GITHUB_ARCHIVES_REPO = "https://github.com/murolem/wplace-archives"


@dataclass(frozen=True)
class BackfillContext:
    """Directory layout for historical backfill workspaces."""

    project_root: Path
    data_dir: Path
    archives_dir: Path
    extracted_dir: Path

    @staticmethod
    def from_base_dir(base_dir: Path) -> "BackfillContext":
        base_dir = base_dir.resolve()
        data_dir = base_dir / "data"
        return BackfillContext(
            project_root=base_dir,
            data_dir=data_dir,
            archives_dir=data_dir / "archives",
            extracted_dir=data_dir / "extracted",
        )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def init_project(ctx: BackfillContext) -> None:
    """Create the base directory layout for backfilling."""
    for path in (ctx.project_root, ctx.data_dir, ctx.archives_dir, ctx.extracted_dir):
        path.mkdir(parents=True, exist_ok=True)
        logging.debug("Ensured directory exists: %s", path)

    logging.info("Initialized historical backfill workspace at %s", ctx.project_root)


def plan_backfill(ctx: BackfillContext, slug: str | None) -> None:
    """Summarize the manual steps required to ingest historical data."""
    logging.info("Historical backfill staging area: %s", ctx.project_root)
    slug_hint = slug or "<slug>"
    instructions = dedent(
        f"""
        Next steps:
        1. Visit {GITHUB_ARCHIVES_REPO}/releases and pick the archive matching your timeframe.
        2. Download the `world-<timestamp>` split tarballs into {ctx.archives_dir}.
        3. Concatenate the split files, e.g. `cat world-2024-08-22.tar.gz.* > archive.tar.gz`.
        4. Extract into {ctx.extracted_dir} (`tar -xzf archive.tar.gz -C {ctx.extracted_dir}`).
        5. Run `python -m historical_backfill generate --slug {slug_hint} ...` with your desired timeframe.
        6. Copy the generated `backups/{slug_hint}` sessions into the main timelapse `backups/` directory.
        """
    ).strip()
    for line in instructions.splitlines():
        logging.info(line)


def _log_release_summary(release, prefix: str = "") -> None:
    asset_count = len(release.assets)
    total_size = sum(asset.size for asset in release.assets)
    published = release.published_at.isoformat() if release.published_at else "unknown"
    logging.info(
        "%s%s (published %s, %s assets, %.2f GiB)",
        prefix,
        release.tag,
        published,
        asset_count,
        total_size / (1024**3),
    )


def download_releases(ctx: BackfillContext, args: argparse.Namespace) -> int:
    tag = args.tag or args.release

    try:
        if args.list or not tag:
            count = max(1, args.count)
            releases = fetch_recent_releases(count)
            if not releases:
                logging.warning("No releases returned by GitHub.")
            else:
                logging.info("Latest %s releases:", min(count, len(releases)))
                for release in releases:
                    _log_release_summary(release, prefix="- ")
            if not tag:
                return 0

        release = fetch_release_by_tag(tag)
        _log_release_summary(release, prefix="Selected release: ")

        destination = (args.output_dir or ctx.archives_dir).resolve()
        skip_existing = args.skip_existing
        if args.dry_run:
            logging.info("Dry run: would download %s assets to %s", len(release.assets), destination)
            return 0

        downloaded = download_assets(release, destination, skip_existing=skip_existing)
        logging.info("Downloaded %s assets into %s", len(downloaded), destination)
        return 0
    except GitHubError as exc:
        logging.error("GitHub API error: %s", exc)
        return 1


def _cleanup_archives(releases: set[str], archives_root: Path) -> None:
    if not releases:
        logging.info("No archives to clean up.")
        return

    if not archives_root.exists():
        logging.info("Archive directory %s does not exist; skipping cleanup.", archives_root)
        return

    for tag in sorted(releases):
        archive_dir = archives_root / tag
        if not archive_dir.exists():
            logging.debug("Archive directory %s is already removed.", archive_dir)
            continue
        if not archive_dir.is_dir():
            logging.warning("Expected %s to be a directory; skipping.", archive_dir)
            continue

        logging.info("Removing archive files for %s at %s", tag, archive_dir)
        shutil.rmtree(archive_dir, ignore_errors=True)

def _load_timelapse_entry(config_path: Path, slug_hint: str | None) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    timelapses = payload.get("timelapses") or []
    if not timelapses:
        raise ValueError(f"No timelapse entries found in {config_path}")

    if slug_hint:
        for entry in timelapses:
            if entry.get("slug") == slug_hint:
                return entry
        raise ValueError(f"Slug '{slug_hint}' not found in {config_path}")

    if len(timelapses) == 1:
        return timelapses[0]

    raise ValueError(
        "Configuration contains multiple timelapses; specify --config-slug to pick one."
    )


def _bounding_box_from_entry(entry: dict) -> BoundingBox:
    coords = entry.get("coordinates") or {}
    required = ("xmin", "xmax", "ymin", "ymax")
    missing = [key for key in required if key not in coords]
    if missing:
        raise ValueError(f"Configuration missing coordinate keys: {', '.join(missing)}")

    return BoundingBox(
        xmin=int(coords["xmin"]),
        xmax=int(coords["xmax"]),
        ymin=int(coords["ymin"]),
        ymax=int(coords["ymax"]),
    )


def _bounding_box_from_args(args: argparse.Namespace) -> BoundingBox | None:
    values = {
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
    }
    provided = [value is not None for value in values.values()]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError("All coordinate bounds (xmin/xmax/ymin/ymax) must be provided.")

    return BoundingBox(
        xmin=int(values["xmin"]),
        xmax=int(values["xmax"]),
        ymin=int(values["ymin"]),
        ymax=int(values["ymax"]),
    )


def generate_backfill(
    parser: argparse.ArgumentParser,
    ctx: BackfillContext,
    args: argparse.Namespace,
) -> int:
    try:
        start = parse_iso_datetime(args.start)
        end = parse_iso_datetime(args.end)
    except ValueError as exc:
        parser.error(str(exc))

    interval = timedelta(minutes=args.interval_minutes)
    dest_root = (args.dest_dir or (args.root / "backups")).resolve()
    source_root = (args.source_dir or ctx.extracted_dir).resolve()
    tiles_subdir = Path(args.tiles_subdir) if args.tiles_subdir else None

    config_entry = None
    if args.config:
        try:
            config_entry = _load_timelapse_entry(
                Path(args.config).resolve(),
                args.config_slug or args.slug,
            )
        except ValueError as exc:
            parser.error(str(exc))

    bounding_box = None
    try:
        bounding_box = _bounding_box_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if bounding_box is None:
        if config_entry is None:
            parser.error("Provide coordinate bounds or supply --config to load them.")
        try:
            bounding_box = _bounding_box_from_entry(config_entry)
        except ValueError as exc:
            parser.error(str(exc))

    region_name = (
        args.name
        or (config_entry.get("name") if isinstance(config_entry, dict) else None)
        or args.slug
    )

    region = RegionSpec(
        slug=args.slug,
        name=region_name,
        bounding_box=bounding_box,
    )

    request = BackfillRequest(
        region=region,
        start=start,
        end=end,
        interval=interval,
        dest_root=dest_root,
        source_root=source_root,
        tiles_subdir=tiles_subdir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        deduplicate=args.deduplicate,
    )

    try:
        request.validate()
    except ValueError as exc:
        parser.error(str(exc))

    releases = scan_local_releases(request.source_root, request.tiles_subdir)
    if not releases:
        logging.error(
            "No releases discovered under %s. Run the plan command for download instructions.",
            request.source_root,
        )
        return 1

    generator = HistoricalGenerator(request, releases, logging.getLogger("historical_backfill"))
    summary = generator.run()
    logging.info("Backfill complete: %s", summary.as_dict())

    if args.cleanup_archives:
        archives_root = (args.archives_dir or ctx.archives_dir).resolve()
        if args.dry_run:
            logging.info(
                "Dry run: would remove archives for releases %s under %s",
                ", ".join(sorted(summary.releases_used)),
                archives_root,
            )
        else:
            _cleanup_archives(summary.releases_used, archives_root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Historical backfill tools for the WPlace timelapse project.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for the timelapse project (default: current working directory).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the local workspace layout.")

    plan_parser = subparsers.add_parser(
        "plan", help="Print a checklist for preparing historical archives."
    )
    plan_parser.add_argument(
        "--slug",
        help="Timelapse slug these archives will feed (for documentation only).",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="List or download releases directly from GitHub.",
    )
    download_parser.add_argument(
        "--tag",
        help="Release tag to download.",
    )
    download_parser.add_argument(
        "--release",
        help="Alias for --tag (maintained for compatibility).",
    )
    download_parser.add_argument(
        "--list",
        action="store_true",
        help="Print the latest releases before downloading.",
    )
    download_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of releases to display when using --list (default: 5).",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store downloaded assets (default: <root>/historical_backfill/data/archives).",
    )
    download_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading assets that already exist on disk.",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which assets would be downloaded without writing files.",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Populate backup sessions from extracted archive releases.",
    )
    generate_parser.add_argument(
        "--slug",
        required=True,
        help="Destination slug inside the backups directory.",
    )
    generate_parser.add_argument(
        "--name",
        help="Override the human-readable name for the region.",
    )
    generate_parser.add_argument(
        "--start",
        required=True,
        help="Inclusive start datetime (ISO format, e.g. 2024-08-21T12:00:00Z).",
    )
    generate_parser.add_argument(
        "--end",
        required=True,
        help="Inclusive end datetime (ISO format).",
    )
    generate_parser.add_argument(
        "--interval-minutes",
        type=float,
        default=5.0,
        help="Minutes between synthetic captures (default: 5).",
    )
    generate_parser.add_argument("--xmin", type=int, help="Minimum tile X coordinate.")
    generate_parser.add_argument("--xmax", type=int, help="Maximum tile X coordinate.")
    generate_parser.add_argument("--ymin", type=int, help="Minimum tile Y coordinate.")
    generate_parser.add_argument("--ymax", type=int, help="Maximum tile Y coordinate.")
    generate_parser.add_argument(
        "--config",
        type=Path,
        help="Path to timelapse config JSON to reuse bounding boxes.",
    )
    generate_parser.add_argument(
        "--config-slug",
        help="Slug inside the config file to pull bounding box information from.",
    )
    generate_parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory with extracted releases (default: <root>/historical_backfill/data/extracted).",
    )
    generate_parser.add_argument(
        "--dest-dir",
        type=Path,
        help="Destination backups root (default: <root>/backups).",
    )
    generate_parser.add_argument(
        "--tiles-subdir",
        help="Subdirectory within each release that contains the tile tree (e.g. 'tiles').",
    )
    generate_parser.add_argument(
        "--archives-dir",
        type=Path,
        help="Directory containing downloaded archive parts (default: <root>/historical_backfill/data/archives).",
    )
    generate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files.",
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sessions instead of skipping them.",
    )
    generate_parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="Copy tiles even if the same release snapshot is reused.",
    )
    generate_parser.add_argument(
        "--cleanup-archives",
        action="store_true",
        help="Delete archive files for releases once they are used.",
    )
    generate_parser.set_defaults(deduplicate=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    ctx = BackfillContext.from_base_dir(args.root / "historical_backfill")
    logging.debug("Resolved backfill context: %s", ctx)

    if args.command == "init":
        init_project(ctx)
        return 0
    if args.command == "plan":
        plan_backfill(ctx, slug=args.slug)
        return 0
    if args.command == "download":
        return download_releases(ctx, args)
    if args.command == "generate":
        return generate_backfill(parser, ctx, args)

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
