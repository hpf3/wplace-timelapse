import io
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timelapse_backup.cleanup import (  # noqa: E402
    BoundingBox,
    CleanupManager,
    ReleaseCandidate,
    SlugConfig,
    collect_sessions,
    determine_timezone,
    ensure_cache_paths,
    find_gaps,
    prune_empty_directories,
    write_tiles_from_archive,
)
from historical_backfill.github_client import RemoteAsset, RemoteRelease  # noqa: E402


def test_prune_empty_directories(tmp_path: Path) -> None:
    root = tmp_path / "backups" / "slug"
    nested = root / "2024-01-01" / "00-00-00"
    nested.mkdir(parents=True)
    (nested / "dummy.txt").write_text("data")
    empty_dir = root / "2024-01-02" / "01-00-00"
    empty_dir.mkdir(parents=True)

    removed = prune_empty_directories(root)

    assert empty_dir in removed
    assert not empty_dir.exists()
    assert nested.exists()


def build_session_tree(base: Path, timestamps: list[datetime]) -> None:
    for ts in timestamps:
        date_dir = base / ts.strftime("%Y-%m-%d")
        session_dir = date_dir / ts.strftime("%H-%M-%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "tile.png").write_bytes(b"tile")


def test_collect_sessions_and_find_gaps(tmp_path: Path) -> None:
    tz = determine_timezone("UTC")
    slug_dir = tmp_path / "backups" / "slug"
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=tz)
    timestamps = [
        base_time,
        base_time + timedelta(hours=1),
        base_time + timedelta(hours=5),
    ]
    build_session_tree(slug_dir, timestamps)

    sessions = collect_sessions(slug_dir, "slug", tz)
    assert len(sessions) == len(timestamps)

    interval = timedelta(minutes=60)
    min_gap = timedelta(hours=3)
    gaps = find_gaps(sessions, interval, min_gap)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.start == timestamps[1] + interval
    assert gap.end == timestamps[-1]


def test_write_tiles_from_split_archive(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives" / "sample"
    archive_dir.mkdir(parents=True)

    with io.BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            data = b"content"
            info = tarfile.TarInfo(name="sample/1/2.png")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buffer.seek(0)
        payload = buffer.read()
        half = len(payload) // 2
        (archive_dir / "sample.tar.gz.00").write_bytes(payload[:half])
        (archive_dir / "sample.tar.gz.01").write_bytes(payload[half:])

    session_dir = tmp_path / "session"
    extracted, missing = write_tiles_from_archive(
        archive_dir=archive_dir,
        session_dir=session_dir,
        coords=[(1, 2)],
        release_tag="sample",
    )

    assert extracted == 1
    assert missing == 0
    tile_path = session_dir / "1_2.png"
    assert tile_path.exists()


def test_cleanup_reuses_single_download(tmp_path: Path, monkeypatch) -> None:
    tz = determine_timezone("UTC")
    backup_root = tmp_path / "backups"

    def make_existing_session(slug: str, when: datetime) -> None:
        session_dir = (
            backup_root
            / slug
            / when.strftime("%Y-%m-%d")
            / when.strftime("%H-%M-%S")
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "existing.txt").write_text("keep")

    slug_configs = {
        "slug-a": SlugConfig("slug-a", "Slug A", BoundingBox(0, 0, 0, 0)),
        "slug-b": SlugConfig("slug-b", "Slug B", BoundingBox(1, 1, 1, 1)),
    }

    start_a = datetime(2024, 1, 1, 0, 0, tzinfo=tz)
    end_a = datetime(2024, 1, 1, 5, 0, tzinfo=tz)
    start_b = datetime(2024, 1, 1, 0, 30, tzinfo=tz)
    end_b = datetime(2024, 1, 1, 5, 30, tzinfo=tz)
    make_existing_session("slug-a", start_a)
    make_existing_session("slug-a", end_a)
    make_existing_session("slug-b", start_b)
    make_existing_session("slug-b", end_b)

    release_tag = "test-release"
    capture_time = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    release = RemoteRelease(
        tag=release_tag,
        name="Test release",
        published_at=None,
        assets=(
            RemoteAsset(
                name=f"{release_tag}.tar.gz",
                size=1024,
                download_url="https://example.com/archive.tar.gz",
                content_type="application/gzip",
            ),
        ),
    )
    candidate = ReleaseCandidate(release=release, capture_time=capture_time)

    def fake_fetch_releases_for_window(start: datetime, end: datetime, limit: int):
        return [candidate]

    monkeypatch.setattr(
        "timelapse_backup.cleanup.fetch_releases_for_window",
        fake_fetch_releases_for_window,
    )

    download_calls: list[str] = []
    tile_coords = {(0, 0), (1, 1)}

    def fake_download_assets(
        remote_release: RemoteRelease,
        destination: Path,
        skip_existing: bool = False,
        **_: object,
    ) -> list[Path]:
        download_calls.append(remote_release.tag)
        target_dir = destination / remote_release.tag
        target_dir.mkdir(parents=True, exist_ok=True)
        tar_path = target_dir / f"{remote_release.tag}.tar.gz"
        if tar_path.exists() and skip_existing:
            return []
        with tarfile.open(tar_path, "w:gz") as tar:
            for x, y in tile_coords:
                data = f"{x},{y}".encode()
                info = tarfile.TarInfo(
                    name=f"{remote_release.tag}/tiles/{x}/{y}.png"
                )
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return [tar_path]

    monkeypatch.setattr(
        "timelapse_backup.cleanup.download_assets",
        fake_download_assets,
    )

    cache_paths = ensure_cache_paths(tmp_path / "cache")
    manager = CleanupManager(
        backup_root=backup_root,
        slug_configs=slug_configs,
        timezone_info=tz,
        interval=timedelta(hours=1),
        min_gap=timedelta(hours=3),
        cache_paths=cache_paths,
        dry_run=False,
        keep_archives=False,
        release_limit=5,
    )

    manager.run()

    assert download_calls == [release_tag]

    session_a = (
        backup_root
        / "slug-a"
        / "2024-01-01"
        / "02-00-00"
        / "0_0.png"
    )
    session_b = (
        backup_root
        / "slug-b"
        / "2024-01-01"
        / "02-00-00"
        / "1_1.png"
    )

    assert session_a.exists()
    assert session_b.exists()

    archive_dir = cache_paths.archives_dir / release_tag
    assert not archive_dir.exists()
