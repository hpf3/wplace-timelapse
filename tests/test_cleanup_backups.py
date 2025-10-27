import io
import sys
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timelapse_backup.cleanup import (  # noqa: E402
    collect_sessions,
    determine_timezone,
    find_gaps,
    prune_empty_directories,
    write_tiles_from_archive,
)


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
