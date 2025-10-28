import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import timelapse_backup.scheduler as scheduler_module  # noqa: E402


class FakeJob:
    def __init__(self):
        self.next_run_time = None

    def modify(self, **kwargs):
        self.next_run_time = kwargs.get("next_run_time")


class FakeScheduler:
    def __init__(self):
        self.jobs = []
        self.shutdown_called = False

    def add_job(self, func, trigger, id=None, name=None, max_instances=None):
        job = FakeJob()
        self.jobs.append(
            {
                "func": func,
                "trigger": trigger,
                "id": id,
                "name": name,
                "max_instances": max_instances,
                "job": job,
            }
        )
        return job

    def start(self):
        raise KeyboardInterrupt

    def shutdown(self):
        self.shutdown_called = True


class DummyBackup:
    def __init__(self, backup_dir: Path, interval_minutes: int, last_capture: datetime | None):
        self.backup_dir = backup_dir
        self.backup_interval = interval_minutes
        self._last_capture = last_capture
        self.logger = logging.getLogger("scheduler-tests")
        self._timelapses = [
            {
                "slug": "test-slug",
                "name": "Test Timelapse",
                "coordinates": {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
            }
        ]

    def get_enabled_timelapses(self):
        return self._timelapses

    def get_last_capture_time(self):
        return self._last_capture

    def backup_tiles(self):
        pass

    def create_daily_timelapse(self):
        pass


def _run_scheduler_for_test(backup: DummyBackup, now: datetime):
    fake_scheduler = FakeScheduler()
    with patch.object(scheduler_module, "BlockingScheduler", return_value=fake_scheduler):
        with patch.object(scheduler_module, "datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            scheduler_module.run(backup)
    return fake_scheduler


def test_scheduler_delays_first_run_until_interval_boundary(tmp_path):
    now = datetime(2025, 1, 1, 12, 0, 0)
    last_capture = now - timedelta(minutes=2)
    backup = DummyBackup(tmp_path, interval_minutes=5, last_capture=last_capture)

    fake_scheduler = _run_scheduler_for_test(backup, now)

    backup_job = fake_scheduler.jobs[0]["job"]
    expected = last_capture + timedelta(minutes=backup.backup_interval)
    assert backup_job.next_run_time == expected
    assert fake_scheduler.shutdown_called is True


def test_scheduler_runs_immediately_when_interval_has_elapsed(tmp_path):
    now = datetime(2025, 1, 1, 12, 0, 0)
    last_capture = now - timedelta(minutes=10)
    backup = DummyBackup(tmp_path, interval_minutes=5, last_capture=last_capture)

    fake_scheduler = _run_scheduler_for_test(backup, now)

    backup_job = fake_scheduler.jobs[0]["job"]
    assert backup_job.next_run_time == now
    assert fake_scheduler.shutdown_called is True
