"""CLI entrypoint for the timelapse backup system."""

from timelapse_backup.app import TimelapseBackup


def main() -> None:
    """Instantiate the backup facade and start the scheduler."""
    backup_system = TimelapseBackup()
    backup_system.run()


if __name__ == "__main__":
    main()
