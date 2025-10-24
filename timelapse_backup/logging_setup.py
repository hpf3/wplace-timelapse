"""Logging configuration helpers for the timelapse backup system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union


def _ensure_sequence(value: Union[logging.Handler, Sequence[logging.Handler]]) -> Sequence[logging.Handler]:
    if isinstance(value, logging.Handler):
        return (value,)
    return value


def configure_logging(
    logger_name: Optional[str] = None,
    *,
    level: int = logging.INFO,
    log_file: Union[str, Path, None] = "timelapse_backup.log",
    include_stream: bool = True,
) -> logging.Logger:
    """Configure application logging and return a ready-to-use logger.

    Parameters
    ----------
    logger_name:
        Name of the logger to return. Defaults to ``"timelapse_backup"`` when omitted.
    level:
        Logging level applied to the configured handlers.
    log_file:
        Optional path to the log file. Pass ``None`` to disable file logging.
    include_stream:
        When ``True`` (default) attach a `logging.StreamHandler` for stdout feedback.
    """

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handlers: list[logging.Handler] = []
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if include_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    logging.basicConfig(level=level, handlers=_ensure_sequence(handlers), force=True)

    target_logger_name = logger_name or "timelapse_backup"
    logger = logging.getLogger(target_logger_name)
    logger.setLevel(level)
    return logger


__all__ = ["configure_logging"]
