"""Logging configuration helpers for the timelapse backup system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

DEFAULT_LOG_FILENAME = "timelapse_backup.log"
DEFAULT_LOG_DIR = Path("/data/logs")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / DEFAULT_LOG_FILENAME


def _ensure_sequence(value: Union[logging.Handler, Sequence[logging.Handler]]) -> Sequence[logging.Handler]:
    if isinstance(value, logging.Handler):
        return (value,)
    return value


def _prepare_file_handler(log_file: Union[str, Path]) -> tuple[Optional[logging.Handler], Optional[str]]:
    """Create a file handler for the given path, returning an optional warning."""
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        return handler, None
    except OSError as exc:
        fallback_path = Path.cwd() / log_path.name
        try:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(fallback_path, encoding="utf-8")
            warning = (
                f"Failed to open log file at '{log_path}'. Falling back to '{fallback_path}'. "
                f"Reason: {exc}"
            )
            return handler, warning
        except OSError as fallback_exc:
            warning = (
                f"Failed to open log file at '{log_path}' "
                f"and fallback '{fallback_path}'. Reason: {fallback_exc}"
            )
            return None, warning


def configure_logging(
    logger_name: Optional[str] = None,
    *,
    level: int = logging.INFO,
    log_file: Union[str, Path, None] = DEFAULT_LOG_FILE,
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
    pending_warning: Optional[str] = None
    if log_file:
        file_handler, pending_warning = _prepare_file_handler(log_file)
        if file_handler:
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

    if pending_warning:
        logger.warning(pending_warning)

    return logger


__all__ = ["configure_logging"]
