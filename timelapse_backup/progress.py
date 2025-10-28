"""Helpers for estimating and formatting progress timing information."""

from __future__ import annotations

from datetime import datetime, timedelta


def _format_duration(seconds: float) -> str:
    """Return a compact human-readable duration string."""
    total_seconds = int(round(seconds))
    if total_seconds <= 0:
        return "<1s"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_remaining = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_remaining:02d}s"
    if minutes:
        return f"{minutes}m{seconds_remaining:02d}s"
    return f"{seconds_remaining}s"


def eta_string(elapsed: float, completed: int, total: int) -> str:
    """Format an ETA string given elapsed time and progress counters."""
    if completed <= 0 or total <= 0 or completed > total or elapsed <= 0.0:
        return "ETA estimating"

    progress = completed / total
    if progress <= 0.0:
        return "ETA estimating"

    estimated_total = elapsed / progress
    remaining = max(0.0, estimated_total - elapsed)
    finish_time = datetime.now() + timedelta(seconds=remaining)
    return f"ETA {_format_duration(remaining)} (finish {finish_time.strftime('%H:%M:%S')})"

