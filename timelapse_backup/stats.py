"""Statistics helpers for the timelapse backup system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class TimelapseStatsCollector:
    """Aggregate per-frame change statistics and timing metadata."""

    frame_datetimes: List[Optional[datetime]]
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        frame_index: int,
        changed_pixels: Optional[int],
        *,
        exclude_from_timing: bool = False,
    ) -> None:
        """Record stats for a single frame."""
        timestamp = None
        if 0 <= frame_index < len(self.frame_datetimes):
            timestamp = self.frame_datetimes[frame_index]
        entry = {
            "timestamp": timestamp,
            "changed_pixels": None if changed_pixels is None else int(changed_pixels),
            "exclude_from_timing": exclude_from_timing,
        }
        self.records.append(entry)

    def summarize(
        self,
        gap_threshold: Optional[timedelta],
        *,
        seconds_per_pixel: int = 30,
    ) -> Dict[str, Any]:
        """Produce a dictionary summary of collected statistics."""
        rendered_frames = len(self.records)
        total_changed_pixels = sum(
            entry["changed_pixels"] or 0 for entry in self.records
        )
        timing_records = [
            entry for entry in self.records if not entry.get("exclude_from_timing", False)
        ]
        if not timing_records:
            timing_records = list(self.records)

        frames_with_change = 0
        frames_without_change = 0
        frames_with_known_change = 0

        for entry in self.records:
            changed = entry["changed_pixels"]
            if changed is None:
                continue
            frames_with_known_change += 1
            if changed > 0:
                frames_with_change += 1
            else:
                frames_without_change += 1
        frames_excluded_from_stats = rendered_frames - frames_with_known_change
        max_change_pixels = 0
        max_change_timestamp: Optional[datetime] = None

        for entry in timing_records:
            changed = entry["changed_pixels"]
            if changed is None:
                continue
            if changed > max_change_pixels:
                max_change_pixels = changed
                max_change_timestamp = entry["timestamp"]

        timestamps = [
            entry["timestamp"]
            for entry in timing_records
            if entry["timestamp"] is not None
        ]
        start_timestamp = timestamps[0] if timestamps else None
        end_timestamp = timestamps[-1] if timestamps else None
        total_duration_seconds = (
            (end_timestamp - start_timestamp).total_seconds()
            if start_timestamp and end_timestamp and end_timestamp >= start_timestamp
            else 0.0
        )

        intervals: List[float] = []
        previous_ts: Optional[datetime] = None
        for entry in timing_records:
            ts = entry["timestamp"]
            if ts is None:
                continue
            if previous_ts is not None:
                intervals.append(max(0.0, (ts - previous_ts).total_seconds()))
            previous_ts = ts

        if intervals:
            average_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
        else:
            average_interval = min_interval = max_interval = 0.0

        longest_idle_run_frames = 0
        longest_idle_run_duration = 0.0
        longest_idle_run_start: Optional[datetime] = None

        current_run_frames = 0
        current_run_duration = 0.0
        current_run_start: Optional[datetime] = None
        previous_in_run_timestamp: Optional[datetime] = None

        for entry in timing_records:
            ts = entry["timestamp"]
            changed = entry["changed_pixels"]
            if changed == 0:
                if current_run_frames == 0:
                    current_run_start = ts
                    current_run_duration = 0.0
                else:
                    if ts is not None and previous_in_run_timestamp is not None:
                        current_run_duration += max(
                            0.0, (ts - previous_in_run_timestamp).total_seconds()
                        )
                current_run_frames += 1

                if current_run_frames > longest_idle_run_frames or (
                    current_run_frames == longest_idle_run_frames
                    and current_run_duration > longest_idle_run_duration
                ):
                    longest_idle_run_frames = current_run_frames
                    longest_idle_run_duration = current_run_duration
                    longest_idle_run_start = current_run_start
            else:
                current_run_frames = 0
                current_run_duration = 0.0
                current_run_start = None

            if changed is not None:
                previous_in_run_timestamp = ts
            elif ts is not None:
                previous_in_run_timestamp = ts

        coverage_gaps: List[Dict[str, Any]] = []
        if gap_threshold is not None and gap_threshold.total_seconds() > 0:
            previous_ts = None
            for entry in timing_records:
                ts = entry["timestamp"]
                if ts is None:
                    continue
                if previous_ts is not None:
                    delta = ts - previous_ts
                    if delta > gap_threshold:
                        coverage_gaps.append({
                            "start": previous_ts,
                            "end": ts,
                            "duration_seconds": delta.total_seconds(),
                        })
                previous_ts = ts

        coverage_gap_summary: Optional[Dict[str, Any]] = None
        if coverage_gaps:
            sorted_gaps = sorted(
                coverage_gaps,
                key=lambda gap: gap["duration_seconds"],
                reverse=True,
            )
            coverage_gap_summary = {
                "count": len(coverage_gaps),
                "max_duration_seconds": sorted_gaps[0]["duration_seconds"],
                "examples": [
                    {
                        "start": gap["start"].isoformat() if gap["start"] else None,
                        "end": gap["end"].isoformat() if gap["end"] else None,
                        "duration_seconds": gap["duration_seconds"],
                    }
                    for gap in sorted_gaps[:3]
                ],
            }

        total_time_invested_seconds = total_changed_pixels * seconds_per_pixel
        average_pixels_per_frame = (
            total_changed_pixels / rendered_frames if rendered_frames else 0.0
        )
        timing_changed_pixels_total = sum(
            entry["changed_pixels"] or 0 for entry in timing_records
        )
        timing_rendered_frames = len(timing_records)
        average_time_per_frame_seconds = (
            (timing_changed_pixels_total * seconds_per_pixel) / timing_rendered_frames
            if timing_rendered_frames
            else 0.0
        )

        return {
            "rendered_frames": rendered_frames,
            "frames_with_change": frames_with_change,
            "frames_without_change": frames_without_change,
            "frames_excluded_from_stats": frames_excluded_from_stats,
            "total_changed_pixels": total_changed_pixels,
            "average_pixels_per_frame": average_pixels_per_frame,
            "total_time_invested_seconds": total_time_invested_seconds,
            "average_time_per_frame_seconds": average_time_per_frame_seconds,
            "max_change_pixels": max_change_pixels,
            "max_change_frame_timestamp": (
                max_change_timestamp.isoformat() if max_change_timestamp else None
            ),
            "start_timestamp": start_timestamp.isoformat() if start_timestamp else None,
            "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
            "total_duration_seconds": total_duration_seconds,
            "capture_interval_seconds": {
                "average": average_interval,
                "minimum": min_interval,
                "maximum": max_interval,
            },
            "longest_idle_run_frames": longest_idle_run_frames,
            "longest_idle_run_duration_seconds": longest_idle_run_duration,
            "longest_idle_run_start_timestamp": (
                longest_idle_run_start.isoformat() if longest_idle_run_start else None
            ),
            "coverage_gaps": coverage_gap_summary,
        }


def format_duration(seconds: Optional[float]) -> str:
    """Convert a second count into a compact d/h/m/s string."""
    if seconds is None:
        return "unknown"
    total_seconds = int(round(max(0.0, seconds)))
    if total_seconds <= 0:
        return "0s"

    parts: List[str] = []
    days, remainder = divmod(total_seconds, 86400)
    if days:
        parts.append(f"{days}d")
    hours, remainder = divmod(remainder, 3600)
    if hours:
        parts.append(f"{hours}h")
    minutes, remainder = divmod(remainder, 60)
    if minutes:
        parts.append(f"{minutes}m")
    if remainder or not parts:
        parts.append(f"{remainder}s")
    return " ".join(parts)


def format_duration_with_seconds(seconds: Optional[float]) -> str:
    """Return duration with both pretty form and raw seconds."""
    if seconds is None:
        return "unknown"
    total_seconds = int(round(max(0.0, seconds)))
    pretty = format_duration(total_seconds)
    return f"{pretty} ({total_seconds:,} seconds)"


def format_float(value: Optional[float], decimals: int = 2) -> str:
    """Format a float with thousands separators and fixed decimals."""
    if value is None:
        return "unknown"
    format_spec = f",.{decimals}f"
    return format(value, format_spec)


def format_timestamp(ts: Optional[str]) -> str:
    """Return timestamp string or fallback."""
    return ts if ts else "unknown"


def build_stats_report(
    *,
    slug: str,
    name: str,
    mode: str,
    label: str,
    output_path: Path,
    generated_at: datetime,
    stats: Mapping[str, Any],
) -> str:
    """Render a human-readable stats report."""
    lines: List[str] = []
    lines.append("Timelapse Report")
    lines.append("----------------")
    lines.append(f"Timelapse: {name} ({slug})")
    lines.append(f"Mode: {mode}")
    lines.append(f"Label: {label}")
    lines.append(f"Output video: {output_path}")
    stamp = generated_at.replace(microsecond=0).isoformat()
    lines.append(f"Generated at: {stamp}Z")
    lines.append("")

    lines.append("Frame Overview")
    lines.append("--------------")
    frames_rendered = stats.get("rendered_frames", 0)
    lines.append(f"Frames rendered: {frames_rendered:,}")
    lines.append(f"Frames with change: {stats.get('frames_with_change', 0):,}")
    lines.append(f"Frames without change: {stats.get('frames_without_change', 0):,}")
    excluded_frames = stats.get("frames_excluded_from_stats", 0)
    if excluded_frames:
        lines.append(f"Frames excluded from change stats: {excluded_frames:,}")
    lines.append("")

    lines.append("Change Metrics")
    lines.append("--------------")
    total_pixels = stats.get("total_changed_pixels", 0)
    lines.append(f"Total changed pixels: {total_pixels:,}")
    avg_pixels = stats.get("average_pixels_per_frame")
    lines.append(f"Average pixels per frame: {format_float(avg_pixels)}")
    total_time = stats.get("total_time_invested_seconds")
    lines.append(f"Total time invested: {format_duration_with_seconds(total_time)}")
    avg_time = stats.get("average_time_per_frame_seconds")
    lines.append(f"Average time per frame: {format_duration_with_seconds(avg_time)}")
    peak_pixels = stats.get("max_change_pixels", 0)
    peak_ts = format_timestamp(stats.get("max_change_frame_timestamp"))
    lines.append(f"Peak change: {peak_pixels:,} pixels at {peak_ts}")
    lines.append("")

    lines.append("Timeline")
    lines.append("--------")
    start_ts = format_timestamp(stats.get("start_timestamp"))
    end_ts = format_timestamp(stats.get("end_timestamp"))
    lines.append(f"First frame: {start_ts}")
    lines.append(f"Last frame: {end_ts}")
    coverage_span = format_duration_with_seconds(stats.get("total_duration_seconds"))
    lines.append(f"Coverage span: {coverage_span}")
    capture = stats.get("capture_interval_seconds", {})
    avg_capture = format_duration_with_seconds(capture.get("average"))
    min_capture = format_duration_with_seconds(capture.get("minimum"))
    max_capture = format_duration_with_seconds(capture.get("maximum"))
    lines.append(f"Capture interval (avg / min / max): {avg_capture} / {min_capture} / {max_capture}")
    lines.append("")

    lines.append("Idle Periods")
    lines.append("------------")
    idle_frames = stats.get("longest_idle_run_frames", 0)
    if idle_frames:
        idle_duration = format_duration_with_seconds(stats.get("longest_idle_run_duration_seconds"))
        idle_start = format_timestamp(stats.get("longest_idle_run_start_timestamp"))
        lines.append(f"Longest idle run: {idle_frames:,} frames over {idle_duration}, starting {idle_start}")
    else:
        lines.append("Longest idle run: none detected")
    lines.append("")

    lines.append("Coverage Gaps")
    lines.append("-------------")
    gaps = stats.get("coverage_gaps")
    if not gaps:
        lines.append("None observed")
    else:
        count = gaps.get("count", 0)
        longest_gap = format_duration_with_seconds(gaps.get("max_duration_seconds"))
        lines.append(f"Tracked gaps: {count} (longest {longest_gap})")
        examples = gaps.get("examples") or []
        for idx, gap in enumerate(examples, start=1):
            duration_text = format_duration_with_seconds(gap.get("duration_seconds"))
            gap_start = format_timestamp(gap.get("start"))
            gap_end = format_timestamp(gap.get("end"))
            lines.append(f"  {idx}. {duration_text} from {gap_start} to {gap_end}")

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "TimelapseStatsCollector",
    "format_duration",
    "format_duration_with_seconds",
    "format_float",
    "format_timestamp",
    "build_stats_report",
]
