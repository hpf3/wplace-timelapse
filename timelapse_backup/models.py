"""Data models used across the timelapse backup system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CompositeFrame:
    """Container for a rendered composite frame and its transparency mask."""

    color: np.ndarray
    alpha: np.ndarray


@dataclass
class FrameManifest:
    """Resolved tile paths for a single frame."""

    session_dir: Path
    tile_paths: Dict[Tuple[int, int], Path]


@dataclass
class PreparedFrame:
    """Prepared frame data persisted to temporary storage."""

    index: int
    session_dir: Path
    temp_path: Optional[Path]
    frame_shape: Optional[Tuple[int, int]]
    alpha_bounds: Optional[Tuple[int, int, int, int]]


@dataclass
class TimelapseStatsCollector:
    """Aggregate per-frame change statistics and timing metadata."""

    frame_datetimes: List[Optional[datetime]]
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, frame_index: int, changed_pixels: Optional[int]) -> None:
        """Record stats for a single frame."""
        timestamp = None
        if 0 <= frame_index < len(self.frame_datetimes):
            timestamp = self.frame_datetimes[frame_index]
        entry = {
            "timestamp": timestamp,
            "changed_pixels": None if changed_pixels is None else int(changed_pixels),
        }
        self.records.append(entry)

    def summarize(
        self,
        gap_threshold: Optional[timedelta],
        seconds_per_pixel: int = 30,
    ) -> Dict[str, Any]:
        """Produce a dictionary summary of collected statistics."""
        rendered_frames = len(self.records)
        total_changed_pixels = sum(
            entry["changed_pixels"] or 0 for entry in self.records
        )
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

        for entry in self.records:
            changed = entry["changed_pixels"]
            if changed is None:
                continue
            if changed > max_change_pixels:
                max_change_pixels = changed
                max_change_timestamp = entry["timestamp"]

        timestamps = [entry["timestamp"] for entry in self.records if entry["timestamp"] is not None]
        start_timestamp = timestamps[0] if timestamps else None
        end_timestamp = timestamps[-1] if timestamps else None
        total_duration_seconds = (
            (end_timestamp - start_timestamp).total_seconds()
            if start_timestamp and end_timestamp and end_timestamp >= start_timestamp
            else 0.0
        )

        intervals: List[float] = []
        previous_ts: Optional[datetime] = None
        for entry in self.records:
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

        for entry in self.records:
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
            for entry in self.records:
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
        average_time_per_frame_seconds = (
            total_time_invested_seconds / rendered_frames if rendered_frames else 0.0
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


__all__ = [
    "CompositeFrame",
    "FrameManifest",
    "PreparedFrame",
    "TimelapseStatsCollector",
]

