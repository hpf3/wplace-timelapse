"""Configuration dataclasses and loading helpers for the timelapse backup system."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def _default_frame_prep_workers() -> int:
    """Determine a sensible default for CPU-bound frame preparation workers."""
    return max(1, min(4, os.cpu_count() or 1))


def _parse_bool(value: Any, default: bool) -> bool:
    """Parse truthy/falsy values from multiple input types."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return default


def _parse_positive_int(value: Any, default: int) -> int:
    """Parse a positive integer with fallback to default."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_float(value: Any, default: float) -> float:
    """Parse a floating point number with fallback to default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_background_color(value: Any) -> Tuple[int, int, int]:
    """Parse and clamp background color definitions to BGR tuples."""
    default = (0, 0, 0)

    def _clamp_triplet(triplet: Any) -> Optional[Tuple[int, int, int]]:
        if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
            return None
        try:
            return tuple(
                max(0, min(255, int(channel)))
                for channel in triplet
            )
        except (TypeError, ValueError):
            return None

    def _to_bgr(channels: Tuple[int, int, int], order: Optional[str]) -> Optional[Tuple[int, int, int]]:
        color_order = (order or "rgb").lower()
        if color_order == "bgr":
            return channels
        if color_order == "rgb":
            return (channels[2], channels[1], channels[0])
        return None

    if isinstance(value, dict):
        if "hex" in value and isinstance(value["hex"], str):
            return _parse_background_color(value["hex"])
        if "value" in value:
            channels = _clamp_triplet(value["value"])
            if channels is None:
                return default
            bgr = _to_bgr(channels, value.get("order") or value.get("color_space"))
            if bgr is not None:
                return bgr
            return default

    if isinstance(value, (list, tuple)):
        channels = _clamp_triplet(value)
        if channels is None:
            return default
        bgr = _to_bgr(channels, "rgb")
        if bgr is not None:
            return bgr
        return default

    if isinstance(value, str):
        hex_value = value.lstrip("#")
        if len(hex_value) == 6:
            try:
                r = int(hex_value[0:2], 16)
                g = int(hex_value[2:4], 16)
                b = int(hex_value[4:6], 16)
                return (b, g, r)
            except ValueError:
                return default

    return default


@dataclass(frozen=True)
class DiffSettings:
    """Fine-grained settings controlling differential frame generation."""

    threshold: int = 10
    visualization: str = "colored"
    fade_frames: int = 3
    enhancement_factor: float = 2.0


@dataclass(frozen=True)
class ReportingSettings:
    """Settings dictating stats reporting behaviour."""

    enable_stats_file: bool = True
    seconds_per_pixel: int = 30
    coverage_gap_multiplier: Optional[float] = None


@dataclass(frozen=True)
class GlobalSettings:
    """Top-level configuration shared by all timelapses."""

    base_url: str
    backup_interval_minutes: int
    backup_dir: Path
    output_dir: Path
    request_delay: float
    timelapse_fps: int
    timelapse_quality: int
    background_color: Tuple[int, int, int]
    auto_crop_transparent_frames: bool
    frame_prep_workers: int
    diff_settings: DiffSettings = field(default_factory=DiffSettings)
    reporting: ReportingSettings = field(default_factory=ReportingSettings)


@dataclass(frozen=True)
class TimelapseModeConfig:
    """Configurable rendering mode for a timelapse."""

    name: str
    enabled: bool = True
    suffix: str = ""
    create_full_timelapse: bool = False


@dataclass(frozen=True)
class Coordinates:
    """Inclusive coordinate bounds describing a tile region."""

    xmin: int
    xmax: int
    ymin: int
    ymax: int

    def iter_tiles(self) -> Iterable[Tuple[int, int]]:
        """Yield tile coordinates covered by the region."""
        for x in range(self.xmin, self.xmax + 1):
            for y in range(self.ymin, self.ymax + 1):
                yield (x, y)


@dataclass(frozen=True)
class TimelapseConfig:
    """Configuration for a single tracked timelapse."""

    slug: str
    name: str
    coordinates: Coordinates
    description: Optional[str] = None
    enabled: bool = True
    timelapse_modes: Dict[str, TimelapseModeConfig] = field(default_factory=dict)

    def enabled_modes(self) -> Iterable[TimelapseModeConfig]:
        """Iterate enabled modes in insertion order."""
        return (
            mode
            for mode in self.timelapse_modes.values()
            if mode.enabled
        )


@dataclass(frozen=True)
class Config:
    """Root configuration object for the timelapse backup system."""

    timelapses: Tuple[TimelapseConfig, ...]
    global_settings: GlobalSettings


def _parse_diff_settings(raw: Mapping[str, Any]) -> DiffSettings:
    default = DiffSettings()
    if not isinstance(raw, Mapping):
        return default
    return DiffSettings(
        threshold=_parse_positive_int(raw.get("threshold"), default.threshold),
        visualization=str(raw.get("visualization", default.visualization)),
        fade_frames=_parse_positive_int(raw.get("fade_frames"), default.fade_frames),
        enhancement_factor=_parse_float(raw.get("enhancement_factor"), default.enhancement_factor),
    )


def _parse_reporting_settings(raw: Mapping[str, Any]) -> ReportingSettings:
    default = ReportingSettings()
    if not isinstance(raw, Mapping):
        return default

    enable_stats = raw.get("enable_stats_file")
    seconds_per_pixel = raw.get("seconds_per_pixel")
    gap_multiplier = raw.get("coverage_gap_multiplier")

    parsed_enable = (
        _parse_bool(enable_stats, default.enable_stats_file)
        if enable_stats is not None
        else default.enable_stats_file
    )

    parsed_seconds = default.seconds_per_pixel
    if seconds_per_pixel is not None:
        try:
            parsed_seconds = max(1, int(seconds_per_pixel))
        except (TypeError, ValueError):
            parsed_seconds = default.seconds_per_pixel

    parsed_multiplier: Optional[float]
    if gap_multiplier is None:
        parsed_multiplier = default.coverage_gap_multiplier
    else:
        try:
            parsed_multiplier = float(gap_multiplier)
        except (TypeError, ValueError):
            parsed_multiplier = default.coverage_gap_multiplier
        if parsed_multiplier is not None and parsed_multiplier <= 0:
            parsed_multiplier = default.coverage_gap_multiplier

    return ReportingSettings(
        enable_stats_file=parsed_enable,
        seconds_per_pixel=parsed_seconds,
        coverage_gap_multiplier=parsed_multiplier,
    )


def _parse_global_settings(data: Mapping[str, Any]) -> GlobalSettings:
    default_workers = _default_frame_prep_workers()
    diff_settings = _parse_diff_settings(data.get("diff_settings", {}))
    reporting_settings = _parse_reporting_settings(data.get("reporting", {}))

    return GlobalSettings(
        base_url=str(data.get("base_url", "https://backend.wplace.live/files/s0/tiles")),
        backup_interval_minutes=_parse_positive_int(data.get("backup_interval_minutes"), 5),
        backup_dir=Path(data.get("backup_dir", "backups")),
        output_dir=Path(data.get("output_dir", "output")),
        request_delay=_parse_float(data.get("request_delay"), 0.5),
        timelapse_fps=_parse_positive_int(data.get("timelapse_fps"), 10),
        timelapse_quality=_parse_positive_int(data.get("timelapse_quality"), 23),
        background_color=_parse_background_color(data.get("background_color")),
        auto_crop_transparent_frames=_parse_bool(
            data.get("auto_crop_transparent_frames"),
            True,
        ),
        frame_prep_workers=_parse_positive_int(
            data.get("frame_prep_workers"),
            default_workers,
        ),
        diff_settings=diff_settings,
        reporting=reporting_settings,
    )


def _parse_timelapse_modes(raw_modes: Mapping[str, Any]) -> Dict[str, TimelapseModeConfig]:
    if not isinstance(raw_modes, Mapping):
        return {"normal": TimelapseModeConfig(name="normal")}
    parsed: Dict[str, TimelapseModeConfig] = {}
    for name, value in raw_modes.items():
        if not isinstance(value, Mapping):
            continue
        parsed[name] = TimelapseModeConfig(
            name=name,
            enabled=_parse_bool(value.get("enabled"), True),
            suffix=str(value.get("suffix", "")),
            create_full_timelapse=_parse_bool(
                value.get("create_full_timelapse"),
                False,
            ),
        )
    if "normal" not in parsed:
        parsed["normal"] = TimelapseModeConfig(name="normal")
    return parsed


def _parse_coordinates(raw: Mapping[str, Any]) -> Coordinates:
    return Coordinates(
        xmin=_parse_positive_int(raw.get("xmin"), 0),
        xmax=_parse_positive_int(raw.get("xmax"), 0),
        ymin=_parse_positive_int(raw.get("ymin"), 0),
        ymax=_parse_positive_int(raw.get("ymax"), 0),
    )


def _parse_timelapses(raw_list: Iterable[Any]) -> Tuple[TimelapseConfig, ...]:
    timelapses: list[TimelapseConfig] = []
    for entry in raw_list or []:
        if not isinstance(entry, Mapping):
            continue
        coordinates = _parse_coordinates(entry.get("coordinates", {}))
        timelapse = TimelapseConfig(
            slug=str(entry.get("slug", "")).strip() or "default",
            name=str(entry.get("name", "Timelapse")).strip() or "Timelapse",
            description=entry.get("description"),
            coordinates=coordinates,
            enabled=_parse_bool(entry.get("enabled"), True),
            timelapse_modes=_parse_timelapse_modes(entry.get("timelapse_modes", {})),
        )
        timelapses.append(timelapse)
    return tuple(timelapses)


def _load_legacy_config(env: Mapping[str, str]) -> Config:
    """Fallback configuration derived from environment variables."""
    base_url = env.get("BASE_URL", "https://backend.wplace.live/files/s0/tiles")
    backup_interval = _parse_positive_int(env.get("BACKUP_INTERVAL_MINUTES"), 5)
    backup_dir = Path(env.get("BACKUP_DIR", "backups"))
    output_dir = Path(env.get("TIMELAPSE_DIR", "timelapses"))
    request_delay = _parse_float(env.get("REQUEST_DELAY"), 0.5)
    timelapse_fps = _parse_positive_int(env.get("TIMELAPSE_FPS"), 10)
    timelapse_quality = _parse_positive_int(env.get("TIMELAPSE_QUALITY"), 23)
    background_color = (0, 0, 0)
    auto_crop = _parse_bool(env.get("AUTO_CROP_TRANSPARENT_FRAMES"), True)
    default_workers = _default_frame_prep_workers()
    frame_prep_workers = _parse_positive_int(env.get("FRAME_PREP_WORKERS"), default_workers)

    reporting = _parse_reporting_settings({
        "enable_stats_file": env.get("ENABLE_STATS_FILE", "true"),
        "seconds_per_pixel": env.get("SECONDS_PER_PIXEL", "30"),
        "coverage_gap_multiplier": env.get("COVERAGE_GAP_MULTIPLIER"),
    })

    diff_settings = _parse_diff_settings({
        "threshold": env.get("DIFF_THRESHOLD"),
        "visualization": env.get("DIFF_VISUALIZATION"),
        "fade_frames": env.get("DIFF_FADE_FRAMES"),
        "enhancement_factor": env.get("DIFF_ENHANCEMENT_FACTOR"),
    })

    global_settings = GlobalSettings(
        base_url=base_url,
        backup_interval_minutes=backup_interval,
        backup_dir=backup_dir,
        output_dir=output_dir,
        request_delay=request_delay,
        timelapse_fps=timelapse_fps,
        timelapse_quality=timelapse_quality,
        background_color=background_color,
        auto_crop_transparent_frames=auto_crop,
        frame_prep_workers=frame_prep_workers,
        diff_settings=diff_settings,
        reporting=reporting,
    )

    coordinates = Coordinates(
        xmin=_parse_positive_int(env.get("XMIN"), 1031),
        xmax=_parse_positive_int(env.get("XMAX"), 1032),
        ymin=_parse_positive_int(env.get("YMIN"), 747),
        ymax=_parse_positive_int(env.get("YMAX"), 748),
    )

    timelapse = TimelapseConfig(
        slug="default",
        name="Default Timelapse",
        coordinates=coordinates,
        timelapse_modes={
            "normal": TimelapseModeConfig(name="normal"),
        },
    )

    return Config(
        timelapses=(timelapse,),
        global_settings=global_settings,
    )


def load_config(config_path: Path | str, env: Mapping[str, str] | None = None) -> Config:
    """Load configuration from JSON file or environment defaults."""
    source_env = env or os.environ
    path = Path(config_path)

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        global_settings = _parse_global_settings(data.get("global_settings", {}))
        timelapses = _parse_timelapses(data.get("timelapses", []))
        return Config(
            timelapses=timelapses,
            global_settings=global_settings,
        )

    return _load_legacy_config(source_env)


__all__ = [
    "Config",
    "Coordinates",
    "DiffSettings",
    "GlobalSettings",
    "ReportingSettings",
    "TimelapseConfig",
    "TimelapseModeConfig",
    "load_config",
    "_parse_bool",
    "_parse_positive_int",
    "_parse_background_color",
]
