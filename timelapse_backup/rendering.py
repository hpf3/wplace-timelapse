"""Frame rendering and encoding pipeline for the timelapse backup system."""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from timelapse_backup.config import DiffSettings
from timelapse_backup.manifests import ManifestBuilder
from timelapse_backup.models import CompositeFrame, FrameManifest, PreparedFrame
from timelapse_backup.progress import eta_string
from timelapse_backup.stats import TimelapseStatsCollector


class Renderer:
    """Render composite frames and encode them into video outputs."""

    def __init__(
        self,
        manifest_builder: ManifestBuilder,
        *,
        logger: logging.Logger,
        frame_prep_workers: int,
        auto_crop_transparent_frames: bool,
        diff_settings: DiffSettings,
        historical_cutoff: Optional[datetime] = None,
    ) -> None:
        self.manifest_builder = manifest_builder
        self.logger = logger
        self.frame_prep_workers = max(1, frame_prep_workers)
        self.auto_crop_transparent_frames = auto_crop_transparent_frames
        self.diff_settings = diff_settings
        self.historical_cutoff = historical_cutoff

    # ------------------------------------------------------------------
    # Frame preparation
    # ------------------------------------------------------------------

    def prepare_frames_from_manifests(
        self,
        manifests: Sequence[FrameManifest],
        coordinates: Sequence[Tuple[int, int]],
        temp_dir: Path,
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> List[PreparedFrame]:
        if not manifests:
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(manifests)
        progress_interval = max(1, total // 20)
        prepared: List[Optional[PreparedFrame]] = [None] * total
        progress_start = perf_counter()

        with ThreadPoolExecutor(max_workers=self.frame_prep_workers) as executor:
            futures = [
                executor.submit(
                    self._render_frame_from_manifest,
                    index,
                    manifest,
                    coordinates,
                    temp_dir,
                )
                for index, manifest in enumerate(manifests)
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                prepared[result.index] = result
                completed += 1
                if completed % progress_interval == 0 or completed == total:
                    percent = (completed / total) * 100.0
                    elapsed = perf_counter() - progress_start
                    eta_text = eta_string(elapsed, completed, total)
                    self.logger.info(
                        "Frame rendering progress for '%s' (%s) %s %s: %s/%s frames (%0.1f%%, %s)",
                        name,
                        slug,
                        mode_name,
                        label,
                        completed,
                        total,
                        percent,
                        eta_text,
                    )

        return [frame for frame in prepared if frame and frame.temp_path is not None]

    def _render_frame_from_manifest(
        self,
        index: int,
        manifest: FrameManifest,
        coordinates: Sequence[Tuple[int, int]],
        temp_dir: Path,
    ) -> PreparedFrame:
        composite = self.manifest_builder.compose_frame(manifest, coordinates)
        if composite is None:
            return PreparedFrame(
                index=index,
                session_dir=manifest.session_dir,
                temp_path=None,
                frame_shape=None,
                alpha_bounds=None,
            )

        frame_bounds = None
        if self.auto_crop_transparent_frames:
            frame_bounds = self._update_content_bounds(composite.alpha, None)

        success, buffer = cv2.imencode(".png", composite.color)
        if not success:
            raise RuntimeError(
                f"Failed to encode frame for manifest index {index} ({manifest.session_dir})"
            )

        temp_path = temp_dir / f"frame_{index:06d}.png"
        temp_path.write_bytes(buffer.tobytes())

        return PreparedFrame(
            index=index,
            session_dir=manifest.session_dir,
            temp_path=temp_path,
            frame_shape=composite.color.shape[:2],
            alpha_bounds=frame_bounds,
        )

    # ------------------------------------------------------------------
    # Cropping utilities
    # ------------------------------------------------------------------

    def compute_crop_bounds(
        self,
        prepared_frames: Sequence[PreparedFrame],
        *,
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
        first_frame_shape: Optional[Tuple[int, int]] = None
        content_bounds: Optional[Tuple[int, int, int, int]] = None

        for frame in prepared_frames:
            if frame.frame_shape is None:
                continue
            if first_frame_shape is None:
                first_frame_shape = frame.frame_shape
            if self.auto_crop_transparent_frames and frame.alpha_bounds is not None:
                content_bounds = self._merge_bounds(content_bounds, frame.alpha_bounds)

        if first_frame_shape is None:
            raise RuntimeError(
                f"No valid frames created for '{name}' ({slug}) {mode_name} {label}"
            )

        frame_height, frame_width = first_frame_shape

        if self.auto_crop_transparent_frames and content_bounds is not None:
            min_x, min_y, max_x, max_y = content_bounds
            min_x = max(0, min(min_x, frame_width - 1))
            min_y = max(0, min(min_y, frame_height - 1))
            max_x = max(min_x + 1, min(max_x, frame_width))
            max_y = max(min_y + 1, min(max_y, frame_height))
            crop_bounds = (min_x, min_y, max_x, max_y)
        else:
            crop_bounds = (0, 0, frame_width, frame_height)

        crop_width = crop_bounds[2] - crop_bounds[0]
        crop_height = crop_bounds[3] - crop_bounds[1]
        if crop_width <= 0 or crop_height <= 0:
            crop_bounds = (0, 0, frame_width, frame_height)
            crop_width = frame_width
            crop_height = frame_height

        x0, y0, x1, y1 = crop_bounds

        if crop_width % 2 != 0:
            if x1 < frame_width:
                x1 = min(frame_width, x1 + 1)
            elif x0 > 0:
                x0 = max(0, x0 - 1)
            else:
                self.logger.warning(
                    "Unable to expand width to even dimension for '%s' (%s) %s %s",
                    name,
                    slug,
                    mode_name,
                    label,
                )

        if crop_height % 2 != 0:
            if y1 < frame_height:
                y1 = min(frame_height, y1 + 1)
            elif y0 > 0:
                y0 = max(0, y0 - 1)
            else:
                self.logger.warning(
                    "Unable to expand height to even dimension for '%s' (%s) %s %s",
                    name,
                    slug,
                    mode_name,
                    label,
                )

        final_bounds = (x0, y0, x1, y1)
        final_shape = (frame_height, frame_width)

        if (
            self.auto_crop_transparent_frames
            and (crop_width != frame_width or crop_height != frame_height)
        ):
            self.logger.info(
                "Cropping '%s' (%s) %s %s to %sx%s from original %sx%s",
                name,
                slug,
                mode_name,
                label,
                x1 - x0,
                y1 - y0,
                frame_width,
                frame_height,
            )

        return final_bounds, final_shape

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def frame_byte_generator(
        self,
        prepared_frames: Sequence[PreparedFrame],
        mode_name: str,
        slug: str,
        name: str,
        label: str,
        frame_datetimes: Sequence[Optional[datetime]],
    ) -> Tuple[Iterator[bytes], TimelapseStatsCollector]:
        total_frames = len(prepared_frames)
        if total_frames == 0:
            return iter(()), TimelapseStatsCollector(list(frame_datetimes))

        progress_interval = max(1, total_frames // 20)
        stats_collector = TimelapseStatsCollector(list(frame_datetimes))
        progress_start = perf_counter()

        def generator() -> Iterator[bytes]:
            prev_composite_color: Optional[np.ndarray] = None

            for zero_based_index, prepared in enumerate(prepared_frames):
                frame_timestamp = (
                    frame_datetimes[zero_based_index]
                    if zero_based_index < len(frame_datetimes)
                    else None
                )
                is_historical = self._is_historical_timestamp(frame_timestamp)

                if prepared.temp_path is None:
                    stats_collector.record(
                        zero_based_index,
                        None,
                        exclude_from_timing=is_historical,
                    )
                    continue

                changed_pixels: Optional[int]
                if mode_name == "diff":
                    composite_color = cv2.imread(str(prepared.temp_path), cv2.IMREAD_COLOR)
                    if composite_color is None:
                        raise RuntimeError(
                            f"Failed to read prepared composite for diff mode: {prepared.temp_path}"
                        )
                    frame, changed_pixels = self.create_differential_frame(
                        prev_composite_color,
                        composite_color,
                        return_stats=True,
                    )
                    prev_composite_color = composite_color
                    success, buffer = cv2.imencode(".png", frame)
                    if not success:
                        raise RuntimeError(
                            f"Failed to encode differential frame for '{name}' ({slug}) {mode_name} {label}"
                        )
                    frame_bytes = buffer.tobytes()
                else:
                    frame_bytes = prepared.temp_path.read_bytes()
                    changed_pixels = None
                stats_collector.record(
                    zero_based_index,
                    changed_pixels,
                    exclude_from_timing=is_historical,
                )
                yield frame_bytes

                frame_number = zero_based_index + 1
                if frame_number % progress_interval == 0 or frame_number == total_frames:
                    percent = (frame_number / total_frames) * 100.0
                    elapsed = perf_counter() - progress_start
                    eta_text = eta_string(elapsed, frame_number, total_frames)
                    self.logger.info(
                        "Encoding progress for '%s' (%s) %s %s: %s/%s frames (%0.1f%%, %s)",
                        name,
                        slug,
                        mode_name,
                        label,
                        frame_number,
                        total_frames,
                        percent,
                        eta_text,
                    )

        return generator(), stats_collector

    def encode_with_ffmpeg(
        self,
        frame_iter: Iterator[bytes],
        output_path: Path,
        fps: int,
        crop_bounds: Tuple[int, int, int, int],
        *,
        quality: int,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg with libx264/libx265/libsvtav1."
            )

        x0, y0, x1, y1 = crop_bounds
        crop_w, crop_h = (x1 - x0), (y1 - y0)
        crop_filter = f"crop={crop_w}:{crop_h}:{x0}:{y0}"

        codec_args = [
            "-c:v",
            "libx264",
            "-crf",
            str(quality),
            "-preset",
            "slow",
            "-tune",
            "animation",
            "-x264-params",
            "keyint=300:min-keyint=300:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]

        temp_output = output_path.with_name(f".tmp_{uuid.uuid4().hex}_{output_path.name}")
        if temp_output.exists():
            temp_output.unlink()

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            crop_filter,
            *codec_args,
            str(temp_output),
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        assert process.stdin is not None
        try:
            for frame_bytes in frame_iter:
                process.stdin.write(frame_bytes)
        finally:
            process.stdin.close()

        try:
            if process.stdin is None:
                raise RuntimeError("FFmpeg stdin unavailable")
            for frame_bytes in frame_iter:
                process.stdin.write(frame_bytes)
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        stderr_bytes = b""
        if process.stderr is not None:
            try:
                stderr_bytes = process.stderr.read()
            finally:
                process.stderr.close()

        if process.stdout is not None:
            process.stdout.close()

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr_bytes)

        temp_output.replace(output_path)

    # ------------------------------------------------------------------
    # Diff helpers
    # ------------------------------------------------------------------

    def create_differential_frame(
        self,
        previous_frame: Optional[np.ndarray],
        current_frame: np.ndarray,
        *,
        return_stats: bool = False,
    ) -> Tuple[np.ndarray, Optional[int]]:
        background_color = self.manifest_builder.background_color
        changed_pixels = 0

        if previous_frame is None:
            frame = np.full_like(current_frame, background_color)
            return (frame, changed_pixels) if return_stats else frame

        diff = cv2.absdiff(previous_frame, current_frame)
        background = np.full_like(current_frame, background_color)
        visualization = self.diff_settings.visualization

        if visualization == "binary":
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, binary_diff = cv2.threshold(
                gray_diff,
                self.diff_settings.threshold,
                255,
                cv2.THRESH_BINARY,
            )
            result = background.copy()
            mask = binary_diff > 0
            result[mask] = (255, 255, 255)
            changed_pixels = int(np.count_nonzero(mask))

        elif visualization == "heatmap":
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(
                gray_diff,
                self.diff_settings.threshold,
                255,
                cv2.THRESH_BINARY,
            )
            enhanced = cv2.multiply(thresholded, self.diff_settings.enhancement_factor)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
            result = background.copy()
            mask = thresholded > 0
            result[mask] = heatmap[mask]
            changed_pixels = int(np.count_nonzero(mask))

        elif visualization == "overlay":
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray_diff,
                self.diff_settings.threshold,
                255,
                cv2.THRESH_BINARY,
            )
            overlay = current_frame.copy()
            overlay[mask > 0] = [0, 255, 255]
            result = cv2.addWeighted(current_frame, 0.7, overlay, 0.3, 0)
            changed_pixels = int(np.count_nonzero(mask))

        elif visualization == "colored":
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray_diff,
                self.diff_settings.threshold,
                255,
                cv2.THRESH_BINARY,
            )
            colored_diff = np.zeros_like(current_frame)
            colored_diff[mask > 0] = [0, 255, 0]
            enhanced_diff = cv2.multiply(colored_diff, self.diff_settings.enhancement_factor)
            result = background.copy()
            enhanced_diff = np.clip(enhanced_diff, 0, 255).astype(np.uint8)
            result[mask > 0] = enhanced_diff[mask > 0]
            changed_pixels = int(np.count_nonzero(mask))

        else:
            result = background.copy()
            diff_mask = np.any(diff > 0, axis=2)
            result[diff_mask] = diff[diff_mask]
            changed_pixels = int(np.count_nonzero(diff_mask))

        return (result, changed_pixels) if return_stats else result

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _is_historical_timestamp(
        self,
        frame_timestamp: Optional[datetime],
    ) -> bool:
        return (
            self.historical_cutoff is not None
            and frame_timestamp is not None
            and frame_timestamp < self.historical_cutoff
        )

    @staticmethod
    def _update_content_bounds(
        alpha_mask: np.ndarray,
        bounds: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if alpha_mask is None or not np.any(alpha_mask):
            return bounds

        if alpha_mask.dtype != np.uint8:
            mask = alpha_mask.astype(np.uint8)
        else:
            mask = alpha_mask

        coords = cv2.findNonZero(mask)
        if coords is None:
            return bounds

        x, y, w, h = cv2.boundingRect(coords)
        updated = (x, y, x + w, y + h)

        if bounds is None:
            return updated

        return (
            min(bounds[0], updated[0]),
            min(bounds[1], updated[1]),
            max(bounds[2], updated[2]),
            max(bounds[3], updated[3]),
        )

    @staticmethod
    def _merge_bounds(
        current: Optional[Tuple[int, int, int, int]],
        new: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if new is None:
            return current
        if current is None:
            return new
        return (
            min(current[0], new[0]),
            min(current[1], new[1]),
            max(current[2], new[2]),
            max(current[3], new[3]),
        )


__all__ = ["Renderer"]
