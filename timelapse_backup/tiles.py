"""Tile download and placeholder logic for the timelapse backup system."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from timelapse_backup.sessions import extract_slug_from_session, get_prior_sessions


class TileDownloader:
    """Handle tile retrieval, duplicate detection, and placeholder management."""

    def __init__(
        self,
        base_url: str,
        logger: logging.Logger,
        *,
        placeholder_suffix: str = ".placeholder",
        http_timeout: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.placeholder_suffix = placeholder_suffix
        self.http_timeout = http_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _placeholder_filename(self, filename: str) -> str:
        return f"{Path(filename).stem}{self.placeholder_suffix}"

    def _placeholder_path(self, session_dir: Path, filename: str) -> Path:
        return session_dir / self._placeholder_filename(filename)

    def _write_placeholder(self, placeholder_path: Path, target_path: Path) -> bool:
        if not target_path.exists():
            self.logger.warning(
                "Creating placeholder for missing target tile: %s",
                target_path,
            )

        data = {
            "type": "placeholder",
            "version": 2,
            "created_at": datetime.utcnow().isoformat(),
        }

        try:
            with placeholder_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)
            return True
        except OSError as exc:
            self.logger.error("Failed to write placeholder %s: %s", placeholder_path, exc)
            return False

    @staticmethod
    def _find_tile_in_sessions(sessions: Sequence[Path], filename: str) -> Optional[Path]:
        for session in sessions:
            candidate = session / filename
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def placeholder_filename(self, filename: str) -> str:
        return self._placeholder_filename(filename)

    def placeholder_path(self, session_dir: Path, filename: str) -> Path:
        return self._placeholder_path(session_dir, filename)

    def write_placeholder(self, placeholder_path: Path, target_path: Path) -> bool:
        return self._write_placeholder(placeholder_path, target_path)

    def build_previous_tile_map(
        self,
        prior_sessions: Iterable[Path],
        coordinates: Iterable[Tuple[int, int]],
    ) -> Dict[str, Path]:
        filenames = {f"{x}_{y}.png" for x, y in coordinates}
        result: Dict[str, Path] = {}
        session_list = list(prior_sessions)

        for index, session_dir in enumerate(session_list):
            remaining = filenames.difference(result.keys())
            if not remaining:
                break

            for filename in list(remaining):
                direct_path = session_dir / filename
                if direct_path.exists():
                    result[filename] = direct_path
                    continue

                placeholder_path = self._placeholder_path(session_dir, filename)
                if placeholder_path.exists():
                    target_path = self._find_tile_in_sessions(
                        session_list[index + 1 :],
                        filename,
                    )
                    if target_path is not None:
                        result[filename] = target_path

        return result

    def resolve_tile_image_path(
        self,
        session_dir: Path,
        x: int,
        y: int,
        *,
        prior_sessions: Optional[Iterable[Path]] = None,
        backup_root: Optional[Path] = None,
        slug: Optional[str] = None,
    ) -> Optional[Path]:
        filename = f"{x}_{y}.png"
        candidate = session_dir / filename
        if candidate.exists():
            return candidate

        placeholder_path = self._placeholder_path(session_dir, filename)
        if not placeholder_path.exists():
            return None

        sessions_to_search: List[Path]
        if prior_sessions is not None:
            sessions_to_search = list(prior_sessions)
        else:
            derived_slug = slug
            if derived_slug is None and backup_root is not None:
                derived_slug = extract_slug_from_session(session_dir, backup_root)
            if derived_slug and backup_root is not None:
                sessions_to_search = get_prior_sessions(
                    backup_root,
                    derived_slug,
                    session_dir,
                )
            else:
                sessions_to_search = []

        target_path = self._find_tile_in_sessions(sessions_to_search, filename)
        if target_path is not None:
            return target_path

        return None

    def download_tile(
        self,
        slug: str,
        x: int,
        y: int,
        session_dir: Path,
        previous_tile_map: Dict[str, Path],
    ) -> Tuple[bool, bool]:
        """Download a single tile image and emit placeholders for duplicates."""

        url = f"{self.base_url}/{x}/{y}.png" if self.base_url else None
        filename = f"{x}_{y}.png"
        filepath = session_dir / filename

        try:
            if url:
                response = requests.get(url, timeout=self.http_timeout)
                response.raise_for_status()
                content = response.content
            else:
                self.logger.error("No base URL configured for tile downloads")
                return False, False

            previous_path = previous_tile_map.get(filename)
            if previous_path and previous_path.exists():
                try:
                    if previous_path.read_bytes() == content:
                        placeholder_path = self._placeholder_path(session_dir, filename)
                        if self._write_placeholder(placeholder_path, previous_path):
                            self.logger.debug(
                                "Skipped duplicate tile %s %s,%s; placeholder -> %s",
                                slug,
                                x,
                                y,
                                previous_path,
                            )
                            return True, True
                        self.logger.debug(
                            "Placeholder creation failed for %s %s,%s; saving tile to disk",
                            slug,
                            x,
                            y,
                        )
                except OSError as exc:
                    self.logger.warning(
                        "Failed to read previous tile %s: %s",
                        previous_path,
                        exc,
                    )

            with filepath.open("wb") as handle:
                handle.write(content)

            placeholder_path = self._placeholder_path(session_dir, filename)
            if placeholder_path.exists():
                placeholder_path.unlink(missing_ok=True)

            self.logger.debug(
                "Downloaded tile %s %s,%s to %s",
                slug,
                x,
                y,
                filepath,
            )
            return True, False

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to download tile %s,%s: %s", x, y, exc)
            return False, False


__all__ = ["TileDownloader"]
