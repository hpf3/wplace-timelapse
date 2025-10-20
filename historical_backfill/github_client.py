"""
GitHub API helpers for interacting with the murolem/wplace-archives releases.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import time

import requests

from .models import parse_iso_datetime

LOGGER = logging.getLogger(__name__)

GITHUB_API_ROOT = "https://api.github.com"
REPO_OWNER = "murolem"
REPO_NAME = "wplace-archives"
RELEASES_ENDPOINT = f"{GITHUB_API_ROOT}/repos/{REPO_OWNER}/{REPO_NAME}/releases"
RELEASE_BY_TAG_ENDPOINT = (
    f"{GITHUB_API_ROOT}/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{{tag}}"
)


def _build_session() -> requests.Session:
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "wplace-timelapse-backfill",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


@dataclass(frozen=True)
class RemoteAsset:
    name: str
    size: int
    download_url: str
    content_type: Optional[str]


@dataclass(frozen=True)
class RemoteRelease:
    tag: str
    name: str
    published_at: Optional[datetime]
    assets: tuple[RemoteAsset, ...]


class GitHubError(RuntimeError):
    """Raised when GitHub API interaction fails."""


def _handle_response(response: requests.Response) -> dict:
    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        reset = response.headers.get("X-RateLimit-Reset")
        raise GitHubError(
            f"GitHub API rate limit exceeded. Reset epoch: {reset}. Provide a GITHUB_TOKEN or retry later."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise GitHubError(str(exc)) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise GitHubError("Failed to parse GitHub API response as JSON") from exc


def _parse_asset(data: dict) -> RemoteAsset:
    return RemoteAsset(
        name=data.get("name", ""),
        size=int(data.get("size") or 0),
        download_url=data.get("browser_download_url", ""),
        content_type=data.get("content_type"),
    )


def _parse_release(data: dict) -> RemoteRelease:
    published_raw = data.get("published_at")
    published = parse_iso_datetime(published_raw) if published_raw else None
    assets = tuple(_parse_asset(asset) for asset in data.get("assets", []))
    return RemoteRelease(
        tag=data.get("tag_name") or data.get("name") or "unknown",
        name=data.get("name") or data.get("tag_name") or "unknown",
        published_at=published,
        assets=assets,
    )


def fetch_recent_releases(limit: int = 5) -> List[RemoteRelease]:
    session = _build_session()
    per_page = max(1, min(limit, 100))
    releases: List[RemoteRelease] = []
    page = 1

    while len(releases) < limit:
        response = session.get(RELEASES_ENDPOINT, params={"per_page": per_page, "page": page})
        payload = _handle_response(response)
        if not isinstance(payload, list) or not payload:
            break
        for entry in payload:
            releases.append(_parse_release(entry))
            if len(releases) >= limit:
                break
        page += 1
        if len(payload) < per_page:
            break

    return releases


def fetch_release_by_tag(tag: str) -> RemoteRelease:
    session = _build_session()
    response = session.get(RELEASE_BY_TAG_ENDPOINT.format(tag=tag))
    payload = _handle_response(response)
    return _parse_release(payload)


def download_assets(
    release: RemoteRelease,
    destination: Path,
    skip_existing: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[Path]:
    session = _build_session()
    destination = destination / release.tag
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []

    for asset in release.assets:
        if not asset.download_url:
            LOGGER.warning("Skipping asset %s - missing download URL", asset.name)
            continue

        target_path = destination / asset.name
        if skip_existing and target_path.exists():
            LOGGER.info("Skipping existing asset %s", target_path)
            continue

        attempts = 0
        completed = False
        while attempts < max_retries and not completed:
            attempts += 1
            offset = 0
            if target_path.exists():
                try:
                    offset = target_path.stat().st_size
                except OSError:
                    offset = 0
                if offset > asset.size:
                    LOGGER.warning("Existing partial %s is larger than expected; restarting", target_path)
                    try:
                        target_path.unlink()
                    except OSError:
                        LOGGER.debug("Failed to remove oversize file %s", target_path, exc_info=True)
                    offset = 0

            headers = {}
            if offset:
                headers["Range"] = f"bytes={offset}-"

            LOGGER.info(
                "Downloading %s (%s bytes) [attempt %s/%s, offset %s]",
                asset.name,
                asset.size,
                attempts,
                max_retries,
                offset,
            )

            try:
                with session.get(asset.download_url, stream=True, headers=headers) as response:
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        raise GitHubError(f"Failed downloading asset {asset.name}: {exc}") from exc

                    if offset and response.status_code == 200:
                        LOGGER.info("Server did not honour range request for %s, restarting", asset.name)
                        try:
                            target_path.unlink()
                        except OSError:
                            LOGGER.debug("Failed to remove file %s during restart", target_path, exc_info=True)
                        offset = 0
                        attempts -= 1
                        time.sleep(retry_delay)
                        continue

                    mode = "ab" if offset else "wb"
                    with open(target_path, mode) as handle:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            handle.write(chunk)

                final_size = target_path.stat().st_size
                if final_size < asset.size:
                    LOGGER.warning(
                        "Download incomplete for %s: got %s of %s bytes",
                        asset.name,
                        final_size,
                        asset.size,
                    )
                    time.sleep(retry_delay)
                    continue

                downloaded.append(target_path)
                completed = True
            except (requests.RequestException, OSError) as exc:
                LOGGER.warning("Download failed for %s: %s", asset.name, exc)
                time.sleep(retry_delay)

        if not completed:
            raise GitHubError(f"Exceeded retries downloading asset {asset.name}")

    return downloaded
