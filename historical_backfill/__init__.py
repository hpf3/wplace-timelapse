"""
Historical backfill utilities for integrating archived WPlace tiles with the
timelapse pipeline.
"""

from .cli import main
from .generator import GenerationSummary, HistoricalGenerator
from .github_client import GitHubError, fetch_recent_releases, fetch_release_by_tag
from .models import BackfillRequest, BoundingBox, RegionSpec

__all__ = [
    "main",
    "BackfillRequest",
    "BoundingBox",
    "RegionSpec",
    "HistoricalGenerator",
    "GenerationSummary",
    "GitHubError",
    "fetch_recent_releases",
    "fetch_release_by_tag",
]
