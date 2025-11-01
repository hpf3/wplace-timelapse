"""
Timelapse Backup System for WPlace Tiles
Automatically downloads tile images every 5 minutes and creates daily timelapses.
"""

import os
import json
import logging
import cv2
import numpy as np
import shutil
import time
import subprocess
import uuid
import threading
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Template
from typing import List, Tuple, Dict, Any, Optional, Iterable, Iterator, Union
from dotenv import load_dotenv
from timelapse_backup.config import (
    DiffSettings,
    GlobalSettings,
    TimelapseConfig,
    _parse_background_color as config_parse_background_color,
    _parse_bool as config_parse_bool,
    _parse_positive_int as config_parse_positive_int,
    load_config,
)
from timelapse_backup.full_timelapse import (
    FullTimelapseSegment,
    FullTimelapseState,
)
from timelapse_backup.logging_setup import configure_logging
from timelapse_backup.models import (
    CompositeFrame,
    FrameManifest,
    PreparedFrame,
    RenderedTimelapseResult,
)
from timelapse_backup.stats import TimelapseStatsCollector, build_stats_report
from timelapse_backup.sessions import (
    get_all_sessions,
    get_prior_sessions,
    get_session_dirs_for_date,
    parse_session_datetime,
)
from timelapse_backup.manifests import ManifestBuilder
from timelapse_backup.rendering import Renderer
from timelapse_backup.tiles import TileDownloader
from timelapse_backup import scheduler as scheduler_module

# Load environment variables
load_dotenv()


@dataclass
class StatsGenerationJob:
    job_id: str
    slug: str
    mode_name: str
    suffix: str
    output_filename: str
    timelapse_config: Dict[str, Any]
    start: Optional[datetime]
    end: Optional[datetime]
    label: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = field(default=None, repr=False)


class TimelapseBackup:
    PLACEHOLDER_SUFFIX = '.placeholder'
    DEFAULT_PUBLIC_LISTING_BASE = ""
    PUBLIC_LISTING_ENV = "PUBLIC_LISTING_BASE_URL"

    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config_path = Path(config_file)
        # Historical data imported prior to this cutoff is treated as baseline-only.
        self.historical_cutoff: Optional[datetime] = datetime(2025, 10, 13)

        config_data = load_config(self.config_path)
        self.config_data = config_data
        self._apply_global_settings(config_data.global_settings)
        self.config = {
            'timelapses': [self._timelapse_to_dict(timelapse) for timelapse in config_data.timelapses],
            'global_settings': self._global_settings_to_dict(config_data.global_settings),
        }
        self.public_listing_base_url = os.environ.get(
            self.PUBLIC_LISTING_ENV,
            self.DEFAULT_PUBLIC_LISTING_BASE,
        )

        # Setup logging
        self.setup_logging()

        # Initialize tile downloader helper
        self._get_tile_downloader()

        # Create directories
        self.backup_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self._ensure_latest_capture_pages()

        # Stats job tracking
        self._stats_jobs: Dict[str, StatsGenerationJob] = {}
        self._stats_job_index: Dict[Tuple[str, str], str] = {}
        self._stats_jobs_lock = threading.Lock()

    def _apply_global_settings(self, settings: GlobalSettings) -> None:
        """Populate instance attributes from parsed global settings."""
        self.base_url = settings.base_url
        self.backup_interval = settings.backup_interval_minutes
        self.backup_dir = settings.backup_dir
        self.output_dir = settings.output_dir
        self.request_delay = settings.request_delay
        self.fps = settings.timelapse_fps
        self.quality = settings.timelapse_quality
        self.background_color = settings.background_color
        self.auto_crop_transparent_frames = settings.auto_crop_transparent_frames
        self.frame_prep_workers = settings.frame_prep_workers

        diff = settings.diff_settings
        self.diff_threshold = diff.threshold
        self.diff_visualization = diff.visualization
        self.diff_fade_frames = diff.fade_frames
        self.diff_enhancement_factor = diff.enhancement_factor

        reporting = settings.reporting
        self.reporting_enabled = reporting.enable_stats_file
        self.seconds_per_pixel = reporting.seconds_per_pixel
        self.coverage_gap_multiplier = reporting.coverage_gap_multiplier

    def _timelapse_to_dict(self, config: TimelapseConfig) -> Dict[str, Any]:
        """Convert a dataclass timelapse config into the legacy dictionary shape."""
        coordinates = {
            'xmin': config.coordinates.xmin,
            'xmax': config.coordinates.xmax,
            'ymin': config.coordinates.ymin,
            'ymax': config.coordinates.ymax,
        }
        modes: Dict[str, Dict[str, Any]] = {}
        for name, mode in config.timelapse_modes.items():
            modes[name] = {
                'enabled': mode.enabled,
                'suffix': mode.suffix,
                'create_full_timelapse': mode.create_full_timelapse,
            }
        return {
            'slug': config.slug,
            'name': config.name,
            'description': config.description,
            'coordinates': coordinates,
            'enabled': config.enabled,
            'timelapse_modes': modes,
        }

    def _global_settings_to_dict(self, settings: GlobalSettings) -> Dict[str, Any]:
        """Convert parsed global settings into a legacy-compatible dictionary."""
        return {
            'base_url': settings.base_url,
            'backup_interval_minutes': settings.backup_interval_minutes,
            'backup_dir': str(settings.backup_dir),
            'output_dir': str(settings.output_dir),
            'request_delay': settings.request_delay,
            'timelapse_fps': settings.timelapse_fps,
            'timelapse_quality': settings.timelapse_quality,
            'background_color': list(settings.background_color),
            'auto_crop_transparent_frames': settings.auto_crop_transparent_frames,
            'frame_prep_workers': settings.frame_prep_workers,
            'diff_settings': {
                'threshold': settings.diff_settings.threshold,
                'visualization': settings.diff_settings.visualization,
                'fade_frames': settings.diff_settings.fade_frames,
                'enhancement_factor': settings.diff_settings.enhancement_factor,
            },
            'reporting': {
                'enable_stats_file': settings.reporting.enable_stats_file,
                'seconds_per_pixel': settings.reporting.seconds_per_pixel,
                'coverage_gap_multiplier': settings.reporting.coverage_gap_multiplier,
            },
        }

    def _background_color_rgb(self) -> Tuple[int, int, int]:
        """Return the configured background color as an RGB tuple."""
        color = getattr(self, "background_color", (0, 0, 0))
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            return (0, 0, 0)
        # Stored as BGR for OpenCV usage; convert to RGB for web output.
        b, g, r = color
        return (int(r), int(g), int(b))

    def _background_color_hex(self) -> str:
        """Return the configured background color as a CSS hex string."""
        r, g, b = self._background_color_rgb()
        return f"#{r:02x}{g:02x}{b:02x}"

    def _listing_url_for_slug(self, slug: str) -> str:
        """Build the remote listing URL for a timelapse slug."""
        base_hint = (self.public_listing_base_url or self.DEFAULT_PUBLIC_LISTING_BASE or "").strip()
        if not base_hint:
            return ""
        base = base_hint.rstrip("/")
        return f"{base}/{slug}/?raw"

    def _build_latest_capture_page_html(
        self,
        slug: str,
        display_name: str,
        listing_url: str,
        listing_base: str,
        tile_grid: List[List[Dict[str, Any]]],
        grid_rows: int,
        grid_columns: int,
    ) -> str:
        """Construct the static HTML page that points to the newest public artifact."""
        background_hex = self._background_color_hex()
        title = f"{display_name} latest backup"
        tile_grid_json = json.dumps(tile_grid or [])
        template = Template(
            r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --tile-bg: $background_hex;
    }
    body {
      min-height: 100vh;
      margin: 0;
      background: var(--tile-bg);
      color: #f8fbff;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem 1.5rem;
    }
    main {
      width: min(720px, 100%);
      background: rgba(0, 0, 0, 0.45);
      border-radius: 18px;
      padding: 2rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
    }
    h1, h2 {
      margin-top: 0;
      font-weight: 600;
    }
    a {
      color: #ffe082;
    }
    a:hover,
    a:focus {
      color: #fff3c1;
    }
    .status {
      margin-bottom: 1.5rem;
      font-size: 1.05rem;
    }
    .latest {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: baseline;
      margin-bottom: 1.5rem;
      font-size: 1.1rem;
    }
    .latest .label {
      font-weight: 600;
    }
    .age {
      opacity: 0.75;
      font-size: 0.95rem;
    }
    #preview-wrapper {
      margin-top: 1rem;
      border-radius: 12px;
      overflow: hidden;
      padding: 0.35rem;
      background: var(--tile-bg);
      display: flex;
      justify-content: center;
    }
    #preview-wrapper img,
    #preview-wrapper video {
      max-width: 100%;
      width: 100%;
      display: block;
      background: var(--tile-bg);
    }
    .tile-grid {
      display: grid;
      grid-template-columns: repeat(var(--grid-columns, 1), minmax(0, 1fr));
      gap: 0;
      background: var(--tile-bg);
      border-radius: 8px;
      overflow: hidden;
    }
    .tile-cell {
      position: relative;
      background: var(--tile-bg);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .tile-cell img {
      width: 100%;
      height: auto;
      display: block;
      background: var(--tile-bg);
    }
    .tile-cell.missing {
      color: rgba(248, 251, 255, 0.7);
      font-size: 0.9rem;
      min-height: 40px;
    }
    .tile-cell .tile-time {
      position: absolute;
      bottom: 6px;
      right: 8px;
      background: rgba(0, 0, 0, 0.55);
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 0.7rem;
    }
    ol {
      padding-left: 1.25rem;
    }
    li + li {
      margin-top: 0.35rem;
    }
    .manual {
      margin-top: 2rem;
      font-size: 0.95rem;
      opacity: 0.85;
    }
    @media (max-width: 640px) {
      body {
        padding: 1.5rem 1rem;
      }
      main {
        padding: 1.5rem;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>$heading latest backup</h1>
    <p id="status" class="status">Looking for the newest file...</p>
    <p class="latest" id="latest-row" hidden>
      <span class="label">Latest:</span>
      <a id="latest-link" href="#" target="_blank" rel="noopener">Loading</a>
      <span id="latest-age" class="age"></span>
    </p>
    <section id="preview" hidden>
      <h2>Preview</h2>
      <div id="preview-wrapper"></div>
    </section>
    <section id="recent" hidden>
      <h2>Recent files</h2>
      <ol id="recent-list"></ol>
    </section>
    <p class="manual">If this page cannot load automatically, open the <a id="listing-link" href="$listing_url" target="_blank" rel="noopener">raw listing</a>.</p>
    <noscript>This page needs JavaScript to locate the latest backup.</noscript>
  </main>
  <script>
  (function () {
    var RAW_LISTING_URL = "$listing_url";
    var TIMELAPSE_SLUG = "$slug";
    var LISTING_BASE_HINT = "$listing_base";
    var TILE_GRID = $tile_grid;
    var GRID_ROWS = $grid_rows;
    var GRID_COLUMNS = $grid_columns;
    var BACKGROUND_HEX = "$background_hex";
    var MAX_RECENT = 5;
    var statusEl = document.getElementById("status");
    var latestRow = document.getElementById("latest-row");
    var latestLink = document.getElementById("latest-link");
    var latestAge = document.getElementById("latest-age");
    var previewSection = document.getElementById("preview");
    var previewWrapper = document.getElementById("preview-wrapper");
    var recentSection = document.getElementById("recent");
    var recentList = document.getElementById("recent-list");
    var listingLink = document.getElementById("listing-link");

    var imageExts = new Set(["png", "jpg", "jpeg", "gif", "webp"]);
    var videoExts = new Set(["mp4", "webm", "mov", "mkv"]);
    var allowedExts = new Set(Array.from(imageExts).concat(Array.from(videoExts)));

    var tileRows = Array.isArray(TILE_GRID) ? TILE_GRID : [];
    var flattenedTiles = [];
    tileRows.forEach(function (row) {
      if (!Array.isArray(row)) {
        return;
      }
      row.forEach(function (cell) {
        if (!cell || typeof cell.x !== "number" || typeof cell.y !== "number") {
          return;
        }
        var key = cell.key || (cell.x + "_" + cell.y);
        flattenedTiles.push({ x: cell.x, y: cell.y, key: key });
      });
    });

    var inferredColumns = GRID_COLUMNS > 0 ? GRID_COLUMNS : (tileRows.length && Array.isArray(tileRows[0]) ? tileRows[0].length : 0);
    if (!inferredColumns && flattenedTiles.length) {
      inferredColumns = flattenedTiles.length;
    }

    function resolveListingUrl() {
      var manual = (RAW_LISTING_URL || "").trim();
      if (manual) {
        return manual;
      }
      var slug = (TIMELAPSE_SLUG || "").trim();
      if (!slug) {
        return "";
      }
      var baseHint = (LISTING_BASE_HINT || "").trim();
      try {
        var baseUrl = baseHint ? new URL(baseHint, window.location.href) : new URL("../images/", window.location.href);
        var href = baseUrl.href;
        if (!href.endsWith("/")) {
          href += "/";
        }
        return new URL(slug + "/?raw", href).href;
      } catch (_error) {
        return "";
      }
    }

    var LISTING_URL = resolveListingUrl();

    var listingBaseSegments = [];
    try {
      if (LISTING_URL) {
        var listingUrlObject = new URL(LISTING_URL);
        listingBaseSegments = listingUrlObject.pathname.split("/").filter(Boolean);
      }
    } catch (error) {
      console.warn("Unable to parse listing URL", error);
    }

    function decodeSegment(segment) {
      try {
        return decodeURIComponent(segment);
      } catch (_error) {
        return segment;
      }
    }

    function normalizedSegments(segments) {
      return segments.map(decodeSegment);
    }

    function extractEpoch(pathname, segments, filename) {
      var isoMatch = pathname.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
      if (isoMatch) {
        var iso = isoMatch[1] + "-" + isoMatch[2] + "-" + isoMatch[3] + "T" + isoMatch[4] + ":" + isoMatch[5] + ":" + isoMatch[6] + "Z";
        var parsedIso = Date.parse(iso);
        if (!isNaN(parsedIso)) {
          return { epoch: parsedIso, iso: iso };
        }
      }

      var datePart = null;
      var timePart = null;
      for (var i = 0; i < segments.length; i++) {
        var seg = segments[i];
        if (!datePart && /^\d{4}-\d{2}-\d{2}$$/.test(seg)) {
          datePart = seg;
          var next = segments[i + 1];
          if (!timePart && next && /^\d{2}[-:]\d{2}[-:]\d{2}$$/.test(next)) {
            timePart = next;
          }
          continue;
        }
        if (!timePart && /^\d{2}[-:]\d{2}[-:]\d{2}$$/.test(seg)) {
          timePart = seg;
        }
      }

      if (datePart && !timePart) {
        var fileMatch = filename.match(/(\d{2})[-_](\d{2})[-_](\d{2})/);
        if (fileMatch) {
          timePart = fileMatch[1] + "-" + fileMatch[2] + "-" + fileMatch[3];
        }
      }

      if (datePart) {
        var isoGuess = datePart + "T" + (timePart ? timePart.replace(/[-_]/g, ":") : "00:00:00") + "Z";
        var parsedGuess = Date.parse(isoGuess);
        if (!isNaN(parsedGuess)) {
          return { epoch: parsedGuess, iso: isoGuess };
        }
      }

      var fallback = filename.match(/(\d{4})-(\d{2})-(\d{2})(?:[T_](\d{2})[-:]?(\d{2})[-:]?(\d{2}))?/);
      if (fallback) {
        var fallbackIso = fallback[1] + "-" + fallback[2] + "-" + fallback[3] + "T" + (fallback[4] || "00") + ":" + (fallback[5] || "00") + ":" + (fallback[6] || "00") + "Z";
        var parsedFallback = Date.parse(fallbackIso);
        if (!isNaN(parsedFallback)) {
          return { epoch: parsedFallback, iso: fallbackIso };
        }
      }

      return { epoch: null, iso: null };
    }

    function parseEntry(raw) {
      if (!raw) {
        return null;
      }
      var trimmed = raw.trim();
      if (!trimmed) {
        return null;
      }
      var urlObject;
      try {
        urlObject = new URL(trimmed);
      } catch (_error) {
        return null;
      }
      var pathname = urlObject.pathname || "";
      if (pathname.endsWith("/")) {
        return null;
      }
      var segments = pathname.split("/").filter(Boolean);
      if (!segments.length) {
        return null;
      }
      var decodedSegmentsFull = normalizedSegments(segments);
      var relativeSegments = segments.slice();
      if (listingBaseSegments.length <= segments.length) {
        var matchesPrefix = true;
        for (var i = 0; i < listingBaseSegments.length; i++) {
          if (segments[i] !== listingBaseSegments[i]) {
            matchesPrefix = false;
            break;
          }
        }
        if (matchesPrefix) {
          relativeSegments = segments.slice(listingBaseSegments.length);
        }
      }
      var decodedRelativeSegments = normalizedSegments(relativeSegments);
      var filename = decodedRelativeSegments.length ? decodedRelativeSegments[decodedRelativeSegments.length - 1] : (decodedSegmentsFull[decodedSegmentsFull.length - 1] || "");
      if (!filename) {
        return null;
      }
      var lower = filename.toLowerCase();
      var dotIndex = lower.lastIndexOf(".");
      if (dotIndex === -1) {
        return null;
      }
      var ext = lower.slice(dotIndex + 1);
      if (!allowedExts.has(ext)) {
        return null;
      }

      var coordMatch = filename.match(/(\d+)_(\d+)/);
      var coordKey = null;
      var coordX = null;
      var coordY = null;
      if (coordMatch) {
        coordX = parseInt(coordMatch[1], 10);
        coordY = parseInt(coordMatch[2], 10);
        if (!Number.isNaN(coordX) && !Number.isNaN(coordY)) {
          coordKey = coordX + "_" + coordY;
        }
      }

      var epochInfo = extractEpoch(pathname, decodedSegmentsFull, filename);
      var relativePath = decodedRelativeSegments.length ? decodedRelativeSegments.join("/") : filename;

      return {
        url: trimmed,
        name: filename,
        ext: ext,
        mediaType: imageExts.has(ext) ? "image" : (videoExts.has(ext) ? "video" : null),
        epoch: epochInfo.epoch,
        iso: epochInfo.iso,
        relativePath: relativePath,
        coordKey: coordKey,
        coordX: coordX,
        coordY: coordY
      };
    }

    function buildTileMap(entries) {
      var map = Object.create(null);
      entries.forEach(function (entry) {
        if (!entry || !entry.coordKey) {
          return;
        }
        var existing = map[entry.coordKey];
        if (!existing) {
          map[entry.coordKey] = entry;
          return;
        }
        if (entry.epoch && (!existing.epoch || entry.epoch > existing.epoch)) {
          map[entry.coordKey] = entry;
        }
      });
      return map;
    }

    function latestFromTileMap(tileMap, fallbackList) {
      var latest = null;
      Object.keys(tileMap).forEach(function (key) {
        var entry = tileMap[key];
        if (!entry) {
          return;
        }
        if (!latest) {
          latest = entry;
          return;
        }
        if (entry.epoch && (!latest.epoch || entry.epoch > latest.epoch)) {
          latest = entry;
        }
      });
      if (!latest && fallbackList.length) {
        latest = fallbackList[0];
      }
      return latest;
    }

    function describeAge(epoch) {
      if (!epoch) {
        return "";
      }
      var diffMs = Date.now() - epoch;
      if (diffMs < 0) {
        diffMs = 0;
      }
      var diffMinutes = Math.floor(diffMs / 60000);
      if (diffMinutes < 1) {
        return "updated just now";
      }
      if (diffMinutes < 60) {
        return "updated " + diffMinutes + " minute" + (diffMinutes === 1 ? "" : "s") + " ago";
      }
      var diffHours = Math.floor(diffMinutes / 60);
      if (diffHours < 48) {
        return "updated " + diffHours + " hour" + (diffHours === 1 ? "" : "s") + " ago";
      }
      var diffDays = Math.floor(diffHours / 24);
      return "updated " + diffDays + " day" + (diffDays === 1 ? "" : "s") + " ago";
    }

    function setLatest(entry) {
      if (!entry) {
        latestRow.hidden = true;
        return;
      }
      latestLink.href = entry.url;
      latestLink.textContent = entry.relativePath || entry.name;
      latestAge.textContent = entry.epoch ? "(" + describeAge(entry.epoch) + ")" : "";
      latestRow.hidden = false;
    }

    function renderComposite(tileMap) {
      previewWrapper.innerHTML = "";
      if (!flattenedTiles.length) {
        return false;
      }
      var grid = document.createElement("div");
      grid.className = "tile-grid";
      var columns = inferredColumns || 1;
      grid.style.setProperty("--grid-columns", String(columns));
      tileRows.forEach(function (row) {
        if (!Array.isArray(row)) {
          return;
        }
        row.forEach(function (cell) {
          var key = cell && (cell.key || (cell.x + "_" + cell.y));
          var entry = key ? tileMap[key] : null;
          var cellEl = document.createElement("div");
          cellEl.className = "tile-cell";
          if (entry && entry.mediaType === "image") {
            var img = document.createElement("img");
            img.src = entry.url;
            img.alt = key ? ("Tile " + key) : "Tile";
            img.loading = "lazy";
            img.style.backgroundColor = BACKGROUND_HEX;
            cellEl.appendChild(img);
            if (entry.iso) {
              var time = document.createElement("time");
              time.dateTime = entry.iso;
              time.textContent = new Date(entry.iso).toLocaleTimeString();
              time.className = "tile-time";
              cellEl.appendChild(time);
            }
          } else {
            cellEl.classList.add("missing");
            var label = document.createElement("span");
            label.textContent = key || "missing";
            cellEl.appendChild(label);
          }
          grid.appendChild(cellEl);
        });
      });
      previewWrapper.appendChild(grid);
      return true;
    }

    function renderRecent(entries) {
      recentList.innerHTML = "";
      var fragment = document.createDocumentFragment();
      entries.forEach(function (entry) {
        var li = document.createElement("li");
        var anchor = document.createElement("a");
        anchor.href = entry.url;
        anchor.textContent = entry.relativePath;
        anchor.target = "_blank";
        anchor.rel = "noopener";
        li.appendChild(anchor);
        if (entry.iso) {
          var time = document.createElement("time");
          time.dateTime = entry.iso;
          time.textContent = new Date(entry.iso).toLocaleString();
          li.appendChild(document.createTextNode(" — "));
          li.appendChild(time);
        }
        fragment.appendChild(li);
      });
      recentList.appendChild(fragment);
      recentSection.hidden = entries.length <= 1;
    }

    function handleError(error) {
      console.error(error);
      statusEl.textContent = "Could not load the latest backup (" + error.message + ").";
      statusEl.hidden = false;
      latestRow.hidden = true;
      previewSection.hidden = true;
      recentSection.hidden = true;
      if (LISTING_URL) {
        listingLink.href = LISTING_URL;
      } else {
        listingLink.removeAttribute("href");
      }
    }

    function updateStatus(message, hidden) {
      statusEl.textContent = message;
      statusEl.hidden = hidden;
    }

    function loadListing() {
      return fetch(LISTING_URL, { cache: "no-store" });
    }

    async function refresh() {
      try {
        updateStatus("Looking for the newest file...", false);
        updateAuxiliaryLinks();
        if (!LISTING_URL) {
          throw new Error("Listing URL is not configured");
        }
        var response = await loadListing();
        if (!response.ok) {
          throw new Error("HTTP " + response.status + " " + response.statusText);
        }
        var text = await response.text();
        var rawEntries = text.split(/\r?\n/).map(function (line) { return line.trim(); }).filter(Boolean);
        var entries = rawEntries.map(parseEntry).filter(function (entry) { return entry; });
        if (!entries.length) {
          throw new Error("No compatible files found");
        }
        entries.sort(function (a, b) {
          if (a.epoch && b.epoch) {
            return b.epoch - a.epoch;
          }
          if (a.epoch) {
            return -1;
          }
          if (b.epoch) {
            return 1;
          }
          return 0;
        });
        var tileMap = buildTileMap(entries);
        var latest = latestFromTileMap(tileMap, entries);
        setLatest(latest);
        var hasComposite = renderComposite(tileMap);
        previewSection.hidden = !hasComposite;
        renderRecent(entries.slice(0, MAX_RECENT));
        updateStatus("Latest file loaded.", true);
      } catch (error) {
        handleError(error);
      }
    }

    function updateAuxiliaryLinks() {
      if (LISTING_URL) {
        listingLink.href = LISTING_URL;
      } else {
        listingLink.removeAttribute("href");
      }
    }

    refresh();
  }());
  </script>
</body>
</html>
"""
        )
        return template.substitute(
            title=title,
            heading=display_name,
            slug=slug,
            listing_url=listing_url,
            listing_base=listing_base,
            background_hex=background_hex,
            tile_grid=tile_grid_json,
            grid_rows=grid_rows,
            grid_columns=grid_columns,
        )

    def _write_latest_capture_page(
        self,
        slug: str,
        display_name: str,
        timelapse_config: Dict[str, Any],
    ) -> Optional[Path]:
        """Ensure the helper HTML page for a given slug is present at the backup root."""
        listing_url = self._listing_url_for_slug(slug)
        listing_base = (self.public_listing_base_url or self.DEFAULT_PUBLIC_LISTING_BASE or "").strip()
        backup_root = Path(self.backup_dir)
        page_path = backup_root / f"click_for_latest_{slug}.html"
        tile_grid = self._tile_grid_for_timelapse(timelapse_config)
        grid_rows = len(tile_grid)
        grid_columns = len(tile_grid[0]) if grid_rows else 0
        html = self._build_latest_capture_page_html(
            slug,
            display_name,
            listing_url,
            listing_base,
            tile_grid,
            grid_rows,
            grid_columns,
        )
        existing: Optional[str] = None
        if page_path.exists():
            try:
                existing = page_path.read_text(encoding="utf-8")
            except Exception:
                existing = None
        if existing == html:
            return page_path
        try:
            page_path.write_text(html, encoding="utf-8")
        except Exception as error:
            self.logger.warning(
                "Unable to write latest page for '%s': %s",
                slug,
                error,
            )
            return None

        legacy_page = backup_root / slug / "click_for_latest_image.html"
        if legacy_page.exists() and legacy_page.is_file():
            try:
                legacy_page.unlink()
            except Exception as error:
                self.logger.debug(
                    "Unable to remove legacy latest page for '%s': %s",
                    slug,
                    error,
                )
        return page_path

    def _build_index_page_html(
        self,
        entries: List[Tuple[str, str, str]],
    ) -> str:
        """Build an index page linking to each helper page."""
        background_hex = self._background_color_hex()
        title = "Latest backup index"
        if entries:
            list_items = "\n".join(
                f'      <li><a href="{filename}" rel="noopener">{display_name}</a></li>'
                for _, display_name, filename in entries
            )
        else:
            list_items = "      <li>No timelapses configured.</li>"
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            f"  <title>{title}</title>\n"
            "  <style>\n"
            "    body {\n"
            "      min-height: 100vh;\n"
            "      margin: 0;\n"
            f"      background: {background_hex};\n"
            "      color: #f8fbff;\n"
            "      display: flex;\n"
            "      align-items: center;\n"
            "      justify-content: center;\n"
            "      font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;\n"
            "      padding: 2rem 1.5rem;\n"
            "    }\n"
            "    main {\n"
            "      width: min(540px, 100%);\n"
            "      background: rgba(0, 0, 0, 0.45);\n"
            "      border-radius: 18px;\n"
            "      padding: 2rem;\n"
            "      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);\n"
            "    }\n"
            "    h1 {\n"
            "      margin-top: 0;\n"
            "      font-weight: 600;\n"
            "    }\n"
            "    ul {\n"
            "      list-style: none;\n"
            "      padding: 0;\n"
            "      margin: 1.5rem 0 0;\n"
            "      display: grid;\n"
            "      gap: 0.75rem;\n"
            "    }\n"
            "    a {\n"
            "      color: #ffe082;\n"
            "      font-size: 1.05rem;\n"
            "      text-decoration: none;\n"
            "    }\n"
            "    a:hover,\n"
            "    a:focus {\n"
            "      color: #fff3c1;\n"
            "      text-decoration: underline;\n"
            "    }\n"
            "    @media (max-width: 640px) {\n"
            "      body {\n"
            "        padding: 1.5rem 1rem;\n"
            "      }\n"
            "      main {\n"
            "        padding: 1.5rem;\n"
            "      }\n"
            "    }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            "  <main>\n"
            f"    <h1>{title}</h1>\n"
            "    <p>Select a timelapse to jump to the latest capture helper page.</p>\n"
            "    <ul>\n"
            f"{list_items}\n"
            "    </ul>\n"
            "  </main>\n"
            "</body>\n"
            "</html>\n"
        )

    def _write_index_page(self, entries: List[Tuple[str, str, str]]) -> None:
        """Write the top-level index pointing to each helper page."""
        index_path = Path(self.backup_dir) / "click_for_latest_image.html"
        html = self._build_index_page_html(entries)
        existing: Optional[str] = None
        if index_path.exists():
            try:
                existing = index_path.read_text(encoding="utf-8")
            except Exception:
                existing = None
        if existing == html:
            return
        try:
            index_path.write_text(html, encoding="utf-8")
        except Exception as error:
            self.logger.warning("Unable to write index helper page: %s", error)

    def _ensure_latest_capture_pages(self) -> None:
        """Generate helper pages at the backup root and an index page."""
        entries: List[Tuple[str, str, str]] = []
        for timelapse in self.config.get("timelapses", []):
            slug = timelapse.get("slug")
            if not slug:
                continue
            name = timelapse.get("name") or slug
            page_path = self._write_latest_capture_page(slug, name, timelapse)
            if page_path is not None:
                entries.append((slug, name, page_path.name))
        self._write_index_page(entries)

    def _parse_background_color(self, value: Any) -> Tuple[int, int, int]:
        """Backward-compatible wrapper for the background color parser."""
        return config_parse_background_color(value)

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        """Backward-compatible wrapper for boolean parsing."""
        return config_parse_bool(value, default)

    @staticmethod
    def _parse_positive_int(value: Any, default: int) -> int:
        """Backward-compatible wrapper for positive integer parsing."""
        return config_parse_positive_int(value, default)

    def get_prior_sessions(self, slug: str, current_session: Path) -> List[Path]:
        """Delegate to session helper for compatibility."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None or not slug:
            return []
        return get_prior_sessions(backup_root, slug, current_session)

    def get_session_dirs_for_date(self, slug: str, date_str: str) -> List[Path]:
        """Delegate to session helper for compatibility."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None:
            return []
        return get_session_dirs_for_date(backup_root, slug, date_str)

    def get_all_sessions(self, slug: str) -> List[Path]:
        """Delegate to session helper for compatibility."""
        backup_root = getattr(self, "backup_dir", None)
        if backup_root is None:
            return []
        return get_all_sessions(backup_root, slug)

    def _get_tile_downloader(self) -> TileDownloader:
        if not hasattr(self, "_tile_downloader"):
            logger = getattr(self, "logger", logging.getLogger(__name__))
            base_url = getattr(self, "base_url", "")
            self._tile_downloader = TileDownloader(
                base_url=base_url,
                logger=logger,
                placeholder_suffix=self.PLACEHOLDER_SUFFIX,
            )
        return self._tile_downloader

    def _get_manifest_builder(self) -> ManifestBuilder:
        tile_downloader = self._get_tile_downloader()
        current_logger = getattr(self, "logger", logging.getLogger(__name__))
        builder = getattr(self, "_manifest_builder", None)
        if (
            builder is None
            or builder.tile_downloader is not tile_downloader
            or builder.background_color != self.background_color
            or builder.logger is not current_logger
        ):
            builder = ManifestBuilder(
                tile_downloader=tile_downloader,
                background_color=self.background_color,
                logger=current_logger,
            )
            self._manifest_builder = builder
        return builder

    def _placeholder_filename(self, filename: str) -> str:
        return self._get_tile_downloader().placeholder_filename(filename)

    def _placeholder_path(self, session_dir: Path, filename: str) -> Path:
        return self._get_tile_downloader().placeholder_path(session_dir, filename)

    def _write_placeholder(self, placeholder_path: Path, target_path: Path) -> bool:
        return self._get_tile_downloader().write_placeholder(placeholder_path, target_path)

    def _get_renderer(self) -> Renderer:
        builder = self._get_manifest_builder()
        current_logger = getattr(self, "logger", logging.getLogger(__name__))

        if hasattr(self, "config_data"):
            diff_settings = self.config_data.global_settings.diff_settings
        else:
            diff_settings = DiffSettings(
                threshold=getattr(self, "diff_threshold", 10),
                visualization=getattr(self, "diff_visualization", "colored"),
                fade_frames=getattr(self, "diff_fade_frames", 3),
                enhancement_factor=getattr(self, "diff_enhancement_factor", 2.0),
            )

        renderer = getattr(self, "_renderer", None)
        if (
            renderer is None
            or renderer.manifest_builder is not builder
            or renderer.logger is not current_logger
            or renderer.frame_prep_workers != max(1, getattr(self, "frame_prep_workers", 1))
            or renderer.auto_crop_transparent_frames != getattr(self, "auto_crop_transparent_frames", True)
            or renderer.diff_settings != diff_settings
            or renderer.historical_cutoff != getattr(self, "historical_cutoff", None)
        ):
            renderer = Renderer(
                manifest_builder=builder,
                logger=current_logger,
                frame_prep_workers=getattr(self, "frame_prep_workers", 1),
                auto_crop_transparent_frames=getattr(self, "auto_crop_transparent_frames", True),
                diff_settings=diff_settings,
                historical_cutoff=getattr(self, "historical_cutoff", None),
            )
            self._renderer = renderer
        return renderer


    def get_enabled_timelapses(self) -> List[Dict[str, Any]]:
        """Get list of enabled timelapse configurations"""
        return [tl for tl in self.config['timelapses'] if tl.get('enabled', True)]

    def get_last_capture_time(self) -> Optional[datetime]:
        """Return the most recent session timestamp across enabled timelapses."""
        latest: Optional[datetime] = None
        for timelapse in self.get_enabled_timelapses():
            slug = timelapse.get("slug")
            if not slug:
                continue
            sessions = get_all_sessions(self.backup_dir, slug)
            if not sessions:
                continue
            session_time = parse_session_datetime(sessions[-1])
            if session_time is None:
                continue
            if latest is None or session_time > latest:
                latest = session_time
        return latest

    def get_enabled_timelapse_modes(
        self,
        timelapse_config: Union[Dict[str, Any], TimelapseConfig],
    ) -> List[Dict[str, Any]]:
        """Return enabled rendering modes for a timelapse in legacy dict format."""
        enabled_modes: List[Dict[str, Any]] = []

        if isinstance(timelapse_config, TimelapseConfig):
            for mode in timelapse_config.enabled_modes():
                enabled_modes.append(
                    {
                        "mode": mode.name,
                        "suffix": mode.suffix,
                        "create_full": mode.create_full_timelapse,
                    }
                )
        else:
            raw_modes = {}
            if isinstance(timelapse_config, dict):
                raw_modes = timelapse_config.get("timelapse_modes", {})

            if isinstance(raw_modes, dict):
                for mode_name, raw_mode in raw_modes.items():
                    if not isinstance(raw_mode, dict):
                        continue
                    if not self._parse_bool(raw_mode.get("enabled"), True):
                        continue
                    create_full_value = raw_mode.get("create_full")
                    if create_full_value is None:
                        create_full_value = raw_mode.get("create_full_timelapse")
                    enabled_modes.append(
                        {
                            "mode": mode_name,
                            "suffix": str(raw_mode.get("suffix", "")),
                            "create_full": self._parse_bool(create_full_value, False),
                        }
                    )
            elif isinstance(raw_modes, list):
                for entry in raw_modes:
                    if isinstance(entry, str):
                        enabled_modes.append(
                            {"mode": entry, "suffix": "", "create_full": False}
                        )
                    elif isinstance(entry, dict):
                        mode_name = str(entry.get("mode") or entry.get("name") or "normal")
                        if not self._parse_bool(entry.get("enabled"), True):
                            continue
                        create_full_value = entry.get("create_full")
                        if create_full_value is None:
                            create_full_value = entry.get("create_full_timelapse")
                        enabled_modes.append(
                            {
                                "mode": mode_name,
                                "suffix": str(entry.get("suffix", "")),
                                "create_full": self._parse_bool(create_full_value, False),
                            }
                        )

        if not enabled_modes:
            enabled_modes.append(
                {
                    "mode": "normal",
                    "suffix": "",
                    "create_full": False,
                }
            )
        return enabled_modes
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = configure_logging(logger_name=__name__)
        
    def get_tile_coordinates(self, timelapse_config: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Generate list of tile coordinates to download for a specific timelapse"""
        coords = timelapse_config['coordinates']
        coordinates = []
        for x in range(coords['xmin'], coords['xmax'] + 1):
            for y in range(coords['ymin'], coords['ymax'] + 1):
                coordinates.append((x, y))
        return coordinates
    
    def _tile_grid_for_timelapse(self, timelapse_config: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Return coordinates arranged by rows (y ascending) for preview rendering."""
        coords = timelapse_config['coordinates']
        grid: List[List[Dict[str, Any]]] = []
        for y in range(coords['ymin'], coords['ymax'] + 1):
            row: List[Dict[str, Any]] = []
            for x in range(coords['xmin'], coords['xmax'] + 1):
                row.append({"x": x, "y": y, "key": f"{x}_{y}"})
            grid.append(row)
        return grid




    def build_previous_tile_map(
        self,
        prior_sessions: Iterable[Path],
        coordinates: Iterable[Tuple[int, int]],
    ) -> Dict[str, Path]:
        """Delegate to tile downloader for compatibility."""
        return self._get_tile_downloader().build_previous_tile_map(
            prior_sessions,
            coordinates,
        )

    def resolve_tile_image_path(
        self,
        session_dir: Path,
        x: int,
        y: int,
        prior_sessions: Optional[List[Path]] = None
    ) -> Optional[Path]:
        """Resolve actual image path for a tile, following placeholders if needed"""
        backup_root = getattr(self, "backup_dir", None)
        return self._get_tile_downloader().resolve_tile_image_path(
            session_dir,
            x,
            y,
            prior_sessions=prior_sessions,
            backup_root=backup_root,
        )
        
    def download_tile(
        self,
        slug: str,
        x: int,
        y: int,
        session_dir: Path,
        previous_tile_map: Dict[str, Path]
    ) -> Tuple[bool, bool]:
        """Download a single tile image and skip duplicates via placeholders.

        Returns:
            Tuple[bool, bool]: (success flag, placeholder created flag)
        """
        return self._get_tile_downloader().download_tile(
            slug,
            x,
            y,
            session_dir,
            previous_tile_map,
        )
            
    def backup_tiles(self):
        """Backup all tiles for current timestamp for all enabled timelapses"""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        
        enabled_timelapses = self.get_enabled_timelapses()
        self.logger.info(f"Starting backup session at {time_str} for {len(enabled_timelapses)} timelapses")
        
        total_successful = 0
        total_tiles = 0
        total_duplicates = 0
        
        for timelapse in enabled_timelapses:
            slug = timelapse['slug']
            name = timelapse['name']
            
            # Create session directory for this timelapse
            session_dir = self.backup_dir / slug / date_str / time_str
            session_dir.mkdir(parents=True, exist_ok=True)
            
            coordinates = self.get_tile_coordinates(timelapse)
            prior_sessions = get_prior_sessions(self.backup_dir, slug, session_dir)
            previous_tile_map = self.build_previous_tile_map(prior_sessions, coordinates)
            successful_downloads = 0
            duplicate_tiles = 0

            self.logger.info(f"Backing up '{name}' ({slug}): {len(coordinates)} tiles")
            
            for i, (x, y) in enumerate(coordinates):
                success, used_placeholder = self.download_tile(
                    slug,
                    x,
                    y,
                    session_dir,
                    previous_tile_map
                )
                if success:
                    successful_downloads += 1
                    if used_placeholder:
                        duplicate_tiles += 1
                
                # Add delay between requests (except for the last one)
                if i < len(coordinates) - 1:
                    time.sleep(self.request_delay)
                    
            if duplicate_tiles:
                self.logger.info(
                    f"'{name}' completed: {successful_downloads}/{len(coordinates)} tiles "
                    f"({duplicate_tiles} duplicates skipped)"
                )
            else:
                self.logger.info(f"'{name}' completed: {successful_downloads}/{len(coordinates)} tiles")
            
            total_successful += successful_downloads
            total_tiles += len(coordinates)
            total_duplicates += duplicate_tiles
            
            # Remove empty session directory if no downloads succeeded
            if successful_downloads == 0:
                try:
                    session_dir.rmdir()
                    # Try to remove parent directories if empty
                    if not any(session_dir.parent.iterdir()):
                        session_dir.parent.rmdir()
                        if not any(session_dir.parent.parent.iterdir()):
                            session_dir.parent.parent.rmdir()
                except:
                    pass
                    
        if total_duplicates:
            self.logger.info(
                f"Backup session completed: {total_successful}/{total_tiles} total tiles processed "
                f"({total_duplicates} duplicates skipped)"
            )
        else:
            self.logger.info(
                f"Backup session completed: {total_successful}/{total_tiles} total tiles downloaded"
            )
                
    def _build_manifest_for_session(
        self,
        session_dir: Path,
        timelapse_config: Dict[str, Any],
        coordinates: List[Tuple[int, int]],
        tile_cache: Optional[Dict[Tuple[int, int], Path]],
        prior_sessions: Optional[List[Path]] = None,
    ) -> Optional[FrameManifest]:
        """Resolve tile paths for a session using existing fallback logic."""
        backup_root = getattr(self, "backup_dir", None)
        builder = self._get_manifest_builder()
        return builder.build_manifest_for_session(
            session_dir,
            coordinates,
            backup_root=backup_root,
            slug=timelapse_config.get('slug'),
            prior_sessions=prior_sessions,
            tile_cache=tile_cache,
        )

    def _compose_frame_from_manifest(
        self,
        manifest: FrameManifest,
        coordinates: List[Tuple[int, int]],
        x_coords: List[int],
        y_coords: List[int],
        x_index_map: Dict[int, int],
        y_index_map: Dict[int, int],
    ) -> Optional[CompositeFrame]:
        """Compose a frame using tile paths resolved in a manifest."""
        return self._get_manifest_builder().compose_frame(manifest, coordinates)

    def create_composite_image(
        self,
        session_dir: Path,
        timelapse_config: Dict[str, Any],
        tile_cache: Optional[Dict[Tuple[int, int], Path]] = None
    ) -> Optional[CompositeFrame]:
        """Create a composite image from individual tiles and retain transparency."""
        coordinates = self.get_tile_coordinates(timelapse_config)
        builder = self._get_manifest_builder()
        backup_root = getattr(self, "backup_dir", None)
        return builder.create_composite_image(
            session_dir,
            coordinates,
            backup_root=backup_root,
            slug=timelapse_config.get('slug'),
            tile_cache=tile_cache,
        )
        


    def _encode_with_ffmpeg(
        self,
        frame_iter: Iterable[bytes],
        output_path: Path,
        fps: int,
        crop_bounds: Tuple[int, int, int, int]
    ) -> None:
        renderer = self._get_renderer()
        renderer.encode_with_ffmpeg(
            iter(frame_iter),
            output_path,
            fps,
            crop_bounds,
            quality=getattr(self, "quality", 20),
        )

    def _remux_segments_with_ffmpeg(
        self,
        concat_path: Path,
        output_path: Path,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg with libx264/libx265/libsvtav1."
            )

        temp_output = output_path.with_name(f".tmp_{uuid.uuid4().hex}_{output_path.name}")
        if temp_output.exists():
            temp_output.unlink()

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
        )

        temp_output.replace(output_path)

    @staticmethod
    def _probe_video_dimensions(video_path: Path) -> Optional[Tuple[int, int]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if width > 0 and height > 0:
            return width, height
        return None

    def _reframe_video_to_bounds(
        self,
        video_path: Path,
        content_width: int,
        content_height: int,
        crop_left: int,
        crop_top: int,
        target_width: int,
        target_height: int,
        left_offset: int,
        top_offset: int,
    ) -> None:
        if (
            target_width <= 0
            or target_height <= 0
            or content_width <= 0
            or content_height <= 0
        ):
            return

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg with libx264/libx265/libsvtav1."
            )

        current_width: Optional[int] = None
        current_height: Optional[int] = None
        current_dims = self._probe_video_dimensions(video_path)
        if current_dims is not None:
            current_width, current_height = current_dims
            if (
                current_width == target_width
                and current_height == target_height
                and crop_left == left_offset
                and crop_top == top_offset
            ):
                return
            if content_width > current_width or content_height > current_height:
                self.logger.warning(
                    "Skipping reframe for %s; content bounds exceed current frame",
                    video_path,
                )
                return

        color_bgr = getattr(self, "background_color", (0, 0, 0))
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        pad_color = f"0x{r:02X}{g:02X}{b:02X}"

        temp_output = video_path.with_name(f".tmp_{uuid.uuid4().hex}_{video_path.name}")
        if temp_output.exists():
            temp_output.unlink()

        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        left_offset = max(0, left_offset)
        top_offset = max(0, top_offset)

        if left_offset + content_width > target_width:
            target_width = left_offset + content_width
        if top_offset + content_height > target_height:
            target_height = top_offset + content_height

        if current_width is not None and crop_left + content_width > current_width:
            self.logger.warning(
                "Skipping reframe for %s; crop exceeds current width",
                video_path,
            )
            return
        if current_height is not None and crop_top + content_height > current_height:
            self.logger.warning(
                "Skipping reframe for %s; crop exceeds current height",
                video_path,
            )
            return

        crop_filter = f"crop={content_width}:{content_height}:{crop_left}:{crop_top}"
        pad_filter = (
            f"pad={target_width}:{target_height}:{left_offset}:{top_offset}:color={pad_color}"
        )
        filter_chain = f"{crop_filter},{pad_filter}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(getattr(self, "quality", 20)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        temp_output.replace(video_path)

    def _ensure_segment_dimensions(self, state: FullTimelapseState) -> None:
        for segment in state.segments:
            if segment.video_width is None or segment.video_height is None:
                dims = self._probe_video_dimensions(state.segment_path(segment))
                if dims is not None:
                    segment.video_width, segment.video_height = dims
            if segment.content_width is None and segment.video_width is not None:
                segment.content_width = segment.video_width
            if segment.content_height is None and segment.video_height is not None:
                segment.content_height = segment.video_height
            if segment.crop_x is None:
                segment.crop_x = 0
            if segment.crop_y is None:
                segment.crop_y = 0
            if segment.pad_left is None:
                segment.pad_left = 0
            if segment.pad_top is None:
                segment.pad_top = 0



    def create_differential_frame(
        self,
        prev_frame: Optional[np.ndarray],
        curr_frame: np.ndarray,
        return_stats: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """Delegate differential frame generation to the renderer."""
        return self._get_renderer().create_differential_frame(
            prev_frame,
            curr_frame,
            return_stats=return_stats,
        )

    def _prepare_frames_from_manifests(
        self,
        manifests: List[FrameManifest],
        coordinates: List[Tuple[int, int]],
        temp_dir: Path,
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> List[PreparedFrame]:
        return self._get_renderer().prepare_frames_from_manifests(
            manifests,
            coordinates,
            temp_dir,
            slug,
            name,
            mode_name,
            label,
        )

    def _build_frame_manifests(
        self,
        session_dirs: List[Path],
        timelapse_config: Dict[str, Any],
        coordinates: List[Tuple[int, int]],
        slug: str,
        name: str,
        mode_name: str,
        label: str,
    ) -> List[FrameManifest]:
        builder = self._get_manifest_builder()
        backup_root = getattr(self, "backup_dir", None)
        return builder.build_frame_manifests(
            session_dirs,
            coordinates,
            backup_root=backup_root,
            slug=slug,
            mode_name=mode_name,
            label=label,
            timelapse_name=name,
        )

    def _frame_byte_generator(
        self,
        prepared_frames: List[PreparedFrame],
        mode_name: str,
        slug: str,
        name: str,
        label: str,
        frame_datetimes: List[Optional[datetime]],
    ) -> Tuple[Iterator[bytes], TimelapseStatsCollector]:
        renderer = self._get_renderer()
        return renderer.frame_byte_generator(
            prepared_frames,
            mode_name,
            slug,
            name,
            label,
            frame_datetimes,
        )

    def _find_timelapse_entry(self, slug: str) -> Optional[Dict[str, Any]]:
        for timelapse in self.config.get("timelapses", []):
            if timelapse.get("slug") == slug:
                return timelapse
        return None

    def _sessions_between(
        self,
        slug: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> List[Path]:
        all_sessions = get_all_sessions(self.backup_dir, slug)
        selected: List[Path] = []
        for session_dir in all_sessions:
            session_dt = parse_session_datetime(session_dir)
            if session_dt is None:
                continue
            if start is not None and session_dt < start:
                continue
            if end is not None and session_dt > end:
                continue
            selected.append(session_dir)
        return selected

    def _collect_stats_for_sessions(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        session_dirs: List[Path],
        mode_name: str,
        label: str,
    ) -> Optional[TimelapseStatsCollector]:
        coordinates = self.get_tile_coordinates(timelapse_config)
        if not coordinates:
            self.logger.warning(
                "Unable to collect stats for '%s' (%s) %s %s; no coordinates defined",
                name,
                slug,
                mode_name,
                label,
            )
            return None

        manifests = self._build_frame_manifests(
            session_dirs,
            timelapse_config,
            coordinates,
            slug,
            name,
            mode_name,
            label,
        )
        if not manifests:
            self.logger.warning(
                "Unable to collect stats for '%s' (%s) %s %s; no frame manifests generated",
                name,
                slug,
                mode_name,
                label,
            )
            return None

        frame_datetimes = [
            parse_session_datetime(manifest.session_dir)
            for manifest in manifests
        ]
        renderer = self._get_renderer()
        return renderer.collect_stats_from_manifests(
            manifests,
            coordinates,
            mode_name,
            slug,
            name,
            label,
            frame_datetimes,
        )

    def _execute_stats_job(self, job: StatsGenerationJob) -> None:
        timelapse_config = job.timelapse_config
        name = timelapse_config.get("name") or job.slug

        session_dirs = self._sessions_between(job.slug, job.start, job.end)
        if not session_dirs:
            self.logger.info(
                "Stats generation job %s skipped; no sessions found between %s and %s",
                job.job_id,
                job.start,
                job.end,
            )
            return

        stats_collector = self._collect_stats_for_sessions(
            job.slug,
            name,
            timelapse_config,
            session_dirs,
            job.mode_name,
            job.label,
        )
        if stats_collector is None:
            self.logger.info(
                "Stats generation job %s produced no data for '%s' (%s)",
                job.job_id,
                name,
                job.slug,
            )
            return

        gap_threshold: Optional[timedelta] = None
        gap_multiplier = getattr(self, "coverage_gap_multiplier", None)
        if (
            gap_multiplier is not None
            and gap_multiplier > 0
            and getattr(self, "backup_interval", 0) > 0
        ):
            gap_threshold = timedelta(
                minutes=self.backup_interval * gap_multiplier
            )

        stats_summary = stats_collector.summarize(
            gap_threshold,
            seconds_per_pixel=getattr(self, "seconds_per_pixel", 30),
        )

        slug_dir = self.output_dir / job.slug
        slug_dir.mkdir(exist_ok=True)
        target_video = slug_dir / job.output_filename
        report_path = target_video.with_suffix(target_video.suffix + ".stats.txt")

        generated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        report_text = build_stats_report(
            slug=job.slug,
            name=name,
            mode=job.mode_name,
            label=job.label,
            output_path=target_video,
            generated_at=generated_at,
            stats=stats_summary,
        )
        report_path.write_text(report_text, encoding="utf-8")

        rendered_frames = stats_summary.get("rendered_frames", 0)
        start_ts = stats_summary.get("start_timestamp")
        end_ts = stats_summary.get("end_timestamp")
        self.logger.info(
            "Stats generation job %s completed for '%s' (%s) %s; %s frames (%s -> %s)",
            job.job_id,
            name,
            job.slug,
            job.mode_name,
            rendered_frames,
            start_ts or "unknown",
            end_ts or "unknown",
        )

    def _run_stats_job(self, job_id: str) -> None:
        with self._stats_jobs_lock:
            job = self._stats_jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            self._execute_stats_job(job)
        except Exception as exc:
            self.logger.exception(
                "Stats generation job %s failed: %s",
                job_id,
                exc,
            )
            with self._stats_jobs_lock:
                job.error = str(exc)
                job.status = "failed"
                job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            return

        with self._stats_jobs_lock:
            job.status = "completed"
            job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)

    def schedule_stats_generation(
        self,
        slug: str,
        mode_name: str,
        suffix: str,
        output_filename: str,
        *,
        timelapse_config: Optional[Dict[str, Any]] = None,
        start: Optional[datetime],
        end: Optional[datetime],
        label: str,
    ) -> Optional[str]:
        if not getattr(self, "reporting_enabled", False):
            self.logger.debug(
                "Skipping stats generation for '%s' (%s); reporting disabled",
                slug,
                mode_name,
            )
            return None

        resolved_config = timelapse_config or self._find_timelapse_entry(slug)
        if resolved_config is None:
            self.logger.warning(
                "Skipping stats generation for '%s' (%s); configuration not found",
                slug,
                mode_name,
            )
            return None

        job_key = (slug, output_filename)
        with self._stats_jobs_lock:
            existing_id = self._stats_job_index.get(job_key)
            if existing_id is not None:
                existing_job = self._stats_jobs.get(existing_id)
                if existing_job and existing_job.status in {"pending", "running"}:
                    self.logger.info(
                        "Stats generation already in progress for '%s' (%s); using job %s",
                        slug,
                        mode_name,
                        existing_id,
                    )
                    return existing_id

        job_id = uuid.uuid4().hex
        job = StatsGenerationJob(
            job_id=job_id,
            slug=slug,
            mode_name=mode_name,
            suffix=suffix,
            output_filename=output_filename,
            timelapse_config=resolved_config,
            start=start,
            end=end,
            label=label,
        )

        thread = threading.Thread(
            target=self._run_stats_job,
            args=(job_id,),
            name=f"stats-job-{slug}-{mode_name}-{job_id[:8]}",
            daemon=True,
        )
        job.thread = thread

        with self._stats_jobs_lock:
            self._stats_jobs[job_id] = job
            self._stats_job_index[job_key] = job_id

        thread.start()
        return job_id

    def list_stats_jobs(self) -> List[StatsGenerationJob]:
        with self._stats_jobs_lock:
            return list(self._stats_jobs.values())

    def render_timelapse_from_sessions(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        session_dirs: List[Path],
        mode_name: str,
        suffix: str,
        output_filename: str,
        label: str
    ) -> Optional[RenderedTimelapseResult]:
        """Create timelapse from a pre-collected list of session directories.

        Returns:
            RenderedTimelapseResult if frames were rendered, otherwise None.
        """
        if not session_dirs:
            self.logger.warning(
                "No session directories found for '%s' (%s) %s",
                name,
                slug,
                label,
            )
            return None

        self.logger.info(
            "Creating %s timelapse for '%s' (%s) %s with %s frames",
            mode_name,
            name,
            slug,
            label,
            len(session_dirs),
        )

        coordinates = self.get_tile_coordinates(timelapse_config)
        if not coordinates:
            self.logger.error(
                "No valid frames created for '%s' (%s) %s %s",
                name,
                slug,
                mode_name,
                label,
            )
            return None

        manifests = self._build_frame_manifests(
            session_dirs,
            timelapse_config,
            coordinates,
            slug,
            name,
            mode_name,
            label,
        )
        if not manifests:
            self.logger.error(
                "No valid frames created for '%s' (%s) %s %s",
                name,
                slug,
                mode_name,
                label,
            )
            return None

        output_dir = self.output_dir / slug
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        renderer = self._get_renderer()
        total_frames = 0
        rendered_session_dirs: List[Path] = []
        rendered_video_width: Optional[int] = None
        rendered_video_height: Optional[int] = None
        rendered_content_width: Optional[int] = None
        rendered_content_height: Optional[int] = None
        rendered_crop_x: Optional[int] = None
        rendered_crop_y: Optional[int] = None

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            prepared_frames = self._prepare_frames_from_manifests(
                manifests,
                coordinates,
                temp_dir,
                slug,
                name,
                mode_name,
                label,
            )

            if not prepared_frames:
                self.logger.error(
                    "No valid frames created for '%s' (%s) %s %s",
                    name,
                    slug,
                    mode_name,
                    label,
                )
                return None

            try:
                crop_bounds, original_shape = renderer.compute_crop_bounds(
                    prepared_frames,
                    slug=slug,
                    name=name,
                    mode_name=mode_name,
                    label=label,
                )
                x0, y0, x1, y1 = crop_bounds
                rendered_crop_x = x0
                rendered_crop_y = y0
                content_width = max(0, x1 - x0)
                content_height = max(0, y1 - y0)
                if content_width == 0 or content_height == 0:
                    content_width = original_shape[1]
                    content_height = original_shape[0]
                rendered_content_width = content_width
                rendered_content_height = content_height
                rendered_video_width = content_width
                rendered_video_height = content_height
            except RuntimeError as exc:
                self.logger.error(str(exc))
                return None

            frame_datetimes: List[Optional[datetime]] = [
                parse_session_datetime(frame.session_dir)
                for frame in prepared_frames
            ]
            rendered_session_dirs = [frame.session_dir for frame in prepared_frames]

            gap_threshold: Optional[timedelta] = None
            gap_multiplier = getattr(self, "coverage_gap_multiplier", None)
            if (
                gap_multiplier is not None
                and gap_multiplier > 0
                and getattr(self, "backup_interval", 0) > 0
            ):
                gap_threshold = timedelta(
                    minutes=self.backup_interval * gap_multiplier
                )

            total_frames = len(prepared_frames)
            self.logger.info(
                "Streaming %s frames to FFmpeg for '%s' (%s) %s %s",
                total_frames,
                name,
                slug,
                mode_name,
                label,
            )

            try:
                frame_iter, stats_collector = self._frame_byte_generator(
                    prepared_frames,
                    mode_name,
                    slug,
                    name,
                    label,
                    frame_datetimes,
                )
                self._encode_with_ffmpeg(frame_iter, output_path, self.fps, crop_bounds)
            except subprocess.CalledProcessError as exc:
                self.logger.error(
                    "FFmpeg encoding failed for '%s' (%s) %s %s: %s",
                    name,
                    slug,
                    mode_name,
                    label,
                    exc.stderr or exc,
                )
                raise
            else:
                if getattr(self, "reporting_enabled", False):
                    stats_summary = stats_collector.summarize(
                        gap_threshold,
                        seconds_per_pixel=getattr(self, "seconds_per_pixel", 30),
                    )
                    generated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    report_text = build_stats_report(
                        slug=slug,
                        name=name,
                        mode=mode_name,
                        label=label,
                        output_path=output_path,
                        generated_at=generated_at,
                        stats=stats_summary,
                    )
                    stats_path = output_path.with_suffix(output_path.suffix + ".stats.txt")
                    try:
                        stats_path.write_text(
                            report_text,
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        self.logger.error(
                            "Failed to write stats file for '%s' (%s) %s %s: %s",
                            name,
                            slug,
                            mode_name,
                            label,
                            exc,
                        )

        self.logger.info(
            "%s timelapse created for '%s' (%s): %s (%s frames)",
            mode_name.title(),
            name,
            slug,
            output_path,
            total_frames,
        )
        return RenderedTimelapseResult(
            output_path=output_path,
            frame_count=total_frames,
            session_dirs=tuple(rendered_session_dirs),
            video_width=rendered_video_width,
            video_height=rendered_video_height,
            content_width=rendered_content_width,
            content_height=rendered_content_height,
            crop_x=rendered_crop_x,
            crop_y=rendered_crop_y,
        )

    def render_incremental_full_timelapse(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        session_dirs: List[Path],
        mode_name: str,
        suffix: str,
        output_filename: str,
        label: str,
    ) -> None:
        """Render and append only new sessions to the full-history timelapse."""
        slug_dir = self.output_dir / slug
        slug_dir.mkdir(exist_ok=True)

        state = FullTimelapseState(
            slug_dir,
            output_filename,
            logger=self.logger,
        )
        state.load()

        pending_sessions = state.pending_sessions(session_dirs)
        if not pending_sessions:
            self.logger.info(
                "Full timelapse up to date for '%s' (%s) %s %s",
                name,
                slug,
                mode_name,
                label,
            )
            return

        state.ensure_segments_dir()
        temp_segment_rel = Path("segments") / state.output_basename / f"tmp_{uuid.uuid4().hex}.mp4"
        temp_segment_rel_str = temp_segment_rel.as_posix()

        try:
            render_result = self.render_timelapse_from_sessions(
                slug=slug,
                name=name,
                timelapse_config=timelapse_config,
                session_dirs=pending_sessions,
                mode_name=mode_name,
                suffix=suffix,
                output_filename=temp_segment_rel_str,
                label=f"{label} (segment)",
            )
        except subprocess.CalledProcessError:
            # render_timelapse_from_sessions already logged details
            return

        temp_segment_path = slug_dir / temp_segment_rel
        temp_stats_path = temp_segment_path.with_suffix(temp_segment_path.suffix + ".stats.txt")
        result_stats_path = (
            render_result.output_path.with_suffix(render_result.output_path.suffix + ".stats.txt")
            if render_result is not None
            else None
        )

        def cleanup_temp_outputs(remove_video: bool = True) -> None:
            """Remove temporary segment artifacts generated during rendering."""
            if remove_video:
                temp_segment_path.unlink(missing_ok=True)
            paths = {temp_stats_path}
            if result_stats_path is not None:
                paths.add(result_stats_path)
            for stats_path in paths:
                stats_path.unlink(missing_ok=True)

        if render_result is None or render_result.frame_count == 0:
            cleanup_temp_outputs()
            self.logger.info(
                "Skipping full timelapse update for '%s' (%s) %s %s; no frames rendered",
                name,
                slug,
                mode_name,
                label,
            )
            return

        encoded_sessions = list(render_result.session_dirs)
        if not encoded_sessions:
            cleanup_temp_outputs()
            self.logger.warning(
                "Skipping full timelapse update for '%s' (%s) %s %s; no encoded sessions returned",
                name,
                slug,
                mode_name,
                label,
            )
            return

        first_dt = parse_session_datetime(encoded_sessions[0])
        last_dt = parse_session_datetime(encoded_sessions[-1])
        if first_dt is None or last_dt is None:
            cleanup_temp_outputs()
            self.logger.warning(
                "Unable to derive session timestamps for full timelapse '%s' (%s) %s %s; skipping update",
                name,
                slug,
                mode_name,
                label,
            )
            return

        segment_path = state.make_segment_filename(first_dt, last_dt)
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure unique segment names; replace duplicates to keep manifest clean.
        if segment_path.exists():
            segment_path.unlink()

        cleanup_temp_outputs(remove_video=False)
        temp_segment_path.replace(segment_path)
        relative_segment_path = segment_path.relative_to(slug_dir)

        # Harmonize segment framing so the concat muxer can copy streams.
        self._ensure_segment_dimensions(state)

        probe_dims = self._probe_video_dimensions(segment_path)
        new_video_width = render_result.video_width or (probe_dims[0] if probe_dims else None)
        new_video_height = render_result.video_height or (probe_dims[1] if probe_dims else None)
        new_content_width = (
            render_result.content_width
            or render_result.video_width
            or (probe_dims[0] if probe_dims else None)
        )
        new_content_height = (
            render_result.content_height
            or render_result.video_height
            or (probe_dims[1] if probe_dims else None)
        )
        new_crop_x = render_result.crop_x if render_result.crop_x is not None else 0
        new_crop_y = render_result.crop_y if render_result.crop_y is not None else 0

        if new_content_width is None or new_content_height is None:
            dims = self._probe_video_dimensions(segment_path)
            if dims is None:
                self.logger.error(
                    "Unable to determine dimensions for new segment %s",
                    segment_path,
                )
                return
            new_content_width = dims[0]
            new_content_height = dims[1]
            if new_video_width is None:
                new_video_width = dims[0]
            if new_video_height is None:
                new_video_height = dims[1]

        existing_bounds = state.content_bounds()
        min_x = new_crop_x
        min_y = new_crop_y
        max_x = new_crop_x + new_content_width
        max_y = new_crop_y + new_content_height
        if existing_bounds is not None:
            min_x = min(min_x, existing_bounds[0])
            min_y = min(min_y, existing_bounds[1])
            max_x = max(max_x, existing_bounds[2])
            max_y = max(max_y, existing_bounds[3])

        target_width = max(0, max_x - min_x)
        target_height = max(0, max_y - min_y)

        if target_width == 0 or target_height == 0:
            dims = self._probe_video_dimensions(segment_path)
            if dims is not None:
                target_width = dims[0]
                target_height = dims[1]
                min_x = min(min_x, 0)
                min_y = min(min_y, 0)

        if target_width % 2 != 0:
            target_width += 1
        if target_height % 2 != 0:
            target_height += 1

        # Reframe existing segments.
        for segment in state.segments:
            content_width = segment.content_width or segment.video_width
            content_height = segment.content_height or segment.video_height
            if content_width is None or content_height is None:
                continue
            left_offset = max(0, segment.crop_x - min_x)
            top_offset = max(0, segment.crop_y - min_y)
            needs_reframe = (
                segment.video_width != target_width
                or segment.video_height != target_height
                or segment.pad_left != left_offset
                or segment.pad_top != top_offset
            )
            if not needs_reframe:
                continue

            seg_path = state.segment_path(segment)
            self.logger.info(
                "Reframing legacy segment %s to %sx%s (offset %s,%s)",
                seg_path,
                target_width,
                target_height,
                left_offset,
                top_offset,
            )
            self._reframe_video_to_bounds(
                seg_path,
                content_width,
                content_height,
                segment.pad_left or 0,
                segment.pad_top or 0,
                target_width,
                target_height,
                left_offset,
                top_offset,
            )
            segment.video_width = target_width
            segment.video_height = target_height
            segment.pad_left = left_offset
            segment.pad_top = top_offset

        # Reframe new segment.
        new_left_offset = max(0, new_crop_x - min_x)
        new_top_offset = max(0, new_crop_y - min_y)
        self._reframe_video_to_bounds(
            segment_path,
            new_content_width,
            new_content_height,
            0,
            0,
            target_width,
            target_height,
            new_left_offset,
            new_top_offset,
        )
        new_video_width = target_width
        new_video_height = target_height

        new_segment = FullTimelapseSegment(
            path=relative_segment_path.as_posix(),
            first_session=first_dt.replace(microsecond=0).isoformat(),
            last_session=last_dt.replace(microsecond=0).isoformat(),
            frame_count=render_result.frame_count,
            video_width=new_video_width,
            video_height=new_video_height,
            content_width=new_content_width,
            content_height=new_content_height,
            crop_x=new_crop_x,
            crop_y=new_crop_y,
            pad_left=new_left_offset,
            pad_top=new_top_offset,
        )

        updated_segments = [*state.segments, new_segment]
        concat_temp_path = state.write_concat_file(
            updated_segments,
            temporary=True,
        )

        full_output_path = slug_dir / output_filename
        try:
            self._remux_segments_with_ffmpeg(concat_temp_path, full_output_path)
        except subprocess.CalledProcessError as exc:
            segment_path.unlink(missing_ok=True)
            self.logger.error(
                "Failed to append to full timelapse for '%s' (%s) %s %s: %s",
                name,
                slug,
                mode_name,
                label,
                exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else exc,
            )
            return
        finally:
            concat_temp_path.unlink(missing_ok=True)

        state.add_segment(new_segment)
        state.write_concat_file(temporary=False)
        state.save()

        self.logger.info(
            "Full timelapse updated for '%s' (%s) %s %s with %s new frames",
            name,
            slug,
            mode_name,
            label,
            render_result.frame_count,
        )

        if getattr(self, "reporting_enabled", False) and state.segments:
            stats_start = state.segments[0].first_datetime()
            stats_end = state.segments[-1].last_datetime()
            self.schedule_stats_generation(
                slug=slug,
                mode_name=mode_name,
                suffix=suffix,
                output_filename=output_filename,
                timelapse_config=timelapse_config,
                start=stats_start,
                end=stats_end,
                label=f"{label} (full)",
            )

    def create_daily_timelapse(self, date: datetime = None):
        """Create timelapse videos from previous day's images for all timelapses."""
        scheduler_module.create_daily_timelapse(self, date)
            
    def create_full_timelapses(self):
        """Create full-history timelapses for modes configured to support them."""
        scheduler_module.create_full_timelapses(self)
            
    def create_timelapse_for_slug(
        self,
        slug: str,
        name: str,
        timelapse_config: Dict[str, Any],
        date_str: str,
        mode_name: str = 'normal',
        suffix: str = '',
    ):
        """Create timelapse video for a specific timelapse slug and mode."""
        scheduler_module.create_timelapse_for_slug(
            self,
            slug,
            name,
            timelapse_config,
            date_str,
            mode_name=mode_name,
            suffix=suffix,
        )
            
    def run(self):
        """Run the backup system with scheduled tasks."""
        scheduler_module.run(self)
