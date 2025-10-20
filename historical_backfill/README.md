# Historical Backfill

The historical backfill toolkit makes it possible to seed the `backups/`
directory with frames that predate your live timelapse capture window. It
expects you to download and extract releases from
[`murolem/wplace-archives`](https://github.com/murolem/wplace-archives) and then
projects those tiles into the same layout that the live collector uses:

```
backups/<slug>/<YYYY-MM-DD>/<HH-MM-SS>/<x>_<y>.png
```

Once populated, the standard timelapse generation scripts can process the
historical frames alongside newly captured sessions without any additional
changes.

## Directory Layout

Running the `init` command creates a basic workspace:

```
historical_backfill/
├── data/
│   ├── archives/   # store downloaded tar parts
│   └── extracted/  # extracted releases go here
└── ...
```

You can place extracted release directories directly inside
`historical_backfill/data/extracted/`. Each extracted folder should follow the
same naming scheme as its release tag, e.g.
`world-2025-10-20T03-30-04.354Z/`.

## Quick Start

```bash
# 1. Prepare the workspace structure
python3 -m historical_backfill init

# 2. Discover and download release archives
python3 -m historical_backfill download --list --count 5
python3 -m historical_backfill download --tag world-2025-10-20T03-30-04.354Z

# 3. (Optional) Print the manual checklist
python3 -m historical_backfill plan --slug demo

# 4. Generate historical sessions
python3 -m historical_backfill generate \
  --slug demo \
  --start 2025-10-19T18:30:00Z \
  --end 2025-10-20T03:30:00Z \
  --interval-minutes 5 \
  --config config.json \
  --config-slug demo \
  --cleanup-archives
```

The `generate` command inspects the extracted releases, picks the latest archive
available at or before each requested timestamp, and copies the relevant tiles
into the `backups/<slug>/` tree. When the same release snapshot is reused for
consecutive frames, placeholder markers are created so the main pipeline
understands that the tile content is unchanged.

## Command Reference

### `python3 -m historical_backfill init`
Creates the `historical_backfill/data/` workspace structure.

### `python3 -m historical_backfill plan [--slug <slug>]`
Prints a short checklist for fetching and extracting releases from GitHub.

### `python3 -m historical_backfill download ...`
Interacts with the GitHub API to list recent releases and fetch their assets.

- `--list`: Display the latest tags, publication times, asset counts, and sizes.
- `--tag`: Download every asset for a specific release into
  `historical_backfill/data/archives/<tag>/`.
- `--skip-existing`: Leave files untouched if they already exist locally.
- `--dry-run`: Preview the plan without writing to disk.

Set a `GITHUB_TOKEN` environment variable if you encounter GitHub rate limits.

### `python3 -m historical_backfill generate ...`
Populates backup sessions using extracted releases. Key options:

- `--slug`: Destination slug under the `backups/` directory.
- `--start`, `--end`: Inclusive ISO timestamps for the frames you want to
  synthesize. Use `Z` or an explicit UTC offset.
- `--interval-minutes`: Artificial cadence between frames (default: 5).
- `--config` and `--config-slug`: Reuse bounding boxes from the main
  `config.json`. Alternatively specify `--xmin/--xmax/--ymin/--ymax` manually.
- `--tiles-subdir`: If your extraction embeds tiles deeper in the folder
  hierarchy (e.g. `<release>/tiles/`), point the generator at that subdirectory.
- `--archives-dir`: Override the location of downloaded archive parts if you
  store them somewhere other than `historical_backfill/data/archives/`.
- `--dry-run`: Preview activity without writing files.
- `--no-deduplicate`: Force every frame to copy PNG tiles even if the
  underlying release snapshot hasn't changed.
- `--cleanup-archives`: Remove the downloaded archive parts for any release
  that was used once generation completes. Pair with `--archives-dir` if your
  downloads live outside the default location.

The generator writes directly into `<root>/backups/` by default. Use
`--dest-dir` to target a different folder if you want to stage frames before
merging them back into the main project.

## Notes & Expectations

- Releases are indexed solely by their timestamped directory names
  (`world-YYYY-MM-DDTHH-MM-SS.mmmZ`). Keep those folder names intact after
  extraction so the generator can infer the capture time.
- The tool only copies the tiles needed for your bounding box; it does not
  stitch them into a single composite image. The main timelapse pipeline already
  knows how to do that.
- Missing tiles fall back to placeholders that reference earlier sessions.
  Ensure you run the generator chronologically so there is always a real frame
  to reference when placeholders are emitted.
- Use `python3 -m historical_backfill download` to fetch releases directly from
  GitHub when you prefer an automated workflow; the plan checklist remains as a
  manual fallback.
- If disk space is tight, pass `--cleanup-archives` to the generate command to
  delete the multi-gigabyte tar parts as soon as they are consumed.
- The helper script `historical_backfill/fetch_nemo_history.sh` automates the
  full workflow for the Point Nemo region, extracting only the necessary tiles
  and backfilling frames in bulk. Review the environment variables at the top of
  the script if you need to customise the time range or target slug. By default
  it writes backups to `/data/backups` and streams archive downloads through
  `/data/cache` (creating subdirectories as needed), deleting each multi-gigabyte
  archive as soon as the tiles are extracted so only one release resides on disk
  at a time.

  Useful overrides:
  - `HF_START_ISO` / `HF_END_ISO` – adjust the historical window.
  - `HF_CACHE_ROOT` – change the cache base directory (for both `archives/` and
    `extracted/`).
  - `HF_DEST_BACKUPS` – choose where the generated frame directories are stored.
  - `HF_KEEP_ARCHIVES=1` – retain downloaded tar parts instead of deleting them.
