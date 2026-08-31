# Face Privacy Redactor 2.4.0

**Live site:** [https://sebby1770.github.io/face-detection-app/](https://sebby1770.github.io/face-detection-app/)

[github.com/Sebby1770/face-detection-app](https://github.com/Sebby1770/face-detection-app)

Detect and **redact** faces in webcam feeds, still images, videos, and folders with [OpenCV](https://opencv.org/) and Python. This is a privacy tool first: saved output uses a padded solid black box by default, and diagnostic overlays never hit disk unless you ask.

The GitHub Pages demo redacts a photo or camera frame in the browser (Chrome/Edge built-in face detector, or drag boxes yourself). Nothing is uploaded. Use the Python CLI for video folders, YuNet, and `--keep-ids`.

Uses [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) — a fast CNN face detector — with an automatic fallback to OpenCV's bundled Haar cascade if the YuNet model can't be downloaded. The YuNet ONNX file is ~340 KB and is fetched on first run.

## Features

- **YuNet DNN detector** (auto-downloaded) with Haar cascade fallback
- **Image, video, webcam, and directory** sources through one processing pipeline
- **Headless file and batch processing** for scripts
- **Solid, blur, and pixelate** redaction with box or landmark-aware ellipse masks and optional feathering
- **Conservative defaults**: solid black masks, 25% padding, two-frame dropout hold, no saved overlays
- **IoU tracker** with EMA smoothing for stable face IDs and redaction regions
- **Selective IDs** so a host can stay visible (`--keep-ids`) while others are redacted; the review HUD lists live IDs and hold-miss progress
- **Review HUD** (boxes, landmarks, live `IDs: 1,2,5`, hold `#N m/H` miss progress, optional pose, session stats, empty-frame `NO FACE this frame`) on the preview only
- **Coverage JSON** for batch runs (including skipped existing outputs, unique ID observations, files with no faces, miss-frame rate, and a high empty-frame warning) and optional session analytics export
- **`--min-size`** to ignore tiny detections before tracking
- Deterministic synthetic tests and GitHub Actions CI; tests need no camera, model, or network

## Requirements

- Python 3.9+
- A working webcam only when using live capture
- macOS webcam users: grant the terminal Camera access in **System Settings → Privacy & Security → Camera**

## Installation

```bash
git clone https://github.com/Sebby1770/face-detection-app.git
cd face-detection-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install the package (console script `face-detection`):

```bash
pip install -e .
```

The YuNet ONNX model downloads automatically into `./models/` on first run. To pre-fetch it (or to refresh it later):

```bash
python download_model.py
```

If the download fails, the app tries OpenCV's bundled Haar cascade. Some OpenCV builds omit cascade data; in that case install a build that includes it or pass an existing XML file with `--haar-path`. You can force Haar with `--detector haar`.

## Usage

```bash
python face_detection.py
```

A positional path is accepted as an alias of `--input`.

```bash
# Redact an image without opening a window
python face_detection.py \
  --input portrait.jpg --output portrait-redacted.png --headless

# Same thing with a positional source
python face_detection.py portrait.jpg --output portrait-redacted.png --headless

# Redact a video with a deliberately chosen visual effect
python face_detection.py \
  --input interview.mp4 --output interview-redacted.mp4 \
  --headless --redaction pixelate --padding 0.35

# Ellipse mask with a soft edge (still opt-in; box remains the default).
# With YuNet landmarks the ellipse also covers forehead and chin.
python face_detection.py \
  --input talk.mp4 --output talk-redacted.mp4 --headless \
  --mask ellipse --feather 8

# Keep the host visible after checking IDs in review mode
python face_detection.py --input interview.mp4 --review
python face_detection.py \
  --input interview.mp4 --output interview-redacted.mp4 \
  --headless --keep-ids 2

# Ignore detections smaller than 30px before tracking
python face_detection.py \
  --input crowd.jpg --output crowd-redacted.png --headless \
  --min-size 30

# Batch a folder (headless is implied). Mirror the tree into --output.
python face_detection.py \
  --input ./inbox --output ./redacted \
  --coverage coverage.json

# Review a webcam with pose indicators and export session stats
python face_detection.py --review --export-stats session.json

# Include diagnostic boxes/IDs in saved output (off by default)
python face_detection.py \
  --input input.mp4 --output reviewed.mp4 --headless --overlays

# Different camera or resolution
python face_detection.py --camera 1 --width 640 --height 480

# Selfie mirror with blur instead of the default solid mask
python face_detection.py --mirror --blur
```

Existing **files** are never replaced unless `--overwrite` is passed. In batch mode the output directory may already exist; individual files are skipped unless `--overwrite` is set. Skipped relative paths are recorded in `--coverage` JSON and printed with a reminder to pass `--overwrite`. Run `python face_detection.py --help` for the full option list.

`--identify` is reserved for identifying exports such as face crops. **Passing `--identify` is an error in this release**; it does not write crops. Review IDs with `--review`, then use `--keep-ids` / `--redact-ids`.

## Controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `s` | Save the current redacted frame to `snapshots/` (no HUD) |
| `r` | Toggle MP4 recording into `recordings/` (no HUD unless `--overlays`) |
| `b` | Toggle face blur (privacy mode) |
| `p` | Toggle face pixelate (privacy mode) |
| `e` | Toggle box / ellipse mask |
| `f` | Toggle feather 0 / 8 |
| `l` | Toggle landmark points |
| `i` | Toggle face ID labels |
| `m` | Toggle mirror |
| `h` | Toggle on-screen help |

The `b`, `p`, `e`, `f`, `l`, `i`, and `m` controls reprocess still images from the untouched source so preview changes never compound. That first still frame is counted once in session analytics; interactive reprocess does not double-count it. Recording (`r`) applies only to webcam and video previews. When `--no-tracker` is used, summaries report `tracking disabled` rather than presenting frame-local detection IDs as unique people.

`--review` and the live HUD (including current `IDs: 1,2,5`, `hold #N m/H` for tracks with misses, pose, and `NO FACE this frame` on empty frames) apply to the window only. Saved recordings, snapshots, and `--output` files stay redacted-without-HUD unless `--overlays` is set. Batch and CLI runs always print a coverage summary (face observations, id observations, files with no faces, miss rate); if the empty-frame rate is 25% or higher they also warn on stderr that faces may have been missed.

## How it works

1. The media pipeline reads a still image, a video, a camera, or every image/video under a directory.
2. The active detector returns axis-aligned bounding boxes (and, for YuNet, a confidence score plus five facial landmarks). Detections whose width or height is below `--min-size` are dropped before tracking.
3. An IoU-based tracker matches detections to existing tracks and smooths each track's bbox with an exponential moving average. New detections get a new monotonically-increasing ID; stale tracks die after a configurable number of missed frames. The preview HUD lists those IDs and, for a held track, `hold #N m/H` misses versus `--hold-frames`.
4. Every current face region is expanded by the configured padding, clamped to the frame, and redacted on a copy of the source frame. Ellipse mode without landmarks draws a filled ellipse 1.20× the padded height and 1.08× the padded width. With YuNet's 5 landmarks, the ellipse instead uses the padded bbox unioned with forehead (~0.55× bbox height above the eyes), chin (~0.35× below the mouth), and cheek padding (landmark x ± 0.25× bbox width). `--feather` blurs that mask before blending. The last known region remains redacted for two missed frames by default to bridge brief detector dropouts. Box mode ignores landmarks so existing solid-box output stays pixel-identical.
5. `--redact-ids` limits redaction to listed tracks; `--keep-ids` leaves listed tracks visible and wins if an ID is in both lists.
6. The clean redacted frame is written to disk. Boxes, landmarks, IDs, pose, and the analytics HUD stay on the preview unless `--overlays` is passed.

## Privacy limitations

This tool reduces accidental face exposure; it does **not** guarantee anonymisation. A detector can miss profiles, small or partially hidden faces, faces under unusual lighting, or frames with motion blur. Anything the detector misses remains unchanged.

- Review sensitive output before sharing it, ideally frame by frame around cuts and rapid movement.
- Keep the default solid box for stronger visual removal. Ellipse covers more of the head (and, with landmarks, forehead and jaw) but still depends on a correct detection. Blur and pixelation preserve more source information and may be inadequate for high-risk material.
- Increase `--padding` (and consider `--mask ellipse --feather`) when hair, ears, masks, or surrounding context could identify someone.
- Raising `--score-threshold` reduces false positives but can also increase dangerous false negatives.
- `--keep-ids` is an identifying choice: those faces stay in the output. Confirm IDs in `--review` first; tracker IDs are not biometric identities.
- Diagnostic overlays do not improve detection and are excluded from saved output by default.
- There is no face-crop gallery. Exporting crops would be an identifying feature and is intentionally omitted.

The app processes media locally. In automatic detector mode it may download the YuNet model on first use. For an offline run, pass `--detector haar --no-download`; if your OpenCV package omits cascade data, also supply `--haar-path`.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

The suite creates synthetic images and videos and validates tracking, padded redaction, ellipse/feather masks, landmark-aware ellipses, ID filters, `--min-size`, batch coverage (including skipped files, id observations, files with no faces, and miss-rate warnings), still-image analytics, preview HUD IDs/hold lines, `e`/`p`/`f` keys, `--identify` rejection, clean output, CLI safety checks, and source-to-output processing without a webcam or network connection.

## Project layout

```
face_detection.py    validated CLI and interactive preview controls
pipeline.py          reusable per-frame tracking/redaction processor
batch.py             directory tree processing and coverage JSON
media.py             image, video, and webcam source/output pipeline
detectors.py         YuNetDetector, HaarDetector, model auto-download
tracker.py           greedy IoU tracker with EMA smoothing
analytics.py         optional session statistics and JSON export
pose.py              optional landmark-based yaw overlay
download_model.py    standalone helper to pre-fetch the YuNet ONNX
tests/               deterministic unit and media integration tests
```

## License

MIT
