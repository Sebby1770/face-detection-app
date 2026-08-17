# Face Detection & Privacy Redaction App

Detect and redact faces in webcam feeds, still images, and videos with [OpenCV](https://opencv.org/) and Python. The app is privacy-first: saved output uses a padded solid mask by default, and diagnostic overlays are opt-in.

Uses [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) — a fast modern CNN face detector — with an automatic fallback to OpenCV's bundled Haar cascade if the YuNet model can't be downloaded. The YuNet ONNX file is ~340 KB and is fetched on first run.

## Features

- **YuNet DNN detector** (auto-downloaded) with Haar cascade fallback
- **Image, video, and webcam sources** through one reusable processing pipeline
- **Headless file processing** for scripts and batch workflows
- **Solid, blur, and pixelate redaction** with configurable bounding-box padding
- **Conservative defaults**: solid black masks, 25% padding, two-frame dropout hold, and no saved overlays
- **IoU tracker** with EMA smoothing for stable face IDs and redaction regions
- **5-point landmarks** and confidence-colored boxes in the preview or optional output overlays
- **Redacted MP4 recording and snapshots** from the interactive preview
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

The YuNet ONNX model downloads automatically into `./models/` on first run. To pre-fetch it (or to refresh it later):

```bash
python download_model.py
```

If the download fails, the app tries OpenCV's bundled Haar cascade. Some OpenCV builds omit cascade data; in that case install a build that includes it or pass an existing XML file with `--haar-path`. You can force Haar with `--detector haar`.

## Usage

```bash
python face_detection.py
```

Common options:

```bash
# Redact an image without opening a window
python face_detection.py \
  --input portrait.jpg --output portrait-redacted.png --headless

# Redact a video with a deliberately chosen visual effect
python face_detection.py \
  --input interview.mp4 --output interview-redacted.mp4 \
  --headless --redaction pixelate --padding 0.35

# Include diagnostic boxes/IDs in saved output (off by default)
python face_detection.py \
  --input input.mp4 --output reviewed.mp4 --headless --overlays

# Different camera or resolution
python face_detection.py --camera 1 --width 640 --height 480

# Selfie mirror with blur instead of the default solid mask
python face_detection.py --mirror --blur

# Work offline when the installed OpenCV package includes its cascade data
python face_detection.py --detector haar --no-download

# Or point to a locally managed cascade explicitly
python face_detection.py --detector haar --no-download \
  --haar-path models/haarcascade_frontalface_default.xml

# Explicitly process without redaction (unsafe for privacy-sensitive output)
python face_detection.py --input input.mp4 --output raw-copy.mp4 \
  --headless --no-redaction
```

Existing output is never replaced unless `--overwrite` is passed. Run `python face_detection.py --help` for the full option list.

## Controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `s` | Save the current frame to `snapshots/` |
| `r` | Toggle MP4 recording into `recordings/` |
| `b` | Toggle face blur (privacy mode) |
| `l` | Toggle landmark points |
| `i` | Toggle face ID labels |
| `m` | Toggle mirror |
| `h` | Toggle on-screen help |

The `b`, `l`, `i`, and `m` controls reprocess still images from the untouched source so preview changes never compound. Recording (`r`) applies only to webcam and video previews. When `--no-tracker` is used, summaries report `tracking disabled` rather than presenting frame-local detection IDs as unique people.

## How it works

1. The media pipeline reads a still image, a video, or frames from the selected camera.
2. The active detector returns axis-aligned bounding boxes (and, for YuNet, a confidence score plus five facial landmarks).
3. An IoU-based tracker matches detections to existing tracks and smooths each track's bbox with an exponential moving average. New detections get a new monotonically-increasing ID; stale tracks die after a configurable number of missed frames.
4. Every current face region is expanded by the configured padding, clamped to the frame, and redacted on a copy of the source frame. The last known region remains redacted for two missed frames by default to bridge brief detector dropouts.
5. The clean redacted frame is written to disk. Boxes, landmarks, IDs, and confidence labels are included only with `--overlays`; the interactive HUD is never baked into normal output.

## Privacy limitations

This tool reduces accidental face exposure; it does **not** guarantee anonymisation. A detector can miss profiles, small or partially hidden faces, faces under unusual lighting, or frames with motion blur. Anything the detector misses remains unchanged.

- Review sensitive output before sharing it, ideally frame by frame around cuts and rapid movement.
- Keep the default solid mode for stronger visual removal. Blur and pixelation preserve more source information and may be inadequate for high-risk material.
- Increase `--padding` when hair, ears, masks, or surrounding context could identify someone.
- Raising `--score-threshold` reduces false positives but can also increase dangerous false negatives.
- Diagnostic overlays do not improve detection and are excluded from saved output by default.

The app processes media locally. In automatic detector mode it may download the YuNet model on first use. For an offline run, pass `--detector haar --no-download`; if your OpenCV package omits cascade data, also supply `--haar-path`.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

The suite creates synthetic images and videos and validates tracking, padded redaction, clean output, CLI safety checks, and source-to-output processing without a webcam or network connection.

## Project layout

```
face_detection.py    validated CLI and interactive preview controls
detectors.py         YuNetDetector, HaarDetector, model auto-download
tracker.py           greedy IoU tracker with EMA smoothing
pipeline.py          reusable per-frame tracking/redaction processor
media.py             image, video, and webcam source/output pipeline
download_model.py    standalone helper to pre-fetch the YuNet ONNX
tests/                deterministic unit and media integration tests
```

## License

MIT
