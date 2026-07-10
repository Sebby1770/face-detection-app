# Face Detection App

Real-time face detection for webcam, video files, and still images — built with [OpenCV](https://opencv.org/) and pure Python.

Uses [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (a fast modern CNN face detector) with automatic fallback to OpenCV’s bundled Haar cascade. The YuNet ONNX file is ~340 KB and is fetched on first run.

## Features

- **YuNet DNN detector** (auto-downloaded) with Haar cascade fallback
- **Multiple inputs** — webcam index, video file, or still image (`--source` / positional)
- **IoU tracker** with EMA smoothing — stable face IDs, no jittery boxes
- **5-point landmarks** (eyes, nose, mouth corners) rendered on each face
- **Head-pose yaw estimate** from landmarks — L / C / R label and small arrow
- **Confidence-colored boxes** — green (high) / yellow (medium) / orange (low)
- **Privacy modes** — Gaussian blur (`b`) and size-aware pixelate (`p`)
- **Face gallery** — optional crop of each stable track (`--gallery` / `g`)
- **Session analytics** — frames, peak faces, unique IDs, avg confidence; JSON export
- **MP4 recording** and **snapshot** capture
- HUD with face count, FPS, detector name, peak faces, unique IDs
- In-app help overlay

## Requirements

- Python 3.9+
- OpenCV 4.8+ (`opencv-python` or `opencv-python-headless`)
- Webcam optional (only for live camera mode)
- macOS users: grant Camera access under **System Settings → Privacy & Security → Camera** when using a webcam

## Installation

```bash
git clone https://github.com/Sebby1770/face-detection-app.git
cd face-detection-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or install as a package (editable):

```bash
pip install -e ".[dev]"
```

The YuNet ONNX model downloads automatically into `./models/` on first run. To pre-fetch it:

```bash
python download_model.py
```

If the download fails (for example without GitHub access), the app falls back to Haar. Force Haar with `--detector haar`.

## Usage

### Webcam (default)

```bash
python face_detection.py
# same as
python face_detection.py --camera 0
```

### Image or video file

```bash
# Still image — process once, show result, press q/Esc to exit
python face_detection.py photo.jpg
python face_detection.py --source path/to/photo.png --save-image out.png

# Video file — frame-by-frame like the webcam loop
python face_detection.py clip.mp4
python face_detection.py --source recordings/demo.mov
```

### Common options

```bash
# Detector selection
python face_detection.py --detector yunet
python face_detection.py --detector haar

# Camera / resolution
python face_detection.py --camera 1 --width 640 --height 480

# Privacy modes at start
python face_detection.py --mirror --blur
python face_detection.py --pixelate

# Gallery: save one crop per stable track into gallery/
python face_detection.py --gallery --gallery-dir gallery --gallery-min-age 15

# Export session analytics JSON on quit
python face_detection.py --export-stats session_stats.json

# Tighter YuNet threshold / disable tracking
python face_detection.py --score-threshold 0.8
python face_detection.py --no-tracker
```

Run `python face_detection.py --help` for the full flag list.

### CLI reference

| Flag | Default | Description |
| --- | --- | --- |
| `source` (positional) | — | Image path, video path, or camera index |
| `--source` | — | Same as positional source |
| `--camera` | `0` | Webcam index when no file source is given |
| `--detector` | `auto` | `auto`, `yunet`, or `haar` |
| `--width` / `--height` | `1280` / `720` | Capture request size (camera only) |
| `--score-threshold` | `0.6` | YuNet confidence threshold |
| `--nms-threshold` | `0.3` | YuNet NMS threshold |
| `--no-tracker` | off | Disable IoU tracking / stable IDs |
| `--mirror` | off | Start mirrored (selfie style) |
| `--blur` | off | Start in Gaussian face-blur mode |
| `--pixelate` | off | Start in pixelate privacy mode |
| `--snapshot-dir` | `snapshots` | Directory for `s` snapshots |
| `--recording-dir` | `recordings` | Directory for `r` recordings |
| `--model-dir` | `./models` | YuNet ONNX search/download path |
| `--no-download` | off | Do not auto-download YuNet |
| `--export-stats` | — | Write analytics JSON on quit |
| `--gallery` | off | Enable face gallery capture |
| `--gallery-dir` | `gallery` | Gallery output directory |
| `--gallery-min-age` | `15` | Frames before a track is cropped once |
| `--save-image` | — | Write annotated still when source is an image |

## Controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `s` | Save the current frame to `snapshots/` |
| `r` | Toggle MP4 recording into `recordings/` (stream modes) |
| `b` | Toggle face blur (privacy) |
| `p` | Toggle face pixelate (privacy; exclusive with blur) |
| `g` | Toggle gallery capture on the fly |
| `l` | Toggle landmark points |
| `i` | Toggle face ID labels |
| `m` | Toggle mirror (stream modes) |
| `h` | Toggle on-screen help |

## Session analytics

Every run tracks:

- Total frames processed
- Faces-per-frame history
- Unique track IDs seen
- Peak concurrent faces
- Average detection confidence
- Session duration

The HUD shows **peak faces** and **unique IDs**. On quit, pass `--export-stats path.json` to write a full JSON report:

```json
{
  "duration_seconds": 12.4,
  "total_frames": 310,
  "peak_concurrent_faces": 3,
  "unique_id_count": 5,
  "average_faces_per_frame": 1.2,
  "average_confidence": 0.91,
  "faces_per_frame": [1, 1, 2, ...]
}
```

## Head pose

When YuNet landmarks are available, a coarse **yaw** estimate is derived from the nose tip relative to the eye midpoint (normalized by inter-ocular distance). Each track shows:

- **L** / **C** / **R** label above the box
- A small left/right arrow (or center bar)

This is a rough geometric cue, not a full 6-DoF pose solver.

## Face gallery

With `--gallery` (or key `g`):

1. A track must live for at least `--gallery-min-age` frames (default 15).
2. The first time it qualifies, a padded crop is written once as `gallery/face_XXXX.jpg`.
3. Crops use the frame *before* blur/pixelate so gallery faces stay sharp.

## How it works

1. Resolve the input: camera index, video file, or still image.
2. The active detector returns axis-aligned boxes (and, for YuNet, scores + five landmarks).
3. An IoU tracker matches detections to tracks and smooths boxes with an EMA.
4. Optional privacy filters (blur / pixelate) or overlays (boxes, landmarks, pose, IDs).
5. Analytics update each frame; gallery may store stable crops; recording writes annotated frames.

## Project layout

```
face_detection.py    main loop, CLI, rendering, key handling
detectors.py         YuNetDetector, HaarDetector, model auto-download
tracker.py           greedy IoU tracker with EMA smoothing
analytics.py         session stats + JSON export
pose.py              landmark yaw estimate + draw helpers
download_model.py    standalone helper to pre-fetch YuNet ONNX
tests/               unit tests (no camera required)
pyproject.toml       packaging metadata
.github/workflows/   CI (pytest)
```

## Development

```bash
pip install -r requirements.txt
# or: pip install -e ".[dev]"
pytest -q
```

CI runs the same tests on Python 3.9 / 3.11 / 3.12 with `opencv-python-headless`.

## License

MIT
