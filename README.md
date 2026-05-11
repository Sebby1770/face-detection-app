# Face Detection App

Real-time face detection from a webcam feed, built with [OpenCV](https://opencv.org/) and Python.

Uses [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) — a fast modern CNN face detector — with an automatic fallback to OpenCV's bundled Haar cascade if the YuNet model can't be downloaded. The YuNet ONNX file is ~340 KB and is fetched on first run.

## Features

- **YuNet DNN detector** (auto-downloaded) with Haar cascade fallback
- **IoU tracker** with EMA smoothing — stable face IDs, no jittery boxes
- **5-point landmarks** (eyes, nose, mouth corners) rendered on each face
- **Confidence-colored boxes** — green (high) / yellow (medium) / orange (low)
- **Live face blur mode** for privacy demos
- **MP4 video recording** with one keypress
- **Snapshot capture** to disk
- HUD with face count, smoothed FPS, detector name, recording indicator
- In-app help overlay

## Requirements

- Python 3.9+
- A working webcam
- macOS users: grant the terminal Camera access in **System Settings → Privacy & Security → Camera**

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

If the download fails — for example on a network without GitHub access — the app falls back to OpenCV's bundled Haar cascade. You can also force Haar with `--detector haar`.

## Usage

```bash
python face_detection.py
```

Common options:

```bash
# Pick the detector explicitly
python face_detection.py --detector yunet
python face_detection.py --detector haar

# Different camera or resolution
python face_detection.py --camera 1 --width 640 --height 480

# Selfie-mirror, start in blur mode
python face_detection.py --mirror --blur

# Tighter confidence threshold (YuNet)
python face_detection.py --score-threshold 0.8

# Disable bbox smoothing / stable IDs
python face_detection.py --no-tracker
```

Run `python face_detection.py --help` for the full list.

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

## How it works

1. `cv2.VideoCapture` grabs frames from the selected camera.
2. The active detector returns axis-aligned bounding boxes (and, for YuNet, a confidence score plus five facial landmarks).
3. An IoU-based tracker matches detections to existing tracks and smooths each track's bbox with an exponential moving average. New detections get a new monotonically-increasing ID; stale tracks die after a configurable number of missed frames.
4. The renderer overlays boxes, landmarks, IDs, and the HUD onto the original frame.
5. If recording is on, the annotated frame is written to an MP4 via `cv2.VideoWriter`.

## Project layout

```
face_detection.py    main loop, CLI, rendering, key handling
detectors.py         YuNetDetector, HaarDetector, model auto-download
tracker.py           greedy IoU tracker with EMA smoothing
download_model.py    standalone helper to pre-fetch the YuNet ONNX
```

## License

MIT
