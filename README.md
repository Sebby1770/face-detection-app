# Face Detection App

Real-time face detection from a webcam feed, built with [OpenCV](https://opencv.org/) and Python. Uses Haar cascade classifiers shipped with OpenCV — no external model downloads required.

## Features

- Real-time face detection from any connected webcam
- Optional eye detection inside each detected face
- Live HUD with face count and smoothed FPS
- Snapshot capture to disk with a single keypress
- Mirror mode, configurable resolution, and detector tuning via CLI flags

## Requirements

- Python 3.8+
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

## Usage

```bash
python face_detection.py
```

Common options:

```bash
# Use a different camera
python face_detection.py --camera 1

# Mirror the feed (selfie-style)
python face_detection.py --mirror

# Lower resolution for slower machines
python face_detection.py --width 640 --height 480

# Disable eye detection for a small speedup
python face_detection.py --no-eyes
```

Run `python face_detection.py --help` for the full list of flags.

## Controls

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `s` | Save the current frame to `snapshots/` |
| `e` | Toggle eye detection |

## How it works

1. OpenCV captures frames from the selected camera.
2. Each frame is converted to grayscale and histogram-equalized to improve detection under varied lighting.
3. A Haar cascade classifier (`haarcascade_frontalface_default.xml`) scans the frame at multiple scales and returns bounding boxes for faces.
4. For each face, an optional eye cascade runs on the face region to find eyes.
5. Boxes, labels, and a HUD are drawn on top of the original color frame before display.

Haar cascades are fast and CPU-only, which makes them well suited to a simple real-time demo. For more robust detection in low light or at extreme angles, consider swapping the cascade for a DNN model (e.g. OpenCV's `dnn` module with a Caffe / ONNX face detector).

## License

MIT
