"""Real-time face detection from a webcam feed, with YuNet/Haar backends and tracking."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2

from detectors import HaarDetector, YuNetDetector, find_haar_path, find_yunet_path
from tracker import IoUTracker, Track


@dataclass
class Recorder:
    writer: cv2.VideoWriter
    path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time webcam face detection.")
    p.add_argument("--detector", choices=["auto", "yunet", "haar"], default="auto")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--score-threshold", type=float, default=0.6)
    p.add_argument("--nms-threshold", type=float, default=0.3)
    p.add_argument("--no-tracker", action="store_true", help="Disable bbox smoothing/IDs")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--blur", action="store_true", help="Start in blur mode")
    p.add_argument("--snapshot-dir", default="snapshots")
    p.add_argument("--recording-dir", default="recordings")
    p.add_argument("--model-dir", default=None, help="Where to look for YuNet ONNX (default: ./models)")
    p.add_argument("--no-download", action="store_true", help="Don't auto-download the YuNet model")
    return p.parse_args()


def build_detector(args: argparse.Namespace):
    if args.detector in ("auto", "yunet"):
        model_dir = Path(args.model_dir) if args.model_dir else (Path(__file__).parent / "models")
        yunet_path = find_yunet_path(model_dir, download=not args.no_download)
        if yunet_path is not None:
            try:
                detector = YuNetDetector(str(yunet_path), args.score_threshold, args.nms_threshold)
                print(f"Using YuNet detector: {yunet_path}")
                return detector
            except Exception as exc:
                print(f"YuNet init failed ({exc}); falling back to Haar.", file=sys.stderr)
        elif args.detector == "yunet":
            print(
                "YuNet model unavailable. Run download_model.py or omit --no-download.",
                file=sys.stderr,
            )
            sys.exit(1)

    haar_path = find_haar_path()
    print(f"Using Haar detector: {haar_path}")
    return HaarDetector(str(haar_path))


def color_for_score(score: float):
    if score >= 0.85:
        return (0, 255, 0)
    if score >= 0.65:
        return (0, 255, 255)
    return (0, 165, 255)


def draw_track(frame, track: Track, show_landmarks: bool, show_id: bool) -> None:
    x, y, w, h = track.bbox_int
    color = color_for_score(track.score)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"#{track.id} {track.score * 100:.0f}%" if show_id else f"{track.score * 100:.0f}%"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x, y - th - bl - 4), (x + tw + 8, y), color, -1)
    cv2.putText(
        frame, label, (x + 4, y - bl - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
    )

    if show_landmarks and track.landmarks:
        for lx, ly in track.landmarks:
            ix, iy = int(lx), int(ly)
            cv2.circle(frame, (ix, iy), 2, (255, 255, 255), -1)
            cv2.circle(frame, (ix, iy), 4, color, 1)


def blur_face(frame, track: Track) -> None:
    x, y, w, h = track.bbox_int
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    kernel = max(15, ((max(x1 - x0, y1 - y0) // 4) | 1))
    frame[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (kernel, kernel), 0)


def overlay_text(frame, lines: Iterable[str], x: int = 10, y: int = 28, color=(255, 255, 255)) -> None:
    for i, line in enumerate(lines):
        yy = y + i * 24
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


HELP_LINES = (
    "q / Esc  quit",
    "s        snapshot",
    "r        toggle recording",
    "b        toggle face blur",
    "l        toggle landmarks",
    "i        toggle IDs",
    "m        toggle mirror",
    "h        toggle this help",
)


def main() -> int:
    args = parse_args()
    detector = build_detector(args)
    tracker = None if args.no_tracker else IoUTracker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera index {args.camera}.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Face Detection — press h for help"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    show_landmarks = True
    show_ids = True
    show_help = False
    mirror = args.mirror
    blur_mode = args.blur

    snapshot_dir = Path(args.snapshot_dir)
    recording_dir = Path(args.recording_dir)
    recorder: Optional[Recorder] = None

    fps = 0.0
    prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to grab frame.", file=sys.stderr)
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            detections = detector.detect(frame)
            if tracker is not None:
                tracks = [t for t in tracker.update(detections) if t.misses == 0]
            else:
                tracks = [Track.from_detection(d) for d in detections]

            if blur_mode:
                for t in tracks:
                    blur_face(frame, t)
            else:
                for t in tracks:
                    draw_track(frame, t, show_landmarks=show_landmarks, show_id=show_ids)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            hud = [f"Faces: {len(tracks)}   FPS: {fps:5.1f}   Detector: {detector.name}"]
            if blur_mode:
                hud.append("BLUR ON")
            if recorder is not None:
                hud.append(f"REC ● {recorder.path.name}")
            overlay_text(frame, hud)

            if show_help:
                overlay_text(frame, HELP_LINES, x=10, y=80, color=(200, 255, 200))

            if recorder is not None:
                recorder.writer.write(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = snapshot_dir / f"snapshot_{int(time.time())}.png"
                cv2.imwrite(str(path), frame)
                print(f"Saved {path}")
            elif key == ord("r"):
                if recorder is None:
                    recording_dir.mkdir(parents=True, exist_ok=True)
                    path = recording_dir / f"recording_{int(time.time())}.mp4"
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h)
                    )
                    if writer.isOpened():
                        recorder = Recorder(writer, path)
                        print(f"Recording → {path}")
                    else:
                        print("Could not open VideoWriter; recording aborted.", file=sys.stderr)
                else:
                    recorder.writer.release()
                    print(f"Saved {recorder.path}")
                    recorder = None
            elif key == ord("b"):
                blur_mode = not blur_mode
            elif key == ord("l"):
                show_landmarks = not show_landmarks
            elif key == ord("i"):
                show_ids = not show_ids
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("h"):
                show_help = not show_help
    finally:
        if recorder is not None:
            recorder.writer.release()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
