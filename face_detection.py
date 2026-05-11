"""Real-time face detection from a webcam feed using OpenCV Haar cascades."""

import argparse
import sys
import time
from pathlib import Path

import cv2


def load_cascade(name: str) -> cv2.CascadeClassifier:
    path = Path(cv2.data.haarcascades) / name
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {path}")
    return cascade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face detection via webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="detectMultiScale scaleFactor")
    parser.add_argument("--min-neighbors", type=int, default=5, help="detectMultiScale minNeighbors")
    parser.add_argument("--min-size", type=int, default=60, help="Minimum face size in pixels")
    parser.add_argument("--no-eyes", action="store_true", help="Disable eye detection inside faces")
    parser.add_argument("--mirror", action="store_true", help="Horizontally flip the frame")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="snapshots",
        help="Directory used when saving snapshots (key: s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    face_cascade = load_cascade("haarcascade_frontalface_default.xml")
    eye_cascade = None if args.no_eyes else load_cascade("haarcascade_eye.xml")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera index {args.camera}.", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    snapshot_dir = Path(args.snapshot_dir)
    window_name = "Face Detection (q to quit, s to save, e to toggle eyes)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    show_eyes = eye_cascade is not None
    fps_smoothed = 0.0
    prev_time = time.time()

    print("Running. Press 'q' to quit, 's' to save a snapshot, 'e' to toggle eye detection.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: failed to grab frame.", file=sys.stderr)
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_size, args.min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face {w}x{h}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 4, y), (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (x + 2, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            if show_eyes and eye_cascade is not None:
                roi_gray = gray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]
                eyes = eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
                )
                for (ex, ey, ew, eh) in eyes:
                    cx, cy = ex + ew // 2, ey + eh // 2
                    radius = max(ew, eh) // 2
                    cv2.circle(roi_color, (cx, cy), radius, (255, 200, 0), 2)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt) if fps_smoothed else 1.0 / dt

        hud = f"Faces: {len(faces)}  FPS: {fps_smoothed:5.1f}  Eyes: {'on' if show_eyes else 'off'}"
        cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("e") and eye_cascade is not None:
            show_eyes = not show_eyes
        if key == ord("s"):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            filename = snapshot_dir / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(filename), frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
