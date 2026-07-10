"""Real-time face detection from webcam, video file, or still image."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import cv2

from analytics import SessionAnalytics
from detectors import HaarDetector, YuNetDetector, find_haar_path, find_yunet_path
from pose import draw_pose_indicator, estimate_yaw
from tracker import IoUTracker, Track


class SourceKind(str, Enum):
    CAMERA = "camera"
    VIDEO = "video"
    IMAGE = "image"


@dataclass
class InputSource:
    kind: SourceKind
    camera_index: Optional[int] = None
    path: Optional[Path] = None

    def describe(self) -> str:
        if self.kind is SourceKind.CAMERA:
            return f"camera:{self.camera_index}"
        return f"{self.kind.value}:{self.path}"


@dataclass
class Recorder:
    writer: cv2.VideoWriter
    path: Path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time face detection (webcam, video file, or image).",
    )
    p.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Image path, video path, or camera index (default: use --camera)",
    )
    p.add_argument(
        "--source",
        dest="source_flag",
        default=None,
        help="Same as positional source: image/video path or camera index",
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    p.add_argument("--detector", choices=["auto", "yunet", "haar"], default="auto")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--score-threshold", type=float, default=0.6)
    p.add_argument("--nms-threshold", type=float, default=0.3)
    p.add_argument("--no-tracker", action="store_true", help="Disable bbox smoothing/IDs")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--blur", action="store_true", help="Start in blur privacy mode")
    p.add_argument("--pixelate", action="store_true", help="Start in pixelate privacy mode")
    p.add_argument("--snapshot-dir", default="snapshots")
    p.add_argument("--recording-dir", default="recordings")
    p.add_argument("--model-dir", default=None, help="Where to look for YuNet ONNX (default: ./models)")
    p.add_argument("--no-download", action="store_true", help="Don't auto-download the YuNet model")
    p.add_argument(
        "--export-stats",
        default=None,
        metavar="PATH",
        help="Write session analytics JSON on quit (also enabled with any path)",
    )
    p.add_argument(
        "--gallery",
        action="store_true",
        help="Save cropped faces for stable tracks into --gallery-dir",
    )
    p.add_argument("--gallery-dir", default="gallery", help="Directory for face gallery crops")
    p.add_argument(
        "--gallery-min-age",
        type=int,
        default=15,
        help="Frames a track must exist before gallery capture (default: 15)",
    )
    p.add_argument(
        "--save-image",
        default=None,
        metavar="PATH",
        help="When processing a still image, also write the annotated result here",
    )
    return p.parse_args(argv)


def resolve_source(args: argparse.Namespace) -> InputSource:
    """Resolve CLI args into a camera, video, or image source."""
    raw = args.source_flag if args.source_flag is not None else args.source
    if raw is None:
        return InputSource(kind=SourceKind.CAMERA, camera_index=args.camera)

    text = str(raw).strip()
    # Pure integer → camera index (including "0")
    if text.lstrip("-").isdigit() and text.count("-") <= 1:
        return InputSource(kind=SourceKind.CAMERA, camera_index=int(text))

    path = Path(text).expanduser()
    if not path.exists():
        print(f"Error: source path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
    if suffix in image_exts:
        return InputSource(kind=SourceKind.IMAGE, path=path)
    if suffix in video_exts:
        return InputSource(kind=SourceKind.VIDEO, path=path)

    # Probe with OpenCV when extension is ambiguous.
    probe = cv2.imread(str(path))
    if probe is not None:
        return InputSource(kind=SourceKind.IMAGE, path=path)
    return InputSource(kind=SourceKind.VIDEO, path=path)


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


def color_for_score(score: float) -> Tuple[int, int, int]:
    if score >= 0.85:
        return (0, 255, 0)
    if score >= 0.65:
        return (0, 255, 255)
    return (0, 165, 255)


def draw_track(
    frame,
    track: Track,
    show_landmarks: bool,
    show_id: bool,
    show_pose: bool = True,
) -> None:
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

    if show_pose and track.landmarks:
        pose = estimate_yaw(track.landmarks)
        if pose is not None:
            draw_pose_indicator(frame, track.bbox_int, pose, color=color)


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


def pixelate_face(frame, track: Track) -> None:
    """Block-pixelate a face; block size scales with face size."""
    x, y, w, h = track.bbox_int
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return
    # Larger faces → larger blocks; keep at least 4px blocks, at most ~1/8 of face.
    block = max(4, min(rh, rw) // 8)
    small_w = max(1, rw // block)
    small_h = max(1, rh // block)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def maybe_save_gallery(
    frame,
    tracks: Iterable[Track],
    gallery_dir: Path,
    min_age: int,
    saved_ids: Set[int],
) -> None:
    """Save one cropped face per track once it is stable enough."""
    height, width = frame.shape[:2]
    for track in tracks:
        if track.id in saved_ids or track.misses != 0 or track.age < min_age:
            continue
        x, y, w, h = track.bbox_int
        # Slight padding around the box.
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + w + pad_x)
        y1 = min(height, y + h + pad_y)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        gallery_dir.mkdir(parents=True, exist_ok=True)
        path = gallery_dir / f"face_{track.id:04d}.jpg"
        if cv2.imwrite(str(path), crop):
            saved_ids.add(track.id)
            print(f"Gallery saved {path}")


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
    "p        toggle face pixelate",
    "g        toggle gallery capture",
    "l        toggle landmarks",
    "i        toggle IDs",
    "m        toggle mirror",
    "h        toggle this help",
)


def open_capture(source: InputSource, width: int, height: int) -> cv2.VideoCapture:
    if source.kind is SourceKind.CAMERA:
        cap = cv2.VideoCapture(source.camera_index)  # type: ignore[arg-type]
        if not cap.isOpened():
            print(f"Error: could not open camera index {source.camera_index}.", file=sys.stderr)
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    assert source.path is not None
    cap = cv2.VideoCapture(str(source.path))
    if not cap.isOpened():
        print(f"Error: could not open video: {source.path}", file=sys.stderr)
        sys.exit(1)
    return cap


def process_frame(
    frame,
    detector,
    tracker: Optional[IoUTracker],
    *,
    blur_mode: bool,
    pixelate_mode: bool,
    show_landmarks: bool,
    show_ids: bool,
) -> List[Track]:
    detections = detector.detect(frame)
    if tracker is not None:
        tracks = [t for t in tracker.update(detections) if t.misses == 0]
    else:
        tracks = [Track.from_detection(d) for d in detections]

    if pixelate_mode:
        for t in tracks:
            pixelate_face(frame, t)
    elif blur_mode:
        for t in tracks:
            blur_face(frame, t)
    else:
        for t in tracks:
            draw_track(frame, t, show_landmarks=show_landmarks, show_id=show_ids)
    return tracks


def run_image(args: argparse.Namespace, source: InputSource, detector) -> int:
    assert source.path is not None
    frame = cv2.imread(str(source.path))
    if frame is None:
        print(f"Error: could not read image: {source.path}", file=sys.stderr)
        return 1

    tracker = None if args.no_tracker else IoUTracker()
    analytics = SessionAnalytics()
    blur_mode = args.blur and not args.pixelate
    pixelate_mode = args.pixelate
    show_landmarks = True
    show_ids = True
    show_help = False
    mirror = args.mirror
    gallery_enabled = args.gallery
    gallery_dir = Path(args.gallery_dir)
    gallery_saved: Set[int] = set()

    if mirror:
        frame = cv2.flip(frame, 1)

    work = frame.copy()
    tracks = process_frame(
        work,
        detector,
        tracker,
        blur_mode=blur_mode,
        pixelate_mode=pixelate_mode,
        show_landmarks=show_landmarks,
        show_ids=show_ids,
    )
    analytics.update(
        len(tracks),
        [t.score for t in tracks],
        [t.id for t in tracks],
    )
    if gallery_enabled:
        # Force age so a single-frame image still captures stable-looking crops.
        for t in tracks:
            t.age = max(t.age, args.gallery_min_age)
        # Always crop from the clean source frame (not privacy-filtered).
        maybe_save_gallery(frame, tracks, gallery_dir, args.gallery_min_age, gallery_saved)

    hud = [
        f"Faces: {len(tracks)}   Detector: {detector.name}   Image",
        *analytics.hud_lines(),
    ]
    if blur_mode:
        hud.append("BLUR ON")
    if pixelate_mode:
        hud.append("PIXELATE ON")
    if gallery_enabled:
        hud.append(f"GALLERY ({len(gallery_saved)} saved)")
    overlay_text(work, hud)

    if args.save_image:
        out = Path(args.save_image)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), work)
        print(f"Saved annotated image → {out}")

    window_name = "Face Detection — press h for help"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            display = work.copy()
            if show_help:
                overlay_text(display, HELP_LINES, x=10, y=100, color=(200, 255, 200))
            cv2.imshow(window_name, display)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                snapshot_dir = Path(args.snapshot_dir)
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = snapshot_dir / f"snapshot_{int(time.time())}.png"
                cv2.imwrite(str(path), work)
                print(f"Saved {path}")
            elif key == ord("b"):
                blur_mode = not blur_mode
                if blur_mode:
                    pixelate_mode = False
                work = frame.copy()
                if mirror:
                    pass  # already mirrored into frame
                tracks = process_frame(
                    work, detector, None if args.no_tracker else IoUTracker(),
                    blur_mode=blur_mode, pixelate_mode=pixelate_mode,
                    show_landmarks=show_landmarks, show_ids=show_ids,
                )
                hud = [
                    f"Faces: {len(tracks)}   Detector: {detector.name}   Image",
                    *analytics.hud_lines(),
                ]
                if blur_mode:
                    hud.append("BLUR ON")
                if pixelate_mode:
                    hud.append("PIXELATE ON")
                overlay_text(work, hud)
            elif key == ord("p"):
                pixelate_mode = not pixelate_mode
                if pixelate_mode:
                    blur_mode = False
                work = frame.copy()
                tracks = process_frame(
                    work, detector, None if args.no_tracker else IoUTracker(),
                    blur_mode=blur_mode, pixelate_mode=pixelate_mode,
                    show_landmarks=show_landmarks, show_ids=show_ids,
                )
                hud = [
                    f"Faces: {len(tracks)}   Detector: {detector.name}   Image",
                    *analytics.hud_lines(),
                ]
                if blur_mode:
                    hud.append("BLUR ON")
                if pixelate_mode:
                    hud.append("PIXELATE ON")
                overlay_text(work, hud)
            elif key == ord("l"):
                show_landmarks = not show_landmarks
                work = frame.copy()
                tracks = process_frame(
                    work, detector, None if args.no_tracker else IoUTracker(),
                    blur_mode=blur_mode, pixelate_mode=pixelate_mode,
                    show_landmarks=show_landmarks, show_ids=show_ids,
                )
                hud = [
                    f"Faces: {len(tracks)}   Detector: {detector.name}   Image",
                    *analytics.hud_lines(),
                ]
                overlay_text(work, hud)
            elif key == ord("i"):
                show_ids = not show_ids
                work = frame.copy()
                tracks = process_frame(
                    work, detector, None if args.no_tracker else IoUTracker(),
                    blur_mode=blur_mode, pixelate_mode=pixelate_mode,
                    show_landmarks=show_landmarks, show_ids=show_ids,
                )
                hud = [
                    f"Faces: {len(tracks)}   Detector: {detector.name}   Image",
                    *analytics.hud_lines(),
                ]
                overlay_text(work, hud)
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("g"):
                gallery_enabled = not gallery_enabled
                print(f"Gallery capture {'ON' if gallery_enabled else 'OFF'}")
                if gallery_enabled:
                    for t in tracks:
                        t.age = max(t.age, args.gallery_min_age)
                    maybe_save_gallery(frame, tracks, gallery_dir, args.gallery_min_age, gallery_saved)
    finally:
        cv2.destroyAllWindows()
        if args.export_stats:
            out = analytics.export_json(args.export_stats)
            print(f"Wrote session stats → {out}")
    return 0


def run_stream(args: argparse.Namespace, source: InputSource, detector) -> int:
    tracker = None if args.no_tracker else IoUTracker()
    analytics = SessionAnalytics()
    cap = open_capture(source, args.width, args.height)

    window_name = "Face Detection — press h for help"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    show_landmarks = True
    show_ids = True
    show_help = False
    mirror = args.mirror
    # Pixelate takes precedence if both flags set at start.
    blur_mode = bool(args.blur) and not args.pixelate
    pixelate_mode = bool(args.pixelate)
    gallery_enabled = bool(args.gallery)
    gallery_dir = Path(args.gallery_dir)
    gallery_saved: Set[int] = set()

    snapshot_dir = Path(args.snapshot_dir)
    recording_dir = Path(args.recording_dir)
    recorder: Optional[Recorder] = None

    fps = 0.0
    prev = time.time()
    is_video_file = source.kind is SourceKind.VIDEO
    # Use source FPS for video recording when available.
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if src_fps <= 1e-3:
        src_fps = 24.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_video_file:
                    print("End of video.")
                else:
                    print("Warning: failed to grab frame.", file=sys.stderr)
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            # Gallery crops from the clean frame before privacy filters.
            clean = frame.copy() if gallery_enabled else frame

            tracks = process_frame(
                frame,
                detector,
                tracker,
                blur_mode=blur_mode,
                pixelate_mode=pixelate_mode,
                show_landmarks=show_landmarks,
                show_ids=show_ids,
            )
            analytics.update(
                len(tracks),
                [t.score for t in tracks],
                [t.id for t in tracks],
            )
            if gallery_enabled:
                maybe_save_gallery(clean, tracks, gallery_dir, args.gallery_min_age, gallery_saved)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            hud = [
                f"Faces: {len(tracks)}   FPS: {fps:5.1f}   Detector: {detector.name}",
                *analytics.hud_lines(),
            ]
            if blur_mode:
                hud.append("BLUR ON")
            if pixelate_mode:
                hud.append("PIXELATE ON")
            if gallery_enabled:
                hud.append(f"GALLERY ({len(gallery_saved)} saved)")
            if recorder is not None:
                hud.append(f"REC ● {recorder.path.name}")
            overlay_text(frame, hud)

            if show_help:
                overlay_text(frame, HELP_LINES, x=10, y=100, color=(200, 255, 200))

            if recorder is not None:
                recorder.writer.write(frame)

            cv2.imshow(window_name, frame)
            # Video files: wait based on source FPS for natural playback; camera: 1ms.
            wait_ms = max(1, int(round(1000.0 / src_fps))) if is_video_file else 1
            key = cv2.waitKey(wait_ms) & 0xFF
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
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(src_fps), (w, h)
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
                if blur_mode:
                    pixelate_mode = False
            elif key == ord("p"):
                pixelate_mode = not pixelate_mode
                if pixelate_mode:
                    blur_mode = False
            elif key == ord("g"):
                gallery_enabled = not gallery_enabled
                print(f"Gallery capture {'ON' if gallery_enabled else 'OFF'}")
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
        if args.export_stats:
            out = analytics.export_json(args.export_stats)
            print(f"Wrote session stats → {out}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    source = resolve_source(args)
    print(f"Input source: {source.describe()}")
    detector = build_detector(args)

    if source.kind is SourceKind.IMAGE:
        return run_image(args, source, detector)
    return run_stream(args, source, detector)


if __name__ == "__main__":
    sys.exit(main())
