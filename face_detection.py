"""Privacy-first face redaction for webcams, images, and video files."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2

from detectors import HaarDetector, YuNetDetector, find_haar_path, find_yunet_path
from media import (
    IMAGE_SUFFIXES,
    VIDEO_OUTPUT_SUFFIXES,
    MediaError,
    MediaPipeline,
    MediaStats,
    media_kind,
)
from pipeline import (
    REDACTION_MODES,
    FrameProcessor,
    FrameResult,
    OverlayConfig,
    RedactionConfig,
    draw_tracks,
)
from tracker import IoUTracker


@dataclass
class Recorder:
    writer: cv2.VideoWriter
    path: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_color(value: str) -> tuple[int, int, int]:
    normalized = value.removeprefix("#")
    if len(normalized) != 6:
        raise argparse.ArgumentTypeError("must be a six-digit hex color such as #000000")
    try:
        red, green, blue = (
            int(normalized[0:2], 16),
            int(normalized[2:4], 16),
            int(normalized[4:6], 16),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a valid six-digit hex color") from exc
    return blue, green, red


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and redact faces in a webcam feed, image, or video."
    )
    parser.add_argument("--input", type=Path, help="Image or video path; omit to use a webcam")
    parser.add_argument("--output", type=Path, help="Write redacted image/video to this path")
    parser.add_argument("--headless", action="store_true", help="Process without opening a window")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index when --input is omitted")
    parser.add_argument("--width", type=positive_int, default=1280, help="Requested webcam width")
    parser.add_argument("--height", type=positive_int, default=720, help="Requested webcam height")
    parser.add_argument("--max-frames", type=positive_int, help="Stop after this many video frames")

    parser.add_argument("--detector", choices=["auto", "yunet", "haar"], default="auto")
    parser.add_argument("--score-threshold", type=unit_interval, default=0.6)
    parser.add_argument("--nms-threshold", type=unit_interval, default=0.3)
    parser.add_argument("--no-tracker", action="store_true", help="Disable stable face tracks")
    parser.add_argument("--model-dir", type=Path, help="Directory containing the YuNet ONNX model")
    parser.add_argument("--haar-path", type=Path, help="Path to a Haar cascade XML file")
    parser.add_argument("--no-download", action="store_true", help="Never download the YuNet model")

    redaction = parser.add_mutually_exclusive_group()
    redaction.add_argument(
        "--redaction",
        choices=REDACTION_MODES,
        default="solid",
        help="Privacy transform (default: solid, the most conservative option)",
    )
    redaction.add_argument(
        "--blur",
        dest="redaction",
        action="store_const",
        const="blur",
        help="Compatibility shortcut for --redaction blur",
    )
    redaction.add_argument(
        "--no-redaction",
        action="store_true",
        help="Explicitly disable privacy redaction",
    )
    parser.add_argument(
        "--padding",
        type=unit_interval,
        default=0.25,
        help="Fractional padding on every side of a detected face (default: 0.25)",
    )
    parser.add_argument(
        "--hold-frames",
        type=nonnegative_int,
        default=2,
        help="Keep redacting the last known region through brief detector misses (default: 2)",
    )
    parser.add_argument("--pixel-size", type=positive_int, default=14)
    parser.add_argument("--solid-color", type=parse_color, default=(0, 0, 0), metavar="#RRGGBB")
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="Include boxes, landmarks, IDs, and confidence labels in saved output",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror frames before processing")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--recording-dir", type=Path, default=Path("recordings"))
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    input_kind = None
    if args.input is not None:
        if not args.input.is_file():
            parser.error(f"input does not exist or is not a file: {args.input}")
        try:
            input_kind = media_kind(args.input)
        except MediaError as exc:
            parser.error(str(exc))

    if args.headless and args.output is None:
        parser.error("--headless requires --output so processed media is not discarded")
    if input_kind == "image" and args.max_frames is not None:
        parser.error("--max-frames applies only to video or webcam input")

    if args.output is not None:
        output_suffix = args.output.suffix.lower()
        allowed_suffixes = IMAGE_SUFFIXES if input_kind == "image" else VIDEO_OUTPUT_SUFFIXES
        if output_suffix not in allowed_suffixes:
            allowed = ", ".join(sorted(allowed_suffixes))
            parser.error(f"output extension must be one of: {allowed}")
        if args.input is not None and args.output.resolve() == args.input.resolve():
            parser.error("input and output paths must be different")
        if args.output.exists() and not args.overwrite:
            parser.error(f"output already exists; pass --overwrite to replace it: {args.output}")

    if args.pixel_size < 2:
        parser.error("--pixel-size must be at least 2")
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return validate_args(parser, parser.parse_args(argv))


def build_detector(args: argparse.Namespace):
    if args.detector in ("auto", "yunet"):
        model_dir = args.model_dir or (Path(__file__).parent / "models")
        yunet_path = find_yunet_path(model_dir, download=not args.no_download)
        if yunet_path is not None:
            try:
                detector = YuNetDetector(
                    str(yunet_path), args.score_threshold, args.nms_threshold
                )
                print(f"Using YuNet detector: {yunet_path}")
                return detector
            except Exception as exc:
                if args.detector == "yunet":
                    raise RuntimeError(f"YuNet initialization failed: {exc}") from exc
                print(f"YuNet initialization failed ({exc}); falling back to Haar.", file=sys.stderr)
        elif args.detector == "yunet":
            raise RuntimeError(
                "YuNet model unavailable. Run download_model.py or omit --no-download."
            )

    haar_path = find_haar_path(args.haar_path)
    print(f"Using Haar detector: {haar_path}")
    return HaarDetector(str(haar_path))


def overlay_text(
    frame,
    lines: Iterable[str],
    x: int = 10,
    y: int = 28,
    color=(255, 255, 255),
) -> None:
    for index, line in enumerate(lines):
        line_y = y + index * 24
        cv2.putText(
            frame,
            line,
            (x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )


HELP_LINES = (
    "q / Esc  quit",
    "s        save redacted snapshot",
    "r        toggle redacted recording",
    "b        toggle blur / configured mode",
    "l        toggle landmark points",
    "i        toggle face ID labels",
    "m        toggle mirror",
    "h        toggle this help",
)


class PreviewSession:
    """Interactive preview state kept separate from persisted output frames."""

    window_name = "Face Privacy Redactor — press h for help"

    def __init__(self, args: argparse.Namespace, processor: FrameProcessor) -> None:
        self.args = args
        self.processor = processor
        self.mirror = args.mirror
        self.show_help = False
        self.recorder: Optional[Recorder] = None
        self._mode_before_blur = (
            processor.redaction.mode if processor.redaction.mode != "blur" else "solid"
        )
        self._last_frame_time = time.perf_counter()

    def transform(self, frame):
        return cv2.flip(frame, 1) if self.mirror else frame

    def _decorate(self, result: FrameResult, stats: MediaStats) -> object:
        preview = result.frame.copy()
        if not self.args.overlays:
            draw_tracks(preview, result.tracks, self.processor.overlays)

        now = time.perf_counter()
        elapsed = max(now - self._last_frame_time, 1e-9)
        self._last_frame_time = now
        mode = self.processor.redaction.mode if self.processor.redaction.enabled else "OFF"
        hud = [
            f"Faces: {result.face_count}  FPS: {1.0 / elapsed:5.1f}  Detector: {result.detector_name}",
            f"Redaction: {mode.upper()}  Padding: {self.processor.redaction.padding:.0%}",
        ]
        if self.recorder is not None:
            hud.append(f"REC - {self.recorder.path.name}")
        overlay_text(preview, hud)
        if self.show_help:
            overlay_text(preview, HELP_LINES, x=10, y=100, color=(200, 255, 200))
        return preview

    def _start_recording(self, result: FrameResult, fps: float) -> None:
        self.args.recording_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.recording_dir / f"recording_{time.time_ns()}.mp4"
        height, width = result.frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            print("Could not start the interactive video writer.", file=sys.stderr)
            return
        self.recorder = Recorder(writer, path)
        print(f"Recording redacted output to {path}")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        self.recorder.writer.release()
        print(f"Saved {self.recorder.path}")
        self.recorder = None

    def _save_snapshot(self, result: FrameResult) -> None:
        self.args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.snapshot_dir / f"snapshot_{time.time_ns()}.png"
        if cv2.imwrite(str(path), result.frame):
            print(f"Saved {path}")
        else:
            print(f"Could not save {path}", file=sys.stderr)

    def _handle_key(self, key: int, result: FrameResult, fps: float) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("s"):
            self._save_snapshot(result)
        elif key == ord("r"):
            if self.recorder is None:
                self._start_recording(result, fps)
            else:
                self._stop_recording()
        elif key == ord("b"):
            if self.processor.redaction.mode == "blur":
                self.processor.redaction.mode = self._mode_before_blur
            else:
                self._mode_before_blur = self.processor.redaction.mode
                self.processor.redaction.mode = "blur"
                self.processor.redaction.enabled = True
        elif key == ord("l"):
            self.processor.overlays.show_landmarks = not self.processor.overlays.show_landmarks
        elif key == ord("i"):
            self.processor.overlays.show_ids = not self.processor.overlays.show_ids
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key == ord("h"):
            self.show_help = not self.show_help
        return True

    def on_frame(self, result: FrameResult, stats: MediaStats, fps: float) -> bool:
        if self.recorder is not None:
            self.recorder.writer.write(result.frame)
        cv2.imshow(self.window_name, self._decorate(result, stats))
        return self._handle_key(cv2.waitKey(1) & 0xFF, result, fps)

    def _process_still(self, source_frame) -> FrameResult:
        """Re-run a still from its untouched source after an interactive change."""
        self.processor.reset()
        return self.processor.process(
            self.transform(source_frame), include_overlays=self.args.overlays
        )

    def show_still(self, source_frame, stats: MediaStats) -> None:
        result = self._process_still(source_frame)
        while True:
            cv2.imshow(self.window_name, self._decorate(result, stats))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                print("Recording is available for webcam and video previews, not still images.")
                continue

            self._handle_key(key, result, 1.0)
            if key in (ord("b"), ord("l"), ord("i"), ord("m")):
                result = self._process_still(source_frame)

    def close(self) -> None:
        self._stop_recording()
        cv2.destroyAllWindows()


def print_summary(stats: MediaStats, output: Optional[Path]) -> None:
    destination = f" -> {output}" if output is not None else ""
    unique_tracks = stats.unique_tracks
    if unique_tracks is None:
        tracking_summary = "tracking disabled"
    else:
        noun = "track" if unique_tracks == 1 else "tracks"
        tracking_summary = f"{unique_tracks} unique {noun}"
    print(
        f"Processed {stats.frames_processed} frame(s), "
        f"{stats.face_observations} face observation(s), "
        f"{tracking_summary} in {stats.elapsed_seconds:.2f}s{destination}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    preview: Optional[PreviewSession] = None
    try:
        detector = build_detector(args)
        tracker = None if args.no_tracker else IoUTracker(max_misses=max(8, args.hold_frames))
        processor = FrameProcessor(
            detector,
            tracker=tracker,
            redaction=RedactionConfig(
                enabled=not args.no_redaction,
                mode=args.redaction,
                padding=args.padding,
                hold_frames=args.hold_frames,
                pixel_size=args.pixel_size,
                solid_color=args.solid_color,
            ),
            overlays=OverlayConfig(),
        )
        pipeline = MediaPipeline(processor)
        if not args.headless:
            preview = PreviewSession(args, processor)

        transform = preview.transform if preview is not None else (
            (lambda frame: cv2.flip(frame, 1)) if args.mirror else None
        )

        if args.input is not None and media_kind(args.input) == "image":
            still_source = None
            if preview is not None:
                still_source = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
                if still_source is None:
                    raise MediaError(f"Could not read image: {args.input}")
            result, stats = pipeline.process_image(
                args.input,
                output_path=args.output,
                include_overlays=args.overlays,
                frame_transform=transform,
            )
            if preview is not None:
                preview.show_still(still_source, stats)
        else:
            source = str(args.input) if args.input is not None else args.camera
            stats = pipeline.process_capture(
                source,
                output_path=args.output,
                include_overlays=args.overlays,
                frame_transform=transform,
                on_frame=preview.on_frame if preview is not None else None,
                requested_size=(args.width, args.height) if args.input is None else None,
                max_frames=args.max_frames,
            )

        print_summary(stats, args.output)
        return 0
    except (MediaError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    sys.exit(main())
