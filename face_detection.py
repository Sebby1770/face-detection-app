"""Privacy-first face redaction for webcams, images, and video files."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2

from analytics import SessionAnalytics
from batch import (
    coverage_entry,
    coverage_totals,
    print_coverage_summary,
    process_tree,
    write_coverage,
)
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
    MASK_SHAPES,
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


class SourceKind(str, Enum):
    CAMERA = "camera"
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class InputSource:
    kind: SourceKind
    path: Optional[Path] = None
    camera_index: Optional[int] = None


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


def parse_id_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids: list[int] = []
    for part in parts:
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be a comma-separated list of integers such as 1,3"
            ) from exc
    return ids


def _source_from_path(path: Path) -> InputSource:
    if not path.exists():
        raise SystemExit(f"input does not exist: {path}")
    if path.is_dir():
        raise SystemExit(f"expected an image or video file, not a directory: {path}")
    try:
        kind = media_kind(path)
    except MediaError as exc:
        raise SystemExit(str(exc)) from exc
    if kind == "image":
        return InputSource(kind=SourceKind.IMAGE, path=path)
    return InputSource(kind=SourceKind.VIDEO, path=path)


def resolve_source(args: argparse.Namespace) -> InputSource:
    """Resolve camera / image / video input from a CLI-style namespace."""
    source_flag = getattr(args, "source_flag", None)
    source = getattr(args, "source", None)
    camera = getattr(args, "camera", 0)

    candidate = source_flag if source_flag is not None else source
    if candidate is not None:
        text = str(candidate)
        if text.isdigit():
            return InputSource(kind=SourceKind.CAMERA, camera_index=int(text))
        return _source_from_path(Path(candidate))

    return InputSource(kind=SourceKind.CAMERA, camera_index=int(camera))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and redact faces in a webcam feed, image, video, or directory."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="Image, video, or directory path; alias of --input",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Image, video, or directory path; omit to use a webcam",
    )
    parser.add_argument("--output", type=Path, help="Write redacted image/video/tree to this path")
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
        "--pixelate",
        dest="redaction",
        action="store_const",
        const="pixelate",
        help="Compatibility shortcut for --redaction pixelate",
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
        "--mask",
        choices=MASK_SHAPES,
        default="box",
        help="Redaction mask shape (default: box)",
    )
    parser.add_argument(
        "--feather",
        type=nonnegative_int,
        default=0,
        help="Soft-edge blur in pixels applied to the redaction mask (default: 0)",
    )
    parser.add_argument(
        "--min-size",
        type=nonnegative_int,
        default=0,
        help="Ignore detections smaller than this width or height in pixels (default: 0)",
    )
    parser.add_argument(
        "--redact-ids",
        type=parse_id_list,
        default=None,
        metavar="IDS",
        help="Only redact these track IDs (comma-separated, e.g. 1,3)",
    )
    parser.add_argument(
        "--keep-ids",
        type=parse_id_list,
        default=None,
        metavar="IDS",
        help="Leave these track IDs unredacted (comma-separated; wins over --redact-ids)",
    )
    parser.add_argument(
        "--overlays",
        action="store_true",
        help="Include boxes, landmarks, IDs, and confidence labels in saved output",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Show boxes, landmarks, IDs, and pose in the preview window only",
    )
    parser.add_argument(
        "--pose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw head-pose yaw (default: on with --review). Saved only with --overlays.",
    )
    parser.add_argument(
        "--export-stats",
        type=Path,
        metavar="PATH",
        help="Write session analytics JSON to this path",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        metavar="PATH",
        help="Write per-file coverage JSON (batch or single-file)",
    )
    parser.add_argument(
        "--identify",
        action="store_true",
        help="Reserved for identifying exports such as face crops; this release does not write crops",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror frames before processing")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--recording-dir", type=Path, default=Path("recordings"))
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.identify:
        parser.error(
            "--identify is reserved; this release does not export face crops. "
            "Use --review then --keep-ids/--redact-ids."
        )

    if args.source is not None:
        if args.input is not None and args.input.resolve() != args.source.resolve():
            parser.error("provide a positional path or --input, not both")
        args.input = args.source

    args.batch = False
    input_kind = None
    if args.input is not None:
        if args.input.is_dir():
            if args.output is None:
                parser.error("directory --input requires a directory --output")
            if args.output.exists() and not args.output.is_dir():
                parser.error("--output must be a directory when --input is a directory")
            if args.output.exists() and args.input.resolve() == args.output.resolve():
                parser.error("input and output directories must be different")
            args.batch = True
            args.headless = True
        elif args.input.is_file():
            try:
                input_kind = media_kind(args.input)
            except MediaError as exc:
                parser.error(str(exc))
        else:
            parser.error(f"input does not exist or is not a file: {args.input}")

    if args.headless and args.output is None:
        parser.error("--headless requires --output so processed media is not discarded")
    if input_kind == "image" and args.max_frames is not None:
        parser.error("--max-frames applies only to video or webcam input")

    if args.output is not None and not args.batch:
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


def pose_enabled(args: argparse.Namespace) -> bool:
    pose = getattr(args, "pose", None)
    if pose is False:
        return False
    if pose is True:
        return True
    return bool(getattr(args, "review", False))


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


def build_processor(args: argparse.Namespace, detector) -> FrameProcessor:
    tracker = None if args.no_tracker else IoUTracker(max_misses=max(8, args.hold_frames))
    return FrameProcessor(
        detector,
        tracker=tracker,
        redaction=RedactionConfig(
            enabled=not args.no_redaction,
            mode=args.redaction,
            padding=args.padding,
            hold_frames=args.hold_frames,
            pixel_size=args.pixel_size,
            solid_color=args.solid_color,
            shape=args.mask,
            feather=args.feather,
            min_size=args.min_size,
        ),
        overlays=OverlayConfig(),
        redact_ids=args.redact_ids,
        keep_ids=args.keep_ids,
        min_size=args.min_size,
    )


def bind_session_hooks(
    processor: FrameProcessor,
    analytics: Optional[SessionAnalytics],
    *,
    draw_pose_on_output: bool,
) -> None:
    inner = processor.process

    def process(
        frame,
        *,
        include_overlays: bool = False,
        record_analytics: bool = True,
        **kwargs,
    ):
        result = inner(frame, include_overlays=include_overlays, **kwargs)
        if analytics is not None and record_analytics:
            analytics.update(
                result.face_count,
                [track.score for track in result.tracks],
                [track.id for track in result.tracks],
            )
        if include_overlays and draw_pose_on_output:
            try_draw_pose(result.frame, result.tracks)
        return result

    processor.process = process  # type: ignore[method-assign]


_pose_import_error_emitted = False


def try_draw_pose(frame, tracks) -> None:
    """Draw yaw indicators when landmarks exist; never raise into the frame loop."""
    global _pose_import_error_emitted
    try:
        from pose import draw_pose_indicator, estimate_yaw
    except Exception as exc:
        if not _pose_import_error_emitted:
            print(f"Pose overlay unavailable ({exc}). Use --no-pose to silence this.", file=sys.stderr)
            _pose_import_error_emitted = True
        return

    for track in tracks:
        if len(track.landmarks) < 3:
            continue
        pose = estimate_yaw(track.landmarks)
        if pose is None:
            continue
        draw_pose_indicator(frame, track.bbox_int, pose)


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
    "p        toggle pixelate / configured mode",
    "e        toggle box / ellipse mask",
    "f        toggle feather 0 / 8",
    "l        toggle landmark points",
    "i        toggle face ID labels",
    "m        toggle mirror",
    "h        toggle this help",
)


class PreviewSession:
    """Interactive preview state kept separate from persisted output frames."""

    window_name = "Face Privacy Redactor — press h for help"

    def __init__(
        self,
        args: argparse.Namespace,
        processor: FrameProcessor,
        analytics: Optional[SessionAnalytics] = None,
    ) -> None:
        self.args = args
        self.processor = processor
        self.analytics = analytics if analytics is not None else SessionAnalytics()
        self.mirror = args.mirror
        self.show_help = False
        self.recorder: Optional[Recorder] = None
        self._mode_before_blur = (
            processor.redaction.mode if processor.redaction.mode != "blur" else "solid"
        )
        self._mode_before_pixelate = (
            processor.redaction.mode if processor.redaction.mode != "pixelate" else "solid"
        )
        self._last_frame_time = time.perf_counter()
        self._pose_enabled = pose_enabled(args)

    def transform(self, frame):
        return cv2.flip(frame, 1) if self.mirror else frame

    def _decorate(self, result: FrameResult, stats: MediaStats) -> object:
        preview = result.frame.copy()
        overlays_baked = bool(getattr(self.args, "overlays", False))
        if not overlays_baked:
            draw_tracks(preview, result.tracks, self.processor.overlays)
            if self._pose_enabled:
                try_draw_pose(preview, result.tracks)

        now = time.perf_counter()
        elapsed = max(now - self._last_frame_time, 1e-9)
        self._last_frame_time = now
        mode = self.processor.redaction.mode if self.processor.redaction.enabled else "OFF"
        hud = [
            f"Faces: {result.face_count}  FPS: {1.0 / elapsed:5.1f}  Detector: {result.detector_name}",
            (
                f"Redaction: {mode.upper()}  Padding: {self.processor.redaction.padding:.0%}  "
                f"Mask: {self.processor.redaction.shape}"
            ),
        ]
        hud.extend(self.analytics.hud_lines())
        if result.tracks:
            ids = ",".join(
                str(track.id) for track in sorted(result.tracks, key=lambda item: item.id)
            )
            hud.append(f"IDs: {ids}")
        hold_frames = self.processor.redaction.hold_frames
        for track in sorted(result.tracks, key=lambda item: item.id):
            if track.misses > 0:
                hud.append(f"hold #{track.id} {track.misses}/{hold_frames}")
        if result.face_count == 0:
            hud.append("NO FACE this frame")
        if self.recorder is not None:
            hud.append(f"REC - {self.recorder.path.name}")
        overlay_text(preview, hud)
        if self.show_help:
            overlay_text(preview, HELP_LINES, x=10, y=160, color=(200, 255, 200))
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
        elif key == ord("p"):
            if self.processor.redaction.mode == "pixelate":
                self.processor.redaction.mode = self._mode_before_pixelate
            else:
                self._mode_before_pixelate = self.processor.redaction.mode
                self.processor.redaction.mode = "pixelate"
                self.processor.redaction.enabled = True
        elif key == ord("e"):
            if self.processor.redaction.shape == "ellipse":
                self.processor.redaction.shape = "box"
            else:
                self.processor.redaction.shape = "ellipse"
        elif key == ord("f"):
            if self.processor.redaction.feather:
                self.processor.redaction.feather = 0
            else:
                self.processor.redaction.feather = 8
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
            self.transform(source_frame),
            include_overlays=self.args.overlays,
            record_analytics=False,
        )

    def show_still(
        self,
        source_frame,
        stats: MediaStats,
        initial_result: Optional[FrameResult] = None,
    ) -> None:
        result = initial_result if initial_result is not None else self._process_still(source_frame)
        while True:
            cv2.imshow(self.window_name, self._decorate(result, stats))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                print("Recording is available for webcam and video previews, not still images.")
                continue

            self._handle_key(key, result, 1.0)
            if key in (ord("b"), ord("p"), ord("e"), ord("f"), ord("l"), ord("i"), ord("m")):
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
    ids = stats.track_ids
    if ids:
        shown = ", ".join(str(track_id) for track_id in ids[:24])
        extra = "" if len(ids) <= 24 else f" (+{len(ids) - 24} more)"
        print(f"IDs        {shown}{extra}")
        print("Use --review then --keep-ids / --redact-ids with those numbers.")


def coverage_from_stats(path_label: str, kind: str, stats: MediaStats) -> dict:
    entry = coverage_entry(path_label, kind, stats)
    skipped: list[str] = []
    return {
        "files": [entry],
        "skipped": skipped,
        "totals": coverage_totals([entry], skipped=skipped),
    }


def maybe_export_reports(
    args: argparse.Namespace,
    analytics: SessionAnalytics,
    coverage: Optional[dict],
) -> None:
    if args.export_stats is not None:
        written = analytics.export_json(args.export_stats)
        print(f"Wrote session stats to {written}")
    if args.coverage is not None and coverage is not None:
        written = write_coverage(args.coverage, coverage)
        print(f"Wrote coverage to {written}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    preview: Optional[PreviewSession] = None
    analytics = SessionAnalytics()
    draw_pose_on_output = bool(args.overlays) and pose_enabled(args)
    try:
        detector = build_detector(args)

        if args.batch:
            def factory() -> FrameProcessor:
                processor = build_processor(args, detector)
                bind_session_hooks(
                    processor, analytics, draw_pose_on_output=draw_pose_on_output
                )
                return processor

            transform = (lambda frame: cv2.flip(frame, 1)) if args.mirror else None
            coverage = process_tree(
                args.input,
                args.output,
                factory,
                overwrite=args.overwrite,
                include_overlays=args.overlays,
                frame_transform=transform,
                max_frames=args.max_frames,
            )
            totals = coverage["totals"]
            print_coverage_summary(coverage, args.output)
            skipped_n = int(totals.get("skipped", 0))
            if skipped_n:
                print(
                    f"Skipped {skipped_n} existing file(s); pass --overwrite to replace them."
                )
            maybe_export_reports(args, analytics, coverage)
            return 0

        processor = build_processor(args, detector)
        bind_session_hooks(processor, analytics, draw_pose_on_output=draw_pose_on_output)
        pipeline = MediaPipeline(processor)
        if not args.headless:
            preview = PreviewSession(args, processor, analytics=analytics)

        transform = preview.transform if preview is not None else (
            (lambda frame: cv2.flip(frame, 1)) if args.mirror else None
        )

        coverage = None
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
            coverage = coverage_from_stats(args.input.name, "image", stats)
            if preview is not None:
                preview.show_still(still_source, stats, initial_result=result)
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
            kind = "video" if args.input is not None else "camera"
            label = args.input.name if args.input is not None else f"camera:{args.camera}"
            coverage = coverage_from_stats(label, kind, stats)

        print_summary(stats, args.output)
        if coverage is not None:
            print_coverage_summary(coverage, args.output)
        maybe_export_reports(args, analytics, coverage)
        return 0
    except (MediaError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    sys.exit(main())
