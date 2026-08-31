"""OpenCV-backed image, video, and camera source/output pipeline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import cv2

from pipeline import FrameProcessor, FrameResult


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
VIDEO_OUTPUT_SUFFIXES = frozenset({".avi", ".m4v", ".mov", ".mp4"})
CaptureSource = Union[int, str]
FrameTransform = Callable[[object], object]
FrameCallback = Callable[[FrameResult, "MediaStats", float], bool]


class MediaError(RuntimeError):
    pass


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise MediaError(f"Unsupported media extension: {suffix or '<none>'}")


@dataclass
class MediaStats:
    frames_processed: int = 0
    face_observations: int = 0
    empty_frames: int = 0
    elapsed_seconds: float = 0.0
    tracking_enabled: bool = True
    _track_ids: set[int] = field(default_factory=set, repr=False)

    @property
    def unique_tracks(self) -> Optional[int]:
        """Return stable track count, or ``None`` when IDs are frame-local."""
        return len(self._track_ids) if self.tracking_enabled else None

    @property
    def unique_id_count(self) -> int:
        return len(self._track_ids)

    @property
    def track_ids(self) -> list:
        return sorted(self._track_ids)

    def observe(self, result: FrameResult) -> None:
        self.frames_processed += 1
        self.face_observations += result.face_count
        if result.face_count == 0:
            self.empty_frames += 1
        self._track_ids.update(track.id for track in result.tracks if track.id > 0)


def _video_codec(output_path: Path) -> int:
    if output_path.suffix.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter_fourcc(*"mp4v")


class MediaPipeline:
    """Run one :class:`FrameProcessor` over file or live capture sources."""

    def __init__(self, processor: FrameProcessor) -> None:
        self.processor = processor

    def process_image(
        self,
        input_path: Path,
        *,
        output_path: Optional[Path] = None,
        include_overlays: bool = False,
        frame_transform: Optional[FrameTransform] = None,
    ) -> tuple[FrameResult, MediaStats]:
        started = time.perf_counter()
        frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise MediaError(f"Could not read image: {input_path}")
        if frame_transform is not None:
            frame = frame_transform(frame)

        self.processor.reset()
        result = self.processor.process(frame, include_overlays=include_overlays)
        stats = MediaStats(tracking_enabled=self.processor.tracker is not None)
        stats.observe(result)
        stats.elapsed_seconds = time.perf_counter() - started

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), result.frame):
                raise MediaError(f"Could not write image: {output_path}")

        return result, stats

    def process_capture(
        self,
        source: CaptureSource,
        *,
        output_path: Optional[Path] = None,
        include_overlays: bool = False,
        frame_transform: Optional[FrameTransform] = None,
        on_frame: Optional[FrameCallback] = None,
        requested_size: Optional[tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ) -> MediaStats:
        started = time.perf_counter()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise MediaError(f"Could not open capture source: {source}")

        if requested_size is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_size[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_size[1])

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 24.0

        writer = None
        stats = MediaStats(tracking_enabled=self.processor.tracker is not None)
        self.processor.reset()

        try:
            while max_frames is None or stats.frames_processed < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_transform is not None:
                    frame = frame_transform(frame)

                result = self.processor.process(frame, include_overlays=include_overlays)
                if output_path is not None and writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = result.frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), _video_codec(output_path), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise MediaError(f"Could not open video writer: {output_path}")

                if writer is not None:
                    writer.write(result.frame)

                stats.observe(result)
                if on_frame is not None and not on_frame(result, stats, fps):
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        stats.elapsed_seconds = time.perf_counter() - started
        if stats.frames_processed == 0:
            raise MediaError(f"Capture source produced no frames: {source}")
        return stats
