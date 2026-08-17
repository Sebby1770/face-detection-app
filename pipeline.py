"""Reusable face detection, tracking, annotation, and privacy redaction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import cv2

from detectors import Detection
from tracker import IoUTracker, Track


Color = Tuple[int, int, int]
REDACTION_MODES = ("blur", "pixelate", "solid")


class FaceDetector(Protocol):
    """The small detector interface consumed by :class:`FrameProcessor`."""

    name: str

    def detect(self, frame_bgr) -> Sequence[Detection]:
        ...


@dataclass
class RedactionConfig:
    """Privacy transformation settings.

    Solid fill and generous padding are deliberately the defaults. Detection
    can still miss a face, so these settings reduce exposure but cannot
    guarantee anonymisation.
    """

    enabled: bool = True
    mode: str = "solid"
    padding: float = 0.25
    hold_frames: int = 2
    pixel_size: int = 14
    solid_color: Color = (0, 0, 0)

    def __post_init__(self) -> None:
        if self.mode not in REDACTION_MODES:
            raise ValueError(f"Unsupported redaction mode: {self.mode}")
        if not 0.0 <= self.padding <= 1.0:
            raise ValueError("padding must be between 0 and 1")
        if self.hold_frames < 0:
            raise ValueError("hold_frames cannot be negative")
        if self.pixel_size < 2:
            raise ValueError("pixel_size must be at least 2")
        if len(self.solid_color) != 3 or any(not 0 <= channel <= 255 for channel in self.solid_color):
            raise ValueError("solid_color must contain three values between 0 and 255")


@dataclass
class OverlayConfig:
    show_boxes: bool = True
    show_landmarks: bool = True
    show_ids: bool = True


@dataclass
class FrameResult:
    frame: object
    tracks: List[Track]
    detector_name: str

    @property
    def face_count(self) -> int:
        return len(self.tracks)


def color_for_score(score: float) -> Color:
    if score >= 0.85:
        return (0, 255, 0)
    if score >= 0.65:
        return (0, 255, 255)
    return (0, 165, 255)


def expanded_bounds(
    bbox: Tuple[float, float, float, float],
    frame_shape: Sequence[int],
    padding: float,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box and clamp it to the frame as ``x0,y0,x1,y1``."""
    x, y, width, height = bbox
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    pad_x, pad_y = width * padding, height * padding
    x0 = max(0, math.floor(x - pad_x))
    y0 = max(0, math.floor(y - pad_y))
    x1 = min(frame_width, math.ceil(x + width + pad_x))
    y1 = min(frame_height, math.ceil(y + height + pad_y))
    return x0, y0, x1, y1


def redact_track(frame, track: Track, config: RedactionConfig) -> None:
    """Apply the configured redaction in place for one tracked face."""
    x0, y0, x1, y1 = expanded_bounds(track.bbox, frame.shape, config.padding)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    if config.mode == "solid":
        roi[...] = config.solid_color
        return

    if config.mode == "pixelate":
        width, height = x1 - x0, y1 - y0
        reduced_width = max(1, math.ceil(width / config.pixel_size))
        reduced_height = max(1, math.ceil(height / config.pixel_size))
        reduced = cv2.resize(roi, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
        roi[...] = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
        return

    sigma = max(8.0, max(x1 - x0, y1 - y0) / 5.0)
    roi[...] = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)


def draw_tracks(frame, tracks: Sequence[Track], config: OverlayConfig) -> None:
    """Draw optional diagnostics on an already-redacted frame."""
    for track in tracks:
        x, y, width, height = track.bbox_int
        color = color_for_score(track.score)

        if config.show_boxes:
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

        label_parts = []
        if config.show_ids:
            label_parts.append(f"#{track.id}")
        label_parts.append(f"{track.score * 100:.0f}%")
        label = " ".join(label_parts)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y - text_height - baseline - 4)
        label_bottom = min(frame.shape[0], label_top + text_height + baseline + 4)
        cv2.rectangle(
            frame,
            (max(0, x), label_top),
            (min(frame.shape[1], x + text_width + 8), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (max(0, x + 4), max(text_height, label_bottom - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if config.show_landmarks:
            for landmark_x, landmark_y in track.landmarks:
                point = (int(landmark_x), int(landmark_y))
                cv2.circle(frame, point, 2, (255, 255, 255), -1)
                cv2.circle(frame, point, 4, color, 1)


class FrameProcessor:
    """Detect, track, redact, and optionally annotate individual frames."""

    def __init__(
        self,
        detector: FaceDetector,
        *,
        tracker: IoUTracker | None = None,
        redaction: RedactionConfig | None = None,
        overlays: OverlayConfig | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.redaction = redaction or RedactionConfig()
        self.overlays = overlays or OverlayConfig()

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def process(self, frame, *, include_overlays: bool = False) -> FrameResult:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame must be a non-empty image")

        detections = list(self.detector.detect(frame))
        if self.tracker is None:
            tracks = [
                Track.from_detection(detection, track_id=index)
                for index, detection in enumerate(detections, start=1)
            ]
        else:
            # Continue redacting a recently observed region through brief
            # detector dropouts. This is intentionally conservative: one
            # missed frame should not immediately expose a face.
            tracks = [
                track
                for track in self.tracker.update(detections)
                if track.misses <= self.redaction.hold_frames
            ]

        output = frame.copy()
        if self.redaction.enabled:
            for track in tracks:
                redact_track(output, track, self.redaction)

        if include_overlays:
            draw_tracks(output, tracks, self.overlays)

        return FrameResult(output, tracks, self.detector.name)
