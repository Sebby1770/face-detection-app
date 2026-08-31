"""Reusable face detection, tracking, annotation, and privacy redaction."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from detectors import Detection
from tracker import IoUTracker, Track


Color = Tuple[int, int, int]
REDACTION_MODES = ("blur", "pixelate", "solid")
MASK_SHAPES = ("box", "ellipse")


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
    shape: str = "box"
    feather: int = 0
    min_size: int = 0

    def __post_init__(self) -> None:
        if self.mode not in REDACTION_MODES:
            raise ValueError(f"Unsupported redaction mode: {self.mode}")
        if self.shape not in MASK_SHAPES:
            raise ValueError(f"Unsupported mask shape: {self.shape}")
        if not 0.0 <= self.padding <= 1.0:
            raise ValueError("padding must be between 0 and 1")
        if self.hold_frames < 0:
            raise ValueError("hold_frames cannot be negative")
        if self.feather < 0:
            raise ValueError("feather cannot be negative")
        if self.min_size < 0:
            raise ValueError("min_size cannot be negative")
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


def _as_id_set(ids: Optional[Collection[int]]) -> Optional[set[int]]:
    if ids is None:
        return None
    return {int(value) for value in ids}


def should_redact_id(
    track_id: int,
    redact_ids: Optional[Collection[int]],
    keep_ids: Optional[Collection[int]],
) -> bool:
    """Return whether ``track_id`` should be redacted.

    ``keep_ids`` wins when an id appears in both collections.
    """
    if keep_ids is not None and int(track_id) in keep_ids:
        return False
    if redact_ids is not None:
        return int(track_id) in redact_ids
    return True


def ellipse_bounds(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_shape: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Stretch a padded box 1.20× vertically and 1.08× horizontally (clamped)."""
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return x0, y0, x1, y1
    new_width = width * 1.08
    new_height = height * 1.20
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    x0_new = max(0, math.floor(center_x - new_width / 2.0))
    x1_new = min(frame_width, math.ceil(center_x + new_width / 2.0))
    y0_new = max(0, math.floor(center_y - new_height / 2.0))
    y1_new = min(frame_height, math.ceil(center_y + new_height / 2.0))
    if x1_new <= x0_new or y1_new <= y0_new:
        return x0, y0, x1, y1
    return x0_new, y0_new, x1_new, y1_new


def landmark_bounds(
    track: Track,
    frame_shape: Sequence[int],
    padding: float,
) -> Tuple[int, int, int, int]:
    """Expand a padded bbox using YuNet landmarks (forehead / chin / cheeks).

    YuNet order: right eye, left eye, nose, right mouth, left mouth. Forehead
    is 0.55× bbox height above the eyes; chin is 0.35× below the mouth. Falls
    back to :func:`expanded_bounds` when fewer than 5 landmarks are present.
    """
    x0, y0, x1, y1 = expanded_bounds(track.bbox, frame_shape, padding)
    landmarks = track.landmarks
    if len(landmarks) < 5:
        return x0, y0, x1, y1

    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    _x, _y, width, height = track.bbox
    eye_y = min(landmarks[0][1], landmarks[1][1])
    mouth_y = max(landmarks[3][1], landmarks[4][1])
    forehead_y = eye_y - 0.55 * height
    chin_y = mouth_y + 0.35 * height
    xs = [point[0] for point in landmarks]
    side = 0.25 * width
    lx0 = min(xs) - side
    lx1 = max(xs) + side

    x0 = max(0, min(x0, math.floor(lx0)))
    y0 = max(0, min(y0, math.floor(forehead_y)))
    x1 = min(frame_width, max(x1, math.ceil(lx1)))
    y1 = min(frame_height, max(y1, math.ceil(chin_y)))
    if x1 <= x0 or y1 <= y0:
        return expanded_bounds(track.bbox, frame_shape, padding)
    return x0, y0, x1, y1


def _fill_roi(roi, config: RedactionConfig) -> None:
    """Apply a rectangular redaction in place (box, feather 0)."""
    if config.mode == "solid":
        roi[...] = config.solid_color
        return

    if config.mode == "pixelate":
        height, width = roi.shape[:2]
        reduced_width = max(1, math.ceil(width / config.pixel_size))
        reduced_height = max(1, math.ceil(height / config.pixel_size))
        reduced = cv2.resize(roi, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
        roi[...] = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
        return

    sigma = max(8.0, max(roi.shape[0], roi.shape[1]) / 5.0)
    roi[...] = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _render_redacted(roi, config: RedactionConfig):
    redacted = roi.copy()
    _fill_roi(redacted, config)
    return redacted


def _redact_with_mask(frame, x0: int, y0: int, x1: int, y1: int, config: RedactionConfig) -> None:
    feather = config.feather
    frame_height, frame_width = frame.shape[:2]
    mx0 = max(0, x0 - feather)
    my0 = max(0, y0 - feather)
    mx1 = min(frame_width, x1 + feather)
    my1 = min(frame_height, y1 + feather)
    if mx1 <= mx0 or my1 <= my0:
        return

    roi = frame[my0:my1, mx0:mx1]
    redacted = _render_redacted(roi, config)

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    lx0, ly0 = x0 - mx0, y0 - my0
    lx1, ly1 = x1 - mx0, y1 - my0
    if config.shape == "box":
        mask[ly0:ly1, lx0:lx1] = 255
    else:
        axis_x = max(1, int(round((lx1 - lx0) / 2.0)))
        axis_y = max(1, int(round((ly1 - ly0) / 2.0)))
        center = (
            int(round((lx0 + lx1) / 2.0)),
            int(round((ly0 + ly1) / 2.0)),
        )
        cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 255, -1)

    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=feather, sigmaY=feather)

    alpha = mask.astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]
    blended = roi.astype(np.float32) * (1.0 - alpha) + redacted.astype(np.float32) * alpha
    frame[my0:my1, mx0:mx1] = np.clip(blended, 0, 255).astype(np.uint8)


def redact_track(frame, track: Track, config: RedactionConfig) -> None:
    """Apply the configured redaction in place for one tracked face."""
    if config.shape == "ellipse" and len(track.landmarks) >= 5:
        x0, y0, x1, y1 = landmark_bounds(track, frame.shape, config.padding)
    else:
        x0, y0, x1, y1 = expanded_bounds(track.bbox, frame.shape, config.padding)
        if config.shape == "ellipse":
            x0, y0, x1, y1 = ellipse_bounds(x0, y0, x1, y1, frame.shape)
    if x1 <= x0 or y1 <= y0:
        return

    if config.shape == "box" and config.feather == 0:
        _fill_roi(frame[y0:y1, x0:x1], config)
        return

    _redact_with_mask(frame, x0, y0, x1, y1, config)


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
        redact_ids: Optional[Collection[int]] = None,
        keep_ids: Optional[Collection[int]] = None,
        min_size: int = 0,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.redaction = redaction or RedactionConfig()
        self.overlays = overlays or OverlayConfig()
        self.redact_ids = _as_id_set(redact_ids)
        self.keep_ids = _as_id_set(keep_ids)
        if min_size < 0:
            raise ValueError("min_size cannot be negative")
        self.min_size = int(min_size)

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def process(
        self,
        frame,
        *,
        include_overlays: bool = False,
        redact_ids: Optional[Collection[int]] = None,
        keep_ids: Optional[Collection[int]] = None,
        **kwargs,
    ) -> FrameResult:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame must be a non-empty image")

        detections = list(self.detector.detect(frame))
        min_size = self.min_size if self.min_size > 0 else self.redaction.min_size
        if min_size > 0:
            detections = [
                detection
                for detection in detections
                if detection.w >= min_size and detection.h >= min_size
            ]
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

        active_redact = self.redact_ids if redact_ids is None else _as_id_set(redact_ids)
        active_keep = self.keep_ids if keep_ids is None else _as_id_set(keep_ids)

        output = frame.copy()
        if self.redaction.enabled:
            for track in tracks:
                if should_redact_id(track.id, active_redact, active_keep):
                    redact_track(output, track, self.redaction)

        if include_overlays:
            draw_tracks(output, tracks, self.overlays)

        return FrameResult(output, tracks, self.detector.name)
