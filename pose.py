"""Rough head-pose estimates from YuNet 5-point facial landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import cv2

# YuNet landmark order: right eye, left eye, nose tip, right mouth, left mouth
RIGHT_EYE, LEFT_EYE, NOSE = 0, 1, 2


class YawLabel(str, Enum):
    LEFT = "L"
    CENTER = "C"
    RIGHT = "R"


@dataclass(frozen=True)
class HeadPose:
    """Coarse yaw estimate derived from eye midpoints vs nose."""

    yaw_ratio: float
    label: YawLabel

    @property
    def short_label(self) -> str:
        return self.label.value


def estimate_yaw(
    landmarks: Sequence[Tuple[float, float]],
    left_threshold: float = -0.25,
    right_threshold: float = 0.25,
) -> Optional[HeadPose]:
    """
    Estimate horizontal looking direction from 5 facial landmarks.

    Uses the horizontal offset of the nose tip relative to the eye midpoint,
    normalized by inter-ocular distance. Positive ratio → looking right
    (from the camera's view of the face); negative → looking left.
    """
    if len(landmarks) < 3:
        return None

    rx, ry = landmarks[RIGHT_EYE]
    lx, ly = landmarks[LEFT_EYE]
    nx, ny = landmarks[NOSE]

    eye_mid_x = (rx + lx) * 0.5
    interocular = math.hypot(lx - rx, ly - ry)
    if interocular < 1e-3:
        return None

    yaw_ratio = (nx - eye_mid_x) / interocular

    if yaw_ratio <= left_threshold:
        label = YawLabel.LEFT
    elif yaw_ratio >= right_threshold:
        label = YawLabel.RIGHT
    else:
        label = YawLabel.CENTER
    return HeadPose(yaw_ratio=yaw_ratio, label=label)


def draw_pose_indicator(
    frame,
    bbox: Tuple[int, int, int, int],
    pose: HeadPose,
    color: Tuple[int, int, int] = (255, 200, 50),
) -> None:
    """Draw a small L/R/C label and directional arrow near the face box."""
    x, y, w, h = bbox
    cx = x + w // 2
    top = max(16, y - 8)

    label = pose.short_label
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_x = cx - tw // 2
    label_y = top
    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Arrow below the label pointing left / right, or a small bar for center.
    ay = label_y + 12
    length = max(10, min(w // 4, 24))
    if pose.label == YawLabel.LEFT:
        cv2.arrowedLine(frame, (cx + length // 2, ay), (cx - length // 2, ay), color, 2, tipLength=0.4)
    elif pose.label == YawLabel.RIGHT:
        cv2.arrowedLine(frame, (cx - length // 2, ay), (cx + length // 2, ay), color, 2, tipLength=0.4)
    else:
        cv2.line(frame, (cx - length // 3, ay), (cx + length // 3, ay), color, 2)
