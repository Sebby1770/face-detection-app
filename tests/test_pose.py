"""Unit tests for landmark-based head pose (no camera required)."""

from __future__ import annotations

from pose import YawLabel, estimate_yaw


def _landmarks(rx: float, lx: float, nx: float, y: float = 10.0) -> list:
    # right eye, left eye, nose, right mouth, left mouth
    return [(rx, y), (lx, y), (nx, y + 5), (rx, y + 15), (lx, y + 15)]


def test_estimate_yaw_center() -> None:
    # Eyes at 0 and 20, nose at midpoint 10
    pose = estimate_yaw(_landmarks(0, 20, 10))
    assert pose is not None
    assert pose.label is YawLabel.CENTER


def test_estimate_yaw_left() -> None:
    # Nose shifted left of eye midpoint
    pose = estimate_yaw(_landmarks(0, 20, 2))
    assert pose is not None
    assert pose.label is YawLabel.LEFT
    assert pose.yaw_ratio < 0


def test_estimate_yaw_right() -> None:
    pose = estimate_yaw(_landmarks(0, 20, 18))
    assert pose is not None
    assert pose.label is YawLabel.RIGHT
    assert pose.yaw_ratio > 0


def test_estimate_yaw_needs_landmarks() -> None:
    assert estimate_yaw([]) is None
    assert estimate_yaw([(0, 0), (1, 0)]) is None


def test_zero_interocular_returns_none() -> None:
    # Both eyes at same point → undefined scale
    assert estimate_yaw([(5, 5), (5, 5), (5, 8), (4, 10), (6, 10)]) is None
