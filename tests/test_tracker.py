"""Unit tests for the IoU tracker (no camera required)."""

from __future__ import annotations

from detectors import Detection
from tracker import IoUTracker, Track, _iou


def test_iou_identical_boxes() -> None:
    box = (10.0, 20.0, 100.0, 80.0)
    assert abs(_iou(box, box) - 1.0) < 1e-6


def test_iou_no_overlap() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 10.0, 10.0)
    assert _iou(a, b) == 0.0


def test_iou_partial_overlap() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 10.0, 10.0)
    # Intersection 5x10=50, union 100+100-50=150 → 1/3
    assert abs(_iou(a, b) - (50.0 / 150.0)) < 1e-6


def test_track_from_detection() -> None:
    det = Detection(10, 20, 30, 40, 0.9, [(11, 21), (25, 22), (20, 30), (12, 35), (24, 36)])
    track = Track.from_detection(det, track_id=7)
    assert track.id == 7
    assert track.bbox == (10.0, 20.0, 30.0, 40.0)
    assert track.score == 0.9
    assert len(track.landmarks) == 5
    assert track.bbox_int == (10, 20, 30, 40)


def test_tracker_assigns_stable_ids() -> None:
    tracker = IoUTracker(iou_threshold=0.3, max_misses=5, ema=0.5)
    d1 = Detection(100, 100, 50, 50, 0.95)
    tracks = tracker.update([d1])
    assert len(tracks) == 1
    first_id = tracks[0].id

    # Slightly moved box should match the same track.
    d2 = Detection(105, 102, 50, 50, 0.92)
    tracks = tracker.update([d2])
    assert len([t for t in tracks if t.misses == 0]) == 1
    assert tracks[0].id == first_id
    assert tracks[0].age >= 1


def test_tracker_new_face_gets_new_id() -> None:
    tracker = IoUTracker()
    tracker.update([Detection(0, 0, 40, 40, 0.9)])
    tracks = tracker.update(
        [
            Detection(0, 0, 40, 40, 0.9),
            Detection(200, 200, 40, 40, 0.8),
        ]
    )
    active = [t for t in tracks if t.misses == 0]
    assert len(active) == 2
    assert {t.id for t in active} == {1, 2}


def test_tracker_drops_after_max_misses() -> None:
    tracker = IoUTracker(max_misses=2)
    tracker.update([Detection(0, 0, 40, 40, 0.9)])
    tracker.update([])  # miss 1
    tracks = tracker.update([])  # miss 2 — still kept
    assert any(t.id == 1 for t in tracks)
    tracks = tracker.update([])  # miss 3 — dropped
    assert all(t.id != 1 for t in tracks)


def test_tracker_ema_smooths_bbox() -> None:
    tracker = IoUTracker(ema=0.5)
    tracker.update([Detection(0, 0, 100, 100, 1.0)])
    tracks = tracker.update([Detection(20, 0, 100, 100, 1.0)])
    x, y, w, h = tracks[0].bbox
    # EMA 0.5: x = 0.5*0 + 0.5*20 = 10
    assert abs(x - 10.0) < 1e-6
    assert abs(y - 0.0) < 1e-6
