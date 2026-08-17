import cv2
import numpy as np
import pytest

from detectors import Detection
from pipeline import FrameProcessor, RedactionConfig, expanded_bounds
from tracker import IoUTracker


class StaticDetector:
    name = "synthetic"

    def __init__(self, detections):
        self.detections = detections

    def detect(self, _frame):
        return list(self.detections)


def patterned_frame(width=80, height=60):
    x = np.arange(width, dtype=np.uint8)
    y = np.arange(height, dtype=np.uint8)[:, None]
    return np.dstack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.broadcast_to((x + y) % 255, (height, width)),
        )
    ).copy()


def test_expanded_bounds_add_padding_and_clamp_to_frame():
    assert expanded_bounds((20, 10, 20, 20), (60, 80, 3), 0.25) == (15, 5, 45, 35)
    assert expanded_bounds((-5, -5, 20, 20), (60, 80, 3), 0.5) == (0, 0, 25, 25)


def test_solid_redaction_is_padded_and_does_not_change_other_pixels():
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 20, 20, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0.25, solid_color=(7, 8, 9)),
    )

    result = processor.process(frame)

    assert np.all(result.frame[5:35, 15:45] == np.array([7, 8, 9], dtype=np.uint8))
    assert np.array_equal(result.frame[:5], frame[:5])
    assert np.array_equal(result.frame[:, :15], frame[:, :15])


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_visual_redaction_modes_change_the_face_region(mode):
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 30, 30, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode=mode, padding=0.1, pixel_size=8),
    )

    result = processor.process(frame)

    assert not np.array_equal(result.frame[7:43, 17:53], frame[7:43, 17:53])
    assert np.array_equal(result.frame[:5], frame[:5])


def test_redaction_can_only_be_disabled_explicitly():
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 20, 20, 0.95)])
    processor = FrameProcessor(detector, redaction=RedactionConfig(enabled=False))

    result = processor.process(frame)

    assert np.array_equal(result.frame, frame)


def test_saved_frame_has_no_overlays_unless_requested():
    frame = np.full((60, 80, 3), 200, dtype=np.uint8)
    detector = StaticDetector([Detection(20, 15, 20, 20, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
    )

    clean = processor.process(frame, include_overlays=False).frame
    annotated = processor.process(frame, include_overlays=True).frame

    assert np.array_equal(clean[:10], frame[:10])
    assert not np.array_equal(annotated, clean)


def test_frame_processor_exposes_new_track_immediately():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detector = StaticDetector([Detection(5, 5, 10, 10, 0.9)])
    processor = FrameProcessor(detector, tracker=IoUTracker())

    result = processor.process(frame)

    assert result.face_count == 1
    assert result.tracks[0].misses == 0


def test_recent_track_remains_redacted_through_a_detector_dropout():
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    detector = StaticDetector([Detection(10, 10, 10, 10, 0.9)])
    processor = FrameProcessor(
        detector,
        tracker=IoUTracker(),
        redaction=RedactionConfig(mode="solid", padding=0, hold_frames=1),
    )
    processor.process(frame)
    detector.detections = []

    held = processor.process(frame)
    exposed = processor.process(frame)

    assert held.face_count == 1
    assert np.all(held.frame[10:20, 10:20] == 0)
    assert exposed.face_count == 0
    assert np.all(exposed.frame[10:20, 10:20] == 200)
