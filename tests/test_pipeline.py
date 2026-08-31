import cv2
import numpy as np
import pytest

from detectors import Detection
from pipeline import (
    FrameProcessor,
    RedactionConfig,
    ellipse_bounds,
    expanded_bounds,
    landmark_bounds,
)
from tracker import IoUTracker, Track


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


def test_ellipse_bounds_stretch_and_clamp():
    assert ellipse_bounds(10, 10, 30, 30, (100, 100, 3)) == (9, 8, 31, 32)
    assert ellipse_bounds(0, 0, 10, 10, (12, 12, 3)) == (0, 0, 11, 11)
    assert ellipse_bounds(10, 10, 10, 20, (40, 40, 3)) == (10, 10, 10, 20)


def test_landmark_bounds_extends_above_padded_bbox():
    track = Track(
        id=1,
        bbox=(30.0, 25.0, 20.0, 20.0),
        score=0.9,
        landmarks=[(32.0, 10.0), (48.0, 10.0), (40.0, 28.0), (34.0, 38.0), (46.0, 38.0)],
    )
    x0, y0, x1, y1 = landmark_bounds(track, (60, 80, 3), 0.25)
    padded = expanded_bounds(track.bbox, (60, 80, 3), 0.25)

    assert y0 < padded[1]
    assert y0 == 0
    assert x0 <= padded[0]
    assert x1 >= padded[2]


def test_ellipse_with_landmarks_covers_pixel_above_bbox():
    frame = patterned_frame()
    landmarks = [(32, 10), (48, 10), (40, 28), (34, 38), (46, 38)]
    detector = StaticDetector(
        [Detection(30, 25, 20, 20, 0.95, landmarks=list(landmarks))]
    )
    box_processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(
            mode="solid", padding=0.25, solid_color=(7, 8, 9), shape="box"
        ),
    )
    ellipse_processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(
            mode="solid", padding=0.25, solid_color=(7, 8, 9), shape="ellipse"
        ),
    )

    box = box_processor.process(frame)
    ellipse = ellipse_processor.process(frame)
    solid = np.array([7, 8, 9], dtype=np.uint8)

    assert np.array_equal(box.frame[5, 40], frame[5, 40])
    assert np.array_equal(ellipse.frame[5, 40], solid)
    assert np.array_equal(ellipse.frame[-2, -2], frame[-2, -2])
    assert np.array_equal(ellipse.frame[:2, :2], frame[:2, :2])


def test_ellipse_redaction_changes_roi_and_leaves_far_pixels():
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 20, 20, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(
            mode="solid", padding=0.25, solid_color=(7, 8, 9), shape="ellipse"
        ),
    )

    result = processor.process(frame)
    solid = np.array([7, 8, 9], dtype=np.uint8)

    assert np.array_equal(result.frame[20, 30], solid)
    assert np.array_equal(result.frame[:2, :2], frame[:2, :2])
    assert np.array_equal(result.frame[-2:, -2:], frame[-2:, -2:])
    assert not np.array_equal(result.frame[5, 15], solid)


@pytest.mark.parametrize("shape", ["box", "ellipse"])
@pytest.mark.parametrize("feather", [0, 8])
def test_feather_does_not_crash(shape, feather):
    frame = patterned_frame()
    processor = FrameProcessor(
        StaticDetector([Detection(20, 10, 20, 20, 0.95)]),
        redaction=RedactionConfig(shape=shape, feather=feather),
    )

    result = processor.process(frame)

    assert result.frame.shape == frame.shape
    assert result.frame.dtype == frame.dtype


def test_keep_ids_leaves_that_track_unredacted():
    frame = patterned_frame()
    detector = StaticDetector(
        [Detection(10, 10, 12, 12, 0.9), Detection(50, 10, 12, 12, 0.9)]
    )
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
        keep_ids={1},
    )

    result = processor.process(frame)

    assert np.array_equal(result.frame[10:22, 10:22], frame[10:22, 10:22])
    assert np.all(result.frame[10:22, 50:62] == 0)


def test_redact_ids_only_redacts_listed_tracks():
    frame = patterned_frame()
    detector = StaticDetector(
        [Detection(10, 10, 12, 12, 0.9), Detection(50, 10, 12, 12, 0.9)]
    )
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
    )

    result = processor.process(frame, redact_ids={2})

    assert np.array_equal(result.frame[10:22, 10:22], frame[10:22, 10:22])
    assert np.all(result.frame[10:22, 50:62] == 0)


def test_keep_ids_wins_over_redact_ids():
    frame = patterned_frame()
    detector = StaticDetector([Detection(10, 10, 12, 12, 0.9)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
        redact_ids={1},
        keep_ids={1},
    )

    result = processor.process(frame)

    assert np.array_equal(result.frame[10:22, 10:22], frame[10:22, 10:22])


def test_invalid_mask_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        RedactionConfig(shape="star")


def test_negative_feather_is_rejected():
    with pytest.raises(ValueError, match="feather"):
        RedactionConfig(feather=-1)


def test_min_size_drops_small_detections_so_they_are_not_redacted():
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 10, 10, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
        min_size=30,
    )

    result = processor.process(frame)

    assert result.face_count == 0
    assert result.tracks == []
    assert np.array_equal(result.frame, frame)


def test_redaction_config_min_size_drops_small_detections():
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 10, 10, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(
            mode="solid", padding=0, solid_color=(0, 0, 0), min_size=30
        ),
    )

    result = processor.process(frame)

    assert result.face_count == 0
    assert np.array_equal(result.frame, frame)


def test_min_size_keeps_detections_meeting_the_threshold():
    frame = patterned_frame()
    detector = StaticDetector([Detection(20, 10, 30, 30, 0.95)])
    processor = FrameProcessor(
        detector,
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
        min_size=30,
    )

    result = processor.process(frame)

    assert result.face_count == 1
    assert np.all(result.frame[10:40, 20:50] == 0)


def test_negative_min_size_is_rejected():
    with pytest.raises(ValueError, match="min_size"):
        FrameProcessor(
            StaticDetector([]),
            redaction=RedactionConfig(),
            min_size=-1,
        )
    with pytest.raises(ValueError, match="min_size"):
        RedactionConfig(min_size=-1)


def test_process_accepts_extra_kwargs():
    frame = patterned_frame()
    processor = FrameProcessor(
        StaticDetector([Detection(20, 10, 20, 20, 0.95)]),
        redaction=RedactionConfig(mode="solid", padding=0),
    )

    result = processor.process(frame, record_analytics=False)

    assert result.face_count == 1
