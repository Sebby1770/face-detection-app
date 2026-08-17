from pathlib import Path

import cv2
import numpy as np

import media
from detectors import Detection
from media import MediaPipeline, media_kind
from pipeline import FrameProcessor, RedactionConfig


class StaticDetector:
    name = "synthetic"

    def detect(self, _frame):
        return [Detection(16, 12, 16, 16, 0.95)]


def processor():
    return FrameProcessor(
        StaticDetector(),
        redaction=RedactionConfig(mode="solid", padding=0, solid_color=(0, 0, 0)),
    )


def test_image_source_to_output_pipeline(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "redacted.png"
    frame = np.full((48, 64, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(input_path), frame)

    result, stats = MediaPipeline(processor()).process_image(
        input_path, output_path=output_path
    )

    written = cv2.imread(str(output_path))
    assert media_kind(input_path) == "image"
    assert stats.frames_processed == 1
    assert stats.face_observations == 1
    assert stats.unique_tracks is None
    assert result.face_count == 1
    assert written is not None
    assert np.all(written[12:28, 16:32] == 0)
    assert np.all(written[:8] == 180)


def test_video_source_to_output_pipeline(tmp_path: Path):
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "redacted.avi"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48)
    )
    assert writer.isOpened(), "MJPG codec is required for the deterministic integration test"
    for index in range(4):
        writer.write(np.full((48, 64, 3), 100 + index * 20, dtype=np.uint8))
    writer.release()

    stats = MediaPipeline(processor()).process_capture(
        str(input_path), output_path=output_path
    )

    capture = cv2.VideoCapture(str(output_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    assert media_kind(input_path) == "video"
    assert stats.frames_processed == 4
    assert stats.face_observations == 4
    assert stats.unique_tracks is None
    assert len(frames) == 4
    assert float(np.mean(frames[0][14:26, 18:30])) < 8.0
    assert float(np.mean(frames[0][:8])) > 90.0


def test_camera_source_uses_index_and_requested_dimensions(monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.frames = [np.full((48, 64, 3), 150, dtype=np.uint8)]
            self.settings = []
            self.released = False

        def isOpened(self):
            return True

        def set(self, property_id, value):
            self.settings.append((property_id, value))
            return True

        def get(self, property_id):
            return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            self.released = True

    capture = FakeCapture()
    opened_sources = []

    def open_capture(source):
        opened_sources.append(source)
        return capture

    monkeypatch.setattr(media.cv2, "VideoCapture", open_capture)

    stats = MediaPipeline(processor()).process_capture(
        2, requested_size=(320, 240), max_frames=1
    )

    assert opened_sources == [2]
    assert capture.settings == [
        (cv2.CAP_PROP_FRAME_WIDTH, 320),
        (cv2.CAP_PROP_FRAME_HEIGHT, 240),
    ]
    assert capture.released
    assert stats.frames_processed == 1
