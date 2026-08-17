from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from detectors import Detection
from face_detection import PreviewSession
from media import MediaStats
from pipeline import FrameProcessor, FrameResult, RedactionConfig


class RecordingDetector:
    name = "synthetic"

    def __init__(self):
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame.copy())
        return [Detection(8, 8, 12, 12, 0.95)]


def preview_args(tmp_path: Path, *, mirror=False, overlays=False):
    return SimpleNamespace(
        mirror=mirror,
        overlays=overlays,
        snapshot_dir=tmp_path / "snapshots",
        recording_dir=tmp_path / "recordings",
    )


def empty_result():
    return FrameResult(np.zeros((32, 32, 3), dtype=np.uint8), [], "synthetic")


def test_blur_toggle_has_safe_fallback_when_preview_starts_blurred(tmp_path: Path):
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(mode="blur")
    )
    session = PreviewSession(preview_args(tmp_path), processor)

    assert session._handle_key(ord("b"), empty_result(), 24.0)
    assert processor.redaction.mode == "solid"

    assert session._handle_key(ord("b"), empty_result(), 24.0)
    assert processor.redaction.mode == "blur"


def test_still_preview_reprocesses_blur_overlay_and_mirror_controls(
    tmp_path: Path, monkeypatch
):
    detector = RecordingDetector()
    processor = FrameProcessor(detector, redaction=RedactionConfig(mode="blur"))
    session = PreviewSession(preview_args(tmp_path), processor)
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    source[:, :8] = (10, 20, 230)
    shown_frames = []
    keys = iter((ord("b"), ord("l"), ord("i"), ord("m"), ord("q")))
    monkeypatch.setattr(cv2, "imshow", lambda _name, frame: shown_frames.append(frame.copy()))
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))

    session.show_still(source, MediaStats(tracking_enabled=False))

    assert processor.redaction.mode == "solid"
    assert not processor.overlays.show_landmarks
    assert not processor.overlays.show_ids
    assert session.mirror
    assert len(detector.frames) == 5
    assert np.array_equal(detector.frames[-1], cv2.flip(source, 1))
    assert len(shown_frames) == 5
