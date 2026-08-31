from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from analytics import SessionAnalytics
from detectors import Detection
from face_detection import HELP_LINES, PreviewSession, bind_session_hooks
from media import MediaStats
from pipeline import FrameProcessor, FrameResult, OverlayConfig, RedactionConfig
from tracker import Track


class RecordingDetector:
    name = "synthetic"

    def __init__(self):
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame.copy())
        return [Detection(8, 8, 12, 12, 0.95)]


def preview_args(tmp_path: Path, *, mirror=False, overlays=False, review=False, pose=None):
    return SimpleNamespace(
        mirror=mirror,
        overlays=overlays,
        review=review,
        pose=pose,
        snapshot_dir=tmp_path / "snapshots",
        recording_dir=tmp_path / "recordings",
    )


def empty_result():
    return FrameResult(np.zeros((32, 32, 3), dtype=np.uint8), [], "synthetic")


def test_pixelate_toggle_has_safe_fallback_when_preview_starts_pixelated(tmp_path: Path):
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(mode="pixelate")
    )
    session = PreviewSession(preview_args(tmp_path), processor)

    assert session._handle_key(ord("p"), empty_result(), 24.0)
    assert processor.redaction.mode == "solid"

    assert session._handle_key(ord("p"), empty_result(), 24.0)
    assert processor.redaction.mode == "pixelate"


def test_e_toggles_mask_shape_between_box_and_ellipse(tmp_path: Path):
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(shape="box")
    )
    session = PreviewSession(preview_args(tmp_path), processor)

    assert session._handle_key(ord("e"), empty_result(), 24.0)
    assert processor.redaction.shape == "ellipse"

    assert session._handle_key(ord("e"), empty_result(), 24.0)
    assert processor.redaction.shape == "box"


def test_help_lines_include_ellipse_and_pixelate_keys():
    joined = "\n".join(HELP_LINES)
    assert "p        toggle pixelate" in joined
    assert "e        toggle box / ellipse" in joined
    assert "f        toggle feather 0 / 8" in joined


def test_f_toggles_feather_between_zero_and_eight(tmp_path: Path):
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(feather=0)
    )
    session = PreviewSession(preview_args(tmp_path), processor)

    assert session._handle_key(ord("f"), empty_result(), 24.0)
    assert processor.redaction.feather == 8

    assert session._handle_key(ord("f"), empty_result(), 24.0)
    assert processor.redaction.feather == 0


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


def test_still_preview_reprocesses_ellipse_and_pixelate_keys(
    tmp_path: Path, monkeypatch
):
    detector = RecordingDetector()
    processor = FrameProcessor(detector, redaction=RedactionConfig(mode="solid"))
    session = PreviewSession(preview_args(tmp_path), processor)
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    keys = iter((ord("e"), ord("p"), ord("q")))
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))

    session.show_still(source, MediaStats(tracking_enabled=False))

    assert processor.redaction.shape == "ellipse"
    assert processor.redaction.mode == "pixelate"
    assert len(detector.frames) == 3


def test_still_preview_reprocesses_feather_key(tmp_path: Path, monkeypatch):
    detector = RecordingDetector()
    processor = FrameProcessor(detector, redaction=RedactionConfig(feather=0))
    session = PreviewSession(preview_args(tmp_path), processor)
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    keys = iter((ord("f"), ord("q")))
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))

    session.show_still(source, MediaStats(tracking_enabled=False))

    assert processor.redaction.feather == 8
    assert len(detector.frames) == 2


def test_empty_frame_hud_mentions_no_face(tmp_path: Path, monkeypatch):
    seen = []

    def capture(_frame, lines, x=10, y=28, color=(255, 255, 255)):
        seen.append(list(lines))

    monkeypatch.setattr("face_detection.overlay_text", capture)
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(enabled=False)
    )
    session = PreviewSession(preview_args(tmp_path), processor)

    session._decorate(empty_result(), MediaStats(tracking_enabled=False))

    assert any(line == "NO FACE this frame" for group in seen for line in group)


def test_hud_omits_no_face_when_tracks_exist(tmp_path: Path, monkeypatch):
    seen = []

    def capture(_frame, lines, x=10, y=28, color=(255, 255, 255)):
        seen.append(list(lines))

    monkeypatch.setattr("face_detection.overlay_text", capture)
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(enabled=False)
    )
    session = PreviewSession(preview_args(tmp_path), processor)
    track = Track(id=1, bbox=(8, 8, 12, 12), score=0.9)
    result = FrameResult(np.zeros((32, 32, 3), dtype=np.uint8), [track], "synthetic")

    session._decorate(result, MediaStats(tracking_enabled=False))

    assert all("NO FACE this frame" not in line for group in seen for line in group)


def test_hud_lists_sorted_track_ids(tmp_path: Path, monkeypatch):
    seen = []

    def capture(_frame, lines, x=10, y=28, color=(255, 255, 255)):
        seen.append(list(lines))

    monkeypatch.setattr("face_detection.overlay_text", capture)
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(enabled=False)
    )
    session = PreviewSession(preview_args(tmp_path), processor)
    tracks = [
        Track(id=5, bbox=(8, 8, 12, 12), score=0.9),
        Track(id=1, bbox=(8, 8, 12, 12), score=0.9),
        Track(id=2, bbox=(8, 8, 12, 12), score=0.9),
    ]
    result = FrameResult(np.zeros((32, 32, 3), dtype=np.uint8), tracks, "synthetic")

    session._decorate(result, MediaStats(tracking_enabled=False))

    assert any(line == "IDs: 1,2,5" for group in seen for line in group)


def test_hud_shows_hold_progress_for_missed_tracks(tmp_path: Path, monkeypatch):
    seen = []

    def capture(_frame, lines, x=10, y=28, color=(255, 255, 255)):
        seen.append(list(lines))

    monkeypatch.setattr("face_detection.overlay_text", capture)
    processor = FrameProcessor(
        RecordingDetector(),
        redaction=RedactionConfig(enabled=False, hold_frames=2),
    )
    session = PreviewSession(preview_args(tmp_path), processor)
    track = Track(id=3, bbox=(8, 8, 12, 12), score=0.9, misses=1)
    result = FrameResult(np.zeros((32, 32, 3), dtype=np.uint8), [track], "synthetic")

    session._decorate(result, MediaStats(tracking_enabled=False))

    assert any(line == "IDs: 3" for group in seen for line in group)
    assert any(line == "hold #3 1/2" for group in seen for line in group)


def test_still_preview_does_not_double_count_analytics(tmp_path: Path, monkeypatch):
    detector = RecordingDetector()
    processor = FrameProcessor(detector, redaction=RedactionConfig(mode="solid"))
    analytics = SessionAnalytics()
    bind_session_hooks(processor, analytics, draw_pose_on_output=False)
    session = PreviewSession(preview_args(tmp_path), processor, analytics=analytics)
    source = np.zeros((32, 40, 3), dtype=np.uint8)

    result = processor.process(source, include_overlays=False)
    assert analytics.total_frames == 1
    assert len(detector.frames) == 1

    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: ord("q"))

    session.show_still(source, MediaStats(tracking_enabled=False), initial_result=result)

    assert analytics.total_frames == 1
    assert len(detector.frames) == 1


def test_still_preview_reprocess_skips_analytics(tmp_path: Path, monkeypatch):
    detector = RecordingDetector()
    processor = FrameProcessor(detector, redaction=RedactionConfig(mode="solid"))
    analytics = SessionAnalytics()
    bind_session_hooks(processor, analytics, draw_pose_on_output=False)
    session = PreviewSession(preview_args(tmp_path), processor, analytics=analytics)
    source = np.zeros((32, 40, 3), dtype=np.uint8)
    result = processor.process(source, include_overlays=False)
    keys = iter((ord("b"), ord("q")))
    monkeypatch.setattr(cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))

    session.show_still(source, MediaStats(tracking_enabled=False), initial_result=result)

    assert analytics.total_frames == 1
    assert len(detector.frames) == 2


def test_bind_session_hooks_honors_record_analytics_false():
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(mode="solid")
    )
    analytics = SessionAnalytics()
    bind_session_hooks(processor, analytics, draw_pose_on_output=False)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    processor.process(frame, record_analytics=False)
    assert analytics.total_frames == 0

    processor.process(frame)
    assert analytics.total_frames == 1


def test_review_hud_draws_pose_on_display_copy_only(tmp_path: Path, monkeypatch):
    drawn_ids = []

    def fake_pose(frame, tracks):
        drawn_ids.append(id(frame))
        frame[0, 0] = (1, 2, 3)

    monkeypatch.setattr("face_detection.try_draw_pose", fake_pose)
    processor = FrameProcessor(
        RecordingDetector(),
        redaction=RedactionConfig(enabled=False),
        overlays=OverlayConfig(show_boxes=False, show_landmarks=False, show_ids=False),
    )
    session = PreviewSession(preview_args(tmp_path, review=True), processor)
    original = np.full((48, 48, 3), 200, dtype=np.uint8)
    landmarks = [(16, 16), (32, 16), (24, 22), (18, 30), (30, 30)]
    track = Track(id=1, bbox=(10, 10, 28, 28), score=0.9, landmarks=landmarks)
    result = FrameResult(original.copy(), [track], "synthetic")

    decorated = session._decorate(result, MediaStats(tracking_enabled=False))

    assert drawn_ids
    assert drawn_ids[0] != id(result.frame)
    assert np.array_equal(result.frame, original)
    assert not np.array_equal(decorated, original)


def test_overlays_preview_does_not_redraw_pose_in_decorate(tmp_path: Path, monkeypatch):
    called = []
    monkeypatch.setattr(
        "face_detection.try_draw_pose", lambda *_args, **_kwargs: called.append(True)
    )
    processor = FrameProcessor(
        RecordingDetector(), redaction=RedactionConfig(enabled=False)
    )
    session = PreviewSession(preview_args(tmp_path, review=True, overlays=True), processor)
    result = FrameResult(np.full((32, 32, 3), 180, dtype=np.uint8), [], "synthetic")

    session._decorate(result, MediaStats(tracking_enabled=False))

    assert called == []
