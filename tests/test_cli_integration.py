import json
from pathlib import Path

import cv2
import numpy as np

import face_detection
from detectors import Detection


class SyntheticDetector:
    name = "synthetic"

    def detect(self, _frame):
        return [Detection(16, 12, 16, 16, 0.99)]


def test_headless_image_cli_processes_and_writes_clean_output(
    tmp_path: Path, monkeypatch, capsys
):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    assert cv2.imwrite(str(input_path), np.full((48, 64, 3), 180, dtype=np.uint8))
    monkeypatch.setattr(face_detection, "build_detector", lambda _args: SyntheticDetector())

    exit_code = face_detection.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--headless",
            "--padding",
            "0",
            "--no-tracker",
        ]
    )

    output = cv2.imread(str(output_path))
    assert exit_code == 0
    assert output is not None
    assert np.all(output[12:28, 16:32] == 0)
    assert np.all(output[:8] == 180)
    assert "tracking disabled" in capsys.readouterr().out


def test_headless_video_cli_honors_frame_limit(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "output.avi"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48)
    )
    assert writer.isOpened()
    for value in (80, 100, 120, 140):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(face_detection, "build_detector", lambda _args: SyntheticDetector())

    exit_code = face_detection.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--headless",
            "--padding",
            "0",
            "--max-frames",
            "2",
        ]
    )

    capture = cv2.VideoCapture(str(output_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    assert exit_code == 0
    assert len(frames) == 2
    assert float(np.mean(frames[0][14:26, 18:30])) < 8.0


def test_headless_batch_cli_writes_mirrored_images_and_coverage(
    tmp_path: Path, monkeypatch
):
    input_dir = tmp_path / "inbox"
    output_dir = tmp_path / "redacted"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    assert cv2.imwrite(str(input_dir / "a.png"), np.full((48, 64, 3), 180, dtype=np.uint8))
    assert cv2.imwrite(str(nested / "b.png"), np.full((48, 64, 3), 180, dtype=np.uint8))
    monkeypatch.setattr(face_detection, "build_detector", lambda _args: SyntheticDetector())
    coverage_path = tmp_path / "coverage.json"
    stats_path = tmp_path / "stats.json"

    exit_code = face_detection.main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--padding",
            "0",
            "--no-tracker",
            "--coverage",
            str(coverage_path),
            "--export-stats",
            str(stats_path),
        ]
    )

    first = cv2.imread(str(output_dir / "a.png"))
    second = cv2.imread(str(output_dir / "nested" / "b.png"))
    assert exit_code == 0
    assert first is not None and second is not None
    assert np.all(first[12:28, 16:32] == 0)
    assert np.all(second[12:28, 16:32] == 0)
    assert coverage_path.is_file()
    assert stats_path.is_file()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["totals"]["frames"] == 2
    assert coverage["totals"]["skipped"] == 0
    assert coverage["skipped"] == []
    assert coverage["totals"]["files_with_no_faces"] == 0


def test_headless_batch_cli_reports_skipped_existing_files(
    tmp_path: Path, monkeypatch, capsys
):
    input_dir = tmp_path / "inbox"
    output_dir = tmp_path / "redacted"
    input_dir.mkdir()
    output_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "a.png"), np.full((48, 64, 3), 180, dtype=np.uint8))
    existing = np.full((48, 64, 3), 15, dtype=np.uint8)
    assert cv2.imwrite(str(output_dir / "a.png"), existing)
    monkeypatch.setattr(face_detection, "build_detector", lambda _args: SyntheticDetector())

    exit_code = face_detection.main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--padding",
            "0",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Skipped 1" in out
    assert "--overwrite" in out
    assert np.array_equal(cv2.imread(str(output_dir / "a.png")), existing)


def test_still_preview_main_counts_one_analytics_frame(tmp_path: Path, monkeypatch):
    class CountingDetector:
        name = "synthetic"

        def __init__(self):
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            return [Detection(16, 12, 16, 16, 0.99)]

    detector = CountingDetector()
    input_path = tmp_path / "input.png"
    stats_path = tmp_path / "session.json"
    assert cv2.imwrite(str(input_path), np.full((48, 64, 3), 180, dtype=np.uint8))
    monkeypatch.setattr(face_detection, "build_detector", lambda _args: detector)
    monkeypatch.setattr(face_detection.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(face_detection.cv2, "waitKey", lambda *_args, **_kwargs: ord("q"))
    monkeypatch.setattr(face_detection.cv2, "destroyAllWindows", lambda: None)

    exit_code = face_detection.main(
        [
            "--input",
            str(input_path),
            "--review",
            "--export-stats",
            str(stats_path),
            "--no-tracker",
            "--padding",
            "0",
        ]
    )

    data = json.loads(stats_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert detector.calls == 1
    assert data["total_frames"] == 1
