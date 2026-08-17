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
