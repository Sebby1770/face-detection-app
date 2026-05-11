"""Face detector backends: YuNet (DNN) and Haar cascade."""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


Bbox = Tuple[float, float, float, float]


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def bbox(self) -> Bbox:
        return (float(self.x), float(self.y), float(self.w), float(self.h))


def find_haar_path() -> Path:
    return Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def _valid_model(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 100_000


def find_yunet_path(model_dir: Path, download: bool = True) -> Optional[Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / YUNET_FILENAME
    if _valid_model(path):
        return path
    if not download:
        return None
    print(f"Downloading YuNet model → {path}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(YUNET_URL, str(path))
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        path.unlink(missing_ok=True)
        return None
    return path if _valid_model(path) else None


class HaarDetector:
    name = "haar"

    def __init__(
        self,
        cascade_path: str,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 60,
    ) -> None:
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = (min_size, min_size)

    def detect(self, frame_bgr) -> List[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [Detection(int(x), int(y), int(w), int(h), 1.0, []) for (x, y, w, h) in rects]


class YuNetDetector:
    name = "yunet"

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Your opencv-python is too old; upgrade to >=4.8.")
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._input_size: Optional[Tuple[int, int]] = None

    def detect(self, frame_bgr) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []
        results: List[Detection] = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            landmarks = [(int(face[4 + i * 2]), int(face[5 + i * 2])) for i in range(5)]
            score = float(face[14])
            results.append(Detection(x, y, fw, fh, score, landmarks))
        return results
