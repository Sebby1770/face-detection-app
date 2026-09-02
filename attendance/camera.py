"""Webcam capture helpers shared by the GUI screens."""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np


class Camera:
    """Tiny wrapper over cv2.VideoCapture with safe open/close semantics."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        # DirectShow is Windows-only; other platforms use the default backend.
        if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)
        return bool(self._cap and self._cap.isOpened())

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        return bool(ok), frame

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
