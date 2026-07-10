"""CLI source resolution tests (no camera required)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from face_detection import InputSource, SourceKind, resolve_source


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {
        "source": None,
        "source_flag": None,
        "camera": 0,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_default_uses_camera() -> None:
    src = resolve_source(_ns())
    assert src.kind is SourceKind.CAMERA
    assert src.camera_index == 0


def test_camera_flag() -> None:
    src = resolve_source(_ns(camera=2))
    assert src.kind is SourceKind.CAMERA
    assert src.camera_index == 2


def test_positional_camera_index() -> None:
    src = resolve_source(_ns(source="1"))
    assert src.kind is SourceKind.CAMERA
    assert src.camera_index == 1


def test_source_flag_overrides_positional_camera_default() -> None:
    src = resolve_source(_ns(source_flag="3", camera=0))
    assert src.kind is SourceKind.CAMERA
    assert src.camera_index == 3


def test_image_path(tmp_path: Path) -> None:
    img = tmp_path / "face.png"
    # Minimal valid PNG via OpenCV if available, else skip path existence with dummy + extension.
    import cv2
    import numpy as np

    cv2.imwrite(str(img), np.zeros((8, 8, 3), dtype=np.uint8))
    src = resolve_source(_ns(source=str(img)))
    assert src.kind is SourceKind.IMAGE
    assert src.path == img


def test_video_extension(tmp_path: Path) -> None:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"not a real video")
    src = resolve_source(_ns(source=str(vid)))
    assert src.kind is SourceKind.VIDEO
    assert src.path == vid


def test_missing_path_exits() -> None:
    with pytest.raises(SystemExit):
        resolve_source(_ns(source="/nonexistent/path/to/face.jpg"))
