"""Face-engine helper tests that do not require a camera or trained model.

OpenCV is still imported by face_engine; if it is missing the whole module is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from attendance import config  # noqa: E402
from attendance.face_engine import (  # noqa: E402
    clear_model_stale_flag,
    invalidate_model_if_needed,
    mark_model_stale,
    model_is_stale,
)


@pytest.fixture()
def trainer_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    faces = tmp_path / "faces"
    trainer = tmp_path / "trainer"
    faces.mkdir()
    trainer.mkdir()
    monkeypatch.setattr(config, "FACES_DIR", faces)
    monkeypatch.setattr(config, "TRAINER_DIR", trainer)
    monkeypatch.setattr(config, "MODEL_PATH", trainer / "lbph_model.yml")
    monkeypatch.setattr(config, "LABEL_MAP_PATH", trainer / "label_map.json")
    monkeypatch.setattr(config, "MODEL_STALE_FLAG", trainer / ".model_stale")
    clear_model_stale_flag()
    return tmp_path


def test_mark_and_clear_stale_flag(trainer_dirs: Path) -> None:
    assert model_is_stale() is False
    mark_model_stale("unit test")
    assert model_is_stale() is True
    assert config.MODEL_STALE_FLAG.exists()
    clear_model_stale_flag()
    assert model_is_stale() is False


def test_invalidate_when_label_map_mismatches_disk(trainer_dirs: Path) -> None:
    # Simulate a trained model for student 1, but faces folder has 1 and 2.
    config.MODEL_PATH.write_text("placeholder", encoding="utf-8")
    config.LABEL_MAP_PATH.write_text(json.dumps({"0": 1}), encoding="utf-8")
    (config.FACES_DIR / "1").mkdir()
    (config.FACES_DIR / "2").mkdir()
    # Touch a sample so directories look real
    (config.FACES_DIR / "1" / "001.png").write_bytes(b"\x89PNG\r\n")
    (config.FACES_DIR / "2" / "001.png").write_bytes(b"\x89PNG\r\n")

    assert invalidate_model_if_needed() is True
    assert model_is_stale() is True


def test_not_stale_when_in_sync(trainer_dirs: Path) -> None:
    config.MODEL_PATH.write_text("placeholder", encoding="utf-8")
    config.LABEL_MAP_PATH.write_text(json.dumps({"0": 7}), encoding="utf-8")
    student = config.FACES_DIR / "7"
    student.mkdir()
    sample = student / "001.png"
    sample.write_bytes(b"\x89PNG\r\n")
    # Ensure sample is not newer than model (write model after sample)
    import time

    time.sleep(0.05)
    config.MODEL_PATH.write_text("placeholder-v2", encoding="utf-8")

    clear_model_stale_flag()
    assert invalidate_model_if_needed() is False
