"""Folder enrollment tests — synthetic grayscale PNGs, no webcam."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from attendance import config  # noqa: E402
from attendance.face_engine import (  # noqa: E402
    FaceEngine,
    FaceEngineError,
    clear_model_stale_flag,
    is_low_quality_sample,
)


@pytest.fixture()
def engine_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def _make_engine() -> FaceEngine:
    try:
        return FaceEngine()
    except FaceEngineError as exc:
        pytest.skip(str(exc))


def _write_gray_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    assert ok, f"failed to write {path}"


def _pattern(kind: str, size: int = 200, seed: int = 0) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    if kind == "left":
        img[:, : size // 2] = 220
        img[:, size // 2 :] = 30
        # Seeded mark so otherwise-identical left crops are unique files.
        img[seed % size, seed % size] = 40 + (seed % 80)
    elif kind == "right":
        img[:, : size // 2] = 30
        img[:, size // 2 :] = 220
        img[seed % size, seed % size] = 40 + (seed % 80)
    elif kind == "stripe":
        img[::8, :] = 200
        img[:, seed % 17 :: 17] = 40
    else:
        img[:] = 128 + (seed % 40)
        cv2.circle(img, (size // 2, size // 2), 50, int(80 + seed % 50), -1)
    return img


def test_enroll_from_folder_writes_samples(engine_dirs: Path) -> None:
    photos = engine_dirs / "photos"
    photos.mkdir()
    for i in range(4):
        _write_gray_png(photos / f"shot_{i}.png", _pattern("circle", seed=i))

    engine = _make_engine()
    written = engine.enroll_from_folder(photos, student_id=7)
    assert written == 4
    dest = config.FACES_DIR / "7"
    samples = sorted(dest.glob("*.png"))
    assert len(samples) == 4
    loaded = cv2.imread(str(samples[0]), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    assert loaded.shape[:2] == config.FACE_IMAGE_SIZE


def test_enroll_appends_and_respects_max_samples(engine_dirs: Path) -> None:
    photos = engine_dirs / "batch"
    photos.mkdir()
    for i in range(5):
        _write_gray_png(photos / f"{i}.png", _pattern("stripe", seed=i))

    engine = _make_engine()
    assert engine.enroll_from_folder(photos, student_id=1, max_samples=2) == 2
    assert engine.enroll_from_folder(photos, student_id=1, max_samples=2) == 2
    assert len(list((config.FACES_DIR / "1").glob("*.png"))) == 4


def test_enroll_skips_identical_pngs(engine_dirs: Path) -> None:
    import shutil

    photos = engine_dirs / "dups"
    photos.mkdir()
    first = photos / "a.png"
    _write_gray_png(first, _pattern("circle", seed=3))
    shutil.copyfile(first, photos / "b.png")

    engine = _make_engine()
    written = engine.enroll_from_folder(photos, student_id=9)
    assert written == 1
    samples = list((config.FACES_DIR / "9").glob("*.png"))
    assert len(samples) == 1


def test_enroll_skips_tiny_and_black_images(engine_dirs: Path) -> None:
    photos = engine_dirs / "mixed"
    photos.mkdir()
    _write_gray_png(photos / "good.png", _pattern("circle", seed=1))
    _write_gray_png(photos / "tiny.png", np.full((20, 20), 200, dtype=np.uint8))
    _write_gray_png(photos / "black.png", np.zeros((200, 200), dtype=np.uint8))

    assert is_low_quality_sample(np.full((20, 20), 200, dtype=np.uint8))
    assert is_low_quality_sample(np.zeros((200, 200), dtype=np.uint8))
    assert not is_low_quality_sample(_pattern("circle", seed=1))

    engine = _make_engine()
    written = engine.enroll_from_folder(photos, student_id=5)
    assert written == 1
    assert engine.last_quality_skipped == 2
    samples = list((config.FACES_DIR / "5").glob("*.png"))
    assert len(samples) == 1


def test_enroll_empty_folder_raises(engine_dirs: Path) -> None:
    empty = engine_dirs / "empty"
    empty.mkdir()
    engine = _make_engine()
    with pytest.raises(FaceEngineError, match="No usable images"):
        engine.enroll_from_folder(empty, student_id=3)


def test_enroll_missing_folder_raises(engine_dirs: Path) -> None:
    engine = _make_engine()
    with pytest.raises(FaceEngineError, match="does not exist"):
        engine.enroll_from_folder(engine_dirs / "nope", student_id=1)


def test_enroll_then_train_and_predict(engine_dirs: Path) -> None:
    left_dir = engine_dirs / "left"
    right_dir = engine_dirs / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    for i in range(8):
        _write_gray_png(left_dir / f"{i}.png", _pattern("left", seed=i))
        _write_gray_png(right_dir / f"{i}.png", _pattern("right", seed=i))

    engine = _make_engine()
    assert engine.enroll_from_folder(left_dir, student_id=1) == 8
    assert engine.enroll_from_folder(right_dir, student_id=2) == 8
    trained = engine.train_from_dataset()
    assert trained == 16

    sid_left, conf_left = engine.predict(_pattern("left"))
    sid_right, conf_right = engine.predict(_pattern("right"))
    assert sid_left == 1
    assert sid_right == 2
    assert conf_left < config.get_confidence_threshold()
    assert conf_right < config.get_confidence_threshold()

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=config.FACE_IMAGE_SIZE, dtype=np.uint8)
    unknown_id, unknown_conf = engine.predict(noise, threshold=8.0)
    assert unknown_id is None
    assert unknown_conf > 8.0


def test_engine_loads_bundled_or_degrades(engine_dirs: Path) -> None:
    engine = _make_engine()
    assert engine.recognizer is not None
    if config.BUNDLED_HAAR_CASCADE.is_file():
        assert engine.detector is not None
        assert not engine.detector.empty()


def test_extract_face_accepts_already_cropped(engine_dirs: Path) -> None:
    engine = _make_engine()
    crop = _pattern("circle", size=200)
    face = engine.extract_face(crop, detect=True)
    assert face is not None
    assert face.shape[:2] == config.FACE_IMAGE_SIZE
