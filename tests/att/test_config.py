"""Basic configuration sanity checks."""
from __future__ import annotations

from pathlib import Path

from attendance import __version__, config


def test_version_matches() -> None:
    assert __version__ == "3.0.0"
    assert config.APP_VERSION == "3.0.0"


def test_paths_are_under_project_root() -> None:
    root = config.PROJECT_ROOT
    assert root.is_dir()
    assert config.DATA_DIR.is_relative_to(root) or str(config.DATA_DIR).startswith(
        str(root)
    )
    assert config.DB_PATH.parent == config.DATA_DIR
    assert config.UNKNOWNS_DIR.parent == config.DATA_DIR
    assert config.MODEL_PATH.parent == config.TRAINER_DIR
    assert config.LABEL_MAP_PATH.parent == config.TRAINER_DIR
    assert config.MODEL_STALE_FLAG.parent == config.TRAINER_DIR
    assert config.BUNDLED_HAAR_CASCADE.parent == config.ASSETS_DIR
    assert config.BUNDLED_HAAR_CASCADE.is_file()


def test_required_dirs_exist() -> None:
    for d in (
        config.DATA_DIR,
        config.FACES_DIR,
        config.TRAINER_DIR,
        config.EXPORTS_DIR,
        config.ASSETS_DIR,
        config.UNKNOWNS_DIR,
    ):
        assert Path(d).is_dir()


def test_tuning_constants_are_sensible() -> None:
    assert config.SAMPLES_PER_STUDENT > 0
    assert config.FACE_IMAGE_SIZE == (200, 200)
    assert config.CONFIDENCE_THRESHOLD > 0
    assert config.RECOGNITION_THRESHOLD > 0
    assert config.CONFIDENCE_THRESHOLD == config.RECOGNITION_THRESHOLD
    assert config.ATTENDANCE_COOLDOWN_SECONDS > 0
    assert config.UNKNOWN_LOG_COOLDOWN_SECONDS > 0
    assert config.LATE_GRACE_MINUTES >= 0
    assert config.DEFAULT_PERIOD_START == "09:00"
    assert config.PERIODS
    assert any(p["name"] == "Morning" for p in config.PERIODS)
    assert config.ENROLL_MIN_SIZE == (40, 40)
    assert config.ENROLL_MIN_MEAN_PIXEL == 8.0
    assert config.AMBIGUOUS_MARGIN == 8.0
    assert config.ENROLL_MIN_LAPLACIAN == 40.0
    assert config.LIVENESS_HISTORY == 12
    assert config.LIVENESS_MIN_MOTION_PX == 16


def test_threshold_override_updates_alias() -> None:
    config.set_confidence_threshold(55.5)
    assert config.get_confidence_threshold() == 55.5
    assert config.CONFIDENCE_THRESHOLD == 55.5
    assert config.RECOGNITION_THRESHOLD == 55.5


def test_threshold_rejects_non_positive() -> None:
    import pytest

    with pytest.raises(ValueError):
        config.set_confidence_threshold(0)
    with pytest.raises(ValueError):
        config.set_confidence_threshold(-3)


def test_dark_and_light_palettes() -> None:
    assert "dark" in config.THEMES
    assert "light" in config.THEMES
    light_bg = config.THEMES["light"]["COLOR_BG"]
    dark_bg = config.THEMES["dark"]["COLOR_BG"]
    assert light_bg != dark_bg

    config.apply_theme("dark")
    assert config.get_theme() == "dark"
    assert config.COLOR_BG == dark_bg
    assert config.COLOR_SURFACE == config.THEMES["dark"]["COLOR_SURFACE"]

    config.apply_theme("light")
    assert config.get_theme() == "light"
    assert config.COLOR_BG == light_bg


def test_unknown_theme_falls_back_to_light() -> None:
    config.apply_theme("dark")
    applied = config.apply_theme("neon-cyber")
    assert applied == "light"
    assert config.get_theme() == "light"
