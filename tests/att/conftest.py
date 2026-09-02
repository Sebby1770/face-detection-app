"""Shared fixtures. Restores config overrides so tests stay isolated."""
from __future__ import annotations

from pathlib import Path

import pytest

from attendance.database import Database


@pytest.fixture(autouse=True)
def _reset_config_runtime() -> None:
    yield
    from attendance import config

    config.reset_runtime_overrides()


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(db_path=tmp_path / "test_attendance.db")
