from pathlib import Path

import pytest

from detectors import find_haar_path


def test_explicit_haar_path_is_used_without_discovery(tmp_path: Path):
    cascade = tmp_path / "cascade.xml"
    cascade.write_text("placeholder", encoding="utf-8")

    assert find_haar_path(cascade) == cascade


def test_missing_explicit_haar_path_has_an_actionable_error(tmp_path: Path):
    missing = tmp_path / "missing.xml"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_haar_path(missing)
