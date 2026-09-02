"""2.8 school-day web parity helpers, roster suffixes, unknown assign, WAL backup."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from attendance.database import (
    Database,
    iso_weekday,
    parse_hhmm,
    parse_weekend_days,
)


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    from attendance import config

    data = tmp_path / "data"
    faces = data / "faces"
    trainer = data / "trainer"
    exports = tmp_path / "exports"
    unknowns = data / "unknowns"
    for folder in (faces, trainer, exports, unknowns):
        folder.mkdir(parents=True, exist_ok=True)
    db_path = data / "api.db"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "FACES_DIR", faces)
    monkeypatch.setattr(config, "TRAINER_DIR", trainer)
    monkeypatch.setattr(config, "EXPORTS_DIR", exports)
    monkeypatch.setattr(config, "UNKNOWNS_DIR", unknowns)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "MODEL_PATH", trainer / "lbph_model.yml")
    monkeypatch.setattr(config, "LABEL_MAP_PATH", trainer / "label_map.json")
    monkeypatch.setattr(config, "MODEL_STALE_FLAG", trainer / ".model_stale")
    return Database(db_path=db_path)


def test_parse_iso_helpers_still_work() -> None:
    assert parse_hhmm("09:00") == (9, 0)
    assert parse_hhmm("9.30") == (9, 30)
    assert iso_weekday("2026-08-17") == 1
    assert iso_weekday("2026-08-22") == 6
    assert iso_weekday("2026-08-23") == 7
    assert parse_weekend_days("sat,sun") == {6, 7}
    assert parse_weekend_days(None) == {6, 7}


def test_roster_counts_jpg_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from attendance import config

    faces = tmp_path / "faces"
    faces.mkdir()
    monkeypatch.setattr(config, "FACES_DIR", faces)
    db = Database(db_path=tmp_path / "roster.db")
    sid = db.add_student("R1", "Ada", section="A")
    folder = faces / str(sid)
    folder.mkdir()
    (folder / "001.jpg").write_bytes(b"jpeg")
    (folder / "note.txt").write_text("ignored", encoding="utf-8")

    rows = db.enrollment_roster(faces_dir=faces, min_samples=8)
    by_roll = {row["roll_number"]: row for row in rows}
    assert by_roll["R1"]["samples"] == 1
    assert by_roll["R1"]["ready"] is False


def test_assign_unknown_crop_copies_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from attendance import config

    faces = tmp_path / "faces"
    trainer = tmp_path / "trainer"
    faces.mkdir()
    trainer.mkdir()
    monkeypatch.setattr(config, "FACES_DIR", faces)
    monkeypatch.setattr(config, "TRAINER_DIR", trainer)
    monkeypatch.setattr(config, "MODEL_STALE_FLAG", trainer / ".model_stale")
    db = Database(db_path=tmp_path / "assign.db")
    sid = db.add_student("R1", "Ada")
    src = tmp_path / "unknown.png"
    payload = b"\x89PNG\r\n\x1a\n"
    src.write_bytes(payload)

    result = db.assign_unknown_crop(src, "R1")
    dest = Path(result["dest"])
    assert result["roll"] == "R1"
    assert result["samples_written"] == 1
    assert dest.is_file()
    assert dest.parent == faces / str(sid)
    assert dest.name == "001.png"
    assert dest.read_bytes() == payload
    assert not src.exists()

    rows = db.enrollment_roster(faces_dir=faces, min_samples=1)
    assert rows[0]["samples"] == 1
    assert (trainer / ".model_stale").is_file()


def test_backup_copies_wal_sidecar(tmp_path: Path) -> None:
    db_path = tmp_path / "attendance.db"
    db = Database(db_path=db_path)
    wal = tmp_path / "attendance.db-wal"
    wal.write_bytes(b"wal-bytes")
    dest = tmp_path / "backup" / "foo.db"
    db.backup_database(dest)
    assert dest.is_file()
    dest_wal = tmp_path / "backup" / "foo.db-wal"
    assert dest_wal.is_file()
    assert dest_wal.read_bytes() == b"wal-bytes"


def test_log_unknown_face_stores_path(db: Database) -> None:
    assert (
        db.log_unknown_face(
            80.0,
            at=datetime(2026, 8, 18, 11, 0, 0),
            path="/tmp/unk.png",
        )
        is True
    )
    rows = db.list_unknown_crops(date="2026-08-18")
    assert len(rows) == 1
    assert rows[0]["path"] == "/tmp/unk.png"
    assert rows[0]["confidence"] == pytest.approx(80.0)


def test_cli_unknowns_list_exit_0(
    isolated_db: Database, capsys: pytest.CaptureFixture
) -> None:
    from attendance import config
    from main import main

    orphan = config.UNKNOWNS_DIR / "orphan.png"
    orphan.write_bytes(b"x")
    rc = main(["unknowns", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "orphan.png" in out


def test_cli_students_export_writes_header(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("R1", "Ada Lovelace", section="A")
    out = tmp_path / "roster.csv"
    rc = main(["students", "export", "-o", str(out)])
    assert rc == 0
    capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    assert "roll_number" in text.splitlines()[0]
    assert "Ada Lovelace" in text
