"""2.3 archive, alerts, holiday import, backup, and API extras."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from attendance.database import Database


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    from attendance import config

    data = tmp_path / "data"
    faces = data / "faces"
    trainer = data / "trainer"
    exports = tmp_path / "exports"
    for folder in (faces, trainer, exports):
        folder.mkdir(parents=True, exist_ok=True)
    db_path = data / "api.db"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "FACES_DIR", faces)
    monkeypatch.setattr(config, "TRAINER_DIR", trainer)
    monkeypatch.setattr(config, "EXPORTS_DIR", exports)
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return Database(db_path=db_path)


def test_archive_hides_student_from_absentees_and_blocks_mark(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    db.add_student("R2", "Bob", section="A")
    db.archive_student("R2")

    absentees = db.get_absentees("2026-08-18")
    assert list(absentees["roll_number"]) == ["R1"]
    assert [s["roll_number"] for s in db.list_students()] == ["R1"]
    assert [s["roll_number"] for s in db.list_students(include_inactive=True)] == [
        "R1",
        "R2",
    ]

    try:
        db.mark_attendance(db.get_student_by_roll("R2")["id"], status="Present")
        raise AssertionError("archived student should not be markable")
    except ValueError as exc:
        assert "archived" in str(exc)

    db.restore_student("R2")
    assert db.mark_attendance(
        db.get_student_by_roll("R2")["id"],
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Present",
    )
    assert int(db.get_student(ada)["id"]) == ada


def test_bulk_excuse_alerts_and_undo(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    db.add_student("R2", "Bob", section="A")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")

    excused = db.bulk_excuse("2026-08-17", section="A")
    assert excused == 1
    digest = db.daily_digest("2026-08-17", section="A")
    assert digest["excused_count"] == 1
    assert digest["absentee_count"] == 0

    for offset in range(4):
        day = datetime(2026, 8, 18) + timedelta(days=offset)
        db.mark_attendance(ada, at=day.replace(hour=9, minute=0), status="Present")

    consecutive = db.consecutive_absences(as_of="2026-08-21", min_days=3, lookback=7)
    rolls = {row["roll_number"] for row in consecutive}
    assert "R2" in rolls

    at_risk = db.at_risk_students("2026-08-17", "2026-08-21", threshold=90.0)
    assert any(row["roll_number"] == "R2" for row in at_risk)

    removed = db.undo_last_mark("R1")
    assert removed is not None
    assert removed["status"] == "Present"


def test_holiday_import_backup_and_purge(db: Database, tmp_path: Path) -> None:
    csv_path = tmp_path / "days.csv"
    csv_path.write_text("date,name\n2026-08-19,Founders\nnot-a-date,x\n", encoding="utf-8")
    result = db.import_holidays_csv(csv_path)
    assert result["added"] == 1
    assert result["skipped"] == 1
    assert db.get_holiday("2026-08-19") is not None

    sid = db.add_student("P1", "Pat")
    db.mark_attendance(sid, at=datetime(2026, 1, 2, 9, 0, 0), status="Present")
    db.mark_attendance(sid, at=datetime(2026, 8, 18, 9, 0, 0), status="Present")
    deleted = db.purge_attendance_before("2026-06-01")
    assert deleted == 1

    backup = tmp_path / "copy.db"
    db.backup_database(backup)
    assert backup.is_file()
    assert backup.stat().st_size > 0


def test_serve_holidays_alerts_and_post_mark(isolated_db: Database) -> None:
    from attendance.server import make_server

    isolated_db.add_student("S1", "Api Kid", section="A")
    isolated_db.add_holiday("2026-08-19", "Break")

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/holidays", timeout=5) as resp:
            holidays = json.loads(resp.read().decode("utf-8"))
        assert holidays["count"] == 1
        assert holidays["holidays"][0]["date"] == "2026-08-19"

        payload = json.dumps(
            {"roll": "S1", "status": "Present", "at": "2026-08-18T09:00:00"}
        ).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{port}/mark",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            marked = json.loads(resp.read().decode("utf-8"))
        assert marked["ok"] is True
        assert marked["inserted"] is True

        with urlopen(
            f"http://127.0.0.1:{port}/alerts?from=2026-08-18&to=2026-08-18&threshold=10",
            timeout=5,
        ) as resp:
            alerts = json.loads(resp.read().decode("utf-8"))
        assert "at_risk" in alerts
        assert "consecutive" in alerts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
