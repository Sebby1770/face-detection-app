"""2.4 PIN kiosk, calendar heatmap, restore-db, and mark source."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from attendance.database import Database, hash_student_pin


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


def test_set_verify_pin_and_hash(db: Database) -> None:
    db.add_student("R1", "Ada")
    db.add_student("R2", "Bob")
    db.set_pin("R1", "1234")
    db.set_pin("R2", "1234")

    assert db.verify_pin("R1", "1234") is True
    assert db.verify_pin("R1", "0000") is False
    assert db.verify_pin("R1", "12") is False
    assert db.verify_pin("NOPE", "1234") is False
    assert db.verify_pin("R1", " 1234 ") is True

    row = db.get_student_by_roll("R1")
    stored = str(row["pin_hash"])
    assert stored.startswith("scrypt$")
    assert "1234" not in stored
    hashed = hash_student_pin("R1", "1234")
    assert hashed.startswith("scrypt$")
    assert "1234" not in hashed
    assert db.get_student_by_roll("R1")["pin_hash"] != db.get_student_by_roll("R2")["pin_hash"]

    legacy = hashlib.sha256(b"R1:1234").hexdigest()
    with db._connect() as conn:
        conn.execute(
            "UPDATE students SET pin_hash = ? WHERE roll_number = ?",
            (legacy, "R1"),
        )
    assert db.verify_pin("R1", "1234") is True
    assert db.verify_pin("R1", "0000") is False


def test_bad_pin_rejected(db: Database) -> None:
    db.add_student("R1", "Ada")
    with pytest.raises(ValueError, match="4"):
        db.set_pin("R1", "12")
    with pytest.raises(ValueError, match="4"):
        db.set_pin("R1", "123456789")
    with pytest.raises(ValueError, match="4"):
        db.set_pin("R1", "12ab")
    with pytest.raises(ValueError, match="no student"):
        db.set_pin("NOPE", "1234")
    assert db.get_student_by_roll("R1")["pin_hash"] is None


def test_mark_with_pin_success_and_fail(db: Database) -> None:
    db.add_student("R1", "Ada")
    db.set_pin("R1", "1234")
    assert db.mark_with_pin(
        "R1",
        "1234",
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Present",
    )
    df = db.get_attendance(date="2026-08-18")
    assert len(df) == 1
    assert df.iloc[0]["status"] == "Present"
    assert df.iloc[0]["source"] == "pin"

    with pytest.raises(ValueError, match="PIN"):
        db.mark_with_pin("R1", "9999", at=datetime(2026, 8, 18, 10, 0, 0))
    with pytest.raises(ValueError, match="no student"):
        db.mark_with_pin("NOPE", "1234")
    with pytest.raises(ValueError, match="4"):
        db.mark_with_pin("R1", "12")


def test_mark_with_pin_archived(db: Database) -> None:
    db.add_student("R1", "Ada")
    db.set_pin("R1", "1234")
    db.archive_student("R1")
    with pytest.raises(ValueError, match="archived"):
        db.mark_with_pin("R1", "1234", status="Present")


def test_calendar_grid_statuses_holiday_and_range_cap(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    bob = db.add_student("R2", "Bob", section="A")
    db.add_student("R3", "Arch", section="A")
    db.archive_student("R3")
    db.add_holiday("2026-08-19", "Break")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 10, 0, 0), status="Late")
    db.mark_attendance(ada, at=datetime(2026, 8, 18, 9, 40, 0), status="Late")
    db.mark_attendance(bob, at=datetime(2026, 8, 17, 9, 0, 0), status="Excused")

    grid = db.calendar_grid("2026-08-17", "2026-08-19", section="A")
    assert grid["from"] == "2026-08-17"
    assert grid["to"] == "2026-08-19"
    assert grid["dates"] == ["2026-08-17", "2026-08-18", "2026-08-19"]
    assert grid["holidays"]["2026-08-19"] == "Break"
    rolls = [s["roll_number"] for s in grid["students"]]
    assert rolls == ["R1", "R2"]
    ada_days = next(s for s in grid["students"] if s["roll_number"] == "R1")["days"]
    assert ada_days["2026-08-17"] == "Present"
    assert ada_days["2026-08-18"] == "Late"
    assert ada_days["2026-08-19"] is None
    bob_days = next(s for s in grid["students"] if s["roll_number"] == "R2")["days"]
    assert bob_days["2026-08-17"] == "Excused"

    with pytest.raises(ValueError, match="on or after"):
        db.calendar_grid("2026-08-20", "2026-08-10")
    with pytest.raises(ValueError, match="93"):
        db.calendar_grid("2026-01-01", "2026-04-04")
    ok = db.calendar_grid("2026-01-01", "2026-04-03")
    assert len(ok["dates"]) == 93


def test_restore_database_roundtrip(db: Database, tmp_path: Path) -> None:
    sid = db.add_student("R1", "Ada", section="A")
    db.set_pin("R1", "1234")
    db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Present",
        source="gui",
    )
    backup = tmp_path / "backup.db"
    db.backup_database(backup)
    assert backup.is_file()

    db.add_student("R2", "Bob")
    assert db.get_student_by_roll("R2") is not None

    restored_path = tmp_path / "restored.db"
    other = Database(db_path=restored_path)
    other.add_student("TEMP", "Temp")
    other.restore_database(backup)

    assert other.get_student_by_roll("R1") is not None
    assert other.get_student_by_roll("TEMP") is None
    assert other.get_student_by_roll("R2") is None
    assert other.verify_pin("R1", "1234")
    att = other.get_attendance()
    assert len(att) == 1
    assert att.iloc[0]["name"] == "Ada"
    assert att.iloc[0]["source"] == "gui"


def test_mark_source_persisted(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    assert db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Present",
        source="camera",
        period="Morning",
    )
    assert db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 10, 0, 0),
        status="Late",
        source="nope",
        period="Afternoon",
    )
    df = db.get_attendance(date="2026-08-18")
    by_time = {str(row["time"]): str(row["source"]) for _, row in df.iterrows()}
    assert by_time["09:00:00"] == "camera"
    assert by_time["10:00:00"] == "cli"
    default_sid = db.add_student("R2", "Bob")
    db.mark_attendance(
        default_sid, at=datetime(2026, 8, 19, 9, 0, 0), status="Present"
    )
    later = db.get_attendance(date="2026-08-19")
    assert later.iloc[0]["source"] == "cli"


def test_cli_pin_calendar_restore(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("R1", "Ada", section="A")
    isolated_db.add_student("R2", "Bob", section="A")
    rc = main(["students", "pin", "R1", "--pin", "1234"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PIN set" in out
    assert "1234" not in out
    assert isolated_db.verify_pin("R1", "1234")

    rc = main(
        [
            "mark",
            "--roll",
            "R1",
            "--pin",
            "1234",
            "--status",
            "Present",
            "--at",
            "2026-08-18T09:00:00",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert isolated_db.get_attendance(date="2026-08-18").iloc[0]["source"] == "pin"

    isolated_db.add_holiday("2026-08-19", "Break")
    isolated_db.mark_attendance(
        isolated_db.get_student_by_roll("R2")["id"],
        at=datetime(2026, 8, 18, 9, 30, 0),
        status="Late",
    )
    rc = main(["calendar", "--from", "2026-08-18", "--to", "2026-08-19"])
    assert rc == 0
    heat = capsys.readouterr().out
    assert "P" in heat
    assert "H" in heat
    assert "L" in heat

    csv_path = tmp_path / "cal.csv"
    rc = main(
        [
            "calendar",
            "--from",
            "2026-08-18",
            "--to",
            "2026-08-19",
            "-o",
            str(csv_path),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert csv_path.is_file()

    backup = tmp_path / "bak.db"
    isolated_db.backup_database(backup)
    isolated_db.add_student("R9", "Zed")
    rc = main(["restore-db", str(backup)])
    assert rc == 0
    capsys.readouterr()
    assert isolated_db.get_student_by_roll("R1") is not None
    assert isolated_db.get_student_by_roll("R9") is None


def test_api_calendar_and_mark_with_pin(isolated_db: Database) -> None:
    from attendance.server import make_server

    isolated_db.add_student("S1", "Api Kid", section="A")
    isolated_db.add_holiday("2026-08-19", "Break")

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        set_pin = json.dumps({"roll": "S1", "pin": "1234"}).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{port}/pin",
            data=set_pin,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            pinned = json.loads(resp.read().decode("utf-8"))
        assert pinned["ok"] is True
        assert "1234" not in json.dumps(pinned)
        assert isolated_db.verify_pin("S1", "1234")

        bad = json.dumps(
            {
                "roll": "S1",
                "pin": "0000",
                "status": "Present",
                "at": "2026-08-18T09:00:00",
            }
        ).encode("utf-8")
        bad_req = Request(
            f"http://127.0.0.1:{port}/mark",
            data=bad,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as excinfo:
            urlopen(bad_req, timeout=5)
        assert excinfo.value.code == 400

        payload = json.dumps(
            {
                "roll": "S1",
                "pin": "1234",
                "status": "Present",
                "at": "2026-08-18T09:00:00",
            }
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
        assert marked["source"] == "pin"
        att = isolated_db.get_attendance(date="2026-08-18")
        assert att.iloc[0]["source"] == "pin"

        with urlopen(
            f"http://127.0.0.1:{port}/calendar?from=2026-08-18&to=2026-08-19",
            timeout=5,
        ) as resp:
            grid = json.loads(resp.read().decode("utf-8"))
        assert grid["from"] == "2026-08-18"
        assert grid["holidays"]["2026-08-19"] == "Break"
        assert grid["students"][0]["days"]["2026-08-18"] == "Present"
        assert grid["students"][0]["days"]["2026-08-19"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
