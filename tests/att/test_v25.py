"""2.5 present streaks, ICS export, and mark notes."""
from __future__ import annotations

import json
import threading
from datetime import datetime
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


def test_present_streak_skips_holidays(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    bob = db.add_student("R2", "Bob", section="A")
    db.add_holiday("2026-08-19", "Break")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(ada, at=datetime(2026, 8, 18, 9, 40, 0), status="Late")
    db.mark_attendance(ada, at=datetime(2026, 8, 20, 9, 0, 0), status="Present")

    assert db.present_streak("R1", as_of="2026-08-20") == 3
    assert db.present_streak("R1", as_of="2026-08-19") == 2
    assert db.present_streak("R1", as_of="2026-08-18") == 2
    assert db.present_streak("R1", as_of="2026-08-21") == 0

    db.mark_attendance(bob, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(bob, at=datetime(2026, 8, 18, 9, 0, 0), status="Present")
    db.mark_attendance(bob, at=datetime(2026, 8, 20, 9, 0, 0), status="Excused")
    assert db.present_streak("R2", as_of="2026-08-20") == 2
    assert db.present_streak("R2", as_of="2026-08-18") == 2

    report = db.streaks_report(as_of="2026-08-20", section="A")
    assert [row["roll_number"] for row in report] == ["R1", "R2"]
    by_roll = {row["roll_number"]: row for row in report}
    assert by_roll["R1"]["streak"] == 3
    assert by_roll["R1"]["name"] == "Ada"
    assert by_roll["R1"]["section"] == "A"
    assert by_roll["R2"]["streak"] == 2

    with pytest.raises(ValueError, match="no student"):
        db.present_streak("NOPE", as_of="2026-08-20")


def test_export_attendance_ics_contains_vevent(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    bob = db.add_student("R2", "Bob", section="A")
    db.add_holiday("2026-08-19", "Break")
    db.mark_attendance(
        ada,
        at=datetime(2026, 8, 17, 9, 0, 0),
        status="Present",
        note="on time",
    )
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 10, 0, 0), status="Late")
    db.mark_attendance(ada, at=datetime(2026, 8, 18, 9, 40, 0), status="Late")
    db.mark_attendance(bob, at=datetime(2026, 8, 18, 9, 0, 0), status="Excused")

    ics = db.export_attendance_ics("2026-08-17", "2026-08-20")
    assert "BEGIN:VCALENDAR" in ics
    assert "BEGIN:VEVENT" in ics
    assert "END:VEVENT" in ics
    assert "END:VCALENDAR" in ics
    assert "DTSTART;VALUE=DATE:20260817" in ics
    assert "UID:R1-2026-08-17@face-recognition-attendance" in ics
    assert "UID:R1-2026-08-18@face-recognition-attendance" in ics
    assert "UID:R2-2026-08-18@face-recognition-attendance" in ics
    assert ics.count("BEGIN:VEVENT") == 3
    assert "on time" in ics
    assert "DTSTART;VALUE=DATE:20260819" not in ics
    assert "R1-2026-08-19" not in ics

    empty = db.export_attendance_ics("2026-08-19", "2026-08-19")
    assert "BEGIN:VCALENDAR" in empty
    assert "BEGIN:VEVENT" not in empty

    with pytest.raises(ValueError, match="on or after"):
        db.export_attendance_ics("2026-08-20", "2026-08-17")


def test_mark_note_persisted(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    assert db.mark_attendance(
        sid,
        at=datetime(2026, 8, 18, 9, 40, 0),
        status="Late",
        note="bus late",
        source="cli",
    )
    df = db.get_attendance(date="2026-08-18")
    assert len(df) == 1
    assert df.iloc[0]["status"] == "Late"
    assert df.iloc[0]["note"] == "bus late"

    db.set_pin("R1", "1234")
    assert db.mark_with_pin(
        "R1",
        "1234",
        at=datetime(2026, 8, 19, 9, 0, 0),
        status="Present",
        note="kiosk",
    )
    later = db.get_attendance(date="2026-08-19")
    assert later.iloc[0]["note"] == "kiosk"
    assert later.iloc[0]["source"] == "pin"


def test_api_get_streaks(isolated_db: Database) -> None:
    from attendance.server import make_server

    ada = isolated_db.add_student("R1", "Ada", section="A")
    isolated_db.add_student("R2", "Bob", section="A")
    isolated_db.add_holiday("2026-08-19", "Break")
    isolated_db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    isolated_db.mark_attendance(ada, at=datetime(2026, 8, 18, 9, 0, 0), status="Late")
    isolated_db.mark_attendance(ada, at=datetime(2026, 8, 20, 9, 0, 0), status="Present")

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{port}/streaks?as_of=2026-08-20&section=A",
            timeout=5,
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["as_of"] == "2026-08-20"
        assert payload["count"] == 2
        by_roll = {row["roll_number"]: row for row in payload["streaks"]}
        assert by_roll["R1"]["streak"] == 3
        assert by_roll["R2"]["streak"] == 0

        with urlopen(
            f"http://127.0.0.1:{port}/calendar.ics?from=2026-08-17&to=2026-08-20",
            timeout=5,
        ) as resp:
            assert "text/calendar" in resp.headers.get("Content-Type", "")
            ics = resp.read().decode("utf-8")
        assert "BEGIN:VEVENT" in ics
        assert "R1-2026-08-17" in ics

        body = json.dumps(
            {
                "roll": "R2",
                "status": "Present",
                "at": "2026-08-18T09:00:00",
                "note": "bus late",
            }
        ).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{port}/mark",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            marked = json.loads(resp.read().decode("utf-8"))
        assert marked["ok"] is True
        att = isolated_db.get_attendance(date="2026-08-18")
        bob = att[att["roll_number"] == "R2"]
        assert bob.iloc[0]["note"] == "bus late"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_streaks_ics_and_note(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    isolated_db.add_student("R1", "Ada", section="A")
    rc = main(
        [
            "mark",
            "--roll",
            "R1",
            "--status",
            "Present",
            "--at",
            "2026-08-17T09:00:00",
            "--note",
            "bus late",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    df = isolated_db.get_attendance(date="2026-08-17")
    assert df.iloc[0]["note"] == "bus late"

    isolated_db.mark_attendance(
        isolated_db.get_student_by_roll("R1")["id"],
        at=datetime(2026, 8, 18, 9, 0, 0),
        status="Late",
    )
    rc = main(["streaks", "--as-of", "2026-08-18", "--section", "A"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ada" in out
    ada_lines = [ln for ln in out.splitlines() if "Ada" in ln]
    assert ada_lines and ada_lines[0].rstrip().endswith("2")
    assert isolated_db.present_streak("R1", as_of="2026-08-18") == 2

    ics_path = tmp_path / "week.ics"
    rc = main(
        [
            "calendar",
            "--from",
            "2026-08-17",
            "--to",
            "2026-08-18",
            "--ics",
            str(ics_path),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    text = ics_path.read_text(encoding="utf-8")
    assert "BEGIN:VEVENT" in text
    assert "bus late" in text

    alias = tmp_path / "alias.ics"
    rc = main(
        [
            "export-ics",
            "--from",
            "2026-08-17",
            "--to",
            "2026-08-18",
            "-o",
            str(alias),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert "BEGIN:VEVENT" in alias.read_text(encoding="utf-8")
