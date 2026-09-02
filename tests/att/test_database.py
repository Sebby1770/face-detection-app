"""Database layer tests — tempfile SQLite, no camera required."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from attendance import config
from attendance.database import Database


def test_register_and_list_students(db: Database) -> None:
    sid = db.add_student(
        roll_number="R001",
        name="Ada Lovelace",
        email="ada@example.com",
        department="Math",
        section="A",
    )
    assert sid >= 1
    row = db.get_student(sid)
    assert row is not None
    assert row["roll_number"] == "R001"
    assert row["name"] == "Ada Lovelace"
    assert row["section"] == "A"

    by_roll = db.get_student_by_roll("R001")
    assert by_roll is not None
    assert by_roll["id"] == sid

    students = db.list_students()
    assert len(students) == 1


def test_section_filter_and_list_sections(db: Database) -> None:
    db.add_student("R1", "Alice", section="A", department="CS")
    db.add_student("R2", "Bob", section="B", department="CS")
    db.add_student("R3", "Cara", section="A", department="EE")
    db.add_student("R4", "Dan", section="", department="EE")

    section_a = db.list_students(section="A")
    assert {r["name"] for r in section_a} == {"Alice", "Cara"}
    assert {r["name"] for r in db.list_students(section="B")} == {"Bob"}
    assert set(db.list_sections()) == {"A", "B"}


def test_duplicate_roll_raises(db: Database) -> None:
    db.add_student("R001", "One")
    with pytest.raises(Exception):
        db.add_student("R001", "Two")


def test_mark_attendance_and_cooldown(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ATTENDANCE_COOLDOWN_SECONDS", 30)
    sid = db.add_student("R010", "Eve", section="A")

    t0 = datetime(2026, 3, 1, 9, 0, 0)
    assert db.mark_attendance(sid, confidence=40.0, at=t0) is True
    # Immediate second mark within cooldown is suppressed
    assert db.mark_attendance(sid, confidence=41.0, at=t0 + timedelta(seconds=5)) is False
    # After cooldown, SAME period is still unique — no second row
    assert db.mark_attendance(sid, confidence=42.0, at=t0 + timedelta(seconds=31)) is False

    df = db.get_attendance(date="2026-03-01")
    assert len(df) == 1
    assert list(df["roll_number"]) == ["R010"]


def test_mark_attendance_allows_second_period_same_day(db: Database) -> None:
    sid = db.add_student("R011", "Two Periods")
    t0 = datetime(2026, 3, 1, 9, 0, 0)
    assert db.mark_attendance(sid, at=t0, period="Morning") is True
    assert db.mark_attendance(
        sid, at=t0 + timedelta(seconds=1), period="Afternoon"
    ) is True
    df = db.get_attendance(date="2026-03-01")
    assert len(df) == 2
    assert set(df["period"]) == {"Morning", "Afternoon"}


def test_get_attendance_section_filter(db: Database) -> None:
    a = db.add_student("A1", "Ann", section="A")
    b = db.add_student("B1", "Ben", section="B")
    t = datetime(2026, 4, 1, 10, 0, 0)
    assert db.mark_attendance(a, at=t)
    assert db.mark_attendance(b, at=t)

    df_a = db.get_attendance(date="2026-04-01", section="A")
    assert len(df_a) == 1
    assert df_a.iloc[0]["name"] == "Ann"
    assert "section" in df_a.columns


def test_absentees(db: Database) -> None:
    present = db.add_student("P1", "Present Pupil", section="A")
    db.add_student("A1", "Absent Alice", section="A")
    db.add_student("A2", "Absent Bob", section="B")

    t = datetime(2026, 5, 11, 8, 30, 0)
    assert db.mark_attendance(present, at=t) is True

    all_abs = db.get_absentees(date="2026-05-11")
    assert {r for r in all_abs["name"]} == {"Absent Alice", "Absent Bob"}

    abs_a = db.get_absentees(date="2026-05-11", section="A")
    assert list(abs_a["name"]) == ["Absent Alice"]

    # No one present on another weekday → everyone absent
    everyone = db.get_absentees(date="2026-05-12")
    assert len(everyone) == 3


def test_export_absentees_csv(db: Database, tmp_path: Path) -> None:
    db.add_student("X1", "Zoe", section="C")
    out = tmp_path / "abs.csv"
    path = db.export_absentees_csv(out, date="2026-06-01")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Zoe" in text
    assert "X1" in text


def test_export_attendance_csv(db: Database, tmp_path: Path) -> None:
    sid = db.add_student("E1", "Export Me", section="Z")
    db.mark_attendance(sid, confidence=33.0, at=datetime(2026, 6, 2, 11, 0, 0))
    out = tmp_path / "att.csv"
    path = db.export_attendance_csv(out, date="2026-06-02")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Export Me" in content


def test_daily_digest(db: Database, tmp_path: Path) -> None:
    p = db.add_student("D1", "Present", section="A")
    db.add_student("D2", "Missing", section="A")
    db.mark_attendance(p, at=datetime(2026, 7, 1, 9, 0, 0))

    digest = db.daily_digest(date="2026-07-01", section="A")
    assert digest["total_students"] == 2
    assert digest["present_count"] == 1
    assert digest["absentee_count"] == 1
    assert digest["attendance_rate"] == pytest.approx(50.0)
    assert list(digest["absentees"]["name"]) == ["Missing"]

    md = db.write_daily_digest(tmp_path / "d.md", date="2026-07-01", fmt="md")
    assert md.exists()
    body = md.read_text(encoding="utf-8")
    assert "Missing" in body
    assert "50.0%" in body or "50%" in body

    txt = db.write_daily_digest(tmp_path / "d.txt", date="2026-07-01", fmt="txt")
    assert "Present" in txt.read_text(encoding="utf-8") or "Absent" in txt.read_text(
        encoding="utf-8"
    )


def test_stats(db: Database) -> None:
    sid = db.add_student("S1", "Stat Student")
    db.mark_attendance(sid, at=datetime.now())
    s = db.stats()
    assert s["total_students"] == 1
    assert s["total_records"] == 1
    assert s["present_today"] == 1
    assert s["attendance_rate_today"] == pytest.approx(100.0)


def test_delete_student_cascades_attendance(db: Database) -> None:
    sid = db.add_student("DEL1", "Doomed")
    db.mark_attendance(sid, at=datetime(2026, 1, 1, 12, 0, 0))
    assert len(db.get_attendance()) == 1
    db.delete_student(sid)
    assert db.get_student(sid) is None
    assert db.get_attendance().empty


def test_section_migration_on_legacy_schema(tmp_path: Path) -> None:
    """Existing DBs without a section column get it via ALTER TABLE."""
    import sqlite3

    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            department TEXT,
            registered_on TEXT NOT NULL
        );
        CREATE TABLE attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        INSERT INTO students (roll_number, name, email, department, registered_on)
        VALUES ('LEG1', 'Legacy Kid', '', 'CS', '2025-01-01 00:00:00');
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_path=legacy)
    rows = db.list_students()
    assert len(rows) == 1
    # section column exists and is readable (NULL/None or empty)
    assert "section" in rows[0].keys()

    sid = db.add_student("NEW1", "New Kid", section="B")
    row = db.get_student(sid)
    assert row is not None
    assert row["section"] == "B"

    marked = db.mark_attendance(sid, at=datetime(2026, 8, 18, 9, 0, 0), status="Present")
    assert marked is True
    df = db.get_attendance()
    assert "period" in df.columns


def test_settings_persist(db: Database) -> None:
    assert db.get_setting("theme") == "light"
    db.set_setting("theme", "dark")
    db.set_setting("confidence_threshold", "62.5")
    assert db.get_setting("theme") == "dark"
    assert db.get_setting("confidence_threshold") == "62.5"
    all_s = db.all_settings()
    assert all_s["grace_minutes"] == str(config.LATE_GRACE_MINUTES)


def test_export_attendance_json(db: Database, tmp_path: Path) -> None:
    import json

    sid = db.add_student("J1", "JSON Kid", section="Z")
    db.mark_attendance(
        sid, confidence=33.0, at=datetime(2026, 6, 2, 11, 0, 0), status="Present"
    )
    out = tmp_path / "att.json"
    path = db.export_attendance_json(out, date="2026-06-02")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "attendance"
    assert payload["count"] == 1
    assert payload["records"][0]["name"] == "JSON Kid"
    assert payload["records"][0]["roll_number"] == "J1"


def test_export_absentees_json(db: Database, tmp_path: Path) -> None:
    import json

    db.add_student("X1", "Zoe", section="C")
    out = tmp_path / "abs.json"
    path = db.export_absentees_json(out, date="2026-06-01")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "absentees"
    assert payload["count"] == 1
    assert payload["records"][0]["name"] == "Zoe"


def test_unknown_faces_log_and_cooldown(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "UNKNOWN_LOG_COOLDOWN_SECONDS", 5)
    t0 = datetime(2026, 8, 18, 10, 0, 0)
    assert db.log_unknown_face(88.0, at=t0) is True
    assert db.log_unknown_face(90.0, at=t0 + timedelta(seconds=2)) is False
    assert db.log_unknown_face(91.0, at=t0 + timedelta(seconds=6)) is True
    assert db.count_unknown_faces("2026-08-18") == 2
    assert db.count_unknown_faces("2026-08-17") == 0
    listed = db.list_unknown_faces(date="2026-08-18")
    assert len(listed) == 2

    s = db.stats(date="2026-08-18")
    assert s["unknown_today"] == 2


def test_digest_includes_late_count(db: Database, tmp_path: Path) -> None:
    on_time = db.add_student("OT1", "On Time", section="A")
    late = db.add_student("LT1", "Late Kid", section="A")
    db.add_student("AB1", "Missing", section="A")
    db.mark_attendance(
        on_time, at=datetime(2026, 8, 18, 9, 0, 0), period="Morning"
    )
    db.mark_attendance(
        late, at=datetime(2026, 8, 18, 9, 30, 0), period="Morning"
    )

    digest = db.daily_digest(date="2026-08-18", section="A")
    assert digest["present_count"] == 2
    assert digest["late_count"] == 1
    assert digest["on_time_count"] == 1
    assert digest["absentee_count"] == 1
    assert digest["attendance_rate"] == pytest.approx(200.0 / 3)

    md = db.write_daily_digest(tmp_path / "late.md", date="2026-08-18", fmt="md")
    body = md.read_text(encoding="utf-8")
    assert "Late" in body
    txt = db.write_daily_digest(tmp_path / "late.txt", date="2026-08-18", fmt="txt")
    assert "Late" in txt.read_text(encoding="utf-8")


def test_mark_excused_and_absentees(db: Database) -> None:
    sid = db.add_student("EX1", "Excused Eve", section="A")
    db.add_student("AB1", "Absent Abe", section="A")
    assert db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 10, 0, 0), status="Excused"
    )
    df = db.get_attendance(date="2026-08-18")
    assert df.iloc[0]["status"] == "Excused"
    absentees = db.get_absentees(date="2026-08-18")
    assert list(absentees["name"]) == ["Absent Abe"]
    s = db.stats(date="2026-08-18")
    assert s["excused_today"] == 1
    assert s["present_today"] == 0


def test_mark_sick_normalizes_to_excused(db: Database) -> None:
    sid = db.add_student("SK1", "Sick Sam")
    assert db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 8, 0, 0), status="Sick"
    )
    assert db.get_attendance().iloc[0]["status"] == "Excused"


def test_digest_includes_excused(db: Database, tmp_path: Path) -> None:
    on_time = db.add_student("OT1", "On Time", section="A")
    excused = db.add_student("EX1", "Home Sick", section="A")
    db.add_student("AB1", "Missing", section="A")
    db.mark_attendance(
        on_time, at=datetime(2026, 8, 18, 9, 0, 0), status="Present"
    )
    db.mark_attendance(
        excused, at=datetime(2026, 8, 18, 9, 5, 0), status="Excused"
    )

    digest = db.daily_digest(date="2026-08-18", section="A")
    assert digest["present_count"] == 1
    assert digest["excused_count"] == 1
    assert digest["absentee_count"] == 1
    assert list(digest["excused"]["name"]) == ["Home Sick"]

    md = db.write_daily_digest(tmp_path / "ex.md", date="2026-08-18", fmt="md")
    body = md.read_text(encoding="utf-8")
    assert "Excused" in body
    assert "Home Sick" in body
    txt = db.write_daily_digest(tmp_path / "ex.txt", date="2026-08-18", fmt="txt")
    assert "Excused" in txt.read_text(encoding="utf-8")


def test_holidays_upsert_and_list(db: Database) -> None:
    db.add_holiday("2026-12-25", "Christmas")
    db.add_holiday("2026-01-01", "New Year")
    db.add_holiday("2026-12-25", "Christmas Day")
    rows = db.list_holidays()
    assert [r["date"] for r in rows] == ["2026-01-01", "2026-12-25"]
    assert db.get_holiday("2026-12-25")["name"] == "Christmas Day"
    assert db.holiday_dates_between("2026-12-01", "2026-12-31") == {"2026-12-25"}


def test_import_students_csv(db: Database, tmp_path: Path) -> None:
    csv_path = tmp_path / "roster.csv"
    csv_path.write_text(
        "roll_number,name,email,department,section\n"
        "R001,Ada Lovelace,ada@example.com,Math,A\n"
        "R002,Alan Turing,alan@example.com,CS,B\n"
        "R001,Duplicate Ada,skip@example.com,Math,A\n",
        encoding="utf-8",
    )
    result = db.import_students_csv(csv_path)
    assert result["added"] == 2
    assert result["skipped"] == 1
    assert db.get_student_by_roll("R001")["name"] == "Ada Lovelace"
    assert db.get_student_by_roll("R002")["name"] == "Alan Turing"

    again = db.import_students_csv(csv_path)
    assert again["added"] == 0
    assert again["skipped"] == 3
    assert again["updated"] == 0
    assert len(db.list_students()) == 2


def test_import_students_csv_update(db: Database, tmp_path: Path) -> None:
    db.add_student(
        "R001", "Old Ada", email="old@example.com", department="X", section="Z"
    )
    csv_path = tmp_path / "roster.csv"
    csv_path.write_text(
        "roll_number,name,email,department,section\n"
        "R001,Ada Lovelace,ada@example.com,Math,A\n"
        "R002,Alan Turing,alan@example.com,CS,B\n",
        encoding="utf-8",
    )
    skipped = db.import_students_csv(csv_path)
    assert skipped["added"] == 1
    assert skipped["skipped"] == 1
    assert skipped["updated"] == 0
    assert db.get_student_by_roll("R001")["name"] == "Old Ada"

    result = db.import_students_csv(csv_path, update=True)
    assert result["added"] == 0
    assert result["updated"] == 2
    assert result["skipped"] == 0
    row = db.get_student_by_roll("R001")
    assert row["name"] == "Ada Lovelace"
    assert row["email"] == "ada@example.com"
    assert row["department"] == "Math"
    assert row["section"] == "A"
    assert db.get_student_by_roll("R002")["name"] == "Alan Turing"


def test_merge_students_moves_attendance_and_deletes_from(
    db: Database, tmp_path: Path
) -> None:
    from_id = db.add_student("OLD1", "Old Name", email="old@x.com")
    to_id = db.add_student("NEW1", "New Name")
    db.mark_attendance(from_id, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(from_id, at=datetime(2026, 8, 18, 9, 0, 0), status="Late")
    db.mark_attendance(to_id, at=datetime(2026, 8, 19, 9, 0, 0), status="Present")

    faces = tmp_path / "faces"
    (faces / str(from_id)).mkdir(parents=True)
    (faces / str(from_id) / "001.png").write_bytes(b"x")

    result = db.merge_students("OLD1", "NEW1", faces_dir=faces)
    assert result["moved"] == 2
    assert result["from_id"] == from_id
    assert result["to_id"] == to_id
    assert db.get_student_by_roll("OLD1") is None
    assert db.get_student_by_roll("NEW1") is not None
    att = db.get_attendance()
    assert len(att) == 3
    assert set(att["roll_number"]) == {"NEW1"}
    assert not (faces / str(from_id)).exists()

    with pytest.raises(ValueError, match="themselves"):
        db.merge_students("NEW1", "NEW1")
    with pytest.raises(ValueError, match="no student"):
        db.merge_students("GONE", "NEW1")


def test_range_report_counts(db: Database, tmp_path: Path) -> None:
    import json

    ada = db.add_student("R1", "Ada", section="A")
    bob = db.add_student("R2", "Bob", section="A")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(ada, at=datetime(2026, 8, 18, 9, 40, 0), status="Late")
    db.mark_attendance(ada, at=datetime(2026, 8, 19, 9, 0, 0), status="Excused")
    db.mark_attendance(bob, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")

    df = db.range_report("2026-08-17", "2026-08-19", section="A")
    assert len(df) == 2
    ada_row = df.loc[df["roll_number"] == "R1"].iloc[0]
    assert int(ada_row["present"]) == 1
    assert int(ada_row["late"]) == 1
    assert int(ada_row["excused"]) == 1
    assert int(ada_row["absent"]) == 0
    assert int(ada_row["days"]) == 3

    csv_path = db.export_range_report_csv(
        tmp_path / "range.csv", "2026-08-17", "2026-08-19"
    )
    assert "Ada" in csv_path.read_text(encoding="utf-8")
    json_path = db.export_range_report_json(
        tmp_path / "range.json", "2026-08-17", "2026-08-19"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["format"] == "range_report"
    assert payload["from"] == "2026-08-17"
    assert payload["to"] == "2026-08-19"
    assert payload["days"] == 3
    assert payload["count"] == 2


def test_export_unknown_faces(db: Database, tmp_path: Path) -> None:
    import json

    db.log_unknown_face(88.0, at=datetime(2026, 8, 18, 10, 0, 0))
    db.log_unknown_face(91.0, at=datetime(2026, 8, 18, 10, 10, 0))
    csv_path = db.export_unknown_faces_csv(tmp_path / "unk.csv", date="2026-08-18")
    text = csv_path.read_text(encoding="utf-8")
    assert "confidence" in text
    json_path = db.export_unknown_faces_json(tmp_path / "unk.json", date="2026-08-18")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["format"] == "unknown_faces"
    assert payload["count"] == 2
    assert payload["date"] == "2026-08-18"


def test_stats_includes_late_and_unknown(db: Database) -> None:
    sid = db.add_student("S1", "Stat Student")
    db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 9, 45, 0), period="Morning"
    )
    db.log_unknown_face(99.0, at=datetime(2026, 8, 18, 9, 46, 0))
    s = db.stats(date="2026-08-18")
    assert s["late_today"] == 1
    assert s["on_time_today"] == 0
    assert s["present_today"] == 1
    assert s["unknown_today"] == 1
    assert "week_rate" in s

