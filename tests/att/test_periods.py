"""Period start times and Present vs Late classification (clock-driven)."""
from __future__ import annotations

from datetime import datetime

import pytest

from attendance import config
from attendance.database import (
    Database,
    classify_attendance_status,
    format_hhmm,
    parse_hhmm,
)


def test_parse_hhmm_variants() -> None:
    assert parse_hhmm("09:00") == (9, 0)
    assert parse_hhmm("9:05") == (9, 5)
    assert parse_hhmm("1300") == (13, 0)
    assert parse_hhmm("900") == (9, 0)
    assert parse_hhmm("13.30") == (13, 30)
    assert format_hhmm(9, 5) == "09:05"


def test_parse_hhmm_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        parse_hhmm("")
    with pytest.raises(ValueError):
        parse_hhmm("25:00")
    with pytest.raises(ValueError):
        parse_hhmm("noon")


@pytest.mark.parametrize(
    "clock, start, grace, expected",
    [
        (datetime(2026, 8, 18, 8, 59, 0), "09:00", 10, "Present"),
        (datetime(2026, 8, 18, 9, 0, 0), "09:00", 10, "Present"),
        (datetime(2026, 8, 18, 9, 10, 0), "09:00", 10, "Present"),
        (datetime(2026, 8, 18, 9, 10, 1), "09:00", 10, "Late"),
        (datetime(2026, 8, 18, 9, 11, 0), "09:00", 10, "Late"),
        (datetime(2026, 8, 18, 13, 5, 0), "13:00", 10, "Present"),
        (datetime(2026, 8, 18, 13, 20, 0), "13:00", 10, "Late"),
        (datetime(2026, 8, 18, 9, 30, 0), "09:00", 0, "Late"),
        (datetime(2026, 8, 18, 9, 0, 0), "09:00", 0, "Present"),
    ],
)
def test_classify_on_time_vs_late(
    clock: datetime, start: str, grace: int, expected: str
) -> None:
    assert classify_attendance_status(clock, start, grace) == expected


def test_default_period_and_grace_constants() -> None:
    assert config.DEFAULT_PERIOD_START == "09:00"
    assert config.LATE_GRACE_MINUTES == 10
    on_time = classify_attendance_status(datetime(2026, 8, 18, 9, 10, 0))
    late = classify_attendance_status(datetime(2026, 8, 18, 9, 11, 0))
    assert on_time == "Present"
    assert late == "Late"


def test_periods_seeded(db: Database) -> None:
    names = {r["name"] for r in db.list_periods()}
    assert "Morning" in names
    assert "Afternoon" in names
    morning = db.get_period("morning")
    assert morning is not None
    assert morning["start_hhmm"] == "09:00"


def test_upsert_period(db: Database) -> None:
    db.upsert_period("Lab", "14:30")
    row = db.get_period("Lab")
    assert row is not None
    assert row["start_hhmm"] == "14:30"
    db.upsert_period("Lab", "15:00")
    assert db.get_period("Lab")["start_hhmm"] == "15:00"


def test_mark_attendance_classifies_from_clock(db: Database) -> None:
    sid = db.add_student("P1", "Punctual")
    late_id = db.add_student("L1", "Lagging")

    assert db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 9, 5, 0), period="Morning"
    )
    assert db.mark_attendance(
        late_id, at=datetime(2026, 8, 18, 9, 25, 0), period="Morning"
    )

    on_time = db.get_attendance(date="2026-08-18", student_id=sid)
    late = db.get_attendance(date="2026-08-18", student_id=late_id)
    assert on_time.iloc[0]["status"] == "Present"
    assert on_time.iloc[0]["period"] == "Morning"
    assert late.iloc[0]["status"] == "Late"
    assert late.iloc[0]["period"] == "Morning"


def test_afternoon_period_uses_its_start(db: Database) -> None:
    sid = db.add_student("A1", "After")
    # 10:00 is late for Morning but on time for Afternoon (13:00 + 10).
    assert db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 10, 0, 0), period="Afternoon"
    )
    df = db.get_attendance(date="2026-08-18")
    assert df.iloc[0]["status"] == "Present"
    assert df.iloc[0]["period"] == "Afternoon"

    sid2 = db.add_student("A2", "After Late")
    assert db.mark_attendance(
        sid2, at=datetime(2026, 8, 18, 13, 20, 0), period="Afternoon"
    )
    df2 = db.get_attendance(student_id=sid2)
    assert df2.iloc[0]["status"] == "Late"


def test_explicit_status_overrides_clock(db: Database) -> None:
    sid = db.add_student("F1", "Forced")
    assert db.mark_attendance(
        sid,
        status="Present",
        at=datetime(2026, 8, 18, 18, 0, 0),
        period="Morning",
    )
    assert db.get_attendance().iloc[0]["status"] == "Present"


def test_explicit_excused_overrides_clock(db: Database) -> None:
    sid = db.add_student("E1", "Excused")
    assert db.mark_attendance(
        sid,
        status="Excused",
        at=datetime(2026, 8, 18, 9, 0, 0),
        period="Morning",
    )
    assert db.get_attendance().iloc[0]["status"] == "Excused"


def test_custom_grace_from_settings(db: Database) -> None:
    db.set_setting("grace_minutes", "30")
    sid = db.add_student("G1", "Grace")
    assert db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 9, 25, 0), period="Morning"
    )
    assert db.get_attendance().iloc[0]["status"] == "Present"
    sid2 = db.add_student("G2", "Too Late")
    assert db.mark_attendance(
        sid2, at=datetime(2026, 8, 18, 9, 31, 0), period="Morning"
    )
    assert db.get_attendance(student_id=sid2).iloc[0]["status"] == "Late"
