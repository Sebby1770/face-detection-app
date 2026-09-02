"""Weekly present / late / absent rates on a temp database."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from attendance.database import Database, week_dates, week_start


def test_week_start_is_monday() -> None:
    # 2026-08-18 is a Tuesday.
    assert week_start("2026-08-18") == "2026-08-17"
    assert week_start("2026-08-17") == "2026-08-17"
    assert week_start("2026-08-23") == "2026-08-17"
    days = week_dates("2026-08-17", days=7)
    assert days[0] == "2026-08-17"
    assert days[-1] == "2026-08-23"
    assert len(week_dates("2026-08-17", days=5)) == 5


def test_weekly_summary_counts(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    bob = db.add_student("R2", "Bob", section="A")
    # Week of Monday 2026-08-17.
    # Ada: present Mon, late Tue, absent the rest.
    db.mark_attendance(
        ada, at=datetime(2026, 8, 17, 8, 50, 0), period="Morning"
    )
    db.mark_attendance(
        ada, at=datetime(2026, 8, 18, 9, 40, 0), period="Morning"
    )
    # Bob: present all 5 weekdays.
    for offset in range(5):
        day = datetime(2026, 8, 17) + timedelta(days=offset)
        db.mark_attendance(
            bob, at=day.replace(hour=9, minute=0, second=0), period="Morning"
        )

    summary = db.weekly_summary("2026-08-17", days=7)
    assert len(summary) == 2
    ada_row = summary.loc[summary["roll_number"] == "R1"].iloc[0]
    bob_row = summary.loc[summary["roll_number"] == "R2"].iloc[0]
    assert int(ada_row["present"]) == 1
    assert int(ada_row["late"]) == 1
    assert int(ada_row["absent"]) == 3
    assert int(ada_row["days"]) == 5
    assert ada_row["attendance_rate"] == pytest.approx(2 / 5 * 100)
    assert int(bob_row["present"]) == 5
    assert int(bob_row["late"]) == 0
    assert int(bob_row["absent"]) == 0
    assert bob_row["attendance_rate"] == pytest.approx(100.0)


def test_weekly_summary_five_day_window(db: Database) -> None:
    sid = db.add_student("W1", "Weekday")
    for offset in range(5):
        day = datetime(2026, 8, 17) + timedelta(days=offset)
        db.mark_attendance(
            sid, at=day.replace(hour=9, minute=0), status="Present"
        )
    five = db.weekly_summary("2026-08-17", days=5)
    row = five.iloc[0]
    assert int(row["present"]) == 5
    assert int(row["absent"]) == 0
    assert int(row["days"]) == 5
    assert row["attendance_rate"] == pytest.approx(100.0)


def test_weekly_summary_section_filter(db: Database) -> None:
    a = db.add_student("A1", "Ann", section="A")
    db.add_student("B1", "Ben", section="B")
    db.mark_attendance(a, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    only_a = db.weekly_summary("2026-08-17", days=7, section="A")
    assert list(only_a["name"]) == ["Ann"]


def test_weekly_summary_empty_db(db: Database) -> None:
    df = db.weekly_summary("2026-08-17", days=7)
    assert df.empty
    assert "attendance_rate" in df.columns
    assert db.weekly_rate("2026-08-17", days=7) == 0.0


def test_weekly_rate_overall(db: Database) -> None:
    a = db.add_student("A1", "Ann")
    b = db.add_student("B1", "Ben")
    db.mark_attendance(a, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(b, at=datetime(2026, 8, 17, 9, 30, 0), status="Late")
    # 2 students × 2 days = 4 slots, 2 attended → 50%
    rate = db.weekly_rate("2026-08-17", days=2)
    assert rate == pytest.approx(50.0)


def test_weekly_excused_not_absent(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    # Mon present, Tue excused, remaining weekdays unmarked.
    db.mark_attendance(
        ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present"
    )
    db.mark_attendance(
        ada, at=datetime(2026, 8, 18, 9, 0, 0), status="Excused"
    )
    row = db.weekly_summary("2026-08-17", days=7).iloc[0]
    assert int(row["present"]) == 1
    assert int(row["late"]) == 0
    assert int(row["excused"]) == 1
    assert int(row["absent"]) == 3
    assert int(row["days"]) == 5
    assert row["attendance_rate"] == pytest.approx(1 / 5 * 100)


def test_weekly_holidays_excluded_from_absent_and_denominator(db: Database) -> None:
    ada = db.add_student("R1", "Ada")
    db.add_holiday("2026-08-19", "Midweek break")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    row = db.weekly_summary("2026-08-17", days=7).iloc[0]
    # Mon–Sun minus Sat/Sun minus 1 midweek holiday = 4 school days.
    assert int(row["present"]) == 1
    assert int(row["excused"]) == 0
    assert int(row["absent"]) == 3
    assert int(row["days"]) == 4
    assert row["attendance_rate"] == pytest.approx(1 / 4 * 100)
    rng = db.range_report("2026-08-17", "2026-08-23")
    assert int(rng.iloc[0]["absent"]) == 3
    assert int(rng.iloc[0]["days"]) == 4
    assert db.weekly_rate("2026-08-17", days=7) == pytest.approx(1 / 4 * 100)


def test_weekly_uses_first_mark_of_day(db: Database) -> None:
    sid = db.add_student("F1", "First")
    db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present", period="Morning"
    )
    db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 13, 45, 0), status="Late", period="Afternoon"
    )
    row = db.weekly_summary("2026-08-17", days=1).iloc[0]
    assert int(row["present"]) == 1
    assert int(row["late"]) == 0
