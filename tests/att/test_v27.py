"""2.7 school week, enrollment roster, HTML digest, calendar weekends."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pytest

from attendance.database import (
    Database,
    iso_weekday,
    parse_weekend_days,
)


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
    monkeypatch.setattr(config, "MODEL_PATH", trainer / "lbph_model.yml")
    monkeypatch.setattr(config, "LABEL_MAP_PATH", trainer / "label_map.json")
    monkeypatch.setattr(config, "MODEL_STALE_FLAG", trainer / ".model_stale")
    return Database(db_path=db_path)


def test_parse_weekend_days_names() -> None:
    assert parse_weekend_days("sat,sun") == {6, 7}
    assert parse_weekend_days("Sat,Sun") == {6, 7}
    assert parse_weekend_days("saturday,sunday") == {6, 7}
    assert parse_weekend_days("6,7") == {6, 7}
    assert parse_weekend_days(None) == {6, 7}
    assert iso_weekday("2026-08-22") == 6
    assert iso_weekday("2026-08-17") == 1


def test_weekend_absentees_empty(db: Database) -> None:
    db.add_student("R1", "Ada")
    # 2026-08-22 is a Saturday.
    absentees = db.get_absentees("2026-08-22")
    assert absentees.empty

    digest = db.daily_digest("2026-08-22")
    assert digest["weekend"] is True
    assert digest["school_day"] is False
    assert digest["holiday"] is False
    assert digest["absentee_count"] == 0
    assert digest["attendance_rate"] == pytest.approx(100.0)

    s = db.stats(date="2026-08-22")
    assert s["weekend"] is True
    assert s["school_day"] is False
    assert s["holiday"] is False
    assert s["attendance_rate_today"] == pytest.approx(100.0)


def test_weekend_excluded_from_weekly_summary_days(db: Database) -> None:
    db.add_student("R1", "Ada")
    summary = db.weekly_summary("2026-08-17", days=7)
    assert int(summary.iloc[0]["days"]) == 5
    assert int(summary.iloc[0]["absent"]) == 5


def test_present_streak_skips_saturday(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    db.mark_attendance(sid, at=datetime(2026, 8, 20, 9, 0, 0), status="Present")
    db.mark_attendance(sid, at=datetime(2026, 8, 21, 9, 0, 0), status="Present")
    assert db.present_streak("R1", as_of="2026-08-21") == 2
    assert db.present_streak("R1", as_of="2026-08-22") == 2
    assert db.present_streak("R1", as_of="2026-08-23") == 2


def test_roster_counts_png_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from attendance import config

    faces = tmp_path / "faces"
    faces.mkdir()
    monkeypatch.setattr(config, "FACES_DIR", faces)
    db = Database(db_path=tmp_path / "roster.db")
    sid = db.add_student("R1", "Ada", section="A")
    db.add_student("R2", "Bob", section="A")
    folder = faces / str(sid)
    folder.mkdir()
    (folder / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (folder / "002.png").write_bytes(b"x")
    (folder / "note.txt").write_text("ignored", encoding="utf-8")

    rows = db.enrollment_roster(faces_dir=faces, min_samples=8)
    by_roll = {row["roll_number"]: row for row in rows}
    assert by_roll["R1"]["samples"] == 2
    assert by_roll["R1"]["ready"] is False
    assert by_roll["R1"]["id"] == sid
    assert by_roll["R1"]["section"] == "A"
    assert by_roll["R2"]["samples"] == 0
    assert by_roll["R2"]["ready"] is False

    ready_rows = db.enrollment_roster(min_samples=2)
    ready = {row["roll_number"]: row["ready"] for row in ready_rows}
    assert ready["R1"] is True
    assert ready["R2"] is False


def test_html_digest_contains_html_and_name(db: Database, tmp_path: Path) -> None:
    sid = db.add_student("R1", "Ada Lovelace", section="A")
    db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 9, 0, 0), status="Present"
    )
    path = db.write_daily_digest(
        tmp_path / "digest.html", date="2026-08-18", fmt="html"
    )
    text = path.read_text(encoding="utf-8")
    assert "<html" in text
    assert "Ada Lovelace" in text
    assert "<table" in text


def test_calendar_grid_weekends_include_saturday(db: Database) -> None:
    db.add_student("R1", "Ada")
    grid = db.calendar_grid("2026-08-17", "2026-08-23")
    assert "2026-08-22" in grid["weekends"]
    assert "2026-08-23" in grid["weekends"]
    assert "2026-08-17" not in grid["weekends"]


def test_cli_html_digest_and_roster(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    sid = isolated_db.add_student("R1", "Ada Lovelace", section="A")
    isolated_db.mark_attendance(
        sid, at=datetime(2026, 8, 18, 9, 0, 0), status="Present"
    )
    digest_md = tmp_path / "digest.md"
    rc = main(
        [
            "report",
            "digest",
            "--date",
            "2026-08-18",
            "--html",
            "-o",
            str(digest_md),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    text = digest_md.read_text(encoding="utf-8")
    assert "<html" in text
    assert "Ada Lovelace" in text

    from attendance import config

    folder = config.FACES_DIR / str(sid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "001.png").write_bytes(b"x")
    roster_csv = tmp_path / "roster.csv"
    rc = main(["roster", "--min", "8", "-o", str(roster_csv)])
    assert rc == 0
    capsys.readouterr()
    body = roster_csv.read_text(encoding="utf-8")
    assert "Ada Lovelace" in body
    assert "samples" in body


def test_api_roster(isolated_db: Database) -> None:
    from attendance import config
    from attendance.server import make_server

    sid = isolated_db.add_student("S1", "Api Kid", section="A")
    folder = config.FACES_DIR / str(sid)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "001.png").write_bytes(b"x")

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/roster?min=8", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        assert payload["count"] == 1
        assert payload["min_samples"] == 8
        assert payload["students"][0]["name"] == "Api Kid"
        assert payload["students"][0]["samples"] == 1
        assert payload["students"][0]["ready"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
