"""2.9 dwell time, liveness helpers, unknown assign cleanup, range HTML, CLI --out."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from attendance.app import motion_liveness_ok, push_motion_sample
from attendance.database import Database, calendar_cell_letter


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


def test_unique_still_one_row_touch_sets_time_out(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    inn = datetime(2026, 8, 17, 9, 0, 0)
    later = datetime(2026, 8, 17, 9, 15, 0)

    assert db.mark_attendance(sid, at=inn, period="Morning", status="Present")
    df = db.get_attendance(date="2026-08-17")
    assert len(df) == 1
    assert df.iloc[0]["duration_seconds"] is None

    assert (
        db.mark_attendance(
            sid,
            at=later,
            period="Morning",
            status="Late",
        )
        is False
    )
    df = db.get_attendance(date="2026-08-17")
    assert len(df) == 1
    assert df.iloc[0]["duration_seconds"] is None

    assert (
        db.mark_attendance(
            sid,
            at=later,
            period="Morning",
            touch=True,
        )
        is False
    )
    df = db.get_attendance(date="2026-08-17")
    assert len(df) == 1
    assert str(df.iloc[0]["time"]) == "09:00:00"
    assert str(df.iloc[0]["time_out"]) == "09:15:00"
    assert str(df.iloc[0]["time_out"]) > str(df.iloc[0]["time"])
    assert int(df.iloc[0]["duration_seconds"]) > 0
    assert int(df.iloc[0]["duration_seconds"]) == 15 * 60


def test_unique_per_period_still_allows_afternoon(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    morning = datetime(2026, 8, 17, 9, 0, 0)
    assert db.mark_attendance(sid, at=morning, period="Morning", status="Present")
    assert (
        db.mark_attendance(
            sid,
            at=morning + timedelta(seconds=40),
            period="Morning",
            status="Late",
        )
        is False
    )
    assert db.mark_attendance(
        sid,
        at=datetime(2026, 8, 17, 13, 0, 0),
        period="Afternoon",
        status="Present",
    )
    df = db.get_attendance(date="2026-08-17")
    assert len(df) == 2
    assert set(df["period"]) == {"Morning", "Afternoon"}


def test_touch_does_not_move_time_out_backwards(db: Database) -> None:
    sid = db.add_student("R1", "Ada")
    db.mark_attendance(sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 10, 0, 0), touch=True
    )
    db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 30, 0), touch=True
    )
    df = db.get_attendance(date="2026-08-17")
    assert str(df.iloc[0]["time_out"]) == "10:00:00"
    assert int(df.iloc[0]["duration_seconds"]) == 3600


def test_assign_unknown_crop_removes_unknown_faces_row(
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
    db.add_student("R1", "Ada")
    src = tmp_path / "unknown.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert db.log_unknown_face(
        80.0, at=datetime(2026, 8, 17, 11, 0, 0), path=str(src)
    )
    assert len(db.list_unknown_crops()) == 1

    result = db.assign_unknown_crop(src, "R1")
    assert Path(result["dest"]).is_file()
    assert db.list_unknown_crops() == []
    assert not src.exists()


def test_calendar_grid_still_works(db: Database) -> None:
    ada = db.add_student("R1", "Ada", section="A")
    bob = db.add_student("R2", "Bob", section="A")
    db.add_holiday("2026-08-19", "Break")
    db.mark_attendance(ada, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(ada, at=datetime(2026, 8, 18, 9, 40, 0), status="Late")
    db.mark_attendance(bob, at=datetime(2026, 8, 17, 9, 0, 0), status="Excused")

    grid = db.calendar_grid("2026-08-17", "2026-08-19", section="A")
    assert grid["from"] == "2026-08-17"
    assert grid["to"] == "2026-08-19"
    assert grid["dates"][0] == "2026-08-17"
    ada_days = next(s for s in grid["students"] if s["roll_number"] == "R1")["days"]
    assert ada_days["2026-08-17"] == "Present"
    assert ada_days["2026-08-18"] == "Late"
    holidays = grid["holidays"]
    weekends = set(grid.get("weekends") or [])
    assert calendar_cell_letter(
        "2026-08-17", ada_days["2026-08-17"], holidays, weekends
    ) == "P"
    assert calendar_cell_letter("2026-08-19", None, holidays, weekends) == "H"


def test_cli_mark_out(isolated_db: Database, capsys: pytest.CaptureFixture) -> None:
    from main import main

    isolated_db.add_student("R1", "Ada")
    rc = main(["mark", "--roll", "R1", "--at", "2026-08-17T09:00:00"])
    assert rc == 0
    capsys.readouterr()
    rc = main(
        ["mark", "--roll", "R1", "--out", "--at", "2026-08-17T09:10:00"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "time_out" in out
    df = isolated_db.get_attendance(date="2026-08-17")
    assert len(df) == 1
    assert str(df.iloc[0]["time_out"]) == "09:10:00"
    assert int(df.iloc[0]["duration_seconds"]) == 600


def test_export_csv_json_include_time_out(db: Database, tmp_path: Path) -> None:
    sid = db.add_student("E1", "Export Me")
    db.mark_attendance(sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present")
    db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 5, 0), touch=True
    )
    csv_path = db.export_attendance_csv(tmp_path / "att.csv", date="2026-08-17")
    text = csv_path.read_text(encoding="utf-8")
    assert "time_out" in text.splitlines()[0]
    assert "09:05:00" in text
    json_path = db.export_attendance_json(tmp_path / "att.json", date="2026-08-17")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["time_out"] == "09:05:00"
    assert payload["records"][0]["duration_seconds"] == 300


def test_range_html_report(
    isolated_db: Database, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from main import main

    sid = isolated_db.add_student("R1", "Ada Lovelace", section="A")
    isolated_db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present"
    )
    out = tmp_path / "range.html"
    rc = main(
        [
            "report",
            "--from",
            "2026-08-17",
            "--to",
            "2026-08-21",
            "--html",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    assert "<html" in text.lower()
    assert "Ada Lovelace" in text
    assert "<table" in text
    assert "color-scheme: dark light" in text


def test_api_mark_touch(isolated_db: Database) -> None:
    from attendance.server import make_server

    isolated_db.add_student("S1", "Api Kid")
    sid = isolated_db.get_student_by_roll("S1")["id"]
    isolated_db.mark_attendance(
        sid, at=datetime(2026, 8, 17, 9, 0, 0), status="Present", source="api"
    )

    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps(
            {
                "roll": "S1",
                "touch": True,
                "at": "2026-08-17T09:20:00",
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
        assert marked["inserted"] is False
        df = isolated_db.get_attendance(date="2026-08-17")
        assert str(df.iloc[0]["time_out"]) == "09:20:00"
        assert int(df.iloc[0]["duration_seconds"]) == 20 * 60
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_motion_liveness_requires_16px() -> None:
    history: dict[int, list[tuple[float, float]]] = {}
    samples = push_motion_sample(history, 1, (10.0, 10.0), cap=12)
    assert motion_liveness_ok(samples) is False
    for i in range(11):
        push_motion_sample(history, 1, (10.0 + i, 10.0), cap=12)
    # 11 extra samples: x from 10 to 20 → range 10 < 16
    assert motion_liveness_ok(history[1]) is False
    push_motion_sample(history, 1, (10.0 + 16, 10.0), cap=12)
    assert motion_liveness_ok(history[1]) is True
    assert len(history[1]) <= 12
